/**
 * This file is part of the "libunicode" project
 *   Copyright (c) 2020 Christian Parpart <christian@parpart.family>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "multistage_generator.h"

#include <libunicode/multistage_table_generator.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "enum_utils.h"
#include "ucd_parser.h"

namespace tablegen
{

namespace
{
    // Total codepoints in Unicode
    constexpr size_t CODEPOINT_COUNT = 0x110'000;
    constexpr size_t BLOCK_SIZE = 256;

    // ---- Internal record type (no dependency on generated enums) ----

    // Flag bits, must match codepoint_properties::Flag* values
    constexpr uint8_t FlagEmoji = 0x01;
    constexpr uint8_t FlagEmojiPresentation = 0x02;
    constexpr uint8_t FlagEmojiComponent = 0x04;
    constexpr uint8_t FlagEmojiModifier = 0x08;
    constexpr uint8_t FlagEmojiModifierBase = 0x10;
    constexpr uint8_t FlagExtendedPictographic = 0x20;
    constexpr uint8_t FlagCoreGraphemeExtend = 0x40;
    constexpr uint8_t FlagVirama = 0x80;

    // `flags` is exhausted; further single-bit properties live in `flags2`.
    constexpr uint8_t Flag2EmojiVariationBase = 0x01;

    // EmojiSegmentationCategory integer values, must match the enum
    constexpr int8_t ESC_Invalid = -1;
    constexpr int8_t ESC_Emoji = 0;
    constexpr int8_t ESC_EmojiTextPresentation = 1;
    constexpr int8_t ESC_EmojiEmojiPresentation = 2;
    constexpr int8_t ESC_EmojiModifierBase = 3;
    constexpr int8_t ESC_EmojiModifier = 4;
    // constexpr int8_t ESC_EmojiVSBase = 5;
    constexpr int8_t ESC_RegionalIndicator = 6;
    constexpr int8_t ESC_KeyCapBase = 7;
    constexpr int8_t ESC_CombiningEnclosingKeyCap = 8;
    constexpr int8_t ESC_CombiningEnclosingCircleBackslash = 9;
    constexpr int8_t ESC_ZWJ = 10;
    constexpr int8_t ESC_VS15 = 11;
    constexpr int8_t ESC_VS16 = 12;
    constexpr int8_t ESC_TagBase = 13;
    constexpr int8_t ESC_TagSequence = 14;
    constexpr int8_t ESC_TagTerm = 15;

// Packed record matching codepoint_properties layout exactly
#pragma pack(push, 1)
    struct CodepointRecord
    {
        uint8_t char_width = 0;
        uint8_t flags = 0;
        uint8_t flags2 = 0;
        uint8_t script = 0;                 // Will be set per-codepoint
        uint8_t grapheme_cluster_break = 0; // Will be set per-codepoint
        uint8_t east_asian_width = 0;       // Will be set per-codepoint
        uint8_t general_category = 0;       // Will be set per-codepoint
        int8_t emoji_segmentation_category = ESC_Invalid;
        uint8_t age = 0;
        uint8_t indic_conjunct_break = 3; // Default: Indic_Conjunct_Break::None
        uint8_t word_break = 18;          // Default: Word_Break::Other
    };
#pragma pack(pop)

    static_assert(sizeof(CodepointRecord) == 11, "CodepointRecord must be exactly 11 bytes");

    inline bool operator==(CodepointRecord const& a, CodepointRecord const& b) noexcept
    {
        return std::memcmp(&a, &b, sizeof(CodepointRecord)) == 0;
    }

    // ---- Enum index builders ----
    // Build name->index maps using the exact same member ordering as the generated
    // ucd_enums.h. This is critical: the integer values in CodepointRecord must
    // match the generated enum class values for the multistage tables to be correct.

    /// Build name -> index map from enum members list.
    /// Build PVA-based enum index: sorted full-name values, with abbreviation mappings.
    /// The abbrev->fullname map from PVA is used to add abbreviation keys.
    std::map<std::string, uint8_t> buildPvaBasedIndex(std::map<std::string, std::string> const& aliases,
                                                      std::string const& firstMember = {})
    {
        // Build sorted member list (same as enum generator)
        std::vector<std::string> sorted;
        for (auto const& [k, v]: aliases)
            sorted.push_back(v);
        std::sort(sorted.begin(), sorted.end());

        std::map<std::string, uint8_t> result;
        uint8_t idx = 0;
        if (!firstMember.empty())
            result[firstMember] = idx++;
        for (auto const& v: sorted)
            if (result.find(v) == result.end())
                result[v] = idx++;
        // Add abbreviation mappings
        for (auto const& [abbrev, full]: aliases)
            if (result.count(full))
                result[abbrev] = result[full];
        return result;
    }

    auto buildScriptIndex(std::vector<PropertyRange> const& scripts) -> std::map<std::string, uint8_t>
    {
        std::map<std::string, uint8_t> result;
        result["Invalid"] = 0;
        result["Unknown"] = 1;
        result["Common"] = 2;
        std::set<std::string> scriptSet;
        for (auto const& r: scripts)
            scriptSet.insert(r.property);
        uint8_t idx = 3;
        for (auto const& s: scriptSet)
            if (s != "Common")
                result[s] = idx++;
        return result;
    }

    auto buildGeneralCategoryIndex(std::vector<PropertyRange> const& gcats) -> std::map<std::string, uint8_t>
    {
        std::map<std::string, uint8_t> result;
        result["Unspecified"] = 0;
        std::set<std::string> cats;
        for (auto const& r: gcats)
            cats.insert(r.property);
        uint8_t idx = 1;
        for (auto const& c: cats)
            result[c] = idx++;
        return result;
    }

    // ---- Derived property computation ----

    /// Compute char_width from EAW and zero-width GC set.
    /// eawWide and eawFullwidth are the indices for Wide and Fullwidth in the PVA enum.
    auto computeCharWidth(CodepointRecord const& rec,
                          std::set<uint8_t> const& zeroWidthGCs,
                          uint8_t eawWide,
                          uint8_t eawFullwidth) -> uint8_t
    {
        // Zero-width categories
        if (zeroWidthGCs.count(rec.general_category))
            return 0;
        // Emoji presentation -> width 2
        if (rec.flags & FlagEmojiPresentation)
            return 2;
        // Wide/Fullwidth -> 2, everything else -> 1
        if (rec.east_asian_width == eawWide || rec.east_asian_width == eawFullwidth)
            return 2;
        return 1;
    }

    auto computeEmojiSegmentationCategory(char32_t codepoint, CodepointRecord const& rec, uint8_t gcbRegionalIndicator) -> int8_t
    {
        if (codepoint == 0x20e3)
            return ESC_CombiningEnclosingKeyCap;
        if (codepoint == 0x20e0)
            return ESC_CombiningEnclosingCircleBackslash;
        if (codepoint == 0x200d)
            return ESC_ZWJ;
        if (codepoint == 0xfe0e)
            return ESC_VS15;
        if (codepoint == 0xfe0f)
            return ESC_VS16;
        if (codepoint == 0x1f3f4)
            return ESC_TagBase;
        if ((codepoint >= 0xE0030 && codepoint <= 0xE0039) || (codepoint >= 0xE0061 && codepoint <= 0xE007A))
            return ESC_TagSequence;
        if (codepoint == 0xE007F)
            return ESC_TagTerm;

        if (rec.flags & FlagEmojiModifierBase)
            return ESC_EmojiModifierBase;
        if (rec.flags & FlagEmojiModifier)
            return ESC_EmojiModifier;
        if (rec.grapheme_cluster_break == gcbRegionalIndicator)
            return ESC_RegionalIndicator;
        if (('0' <= codepoint && codepoint <= '9') || codepoint == '#' || codepoint == '*')
            return ESC_KeyCapBase;
        if (rec.flags & FlagEmojiPresentation)
            return ESC_EmojiEmojiPresentation;
        if ((rec.flags & FlagEmoji) && !(rec.flags & FlagEmojiPresentation))
            return ESC_EmojiTextPresentation;
        if (rec.flags & FlagEmoji)
            return ESC_Emoji;

        return ESC_Invalid;
    }

    // ---- Name mappings for output ----

    auto buildScriptNames(std::vector<PropertyRange> const& scripts) -> std::vector<std::string>
    {
        std::vector<std::string> result;
        result.push_back("Invalid");
        result.push_back("Unknown");
        result.push_back("Common");
        std::set<std::string> scriptSet;
        for (auto const& r: scripts)
            scriptSet.insert(r.property);
        for (auto const& s: scriptSet)
            if (s != "Common")
                result.push_back(s);
        return result;
    }

    auto buildGeneralCategoryNames(std::vector<PropertyRange> const& gcats) -> std::vector<std::string>
    {
        std::vector<std::string> result;
        result.push_back("Unspecified");
        std::set<std::string> cats;
        for (auto const& r: gcats)
            cats.insert(r.property);
        for (auto const& c: cats)
            result.push_back(c);
        return result;
    }

    /// Reverse lookup: index -> name for a PVA-based enum.
    /// Prefers the longest matching name to avoid returning abbreviations.
    std::string reverseLookup(std::map<std::string, uint8_t> const& index,
                              uint8_t value,
                              std::string const& defaultName = "Unknown")
    {
        std::string bestName;
        for (auto const& [name, idx]: index)
        {
            if (idx == value && name.size() > bestName.size())
                bestName = name;
        }
        return bestName.empty() ? defaultName : bestName;
    }

    std::string_view escName(int8_t idx)
    {
        switch (idx)
        {
            case -1: return "Invalid";
            case 0: return "Emoji";
            case 1: return "EmojiTextPresentation";
            case 2: return "EmojiEmojiPresentation";
            case 3: return "EmojiModifierBase";
            case 4: return "EmojiModifier";
            case 5: return "EmojiVSBase";
            case 6: return "RegionalIndicator";
            case 7: return "KeyCapBase";
            case 8: return "CombiningEnclosingKeyCap";
            case 9: return "CombiningEnclosingCircleBackslash";
            case 10: return "ZWJ";
            case 11: return "VS15";
            case 12: return "VS16";
            case 13: return "TagBase";
            case 14: return "TagSequence";
            case 15: return "TagTerm";
            default: return "Invalid";
        }
    }

    // ---- Binary string for flags ----
    auto binstr(uint8_t n) -> std::string
    {
        std::string ss;
        ss += "0b";
        for (int i = 7; i >= 0; --i)
            ss += ((n & (1u << i)) != 0) ? '1' : '0';
        return ss;
    }

    template <typename T>
    auto minimumUintType(std::vector<T> const& values) -> std::string_view
    {
        auto const v = static_cast<uint64_t>(*std::max_element(values.begin(), values.end()));
        if (v <= 0xFF)
            return "uint8_t";
        if (v <= 0xFFFF)
            return "uint16_t";
        if (v <= 0xFFFFFFFF)
            return "uint32_t";
        return "uint64_t";
    }

    template <typename T>
    void writeCxxTable(
        std::ostream& header, std::ostream& impl, std::vector<T> const& table, std::string_view name, bool commentOnBlock)
    {
        constexpr auto ColumnCount = 16;
        auto const elementTypeName = minimumUintType(table);

        header << "extern std::array<" << elementTypeName << ", " << table.size() << "> const " << name << ";\n";
        impl << "std::array<" << elementTypeName << ", " << table.size() << "> const " << name << " {";
        for (size_t i = 0; i < table.size(); ++i)
        {
            if (i % ColumnCount == 0)
                impl << "\n    ";
            if (commentOnBlock && i % BLOCK_SIZE == 0)
                impl << "// block number: " << (i / BLOCK_SIZE) << "\n    ";
            impl << std::right << std::setw(4) << unsigned(table[i]) << ',';
        }
        impl << "\n};\n\n";
    }

} // anonymous namespace

void generateMultistageFiles(UcdParser const& parser, std::string const& outputDir, std::string const& namespaceName)
{
    std::cout << "[tablegen]   Building enum indices...\n";

    auto const& pva = parser.propertyValueAliases();
    auto const emptyMap = std::map<std::string, std::string> {};

    auto const findPva = [&](std::string const& key) -> std::map<std::string, std::string> const& {
        if (auto it = pva.find(key); it != pva.end())
            return it->second;
        return emptyMap;
    };

    // Build enum indices matching the generated ucd_enums.h exactly
    auto const scriptIndex = buildScriptIndex(parser.scripts());
    auto const gcIndex = buildGeneralCategoryIndex(parser.generalCategories());
    auto const eawIndex = buildPvaBasedIndex(findPva("East_Asian_Width"));
    auto const ageIndex = buildAgeIndex(findPva("Age"));
    auto const gcbIndex = buildPvaBasedIndex(findPva("Grapheme_Cluster_Break"), "Undefined");
    auto const incbIndex = buildPvaBasedIndex(findPva("Indic_Conjunct_Break"));
    auto const wbIndex = buildPvaBasedIndex(findPva("Word_Break"));

    // Name vectors for output
    auto const scriptNames = buildScriptNames(parser.scripts());
    auto const gcNames = buildGeneralCategoryNames(parser.generalCategories());
    auto const ageNames = buildAgeEnumMembers(findPva("Age"));

    // Find GCB Regional_Indicator index
    uint8_t gcbRegionalIndicator = 0;
    if (auto it = gcbIndex.find("Regional_Indicator"); it != gcbIndex.end())
        gcbRegionalIndicator = it->second;

    // Find EAW indices for Wide and Fullwidth
    uint8_t eawWide = 0, eawFullwidth = 0;
    if (auto it = eawIndex.find("Wide"); it != eawIndex.end())
        eawWide = it->second;
    if (auto it = eawIndex.find("Fullwidth"); it != eawIndex.end())
        eawFullwidth = it->second;

    // Find zero-width general categories
    std::set<uint8_t> zeroWidthGCs;
    for (auto const& name: { "Control",
                             "Enclosing_Mark",
                             "Format",
                             "Line_Separator",
                             "Nonspacing_Mark",
                             "Paragraph_Separator",
                             "Spacing_Mark",
                             "Surrogate" })
    {
        if (auto it = gcIndex.find(name); it != gcIndex.end())
            zeroWidthGCs.insert(it->second);
    }

    // ---- Populate codepoint records ----
    std::cout << "[tablegen]   Populating " << CODEPOINT_COUNT << " codepoint records...\n";

    std::vector<CodepointRecord> records(CODEPOINT_COUNT);
    std::vector<std::string> names(CODEPOINT_COUNT);

    // Set defaults matching codepoint_properties defaults
    {
        auto scriptUnknown = scriptIndex.count("Unknown") ? scriptIndex.at("Unknown") : uint8_t(1);
        auto gcbOther = gcbIndex.count("Other") ? gcbIndex.at("Other") : uint8_t(0);
        auto eawNarrow = eawIndex.count("Narrow") ? eawIndex.at("Narrow") : uint8_t(0);
        // General_Category::Unassigned is a member name in the GC enum
        auto gcUnassigned = gcIndex.count("Unassigned") ? gcIndex.at("Unassigned") : uint8_t(0);
        auto incbNone = incbIndex.count("None") ? incbIndex.at("None") : uint8_t(3);
        auto wbOther = wbIndex.count("Other") ? wbIndex.at("Other") : uint8_t(18);
        for (auto& rec: records)
        {
            rec.script = scriptUnknown;
            rec.grapheme_cluster_break = gcbOther;
            rec.east_asian_width = eawNarrow;
            rec.general_category = gcUnassigned;
            rec.indic_conjunct_break = incbNone;
            rec.word_break = wbOther;
        }
    }

    // Scripts
    for (auto const& r: parser.scripts())
    {
        auto it = scriptIndex.find(r.property);
        if (it == scriptIndex.end())
            continue;
        for (auto cp = r.first; cp <= r.last; ++cp)
            records[static_cast<size_t>(cp)].script = it->second;
    }

    // DerivedCoreProperties (Grapheme_Extend flag)
    for (auto const& [propName, ranges]: parser.coreProperties())
    {
        if (propName == "Grapheme_Extend")
        {
            for (auto const& r: ranges)
                for (auto cp = r.first; cp <= r.last; ++cp)
                    records[static_cast<size_t>(cp)].flags |= FlagCoreGraphemeExtend;
        }
    }

    // IndicSyllabicCategory (Virama flag)
    //
    // The FULL virama set, deliberately, rather than Indic_Conjunct_Break=Linker. Linker is not
    // restricted to a handful of scripts -- in UCD 18.0 it already covers Khmer, Myanmar, Javanese,
    // Chakma and Tai Tham -- but it is a smaller set: 20 codepoints against 41. The 21 it leaves out
    // are viramas that do not form InCB conjuncts, among them Gurmukhi U+0A4D, Tamil U+0BCD,
    // Kannada U+0CCD, Sinhala U+0DCA and the Brahmi-family stackers. They still stack a consonant
    // onto the previous one, so they still widen the cluster.
    //
    // Invisible_Stacker is included because it IS a virama for measuring purposes -- Khmer U+17D2,
    // Myanmar U+1039, Chakma U+11133 and Tai Tham U+1A60 stack a consonant onto the previous one
    // exactly as a Virama does, and are only categorised apart because they are never rendered.
    //
    // Together this reproduces wcwidth's _ISC_VIRAMA_SET exactly (41 codepoints).
    for (auto const& r: parser.indicSyllabicCategory())
        if (r.property == "Virama" || r.property == "Invisible_Stacker")
            for (auto cp = r.first; cp <= r.last && cp < CODEPOINT_COUNT; ++cp)
                records[static_cast<size_t>(cp)].flags |= FlagVirama;

    // Emoji variation sequence bases: the codepoints a following VS15/VS16 is defined to re-present.
    for (auto const cp: parser.emojiVariationBases())
        if (cp < CODEPOINT_COUNT)
            records[static_cast<size_t>(cp)].flags2 |= Flag2EmojiVariationBase;

    // DerivedAge
    for (auto const& r: parser.ageRanges())
    {
        auto const& ageAliases = findPva("Age");
        // r.property is the version string directly
        std::string enumName;
        for (auto const& [abbrev, full]: ageAliases)
        {
            if (abbrev == r.property)
            {
                enumName = full;
                break;
            }
        }
        if (enumName.empty())
            continue;
        auto it = ageIndex.find(enumName);
        if (it == ageIndex.end())
            continue;
        for (auto cp = r.first; cp <= r.last; ++cp)
            records[static_cast<size_t>(cp)].age = it->second;
    }

    // General Category (uses abbreviations in the property field from DerivedGeneralCategory)
    // But our parser already resolved abbreviations to full names via PropertyValueAliases
    for (auto const& r: parser.generalCategories())
    {
        auto it = gcIndex.find(r.property);
        if (it == gcIndex.end())
            continue;
        for (auto cp = r.first; cp <= r.last; ++cp)
            records[static_cast<size_t>(cp)].general_category = it->second;
    }

    // Names
    for (auto const& [cp, name]: parser.derivedNames())
        names[static_cast<size_t>(cp)] = name;

    // Grapheme Cluster Break
    for (auto const& [propName, ranges]: parser.graphemeBreakProps())
    {
        for (auto const& r: ranges)
        {
            auto it = gcbIndex.find(r.property);
            if (it == gcbIndex.end())
                continue;
            for (auto cp = r.first; cp <= r.last; ++cp)
                records[static_cast<size_t>(cp)].grapheme_cluster_break = it->second;
        }
    }

    // Indic Conjunct Break
    for (auto const& [propValue, ranges]: parser.indicConjunctBreak())
    {
        auto it = incbIndex.find(propValue);
        if (it == incbIndex.end())
            continue;
        for (auto const& r: ranges)
            for (auto cp = r.first; cp <= r.last; ++cp)
                records[static_cast<size_t>(cp)].indic_conjunct_break = it->second;
    }

    // Word Break
    for (auto const& [propName, ranges]: parser.wordBreakProps())
    {
        for (auto const& r: ranges)
        {
            auto it = wbIndex.find(r.property);
            if (it == wbIndex.end())
                continue;
            for (auto cp = r.first; cp <= r.last; ++cp)
                records[static_cast<size_t>(cp)].word_break = it->second;
        }
    }

    // East Asian Width
    for (auto const& r: parser.eastAsianWidths())
    {
        auto it = eawIndex.find(r.property);
        if (it == eawIndex.end())
            continue;
        for (auto cp = r.first; cp <= r.last; ++cp)
            records[static_cast<size_t>(cp)].east_asian_width = it->second;
    }

    // Emoji properties
    for (auto const& [propName, ranges]: parser.emojiProps())
    {
        uint8_t flag = 0;
        if (propName == "Emoji")
            flag = FlagEmoji;
        else if (propName == "Emoji_Component")
            flag = FlagEmojiComponent;
        else if (propName == "Emoji_Modifier")
            flag = FlagEmojiModifier;
        else if (propName == "Emoji_Modifier_Base")
            flag = FlagEmojiModifierBase;
        else if (propName == "Emoji_Presentation")
            flag = FlagEmojiPresentation;
        else if (propName == "Extended_Pictographic")
            flag = FlagExtendedPictographic;
        else
            continue;

        for (auto const& r: ranges)
            for (auto cp = r.first; cp <= r.last; ++cp)
                records[static_cast<size_t>(cp)].flags |= flag;
    }

    // Derived: EmojiSegmentationCategory
    std::cout << "[tablegen]   Computing derived properties...\n";
    for (char32_t cp = 0; cp < CODEPOINT_COUNT; ++cp)
        records[static_cast<size_t>(cp)].emoji_segmentation_category =
            computeEmojiSegmentationCategory(cp, records[static_cast<size_t>(cp)], gcbRegionalIndicator);

    // Derived: char_width
    for (char32_t cp = 0; cp < CODEPOINT_COUNT; ++cp)
        records[static_cast<size_t>(cp)].char_width =
            computeCharWidth(records[static_cast<size_t>(cp)], zeroWidthGCs, eawWide, eawFullwidth);

    // A format character that is actually RENDERED takes a column.
    //
    // General_Category=Cf is zero-width as a rule, and for the invisible ones -- ZWJ, ZWNJ, the
    // bidi marks, BOM -- that is right. But a handful of Cf codepoints do draw something: the Arabic
    // prepended concatenation marks (U+0600..U+0605, U+06DD, U+0890, U+0891, U+08E2), U+070F, and
    // the Kaithi number signs (U+110BD, U+110CD). What they have in common is Grapheme_Cluster_Break
    // =Prepend: they attach to and are drawn before the cluster that follows.
    //
    // Default_Ignorable_Code_Point looks like it separates the same two groups but is 19 codepoints
    // too generous: the interlinear annotation marks U+FFF9..U+FFFB and the Egyptian hieroglyph
    // format controls U+13430..U+1343F are Cf, are not Default_Ignorable, and still render nothing.
    // Python wcwidth measures those as zero. Cf AND Prepend reproduces wcwidth's set exactly.
    {
        auto prepend = std::vector<bool>(CODEPOINT_COUNT, false);
        for (auto const& [_, ranges]: parser.graphemeBreakProps())
            for (auto const& r: ranges)
                if (r.property == "Prepend")
                    for (auto cp = r.first; cp <= r.last && cp < CODEPOINT_COUNT; ++cp)
                        prepend[static_cast<size_t>(cp)] = true;

        if (auto const it = gcIndex.find("Format"); it != gcIndex.end())
        {
            auto const gcFormat = it->second;
            for (char32_t cp = 0; cp < CODEPOINT_COUNT; ++cp)
                if (records[static_cast<size_t>(cp)].general_category == gcFormat && prepend[static_cast<size_t>(cp)])
                    records[static_cast<size_t>(cp)].char_width = 1;
        }

        // U+00AD SOFT HYPHEN is Cf but not Prepend, so the rule above misses it, yet a terminal has
        // no line breaking to hide it with and draws it as a hyphen. wcwidth measures it as one
        // column for the same reason; Windows Terminal carves it out likewise.
        records[0x00AD].char_width = 1;
    }

    // Conjoining Hangul V/T Jamo must be width 0 so that decomposed syllables
    // (L + V + T) sum to the same width as their precomposed forms (issue #32).
    // Only V and T need override; L, LV, LVT already get width 2 from EAW=Wide.
    for (auto const& r: parser.hangulSyllableType())
    {
        if (r.property == "V" || r.property == "T")
            for (auto cp = r.first; cp <= r.last; ++cp)
                records[static_cast<size_t>(cp)].char_width = 0;
    }

    // ---- Generate multistage tables ----
    std::cout << "[tablegen]   Generating multistage tables (properties)...\n";

    // FNV-1a hasher for CodepointRecord (trivially copyable, 8 bytes)
    struct RecordHasher
    {
        size_t operator()(CodepointRecord const& r) const noexcept
        {
            size_t hash = 14695981039346656037ULL;
            auto const* bytes = reinterpret_cast<char const*>(&r);
            for (size_t i = 0; i < sizeof(r); ++i)
            {
                hash ^= static_cast<size_t>(static_cast<unsigned char>(bytes[i]));
                hash *= 1099511628211ULL;
            }
            return hash;
        }
    };

    using PropsTable = support::multistage_table<CodepointRecord, uint32_t, uint8_t, uint16_t, BLOCK_SIZE, CODEPOINT_COUNT - 1>;
    PropsTable propsTable {};
    support::generate(records.data(), records.size(), propsTable, RecordHasher {});

    std::cout << "[tablegen]   Generating multistage tables (names)...\n";

    using NamesTable = support::multistage_table<std::string, uint32_t, uint8_t, uint16_t, BLOCK_SIZE, CODEPOINT_COUNT - 1>;
    NamesTable namesTable {};
    support::generate(names.data(), names.size(), namesTable, std::hash<std::string> {});

    // ---- Write output files ----
    std::cout << "[tablegen]   Writing output files...\n";

    auto const disclaimer = std::string("// This file was auto-generated by unicode_tablegen.\n");

    auto headerPath = outputDir + "/codepoint_properties_data.h";
    auto implPath = outputDir + "/codepoint_properties_data.cpp";
    auto namesPath = outputDir + "/codepoint_properties_names.cpp";

    auto header = std::ofstream(headerPath);
    auto impl = std::ofstream(implPath);
    auto namesFile = std::ofstream(namesPath);

    // Header
    header << disclaimer;
    header << "#pragma once\n\n";
    header << "#include <libunicode/codepoint_properties.h>\n\n";
    header << "#include <array>\n";
    header << "#include <cstdint>\n\n";
    header << "namespace " << namespaceName << "\n{\n\n";

    // Implementation
    impl << disclaimer;
    impl << "#include <libunicode/codepoint_properties.h>\n";
    impl << "#include <libunicode/codepoint_properties_data.h>\n";
    impl << "#include <libunicode/emoji_segmenter.h>\n";
    impl << "#include <libunicode/ucd_enums.h>\n\n";
    impl << "#include <array>\n";
    impl << "#include <cstdint>\n";
    impl << "#include <string_view>\n\n";
    impl << "using namespace unicode;\n\n";
    impl << "namespace " << namespaceName << "\n{\n\n";

    // Stage 1 & 2
    writeCxxTable(header, impl, propsTable.stage1, "stage1", false);
    writeCxxTable(header, impl, propsTable.stage2, "stage2", true);

    // Properties table (stage 3)
    header << "extern std::array<codepoint_properties, " << propsTable.stage3.size() << "> const properties;\n";
    impl << "std::array<codepoint_properties, " << propsTable.stage3.size() << "> const properties{{\n";
    for (auto const& rec: propsTable.stage3)
    {
        impl << "    {" << static_cast<unsigned>(rec.char_width) << ", " << (!rec.flags ? "0" : binstr(rec.flags)) << ", "
             << (!rec.flags2 ? "0" : binstr(rec.flags2)) << ", "
             << "Script::" << (rec.script < scriptNames.size() ? scriptNames[rec.script] : "Unknown") << ", "
             << "Grapheme_Cluster_Break::" << reverseLookup(gcbIndex, rec.grapheme_cluster_break, "Other") << ", "
             << "East_Asian_Width::" << reverseLookup(eawIndex, rec.east_asian_width, "Narrow") << ", "
             << "General_Category::" << (rec.general_category < gcNames.size() ? gcNames[rec.general_category] : "Unspecified")
             << ", "
             << "EmojiSegmentationCategory::" << escName(rec.emoji_segmentation_category) << ", "
             << "Age::" << (rec.age < ageNames.size() ? ageNames[rec.age] : "Unassigned") << ", "
             << "Indic_Conjunct_Break::" << reverseLookup(incbIndex, rec.indic_conjunct_break, "None") << ", "
             << "Word_Break::" << reverseLookup(wbIndex, rec.word_break, "Other") << "},\n";
    }
    impl << "}};\n\n";

    impl << "} // end namespace " << namespaceName << "\n";

    // Names file
    namesFile << disclaimer;
    namesFile << "#include <libunicode/codepoint_properties_data.h>\n\n";
    namesFile << "#include <array>\n";
    namesFile << "#include <string_view>\n";
    namesFile << "#include <cstdint>\n\n";
    namesFile << "using namespace unicode;\n";
    namesFile << "using namespace std::string_view_literals;\n\n";
    namesFile << "namespace " << namespaceName << "\n{\n\n";

    writeCxxTable(header, namesFile, namesTable.stage1, "names_stage1", false);
    writeCxxTable(header, namesFile, namesTable.stage2, "names_stage2", true);

    // Names stage 3
    header << "extern std::array<std::string_view, " << namesTable.stage3.size() << "> const names_stage3;\n";
    namesFile << "std::array<std::string_view, " << namesTable.stage3.size() << "> const names_stage3{{\n";
    for (auto const& name: namesTable.stage3)
        namesFile << "    \"" << name << "\"sv,\n";
    namesFile << "}};\n\n";

    namesFile << "} // end namespace " << namespaceName << "\n";

    header << "\n} // end namespace " << namespaceName << "\n";
}

} // namespace tablegen
