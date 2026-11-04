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
#include "ucd_api_generator.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <map>
#include <set>
#include <string>

#include "enum_utils.h"
#include "ucd_parser.h"

namespace tablegen
{

namespace
{
    constexpr auto FOLD_OPEN = "{{{";
    constexpr auto FOLD_CLOSE = "}}}";

    /// Represents a merged range (consecutive ranges with the same value collapsed).
    struct MergedRange
    {
        char32_t rangeFrom {};
        char32_t rangeTo {};
        std::string value;
        std::vector<std::string> comments;
        int count {};
    };

    auto mergeRanges(std::vector<PropertyRange> const& table) -> std::vector<MergedRange>
    {
        std::vector<MergedRange> result;
        MergedRange current;
        current.count = 0;

        for (auto const& r: table)
        {
            if (current.count > 0 && current.rangeTo + 1 == r.first && current.value == r.property)
            {
                // Adjacent range with same value
                if (current.comments.size() == 1)
                    current.comments[0] = std::format("0x{:04X} .. 0x{:04X}: {}",
                                                      static_cast<unsigned>(current.rangeFrom),
                                                      static_cast<unsigned>(current.rangeTo),
                                                      current.comments[0]);
                if (!current.comments.empty())
                {
                    current.comments.push_back(std::format(
                        "0x{:04X} .. 0x{:04X}: {}", static_cast<unsigned>(r.first), static_cast<unsigned>(r.last), r.comment));
                }
                current.rangeTo = r.last;
            }
            else
            {
                if (current.count > 0)
                    result.push_back(current);
                current.rangeFrom = r.first;
                current.rangeTo = r.last;
                current.value = r.property;
                current.comments = { r.comment };
                current.count = 0;
            }
            current.count++;
        }
        if (current.count > 0)
            result.push_back(current);
        return result;
    }

    /// East Asian Width abbreviation -> enum member name
    auto widthName(std::string const& abbrev) -> std::string
    {
        if (abbrev == "A")
            return "Ambiguous";
        if (abbrev == "F")
            return "FullWidth";
        if (abbrev == "H")
            return "HalfWidth";
        if (abbrev == "N")
            return "Neutral";
        if (abbrev == "Na")
            return "Narrow";
        if (abbrev == "W")
            return "Wide";
        return "Unspecified";
    }

} // anonymous namespace

void generateUcdApiFiles(UcdParser const& parser, std::string const& outputDir)
{
    auto header = std::ofstream(outputDir + "/ucd.h", std::ios::binary);
    auto impl = std::ofstream(outputDir + "/ucd.cpp", std::ios::binary);

    // ---- File headers ----
    header << licenseHeader;
    header << R"(#pragma once

#include <libunicode/ucd_enums.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace unicode
{

)";

    impl << licenseHeader;
    impl << R"(
#include <libunicode/ucd.h>
#include <libunicode/ucd_private.h>

#include <algorithm>
#include <array>
#include <span>
#include <string_view>

namespace unicode
{

)";

    // ---- Planes ----
    {
        auto const& planes = parser.planes();
        auto elementType = std::string("Prop<::unicode::Plane>");
        impl << "namespace tables\n{\n";
        impl << "    // clang-format off\n";
        impl << std::format("    auto static const Plane = std::array<{}, {}>{{ // {}\n", elementType, planes.size(), FOLD_OPEN);
        for (auto const& p: planes)
        {
            impl << std::format("        {} {{ {{ 0x{:04X}, 0x{:04X} }}, unicode::Plane::{} }},",
                                elementType,
                                static_cast<unsigned>(p.start),
                                static_cast<unsigned>(p.end),
                                sanitizeIdentifier(p.name));
            if (p.shortName.empty())
                impl << std::format(" // Plane {}\n", p.number);
            else
                impl << std::format(" // Plane {} {}\n", p.number, p.shortName);
        }
        impl << std::format("    }}; // {}\n", FOLD_CLOSE);
        impl << "    // clang-format off\n";
        impl << "} // namespace tables\n\n";

        header << "Plane plane(char32_t codepoint) noexcept;\n\n";
        impl << "Plane plane(char32_t codepoint) noexcept {\n";
        impl << "    return search(tables::Plane, codepoint).value_or(Plane::Unassigned);\n";
        impl << "}\n\n";
    }

    // ---- Core Properties ----
    {
        auto const& props = parser.coreProperties();
        impl << "namespace tables {\n";
        for (auto const& [name, ranges]: props)
        {
            impl << std::format(
                "auto static const {} = std::array<{}, {}>{{ // {}\n", name, "Interval", ranges.size(), FOLD_OPEN);
            for (auto const& r: ranges)
                impl << std::format("    Interval{{ 0x{:04X}, 0x{:04X} }}, // {}\n",
                                    static_cast<unsigned>(r.first),
                                    static_cast<unsigned>(r.last),
                                    r.comment);
            impl << std::format("}}; // {}\n", FOLD_CLOSE);
        }
        impl << "} // end namespace tables\n\n";

        impl << "bool contains(Core_Property prop, char32_t codepoint) noexcept {\n";
        impl << "    switch (prop)\n";
        impl << "    {\n";
        for (auto const& [name, ranges]: props)
            impl << std::format("        case Core_Property::{0:}: return contains(tables::{0:}, codepoint);\n", name);
        impl << "    }\n";
        impl << "    return false;\n";
        impl << "}\n\n";

        header << "bool contains(Core_Property prop, char32_t codepoint) noexcept;\n\n";
    }

    // ---- General Categories ----
    {
        auto const& gcats = parser.generalCategories();
        auto const& catsMap = parser.generalCategoryMap();
        auto typeName = std::string("General_Category");
        auto fqdnTypeName = "::unicode::General_Category";
        auto elementType = std::string("Prop<") + fqdnTypeName + ">";

        // Main lookup table
        impl << "namespace tables {\n";
        impl << std::format("auto const {} = std::array<{}, {}>{{\n", typeName, elementType, gcats.size());
        for (auto const& cat: gcats)
        {
            impl << std::format("    {} {{ {{ 0x{:04X}, 0x{:04X} }}, {}::{} }}, // {}\n",
                                elementType,
                                static_cast<unsigned>(cat.first),
                                static_cast<unsigned>(cat.last),
                                fqdnTypeName,
                                cat.property,
                                cat.comment);
        }
        impl << "};\n";
        impl << "} // end namespace tables\n\n";

        // Getter
        impl << std::format("namespace {}\n", "general_category");
        impl << "{\n";
        impl << std::format("    {} get(char32_t value) noexcept {{\n", typeName);
        impl << std::format("        return search(tables::{}, value).value_or({}::Unspecified);\n", typeName, typeName);
        impl << "    }\n";
        impl << "}\n\n";

        // Per-category interval tables
        impl << "namespace tables {\n";
        for (auto const& [name, ranges]: catsMap)
        {
            impl << std::format(
                "auto static const {} = std::array<{}, {}>{{ // {}\n", name, "Interval", ranges.size(), FOLD_OPEN);
            for (auto const& r: ranges)
                impl << std::format("    Interval{{ 0x{:04X}, 0x{:04X} }}, // {}\n",
                                    static_cast<unsigned>(r.first),
                                    static_cast<unsigned>(r.last),
                                    r.comment);
            impl << std::format("}}; // {}\n", FOLD_CLOSE);
        }
        impl << "} // end namespace tables\n\n";

        // contains(General_Category, char32_t)
        impl << "bool contains(General_Category generalCategory, char32_t codepoint) noexcept {\n";
        impl << "    switch (generalCategory)\n";
        impl << "    {\n";
        for (auto const& [name, ranges]: catsMap)
            impl << std::format("        case General_Category::{0:}: return contains(tables::{0:}, codepoint);\n", name);
        impl << "        case General_Category::Unspecified: return false;\n";
        impl << "    }\n";
        impl << "    return false;\n";
        impl << "}\n\n";

        header << "bool contains(General_Category generalCategory, char32_t codepoint) noexcept;\n\n";

        // Inline helpers
        header << "// Disabling clang-format to avoid single-line folding implementations.\n";
        header << "// clang-format off\n";
        header << "namespace general_category\n";
        header << "{\n";
        header << std::format("    {} get(char32_t value) noexcept;\n\n", typeName);
        for (auto const& [name, ranges]: catsMap)
        {
            header << std::format(
                "    inline bool is_{}(char32_t codepoint) noexcept\n"
                "    {{\n"
                "        return contains(General_Category::{}, codepoint);\n"
                "    }}\n\n",
                [&name = name] {
                    auto lower = name;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    return lower;
                }(),
                name);
        }
        header << "} // namespace general_category\n";
        header << "// clang-format on\n";
        header << "\n";
    }

    // ---- Scripts ----
    {
        auto const& scripts = parser.scripts();
        auto elementType = std::string("Prop<unicode::Script>");

        impl << "namespace tables {\n";
        impl << std::format("auto static const Script = std::array<{}, {}>{{ // {}\n", elementType, scripts.size(), FOLD_OPEN);
        for (auto const& r: scripts)
        {
            impl << std::format("    {} {{ {{ 0x{:04X}, 0x{:04X} }}, unicode::Script::{} }}, // {}\n",
                                elementType,
                                static_cast<unsigned>(r.first),
                                static_cast<unsigned>(r.last),
                                r.property,
                                r.comment);
        }
        impl << std::format("}}; // {}\n", FOLD_CLOSE);
        impl << "} // end namespace tables\n\n";

        header << "Script script(char32_t codepoint) noexcept;\n\n";
        impl << "Script script(char32_t codepoint) noexcept {\n";
        impl << "    return search(tables::Script, codepoint).value_or(Script::Unknown);\n";
        impl << "}\n\n";
    }

    // ---- Script Extensions ----
    {
        auto const& sces = parser.scriptExtensions();
        auto const& pva = parser.propertyValueAliases().at("Script");

        impl << std::format("namespace tables {{ // {} ScriptExtensions\n", FOLD_OPEN);

        // Indirected lists
        std::vector<std::string> doneList;
        for (auto const& sce: sces)
        {
            auto key = std::string("sce");
            for (auto const& s: sce.properties)
                key += "_" + s;
            if (std::find(doneList.begin(), doneList.end(), key) != doneList.end())
                continue;
            doneList.push_back(key);

            impl << std::format("auto static const {} = std::array<{}, {}>{{\n", key, "unicode::Script", sce.properties.size());
            for (auto const& scriptAbbrev: sce.properties)
            {
                auto it = pva.find(scriptAbbrev);
                auto fullName = (it != pva.end()) ? it->second : scriptAbbrev;
                impl << std::format("    unicode::Script::{},\n", fullName);
            }
            impl << "};\n\n";
        }

        // Main lookup table
        auto elementType = std::string("Prop<std::pair<unicode::Script const*, std::size_t>>");
        impl << std::format("static const std::array<{}, {}> {} {{ {{\n", elementType, sces.size(), "sce");
        for (auto const& sce: sces)
        {
            auto key = std::string("sce");
            for (auto const& s: sce.properties)
                key += "_" + s;
            impl << std::format("    {} {{ {{ 0x{:04X}, 0x{:04X} }}, {{ {}.data(), {}.size() }} }}, // {}\n",
                                elementType,
                                static_cast<unsigned>(sce.first),
                                static_cast<unsigned>(sce.last),
                                key,
                                key,
                                sce.comment);
        }
        impl << "} };\n";
        impl << std::format("}} // {}\n\n", FOLD_CLOSE);

        // Getter
        header << "std::optional<std::span<Script const>> script_extensions(char32_t codepoint) noexcept;\n\n";
        impl << "std::optional<std::span<Script const>> script_extensions(char32_t codepoint) noexcept {\n";
        impl << "    auto const p = search(tables::sce, codepoint);\n";
        impl << "    if (!p.has_value())\n";
        impl << "        return std::nullopt;\n";
        impl << "    return std::span<Script const>{ p->first, p->second };\n";
        impl << "}\n\n";
    }

    // ---- Blocks ----
    {
        auto const& blocks = parser.blocks();
        auto elementType = std::string("Prop<::unicode::Block>");

        impl << "namespace tables {\n";
        impl << std::format("auto static const Block = std::array<{}, {}>{{ // {}\n", elementType, blocks.size(), FOLD_OPEN);
        for (auto const& b: blocks)
        {
            impl << std::format("    {} {{ {{ 0x{:04X}, 0x{:04X} }}, {}::{} }},\n",
                                elementType,
                                static_cast<unsigned>(b.first),
                                static_cast<unsigned>(b.last),
                                "::unicode::Block",
                                sanitizeIdentifier(b.title));
        }
        impl << "};\n";
        impl << "} // end namespace tables {}\n\n";

        header << "Block block(char32_t codepoint) noexcept;\n\n";
        impl << "Block block(char32_t codepoint) noexcept {\n";
        impl << "    return search(tables::Block, codepoint).value_or(::unicode::Block::Unspecified);\n";
        impl << "}\n\n";
    }

    // ---- Grapheme Break Properties ----
    {
        auto const& props = parser.graphemeBreakProps();
        impl << "namespace tables {\n";
        for (auto const& [name, ranges]: props)
        {
            auto elementType = std::format("Prop<::unicode::{}>", name);
            impl << std::format(
                "auto static const {} = std::array<{}, {}>{{ // {}\n", name, elementType, ranges.size(), FOLD_OPEN);
            for (auto const& r: ranges)
            {
                impl << std::format("    {} {{ {{ 0x{:04X}, 0x{:04X} }}, ::unicode::{}::{} }}, // {}\n",
                                    elementType,
                                    static_cast<unsigned>(r.first),
                                    static_cast<unsigned>(r.last),
                                    name,
                                    r.property,
                                    r.comment);
            }
            impl << std::format("}}; // {}\n", FOLD_CLOSE);
        }
        impl << "} // end namespace tables\n\n";

        for (auto const& [name, ranges]: props)
        {
            header << std::format("{} {}(char32_t codepoint) noexcept;\n", name, [&name = name] {
                auto lower = name;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                return lower;
            }());
            impl << std::format("{} {}(char32_t codepoint) noexcept {{\n", name, [&name = name] {
                auto lower = name;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                return lower;
            }());
            impl << std::format("    return search(tables::{}, codepoint).value_or({}::Undefined);\n", name, name);
            impl << "}\n\n";
        }
        header << "\n";
        impl << "\n";
    }

    // ---- East Asian Width ----
    {
        auto const& rawWidths = parser.eastAsianWidths();
        auto compactRanges = mergeRanges(rawWidths);
        auto propType = std::string("::unicode::EastAsianWidth");
        auto elementType = std::string("Prop<") + propType + ">";

        // Header: to_string inline
        header << "inline std::string to_string(EastAsianWidth value)\n";
        header << "{\n";
        header << "    switch (value)\n";
        header << "    {\n";
        for (auto const& v: { "Ambiguous", "FullWidth", "HalfWidth", "Neutral", "Narrow", "Wide", "Unspecified" })
            header << std::format("        case EastAsianWidth::{}: return \"{}\";\n", v, v);
        header << "    }\n";
        header << "    return \"Unknown\";\n";
        header << "}\n\n";

        header << "EastAsianWidth east_asian_width(char32_t codepoint) noexcept;\n\n";

        // Impl: range tables
        impl << "namespace tables {\n";
        impl << std::format(
            "auto static const EastAsianWidth = std::array<{}, {}>{{ // {}\n", elementType, compactRanges.size(), FOLD_OPEN);
        for (auto const& range: compactRanges)
        {
            if (range.comments.size() > 1)
                for (auto const& comment: range.comments)
                    impl << std::format("    // {}\n", comment);
            impl << std::format("    {} {{ {{ 0x{:04X}, 0x{:04X} }}, {}::{} }},",
                                elementType,
                                static_cast<unsigned>(range.rangeFrom),
                                static_cast<unsigned>(range.rangeTo),
                                propType,
                                widthName(range.value));
            if (range.count == 1 && range.comments.size() == 1)
                impl << std::format(" // {}", range.comments[0]);
            else if (range.count > 1)
                impl << std::format(" // #{}", range.count);
            impl << "\n";
        }
        impl << std::format("}}; // {}\n", FOLD_CLOSE);
        impl << "} // end namespace tables\n\n";

        impl << "EastAsianWidth east_asian_width(char32_t codepoint) noexcept {\n";
        impl << "    return search(tables::EastAsianWidth, codepoint).value_or(EastAsianWidth::Unspecified);\n";
        impl << "}\n\n";
    }

    // ---- Emoji Properties ----
    {
        auto const& emojiProps = parser.emojiProps();

        impl << "namespace tables {\n";
        for (auto const& [name, ranges]: emojiProps)
        {
            impl << std::format(
                "auto static const {} = std::array<{}, {}>{{ // {}\n", name, "Interval", ranges.size(), FOLD_OPEN);
            for (auto const& r: ranges)
                impl << std::format("    Interval{{ 0x{:04X}, 0x{:04X} }}, // {}\n",
                                    static_cast<unsigned>(r.first),
                                    static_cast<unsigned>(r.last),
                                    r.comment);
            impl << std::format("}}; // {}\n", FOLD_CLOSE);
        }
        impl << "} // end namespace tables\n\n";

        for (auto const& [name, ranges]: emojiProps)
        {
            auto lowerName = name;
            std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
            impl << std::format("bool is_{}(char32_t codepoint) noexcept {{\n", lowerName);
            impl << std::format("    return contains(tables::{}, codepoint);\n", name);
            impl << "}\n\n";
        }

        for (auto const& [name, ranges]: emojiProps)
        {
            auto lowerName = name;
            std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
            header << std::format("bool is_{}(char32_t codepoint) noexcept;\n", lowerName);
        }
        header << "\n";
    }

    // ---- Bidi Mirroring ----
    {
        auto const& intervals = parser.bidiMirroredIntervals();
        auto const& pairs = parser.bidiMirroringGlyphPairs();

        impl << "namespace tables {\n";

        // Bidi_Mirrored interval table
        impl << std::format("auto static const Bidi_Mirrored = std::array<Interval, {}>{{ // {}\n", intervals.size(), FOLD_OPEN);
        for (auto const& [start, end]: intervals)
            impl << std::format(
                "    Interval{{ 0x{:04X}, 0x{:04X} }},\n", static_cast<unsigned>(start), static_cast<unsigned>(end));
        impl << std::format("}}; // {}\n", FOLD_CLOSE);

        // Bidi_Mirroring_Glyph pair table
        impl << std::format("auto static const Bidi_Mirroring_Glyph = std::array<std::pair<char32_t, char32_t>, {}>{{ // {}\n",
                            pairs.size(),
                            FOLD_OPEN);
        for (auto const& [source, target]: pairs)
            impl << std::format("    std::pair<char32_t, char32_t>{{ 0x{:04X}, 0x{:04X} }},\n",
                                static_cast<unsigned>(source),
                                static_cast<unsigned>(target));
        impl << std::format("}}; // {}\n", FOLD_CLOSE);

        impl << "} // end namespace tables\n\n";

        // Functions
        impl << "bool is_mirrored(char32_t codepoint) noexcept {\n";
        impl << "    return contains(tables::Bidi_Mirrored, codepoint);\n";
        impl << "}\n\n";

        impl << "char32_t bidi_mirroring_glyph(char32_t codepoint) noexcept {\n";
        impl << "    auto const it = std::lower_bound(\n";
        impl << "        tables::Bidi_Mirroring_Glyph.begin(),\n";
        impl << "        tables::Bidi_Mirroring_Glyph.end(),\n";
        impl << "        codepoint,\n";
        impl << "        [](auto const& pair, char32_t cp) { return pair.first < cp; });\n";
        impl << "    if (it != tables::Bidi_Mirroring_Glyph.end() && it->first == codepoint)\n";
        impl << "        return it->second;\n";
        impl << "    return codepoint;\n";
        impl << "}\n\n";

        header << "/// Returns true if the codepoint has the Bidi_Mirrored property.\n";
        header << "[[nodiscard]] bool is_mirrored(char32_t codepoint) noexcept;\n\n";
        header << "/// Returns the Bidi_Mirroring_Glyph for a codepoint.\n";
        header << "/// If no mirroring glyph exists, returns the input codepoint unchanged.\n";
        header << "[[nodiscard]] char32_t bidi_mirroring_glyph(char32_t codepoint) noexcept;\n\n";
    }

    // ---- Age (make_age: version string -> Age enum) ----
    {
        auto const& pva = parser.propertyValueAliases();
        auto const it = pva.find("Age");
        if (it != pva.end())
        {
            auto const& ageAliases = it->second;

            // Collect abbreviation -> full name pairs, excluding "NA" (Unassigned)
            struct AgeEntry
            {
                std::string abbrev;
                std::string enumName;
            };
            std::vector<AgeEntry> entries;
            for (auto const& [abbrev, full]: ageAliases)
            {
                if (full == "Unassigned")
                    continue;
                entries.push_back({ abbrev, full });
            }
            std::sort(entries.begin(), entries.end(), [](auto const& a, auto const& b) { return a.abbrev < b.abbrev; });

            // Header declaration
            header << "/// Parses a Unicode version string (e.g., \"1.1\", \"18.0\") to an Age enum value.\n";
            header << "/// @param value The version string from DerivedAge.txt.\n";
            header << "/// @return The corresponding Age enum value, or std::nullopt if not recognized.\n";
            header << "[[nodiscard]] std::optional<Age> make_age(std::string_view value) noexcept;\n\n";

            // Implementation
            impl << "std::optional<Age> make_age(std::string_view value) noexcept {\n";
            impl << "    using namespace std::string_view_literals;\n";
            impl << "    // clang-format off\n";
            impl << std::format("    static constexpr auto mappings = std::array<std::pair<std::string_view, Age>, {}>{{\n",
                                entries.size());
            for (auto const& entry: entries)
                impl << std::format("        std::pair{{ \"{}\"sv, Age::{} }},\n", entry.abbrev, entry.enumName);
            impl << "    };\n";
            impl << "    // clang-format on\n";
            impl << "    for (auto const& [key, val] : mappings)\n";
            impl << "        if (key == value)\n";
            impl << "            return val;\n";
            impl << "    return std::nullopt;\n";
            impl << "}\n\n";
        }
    }

    // ---- File footers ----
    header << "} // namespace unicode\n";
    impl << "} // namespace unicode\n";
}

} // namespace tablegen
