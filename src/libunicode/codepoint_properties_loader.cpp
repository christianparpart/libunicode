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
#include <libunicode/codepoint_properties_loader.h>
#include <libunicode/multistage_table_generator.h>
#include <libunicode/scoped_timer.h>
#include <libunicode/ucd_enums.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string_view>
#include <utility>

using namespace std;
using namespace std::string_view_literals;

namespace unicode
{

namespace
{
    // {{{ string-to-enum convert helper
    constexpr optional<unicode::Age> make_age(string_view value) noexcept
    {
        // clang-format off
        auto /*static*/ constexpr mappings = array {
            pair { "1.1"sv, Age::V1_1 },
            pair { "2.0"sv, Age::V2_0 },
            pair { "2.1"sv, Age::V2_1 },
            pair { "3.0"sv, Age::V3_0 },
            pair { "3.1"sv, Age::V3_1 },
            pair { "3.2"sv, Age::V3_2 },
            pair { "4.0"sv, Age::V4_0 },
            pair { "4.1"sv, Age::V4_1 },
            pair { "5.0"sv, Age::V5_0 },
            pair { "5.1"sv, Age::V5_1 },
            pair { "5.2"sv, Age::V5_2 },
            pair { "6.0"sv, Age::V6_0 },
            pair { "6.1"sv, Age::V6_1 },
            pair { "6.2"sv, Age::V6_2 },
            pair { "6.3"sv, Age::V6_3 },
            pair { "7.0"sv, Age::V7_0 },
            pair { "8.0"sv, Age::V8_0 },
            pair { "9.0"sv, Age::V9_0 },
            pair { "10.0"sv, Age::V10_0 },
            pair { "11.0"sv, Age::V11_0 },
            pair { "12.0"sv, Age::V12_0 },
            pair { "12.1"sv, Age::V12_1 },
            pair { "13.0"sv, Age::V13_0 },
            pair { "14.0"sv, Age::V14_0 },
            pair { "15.0"sv, Age::V15_0 },
            pair { "16.0"sv, Age::V16_0 },
            pair { "17.0"sv, Age::V17_0 },
        };
        // clang-format on

        for (auto const& mapping: mappings)
            if (mapping.first == value)
                return { mapping.second };

        return nullopt;
    }

    constexpr optional<unicode::General_Category> make_general_category(string_view value) noexcept
    {
        auto /*static*/ constexpr mappings = array {
            pair { "Cn"sv, General_Category::Unassigned },

            pair { "Lu"sv, General_Category::Uppercase_Letter },
            pair { "Ll"sv, General_Category::Lowercase_Letter },
            pair { "Lt"sv, General_Category::Titlecase_Letter },
            pair { "Lm"sv, General_Category::Modifier_Letter },
            pair { "Lo"sv, General_Category::Other_Letter },

            pair { "Mn"sv, General_Category::Nonspacing_Mark },
            pair { "Me"sv, General_Category::Enclosing_Mark },
            pair { "Mc"sv, General_Category::Spacing_Mark },

            pair { "Nd"sv, General_Category::Decimal_Number },
            pair { "Nl"sv, General_Category::Letter_Number },
            pair { "No"sv, General_Category::Other_Number },

            pair { "Zs"sv, General_Category::Space_Separator },
            pair { "Zl"sv, General_Category::Line_Separator },
            pair { "Zp"sv, General_Category::Paragraph_Separator },

            pair { "Cc"sv, General_Category::Control },
            pair { "Cf"sv, General_Category::Format },
            pair { "Co"sv, General_Category::Private_Use },
            pair { "Cs"sv, General_Category::Surrogate },

            pair { "Pd"sv, General_Category::Dash_Punctuation },
            pair { "Ps"sv, General_Category::Open_Punctuation },
            pair { "Pe"sv, General_Category::Close_Punctuation },
            pair { "Pc"sv, General_Category::Connector_Punctuation },
            pair { "Po"sv, General_Category::Other_Punctuation },

            pair { "Sm"sv, General_Category::Math_Symbol },
            pair { "Sc"sv, General_Category::Currency_Symbol },
            pair { "Sk"sv, General_Category::Modifier_Symbol },
            pair { "So"sv, General_Category::Other_Symbol },

            pair { "Pi"sv, General_Category::Initial_Punctuation },
            pair { "Pf"sv, General_Category::Final_Punctuation },
        };

        for (auto const& mapping: mappings)
            if (mapping.first == value)
                return { mapping.second };

        return nullopt;
    }

    constexpr optional<unicode::Script> make_script(string_view value) noexcept
    {
        auto /*static*/ constexpr mappings = array {
            pair { "Adlam"sv, Script::Adlam },
            pair { "Ahom"sv, Script::Ahom },
            pair { "Anatolian_Hieroglyphs"sv, Script::Anatolian_Hieroglyphs },
            pair { "Arabic"sv, Script::Arabic },
            pair { "Armenian"sv, Script::Armenian },
            pair { "Avestan"sv, Script::Avestan },
            pair { "Balinese"sv, Script::Balinese },
            pair { "Bamum"sv, Script::Bamum },
            pair { "Bassa_Vah"sv, Script::Bassa_Vah },
            pair { "Batak"sv, Script::Batak },
            pair { "Bengali"sv, Script::Bengali },
            pair { "Bhaiksuki"sv, Script::Bhaiksuki },
            pair { "Bopomofo"sv, Script::Bopomofo },
            pair { "Brahmi"sv, Script::Brahmi },
            pair { "Braille"sv, Script::Braille },
            pair { "Buginese"sv, Script::Buginese },
            pair { "Buhid"sv, Script::Buhid },
            pair { "Canadian_Aboriginal"sv, Script::Canadian_Aboriginal },
            pair { "Carian"sv, Script::Carian },
            pair { "Caucasian_Albanian"sv, Script::Caucasian_Albanian },
            pair { "Chakma"sv, Script::Chakma },
            pair { "Cham"sv, Script::Cham },
            pair { "Cherokee"sv, Script::Cherokee },
            pair { "Chorasmian"sv, Script::Chorasmian },
            pair { "Common"sv, Script::Common },
            pair { "Coptic"sv, Script::Coptic },
            pair { "Cuneiform"sv, Script::Cuneiform },
            pair { "Cypriot"sv, Script::Cypriot },
            pair { "Cypro_Minoan"sv, Script::Cypro_Minoan },
            pair { "Cyrillic"sv, Script::Cyrillic },
            pair { "Deseret"sv, Script::Deseret },
            pair { "Devanagari"sv, Script::Devanagari },
            pair { "Dives_Akuru"sv, Script::Dives_Akuru },
            pair { "Dogra"sv, Script::Dogra },
            pair { "Duployan"sv, Script::Duployan },
            pair { "Egyptian_Hieroglyphs"sv, Script::Egyptian_Hieroglyphs },
            pair { "Elbasan"sv, Script::Elbasan },
            pair { "Elymaic"sv, Script::Elymaic },
            pair { "Ethiopic"sv, Script::Ethiopic },
            pair { "Georgian"sv, Script::Georgian },
            pair { "Glagolitic"sv, Script::Glagolitic },
            pair { "Gothic"sv, Script::Gothic },
            pair { "Grantha"sv, Script::Grantha },
            pair { "Greek"sv, Script::Greek },
            pair { "Gujarati"sv, Script::Gujarati },
            pair { "Gunjala_Gondi"sv, Script::Gunjala_Gondi },
            pair { "Gurmukhi"sv, Script::Gurmukhi },
            pair { "Han"sv, Script::Han },
            pair { "Hangul"sv, Script::Hangul },
            pair { "Hanifi_Rohingya"sv, Script::Hanifi_Rohingya },
            pair { "Hanunoo"sv, Script::Hanunoo },
            pair { "Hatran"sv, Script::Hatran },
            pair { "Hebrew"sv, Script::Hebrew },
            pair { "Hiragana"sv, Script::Hiragana },
            pair { "Imperial_Aramaic"sv, Script::Imperial_Aramaic },
            pair { "Inherited"sv, Script::Inherited },
            pair { "Inscriptional_Pahlavi"sv, Script::Inscriptional_Pahlavi },
            pair { "Inscriptional_Parthian"sv, Script::Inscriptional_Parthian },
            pair { "Javanese"sv, Script::Javanese },
            pair { "Jurchen"sv, Script::Jurchen },
            pair { "Kaithi"sv, Script::Kaithi },
            pair { "Kannada"sv, Script::Kannada },
            pair { "Katakana"sv, Script::Katakana },
            pair { "Kawi"sv, Script::Kawi },
            pair { "Kayah_Li"sv, Script::Kayah_Li },
            pair { "Kharoshthi"sv, Script::Kharoshthi },
            pair { "Khitan_Small_Script"sv, Script::Khitan_Small_Script },
            pair { "Khmer"sv, Script::Khmer },
            pair { "Khojki"sv, Script::Khojki },
            pair { "Khudawadi"sv, Script::Khudawadi },
            pair { "Lao"sv, Script::Lao },
            pair { "Latin"sv, Script::Latin },
            pair { "Lepcha"sv, Script::Lepcha },
            pair { "Limbu"sv, Script::Limbu },
            pair { "Linear_A"sv, Script::Linear_A },
            pair { "Linear_B"sv, Script::Linear_B },
            pair { "Lisu"sv, Script::Lisu },
            pair { "Lycian"sv, Script::Lycian },
            pair { "Lydian"sv, Script::Lydian },
            pair { "Mahajani"sv, Script::Mahajani },
            pair { "Makasar"sv, Script::Makasar },
            pair { "Malayalam"sv, Script::Malayalam },
            pair { "Mandaic"sv, Script::Mandaic },
            pair { "Manichaean"sv, Script::Manichaean },
            pair { "Marchen"sv, Script::Marchen },
            pair { "Masaram_Gondi"sv, Script::Masaram_Gondi },
            pair { "Medefaidrin"sv, Script::Medefaidrin },
            pair { "Meetei_Mayek"sv, Script::Meetei_Mayek },
            pair { "Mende_Kikakui"sv, Script::Mende_Kikakui },
            pair { "Meroitic_Cursive"sv, Script::Meroitic_Cursive },
            pair { "Meroitic_Hieroglyphs"sv, Script::Meroitic_Hieroglyphs },
            pair { "Miao"sv, Script::Miao },
            pair { "Modi"sv, Script::Modi },
            pair { "Mongolian"sv, Script::Mongolian },
            pair { "Mro"sv, Script::Mro },
            pair { "Multani"sv, Script::Multani },
            pair { "Myanmar"sv, Script::Myanmar },
            pair { "Nabataean"sv, Script::Nabataean },
            pair { "Nag_Mundari"sv, Script::Nag_Mundari },
            pair { "Nandinagari"sv, Script::Nandinagari },
            pair { "New_Tai_Lue"sv, Script::New_Tai_Lue },
            pair { "Newa"sv, Script::Newa },
            pair { "Nko"sv, Script::Nko },
            pair { "Nushu"sv, Script::Nushu },
            pair { "Nyiakeng_Puachue_Hmong"sv, Script::Nyiakeng_Puachue_Hmong },
            pair { "Ogham"sv, Script::Ogham },
            pair { "Ol_Chiki"sv, Script::Ol_Chiki },
            pair { "Old_Hungarian"sv, Script::Old_Hungarian },
            pair { "Old_Italic"sv, Script::Old_Italic },
            pair { "Old_North_Arabian"sv, Script::Old_North_Arabian },
            pair { "Old_Permic"sv, Script::Old_Permic },
            pair { "Old_Persian"sv, Script::Old_Persian },
            pair { "Old_Sogdian"sv, Script::Old_Sogdian },
            pair { "Old_South_Arabian"sv, Script::Old_South_Arabian },
            pair { "Old_Turkic"sv, Script::Old_Turkic },
            pair { "Old_Uyghur"sv, Script::Old_Uyghur },
            pair { "Oriya"sv, Script::Oriya },
            pair { "Osage"sv, Script::Osage },
            pair { "Osmanya"sv, Script::Osmanya },
            pair { "Pahawh_Hmong"sv, Script::Pahawh_Hmong },
            pair { "Palmyrene"sv, Script::Palmyrene },
            pair { "Pau_Cin_Hau"sv, Script::Pau_Cin_Hau },
            pair { "Phags_Pa"sv, Script::Phags_Pa },
            pair { "Phoenician"sv, Script::Phoenician },
            pair { "Proto_Cuneiform"sv, Script::Proto_Cuneiform },
            pair { "Psalter_Pahlavi"sv, Script::Psalter_Pahlavi },
            pair { "Rejang"sv, Script::Rejang },
            pair { "Runic"sv, Script::Runic },
            pair { "Samaritan"sv, Script::Samaritan },
            pair { "Saurashtra"sv, Script::Saurashtra },
            pair { "Seal"sv, Script::Seal },
            pair { "Sharada"sv, Script::Sharada },
            pair { "Shavian"sv, Script::Shavian },
            pair { "Siddham"sv, Script::Siddham },
            pair { "SignWriting"sv, Script::SignWriting },
            pair { "Sinhala"sv, Script::Sinhala },
            pair { "Sogdian"sv, Script::Sogdian },
            pair { "Sora_Sompeng"sv, Script::Sora_Sompeng },
            pair { "Soyombo"sv, Script::Soyombo },
            pair { "Sundanese"sv, Script::Sundanese },
            pair { "Syloti_Nagri"sv, Script::Syloti_Nagri },
            pair { "Syriac"sv, Script::Syriac },
            pair { "Tagalog"sv, Script::Tagalog },
            pair { "Tagbanwa"sv, Script::Tagbanwa },
            pair { "Tai_Le"sv, Script::Tai_Le },
            pair { "Tai_Tham"sv, Script::Tai_Tham },
            pair { "Tai_Viet"sv, Script::Tai_Viet },
            pair { "Takri"sv, Script::Takri },
            pair { "Tamil"sv, Script::Tamil },
            pair { "Tangsa"sv, Script::Tangsa },
            pair { "Tangut"sv, Script::Tangut },
            pair { "Telugu"sv, Script::Telugu },
            pair { "Thaana"sv, Script::Thaana },
            pair { "Thai"sv, Script::Thai },
            pair { "Tibetan"sv, Script::Tibetan },
            pair { "Tifinagh"sv, Script::Tifinagh },
            pair { "Tirhuta"sv, Script::Tirhuta },
            pair { "Toto"sv, Script::Toto },
            pair { "Ugaritic"sv, Script::Ugaritic },
            pair { "Vai"sv, Script::Vai },
            pair { "Vithkuqi"sv, Script::Vithkuqi },
            pair { "Wancho"sv, Script::Wancho },
            pair { "Warang_Citi"sv, Script::Warang_Citi },
            pair { "Yezidi"sv, Script::Yezidi },
            pair { "Yi"sv, Script::Yi },
            pair { "Zanabazar_Square"sv, Script::Zanabazar_Square },
        };

        for (auto const& mapping: mappings)
            if (mapping.first == value)
                return { mapping.second };

        return nullopt;
    }

    constexpr optional<unicode::East_Asian_Width> make_width(string_view value) noexcept
    {
        auto /*static*/ constexpr mappings = array {
            pair { "A"sv, unicode::East_Asian_Width::Ambiguous }, pair { "F"sv, unicode::East_Asian_Width::Fullwidth },
            pair { "H"sv, unicode::East_Asian_Width::Halfwidth }, pair { "N"sv, unicode::East_Asian_Width::Neutral },
            pair { "Na"sv, unicode::East_Asian_Width::Narrow },   pair { "W"sv, unicode::East_Asian_Width::Wide },
        };

        for (auto const& mapping: mappings)
            if (mapping.first == value)
                return mapping.second;

        return nullopt;
    }

    constexpr optional<unicode::Grapheme_Cluster_Break> make_gb(string_view value) noexcept
    {
        auto /*static*/ constexpr mappings = array {
            pair { "LV"sv, unicode::Grapheme_Cluster_Break::LV },
            pair { "Undefined"sv, unicode::Grapheme_Cluster_Break::Undefined },
            pair { "CR"sv, unicode::Grapheme_Cluster_Break::CR },
            pair { "Control"sv, unicode::Grapheme_Cluster_Break::Control },
            pair { "E_Base"sv, unicode::Grapheme_Cluster_Break::E_Base },
            pair { "E_Base_GAZ"sv, unicode::Grapheme_Cluster_Break::E_Base_GAZ },
            pair { "E_Modifier"sv, unicode::Grapheme_Cluster_Break::E_Modifier },
            pair { "Extend"sv, unicode::Grapheme_Cluster_Break::Extend },
            pair { "Glue_After_Zwj"sv, unicode::Grapheme_Cluster_Break::Glue_After_Zwj },
            pair { "L"sv, unicode::Grapheme_Cluster_Break::L },
            pair { "LF"sv, unicode::Grapheme_Cluster_Break::LF },
            pair { "LV"sv, unicode::Grapheme_Cluster_Break::LV },
            pair { "LVT"sv, unicode::Grapheme_Cluster_Break::LVT },
            pair { "Other"sv, unicode::Grapheme_Cluster_Break::Other },
            pair { "Prepend"sv, unicode::Grapheme_Cluster_Break::Prepend },
            pair { "Regional_Indicator"sv, unicode::Grapheme_Cluster_Break::Regional_Indicator },
            pair { "SpacingMark"sv, unicode::Grapheme_Cluster_Break::SpacingMark },
            pair { "T"sv, unicode::Grapheme_Cluster_Break::T },
            pair { "V"sv, unicode::Grapheme_Cluster_Break::V },
            pair { "ZWJ"sv, unicode::Grapheme_Cluster_Break::ZWJ },
        };

        for (auto const& mapping: mappings)
            if (mapping.first == value)
                return mapping.second;

        return nullopt;
    }
    // }}}

    using scoped_timer = support::scoped_timer;

    inline uint8_t toCharWidth(codepoint_properties const& properties) noexcept
    {
        switch (properties.general_category)
        {
            case General_Category::Control: // XXX really?
            case General_Category::Enclosing_Mark:
            case General_Category::Format:
            case General_Category::Line_Separator:
            // case General_Category::Modifier_Symbol:
            case General_Category::Nonspacing_Mark:
            case General_Category::Paragraph_Separator:
            case General_Category::Spacing_Mark:
            case General_Category::Surrogate: return 0;
            default: break;
        }

        if (properties.is_emoji_presentation())
            // UAX #11 §5 Recommendations:
            //     [UTS51] emoji presentation sequences behave as though they were East Asian Wide,
            //     regardless of their assigned East_Asian_Width property value.
            return 2;

        switch (properties.east_asian_width)
        {
            case East_Asian_Width::Narrow:
            case East_Asian_Width::Ambiguous:
            case East_Asian_Width::Halfwidth:
            case East_Asian_Width::Neutral:
                //.
                return 1;
            case East_Asian_Width::Wide:
            case East_Asian_Width::Fullwidth:
                //.
                return 2;
        }

        // Should never be reached.
        return 1;
    }

    inline EmojiSegmentationCategory toEmojiSegmentationCategory(char32_t codepoint, codepoint_properties const& props) noexcept
    {
        if (codepoint == 0x20e3)
            return EmojiSegmentationCategory::CombiningEnclosingKeyCap;
        if (codepoint == 0x20e0)
            return EmojiSegmentationCategory::CombiningEnclosingCircleBackslash;
        if (codepoint == 0x200d)
            return EmojiSegmentationCategory::ZWJ;
        if (codepoint == 0xfe0e)
            return EmojiSegmentationCategory::VS15;
        if (codepoint == 0xfe0f)
            return EmojiSegmentationCategory::VS16;
        if (codepoint == 0x1f3f4)
            return EmojiSegmentationCategory::TagBase;
        if ((codepoint >= 0xE0030 && codepoint <= 0xE0039) || (codepoint >= 0xE0061 && codepoint <= 0xE007A))
            return EmojiSegmentationCategory::TagSequence;
        if (codepoint == 0xE007F)
            return EmojiSegmentationCategory::TagTerm;

        if (props.is_emoji_modifier_base())
            return EmojiSegmentationCategory::EmojiModifierBase;
        if (props.is_emoji_modifier())
            return EmojiSegmentationCategory::EmojiModifier;
        if (props.grapheme_cluster_break == Grapheme_Cluster_Break::Regional_Indicator)
            return EmojiSegmentationCategory::RegionalIndicator;
        if (('0' <= codepoint && codepoint <= '9') || codepoint == '#' || codepoint == '*')
            return EmojiSegmentationCategory::KeyCapBase;
        if (props.is_emoji_presentation())
            return EmojiSegmentationCategory::EmojiEmojiPresentation;
        if (props.is_emoji() && !props.is_emoji_presentation())
            return EmojiSegmentationCategory::EmojiTextPresentation;
        if (props.is_emoji())
            return EmojiSegmentationCategory::Emoji;

        return EmojiSegmentationCategory::Invalid;
    }

    class codepoint_properties_loader
    {
      public:
        static std::tuple<codepoint_properties_table, codepoint_names_table> load_from_directory(string const& ucdDataDirectory,
                                                                                                 std::ostream* log = nullptr);

      private:
        using tables_view = codepoint_properties::tables_view;

        codepoint_properties_loader(string ucdDataDirectory, std::ostream* log = nullptr);

        void load();
        void create_multistage_tables();

        [[nodiscard]] codepoint_properties& properties(char32_t codepoint) noexcept
        {
            return _codepoints[static_cast<size_t>(codepoint)];
        }

        template <typename T>
        void process_properties(string const& filePathSuffix, T callback)
        {
            auto const _ = scoped_timer { _log, "Loading file " + filePathSuffix };

            // clang-format off
            // [SPACE] ALNUMDOT ([SPACE] ALNUMDOT)::= (\s+[A-Za-z_0-9\.]+)*
            auto const singleCodepointPattern = regex(R"(^([0-9A-F]+)\s*;\s*([A-Za-z_0-9\.]+(\s+[A-Za-z_0-9\.]+)*))");
            auto const codepointRangePattern = regex(R"(^([0-9A-F]+)\.\.([0-9A-F]+)\s*;\s*([A-Za-z_0-9\.]+))");
            // clang-format on

            auto const filePath = _ucdDataDirectory + "/" + filePathSuffix;
            auto f = ifstream(filePath);
            if (!f.good())
                throw std::runtime_error("Could not open file: "s + filePath);
            while (f.good())
            {
                string line;
                getline(f, line);
                auto sm = smatch {};
                if (regex_search(line, sm, singleCodepointPattern))
                {
                    auto const codepoint = static_cast<char32_t>(stoul(sm[1], nullptr, 16));
                    callback(codepoint, sm.str(2));
                }
                else if (regex_search(line, sm, codepointRangePattern))
                {
                    auto const first = static_cast<char32_t>(stoul(sm[1], nullptr, 16));
                    auto const last = static_cast<char32_t>(stoul(sm[2], nullptr, 16));
                    for (auto codepoint = first; codepoint <= last; ++codepoint)
                        callback(codepoint, sm.str(3));
                }
            }
        }

        string _ucdDataDirectory;
        std::ostream* _log;
        vector<codepoint_properties> _codepoints {}; // Meh!
        codepoint_properties_table _output {};

        vector<std::string> _names {};
        codepoint_names_table _outputNames {};
    };

    codepoint_properties_loader::codepoint_properties_loader(string ucdDataDirectory, std::ostream* log):
        _ucdDataDirectory { std::move(ucdDataDirectory) }, _log { log }
    {
        _codepoints.resize(0x110'000);
        _names.resize(0x110'000);

        // _output.names.emplace_back(""); // All unassigned codepoints point here.
    }

    void codepoint_properties_loader::load()
    {
        process_properties("Scripts.txt", [&](char32_t codepoint, string_view value) {
            properties(codepoint).script = make_script(value).value_or(unicode::Script::Invalid);
        });

        process_properties("DerivedCoreProperties.txt", [&](char32_t codepoint, string_view value) {
            // Generically written such that we can easily add more core properties here, once relevant.
            auto static constexpr mappings = array {
                pair { "Grapheme_Extend", codepoint_properties::FlagCoreGraphemeExtend },
            };
            auto const equalName = [=](auto x) {
                return x.first == value;
            };

            if (auto const i = find_if(begin(mappings), end(mappings), equalName); i != end(mappings))
                properties(codepoint).flags |= i->second;
        });

        process_properties("DerivedAge.txt", [&](char32_t codepoint, string_view value) {
            properties(codepoint).age = make_age(value).value_or(unicode::Age::Unassigned);
        });

        process_properties("extracted/DerivedGeneralCategory.txt", [&](char32_t codepoint, string_view value) {
            properties(codepoint).general_category = make_general_category(value).value();
        });

        // Prep-work for names loading
        process_properties("extracted/DerivedName.txt", [&](char32_t codepoint, string_view value) {
            _names[static_cast<size_t>(codepoint)] = string(value.data(), value.size());
        });

        process_properties("auxiliary/GraphemeBreakProperty.txt", [&](char32_t codepoint, string_view value) {
            properties(codepoint).grapheme_cluster_break = make_gb(value).value();
        });

        process_properties("EastAsianWidth.txt", [&](char32_t codepoint, string_view value) {
            properties(codepoint).east_asian_width = make_width(value).value();
        });

        // {{{ fill EmojiSegmentationCategory and other emoji related properties
        // clang-format off
        properties(0x20e3).emoji_segmentation_category = EmojiSegmentationCategory::CombiningEnclosingKeyCap;
        properties(0x20e0).emoji_segmentation_category = EmojiSegmentationCategory::CombiningEnclosingCircleBackslash;
        properties(0x200d).emoji_segmentation_category = EmojiSegmentationCategory::ZWJ;
        properties(0xfe0e).emoji_segmentation_category = EmojiSegmentationCategory::VS15;
        properties(0xfe0f).emoji_segmentation_category = EmojiSegmentationCategory::VS16;

        properties(0x1f3f4).emoji_segmentation_category = EmojiSegmentationCategory::TagBase;
        for (char32_t ch = 0xE0030; ch <= 0xE0039; ++ch)
            properties(ch).emoji_segmentation_category = EmojiSegmentationCategory::TagSequence;
        for (char32_t ch = 0xE0061; ch <= 0xE007A; ++ch)
            properties(ch).emoji_segmentation_category = EmojiSegmentationCategory::TagSequence;
        properties(0xE007F).emoji_segmentation_category = EmojiSegmentationCategory::TagTerm;

        for (char32_t const codepoint: array<char32_t, 12> { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#', '*' })
            properties(codepoint).emoji_segmentation_category = EmojiSegmentationCategory::KeyCapBase;
        // clang-format on

        process_properties("emoji/emoji-data.txt", [&](char32_t codepoint, string_view value) -> void {
            auto static constexpr mappings = array {
                pair { "Emoji", codepoint_properties::FlagEmoji },
                pair { "Emoji_Component", codepoint_properties::FlagEmojiComponent },
                pair { "Emoji_Modifier", codepoint_properties::FlagEmojiModifier },
                pair { "Emoji_Modifier_Base", codepoint_properties::FlagEmojiModifierBase },
                pair { "Emoji_Presentation", codepoint_properties::FlagEmojiPresentation },
                pair { "Extended_Pictographic", codepoint_properties::FlagExtendedPictographic },
            };
            auto const equalName = [=](auto x) {
                return x.first == value;
            };

            if (auto const i = find_if(begin(mappings), end(mappings), equalName); i != end(mappings))
                properties(codepoint).flags |= i->second;
        });

        {
            auto const _ = scoped_timer { _log, "Assigning EmojiSegmentationCategory" };
            for (char32_t codepoint = 0; codepoint < 0x110'000; ++codepoint)
                properties(codepoint).emoji_segmentation_category = toEmojiSegmentationCategory(codepoint, properties(codepoint));
        }
        // }}}

        // {{{ assign char_width
        {
            auto const _ = scoped_timer { _log, "Assigning char_width" };
            for (char32_t codepoint = 0; codepoint < 0x110'000; ++codepoint)
                properties(codepoint).char_width = toCharWidth(properties(codepoint));
        }
        // }}}
    }

    std::tuple<codepoint_properties_table, codepoint_names_table> codepoint_properties_loader::load_from_directory(
        string const& ucdDataDirectory, std::ostream* log)
    {
        auto loader = codepoint_properties_loader { ucdDataDirectory, log };
        loader.load();
        loader.create_multistage_tables();

        return { std::move(loader._output), std::move(loader._outputNames) };
    }

    void codepoint_properties_loader::create_multistage_tables()
    {
        {
            auto const _ = scoped_timer { _log, "Creating multistage tables (properties)" };
            support::generate(
                _codepoints.data(), _codepoints.size(), _output, [](auto const& begin, auto const& end, auto value) noexcept {
                    return std::find(begin, end, value);
                });
        }

        {
            auto const _ = scoped_timer { _log, "Creating multistage tables (names)" };
            support::generate(
                _names.data(), _names.size(), _outputNames, [&](auto const& begin, auto const& end, auto value) noexcept {
#if defined(LIBUNICODE_TABLEGEN_FASTBUILD)
                    if (value.empty())
                        // This case is happening for unassigned codepoints (and quite a lot)
                        // We want to keep the names table as small as possible, so we don't
                        // re-add empty strings to it.
                        return std::find(begin, end, value);
                    // Non-empty names are mostly unique (~1.6% are duplicates, that's okay)
                    return end;
#else
                                  return std::find(begin, end, value);
#endif
                });
        }
    }
} // namespace

std::tuple<codepoint_properties_table, codepoint_names_table> load_from_directory(std::string const& ucdDataDirectory,
                                                                                  std::ostream* log)
{
    return codepoint_properties_loader::load_from_directory(ucdDataDirectory, log);
}

} // namespace unicode
