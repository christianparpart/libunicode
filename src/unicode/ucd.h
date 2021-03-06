/**
 * This file is part of the "libunicode" project
 *   Copyright (c) 2020-2021 Christian Parpart <christian@parpart.family>
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
#pragma once

#include <unicode/ucd_enums.h>

#include <array>
#include <optional>
#include <string>
#include <utility>

namespace unicode {

Plane plane(char32_t _codepoint) noexcept;

bool contains(Core_Property _prop, char32_t _codepoint) noexcept;

bool contains(General_Category _cat, char32_t _codepoint) noexcept;

namespace general_category {
    General_Category get(char32_t _value) noexcept;

    inline bool close_punctuation(char32_t _codepoint) { return contains(General_Category::Close_Punctuation, _codepoint); }
    inline bool connector_punctuation(char32_t _codepoint) { return contains(General_Category::Connector_Punctuation, _codepoint); }
    inline bool control(char32_t _codepoint) { return contains(General_Category::Control, _codepoint); }
    inline bool currency_symbol(char32_t _codepoint) { return contains(General_Category::Currency_Symbol, _codepoint); }
    inline bool dash_punctuation(char32_t _codepoint) { return contains(General_Category::Dash_Punctuation, _codepoint); }
    inline bool decimal_number(char32_t _codepoint) { return contains(General_Category::Decimal_Number, _codepoint); }
    inline bool enclosing_mark(char32_t _codepoint) { return contains(General_Category::Enclosing_Mark, _codepoint); }
    inline bool final_punctuation(char32_t _codepoint) { return contains(General_Category::Final_Punctuation, _codepoint); }
    inline bool format(char32_t _codepoint) { return contains(General_Category::Format, _codepoint); }
    inline bool initial_punctuation(char32_t _codepoint) { return contains(General_Category::Initial_Punctuation, _codepoint); }
    inline bool letter_number(char32_t _codepoint) { return contains(General_Category::Letter_Number, _codepoint); }
    inline bool line_separator(char32_t _codepoint) { return contains(General_Category::Line_Separator, _codepoint); }
    inline bool lowercase_letter(char32_t _codepoint) { return contains(General_Category::Lowercase_Letter, _codepoint); }
    inline bool math_symbol(char32_t _codepoint) { return contains(General_Category::Math_Symbol, _codepoint); }
    inline bool modifier_letter(char32_t _codepoint) { return contains(General_Category::Modifier_Letter, _codepoint); }
    inline bool modifier_symbol(char32_t _codepoint) { return contains(General_Category::Modifier_Symbol, _codepoint); }
    inline bool nonspacing_mark(char32_t _codepoint) { return contains(General_Category::Nonspacing_Mark, _codepoint); }
    inline bool open_punctuation(char32_t _codepoint) { return contains(General_Category::Open_Punctuation, _codepoint); }
    inline bool other_letter(char32_t _codepoint) { return contains(General_Category::Other_Letter, _codepoint); }
    inline bool other_number(char32_t _codepoint) { return contains(General_Category::Other_Number, _codepoint); }
    inline bool other_punctuation(char32_t _codepoint) { return contains(General_Category::Other_Punctuation, _codepoint); }
    inline bool other_symbol(char32_t _codepoint) { return contains(General_Category::Other_Symbol, _codepoint); }
    inline bool paragraph_separator(char32_t _codepoint) { return contains(General_Category::Paragraph_Separator, _codepoint); }
    inline bool private_use(char32_t _codepoint) { return contains(General_Category::Private_Use, _codepoint); }
    inline bool space_separator(char32_t _codepoint) { return contains(General_Category::Space_Separator, _codepoint); }
    inline bool spacing_mark(char32_t _codepoint) { return contains(General_Category::Spacing_Mark, _codepoint); }
    inline bool surrogate(char32_t _codepoint) { return contains(General_Category::Surrogate, _codepoint); }
    inline bool titlecase_letter(char32_t _codepoint) { return contains(General_Category::Titlecase_Letter, _codepoint); }
    inline bool unassigned(char32_t _codepoint) { return contains(General_Category::Unassigned, _codepoint); }
    inline bool uppercase_letter(char32_t _codepoint) { return contains(General_Category::Uppercase_Letter, _codepoint); }
}

Script script(char32_t _codepoint) noexcept;

size_t script_extensions(char32_t _codepoint, Script* _result, size_t _capacity) noexcept;

Block block(char32_t _codepoint) noexcept;

namespace grapheme_cluster_break {
    bool cr(char32_t _codepoint) noexcept;
    bool control(char32_t _codepoint) noexcept;
    bool extend(char32_t _codepoint) noexcept;
    bool l(char32_t _codepoint) noexcept;
    bool lf(char32_t _codepoint) noexcept;
    bool lv(char32_t _codepoint) noexcept;
    bool lvt(char32_t _codepoint) noexcept;
    bool prepend(char32_t _codepoint) noexcept;
    bool regional_indicator(char32_t _codepoint) noexcept;
    bool spacingmark(char32_t _codepoint) noexcept;
    bool t(char32_t _codepoint) noexcept;
    bool v(char32_t _codepoint) noexcept;
    bool zwj(char32_t _codepoint) noexcept;
}

inline std::string to_string(EastAsianWidth _value) {
    switch (_value) {
        case EastAsianWidth::Ambiguous: return "Ambiguous";
        case EastAsianWidth::FullWidth: return "FullWidth";
        case EastAsianWidth::HalfWidth: return "HalfWidth";
        case EastAsianWidth::Neutral: return "Neutral";
        case EastAsianWidth::Narrow: return "Narrow";
        case EastAsianWidth::Wide: return "Wide";
        case EastAsianWidth::Unspecified: return "Unspecified";
    }
    return "Unknown";
}

EastAsianWidth east_asian_width(char32_t _codepoint) noexcept;

bool emoji(char32_t _codepoint) noexcept;
bool emoji_component(char32_t _codepoint) noexcept;
bool emoji_modifier(char32_t _codepoint) noexcept;
bool emoji_modifier_base(char32_t _codepoint) noexcept;
bool emoji_presentation(char32_t _codepoint) noexcept;
bool extended_pictographic(char32_t _codepoint) noexcept;

} // end namespace
