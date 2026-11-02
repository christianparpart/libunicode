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
#include <libunicode/case_normalization_data.h>
#include <libunicode/convert.h>
#include <libunicode/normalization.h>

#include <algorithm>
#include <iterator>
#include <span>

namespace unicode
{

// ============================================================================
// Hangul constants (from Unicode Standard, Chapter 3)
// ============================================================================
namespace hangul
{
    constexpr char32_t SBase = 0xAC00;
    constexpr char32_t LBase = 0x1100;
    constexpr char32_t VBase = 0x1161;
    constexpr char32_t TBase = 0x11A7;
    constexpr char32_t LCount = 19;
    constexpr char32_t VCount = 21;
    constexpr char32_t TCount = 28;
    constexpr char32_t NCount = VCount * TCount; // 588
    constexpr char32_t SCount = LCount * NCount; // 11172
} // namespace hangul

// ============================================================================
// Helper functions
// ============================================================================

namespace
{
    template <size_t N>
    [[nodiscard]] bool contains_codepoint(std::array<char32_t, N> const& table, char32_t codepoint) noexcept
    {
        return std::binary_search(table.begin(), table.end(), codepoint);
    }

    [[nodiscard]] uint8_t lookup_ccc(char32_t codepoint) noexcept
    {
        auto const& table = detail::ccc_table;
        auto const it = std::lower_bound(
            table.begin(), table.end(), codepoint, [](auto const& pair, char32_t cp) { return pair.first < cp; });

        if (it != table.end() && it->first == codepoint)
            return it->second;

        return 0;
    }

    /// Binary search a decomposition table, returning a span into the static table data.
    /// Returns an empty span if the codepoint has no decomposition in this table.
    template <typename Table>
    [[nodiscard]] std::span<char32_t const> find_decomposition(Table const& table, char32_t codepoint) noexcept
    {
        auto const it = std::lower_bound(
            table.begin(), table.end(), codepoint, [](auto const& entry, char32_t cp) { return entry.source < cp; });

        if (it != table.end() && it->source == codepoint)
            return { it->targets, it->length };

        return {};
    }

    [[nodiscard]] Decomposition_Type lookup_compatibility_decomp_type(char32_t codepoint) noexcept
    {
        auto const& table = detail::compatibility_decomposition_table;
        auto const it = std::lower_bound(
            table.begin(), table.end(), codepoint, [](auto const& entry, char32_t cp) { return entry.source < cp; });

        if (it != table.end() && it->source == codepoint)
            return static_cast<Decomposition_Type>(it->decomp_type);

        return Decomposition_Type::None;
    }

    // Try to compose two codepoints
    [[nodiscard]] char32_t try_compose(char32_t first, char32_t second) noexcept
    {
        // Check Hangul composition first (L + V or LV + T), Unicode Standard section 3.12. The jamo
        // predicates carry the ranges: T is U+11A8..U+11C2, so TBase (U+11A7) never composes.
        if (is_hangul_l_jamo(first))
        {
            if (is_hangul_v_jamo(second))
            {
                // L + V -> LV
                return hangul::SBase + ((first - hangul::LBase) * hangul::VCount + (second - hangul::VBase)) * hangul::TCount;
            }
        }
        else if (is_hangul_syllable(first))
        {
            if ((first - hangul::SBase) % hangul::TCount == 0 && is_hangul_t_jamo(second))
            {
                // LV + T -> LVT
                return first + (second - hangul::TBase);
            }
        }

        // Look up in composition table (binary search)
        auto const& table = detail::composition_table;
        auto it = table.begin();
        auto end = table.end();

        while (it != end)
        {
            auto mid = it + (end - it) / 2;
            if (mid->first < first || (mid->first == first && mid->second < second))
                it = mid + 1;
            else if (mid->first > first || (mid->first == first && mid->second > second))
                end = mid;
            else
                return mid->composed;
        }

        return 0; // No composition found
    }

    void decompose_recursive(char32_t cp, std::u32string& output, bool compatibility)
    {
        if (is_hangul_syllable(cp))
        {
            char32_t jamos[3];
            auto const count = hangul_decompose(cp, jamos);
            output.append(jamos, jamos + count);
            return;
        }

        auto decomp = find_decomposition(detail::canonical_decomposition_table, cp);

        if (decomp.empty() && compatibility)
            decomp = find_decomposition(detail::compatibility_decomposition_table, cp);

        if (decomp.empty())
        {
            output.push_back(cp);
            return;
        }

        for (char32_t c: decomp)
            decompose_recursive(c, output, compatibility);
    }

    // Canonical ordering of combining marks
    void canonical_order(std::u32string& text)
    {
        if (text.size() < 2)
            return;

        // Bubble sort adjacent combining marks by CCC (stable)
        for (size_t i = 1; i < text.size(); ++i)
        {
            auto const ccc_i = canonical_combining_class(text[i]);
            if (ccc_i == 0)
                continue; // Starter, no reordering needed

            size_t j = i;
            while (j > 0)
            {
                auto const ccc_prev = canonical_combining_class(text[j - 1]);
                if (ccc_prev == 0 || ccc_prev <= ccc_i)
                    break;
                std::swap(text[j - 1], text[j]);
                --j;
            }
        }
    }

    // Compose a decomposed (and canonically-ordered) string.
    //
    // Every input character is appended to `result` immediately, in its
    // decomposed-and-reordered position -- composition never delays writing a
    // character, it only merges a later combining mark back into an
    // already-written starter (`result[starterIndex]`). This keeps relative
    // ordering correct by construction: a combining mark that fails to compose
    // can never end up ahead of the starter it followed, because the starter
    // was placed in `result` before that mark was even considered.
    std::u32string compose(std::u32string const& decomposed)
    {
        if (decomposed.empty())
            return {};

        std::u32string result;
        result.reserve(decomposed.size());

        size_t starterIndex = std::u32string::npos;
        uint8_t last_ccc = 0;

        for (char32_t cp: decomposed)
        {
            uint8_t const ccc = canonical_combining_class(cp);

            char32_t composed = 0;
            if (starterIndex != std::u32string::npos && (last_ccc < ccc || last_ccc == 0))
                composed = try_compose(result[starterIndex], cp);

            if (composed != 0 && !is_composition_exclusion(composed))
            {
                // Composition succeeded: update the starter in place, do not
                // advance last_ccc -- the (now composed) starter may still
                // absorb a later mark with a higher CCC.
                result[starterIndex] = composed;
                continue;
            }

            result.push_back(cp);

            if (ccc == 0)
            {
                // New starter.
                starterIndex = result.size() - 1;
                last_ccc = 0;
            }
            else
            {
                // Blocked from the current starter; a later mark may still
                // compose with it (per UAX #15 blocking is CCC-relative, not
                // absolute), so the starter stays put -- only our "highest
                // CCC seen since the starter" watermark advances.
                last_ccc = ccc;
            }
        }

        return result;
    }

} // anonymous namespace

// ============================================================================
// Normalization properties
// ============================================================================

uint8_t canonical_combining_class(char32_t codepoint) noexcept
{
    return lookup_ccc(codepoint);
}

std::vector<char32_t> canonical_decomposition(char32_t codepoint)
{
    if (is_hangul_syllable(codepoint))
    {
        std::vector<char32_t> result(3);
        auto const count = hangul_decompose(codepoint, result.data());
        result.resize(count);
        return result;
    }

    auto span = find_decomposition(detail::canonical_decomposition_table, codepoint);
    return { span.begin(), span.end() };
}

std::vector<char32_t> compatibility_decomposition(char32_t codepoint)
{
    auto result = canonical_decomposition(codepoint);
    if (result.empty())
    {
        auto span = find_decomposition(detail::compatibility_decomposition_table, codepoint);
        result.assign(span.begin(), span.end());
    }
    return result;
}

Decomposition_Type decomposition_type(char32_t codepoint) noexcept
{
    if (is_hangul_syllable(codepoint))
        return Decomposition_Type::Canonical;

    if (!find_decomposition(detail::canonical_decomposition_table, codepoint).empty())
        return Decomposition_Type::Canonical;

    return lookup_compatibility_decomp_type(codepoint);
}

bool is_composition_exclusion(char32_t codepoint) noexcept
{
    return contains_codepoint(detail::composition_exclusions_table, codepoint);
}

bool is_full_composition_exclusion(char32_t codepoint) noexcept
{
    return is_composition_exclusion(codepoint);
}

// ============================================================================
// Quick Check properties
// ============================================================================

NFC_Quick_Check nfc_quick_check(char32_t codepoint) noexcept
{
    if (contains_codepoint(detail::nfc_qc_no_table, codepoint))
        return NFC_Quick_Check::No;
    if (contains_codepoint(detail::nfc_qc_maybe_table, codepoint))
        return NFC_Quick_Check::Maybe;
    return NFC_Quick_Check::Yes;
}

bool nfd_quick_check(char32_t codepoint) noexcept
{
    return !contains_codepoint(detail::nfd_qc_no_table, codepoint);
}

NFKC_Quick_Check nfkc_quick_check(char32_t codepoint) noexcept
{
    if (contains_codepoint(detail::nfkc_qc_no_table, codepoint))
        return NFKC_Quick_Check::No;
    if (contains_codepoint(detail::nfkc_qc_maybe_table, codepoint))
        return NFKC_Quick_Check::Maybe;
    return NFKC_Quick_Check::Yes;
}

bool nfkd_quick_check(char32_t codepoint) noexcept
{
    return !contains_codepoint(detail::nfkd_qc_no_table, codepoint);
}

// ============================================================================
// String normalization
// ============================================================================

std::u32string normalize(std::u32string_view text, Normalization_Form form)
{
    if (text.empty())
        return {};

    bool const use_compatibility = (form == Normalization_Form::NFKC || form == Normalization_Form::NFKD);

    // Step 1: Decompose
    std::u32string decomposed;
    decomposed.reserve(text.size() * 2);

    for (char32_t cp: text)
        decompose_recursive(cp, decomposed, use_compatibility);

    // Step 2: Canonical ordering
    canonical_order(decomposed);

    // Step 3: Compose (for NFC/NFKC only)
    if (form == Normalization_Form::NFC || form == Normalization_Form::NFKC)
        return compose(decomposed);

    return decomposed;
}

std::string normalize(std::string_view text, Normalization_Form form)
{
    auto const u32text = convert_to<char32_t>(text);
    auto const u32result = normalize(std::u32string_view(u32text), form);
    return convert_to<char>(std::u32string_view(u32result));
}

// ============================================================================
// Normalization checks
// ============================================================================

Quick_Check_Result quick_check(std::u32string_view text, Normalization_Form form)
{
    Quick_Check_Result result = Quick_Check_Result::Yes;
    uint8_t last_ccc = 0;

    for (char32_t cp: text)
    {
        uint8_t ccc = canonical_combining_class(cp);

        // Check canonical ordering
        if (ccc != 0 && last_ccc > ccc)
            return Quick_Check_Result::No;

        last_ccc = ccc;

        // Check quick check property
        switch (form)
        {
            case Normalization_Form::NFC: {
                auto qc = nfc_quick_check(cp);
                if (qc == NFC_Quick_Check::No)
                    return Quick_Check_Result::No;
                if (qc == NFC_Quick_Check::Maybe)
                    result = Quick_Check_Result::Maybe;
                break;
            }
            case Normalization_Form::NFD:
                if (!nfd_quick_check(cp))
                    return Quick_Check_Result::No;
                break;
            case Normalization_Form::NFKC: {
                auto qc = nfkc_quick_check(cp);
                if (qc == NFKC_Quick_Check::No)
                    return Quick_Check_Result::No;
                if (qc == NFKC_Quick_Check::Maybe)
                    result = Quick_Check_Result::Maybe;
                break;
            }
            case Normalization_Form::NFKD:
                if (!nfkd_quick_check(cp))
                    return Quick_Check_Result::No;
                break;
        }
    }

    return result;
}

Quick_Check_Result quick_check(std::string_view text, Normalization_Form form)
{
    auto const u32text = convert_to<char32_t>(text);
    return quick_check(std::u32string_view(u32text), form);
}

bool is_normalized(std::u32string_view text, Normalization_Form form)
{
    auto qc = quick_check(text, form);
    if (qc == Quick_Check_Result::Yes)
        return true;
    if (qc == Quick_Check_Result::No)
        return false;

    // Quick check returned Maybe, do full check
    auto const normalized = normalize(text, form);
    return normalized == text;
}

bool is_normalized(std::string_view text, Normalization_Form form)
{
    auto const u32text = convert_to<char32_t>(text);
    return is_normalized(std::u32string_view(u32text), form);
}

// ============================================================================
// Canonical equivalence
// ============================================================================

bool is_canonically_equivalent(std::u32string_view a, std::u32string_view b)
{
    return to_nfd(a) == to_nfd(b);
}

bool is_canonically_equivalent(std::string_view a, std::string_view b)
{
    return to_nfd(a) == to_nfd(b);
}

bool is_compatibility_equivalent(std::u32string_view a, std::u32string_view b)
{
    return to_nfkd(a) == to_nfkd(b);
}

bool is_compatibility_equivalent(std::string_view a, std::string_view b)
{
    return to_nfkd(a) == to_nfkd(b);
}

// ============================================================================
// Hangul algorithmic decomposition/composition
// ============================================================================

bool is_hangul_syllable(char32_t codepoint) noexcept
{
    return codepoint >= hangul::SBase && codepoint < hangul::SBase + hangul::SCount;
}

bool is_hangul_l_jamo(char32_t codepoint) noexcept
{
    return codepoint >= hangul::LBase && codepoint < hangul::LBase + hangul::LCount;
}

bool is_hangul_v_jamo(char32_t codepoint) noexcept
{
    return codepoint >= hangul::VBase && codepoint < hangul::VBase + hangul::VCount;
}

bool is_hangul_t_jamo(char32_t codepoint) noexcept
{
    return codepoint > hangul::TBase && codepoint < hangul::TBase + hangul::TCount;
}

size_t hangul_decompose(char32_t syllable, char32_t* output) noexcept
{
    if (!is_hangul_syllable(syllable))
        return 0;

    auto const sIndex = syllable - hangul::SBase;
    auto const lIndex = sIndex / hangul::NCount;
    auto const vIndex = (sIndex % hangul::NCount) / hangul::TCount;
    auto const tIndex = sIndex % hangul::TCount;

    output[0] = hangul::LBase + lIndex;
    output[1] = hangul::VBase + vIndex;

    if (tIndex > 0)
    {
        output[2] = hangul::TBase + tIndex;
        return 3;
    }

    return 2;
}

char32_t hangul_compose(char32_t l, char32_t v, char32_t t) noexcept
{
    if (!is_hangul_l_jamo(l) || !is_hangul_v_jamo(v))
        return 0;

    auto const lIndex = l - hangul::LBase;
    auto const vIndex = v - hangul::VBase;
    auto const lvIndex = lIndex * hangul::NCount + vIndex * hangul::TCount;

    if (t == 0)
        return hangul::SBase + lvIndex;

    if (!is_hangul_t_jamo(t))
        return 0;

    auto const tIndex = t - hangul::TBase;
    return hangul::SBase + lvIndex + tIndex;
}

// ============================================================================
// Streaming normalizer
// ============================================================================

normalizer::normalizer(Normalization_Form form) noexcept: _form(form)
{
}

bool normalizer::is_boundary(char32_t codepoint) const noexcept
{
    // Non-starters are never boundaries
    if (canonical_combining_class(codepoint) != 0)
        return false;

    // For decomposition forms, every starter is a safe boundary
    if (_form == Normalization_Form::NFD || _form == Normalization_Form::NFKD)
        return true;

    // For composition forms, a starter is safe only if its quick-check value
    // is Yes (it cannot compose with the preceding segment)
    if (_form == Normalization_Form::NFC)
        return nfc_quick_check(codepoint) == NFC_Quick_Check::Yes;

    return nfkc_quick_check(codepoint) == NFKC_Quick_Check::Yes;
}

std::u32string_view normalizer::emit_pending()
{
    if (_pending.empty())
        return {};

    _output = normalize(std::u32string_view(_pending), _form);
    _pending.clear();
    return _output;
}

std::u32string_view normalizer::feed(char32_t codepoint)
{
    if (!_pending.empty() && is_boundary(codepoint))
    {
        auto result = emit_pending();
        _pending.push_back(codepoint);
        return result;
    }

    _pending.push_back(codepoint);
    return {};
}

std::u32string_view normalizer::flush()
{
    return emit_pending();
}

void normalizer::reset() noexcept
{
    _pending.clear();
    _output.clear();
}

// ============================================================================
// UTF-8 streaming normalizer
// ============================================================================

utf8_normalizer::utf8_normalizer(Normalization_Form form) noexcept: _inner(form)
{
}

std::string_view utf8_normalizer::feed(std::string_view utf8Data)
{
    _utf8Output.clear();

    for (auto byte: utf8Data)
    {
        auto const feedCodepoint = [&](char32_t cp) {
            auto segment = _inner.feed(cp);
            if (!segment.empty())
                convert_to<char>(segment, std::back_inserter(_utf8Output));
        };
        // Replaces invalid sequences with U+FFFD; Incomplete continues buffering UTF-8 bytes.
        auto const feedResult = [&](ConvertResult const& result) {
            if (std::holds_alternative<Success>(result))
                feedCodepoint(std::get<Success>(result).value);
            else if (std::holds_alternative<Invalid>(result))
                feedCodepoint(char32_t { 0xFFFD });
        };

        auto const expectedBefore = _utf8State.expectedLength;
        auto const result = from_utf8(_utf8State, static_cast<uint8_t>(byte));
        feedResult(result);
        if (std::holds_alternative<Invalid>(result) && from_utf8_left_byte_unconsumed(expectedBefore, _utf8State))
            feedResult(from_utf8(_utf8State, static_cast<uint8_t>(byte)));
    }

    return _utf8Output;
}

std::string_view utf8_normalizer::flush()
{
    _utf8Output.clear();

    auto segment = _inner.flush();
    if (!segment.empty())
        convert_to<char>(segment, std::back_inserter(_utf8Output));

    return _utf8Output;
}

void utf8_normalizer::reset() noexcept
{
    _inner.reset();
    _utf8State = {};
    _utf8Output.clear();
}

} // namespace unicode
