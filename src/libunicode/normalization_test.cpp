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
#include <libunicode/normalization.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <format>
#include <fstream>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace unicode;
using namespace std::string_view_literals;

TEST_CASE("normalization.canonical_combining_class", "[normalization]")
{
    // Base characters have CCC 0
    CHECK(canonical_combining_class('A') == 0);
    CHECK(canonical_combining_class('a') == 0);
    CHECK(canonical_combining_class(' ') == 0);

    // Combining diacritical marks have non-zero CCC
    CHECK(canonical_combining_class(U'\u0300') == 230); // Combining grave accent
    CHECK(canonical_combining_class(U'\u0301') == 230); // Combining acute accent
    CHECK(canonical_combining_class(U'\u0327') == 202); // Combining cedilla
    CHECK(canonical_combining_class(U'\u0328') == 202); // Combining ogonek
}

TEST_CASE("normalization.hangul_syllable", "[normalization]")
{
    // Hangul syllable "가" (U+AC00) = L (U+1100) + V (U+1161)
    CHECK(is_hangul_syllable(U'\uAC00'));
    CHECK_FALSE(is_hangul_syllable('A'));
    CHECK_FALSE(is_hangul_syllable(U'\u1100')); // Leading jamo

    CHECK(is_hangul_l_jamo(U'\u1100'));
    CHECK(is_hangul_v_jamo(U'\u1161'));
    CHECK(is_hangul_t_jamo(U'\u11A8'));
}

TEST_CASE("normalization.hangul_decompose", "[normalization]")
{
    // "가" (U+AC00) = ᄀ (U+1100) + ᅡ (U+1161)
    char32_t jamos[3];
    auto count = hangul_decompose(U'\uAC00', jamos);
    CHECK(count == 2);
    CHECK(jamos[0] == U'\u1100');
    CHECK(jamos[1] == U'\u1161');

    // "각" (U+AC01) = ᄀ (U+1100) + ᅡ (U+1161) + ᆨ (U+11A8)
    count = hangul_decompose(U'\uAC01', jamos);
    CHECK(count == 3);
    CHECK(jamos[0] == U'\u1100');
    CHECK(jamos[1] == U'\u1161');
    CHECK(jamos[2] == U'\u11A8');
}

TEST_CASE("normalization.hangul_compose", "[normalization]")
{
    // ᄀ (U+1100) + ᅡ (U+1161) = 가 (U+AC00)
    CHECK(hangul_compose(U'\u1100', U'\u1161', 0) == U'\uAC00');

    // ᄀ + ᅡ + ᆨ = 각
    CHECK(hangul_compose(U'\u1100', U'\u1161', U'\u11A8') == U'\uAC01');
}

TEST_CASE("normalization.hangul_compose_tbase_boundary", "[normalization]")
{
    // U+11A7 is JAMO_T_BASE, the codepoint immediately preceding the valid
    // trailing-jamo range -- it is not itself a trailing consonant (that range
    // starts at U+11A8) and must not compose into an LVT syllable.
    CHECK(hangul_compose(U'\u1100', U'\u1161', U'\u11A7') == 0);

    // U+11A8 is the first valid trailing jamo (T index 1).
    CHECK(hangul_compose(U'\u1100', U'\u1161', U'\u11A8') != 0);
}

TEST_CASE("normalization.nfc_keeps_tbase_after_an_lv_syllable", "[normalization]")
{
    // NormalizationTest-18.0.0.txt line 69: U+11A7 HANGUL JUNGSEONG O-YAE sits at TBase, one below the
    // trailing consonants U+11A8..U+11C2, so it never joins the LV syllable before it (T index 0 means
    // "no trailing consonant"). Composing it anyway added T index 0 and silently dropped the U+11A7.
    auto const composed = std::u32string_view { U"\uAC00\u11A7" };
    auto const decomposed = std::u32string_view { U"\u1100\u1161\u11A7" };

    CHECK(to_nfc(composed) == composed);
    CHECK(to_nfc(decomposed) == composed);
    CHECK(to_nfkc(composed) == composed);
    CHECK(to_nfd(composed) == decomposed);
    CHECK(to_nfkd(composed) == decomposed);
    CHECK(is_nfc(composed));
}

TEST_CASE("normalization.nfc_composes_an_lv_syllable_with_exactly_the_trailing_consonants", "[normalization]")
{
    // Unicode Standard, section 3.12, Hangul Syllable Composition: <LVPart, TPart> maps to LVPart + TIndex
    // only where TPart is in U+11A8..U+11C2 (0 < TIndex < TCount). Every LV syllable, and the codepoints
    // just outside that range on both sides, are checked.
    constexpr auto SBase = char32_t { 0xAC00 };
    constexpr auto TBase = char32_t { 0x11A7 };
    constexpr auto TCount = char32_t { 28 };
    constexpr auto LVCount = char32_t { 19 * 21 };

    for (auto const lvIndex: std::views::iota(char32_t { 0 }, LVCount))
    {
        auto const lv = static_cast<char32_t>(SBase + (lvIndex * TCount));
        for (auto const t: std::views::iota(char32_t { TBase - 1 }, char32_t { TBase + TCount + 1 }))
        {
            auto const input = std::u32string { lv, t };
            auto const isTrailingConsonant = t > TBase && t < TBase + TCount;
            auto const lvt = std::u32string(1, static_cast<char32_t>(lv + (t - TBase)));
            auto const expected = isTrailingConsonant ? lvt : input;
            INFO(std::format("U+{:04X} U+{:04X}", static_cast<uint32_t>(lv), static_cast<uint32_t>(t)));
            CHECK(to_nfc(input) == expected);
            CHECK(to_nfkc(input) == expected);
        }
    }
}

TEST_CASE("normalization.canonical_decomposition", "[normalization]")
{
    // é (U+00E9) decomposes to e + combining acute accent
    auto decomp = canonical_decomposition(U'\u00E9');
    REQUIRE(decomp.size() == 2);
    CHECK(decomp[0] == 'e');
    CHECK(decomp[1] == U'\u0301');

    // ñ (U+00F1) decomposes to n + combining tilde
    decomp = canonical_decomposition(U'\u00F1');
    REQUIRE(decomp.size() == 2);
    CHECK(decomp[0] == 'n');
    CHECK(decomp[1] == U'\u0303');

    // ASCII has no decomposition
    decomp = canonical_decomposition('A');
    CHECK(decomp.empty());
}

TEST_CASE("normalization.tibetan_recursive_compatibility_decomposition", "[normalization]")
{
    // U+0F77 TIBETAN VOWEL SIGN VOCALIC RR has a *compatibility* decomposition
    // to U+0FB2 U+0F81 (per UnicodeData.txt), and U+0F81 itself further
    // canonically decomposes to U+0F71 U+0F80. NFKD must recurse through both
    // levels, giving a final 3-codepoint result -- exercising the same
    // two-level recursive decomposition ICU's tstnorm.cpp TestTibetan checks.
    auto const result = to_nfkd(std::u32string_view { U"ཷ" });
    CHECK(result == std::u32string { U"ྲཱྀ" });
}

TEST_CASE("normalization.to_nfd", "[normalization]")
{
    // é -> e + combining acute
    auto result = to_nfd(U"\u00E9");
    CHECK(result == U"e\u0301");

    // Already decomposed stays the same
    result = to_nfd(U"e\u0301");
    CHECK(result == U"e\u0301");

    // ASCII is unchanged
    result = to_nfd(U"hello");
    CHECK(result == U"hello");
}

TEST_CASE("normalization.to_nfc", "[normalization]")
{
    // e + combining acute -> é
    auto result = to_nfc(U"e\u0301");
    CHECK(result == U"\u00E9");

    // Already composed stays the same
    result = to_nfc(U"\u00E9");
    CHECK(result == U"\u00E9");

    // ASCII is unchanged
    result = to_nfc(U"hello");
    CHECK(result == U"hello");
}

TEST_CASE("normalization.to_nfd_utf8", "[normalization]")
{
    // UTF-8 version
    auto result = to_nfd("café"sv);
    // café with composed é becomes café with decomposed e + accent
    CHECK(result.length() > 4); // Should be longer due to decomposition
}

TEST_CASE("normalization.to_nfc_utf8", "[normalization]")
{
    // Compose
    auto nfd_input = to_nfd("café"sv);
    auto result = to_nfc(nfd_input);
    CHECK(result == "café");
}

TEST_CASE("normalization.quick_check", "[normalization]")
{
    // ASCII is always normalized
    CHECK(quick_check(U"hello", Normalization_Form::NFC) == Quick_Check_Result::Yes);
    CHECK(quick_check(U"hello", Normalization_Form::NFD) == Quick_Check_Result::Yes);

    // Pre-composed character
    CHECK(quick_check(U"\u00E9", Normalization_Form::NFC) == Quick_Check_Result::Yes);
    // Note: full is_normalized check works correctly even if quick_check is lenient
}

TEST_CASE("normalization.is_normalized", "[normalization]")
{
    // ASCII
    CHECK(is_nfc(U"hello"));
    CHECK(is_nfd(U"hello"));

    // Pre-composed é is NFC but not NFD
    CHECK(is_nfc(U"\u00E9"));

    // Decomposed e + accent is NFD but might not be NFC
    CHECK(is_nfd(U"e\u0301"));
}

TEST_CASE("normalization.is_canonically_equivalent", "[normalization]")
{
    // é composed vs decomposed
    CHECK(is_canonically_equivalent(U"\u00E9", U"e\u0301"));

    // Same string
    CHECK(is_canonically_equivalent(U"hello", U"hello"));

    // Different strings
    CHECK_FALSE(is_canonically_equivalent(U"hello", U"world"));
}

TEST_CASE("normalization.is_canonically_equivalent_utf8", "[normalization]")
{
    // UTF-8 versions
    CHECK(is_canonically_equivalent("hello"sv, "hello"sv));
    CHECK_FALSE(is_canonically_equivalent("hello"sv, "world"sv));
}

TEST_CASE("normalization.canonical_ordering", "[normalization]")
{
    // Multiple combining marks should be reordered by CCC
    // This tests that marks with lower CCC come before marks with higher CCC
    auto result = to_nfd(U"a\u0327\u0301"); // a + cedilla + acute
    // After canonical ordering, the cedilla (CCC=202) should come before acute (CCC=230)
    // Wait, that's backwards... let me check. Actually 202 < 230, so cedilla should come first.
    // But the NFD ordering is by CCC value, lower first.

    // Let's test with a character that has marks in wrong order
    auto input = U"a\u0301\u0327"; // acute (230) before cedilla (202) - wrong order
    result = to_nfd(input);
    // After normalization, cedilla should come first
    REQUIRE(result.size() == 3);
    CHECK(result[0] == 'a');
    // The order of combining marks should be by CCC (202 before 230)
    CHECK(canonical_combining_class(result[1]) <= canonical_combining_class(result[2]));
}

TEST_CASE("normalization.is_composition_exclusion", "[normalization]")
{
    // U+0344 COMBINING GREEK DIALYTIKA TONOS is a Full_Composition_Exclusion
    // (DerivedNormalizationProps.txt: "0343..0344 ; Full_Composition_Exclusion"),
    // even though it isn't listed in the smaller "primary" CompositionExclusions.txt.
    CHECK(is_composition_exclusion(U'\u0344'));

    // A character with no decomposition at all is trivially not an exclusion.
    CHECK_FALSE(is_composition_exclusion('A'));

    // U+00C0 (LATIN CAPITAL LETTER A WITH GRAVE) has a canonical decomposition
    // and is NOT excluded -- it's a normal composable character.
    CHECK_FALSE(is_composition_exclusion(U'\u00c0'));
}

TEST_CASE("normalization.decomposition_type", "[normalization]")
{
    // Pre-composed characters have canonical decomposition
    CHECK(decomposition_type(U'\u00E9') == Decomposition_Type::Canonical);

    // ASCII has no decomposition
    CHECK(decomposition_type('A') == Decomposition_Type::None);

    // Hangul syllables have canonical decomposition
    CHECK(decomposition_type(U'\uAC00') == Decomposition_Type::Canonical);
}

// ============================================================================
// NFKC/NFKD tests (compatibility decomposition)
// ============================================================================

TEST_CASE("normalization.to_nfkd", "[normalization]")
{
    // fi ligature (U+FB01) decomposes under NFKD to "fi"
    auto result = to_nfkd(U"\uFB01");
    CHECK(result == U"fi");

    // Superscript 2 (U+00B2) decomposes to "2"
    result = to_nfkd(U"\u00B2");
    CHECK(result == U"2");

    // Fullwidth A (U+FF21) decomposes to "A"
    result = to_nfkd(U"\uFF21");
    CHECK(result == U"A");

    // Canonical decomposition still works in NFKD
    result = to_nfkd(U"\u00E9");
    CHECK(result == U"e\u0301");

    // ASCII is unchanged
    result = to_nfkd(U"hello");
    CHECK(result == U"hello");
}

TEST_CASE("normalization.to_nfkc", "[normalization]")
{
    // fi ligature becomes "fi"
    auto result = to_nfkc(U"\uFB01");
    CHECK(result == U"fi");

    // Superscript 2 becomes "2"
    result = to_nfkc(U"\u00B2");
    CHECK(result == U"2");

    // Fullwidth A becomes "A"
    result = to_nfkc(U"\uFF21");
    CHECK(result == U"A");

    // Composed character stays composed
    result = to_nfkc(U"\u00E9");
    CHECK(result == U"\u00E9");

    // Decomposed character gets recomposed
    result = to_nfkc(U"e\u0301");
    CHECK(result == U"\u00E9");

    // ASCII is unchanged
    result = to_nfkc(U"hello");
    CHECK(result == U"hello");
}

TEST_CASE("normalization.nfkd_does_not_equal_nfd", "[normalization]")
{
    // fi ligature: NFD leaves it unchanged, NFKD decomposes it
    CHECK(to_nfd(U"\uFB01") == U"\uFB01");
    CHECK(to_nfkd(U"\uFB01") == U"fi");
}

TEST_CASE("normalization.compatibility_equivalence", "[normalization]")
{
    CHECK(is_compatibility_equivalent(U"\uFB01", U"fi"));
    CHECK(is_compatibility_equivalent(U"\u00B2", U"2"));
    CHECK(is_compatibility_equivalent(U"\uFF21", U"A"));
}

// ============================================================================
// Quick check tests (with populated tables)
// ============================================================================

TEST_CASE("normalization.quick_check_nfd", "[normalization]")
{
    // Composed e-acute: NFD_QC=No (has canonical decomposition)
    CHECK(nfd_quick_check(U'\u00E9') == false);

    // ASCII: NFD_QC=Yes
    CHECK(nfd_quick_check('A') == true);

    // Already-decomposed combining mark: NFD_QC=Yes
    CHECK(nfd_quick_check(U'\u0301') == true);
}

TEST_CASE("normalization.quick_check_nfc", "[normalization]")
{
    // Combining acute accent: NFC_QC=Maybe (it can compose with a preceding starter)
    CHECK(nfc_quick_check(U'\u0301') == NFC_Quick_Check::Maybe);

    // ASCII: NFC_QC=Yes
    CHECK(nfc_quick_check('A') == NFC_Quick_Check::Yes);
}

TEST_CASE("normalization.quick_check_nfkd", "[normalization]")
{
    // fi ligature: NFKD_QC=No (has compatibility decomposition)
    CHECK(nfkd_quick_check(U'\uFB01') == false);

    // Composed e-acute: NFKD_QC=No (has canonical decomposition)
    CHECK(nfkd_quick_check(U'\u00E9') == false);

    // ASCII: NFKD_QC=Yes
    CHECK(nfkd_quick_check('A') == true);
}

TEST_CASE("normalization.quick_check_nfkc", "[normalization]")
{
    // fi ligature: NFKC_QC=No
    CHECK(nfkc_quick_check(U'\uFB01') == NFKC_Quick_Check::No);

    // ASCII: NFKC_QC=Yes
    CHECK(nfkc_quick_check('A') == NFKC_Quick_Check::Yes);
}

TEST_CASE("normalization.quick_check_string", "[normalization]")
{
    // fi ligature: NFC=Yes (no canonical decomposition), NFKC=No
    CHECK(quick_check(U"\uFB01", Normalization_Form::NFC) == Quick_Check_Result::Yes);
    CHECK(quick_check(U"\uFB01", Normalization_Form::NFKC) == Quick_Check_Result::No);
    CHECK(quick_check(U"\uFB01", Normalization_Form::NFD) == Quick_Check_Result::Yes);
    CHECK(quick_check(U"\uFB01", Normalization_Form::NFKD) == Quick_Check_Result::No);
}

// ============================================================================
// Decomposition type for compatibility types
// ============================================================================

TEST_CASE("normalization.decomposition_type_compat", "[normalization]")
{
    // fi ligature has compat decomposition
    CHECK(decomposition_type(U'\uFB01') == Decomposition_Type::Compat);

    // Superscript 2 has super decomposition
    CHECK(decomposition_type(U'\u00B2') == Decomposition_Type::Super);

    // Fullwidth A has wide decomposition
    CHECK(decomposition_type(U'\uFF21') == Decomposition_Type::Wide);

    // Halfwidth katakana wo (U+FF66) has narrow decomposition
    CHECK(decomposition_type(U'\uFF66') == Decomposition_Type::Narrow);

    // Circled digit 1 (U+2460) has circle decomposition
    CHECK(decomposition_type(U'\u2460') == Decomposition_Type::Circle);
}

// ============================================================================
// Streaming normalizer
// ============================================================================

TEST_CASE("normalization_stream.nfc_basic", "[normalization]")
{
    normalizer norm(Normalization_Form::NFC);

    // Feed decomposed e + combining acute, then a new starter to trigger emission
    auto result = norm.feed(U'e');
    CHECK(result.empty()); // still buffering

    result = norm.feed(U'\u0301');
    CHECK(result.empty()); // still buffering (combining mark)

    result = norm.feed(U'x'); // starter triggers emission of previous segment
    REQUIRE_FALSE(result.empty());
    CHECK(result == U"\u00E9"); // e + acute composed to e-acute

    result = norm.flush();
    CHECK(result == U"x"); // final segment
}

TEST_CASE("normalization_stream.nfd_basic", "[normalization]")
{
    normalizer norm(Normalization_Form::NFD);

    auto result = norm.feed(U'\u00E9'); // pre-composed e-acute
    CHECK(result.empty());              // first codepoint, no boundary yet

    result = norm.flush();
    CHECK(result == U"e\u0301"); // decomposed
}

TEST_CASE("normalization_stream.empty_input", "[normalization]")
{
    normalizer norm(Normalization_Form::NFC);

    auto result = norm.flush();
    CHECK(result.empty());
}

TEST_CASE("normalization_stream.all_starters", "[normalization]")
{
    normalizer norm(Normalization_Form::NFC);
    std::u32string output;

    for (char32_t cp: std::u32string_view(U"hello"))
    {
        auto segment = norm.feed(cp);
        output.append(segment);
    }
    output.append(norm.flush());

    CHECK(output == U"hello");
}

TEST_CASE("normalization_stream.all_combining", "[normalization]")
{
    normalizer norm(Normalization_Form::NFD);

    // All combining marks, no starters — everything buffers until flush
    (void) norm.feed(U'\u0301'); // CCC 230
    (void) norm.feed(U'\u0327'); // CCC 202

    auto result = norm.flush();
    // After normalization, canonical ordering: cedilla (202) before acute (230)
    REQUIRE(result.size() == 2);
    CHECK(result[0] == U'\u0327');
    CHECK(result[1] == U'\u0301');
}

TEST_CASE("normalization_stream.hangul_composition", "[normalization]")
{
    normalizer norm(Normalization_Form::NFC);
    std::u32string output;

    // Feed L + V jamos
    output.append(norm.feed(U'\u1100')); // L jamo
    output.append(norm.feed(U'\u1161')); // V jamo
    output.append(norm.flush());

    CHECK(output == U"\uAC00"); // Composed syllable
}

TEST_CASE("normalization_stream.equivalence_with_batch", "[normalization]")
{
    std::u32string input = U"caf\u00E9 na\u00EFve r\u00E9sum\u00E9";

    for (auto form: { Normalization_Form::NFC, Normalization_Form::NFD, Normalization_Form::NFKC, Normalization_Form::NFKD })
    {
        auto batchResult = normalize(std::u32string_view(input), form);

        std::u32string streamResult;
        normalizer norm(form);
        for (char32_t cp: input)
        {
            auto segment = norm.feed(cp);
            streamResult.append(segment);
        }
        streamResult.append(norm.flush());

        CHECK(streamResult == batchResult);
    }
}

TEST_CASE("normalization_stream.reset", "[normalization]")
{
    normalizer norm(Normalization_Form::NFC);

    (void) norm.feed(U'e');
    (void) norm.feed(U'\u0301');
    norm.reset(); // discard buffered data

    auto result = norm.flush();
    CHECK(result.empty());
}

TEST_CASE("normalization_stream.multiple_segments", "[normalization]")
{
    normalizer norm(Normalization_Form::NFC);
    std::u32string output;

    // "café" with decomposed e-acute
    for (char32_t cp: std::u32string_view(U"cafe\u0301"))
    {
        auto segment = norm.feed(cp);
        output.append(segment);
    }
    output.append(norm.flush());

    CHECK(output == U"caf\u00E9");
}

// ============================================================================
// UTF-8 streaming normalizer
// ============================================================================

TEST_CASE("normalization_stream.utf8_basic", "[normalization]")
{
    utf8_normalizer norm(Normalization_Form::NFC);

    // Feed "cafe" + combining acute (UTF-8: CC 81) + "!" to trigger final segment
    std::string output;
    output += norm.feed("cafe\xCC\x81!");
    output += norm.flush();

    CHECK(output == "caf\xC3\xA9!"); // "café!"
}

TEST_CASE("normalization_stream.utf8_byte_by_byte", "[normalization]")
{
    utf8_normalizer norm(Normalization_Form::NFC);

    // Feed composed é (UTF-8: C3 A9) byte by byte, then "x"
    std::string input = "\xC3\xA9x";
    std::string output;
    for (char c: input)
        output += norm.feed(std::string_view(&c, 1));
    output += norm.flush();

    CHECK(output == "\xC3\xA9x");
}

TEST_CASE("normalization_stream.utf8_reset", "[normalization]")
{
    utf8_normalizer norm(Normalization_Form::NFC);

    (void) norm.feed("hello");
    norm.reset();

    auto result = norm.flush();
    CHECK(result.empty());
}

// ============================================================================
// Conformance test vectors (representative subset from NormalizationTest.txt)
// ============================================================================

TEST_CASE("normalization.conformance_subset", "[normalization]")
{
    struct TestCase
    {
        std::u32string source;
        std::u32string nfc;
        std::u32string nfd;
        std::u32string nfkc;
        std::u32string nfkd;
    };

    // clang-format off
    auto const testVectors = std::vector<TestCase> {
        // Basic canonical decomposition/composition
        { U"\u00E9",            U"\u00E9",            U"e\u0301",            U"\u00E9",            U"e\u0301" },
        { U"e\u0301",           U"\u00E9",            U"e\u0301",            U"\u00E9",            U"e\u0301" },
        // fi ligature (compatibility only)
        { U"\uFB01",            U"\uFB01",            U"\uFB01",            U"fi",                U"fi" },
        // Fullwidth A
        { U"\uFF21",            U"\uFF21",            U"\uFF21",            U"A",                 U"A" },
        // Superscript 2
        { U"\u00B2",            U"\u00B2",            U"\u00B2",            U"2",                 U"2" },
        // Hangul syllable
        { U"\uAC00",            U"\uAC00",            U"\u1100\u1161",      U"\uAC00",            U"\u1100\u1161" },
        { U"\uAC01",            U"\uAC01",            U"\u1100\u1161\u11A8", U"\uAC01",           U"\u1100\u1161\u11A8" },
        // Hangul jamos -> syllable
        { U"\u1100\u1161",      U"\uAC00",            U"\u1100\u1161",      U"\uAC00",            U"\u1100\u1161" },
        { U"\u1100\u1161\u11A8", U"\uAC01",           U"\u1100\u1161\u11A8", U"\uAC01",           U"\u1100\u1161\u11A8" },
    };
    // clang-format on

    for (size_t i = 0; i < testVectors.size(); ++i)
    {
        auto const& tc = testVectors[i];
        INFO("Test vector index: " << i);

        CHECK(to_nfc(tc.source) == tc.nfc);
        CHECK(to_nfd(tc.source) == tc.nfd);
        CHECK(to_nfkc(tc.source) == tc.nfkc);
        CHECK(to_nfkd(tc.source) == tc.nfkd);
    }
}

// ============================================================================
// Official Unicode Conformance Tests (live parse of NormalizationTest.txt)
// ============================================================================

namespace
{

/// Parse a whitespace-separated list of hex codepoints, e.g. "1E0A 0323", into a u32string.
std::u32string parse_hex_codepoints(std::string_view field)
{
    std::u32string result;
    size_t i = 0;
    while (i < field.size())
    {
        while (i < field.size() && field[i] == ' ')
            ++i;
        auto start = i;
        while (i < field.size() && field[i] != ' ')
            ++i;
        if (i > start)
        {
            unsigned long value = 0;
            for (auto j = start; j < i; ++j)
            {
                auto c = field[j];
                value <<= 4;
                if (c >= '0' && c <= '9')
                    value |= static_cast<unsigned long>(c - '0');
                else if (c >= 'A' && c <= 'F')
                    value |= static_cast<unsigned long>(c - 'A' + 10);
                else if (c >= 'a' && c <= 'f')
                    value |= static_cast<unsigned long>(c - 'a' + 10);
            }
            result.push_back(static_cast<char32_t>(value));
        }
    }
    return result;
}

struct NormalizationTestCase
{
    std::u32string c1, c2, c3, c4, c5;
    std::string comment;
    int line_number = 0;
};

/// Parse NormalizationTest.txt into test cases.
/// Format: c1;c2;c3;c4;c5; # comment
/// @Part headers and blank/comment lines are skipped; every data line uses the
/// same 5-column conformance invariants regardless of which Part it came from.
std::vector<NormalizationTestCase> parse_normalization_test_file(std::string const& path)
{
    std::vector<NormalizationTestCase> cases;
    std::ifstream file(path);
    if (!file.is_open())
        return cases;

    std::string line;
    auto lineNo = 0;
    while (std::getline(file, line))
    {
        ++lineNo;
        if (line.empty() || line[0] == '#' || line[0] == '@')
            continue;

        NormalizationTestCase tc;
        tc.line_number = lineNo;

        auto commentPos = line.find('#');
        auto dataStr = (commentPos != std::string::npos) ? line.substr(0, commentPos) : line;
        if (commentPos != std::string::npos)
            tc.comment = line.substr(commentPos);

        std::vector<std::string> fields;
        size_t start = 0;
        for (size_t i = 0; i <= dataStr.size(); ++i)
        {
            if (i == dataStr.size() || dataStr[i] == ';')
            {
                fields.push_back(dataStr.substr(start, i - start));
                start = i + 1;
            }
        }

        if (fields.size() < 5)
            continue;

        tc.c1 = parse_hex_codepoints(fields[0]);
        tc.c2 = parse_hex_codepoints(fields[1]);
        tc.c3 = parse_hex_codepoints(fields[2]);
        tc.c4 = parse_hex_codepoints(fields[3]);
        tc.c5 = parse_hex_codepoints(fields[4]);

        if (!tc.c1.empty())
            cases.push_back(std::move(tc));
    }
    return cases;
}

} // namespace

// ============================================================================
// Cases ported from icu4x components/normalizer/tests/tests.rs
// (test_nfd_basic / test_nfkd_basic / test_nfc_basic / test_nfkc_basic,
// test_accented_digraph, test_ddd).
// ============================================================================

TEST_CASE("normalization.icu4x_singleton_decompositions", "[normalization]")
{
    // Ohm sign (U+2126) is a singleton canonical decomposition to Omega (U+03A9).
    CHECK(to_nfd(std::u32string_view { U"\U00002126" }) == std::u32string { U"\U000003A9" });
    CHECK(to_nfkd(std::u32string_view { U"\U00002126" }) == std::u32string { U"\U000003A9" });
    CHECK(to_nfc(std::u32string_view { U"\U00002126" }) == std::u32string { U"\U000003A9" });
    CHECK(to_nfkc(std::u32string_view { U"\U00002126" }) == std::u32string { U"\U000003A9" });

    // Angstrom sign (U+212B) has a single-step ("singleton") canonical
    // decomposition to A-with-ring (U+00C5) per UnicodeData.txt. This is the
    // one-step mapping exercised by icu4x's CanonicalDecomposition::decompose
    // (Decomposed::Singleton) and matches libunicode's canonical_decomposition().
    // Note this is NOT what to_nfd() returns for U+212B: NFD is fully
    // recursive, and U+00C5 itself further decomposes to U+0041 U+030A, so
    // to_nfd(U+212B) correctly yields "A" + combining ring above, not U+00C5.
    CHECK(canonical_decomposition(U'\U0000212B') == std::vector<char32_t> { U'\U000000C5' });
    CHECK(to_nfd(std::u32string_view { U"\U0000212B" }) == std::u32string { U"\U00000041\U0000030A" });

    // Half-width katakana HE (U+FF8D) + half-width voiced sound mark (U+FF9E):
    // unchanged under NFD/NFC, but NFKD maps it to full-width kana HE
    // (U+30D8) followed by a combining sound mark (U+3099), and NFKC composes
    // that pair into the fully composed full-width kana BE (U+30D9).
    CHECK(to_nfd(std::u32string_view { U"\U0000FF8D\U0000FF9E" }) == std::u32string { U"\U0000FF8D\U0000FF9E" });
    CHECK(to_nfc(std::u32string_view { U"\U0000FF8D\U0000FF9E" }) == std::u32string { U"\U0000FF8D\U0000FF9E" });
    CHECK(to_nfkd(std::u32string_view { U"\U0000FF8D\U0000FF9E" }) == std::u32string { U"\U000030D8\U00003099" });
    CHECK(to_nfkc(std::u32string_view { U"\U0000FF8D\U0000FF9E" }) == std::u32string { U"\U000030D9" });

    // "fi" ligature (U+FB01): unchanged under NFD/NFC, expands to "fi" under NFKD/NFKC.
    CHECK(to_nfd(std::u32string_view { U"\U0000FB01" }) == std::u32string { U"\U0000FB01" });
    CHECK(to_nfc(std::u32string_view { U"\U0000FB01" }) == std::u32string { U"\U0000FB01" });
    CHECK(to_nfkd(std::u32string_view { U"\U0000FB01" }) == std::u32string { U"\U00000066\U00000069" });
    CHECK(to_nfkc(std::u32string_view { U"\U0000FB01" }) == std::u32string { U"\U00000066\U00000069" });

    // Arabic ligature U+FDFA ("Sallallahou Alayhi Wasallam"): unchanged under
    // NFD/NFC, expands to its 18-codepoint compatibility sequence under NFKD/NFKC.
    CHECK(to_nfd(std::u32string_view { U"\U0000FDFA" }) == std::u32string { U"\U0000FDFA" });
    CHECK(to_nfkd(std::u32string_view { U"\U0000FDFA" })
          == std::u32string { U"\U00000635\U00000644\U00000649\U00000020\U00000627\U00000644\U00000644\U00000647"
                              U"\U00000020\U00000639\U00000644\U0000064A\U00000647\U00000020\U00000648\U00000633"
                              U"\U00000644\U00000645" });

    // Iota subscript (U+0345) never decomposes under any of the four forms
    // (it is only remapped by UTS46, which libunicode does not implement).
    CHECK(to_nfd(std::u32string_view { U"\U00000345" }) == std::u32string { U"\U00000345" });
    CHECK(to_nfc(std::u32string_view { U"\U00000345" }) == std::u32string { U"\U00000345" });
    CHECK(to_nfkd(std::u32string_view { U"\U00000345" }) == std::u32string { U"\U00000345" });
    CHECK(to_nfkc(std::u32string_view { U"\U00000345" }) == std::u32string { U"\U00000345" });
}

TEST_CASE("normalization.icu4x_accented_digraph", "[normalization]")
{
    // U+01C4 (LATIN CAPITAL LETTER DZ WITH CARON) + combining dot-below
    // (U+0323), under NFKD, decomposes to "DZ" followed by the two combining
    // marks in canonical order (dot-below U+0323, CCC=220, before caron
    // U+030C, CCC=230).
    CHECK(to_nfkd(std::u32string_view { U"\U000001C4\U00000323" })
          == std::u32string { U"\U00000044\U0000005A\U00000323\U0000030C" });

    // Feeding the marks in the opposite order yields the same canonically-ordered result.
    CHECK(to_nfkd(std::u32string_view { U"\U00000044\U0000005A\U0000030C\U00000323" })
          == std::u32string { U"\U00000044\U0000005A\U00000323\U0000030C" });
}

TEST_CASE("normalization.icu4x_sinhala_ddd", "[normalization]")
{
    // Sinhala U+0DDD (SINHALA VOWEL SIGN KOMBUVA HAA GAYANUKITTA) followed by
    // U+0334 (combining tilde overlay) expands under NFD into a 4-codepoint
    // sequence: the multi-codepoint canonical decomposition of U+0DDD
    // (U+0DD9 U+0DCF) interleaved with the combining mark, canonically
    // reordered with the trailing U+0DCA.
    CHECK(to_nfd(std::u32string_view { U"\U00000DDD\U00000334" })
          == std::u32string { U"\U00000DD9\U00000DCF\U00000334\U00000DCA" });
}

TEST_CASE("normalization.unicode_conformance", "[normalization]")
{
    auto const path = std::string(LIBUNICODE_UCD_DIR) + "/NormalizationTest.txt";
    auto const testCases = parse_normalization_test_file(path);

    if (testCases.empty())
    {
        WARN("Skipping conformance tests: NormalizationTest.txt not found at " << path);
        return;
    }
    INFO("Loaded " << testCases.size() << " test cases from NormalizationTest.txt");

    for (auto const& tc: testCases)
    {
        // NFC: c2 == NFC(c1) == NFC(c2) == NFC(c3)
        if (to_nfc(tc.c1) != tc.c2 || to_nfc(tc.c2) != tc.c2 || to_nfc(tc.c3) != tc.c2)
            FAIL("NFC mismatch at line " << tc.line_number << ": " << tc.comment);

        // NFD: c3 == NFD(c1) == NFD(c2) == NFD(c3)
        if (to_nfd(tc.c1) != tc.c3 || to_nfd(tc.c2) != tc.c3 || to_nfd(tc.c3) != tc.c3)
            FAIL("NFD mismatch at line " << tc.line_number << ": " << tc.comment);

        // NFKC: c4 == NFKC(c1) == NFKC(c2) == NFKC(c3) == NFKC(c4) == NFKC(c5)
        if (to_nfkc(tc.c1) != tc.c4 || to_nfkc(tc.c2) != tc.c4 || to_nfkc(tc.c3) != tc.c4 || to_nfkc(tc.c4) != tc.c4
            || to_nfkc(tc.c5) != tc.c4)
            FAIL("NFKC mismatch at line " << tc.line_number << ": " << tc.comment);

        // NFKD: c5 == NFKD(c1) == NFKD(c2) == NFKD(c3) == NFKD(c4) == NFKD(c5)
        if (to_nfkd(tc.c1) != tc.c5 || to_nfkd(tc.c2) != tc.c5 || to_nfkd(tc.c3) != tc.c5 || to_nfkd(tc.c4) != tc.c5
            || to_nfkd(tc.c5) != tc.c5)
            FAIL("NFKD mismatch at line " << tc.line_number << ": " << tc.comment);
    }
}

TEST_CASE("normalization_stream.utf8_ill_formed_byte_is_fed_again_as_itself", "[normalization]")
{
    auto const run = [](std::string_view input) {
        utf8_normalizer norm(Normalization_Form::NFC);
        auto output = std::string(norm.feed(input));
        output += norm.flush();
        return output;
    };

    // C3 is cut off by 'A', which must not be lost.
    CHECK(run("\xC3"
              "A")
          == "\xEF\xBF\xBD"
             "A");
    // Maximal subparts: C2, C0, C2 yield three replacement characters.
    CHECK(run("\xC2\xC0\xC2"
              "A")
          == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"
             "A");
}
