/**
 * This file is part of the "libterminal" project
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
#include <libunicode/convert.h>
#include <libunicode/scan.h>
#include <libunicode/utf8.h>
#include <libunicode/width.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using std::string_view;

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace
{

auto constexpr FamilyEmoji = U"\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466"sv;
auto constexpr SmileyEmoji = U"\U0001F600"sv;
auto constexpr CopyrightSign = U"\u00A9"sv;
auto constexpr ControlCodes = "\r\n"sv; // Used to ensure that C0 codes properly cancel scanning.

template <typename T>
auto u8(T text)
{
    return unicode::convert_to<char>(text);
}

std::string escape(uint8_t ch)
{
    switch (ch)
    {
        case '\\': return "\\\\";
        case 0x1B: return "\\e";
        case '\t': return "\\t";
        case '\r': return "\\r";
        case '\n': return "\\n";
        case '"': return "\\\"";
        default:
            if (ch < 0x20)
                return std::format("\\{:03o}", static_cast<uint8_t>(ch) & 0xFF);
            else if (ch < 0x80)
                return std::format("{}", static_cast<char>(ch));
            else
                return std::format("\\x{:02x}", static_cast<uint8_t>(ch) & 0xFF);
    }
}

template <typename T>
std::string escape(T begin, T end)
{
    static_assert(sizeof(*std::declval<T>()) == 1, "should be only 1 byte, such as: char, char8_t, uint8_t, byte, ...");
    auto result = std::string {};
    while (begin != end)
        result += escape(static_cast<uint8_t>(*begin++));
    return result;
}

inline std::string escape(std::string_view s)
{
    return escape(begin(s), end(s));
}

class grapheme_cluster_collector final: public unicode::grapheme_cluster_receiver
{
  public:
    std::vector<std::u32string> output;

    void receiveAsciiSequence(std::string_view sequence) noexcept override
    {
        for (char const ch: sequence)
            output.emplace_back(1, static_cast<char32_t>(ch));
    }

    void receiveGraphemeCluster(std::string_view cluster, size_t) noexcept override
    {
        output.emplace_back(unicode::convert_to<char32_t>(cluster));
    }

    void receiveInvalidGraphemeCluster() noexcept override
    {
        auto constexpr ReplacementCharacter = U'\uFFFD';
        output.emplace_back(1, ReplacementCharacter);
    }
};

} // namespace

using unicode::detail::scan_for_text_ascii;

TEST_CASE("scan.ascii.empty")
{
    CHECK(scan_for_text_ascii("", 0) == 0);
    CHECK(scan_for_text_ascii("", 1) == 0);
}

TEST_CASE("scan.ascii.32")
{
    auto const text = "0123456789ABCDEF0123456789ABCDEF"sv;
    CHECK(scan_for_text_ascii(text, 32) == 32);
    CHECK(scan_for_text_ascii(text, 16) == 16);
    CHECK(scan_for_text_ascii(text, 8) == 8);
    CHECK(scan_for_text_ascii(text, 1) == 1);
}

TEST_CASE("scan.ascii.64")
{
    auto const text = "0123456789ABCDEF0123456789ABCDEF"
                      "0123456789ABCDEF0123456789ABCDEF"sv;
    CHECK(scan_for_text_ascii(text, 64) == 64);
    CHECK(scan_for_text_ascii(text, 32) == 32);
    CHECK(scan_for_text_ascii(text, 16) == 16);
    CHECK(scan_for_text_ascii(text, 8) == 8);
    CHECK(scan_for_text_ascii(text, 1) == 1);
}

TEST_CASE("scan.ascii.128")
{
    auto const text = "0123456789ABCDEF0123456789ABCDEF"
                      "0123456789ABCDEF0123456789ABCDEF"
                      "0123456789ABCDEF0123456789ABCDEF"
                      "0123456789ABCDEF0123456789ABCDEF"sv;
    CHECK(scan_for_text_ascii(text, 128) == 128);
    CHECK(scan_for_text_ascii(text, 64) == 64);
    CHECK(scan_for_text_ascii(text, 32) == 32);
    CHECK(scan_for_text_ascii(text, 16) == 16);
    CHECK(scan_for_text_ascii(text, 8) == 8);
    CHECK(scan_for_text_ascii(text, 1) == 1);
}

TEST_CASE("scan.ascii.mixed_with_controls")
{
    CHECK(scan_for_text_ascii("\0331234", 80) == 0);
    CHECK(scan_for_text_ascii("1234\033", 80) == 4);
    CHECK(scan_for_text_ascii("12345678\033", 80) == 8);
    CHECK(scan_for_text_ascii("0123456789ABCDEF\033", 80) == 16);
    CHECK(scan_for_text_ascii("0123456789ABCDEF1\033", 80) == 17);
    auto text = "0123456789ABCDEF0\033123456789ABCDEF"
                "0123456789ABCDEF0123456789ABCDEF"
                "0123456789ABCDEF0123456789ABCDEF"
                "0123456789ABCDEF0123456789ABCDEF"sv;
    CHECK(scan_for_text_ascii(text, 80) == 17);
    text = "0123456789ABCDEF0123456789ABCDEF"
           "0123456789ABCDEF0123456789ABCDEF"
           "0123456789ABCDEF0123456789ABCDEF"
           "0123456789ABCDEF\0330123456789ABCDEF"sv;
    CHECK(scan_for_text_ascii(text, 128) == 112);
}

TEST_CASE("scan.ascii.until_complex")
{
    CHECK(scan_for_text_ascii("1234\x80", 80) == 4);
    CHECK(scan_for_text_ascii("0123456789{\xE2\x94\x80}ABCDEF", 80) == 11);
    constexpr auto text = "0123456789{\xE2\x94\x80}ABCDEF0323456789ABCDEF"
                          "0123456789ABCDEF0123456789ABCDEF"
                          "0123456789ABCDEF0123456789ABCDEF"
                          "0123456789ABCDEF0123456789ABCDEF"sv;
    CHECK(scan_for_text_ascii(text, 80) == 11);
}

namespace
{

// The largest offset and length the per-width scanner tests try: three whole 512-bit vectors and part of a fourth.
auto constexpr MaxTestedLength = size_t { 200 };

// One whole 512-bit vector longer than any offset or length tested, so that a stop byte at any tested offset has text
// after it and lies inside a full vector load of every width rather than in the scalar tail.
auto constexpr TestTextLength = MaxTestedLength + (512 / 8);

} // namespace

TEST_CASE("scan.ascii.every_simd_width_stops_exactly_at_a_non_text_byte")
{
    // Bytes that end ASCII text: C0 controls, and the lead and continuation bytes of UTF-8.
    auto constexpr Stoppers = std::array { '\x00', '\x1B', '\x1F', '\x80', '\xC3', '\xFF' };

    for (auto const& scanner: unicode::detail::ascii_scanners())
    {
        INFO("SIMD width: " << scanner.simd_bit_width);
        for (auto const stopper: Stoppers)
        {
            INFO("stop byte: " << escape(static_cast<uint8_t>(stopper)));
            for (auto const offset: std::views::iota(size_t { 0 }, MaxTestedLength + 1))
            {
                INFO("offset: " << offset);
                auto text = std::string(TestTextLength, 'a');
                text[offset] = stopper;
                CHECK(scanner.scan(text, text.size()) == offset);
            }
        }
    }
}

TEST_CASE("scan.ascii.every_simd_width_scans_past_the_edges_of_printable_ascii")
{
    // The first and last printable ASCII bytes, which must not end the text.
    auto constexpr NonStoppers = std::array { ' ', '~' };

    for (auto const& scanner: unicode::detail::ascii_scanners())
    {
        INFO("SIMD width: " << scanner.simd_bit_width);
        for (auto const nonStopper: NonStoppers)
        {
            INFO("text byte: " << escape(static_cast<uint8_t>(nonStopper)));
            // Every offset, the scalar tail's included: a vector stop is only a candidate that the scalar loop
            // confirms, so an off-by-one there shows only when the byte reaches it.
            for (auto const offset: std::views::iota(size_t { 0 }, TestTextLength))
            {
                INFO("offset: " << offset);
                auto text = std::string(TestTextLength, 'a');
                text[offset] = nonStopper;
                CHECK(scanner.scan(text, text.size()) == text.size());
            }
        }
    }
}

TEST_CASE("scan.ascii.every_simd_width_covers_text_of_every_length")
{
    auto const longText = std::string(TestTextLength, 'a');
    for (auto const& scanner: unicode::detail::ascii_scanners())
    {
        INFO("SIMD width: " << scanner.simd_bit_width);
        // Empty text with a null data pointer, as a default-constructed string_view has, must not be loaded from.
        CHECK(scanner.scan(std::string_view {}, TestTextLength) == 0);
        for (auto const length: std::views::iota(size_t { 0 }, MaxTestedLength + 1))
        {
            INFO("length: " << length);
            // A heap buffer of exactly this many bytes: under ASan a load past its end is an error, which a
            // std::string's spare capacity would hide.
            auto const exact = std::vector<char>(length, 'a');
            CHECK(scanner.scan(std::string_view(exact.data(), exact.size()), TestTextLength) == length);
            // The same length reached through the column limit instead of the end of the text.
            CHECK(scanner.scan(longText, length) == length);
        }
    }
}

TEST_CASE("scan.ascii.scanners_are_listed_widest_first_down_to_128_bits")
{
    auto const scanners = unicode::detail::ascii_scanners();
    REQUIRE(!scanners.empty());
    CHECK(scanners.back().simd_bit_width == 128);
    CHECK(std::ranges::is_sorted(scanners, std::ranges::greater {}, &unicode::detail::ascii_scanner::simd_bit_width));
}

TEST_CASE("scan.ascii.runnable_scanners_start_at_the_widest_the_cpu_can_run")
{
    using unicode::detail::ascii_scanner;

    // Only the widths matter to the filter, so every row borrows a scan function that every build has.
    auto constexpr Table = std::array {
        ascii_scanner { .simd_bit_width = 512, .scan = &unicode::detail::scan_for_text_ascii },
        ascii_scanner { .simd_bit_width = 256, .scan = &unicode::detail::scan_for_text_ascii },
        ascii_scanner { .simd_bit_width = 128, .scan = &unicode::detail::scan_for_text_ascii },
    };

    struct Expectation
    {
        size_t maxSimdBitWidth;
        size_t firstWidth;
        size_t count;
    };
    auto constexpr Expectations = std::array {
        Expectation { .maxSimdBitWidth = 128, .firstWidth = 128, .count = 1 },
        Expectation { .maxSimdBitWidth = 256, .firstWidth = 256, .count = 2 },
        Expectation { .maxSimdBitWidth = 512, .firstWidth = 512, .count = 3 },
    };

    for (auto const& expectation: Expectations)
    {
        INFO("CPU's widest vector: " << expectation.maxSimdBitWidth);
        auto const runnable = unicode::detail::runnable_ascii_scanners(Table, expectation.maxSimdBitWidth);
        REQUIRE(runnable.size() == expectation.count);
        CHECK(runnable.front().simd_bit_width == expectation.firstWidth);
    }
}

TEST_CASE("scan.complex.grapheme_cluster.1")
{
    auto state = unicode::scan_state {};
    auto const familyEmoji8 = u8(FamilyEmoji);
    auto const result = unicode::detail::scan_for_text_nonascii(state, familyEmoji8, 80, unicode::null_receiver::get());
    CHECK(result.count == 2);
    CHECK(state.next == familyEmoji8.data() + familyEmoji8.size());
}

TEST_CASE("scan.complex.grapheme_cluster.2")
{
    auto state = unicode::scan_state {};
    auto const familyEmoji8 = u8(FamilyEmoji) + u8(FamilyEmoji);
    auto const result = unicode::detail::scan_for_text_nonascii(state, familyEmoji8, 80, unicode::null_receiver::get());
    CHECK(result.count == 4);
    CHECK(state.next == familyEmoji8.data() + familyEmoji8.size());
}

TEST_CASE("scan.complex.mixed")
{
    auto state = unicode::scan_state {};
    auto const text = u8(FamilyEmoji) + "ABC"s + u8(FamilyEmoji);
    auto const result = unicode::detail::scan_for_text_nonascii(state, text, 80, unicode::null_receiver::get());
    CHECK(result.count == 2);
    CHECK(state.next == text.data() + u8(FamilyEmoji).size());
}

TEST_CASE("scan.complex.half-overflowing")
{
    auto state = unicode::scan_state {};
    auto const oneEmoji = u8(SmileyEmoji);
    auto const text = oneEmoji + oneEmoji + oneEmoji;

    // match at boundary
    auto const result2 = unicode::detail::scan_for_text_nonascii(state, text, 2, unicode::null_receiver::get());
    CHECK(result2.count == 2);
    CHECK(state.next == text.data() + oneEmoji.size());

    // one grapheme cluster is half overflowing
    auto const result3 = unicode::detail::scan_for_text_nonascii(state, text, 3, unicode::null_receiver::get());
    CHECK(result3.count == 2);
    CHECK(state.next == text.data() + oneEmoji.size());

    // match buondary
    auto const result4 = unicode::detail::scan_for_text_nonascii(state, text, 4, unicode::null_receiver::get());
    CHECK(result4.count == 4);
    CHECK(state.next == text.data() + 2 * oneEmoji.size());
}

TEST_CASE("scan.any.tiny")
{
    // Ensure that we're really only scanning up to the input's size (1 byte, here).
    auto state = unicode::scan_state {};
    auto const storage = "X{0123456789ABCDEF}"sv;
    auto const input = storage.substr(0, 1);
    auto const result = unicode::scan_text(state, input, 80);
    CHECK(result.count == 1);
    CHECK(state.next == input.data() + input.size());
    CHECK(*state.next == '{');
}

TEST_CASE("scan.complex.sliced_calls")
{
    auto state = unicode::scan_state {};
    auto const text = "\xF0\x9F\x98\x80\033\\0123456789ABCDEF"sv; // U+1F600
    auto constexpr splitOffset = 3;
    auto const chunkOne = std::string_view(text.data(), splitOffset);

    auto result = unicode::scan_text(state, chunkOne, 80);

    REQUIRE(state.utf8.expectedLength == 4);
    REQUIRE(state.utf8.currentLength == 3);
    CHECK(result.count == 0);
    CHECK(result.start == text.data());
    CHECK(result.end == text.data());
    CHECK(state.next == (text.data() + splitOffset));

    auto const chunkTwo = std::string_view(state.next, (size_t) std::distance(state.next, text.data() + text.size()));
    result = unicode::scan_text(state, chunkTwo, 80, unicode::null_receiver::get());

    REQUIRE(state.utf8.expectedLength == 0);
    CHECK(result.count == 2);
    REQUIRE(result.start == text.data());
    REQUIRE(result.end == text.data() + 4);
    REQUIRE(state.next == text.data() + 4);
    auto const resultingText = string_view(result.start, static_cast<size_t>(std::distance(result.start, result.end)));
    REQUIRE(resultingText == text.substr(0, 4));
}

TEST_CASE("scan.any.ascii_complex_repeat")
{
    auto const oneComplex = u8(SmileyEmoji);        // 2
    auto const oneSimple = "0123456789ABCDEF0123"s; // 20

    for (size_t i = 1; i <= 6; ++i)
    {
        auto s = ""s;
        for (size_t k = 1; k <= i; ++k)
            s += (k % 2) != 0 ? oneSimple : oneComplex;
        s += ControlCodes;

        auto state = unicode::scan_state {};
        auto const result = scan_text(state, s, 80);
        auto const countSimple = ((i + 1) / 2) * 20;
        auto const countComplex = (i / 2) * 2;

        INFO(std::format("i = {}, ascii# {}, complex# {}, count {}, actual {}, s = \"{}\"",
                         i,
                         countSimple,
                         countComplex,
                         result.count,
                         countSimple + countComplex,
                         escape(s)));

        CHECK(result.count == countSimple + countComplex);
        CHECK(state.next == s.data() + s.size() - ControlCodes.size());
    }
}

TEST_CASE("scan.any.complex_ascii_repeat")
{
    auto const oneComplex = u8(SmileyEmoji);        // 2
    auto const oneSimple = "0123456789ABCDEF0123"s; // 20

    for (size_t i = 1; i <= 6; ++i)
    {
        auto s = ""s;
        for (size_t k = 1; k <= i; ++k)
            s += (k % 2) != 0 ? oneComplex : oneSimple;
        s += ControlCodes;

        auto state = unicode::scan_state {};
        auto const result = unicode::scan_text(state, s, 80);
        CHECK(result.count == (i / 2) * 20 + ((i + 1) / 2) * 2);
        CHECK(state.next == s.data() + s.size() - ControlCodes.size());
    }
}

TEST_CASE("scan.complex.VS16")
{
    auto const oneComplex = u8(CopyrightSign);
    auto const modifierVS16 = u8(U"\uFE0F"sv);

    // // narrow copyright sign
    auto state = unicode::scan_state {};
    auto const result1 = unicode::scan_text(state, oneComplex, 80);
    CHECK(result1.count == 1);
    CHECK(state.next == oneComplex.data() + oneComplex.size());

    // copyright sign in emoji presentation
    state = {};
    auto const s = oneComplex + modifierVS16;
    auto const result = unicode::scan_text(state, s, 80);
    CHECK(result.count == 2);
    CHECK(state.next == s.data() + s.size());

    state = {};
    auto const result3 = unicode::scan_text(state, s, 1);
    CHECK(result3.count == 0);
    CHECK(state.next == s.data());
}

TEST_CASE("scan.width_agrees_with_grapheme_cluster_width")
{
    // scan_text() and grapheme_cluster_width() are the library's two ways of asking how wide a piece
    // of text is, and they must never answer differently: a terminal scans with the former to advance
    // the cursor and measures with the latter when laying out, so any disagreement draws text at a
    // different width than was reserved for it and corrupts the rest of the line.
    auto const cases = std::array {
        U"का"sv,                   // Devanagari ka + AA matra (spacing mark)
        U"क्न"sv,                   // ka + virama + na (conjunct)
        U"กำ"sv,                   // Thai ko kai + sara am
        U"ຠຳ"sv,                   // Lao pho + sign AM
        U"ក្រ"sv,                   // Khmer conjunct (Invisible_Stacker)
        U"©️"sv,                    // copyright + VS16 (a defined variation base)
        U"✓️"sv,                    // check mark + VS16 (NOT a variation base)
        U"⌚︎"sv,                   // watch + VS15
        U"漢︎"sv,                   // CJK ideograph + VS15 (must not narrow)
        U"❤\U0001F3FB"sv,          // heart + skin tone modifier
        U"❤‍\U0001F525"sv,     // heart ZWJ fire
        U"\U0001F1E9\U0001F1EA"sv, // regional indicator pair
        U"中é"sv,                  // wide then narrow: the narrow one must not inherit width 2
        U"中éé"sv,                 // ...and the error must not compound
        U"\U0001F600é"sv,          // emoji then narrow
        U"中α"sv,                  // CJK then Greek alpha
        U"❤‍A"sv,              // heart ZWJ 'A': GB11 does not join, so this is two clusters
    };

    for (auto const cluster: cases)
    {
        auto const utf8 = u8(cluster);
        auto state = unicode::scan_state {};
        auto const scanned = unicode::scan_text(state, utf8, 80);
        INFO("cluster: " << escape(utf8));
        // The oracle is the std::string_view overload, which segments. The u32string_view one does
        // not -- it is documented as taking a *single* cluster -- and several cases above are two,
        // so measuring them with it would let a ZWJ swallow the codepoint across a cluster boundary
        // and report on the oracle rather than on scan_text().
        CHECK(scanned.count == unicode::grapheme_cluster_width(utf8));
    }
}

TEST_CASE("scan.cluster_split_across_calls_is_measured_once")
{
    // A grapheme cluster that straddles two scan_text() calls must total exactly what the whole
    // string totals: chunked input is how a terminal reads a PTY, so any drift here mis-advances the
    // cursor on ordinary text.
    //
    // Regression: the width accumulator was a local of scan_for_text_nonascii() while the grapheme
    // state it tracks lives in scan_state, so the second call started from an empty accumulator and
    // the selector it carried widened nothing.
    auto const text = u8(U"©️"sv); // copyright (2 bytes) + VS16 (3 bytes)
    REQUIRE(text.size() == 5);

    auto state = unicode::scan_state {};
    auto const first = unicode::scan_text(state, string_view(text.data(), 2), 80);
    auto const rest = string_view(state.next, static_cast<size_t>(std::distance(state.next, text.data() + text.size())));
    auto const second = unicode::scan_text(state, rest, 80);

    CHECK(first.count + second.count == unicode::grapheme_cluster_width(text));
    CHECK(first.count + second.count == 2);
    CHECK(state.next == text.data() + text.size());
}

TEST_CASE("scan.growing_cluster_does_not_overrun_the_scan_end")
{
    // scan_text() promises result.end <= state.next. The overflow rewind used to reset the input
    // pointer to a cluster start that lagged one cluster behind while result.end had already
    // advanced, so the caller was handed an end past what was scanned and re-rendered those bytes.
    //
    // The rewind was reachable only through VS16 before this path learned to honour spacing marks
    // and conjuncts, which is why ordinary Devanagari now reaches it.
    auto const text = u8(U"中काab"sv); // CJK + ka + AA matra + "ab"
    auto state = unicode::scan_state {};
    auto const result = unicode::scan_text(state, text, 3);

    CHECK(result.start <= result.end);
    CHECK(result.end <= state.next);

    // 中 is two columns and का is two more, so only 中 fits in three.
    CHECK(result.count == 2);
    CHECK(result.end == text.data() + 3);
}

TEST_CASE("scan.ascii_base_carries_into_a_following_spacing_mark")
{
    // The ASCII fast path is a separate scanner, but a cluster does not have to respect that
    // boundary: an ASCII base followed by a non-ASCII spacing mark is one cluster two columns wide.
    // Regression: the fast path recorded no last codepoint, so the mark looked like the start of a
    // fresh cluster and the pair measured 1.
    struct Case
    {
        std::u32string_view text;
        size_t expected;
        std::string_view what;
    };
    auto const cases = std::array {
        Case { U"aः"sv, 2, "a + Devanagari visarga" },
        Case { U"aா"sv, 2, "a + Tamil AA matra" },
        Case { U"abः"sv, 3, "trailing ASCII char is the base" },
        Case { U"abc"sv, 3, "pure ASCII still counts every column" },
        Case { U"ab中"sv, 4, "ASCII then a wide char that opens its own cluster" },
    };

    for (auto const& testCase: cases)
    {
        auto const utf8 = u8(testCase.text);
        auto state = unicode::scan_state {};
        auto const result = unicode::scan_text(state, utf8, 80);
        INFO(testCase.what);
        CHECK(result.count == testCase.expected);
        CHECK(result.count == unicode::grapheme_cluster_width(utf8));
    }
}

TEST_CASE("scan.ascii_base_keeps_a_zero_width_continuation")
{
    // The neighbouring case above widens its cluster, which is what used to make it work. A
    // continuation that adds NO column -- a combining mark, a variation selector on a base with no
    // variation sequence -- consumed its bytes and reported none of them: the scan advanced
    // state.next past the mark while result.end stayed at the end of the ASCII run, so a caller
    // printing [start, end) and resuming at next lost the mark entirely. "e" + U+0301 arrived as a
    // bare "e".
    //
    // Reachable only with an ASCII base: without one the whole cluster stays inside a single
    // scan_for_text_nonascii() call and never crosses the handoff between the two scanners.
    struct Case
    {
        std::u32string_view text;
        size_t expected;
        std::string_view what;
    };
    auto const cases = std::array {
        Case { U"éx"sv, 2, "e + combining acute, then ASCII" },
        Case { U"é"sv, 1, "e + combining acute at the very end" },
        Case { U"abćd"sv, 4, "the mark attaches to the last ASCII char of a run" },
        Case { U"e️"sv, 1, "VS16 on a base with no variation sequence adds nothing" },
        Case { U"1️"sv, 2, "VS16 on a defined variation base still widens" },
    };

    for (auto const& testCase: cases)
    {
        auto const utf8 = u8(testCase.text);
        auto state = unicode::scan_state {};
        auto const result = unicode::scan_text(state, utf8, 80);
        INFO(testCase.what << ": " << escape(utf8));
        CHECK(result.count == testCase.expected);
        // Every byte the scan consumed has to be inside the range it reports.
        CHECK(result.end == state.next);
        CHECK(state.next == utf8.data() + utf8.size());
    }
}

TEST_CASE("scan.everything_consumed_is_reported")
{
    // The general form of the case above, and the property whose absence let it through: unless the
    // scan stopped inside an incomplete UTF-8 sequence, the bytes it consumed (up to state.next) and
    // the bytes it hands back ([start, end)) are the same bytes. A caller prints the latter and
    // resumes at the former, so any gap between them is text that is silently swallowed -- counting
    // columns alone cannot detect it.
    auto const cases = std::array {
        U"éx"sv,                     // zero-width continuation after ASCII
        U"aः"sv,                     // widening continuation after ASCII
        U"abc"sv,                    // pure ASCII
        U"中éé"sv,                   // wide, then two narrow clusters
        U"a中b"sv,                   // alternating scanners
        U"éé"sv,                     // two zero-width continuations in a row
        U"a‍b"sv,                  // ZWJ between ASCII: joins nothing, still consumes bytes
        U"⌚︎x"sv,                    // VS15 after a non-ASCII base
        FamilyEmoji,                 // long ZWJ sequence
        U"a\U0001F1E9\U0001F1EAb"sv, // regional indicator pair between ASCII runs
    };

    for (auto const& text: cases)
    {
        auto const utf8 = u8(text);
        auto state = unicode::scan_state {};
        auto const result = unicode::scan_text(state, utf8, 80);
        INFO(escape(utf8));
        REQUIRE(state.utf8.expectedLength == 0); // nothing left half-decoded
        CHECK(result.start <= result.end);
        CHECK(result.end == state.next);
    }
}

namespace
{
struct cluster_collector final: public unicode::grapheme_cluster_receiver
{
    std::vector<std::string> clusters;
    std::vector<size_t> widths;

    void receiveAsciiSequence(std::string_view text) noexcept override
    {
        for (auto const ch: text)
        {
            clusters.emplace_back(1, ch);
            widths.push_back(1);
        }
    }
    void receiveGraphemeCluster(std::string_view text, size_t columnCount) noexcept override
    {
        clusters.emplace_back(text);
        widths.push_back(columnCount);
    }
    void receiveInvalidGraphemeCluster() noexcept override
    {
        clusters.emplace_back("<invalid>");
        widths.push_back(1);
    }
};
} // namespace

TEST_CASE("scan.receiver_gets_whole_clusters")
{
    // Regression: the byte count handed to receiveGraphemeCluster() was reset only *after* the call,
    // so each cluster was delivered with the byte length of the next cluster's first codepoint --
    // truncated UTF-8 whenever the two differ in length, and the final cluster was dropped entirely.
    // Clusters of equal byte length hid it, which is all the older tests used.
    auto const text = u8(U"中é"sv); // CJK (3 bytes) + e-acute (2 bytes)
    auto collector = cluster_collector {};
    auto state = unicode::scan_state {};
    auto const result = unicode::scan_text(state, text, 80, collector);

    CHECK(result.count == 3);
    REQUIRE(collector.clusters.size() == 2);
    CHECK(collector.clusters[0] == u8(U"中"sv));
    CHECK(collector.clusters[1] == u8(U"é"sv));
    CHECK(collector.widths[0] == 2);
    CHECK(collector.widths[1] == 1);
}

TEST_CASE("scan.c1_control.stops_mid_run")
{
    // A raw C1 control byte (0x80..0x9F) sitting at a character boundary is a control, not text, so
    // scan_text stops at it and hands it back to the caller -- even when text follows. Regression:
    // with trailing bytes present the C1 byte used to be consumed as U+FFFD (a terminal parser would
    // then lose a mid-run 8-bit SPA/EPA, i.e. 0x96/0x97).
    auto state = unicode::scan_state {};
    auto const text = "ab\x96"
                      "cd"sv; // 'a', 'b', C1 SPA (0x96), then more text
    auto const result = unicode::scan_text(state, text, 80);
    CHECK(result.count == 2);
    CHECK(result.end == text.data() + 2);
    CHECK(state.next == text.data() + 2);
    CHECK(static_cast<uint8_t>(*state.next) == 0x96);
}

TEST_CASE("scan.c1_control.continuation_byte_stays_text")
{
    // A byte in 0x80..0x9F that is a UTF-8 *continuation* byte must remain part of its codepoint, not
    // be mistaken for a C1 control: U+2018 encodes as 0xE2 0x80 0x98, both trailing bytes falling in
    // the C1 range. The scan consumes the whole codepoint and stops at the following C0 control.
    auto state = unicode::scan_state {};
    auto const text = u8(U"\u2018"sv) + "\n"s; // LEFT SINGLE QUOTATION MARK, then LF
    auto const result = unicode::scan_text(state, text, 80);
    CHECK(result.count == 1);
    CHECK(state.next == text.data() + 3);
    CHECK(*state.next == '\n');
}

// {{{ ill-formed UTF-8
//
// scan_text() ends the run before an ill-formed UTF-8 subsequence, exactly as it ends it before a control,
// and leaves state.next at the subsequence's first byte: the caller decodes it under its own replacement
// policy, and every column counted is a column of the returned text. A character that the end of the input
// cuts off is not ill-formed; it stays pending in state.utf8 for the next call to complete.
//
// Regression: an ill-formed byte was counted as a column, with no check against the limit, and left
// outside the returned run unless valid text followed it. A trailing one was therefore consumed and lost,
// a mid-run one was handed back raw inside the run, and the count could exceed the caller's limit.

namespace
{

/// One row of the Unicode Standard's Table 3-7, "Well-Formed UTF-8 Byte Sequences".
struct well_formed_utf8_sequence
{
    uint8_t leadFirst;
    uint8_t leadLast;
    size_t length;
    // The range of the second byte. It is narrower than 0x80..0xBF after E0, ED, F0 and F4, which excludes
    // overlong forms, surrogates and codepoints past U+10FFFF.
    uint8_t secondFirst;
    uint8_t secondLast;
};

auto constexpr WellFormedUtf8Sequences = std::array {
    well_formed_utf8_sequence { 0x00, 0x7F, 1, 0x00, 0x00 }, // U+0000..U+007F
    well_formed_utf8_sequence { 0xC2, 0xDF, 2, 0x80, 0xBF }, // U+0080..U+07FF
    well_formed_utf8_sequence { 0xE0, 0xE0, 3, 0xA0, 0xBF }, // U+0800..U+0FFF
    well_formed_utf8_sequence { 0xE1, 0xEC, 3, 0x80, 0xBF }, // U+1000..U+CFFF
    well_formed_utf8_sequence { 0xED, 0xED, 3, 0x80, 0x9F }, // U+D000..U+D7FF
    well_formed_utf8_sequence { 0xEE, 0xEF, 3, 0x80, 0xBF }, // U+E000..U+FFFF
    well_formed_utf8_sequence { 0xF0, 0xF0, 4, 0x90, 0xBF }, // U+10000..U+3FFFF
    well_formed_utf8_sequence { 0xF1, 0xF3, 4, 0x80, 0xBF }, // U+40000..U+FFFFF
    well_formed_utf8_sequence { 0xF4, 0xF4, 4, 0x80, 0x8F }, // U+100000..U+10FFFF
};

/// Tests whether @p text is well-formed UTF-8 by Table 3-7 -- independently of the decoder under test,
/// which is more lenient: it accepts overlong forms, for one.
bool is_well_formed_utf8(std::string_view text)
{
    auto const isContinuation = [](char ch) {
        return (static_cast<uint8_t>(ch) & 0b1100'0000) == 0b1000'0000;
    };
    while (!text.empty())
    {
        auto const lead = static_cast<uint8_t>(text.front());
        auto const sequence = std::ranges::find_if(
            WellFormedUtf8Sequences, [lead](auto const& row) { return row.leadFirst <= lead && lead <= row.leadLast; });
        if (sequence == WellFormedUtf8Sequences.end() || text.size() < sequence->length)
            return false;
        if (sequence->length > 1)
        {
            auto const second = static_cast<uint8_t>(text[1]);
            if (second < sequence->secondFirst || sequence->secondLast < second)
                return false;
            if (!std::ranges::all_of(text.substr(2, sequence->length - 2), isContinuation))
                return false;
        }
        text.remove_prefix(sequence->length);
    }
    return true;
}

/// The ill-formed subsequences the sweeps below insert into valid text. Text always follows them, so
/// none is a character merely cut off by the end of the input.
auto constexpr IllFormedSubsequences = std::array {
    "\x80"sv,     // a continuation byte with no lead byte before it (at a character boundary, also a C1 control)
    "\xBF"sv,     // a continuation byte with no lead byte before it
    "\xC0"sv,     // a lead byte, cut off by whatever follows it in the sweeps
    "\xFF"sv,     // a byte that UTF-8 never uses
    "\xE2\x94"sv, // two bytes of a three-byte character, cut off by whatever follows them in the sweeps
};

/// Column limits the sweeps scan with: tighter than the text before the ill-formed bytes, and looser.
auto constexpr ColumnLimits = std::array<size_t, 6> { 1, 2, 3, 7, 80, 100'000 };

/// The characters the sweeps build their text from, each one a whole grapheme cluster.
auto constexpr AsciiUnits = std::array { "a"sv };

auto constexpr MixedUnits = std::array {
    "a"sv,
    "\xE4\xB8\xAD"sv,         // U+4E2D, two columns
    "\xC3\xA9"sv,             // U+00E9
    "e\xCC\x81"sv,            // e + U+0301: a cluster across the handoff from the ASCII scanner
    "\xF0\x9F\x98\x80"sv,     // U+1F600, two columns
    "\xE2\x80\x98"sv,         // U+2018, whose continuation bytes fall in the C1 range
    "\xC2\xA9\xEF\xB8\x8F"sv, // U+00A9 + VS16: a cluster that widens to two columns
    "bc"sv,
};

/// No ASCII at all, so that the non-ASCII scanner, which stops at ASCII, sees the whole text.
auto constexpr NonAsciiUnits = std::array {
    "\xE4\xB8\xAD"sv, "\xC3\xA9"sv, "\xF0\x9F\x98\x80"sv, "\xE2\x80\x98"sv, "\xC2\xA9\xEF\xB8\x8F"sv, "\xCE\xB1"sv,
};

/// Valid text built from whole units, and the offset of every boundary between them.
struct unit_text
{
    std::string text;
    std::vector<size_t> boundaries;
};

/// Repeats @p units until the text is at least TestTextLength bytes long.
unit_text build_unit_text(std::span<std::string_view const> units)
{
    auto result = unit_text {};
    while (result.text.size() < TestTextLength)
    {
        for (auto const unit: units)
        {
            result.boundaries.push_back(result.text.size());
            result.text += unit;
        }
    }
    return result;
}

using scan_function = std::function<unicode::scan_result(unicode::scan_state&, std::string_view, size_t)>;

/// Inserts every ill-formed subsequence at every unit boundary of @p base up to MaxTestedLength, and checks
/// that @p scan, at every column limit, ends no later than there -- and exactly as if the text ended there.
void check_every_ill_formed_subsequence_ends_the_scan(scan_function const& scan, unit_text const& base)
{
    for (auto const illFormed: IllFormedSubsequences)
    {
        INFO("ill-formed: " << escape(illFormed));
        for (auto const offset: base.boundaries | std::views::take_while([](size_t at) { return at <= MaxTestedLength; }))
        {
            INFO("at offset: " << offset);
            auto text = base.text;
            text.insert(offset, illFormed);
            auto const valid = std::string_view(text).substr(0, offset);
            for (auto const limit: ColumnLimits)
            {
                INFO("limit: " << limit);
                auto state = unicode::scan_state {};
                auto const result = scan(state, text, limit);
                auto const run = std::string_view(result.start, result.end);

                CHECK(result.count <= limit);
                CHECK(result.start == text.data());
                CHECK(result.end <= text.data() + offset);
                // Nothing is consumed that is not handed back: no ill-formed byte, and no text behind one.
                CHECK(state.next == result.end);
                CHECK(state.utf8.expectedLength == 0);
                CHECK(is_well_formed_utf8(run));
                CHECK(result.count == unicode::grapheme_cluster_width(run));

                auto validState = unicode::scan_state {};
                auto const expected = scan(validState, valid, limit);
                CHECK(result.count == expected.count);
                CHECK(run.size() == static_cast<size_t>(std::distance(expected.start, expected.end)));
            }
        }
    }
}

/// The bytes of a character that the end of the text cut off, which the scan keeps in its decoder state.
unsigned pending_bytes(unicode::scan_state const& state)
{
    return state.utf8.expectedLength != 0 ? state.utf8.currentLength : 0;
}

} // namespace

TEST_CASE("scan.ill_formed_utf8.ends_the_run_before_it")
{
    struct ill_formed_case
    {
        std::string_view what;
        std::string_view text;
        size_t limit;
        std::string_view run; // the text returned, which starts where the input does
        size_t count;
        size_t next; // the offset state.next is left at
        unsigned pending;
    };

    // "Reported" rows reproduce what Contour's bulk-scan differential harness found.
    auto constexpr Cases = std::array {
        ill_formed_case { "a lead byte cut off by ASCII, at the start (reported)", "\xE2hello\xBF\xFF", 7, "", 0, 0, 0 },
        ill_formed_case { "the same text once the caller has decoded the lead byte", "hello\xBF\xFF", 7, "hello", 5, 5, 0 },
        ill_formed_case { "two FF bytes behind ASCII, with a tight limit (reported)", "ab\xFF\xFF", 3, "ab", 2, 2, 0 },
        ill_formed_case { "two FF bytes behind ASCII (reported)", "ab\xFF\xFF", 7, "ab", 2, 2, 0 },
        ill_formed_case { "a stray continuation byte and FF behind ASCII (reported)", "hello\xBF\xFF", 7, "hello", 5, 5, 0 },
        ill_formed_case { "the same with the limit inside the ASCII (reported)", "hello\xBF\xFF", 2, "he", 2, 2, 0 },
        ill_formed_case { "an ill-formed byte right where the limit is reached", "\xE4\xB8\xAD\xFF", 2, "\xE4\xB8\xAD", 2, 3, 0 },
        ill_formed_case { "the lead byte is ill-formed, not the ASCII that cuts it off", "ab\xE2hi", 80, "ab", 2, 2, 0 },
        ill_formed_case { "two bytes of three, cut off by ASCII", "ab\xE2\x94hi", 80, "ab", 2, 2, 0 },
        ill_formed_case { "cut off by a control", "ab\xE2\x1B[m", 80, "ab", 2, 2, 0 },
        ill_formed_case { "cut off by FF, which can never continue a character", "ab\xE2\xFF\xBFxy", 80, "ab", 2, 2, 0 },
        ill_formed_case { "cut off by another lead byte, whose character is left whole", "ab\xE2\xC3\xA9", 80, "ab", 2, 2, 0 },
        ill_formed_case { "a stray continuation byte behind a wide character", "\xE4\xB8\xAD\xBFx", 80, "\xE4\xB8\xAD", 2, 3, 0 },
        ill_formed_case {
            "a lead byte cut off between two wide characters", "\xE4\xB8\xAD\xE2\xE6\x96\x87", 80, "\xE4\xB8\xAD", 2, 3, 0 },
        ill_formed_case { "three bytes of four, cut off by ASCII", "\xF0\x9F\x98x", 80, "", 0, 0, 0 },
        ill_formed_case { "behind a zero-width continuation of an ASCII base", "e\xCC\x81\xBF", 80, "e\xCC\x81", 1, 3, 0 },
        ill_formed_case { "behind a cluster that widened", "\xC2\xA9\xEF\xB8\x8F\xFF", 80, "\xC2\xA9\xEF\xB8\x8F", 2, 5, 0 },
        ill_formed_case { "behind a cluster that widened past the limit", "\xC2\xA9\xEF\xB8\x8F\xFF", 1, "", 0, 0, 0 },
        ill_formed_case { "a stray continuation byte alone", "\xBF", 80, "", 0, 0, 0 },
        // The next two are adapted from scan.invalid, a test long compiled out, which expected U+FFFD columns.
        ill_formed_case { "B1 between ASCII", "A\xB1Z", 80, "A", 1, 1, 0 },
        ill_formed_case { "an F5 lead byte cut off by ASCII", "A\xF5Z", 80, "A", 1, 1, 0 },
        ill_formed_case {
            "a character cut off by the end of the text is pending, not ill-formed", "a\xE2\x94", 80, "a", 1, 3, 2 },
        ill_formed_case { "the same behind a wide character", "\xE4\xB8\xAD\xE2\x94", 80, "\xE4\xB8\xAD", 2, 5, 2 },
        ill_formed_case { "a lone lead byte at the end of the text", "\xE2", 80, "", 0, 1, 1 },
    };

    for (auto const& testCase: Cases)
    {
        INFO(testCase.what << ": \"" << escape(testCase.text) << "\", limit " << testCase.limit);
        auto state = unicode::scan_state {};
        auto const result = unicode::scan_text(state, testCase.text, testCase.limit);
        CHECK(result.count <= testCase.limit);
        CHECK(result.count == testCase.count);
        CHECK(result.start == testCase.text.data());
        CHECK(escape(std::string_view(result.start, result.end)) == escape(testCase.run));
        CHECK(state.next == testCase.text.data() + testCase.next);
        CHECK(pending_bytes(state) == testCase.pending);
    }
}

TEST_CASE("scan.ill_formed_utf8.every_ascii_scanner_stops_before_it")
{
    for (auto const& scanner: unicode::detail::ascii_scanners())
    {
        INFO("SIMD width: " << scanner.simd_bit_width);
        for (auto const illFormed: IllFormedSubsequences)
        {
            INFO("ill-formed: " << escape(illFormed));
            for (auto const offset: std::views::iota(size_t { 0 }, MaxTestedLength + 1))
            {
                INFO("at offset: " << offset);
                auto text = std::string(TestTextLength, 'a');
                text.insert(offset, illFormed);
                for (auto const limit: ColumnLimits)
                {
                    INFO("limit: " << limit);
                    CHECK(scanner.scan(text, limit) == std::min(offset, limit));
                }
            }
        }
    }
}

TEST_CASE("scan.ill_formed_utf8.ends_the_scan_wherever_it_sits")
{
    auto const scanText = [](unicode::scan_state& state, std::string_view text, size_t limit) {
        return unicode::scan_text(state, text, limit);
    };

    SECTION("in ASCII text")
    {
        check_every_ill_formed_subsequence_ends_the_scan(scanText, build_unit_text(AsciiUnits));
    }
    SECTION("in mixed text")
    {
        check_every_ill_formed_subsequence_ends_the_scan(scanText, build_unit_text(MixedUnits));
    }
}

TEST_CASE("scan.ill_formed_utf8.ends_the_non_ascii_scan_wherever_it_sits")
{
    auto const scanNonAscii = [](unicode::scan_state& state, std::string_view text, size_t limit) {
        return unicode::detail::scan_for_text_nonascii(state, text, limit, unicode::null_receiver::get());
    };
    check_every_ill_formed_subsequence_ends_the_scan(scanNonAscii, build_unit_text(NonAsciiUnits));
}

TEST_CASE("scan.ill_formed_utf8.is_not_delivered_to_the_receiver")
{
    // The receiver hears of exactly the text the scan returns, so receiveInvalidGraphemeCluster() is not
    // called: the ill-formed bytes are the caller's to decode.
    struct receiver_case
    {
        std::string_view text;
        std::vector<std::u32string> received;
    };
    auto const cases = std::array {
        receiver_case { "ab\xFF\xE4\xB8\xAD"sv, { U"a", U"b" } },
        receiver_case { "\xE4\xB8\xAD\xBF"sv, { U"中" } },
        receiver_case { "\xE4\xB8\xAD\xE2h"sv, { U"中" } },
    };

    for (auto const& testCase: cases)
    {
        INFO(escape(testCase.text));
        auto collector = grapheme_cluster_collector {};
        auto state = unicode::scan_state {};
        unicode::scan_text(state, testCase.text, 80, collector);
        CHECK(collector.output == testCase.received);
    }
}

TEST_CASE("scan.ill_formed_utf8.cutting_off_a_character_an_earlier_call_began")
{
    // The first call ends inside a character and keeps its lead byte pending; only the second learns that
    // the character is ill-formed. It hands the lead byte back all the same, through state.next, which
    // therefore lies before the second call's text: in the first call's, where result.start already
    // points when such a character completes instead.
    auto constexpr Continuations = std::array {
        "hi"sv,       // ASCII
        "\x94hi"sv,   // one more byte of the character, then ASCII
        "\xC3\xA9"sv, // another lead byte
        "\xFF"sv,     // a byte that never continues a character
    };

    for (auto const continuation: Continuations)
    {
        auto const buffer = "ab\xE2"s + std::string(continuation);
        INFO(escape(buffer));
        auto state = unicode::scan_state {};
        auto const first = unicode::scan_text(state, std::string_view(buffer).substr(0, 3), 80);
        REQUIRE(first.count == 2);
        REQUIRE(state.next == buffer.data() + 3);
        REQUIRE(pending_bytes(state) == 1);

        auto const second = unicode::scan_text(state, std::string_view(buffer).substr(3), 80);
        CHECK(second.count == 0);
        CHECK(second.start == buffer.data() + 2);
        CHECK(second.end == buffer.data() + 2);
        CHECK(state.next == buffer.data() + 2);
        CHECK(pending_bytes(state) == 0);
    }
}

// {{{ Table 3-7 enforcement

namespace
{
/// Ill-formed byte sequences that the decoder must reject at the second byte or the lead byte itself.
auto constexpr TableThreeSevenViolations = std::array {
    "\xC0\x80"sv,         // overlong NUL
    "\xC1\xBF"sv,         // overlong U+007F
    "\xC0\x9B"sv,         // overlong ESC (0x9B is a C1 control on its own)
    "\xE0\x80\xAF"sv,     // overlong '/'
    "\xE0\x9F\xBF"sv,     // overlong U+07FF
    "\xED\xA0\x80"sv,     // surrogate U+D800
    "\xED\xBF\xBF"sv,     // surrogate U+DFFF
    "\xF0\x80\x80\x80"sv, // overlong NUL
    "\xF0\x8F\xBF\xBF"sv, // overlong U+FFFF
    "\xF4\x90\x80\x80"sv, // U+110000
    "\xF5\x80\x80\x80"sv, // lead byte past U+10FFFF
    "\xFF"sv,             // never used
};

/// Sequences at the edges of Table 3-7 that remain well-formed.
auto constexpr TableThreeSevenBoundaries = std::array {
    "\xC2\x80"sv,     "\xDF\xBF"sv,     "\xE0\xA0\x80"sv,     "\xED\x9F\xBF"sv,
    "\xEE\x80\x80"sv, "\xEF\xBF\xBF"sv, "\xF0\x90\x80\x80"sv, "\xF4\x8F\xBF\xBF"sv,
};
} // namespace

TEST_CASE("scan.table_3_7.violations_end_the_scan_in_front_of_them")
{
    for (auto const violation: TableThreeSevenViolations)
    {
        INFO("violation: " << escape(violation));
        auto const text = "ab"s + std::string(violation) + "cd"s;
        auto state = unicode::scan_state {};
        auto const result = unicode::scan_text(state, text, 80);
        CHECK(result.count == 2);
        CHECK(result.start == text.data());
        CHECK(result.end == text.data() + 2);
        CHECK(state.next == text.data() + 2);
        CHECK(pending_bytes(state) == 0);

        // The same at the very start of the text.
        auto const bare = std::string(violation) + "cd"s;
        auto bareState = unicode::scan_state {};
        auto const bareResult = unicode::scan_text(bareState, bare, 80);
        CHECK(bareResult.count == 0);
        CHECK(bareResult.end == bare.data());
        CHECK(bareState.next == bare.data());
    }
}

TEST_CASE("scan.table_3_7.boundary_values_stay_valid")
{
    for (auto const sequence: TableThreeSevenBoundaries)
    {
        INFO("sequence: " << escape(sequence));
        auto const text = "a"s + std::string(sequence) + "b"s;
        auto state = unicode::scan_state {};
        auto const result = unicode::scan_text(state, text, 80);
        CHECK(std::string_view(result.start, result.end) == text);
        CHECK(result.end == text.data() + text.size());
        CHECK(pending_bytes(state) == 0);
    }
}

TEST_CASE("scan.table_3_7.violation_split_across_calls")
{
    for (auto const violation: TableThreeSevenViolations)
    {
        if (violation.size() < 2 || static_cast<uint8_t>(violation[0]) == 0xC0 || static_cast<uint8_t>(violation[0]) == 0xC1
            || static_cast<uint8_t>(violation[0]) == 0xF5)
            continue; // rejected at the lead byte; nothing to split
        INFO("violation: " << escape(violation));
        auto state = unicode::scan_state {};
        // One buffer: a carried lead byte sits directly in front of the next call's text.
        auto const buffer = std::string(violation) + "cd"s;

        auto const r0 = unicode::scan_text(state, std::string_view(buffer).substr(0, 1), 80);
        CHECK(r0.count == 0);
        CHECK(pending_bytes(state) == 1);

        auto const r1 = unicode::scan_text(state, std::string_view(buffer).substr(1), 80);
        CHECK(r1.count == 0);
        CHECK(r1.start == buffer.data());
        CHECK(r1.end == buffer.data());
        CHECK(state.next == buffer.data());
        CHECK(pending_bytes(state) == 0);
    }

    SECTION("a valid second byte completes the character")
    {
        auto state = unicode::scan_state {};
        auto const buffer = "\xE0\xA0\x80"s;
        CHECK(unicode::scan_text(state, std::string_view(buffer).substr(0, 1), 80).count == 0);
        auto const r = unicode::scan_text(state, std::string_view(buffer).substr(1), 80);
        CHECK(r.count == 1);
        CHECK(r.end == buffer.data() + buffer.size());
    }
}

// }}}
// }}}
