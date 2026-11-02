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
#include <libunicode/support.h>
#include <libunicode/utf8.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cassert>
#include <cstdlib>
#include <format>
#include <variant>

using namespace std;
using namespace unicode;

namespace
{

std::string binstr(unsigned n)
{
    string ss;
    for (int i = 7; i >= 0; --i)
        ss += ((n & (1u << i)) != 0) ? '1' : '0';
    return ss;
}

} // namespace

TEST_CASE("utf8.bytes_1", "[utf8]")
{
    uint8_t constexpr C = '[';

    // encode
    uint8_t encoded {};
    unsigned const count = to_utf8(C, &encoded);
    REQUIRE(count == 1);
    CHECK(encoded == C);

    // decode
    auto state = utf8_decoder_state {};
    auto const r1 = from_utf8(state, C);
    REQUIRE(holds_alternative<Success>(r1));
    char32_t const b = get<Success>(r1).value;
    INFO(std::format("char32_t : 0x{:04X}", (unsigned) b));
    REQUIRE(b == C);
}

TEST_CASE("utf8.bytes_2", "[utf8]")
{
    char32_t constexpr C { 0xF6 }; // 0xC3 0xB6, 'ö'
    INFO(std::format("codepoint : 0x{:04X}", (unsigned) C));

    // encode
    uint8_t actual[2] = {};
    auto const count = to_utf8(C, actual);
    INFO(std::format("utf8      : {:X} {:X}", actual[0], actual[1]));
    INFO(std::format("binary    : {} {}", binstr(actual[0]), binstr(actual[1])));
    REQUIRE(count == 2);
    CHECK(actual[0] == 0xC3);
    CHECK(actual[1] == 0xB6);

    // decode
    size_t len = 0;
    auto const a = from_utf8(actual, &len);
    REQUIRE(holds_alternative<Success>(a));
    char32_t b = get<Success>(a).value;
    INFO(std::format("char32_t : 0x{:04X} ==? 0x{:04X}", (unsigned) b, (unsigned) C));
    REQUIRE(b == C);
}

TEST_CASE("utf8.bytes_3", "[utf8]")
{
    // encode
    uint8_t b3[4] = {};
    auto const count = to_utf8(0x20AC, b3); // EURO sign: €
    REQUIRE(count == 3);
    CHECK(b3[0] == 0xE2);
    CHECK(b3[1] == 0x82);
    CHECK(b3[2] == 0xAC);

    INFO(std::format("{:02X} {:02X} {:02X}", b3[0], b3[1], b3[2]));
    INFO(std::format("{} {} {}", binstr(b3[0]), binstr(b3[1]), binstr(b3[2])));

    // decode
    size_t len = 0;
    auto const a = from_utf8(b3, &len);
    REQUIRE(holds_alternative<Success>(a));
    char32_t const b = get<Success>(a).value;
    INFO(std::format("char32_t : 0x{:04X}", (unsigned) b));
    REQUIRE(b == 0x20AC);
}

TEST_CASE("utf8.bytes_3_dash", "[utf8]")
{
    // encode
    auto constexpr codepoint = 0x251C;
    uint8_t d3[3] { 0, 0, 0 };
    auto const len3 = to_utf8(codepoint, d3);
    REQUIRE(len3 == 3);
    CHECK(d3[0] == 0xE2);
    CHECK(d3[1] == 0x94);
    CHECK(d3[2] == 0x9C);

    INFO(std::format("{:02X} {:02X} {:02X}", d3[0], d3[1], d3[2]));
    INFO(std::format("{} {} {}", binstr(d3[0]), binstr(d3[1]), binstr(d3[2])));

    // decode
    auto constexpr u8s = array<uint8_t, 4> { 0xe2, 0x94, 0x9c, 0x61 /*a*/ };
    size_t len = 0;
    auto const a = from_utf8(u8s.data(), &len);

    CHECK(len == 3);
    REQUIRE(holds_alternative<Success>(a));
    CHECK(get<Success>(a).value == codepoint);
}

TEST_CASE("utf8.from_utf8", "[utf8]")
{
    auto const a8 = string { "Hello, World!" };
    auto const a32 = from_utf8(a8);
    CHECK(a32 == U"Hello, World!");

    auto const b8 = string { (char const*) u8"😖:-)" };
    auto const b32 = from_utf8(b8);
    CHECK(b32 == U"😖:-)");
}

TEST_CASE("utf8.from_utf8.invalid", "[utf8]")
{
    // Ensure invalid bytes are consumed and ignored accordingly.
    auto const a8 = string { "Hi\xb1Ho" };
    auto const a32 = from_utf8(a8);
    CHECK(a32 == U"HiHo");
}

TEST_CASE("utf8.from_utf8.incomplete.2", "[utf8]")
{
    // Ensure incomplete bytes are consumed and reported as Invalid accordingly.
    auto state = utf8_decoder_state {};

    // We start with an incomplete 2-byte sequence.
    auto const r0 = from_utf8(state, 0xC7);
    REQUIRE(holds_alternative<Incomplete>(r0));

    // Continue with another 2-byte sequence,
    // while the first one is still incomplete.
    auto const r1 = from_utf8(state, 0xC7);
    REQUIRE(holds_alternative<Invalid>(r1));
    auto const r2 = from_utf8(state, 0x8E);
    REQUIRE(holds_alternative<Success>(r2));
    REQUIRE((unsigned) get<Success>(r2).value == 0x01CE);
}

TEST_CASE("utf8.from_utf8.incomplete.3", "[utf8]")
{
    // Ensure incomplete bytes are consumed and reported as Invalid accordingly.
    auto state = utf8_decoder_state {};

    // We start with an incomplete 2-byte sequence.
    auto const r0 = from_utf8(state, 0xE2);
    REQUIRE(holds_alternative<Incomplete>(r0));
    auto const r1 = from_utf8(state, 0x82);
    REQUIRE(holds_alternative<Incomplete>(r1));

    // Continue with another 2-byte sequence,
    // while the first one is still incomplete.
    auto const r2 = from_utf8(state, 0xE2);
    REQUIRE(holds_alternative<Invalid>(r2));
    auto const r3 = from_utf8(state, 0x82);
    REQUIRE(holds_alternative<Incomplete>(r3));
    auto const r4 = from_utf8(state, 0xAC);
    REQUIRE(holds_alternative<Success>(r4));
    REQUIRE((unsigned) get<Success>(r4).value == 0x20AC);
}

TEST_CASE("utf8.from_utf8.incomplete.4", "[utf8]")
{
    auto constexpr sequence = "\xF0\x9F\x8D\xA3"sv;
    auto constexpr codepoint = 0x1F363;

    auto state = utf8_decoder_state {};

    // Generate an incomplete multi-byte sequence.
    for (size_t i = 0; i < sequence.size() - 1; ++i)
    {
        CAPTURE(i, unsigned(sequence[i]));
        auto const r = from_utf8(state, (uint8_t) sequence[i]);
        REQUIRE(holds_alternative<Incomplete>(r));
    }

    // Now fill the multi-byte sequence, but completely.
    auto const r0 = from_utf8(state, (uint8_t) sequence[0]);
    REQUIRE(holds_alternative<Invalid>(r0));

    for (size_t i = 1; i < sequence.size() - 1; ++i)
    {
        CAPTURE(i, unsigned(sequence[i]));
        auto const ri = from_utf8(state, (uint8_t) sequence[i]);
        REQUIRE(holds_alternative<Incomplete>(ri));
    }

    auto const last = from_utf8(state, (uint8_t) sequence.back());
    REQUIRE(holds_alternative<Success>(last));
    REQUIRE(get<Success>(last).value == codepoint);
}

TEST_CASE("utf8.iter", "[utf8]")
{
    auto constexpr values = string_view {
        "["
        "\xC3\xB6"         // ö  - german o-umlaut
        "\xE2\x82\xAC"     // €  - EURO sign U+20AC
        "\xF0\x9F\x98\x80" // 😀 - U+1F600
    };
    auto const* p = (char8_type const*) (values.data());
    auto state = utf8_decoder_state {};

    // single-byte
    auto result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Success>(result));
    REQUIRE(get<Success>(result).value == '[');

    // double-byte
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Incomplete>(result));
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Success>(result));
    REQUIRE(get<Success>(result).value == U'\u00F6'); // ö

    // 3 bytes
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Incomplete>(result));
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Incomplete>(result));
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Success>(result));
    REQUIRE(get<Success>(result).value == U'\u20AC'); // €

    // 4 bytes
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Incomplete>(result));
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Incomplete>(result));
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Incomplete>(result));
    result = from_utf8(state, *p++);
    REQUIRE(holds_alternative<Success>(result));
    REQUIRE(get<Success>(result).value == U'\U0001F600');
}

namespace
{
/// Decodes @p bytes the way a caller of from_utf8() must: re-feeding a byte that broke a character off
/// and was not consumed, and replacing each Invalid with one U+FFFD.
std::u32string decode_with_replacement(std::string_view bytes)
{
    auto out = std::u32string {};
    auto state = utf8_decoder_state {};
    for (auto const ch: bytes)
    {
        auto const value = static_cast<uint8_t>(ch);
        auto const expectedBefore = state.expectedLength;
        auto result = from_utf8(state, value);
        if (holds_alternative<Invalid>(result))
        {
            out.push_back(U'\uFFFD');
            if (from_utf8_left_byte_unconsumed(expectedBefore, state))
            {
                result = from_utf8(state, value);
                if (holds_alternative<Invalid>(result))
                    out.push_back(U'\uFFFD');
            }
        }
        if (holds_alternative<Success>(result))
            out.push_back(get<Success>(result).value);
    }
    if (state.expectedLength != 0)
        out.push_back(U'\uFFFD');
    return out;
}
} // namespace

TEST_CASE("utf8.from_utf8.table_3_7.violations", "[utf8]")
{
    CHECK(decode_with_replacement("\xC0\x80") == U"\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xC1\xBF") == U"\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xC0\x9B") == U"\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xE0\x80\xAF") == U"\uFFFD\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xE0\x9F\xBF") == U"\uFFFD\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xED\xA0\x80") == U"\uFFFD\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xF0\x80\x80\x80") == U"\uFFFD\uFFFD\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xF4\x90\x80\x80") == U"\uFFFD\uFFFD\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xF5\x80\x80\x80") == U"\uFFFD\uFFFD\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xFF") == U"\uFFFD");
}

TEST_CASE("utf8.from_utf8.table_3_7.maximal_subparts", "[utf8]")
{
    CHECK(decode_with_replacement("\xE0\x80\x41") == U"\uFFFD\uFFFDA");
    CHECK(decode_with_replacement("\xE2\x82\x41") == U"\uFFFDA");          // one subpart: E2 82
    CHECK(decode_with_replacement("\xF0\x9F\x98\x41") == U"\uFFFDA");      // one subpart: F0 9F 98
    CHECK(decode_with_replacement("\xE2\xE2\x82\xAC") == U"\uFFFD\u20AC"); // E2 then a whole euro sign
    CHECK(decode_with_replacement("\xC3\xFF") == U"\uFFFD\uFFFD");
    CHECK(decode_with_replacement("\xE1\x80\xE2\xF0\x91\x92\xF1\xBF\x41") == U"\uFFFD\uFFFD\uFFFD\uFFFDA"); // Unicode 3.9 example
}

TEST_CASE("utf8.from_utf8.table_3_7.boundaries", "[utf8]")
{
    CHECK(decode_with_replacement("\xC2\x80") == U"\u0080");
    CHECK(decode_with_replacement("\xDF\xBF") == U"\u07FF");
    CHECK(decode_with_replacement("\xE0\xA0\x80") == U"\u0800");
    CHECK(decode_with_replacement("\xED\x9F\xBF") == U"\uD7FF");
    CHECK(decode_with_replacement("\xEE\x80\x80") == U"\uE000");
    CHECK(decode_with_replacement("\xEF\xBF\xBF") == U"\uFFFF");
    CHECK(decode_with_replacement("\xF0\x90\x80\x80") == U"\U00010000");
    CHECK(decode_with_replacement("\xF4\x8F\xBF\xBF") == U"\U0010FFFF");
}
