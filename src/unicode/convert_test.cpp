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
#include <unicode/convert.h>
#include <unicode/utf8.h>
#include <unicode/support.h>
#include <catch2/catch_all.hpp>
#include <fmt/format.h>
#include <iterator>

using namespace unicode;
using namespace std::string_literals;
using namespace std;

TEST_CASE("convert.same", "[convert]")
{
    auto const s8 = "Hello, 😀"sv;
    auto t8 = string{};
    unicode::convert_to<char>(s8, back_insert_iterator(t8));
    CHECK(s8 == t8);

    auto const s16 = u"Hello, 😀"sv;
    auto t16 = u16string{};
    unicode::convert_to<char16_t>(s16, back_insert_iterator(t16));
    CHECK(s16 == t16);

    auto const s32 = U"Hello, 😀"sv;
    auto t32 = u32string{};
    unicode::convert_to<char32_t>(s32, back_insert_iterator(t32));
    CHECK(s32 == t32);
}

TEST_CASE("convert.8_to_16", "[convert]")
{
    auto constexpr input = string_view{
        "["
        "\xC3\xB6"          // ö  - german o-umlaut
        "\xE2\x82\xAC"      // €  - EURO sign U+20AC
        "\xF0\x9F\x98\x80"  // 😀 - U+1F600
    };
    u16string output;
    auto bi = back_inserter(output);
    unicode::convert_to<char16_t>(input, bi); // back_inserter(output));
    CHECK(output.size() == 5);
    CHECK(output == u"[ö€😀");
}

TEST_CASE("convert.8_to_32", "[convert]")
{
    auto constexpr input = string_view{
        "["
        "\xC3\xB6"          // ö  - german o-umlaut
        "\xE2\x82\xAC"      // €  - EURO sign U+20AC
        "\xF0\x9F\x98\x80"  // 😀 - U+1F600
    };
    u32string output;
    auto bi = back_inserter(output);
    unicode::convert_to<char32_t>(input, bi); // back_inserter(output));
    CHECK(output.size() == 4);
    CHECK(output == U"[ö€😀");
}

TEST_CASE("convert.utf8.ascii_sse.1")
{
    auto constexpr input    =  "0123456789ABCDEF"sv;
    auto constexpr expected = U"0123456789ABCDEF"sv;

    auto const* i = (char8_type const*) input.data();
    auto const* e = (char8_type const*) input.data() + input.size();

    auto decode = unicode::decoder<char>{};

    u32string output;
    output.resize(input.size());
    auto const [success, nread, nwritten] = decode(i, e, output.data());

    CHECK(success);
    CHECK(nread == input.size());
    output.resize(nwritten);

    CHECK(output == expected);
}

TEST_CASE("convert.utf8.ascii_sse.1.1")
{
    auto constexpr input    =  "0123456789ABCDEFa"sv;
    auto constexpr expected = U"0123456789ABCDEFa"sv;

    auto const* i = (char8_type const*) input.data();
    auto const* e = (char8_type const*) input.data() + input.size();

    auto decode = unicode::decoder<char>{};

    u32string output;
    output.resize(input.size());
    auto const [success, nread, nwritten] = decode(i, e, output.data());

    CHECK(success);
    CHECK(nread == input.size());
    output.resize(nwritten);

    CHECK(output == expected);
}

template <size_t N>
std::string join(array<string_view, N> _chunks)
{
    std::stringstream ss;
    for (auto chunk: _chunks)
        ss << chunk;
    return ss.str();
}

std::u32string ascii2utf32(std::string _input)
{
    std::u32string s;
    s.reserve(_input.size());
    for (char ch: _input)
        s += static_cast<char32_t>(ch);
    return s;
}

TEST_CASE("convert.utf8.ascii_sse.chunked")
{
    auto constexpr input = array<string_view, 3>{
        "0123456789ABCDEF"sv,
        "\\n\\033[1"sv,
        ";2m"sv
    };
    auto expected = ascii2utf32(join(input));

    auto decode = unicode::decoder<char>{};

    u32string output;
    output.resize(expected.size());
    size_t totalWritten = 0;
    for (size_t k = 0; k < input.size(); ++k)
    {
        auto chunk = input[k];
        auto const* i = (char8_type const*) chunk.data();
        auto const* e = (char8_type const*) chunk.data() + chunk.size();

        auto chunkStr = string(chunk);
        auto const [success, nread, nwritten] = decode(i, e, output.data() + totalWritten);
        CHECK(success);
        CHECK(nread == chunk.size());
        CHECK(nwritten == nread); // because it's ASCII
        totalWritten += nwritten;
    }
    CHECK(totalWritten == expected.size()); // because it's ASCII
    output.resize(expected.size());

    CHECK(output == expected);
}

TEST_CASE("convert.utf8.ascii_sse.2")
{
    auto constexpr input    =  "0123456789ABCDEF0123456789ABCDEF"sv;
    auto constexpr expected = U"0123456789ABCDEF0123456789ABCDEF"sv;

    auto const* i = (char8_type const*) input.data();
    auto const* e = (char8_type const*) input.data() + input.size();

    auto decode = unicode::decoder<char>{};

    u32string output;
    output.resize(input.size() * 2);
    auto const [success, nread, nwritten] = decode(i, e, output.data());

    CHECK(success);
    CHECK(nread == input.size());
    output.resize(nwritten);

    CHECK(output == expected);
}

TEST_CASE("convert.utf8.incremental_decode", "[utf8]")
{
    auto constexpr values = string_view{
        "["
        "\xC3\xB6"          // ö  - german o-umlaut
        "\xE2\x82\xAC"      // €  - EURO sign U+20AC
        "\xF0\x9F\x98\x80"  // 😀 - U+1F600
    };
    auto const* p = (char8_type const*) (values.data());
    auto decode = unicode::decoder<char>{};

    // single-byte
    auto result = decode(*p++);
    REQUIRE(result.has_value());
    REQUIRE(result.value() == '[');

    // double-byte
    result = decode(*p++);
    REQUIRE(!result.has_value());
    result = decode(*p++);
    REQUIRE(result.has_value());
    REQUIRE(result.value() == U'\u00F6'); // ö

    // 3 bytes
    result = decode(*p++);
    REQUIRE(!result.has_value());
    result = decode(*p++);
    REQUIRE(!result.has_value());
    result = decode(*p++);
    REQUIRE(result.has_value());
    REQUIRE(result.value() == U'\u20AC'); // €

    // 4 bytes
    result = decode(*p++);
    REQUIRE(!result.has_value());
    result = decode(*p++);
    REQUIRE(!result.has_value());
    result = decode(*p++);
    REQUIRE(!result.has_value());
    result = decode(*p++);
    REQUIRE(result.has_value());
    REQUIRE(result.value() == U'\U0001F600'); // 😀
}
