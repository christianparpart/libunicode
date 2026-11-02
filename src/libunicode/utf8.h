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
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

namespace unicode
{

/// Converts an UTF-32 codepoint into a UTF-8 sequence.
///
/// @param character UTF-32 character to encode to UTF-8
/// @param result target memory location to start writing to (up to 4 chars)
///
/// @return number of bytes written to.
constexpr inline unsigned to_utf8(char32_t character, uint8_t* result)
{
    if (character <= 0x7F)
    {
        result[0] = static_cast<uint8_t>(character & 0b0111'1111);
        return 1;
    }
    else if (character <= 0x07FF)
    {
        result[0] = static_cast<uint8_t>(((character >> 6) & 0b0001'1111) | 0b1100'0000);
        result[1] = static_cast<uint8_t>(((character >> 0) & 0b0011'1111) | 0b1000'0000);
        return 2;
    }
    if (character <= 0xFFFF)
    {
        result[0] = static_cast<uint8_t>(((character >> 12) & 0b0000'1111) | 0b1110'0000);
        result[1] = static_cast<uint8_t>(((character >> 6) & 0b0011'1111) | 0b1000'0000);
        result[2] = static_cast<uint8_t>(((character >> 0) & 0b0011'1111) | 0b1000'0000);
        return 3;
    }
    else
    {
        result[0] = static_cast<uint8_t>(((character >> 18) & 0b0000'0111) | 0b1111'0000);
        result[1] = static_cast<uint8_t>(((character >> 12) & 0b0011'1111) | 0b1000'0000);
        result[2] = static_cast<uint8_t>(((character >> 6) & 0b0011'1111) | 0b1000'0000);
        result[3] = static_cast<uint8_t>(((character >> 0) & 0b0011'1111) | 0b1000'0000);
        return 4;
    }
}

/// Converts a UTF-32 string into an UTF-8 sring.
inline std::string to_utf8(char32_t const* characters, size_t n)
{
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        uint8_t bytes[4];
        unsigned const len = to_utf8(characters[i], bytes);
        s.append((char const*) bytes, len);
    }

    return s;
}

inline std::string to_utf8(char32_t character)
{
    return to_utf8(&character, 1);
}

inline std::string to_utf8(std::u32string const& characters)
{
    return to_utf8(characters.data(), characters.size());
}

inline std::string to_utf8(std::u32string_view const& characters)
{
    return to_utf8(characters.data(), characters.size());
}

struct utf8_decoder_state
{
    char32_t character = 0;
    unsigned expectedLength = 0;
    unsigned currentLength = 0;
};

// clang-format off
// NOLINTBEGIN(readability-identifier-naming)
struct Invalid { };
struct Incomplete { };
struct Success { char32_t value; };
// NOLINTEND(readability-identifier-naming)
// clang-format on

using ConvertResult = std::variant<Invalid, Incomplete, Success>;

/// Feature macro: from_utf8() accepts well-formed UTF-8 only (Unicode Table 3-7) and reports an
/// Invalid, rather than a codepoint, for overlong forms, surrogates, codepoints past U+10FFFF, and
/// a character broken off by a byte that cannot continue it. A byte that breaks a character off
/// and cannot start one is not consumed; see from_utf8_left_byte_unconsumed().
#define LIBUNICODE_FROM_UTF8_STRICT 1

/// Progressively decodes a UTF-8 codepoint, accepting well-formed UTF-8 only (Unicode Table 3-7):
/// C0, C1 and F5..FF never start a character, and the second byte after E0, ED, F0 and F4 is
/// limited to A0..BF, 80..9F, 90..BF and 80..8F, so overlong forms, surrogates and codepoints
/// past U+10FFFF are Invalid.
///
/// An ill-formed sequence ends at the first byte that cannot continue it, yielding one Invalid for
/// the bytes before it (a maximal subpart). That byte is then processed as itself:
/// - Called at a character boundary, the byte that is Invalid on its own is consumed.
/// - Called inside a character, a byte that may start one (C2..F4) is consumed and starts the next
///   character, so the following call continues it. Any other byte is NOT consumed: the state is
///   reset and the caller feeds the byte again.
ConvertResult from_utf8(utf8_decoder_state& state, uint8_t value) noexcept;

/// Tells a caller of from_utf8() whether the Invalid it just got left the byte it fed unconsumed, so the
/// caller must feed the same byte again, now at a character boundary.
///
/// @param expectedLengthBefore The decoder state's expectedLength before the byte was fed.
/// @param state The decoder state after the byte was fed.
/// @return true if the byte broke a character off and cannot start one itself.
[[nodiscard]] constexpr bool from_utf8_left_byte_unconsumed(unsigned expectedLengthBefore,
                                                            utf8_decoder_state const& state) noexcept
{
    return expectedLengthBefore != 0 && state.expectedLength == 0;
}

inline unsigned from_utf8i(utf8_decoder_state& state, uint8_t value)
{
    auto const result = from_utf8(state, value);

    if (std::holds_alternative<Incomplete>(result))
        return static_cast<unsigned>(-1);

    if (std::holds_alternative<Invalid>(result))
        return static_cast<unsigned>(-2);

    return std::get<Success>(result).value;
}

inline ConvertResult from_utf8(uint8_t const* bytes, size_t* size)
{
    auto state = utf8_decoder_state {};
    auto result = ConvertResult {};

    do
        result = from_utf8(state, *bytes++);
    while (std::holds_alternative<Incomplete>(result));

    if (size)
        *size = state.currentLength;

    return result;
}

#if 0 // TODO(do that later) __cplusplus > 201703L // C++20 (char8_t)
inline ConvertResult from_utf8(char8_t const* bytes, size_t* size)
{
    return from_utf8((uint8_t const*)(bytes), size);
}
#endif

inline ConvertResult from_utf8(char const* bytes, size_t* size)
{
    return from_utf8((uint8_t const*) (bytes), size);
}

namespace detail
{
    // Forward declaration of SIMD-accelerated UTF-8 -> UTF-32 dispatcher (defined in convert.cpp).
    size_t convert_utf8_to_utf32(char const* input, size_t inputSize, char32_t* output) noexcept;
} // namespace detail

/// Converts a UTF-8 string to UTF-32, using SIMD acceleration when available.
template <typename T = char32_t>
inline std::basic_string<T> from_utf8(std::string_view bytes)
{
    static_assert(sizeof(T) == 4);
    if (bytes.empty())
        return {};
    std::basic_string<T> s;
    s.resize(bytes.size()); // worst case: 1 codepoint per byte (all ASCII)
    auto const written = detail::convert_utf8_to_utf32(bytes.data(), bytes.size(), reinterpret_cast<char32_t*>(s.data()));
    s.resize(written);
    return s;
}

} // namespace unicode
