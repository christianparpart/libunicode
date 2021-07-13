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

#include <string_view>
#include <array>
#include <type_traits>
#include <iterator>
#include <optional>

#if defined(__linux__)
    #include <emmintrin.h>
    #include <immintrin.h>
    #include <xmmintrin.h>
#endif

namespace unicode {

template <typename T> struct decoder;
template <typename T> struct encoder;

struct decoder_status
{
    bool success;
    size_t read_offset;
    size_t write_offset;
};

// {{{ SSE optimizations

#if !defined(_MSC_VER)
    #define KEWB_ALIGN_FN  __attribute__((aligned (128)))
    #ifdef __OPTIMIZE__
        #define LIBUNICODE_FORCE_INLINE   inline __attribute__((always_inline))
    #else
        #define LIBUNICODE_FORCE_INLINE   inline
    #endif
#else
    #define KEWB_ALIGN_FN
#endif

LIBUNICODE_FORCE_INLINE uint32_t trailingZeros(int32_t x) noexcept
{
    return  __builtin_ctz((unsigned int) x);
}

namespace accelerator
{
    struct sse {};
    struct avx512bw {};
};

template <typename Accelerator>
void convertAsciiBlockOnce(unsigned char const*& _begin, char32_t*& _output) noexcept;

#if defined(__AVX512BW__)
template <>
LIBUNICODE_FORCE_INLINE
void convertAsciiBlockOnce<accelerator::avx512bw>(unsigned char const*& _begin, char32_t*& _output) noexcept
{
    __m128i  input    = _mm_loadu_si128((__m128i const*) _begin); // VMOVDQU: load 16 bytes
    uint32_t mask     = _mm_movemask_epi8(input);                 // VPMOVMSKB: Determine which octets have high bit set
    __m512i  extended = _mm512_cvtepu8_epi32(input);              // VPMOVXZBD: packed zero-extend bytes to DWORD's
    _mm512_store_epi64(_output, extended);                        // VMOVDQA32: Write to memory

#if 1
    auto const incr = /* mask == 0 ? 16 : */ trailingZeros(mask);
    _begin += incr;
    _output += incr;
#else
    if (mask == 0) {
        _begin += 16;
        _output += 16;
    } else {
        auto const incr = trailingZeros(mask);
        _begin += incr;
        _output += incr;
    }
#endif
}
#endif

#if defined(__SSE__)
template <>
LIBUNICODE_FORCE_INLINE
void convertAsciiBlockOnce<accelerator::sse>(unsigned char const*& _begin, char32_t*& _output) noexcept
{
#if defined(__linux__)
    __m128i     chunk, half, qrtr, zero;
    int32_t     mask, incr;

    zero  = _mm_set1_epi8(0);                           //- Zero out the interleave register
    chunk = _mm_loadu_si128((__m128i const*) _begin);     //- Load a register with 8-bit bytes
    mask  = _mm_movemask_epi8(chunk);                   //- Determine which octets have high bit set

    half = _mm_unpacklo_epi8(chunk, zero);              //- Unpack bytes 0-7 into 16-bit words
    qrtr = _mm_unpacklo_epi16(half, zero);              //- Unpack words 0-3 into 32-bit dwords
    _mm_storeu_si128((__m128i*) _output, qrtr);            //- Write to memory
    qrtr = _mm_unpackhi_epi16(half, zero);              //- Unpack words 4-7 into 32-bit dwords
    _mm_storeu_si128((__m128i*) (_output + 4), qrtr);      //- Write to memory

    half = _mm_unpackhi_epi8(chunk, zero);              //- Unpack bytes 8-15 into 16-bit words
    qrtr = _mm_unpacklo_epi16(half, zero);              //- Unpack words 8-11 into 32-bit dwords
    _mm_storeu_si128((__m128i*) (_output + 8), qrtr);      //- Write to memory
    qrtr = _mm_unpackhi_epi16(half, zero);              //- Unpack words 12-15 into 32-bit dwords
    _mm_storeu_si128((__m128i*) (_output + 12), qrtr);     //- Write to memory

    //- If no bits were set in the mask, then all 16 code units were ASCII, and therefore
    //  both pointers are advanced by 16.
    //
    if (mask == 0)
    {
        _begin += 16;
        _output += 16;
    }

    //- Otherwise, the number of trailing (low-order) zero bits in the mask indicates the number
    //  of ASCII code units starting from the lowest byte address.
    else
    {
        incr = trailingZeros(mask);
        _begin += incr;
        _output += incr;
    }
#else
    // other platforms?
#endif
}
#endif

// void sseConvertAsciiBlock(unsigned char const*& _begin, unsigned char const* _end, uint32_t* _output)
// {
//     static_assert(sizeof(__m128i) == 16);
//     assert(std::distance(_begin, _end) % 16 == 0)
//     while (_begin <= _end - sizeof(__m128i))
//     {
//         if (*_begin < 0x80)
//             sseConvertAsciiBlockOnce(_begin, _output);
//         else if (auto const opt = (*this)(_begin, _end))
//             *_output++ = *opt;
//         else
//             return decoder_status{
//                 false,
//                 static_cast<size_t>(_begin - inputBegin),
//                 static_cast<size_t>(_output - outputBegin),
//             };
//     }
// }

// }}}
template<> struct encoder<char> // {{{
{
    template <typename OutputIterator>
    constexpr OutputIterator operator()(char32_t _input, OutputIterator _output)
    {
        if (_input <= 0x7F)
        {
            *_output++ = static_cast<char>(_input & 0b0111'1111);
        }
        else if (_input <= 0x07FF)
        {
            *_output++ = static_cast<char>(((_input >> 6) & 0b0001'1111) | 0b1100'0000);
            *_output++ = static_cast<char>(((_input >> 0) & 0b0011'1111) | 0b1000'0000);
        }
        else if (_input <= 0xFFFF)
        {
            *_output++ = static_cast<char>(((_input >> 12) & 0b0000'1111) | 0b1110'0000);
            *_output++ = static_cast<char>(((_input >>  6) & 0b0011'1111) | 0b1000'0000);
            *_output++ = static_cast<char>(((_input >>  0) & 0b0011'1111) | 0b1000'0000);
        }
        else
        {
            *_output++ = static_cast<char>(((_input >> 18) & 0b0000'0111) | 0b1111'0000);
            *_output++ = static_cast<char>(((_input >> 12) & 0b0011'1111) | 0b1000'0000);
            *_output++ = static_cast<char>(((_input >>  6) & 0b0011'1111) | 0b1000'0000);
            *_output++ = static_cast<char>(((_input >>  0) & 0b0011'1111) | 0b1000'0000);
        }
        return _output;
    }
}; // }}}
template<> struct decoder<char> // {{{
{
    char32_t character = 0;
    unsigned expectedLength = 0;
    unsigned currentLength = 0;

    decoder_status operator()(uint8_t const* _begin,
                              uint8_t const* _end,
                              char32_t* _output)
    {
        uint8_t const* inputBegin = _begin;
        char32_t const* outputBegin = _output;

#if defined(__SSE__) // TODO: does this work on Windows?
        // TODO: ensure we can provide more accelerators: SSE4, AVX, AVX512
        while (_begin <= _end - sizeof(__m128i))
        {
            if (*_begin < 0x80)
                convertAsciiBlockOnce<accelerator::sse>(_begin, _output);
                //convertAsciiBlockOnce<accelerator::avx512bw>(_begin, _output);
            else if (auto const opt = (*this)(_begin, _end))
                *_output++ = *opt;
            else
                return decoder_status{
                    false,
                    static_cast<size_t>(_begin - inputBegin),
                    static_cast<size_t>(_output - outputBegin),
                };
        }
#endif

        while (_begin < _end)
        {
            if (*_begin < 0x80)
                *_output++ = *_begin++;
            else if (auto const opt = (*this)(_begin, _end))
                *_output++ = *opt;
            else
                return decoder_status{
                    false,
                    static_cast<size_t>(_begin - inputBegin),
                    static_cast<size_t>(_output - outputBegin),
                };
        }

        return decoder_status{
            true,
            static_cast<size_t>(_begin - inputBegin),
            static_cast<size_t>(_output - outputBegin),
        };
    }

    constexpr std::optional<char32_t> operator()(uint8_t _byte)
    {
        if (!expectedLength)
        {
            if ((_byte & 0b1000'0000) == 0)
            {
                currentLength = 1;
                return char32_t(_byte);
            }
            else if ((_byte & 0b1110'0000) == 0b1100'0000)
            {
                currentLength = 1;
                expectedLength = 2;
                character = _byte & 0b0001'1111;
            }
            else if ((_byte & 0b1111'0000) == 0b1110'0000)
            {
                currentLength = 1;
                expectedLength = 3;
                character = _byte & 0b0000'1111;
            }
            else if ((_byte & 0b1111'1000) == 0b1111'0000)
            {
                currentLength = 1;
                expectedLength = 4;
                character = _byte & 0b0000'0111;
            }
            else
                return std::nullopt; // invalid
        }
        else
        {
            character <<= 6;
            character |= _byte & 0b0011'1111;
            currentLength++;
        }

        if  (currentLength < expectedLength)
            return std::nullopt; // incomplete

        expectedLength = 0; // reset state
        return character;
    }

    template <
        typename InputIterator,
        typename InputSentinel,
        std::enable_if_t<std::is_convertible_v<decltype(*std::declval<InputIterator>()), char>, int> = 0
    >
    constexpr std::optional<char32_t> operator()(InputIterator& _input, InputSentinel _end)
    {
        using std::nullopt;

        if (_input == _end)
            return std::nullopt;

        auto const ch0 = uint8_t(*_input++);
        if (ch0 < 0x80) // 0xxx_xxxx
            return static_cast<char32_t>(ch0);

        if (ch0 < 0xC0)
            return nullopt;

        if (ch0 < 0xE0) // 110x_xxxx 10xx_xxxx
        {
            if (_input == _end)
                return std::nullopt;

            auto const ch1 = uint8_t(*_input++);
            if ((ch1 >> 6) != 2)
                return nullopt;
            return static_cast<char32_t>((ch0 << 6) + ch1 - 0x3080);
        }

        if (ch0 < 0xF0) // 1110_xxxx 10xx_xxxx 10xx_xxxx
        {
            if (!(_input + 1 < _end))
                return std::nullopt;

            auto const ch1 = uint8_t(*_input++);
            if (ch1 >> 6 != 2)
                return nullopt;
            auto const ch2 = uint8_t(*_input++);
            if (ch2 >> 6 != 2)
                return nullopt;
            return static_cast<char32_t>((ch0 << 12) + (ch1 << 6) + ch2 - 0xE2080);
        }
        if (ch0 < 0xF8) // 1111_0xxx 10xx_xxxx 10xx_xxxx 10xx_xxxx
        {
            if (!(_input + 2 < _end))
                return std::nullopt;
            auto const ch1 = uint8_t(*_input++);
            if (ch1 >> 6 != 2)
                return nullopt;
            auto const ch2 = uint8_t(*_input++);
            if (ch2 >> 6 != 2)
                return nullopt;
            auto const ch3 = uint8_t(*_input++);
            if (ch3 >> 6 != 2)
                return nullopt;
            return static_cast<char32_t>((ch0 << 18) + (ch1 << 12) + (ch2 << 6) + ch3 - 0x3C82080);
        }
        if (ch0 < 0xFC) // 1111_10xx 10xx_xxxx 10xx_xxxx 10xx_xxxx 10xx_xxxx
        {
            if (!(_input + 3 < _end))
                return std::nullopt;
            auto const ch1 = uint8_t(*_input++);
            if (ch1 >> 6 != 2)
                return nullopt;
            auto const ch2 = uint8_t(*_input++);
            if (ch2 >> 6 != 2)
                return nullopt;
            auto const ch3 = uint8_t(*_input++);
            if (ch3 >> 6 != 2)
                return nullopt;
            auto const ch4 = uint8_t(*_input++);
            if (ch4 >> 6 != 2)
                return nullopt;
            auto const a = static_cast<uint32_t>((ch0 << 24u) + (ch1 << 18u) + (ch2 << 12u) + (ch3 << 6u) + ch4);
            return static_cast<char32_t>(a - 0xFA082080lu);
        }
        if (ch0 < 0xFE) // 1111_110x 10xx_xxxx 10xx_xxxx 10xx_xxxx 10xx_xxxx 10xx_xxxx
        {
            auto const ch1 = uint8_t(*_input++);
            if (ch1 >> 6 != 2)
                return nullopt;
            auto const ch2 = uint8_t(*_input++);
            if (ch2 >> 6 != 2)
                return nullopt;
            auto const ch3 = uint8_t(*_input++);
            if (ch3 >> 6 != 2)
                return nullopt;
            auto const ch4 = uint8_t(*_input++);
            if (ch4 >> 6 != 2)
                return nullopt;
            auto const ch5 = uint8_t(*_input++);
            if (ch5 >> 6 != 2)
                return nullopt;
            auto const a = static_cast<uint32_t>((ch0 << 30) + (ch1 << 24) + (ch2 << 18) + (ch3 << 12) + (ch4 << 6) + ch5);
            return static_cast<char32_t>(a - 0x82082080);
        }
        return nullopt;
    }
}; // }}}
template<> struct encoder<char16_t> // {{{
{
    using char_type = char16_t;

    template <typename OutputIterator>
    constexpr OutputIterator operator()(char32_t _input, OutputIterator _output)
    {
        if (_input < 0xD800) // [0x0000 .. 0xD7FF]
        {
            *_output++ = char_type(_input);
            return _output;
        }
        else if (_input < 0x10000)
        {
            if (_input < 0xE000)
                return _output; // The UTF-16 code point can not be in surrogate range.

            // [0xE000 .. 0xFFFF]
            *_output++ = char_type(_input);
            return _output;
        }
        else if (_input < 0x110000) // [0xD800 .. 0xDBFF] [0xDC00 .. 0xDFFF]
        {
            *_output++ = char_type(0xD7C0 + (_input >> 10));
            *_output++ = char_type(0xDC00 + (_input & 0x3FF));
            return _output;
        }
        else
            return _output; // Too large the UTF-16  code point.
    }
}; // }}}
template<> struct decoder<char16_t> // {{{
{
    template <typename InputIterator>
    constexpr std::optional<char32_t> operator()(InputIterator& _input)
    {
        auto const ch0 = *_input++;

        if (ch0 < 0xD800) // [0x0000 .. 0xD7FF]
            return ch0;

        if (ch0 < 0xDC00) // [0xD800 .. 0xDBFF], [0xDC00 .. 0xDFFF]
        {
            auto const ch1 = *_input++;
            if ((ch1 >> 10) != 0x37)
                return std::nullopt; // The low UTF-16 surrogate character is expected.

            return (ch0 << 10) + ch1 - 0x35FDC00;
        }

        if (ch0 < 0xE000)
            return std::nullopt; // The high UTF-16 surrogate character is expected.

        // [0xE000 .. 0xFFFF]
        return ch0;
    }
}; // }}}
template<> struct encoder<char32_t> // {{{ (no-op)
{
    using char_type = char32_t;

    template <typename OutputIterator>
    constexpr OutputIterator operator()(char32_t _input, OutputIterator _output)
    {
        *_output++ = _input;
        return _output;
    }
}; // }}}
template<> struct decoder<char32_t> // {{{ (no-op)
{

    template <typename InputIterator, typename InputSentinel>
    constexpr std::optional<char32_t> operator()(InputIterator& _input, InputSentinel _end)
    {
        if (_input != _end)
            return *_input++;
        else
            return std::nullopt;
    }
}; // }}}
template<> struct encoder<wchar_t> // {{{
{
    using char_type = wchar_t;

    template <typename OutputIterator>
    constexpr OutputIterator operator()(char32_t _input, OutputIterator _output)
    {
        static_assert(sizeof(wchar_t) == 2 || sizeof(wchar_t) == 4);

        if constexpr (sizeof(wchar_t) == 2)
            return encoder<char16_t>{}(_input, _output);
        else
            return encoder<char32_t>{}(_input, _output);
    }
}; // }}}
template<> struct decoder<wchar_t> // {{{
{
    template <typename InputIterator>
    constexpr std::optional<char32_t> operator()(InputIterator& _input)
    {
        static_assert(sizeof(wchar_t) == 2 || sizeof(wchar_t) == 4);

        if constexpr (sizeof(wchar_t) == 2)
            return decoder<char16_t>{}(_input);
        else
            return decoder<char32_t>{}(_input);
    }
}; // }}}

namespace detail // {{{
{
    template <typename SourceRange, typename OutputIterator>
    OutputIterator  convert_identity(SourceRange&& s, OutputIterator t)
    {
        for (auto const c : s)
            *t++ = c;
        return t;
    }
} // }}}

/// @p _input with element type @p S to the appropricate type of @p _output.
template <typename T, typename OutputIterator, typename S>
OutputIterator convert_to(std::basic_string_view<S> _input, OutputIterator _output)
{
    if constexpr (std::is_same_v<S, T>)
        return detail::convert_identity(_input, _output);
    else
    {
        auto i = begin(_input);
        auto e = end(_input);
        decoder<S> read{};
        encoder<T> write{};
        while (i != e)
        {
            auto const outChar = read(i, e);
            if (outChar.has_value())
                _output = write(outChar.value(), _output);
        }
        return _output;
    }
}

/// Converts a string of element type @p <S> into string of element type @p <T>.
template <typename T, typename S>
std::basic_string<T> convert_to(std::basic_string_view<S> in)
{
    std::basic_string<T> out;
    convert_to<T>(in, std::back_inserter(out));
    return out;
}

template <typename T, typename S,
    std::enable_if_t<
        std::is_same_v<S, char> ||
        std::is_same_v<S, char16_t> ||
        std::is_same_v<S, char32_t>
        , int
    > = 0
>
std::basic_string<T> convert_to(S _in)
{
    std::basic_string_view<S> in(&_in, 1);
    std::basic_string<T> out;
    convert_to<T>(in, std::back_inserter(out));
    return out;
}

}
