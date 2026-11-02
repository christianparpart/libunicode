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
#include <libunicode/utf8.h>

namespace unicode
{

namespace
{
    // Tests whether @p value is a byte that may start a character (Unicode Table 3-7): C2..F4.
    // C0 and C1 would start overlong forms, and F5..FF would encode past U+10FFFF.
    constexpr bool is_lead_byte(uint8_t value) noexcept
    {
        return 0xC2 <= value && value <= 0xF4;
    }

    // Tests whether @p value may follow the lead byte the state holds, as the character's second byte.
    // The lead's payload bits identify the four leads whose second byte is narrower than 80..BF:
    // E0 (A0..BF), ED (80..9F), F0 (90..BF) and F4 (80..8F); the rest of 80..BF is Table 3-7's
    // exclusion of overlong forms, surrogates and codepoints past U+10FFFF.
    constexpr bool is_valid_second_byte(utf8_decoder_state const& state, uint8_t value) noexcept
    {
        if (state.expectedLength == 3)
        {
            if (state.character == 0x0)
                return value >= 0xA0;
            if (state.character == 0xD)
                return value <= 0x9F;
        }
        else if (state.expectedLength == 4)
        {
            if (state.character == 0x0)
                return value >= 0x90;
            if (state.character == 0x4)
                return value <= 0x8F;
        }
        return true;
    }
} // namespace

ConvertResult from_utf8(utf8_decoder_state& state, uint8_t value) noexcept
{
    if (!state.expectedLength)
    {
        state.currentLength = 1;
        if ((value & 0b1000'0000) == 0)
            return Success { value };
        if (!is_lead_byte(value))
            return Invalid {}; // a stray continuation byte, C0, C1, or F5..FF
        if (value < 0xE0)
        {
            state.expectedLength = 2;
            state.character = value & 0b0001'1111;
        }
        else if (value < 0xF0)
        {
            state.expectedLength = 3;
            state.character = value & 0b0000'1111;
        }
        else
        {
            state.expectedLength = 4;
            state.character = value & 0b0000'0111;
        }
    }
    else if ((value & 0b1100'0000) != 0b1000'0000 || (state.currentLength == 1 && !is_valid_second_byte(state, value)))
    {
        // The character is broken off. Its bytes so far are one maximal subpart.
        state.expectedLength = 0;
        // A byte that may start a character does so right here, and is consumed. Any other byte
        // (ASCII, a continuation byte out of range, C0, C1, F5..FF) is not consumed: the caller
        // feeds it again, now at a character boundary.
        if (is_lead_byte(value))
            from_utf8(state, value);
        return { Invalid {} };
    }
    else
    {
        state.character <<= 6;
        state.character |= value & 0b0011'1111;
        state.currentLength++;
    }

    if (state.currentLength < state.expectedLength)
        return { Incomplete {} };

    state.expectedLength = 0; // reset state
    return { Success { state.character } };
}

} // namespace unicode
