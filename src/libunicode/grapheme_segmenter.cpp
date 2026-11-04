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
#include <libunicode/utf8_grapheme_segmenter.h>

namespace unicode
{

void grapheme_process_init(char32_t nextCodepoint, grapheme_segmenter_state& state) noexcept
{
    auto const Pb = codepoint_properties::get(nextCodepoint);
    auto const B = Pb.grapheme_cluster_break;

    state.previousCodepoint = nextCodepoint;
    state.previousProperties = Pb;
    state.ri_counter = (B == Grapheme_Cluster_Break::Regional_Indicator) ? 1 : 0;
    state.incb_state = (Pb.indic_conjunct_break == Indic_Conjunct_Break::Consonant) ? 1 : 0;
    state.extpic_state = Pb.is_extended_pictographic() ? 1 : 0;
}

bool grapheme_process_breakable(char32_t nextCodepoint, grapheme_segmenter_state& state) noexcept
{
    auto const a = state.previousCodepoint;
    auto const Pa = state.previousProperties;
    auto const A = Pa.grapheme_cluster_break;

    auto const b = nextCodepoint;
    auto const Pb = codepoint_properties::get(b);
    auto const B = Pb.grapheme_cluster_break;

    state.previousCodepoint = b;
    state.previousProperties = Pb;

    // Track the parity of the Regional_Indicator run ending at the current codepoint. GB12/GB13 count
    // the RIs of the whole run, not the pairs formed so far, so this happens ahead of every early return.
    // Only a Regional_Indicator can extend a run, so every other codepoint just clears the counter.
    if (B != Grapheme_Cluster_Break::Regional_Indicator)
        state.ri_counter = 0;
    else [[unlikely]]
    {
        auto const riParityBefore = state.ri_counter;
        auto const followsRegionalIndicator = A == Grapheme_Cluster_Break::Regional_Indicator;
        state.ri_counter = static_cast<uint8_t>(!followsRegionalIndicator || riParityBefore == 0);

        // GB12/GB13: Do not break within emoji flag sequences.
        // That is, do not break between regional indicator (RI) symbols
        // if there is an odd number of RI characters before the break point.
        //
        // Deciding this here is equivalent to deciding it last: with both A and B being Regional_Indicator,
        // no earlier rule can fire (GB3-GB5 need CR/LF/Control, GB6-GB8 need Hangul, GB9/GB9a need
        // Extend/ZWJ/SpacingMark as B, GB9b needs Prepend as A, GB9c needs an InCB=Consonant B, GB11 needs
        // ZWJ as A). The state updates below are fixed for such a pair, as Regional_Indicator is neither
        // InCB=Consonant/Linker/Extend nor Extended_Pictographic nor Extend/ZWJ, so they are replayed here.
        if (followsRegionalIndicator && riParityBefore == 1)
        {
            state.incb_state = 0;
            state.extpic_state = 0;
            return false;
        }
    }

    // GB3: Do not break between a CR and LF. Otherwise, break before and after controls.
    if (A == Grapheme_Cluster_Break::CR && B == Grapheme_Cluster_Break::LF)
        return false;

    // GB4 (a) + GB5 (b) part 1 (C0 characers) + US-ASCII shortcut
    // The US-ASCII part is a pure optimization improving performance
    // in standard Latin text.
    if (a < 128 && b < 128)
    {
        state.incb_state = 0;   // ASCII resets InCB tracking
        state.extpic_state = 0; // ASCII resets ExtPic tracking
        return true;
    }

    // GB9c: Indic conjunct break state machine update.
    // Capture previous state before updating, as GB9c check needs the pre-transition state.
    auto const prev_incb_state = state.incb_state;
    {
        auto const incb = Pb.indic_conjunct_break;
        if (incb == Indic_Conjunct_Break::Consonant)
            state.incb_state = 1;
        else if (incb == Indic_Conjunct_Break::Linker && prev_incb_state >= 1)
            state.incb_state = 2;
        else if (incb == Indic_Conjunct_Break::Extend && prev_incb_state >= 1)
            ; // keep current state — extends do not change conjunct tracking
        else
            state.incb_state = 0;
    }

    // GB11: Extended Pictographic tracking state machine.
    // Capture previous state before updating, as GB11 check needs the pre-transition state.
    auto const prev_extpic_state = state.extpic_state;
    if (prev_extpic_state != 0 || Pb.is_extended_pictographic())
    {
        if (Pb.is_extended_pictographic())
            state.extpic_state = 1;
        else if ((B == Grapheme_Cluster_Break::Extend || B == Grapheme_Cluster_Break::ZWJ) && prev_extpic_state == 1
                 && A != Grapheme_Cluster_Break::ZWJ)
            ; // keep state - Extend* ZWJ extends the ExtPic chain for GB11, but only one terminal ZWJ
        else
            state.extpic_state = 0;
    }

    // GB4: Break after (Control | CR | LF)
    if (A == Grapheme_Cluster_Break::Control || A == Grapheme_Cluster_Break::CR || A == Grapheme_Cluster_Break::LF)
        return true;

    // GB5: Break before (Control | CR | LF)
    if (B == Grapheme_Cluster_Break::Control || B == Grapheme_Cluster_Break::CR || B == Grapheme_Cluster_Break::LF)
        return true;

    // Do not break Hangul syllable sequences.
    // GB6:
    if (A == Grapheme_Cluster_Break::L
        && (B == Grapheme_Cluster_Break::L || B == Grapheme_Cluster_Break::V || B == Grapheme_Cluster_Break::LV
            || B == Grapheme_Cluster_Break::LVT))
        return false;

    // GB7:
    if ((A == Grapheme_Cluster_Break::LV || A == Grapheme_Cluster_Break::V)
        && (B == Grapheme_Cluster_Break::V || B == Grapheme_Cluster_Break::T))
        return false;

    // GB8:
    if ((A == Grapheme_Cluster_Break::LVT || A == Grapheme_Cluster_Break::T) && B == Grapheme_Cluster_Break::T)
        return false;

    // GB9: Do not break before extending characters.
    if (B == Grapheme_Cluster_Break::Extend || B == Grapheme_Cluster_Break::ZWJ)
        return false;

    // GB9a: Do not break before SpacingMarks
    if (B == Grapheme_Cluster_Break::SpacingMark)
        return false;

    // GB9b: or after Prepend characters.
    if (A == Grapheme_Cluster_Break::Prepend)
        return false;

    // GB9c: Do not break within Indic conjunct clusters.
    // Pattern: \p{InCB=Consonant} [\p{InCB=Extend}\p{InCB=Linker}]* \p{InCB=Linker}
    //          [\p{InCB=Extend}\p{InCB=Linker}]* × \p{InCB=Consonant}
    if (Pb.indic_conjunct_break == Indic_Conjunct_Break::Consonant && prev_incb_state == 2)
        return false;

    // GB11: Do not break within emoji modifier sequences or emoji zwj sequences.
    // Full rule: \p{Extended_Pictographic} Extend* ZWJ × \p{Extended_Pictographic}
    if (A == Grapheme_Cluster_Break::ZWJ && Pb.is_extended_pictographic() && prev_extpic_state == 1)
        return false;

    // GB999: Otherwise, break everywhere.
    return true;
}

} // namespace unicode
