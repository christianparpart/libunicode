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
#include <libunicode/grapheme_segmenter.h>
#include <libunicode/scan.h>
#include <libunicode/scan_simd_impl.h>
#include <libunicode/simd_detector.h>
#include <libunicode/utf8.h>
#include <libunicode/width.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <span>
#include <string_view>

using std::distance;
using std::get;
using std::holds_alternative;
using std::max;
using std::min;
using std::string_view;

namespace unicode
{

namespace
{
    template <typename T>
    constexpr bool ascending(T low, T val, T high) noexcept
    {
        return low <= val && val <= high;
    }

    constexpr bool is_control(char ch) noexcept
    {
        return static_cast<uint8_t>(ch) < 0x20;
    }

    // Tests if the given byte is a C1 control (0x80..0x9F). At a UTF-8 character boundary such a byte
    // is not text -- it is a control, the ISO 6429 counterpart of the C0 range -- so scan_text must
    // stop at it exactly as it stops at a C0 control, leaving it for the caller to interpret.
    constexpr bool is_c1_control(char ch) noexcept
    {
        auto const value = static_cast<uint8_t>(ch);
        return value >= 0x80 && value <= 0x9F;
    }

    // Tests if given UTF-8 byte is part of a complex Unicode codepoint, that is, a value greater than U+7E.
    constexpr bool is_complex(char ch) noexcept
    {
        return static_cast<uint8_t>(ch) & 0x80;
    }

    // Tests if the given byte is a UTF-8 continuation byte (0b10xx'xxxx), the only kind of byte that may
    // follow a lead byte inside a character.
    constexpr bool is_continuation(char ch) noexcept
    {
        return (static_cast<uint8_t>(ch) & 0b1100'0000) == 0b1000'0000;
    }
} // namespace

// The x86-64 SIMD builds also compile the 256- and 512-bit scanners, each in its own translation unit with its ISA
// flags (scan256.cpp, scan512.cpp), and pick the widest one the CPU can run.
#if (defined(LIBUNICODE_USE_STD_SIMD) || defined(LIBUNICODE_USE_INTRINSICS)) && (defined(__x86_64__) || defined(_M_AMD64))
    #define LIBUNICODE_SCAN_HAS_WIDE_SIMD
#endif

std::span<detail::ascii_scanner const> detail::runnable_ascii_scanners(std::span<ascii_scanner const> all,
                                                                       size_t maxSimdBitWidth) noexcept
{
    auto const canRun = [maxSimdBitWidth](ascii_scanner const& scanner) {
        return scanner.simd_bit_width <= maxSimdBitWidth;
    };
    return { std::ranges::find_if(all, canRun), all.end() };
}

std::span<detail::ascii_scanner const> detail::ascii_scanners() noexcept
{
    static constexpr auto all = std::array {
#if defined(LIBUNICODE_SCAN_HAS_WIDE_SIMD)
        ascii_scanner { .simd_bit_width = 512, .scan = &scan_for_text_ascii_512 },
        ascii_scanner { .simd_bit_width = 256, .scan = &scan_for_text_ascii_256 },
#endif
        ascii_scanner { .simd_bit_width = 128, .scan = &scan_for_text_ascii_simd<128> },
    };
#if defined(LIBUNICODE_SCAN_HAS_WIDE_SIMD)
    // The CPU's widest vector decides where the list starts; max_simd_size() is 128 at least, so the
    // 128-bit scanner is always runnable.
    return runnable_ascii_scanners(all, max_simd_size());
#else
    return all;
#endif
}

size_t detail::scan_for_text_ascii(string_view text, size_t maxColumnCount) noexcept
{
#if defined(LIBUNICODE_SCAN_HAS_WIDE_SIMD)
    static auto const widest = ascii_scanners().front().scan;
    return widest(text, maxColumnCount);
#else
    return scan_for_text_ascii_simd<128>(text, maxColumnCount);
#endif
}

scan_result detail::scan_for_text_nonascii(scan_state& state,
                                           string_view text,
                                           size_t maxColumnCount,
                                           grapheme_cluster_receiver& receiver) noexcept
{
    size_t count = 0;

    char const* start = text.data();
    char const* end = start + text.size();
    char const* input = start;
    char const* clusterStart = start;

    unsigned byteCount = 0; // bytes consumed for the current codepoint
    // Bytes of the first character that an earlier call consumed; they lie in front of this call's text.
    unsigned carriedBytes = state.utf8.expectedLength ? state.utf8.currentLength : 0;

    // How much of the open cluster's width THIS call has contributed. The cluster may have been
    // opened by an earlier call, whose columns are already spent and cannot be taken back here, so
    // this -- not state.reportedClusterWidth -- bounds what a rewind may subtract from count.
    size_t clusterWidthCountedHere = 0;

    char const* resultStart = state.utf8.expectedLength ? start - state.utf8.currentLength : start;
    char const* resultEnd = resultStart;

    // Hands over the cluster spanning [clusterStart, clusterEnd). Called at every cluster boundary
    // and once more when the scan ends, so the last cluster is not dropped. A cluster put back
    // because it did not fit leaves the range empty and is correctly not delivered.
    auto const flushOpenCluster = [&](char const* clusterEnd) noexcept {
        if (clusterEnd > clusterStart)
            receiver.receiveGraphemeCluster(string_view(clusterStart, static_cast<size_t>(clusterEnd - clusterStart)),
                                            state.reportedClusterWidth);
        clusterStart = clusterEnd;
    };

    // Ends the scan at @p stop, in front of a byte that is not text: a control, or the first byte of an
    // ill-formed UTF-8 subsequence. Both are the caller's to interpret, so nothing from @p stop on is
    // counted or consumed. Whatever cluster was open ends there, so hand it over before dropping the state.
    auto const stopAt = [&](char const* stop) noexcept {
        flushOpenCluster(stop);
        state.utf8 = {};
        state.lastCodepointHint = 0;
        state.graphemeState = {};
        state.clusterWidth.reset();
        state.reportedClusterWidth = 0;
        resultEnd = stop;
        input = stop;
    };

    while (input != end && count <= maxColumnCount)
    {
        if (state.utf8.expectedLength != 0)
        {
            // Inside a character, only a continuation byte may follow; one that merely falls in
            // 0x80..0x9F (e.g. the middle byte of U+2018) is part of its codepoint, not a C1 control.
            // Any other byte -- ASCII, a control, another lead byte, or 0xF8..0xFF -- breaks the
            // character off and leaves it ill-formed. (from_utf8() rejects those too, but this check is
            // cheaper than a decode.) The caller decodes the broken character under
            // its own replacement policy, so stop in front of its first byte. When an earlier call
            // began the character, that byte lies in the earlier call's text, where resultStart points.
            if (!is_continuation(*input))
            {
                stopAt(input - state.utf8.currentLength);
                break;
            }
        }
        else if (is_control(*input) || !is_complex(*input) || is_c1_control(*input))
        {
            // Stop at a C1 control (0x80..0x9F) sitting at a character boundary: it is a control, not
            // text, so -- like a C0 control -- it is left for the caller.
            stopAt(input);
            break;
        }

        auto const result = from_utf8(state.utf8, static_cast<uint8_t>(*input++));
        ++byteCount;

        if (holds_alternative<Incomplete>(result))
            continue;

        if (holds_alternative<Success>(result))
        {
            auto const prevCodepoint = state.lastCodepointHint;
            auto const nextCodepoint = get<Success>(result).value;
            char const* const codepointStart = input - byteCount;
            byteCount = 0;
            carriedBytes = 0;
            state.lastCodepointHint = nextCodepoint;

            bool const breakable = [&] {
                if (!prevCodepoint)
                {
                    grapheme_process_init(nextCodepoint, state.graphemeState);
                    return true;
                }
                return grapheme_process_breakable(nextCodepoint, state.graphemeState);
            }();
            if (breakable)
            {
                // The incoming codepoint opens the next cluster, so it is measured on its own rather
                // than carrying anything over from the cluster just closed.
                auto nextCluster = grapheme_cluster_width_accumulator {};
                nextCluster.push(nextCodepoint);
                auto const nextWidth = static_cast<size_t>(nextCluster.width());

                if (count + nextWidth > maxColumnCount)
                {
                    // Currently scanned grapheme cluster won't fit. Break at its start.
                    input = codepointStart;
                    break;
                }

                // The cluster that just ended spans everything up to this codepoint.
                flushOpenCluster(codepointStart);

                count += nextWidth;
                clusterWidthCountedHere = nextWidth;
                state.clusterWidth = nextCluster;
                state.reportedClusterWidth = nextWidth;
                resultEnd = input;
            }
            else
            {
                // The codepoint joins the current cluster, which may widen it (a spacing mark, a
                // conjunct, VS16) or narrow it (VS15). Only the difference is counted, so a cluster
                // split across two calls is not measured twice.
                state.clusterWidth.push(nextCodepoint);
                auto const updatedWidth = static_cast<size_t>(state.clusterWidth.width());

                if (updatedWidth >= state.reportedClusterWidth)
                {
                    auto const added = updatedWidth - state.reportedClusterWidth;
                    if (added != 0 && count + added > maxColumnCount)
                    {
                        // The cluster grew past what is left of the line. Put the whole cluster
                        // back -- a caller cannot render part of one -- and un-count only what this
                        // call contributed to it.
                        count -= clusterWidthCountedHere;
                        input = clusterStart;
                        resultEnd = clusterStart;
                        state.clusterWidth.reset();
                        state.reportedClusterWidth = 0;
                        break;
                    }
                    count += added;
                    clusterWidthCountedHere += added;
                }
                else
                {
                    auto const removed = min(state.reportedClusterWidth - updatedWidth, clusterWidthCountedHere);
                    count -= removed;
                    clusterWidthCountedHere -= removed;
                }
                state.reportedClusterWidth = updatedWidth;
                resultEnd = input;
            }
        }
        else
        {
            // Either a byte that cannot begin a character (0x80..0xBF, C0, C1, F5..FF), or the second byte
            // of a character that Table 3-7 rules out (an overlong form, a surrogate, or a codepoint past
            // U+10FFFF). Stop in front of the character's first byte and leave it to the caller likewise.
            assert(holds_alternative<Invalid>(result));
            stopAt(input - byteCount - carriedBytes);
            break;
        }
    }

    // Hand over whatever cluster the scan ended inside of.
    flushOpenCluster(input);

    assert(resultStart <= resultEnd);

    state.next = input;
    return { count, resultStart, resultEnd };
}

scan_result scan_text(scan_state& state, std::string_view text, size_t maxColumnCount) noexcept
{
    return scan_text(state, text, maxColumnCount, null_receiver::get());
}

scan_result scan_text(scan_state& state,
                      std::string_view text,
                      size_t maxColumnCount,
                      grapheme_cluster_receiver& receiver) noexcept
{
    //       ----(a)--->   A   -------> END
    //                   ^   |
    //                   |   |
    // Start            (a) (b)
    //                   |   |
    //                   |   v
    //       ----(b)--->   B   -------> END

    enum class NextState
    {
        Trivial,
        Complex
    };

    auto result = scan_result { 0, text.data(), text.data() };

    if (state.next == nullptr)
        state.next = text.data();

    // If state indicates that we previously started consuming a UTF-8 sequence but did not complete yet,
    // attempt to finish that one first.
    if (state.utf8.expectedLength != 0)
    {
        result = detail::scan_for_text_nonascii(state, text, maxColumnCount, receiver);
        text = std::string_view(result.end, static_cast<size_t>(std::distance(result.end, text.data() + text.size())));
    }

    if (text.empty())
        return result;

    auto nextState = is_complex(text.front()) ? NextState::Complex : NextState::Trivial;
    while (result.count < maxColumnCount && state.next != (text.data() + text.size()))
    {
        switch (nextState)
        {
            case NextState::Trivial: {
                auto const count = detail::scan_for_text_ascii(text, maxColumnCount - result.count);
                if (!count)
                    return result;
                receiver.receiveAsciiSequence(text.substr(0, count));

                // A cluster does not have to respect the boundary between the two scanners: the last
                // character of this run may be the base of one that continues into the non-ASCII
                // bytes behind it, as `a` does in "a<visarga>". Seed the segmenter and the width
                // accumulator with it so the scan below joins the two instead of treating the mark
                // as opening a fresh cluster -- and record the one column already counted for it, so
                // only what the mark adds is counted a second time.
                auto const lastAscii = static_cast<char32_t>(static_cast<uint8_t>(text[count - 1]));
                state.lastCodepointHint = lastAscii;
                grapheme_process_init(lastAscii, state.graphemeState);
                state.clusterWidth.reset();
                state.clusterWidth.push(lastAscii);
                state.reportedClusterWidth = 1;

                result.count += count;
                state.next += count;
                result.end += count;
                nextState = NextState::Complex;
                text.remove_prefix(count);
                break;
            }
            case NextState::Complex: {
                auto const sub = detail::scan_for_text_nonascii(state, text, maxColumnCount - result.count, receiver);

                // Zero COLUMNS is not zero BYTES. A codepoint that joins the open cluster without
                // widening it -- a combining mark, a variation selector on a base that has no
                // variation sequence -- consumes its bytes and adds nothing, and after the ASCII
                // scan above handed over, it arrives here on its own. Treating `count` as the
                // measure of progress dropped it: state.next had moved past those bytes while
                // result.end had not, and a caller that prints [start, end) and resumes at next
                // never saw them. "e" + U+0301 reached such a caller as a bare "e".
                //
                // The trivial arm may use count for this because there it means the same thing:
                // no ASCII bytes at the front, hence nothing consumed.
                auto const consumed = static_cast<size_t>(std::distance(sub.start, sub.end));
                result.count += sub.count;
                result.end = sub.end;
                if (!consumed)
                    return result;
                nextState = NextState::Trivial;
                text.remove_prefix(consumed);
                break;
            }
        }
    }

    assert(result.start <= result.end);
    assert(result.end <= state.next);

    return result;
}

} // namespace unicode
