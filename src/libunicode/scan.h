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

#include <libunicode/grapheme_segmenter.h>
#include <libunicode/utf8.h>
#include <libunicode/width.h>

#include <span>
#include <string_view>

namespace unicode
{

/// Holds the result of a call to scan_test().
struct scan_result
{
    /// Number of columns scanned.
    /// One column equals a single narrow-width codepoint.
    /// Codepoints with property East Asian Width Wide are treated as two columns.
    size_t count;

    /// Pointer to UTF-8 grapheme cluster start.
    char const* start;

    /// Pointer to UTF-8 grapheme cluster end, i.e. one byte behind
    /// the last successfuly processed complete UTF-8 byte..
    char const* end;
};

/// Holds the state to keep through a consecutive sequence of calls to scan_test().
///
/// This state holds the UTF-8 decoding state, if processing had to be stopped
/// at an incomplete UTF-8 byte sequence,
/// and the last decoded Unicode codepoint necessary for grapheme cluster segmentation.
struct scan_state
{
    utf8_decoder_state utf8 {};
    char32_t lastCodepointHint {};
    grapheme_segmenter_state graphemeState {};

    /// Measures the grapheme cluster currently being scanned.
    ///
    /// This belongs to the state rather than to scan_text()'s frame because a cluster may straddle
    /// two calls, and it has to stay in step with graphemeState above: measuring the second half of
    /// a cluster with a fresh accumulator loses whatever the first half contributed.
    grapheme_cluster_width_accumulator clusterWidth {};

    /// Columns already reported for the cluster clusterWidth is measuring.
    ///
    /// A cluster that grows contributes only the difference, so the counts returned by a sequence of
    /// calls stay additive no matter where the chunk boundaries fall.
    size_t reportedClusterWidth {};

    /// Pointer to one byte after the last scanned codepoint.
    char const* next {};
};

/// Callback-interface that allows precisely understanding the structure of a UTF-8 sequence.
class grapheme_cluster_receiver
{
  public:
    virtual ~grapheme_cluster_receiver() = default;

    virtual void receiveAsciiSequence(std::string_view codepoints) noexcept = 0;
    virtual void receiveGraphemeCluster(std::string_view codepoints, size_t columnCount) noexcept = 0;

    /// Not called by scan_text(), which stops in front of an ill-formed UTF-8 subsequence instead of
    /// delivering it (see LIBUNICODE_SCAN_TEXT_STOPS_AT_ILL_FORMED_UTF8). Kept so that receivers written
    /// against earlier releases still compile.
    virtual void receiveInvalidGraphemeCluster() noexcept = 0;
};

/// Quite obviousely, this grapheme_cluster_receiver will do nothing.
class null_receiver final: public grapheme_cluster_receiver
{
  public:
    void receiveAsciiSequence(std::string_view) noexcept override {}
    void receiveGraphemeCluster(std::string_view, size_t) noexcept override {}
    void receiveInvalidGraphemeCluster() noexcept override {}

    static null_receiver& get() noexcept
    {
        static null_receiver instance {};
        return instance;
    }
};

namespace detail
{
    size_t scan_for_text_ascii(std::string_view text, size_t maxColumnCount) noexcept;

    template <size_t SimdBitWidth>
    size_t scan_for_text_ascii_simd(std::string_view text, size_t maxColumnCount) noexcept;
    size_t scan_for_text_ascii_256(std::string_view text, size_t maxColumnCount) noexcept;
    size_t scan_for_text_ascii_512(std::string_view text, size_t maxColumnCount) noexcept;

    /// One ASCII scanner this build provides.
    struct ascii_scanner
    {
        /// The vector width it was built for, in bits; on builds without SIMD the single 128 row is the scalar loop.
        size_t simd_bit_width;

        /// Counts the ASCII text at the start of @p text, stopping at @p maxColumnCount.
        size_t (*scan)(std::string_view text, size_t maxColumnCount) noexcept;
    };

    /// Selects the scanners a CPU can run from a table of scanners listed widest first.
    ///
    /// @param all Every scanner a build provides, widest first, ending with the 128-bit one.
    /// @param maxSimdBitWidth The widest vector the CPU can run, in bits, as max_simd_size() reports it.
    /// @return The tail of @p all that starts at its first scanner no wider than @p maxSimdBitWidth.
    std::span<ascii_scanner const> runnable_ascii_scanners(std::span<ascii_scanner const> all, size_t maxSimdBitWidth) noexcept;

    /// Lists every ASCII scanner this build provides that this CPU can run, widest first.
    ///
    /// scan_for_text_ascii() runs the first. Tests run them all, so that each vector width is covered
    /// on a machine that has it, not only the widest one.
    ///
    /// Probes the CPU on every call; hot-path callers cache the result.
    std::span<ascii_scanner const> ascii_scanners() noexcept;

    scan_result scan_for_text_nonascii(scan_state& state,
                                       std::string_view text,
                                       size_t maxColumnCount,
                                       grapheme_cluster_receiver& receiver) noexcept;
} // namespace detail

/// Scans a sequence of UTF-8 encoded bytes.
///
/// This call will return early one of the conditions is met:
///
/// - given the input sequence, the right most complete UTF-8 sequence is processed,
/// - maxColumnCount is reached and the next grapheme cluster would exceed the given limit,
/// - a control character is about to be processed. Both the C0 (0x00..0x1F) and the C1
///   (0x80..0x9F) control ranges count; a C1 byte only stops the scan at a UTF-8 character
///   boundary, so a continuation byte that happens to fall in 0x80..0x9F stays part of its codepoint.
/// - an ill-formed UTF-8 subsequence is about to be processed: a byte that cannot begin a
///   character (0x80..0xBF, 0xC0, 0xC1, 0xF5..0xFF), a character that Table 3-7 rules out, or a
///   character broken off by a byte that cannot continue it. As at a control, the scan stops in front of the subsequence's first
///   byte and leaves state.next there, so the caller decodes it under its own replacement policy. Nothing ill-formed is counted
///   or returned, so no column counted lies outside the returned text. A character that an earlier call began has its first byte
///   in that call's text, before
///   @p text: the same place the returned start points to when such a character completes.
///
/// When this function returns, it is guaranteed to not contain an incomplete UTF-8 sequence
/// at the end of the output sequence. A character cut off by the end of @p text is not
/// ill-formed: its bytes are consumed into the passed UTF-8 decoder state, and calling this
/// function again with more bytes will resume decoding it.
///
/// Well-formedness is that of the Unicode Standard's Table 3-7: C0, C1 and F5..FF never begin a
/// character, and the second byte after E0, ED, F0 and F4 is limited to A0..BF, 80..9F, 90..BF and
/// 80..8F. Overlong forms, surrogates and codepoints past U+10FFFF are therefore ill-formed and end
/// the scan like any other ill-formed subsequence; the caller sees them as the maximal subpart they
/// are and, per the Standard, replaces each one with a single U+FFFD.
///
/// @return scanned textual result. This is, a sequence of complete UTF-8 codepoints, none of
///         them ill-formed and none incomplete at the end.
scan_result scan_text(scan_state& state, std::string_view text, size_t maxColumnCount) noexcept;

/// Feature macro: scan_text() stops at a C1 control byte (0x80..0x9F) sitting at a UTF-8 character
/// boundary, rather than rendering it as U+FFFD. Downstream terminal parsers key off this to know
/// whether they must guard against a mid-run 8-bit C1 control being swallowed as text.
#define LIBUNICODE_SCAN_TEXT_STOPS_AT_C1_CONTROLS 1

/// Feature macro: scan_text() stops in front of an ill-formed UTF-8 subsequence and leaves it to the
/// caller, rather than counting it as a column -- possibly one past maxColumnCount -- and consuming it
/// outside the returned text. Downstream terminal parsers key off this to know that every column
/// counted belongs to the returned text, and that the only bytes consumed but not returned are those of
/// a character cut off by the end of the input.
#define LIBUNICODE_SCAN_TEXT_STOPS_AT_ILL_FORMED_UTF8 1

scan_result scan_text(scan_state& state,
                      std::string_view text,
                      size_t maxColumnCount,
                      grapheme_cluster_receiver& receiver) noexcept;

} // namespace unicode
