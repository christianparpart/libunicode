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
#include <libunicode/convert.h>
#include <libunicode/grapheme_segmenter.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace unicode;
using namespace std::string_literals;
using namespace std;

TEST_CASE("latin_common", "[grapheme_segmenter]")
{
    // Each ASCII character is its own grapheme cluster
    auto const text = u32string_view { U"ab!." };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == U"a");
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == U"b");
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == U"!");
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == U".");
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("combining character sequences", "[grapheme_segmenter]")
{
    // g + combining diaeresis = one cluster
    auto const text = u32string_view { U"\u0067\u0308" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == text);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("emoji.speaking-eye", "[grapheme_segmenter]")
{
    // 👁️‍🗨️ = Eye + VS16 + ZWJ + Left Speech Bubble + VS16
    auto const text = u32string_view { U"\U0001F441\uFE0F\u200D\U0001F5E8\uFE0F" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == text);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("emoji", "[grapheme_segmenter]")
{
    // 👨‍🦰 = Man + ZWJ + Red Hair
    auto const text1 = u32string_view { U"\U0001F468\u200D\U0001F9B0" };
    auto gs1 = grapheme_segmenter { text1 };
    CHECK(*gs1 == text1);
    CHECK_FALSE(gs1.codepointsAvailable());

    // 👨‍👩‍👧 = Man + ZWJ + Woman + ZWJ + Girl
    auto const text2 = u32string_view { U"\U0001F468\u200D\U0001F469\u200D\U0001F467" };
    auto gs2 = grapheme_segmenter { text2 };
    CHECK(*gs2 == text2);
    CHECK_FALSE(gs2.codepointsAvailable());
}

TEST_CASE("emoji: Man Facepalming: Medium-Light Skin Tone", "[grapheme_segmenter]")
{
    // 🤦🏼‍♂️ = Facepalm + Skin Tone + ZWJ + Male Sign + VS16
    auto const text = u32string_view { U"\U0001F926\U0001F3FC\u200D\u2642\uFE0F" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == text);
    CHECK_FALSE(gs.codepointsAvailable());
}

// ---- GB11: Extended Pictographic + ZWJ sequence tests ----

TEST_CASE("grapheme_segmenter.gb11_extpic_zwj_extpic", "[grapheme_segmenter]")
{
    // ExtPic + ZWJ + ExtPic = one cluster (GB11 applies)
    // 🛑 + ZWJ + 🛑
    auto const text = u32string_view { U"\U0001F6D1\u200D\U0001F6D1" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == text);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb11_extpic_extend_zwj_extpic", "[grapheme_segmenter]")
{
    // ExtPic + Extend + ZWJ + ExtPic = one cluster (GB11 with Extend chain)
    // 🛑 + combining diaeresis + ZWJ + 🛑
    auto const text = u32string_view { U"\U0001F6D1\u0308\u200D\U0001F6D1" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == text);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb11_no_preceding_extpic", "[grapheme_segmenter]")
{
    // ZWJ + ExtPic without preceding ExtPic = two clusters (GB11 does NOT apply)
    // ZWJ + © (copyright sign, which is ExtPic)
    auto const text = u32string_view { U"\u200D\u00A9" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == u32string_view { U"\u200D" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\u00A9" });
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb11_non_extpic_zwj_extpic", "[grapheme_segmenter]")
{
    // 'a' + ZWJ + ExtPic = two clusters: [a ZWJ] [ExtPic]
    // GB9 keeps ZWJ with 'a', but GB11 does not fire (no preceding ExtPic)
    auto const text = u32string_view { U"\u0061\u200D\U0001F6D1" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == u32string_view { U"\u0061\u200D" }); // 'a' + ZWJ
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F6D1" }); // 🛑
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb11_extpic_double_zwj_extpic", "[grapheme_segmenter]")
{
    // GB9 attaches both ZWJs to MAN, but GB11 does not match multiple ZWJs.
    // The trailing GIRL therefore starts a new cluster.
    auto const text = u32string_view { U"\U0001F468\u200D\u200D\U0001F467" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == u32string_view { U"\U0001F468\u200D\u200D" }); // MAN + ZWJ + ZWJ
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F467" }); // GIRL
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb11_extend_and_zwj", "[grapheme_segmenter]")
{
    // An Extend between ZWJs breaks the GB11 ExtPic chain.
    auto const text1 = u32string_view { U"\U0001F468\u200D\uFE0F\u200D\U0001F467" };
    auto gs1 = grapheme_segmenter { text1 };
    CHECK(*gs1 == u32string_view { U"\U0001F468\u200D\uFE0F\u200D" });
    CHECK(gs1.codepointsAvailable());
    ++gs1;
    CHECK(*gs1 == u32string_view { U"\U0001F467" });
    CHECK_FALSE(gs1.codepointsAvailable());

    // A skin-tone Extend before a single ZWJ still permits GB11.
    auto const text2 = u32string_view { U"\U0001F468\U0001F3FB\u200D\U0001F467" };
    auto gs2 = grapheme_segmenter { text2 };
    CHECK(*gs2 == text2);
    CHECK_FALSE(gs2.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb11_ascii_resets_extpic_chain", "[grapheme_segmenter]")
{
    // ExtPic + ASCII + ZWJ + ExtPic = three clusters: [ExtPic] [a ZWJ] [ExtPic]
    // ASCII breaks the ExtPic chain, so the later ZWJ + ExtPic must not trigger GB11
    auto const text = u32string_view { U"\U0001F6D1\u0061\u200D\U0001F6D1" };
    auto gs = grapheme_segmenter { text };
    CHECK(*gs == u32string_view { U"\U0001F6D1" }); // 🛑
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\u0061\u200D" }); // 'a' + ZWJ
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F6D1" }); // 🛑
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.iterator_1", "[grapheme_segmenter]")
{
    auto const codepoints = u32string_view { U"\U0001F926\U0001F3FC\u200D\u2642\uFE0F" };
    auto gs = grapheme_segmenter { codepoints };

    // initially first token already process
    CHECK(*gs == codepoints);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.iterator_2", "[grapheme_segmenter]")
{
    auto const grapheme_cluster2 = u32string_view { U"\U0001F926\U0001F3FC\u200D\u2642\uFE0F" };
    auto const codepoints = u32string_view { U"X\U0001F926\U0001F3FC\u200D\u2642\uFE0F5" };
    auto gs = grapheme_segmenter { codepoints };

    // first grapheme cluster
    CHECK(*gs == U"X");
    CHECK(gs.codepointsAvailable());

    // second grapheme cluster
    ++gs;
    CHECK(*gs == grapheme_cluster2);
    CHECK(gs.codepointsAvailable());

    // 3rd grapheme cluster
    ++gs;
    CHECK(*gs == U"5");
    CHECK_FALSE(gs.codepointsAvailable());

    // incrementing beyond end of stream
    ++gs;
    CHECK(*gs == U"");
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.iterator_3: regional flags", "[grapheme_segmenter]")
{
    auto const ri_DE = u32string { U"\U0001F1E9\U0001F1E9" };
    auto const ri_JP = u32string { U"\U0001F1EF\U0001F1F5" };
    auto const codepoints = ri_DE + ri_DE + ri_JP;
    auto gs = grapheme_segmenter { codepoints };

    // first grapheme cluster
    REQUIRE(*gs == ri_DE);
    REQUIRE(gs.codepointsAvailable());

    // second grapheme cluster
    ++gs;
    REQUIRE(*gs == ri_DE);
    REQUIRE(gs.codepointsAvailable());

    // 3rd grapheme cluster
    ++gs;
    REQUIRE(*gs == ri_JP);
    REQUIRE_FALSE(gs.codepointsAvailable());

    // incrementing beyond end of stream
    ++gs;
    REQUIRE(*gs == U"");
    REQUIRE_FALSE(gs.codepointsAvailable());
}

namespace
{
/// Streams @p codepoints the way a terminal does: one init() for the first codepoint, then
/// breakable() for each following one. Returns the indices i for which a break precedes codepoints[i].
std::vector<size_t> streamingBreaks(std::u32string_view codepoints)
{
    auto state = grapheme_segmenter_state {};
    auto breaks = std::vector<size_t> {};
    grapheme_process_init(codepoints.front(), state);
    for (auto i = size_t { 1 }; i < codepoints.size(); ++i)
        if (grapheme_process_breakable(codepoints[i], state))
            breaks.push_back(i);
    return breaks;
}
} // namespace

TEST_CASE("grapheme_segmenter.streaming_regional_flags_after_non_ri", "[grapheme_segmenter]")
{
    // GB13: [^RI] (RI RI)* RI x RI -- the RI run restarts after every break.
    auto const input = u32string { U"a🇩🇪🇺🇸" };
    CHECK(streamingBreaks(input) == std::vector<size_t> { 1, 3 });
}

TEST_CASE("grapheme_segmenter.streaming_three_regional_indicators", "[grapheme_segmenter]")
{
    auto const input = u32string { U"🇩🇪🇺" };
    CHECK(streamingBreaks(input) == std::vector<size_t> { 2 });
}

TEST_CASE("grapheme_segmenter.streaming_four_regional_indicators_after_ascii", "[grapheme_segmenter]")
{
    auto const input = u32string { U"ab🇩🇪🇺🇸" };
    CHECK(streamingBreaks(input) == std::vector<size_t> { 1, 2, 4 });
}

TEST_CASE("grapheme_segmenter.prepend_then_regional_pair", "[grapheme_segmenter]")
{
    // GB9b joins Prepend x RI; the RI run then still has odd length at the second RI (GB12/GB13).
    auto const input = u32string { U"؀🇦🇧" };
    auto gs = grapheme_segmenter { input };
    CHECK(*gs == input);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.iterator_3: regional flags invalid 1", "[grapheme_segmenter]")
{
    auto const ri_DE = u32string { U"\U0001F1E9\U0001F1E9" };
    auto const ri_J = u32string { U"\U0001F1EF" };
    auto const codepoints = ri_DE + ri_DE + ri_J + U"P";
    auto gs = grapheme_segmenter { codepoints };

    // first grapheme cluster
    REQUIRE(*gs == ri_DE);
    REQUIRE(gs.codepointsAvailable());

    // second grapheme cluster
    ++gs;
    REQUIRE(*gs == ri_DE);
    REQUIRE(gs.codepointsAvailable());

    // 3rd grapheme cluster
    ++gs;
    REQUIRE(*gs == ri_J);
    REQUIRE(gs.codepointsAvailable());

    // 4th grapheme cluster
    ++gs;
    REQUIRE(*gs == U"P");
    REQUIRE_FALSE(gs.codepointsAvailable());

    // incrementing beyond end of stream
    ++gs;
    REQUIRE(*gs == U"");
    REQUIRE_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.iterator_3: regional flags invalid 2", "[grapheme_segmenter]")
{
    auto const ri_DE = u32string { U"\U0001F1E9\U0001F1E9" };
    auto const ri_J = u32string { U"\U0001F1EF" };
    auto const codepoints = ri_DE + ri_DE + U"Q" + ri_J + U"P";
    auto gs = grapheme_segmenter { codepoints };

    // 1
    REQUIRE(*gs == ri_DE);
    REQUIRE(gs.codepointsAvailable());

    // 2
    ++gs;
    REQUIRE(*gs == ri_DE);
    REQUIRE(gs.codepointsAvailable());

    // 3
    ++gs;
    REQUIRE(*gs == U"Q");
    REQUIRE(gs.codepointsAvailable());

    // 4
    ++gs;
    REQUIRE(*gs == ri_J);
    REQUIRE(gs.codepointsAvailable());

    // 5
    ++gs;
    REQUIRE(*gs == U"P");
    REQUIRE_FALSE(gs.codepointsAvailable());

    // incrementing beyond end of stream
    ++gs;
    REQUIRE(*gs == U"");
    REQUIRE_FALSE(gs.codepointsAvailable());
}

// ---- GB9c: Indic conjunct break rule tests ----

TEST_CASE("grapheme_segmenter.gb9c_basic_devanagari", "[grapheme_segmenter]")
{
    // KA + VIRAMA + TA should be one cluster
    auto const conjunct = u32string_view { U"\u0915\u094D\u0924" };
    auto gs = grapheme_segmenter { conjunct };
    CHECK(*gs == conjunct);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_chained_devanagari", "[grapheme_segmenter]")
{
    // KA + VIRAMA + TA + VIRAMA + YA should be one cluster
    auto const conjunct = u32string_view { U"\u0915\u094D\u0924\u094D\u092F" };
    auto gs = grapheme_segmenter { conjunct };
    CHECK(*gs == conjunct);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_with_extend", "[grapheme_segmenter]")
{
    // KA + NUKTA + ZWJ + VIRAMA + TA should be one cluster
    auto const conjunct = u32string_view { U"\u0915\u093C\u200D\u094D\u0924" };
    auto gs = grapheme_segmenter { conjunct };
    CHECK(*gs == conjunct);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_double_virama", "[grapheme_segmenter]")
{
    // KA + VIRAMA + VIRAMA + TA should be one cluster
    auto const conjunct = u32string_view { U"\u0915\u094D\u094D\u0924" };
    auto gs = grapheme_segmenter { conjunct };
    CHECK(*gs == conjunct);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_virama_zwj", "[grapheme_segmenter]")
{
    // KA + VIRAMA + ZWJ + TA should be one cluster
    auto const conjunct = u32string_view { U"\u0915\u094D\u200D\u0924" };
    auto gs = grapheme_segmenter { conjunct };
    CHECK(*gs == conjunct);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_negative_no_consonant_start", "[grapheme_segmenter]")
{
    // 'a' + VIRAMA + TA should be two clusters: [a, VIRAMA] and [TA]
    // ('a' is not InCB=Consonant, so GB9c does not apply)
    auto const seq = u32string_view { U"\u0061\u094D\u0924" };
    auto gs = grapheme_segmenter { seq };
    CHECK(*gs == u32string_view { U"\u0061\u094D" }); // 'a' + VIRAMA (GB9 keeps VIRAMA with 'a')
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\u0924" }); // TA alone
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_negative_consonant_without_linker", "[grapheme_segmenter]")
{
    // KA + TA should be two clusters (no virama/linker between them)
    auto const seq = u32string_view { U"\u0915\u0924" };
    auto gs = grapheme_segmenter { seq };
    CHECK(*gs == u32string_view { U"\u0915" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\u0924" });
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_negative_virama_latin", "[grapheme_segmenter]")
{
    // KA + VIRAMA + 'a': should be two clusters [KA, VIRAMA] and [a]
    // ('a' is not InCB=Consonant)
    auto const seq = u32string_view { U"\u0915\u094D\u0061" };
    auto gs = grapheme_segmenter { seq };
    CHECK(*gs == u32string_view { U"\u0915\u094D" }); // KA + VIRAMA
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\u0061" }); // 'a'
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_myanmar", "[grapheme_segmenter]")
{
    // MYANMAR LETTER MA + MYANMAR SIGN VIRAMA + MYANMAR LETTER BHA
    auto const conjunct = u32string_view { U"\u1019\u1039\u1018" };
    auto gs = grapheme_segmenter { conjunct };
    CHECK(*gs == conjunct);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_khmer", "[grapheme_segmenter]")
{
    // KHMER LETTER SA + KHMER SIGN COENG + KHMER LETTER TA + KHMER SIGN COENG + KHMER LETTER RO
    auto const conjunct = u32string_view { U"\u179F\u17D2\u178F\u17D2\u179A" };
    auto gs = grapheme_segmenter { conjunct };
    CHECK(*gs == conjunct);
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.gb9c_gujarati_with_shadda", "[grapheme_segmenter]")
{
    // GUJARATI LETTER SA + GUJARATI SIGN SHADDA + GUJARATI SIGN VIRAMA + GUJARATI LETTER SA + GUJARATI SIGN SHADDA
    auto const conjunct = u32string_view { U"\u0AB8\u0AFB\u0ACD\u0AB8\u0AFB" };
    auto gs = grapheme_segmenter { conjunct };
    CHECK(*gs == conjunct);
    CHECK_FALSE(gs.codepointsAvailable());
}

// ---- Ported from apple/swift test/stdlib/Character.swift: "Unicode 9 grapheme breaking" ----

TEST_CASE("grapheme_segmenter.swift_flags_cluster_count", "[grapheme_segmenter]")
{
    // US flag + CA flag + DK flag + rainbow pride flag (WHITE FLAG + VS16 + ZWJ + RAINBOW)
    // ported from apple/swift test/stdlib/Character.swift "Unicode 9 grapheme breaking"
    // (flags.count == 4).
    auto const text =
        u32string_view { U"\U0001F1FA\U0001F1F8\U0001F1E8\U0001F1E6\U0001F1E9\U0001F1F0\U0001F3F3\uFE0F\u200D\U0001F308" };

    auto gs = grapheme_segmenter { text };
    CHECK(*gs == u32string_view { U"\U0001F1FA\U0001F1F8" }); // US
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F1E8\U0001F1E6" }); // CA
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F1E9\U0001F1F0" }); // DK
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F3F3\uFE0F\u200D\U0001F308" }); // rainbow pride flag
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.swift_family_cluster_count", "[grapheme_segmenter]")
{
    // Six family emoji ZWJ sequences back-to-back, ported from apple/swift
    // test/stdlib/Character.swift "Unicode 9 grapheme breaking" (family.count == 6):
    // FAMILY, MAN+ZWJ+GIRL+ZWJ+GIRL, WOMAN+ZWJ+WOMAN+ZWJ+GIRL+ZWJ+BOY,
    // MAN+ZWJ+MAN+ZWJ+BOY+ZWJ+BOY, MAN+ZWJ+GIRL, WOMAN+ZWJ+BOY+ZWJ+BOY.
    auto const text = u32string_view { U"\U0001F46A"
                                       U"\U0001F468\u200D\U0001F467\u200D\U0001F467"
                                       U"\U0001F469\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466"
                                       U"\U0001F468\u200D\U0001F468\u200D\U0001F466\u200D\U0001F466"
                                       U"\U0001F468\u200D\U0001F467"
                                       U"\U0001F469\u200D\U0001F466\u200D\U0001F466" };

    auto gs = grapheme_segmenter { text };
    CHECK(*gs == u32string_view { U"\U0001F46A" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F468\u200D\U0001F467\u200D\U0001F467" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F469\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F468\u200D\U0001F468\u200D\U0001F466\u200D\U0001F466" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F468\u200D\U0001F467" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F469\u200D\U0001F466\u200D\U0001F466" });
    CHECK_FALSE(gs.codepointsAvailable());
}

TEST_CASE("grapheme_segmenter.swift_skin_tone_cluster_count", "[grapheme_segmenter]")
{
    // WAVING HAND with each of the five Fitzpatrick skin tone modifiers, ported
    // from apple/swift test/stdlib/Character.swift "Unicode 9 grapheme breaking"
    // (skinTone.count == 6).
    auto const text = u32string_view { U"\U0001F44B"
                                       U"\U0001F44B\U0001F3FB"
                                       U"\U0001F44B\U0001F3FC"
                                       U"\U0001F44B\U0001F3FD"
                                       U"\U0001F44B\U0001F3FE"
                                       U"\U0001F44B\U0001F3FF" };

    auto gs = grapheme_segmenter { text };
    CHECK(*gs == u32string_view { U"\U0001F44B" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F44B\U0001F3FB" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F44B\U0001F3FC" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F44B\U0001F3FD" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F44B\U0001F3FE" });
    CHECK(gs.codepointsAvailable());
    ++gs;
    CHECK(*gs == u32string_view { U"\U0001F44B\U0001F3FF" });
    CHECK_FALSE(gs.codepointsAvailable());
}

// =============================================================================
// Official Unicode Conformance Tests
// =============================================================================

namespace
{

/// Collects break positions (offsets from start) for a codepoint sequence.
/// Returns a sorted list of offsets where grapheme cluster boundaries occur.
/// Includes position 0 (sot) and position N (eot).
std::vector<size_t> collect_break_positions(std::u32string_view text)
{
    std::vector<size_t> breaks;
    breaks.push_back(0); // sot

    if (text.empty())
        return breaks;

    auto gs = grapheme_segmenter(text);
    size_t offset = 0;
    while (true)
    {
        auto const cluster = *gs;
        offset += cluster.size();
        breaks.push_back(offset);
        if (!gs.codepointsAvailable())
            break;
        ++gs;
    }
    return breaks;
}

/// Parse a hex string like "000D" to a char32_t.
char32_t parse_hex(std::string_view sv)
{
    unsigned long val = 0;
    for (auto c: sv)
    {
        val <<= 4;
        if (c >= '0' && c <= '9')
            val |= static_cast<unsigned long>(c - '0');
        else if (c >= 'A' && c <= 'F')
            val |= static_cast<unsigned long>(c - 'A' + 10);
        else if (c >= 'a' && c <= 'f')
            val |= static_cast<unsigned long>(c - 'a' + 10);
    }
    return static_cast<char32_t>(val);
}

struct GraphemeBreakTestCase
{
    std::u32string codepoints;
    std::vector<size_t> expected_breaks; // offsets where breaks occur
    std::string comment;
    int line_number = 0;
};

/// Parse GraphemeBreakTest.txt into test cases.
/// Format: ÷ 000D × 000A ÷ (# comment)
/// ÷ = break, × = no break
std::vector<GraphemeBreakTestCase> parse_grapheme_break_test_file(std::string const& path)
{
    std::vector<GraphemeBreakTestCase> cases;
    std::ifstream file(path);
    if (!file.is_open())
        return cases;

    std::string line;
    auto lineNo = 0;
    while (std::getline(file, line))
    {
        ++lineNo;
        if (line.empty() || line[0] == '#')
            continue;

        // The line starts with ÷ or × characters (UTF-8 encoded)
        // ÷ = U+00F7 (UTF-8: C3 B7), × = U+00D7 (UTF-8: C3 97)

        GraphemeBreakTestCase tc;
        tc.line_number = lineNo;

        // Extract comment part
        auto commentPos = line.find('#');
        auto dataStr = (commentPos != std::string::npos) ? line.substr(0, commentPos) : line;
        if (commentPos != std::string::npos)
            tc.comment = line.substr(commentPos);

        // Parse the data portion: sequence of (÷|×) XXXX pairs
        size_t cpIndex = 0;
        auto i = size_t { 0 };
        while (i < dataStr.size())
        {
            // Skip whitespace
            while (i < dataStr.size() && (dataStr[i] == ' ' || dataStr[i] == '\t'))
                ++i;
            if (i >= dataStr.size())
                break;

            // Check for ÷ (C3 B7) or × (C3 97)
            if (i + 1 < dataStr.size() && static_cast<unsigned char>(dataStr[i]) == 0xC3)
            {
                auto const secondByte = static_cast<unsigned char>(dataStr[i + 1]);
                if (secondByte == 0xB7) // ÷ = break
                {
                    tc.expected_breaks.push_back(cpIndex);
                    i += 2;
                    continue;
                }
                if (secondByte == 0x97) // × = no break
                {
                    i += 2;
                    continue;
                }
            }

            // Must be a hex codepoint
            auto start = i;
            while (i < dataStr.size() && dataStr[i] != ' ' && dataStr[i] != '\t'
                   && static_cast<unsigned char>(dataStr[i]) != 0xC3)
                ++i;

            if (i > start)
            {
                auto hexStr = dataStr.substr(start, i - start);
                tc.codepoints.push_back(parse_hex(hexStr));
                ++cpIndex;
            }
        }

        if (!tc.codepoints.empty())
            cases.push_back(std::move(tc));
    }
    return cases;
}

} // namespace

TEST_CASE("grapheme_segmenter.unicode_conformance", "[grapheme_segmenter]")
{
    auto const path = std::string(LIBUNICODE_UCD_DIR) + "/auxiliary/GraphemeBreakTest.txt";
    auto const testCases = parse_grapheme_break_test_file(path);

    if (testCases.empty())
    {
        WARN("Skipping conformance tests: GraphemeBreakTest.txt not found at " << path);
        return;
    }
    INFO("Loaded " << testCases.size() << " test cases from GraphemeBreakTest.txt");

    for (auto const& tc: testCases)
    {
        auto const actual = collect_break_positions(tc.codepoints);
        if (actual != tc.expected_breaks)
        {
            std::ostringstream msg;
            msg << "FAIL line " << tc.line_number << ": ";

            msg << "codepoints: ";
            for (auto cp: tc.codepoints)
                msg << std::hex << "U+" << static_cast<uint32_t>(cp) << " ";

            msg << " expected breaks: ";
            for (auto b: tc.expected_breaks)
                msg << std::dec << b << " ";

            msg << " actual breaks: ";
            for (auto b: actual)
                msg << std::dec << b << " ";

            msg << tc.comment;
            FAIL(msg.str());
        }
    }
}
