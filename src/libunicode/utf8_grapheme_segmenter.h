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
#include <libunicode/utf8.h>

#include <ostream>
#include <string_view>

namespace unicode
{

struct utf8_grapheme_segmenter
{
    class iterator;

    explicit utf8_grapheme_segmenter(std::string_view text) noexcept;
    utf8_grapheme_segmenter(utf8_grapheme_segmenter const&) noexcept = default;
    utf8_grapheme_segmenter(utf8_grapheme_segmenter&&) noexcept = default;
    utf8_grapheme_segmenter& operator=(utf8_grapheme_segmenter const&) noexcept = default;
    utf8_grapheme_segmenter& operator=(utf8_grapheme_segmenter&&) noexcept = default;

    iterator begin() const noexcept;
    iterator end() const noexcept;

  private:
    std::string_view _text;
};

class utf8_grapheme_segmenter::iterator
{
  public:
    using value_type = std::u32string;

    iterator(char const* data, char const* end) noexcept;
    iterator(iterator const&) = default;
    iterator(iterator&&) noexcept = default;
    iterator& operator=(iterator const&) = default;
    iterator& operator=(iterator&&) noexcept = default;

    value_type const& value() const noexcept;
    value_type const& operator*() const noexcept;

    iterator& operator++() noexcept;
    iterator operator++(int) noexcept;

    bool operator==(iterator const& other) const noexcept;
    bool operator!=(iterator const& other) const noexcept;

    // private:
    char32_t consumeCodepoint() noexcept;
    void consumeGraphemeCluster() noexcept;

    char const* _start;
    char const* _clusterStart;
    char const* _nextCodepointStart;
    char const* _nextUtf8;
    char const* _end;
    utf8_decoder_state _utf8_decoder_state {};
    ConvertResult _result = Incomplete {};
    char32_t _nextCodepoint {};
    value_type _cluster {};
    grapheme_segmenter_state _segmenter_state {};
};

// {{{ utf8_grapheme_segmenter implementation
inline utf8_grapheme_segmenter::utf8_grapheme_segmenter(std::string_view text) noexcept: _text { text }
{
}

inline utf8_grapheme_segmenter::iterator utf8_grapheme_segmenter::begin() const noexcept
{
    return iterator { _text.data(), _text.data() + _text.size() };
}

inline utf8_grapheme_segmenter::iterator utf8_grapheme_segmenter::end() const noexcept
{
    return iterator { _text.data() + _text.size(), _text.data() + _text.size() };
}
// }}}

// {{{ iterator implementation
inline utf8_grapheme_segmenter::iterator::iterator(char const* data, char const* end) noexcept:
    _start { data }, _clusterStart { data }, _nextCodepointStart { data }, _nextUtf8 { data }, _end { end }
{
    if (data != end)
    {
        consumeCodepoint();
        consumeGraphemeCluster();
    }
}

inline utf8_grapheme_segmenter::iterator::value_type const& utf8_grapheme_segmenter::iterator::value() const noexcept
{
    return _cluster;
}

inline utf8_grapheme_segmenter::iterator::value_type const& utf8_grapheme_segmenter::iterator::operator*() const noexcept
{
    return _cluster;
}

inline char32_t utf8_grapheme_segmenter::iterator::consumeCodepoint() noexcept
{
    auto constexpr ReplacementChar = char32_t { 0xFFFD };
    _nextCodepointStart = _nextUtf8;
    while (_nextUtf8 != _end)
    {
        auto const expectedBefore = _utf8_decoder_state.expectedLength;
        _result = from_utf8(_utf8_decoder_state, uint8_t(*_nextUtf8++));
        if (std::holds_alternative<Success>(_result))
        {
            auto const result = _nextCodepoint;
            _nextCodepoint = std::get<Success>(_result).value;
            return result;
        }
        if (std::holds_alternative<Invalid>(_result))
        {
            // A byte that broke a character off and cannot start one is decoded again, as itself.
            if (from_utf8_left_byte_unconsumed(expectedBefore, _utf8_decoder_state))
                --_nextUtf8;
            auto const result = _nextCodepoint;
            _nextCodepoint = ReplacementChar;
            return result;
        }
    }
    auto const result = _nextCodepoint;
    _nextCodepoint = 0;
    return result;
}

inline void utf8_grapheme_segmenter::iterator::consumeGraphemeCluster() noexcept
{
    _clusterStart = _nextCodepointStart;
    _cluster.clear();

    if (_nextCodepointStart == _end)
        return;

    // Consume first codepoint and initialize segmenter state
    _cluster.push_back(consumeCodepoint());
    grapheme_process_init(_cluster.back(), _segmenter_state);

    // Continue consuming while the next codepoint doesn't break the cluster
    while (_nextCodepointStart != _end && !grapheme_process_breakable(_nextCodepoint, _segmenter_state))
        _cluster.push_back(consumeCodepoint());
}

inline utf8_grapheme_segmenter::iterator& utf8_grapheme_segmenter::iterator::operator++() noexcept
{
    consumeGraphemeCluster();
    return *this;
}

inline utf8_grapheme_segmenter::iterator utf8_grapheme_segmenter::iterator::operator++(int) noexcept
{
    auto tmp(*this);
    ++*this;
    return tmp;
}

inline bool utf8_grapheme_segmenter::iterator::operator==(iterator const& other) const noexcept
{
    return _clusterStart == other._clusterStart;
}

inline bool utf8_grapheme_segmenter::iterator::operator!=(iterator const& other) const noexcept
{
    return !(*this == other);
}
// }}}

} // namespace unicode

namespace std
{
inline ostream& operator<<(ostream& os, unicode::utf8_grapheme_segmenter::iterator const& i)
{
    os << '"' << unicode::convert_to<char>(std::u32string_view(i.value())) << '"';
    return os;
}
} // namespace std
