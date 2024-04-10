/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "ftos_converter.cuh"
#include "string_to_float_cudf.cuh"

#include <cudf/strings/detail/convert/string_to_float.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/types.hpp>

#include <thrust/pair.h>
#include <thrust/tuple.h>

namespace spark_rapids_jni {

/**
 * write style when writing out JSON string
 */
enum class write_style {
  // e.g.: '\\r' is a string with 2 chars '\' 'r', writes 1 char '\r'
  unescaped,

  // * e.g.: '"' is a string with 1 char '"', writes out 4 chars '"' '\' '\"'
  // '"'
  escaped
};

// allow single quotes to represent strings in JSON
// e.g.: {'k': 'v'} is valid when it's true
constexpr bool allow_single_quotes = true;

// Whether allow unescaped control characters in JSON Strings.
// Unescaped control characters are ASCII characters with value less than 32,
// including tab and line feed characters. ASCII values range is [0, 32)
// e.g.: ["\n"] is valid, here \n is one char
// If true, JSON is not conventional format.
// e.g., how to represent carriage return and newline characters:
//   if true, allow "\n\r" two control characters without escape directly
//   if false, "\n\r" are not allowed, should use escape characters: "\\n\\r"
constexpr bool allow_unescaped_control_chars = true;

/**
 * @brief Maximum JSON nesting depth
 * JSON with a greater depth is invalid
 * If set this to be a greater value, should update `context_stack`
 */
constexpr int max_json_nesting_depth = 64;

// Define the maximum JSON String length, counts utf8 bytes.
// By default, maximum JSON String length is negative one, means no
// limitation. e.g.: The length of String "\\n" is 1, JSON parser does not
// count escape characters.
constexpr int max_string_utf8_bytes = 20000000;

//
/**
 * Define the maximum JSON number length. Negative or zero means no
 * limitation.
 *
 * By default, maximum JSON number length is negative one, means no
 * limitation.
 *
 * e.g.: The length of number -123.45e-67 is 7. if maximum JSON number length
 * is 6, then this number is a invalid number.
 */
constexpr int max_num_len = 1000;

/**
 * whether allow tailing useless sub-string in JSON.
 *
 * If true, e.g., the following invalid JSON is allowed, because prefix {'k' :
 * 'v'} is valid.
 *   {'k' : 'v'}_extra_tail_sub_string
 */
constexpr bool allow_tailing_sub_string = true;

/**
 * JSON token enum
 */
enum class json_token {
  // start token
  INIT = 0,

  // successfully parsed the whole JSON string
  SUCCESS,

  // get error when parsing JSON string
  ERROR,

  // '{'
  START_OBJECT,

  // '}'
  END_OBJECT,

  // '['
  START_ARRAY,

  // ']'
  END_ARRAY,

  // e.g.: key1 in {"key1" : "value1"}
  FIELD_NAME,

  // e.g.: value1 in {"key1" : "value1"}
  VALUE_STRING,

  // e.g.: 123 in {"key1" : 123}
  VALUE_NUMBER_INT,

  // e.g.: 1.25 in {"key1" : 1.25}
  VALUE_NUMBER_FLOAT,

  // e.g.: true in {"key1" : true}
  VALUE_TRUE,

  // e.g.: false in {"key1" : false}
  VALUE_FALSE,

  // e.g.: null in {"key1" : null}
  VALUE_NULL

};

/**
 * JSON parser, provides token by token parsing.
 * Follow Jackson JSON format by default.
 *
 *
 * For JSON format:
 * Refer to https://www.json.org/json-en.html.
 *
 * Note: when setting `allow_single_quotes` or `allow_unescaped_control_chars`,
 * then JSON format is not conventional.
 *
 * White space can only be 4 chars: ' ', '\n', '\r', '\t',
 * Jackson does not allow other control chars as white spaces.
 *
 * Valid number examples:
 *   0, 102, -0, -102, 0.3, -0.3
 *   1e-5, 1E+5, 1e0, 1E0, 1.3e5
 *   1e01 : allow leading zeor after 'e'
 *
 * Invalid number examples:
 *   00, -00   Leading zeroes not allowed
 *   infinity, +infinity, -infinity
 *   1e, 1e+, 1e-, -1., 1.
 *
 * When `allow_single_quotes` is true:
 *   Valid string examples:
 *     "\'" , "\"" ,  '\'' , '\"' , '"' , "'"
 *
 *  When `allow_single_quotes` is false:
 *   Invalid string examples:
 *     "\'"
 *
 *  When `allow_unescaped_control_chars` is true:
 *    Valid string: "asscii_control_chars"
 *      here `asscii_control_chars` represents control chars which in Ascii code
 * range: [0, 32)
 *
 *  When `allow_unescaped_control_chars` is false:
 *    Invalid string: "asscii_control_chars"
 *      here `asscii_control_chars` represents control chars which in Ascii code
 * range: [0, 32)
 *
 */
class json_parser {
 public:
  __device__ inline json_parser(char const* const _json_start_pos, cudf::size_type const _json_len)
    : json_start_pos(_json_start_pos),
      json_end_pos(_json_start_pos + _json_len),
      curr_pos(_json_start_pos)
  {
  }

 private:
  /**
   * @brief get the bit value for specified bit from a int64 number
   */
  __device__ inline bool get_bit_value(int64_t number, int bitIndex)
  {
    // Shift the number right by the bitIndex to bring the desired bit to the rightmost position
    long shifted = number >> bitIndex;

    // Extract the rightmost bit by performing a bitwise AND with 1
    bool bit_value = shifted & 1;

    return bit_value;
  }

  /**
   * @brief set the bit value for specified bit to a int64 number
   */
  __device__ inline void set_bit_value(int64_t& number, int bit_index, bool bit_value)
  {
    // Create a mask with a 1 at the desired bit index
    long mask = 1L << bit_index;

    if (bit_value) {
      // Set the bit to 1 by performing a bitwise OR with the mask
      number |= mask;
    } else {
      // Set the bit to 0 by performing a bitwise AND with the complement of the mask
      number &= ~mask;
    }
  }

  /**
   * is current position EOF
   */
  __device__ inline bool eof(char const* pos) { return pos >= json_end_pos; }

  /**
   * is hex digits: 0-9, A-F, a-f
   */
  __device__ inline bool is_hex_digit(char c)
  {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
  }

  /**
   * is 0 to 9 digit
   */
  __device__ inline bool is_digit(char c) { return (c >= '0' && c <= '9'); }

  /**
   * is white spaces: ' ', '\t', '\n' '\r'
   */
  __device__ inline bool is_whitespace(char c)
  {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  /**
   * skips 4 characters: ' ', '\t', '\n' '\r'
   */
  __device__ inline void skip_whitespaces(char const*& pos)
  {
    while (!eof(pos) && is_whitespace(*pos)) {
      pos++;
    }
  }

  /**
   * check current char, if it's expected, then plus the position
   */
  __device__ inline bool try_skip(char const*& pos, char expected)
  {
    if (!eof(pos) && *pos == expected) {
      pos++;
      return true;
    }
    return false;
  }

  /**
   * try to push current context into stack
   * if nested depth exceeds limitation, return false
   */
  __device__ inline bool try_push_context(json_token token)
  {
    if (stack_size < max_json_nesting_depth) {
      push_context(token);
      return true;
    } else {
      return false;
    }
  }

  /**
   * record the nested state into stack: JSON object or JSON array
   */
  __device__ inline void push_context(json_token token)
  {
    bool v = json_token::START_OBJECT == token ? true : false;
    set_bit_value(context_stack, stack_size, v);
    stack_size++;
  }

  /**
   * whether the top of nested context stack is JSON object context
   * true is object, false is array
   * only has two contexts: object or array
   */
  __device__ inline bool is_object_context()
  {
    return get_bit_value(context_stack, stack_size - 1);
  }

  /**
   * pop top context from stack
   */
  __device__ inline void pop_curr_context() { stack_size--; }

  /**
   * is context stack is empty
   */
  __device__ inline bool is_context_stack_empty() { return stack_size == 0; }

  /**
   * parse the first value token from current position
   * e.g., after finished this function:
   *   current token is START_OBJECT if current value is object
   *   current token is START_ARRAY if current value is array
   *   current token is string/num/true/false/null if current value is terminal
   *   current token is ERROR if parse failed
   */
  __device__ inline void parse_first_token_in_value()
  {
    // already checked eof
    char c = *curr_pos;
    switch (c) {
      case '{':
        if (!try_push_context(json_token::START_OBJECT)) {
          curr_token = json_token::ERROR;
          return;
        }
        curr_pos++;
        curr_token = json_token::START_OBJECT;
        break;

      case '[':
        if (!try_push_context(json_token::START_ARRAY)) {
          curr_token = json_token::ERROR;
          return;
        }
        curr_pos++;
        curr_token = json_token::START_ARRAY;
        break;

      case '"': parse_double_quoted_string(); break;

      case '\'':
        if (allow_single_quotes) {
          parse_single_quoted_string();
        } else {
          curr_token = json_token::ERROR;
        }
        break;

      case 't':
        curr_pos++;
        parse_true();
        break;

      case 'f':
        curr_pos++;
        parse_false();
        break;

      case 'n':
        curr_pos++;
        parse_null();
        break;

      default: parse_number();
    }
  }

  // =========== Parse string begin ===========

  /**
   * parse ' quoted string
   */
  __device__ inline void parse_single_quoted_string()
  {
    auto [success, end_char_pos] =
      try_parse_single_quoted_string(curr_pos, nullptr, nullptr, nullptr, write_style::unescaped);
    if (success) {
      curr_pos   = end_char_pos;
      curr_token = json_token::VALUE_STRING;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse " quoted string
   */
  __device__ inline void parse_double_quoted_string()
  {
    auto [success, end_char_pos] =
      try_parse_double_quoted_string(curr_pos, nullptr, nullptr, nullptr, write_style::unescaped);
    if (success) {
      curr_pos   = end_char_pos;
      curr_token = json_token::VALUE_STRING;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /*
   * try parse ' or " quoted string
   *
   * when allow single quote, first try single quote
   * @param str_pos str start position for parsing, should be a position in JSON
   * string
   * @param to_match_str_pos expected match str position, nullptr means do not
   * match
   * @param to_match_str_end expected match str end
   * @param copy_destination copy unescaped str to destination, nullptr means do
   * not copy
   * @return whether passed successfully and the end position of parsed str
   *
   */
  __device__ inline std::pair<bool, char const*> try_parse_string(
    char const* str_pos,
    char const* to_match_str_pos,
    char const* const to_match_str_end,
    char* copy_destination,
    write_style w_style)
  {
    if (!eof(str_pos)) {
      if (allow_single_quotes && *str_pos == '\'') {
        return try_parse_single_quoted_string(
          str_pos, to_match_str_pos, to_match_str_end, copy_destination, w_style);
      } else {
        return try_parse_double_quoted_string(
          str_pos, to_match_str_pos, to_match_str_end, copy_destination, w_style);
      }
    } else {
      return std::make_pair(false, nullptr);
    }
  }

  /**
   * try parse ' quoted string
   *
   * when allow single quote, first try single quote
   * @param str_pos str start position for parsing, should be a position in JSON
   * string
   * @param to_match_str_pos expected match str position, nullptr means do not
   * match
   * @param to_match_str_end expected match str end
   * @param copy_destination copy unescaped str to destination, nullptr means do
   * not copy
   *
   */
  __device__ inline std::pair<bool, char const*> try_parse_single_quoted_string(
    char const* str_pos,
    char const* to_match_str_pos,
    char const* const to_match_str_end,
    char* copy_destination,
    write_style w_style)
  {
    return try_parse_quoted_string(str_pos,
                                   '\'',
                                   to_match_str_pos,  // match str pos, nullptr means do not match
                                   to_match_str_end,  // match str end
                                   copy_destination,  // copy destination while parsing, nullptr
                                                      // means do not copy
                                   w_style);
  }

  /**
   * try parse " quoted string.
   *
   * when allow single quote, first try single quote
   * @param str_pos str start position for parsing, should be a position in JSON
   * string
   * @param to_match_str_pos expected match str position, nullptr means do not
   * match
   * @param to_match_str_end expected match str end
   * @param copy_destination copy unescaped str to destination, nullptr means do
   * not copy
   *
   */
  __device__ inline std::pair<bool, char const*> try_parse_double_quoted_string(
    char const* str_pos,
    char const* to_match_str_pos,
    char const* const to_match_str_end,
    char* copy_destination,
    write_style w_style)
  {
    return try_parse_quoted_string(str_pos,
                                   '\"',
                                   to_match_str_pos,  // match str pos, nullptr means do not match
                                   to_match_str_end,  // match str end
                                   copy_destination,  // copy destination while parsing, nullptr
                                                      // means do not copy
                                   w_style);
  }

  /**
   * transform int value from [0, 15] to hex char
   */
  __device__ inline char to_hex_char(unsigned int v)
  {
    if (v < 10)
      return '0' + v;
    else
      return 'A' + (v - 10);
  }

  /**
   * escape control char ( ASCII code value [0, 32) )
   * e.g.: \0  (ASCII code 0) will be escaped to 6 chars: \u0000
   * e.g.: \10 (ASCII code 0) will be escaped to 2 chars: \n
   * @param char to be escaped, c should in range [0, 31)
   * @param[out] escape output
   */
  __device__ inline int escape_char(unsigned char c, char* output)
  {
    if (nullptr == output) {
      switch (c) {
        case 8:             // \b
        case 9:             // \t
        case 10:            // \n
        case 12:            // \f
        case 13: return 2;  // \r
        default: return 6;  // \u0000
      }
    }
    switch (c) {
      case 8:
        output[0] = '\\';
        output[1] = 'b';
        return 2;
      case 9:
        output[0] = '\\';
        output[1] = 't';
        return 2;
      case 10:
        output[0] = '\\';
        output[1] = 'n';
        return 2;
      case 12:
        output[0] = '\\';
        output[1] = 'f';
        return 2;
      case 13:
        output[0] = '\\';
        output[1] = 'r';
        return 2;
      default:
        output[0] = '\\';
        output[1] = 'u';
        output[2] = '0';
        output[3] = '0';

        // write high digit
        if (c >= 16) {
          output[4] = '1';
        } else {
          output[4] = '0';
        }

        // write low digit
        unsigned int v = c % 16;
        output[5]      = to_hex_char(v);
        return 6;
    }
  }

  /**
   * utility for parsing string, this function does not update the parser
   * internal try parse quoted string using passed `quote_char` `quote_char` can
   * be ' or " For UTF-8 encoding: Single byte char: The most significant bit of
   * the byte is always 0 Two-byte characters: The leading bits of the first
   * byte are 110, and the leading bits of the second byte are 10. Three-byte
   * characters: The leading bits of the first byte are 1110, and the leading
   * bits of the second and third bytes are 10. Four-byte characters: The
   * leading bits of the first byte are 11110, and the leading bits of the
   * second, third, and fourth bytes are 10. Because JSON structural chars([ ] {
   * } , :), string quote char(" ') and Escape char \ are all Ascii(The leading
   * bit is 0), so it's safe that do not convert byte array to UTF-8 char.
   *
   * When quote is " and allow_unescaped_control_chars is false, grammar is:
   *
   *   STRING
   *     : '"' (ESC | SAFECODEPOINT)* '"'
   *     ;
   *
   *   fragment ESC
   *     : '\\' (["\\/bfnrt] | UNICODE)
   *     ;
   *
   *   fragment UNICODE
   *     : 'u' HEX HEX HEX HEX
   *     ;
   *
   *   fragment HEX
   *     : [0-9a-fA-F]
   *     ;
   *
   *   fragment SAFECODEPOINT
   *       // 1 not " or ' depending to allow_single_quotes
   *       // 2 not \
   *       // 3 non control character: Ascii value not in [0, 32)
   *     : ~ ["\\\u0000-\u001F]
   *     ;
   *
   * When allow_unescaped_control_chars is true:
   *   Allow [0-32) control Ascii chars directly without escape
   * When allow_single_quotes is true:
   *   These strings are allowed: '\'' , '\"' , '"' , "\"" , "\'" , "'"
   * @param str_pos str start position for parsing, should be a position in JSON
   * string
   * @param quote_char expected quote char
   * @param to_match_str_pos expected match str position, nullptr means do not
   * match
   * @param to_match_str_end expected match str end
   * @param copy_destination copy unescaped str to destination, nullptr means do
   * not copy
   */
  __device__ inline std::pair<bool, char const*> try_parse_quoted_string(
    char const* str_pos,
    char const quote_char,
    char const* to_match_str_pos,
    char const* const to_match_str_end,
    char* copy_destination,
    write_style w_style)
  {
    // update state
    string_token_utf8_bytes       = 0;
    bytes_diff_for_escape_writing = 0;

    // write the first " if write style is escaped
    if (write_style::escaped == w_style) {
      bytes_diff_for_escape_writing++;
      if (nullptr != copy_destination) { *copy_destination++ = '"'; }
    }

    // skip left quote char
    if (!try_skip(str_pos, quote_char)) { return std::make_pair(false, nullptr); }

    // scan string content
    while (!eof(str_pos)) {
      char c = *str_pos;
      int v  = static_cast<int>(c);
      if (c == quote_char) {
        // path 1: match closing quote char
        str_pos++;

        // check max str len
        if (!check_string_max_utf8_bytes()) { return std::make_pair(false, nullptr); }

        // match check, the last char in match_str is quote_char
        if (nullptr != to_match_str_pos) {
          // match check, the last char in match_str is quote_char
          if (to_match_str_pos != to_match_str_end) { return std::make_pair(false, nullptr); }
        }

        // write the end " if write style is escaped
        if (write_style::escaped == w_style) {
          bytes_diff_for_escape_writing++;
          if (nullptr != copy_destination) { *copy_destination++ = '"'; }
        }

        return std::make_pair(true, str_pos);
      } else if (v >= 0 && v < 32 && allow_unescaped_control_chars) {
        // path 2: unescaped control char

        // copy if enabled, unescape mode, write 1 char
        if (copy_destination != nullptr && write_style::unescaped == w_style) {
          *copy_destination++ = *str_pos;
        }

        // copy if enabled, escape mode, write more chars
        if (write_style::escaped == w_style) {
          int escape_chars = escape_char(*str_pos, copy_destination);
          if (copy_destination != nullptr) copy_destination += escape_chars;
          bytes_diff_for_escape_writing += (escape_chars - 1);
        }

        // check match if enabled
        if (!try_match_char(to_match_str_pos, to_match_str_end, *str_pos)) {
          return std::make_pair(false, nullptr);
        }

        str_pos++;
        string_token_utf8_bytes++;
        continue;
      } else if ('\\' == c) {
        // path 3: escape path
        str_pos++;
        if (!try_skip_escape_part(
              str_pos, to_match_str_pos, to_match_str_end, copy_destination, w_style)) {
          return std::make_pair(false, nullptr);
        }
      } else {
        // path 4: safe code point

        // handle single unescaped " char; happens when string is quoted by char '
        // e.g.:  'A"' string, escape to "A\\"" (5 chars: " A \ " ")
        if ('\"' == c && write_style::escaped == w_style) {
          if (copy_destination != nullptr) { *copy_destination++ = '\\'; }
          bytes_diff_for_escape_writing++;
        }

        if (!try_skip_safe_code_point(str_pos, c)) { return std::make_pair(false, nullptr); }
        if (copy_destination != nullptr) { *copy_destination++ = c; }
        // check match if enabled
        if (!try_match_char(to_match_str_pos, to_match_str_end, c)) {
          return std::make_pair(false, nullptr);
        }
        string_token_utf8_bytes++;
      }
    }

    return std::make_pair(false, nullptr);
  }

  __device__ inline bool try_match_char(char const*& char_pos,
                                        char const* const char_end_pos,
                                        char c)
  {
    if (nullptr != char_pos) {
      if (char_pos < char_end_pos && *char_pos == c) {
        char_pos++;
        return true;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  /**
   * skip the second char in \", \', \\, \/, \b, \f, \n, \r, \t;
   * skip the HEX chars in \u HEX HEX HEX HEX.
   * @return positive escaped ASCII value if success, -1 otherwise
   */
  __device__ inline bool try_skip_escape_part(char const*& str_pos,
                                              char const*& to_match_str_pos,
                                              char const* const to_match_str_end,
                                              char*& copy_dest,
                                              write_style w_style)
  {
    // already skipped the first '\'
    // try skip second part
    if (!eof(str_pos)) {
      char c = *str_pos;
      switch (*str_pos) {
        // path 1: \", \', \\, \/, \b, \f, \n, \r, \t
        case '\"':
          if (nullptr != copy_dest && write_style::unescaped == w_style) { *copy_dest++ = c; }
          if (write_style::escaped == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = '"';
            }
            bytes_diff_for_escape_writing++;
          }
          if (!try_match_char(to_match_str_pos, to_match_str_end, c)) { return false; }
          string_token_utf8_bytes++;
          str_pos++;
          return true;
        case '\'':
          // only allow escape ' when `allow_single_quotes`
          if (allow_single_quotes) {
            // for both unescaped/escaped writes a single char '
            if (nullptr != copy_dest) { *copy_dest++ = c; }
            if (!try_match_char(to_match_str_pos, to_match_str_end, c)) { return false; }

            string_token_utf8_bytes++;
            str_pos++;
            return true;
          } else {
            return false;
          }
        case '\\':
          if (nullptr != copy_dest && write_style::unescaped == w_style) { *copy_dest++ = c; }
          if (write_style::escaped == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = '\\';
            }
            bytes_diff_for_escape_writing++;
          }
          if (!try_match_char(to_match_str_pos, to_match_str_end, c)) { return false; }
          string_token_utf8_bytes++;
          str_pos++;
          return true;
        case '/':
          // for both unescaped/escaped writes a single char /
          if (nullptr != copy_dest) { *copy_dest++ = c; }
          if (!try_match_char(to_match_str_pos, to_match_str_end, c)) { return false; }
          string_token_utf8_bytes++;
          str_pos++;
          return true;
        case 'b':
          if (nullptr != copy_dest && write_style::unescaped == w_style) { *copy_dest++ = '\b'; }
          if (write_style::escaped == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'b';
            }
            bytes_diff_for_escape_writing++;
          }
          if (!try_match_char(to_match_str_pos, to_match_str_end, '\b')) { return false; }
          string_token_utf8_bytes++;
          str_pos++;
          return true;
        case 'f':
          if (nullptr != copy_dest && write_style::unescaped == w_style) { *copy_dest++ = '\f'; }
          if (write_style::escaped == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'f';
            }
            bytes_diff_for_escape_writing++;
          }
          if (!try_match_char(to_match_str_pos, to_match_str_end, '\f')) { return false; }
          string_token_utf8_bytes++;
          str_pos++;
          return true;
        case 'n':
          if (nullptr != copy_dest && write_style::unescaped == w_style) { *copy_dest++ = '\n'; }
          if (write_style::escaped == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'n';
            }
            bytes_diff_for_escape_writing++;
          }
          if (!try_match_char(to_match_str_pos, to_match_str_end, '\n')) { return false; }
          string_token_utf8_bytes++;
          str_pos++;
          return true;
        case 'r':
          if (nullptr != copy_dest && write_style::unescaped == w_style) { *copy_dest++ = '\r'; }
          if (write_style::escaped == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'r';
            }
            bytes_diff_for_escape_writing++;
          }
          if (!try_match_char(to_match_str_pos, to_match_str_end, '\r')) { return false; }
          string_token_utf8_bytes++;
          str_pos++;
          return true;
        case 't':
          if (nullptr != copy_dest && write_style::unescaped == w_style) { *copy_dest++ = '\t'; }
          if (write_style::escaped == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 't';
            }
            bytes_diff_for_escape_writing++;
          }
          if (!try_match_char(to_match_str_pos, to_match_str_end, '\t')) { return false; }
          string_token_utf8_bytes++;
          str_pos++;
          return true;
        // path 1 done: \", \', \\, \/, \b, \f, \n, \r, \t
        case 'u':
          // path 2: \u HEX HEX HEX HEX
          str_pos++;

          // for both unescaped/escaped writes corresponding utf8 bytes, no need
          // to pass in write style
          return try_skip_unicode(str_pos, to_match_str_pos, to_match_str_end, copy_dest);
        default:
          // path 3: invalid
          return false;
      }
    } else {
      // eof, no escaped char after char '\'
      return false;
    }
  }

  /**
   * parse:
   *   fragment SAFECODEPOINT
   *       // 1 not " or ' depending to allow_single_quotes
   *       // 2 not \
   *       // 3 non control character: Ascii value not in [0, 32)
   *     : ~ ["\\\u0000-\u001F]
   *     ;
   */
  __device__ inline bool try_skip_safe_code_point(char const*& str_pos, char c)
  {
    // 1 the char is not quoted(' or ") char, here satisfy, do not need to check
    // again

    // 2. the char is not \, here satisfy, do not need to check again

    // 3. chars not in [0, 32)
    int v = static_cast<int>(c);
    if (!(v >= 0 && v < 32)) {
      str_pos++;
      return true;
    } else {
      return false;
    }
  }

  /**
   * convert chars 0-9, a-f, A-F to int value
   */
  __device__ inline uint8_t hex_value(char c)
  {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  }

  /**
   * parse four HEX chars to unsigned int
   */
  __device__ inline cudf::char_utf8 parse_code_point(char const* p)
  {
    cudf::char_utf8 v = 0;
    for (size_t i = 0; i < 4; i++) {
      v = v * 16 + hex_value(p[i]);
    }
    return v;
  }

  /**
   * @brief Returns the number of bytes in the specified character.
   *
   * @param character Single character
   * @return Number of bytes
   */
  __device__ cudf::size_type bytes_in_char_utf8(cudf::char_utf8 character)
  {
    return 1 + static_cast<cudf::size_type>((character & 0x0000'FF00u) > 0) +
           static_cast<cudf::size_type>((character & 0x00FF'0000u) > 0) +
           static_cast<cudf::size_type>((character & 0xFF00'0000u) > 0);
  }

  /**
   * @brief Converts a character code-point value into a UTF-8 character.
   *
   * @param unchr Character code-point to convert.
   * @return Single UTF-8 character.
   */
  __device__ cudf::char_utf8 codepoint_to_utf8(uint32_t unchr)
  {
    cudf::char_utf8 utf8 = 0;
    if (unchr < 0x0000'0080) {
      // single byte utf8
      utf8 = unchr;
    } else if (unchr < 0x0000'0800) {
      // double byte utf8
      utf8 = (unchr << 2) & 0x1F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0x0000'C080;
    } else if (unchr < 0x0001'0000) {
      // triple byte utf8
      utf8 = (unchr << 4) & 0x0F'0000;
      utf8 |= (unchr << 2) & 0x00'3F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0x00E0'8080;
    } else if (unchr < 0x0011'0000) {
      // quadruple byte utf8
      utf8 = (unchr << 6) & 0x0700'0000;
      utf8 |= (unchr << 4) & 0x003F'0000;
      utf8 |= (unchr << 2) & 0x0000'3F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0xF080'8080u;
    }
    return utf8;
  }

  /**
   * @brief Place a char_utf8 value into a char array.
   *
   * @param character Single character
   * @param[out] str Output array.
   * @return The number of bytes in the character
   */
  __device__ cudf::size_type from_char_utf8(cudf::char_utf8 character, char* str)
  {
    cudf::size_type const chr_width = bytes_in_char_utf8(character);
    for (cudf::size_type idx = 0; idx < chr_width; ++idx) {
      str[chr_width - idx - 1] = static_cast<char>(character) & 0xFF;
      character                = character >> 8;
    }
    return chr_width;
  }

  /**
   * try skip 4 HEX chars
   * in pattern: '\\' 'u' HEX HEX HEX HEX, it's a code point of unicode
   */
  __device__ bool try_skip_unicode(char const*& str_pos,
                                   char const*& to_match_str_pos,
                                   char const* const to_match_str_end,
                                   char*& copy_dest)
  {
    // already parsed u
    bool is_success = try_skip_hex(str_pos) && try_skip_hex(str_pos) && try_skip_hex(str_pos) &&
                      try_skip_hex(str_pos);
    if (is_success) {
      // parse 4 HEX chars to uint32_t value
      auto code_point = parse_code_point(str_pos - 4);
      auto utf_char   = codepoint_to_utf8(code_point);
      // write utf8 bytes.
      // In UTF-8, the maximum number of bytes used to encode a single character
      // is 4
      char buff[4];
      cudf::size_type bytes = from_char_utf8(utf_char, buff);
      string_token_utf8_bytes += bytes;

      if (nullptr != copy_dest) {
        for (cudf::size_type i = 0; i < bytes; i++) {
          *copy_dest++ = buff[i];
        }
      }

      if (nullptr != to_match_str_pos) {
        for (cudf::size_type i = 0; i < bytes; i++) {
          if (!(to_match_str_pos < to_match_str_end && *to_match_str_pos == buff[i])) {
            return false;
          }
          to_match_str_pos++;
        }
      }

      return true;
    } else {
      return false;
    }
  }

  /**
   * try skip HEX
   */
  __device__ inline bool try_skip_hex(char const*& str_pos)
  {
    if (!eof(str_pos) && is_hex_digit(*str_pos)) {
      str_pos++;
      return true;
    }
    return false;
  }

  // =========== Parse string end ===========

  // =========== Parse number begin ===========

  /**
   * parse number, grammar is:
   * NUMBER
   *   : '-'? INT ('.' [0-9]+)? EXP?
   *   ;
   *
   * fragment INT
   *   // integer part forbis leading 0s (e.g. `01`)
   *   : '0'
   *   | [1-9] [0-9]*
   *   ;
   *
   * fragment EXP
   *   : [Ee] [+\-]? [0-9]+
   *   ;
   *
   * valid number:    0, 0.3, 0e005, 0E005
   * invalid number:  0., 0e, 0E
   *
   * Note: Leading zeroes are not allowed, keep consistent with Spark, e.g.: 00, -01 are invalid
   */
  __device__ inline void parse_number()
  {
    // parse sign
    try_skip(curr_pos, '-');

    // parse unsigned number
    bool is_float = false;
    // store number digits length
    // e.g.: +1.23e-45 length is 5
    int number_digits_length = 0;
    if (try_unsigned_number(is_float, number_digits_length)) {
      if (check_max_num_len(number_digits_length)) {
        curr_token = (is_float ? json_token::VALUE_NUMBER_FLOAT : json_token::VALUE_NUMBER_INT);
        // success parsed a number, update the token length
        number_token_len = curr_pos - current_token_start_pos;
      } else {
        curr_token = json_token::ERROR;
      }
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * verify max number digits length if enabled
   * e.g.: +1.23e-45 length is 5
   */
  __device__ inline bool check_max_num_len(int number_digits_length)
  {
    return
      // disabled num len check
      max_num_len <= 0 ||
      // enabled num len check
      (max_num_len > 0 && number_digits_length <= max_num_len);
  }

  /**
   * verify max string length if enabled
   */
  __device__ inline bool check_string_max_utf8_bytes()
  {
    return
      // disabled str len check
      max_string_utf8_bytes <= 0 ||
      // enabled str len check
      (max_string_utf8_bytes > 0 && string_token_utf8_bytes <= max_string_utf8_bytes);
  }

  /**
   * parse:  INT ('.' [0-9]+)? EXP?
   * and verify leading zeroes
   *
   * @param[out] is_float, if contains `.` or `e`, set true
   */
  __device__ inline bool try_unsigned_number(bool& is_float, int& number_digits_length)
  {
    if (!eof(curr_pos)) {
      char c = *curr_pos;
      if (c >= '1' && c <= '9') {
        curr_pos++;
        number_digits_length++;
        // first digit is [1-9]
        // path: INT = [1-9] [0-9]*
        number_digits_length += skip_zero_or_more_digits();
        return parse_number_from_fraction(is_float, number_digits_length);
      } else if (c == '0') {
        curr_pos++;
        number_digits_length++;

        // check leading zeros
        if (!eof(curr_pos)) {
          char next_char_after_zero = *curr_pos;
          if (next_char_after_zero >= '0' && next_char_after_zero <= '9') {
            // e.g.: 01 is invalid
            return false;
          }
        }

        // first digit is [0]
        // path: INT = '0'
        return parse_number_from_fraction(is_float, number_digits_length);
      } else {
        // first digit is non [0-9]
        return false;
      }
    } else {
      // eof, has no digits
      return false;
    }
  }

  /**
   * parse: ('.' [0-9]+)? EXP?
   * @param[is_float] is float
   */
  __device__ inline bool parse_number_from_fraction(bool& is_float, int& number_digits_length)
  {
    // parse fraction
    if (try_skip(curr_pos, '.')) {
      // has fraction
      is_float = true;
      // try pattern: [0-9]+
      if (!try_skip_one_or_more_digits(number_digits_length)) { return false; }
    }

    // parse exp
    if (!eof(curr_pos) && (*curr_pos == 'e' || *curr_pos == 'E')) {
      curr_pos++;
      is_float = true;
      return try_parse_exp(number_digits_length);
    }

    return true;
  }

  /**
   * parse: [0-9]*
   * skip zero or more [0-9]
   */
  __device__ inline int skip_zero_or_more_digits()
  {
    int digits = 0;
    while (!eof(curr_pos)) {
      if (is_digit(*curr_pos)) {
        digits++;
        curr_pos++;
      } else {
        // point to first non-digit char
        break;
      }
    }
    return digits;
  }

  /**
   * parse: [0-9]+
   * try skip one or more [0-9]
   * @param[out] len: skipped num of digits
   */
  __device__ inline bool try_skip_one_or_more_digits(int& number_digits_length)
  {
    if (!eof(curr_pos) && is_digit(*curr_pos)) {
      curr_pos++;
      number_digits_length++;
      number_digits_length += skip_zero_or_more_digits();
      return true;
    } else {
      return false;
    }
  }

  /**
   * parse [eE][+-]?[0-9]+
   * @param[out] exp_len exp len
   */
  __device__ inline bool try_parse_exp(int& number_digits_length)
  {
    // already parsed [eE]

    // parse [+-]?
    if (!eof(curr_pos) && (*curr_pos == '+' || *curr_pos == '-')) { curr_pos++; }

    // parse [0-9]+
    return try_skip_one_or_more_digits(number_digits_length);
  }

  // =========== Parse number end ===========

  /**
   * parse true
   */
  __device__ inline void parse_true()
  {
    // already parsed 't'
    if (try_skip(curr_pos, 'r') && try_skip(curr_pos, 'u') && try_skip(curr_pos, 'e')) {
      curr_token = json_token::VALUE_TRUE;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse false
   */
  __device__ inline void parse_false()
  {
    // already parsed 'f'
    if (try_skip(curr_pos, 'a') && try_skip(curr_pos, 'l') && try_skip(curr_pos, 's') &&
        try_skip(curr_pos, 'e')) {
      curr_token = json_token::VALUE_FALSE;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse null
   */
  __device__ inline void parse_null()
  {
    // already parsed 'n'
    if (try_skip(curr_pos, 'u') && try_skip(curr_pos, 'l') && try_skip(curr_pos, 'l')) {
      curr_token = json_token::VALUE_NULL;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse the key string in key:value pair
   */
  __device__ inline void parse_field_name()
  {
    auto [success, end_char_pos] =
      try_parse_string(curr_pos, nullptr, nullptr, nullptr, write_style::unescaped);
    if (success) {
      curr_pos   = end_char_pos;
      curr_token = json_token::FIELD_NAME;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * continute parsing the next token and update current token
   * Note: only parse one token at a time
   * @param[out] has_comma_before_token has comma before next token
   * @param[out] has_colon_before_token has colon before next token
   */
  __device__ inline json_token parse_next_token(bool& has_comma_before_token,
                                                bool& has_colon_before_token)
  {
    skip_whitespaces(curr_pos);
    if (!eof(curr_pos)) {
      char c = *curr_pos;
      if (is_context_stack_empty()) {
        // stack is empty

        if (curr_token == json_token::INIT) {
          // main root entry point
          current_token_start_pos = curr_pos;
          parse_first_token_in_value();
        } else {
          if (allow_tailing_sub_string) {
            // previous token is not INIT, means already get a token; stack is
            // empty; Successfully parsed. Note: ignore the tailing sub-string
            curr_token = json_token::SUCCESS;
          } else {
            // not eof, has extra useless tailing characters.
            curr_token = json_token::ERROR;
          }
        }
      } else {
        // stack is non-empty

        if (is_object_context()) {
          // in JSON object context
          if (curr_token == json_token::START_OBJECT) {
            // previous token is '{'
            if (c == '}') {
              // empty object
              // close curr object context
              current_token_start_pos = curr_pos;
              curr_pos++;
              curr_token = json_token::END_OBJECT;
              pop_curr_context();
            } else {
              // parse key in key:value pair
              current_token_start_pos = curr_pos;
              parse_field_name();
            }
          } else if (curr_token == json_token::FIELD_NAME) {
            if (c == ':') {
              has_colon_before_token = true;
              // skip ':' and parse value in key:value pair
              curr_pos++;
              skip_whitespaces(curr_pos);
              current_token_start_pos = curr_pos;
              parse_first_token_in_value();
            } else {
              curr_token = json_token::ERROR;
            }
          } else {
            // expect next key:value pair or '}'
            if (c == '}') {
              // end of object
              current_token_start_pos = curr_pos;
              curr_pos++;
              curr_token = json_token::END_OBJECT;
              pop_curr_context();
            } else if (c == ',') {
              has_comma_before_token = true;
              // parse next key:value pair
              curr_pos++;
              skip_whitespaces(curr_pos);
              current_token_start_pos = curr_pos;
              parse_field_name();
            } else {
              curr_token = json_token::ERROR;
            }
          }
        } else {
          // in Json array context
          if (curr_token == json_token::START_ARRAY) {
            // previous token is '['
            if (c == ']') {
              // curr: ']', empty array
              current_token_start_pos = curr_pos;
              curr_pos++;
              curr_token = json_token::END_ARRAY;
              pop_curr_context();
            } else {
              // non-empty array, parse the first value in the array
              current_token_start_pos = curr_pos;
              parse_first_token_in_value();
            }
          } else {
            if (c == ',') {
              has_comma_before_token = true;
              // skip ',' and parse the next value
              curr_pos++;
              skip_whitespaces(curr_pos);
              current_token_start_pos = curr_pos;
              parse_first_token_in_value();
            } else if (c == ']') {
              // end of array
              current_token_start_pos = curr_pos;
              curr_pos++;
              curr_token = json_token::END_ARRAY;
              pop_curr_context();
            } else {
              curr_token = json_token::ERROR;
            }
          }
        }
      }
    } else {
      // eof
      if (is_context_stack_empty() && curr_token != json_token::INIT) {
        // reach eof; stack is empty; previous token is not INIT
        curr_token = json_token::SUCCESS;
      } else {
        // eof, and meet the following cases:
        //   - has unclosed JSON array/object;
        //   - the whole JSON is empty
        curr_token = json_token::ERROR;
      }
    }
    return curr_token;
  }

 public:
  /**
   * continute parsing, get next token.
   * The final tokens are ERROR or SUCCESS;
   */
  __device__ json_token next_token()
  {
    // parse next token
    bool has_comma_before_token;  // no-initialization because of do not care here
    bool has_colon_before_token;  // no-initialization because of do not care here
    return parse_next_token(has_comma_before_token, has_colon_before_token);
  }

  /**
   * get current token
   */
  __device__ json_token get_current_token() { return curr_token; }

  /**
   * is valid JSON by parsing through all tokens
   */
  __device__ bool is_valid()
  {
    while (curr_token != json_token::ERROR && curr_token != json_token::SUCCESS) {
      next_token();
    }
    return curr_token == json_token::SUCCESS;
  }

  /**
   * skip children if current token is [ or {, or do nothing otherwise.
   * after this call, the current token is ] or } if token is { or [
   * @return true if JSON is valid so far, false otherwise.
   */
  __device__ bool try_skip_children()
  {
    if (curr_token == json_token::ERROR || curr_token == json_token::INIT ||
        curr_token == json_token::SUCCESS) {
      return false;
    }

    if (curr_token != json_token::START_OBJECT && curr_token != json_token::START_ARRAY) {
      return true;
    }

    int open = 1;
    while (true) {
      json_token t = next_token();
      if (t == json_token::START_OBJECT || t == json_token::START_ARRAY) {
        ++open;
      } else if (t == json_token::END_OBJECT || t == json_token::END_ARRAY) {
        if (--open == 0) { return true; }
      } else if (t == json_token::ERROR) {
        return false;
      }
    }
  }

  __device__ cudf::size_type compute_unescaped_len() { return write_unescaped_text(nullptr); }

  /**
   * unescape current token text, then write to destination
   * e.g.: '\\r' is a string with 2 chars '\' 'r', writes 1 char '\r'
   * e.g.: "\u4e2d\u56FD" are code points for Chinese chars "中国",
   *   writes 6 utf8 bytes: -28  -72 -83 -27 -101 -67
   * For number, write verbatim without normalization
   */
  __device__ cudf::size_type write_unescaped_text(char* destination)
  {
    switch (curr_token) {
      case json_token::VALUE_STRING:
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        try_parse_string(
          current_token_start_pos, nullptr, nullptr, destination, write_style::unescaped);
        return string_token_utf8_bytes;
      case json_token::VALUE_NUMBER_INT:
        if (number_token_len == 2 && current_token_start_pos[0] == '-' &&
            current_token_start_pos[1] == '0') {
          if (nullptr != destination) *destination++ = '0';
          return 1;
        }
        if (nullptr != destination) {
          for (cudf::size_type i = 0; i < number_token_len; ++i) {
            *destination++ = *(current_token_start_pos + i);
          }
        }
        return number_token_len;
      case json_token::VALUE_NUMBER_FLOAT: {
        // number normalization:
        // 0.03E-2 => 0.3E-5; infinity;
        // 200.000 => 200.0, 351.980 => 351.98, 12345678900000000000.0
        // => 1.23456789E19 0.0000000000003 => 3.0E-13; 0.003 => 0.003; 0.0003
        // => 3.0E-4 leverage function: `get_current_float_parts`
        double d_value = spark_rapids_jni::detail::stod(
          cudf::string_view(current_token_start_pos, number_token_len));
        return spark_rapids_jni::ftos_converter::double_normalization(d_value, destination);
      }
      case json_token::VALUE_TRUE:
        if (nullptr != destination) {
          *destination++ = 't';
          *destination++ = 'r';
          *destination++ = 'u';
          *destination++ = 'e';
        }
        return 4;
      case json_token::VALUE_FALSE:
        if (nullptr != destination) {
          *destination++ = 'f';
          *destination++ = 'a';
          *destination++ = 'l';
          *destination++ = 's';
          *destination++ = 'e';
        }
        return 5;
      case json_token::VALUE_NULL:
        if (nullptr != destination) {
          *destination++ = 'n';
          *destination++ = 'u';
          *destination++ = 'l';
          *destination++ = 'l';
        }
        return 4;
      case json_token::FIELD_NAME:
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        try_parse_string(
          current_token_start_pos, nullptr, nullptr, destination, write_style::unescaped);
        return string_token_utf8_bytes;
      case json_token::START_ARRAY:
        if (nullptr != destination) { *destination++ = '['; }
        return 1;
      case json_token::END_ARRAY:
        if (nullptr != destination) { *destination++ = ']'; }
        return 1;
      case json_token::START_OBJECT:
        if (nullptr != destination) { *destination++ = '{'; }
        return 1;
      case json_token::END_OBJECT:
        if (nullptr != destination) { *destination++ = '}'; }
        return 1;
      // for the following tokens, return false
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return 0;
    }
    return 0;
  }

  __device__ cudf::size_type compute_escaped_len() { return write_escaped_text(nullptr); }
  /**
   * escape current token text, then write to destination
   * e.g.: '"' is a string with 1 char '"', writes out 4 chars '"' '\' '\"' '"'
   * e.g.: "\u4e2d\u56FD" are code points for Chinese chars "中国",
   *   writes 8 utf8 bytes: '"' -28  -72 -83 -27 -101 -67 '"'
   * For number, write verbatim without normalization
   */
  __device__ cudf::size_type write_escaped_text(char* destination)
  {
    switch (curr_token) {
      case json_token::VALUE_STRING:
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        try_parse_string(
          current_token_start_pos, nullptr, nullptr, destination, write_style::escaped);
        return string_token_utf8_bytes + bytes_diff_for_escape_writing;
      case json_token::VALUE_NUMBER_INT:
        if (number_token_len == 2 && current_token_start_pos[0] == '-' &&
            current_token_start_pos[1] == '0') {
          if (nullptr != destination) *destination++ = '0';
          return 1;
        }
        if (nullptr != destination) {
          for (cudf::size_type i = 0; i < number_token_len; ++i) {
            *destination++ = *(current_token_start_pos + i);
          }
        }
        return number_token_len;
      case json_token::VALUE_NUMBER_FLOAT: {
        double d_value = spark_rapids_jni::detail::stod(
          cudf::string_view(current_token_start_pos, number_token_len));
        return spark_rapids_jni::ftos_converter::double_normalization(d_value, destination);
      }
      case json_token::VALUE_TRUE:
        if (nullptr != destination) {
          *destination++ = 't';
          *destination++ = 'r';
          *destination++ = 'u';
          *destination++ = 'e';
        }
        return 4;
      case json_token::VALUE_FALSE:
        if (nullptr != destination) {
          *destination++ = 'f';
          *destination++ = 'a';
          *destination++ = 'l';
          *destination++ = 's';
          *destination++ = 'e';
        }
        return 5;
      case json_token::VALUE_NULL:
        if (nullptr != destination) {
          *destination++ = 'n';
          *destination++ = 'u';
          *destination++ = 'l';
          *destination++ = 'l';
        }
        return 4;
      case json_token::FIELD_NAME:
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        try_parse_string(
          current_token_start_pos, nullptr, nullptr, destination, write_style::escaped);
        return string_token_utf8_bytes + bytes_diff_for_escape_writing;
      case json_token::START_ARRAY:
        if (nullptr != destination) { *destination++ = '['; }
        return 1;
      case json_token::END_ARRAY:
        if (nullptr != destination) { *destination++ = ']'; }
        return 1;
      case json_token::START_OBJECT:
        if (nullptr != destination) { *destination++ = '{'; }
        return 1;
      case json_token::END_OBJECT:
        if (nullptr != destination) { *destination++ = '}'; }
        return 1;
      // for the following tokens, return false
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return 0;
    }
    return 0;
  }

  /**
   * reset the parser
   */
  __device__ void reset()
  {
    curr_pos   = json_start_pos;
    curr_token = json_token::INIT;
    stack_size = 0;
  }

  /**
   * match field name string when current token is FIELD_NAME,
   * return true if current token is FIELD_NAME and match successfully.
   * return false otherwise,
   * Note: to_match_str_ptr should not be nullptr
   */
  __device__ bool match_current_field_name(cudf::string_view name)
  {
    return match_current_field_name(name.data(), name.size_bytes());
  }

  /**
   * match current field name
   */
  __device__ bool match_current_field_name(char const* to_match_str_ptr, cudf::size_type len)
  {
    if (json_token::FIELD_NAME == curr_token) {
      auto [b, end_pos] = try_parse_string(current_token_start_pos,
                                           to_match_str_ptr,
                                           to_match_str_ptr + len,
                                           nullptr,
                                           write_style::unescaped);
      return b;
    } else {
      return false;
    }
  }

  /**
   * copy current structure to destination.
   * return false if meets JSON format error,
   * reurn true otherwise.
   * @param[out] copy_to
   */
  __device__ thrust::pair<bool, size_t> copy_current_structure(char* copy_to)
  {
    switch (curr_token) {
      case json_token::INIT:
      case json_token::ERROR:
      case json_token::SUCCESS:
      case json_token::FIELD_NAME:
      case json_token::END_ARRAY:
      case json_token::END_OBJECT: return thrust::make_pair(false, 0);
      case json_token::VALUE_NUMBER_INT:
      case json_token::VALUE_NUMBER_FLOAT:
      case json_token::VALUE_STRING:
      case json_token::VALUE_TRUE:
      case json_token::VALUE_FALSE:
      case json_token::VALUE_NULL:
        // copy terminal token
        if (nullptr != copy_to) {
          size_t copy_len = write_escaped_text(copy_to);
          return thrust::make_pair(true, copy_len);
        } else {
          size_t copy_len = compute_escaped_len();
          return thrust::make_pair(true, copy_len);
        }
      case json_token::START_ARRAY:
      case json_token::START_OBJECT:
        // stack size increased by 1 when meet start object/array
        // copy until meet matched end object/array
        size_t sum_copy_len   = 0;
        int backup_stack_size = stack_size;

        // copy start object/array
        if (nullptr != copy_to) {
          int len = write_escaped_text(copy_to);
          sum_copy_len += len;
          copy_to += len;
        } else {
          sum_copy_len += compute_escaped_len();
        }

        while (true) {
          bool has_comma_before_token = false;
          bool has_colon_before_token = false;

          // parse and get has_comma_before_token, has_colon_before_token
          parse_next_token(has_comma_before_token, has_colon_before_token);

          // check the JSON format
          if (curr_token == json_token::ERROR) { return thrust::make_pair(false, 0); }

          // write out the token
          if (nullptr != copy_to) {
            if (has_comma_before_token) {
              sum_copy_len++;
              *copy_to++ = ',';
            }
            if (has_colon_before_token) {
              sum_copy_len++;
              *copy_to++ = ':';
            }
            int len = write_escaped_text(copy_to);
            sum_copy_len += len;
            copy_to += len;
          } else {
            if (has_comma_before_token) { sum_copy_len++; }
            if (has_colon_before_token) { sum_copy_len++; }
            sum_copy_len += compute_escaped_len();
          }

          if (backup_stack_size - 1 == stack_size) {
            // indicate meet the matched end object/array
            return thrust::make_pair(true, sum_copy_len);
          }
        }
        return thrust::make_pair(false, 0);
    }

    // never happen
    return thrust::make_pair(false, 0);
  }

 private:
  char const* const json_start_pos;
  char const* const json_end_pos;
  char const* curr_pos;
  json_token curr_token{json_token::INIT};

  // 64 bits long saves the nested object/array contexts
  // true(bit value 1) is JSON object context
  // false(bit value 0) is JSON array context
  // JSON parser checks array/object are mached, e.g.: [1,2) are wrong
  int64_t context_stack;
  int stack_size = 0;

  // save current token start pos, used by coping current token text
  char const* current_token_start_pos;
  // used to store number token length
  cudf::size_type number_token_len;

  // Records string/field name token utf8 bytes size after unescaped
  // e.g.: For JSON 4 chars string "\\n", after unescaped, get 1 char '\n'
  // used by checking the max string length
  int string_token_utf8_bytes;

  // Records bytes diff between escape writing and unescape writing
  // e.g.: 4 chars string "\\n", string_token_utf8_bytes is 1,
  // when `write_escaped_text`, will write out 4 chars: " \ n ",
  // then this diff will be 4 - 1 = 3
  int bytes_diff_for_escape_writing;
};

}  // namespace spark_rapids_jni
