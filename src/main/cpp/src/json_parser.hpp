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

#include <cudf/strings/detail/utf8.hpp>
#include <cudf/types.hpp>

#include <thrust/pair.h>

namespace spark_rapids_jni {

/**
 * @brief Settings for `json_parser_options()`.
 */
class json_parser_options {
  // allow single quotes to represent strings in JSON
  bool allow_single_quotes = false;

  // Whether allow unescaped control characters in JSON Strings.
  bool allow_unescaped_control_chars = false;

  // Define the maximum JSON String length.
  int max_string_len = -1;

  // Define the maximum JSON number length.
  int max_num_len = -1;

  // Whether allow tailing useless sub-string
  bool allow_tailing_sub_string = false;

 public:
  /**
   * @brief Default constructor.
   */
  explicit json_parser_options() = default;

  /**
   * @brief Returns true/false depending on whether single-quotes for representing strings
   * are allowed.
   *
   * @return true if single-quotes are allowed, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_allow_single_quotes() const
  {
    return allow_single_quotes;
  }

  /**
   * @brief Returns true/false depending on whether unescaped characters for representing strings
   * are allowed.
   *
   * Unescaped control characters are ASCII characters with value less than 32,
   * including tab and line feed characters.
   *
   * If true, JSON is not conventional format.
   * e.g., how to represent carriage return and newline characters:
   *   if true, allow "\n\r" two control characters without escape directly
   *   if false, "\n\r" are not allowed, should use escape characters: "\\n\\r"
   *
   * @return true if unescaped characters are allowed, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_allow_unescaped_control_chars() const
  {
    return allow_unescaped_control_chars;
  }

  /**
   * @brief Returns maximum JSON String length, negative or zero means no limitation.
   *
   * By default, maximum JSON String length is negative one, means no limitation.
   * e.g.: The length of String "\\n" is 1, JSON parser does not count escape characters.
   *
   * @return integer value of allowed maximum JSON String length
   */
  [[nodiscard]] CUDF_HOST_DEVICE int get_max_string_len() const { return max_string_len; }

  /**
   * @brief Returns maximum JSON number length, negative or zero means no limitation.
   *
   * By default, maximum JSON number length is negative one, means no limitation.
   *
   * e.g.: The length of number -123.45e-67 is 7. if maximum JSON number length is 6,
   * then this number is a invalid number.
   *
   * @return integer value of allowed maximum JSON number length
   */
  [[nodiscard]] CUDF_HOST_DEVICE int get_max_num_len() const { return max_num_len; }

  /**
   * @brief Returns whether allow tailing useless sub-string in JSON.
   *
   * If true, e.g., the following invalid JSON is allowed, because prefix {'k' : 'v'} is valid.
   *   {'k' : 'v'}_extra_tail_sub_string
   *
   * @return true if alow tailing useless sub-string, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE int get_allow_tailing_sub_string() const
  {
    return allow_tailing_sub_string;
  }

  /**
   * @brief Set whether single-quotes for strings are allowed.
   *
   * @param _allow_single_quotes bool indicating desired behavior.
   */
  void set_allow_single_quotes(bool _allow_single_quotes)
  {
    allow_single_quotes = _allow_single_quotes;
  }

  /**
   * @brief Set whether alow unescaped control characters.
   *
   * @param _allow_unescaped_control_chars bool indicating desired behavior.
   */
  void set_allow_unescaped_control_chars(bool _allow_unescaped_control_chars)
  {
    allow_unescaped_control_chars = _allow_unescaped_control_chars;
  }

  /**
   * @brief Set maximum JSON String length.
   *
   * @param _max_string_len integer indicating desired behavior.
   */
  void set_max_string_len(int _max_string_len) { max_string_len = _max_string_len; }

  /**
   * @brief Set maximum JSON number length.
   *
   * @param _max_num_len integer indicating desired behavior.
   */
  void set_max_num_len(int _max_num_len) { max_num_len = _max_num_len; }

  /**
   * @brief Set whether allow tailing useless sub-string.
   *
   * @param _allow_tailing_sub_string bool indicating desired behavior.
   */
  void set_allow_tailing_sub_string(bool _allow_tailing_sub_string)
  {
    allow_tailing_sub_string = _allow_tailing_sub_string;
  }
};

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
 * Note: when setting `allow_single_quotes` or `allow_unescaped_control_chars`, then JSON
 * format is not conventional.
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
 *      here `asscii_control_chars` represents control chars which in Ascii code range: [0, 32)
 *
 *  When `allow_unescaped_control_chars` is false:
 *    Invalid string: "asscii_control_chars"
 *      here `asscii_control_chars` represents control chars which in Ascii code range: [0, 32)
 *
 */
template <int max_json_nesting_depth>
class json_parser {
 public:
  CUDF_HOST_DEVICE inline json_parser(json_parser_options const& _options,
                                      char const* const _json_start_pos,
                                      cudf::size_type const _json_len)
    : options(_options),
      json_start_pos(_json_start_pos),
      json_end_pos(_json_start_pos + _json_len),
      curr_pos(_json_start_pos)
  {
  }

 private:
  /**
   * is current position EOF?
   */
  CUDF_HOST_DEVICE inline bool eof() { return curr_pos >= json_end_pos; }

  /**
   * is hex digits: 0-9, A-F, a-f
   */
  CUDF_HOST_DEVICE inline bool is_hex_digit(char c)
  {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
  }

  /**
   * is 0 to 9 digit
   */
  CUDF_HOST_DEVICE inline bool is_digit(char c) { return (c >= '0' && c <= '9'); }

  /**
   * is white spaces: ' ', '\t', '\n' '\r'
   */
  CUDF_HOST_DEVICE inline bool is_whitespace(char c)
  {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  /**
   * skips 4 characters: ' ', '\t', '\n' '\r'
   */
  CUDF_HOST_DEVICE inline void skip_whitespaces()
  {
    while (!eof() && is_whitespace(*curr_pos)) {
      curr_pos++;
    }
  }

  /**
   * check current char, if it's expected, then plus the position
   */
  CUDF_HOST_DEVICE inline bool try_skip(char expected)
  {
    if (!eof() && *curr_pos == expected) {
      curr_pos++;
      return true;
    }
    return false;
  }

  /**
   * try to push current context into stack
   * if nested depth exceeds limitation, return false
   */
  CUDF_HOST_DEVICE inline bool try_push_context(json_token token)
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
  CUDF_HOST_DEVICE inline void push_context(json_token token)
  {
    bool v              = json_token::START_OBJECT == token ? true : false;
    stack[stack_size++] = v;
  }

  /**
   * whether the top of nested context stack is JSON object context
   * true is object, false is array
   * only has two contexts: object or array
   */
  CUDF_HOST_DEVICE inline bool is_object_context() { return stack[stack_size - 1]; }

  /**
   * pop top context from stack
   */
  CUDF_HOST_DEVICE inline void pop_curr_context() { stack_size--; }

  /**
   * is context stack is empty
   */
  CUDF_HOST_DEVICE inline bool is_context_stack_empty() { return stack_size == 0; }

  /**
   * parse the first value token from current position
   * e.g., after finished this function:
   *   current token is START_OBJECT if current value is object
   *   current token is START_ARRAY if current value is array
   *   current token is string/num/true/false/null if current value is terminal
   *   current token is ERROR if parse failed
   */
  CUDF_HOST_DEVICE inline void parse_first_token_in_value()
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
        if (options.get_allow_single_quotes()) {
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
  CUDF_HOST_DEVICE inline void parse_single_quoted_string()
  {
    if (try_parse_single_quoted_string()) {
      curr_token = json_token::VALUE_STRING;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse " quoted string
   */
  CUDF_HOST_DEVICE inline void parse_double_quoted_string()
  {
    if (try_parse_double_quoted_string()) {
      curr_token = json_token::VALUE_STRING;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /*
   * try parse ' or " quoted string
   * when allow single quote, first try single quote
   */
  CUDF_HOST_DEVICE inline bool try_parse_string()
  {
    if (options.get_allow_single_quotes() && *curr_pos == '\'') {
      return try_parse_single_quoted_string();
    } else {
      return try_parse_double_quoted_string();
    }
  }

  /**
   * try parse ' quoted string
   */
  CUDF_HOST_DEVICE inline bool try_parse_single_quoted_string() { return try_parse_string('\''); }

  /**
   * try parse " quoted string
   */
  CUDF_HOST_DEVICE inline bool try_parse_double_quoted_string() { return try_parse_string('\"'); }

  /**
   * try parse quoted string using passed `quote_char`
   * `quote_char` can be ' or "
   * For UTF-8 encoding:
   *   Single byte char: The most significant bit of the byte is always 0
   *   Two-byte characters: The leading bits of the first byte are 110,
   *     and the leading bits of the second byte are 10.
   *   Three-byte characters: The leading bits of the first byte are 1110,
   *     and the leading bits of the second and third bytes are 10.
   *   Four-byte characters: The leading bits of the first byte are 11110,
   *     and the leading bits of the second, third, and fourth bytes are 10.
   * Because JSON structural chars([ ] { } , :), string quote char(" ') and
   * Escape char \ are all Ascii(The leading bit is 0), so it's safe that do
   * not convert byte array to UTF-8 char.
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
   */
  CUDF_HOST_DEVICE inline bool try_parse_string(char quote_char,
                                                bool copy       = false,
                                                char* copy_dest = nullptr)
  {
    // copy start position for VALUE_STRING/FIELD_NAME token
    token_start_pos = curr_pos;
    if (!try_skip(quote_char)) { return false; }

    int curr_str_len = 0;
    // scan string content
    while (!eof()) {
      char c = *curr_pos;
      int v  = static_cast<int>(c);
      if (c == quote_char) {
        // path 1: close string
        curr_pos++;
        return verify_max_string_len(curr_str_len);
      } else if (v >= 0 && v < 32 && options.get_allow_unescaped_control_chars()) {
        // path 2: unescaped control char
        if (copy) { *copy_dest++ = *curr_pos; }
        curr_pos++;
        curr_str_len++;
        continue;
      } else {
        switch (c) {
          case '\\':
            // path 3: escape path
            curr_pos++;
            if (try_skip_escape_part(copy, copy_dest)) {
              return false;
            } else {
              curr_str_len++;
            }
            break;

          default:
            // path 4: safe code point
            if (!try_skip_safe_code_point(c)) {
              return false;
            } else {
              if (copy) { *copy_dest++ = c; }
              curr_str_len++;
            }
        }
      }
    }

    return false;
  }

  /**
   * skip the second char in \", \', \\, \/, \b, \f, \n, \r, \t;
   * skip the HEX chars in \u HEX HEX HEX HEX.
   * @return positive escaped ASCII value if success, -1 otherwise
   */
  CUDF_HOST_DEVICE inline bool try_skip_escape_part(bool copy = false, char*& copy_dest = nullptr)
  {
    // already skipped the first '\'
    // try skip second part
    if (!eof()) {
      char c = *curr_pos;
      switch (*curr_pos) {
        case '\"':
          if (copy) { *copy_dest++ = c; }
          curr_pos++;
          return true;
        case '\'':
          // only allow escape ' when `allow_single_quotes`
          if (copy) { *copy_dest++ = c; }
          curr_pos++;
          return options.get_allow_single_quotes();
        case '\\':
        case '/':
          if (copy) { *copy_dest++ = c; }
        case 'b':
          if (copy) { *copy_dest++ = '\f'; }
        case 'f':
          if (copy) { *copy_dest++ = '\f'; }
        case 'n':
          if (copy) { *copy_dest++ = '\n'; }
        case 'r':
          if (copy) { *copy_dest++ = '\r'; }
        case 't':
          if (copy) { *copy_dest++ = '\t'; }
          // path 1: \", \', \\, \/, \b, \f, \n, \r, \t
          curr_pos++;
          return true;
        case 'u':
          // path 2: \u HEX HEX HEX HEX
          curr_pos++;
          return try_skip_unicode(copy, copy_dest);
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
  CUDF_HOST_DEVICE inline bool try_skip_safe_code_point(char c)
  {
    // 1 the char is not quoted(' or ") char, here satisfy, do not need to check again

    // 2. the char is not \, here satisfy, do not need to check again

    // 3. chars not in [0, 32)
    int v = static_cast<int>(c);
    if (!(v >= 0 && v < 32)) {
      curr_pos++;
      return true;
    } else {
      return false;
    }
  }

  /**
   * convert chars 0-9, a-f, A-F to int value
   */
  CUDF_HOST_DEVICE inline uint8_t hex_value(char c)
  {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  }

  /**
   * parse four HEX chars to unsigned int
   */
  CUDF_HOST_DEVICE inline cudf::char_utf8 parse_utf_char(char const* p)
  {
    cudf::char_utf8 v = 0;
    for (size_t i = 0; i < 4; i++) {
      v = v * 16 + hex_value(p[i]);
    }
    return v;
  }

  /**
   * try skip 4 HEX chars
   * in pattern: '\\' 'u' HEX HEX HEX HEX
   */
  CUDF_HOST_DEVICE inline bool try_skip_unicode(bool copy = false, char*& copy_dest = nullptr)
  {
    // already parsed u
    bool is_success = try_skip_hex() && try_skip_hex() && try_skip_hex() && try_skip_hex();
    if (is_success) {
      if (copy) {
        // parse 4 HEX chars to uint32_t value
        auto utf_char = parse_utf_char(curr_pos - 4);
        // write utf8 bytes
        cudf::size_type bytes = cudf::strings::detail::from_char_utf8(utf_char, copy_dest);
        copy_dest += bytes;
      }
      return true;
    } else {
      return false;
    }
  }

  /**
   * try skip HEX
   */
  CUDF_HOST_DEVICE inline bool try_skip_hex()
  {
    if (!eof() && is_hex_digit(*curr_pos)) {
      curr_pos++;
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
   */
  CUDF_HOST_DEVICE inline void parse_number()
  {
    // copy start position for number token
    token_start_pos = curr_pos;

    // parse sign
    try_skip('-');

    // parse unsigned number
    bool is_float = false;
    if (try_unsigned_number(is_float)) {
      curr_token = (is_float ? json_token::VALUE_NUMBER_FLOAT : json_token::VALUE_NUMBER_INT);
      // success parsed a number, update the token length
      token_str_len = curr_pos - token_start_pos;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * verify max number length if enabled
   * e.g.: -1.23e-456, int len is 1, fraction len is 2, exp len is 3
   */
  CUDF_HOST_DEVICE inline bool verify_max_num_len(int int_len, int fraction_len, int exp_len)
  {
    int sum_len = int_len + fraction_len + exp_len;
    return
      // disabled num len check
      options.get_max_num_len() <= 0 ||
      // enabled num len check
      (options.get_max_num_len() > 0 && sum_len <= options.get_max_num_len());
  }

  /**
   * verify max string length if enabled
   */
  CUDF_HOST_DEVICE inline bool verify_max_string_len(int str_len)
  {
    return
      // disabled str len check
      options.get_max_string_len() <= 0 ||
      // enabled str len check
      (options.get_max_string_len() > 0 && str_len <= options.get_max_string_len());
  }

  /**
   * parse:  INT ('.' [0-9]+)? EXP?
   *
   * @param[out] is_float, if contains `.` or `e`, set true
   */
  CUDF_HOST_DEVICE inline bool try_unsigned_number(bool& is_float)
  {
    int int_len      = 0;
    int fraction_len = 0;
    int exp_len      = 0;

    if (!eof()) {
      char c = *curr_pos;
      if (c >= '1' && c <= '9') {
        curr_pos++;
        int_len++;
        // first digit is [1-9]
        // path: INT = [1-9] [0-9]*
        int_len += skip_zero_or_more_digits();
        return parse_number_from_fraction(is_float, fraction_len, exp_len) &&
               verify_max_num_len(int_len, fraction_len, exp_len);
      } else if (c == '0') {
        curr_pos++;
        int_len++;
        // first digit is [0]
        // path: INT = '0'
        return parse_number_from_fraction(is_float, fraction_len, exp_len) &&
               verify_max_num_len(int_len, fraction_len, exp_len);
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
   * @param[fraction_len] fraction len in number, e.g.: 1.23 23 is fraction part
   * @param[exp_len] exp len in number, e.g.: 1e234 234 is exp part
   */
  CUDF_HOST_DEVICE inline bool parse_number_from_fraction(bool& is_float,
                                                          int& fraction_len,
                                                          int& exp_len)
  {
    // parse fraction
    if (try_skip('.')) {
      // has fraction
      is_float = true;
      // try pattern: [0-9]+
      if (!try_skip_one_or_more_digits(fraction_len)) { return false; }
    }

    // parse exp
    if (!eof() && (*curr_pos == 'e' || *curr_pos == 'E')) {
      curr_pos++;
      is_float = true;
      return try_parse_exp(exp_len);
    }

    return true;
  }

  /**
   * parse: [0-9]*
   * skip zero or more [0-9]
   */
  CUDF_HOST_DEVICE inline int skip_zero_or_more_digits()
  {
    int digits = 0;
    while (!eof()) {
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
  CUDF_HOST_DEVICE inline bool try_skip_one_or_more_digits(int& len)
  {
    if (!eof() && is_digit(*curr_pos)) {
      curr_pos++;
      len++;
      len += skip_zero_or_more_digits();
      return true;
    } else {
      return false;
    }
  }

  /**
   * parse [eE][+-]?[0-9]+
   * @param[out] exp_len exp len
   */
  CUDF_HOST_DEVICE inline bool try_parse_exp(int& exp_len)
  {
    // already parsed [eE]

    // parse [+-]?
    if (!eof() && (*curr_pos == '+' || *curr_pos == '-')) { curr_pos++; }

    // parse [0-9]+
    return try_skip_one_or_more_digits(exp_len);
  }

  // =========== Parse number end ===========

  /**
   * parse true
   */
  CUDF_HOST_DEVICE inline void parse_true()
  {
    // already parsed 't'
    if (try_skip('r') && try_skip('u') && try_skip('e')) {
      curr_token = json_token::VALUE_TRUE;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse false
   */
  CUDF_HOST_DEVICE inline void parse_false()
  {
    // already parsed 'f'
    if (try_skip('a') && try_skip('l') && try_skip('s') && try_skip('e')) {
      curr_token = json_token::VALUE_FALSE;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse null
   */
  CUDF_HOST_DEVICE inline void parse_null()
  {
    // already parsed 'n'
    if (try_skip('u') && try_skip('l') && try_skip('l')) {
      curr_token = json_token::VALUE_NULL;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse the key string in key:value pair
   */
  CUDF_HOST_DEVICE inline void parse_field_name()
  {
    if (try_parse_string()) {
      curr_token = json_token::FIELD_NAME;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * continute parsing the next token and update current token
   * Note: only parse one token at a time
   */
  CUDF_HOST_DEVICE inline json_token parse_next_token()
  {
    // SUCCESS or ERROR means parsing is completed,
    // should not call this function again.
    assert(curr_token != json_token::SUCCESS && curr_token != json_token::ERROR);

    skip_whitespaces();

    if (!eof()) {
      char c = *curr_pos;
      if (is_context_stack_empty()) {
        // stack is empty

        if (curr_token == json_token::INIT) {
          // main root entry point
          parse_first_token_in_value();
        } else {
          if (options.get_allow_tailing_sub_string()) {
            // previous token is not INIT, means already get a token; stack is empty;
            // Successfully parsed.
            // Note: ignore the tailing sub-string
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
              curr_pos++;
              curr_token = json_token::END_OBJECT;
              pop_curr_context();
            } else {
              // parse key in key:value pair
              parse_field_name();
            }
          } else if (curr_token == json_token::FIELD_NAME) {
            if (c == ':') {
              // skip ':' and parse value in key:value pair
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value();
            } else {
              curr_token = json_token::ERROR;
            }
          } else {
            // expect next key:value pair or '}'
            if (c == '}') {
              // end of object
              curr_pos++;
              curr_token = json_token::END_OBJECT;
              pop_curr_context();
            } else if (c == ',') {
              // parse next key:value pair
              curr_pos++;
              skip_whitespaces();
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
              curr_pos++;
              curr_token = json_token::END_ARRAY;
              pop_curr_context();
            } else {
              // non-empty array, parse the first value in the array
              parse_first_token_in_value();
            }
          } else {
            if (c == ',') {
              // skip ',' and parse the next value
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value();
            } else if (c == ']') {
              // end of array
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
  CUDF_HOST_DEVICE json_token next_token() { return parse_next_token(); }

  /**
   * get current token
   */
  CUDF_HOST_DEVICE json_token get_current_token() { return curr_token; }

  /**
   * is valid JSON by parsing through all tokens
   */
  CUDF_HOST_DEVICE bool is_valid()
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
  CUDF_HOST_DEVICE bool try_skip_children()
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

  /**
   * try to copy current text
   * @return false if current token is SUCCESS/ERROR/INIT, true otherwise
   */
  CUDF_HOST_DEVICE bool try_copy_raw_text(char* destination)
  {
    switch (curr_token) {
      case json_token::VALUE_STRING:
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        curr_pos = token_start_pos;
        // token_start_pos should be ' or "
        try_parse_string(*token_start_pos, true, destination);
        return true;
      case json_token::VALUE_NUMBER_INT:
      case json_token::VALUE_NUMBER_FLOAT:
        // number can be copied from JSON string directly
        for (cudf::size_type i = 0; i < token_str_len; ++i) {
          *destination++ = *(token_start_pos + i);
        }
        return true;
      case json_token::VALUE_TRUE:
        *destination++ = 't';
        *destination++ = 'r';
        *destination++ = 'u';
        *destination++ = 'e';
        return true;
      case json_token::VALUE_FALSE:
        *destination++ = 'f';
        *destination++ = 'a';
        *destination++ = 'l';
        *destination++ = 's';
        *destination++ = 'e';
        return true;
      case json_token::VALUE_NULL:
        *destination++ = 'n';
        *destination++ = 'u';
        *destination++ = 'l';
        *destination++ = 'l';
        return true;
      case json_token::FIELD_NAME:
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        curr_pos = token_start_pos;
        // token_start_pos should be ' or "
        try_parse_string(*token_start_pos, true, destination);
        return true;
      case json_token::START_ARRAY: *destination++ = '['; return true;
      case json_token::END_ARRAY: *destination++ = ']'; return true;
      case json_token::START_OBJECT: *destination++ = '{'; return true;
      case json_token::END_OBJECT: *destination++ = '}'; return true;
      // for the following tokens, return false
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return false;
    }
  }

  /**
   * reset the parser
   */
  CUDF_HOST_DEVICE void reset()
  {
    curr_pos        = json_start_pos;
    curr_token      = json_token::INIT;
    stack_size      = 0;
    token_start_pos = json_start_pos;
    token_str_len   = 0;
  }

  /**
   * get current number text.
   * if current token is not number, return pair(nullptr, -1)
   */
  CUDF_HOST_DEVICE thrust::pair<char const *, cudf::size_type> get_current_number_text()
  {
    if (curr_token != json_token::VALUE_NUMBER_FLOAT &&
        curr_token != json_token::VALUE_NUMBER_INT)
    {
      return thrust::make_pair(nullptr, -1);
    }

    return thrust::make_pair(token_start_pos, token_str_len);
  }

 private:
  json_parser_options const& options;
  char const* const json_start_pos{nullptr};
  char const* const json_end_pos{nullptr};

  char const* curr_pos{nullptr};
  json_token curr_token{json_token::INIT};

  // saves the nested contexts: JSON object context or JSON array context
  // true is JSON object context; false is JSON array context
  // When encounter EOF and this stack is non-empty, means non-closed JSON object/array,
  // then parsing will fail.
  bool stack[max_json_nesting_depth];
  int stack_size = 0;

  // used by copy text
  char const* token_start_pos;
  cudf::size_type token_str_len;
};

}  // namespace spark_rapids_jni
