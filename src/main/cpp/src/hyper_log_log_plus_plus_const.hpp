/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

namespace spark_rapids_jni {

/**
 * The number of bits that is required for a HLLPP register value.
 *
 * This number is determined by the maximum number of leading binary zeros a
 * hashcode can produce. This is equal to the number of bits the hashcode
 * returns. The current implementation uses a 64-bit hashcode, this means 6-bits
 * are (at most) needed to store the number of leading zeros.
 */
constexpr int REGISTER_VALUE_BITS = 6;

// MASK binary 6 bits: 111-111
constexpr uint64_t MASK = (1L << REGISTER_VALUE_BITS) - 1L;

// This value is 10, one long stores 10 register values
constexpr int REGISTERS_PER_LONG = 64 / REGISTER_VALUE_BITS;

// XXHash seed, consistent with Spark
constexpr int64_t SEED = 42L;

// max precision, if require a precision bigger than 18, then use 18.
constexpr int MAX_PRECISION = 18;

}  // namespace spark_rapids_jni
