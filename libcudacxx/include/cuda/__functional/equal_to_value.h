//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H
#define _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Checks if two values are equal to a specified value
template <typename T>
struct equal_to_value
{
  T value_;

  explicit constexpr equal_to_value(const T& value) noexcept
      : value_(value)
  {}

  __host__ __device__ constexpr bool operator()(const T& lhs) const noexcept {
    return lhs == value_;
  }
};

// Checks if a value is equal to one
template <typename T>
struct equal_to_one : equal_to_value<T>
{
  constexpr equal_to_one() noexcept
      : equal_to_value<T>(static_cast<T>(1))
  {}
};

// Checks if a value is equal to its default constructed value
template <typename T>
struct equal_to_default : equal_to_value<T>
{
  constexpr equal_to_default() noexcept
      : equal_to_value<T>(T{})
  {}
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H
