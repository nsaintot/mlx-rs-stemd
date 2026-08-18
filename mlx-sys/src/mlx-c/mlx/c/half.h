/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_HALF_H
#define MLX_HALF_H

#ifdef __cplusplus
extern "C" {
#endif

/* Upstream defines these types on aarch64 only, preprocessing the four
   half-precision entry points below out of existence elsewhere; mlx-rs calls
   all four unconditionally. Both are 2-byte storage here (array.cpp goes
   through mlx::core's own half types), so any 2-byte POD of alignment 2 is a
   correct ABI stand-in.

   Split on compiler, not platform: two read this header. cl.exe compiles
   array.cpp and has neither _Float16 nor __bf16 at any version. libclang, via
   bindgen, has both, and renders them as __BindgenFloat16(pub u16) and u16,
   the shapes mlx-rs's Guard impls destructure. clang-cl defines _MSC_VER and
   __clang__, and takes the clang branch. */
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__aarch64__)
#define HAS_FLOAT16
#include <arm_fp16.h>
typedef __fp16 float16_t;
#elif defined(_MSC_VER) && !defined(__clang__)
#define HAS_FLOAT16
typedef unsigned short float16_t;
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#define HAS_FLOAT16
typedef _Float16 float16_t;
#endif

#if defined(__ARM_FEATURE_BF16) || defined(__aarch64__)
#define HAS_BFLOAT16
#include <arm_bf16.h>
typedef __bf16 bfloat16_t;
#elif defined(_MSC_VER) && !defined(__clang__)
#define HAS_BFLOAT16
typedef unsigned short bfloat16_t;
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#define HAS_BFLOAT16
typedef __bf16 bfloat16_t;
#endif

#ifdef __cplusplus
}
#endif

#endif
