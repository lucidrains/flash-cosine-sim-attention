#pragma once

// Custom dispatch inspired from
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
// https://github.com/swansontec/map-macro

// Macro utilities:

#include <ATen/Dispatch.h>

#define REMOVE_PAREN_IMPL(...) __VA_ARGS__
#define REMOVE_PAREN(args) REMOVE_PAREN_IMPL args

#define EVAL0(...) __VA_ARGS__
#define EVAL1(...) EVAL0(EVAL0(EVAL0(__VA_ARGS__)))
#define EVAL2(...) EVAL1(EVAL1(EVAL1(__VA_ARGS__)))
#define EVAL3(...) EVAL2(EVAL2(EVAL2(__VA_ARGS__)))
#define EVAL4(...) EVAL3(EVAL3(EVAL3(__VA_ARGS__)))
#define EVAL(...)  EVAL4(EVAL4(EVAL4(__VA_ARGS__)))

#define MAP_END(...)
#define MAP_OUT

#define MAP_GET_END2() 0, MAP_END
#define MAP_GET_END1(...) MAP_GET_END2
#define MAP_GET_END(...) MAP_GET_END1
#define MAP_NEXT0(test, next, ...) next MAP_OUT
#define MAP_NEXT1(test, next) MAP_NEXT0(test, next, 0)
#define MAP_NEXT(test, next)  MAP_NEXT1(MAP_GET_END test, next)

#define MAP0(f, TYPE_NAME, CASE_CODE, x, peek, ...) f(TYPE_NAME, CASE_CODE, x) MAP_NEXT(peek, MAP1)(f, TYPE_NAME, CASE_CODE, peek, __VA_ARGS__)
#define MAP1(f, TYPE_NAME, CASE_CODE, x, peek, ...) f(TYPE_NAME, CASE_CODE, x) MAP_NEXT(peek, MAP0)(f, TYPE_NAME, CASE_CODE, peek, __VA_ARGS__)
#define MAP(f, TYPE_NAME, CASE_CODE, ...) EVAL(MAP1(f, TYPE_NAME, CASE_CODE, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// Type dispatch

#define AT_TYPE_DISPATCH_CASE(TYPE_NAME, CASE_CODE, x)                            \
    case x: {                                                                     \
        using TYPE_NAME C10_UNUSED_DISPATCH_CUDA_WORKAROUND =                     \
          typename c10::impl::ScalarTypeToCPPType<x>::type;                       \
        REMOVE_PAREN(CASE_CODE)                                                   \
        break;                                                                    \
    }

#define AT_TYPE_DISPATCH_SWITCH(TYPE, TYPE_NAME, TYPES, CASE_CODE, DEFAULT_CODE)  \
  {                                                                               \
    switch (TYPE) {                                                               \
        MAP(AT_TYPE_DISPATCH_CASE, TYPE_NAME, CASE_CODE, REMOVE_PAREN(TYPES))     \
        default: {                                                                \
            REMOVE_PAREN(DEFAULT_CODE)                                            \
        }                                                                         \
    }                                                                             \
  }

// Value dispatch

#define VALUE_DISPATCH_CASE(VALUE_NAME, CASE_CODE, x)                             \
    case x: {                                                                     \
        constexpr const auto VALUE_NAME = x;                                      \
        REMOVE_PAREN(CASE_CODE)                                                   \
        break;                                                                    \
    }

#define VALUE_DISPATCH_SWITCH(VALUE, VALUE_NAME, VALUES, CASE_CODE, DEFAULT_CODE) \
  {                                                                               \
    switch (VALUE) {                                                              \
        MAP(VALUE_DISPATCH_CASE, VALUE_NAME, CASE_CODE, REMOVE_PAREN(VALUES))     \
        default: {                                                                \
            REMOVE_PAREN(DEFAULT_CODE)                                            \
        }                                                                         \
    }                                                                             \
  }
