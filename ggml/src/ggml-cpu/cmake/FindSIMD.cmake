include(CheckCSourceRuns)

set(AVX_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 a;
        a = _mm256_set1_ps(0);
        return 0;
    }
")

set(AVX512_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0);
        __m512i b = a;
        __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
        return 0;
    }
")

set(AVX512_VNNI_CODE "
    #include <immintrin.h>
    int main()
    {
        volatile int v = 1;
        __m512i a = _mm512_set1_epi8((char) v);
        a = _mm512_dpbusd_epi32(a, a, a);
        return _mm512_reduce_add_epi32(a) == 0;
    }
")

set(AVX512_VBMI_CODE "
    #include <immintrin.h>
    int main()
    {
        volatile int v = 1;
        __m512i a = _mm512_set1_epi8((char) v);
        a = _mm512_permutexvar_epi8(a, a);
        return _mm512_reduce_add_epi32(a) == 0;
    }
")

set(AVX512_BF16_CODE "
    #include <immintrin.h>
    int main()
    {
        volatile float v = 1.0f;
        __m512 a = _mm512_set1_ps(v);
        __m512bh b = _mm512_cvtne2ps_pbh(a, a);
        a = _mm512_dpbf16_ps(a, b, b);
        return _mm512_reduce_add_ps(a) == 0.0f;
    }
")

set(AVX2_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256i a = {0};
        a = _mm256_abs_epi16(a);
        __m256i x;
        _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
        return 0;
    }
")

set(FMA_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 acc = _mm256_setzero_ps();
        const __m256 d = _mm256_setzero_ps();
        const __m256 p = _mm256_setzero_ps();
        acc = _mm256_fmadd_ps( d, p, acc );
        return 0;
    }
")

macro(check_sse type flags)
    set(__FLAG_I 1)
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    foreach (__FLAG ${flags})
        if (NOT ${type}_FOUND)
            set(CMAKE_REQUIRED_FLAGS ${__FLAG})
            check_c_source_runs("${${type}_CODE}" HAS_${type}_${__FLAG_I})
            if (HAS_${type}_${__FLAG_I})
                set(${type}_FOUND TRUE CACHE BOOL "${type} support")
                set(${type}_FLAGS "${__FLAG}" CACHE STRING "${type} flags")
            endif()
            math(EXPR __FLAG_I "${__FLAG_I}+1")
        endif()
    endforeach()
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if (NOT ${type}_FOUND)
        set(${type}_FOUND FALSE CACHE BOOL "${type} support")
        set(${type}_FLAGS "" CACHE STRING "${type} flags")
    endif()

    mark_as_advanced(${type}_FOUND ${type}_FLAGS)
endmacro()

# flags are for MSVC only!
check_sse("AVX" " ;/arch:AVX")
if (NOT ${AVX_FOUND})
    set(GGML_AVX OFF)
else()
    set(GGML_AVX ON)
endif()

check_sse("AVX2" " ;/arch:AVX2")
check_sse("FMA" " ;/arch:AVX2")
if ((NOT ${AVX2_FOUND}) OR (NOT ${FMA_FOUND}))
    set(GGML_AVX2 OFF)
else()
    set(GGML_AVX2 ON)
endif()

check_sse("AVX512" " ;/arch:AVX512")
if (NOT ${AVX512_FOUND})
    set(GGML_AVX512 OFF)
else()
    set(GGML_AVX512 ON)

    # MSVC defines no macros for these extensions, so detect them by running code
    check_sse("AVX512_VNNI" "/arch:AVX512")
    check_sse("AVX512_VBMI" "/arch:AVX512")
    check_sse("AVX512_BF16" "/arch:AVX512")
    foreach (ext VNNI VBMI BF16)
        if (${AVX512_${ext}_FOUND})
            set(GGML_AVX512_${ext} ON)
        endif()
    endforeach()
endif()
