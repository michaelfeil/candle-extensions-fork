#pragma once

#include <stdint.h>

extern "C" {

int cudnn_thd_sdpa_fwd(
    void* q,
    void* k,
    void* v,
    void* o,
    const int32_t* seq_q,
    const int32_t* seq_kv,
    const int64_t* q_ragged,
    const int64_t* k_ragged,
    const int64_t* v_ragged,
    const int64_t* o_ragged,
    int64_t b,
    int64_t h,
    int64_t s,
    int64_t d,
    float attn_scale,
    int causal,
    int is_bf16,
    void* stream);

const char* cudnn_thd_last_error(void);

int64_t cudnn_thd_cache_plan_count(void);
int64_t cudnn_thd_cache_workspace_bytes(void);
int cudnn_thd_cuda_mem_info(int64_t* free_bytes, int64_t* total_bytes);

}
