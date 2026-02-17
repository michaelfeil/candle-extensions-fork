#include "cudnn_thd_frontend.h"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cudnn_frontend_utils.h"
#include "cudnn_frontend_Heuristics.h"
#include "cudnn_frontend_find_plan.h"
#include "cudnn_frontend/graph_interface.h"
#include "cudnn_frontend/backend/kernel_cache.h"

namespace {
thread_local std::string g_last_error;

int fail(const std::string& msg) {
    g_last_error = msg;
    return -1;
}

struct CacheKey {
    int64_t b;
    int64_t h;
    int64_t s;
    int64_t d;
    int causal;
    int is_bf16;
    uint32_t scale_bits;

    bool operator==(const CacheKey& other) const {
        return b == other.b && h == other.h && s == other.s && d == other.d && causal == other.causal &&
               is_bf16 == other.is_bf16 && scale_bits == other.scale_bits;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const {
        std::size_t seed = 0;
        auto mix = [&seed](std::size_t v) {
            seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        mix(std::hash<int64_t>{}(k.b));
        mix(std::hash<int64_t>{}(k.h));
        mix(std::hash<int64_t>{}(k.s));
        mix(std::hash<int64_t>{}(k.d));
        mix(std::hash<int>{}(k.causal));
        mix(std::hash<int>{}(k.is_bf16));
        mix(std::hash<uint32_t>{}(k.scale_bits));
        return seed;
    }
};

struct CacheEntry {
    std::shared_ptr<cudnn_frontend::graph::Graph> graph;
    void* workspace = nullptr;
    int64_t workspace_bytes = 0;
};

struct CacheValue {
    CacheEntry entry;
    std::list<CacheKey>::iterator lru_it;
};

struct ThreadContext {
    static constexpr std::size_t kMaxPlans = 2048;

    cudnnHandle_t handle = nullptr;
    std::shared_ptr<cudnn_frontend::KernelCache> kernel_cache;
    std::unordered_map<CacheKey, CacheValue, CacheKeyHash> cache;
    std::list<CacheKey> lru;

    ~ThreadContext() {
        for (auto& kv : cache) {
            if (kv.second.entry.workspace) {
                cudaFree(kv.second.entry.workspace);
            }
        }
        if (handle) {
            cudnnDestroy(handle);
        }
    }
};

thread_local ThreadContext g_ctx;

bool dynamic_kernel_cache_enabled() {
    static int enabled = []() {
        const char* v = std::getenv("CUDNN_THD_DYNAMIC_KERNEL_CACHE");
        if (!v) {
            return 1;
        }
        return (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0) ? 0
                                                                                                              : 1;
    }();
    return enabled != 0;
}

CacheEntry build_cache_entry(ThreadContext& ctx, const CacheKey& key, float attn_scale) {
    namespace fe = cudnn_frontend;
    CacheEntry entry;
    entry.graph = std::make_shared<fe::graph::Graph>();
    entry.graph->set_io_data_type(key.is_bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Try to use dynamic-kernel-cache path to reduce first-build latency for shape churn.
    if (dynamic_kernel_cache_enabled()) {
        if (!ctx.kernel_cache) {
            ctx.kernel_cache = std::make_shared<fe::KernelCache>();
        }
        entry.graph->set_dynamic_shape_enabled(true).set_kernel_cache(ctx.kernel_cache);
    }

    constexpr int64_t Q_UID = 1;
    constexpr int64_t K_UID = 2;
    constexpr int64_t V_UID = 3;
    constexpr int64_t O_UID = 4;
    constexpr int64_t SEQ_Q_UID = 5;
    constexpr int64_t SEQ_KV_UID = 6;
    constexpr int64_t Q_RAGGED_UID = 7;
    constexpr int64_t K_RAGGED_UID = 8;
    constexpr int64_t V_RAGGED_UID = 9;
    constexpr int64_t O_RAGGED_UID = 10;

    auto q_rag = entry.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("q_ragged")
                                         .set_uid(Q_RAGGED_UID)
                                         .set_data_type(fe::DataType_t::INT64)
                                         .set_dim({key.b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1}));
    auto k_rag = entry.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("k_ragged")
                                         .set_uid(K_RAGGED_UID)
                                         .set_data_type(fe::DataType_t::INT64)
                                         .set_dim({key.b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1}));
    auto v_rag = entry.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("v_ragged")
                                         .set_uid(V_RAGGED_UID)
                                         .set_data_type(fe::DataType_t::INT64)
                                         .set_dim({key.b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1}));
    auto o_rag = entry.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("o_ragged")
                                         .set_uid(O_RAGGED_UID)
                                         .set_data_type(fe::DataType_t::INT64)
                                         .set_dim({key.b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1}));

    std::vector<int64_t> bhsd_dim = {key.b, key.h, key.s, key.d};
    std::vector<int64_t> bhsd_stride = {key.h * key.s * key.d, key.d, key.h * key.d, 1};

    auto Q = entry.graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Q")
                                     .set_uid(Q_UID)
                                     .set_dim(bhsd_dim)
                                     .set_stride(bhsd_stride)
                                     .set_ragged_offset(q_rag));
    auto K = entry.graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("K")
                                     .set_uid(K_UID)
                                     .set_dim(bhsd_dim)
                                     .set_stride(bhsd_stride)
                                     .set_ragged_offset(k_rag));
    auto V = entry.graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("V")
                                     .set_uid(V_UID)
                                     .set_dim(bhsd_dim)
                                     .set_stride(bhsd_stride)
                                     .set_ragged_offset(v_rag));

    auto seqQ = entry.graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_q")
                                        .set_uid(SEQ_Q_UID)
                                        .set_data_type(fe::DataType_t::INT32)
                                        .set_dim({key.b, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1}));
    auto seqKV = entry.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("seq_kv")
                                         .set_uid(SEQ_KV_UID)
                                         .set_data_type(fe::DataType_t::INT32)
                                         .set_dim({key.b, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("thd_sdpa")
                            .set_generate_stats(false)
                            .set_attn_scale(attn_scale)
                            .set_padding_mask(true)
                            .set_seq_len_q(seqQ)
                            .set_seq_len_kv(seqKV);
    if (key.causal) {
        sdpa_options.set_causal_mask(true);
    }

    auto out = entry.graph->sdpa(Q, K, V, sdpa_options);
    auto O = out[0];
    O->set_output(true)
        .set_uid(O_UID)
        .set_dim(bhsd_dim)
        .set_stride(bhsd_stride)
        .set_ragged_offset(o_rag);

    // Favor heuristics-A engines for steady-state speed.
    std::vector<fe::HeurMode_t> modes = {fe::HeurMode_t::A};
    auto build_status = entry.graph->build(ctx.handle, modes);
    if (build_status.is_bad()) {
        throw std::runtime_error(std::string("graph build failed: ") + build_status.get_message());
    }

    int64_t workspace_size = 0;
    auto ws_status = entry.graph->get_workspace_size(workspace_size);
    if (ws_status.is_bad()) {
        throw std::runtime_error(std::string("get_workspace_size failed: ") + ws_status.get_message());
    }

    if (workspace_size > 0) {
        cudaError_t cerr = cudaMalloc(&entry.workspace, static_cast<size_t>(workspace_size));
        if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc workspace failed: ") + cudaGetErrorString(cerr));
        }
        entry.workspace_bytes = workspace_size;
    }

    return entry;
}
}  // namespace

extern "C" int cudnn_thd_sdpa_fwd(
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
    void* stream) {
    try {
        if (!q || !k || !v || !o || !seq_q || !seq_kv || !q_ragged || !k_ragged || !v_ragged || !o_ragged) {
            return fail("null pointer passed to cudnn_thd_sdpa_fwd");
        }
        if (b <= 0 || h <= 0 || s <= 0 || d <= 0) {
            return fail("invalid dimensions in cudnn_thd_sdpa_fwd");
        }

        ThreadContext& ctx = g_ctx;
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (ctx.handle == nullptr) {
            st = cudnnCreate(&ctx.handle);
            if (st != CUDNN_STATUS_SUCCESS) {
                return fail(std::string("cudnnCreate failed: ") + cudnnGetErrorString(st));
            }
        }

        if (stream) {
            st = cudnnSetStream(ctx.handle, reinterpret_cast<cudaStream_t>(stream));
            if (st != CUDNN_STATUS_SUCCESS) {
                return fail(std::string("cudnnSetStream failed: ") + cudnnGetErrorString(st));
            }
        }

        uint32_t scale_bits = 0;
        std::memcpy(&scale_bits, &attn_scale, sizeof(scale_bits));
        CacheKey key{b, h, s, d, causal, is_bf16, scale_bits};
        auto it = ctx.cache.find(key);
        if (it != ctx.cache.end()) {
            ctx.lru.splice(ctx.lru.begin(), ctx.lru, it->second.lru_it);
            it->second.lru_it = ctx.lru.begin();
        } else {
            try {
                auto entry = build_cache_entry(ctx, key, attn_scale);
                ctx.lru.push_front(key);
                CacheValue v{std::move(entry), ctx.lru.begin()};
                it = ctx.cache.emplace(key, std::move(v)).first;

                if (ctx.cache.size() > ThreadContext::kMaxPlans) {
                    CacheKey evict_key = ctx.lru.back();
                    ctx.lru.pop_back();
                    auto evict_it = ctx.cache.find(evict_key);
                    if (evict_it != ctx.cache.end()) {
                        if (evict_it->second.entry.workspace) {
                            cudaFree(evict_it->second.entry.workspace);
                        }
                        ctx.cache.erase(evict_it);
                    }
                }
            } catch (const std::exception& e) {
                return fail(std::string("cache build exception: ") + e.what());
            }
        }

        constexpr int64_t Q_UID = 1;
        constexpr int64_t K_UID = 2;
        constexpr int64_t V_UID = 3;
        constexpr int64_t O_UID = 4;
        constexpr int64_t SEQ_Q_UID = 5;
        constexpr int64_t SEQ_KV_UID = 6;
        constexpr int64_t Q_RAGGED_UID = 7;
        constexpr int64_t K_RAGGED_UID = 8;
        constexpr int64_t V_RAGGED_UID = 9;
        constexpr int64_t O_RAGGED_UID = 10;
        std::unordered_map<int64_t, void*> variant_pack = {
            {Q_UID, q},
            {K_UID, k},
            {V_UID, v},
            {O_UID, o},
            {SEQ_Q_UID, const_cast<int32_t*>(seq_q)},
            {SEQ_KV_UID, const_cast<int32_t*>(seq_kv)},
            {Q_RAGGED_UID, const_cast<int64_t*>(q_ragged)},
            {K_RAGGED_UID, const_cast<int64_t*>(k_ragged)},
            {V_RAGGED_UID, const_cast<int64_t*>(v_ragged)},
            {O_RAGGED_UID, const_cast<int64_t*>(o_ragged)},
        };

        auto exec_status = it->second.entry.graph->execute(ctx.handle, variant_pack, it->second.entry.workspace);
        if (exec_status.is_bad()) {
            return fail(std::string("graph execute failed: ") + exec_status.get_message());
        }

        return 0;
    } catch (const std::exception& e) {
        return fail(std::string("exception: ") + e.what());
    } catch (...) {
        return fail("unknown exception");
    }
}

extern "C" const char* cudnn_thd_last_error(void) {
    return g_last_error.c_str();
}

extern "C" int64_t cudnn_thd_cache_plan_count(void) {
    return static_cast<int64_t>(g_ctx.cache.size());
}

extern "C" int64_t cudnn_thd_cache_workspace_bytes(void) {
    int64_t total = 0;
    for (const auto& kv : g_ctx.cache) {
        total += kv.second.entry.workspace_bytes;
    }
    return total;
}

extern "C" int cudnn_thd_cuda_mem_info(int64_t* free_bytes, int64_t* total_bytes) {
    if (!free_bytes || !total_bytes) {
        return -1;
    }
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t st = cudaMemGetInfo(&free_b, &total_b);
    if (st != cudaSuccess) {
        return -1;
    }
    *free_bytes = static_cast<int64_t>(free_b);
    *total_bytes = static_cast<int64_t>(total_b);
    return 0;
}
