#include "cudnn_thd_frontend.h"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cudnn_frontend_utils.h"
#include "cudnn_frontend_Heuristics.h"
#include "cudnn_frontend_find_plan.h"
#include "cudnn_frontend/graph_interface.h"

namespace {
thread_local std::string g_last_error;

int fail(const std::string& msg) {
    g_last_error = msg;
    return -1;
}

struct HandleGuard {
    cudnnHandle_t handle = nullptr;
    ~HandleGuard() {
        if (handle) {
            cudnnDestroy(handle);
        }
    }
};
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

        HandleGuard guard;
        cudnnStatus_t st = cudnnCreate(&guard.handle);
        if (st != CUDNN_STATUS_SUCCESS) {
            return fail(std::string("cudnnCreate failed: ") + cudnnGetErrorString(st));
        }
        if (stream) {
            st = cudnnSetStream(guard.handle, reinterpret_cast<cudaStream_t>(stream));
            if (st != CUDNN_STATUS_SUCCESS) {
                return fail(std::string("cudnnSetStream failed: ") + cudnnGetErrorString(st));
            }
        }

        namespace fe = cudnn_frontend;
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(is_bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

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

        auto q_rag = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("q_ragged")
                                       .set_uid(Q_RAGGED_UID)
                                       .set_data_type(fe::DataType_t::INT64)
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1}));
        auto k_rag = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("k_ragged")
                                       .set_uid(K_RAGGED_UID)
                                       .set_data_type(fe::DataType_t::INT64)
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1}));
        auto v_rag = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("v_ragged")
                                       .set_uid(V_RAGGED_UID)
                                       .set_data_type(fe::DataType_t::INT64)
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1}));
        auto o_rag = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("o_ragged")
                                       .set_uid(O_RAGGED_UID)
                                       .set_data_type(fe::DataType_t::INT64)
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1}));

        std::vector<int64_t> bhsd_dim = {b, h, s, d};
        std::vector<int64_t> bhsd_stride = {h * s * d, d, h * d, 1};

        auto Q = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q")
                                   .set_uid(Q_UID)
                                   .set_dim(bhsd_dim)
                                   .set_stride(bhsd_stride)
                                   .set_ragged_offset(q_rag));
        auto K = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K")
                                   .set_uid(K_UID)
                                   .set_dim(bhsd_dim)
                                   .set_stride(bhsd_stride)
                                   .set_ragged_offset(k_rag));
        auto V = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V")
                                   .set_uid(V_UID)
                                   .set_dim(bhsd_dim)
                                   .set_stride(bhsd_stride)
                                   .set_ragged_offset(v_rag));

        auto seqQ = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("seq_q")
                                      .set_uid(SEQ_Q_UID)
                                      .set_data_type(fe::DataType_t::INT32)
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1}));
        auto seqKV = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_kv")
                                       .set_uid(SEQ_KV_UID)
                                       .set_data_type(fe::DataType_t::INT32)
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1}));

        auto sdpa_options = fe::graph::SDPA_attributes()
                                .set_name("thd_sdpa")
                                .set_generate_stats(false)
                                .set_attn_scale(attn_scale)
                                .set_padding_mask(true)
                                .set_seq_len_q(seqQ)
                                .set_seq_len_kv(seqKV);
        if (causal) {
            sdpa_options.set_causal_mask(true);
        }

        auto out = graph->sdpa(Q, K, V, sdpa_options);
        auto O = out[0];
        O->set_output(true)
            .set_uid(O_UID)
            .set_dim(bhsd_dim)
            .set_stride(bhsd_stride)
            .set_ragged_offset(o_rag);

        std::vector<fe::HeurMode_t> modes = {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK};
        auto build_status = graph->build(guard.handle, modes);
        if (build_status.is_bad()) {
            return fail(std::string("graph build failed: ") + build_status.get_message());
        }

        int64_t workspace_size = 0;
        auto ws_status = graph->get_workspace_size(workspace_size);
        if (ws_status.is_bad()) {
            return fail(std::string("get_workspace_size failed: ") + ws_status.get_message());
        }

        void* workspace = nullptr;
        if (workspace_size > 0) {
            cudaError_t cerr = cudaMalloc(&workspace, static_cast<size_t>(workspace_size));
            if (cerr != cudaSuccess) {
                return fail(std::string("cudaMalloc workspace failed: ") + cudaGetErrorString(cerr));
            }
        }

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

        auto exec_status = graph->execute(guard.handle, variant_pack, workspace);
        if (workspace) {
            cudaFree(workspace);
        }
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
