#pragma once

// Include cuDNN frontend headers
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_frontend/Tensor.h>
#include <cudnn_frontend/Operation.h>
#include <cudnn_frontend/Engine.h>
#include <cudnn_frontend/ExecutionPlan.h>
#include <cudnn_frontend/node/scaled_dot_product_flash_attention.h>

// C wrapper functions for Rust FFI
extern "C" {

// Graph operations
void* cudnn_graph_create(int io_data_type, int intermediate_data_type, int compute_data_type);
void cudnn_graph_destroy(void* graph);
int cudnn_graph_build(void* graph, void** execution_plan, size_t* workspace_size);

// Tensor operations
void* cudnn_tensor_create(void* graph, const int64_t* dims, int nb_dims, int data_type);
void cudnn_tensor_destroy(void* tensor);
void cudnn_tensor_set_output(void* tensor, int is_output);
void cudnn_tensor_set_ragged_offset(void* tensor, void* ragged_offset);

// SDPA operations
int cudnn_sdpa_forward(void* graph, void* q, void* k, void* v, 
                       float attn_scale, int causal, int use_padding_mask,
                       void** output, void** stats);

// Execution
int cudnn_graph_execute(void* graph, void* execution_plan, void* workspace, size_t workspace_size);

// Error handling
const char* cudnn_get_error_string(int error_code);

}