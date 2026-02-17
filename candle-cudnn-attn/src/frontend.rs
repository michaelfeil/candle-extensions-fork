use std::ffi::CStr;

unsafe extern "C" {
    fn cudnn_thd_sdpa_fwd(
        q: *mut core::ffi::c_void,
        k: *mut core::ffi::c_void,
        v: *mut core::ffi::c_void,
        o: *mut core::ffi::c_void,
        seq_q: *const i32,
        seq_kv: *const i32,
        q_ragged: *const i64,
        k_ragged: *const i64,
        v_ragged: *const i64,
        o_ragged: *const i64,
        b: i64,
        h: i64,
        s: i64,
        d: i64,
        attn_scale: f32,
        causal: i32,
        is_bf16: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn cudnn_thd_last_error() -> *const core::ffi::c_char;
    fn cudnn_thd_cache_plan_count() -> i64;
    fn cudnn_thd_cache_workspace_bytes() -> i64;
    fn cudnn_thd_cuda_mem_info(free_bytes: *mut i64, total_bytes: *mut i64) -> i32;
}

pub(crate) struct ThdSdpaFwdParams {
    pub(crate) q: *mut core::ffi::c_void,
    pub(crate) k: *mut core::ffi::c_void,
    pub(crate) v: *mut core::ffi::c_void,
    pub(crate) o: *mut core::ffi::c_void,
    pub(crate) seq_q: *const i32,
    pub(crate) seq_kv: *const i32,
    pub(crate) q_ragged: *const i64,
    pub(crate) k_ragged: *const i64,
    pub(crate) v_ragged: *const i64,
    pub(crate) o_ragged: *const i64,
    pub(crate) b: i64,
    pub(crate) h: i64,
    pub(crate) s: i64,
    pub(crate) d: i64,
    pub(crate) attn_scale: f32,
    pub(crate) causal: bool,
    pub(crate) is_bf16: bool,
    pub(crate) stream: *mut core::ffi::c_void,
}

pub(crate) fn run_thd_sdpa_fwd(params: &ThdSdpaFwdParams) -> Result<(), String> {
    let status = unsafe {
        cudnn_thd_sdpa_fwd(
            params.q,
            params.k,
            params.v,
            params.o,
            params.seq_q,
            params.seq_kv,
            params.q_ragged,
            params.k_ragged,
            params.v_ragged,
            params.o_ragged,
            params.b,
            params.h,
            params.s,
            params.d,
            params.attn_scale,
            if params.causal { 1 } else { 0 },
            if params.is_bf16 { 1 } else { 0 },
            params.stream,
        )
    };

    if status == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = cudnn_thd_last_error();
            if ptr.is_null() {
                "unknown cudnn frontend error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(msg)
    }
}

pub fn cache_plan_count() -> usize {
    let v = unsafe { cudnn_thd_cache_plan_count() };
    if v < 0 { 0 } else { v as usize }
}

pub fn cache_workspace_bytes() -> usize {
    let v = unsafe { cudnn_thd_cache_workspace_bytes() };
    if v < 0 { 0 } else { v as usize }
}

pub fn cuda_mem_info() -> Result<(usize, usize), String> {
    let mut free_b = 0i64;
    let mut total_b = 0i64;
    let rc = unsafe { cudnn_thd_cuda_mem_info(&mut free_b as *mut i64, &mut total_b as *mut i64) };
    if rc == 0 {
        Ok((free_b.max(0) as usize, total_b.max(0) as usize))
    } else {
        Err("cudaMemGetInfo failed".to_string())
    }
}
