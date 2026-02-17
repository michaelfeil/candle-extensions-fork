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
}

pub fn run_thd_sdpa_fwd(
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
    causal: bool,
    is_bf16: bool,
    stream: *mut core::ffi::c_void,
) -> Result<(), String> {
    let status = unsafe {
        cudnn_thd_sdpa_fwd(
            q,
            k,
            v,
            o,
            seq_q,
            seq_kv,
            q_ragged,
            k_ragged,
            v_ragged,
            o_ragged,
            b,
            h,
            s,
            d,
            attn_scale,
            if causal { 1 } else { 0 },
            if is_bf16 { 1 } else { 0 },
            stream,
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
