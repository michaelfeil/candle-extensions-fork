use candle::{DType, Device, Tensor};
use candle_cudnn_attn::flash_attn_varlen;

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_dtype() -> DType {
    match std::env::var("DTYPE").ok().as_deref() {
        Some("bf16") | Some("BF16") => DType::BF16,
        _ => DType::F16,
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("on") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("off") => false,
        _ => default,
    }
}

fn sync_tensor(t: &Tensor) -> candle::Result<()> {
    let _ = t.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    Ok(())
}

struct CaseStats {
    first_call_ms: f64,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    toks_per_s: f64,
    warmup_total_ms: f64,
    warmup_avg_ms: f64,
    extra_warmup: usize,
}

fn process_rss_bytes() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb = line
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse::<usize>().ok())?;
    Some(kb * 1024)
}

fn print_mem_snapshot(tag: &str) {
    let cpu = process_rss_bytes().unwrap_or(0);
    let (gpu_free, gpu_total) = candle_cudnn_attn::debug_cuda_mem_info().unwrap_or((0, 0));
    let plans = candle_cudnn_attn::debug_cache_plan_count();
    let ws = candle_cudnn_attn::debug_cache_workspace_bytes();
    let gpu_used = gpu_total.saturating_sub(gpu_free);
    println!(
        "[mem:{tag}] cpu_rss={:.2} MiB gpu_used={:.2} MiB gpu_free={:.2} MiB gpu_total={:.2} MiB cache_plans={} cache_ws={:.2} MiB",
        bytes_to_mib(cpu),
        bytes_to_mib(gpu_used),
        bytes_to_mib(gpu_free),
        bytes_to_mib(gpu_total),
        plans,
        bytes_to_mib(ws),
    );
}

fn make_uniform_inputs(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> candle::Result<(Tensor, Tensor, Tensor, Tensor)> {
    let total_tokens = batch_size * seq_len;
    let q = Tensor::randn(0.0f32, 1.0f32, (total_tokens, heads, head_dim), device)?.to_dtype(dtype)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (total_tokens, heads, head_dim), device)?.to_dtype(dtype)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (total_tokens, heads, head_dim), device)?.to_dtype(dtype)?;

    let mut host_seqlens = Vec::with_capacity(batch_size + 1);
    host_seqlens.push(0u32);
    for i in 1..=batch_size {
        host_seqlens.push((i * seq_len) as u32);
    }
    let seqlens = Tensor::new(&host_seqlens[..], device)?;
    Ok((q, k, v, seqlens))
}

fn run_case(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens: &Tensor,
    max_seqlen: usize,
    softmax_scale: f32,
    causal: bool,
    warmup: usize,
    iters: usize,
) -> candle::Result<CaseStats> {
    let first_start = std::time::Instant::now();
    let out = flash_attn_varlen(q, k, v, seqlens, max_seqlen, softmax_scale, causal)?;
    sync_tensor(&out)?;
    let first_call_ms = first_start.elapsed().as_secs_f64() * 1000.0;

    let warmup_start = std::time::Instant::now();
    let extra_warmup = warmup.saturating_sub(1);
    for _ in 0..extra_warmup {
        let out = flash_attn_varlen(q, k, v, seqlens, max_seqlen, softmax_scale, causal)?;
        sync_tensor(&out)?;
    }
    let warmup_total_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
    let warmup_avg_ms = if extra_warmup > 0 {
        warmup_total_ms / extra_warmup as f64
    } else {
        0.0
    };

    let mut mean_ms = 0.0;
    let mut p50_ms = 0.0;
    let mut p95_ms = 0.0;
    let mut toks_per_s = 0.0;
    if iters > 0 {
        let mut times_ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = std::time::Instant::now();
            let out = flash_attn_varlen(q, k, v, seqlens, max_seqlen, softmax_scale, causal)?;
            sync_tensor(&out)?;
            let elapsed = start.elapsed();
            times_ms.push(elapsed.as_secs_f64() * 1000.0);
        }

        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        mean_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        p50_ms = times_ms[times_ms.len() / 2];
        let p95_idx = ((times_ms.len() as f64) * 0.95).floor() as usize;
        p95_ms = times_ms[p95_idx.min(times_ms.len() - 1)];
        let tokens = q.dim(0)? as f64;
        toks_per_s = tokens / (mean_ms / 1000.0);
    }

    Ok(CaseStats {
        first_call_ms,
        mean_ms,
        p50_ms,
        p95_ms,
        toks_per_s,
        warmup_total_ms,
        warmup_avg_ms,
        extra_warmup,
    })
}

fn print_case_stats(causal: bool, s: &CaseStats) {
    println!(
        "causal={:<5} first={:>8.3} ms  warmup_total({:>3})={:>8.3} ms  warmup_avg={:>8.3} ms  mean={:>8.3} ms  p50={:>8.3} ms  p95={:>8.3} ms  tok/s={:>12.1}",
        causal,
        s.first_call_ms,
        s.extra_warmup,
        s.warmup_total_ms,
        s.warmup_avg_ms,
        s.mean_ms,
        s.p50_ms,
        s.p95_ms,
        s.toks_per_s
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !candle_cudnn_attn::is_available() {
        eprintln!("cuDNN attention is not available");
        return Ok(());
    }

    let batch_size = env_usize("BATCH_SIZE", 4);
    let seq_len = env_usize("SEQ_LEN", 2048);
    let heads = env_usize("HEADS", 8);
    let head_dim = env_usize("HEAD_DIM", 128);
    let warmup = env_usize("WARMUP", 10);
    let iters = env_usize("ITERS", 100);
    let sweep_start = env_usize("SWEEP_SEQ_START", 0);
    let sweep_end = env_usize("SWEEP_SEQ_END", 0);
    let cold_probe = env_bool("COLD_PROBE", true);
    let dummy_batch = env_usize("DUMMY_BATCH_SIZE", 1);
    let dummy_seq = env_usize("DUMMY_SEQ_LEN", 16);
    let dtype = env_dtype();
    let total_tokens = batch_size * seq_len;

    let device = Device::new_cuda(0)?;
    let (q, k, v, seqlens) =
        make_uniform_inputs(&device, dtype, batch_size, seq_len, heads, head_dim)?;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    println!("candle-cudnn-attn perf bench");
    println!(
        "dtype={:?} B={} S={} H={} D={} total_tokens={} warmup={} iters={}",
        dtype, batch_size, seq_len, heads, head_dim, total_tokens, warmup, iters
    );
    println!("cold_probe={} dummy_B={} dummy_S={}", cold_probe, dummy_batch, dummy_seq);
    println!("results:");
    print_mem_snapshot("start");

    if cold_probe {
        let (dq, dk, dv, dseqlens) =
            make_uniform_inputs(&device, dtype, dummy_batch, dummy_seq, heads, head_dim)?;
        let dscale = 1.0 / (head_dim as f32).sqrt();
        let t0 = std::time::Instant::now();
        let dout = flash_attn_varlen(&dq, &dk, &dv, &dseqlens, dummy_seq, dscale, false)?;
        sync_tensor(&dout)?;
        let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "cold_start_dummy causal=false B={} S={} H={} D={} -> {:>8.3} ms",
            dummy_batch, dummy_seq, heads, head_dim, cold_ms
        );
        print_mem_snapshot("after_dummy_cold");
    }

    let s0 = run_case(
        &q, &k, &v, &seqlens, seq_len, softmax_scale, false, warmup, iters,
    )?;
    print_case_stats(false, &s0);
    print_mem_snapshot("after_causal_false");
    let s1 = run_case(
        &q, &k, &v, &seqlens, seq_len, softmax_scale, true, warmup, iters,
    )?;
    print_case_stats(true, &s1);
    print_mem_snapshot("after_causal_true");

    if sweep_start > 0 && sweep_end >= sweep_start {
        println!("\nsweep_seq_len start={} end={} step=1", sweep_start, sweep_end);
        println!("seq,causal,first_ms,mean_ms,p50_ms,p95_ms,tok_per_s");
        for s in sweep_start..=sweep_end {
            let (sq, sk, sv, sseq) = make_uniform_inputs(&device, dtype, batch_size, s, heads, head_dim)?;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let st_false = run_case(&sq, &sk, &sv, &sseq, s, scale, false, warmup, iters)?;
            println!(
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.1}",
                s, false, st_false.first_call_ms, st_false.mean_ms, st_false.p50_ms, st_false.p95_ms, st_false.toks_per_s
            );
            let st_true = run_case(&sq, &sk, &sv, &sseq, s, scale, true, warmup, iters)?;
            println!(
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.1}",
                s, true, st_true.first_call_ms, st_true.mean_ms, st_true.p50_ms, st_true.p95_ms, st_true.toks_per_s
            );
        }
    }
    print_mem_snapshot("end");

    Ok(())
}
