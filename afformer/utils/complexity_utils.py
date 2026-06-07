import time
import torch
import torch.nn as nn

def flops_via_thop(model: nn.Module, x: torch.Tensor):
    """Use thop to count MACs (multiply-accumulates)."""
    try:
        from thop import profile as thop_profile
        HAS_THOP = True
    except ImportError:
        HAS_THOP = False

    if not HAS_THOP:
        return None, None
    model.eval()
    with torch.no_grad():
        macs, params = thop_profile(model, inputs=(x,), verbose=False)
    return macs, params

def profile_model(model, input_data):
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, input_data)
    macs, _ = flops_via_thop(model, input_data)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    print(f"Total MACs: {macs / 1e9:.2f} G")
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} G")

def make_dummy_data_dict(
    device="cuda",
    B=1,
    record_len_value=2,      
    k_processed_len=2,      
    M=5000,                  # 随机生成的 voxel 数量（可以改大一点/小一点）
    max_points_per_voxel=32,
    num_point_features=4,   # PillarVFE 里默认 num_point_features=4
    grid_size=(504, 200, 1),# (W, H, nz) 对应 scatter 用的 nx, ny, nz
    k_frame=1,               # forward 里取 [:,0,...]，所以让第二维至少有 0
    ):
    assert B == 1, "当前 dummy 只按 B=1 构造"
    L = record_len_value  # max cav（这里就是 2）

    nx, ny, nz = grid_size

    # voxel_features: [M, 32, 4]
    voxel_features = torch.randn(
        M, max_points_per_voxel, num_point_features,
        device=device, dtype=torch.float32
    )

    # voxel_num_points: [M]
    voxel_num_points = torch.randint(
        low=1, high=max_points_per_voxel + 1, size=(M,),
        device=device, dtype=torch.int64
    )

    # voxel_coords: [M, 4]，scatter 里按 [batch_idx, z, y, x] 用
    # 你的 record_len=2，所以 batch_idx（这里对应“cav槽位”）取 0/1
    batch_idx = torch.randint(0, L, (M,), device=device, dtype=torch.int64)
    z = torch.zeros((M,), device=device, dtype=torch.int64)  # nz=1，所以永远是 0
    y = torch.randint(0, ny, (M,), device=device, dtype=torch.int64)
    x = torch.randint(0, nx, (M,), device=device, dtype=torch.int64)

    voxel_coords = torch.stack([batch_idx, z, y, x], dim=1)  # [M,4]

    # k_processed_lidar: list，长度=3，包含索引 0/1/2
    k_processed_lidar = []
    for _ in range(k_processed_len):
        k_processed_lidar.append({
            "voxel_features": voxel_features.clone(),
            "voxel_coords": voxel_coords.clone(),
            "voxel_num_points": voxel_num_points.clone(),
        })

    # record_len: [B]，你要 record_len==2（Pdb里看到的是 shape=[1]）
    record_len = torch.tensor([record_len_value], device=device, dtype=torch.int64)

    # fading_gain / SNR_db：长度应为 sum(record_len)=2（Pdb里也是 shape=[2]）
    fading_gain = torch.ones(L, device=device, dtype=torch.float32)
    SNR_db = torch.ones(L, device=device, dtype=torch.float32)

    # k_pairwise_t_matrix: 需要能让下面这句工作：
    # pairwise_t_matrix = data_dict['k_pairwise_t_matrix'][:,0,:,:,:,:]
    # 所以原始 shape 设为 [B, k_frame, L, L, 4, 4]
    k_pairwise_t_matrix = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 1, 1, 4, 4)
    k_pairwise_t_matrix = k_pairwise_t_matrix.repeat(B, k_frame, L, L, 1, 1).contiguous()

    data_dict = {
        "k_processed_lidar": k_processed_lidar,
        "record_len": record_len,
        "fading_gain": fading_gain,
        "SNR_db": SNR_db,
        "k_pairwise_t_matrix": k_pairwise_t_matrix,
    }
    return data_dict


def _gpu_benchmark(model, x, warmup=50, repeats=200):
    """CUDA-event based latency measurement. Returns list of ms values."""
    model.eval()
    latencies = []

    # warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # benchmark
    with torch.no_grad():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))   # ms

    return latencies


def _cpu_benchmark(model, x, warmup=20, repeats=100):
    """CPU wall-clock latency measurement."""
    model.eval()
    latencies = []

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(x)
            latencies.append((time.perf_counter() - t0) * 1000)   # ms

    return latencies


def benchmark_model(model, x, device, warmup, repeats):
    if device.type == "cuda":
        return _gpu_benchmark(model, x, warmup, repeats)
    else:
        return _cpu_benchmark(model, x, warmup, repeats)


def summarize(latencies: list) -> dict:
    arr = np.array(latencies)
    return {
        "mean_ms"  : float(arr.mean()),
        "std_ms"   : float(arr.std()),
        "min_ms"   : float(arr.min()),
        "p50_ms"   : float(np.percentile(arr, 50)),
        "p95_ms"   : float(np.percentile(arr, 95)),
        "p99_ms"   : float(np.percentile(arr, 99)),
        "max_ms"   : float(arr.max()),
    }


def measure_latency(model, x, device):
    dev_type = device.type if isinstance(device, torch.device) else str(device)

    if dev_type == "cuda":
        warmup = 50
        repeats = 200
    else:  # 默认为 CPU 设置
        warmup = 20
        repeats = 100

    lats = benchmark_model(model, x, device, warmup, repeats)
    stats = summarize(lats)

    print(stats)

    latencies = stats["mean_ms"]
    print(f"Inference Latency: {latencies:.2f} ms")

    return lats

def plot_latency_distribution_histogram(ds, ss):
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path


    fig, axes = plt.subplots(2, 1, figsize=(6, 7))
    labels = {"dual": "DualSA (Ours)", "standard": "Traditional SA"}
    colors = {"dual": "#2196F3", "standard": "#FF5722"}

    for ax, (k, lats) in zip(axes, [("dual", ds), ("standard", ss)]):
        ax.hist(lats, bins=40, color=colors[k], alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(lats), color="black", lw=2, linestyle="--",
                   label=f"Mean={np.mean(lats):.2f}ms")
        ax.set_title(labels[k], fontsize=12, fontweight="bold")
        ax.set_xlabel("Inference Latency (ms)")
        ax.set_ylabel("Count")
        ax.legend()

    # fig.suptitle("Latency Distribution", fontsize=12)
    fig.tight_layout()

    plot_path = Path(__file__).parent / "results" / "latency_dist.pdf"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Latency distribution plot saved to {plot_path}")
 

def measure_memory(model, B, C, H, W, device, name):
    """
    Returns (peak_forward_MB, peak_backward_MB).
    If device is CPU, returns (nan, nan).
    """
    if device.type != "cuda":
        return float("nan"), float("nan")

    import gc

    x = torch.randn(B, C, H, W, device=device, requires_grad=True)

    # ── forward peak ──────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    model.train()
    if name=='DualSA':
        out1, out2 = model(x)
        out = (out1+out2)/2
    else:
        out = model(x)
    torch.cuda.synchronize()
    fwd_peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"Peak forward GPU Memory: {fwd_peak_gb:.2f} Gb")
    

    # ── forward + backward peak ────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    bwd_peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    
    print(f"Peak backward GPU Memory: {bwd_peak_gb:.2f} Gb")

    del model, x, out, loss
    gc.collect()
    torch.cuda.empty_cache()

    return fwd_peak_gb, bwd_peak_gb