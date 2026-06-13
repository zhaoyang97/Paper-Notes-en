---
title: >-
  [Paper Note] TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization
description: >-
  [ICML 2026][3D Vision][3DGS] TideGS moves the 3DGS parameter table to SSD, virtualizing it into "blocks" while using GPU VRAM as a cache for the frustum-visible working set. Combined with a three-tier asynchronous pipeli…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "3DGS"
  - "out-of-core"
  - "SSD-CPU-GPU three-tier storage"
  - "visibility sparsity"
  - "trajectory differential streaming"
date: 2026-05-08
content_hash: 3946e6f366867a80
---

# TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.20150](https://arxiv.org/abs/2605.20150)  
**Code**: To be confirmed  
**Area**: 3D Vision / 3D Gaussian Splatting / System Optimization  
**Keywords**: 3DGS, out-of-core, SSD-CPU-GPU three-tier storage, visibility sparsity, trajectory differential streaming  

## TL;DR
TideGS moves the 3DGS parameter table to SSD, virtualizing it into "blocks" while using GPU VRAM as a cache for the frustum-visible working set. Combined with a three-tier asynchronous pipeline and trajectory-adaptive differential streaming, it scales the number of trainable Gaussians from approximately 11M (Native 3DGS) / 105M (CLM) to **over 1 billion** on a single 24 GB GPU, achieving large-scene reconstruction quality superior to all evaluated single-card baselines.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the mainstream explicit representation for novel view synthesis, where each Gaussian point carries a set of learnable parameters, and real-time rasterization is achieved via splatting. Unlike implicit representations like NeRF, 3DGS places model capacity directly into a "primitive table"—the more Gaussians, the finer the reconstruction, but at the cost of significantly higher VRAM pressure.

**Limitations of Prior Work**: Under a standard SH-degree-3 configuration, each Gaussian has $D=59$ fp32 parameters. Including gradients and Adam first/second moments, the storage requirement is roughly 8x. A scene with 100 million Gaussians requires ~90 GB, far exceeding the 24 GB limit of consumer-grade cards. Empirically, native 3DGS is limited to ~11.5M Gaussians on a 24 GB card, ZeRO-Offload style Naive Offload hits a ceiling at ~50M (as it still moves all parameters to the GPU per step), and even CLM (which offloads SH coefficients to CPU) only reaches ~105M before the radix-sort buffer exhausts VRAM.

**Key Challenge**: 3DGS binds model capacity to GPU memory, but **the number of Gaussians truly accessed in a single iteration is minimal**. In city-scale scenes like MatrixCity BigCity, an average single view activates only 0.39% of Gaussians (worst-case 1.06%), showing extreme visibility sparsity. Furthermore, adjacent camera frustums overlap significantly, exhibiting strong temporal locality in the active set. Thus, keeping all parameters permanently in VRAM is highly inefficient.

**Goal**: (i) Transition from "VRAM-resident parameters" to "VRAM as a cache for the current working set"; (ii) Extend the storage hierarchy from GPU-CPU to include SSD, enabling billion-scale training on a single consumer GPU; (iii) Maintain forward/backward semantics and reconstruction quality identical to native 3DGS.

**Key Insight**: Treat 3DGS training as **sparse embedding-table training**—streaming only the "rows used in the current batch" into the GPU while keeping the rest on CPU/SSD. Since SSD bandwidth is low and latency is high, a naive offload would fail; this necessitates a triad of "block alignment + asynchronous pipelining + differential streaming."

**Core Idea**: Use blocks as unified units for storage, caching, and transmission, leveraging Morton coding to preserve spatial locality. The CPU performs coarse-grained frustum culling to determine the GPU working set. **Trajectory-adaptive differential streaming** is then used to transmit only the increment $\mathcal{S}_t^+ = \mathcal{R}_{t+1} \setminus \mathcal{R}_t$ between adjacent iterations, allowing cross-tier traffic to scale with "working set changes" rather than "model size."

## Method

### Overall Architecture
TideGS is an out-of-core 3DGS training system built on a three-tier SSD–CPU–GPU storage architecture. The complete parameter table $\Theta \in \mathbb{R}^{N \times D}$ ($D=59$) resides on SSD in **blocks** (log-structured append-only segments). The CPU DRAM maintains an LRU warm cache with dirty bits, while GPU VRAM holds only the **capacity-bounded resident set** $\mathcal{R}_t$ required for the current iteration. Each training step consists of four stages: (1) Block-level frustum culling on CPU to identify the candidate working set $\mathcal{K}_t$; (2) Asynchronous prefetch and H2D transfer of missing blocks to VRAM; (3) Standard 3DGS forward/backward on GPU; (4) Asynchronous D2H transfer of evicted dirty blocks back to the CPU cache, with bulk writes to SSD when the CPU cache is full. Stages (1), (2), and (4) overlap with (3) using independent CUDA streams and I/O threads to hide SSD/PCIe latency.

### Key Designs

1.  **Block virtualization & 2-stage visibility**:
    - **Function**: Converts random "per-Gaussian" accesses into sequential "per-block" I/O and filters out most invisible blocks on the CPU to avoid redundant SSD/PCIe traffic.
    - **Mechanism**: Gaussians are sorted by Morton codes of their centers and partitioned into contiguous blocks of size $B=4096$. A single block load is $4096 \times 59 \times 4$ B ≈ 944 KiB, aligning naturally with filesystem/page cache granularities. Each block is summarized by a bounding sphere $(\mathbf{c}_k, r_k)$. The first stage performs 6-plane frustum-sphere culling on the CPU for the camera batch $\mathcal{B}_t$: if $d < -r_k$, the block is discarded, leaving the conservative visible set $\mathcal{K}_t = \bigcup_{c \in \mathcal{B}_t}\{k \mid \mathrm{visible}(k, c)\}$. The second stage performs standard fine-grained 3DGS culling and rasterization on the GPU, resulting in the contribution set $\mathcal{I}_t \subseteq \bigcup_{k \in \mathcal{R}_t \cap \mathcal{K}_t} \mathrm{Block}(k)$. Block assignments are fixed after initialization; Gaussian position updates only refresh block bounds, ensuring batch I/O **without altering native 3DGS rendering semantics**.
    - **Design Motivation**: SSDs perform poorly with fragmented random reads; large block alignment is required to saturate the 3.3 GB/s bandwidth. Filtering must occur **before** data transport, or PCIe traffic would still scale linearly with $N$.

2.  **Three-tier SSD–CPU–GPU async engine + log-structured writeback**:
    - **Function**: Maintains high GPU utilization despite high-latency SSD backends and converts frequent small parameter updates into sequential bulk writes to avoid SSD write amplification.
    - **Mechanism**: SSDs use append-only segments—the initial model is written as an immutable base segment, while updated blocks are appended to patch segments. An index table $\mathrm{Index}[k] = (\mathrm{file\_id}, \mathrm{offset}, \mathrm{size}, \mathrm{version})$ points to the latest version, avoiding in-place overwrites. The CPU cache uses LRU and dirty bits; dirty blocks are only written to SSD upon eviction or checkpoints. The eviction path is VRAM → CPU → SSD. Execution is distributed across dedicated I/O threads and independent CUDA streams, using double-buffered slots to prefetch $\mathcal{S}_t^+$ for the next iteration while the current one computes.
    - **Design Motivation**: Naive offloading stalls on serial I/O. Log-structured writing is a classic SSD optimization that maintains peak write bandwidth during frequent parameter updates. The two-step writeback allows hot blocks in VRAM to be modified repeatedly in the warm CPU cache without touching the disk.

3.  **Tide: Trajectory-adaptive differential streaming**:
    - **Function**: Reuses resident blocks across iterations within a VRAM budget $C$, scaling PCIe/SSD traffic with the **resident set change** rather than the candidate working set size.
    - **Mechanism**: Camera sequences are sorted via TSP-clustering to increase working set overlap between adjacent batches. In each iteration, the next resident set $\mathcal{R}_{t+1}$ is selected from $\mathcal{C}_t = \mathcal{R}_t \cup \mathcal{K}_{t+1}$ based on a score $s(k) = \lambda \cdot \mathbf{1}[k \in \mathcal{K}_{t+1}] + (1-\lambda) \cdot \mathrm{Recency}(k)$. If $|\mathcal{K}_{t+1}| > C$, a camera-balanced Top-$C$ selection ensures view coverage. Only the difference $\mathcal{S}_t^+ = \mathcal{R}_{t+1} \setminus \mathcal{R}_t$ is transported, $\mathcal{S}_t^-$ is evicted, and the intersection $\Omega_t^R$ is retained. Optimizer states are only instantiated for resident blocks; evicted states are discarded and cold-restarted if re-entering.
    - **Design Motivation**: Re-transmitting $\mathcal{K}_t$ at each step still incurs hundreds of megabytes of PCIe traffic in city-scale scenes. With smooth camera trajectories, $|\mathcal{S}_t^+|$ is usually much smaller than $|\mathcal{K}_t|$, reducing traffic by another order of magnitude.

### Loss & Training
The system uses the standard 3DGS photometric loss, SH-degree-3 representation, and Adam optimizer. Block size $B = 4096$. Training runs for 30k steps on Mip-NeRF 360 with batch=4; MatrixCity defaults to batch=64 with a 16 GB CPU cache. Morton sorting and initial base segment writing are one-time pre-processing steps: 102M Gaussians take 1.9 minutes, 1.1B Gaussians take 21.2 minutes (<0.5% of total training time).

## Key Experimental Results

### Main Results

Scalability limits (Maximum trainable scale on a single 24 GB GPU):

| Method | VRAM Complexity | Limiting Factor | $N_{\max}$ |
| :--- | :--- | :--- | :--- |
| Native 3DGS | $O(N)$ | Resident state | ~11.5M |
| Naive Offload | $O(N)$ | Per-step params (SH) | ~50M |
| CLM | $O(N)$ | Rasterization buffer | ~105M |
| **TideGS (Ours)** | $O(\|\mathcal{R}_t\|)$ / $O(\|\mathcal{I}_t\|)$ | Resident budget / SSD | **>1B** |

MatrixCity Throughput and Cross-tier Traffic:

| Method | Scale $N$ | Backend | PCIe (GB/iter) ↓ | GPU Util (%) ↑ | Iter (ms) ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Naive Offload | ~102M | DRAM | — OOM — | — | — |
| CLM | ~102M | DRAM | 0.41 | 37.0 | 100.8 |
| **TideGS** | ~102M | NVMe SSD | **0.10** | 43.3 | **90.7** |
| CLM | ~1.1B | DRAM | — OOM — | — | — |
| **TideGS** | ~1.1B | NVMe SSD | 0.97 | 49.5 | 525.6 |

Quality Alignment (Mip-NeRF 360): TideGS achieved 28.92 dB PSNR vs. Native 3DGS 29.03 dB (a difference of only 0.11 dB), proving virtualization does not compromise the optimization objective. On MatrixCity, TideGS at ~1.1B Gaussians reached **26.1 dB**, whereas CLM OOM'd at 25.0 dB (~102M), showing that higher Gaussian counts yield tangible reconstruction gains.

### Ablation Study

| Configuration | Iter (ms) ↓ | PCIe (GB/iter) ↓ | CPU Cache Hit (%) ↑ |
| :--- | :--- | :--- | :--- |
| Full TideGS | **90.7** | **0.10** | **95.2** |
| w/o Tide (Differential) | 145.3 | 0.85 | 95.2 |
| w/o Overlap (Async) | 210.5 | 0.10 | 95.2 |
| w/o Morton (Locality) | 115.8 | 0.45 | 42.1 |

### Key Findings
- **Differential streaming handles the "Traffic Axis"**: Disabling Tide increases PCIe traffic by 8.5x and nearly doubles iteration time.
- **Asynchronous pipelining handles the "Latency Axis"**: Without overlapping, iteration time jumps from 90.7 ms to 210.5 ms, dominated by SSD/PCIe latency.
- **Morton sorting is the lifeblood of the CPU cache**: Random block layouts cause the CPU cache hit rate to drop from 95.2% to 42.1%.
- **Negligible impact on quality**: Ablations show that these system-level optimizations do not significantly affect PSNR/SSIM/LPIPS, as they do not alter the visible Gaussian set or the objective.

## Highlights & Insights
- Viewing 3DGS training as **sparse embedding-table training** is an excellent analogy, allowing mature mechanisms like LRU working-set caches and log-structured writes to be mapped directly to 3D spatial blocks.
- **TSP-clustering** for camera trajectories is a clever "systems-first" trade-off; it sacrifices i.i.d. sampling for an 8.5x reduction in PCIe traffic.
- **Two-stage visibility filtering** serves as the primary leverage for scaling from 100M to 1 billion Gaussians, as it decouples I/O from the total model size $N$. This can be generalized to any scene with sparse access patterns.
- The choice to **cold-restart optimizer states** significantly reduces the VRAM footprint and cross-tier traffic, with minimal impact on convergence due to trajectory-based block retention.

## Limitations & Future Work
- **Dependency on Trajectory Smoothness**: The benefits of differential streaming rely on working-set overlap. Highly disordered views or random sampling tasks would neutralize the traffic advantages.
- **Long-term Impact of Cold-Restarting**: While cumulative statistics are provided, the potential for local "texture flickering" or slow convergence in certain regions was not deeply analyzed.
- **Fixed Block Assignment**: If Gaussians drift significantly, bounding spheres may expand, leading to conservative culling. Periodic re-blocking might be necessary.
- **Hardware Dependency**: The system assumes high-performance NVMe SSDs; performance on lower-end QLC or SATA drives remains unquantified.

## Related Work & Insights
- **vs. CLM**: TideGS further offloads the entire parameter table to SSD and uses a capacity-bounded resident set, whereas CLM is limited by the rasterization buffer size and GPU-resident geometry.
- **vs. Naive Offload**: TideGS recognizes the sparsity and trajectory dependence of 3DGS parameter access, which the ZeRO-style offloading used in LLMs fails to exploit.
- **vs. Multi-GPU 3DGS**: While multi-GPU systems scale via horizontal VRAM expansion, TideGS provides vertical storage expansion, making billion-scale modeling accessible on a single consumer-grade card.

## Rating
- Novelty: ⭐⭐⭐⭐ (Solid application of out-of-core system principles to 3DGS).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive metrics across system throughput and reconstruction quality).
- Writing Quality: ⭐⭐⭐⭐ (Clear logic and well-supported design decisions).
- Value: ⭐⭐⭐⭐⭐ (Significantly lowers the hardware barrier for large-scale 3DGS research).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](../../CVPR2026/3d_vision/fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[ICCV 2025\] A Unified Interpretation of Training-Time Out-of-Distribution Detection](../../ICCV2025/3d_vision/a_unified_interpretation_of_training-time_out-of-distribution_detection.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](../../CVPR2026/3d_vision/off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](../../CVPR2026/3d_vision/ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [\[CVPR 2026\] 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds](../../CVPR2026/3d_vision/3d_sans_3d_scans_scalable_pre-training_from_video-generated_point_clouds.md)

</div>

<!-- RELATED:END -->
