---
title: >-
  [Paper Note] (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models
description: >-
  [ICML 2026][Scientific Computing][Weather Forecasting] MOSAIC addresses two types of spectral degradation in ML-based weather forecasting models—spectral damping from deterministic averaging and high-frequency aliasing f…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Weather Forecasting"
  - "Sparse Attention"
  - "HEALPix mesh"
  - "Spectral Fidelity"
  - "Probabilistic Ensemble Forecast"
date: 2026-05-08
content_hash: 2d499fc2090c0f3f
---

# (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models

**Conference**: ICML 2026  
**arXiv**: [2604.16429](https://arxiv.org/abs/2604.16429)  
**Code**: https://github.com/maxxxzdn/mosaic (available)  
**Area**: 3D Vision / Physical Modeling / Probabilistic Weather Forecasting / Sparse Attention  
**Keywords**: Weather Forecasting, Sparse Attention, HEALPix mesh, Spectral Fidelity, Probabilistic Ensemble Forecast

## TL;DR
MOSAIC addresses two types of spectral degradation in ML-based weather forecasting models—spectral damping from deterministic averaging and high-frequency aliasing from latent space compression—by combining probabilistic perturbation with mesh-aligned block-sparse attention on the HEALPix spherical mesh. With only 214M parameters at 1.5° resolution, it matches or surpasses models at 6× higher resolution, generating 24-member 10-day forecasts in 12 seconds on a single H100.

## Background & Motivation

**Background**: Traditional numerical weather prediction (NWP) solves fluid dynamics equations for 10-day forecasts, achieving high accuracy but with computational cost scaling cubically with resolution. In the past three years, ML models such as GraphCast, Pangu, AIFS, GenCast, and Aurora have reduced inference time to under a minute—1000–10000× faster than NWP. However, these models struggle at fine scales: features like 50–80 km fronts and tropical cyclones are poorly reproduced, and mesoscale (10–100 km) spectral energy is systematically underestimated.

**Limitations of Prior Work**: The authors categorize spectral failures in MLWP into two types. The first, "spectral damping," is statistical: deterministic models are trained to predict conditional expectations, which are inherently smoother than any single realization, thus suppressing high frequencies. The second, "high-frequency aliasing," is architectural: MLWP models typically use "compression encoding"—projecting high-resolution meteorological fields into a latent space with much lower spatial than channel resolution, where most computation occurs. If the latent grid's Nyquist frequency is too low, nonlinear activations fold high-frequency content back to low wavenumbers, manifesting as spurious spectral energy near Nyquist upon decoding (clearly visible in GenCast spectra).

**Key Challenge**: Eliminating spectral damping requires probabilistic models generating single realizations rather than expectations; eliminating aliasing requires spatial mixing at native resolution, not after compression. However, standard self-attention at native 0.25° resolution is $O(N^2)$ and infeasible—linear attention sacrifices input-dependent selectivity, making it hard to balance long-range dependencies and computational cost.

**Goal**: (i) Eliminate spectral damping probabilistically; (ii) eliminate compression-induced aliasing architecturally; (iii) provide $O(N)$ complexity, softmax-expressive global attention on the native grid, enabling spatial interactions before compression.

**Key Insight**: Tobler's First Law of Geography—"near things are more related than distant things"—naturally supports two engineering choices: (1) placing data on the HEALPix spherical mesh so spatial neighbors are contiguous in memory; (2) having neighboring queries share key-value selection, replacing "per-token KV selection" with "per-block KV selection," amortizing sparse attention selection cost over blocks.

**Core Idea**: Extend Native Sparse Attention (NSA) from 1D sequences to the sphere, constructing mesh-aligned block-sparse attention (BSA) on the HEALPix mesh to model global long-range dependencies at $O(N)$ cost, and combine with learned functional perturbation for probabilistic ensemble forecasting, jointly eliminating both spectral failure modes.

## Method

### Overall Architecture
MOSAIC's forward process: (1) Interpolate latitude-longitude grid inputs to the HEALPix mesh via cross-attention (NESTED indexing ensures neighborhood continuity); (2) Run several BSA encoder layers at native resolution to capture spatial interactions; (3) Optionally downsample to coarser HEALPix resolution for transformer backbone processing; (4) Decode back to latitude-longitude; (5) Inject randomness via learned functional perturbation to obtain ensemble members; (6) Autoregressive rollout for 10-day ensemble forecasts. The key is that BSA replaces standard transformer self-attention, making step (2) feasible at high native resolutions (0.46°–0.23°).

### Key Designs

1. **HEALPix Spherical Tessellation + NESTED Indexing (Spatial Neighbors = Memory Neighbors)**:

    - Function: Addresses the issue that spatial neighbors on a standard lat-lon grid are far apart in memory, enabling block-based computation to merge each block with a single memory read.
    - Mechanism: HEALPix divides the sphere into 12 equal-area base pixels, recursively subdivided; at $N_{side}$ resolution, there are $12 N_{side}^2$ equal-area pixels, with adjacent pixels occupying contiguous indices along a Z-order curve (pixel $p$'s four children are $4p, 4p{+}1, 4p{+}2, 4p{+}3$). $N_{side}\in\{32,64,128,256\}$ corresponds to 1.83°/0.92°/0.46°/0.23°. Cross-attention interpolation maps lat-lon to HEALPix: for each HEALPix target $i$, relative position $p_{ij}$ is the query, neighboring source features are key/value, and $o_i = \sum_{j\in N_i}\mathrm{softmax}_j(q_{ij}^T k_j/\sqrt d) v_j$.
    - Design Motivation: Standard lat-lon grids oversample at the poles and have distant neighbor indices; block data loading requires scattered memory access, reducing GPU efficiency. HEALPix NESTED indexing unifies "spatially adjacent" with "index adjacent" and "memory adjacent," a geometric prerequisite for block-sparse attention on the sphere, directly enabling BSA.

2. **Mesh-aligned Block-Sparse Attention (BSA)**:

    - Function: Upgrades NSA's "per-token KV selection" to "per-block shared selection," mapping sparse attention to physical block-to-block interactions and reducing complexity to linear on high-dimensional native grids.
    - Mechanism: Partition $N$ tokens in HEALPix NESTED order into $m$ non-overlapping blocks $\{B_1,\dots,B_m\}$; pool within each block via $\phi$ to obtain block-level $\bar q_i, \bar k_j, \bar v_j$. BSA has three complementary branches fused by learnable gates $g_{CG},g_{FG},g_L$: (a) **Coarse-grained branch**—compute block-level attention $\bar a_{ij} = \mathrm{softmax}(\bar q_i^T \bar k_j/\sqrt{d_k})$, broadcasting scores to all tokens in the block; (b) **Fine-grained selection branch**—each query block selects top-$n$ key blocks by coarse scores, then performs full-resolution attention within selected blocks, capturing long-range fine interactions; (c) **Local branch**—standard attention within a sliding window for short-range details. Final output: $o_i = g_{CG}o_i^{CG} + g_{FG}o_i^{FG} + g_L o_i^L$.
    - Design Motivation: NSA is effective on 1D sequences because contiguous indices naturally correspond to contiguous semantics; on the sphere, this only holds with HEALPix. Elevating selection from token to block enforces that geographically close queries attend to the same distant regions, aligning with atmospheric physics and amortizing selection cost by $\approx B$ (block size), leveraging both geometric priors and hardware efficiency.

3. **Learned Functional Perturbation (Probabilistic Component to Eliminate Spectral Damping)**:

    - Function: Injects learnable random perturbations into the input to generate an ensemble of members, each a plausible realization with authentic high-frequency detail, restoring spectral truth.
    - Mechanism: Following Alet et al. 2025, add a learnable global perturbation field at the input layer; different seeds yield different members. Each member is rolled out independently; ensemble-mean RMSE/ACC and single-member spectra are evaluated separately. Ensemble member spectra nearly coincide with ERA5 ground truth at all resolutions (see Fig. 2a), while deterministic models systematically underestimate high frequencies.
    - Design Motivation: Deterministic models minimize MSE against the expectation, which is inherently smooth; post-processing cannot recover true high-frequency energy. Switching from "single output" to "sampled trajectories" directly achieves spectral fidelity: each trajectory is a plausible high-frequency realization, and the ensemble mean naturally reflects correct uncertainty. Combined with BSA's native-resolution processing, both "statistical perturbation" and "geometric fidelity" are addressed concurrently.

### Loss & Training
The training paradigm follows ArchesWeather/GenCast: autoregressive training on ERA5 (2013–2019) with 6-hour steps; evaluation on 2020, following WeatherBench2 protocol; 24-member, 10-day forecasts; probabilistic objectives (see Appendix). The model has 214M parameters, 1.5° input resolution, and is benchmarked on a single H100 GPU.

## Key Experimental Results

### Main Results
MOSAIC at 1.5° resolution, 214M parameters vs. 1.5° / 0.25° MLWP models:

| Model | Resolution | Spectral Fidelity (10 m wind 24h spectrum ratio) | nRMSE @ 240h | Inference Time / Member / Step | Memory |
|-------|------------|-----------------------------------------------|--------------|-------------------------------|--------|
| Pangu-Weather | 0.25° | Significant high-frequency underestimation | ≈ baseline | Fast | ≈ 10 GB |
| GraphCast (oper.) | 0.25° | High-frequency underestimation | Good | Moderate | ≈ 10 GB |
| GenCast (1st, oper.) | 0.25° | Near ground truth but Nyquist spike (aliasing) | Strong | Slow ($\approx 20\times$) | ≈ 70 GB |
| Stormer | 1.5° | High-frequency underestimation | Baseline | Fast | Small |
| ArchesWeather-Gen | 1.5° | Near ground truth | Strong | Moderate | Large |
| **MOSAIC (Ours)** | **1.5°** | **Nearly perfect alignment** | **Matches/exceeds 6× resolution models** | **≤ 12s / 24 members / 10 days** | **≈ 3 GB** |
| MOSAIC-C (compression ablation) | 1.5° | Nyquist spike (aliasing) | Significant drop | — | — |

### Ablation Study

| Configuration | Result |
|---------------|--------|
| Full MOSAIC (BSA + perturbation + native grid) | Spectrum nearly identical to ERA5; optimal nRMSE |
| MOSAIC-C (compression to coarse latent) | Nyquist spectral spike, confirming compression→aliasing causality |
| Remove functional perturbation (deterministic) | Clear spectral damping, confirming probabilistic perturbation is key |
| Replace BSA with dense attention | Feasible at 1.5°, but memory/latency greatly increased; infeasible at higher resolutions |
| BSA without HEALPix (lat-lon block) | Block tokens no longer geographically adjacent, block-shared selection fails, performance drops |

### Key Findings
- Spectra of probabilistic members nearly perfectly align with ERA5 at all resolvable frequencies, while deterministic models systematically underestimate high frequencies; this is the first 1.5° model to "look like real atmosphere" spectrally.
- MOSAIC-C (forced compression) shows Nyquist spectral spikes, matching GenCast and other 0.25° models, confirming compression encoding as the direct cause of aliasing—guiding future MLWP architecture choices.
- MOSAIC at 1.5° matches or exceeds 0.25° models, challenging the "higher resolution is always better" assumption: architectural correctness may outweigh data resolution.
- Single H100 card produces 24-member 10-day forecasts in 12 seconds, making it one of the most efficient probabilistic weather models, with 3 GB memory enabling inference even on consumer GPUs.

## Highlights & Insights
- Adapting NSA, a recent sparse attention method from language models, to spherical physical modeling hinges on recognizing NSA's implicit assumption—"contiguous indices = contiguous semantics"—which HEALPix satisfies. This paradigm of identifying implicit assumptions in existing methods and finding analogous geometry in another domain is noteworthy.
- Spectral damping and aliasing are two causes of the same "spectral failure" phenomenon; the authors are the first to clearly distinguish and address both. The MOSAIC-C reverse ablation (deliberate compression) is highly convincing.
- Block-sharing selection better matches GPU memory access patterns, achieving both hardware efficiency and geometric correctness—this "hardware-first, geometry-prior" design is valuable for efficient transformer design.
- Independently proposed in concurrent video diffusion work (Gu 2026, Meituan 2025), block-shared sparse attention is shown here to be a general design principle for physical data with spatial/temporal continuity.

## Limitations & Future Work
- Training data (2013–2019) and compute are still smaller than SOTA (e.g., GraphCast); the paper acknowledges that Z500 performance still lags ECMWF SOTA.
- The 1.5° input resolution is an engineering compromise; scalability at 0.25°/0.5° remains to be validated (BSA is theoretically linear, but functional perturbation and decoder costs need reassessment).
- BSA block size, top-$n$ selection, and window size are hand-tuned hyperparameters, lacking adaptive versions as in NSA; allowing block size to follow physical spatiotemporal scales (e.g., wider ITCZ) may further improve results.
- Probabilistic perturbation is "input-level" and does not explicitly model process noise (e.g., convective-scale stochastic parameterization), possibly under-sampling extreme events.

## Related Work & Insights
- **vs GenCast / GraphCast (Google DeepMind 0.25°)**: GenCast is probabilistic but still uses compressed latent space, so spectra still show aliasing; GraphCast is deterministic, with spectral damping. MOSAIC solves both, matching or exceeding their performance at 1.5° with far fewer parameters/memory.
- **vs NSA (Yuan 2025) / Block-shared sparse attention for video (Gu 2026, Meituan 2025) / Long-context LLM (Wang 2026)**: These methods use block-level sparsity on 1D sequences or regular grids; MOSAIC is the first to robustly apply this to spherical meshes, explicitly incorporating HEALPix as a geometric prerequisite.
- **vs Subich 2025 / Bonev 2025 (spectral loss soft constraints)**: They restore effective resolution by penalizing spectral differences in the loss; MOSAIC addresses both issues via architecture and probabilistic modeling, avoiding heavy manual loss weighting.
- **vs Banño-Medina 2025 / Nordhagen 2025 (native grid message-passing)**: Similar idea (native resolution processing) but using MPNN, lacking transformer's long-range dependencies; MOSAIC replaces this with attention, significantly boosting expressiveness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify spherical physical modeling, native sparse attention, HEALPix memory continuity, and probabilistic perturbation in one framework; highly innovative in both geometry and hardware alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evidence across spectral analysis, nRMSE/ACC, inference time/memory, and ablation (MOSAIC-C), with direct comparison to 0.25° models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear diagnosis and targeted solutions for "two failure modes," with Fig.2 (a/b/c) visually presenting both spectral failure symptoms convincingly.
- Value: ⭐⭐⭐⭐⭐ 1.5° model delivers 24-member 10-day forecasts in 12s with realistic spectra, offering practical advances for both academic and operational weather forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting](../../NeurIPS2025/3d_vision/mesh_interpolation_graph_network_for_dynamic_and_spatially_irregular_global_weat.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)
- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](../../CVPR2026/3d_vision/speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [\[ICML 2026\] Smoothness Errors in Dynamics Models and How to Avoid Them](smoothness_errors_in_dynamics_models_and_how_to_avoid_them.md)
- [\[ICML 2026\] SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion](splattn_bridging_2d_and_3d_with_gaussian_soft_splatting_and_attention_for_point_.md)

</div>

<!-- RELATED:END -->
