---
title: >-
  [Paper Note] (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models
description: >-
  [ICML 2026][Earth Science][Weather Forecasting] MOSAIC addresses two types of spectral degradation in ML weather models (spectral damping from deterministic averaging and high-frequency aliasing from coarsened latent spa…
tags:
  - "ICML 2026"
  - "Earth Science"
  - "Weather Forecasting"
  - "Sparse Attention"
  - "HEALPix mesh"
  - "Spectral Fidelity"
  - "Probabilistic Ensemble Forecasting"
date: 2026-05-08
content_hash: 76bce62cfc13bca8
---

# (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models

**Conference**: ICML 2026  
**arXiv**: [2604.16429](https://arxiv.org/abs/2604.16429)  
**Code**: https://github.com/maxxxzdn/mosaic (Available)  
**Area**: 3D Vision / Physical Modeling / Probabilistic Weather Forecasting / Sparse Attention  
**Keywords**: Weather Forecasting, Sparse Attention, HEALPix mesh, Spectral Fidelity, Probabilistic Ensemble Forecasting

## TL;DR
MOSAIC addresses two types of spectral degradation in ML weather models (spectral damping from deterministic averaging and high-frequency aliasing from coarsened latent spaces) using "probabilistic perturbation + mesh-aligned block-sparse attention on HEALPix spherical grids." At 1.5° resolution with only 214M parameters, it matches or exceeds models with 6× higher resolution, generating a 24-member, 10-day forecast in 12 seconds on a single H100.

## Background & Motivation

**Background**: Traditional Numerical Weather Prediction (NWP) produces 10-day forecasts by solving fluid dynamics equations, offering high precision but at a cubic computational cost relative to resolution. Recently, ML models (MLWP) such as GraphCast, Pangu, AIFS, GenCast, and Aurora have reduced inference time to under one minute—1,000–10,000× faster than NWP. However, they generally fail to resolve fine-scale details; frontal systems and tropical cyclones at 50–80 km scales are poorly represented, with systematic energy underestimates in the mesoscale (10–100 km) spectrum.

**Limitations of Prior Work**: The authors specifically attribute the spectral failure of existing MLWPs to two categories. The first is "spectral damping," which is statistical: deterministic models are trained to predict the "conditional expectation," which is inherently smoother than any single realization, leading to the automatic erasure of high frequencies. The second is "high-frequency aliasing," which is architectural: nearly all MLWPs use "compressive encoding"—projecting high-resolution fields into a coarsened latent space where spatial reduction exceeds channel expansion. If the Nyquist frequency of the latent grid is insufficient, non-linear activations "fold" high-frequency content back into low wavenumbers, manifesting as non-physical spectral energy "bumps" near the Nyquist frequency (clearly visible in GenCast's spectrum).

**Key Challenge**: Eliminating spectral damping requires probabilistic models to generate single realizations rather than expectations. Eliminating aliasing requires spatial mixing at native resolution rather than compression. However, standard self-attention at native 0.25° resolution is $O(N^2)$ and computationally infeasible, while linear attention sacrifices input-dependent selectivity and fails to balance long-range dependencies with computational efficiency.

**Goal**: (i) Eliminate spectral damping at the probabilistic level; (ii) Eliminate compression-induced aliasing at the architectural level; (iii) Provide global attention with $O(N)$ complexity and softmax expressivity on the native grid to enable spatial interaction "before compression."

**Key Insight**: Leveraging Tobler’s First Law of Geography—"near things are more related than distant things"—supports two engineering designs: (1) Organizing data on a HEALPix spherical mosaic so that spatially adjacent pixels are contiguous in memory; (2) Sharing key-value selections across adjacent queries to amortize the cost of sparse attention from "per-token" to "per-block."

**Core Idea**: Extend Native Sparse Attention (NSA) from 1D sequences to the sphere by constructing mesh-aligned block-sparse attention (BSA). This achieves global long-range dependency modeling at $O(N)$ cost on the HEALPix mesh. Combined with learned functional perturbations for probabilistic ensemble forecasting, it simultaneously eliminates both modes of spectral failure.

## Method

### Overall Architecture
The MOSAIC forward pass: (1) Input from latitude-longitude grids is moved to the HEALPix mesh via cross-attention interpolation (NESTED indexing ensures neighborhood contiguity); (2) Multiple BSA encoder layers run at the native resolution to capture spatial interactions; (3) The representation can then be downsampled to coarser HEALPix resolutions for processing by a transformer backbone; (4) It is decoded back to latitude-longitude; (5) Stochasticity is injected via learned functional perturbations to obtain ensemble members; (6) Autoregressive rollout generates a 10-day ensemble forecast. The key is that BSA replaces standard self-attention, making step (2) feasible at high native resolutions such as 0.46°–0.23°.

### Key Designs

1.  **HEALPix Spherical Mosaic + NESTED Indexing (Spatially adjacent = Memory adjacent)**:
    - **Function**: Solves the problem where spatial neighbors are distant in indexing on standard lat-lon grids, allowing block-based computation to use single coalesced memory reads per block.
    - **Mechanism**: HEALPix divides the sphere into 12 equal-area base pixels, each recursively subdivided. At resolution $N_{side}$, there are $12 N_{side}^2$ equal-area pixels. Adjacent pixels along a Z-order curve occupy contiguous indices (sub-pixels of pixel $p$ are indexed $4p, 4p+1, 4p+2, 4p+3$). $N_{side}\in\{32,64,128,256\}$ corresponds to 1.83°/0.92°/0.46°/0.23°. Conversion from lat-lon grids uses cross-attention interpolation: for each HEALPix target $i$, relative positions $p_{ij}$ serve as queries, and neighboring source features as key/values: $o_i = \sum_{j\in N_i}\mathrm{softmax}_j(q_{ij}^T k_j/\sqrt d) v_j$.
    - **Design Motivation**: Lat-lon grids oversample at the poles and have distant neighbor indices across rows, leading to scattered memory access and low GPU utilization. HEALPix NESTED indexing unifies "spatial proximity" $\Rightarrow$ "index proximity" $\Rightarrow$ "memory proximity," serving as the geometric prerequisite for block-sparse attention on the sphere.

2.  **Mesh-aligned Block-Sparse Attention (BSA)**:
    - **Function**: Upgrades NSA's "independent selection per token" to "shared selection per block," mapping sparse attention to physical block-to-block interactions and reducing complexity to linear on high-res grids.
    - **Mechanism**: $N$ tokens are partitioned into $m$ non-overlapping blocks $\{B_1,\dots,B_m\}$ following the HEALPix NESTED order. Block-level $\bar q_i, \bar k_j, \bar v_j$ are obtained via pooling $\phi$. BSA uses three complementary branches fused via learnable gates $g_{CG}, g_{FG}, g_L$: (a) **Compression Branch**—computes coarse-grained block-to-block attention $\bar a_{ij} = \mathrm{softmax}(\bar q_i^T \bar k_j/\sqrt{d_k})$ and broadcasts scores to all tokens within the block; (b) **Selection Branch**—each query block selects top-$n$ key blocks based on compression scores and performs full-resolution fine-grained attention within selected blocks; (c) **Local Branch**—standard sliding window attention for short-range details. Final output: $o_i = g_{CG}o_i^{CG} + g_{FG}o_i^{FG} + g_L o_i^L$.
    - **Design Motivation**: SHA/NSA works on 1D sequences because contiguous indices naturally correspond to contiguous semantics; on a sphere, this only holds if using HEALPix. Lifting "selection" from token to block assumes "spatially close queries attend to similar distant regions," which aligns with atmospheric physics and amortizes selection costs by $\approx B$ (block size).

3.  **Learned Functional Perturbation (Probabilistic component to eliminate spectral damping)**:
    - **Function**: Injects learnable stochastic perturbations into the input to generate ensemble members, ensuring each member is a physically plausible realization with realistic high-frequency details.
    - **Mechanism**: Following Alet et al. (2025), a learnable global perturbation field is added at the input layer. Different seeds yield different members; each member rolls out independently. The ensemble average (for RMSE/ACC) and single-member spectra are evaluated separately.
    - **Design Motivation**: Deterministic losses (MSE against ground truth) target the conditional expectation, which is inherently smooth. Shifting from "single point output" to "trajectory sampling" achieves spectral fidelity: each trajectory is a valid high-frequency realization. Coupled with BSA, it addresses both "statistical damping" and "geometric aliasing."

### Loss & Training
The authors follow the ArchesWeather/GenCast training paradigm: autoregressive training on ERA5 (2013-2019) with a 6-hour step. Evaluation follows the WeatherBench2 protocol on the year 2020. Objective functions are based on the probabilistic framework (see Appendix). Total parameters: 214M; input spatial resolution: 1.5°; hardware evaluation on a single H100 GPU.

## Key Experimental Results

### Main Results
MOSAIC (1.5°, 214M params) vs. various MLWPs:

| Model | Resolution | Spectral Fidelity (10m Wind 24h) | nRMSE @ 240h | Inference Time / member / step | VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pangu-Weather | 0.25° | Significant HF Underestimate | ≈ baseline | Fast | ≈ 10 GB |
| GraphCast (oper.) | 0.25° | HF Underestimate | Good | Medium | ≈ 10 GB |
| GenCast (1st, oper.) | 0.25° | Near-truth but Nyquist "bump" (aliasing) | Strong | Slow ($\approx 20\times$) | ≈ 70 GB |
| Stormer | 1.5° | HF Underestimate | baseline | Fast | Small |
| ArchesWeather-Gen | 1.5° | Near-truth | Strong | Medium | Large |
| **MOSAIC (Ours)** | **1.5°** | **Near-perfect alignment** | **Matches/exceeds 6× res models** | **≤ 12s / 24 members / 10 days** | **≈ 3 GB** |
| MOSAIC-C (Ablation) | 1.5° | Aliasing bump at Nyquist | Significant Drop | — | — |

### Ablation Study

| Configuration | Results |
| :--- | :--- |
| Full MOSAIC (BSA + Perturbation + Native Grid) | Spectrum aligns almost perfectly with ERA5; optimal nRMSE. |
| MOSAIC-C (Compress to coarse latent space first) | Energy bump at Nyquist frequency, confirming compression $\to$ aliasing. |
| Without functional perturbation (Deterministic) | Obvious spectral damping, confirming perturbation is key to solving damping. |
| Replace BSA with standard dense attention | Feasible at 1.5° but VRAM and latency skyrocket; infeasible at high res. |
| BSA without HEALPix (Lat-lon blocks) | Tokens in blocks no longer spatially adjacent; "shared selection" fails; performance drops. |

### Key Findings
- Single-member spectra align almost perfectly with ERA5 across all resolvable frequencies; this is the first 1.5° model that "looks like the real atmosphere" spectrally.
- MOSAIC-C (forced input compression) exhibits the spectral energy bump near Nyquist, consistent with 0.25° models like GenCast, proving compressive encoding is the direct cause of aliasing.
- MOSAIC at 1.5° matches or exceeds 0.25° models, challenging the "higher resolution is always better" trope: architectural correctness may outweigh data resolution.
- 12 seconds for a 24-member 10-day forecast on a single H100 makes it one of the most efficient probabilistic models; 3 GB VRAM allows inference on consumer GPUs.

## Highlights & Insights
- "Grafts" NSA—a recent sparse attention from LLMs—onto spherical physical modeling by recognizing that HEALPix satisfies NSA’s implicit assumption: "contiguous index = contiguous semantics."
- Identifies spectral damping and aliasing as two distinct causes of "spectral failure" and provides targeted cures; the MOSAIC-C "reverse ablation" is highly convincing.
- Block-sharing selection is more compatible with GPU memory access patterns than token-level selection; the design achieves hardware friendliness and geometric correctness simultaneously.
- Independently proposes block-shared sparse attention similar to concurrent video diffusion work (Gu 2026), suggesting this is a universal design principle for spatiotemporal physical data.

## Limitations & Future Work
- Data scale (2013–2019) and compute are still smaller than SOTA models like GraphCast; nRMSE on Z500 has not yet surpassed ECMWF SOTA.
- 1.5° resolution is an engineering compromise; scalability to 0.25°/0.5° native grids remains to be verified (though BSA is linear, perturbation and decoder overheads may grow).
- BSA parameters (block size, top-$n$) are manually tuned and lack adaptive versions (e.g., varying block size by latitude).
- Probabilistic perturbation is "input-level" and does not explicitly model the physical structure of process noise (e.g., stochastic parameterization of convection), which may under-sample extreme events.

## Related Work & Insights
- **vs. GenCast / GraphCast (Google DeepMind 0.25°)**: GenCast is probabilistic but uses a compressed latent space (aliasing exists); GraphCast is deterministic (damping exists). MOSAIC solves both and matches performance at 1.5° with much smaller VRAM/params.
- **vs. NSA (Yuan 2025) / Video Block-sparse (Gu 2026, Meituan 2025)**: These use block-level sparsity on 1D or regular grids. MOSAIC is the first to implement this on spherical meshes using HEALPix as an explicit geometric prerequisite.
- **vs. Subich 2025 / Bonev 2025 (Spectral Soft Constraints)**: They use loss functions to penalize spectral differences. MOSAIC solves it via architecture + probability, avoiding complex manual weighting in the loss.
- **vs. Banño-Medina 2025 / Nordhagen 2025 (Native grid message-passing)**: Consistent in native resolution processing but uses MPNN, which lacks the long-range dependencies of transformers; MOSAIC provides a significant expressivity boost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify spherical meshes, native sparse attention, HEALPix memory contiguity, and probabilistic perturbations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evidence across spectral analysis, nRMSE/ACC, efficiency, and causality-focused ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear diagnosis of "two failure modes" with visual evidence in Fig. 2.
- Value: ⭐⭐⭐⭐⭐ High-fidelity spectral results at 1.5° with extreme efficiency (12s per forecast) provide a practical path forward for both academia and operations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](../../CVPR2026/earth_science/geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)
- [\[AAAI 2026\] MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics](../../AAAI2026/earth_science/mdaif_robust_one-stop_multi-degradation-aware_image_fusion_with_language-driven_.md)
- [\[AAAI 2026\] RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways](../../AAAI2026/earth_science/renew_risk-_and_energy-aware_navigation_in_dynamic_waterways.md)
- [\[NeurIPS 2025\] Power Ensemble Aggregation for Improved Extreme Event AI Prediction](../../NeurIPS2025/earth_science/power_ensemble_aggregation_for_improved_extreme_event_ai_prediction.md)
- [\[NeurIPS 2025\] A Probabilistic U-Net Approach to Downscaling Climate Simulations](../../NeurIPS2025/earth_science/a_probabilistic_unet_approach_to_downscaling_climate_simulat.md)

</div>

<!-- RELATED:END -->
