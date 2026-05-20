---
title: >-
  [Paper Note] ANTIC: Adaptive Neural Temporal In-situ Compressor
description: >-
  [ICML 2026][Scientific Computing][Online Compression] To enable "on-the-fly" compression of PB-EB scale PDE simulation data, this work proposes ANTIC: a physics-aware temporal selector retains only physically important s…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Online Compression"
  - "Neural Fields"
  - "Continual Fine-tuning"
  - "LoRA"
  - "PDE Simulation"
date: 2026-05-08
content_hash: fae110db21d6cc53
---

# ANTIC: Adaptive Neural Temporal In-situ Compressor

**Conference**: ICML 2026  
**arXiv**: [2604.09543](https://arxiv.org/abs/2604.09543)  
**Code**: https://github.com/AndreiB137/ANTIC  
**Area**: Scientific Computing / Neural Compression / Neural Fields  
**Keywords**: Online Compression, Neural Fields, Continual Fine-tuning, LoRA, PDE Simulation

## TL;DR
To enable "on-the-fly" compression of PB-EB scale PDE simulation data, this work proposes ANTIC: a physics-aware temporal selector retains only physically important snapshots, and a neural field with LoRA continually fine-tunes to encode residuals between adjacent snapshots. Achieves 435× compression on 2D Kolmogorov flow and 6807× spatiotemporal compression on a 4.2 TiB 3D binary black hole merger simulation.

## Background & Motivation
**Background**: High-resolution transient simulations in CFD, MHD, plasma physics, and numerical relativity often generate single trajectories of several TB to hundreds of TB. Traditional approaches output raw simulation data and perform post-hoc compression (JPEG2000, DWT, FPZIP, ZFP, etc.), with both codec-based and low-rank tensor decomposition methods used for spatial dimensions.

**Limitations of Prior Work**: (1) Offline compression is infeasible for petascale/exascale simulations—there is simply not enough disk space to store raw data first. (2) Existing in-situ compression either samples time uniformly (missing transient events or oversampling slow periods) or uses fixed spatial representations (autoencoder latents with fixed resolution, traditional codecs struggle with multiscale correlations). (3) Most methods are not "physics-aware"—they neither identify important snapshots nor exploit continuity between adjacent snapshots.

**Key Challenge**: Stiff/multi-rate PDEs exhibit both temporal multiscale (coexistence of fast and slow phase transitions) and spatial multiscale (nonlinearity, nonstationarity). Single time sampling strategies or spatial representations cannot simultaneously balance storage, accuracy, and throughput—necessitating a spatiotemporally joint, physics-aware in-situ framework.

**Goal**: (1) Design a parameter-free, PDE-metric-injectable snapshot selector along the time axis; (2) Use neural field representation with continual fine-tuning of residuals between adjacent snapshots along the spatial axis; (3) Merge into a single streaming in-situ pipeline, exposing the rate-distortion Pareto frontier for user selection.

**Key Insight**: The authors observe that stiff PDE solutions between adjacent time steps are "mainly smooth, small perturbations," so "compressing the $t+\Delta t$ snapshot" can be reframed as "performing a low-rank residual update to a neural field already fit to the $t$ snapshot"—naturally suited for LoRA. Physical quantities (vorticity, Weyl scalar) serve as cheap saliency indicators to instantly judge whether the system is steady or undergoing a phase transition.

**Core Idea**: Combine a Physics-aware Temporal Selector (PATS) with continual fine-tuning of neural fields (CFT / CFT+LoRA) to achieve simultaneous online temporal and spatial compression, exposing the rate-distortion Pareto to users.

## Method
ANTIC consists of two asynchronous modules: (i) PATS decides "whether to compress this frame," (ii) Spatial Neural Compression decides "how to compress." The entire pipeline is a single streaming pass, with no need to write the raw trajectory to disk.

### Overall Architecture
- **Streaming Input**: The simulator outputs snapshots $u(t)$ at each time step.
- **PATS Sub-pipeline**: The Metric extracts physics-of-interest $\phi_t$ (e.g., enstrophy of vorticity / modulus of Weyl scalar) from each snapshot. The Regulator dynamically adjusts the Queue window size $W$ based on $\phi_t$. The Gate uses the current truncated context and $\phi_t$ to form a dynamic threshold to decide whether to accept the snapshot.
- **Spatial Neural Compression**: Selected snapshots are encoded by continual fine-tuning (CFT) of the existing neural field $W_t \to W_t + \Delta W_{\Delta t}$. $\Delta W_{\Delta t}$ can be full fine-tune (more accurate, higher memory) or low-rank $\mathbf{A}^{(\Delta t)}\mathbf{B}^{(\Delta t)}$ (more efficient, slight accuracy loss), allowing users to slide along the Pareto frontier.
- **Output**: Only the sparse sequence of neural field weights is stored on disk. For decompression, each segment's weights are incrementally added to the base network, and querying coordinates reconstructs the field at any time.

### Key Designs

1. **Physics-aware Temporal Selector (PATS)**:

    - **Function**: Online determination of whether each new snapshot contains sufficient "new physical information" to warrant storage, skipping redundant slow-evolving phases and retaining rapid transients.
    - **Mechanism**: Four-component parameter-free architecture. The Metric is a PDE-specific scalar—enstrophy $\mathcal{E}(t) = \frac{1}{2}\int_\Omega \|\omega\|^2 dA$ for turbulent flows, Weyl scalar $\Psi_4(t,\mathbf{r})$ for binary black hole mergers (outgoing gravitational wave strength). The Queue is a sliding window storing the most recent $W$ metric values. The Regulator detects phase transitions, truncates the Queue, and resets the reference anchor. The Gate forms a dynamic threshold from the truncated context—deciding whether to compress the current frame and feeding back to the Regulator to adjust the next window size. The entire mechanism requires no trainable parameters; all "intelligence" comes from PDE-specific physical quantities.
    - **Design Motivation**: Traditional time sampling is either fixed-interval (misses fast changes) or uses generic heuristics (e.g., inter-frame pixel difference, physically meaningless). Using intrinsic PDE invariants/excitation indicators as saliency enables adaptivity for stiff/multi-rate systems—sparse sampling during slow evolution, dense sampling during transients, and only the Metric function needs to be changed for different PDEs.

2. **Neural Field + Continual Fine-tuning of Residuals between Adjacent Snapshots**:

    - **Function**: Use a coordinate-based MLP to compress each snapshot into $\sim 0.4$M parameters; only the difference between adjacent snapshots is learned, requiring a small-scale fine-tune.
    - **Mechanism**: Reframe spatial compression as "performing a residual update to the neural field already fit to $u(t)$ to fit $u(t+\Delta t)$." Due to the smoothness of PDE solutions $u(t+\Delta t) - u(t) \approx \Delta u(t)$, the residual is much smaller than the original field, so only a few gradient steps and parameter updates are needed for convergence. The architecture is a $256\times 6$ MLP + SiLU + Fourier Feature Mapping (embedding dim 256, alleviating high-frequency bias), trained with the SOAP second-order preconditioned optimizer + cosine annealing. To stabilize CFT, LayerNorm (suppressing activation drift) and weight decay (suppressing unbounded weight growth) are added.
    - **Design Motivation**: Fitting a separate network for each snapshot is redundant (adjacent frame weights should be similar); fitting only the first frame and extrapolating in time leads to error explosion. "Continual fine-tuning of residuals" strikes a balance—using temporal correlation as a prior, while allowing per-frame correction.

3. **LoRA Low-rank Residuals Exposing Rate-distortion Pareto**:

    - **Function**: Parameterize the residual update $\Delta W_{\Delta t}$ as low-rank $\mathbf{A}\mathbf{B}$, allowing trade-off between memory and reconstruction accuracy by adjusting rank $r$.
    - **Mechanism**: $\Delta W_{\Delta t} = \mathbf{A}^{(\Delta t)}\mathbf{B}^{(\Delta t)}$ where $\mathbf{A}\in\mathbb{R}^{n\times r}$, $\mathbf{B}\in\mathbb{R}^{r\times k}$, $r\ll \min(n,k)$. LoRA initial learning rate is an order of magnitude higher than full FT ($10^{-2}$ vs $10^{-3}$), consistent with recent findings by Hayou et al. Changing $r$ moves along an accuracy-memory Pareto frontier—large $r$ approaches full FT, small $r$ yields extreme compression. On 3D BBH, $r=16$ achieves 3744× single snapshot spatial compression.
    - **Design Motivation**: Full FT updates all parameters, making storage control impossible; LoRA has shown that small rank suffices to approach full FT performance in LLMs—here, the same idea is applied to neural fields, with the fine-tune target being "fit the next time step's field." The adjustable Pareto frontier allows ANTIC to adapt to scenarios ranging from storage-constrained to accuracy-prioritized.

### Loss & Training
Neural field training: standard coordinate-to-value regression loss (L2 on physical quantity values at sampled coordinates). CFT phase learning rate anneals from $10^{-3}$ to $10^{-5}$; LoRA version uses initial learning rate $10^{-2}$. Each snapshot is fine-tuned independently before proceeding to the next; intermediate results (loss curve) are used by PATS to decide whether to trigger.

## Key Experimental Results

### Main Results (2D Kolmogorov + 3D BBH Merger)
PATS-LoRA outperforms both traditional compressors and uniform-sampling neural compression on two stress tests. TR=Temporal Retention, SC=Spatial Compression, TC=Total Compression.

| Method | Dataset | TR | PA | SC | TC |
|--------|---------|----|----|-----|-----|
| Sparse + ZFP | 2D Kolmogorov | 20% | ✗ | 13× | 65× |
| PATS + ZFP | same | 37% | ✓ | 13× | 120× |
| Sparse + LoRA(r=32) | same | 20% | ✗ | 47× | 235× |
| **ANTIC-LoRA (Ours)** | same | 37% | ✓ | **47×** | **435×** |
| Sparse + FT | 3D BBH (4.2 TiB) | 20% | ✗ | 471× | 2457× |
| Sparse + LoRA(r=16) | same | 20% | ✗ | 3744× | 18720× |
| Dense + LoRA(r=16) | same | 100% | ✓ | 3744× | 3744× |
| **ANTIC-LoRA (Ours)** | same | 55% | ✓ | **3744×** | **6807×** |

### Ablation Study

| Configuration | Key Result | Note |
|---------------|------------|------|
| Dense + ZFP (no PATS, traditional compression) | 13× / 27× | baseline, spatial compression upper bound |
| Dense + FT (no PATS, neural compression) | 12× / 471× | neural field outperforms traditional in 3D |
| PATS + ZFP | TC up to 120× / 52× | temporal selection alone greatly boosts traditional codec |
| ANTIC-FT (37% / 55% TR) | 111× / 860× | temporal selection + full FT neural compression |
| ANTIC-LoRA | 435× / 6807× | LoRA adds another order of magnitude |

### Key Findings
- Temporal and spatial axes are multiplicative: PATS alone yields 2.5–3× TC gain, neural field alone yields 30–470× SC, together reaching 100–1000×.
- On 3D BBH, a multi-rate system, PATS achieves 45% temporal compression without missing key physical events (merger transients), showing Weyl scalar is an effective saliency indicator—uniform sampling would miss merger peaks.
- LoRA rank $r$ provides a smooth Pareto; $r=16$ suffices for 3D, larger $r$ brings little benefit; this "rank proportional to PDE intrinsic dimension" may generalize to other scientific domains.
- LayerNorm + weight decay are critical for CFT stability—without them, weight norms explode and the network diverges after multiple fine-tunes.

## Highlights & Insights
- **Reframing "Residual as Compression"**: Instead of fitting an independent network per frame, perform LoRA residual updates to the previous frame's network—a simple yet powerful reframe, turning "spatial neural compression" into "joint temporal-spatial neural compression," naturally supporting streaming input.
- **PATS is Fully Parameter-free**: All decisions derive from PDE physical quantities and sliding window thresholds, with no training cost or hyperparameter tuning, and only the Metric needs to be changed for different PDEs—highly engineer-friendly.
- **Exposing Pareto Frontier Instead of Single Compression Ratio**: Users can select LoRA rank according to storage budget or accuracy needs; this "tunable" design is valuable in scientific computing, where accuracy tolerance varies widely across experiments.

## Limitations & Future Work
- The Metric is PDE-specific; new PDEs require expert-selected indicators. Whether data-driven methods can automatically learn saliency from trajectories is a future direction.
- LoRA rank is manually swept; adaptive rank allocation could further improve compression.
- Experiments cover only 2D Kolmogorov and 3D BBH PDEs; validation on more stiff systems (MHD, chemical reactions, etc.) is limited.
- Decompression requires sequentially loading each weight segment and coordinate query, making random time access unfriendly; streaming decoding is an advantage, but point queries are a disadvantage.
- Neural fields may still exhibit oscillations (Gibbs-like) for sharp features/shocks; this is not deeply discussed in the paper.

## Related Work & Insights
- **vs ZFP / FPZIP / MGARD (Traditional Compression)**: Traditional methods are transform-based and unaware of PDE multiscale structure; neural field + LoRA achieves one to two orders of magnitude higher SC in 3D.
- **vs MGARD (Feature-aware Error Bound Codec)**: MGARD is feature-aware in error bounds but still uniform in time; this work is non-uniform in time + neural in space, orthogonal to MGARD.
- **vs PINN / Physics-informed Neural Fields (Galletti 2025)**: Those use VQ + physics loss to achieve 70,000× compression but are offline only; this work is online and supports arbitrary PDEs without explicit loss design.
- **vs Neural Video Compression**: Conceptually similar (keyframe + residual), but NVC is perceptual-optimized, this work is physics-optimized; NVC keyframe selection is based on motion/information entropy, here it is based on PDE invariants.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of temporal PATS and spatial LoRA residual neural fields is a clear and effective innovation
- Experimental Thoroughness: ⭐⭐⭐⭐ Two PDEs of completely different scales (2D 16GB / 3D 4TB) + multiple baselines, results are convincing
- Writing Quality: ⭐⭐⭐⭐ Modular breakdown is clear, pseudocode is complete; some physics background may be deep for non-NR readers
- Value: ⭐⭐⭐⭐⭐ Directly addresses the storage crisis in scientific computing, engineering-ready and open source

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](../../NeurIPS2025/scientific_computing/f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](../../NeurIPS2025/scientific_computing/hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/scientific_computing/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)

</div>

<!-- RELATED:END -->
