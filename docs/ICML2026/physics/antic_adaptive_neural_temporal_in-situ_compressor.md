---
title: >-
  [Paper Note] ANTIC: Adaptive Neural Temporal In-situ Compressor
description: >-
  [ICML 2026][Physics & Scientific Computing][Online Compression] To achieve "on-the-fly" compression of PB-EB scale PDE simulation data, this paper proposes ANTIC: using a physics-aware temporal selector to retain only ph…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Online Compression"
  - "Neural Fields"
  - "Continual Fine-tuning"
  - "LoRA"
  - "PDE Simulation"
date: 2026-05-08
content_hash: b6079105dc40ac76
---

# ANTIC: Adaptive Neural Temporal In-situ Compressor

**Conference**: ICML 2026  
**arXiv**: [2604.09543](https://arxiv.org/abs/2604.09543)  
**Code**: https://github.com/AndreiB137/ANTIC  
**Area**: Scientific Computing / Neural Compression / Neural Fields  
**Keywords**: Online Compression, Neural Fields, Continual Fine-tuning, LoRA, PDE Simulation  

## TL;DR
To achieve "on-the-fly" compression of PB-EB scale PDE simulation data, this paper proposes ANTIC: using a physics-aware temporal selector to retain only physically critical snapshots, followed by neural fields with LoRA-based continual fine-tuning to encode residuals between adjacent snapshots. It achieves 435× compression on 2D Kolmogorov flows and 6807× spatio-temporal joint compression on 4.2 TiB 3D binary black hole merger simulations.

## Background & Motivation
**Background**: High-resolution transient simulations in CFD, magnetohydrodynamics, plasma physics, and numerical relativity often produce single trajectories ranging from several TB to hundreds of TB. Conventional solutions involve post-hoc compression (JPEG2000, DWT, FPZIP, ZFP, etc.) after the simulation outputs raw data. In the spatial dimension, both codec-based and low-rank tensor decomposition methods are widely used.

**Limitations of Prior Work**: (1) Offline compression is no longer feasible for petascale/exascale simulations—there is insufficient disk space to store raw data before compression. (2) Existing in-situ compression either uses fixed-interval temporal sampling (missing transient events or oversampling stable periods) or utilizes fixed spatial representations (autoencoder latents lack resolution invariance, and traditional codecs struggle with multiscale correlations). (3) Most methods lack "physics awareness"—they neither identify critical snapshots nor exploit continuity between adjacent snapshots.

**Key Challenge**: Stiff/multi-rate PDEs exhibit both temporal multiscale (simultaneous fast and slow phase transitions) and spatial multiscale (non-linear, non-stationary) characteristics. A single temporal sampling strategy or spatial representation cannot simultaneously optimize storage, accuracy, and throughput—necessitating a physics-aware in-situ framework for joint spatio-temporal optimization.

**Goal**: (1) Design a parameter-free snapshot selector on the temporal axis capable of injecting PDE-specific metrics; (2) Use neural field representations with continual fine-tuning for residuals between adjacent snapshots in the spatial axis; (3) Combine these into a single streaming pass in-situ pipeline, exposing a rate-distortion Pareto front for user-defined selection.

**Key Insight**: The authors observe that solutions to stiff PDEs primarily consist of "smooth, small-magnitude perturbations" between adjacent timesteps. Compressing snapshot $t+\Delta t$ can be re-interpreted as performing a low-rank residual update on a neural field already fitted to snapshot $t$—a task naturally suited for LoRA. Simultaneously, physical quantities (enstrophy, Weyl scalar) serve as cheap saliency indicators to instantly judge whether the system is in a steady state or a phase transition.

**Core Idea**: A combination of a Physics-aware Temporal Selector (PATS) and Continual Fine-Tuning (CFT / CFT+LoRA) of neural fields to complete spatio-temporal compression online, providing a rate-distortion Pareto front to the user.

## Method
ANTIC consists of two asynchronous modules: (i) PATS determines "whether to compress this frame," and (ii) Spatial Neural Compression determines "how to compress it." The entire pipeline operates in a single streaming pass without saving raw trajectories to disk.

### Overall Architecture
- **Streaming Input**: The simulator outputs snapshots $u(t)$ step-by-step.
- **PATS Sub-pipeline**: A Metric extracts physics-of-interest $\phi_t$ (e.g., enstrophy $\mathcal{E}$ or Weyl scalar magnitude) from the snapshot. A Regulator dynamically adjusts the Queue window size $W$ based on $\phi_t$. A Gate uses the current truncated context and $\phi_t$ to form a dynamic threshold for accepting the snapshot.
- **Spatial Neural Compression**: Selected snapshots update an existing neural field $W_t \to W_t + \Delta W_{\Delta t}$ via Continual Fine-Tuning (CFT). $\Delta W_{\Delta t}$ can be a full fine-tune (higher accuracy, higher memory) or a low-rank $\mathbf{A}^{(\Delta t)}\mathbf{B}^{(\Delta t)}$ (more efficient, slight loss in accuracy), allowing users to select points on the Pareto front.
- **Output**: Only sparse sequences of neural field weights are stored on disk. During decompression, weights are incrementally added back to the base network, and field values at any moment can be reconstructed by querying coordinates.

### Key Designs

1. **Physics-aware Temporal Selector (PATS)**:
    - **Function**: Online assessment of whether a new snapshot contains sufficient "new physical information" to merit storage, skipping redundant slow evolution phases while preserving rapid transients.
    - **Mechanism**: A parameter-free four-part architecture. The Metric is a PDE-specific scalar indicator—enstrophy $\mathcal{E}(t) = \frac{1}{2}\int_\Omega \|\omega\|^2 dA$ for turbulence, or the Weyl scalar $\Psi_4(t,\mathbf{r})$ for binary black hole mergers. The Queue is a sliding window storing the $W$ most recent metric values. The Regulator truncates the Queue and resets the reference anchor upon detecting a phase transition. The Gate forms a dynamic threshold based on the truncated context.
    - **Design Motivation**: Traditional temporal sampling is either fixed-interval (missing fast changes) or uses generic heuristics (e.g., pixel differences) that ignore physical meaning. Using intrinsic PDE conservation laws or excitation indicators as saliency enables adaptive sampling—low frequency during slow evolution and high frequency during transients—requiring only a change in the Metric function for different PDEs.

2. **Neural Fields + Continual Fine-Tuning for Residuals**:
    - **Function**: Use a coordinate-based MLP to compress each snapshot into $\sim 0.4$M parameters; only differences between adjacent snapshots are learned via small-scale fine-tuning.
    - **Mechanism**: Spatial compression is reframed as "residual updates to a neural field fitted for $u(t)$ to fit $u(t+\Delta t)$." Due to PDE smoothness $u(t+\Delta t) - u(t) \approx \Delta u(t)$, the residual magnitude is much smaller than the original field, allowing convergence with few gradient steps and minimal parameter updates. The architecture uses a $256\times 6$ MLP + SiLU + Fourier Feature Mapping with a SOAP second-order preconditioned optimizer and cosine annealing. Stability is ensured via LayerNorm and weight decay.
    - **Design Motivation**: Fitting an independent network for every snapshot is redundant; relying solely on temporal extrapolation leads to error accumulation. Continual residual fine-tuning leverages temporal correlation as a prior while performing frame-by-frame corrections.

3. **LoRA Residuals for Rate-Distortion Pareto**:
    - **Function**: Parameterize the residual update $\Delta W_{\Delta t}$ as a low-rank $\mathbf{A}\mathbf{B}$ structure, enabling a trade-off between memory and reconstruction accuracy by adjusting rank $r$.
    - **Mechanism**: $\Delta W_{\Delta t} = \mathbf{A}^{(\Delta t)}\mathbf{B}^{(\Delta t)}$ where $\mathbf{A}\in\mathbb{R}^{n\times r}$ and $\mathbf{B}\in\mathbb{R}^{r\times k}$. LoRA uses a higher initial learning rate ($10^{-2}$ vs $10^{-3}$). Adjusting $r$ moves the system along an accuracy-memory Pareto front—large $r$ approaches full FT, while small $r$ provides extreme compression.
    - **Design Motivation**: Full FT updates all parameters, preventing storage control. LoRA allows low-rank approximation of full FT performance, adapted here for fitting subsequent timesteps. This makes ANTIC adaptable to scenarios ranging from tight storage constraints to high-precision requirements.

### Loss & Training
Neural field training uses a standard coordinate-to-value regression loss ($L2$ on physical values at sampled coordinates). During the CFT stage, the learning rate undergoes cosine annealing. Each snapshot completes fine-tuning independently before proceeding to the next; intermediate loss curves determine if the PATS decision was triggered.

## Key Experimental Results

### Main Results (2D Kolmogorov + 3D BBH Merger)
PATS-LoRA significantly outperforms traditional compressors and fixed-interval neural compression on two stress tests. TR=Temporal Retention, SC=Spatial Compression, TC=Total Compression.

| Method | Dataset | TR | PA | SC | TC |
|------|--------|----|----|-----|-----|
| Sparse + ZFP | 2D Kolmogorov | 20% | ✗ | 13× | 65× |
| PATS + ZFP | Same | 37% | ✓ | 13× | 120× |
| Sparse + LoRA(r=32) | Same | 20% | ✗ | 47× | 235× |
| **ANTIC-LoRA (Ours)** | Same | 37% | ✓ | **47×** | **435×** |
| Sparse + FT | 3D BBH (4.2 TiB) | 20% | ✗ | 471× | 2457× |
| Sparse + LoRA(r=16) | Same | 20% | ✗ | 3744× | 18720× |
| Dense + LoRA(r=16) | Same | 100% | ✓ | 3744× | 3744× |
| **ANTIC-LoRA (Ours)** | Same | 55% | ✓ | **3744×** | **6807×** |

### Ablation Study

| Configuration | Key Result | Description |
|------|----------|------|
| Dense + ZFP | 13× / 27× | Baseline, spatial compression ceiling for traditional codecs |
| Dense + FT | 12× / 471× | Neural fields significantly outperform traditional methods in 3D |
| PATS + ZFP | TC up to 120× / 52× | Temporal selection significantly extends traditional codecs |
| ANTIC-FT | 111× / 860× | Temporal selection + full FT neural compression |
| ANTIC-LoRA | 435× / 6807× | LoRA adds another order of magnitude to compression |

### Key Findings
- Temporal and spatial axes provide multiplicative gains: PATS adds 2.5~3× TC, while neural fields add 30~470× SC, reaching 100~1000× range combined.
- On the multi-rate 3D BBH system, PATS achieved 45% temporal compression without losing critical physical events (merger transients), proving the Weyl scalar is an effective saliency metric.
- LoRA rank $r$ provides a smooth Pareto front; $r=16$ is sufficient for 3D, with diminishing returns for larger ranks—suggesting rank correlates with intrinsic PDE dimensionality.
- LayerNorm and weight decay are crucial for CFT stability; without them, weight norms explode, causing the network to diverge after multiple fine-tuning steps.

## Highlights & Insights
- **"Residual as Compression" Perspective**: Shifting from "fitting a network per frame" to "LoRA residual updates" is a simple but powerful reframing that turns spatial neural compression into joint spatio-temporal compression with native streaming support.
- **Paramter-free PATS**: Since decisions rely on PDE physics and sliding window thresholds, there is no training cost or hyperparameter tuning "black magic," and it is easily adaptable to different PDEs by simply changing the Metric.
- **Exposing a Pareto Front**: Allowing users to choose LoRA rank based on storage budgets or precision needs is highly practical for scientific computing, where tolerance for accuracy varies significantly between experiments.

## Limitations & Future Work
- Metrics are PDE-specific and require expert selection; future work could explore data-driven saliency learning from trajectories.
- LoRA rank is currently manually swept; adaptive rank allocation could further improve compression.
- Experiments are limited to 2D Kolmogorov and 3D BBH; verification across more stiff systems (magnetohydrodynamics, etc.) is needed.
- Decompression requires sequential loading of weights, which is unfriendly to random temporal access. Point queries are also slower than traditional codecs.
- Neural fields may still exhibit oscillations (Gibbs-like) for sharp features or shocks.

## Related Work & Insights
- **vs ZFP / FPZIP / MGARD**: Traditional methods are transform-based and unaware of PDE multiscale structures; ANTIC's SC is 1-2 orders of magnitude higher in 3D.
- **vs MGARD (Adaptive Accuracy)**: MGARD provides feature-aware error bounds but remains temporally uniform; ANTIC provides non-uniform temporal and neural spatial compression.
- **vs PINN / Physics-Informed Neural Fields (Galletti 2025)**: While some offline methods reach 70,000× using VQ and physics losses, ANTIC supports online streaming and arbitrary PDEs without explicit loss derivation.
- **vs Neural Video Compression**: Similar in concept (keyframe + residuals), but NVC targets perceptual quality while ANTIC targets physics; PATS selection is based on conservation laws rather than motion.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of PATS and LoRA-based residual marks a clear and effective innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong results across different scales (2D 16GB vs 3D 4TB) and multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear modular breakdown and complete pseudocode.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the storage crisis in scientific computing; engineering-ready and open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)
- [\[ICML 2026\] BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields](ballast_bayesian_active_learning_with_look-ahead_amendment_for_sea-drifter_traje.md)
- [\[NeurIPS 2025\] FEAT: Free Energy Estimators with Adaptive Transport](../../NeurIPS2025/physics/feat_free_energy_estimators_with_adaptive_transport.md)
- [\[AAAI 2026\] Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness](../../AAAI2026/physics/adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)

</div>

<!-- RELATED:END -->
