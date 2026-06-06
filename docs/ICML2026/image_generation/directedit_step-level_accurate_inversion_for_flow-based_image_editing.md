---
title: >-
  [Paper Note] DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing
description: >-
  [ICML 2026][Image Generation][Flow Model Inversion] DirectEdit achieves "step-level accurate reconstruction" without increasing NFE by recording latent residuals $\Delta\mathbf{Z}_t$ during the Rectified Flow inversion p…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Flow Model Inversion"
  - "Training-free Image Editing"
  - "Step-level Accurate Reconstruction"
  - "Feature Injection"
  - "Mask-guided Blending"
date: 2026-05-08
content_hash: 554cf7745204078d
---

# DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing

**Conference**: ICML 2026  
**arXiv**: [2605.02417](https://arxiv.org/abs/2605.02417)  
**Code**: https://desongyang.github.io/Directedit (Available)  
**Area**: Diffusion Models / Image Editing / Rectified Flow  
**Keywords**: Flow Model Inversion, Training-free Image Editing, Step-level Accurate Reconstruction, Feature Injection, Mask-guided Blending

## TL;DR
DirectEdit achieves "step-level accurate reconstruction" without increasing NFE by recording latent residuals $\Delta\mathbf{Z}_t$ during the Rectified Flow inversion process and pre-injecting them into the forward path. Combined with MLLM+SAM multi-branch mask noise blending and attention Value injection, it significantly outperforms existing training-free methods like RF-Inversion, FireFlow, FTEdit, and DNAEdit, ranking 4.0 (FLUX) and 2.43 (SD3.5) on PIE-Bench.

## Background & Motivation

**Background**: Rectified Flow (RF) has become the mainstream framework for large-scale T2I models like SD3.5 and FLUX. Training-free image editing based on these pretrained flow models typically follows the "Inversion → Denoising Reconstruction + Editing Dual-Path" paradigm: Euler inversion maps a clean image to a noise latent, then forward denoising simultaneously runs a source reconstruction path and a target editing path, injecting source features (attention KV, latents, etc.) into the target path to preserve original information.

**Limitations of Prior Work**: The core approximation of Euler inversion, $\mathbf{Z}^{inv}_t = \mathbf{Z}^{inv}_{t+1} - (\sigma_{t+1}-\sigma_t)\,v_\theta(\mathbf{Z}^{inv}_{t+1})$, uses velocity at $t+1$ to approximate velocity at $t$. Although per-step error is small, it accumulates along the denoising steps, causing the reconstruction path to deviate entirely from the inversion trajectory. Subsequent works (high-order ODE solvers like RFEdit, fixed-point iteration in FTEdit, interpolated noise optimization in DNAEdit) only mitigate overall trajectory drift, while the **internal step-level error still persists**. This means the source features injected into the editing path at each step are "drifted features," leading to background distortion and artifacts. Inversion-free methods like FlowEdit sacrifice fidelity due to reliance on random noise interpolation. The authors demonstrate in Table 2 that standard Stepwise Correction has an average step-level MSE of 0.2857 (max 11.73), and even FTEdit reaches 0.0881.

**Key Challenge**: Existing methods focus on "correcting the inversion path to approximate the reconstruction path." However, RF's Euler inversion is **far more sensitive to single-step approximation errors than DDIM inversion**. No matter how the inversion side is optimized, the velocity $v_\theta(\mathbf{Z}_t)$ used in forward denoising and $v_\theta(\mathbf{Z}^{inv}_{t+1})$ recorded during inversion are inherently different, leaving step-level errors unresolvable.

**Goal**: Achieve step-level accurate reconstruction—ensuring the latent at each step of the forward path strictly equals the corresponding step of the inversion path $\mathbf{Z}_{t+1} = \mathbf{Z}^{inv}_{t+1}$—without adding any extra Neural Function Evaluations (NFE).

**Key Insight**: Instead of correcting the inversion path, why not **directly align the forward path**? If the velocity used in forward denoising can be made exactly equal to the velocity used in inversion $v_\theta(\mathbf{Z}_t) = v_\theta(\mathbf{Z}^{inv}_{t+1})$, step-wise alignment follows automatically. Since $\mathbf{Z}^{inv}_{t+1}$ is fully accessible during the inversion phase, one can simply buffer the latent residual $\Delta\mathbf{Z}_t = \mathbf{Z}^{inv}_{t+1} - \mathbf{Z}^{inv}_t$ at each step.

**Core Idea**: Use the latent residuals $\Delta\mathbf{Z}_t$ cached during inversion to temporarily shift the current latent before velocity prediction in the forward path, obtaining $\hat{\mathbf{Z}}_t = \mathbf{Z}_t + \Delta\mathbf{Z}_t$. Then, use $v_\theta(\hat{\mathbf{Z}}_t)$ to update $\mathbf{Z}_t$, forcibly aligning the velocity with the inversion trajectory at zero cost.

## Method

### Overall Architecture
Input: Source image $\mathbf{I}_{src}$, source prompt $\psi_{src}$, target prompt $\psi_{tar}$, $T$-step schedule $\{\sigma_0,\dots,\sigma_T\}$. The process consists of two stages:

1.  **Inversion Phase**: VAE encoding yields $\mathbf{Z}_T$. Standard Euler inversion is performed for $T$ steps to obtain the trajectory $\{\mathbf{Z}^{inv}_t\}$, with the **key step of caching each latent residual $\Delta\mathbf{Z}_t$**. Simultaneously, an MLLM identifies the edit type $O \in \{\text{Local, Background, Global, Other}\}$ and coordinates $P,Q$ for the region of interest, combined with SAM to generate a mask $\mathcal{M}$.
2.  **Editing Phase**: Starting from $\mathbf{Z}^{inv}_0$, source and target forward paths are run simultaneously. At each step, $\Delta\mathbf{Z}_t$ pushes the latent to $\hat{\mathbf{Z}}_t$ for velocity prediction (Direct Alignment). In the target path, the Value in self-attention is replaced with the source path's Value (Attention Injection). After the update, source and target latents are merged using the mask (Noise Blending). Finally, VAE decoding produces the edited result.

The extra overhead is limited to caching $\{\Delta\mathbf{Z}_t\}$, **introducing no additional NFE**—a key efficiency advantage over FTEdit (120 NFE) and RFEdit (120 NFE).

### Key Designs

1.  **Direct Alignment for Accurate Inversion**:
    - **Function**: Ensures the forward reconstruction path strictly matches the inversion trajectory latent at every step, eliminating step-level reconstruction errors at the source.
    - **Mechanism**: From the Euler update formula, $\mathbf{Z}_{t+1} = \mathbf{Z}^{inv}_{t+1}$ is equivalent to $v_\theta(\mathbf{Z}_t) = v_\theta(\mathbf{Z}^{inv}_{t+1})$, where $\mathbf{Z}^{inv}_{t+1}$ is known from inversion. During inversion, the latent residual $\Delta\mathbf{Z}_t = \mathbf{Z}^{inv}_{t+1} - \mathbf{Z}^{inv}_t$ is recorded. During forward denoising, an alignment latent $\hat{\mathbf{Z}}_t = \mathbf{Z}_t + \Delta\mathbf{Z}_t$ is constructed for velocity prediction, and the update is performed as $\mathbf{Z}_{t+1} = \mathbf{Z}_t + (\sigma_{t+1}-\sigma_t)\,v_\theta(\hat{\mathbf{Z}}_t)$. Note that velocity prediction uses $\hat{\mathbf{Z}}_t$ while the update applies to the original $\mathbf{Z}_t$; this misalignment ensures the velocity field matches the inversion stage exactly.
    - **Design Motivation**: Abandons the old approach of "fixing the inversion path" in favor of correcting the velocity input of the forward path. It requires no additional $v_\theta$ calls, keeping NFE identical to vanilla Euler (60 NFE vs 120 NFE for FTEdit), while reducing step-level MSE from the $10^{-1}$ magnitude to $10^{-4}$ (Table 2).

2.  **Multi-branch Mask-guided Noise Blending**:
    - **Function**: Automatically generates spatial masks based on the edit type to merge source and target paths in latent space, precisely protecting the background.
    - **Mechanism**: MLLM processes $(\mathbf{I}_{src}, \psi_{src}, \psi_{tar})$ to output edit type $O$ and bounding box $(P, Q)$. Different branches generate the mask based on $O$: Local uses SAM for object segmentation; Background uses the inverse; Global (e.g., style transfer) uses a full mask of 1; Other (e.g., object addition) uses the rectangular box $\mathcal{B}(P,Q)$. After each denoising step, fusion occurs: $\mathbf{Z}^{far}_{t+1} \leftarrow \mathbf{Z}^{src}_{t+1} \odot (\mathbf{1}-\mathcal{M}) + \mathbf{Z}^{tar}_{t+1} \odot \mathcal{M}$. Combined with DirectEdit's accurate reconstruction, non-edited areas achieve near-zero distortion.
    - **Design Motivation**: Single rectangular or segmentation masks cannot adapt to all edit types. The multi-branch mechanism automatically matches mask generation with editing intent. Fusing step-wise in latent space avoids boundary artifacts caused by one-time final fusion. Ablations show PSNR drops by 4.77 dB (32.63 → 27.86) without multi-branch support.

3.  **Attention Feature Injection**:
    - **Function**: Injects source path Values into the target path self-attention layers to preserve fine-grained semantic details of the edited object.
    - **Mechanism**: For the first $t_{inj}$ steps, the self-attention output of each MM-DiT block in the target path is replaced with $\text{Attention}(\mathbf{Q}^{tar}_t, \mathbf{K}^{tar}_t, \mathbf{V}^{src}_t)$, followed by a return to standard self-attention. The resulting features $\hat{\mathbf{F}}^{tar}_t$ refresh the target velocity $\hat{v}^{tar}_t$. Architecturally, SD3.5 injects into all MM-DiT blocks (except the last), while FLUX only injects into single blocks (as they process image and text features simultaneously, better preserving source information). $t_{inj}=3$ is generally optimal.
    - **Design Motivation**: While mask blending handles the background, the editing area needs texture and identity preservation. Value injection preserves "content" (V) while allowing new "relationships" (from target Q, K). DirectEdit provides drift-free V, making the injection "clean."

### Loss & Training
**Entirely training-free**, involving no loss functions or backpropagation. Inference settings: FLUX.1-dev / SD3.5-medium backbones, 30-step denoising, CFG=1 (Inversion) / CFG=2 (Editing), $t_{inj}=3$.

## Key Experimental Results

### Main Results
Comparison of training-free editing on PIE-Bench (700 images, 9 edit categories; lower average rank is better):

| Backbone | Method | Structure↓ | PSNR↑ | LPIPS↓ | MSE↓ | CLIP-Whole↑ | Avg. Rank↓ |
|----------|------|-----------|-------|--------|------|-------------|-------------|
| FLUX | RF-Inversion | 41.17 | 20.86 | 187.01 | 120.12 | 25.08 | 13.57 |
| FLUX | RFEdit | 25.15 | 24.33 | 121.59 | 56.98 | 25.57 | 9.14 |
| FLUX | FireFlow | 27.40 | 23.11 | 128.46 | 70.75 | 26.13 | 9.43 |
| FLUX | FlowEdit | 27.83 | 21.96 | 112.15 | 94.94 | 25.26 | 10.57 |
| FLUX | DNAEdit | 16.81 | 25.20 | 86.68 | 48.35 | 24.81 | 7.71 |
| FLUX | **DirectEdit** | **17.94** | **32.63** | **35.45** | **25.05** | 25.39 | **4.00** |
| SD3.5 | FTEdit | 21.06 | 23.49 | 90.25 | 61.78 | 25.21 | 9.29 |
| SD3.5 | FlowEdit | 23.13 | 23.29 | 92.81 | 69.09 | 26.71 | 7.29 |
| SD3.5 | DNAEdit | 11.03 | 27.71 | 60.51 | 26.28 | 25.20 | 5.14 |
| SD3.5 | **DirectEdit** | **14.65** | **31.82** | **31.36** | **21.64** | 25.64 | **2.43** |

**Reconstruction Error Comparison** (FLUX, 60 NFE vs 120 NFE, key metric: Step-Level MSE):

| Method | NFE↓ | PSNR↑ | Step-Level MSE Avg↓ | Step-Level MSE Max↓ |
|------|------|-------|---------------------|---------------------|
| VAE (Theoretical Limit) | - | 34.38 | - | - |
| Vanilla Euler | 60 | 14.59 | 1177.73 | 39511.72 |
| Stepwise Correction | 60 | 34.38 | 0.2857 | 11.73 |
| FTEdit | 120 | 34.38 | 0.0881 | 14.82 |
| RFEdit | 120 | 21.92 | 231.72 | 20156.25 |
| **DirectEdit** | **60** | **34.38** | **0.0006** | **0.0757** |

DirectEdit uses half the NFE to reduce step-level MSE average/max by **2–4 orders of magnitude**, reaching the theoretical limit of VAE reconstruction.

### Ablation Study (FLUX, PIE-Bench)

| Configuration | Struct.↓ | PSNR↑ | LPIPS↓ | MSE↓ | CLIP-Whole↑ |
|------|----------|-------|--------|------|-------------|
| Vanilla | 75.95 | 16.81 | 276.29 | 332.65 | 23.57 |
| w/o alignment (fallback to Stepwise Correction) | 29.22 | 31.12 | 53.16 | 48.17 | 25.24 |
| w/o attention | 23.75 | 31.93 | 39.60 | 33.97 | 25.60 |
| w/o mask | 21.93 | 24.70 | 102.92 | 56.76 | 25.89 |
| w/o multi-branch (fallback to rectangle) | 19.15 | 27.86 | 60.92 | 38.94 | 25.71 |
| **DirectEdit (full)** | **17.94** | **32.63** | **35.45** | **25.05** | 25.39 |

### Key Findings
- **Direct Alignment is the critical core**: Removing it causes all metrics to collapse (PSNR drops 32.63 → 31.12, Structure distance rises 17.94 → 29.22), confirming that step-level reconstruction error is the root cause of feature injection drift.
- **Mask blending dominates background fidelity**: Without masking, PSNR drops by 7.93 dB; multi-branch masking further contributes approximately 4.77 dB.
- **Attention injection trades off detail preservation with prompt following**: Removing it slightly increases CLIP scores (25.60 vs 25.39) but yields a loss of source texture. $t_{inj}=3$ is the empirical sweet spot.
- **Significant efficiency advantage**: It achieves better step-level MSE at 60 NFE than FTEdit at 120 NFE (0.0006 vs 0.0881), being both faster and more accurate.

## Highlights & Insights
- **The "Reverse Thinking" Inversion Philosophy**: While predecessors attempted to fix the inversion path to match the reconstruction path, the authors treat the inversion path as the pivot and align the forward path to it. This shift in perspective leads to an elegant solution: simply caching and adding back residuals.
- **Zero-NFE Precision Gain**: Accurate step-level alignment is achieved purely through algebraic identity at the latent level ($v_\theta(\mathbf{Z}_t + \Delta\mathbf{Z}_t) = v_\theta(\mathbf{Z}^{inv}_{t+1})$). No extra neural forwards, no fixed-point iterations, and no high-order solvers are required.
- **"Semantic Mask Router" via MLLM+SAM**: Upgrading mask generation from a binary choice (SAM vs. rectangle) to multi-branch routing based on edit intent is a pattern that can be transferred to other training-free tasks like video or 3D editing.
- **Reusable Trick**: The paradigm of caching inversion-side intermediates to "compensate" the forward side can be generalized to any inversion task using Euler/RK discrete ODE solvers.

## Limitations & Future Work
- The editing quality ceiling is locked by the priors of the backbone T2I models; the method still struggles with edits requiring "content reorganization," such as size changes or complex contextual reasoning.
- Implicit Limitation: The accuracy of MLLM in identifying edit types and coordinates directly determines mask quality. The paper does not quantify the impact of MLLM misclassification.
- Implicit Limitation: Caching $\{\Delta\mathbf{Z}_t\}$ ($T$ latents) imposes memory pressure for long schedules or high-resolution FLUX models. 
- Future Directions: (1) Combine Direct Alignment with higher-order solvers for even fewer steps; (2) Extend mask routing from discrete branches to learnable continuous generators; (3) Hybridize with inversion-free ideas (e.g., FlowEdit) for large-scale edits in early steps and DirectEdit for refinement.

## Related Work & Insights
- **vs Stepwise Correction (Direct Inversion, 2023)**: Both recognize the need for path alignment, but the former forces $\mathbf{Z}_t = \mathbf{Z}^{inv}_t$ after reconstruction, leaving velocities inconsistent within each step. DirectEdit ensures velocity consistency *before* reconstruction, reaching MSE levels two orders of magnitude lower.
- **vs FTEdit / DNAEdit**: FTEdit uses fixed-point iteration; DNAEdit uses interpolated velocity estimation. Both reduce but do not eliminate step-level errors while doubling NFE to 120. DirectEdit achieves $10^{-4}$ error at 60 NFE with superior rankings.
- **vs FlowEdit (inversion-free)**: FlowEdit bypasses inversion via noise interpolation, which is simple but lacks fidelity (PSNR 21.96 on FLUX). DirectEdit proves that "perfecting inversion" remains a superior route for RF.
- **vs RF-Inversion**: RF-Inversion uses LQR control theory for auxiliary velocity fields, but DirectEdit's "residual caching" is simpler and achieves much higher PSNR (32.63 vs 20.86), showing that engineering simplicity with the right perspective often wins.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perspective shift of "aligning the forward path" and the zero-NFE residual trick are highly intuitive yet transformative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive PIE-Bench metrics across two backbones (FLUX/SD3.5) with specific step-level MSE validation, though memory data for high-res scenarios is missing.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations and helpful algorithms; some proofs are relocated to the Appendix.
- Value: ⭐⭐⭐⭐⭐ Sets a new benchmark for efficiency (60 vs 120 NFE) and accuracy (2-order MSE reduction) in RF-based editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](../../NeurIPS2025/image_generation/splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)
- [\[ICML 2026\] Semantic Granularity Navigation in Image Editing](semantic_granularity_navigation_in_image_editing.md)
- [\[NeurIPS 2025\] FALCON: Few-step Accurate Likelihoods for Continuous Flows](../../NeurIPS2025/image_generation/falcon_few-step_accurate_likelihoods_for_continuous_flows.md)
- [\[ICLR 2026\] Free Lunch for Stabilizing Rectified Flow Inversion](../../ICLR2026/image_generation/free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)

</div>

<!-- RELATED:END -->
