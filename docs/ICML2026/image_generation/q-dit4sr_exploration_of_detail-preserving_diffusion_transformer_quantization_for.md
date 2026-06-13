---
title: >-
  [Paper Note] Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution
description: >-
  [ICML 2026][Image Generation][PTQ] This paper introduces Q-DiT4SR, the first PTQ framework specifically designed for DiT-based Real-World Image Super-Resolution (Real-ISR). It preserves high-frequency details through "gl…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "PTQ"
  - "Diffusion Transformer"
  - "Hierarchical SVD"
  - "Mixed Precision"
  - "Real-ISR"
date: 2026-05-08
content_hash: 5c08db57dd638930
---

# Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution

**Conference**: ICML 2026  
**arXiv**: [2602.01273](https://arxiv.org/abs/2602.01273)  
**Code**: https://github.com/xunzhang1128/Q-DiT4SR (To be released)  
**Area**: Model Compression / Diffusion Model Quantization / Real-World Image Super-Resolution  
**Keywords**: PTQ, Diffusion Transformer, Hierarchical SVD, Mixed Precision, Real-ISR

## TL;DR
This paper introduces Q-DiT4SR, the first PTQ framework specifically designed for DiT-based Real-World Image Super-Resolution (Real-ISR). It preserves high-frequency details through "global low-rank + local block-wise rank-1" hierarchical SVD decomposition. It further proposes data-free inter-layer weight bit-width allocation (VaSMP) and DP-based temporal activation bit-width scheduling (VaTMP) based on Rate-Distortion theory. Q-DiT4SR achieves SOTA performance under W4A6/W4A4 settings, compressing the model by 5.8$\times$ and reducing computation by 6.14$\times$.

## Background & Motivation

**Background**: Real-ISR has evolved from CNN/Transformer to diffusion models. Recent methods based on Diffusion Transformer (DiT), such as DiT4SR and DreamClear, use pure DiT architectures with linear layers and self-attention to significantly improve texture recovery. however, DiT models suffer from massive parameter counts and computational overhead, and iterative denoising further escalates inference costs, hindering deployment.

**Limitations of Prior Work**: PTQ is a recognized low-cost acceleration solution, but current methods are not directly applicable to DiT-based Real-ISR: (1) General diffusion PTQ (e.g., Q-Diffusion, PTQD, TDQ) designed for U-Net and text-to-image tasks leads to severe high-frequency texture degradation in DiT super-resolution; (2) DiT-specific PTQ (e.g., PTQ4DiT, Q-DiT, SVDQuant) targeting text-to-image generation is unsuitable for "pixel-level fidelity" tasks like SR, often failing under W4A4 settings.

**Key Challenge**: The authors identify three specific deficiencies: ① Existing SVD low-rank decomposition is too "global," discarding high-frequency residues as noise despite SR's reliance on them; ② Weight variances differ drastically across DiT layers (by orders of magnitude), yet layers are assigned uniform bit-widths; ③ Activation variances vary significantly across diffusion timesteps, while existing methods use "time-invariant" static precision allocation.

**Goal**: To address weight reconstruction accuracy, inter-layer weight bit-width allocation, and temporal activation bit-width scheduling for DiT-based Real-ISR under ultra-low W4A6/W4A4 settings, while minimizing expensive calibration requirements.

**Key Insight**: Two key observations: (a) PCA analysis reveals that removing the top 128 principal components of DiT layer outputs significantly damages SR quality, indicating dominant components must remain in FP; (b) After Hadamard transform, weights and activations approximate a Gaussian distribution, where variance directly determines uniform quantization distortion. Thus, "variance" serves as a natural sensitivity proxy.

**Core Idea**: Use "Global low-rank + Local block-wise rank-1" hierarchical SVD (H-SVD) to preserve FP information flow; employ "variance-driven Rate-Distortion" closed-form solutions with greedy discretization for data-free inter-layer bit-width allocation (VaSMP), and use "variance-driven dynamic programming" for intra-layer temporal activation bit-width scheduling (VaTMP).

## Method

### Overall Architecture
Q-DiT4SR uses DiT4SR as the backbone, where all MM-DiT blocks are quantized (Softmax is kept at 8-bit for numerical stability). Each linear layer first undergoes a Hadamard transform to make weights/activations approximately Gaussian. Subsequently, three independent but cascaded processes are executed: ① H-SVD decomposes weights into "Global SVD-G + Local block rank-1 SVD-L" FP branches, with quantization applied only to the residue. ② VaSMP determines the bit-width $b_\ell$ for each layer $\ell$ during the offline stage by solving a Rate-Distortion problem using layer-wise weight variance $\bar{\sigma}_\ell^2$. ③ VaTMP (activated for W4A4) calculates a "piecewise constant" temporal bit-width schedule using dynamic programming based on activation variances $v_{\ell,t}$ collected from a small calibration set. These steps are orthogonal, resulting in a (layer $\times$ timestep) bit-width grid for inference.

### Key Designs

1.  **H-SVD (Hierarchical SVD) Weight Decomposition**:
    - **Function**: Decomposes each Hadamard-transformed weight $\mathbf{W}_H = \mathbf{W}\mathbf{H}_n$ into "Global SVD-G + Local block SVD-L + Quantized residue" to ensure local high-frequency information required for SR remains in FP branches.
    - **Mechanism**: First, truncated SVD produces the global rank-$r$ branch $\mathbf{W}_{\text{SVD-G}}$. The residue $\mathbf{W}_{\text{res}} = \mathbf{W}_H - \mathbf{W}_{\text{SVD-G}}$ is then split into $s_o \times s_i$ small blocks, each approximated by a rank-1 SVD $\mathbf{W}^{(p,q)} \approx \sigma_{p,q}\mathbf{u}_{p,q}\mathbf{v}_{p,q}^\top$ to form SVD-L. Block sizes $(s_o, s_i)$ are grid-searched under the constraint $P_{\text{SVD-L}} \lesssim P_{\text{SVD-G}}(r)$. The final reconstruction is $\hat{\mathbf{W}} = (\mathbf{W}_{\text{SVD-G}} + \mathbf{W}_{\text{SVD-L}} + Q_w(\mathbf{W}_{\text{res}} - \mathbf{W}_{\text{SVD-L}}))\mathbf{H}_n^\top$.
    - **Design Motivation**: Global-only SVD (like SVDQuant) lumps all "non-low-rank" components into the quantizer, losing local details. By explicitly modeling local blocks as rank-1 FP branches, H-SVD output projections in the principal component space stay closer to the FP model.

2.  **VaSMP (Variance-aware Spatio Mixed Precision) Data-free Inter-layer Bit-width Allocation**:
    - **Function**: Decides weight bit-width $b_\ell$ for each DiT layer $\ell$ without calibration data, minimizing total distortion under a target average bit-width budget $B_{\text{target}}$.
    - **Mechanism**: Based on high-rate approximation $\mathbb{E}[e^2] \propto \sigma^2 \cdot 2^{-2b}$, layer-wise distortion is defined as $D_\ell(b_\ell) \propto N_\ell \bar{\sigma}_\ell^2 2^{-2b_\ell}$. Solving the Lagrangian under the budget $\sum_\ell w_\ell b_\ell = B_{\text{target}} \sum_\ell w_\ell$ yields the continuous solution $b_\ell^* = B_{\text{target}} + \tfrac{1}{2}(\log_2 \bar{\sigma}_\ell^2 - \overline{\log_2 \bar{\sigma}})$. A greedy strategy then allocates remaining bits based on gain $\text{Gain}_\ell \propto \bar{\sigma}_\ell^2 \cdot 4^{-b_\ell}$.
    - **Design Motivation**: DiT weight variances vary across layers while remaining stable within output channels. VaSMP utilizes weight statistics directly, reducing mixed-precision costs to near zero compared to Hessian-based methods.

3.  **VaTMP (Variance-aware Temporal Mixed Precision) Timestep Activation DP Scheduling**:
    - **Function**: For W4A4, it segments diffusion timesteps $T_\ell$ for each layer and assigns activation bit-widths $b_{\ell,t} \in \{2, \dots, 8\}$ to minimize total activation quantization distortion.
    - **Mechanism**: Assuming Gaussian distribution $z \sim \mathcal{N}(0, v_{\ell,t})$, distortion is $D_{\ell,t}(b) = v_{\ell,t} \cdot \kappa(b)$ where $\kappa(b)$ is the normalized distortion coefficient. Token variances $v_{\ell,t} = \tfrac{1}{NC}\sum_n \|\mathbf{z}_n^{(\ell,t)}\|_2^2$ are collected on a small calibration set (32 images). The optimal "piecewise constant" schedule is then solved using DP with cost $\text{SegCost}(i,j;b) = \kappa(b)\sum_{t=i}^{j-1} v_{\ell,t}$.
    - **Design Motivation**: Activation variance in diffusion SR exhibits a "rise-then-fall" trajectory. Static bit-widths are either wasteful or insufficient. VaTMP ensures high-variance timesteps receive higher precision.

### Loss & Training
Q-DiT4SR is a PTQ framework requiring no retraining. VaSMP is entirely data-free. VaTMP requires only a small calibration set of 32 LR images ($128 \times 128$ crops) to collect activation variances. Evaluations are performed on NVIDIA RTX A6000 with DiT4SR $\times 4$ SR as the backbone.

## Key Experimental Results

### Main Results
Under W4A6/W4A4, Q-DiT4SR was compared against Q-Diffusion, SVDQuant, Q-DiT, etc., on DrealSR, RealSR, RealLR200, and RealLQ250.

| Dataset / Setting | Metric | FP | SVDQuant | Q-DiT | FlatQuant | Q-DiT4SR (ours) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RealSR W4A6 | MUSIQ $\uparrow$ | 67.89 | 66.63 | 59.02 | 57.11 | **67.72** |
| RealSR W4A6 | LIQE $\uparrow$ | 3.988 | 3.434 | 1.790 | 2.455 | **3.980** |
| RealSR W4A4 | MUSIQ $\uparrow$ | 67.89 | 63.14 | 59.97 | 59.41 | **66.36** |
| RealSR W4A4 | LIQE $\uparrow$ | 3.988 | 3.115 | 2.009 | 1.996 | **3.179** |

For W4A4, Q-DiT4SR reduced peak memory from 15086 to 3974 MiB with $\sim 4.5\times$ end-to-end acceleration and **$8.99\times$** single-layer speedup.

### Ablation Study

| Config (RealSR W4A4) | MUSIQ $\uparrow$ | MANIQA $\uparrow$ | CLIP-IQA $\uparrow$ | LIQE $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Naive PTQ) | 64.94 | 0.4111 | 0.4899 | 3.191 |
| + H-SVD + VaSMP | 65.83 | 0.4227 | 0.4922 | 3.091 |
| + H-SVD + VaSMP + VaTMP (Full) | **66.36** | **0.4367** | **0.4956** | **3.179** |

### Key Findings
- Modules address separate error sources: H-SVD for weight reconstruction, VaSMP for inter-layer budget, and VaTMP for temporal activation precision. VaTMP is most critical under W4A4 activation constraints.
- Naive mixed precision (optimizing global MSE) can underperform against H-SVD alone, whereas VaSMP's variance-driven closed-form approach is more stable in non-convex PTQ scenarios.
- Some IQA metrics (e.g., MANIQA) show mismatch with human perception in heavily quantized diffusion SR by rewarding sharp noise.

## Highlights & Insights
- **Dual FP Branch Philosophy**: H-SVD explicitly decomposes representation space into "global structure + local texture + residue," leaving the quantizer to handle only fine-grained noise.
- **Data-free Mixed Precision**: Deriving bit-widths from analytic optimization and greedy discretization provides a low-cost, effective baseline for mixed-precision research.
- **Variance as Universal Proxy**: Using variance in both spatio and temporal dimensions, justified by Hadamard-induced Gaussianity, makes for a clean and mathematically sound engineering framework.

## Limitations & Future Work
- Existing no-reference IQA metrics mismatch visual perception for quantized models.
- Strong dependency on Hadamard Gaussian approximation; performance may vary for MoE or heavy-tailed activation networks.
- DP complexity in VaTMP may require pruning for 100+ step samplers.
- Transferability to larger models like SD3 or Flux remains to be verified.

## Related Work & Insights
- **vs SVDQuant (ICLR 2025)**: Adds local block-wise rank-1 branches to preserve high-frequency details lost in global-only SVD.
- **vs PTQ4DiT / Q-DiT**: General DiT schemes lack focus on SR sensitivity, failing under W4A4.
- **vs HAWQ / MixDQ**: VaSMP provides a data-free alternative to compute-intensive mixed-precision methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative H-SVD and VaSMP/VaTMP formulas for DiT SR.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive baselines and ablations, though limited to DiT4SR.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous derivation.
- Value: ⭐⭐⭐⭐ Significant practical speedup (8.99$\times$ layer-wise) and reusable mixed-precision formulas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](../../AAAI2026/image_generation/realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](../../CVPR2026/image_generation/oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)
- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](../../AAAI2026/image_generation/continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[CVPR 2026\] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution](../../CVPR2026/image_generation/framer_frequency-aligned_self-distillation_with_adaptive_modulation_leveraging_d.md)

</div>

<!-- RELATED:END -->
