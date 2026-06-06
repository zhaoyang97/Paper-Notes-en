---
title: >-
  [Paper Note] Visual Implicit Autoregressive Modeling
description: >-
  [ICML 2026][Image Generation][VAR] This paper integrates Deep Equilibrium (DEQ) implicit fixed-point layers into the next-scale autoregressive framework of VAR. By employing Jacobian-Free Backpropagation to achieve const…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "VAR"
  - "Deep Equilibrium"
  - "Jacobian-Free Backprop"
  - "Next-scale prediction"
  - "Adaptive Inference"
date: 2026-05-08
content_hash: 4e7ee8b6e590a430
---

# Visual Implicit Autoregressive Modeling

**Conference**: ICML 2026  
**arXiv**: [2605.01220](https://arxiv.org/abs/2605.01220)  
**Code**: https://github.com/mobiushy/VIAR  
**Area**: Image Generation / Autoregressive Generation / Implicit Deep Models  
**Keywords**: VAR, Deep Equilibrium, Jacobian-Free Backprop, Next-scale prediction, Adaptive Inference

## TL;DR
This paper integrates Deep Equilibrium (DEQ) implicit fixed-point layers into the next-scale autoregressive framework of VAR. By employing Jacobian-Free Backpropagation to achieve constant memory training, the authors compress the 2 billion parameters of VAR-d30 to 770 million. Meanwhile, the iteration count for each scale is transformed into an "adjustable knob" during inference. On ImageNet-256, while maintaining FID 2.16/sFID 8.07, the peak GPU memory on a single 4090 is reduced from 19.24GB to 8.53GB, and throughput is increased from 15.16 to 32.08 img/s.

## Background & Motivation

**Background**: The two mainstream paradigms for image generation are diffusion and AR (Autoregressive). VAR (Tian et al. 2024) shifted image AR from next-token to **next-scale prediction**—generating multi-scale token maps from coarse to fine. By predicting tokens in parallel within each scale, it reduces the total attention overhead from $O(n^6)$ to $O(n^4)$ while preserving spatial locality, making it one of the most powerful AR image generation paradigms currently available.

**Limitations of Prior Work**: VAR still utilizes a **deeply stacked explicit Transformer** within each scale (e.g., 30 layers for VAR-d30). This presents three engineering challenges: (1) Large parameter count, with the d30 model having 2 billion parameters; (2) Training memory grows linearly with depth as activations and optimizer states increase; (3) The computational depth for each scale is fixed at 30, preventing adaptive allocation of compute resources based on scale size—even though the largest scale (high resolution) is the primary bottleneck for KV cache and latency.

**Key Challenge**: Model depth is both the "source of quality" and the "source of overhead." In the VAR paradigm, these two are coupled—improving quality requires increasing depth, which inevitably leads to higher memory and latency. Furthermore, Figure 3 shows that on the largest scale, the cosine similarity exceeds 0.98 after 5 iterations and approaches 0.999 after 10, implying that **deep stacking on high-resolution scales constitutes redundant computation**.

**Goal**: Replace the "fixed-depth stacking" of VAR with a module that is "equivalent to infinite depth but has adjustable iterations," while preserving the parallelism and spatial locality of next-scale prediction.

**Key Insight**: Deep Equilibrium Models (DEQ) provide exactly this property—using an implicit fixed-point layer $z^* = f(z^*, x)$ to replace deep stacking. Combined with Jacobian-Free Backprop (JFB), which backpropagates only through the final steps, it decouples **training memory from "equivalent depth."** During inference, the number of iterations becomes an adjustable knob, allowing a single trained model to simulate networks of various depths.

**Core Idea**: Replace the "first p blocks + deep stack + last p blocks" structure of VAR with "first p blocks + **one implicit equilibrium layer** + last p blocks." The shallow transformers are kept as interfaces at both ends, while the fixed-point iteration in the middle handles the "infinite depth." The number of iterations for each scale can be scheduled independently.

## Method

### Overall Architecture
The process of VIAR on each scale $k$ is as follows: (1) Input Injection: $p=5$ pre-blocks project the output of the previous scale $e_{k-1}$ into $x_k = f_{\text{pre}}(e_{k-1}, c)$; (2) Implicit Equilibrium: Initialized from $z^0 = x_k$, it repeatedly iterates $z^{t+1} = f_{\text{imp}}(\text{Proj}([z^t, x_k]), c)$ until the fixed point $z_k^*$ is reached; (3) Post-projection: $\hat{r}_k = f_{\text{post}}(z_k^*, c)$ uses $p=5$ post-blocks to output token predictions. The factorization of the next-scale autoregression $p(r_1,\cdots,r_K) = \prod_k p(r_k|r_{<k})$ remains consistent with VAR. The VAE tokenizer reuses the multi-scale VQVAE from VAR and is frozen.

### Key Designs

1.  **Function: Implicit Equilibrium Layer Replacing Explicit Stacking**:
    - **Mechanism**: Collapses the explicit Transformer middle stack of 20+ layers in VAR into a single-layer fixed-point operator (parameter sharing, recurrent unfolding).
    - **Core Idea**: Defines a contractive mapping $f_{\text{imp}}(z, x, c)$ implemented as a Transformer block plus an input injection projection $\text{Proj}([z, x_k])$, solving for the fixed point $z_k^* = f_{\text{imp}}(z_k^*, x_k, c)$. A trained block can iterate any number of times during inference—the "equivalent depth" is controlled by the iteration count at test time rather than being fixed during training. Figure 6 shows that parameter/gradient memory is approximately 2.87GB (VIAR) vs 7.49GB (VAR-d30), and optimizer states are 5.74GB vs 14.98GB, a saving of about 61.6%.
    - **Design Motivation**: Transforms "depth" from an architectural hyperparameter into an inference hyperparameter, enabling the deployment of multiple depths from a single model. Simultaneously reduces middle stack parameters to the equivalent of one block (93.3% middle parameter reduction, 61.6% total parameter reduction).

2.  **Function: Stochastic Jacobian-Free Backprop (S-JFB)**:
    - **Mechanism**: Provides stable and low-bias gradients for the implicit layer without storing intermediate activations.
    - **Core Idea**: For each training step, $n \sim U\{0, N\}$ "gradient-free" iterations are sampled to approach the fixed point, followed by $m \sim U\{1, M\}$ "gradient-aware" iterations where backpropagation occurs only through the last $m$ steps ($$\partial \mathcal{L}/\partial \theta_{\text{imp}} \approx \sum_k (\partial \mathcal{L}_k/\partial \hat{z}_k) \cdot (\partial \hat{z}_k/\partial \theta_{\text{imp}})|_{\text{last } m}$$). The defaults are $N=10, M=12$. Standard backprop is used for shallow blocks, while S-JFB is used for the implicit block.
    - **Design Motivation**: Pure 1-step JFB has high bias, and full unfolding is memory-prohibitive. Stochastic multi-step JFB approximates the true gradient in expectation while maintaining constant memory. The authors note that an excessively large $m$ can compromise stability (when the operator's local Lipschitz constant is large), identifying a sweet spot at moderate $m$.

3.  **Function: Cross-scale Adaptive Iteration Scheduling**:
    - **Mechanism**: Reallocates computational resources during inference based on scale size—using more iterations for coarse scales to stabilize global structure and fewer for fine scales to save KV cache.
    - **Core Idea**: Defines a total budget $\mathcal{C} = \sum_k (p_{\text{pre}} + c_k + p_{\text{post}})$. Different schedules $\{c_k\}$ can be chosen: Constant $\text{Con.}_{(c,c)}$, Decreasing $\text{Dec._{(a,b)}}$ (a iterations for coarse scales, b for fine), or adaptive threshold control $\|G(z) - z\|_2 \le \tau_k$. Convergence analysis in Figure 3 shows that on the largest scale, 5 iterations yield a cosine similarity >0.98, and 10 iterations approach 0.999, allowing for a bold reduction in iterations for high-resolution scales. The experiments find that when high resolution has not converged, **adding iterations to coarse scales** leads to continuous FID improvement, suggesting global structure influences detail quality more than detail-level self-iteration.
    - **Design Motivation**: Addresses the fundamental issue of uneven resource allocation in VAR—high-resolution scales have the largest KV cache and token count but converge fastest. Scheduling iterations as a resource is more flexible than fixing depth during training.

### Loss & Training
Standard next-scale cross-entropy $\mathcal{L} = -\sum_k \log p_\theta(\hat{r}_k|r_{<k})$. Implicit layers use S-JFB, while shallow layers use standard backprop. Global batch size 512, lr 8e-5, with other optimizers/schedules following VAR. Tokenizer is frozen. The base uses the VQVAE from the 2B-parameter VAR-d30 with the custom pre/imp/post architecture.

## Key Experimental Results

### Main Results
Class-conditional generation on ImageNet 256×256. Comparison of FID/sFID/IS/Precision/Recall using 50K samples.

| Model | FID ↓ | sFID ↓ | IS ↑ | Pre ↑ | Rec ↑ | #Params | Inference Memory |
|------|-------|--------|------|-------|-------|---------|----------|
| VAR-d30 (cfg=2.0) | 2.05 | 8.86 | 328.5 | 0.82 | 0.59 | 2010M | 19.24GB |
| VAR-d30 (cfg=1.5) | 2.08 | 8.82 | 306.8 | 0.82 | 0.59 | 2010M | 19.24GB |
| VIAR (cfg=2.0) | 2.35 | **7.92** | **330.7** | **0.83** | 0.58 | **770.9M** | 11.16GB |
| VIAR (cfg=1.5) | **2.16** | 8.07 | 300.1 | 0.81 | 0.59 | **770.9M** | 11.16GB |

Ours uses only 38.4% parameters (770.9M vs 2010M), while FID drops by only 0.08 (2.16 vs 2.08) and sFID actually improves (7.92 vs 8.86), indicating superior spatial structure.

### Ablation Study
Comparison of throughput and memory on a single 4090 card (varying schedule aggressiveness, s1 is most conservative, s4 is most aggressive):

| Method | FID ↓ | sFID ↓ | Memory (GB) ↓ | Throughput (img/s) ↑ |
|------|-------|--------|------|----------|
| VAR | 2.08 | 8.82 | 19.24 | 15.16 |
| VIAR_s1 | 2.16 | 8.07 | 11.16 | 21.50 |
| VIAR_s2 | 2.22 | 8.08 | 9.60 | 26.92 |
| VIAR_s3 | 2.27 | 8.02 | 9.40 | 28.12 |
| VIAR_s4 | 2.43 | 8.28 | **8.53** | **32.08** |

The most aggressive s4 achieves a 2.1× speedup and a 2.26× memory reduction, with FID only dropping by 0.35.

Cross-scale scheduling (coarse-to-fine iterations):

| Schedule | FID ↓ | sFID ↓ | IS ↑ |
|------|-------|--------|------|
| Dec.(20, 5) | 2.18 | 8.04 | 299.2 |
| Dec.(20, 10) | 2.16 | 8.07 | 294.8 |
| Dec.(10, 5) | 2.22 | 8.08 | 303.4 |
| Con.(20, 20) | 2.16 | 8.17 | 294.8 |
| Con.(5, 5) | 2.27 | 8.02 | 307.1 |
| Con.(10, 10) | **2.16** | 8.07 | 300.1 |

### Key Findings
- **Coarse scale iterations > Fine scale iterations**: When fine scales have not converged, increasing coarse scale iterations improves FID more effectively (Dec.(20,5) vs Con.(5,5)), suggesting global structure is the bottleneck for quality.
- **Constant Training Memory**: Figure 6 shows VIAR memory remains nearly constant regardless of "equivalent depth" (plateau ~2.87GB), while VAR memory scales linearly with depth.
- **Enhanced Zero-shot Editing**: Figure 7 shows VIAR achieves smoother boundary fusion and sharper details in in-painting and class-conditional editing, attributed to the "long-range context aggregation" of implicit layers being more stable than fixed depth.
- **Fast Fixed-Point Convergence**: Reaches a cosine similarity of 0.985 at 5 iterations and 0.999 at 10 on the largest scale, providing the physical basis for reducing iteration compute.

## Highlights & Insights
- Successfully scales DEQ on large-scale generation tasks. While DEQ was previously used mostly for demos in classification or optical flow, VIAR proves that implicit layers can replace deep stacks in ImageNet-level AR generation without performance loss.
- **"Train once, infer at multiple depths"**: A single trained VIAR model can run different iteration counts depending on hardware budgets, effectively providing a family of "small-medium-large" models for free; this elasticity is invaluable for edge deployment.
- The concept of turning "depth vs. computation" into a continuous knob can be transferred to diffusion (already being explored by Bai & Melas-Kyriazi 2024), middle layers of long-context LLMs, or even neural rendering.
- The conclusion regarding cross-scale resource reallocation (coarse scales being more important) is counter-intuitive—one might expect high-resolution scales to require more compute due to information density, but experiments show they converge fast while coarse scales set the tone for global structure.

## Limitations & Future Work
- The main FID of 2.16 is slightly inferior to VAR-d30's 2.05. The authors have not verified if VIAR maintains the parameter ratio on larger models (d36+); the stability of DEQ at larger scales remains an open question.
- S-JFB is a biased estimator. Although it is stable in experiments, it theoretically requires specific Lipschitz properties of $f_{\text{imp}}$. The paper lacks theoretical analysis on convergence guarantees or local stability.
- Only tested on ImageNet 256×256. Performance on 512×512 or text-to-image (e.g., replacing VIAR with LlamaGen / MAR bases) is unknown; whether DEQ iteration counts surge with resolution is the true test.
- The adaptive $\tau$ threshold control is only sketched and lacks systematic ablation. The design of a "controller" for stopping criteria in practical deployment is still missing.
- Although iterations are adjustable during inference, each iteration still traverses the entire Transformer block. There is no discussion on whether FlashAttention/Triton optimizations could further reduce latency.

## Related Work & Insights
- **vs VAR (Tian et al. 2024)**: The direct baseline. Maintains the next-scale paradigm while only replacing the middle stack with an implicit layer. This is a "plug-and-play" modification; almost all follow-up work on VAR (CAR, speculative VAR) can migrate seamlessly.
- **vs Fixed-Point Diffusion (Bai & Melas-Kyriazi 2024)**: They insert DEQ layers into the diffusion denoiser for compute reallocation across timesteps; VIAR applies the same logic to spatial scales, making them complementary.
- **vs Pokle et al. 2022 (Diffusion trajectory as DEQ)**: Coarser granularity where the entire reverse trajectory is solved at once; VIAR is DEQ within a single scale, allowing finer adjustment.
- **vs Collaborative Decoding (Chen et al. 2025b) / Cached-token Pruning (Guo et al. 2025)**: These are acceleration schemes for the decoding side (saving KV cache or parallel decoding). They are orthogonal to the "structural" acceleration of VIAR and can theoretically be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ — Grafting DEQ onto VAR is a first, and the engineering of JFB is solid; however, the combination of DEQ and AR is not a brand-new concept in NLP/diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Main results, 5 scheduling types, training memory curves, zero-shot editing, and convergence analysis are all present; lacks scaling to larger models and 512×512 validation.
- Writing Quality: ⭐⭐⭐⭐ — Figure 1 summarizes resource savings effectively, and the method section has clear formulas; however, the S-JFB algorithm description might have a high barrier for readers unfamiliar with DEQ.
- Value: ⭐⭐⭐⭐⭐ — "Train once for multiple inference depths," "61.6% parameter reduction," and "minimal FID loss" offer genuine industrial value, significant for edge deployment and elastic inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](../../AAAI2026/image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)
- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](../../CVPR2026/image_generation/depthvar_depth_adaptive_var.md)
- [\[ICML 2026\] Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models](compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)

</div>

<!-- RELATED:END -->
