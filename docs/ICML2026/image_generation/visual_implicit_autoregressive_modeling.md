---
title: >-
  [Paper Note] Visual Implicit Autoregressive Modeling
description: >-
  [ICML 2026][Image Generation][VAR] This work embeds Deep Equilibrium (DEQ) implicit fixed-point layers into the next-scale autoregressive framework of VAR…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "VAR"
  - "Deep Equilibrium"
  - "Jacobian-Free Backprop"
  - "Next-scale prediction"
  - "Adaptive Inference"
date: 2026-05-08
content_hash: 504c82d15142aec0
---

# Visual Implicit Autoregressive Modeling

**Conference**: ICML 2026  
**arXiv**: [2605.01220](https://arxiv.org/abs/2605.01220)  
**Code**: https://github.com/mobiushy/VIAR  
**Area**: Image Generation / Autoregressive Generation / Implicit Deep Models  
**Keywords**: VAR, Deep Equilibrium, Jacobian-Free Backprop, Next-scale prediction, Adaptive Inference

## TL;DR
This work embeds Deep Equilibrium (DEQ) implicit fixed-point layers into the next-scale autoregressive framework of VAR, using Jacobian-Free Backpropagation to achieve constant memory training, compressing the 2B parameters of VAR-d30 down to 770M. During inference, the number of iterations per scale becomes a "tunable knob"—on ImageNet-256, FID 2.16/sFID 8.07 are maintained, while peak memory on a single 4090 drops from 19.24GB to 8.53GB and throughput increases from 15.16 to 32.08 img/s.

## Background & Motivation

**Background**: The two mainstream approaches in image generation are diffusion and AR (autoregressive). VAR (Tian et al. 2024) shifts image AR from next-token to **next-scale prediction**—generating multi-scale token maps from coarse to fine, with parallel prediction within each scale, reducing total attention cost from $O(n^6)$ to $O(n^4)$ while preserving spatial locality. It is currently one of the strongest AR image generation paradigms.

**Limitations of Prior Work**: VAR still uses a **deeply stacked explicit Transformer** within each scale (e.g., 30 layers in VAR-d30), leading to three engineering challenges: (1) large parameter count (2B for d30); (2) training memory grows linearly with depth, with activations and optimizer states exploding accordingly; (3) computation depth per scale is fixed at 30, preventing adaptive allocation of compute based on scale size—yet the largest scale (high resolution) is the real bottleneck for KV cache and latency.

**Key Challenge**: Model depth is both the "source of quality" and "source of cost," but in the VAR paradigm, the two are tightly coupled—better quality requires deeper stacks, which inevitably increases memory and latency. Figure 3 shows that on the largest scale, cosine similarity exceeds 0.98 after 5 iterations and approaches 0.999 after 10, indicating that **deep stacking at high-resolution scales is actually over-computation**.

**Goal**: Replace VAR's "fixed-depth stacking" with a module "equivalent to infinite depth but with tunable iteration count," while retaining next-scale parallelism and spatial locality.

**Key Insight**: The Deep Equilibrium Model (DEQ) provides exactly this property—using an implicit fixed-point layer $z^* = f(z^*, x)$ to replace deep stacking, and with Jacobian-Free Backpropagation (JFB), only the last few steps are backpropagated, thus **decoupling training memory from 'effective depth'**. At inference, the number of iterations becomes a tunable knob, allowing a single trained model to simulate networks of different depths.

**Core Idea**: Replace VAR's "first p blocks + deep stack + last p blocks" with "first p blocks + **one implicit equilibrium layer** + last p blocks," retaining shallow transformers as interfaces at both ends, with the fixed-point iteration in the middle acting as "infinite depth." The number of iterations per scale can be scheduled independently.

## Method

### Overall Architecture
At each scale $k$, VIAR proceeds as follows: (1) Input injection: use $p=5$ pre-blocks to project the previous scale output $e_{k-1}$ into $x_k = f_{\text{pre}}(e_{k-1}, c)$; (2) Implicit equilibrium: initialize $z^0 = x_k$, then iteratively update $z^{t+1} = f_{\text{imp}}(\text{Proj}([z^t, x_k]), c)$ until reaching the fixed point $z_k^*$; (3) Post-projection: $\hat{r}_k = f_{\text{post}}(z_k^*, c)$ uses $p=5$ post-blocks to output token predictions. The overall next-scale autoregressive factorization $p(r_1,\cdots,r_K) = \prod_k p(r_k|r_{<k})$ is consistent with VAR, and the VAE tokenizer reuses VAR's multi-scale VQVAE and is frozen.

### Key Designs

1. **Implicit Equilibrium Layer Replacing Explicit Stacking**:

    - **Function**: Collapses the 20+ explicit Transformer middle stack layers in VAR into a single (parameter-shared, recurrently unfolded) fixed-point operator.
    - **Mechanism**: Defines a contractive mapping $f_{\text{imp}}(z, x, c)$ implemented as a Transformer block plus an input injection projection $\text{Proj}([z, x_k])$, solving for the fixed point $z_k^* = f_{\text{imp}}(z_k^*, x_k, c)$. A single trained block can be iterated any number of times at inference—the model's "effective depth" is controlled by the number of iterations at test time, not fixed at training. Figure 6 shows parameter/gradient memory is about 2.87GB (VIAR) vs 7.49GB (VAR-d30), optimizer state 5.74GB vs 14.98GB, saving about 61.6%.
    - **Design Motivation**: Turns "depth" from an architectural hyperparameter into an inference hyperparameter, enabling multi-depth deployment with a single model; also compresses the middle stack parameters to that of a single block (93.3% reduction in middle parameters, 61.6% total parameter reduction).

2. **Stochastic Jacobian-Free Backprop (S-JFB)**:

    - **Function**: Provides stable and low-bias gradients for the implicit layer without storing intermediate activations.
    - **Mechanism**: For each training step, sample $n \sim U\{0, N\}$ steps of "no-gradient" iteration to approach the fixed point, then sample $m \sim U\{1, M\}$ steps of "with-gradient" iteration and only backpropagate through the last $m$ steps ($\partial \mathcal{L}/\partial \theta_{\text{imp}} \approx \sum_k (\partial \mathcal{L}_k/\partial \hat{z}_k) \cdot (\partial \hat{z}_k/\partial \theta_{\text{imp}})|_{\text{last } m}$). The paper defaults to $N=10, M=12$. The shallow blocks before and after use standard backprop, while the implicit block uses S-JFB.
    - **Design Motivation**: Pure 1-step JFB has high bias, while full unrolling is infeasible. Random multi-step JFB approximates the true gradient in expectation with constant memory. The authors note that too large $m$ can harm stability (when the operator's local Lipschitz constant is large), with the sweet spot at moderate $m$.

3. **Cross-scale Adaptive Iteration Scheduling**:

    - **Function**: Reallocates compute at inference based on scale size—more iterations for coarse scales to stabilize global structure, fewer for fine scales to save KV cache.
    - **Mechanism**: Defines a total budget $\mathcal{C} = \sum_k (p_{\text{pre}} + c_k + p_{\text{post}})$, allowing different schedules $\{c_k\}$: constant $\text{Con.}_{(c,c)}$, decreasing $\text{Dec.}_{(a,b)}$ (coarse scale $a$ times, fine scale $b$ times), or adaptive threshold control $\|G(z) - z\|_2 \le \tau_k$. Convergence analysis in Figure 3 shows that on the largest scale, 5 iterations yield cosine similarity >0.98, 10 iterations approach 0.999, so the number of iterations at high-resolution scales can be safely reduced. Experiments show that when high-resolution scales have not converged, **increasing iterations at coarse scales** continues to improve FID, indicating that global structure has a greater impact on detail quality than local self-iteration.
    - **Design Motivation**: Addresses the fundamental issue of uneven compute allocation across scales in VAR—the high-resolution scale has the largest KV cache and parallel token count, but actually converges fastest. Treating iteration as a resource to be scheduled is more flexible than fixing depth at training.

### Loss & Training
Standard next-scale cross-entropy $\mathcal{L} = -\sum_k \log p_\theta(\hat{r}_k|r_{<k})$. The implicit layer uses S-JFB, while the shallow layers before and after use standard backprop. Global batch size is 512, learning rate 8e-5, and other optimizer/scheduler settings follow VAR. The tokenizer is frozen. The backbone is the 2B-parameter VAR-d30 VQVAE plus custom pre/imp/post structures.

## Key Experimental Results

### Main Results
Class-conditional generation on ImageNet 256×256, 50K samples compared for FID/sFID/IS/Precision/Recall.

| Model | FID ↓ | sFID ↓ | IS ↑ | Pre ↑ | Rec ↑ | #Params | Inference Mem |
|-------|-------|--------|------|-------|-------|---------|--------------|
| VAR-d30 (cfg=2.0) | 2.05 | 8.86 | 328.5 | 0.82 | 0.59 | 2010M | 19.24GB |
| VAR-d30 (cfg=1.5) | 2.08 | 8.82 | 306.8 | 0.82 | 0.59 | 2010M | 19.24GB |
| VIAR (cfg=2.0) | 2.35 | **7.92** | **330.7** | **0.83** | 0.58 | **770.9M** | 11.16GB |
| VIAR (cfg=1.5) | **2.16** | 8.07 | 300.1 | 0.81 | 0.59 | **770.9M** | 11.16GB |

VIAR uses only 38.4% of the parameters (770.9M vs 2010M), with FID dropping by only 0.08 (2.16 vs 2.08), and sFID even better (7.92 vs 8.86), indicating even better spatial structure.

### Ablation Study
4090 single-card throughput and memory comparison (varying schedule aggressiveness, s1 most conservative, s4 most aggressive):

| Method | FID ↓ | sFID ↓ | Mem (GB) ↓ | Throughput (img/s) ↑ |
|--------|-------|--------|------------|----------------------|
| VAR | 2.08 | 8.82 | 19.24 | 15.16 |
| VIAR_s1 | 2.16 | 8.07 | 11.16 | 21.50 |
| VIAR_s2 | 2.22 | 8.08 | 9.60 | 26.92 |
| VIAR_s3 | 2.27 | 8.02 | 9.40 | 28.12 |
| VIAR_s4 | 2.43 | 8.28 | **8.53** | **32.08** |

The most aggressive s4 achieves 2.1× speedup and 2.26× memory reduction, with FID dropping by only 0.35.

Cross-scale scheduling (coarse-fine iterations):

| Schedule | FID ↓ | sFID ↓ | IS ↑ |
|----------|-------|--------|------|
| Dec.(20, 5) | 2.18 | 8.04 | 299.2 |
| Dec.(20, 10) | 2.16 | 8.07 | 294.8 |
| Dec.(10, 5) | 2.22 | 8.08 | 303.4 |
| Con.(20, 20) | 2.16 | 8.17 | 294.8 |
| Con.(5, 5) | 2.27 | 8.02 | 307.1 |
| Con.(10, 10) | **2.16** | 8.07 | 300.1 |

### Key Findings
- **More iterations at coarse scales > more at fine scales**: When fine scales have not converged, increasing coarse scale iterations improves FID more (Dec.(20,5) vs Con.(5,5)), suggesting global structure is the bottleneck for detail quality.
- **Constant training memory**: Figure 6 shows VIAR memory is almost flat with increasing "effective depth" (plateau ~2.87GB), while VAR memory grows nearly linearly with depth.
- **Enhanced zero-shot editing**: Figure 7 shows VIAR achieves smoother boundary fusion and sharper details in in-painting and class-conditional editing, attributed to the implicit layer's "long-range context aggregation" being more stable than fixed depth.
- **Fixed-point iteration converges extremely fast**: On the largest scale, 5 iterations yield cosine similarity 0.985, 10 iterations 0.999, providing the physical basis for saving compute.

## Highlights & Insights
- Successfully runs DEQ on large-scale generation tasks—for the first time, VIAR demonstrates on ImageNet-scale AR image generation that "implicit layers can replace deep stacking without loss," which is of greater engineering than theoretical significance.
- **"Train once, infer at multiple depths"**: A single trained VIAR model can run at different iteration counts under different hardware budgets, effectively providing a family of "small-medium-large" models for free; this elasticity is highly valuable for edge deployment.
- The idea of making "depth vs compute" a continuous knob can be transferred to diffusion (in fact, Bai & Melas-Kyriazi 2024 are already doing this), to intermediate layers of long-context LLMs (implicitizing each transformer block), or even neural rendering (per-pixel adaptive iteration).
- The conclusion on cross-scale compute reallocation (coarse scales matter more) is counterintuitive—one would expect high-resolution scales to require more compute due to denser information, but experiments show they converge faster, while coarse scales set the global structure. This insight is valuable for all "hierarchical generation" architectures.

## Limitations & Future Work
- The main result FID 2.16 is slightly inferior to VAR-d30's 2.05; the authors did not verify whether VIAR maintains the parameter ratio at larger model sizes (d36+); DEQ stability at larger scales remains an open question.
- S-JFB is a biased estimator; although empirically stable, it theoretically requires certain Lipschitz properties of $f_{\text{imp}}$; the paper does not provide convergence guarantees or local stability analysis.
- Only tested on ImageNet 256×256; performance on 512×512 or text-to-image (e.g., replacing VIAR's backbone with LlamaGen / MAR) is unknown; whether DEQ's iteration count explodes with resolution is a real test.
- Adaptive $\tau$ threshold control is only sketched in the paper, with no systematic ablation; in real deployment, the design of the "when to stop" controller remains unaddressed.
- Although the number of inference iterations is tunable, each iteration still requires a full Transformer block pass; whether FlashAttention/Triton optimizations can further reduce latency is not discussed.

## Related Work & Insights
- **vs VAR (Tian et al. 2024)**: Direct baseline, retaining the next-scale paradigm and only replacing the middle stack with an implicit layer, making it a truly "plug-in" modification; nearly all subsequent VAR works (CAR, speculative VAR) can be seamlessly migrated.
- **vs Fixed-Point Diffusion (Bai & Melas-Kyriazi 2024)**: They insert a DEQ layer into the diffusion denoiser for compute reallocation over timesteps; VIAR applies the same idea to spatial scales, making the approaches complementary rather than competitive.
- **vs Pokle et al. 2022 (diffusion trajectory as DEQ)**: Coarser granularity, solving the entire reverse trajectory at once; VIAR applies DEQ within a single scale, offering finer granularity and more tunability.
- **vs Collaborative Decoding (Chen et al. 2025b) / Cached-token Pruning (Guo et al. 2025)**: These are VAR decoding-side acceleration methods (saving KV cache or parallel decoding), orthogonal to VIAR's "structural" acceleration and theoretically stackable.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to graft DEQ onto VAR, with solid engineering of JFB; though the DEQ+AR combination is not entirely new in NLP/diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Main results + 5 scheduling strategies + training memory curves + zero-shot editing + convergence analysis are all present; lacks scaling to larger models and 512×512 validation.
- Writing Quality: ⭐⭐⭐⭐ — Figure 1 clearly illustrates resource savings, and the method section is formulaic and clear; however, the S-JFB algorithm description may be challenging for readers unfamiliar with DEQ.
- Value: ⭐⭐⭐⭐⭐ — "Train once, infer at multiple depths" + "61.6% parameter reduction" + "almost no FID drop" are of direct industrial value, highly significant for edge deployment and elastic inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](../../AAAI2026/image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)
- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](../../CVPR2026/image_generation/depthvar_depth_adaptive_var.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[ICLR 2026\] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](../../ICLR2026/image_generation/ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)

</div>

<!-- RELATED:END -->
