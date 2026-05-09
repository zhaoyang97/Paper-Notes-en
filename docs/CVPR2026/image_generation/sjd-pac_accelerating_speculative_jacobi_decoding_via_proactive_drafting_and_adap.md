---
title: >-
  [Paper Note] SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation
description: >-
  [CVPR 2026][Image Generation][Autoregressive image generation] This paper analyzes the bottleneck of severely skewed acceptance-length distributions in Speculative Jacobi Decoding (SJD) for text-to-image generation, and proposes the SJD-PAC framework. By introducing two techniques—Proactive Drafting (PD) and Adaptive Continuation (AC)—SJD-PAC achieves a strictly lossless 3.8× inference speedup, substantially surpassing the ~2× acceleration of vanilla SJD.
tags:
  - CVPR 2026
  - Image Generation
  - Autoregressive image generation
  - inference acceleration
  - speculative decoding
  - Jacobi decoding
  - lossless acceleration
date: 2026-05-08
content_hash: 527f5a987ded36ef
---

# SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation

**Conference**: CVPR 2026
**arXiv**: [2603.18599](https://arxiv.org/abs/2603.18599)
**Authors**: Jialiang Kang (Peking University), Han Shu, Wenshuo Li, Yingjie Zhai, Xinghao Chen (Huawei)
**Code**: Unavailable
**Area**: Image Generation
**Keywords**: Autoregressive image generation, inference acceleration, speculative decoding, Jacobi decoding, lossless acceleration

## TL;DR
This paper analyzes the bottleneck of severely skewed acceptance-length distributions in Speculative Jacobi Decoding (SJD) for text-to-image generation, and proposes the SJD-PAC framework. By introducing two techniques—Proactive Drafting (PD) and Adaptive Continuation (AC)—SJD-PAC achieves a strictly lossless 3.8× inference speedup, substantially surpassing the ~2× acceleration of vanilla SJD.

## Background & Motivation

**Background**: Autoregressive (AR) text-to-image models (e.g., Lumina-mGPT, Emu3) have achieved generation quality competitive with diffusion models, but suffer from severe inference latency due to the need to serialize the generation of thousands of tokens. Speculative Decoding (SD) is the dominant approach to accelerating LLM inference, yet it performs poorly in the T2I setting.

**Limitations of Prior Work**: Standard SD methods (e.g., EAGLE) provide almost no speedup on T2I models, because the high-entropy nature of image generation leads to extremely low acceptance rates for draft tokens—even at standard sampling temperatures, many candidate image tokens are nearly equiprobable. Existing SJD methods are training-free and lossless, but deliver only a moderate ~2× speedup.

**Key Challenge**: A detailed analysis reveals that the acceptance-length distribution of SJD is highly skewed—approximately 50% of forward passes accept only a single token (i.e., yield zero acceleration), and the average 2× speedup is driven primarily by a small fraction of steps that successfully accept many tokens. This "long-tail distribution" is the fundamental performance bottleneck.

**Goal**
- How to reduce the frequency of single-token acceptance (inefficient steps)?
- How to increase the number of tokens successfully verified per step?

**Key Insight**: The root cause of single-token acceptance is the "cascading effect of context mismatch"—when position $i$ is rejected, the contexts of all subsequent proposals become invalid. Two complementary directions of optimization are thus needed: (1) providing diverse candidates at the rejection point to reduce subsequent cascading rejections (PD), and (2) continuing verification of subsequent tokens rather than terminating immediately upon rejection (AC).

**Core Idea**: Instead of stopping after a rejection, continue verification and simultaneously draft multiple candidate paths—a two-pronged approach to maximizing per-step acceptance length.

## Method

### Overall Architecture
SJD-PAC modifies the verification loop and drafting strategy of the standard SJD framework. Each iteration proceeds as follows: (1) a single parallel forward pass over the entire draft sequence yields the target distribution $P^t$; (2) an AC loop traverses all $L$ positions for accept/reject decisions without interrupting upon rejection; (3) if any rejection occurs, PD is triggered at the first rejection point to construct a multi-path tree-shaped proposal for the next iteration.

### Key Designs

1. **Adaptive Continuation (AC)**:

    - **Function**: Eliminates the "first-reject break" mechanism in the standard SJD verification loop.
    - **Mechanism**: Standard SJD terminates immediately upon rejection at position $i$, discarding all subsequent tokens $X_{i+1:L}^{t-1}$. AC removes this break—upon rejection at position $i$, it resamples $x_i^t \sim \max(p_i - q_i, 0)$ and then continues applying standard rejection sampling to positions $j > i$. Subsequent tokens that pass verification are retained (accepted); those that do not are also resampled.
    - **Design Motivation**: Image tokens exhibit strong locality. By measuring the Total Variation distance $d_{TV}$ of output distributions under context perturbations at varying distances, the authors find that $d_{TV}$ rapidly approaches zero with increasing distance for image tokens (whereas text tokens remain highly sensitive). This implies that even if the context is modified at position $i$, the target distribution at distant positions changes negligibly. Verification using a stale distribution is therefore effective.
    - **Distinction from Standard SJD**: Standard SJD resamples the entire suffix upon rejection, whereas AC replaces only the individually rejected tokens while retaining accepted subsequent tokens. Each retained token is accepted with probability $1 - d_{TV}(p_i^{t-1}, p_i^t) > 0.7$, far exceeding the probability of resampling the same token from scratch ($< 0.01$, due to the high entropy of image tokens).

2. **Proactive Drafting (PD)**:

    - **Function**: Constructs a multi-path tree-shaped proposal at the rejection point to reduce cascading rejections in subsequent iterations.
    - **Mechanism**: Upon rejection at position $i$, rather than sampling a single continuation, PD constructs a "shallow and wide" tree:
        - **Tree portion** (depth $D=3$, width $K=4$): For positions $i+1$ to $i+D$, $K$ candidate tokens are sampled without replacement from the target distribution $p(\cdot|X_{<j}^{t-1})$ at each position.
        - **Chain portion**: One of the $K$ paths is selected and autoregressively extended to the full length $L$.
    - **Design Motivation**: The context mismatch between the newly resampled $x_i^t$ and subsequent tokens is the root cause of cascading rejections. By providing $K$ diverse choices at the critical post-rejection boundary, the probability that at least one path is valid in the next verification round is increased. The tree is constructed locally at the rejection point and requires no additional model forward passes.
    - **vs. Standard Tree-Based Speculative Decoding**: Standard tree-based SD requires multiple forward passes to construct the tree, whereas PD's tree is built from the current (possibly stale) distribution, incurring only sampling overhead with no additional computational cost.

3. **Synergy of PD and AC**:

    - AC stabilizes the sequence after rejection (more tokens are retained as drafts for the next iteration), while PD provides diverse candidates at the rejection point to increase acceptance probability.
    - Both components strictly maintain losslessness: AC's continued verification uses standard rejection sampling to guarantee distributional correctness, and PD's tree serves only as a draft without affecting the final output distribution.

### Algorithm
The complete algorithm (Algorithm 1): parallel forward pass → AC loop (traverse all $L$ positions, record `first_rej_idx` without breaking) → if rejection occurs, trigger PD at `first_rej_idx` → return new sequence and probabilities.

## Key Experimental Results

### Main Results (Lumina-mGPT, MS-COCO 2017)

| Method | Training-Free? | Lossless? | Step Compression↑ | Latency Speedup↑ | FID↓ | CLIP↑ |
|--------|---------------|-----------|-------------------|------------------|------|-------|
| Vanilla AR | ✓ | ✓ | 1.00× | 1.00× | 30.79 | 31.31 |
| EAGLE | ✗ | ✓ | 2.94× | 2.10× | 30.68 | 31.73 |
| SJD | ✓ | ✓ | 2.22× | 2.05× | 31.13 | 31.33 |
| GSD (lossy) | ✓ | ✗ | 3.39× | 3.62× | 33.12 | 31.25 |
| SJD2 (lossy) | ✗ | ✗ | 4.02× | 2.81× | 31.40 | 31.80 |
| **SJD-PAC** | **✓** | **✓** | **4.51×** | **3.80×** | **30.69** | **31.21** |

SJD-PAC achieves a lossless speedup that surpasses all lossy methods.

### Cross-Model Validation (Emu3, MS-COCO 2017)

| Method | Lossless? | Step Compression↑ | Latency Speedup↑ | FID↓ |
|--------|-----------|-------------------|------------------|------|
| SJD | ✓ | 2.32× | 2.01× | 30.74 |
| SJD2 | ✗ | 5.62× | 2.54× | 31.50 |
| **SJD-PAC** | **✓** | **4.31×** | **3.25×** | **31.10** |

Although SJD2 achieves higher step compression, doubling the window length introduces substantial overhead, resulting in a wall-clock speedup far below that of SJD-PAC.

### Ablation Study

| Configuration | Step Compression↑ | Notes |
|--------------|-------------------|-------|
| SJD baseline ($L=32$) | 2.31× | Vanilla method |
| + PD | 2.71× | Proactive drafting reduces cascading rejections |
| + PD + AC | 3.52× | Adaptive continuation yields substantial gains |
| + PD + AC ($L=64$) | **4.51×** | Larger window fully exploits AC |

### Key Findings
- AC is the dominant component (contributing +0.81× compression ratio vs. +0.40× for PD), as it directly retains more valid tokens.
- Enabling AC makes $L=32$ a bottleneck—tokens stabilize faster, necessitating a larger window ($L=64$) to fully realize the gains.
- The rapid decay of Total Variation distance $d_{TV}$ with distance for image tokens is the key theoretical foundation for AC's effectiveness—in stark contrast to text generation.
- Modifying a single token (0.04% of the total) can introduce severe visual artifacts, demonstrating the necessity of lossless guarantees for T2I generation.
- PD tree parameters $D=3$, $K=4$ represent the sweet spot—too deep wastes sampling budget; too shallow provides insufficient diversity.

## Highlights & Insights
- **Fine-grained analysis of SJD acceptance-length distribution** reveals the insight that "50% of steps contribute 0% speedup." This analysis is independently valuable, providing a clear optimization target for subsequent acceleration methods.
- **AC's exploitation of image token locality** is an elegant observation—the long-range dependencies in text tokens make analogous approaches infeasible for LLMs, whereas the strong locality of image tokens makes stale-distribution verification effective.
- **The orthogonal design of PD and AC** enables independent analysis and combination; this modular design is worth emulating.

## Limitations & Future Work
- Evaluation is limited to Lumina-mGPT and Emu3; generalizability to newer AR T2I models remains unknown.
- Benefits diminish for window sizes $L > 64$ while computational overhead increases—the hardware-specific optimal $L$ is not universally applicable.
- PD's tree is constructed from a stale distribution, which is theoretically less accurate than one built with a full forward pass; this may leave room for improvement under stricter quality requirements.
- Adaptive $D$ and $K$ parameters could be explored—dynamically adjusting tree depth and width based on the local entropy of the current region.

## Related Work & Insights
- **vs. Vanilla SJD**: SJD-PAC modifies the verification loop and drafting strategy, improving speedup from ~2× to 3.8× while remaining training-free and lossless.
- **vs. EAGLE**: EAGLE requires training a draft model and performs poorly on T2I (2.10×); SJD-PAC requires no training and achieves superior acceleration (3.80×).
- **vs. GSD/LANTERN++**: These lossy methods accelerate by relaxing acceptance criteria, potentially introducing visual artifacts. SJD-PAC achieves an even higher speedup than these methods while remaining lossless.
- **vs. SJD2**: SJD2 requires training and is lossy; although its step compression is high, actual wall-clock speedup is low due to large window size. SJD-PAC is more practical.

## Rating
- Novelty: ⭐⭐⭐⭐ AC and PD are individually not particularly novel, but their combination is well-motivated for the high-entropy characteristics of T2I generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two models, two benchmarks, detailed ablations and analysis, though evaluation on larger-scale models is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem analysis is thorough; the logical chain from skewed-distribution observations to the two proposed solutions is exceptionally clear.
- Value: ⭐⭐⭐⭐ Directly practical for AR T2I inference acceleration; the training-free + lossless combination is a strong selling point.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](../../ICCV2025/image_generation/grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](depthvar_depth_adaptive_var.md)
- [\[CVPR 2026\] Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective](accelerating_diffusion_model_training_under_minimal_budgets_a_condensation-based.md)
- [\[CVPR 2026\] D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](d2c_diffusion_dataset_condensation.md)

<!-- RELATED:END -->
