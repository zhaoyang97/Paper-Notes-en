---
title: >-
  [Paper Note] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation
description: >-
  [AAAI 2026][Image Generation][speculative decoding] This paper proposes Cool-SD, a theoretically grounded annealed relaxation framework for speculative decoding. By deriving a tight upper bound on the TV distance, it obtains the optimal resampling distribution and proves that a decreasing acceptance probability schedule yields smaller distributional shift than a uniform schedule. Cool-SD achieves a superior speed–quality trade-off over LANTERN++ on LlamaGen and Lumina-mGPT.
tags:
  - AAAI 2026
  - Image Generation
  - speculative decoding
  - autoregressive image generation
  - annealed relaxation
  - total variation distance
  - inference acceleration
date: 2026-05-08
content_hash: 0e9944bca88c854f
---

# Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation

**Conference**: AAAI 2026
**arXiv**: [2601.09212v1](https://arxiv.org/abs/2601.09212v1)
**Code**: [https://github.com/xingyaoL/COOL-SD](https://github.com/xingyaoL/COOL-SD)
**Area**: Image Generation / Efficient Inference
**Keywords**: speculative decoding, autoregressive image generation, annealed relaxation, total variation distance, inference acceleration

## TL;DR
This paper proposes Cool-SD, a theoretically grounded annealed relaxation framework for speculative decoding. By deriving a tight upper bound on the TV distance, it obtains the optimal resampling distribution and proves that a decreasing acceptance probability schedule yields smaller distributional shift than a uniform schedule. Cool-SD achieves a superior speed–quality trade-off over LANTERN++ on LlamaGen and Lumina-mGPT.

## Background & Motivation
Autoregressive (AR) image generation models (e.g., LlamaGen, Lumina-mGPT) achieve quality comparable to diffusion models, but token-by-token generation makes inference slow. Speculative Decoding (SD) is a well-established acceleration technique in the LLM domain: a small draft model generates a candidate token sequence, which is then verified in parallel by the target model. However, SD is less effective for image generation due to the **semantic ambiguity of image tokens** — many positions have multiple candidate tokens with similar probabilities, making it difficult for the draft model to match the target model's selection, resulting in low acceptance rates.

LANTERN and LANTERN++ improve acceptance rates by relaxing the verification criterion (allowing neighboring tokens to be accepted), but both designs are heuristic and **lack a theoretical basis** to bound the distributional deviation introduced by relaxation. Moreover, their resampling distributions are suboptimal, making it difficult to control distributional shift.

## Core Problem
**How can the verification criterion in speculative decoding be relaxed to improve acceptance rates while minimizing, with theoretical guarantees, the deviation between the output distribution and the target distribution?** Two sub-problems must be addressed simultaneously: given a relaxed acceptance criterion, what is the optimal resampling distribution? And how should the acceptance probability vary across positions?

## Method

### Overall Architecture
Cool-SD introduces two modifications to the standard SD framework:
- **Input**: target model $P$, draft model $Q$, prefix, draft length $L$
- **Modification 1**: replaces the acceptance probability $f_i$ in vanilla SD's strict matching with a version scaled by a relaxation factor $\omega_i$ that decays exponentially
- **Modification 2**: when a token is rejected, uses the derived optimal resampling distribution $G_i^*$ to correct the bias
- **Output**: a generated token sequence that approximates the target distribution with longer accepted lengths

### Key Designs

1. **TV Distance Upper Bound and Optimal Resampling Distribution (Theorem 2)**: For any relaxed acceptance criterion $\{f_i\}_{i=1}^L$, a nearly tight upper bound on the TV distance between the output distribution $\hat{P}$ and the target distribution $P$ is derived. Minimizing this bound yields the optimal resampling distribution at each position: $G_{i+1}^* = \text{Norm}([P(\cdot|x_{1:i}) - Q(\cdot|x_{1:i}) \cdot f_{i+1}((x_{1:i}, \cdot), P, Q)]_+)$. When the acceptance probability is no less than that of vanilla SD, this optimal distribution coincides exactly with vanilla SD's resampling distribution (Proposition 1), yielding an elegant formulation.

2. **Annealing Property and Exponential Decay Schedule (Proposition 2)**: A perturbation analysis establishes an annealing property: under a fixed expected number of accepted tokens, **relaxing more at earlier positions and less at later positions** yields a smaller TV distance upper bound than the reverse. Based on this, an exponential decay schedule $\omega_i = \delta \cdot \exp(-\nu \cdot i - \mu)$ is designed, where $\delta$ controls the total relaxation and $\nu = 0.7$ controls the decay rate.

3. **Parameterized Acceptance Criterion**: $f_i(x_{1:i}, P, Q; \omega_i) = \min\{1, \omega_i \cdot P(\tilde{x}_i | x_{1:i-1}) / Q(\tilde{x}_i | x_{1:i-1})\}$, where $\omega_i > 1$ relaxes the acceptance condition and $\omega_i = 1$ recovers vanilla SD. $\delta$ is the only hyperparameter to tune, controlling the speed–quality trade-off.

### Loss & Training
Cool-SD requires no additional training; it only modifies the verification and resampling strategy of SD. Draft model training follows the Eagle procedure. The implementation is simpler than LANTERN++, requiring no nearest-neighbor probability retrieval or aggregation.

## Key Experimental Results

| Target Model | Method | CLIP↑ | FID↓ | IR↑ | Accept Len↑ | Latency (s)↓ | Speedup↑ |
|---|---|---|---|---|---|---|---|
| Lumina-mGPT (7B) | Target Model | 0.3330 | 28.99 | 0.6855 | 1.00 | 170.14 | 1.00× |
| Lumina-mGPT | Eagle-1 (no relaxation) | 0.3330 | 29.05 | 0.6883 | 2.76 | 71.66 | 2.37× |
| Lumina-mGPT | LANTERN++ | 0.3328 | 30.31 | 0.6697 | 2.99 | 68.64 | 2.48× |
| Lumina-mGPT | Cool-SD (δ=1.1) | 0.3325 | 30.30 | 0.6699 | 3.11 | 63.24 | **2.69×** |
| LlamaGen-XL (775M) | Eagle-1 | 0.3157 | 20.97 | -0.0859 | 2.42 | 4.99 | 2.03× |
| LlamaGen-XL | LANTERN++ | 0.3157 | 21.17 | -0.1155 | 2.67 | 4.70 | 2.15× |
| LlamaGen-XL | Cool-SD (δ=1.1) | 0.3167 | 21.02 | -0.0997 | 2.73 | 4.46 | 2.27× |
| LlamaGen-XL | Cool-SD (δ=2) | 0.3154 | 21.20 | -0.1353 | 3.34 | 3.72 | **2.72×** |

### Ablation Study
- **Annealed vs. uniform relaxation**: Cool-SD (exponential decay) uniformly dominates UniformRSD (uniform relaxation) on the FID–acceptance length curve, validating the practical benefit of the annealing property.
- **Effect of $G_i^*$ on LANTERN++**: Applying $G_i^*$ to LANTERN++ improves its performance across different $k$ settings, with larger improvements at larger $k$ (since LANTERN++'s own resampling bias grows with $k$).
- **Linear vs. exponential decay**: The exponential schedule outperforms the linear schedule on LlamaGen; the two are comparable on Lumina-mGPT, and both outperform the uniform schedule.
- **Comparison with SJD**: Cool-SD (δ=1.1) achieves 2.69× speedup on Lumina-mGPT vs. SJD's 2.06×, with comparable quality.
- **Statistical significance**: The difference in acceptance length between Cool-SD and LANTERN++ over 1,000 samples yields $t=18.11, p < 10^{-60}$.

## Highlights & Insights
- **Seamless integration of theory and practice**: Rather than post-hoc justifying a design with theory, the optimal resampling distribution is derived directly from minimizing the TV distance upper bound, and the annealing property emerges from perturbation analysis — theory directly drives design.
- **Plug-and-play optimal resampling distribution**: $G_i^*$ can directly improve existing methods such as LANTERN++, serving as a general drop-in component.
- **Theoretical foundation for the annealing intuition**: The "relax early, tighten late" acceptance strategy has intuitive appeal (early positions have broader influence; relaxing late positions yields diminishing returns), and the paper provides a rigorous proof.
- **Simpler implementation**: Unlike LANTERN++, which requires building $k$-nearest-neighbor structures and aggregating probabilities, Cool-SD only modifies the acceptance threshold and resampling distribution, incurring virtually no additional overhead.

## Limitations & Future Work
- **Hyperparameter $\delta$ still requires manual tuning**: Although only one hyperparameter is involved, its optimal value varies across models and applications, precluding adaptive selection.
- **Rigorous proof of the annealing property is limited to $L=2$**: Although the paper claims the result extends to $L > 2$ via pairwise comparisons, the formal proof covers only the two-step case.
- **Assumes draft and target distributions are close** (Assumption 2: TV ≤ 2/5): Theoretical guarantees may weaken when the draft model quality is poor.
- **Evaluated only on image generation**: While the theoretical framework applies to any AR task, experiments are conducted solely on image generation; effectiveness on LLM text generation remains unknown.
- **Quality degrades as $\delta$ increases**: Image quality noticeably deteriorates at high speedup ratios (>3.5×), precluding unbounded acceleration.

## Related Work & Insights

| Method | Core Idea | Key Difference from Cool-SD |
|---|---|---|
| **Vanilla SD (Eagle-1)** | Unbiased decoding via strict matching | Cool-SD trades controlled relaxation for higher speedup, with a theoretical bound on the degree of bias |
| **LANTERN** | Probability aggregation relaxation via $k$-nearest neighbors | Heuristic design with no theoretical guarantee; resampling distribution is suboptimal |
| **LANTERN++** | LANTERN + static tree structure | Cool-SD has stronger theoretical grounding, simpler implementation (no neighbor retrieval), and a superior Pareto frontier |
| **SJD** | Jacobi decoding + speculative | Lossless but limited speedup (2.06× vs. Cool-SD's 2.69×) |

## Related Inspirations
- The derivation of the optimal resampling distribution is transferable to other approximate sampling scenarios (e.g., resampling in guided diffusion).
- The annealing schedule shares conceptual spirit with simulated annealing and temperature scheduling in optimization — "relax early, tighten late" is a broadly applicable heuristic.
- The theoretical framework applies to speculative decoding acceleration for any AR model, not limited to image generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The TV distance upper bound derivation and proof of the annealing property constitute substantive theoretical contributions, though the overall direction of improvement is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two target models, multiple baselines, and comprehensive ablation analysis; however, validation beyond image generation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous, experimental presentation is clear, and the transition from theory to practice is natural.
- **Value**: ⭐⭐⭐⭐ Provides the first theoretical framework for relaxed SD; the plug-and-play optimal resampling distribution offers direct value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](../../ICCV2025/image_generation/grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[CVPR 2026\] SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation](../../CVPR2026/image_generation/sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)
- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](../../ICLR2026/image_generation/autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[ICLR 2026\] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](../../ICLR2026/image_generation/locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)
- [\[ICLR 2026\] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](../../ICLR2026/image_generation/from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)

</div>

<!-- RELATED:END -->
