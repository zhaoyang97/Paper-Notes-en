---
title: >-
  [Paper Note] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving
description: >-
  [CVPR 2026][Multimodal VLM][Multi-view VLM] Prune2Drive is the first plug-and-play token pruning framework designed for multi-view autonomous driving VLMs. It combines T-FPS (Token-wise Farthest Point Sampling) to preserve semantic and spatial diversity with view-adaptive pruning rate optimization to automatically allocate token budgets across camera views. Retaining only 10% of tokens on DriveLM, it achieves 6.40× prefill speedup with only a 3% performance drop.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multi-view VLM
  - visual token pruning
  - diversity-aware sampling
  - view-adaptive pruning
  - autonomous driving
date: 2026-05-08
content_hash: 5b32fabbc2e2be57
---

# Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2508.13305](https://arxiv.org/abs/2508.13305)
**Code**: [https://github.com/MinhaoXiong/Prune2Drive.git](https://github.com/MinhaoXiong/Prune2Drive.git)
**Area**: Autonomous Driving / VLM Acceleration / Token Pruning
**Keywords**: Multi-view VLM, visual token pruning, diversity-aware sampling, view-adaptive pruning, autonomous driving

## TL;DR
Prune2Drive is the first plug-and-play token pruning framework designed for multi-view autonomous driving VLMs. It combines T-FPS (Token-wise Farthest Point Sampling) to preserve semantic and spatial diversity with view-adaptive pruning rate optimization to automatically allocate token budgets across camera views. Retaining only 10% of tokens on DriveLM, it achieves 6.40× prefill speedup with only a 3% performance drop.

## Background & Motivation
Autonomous driving VLMs (e.g., DriveMM, DriveLMM-o1) must process high-resolution images from six surround-view cameras (729 tokens per image), yielding over 4,000 visual tokens in total, resulting in prohibitively slow $O(n^2)$ attention computation. Existing token pruning methods (FastV, SparseVLM) are designed for single-image settings and suffer from three shortcomings: (1) reliance on attention weights, making them incompatible with FlashAttention; (2) positional bias, causing tokens at later positions to be systematically retained; and (3) neglect of differential view contributions — front and rear cameras differ in importance for driving decisions and should not be pruned uniformly.

## Core Problem
How to design a training-free token pruning method for multi-view autonomous driving that does not rely on attention weights and accounts for differential view contributions?

## Method

### Overall Architecture
The framework consists of two core components: (1) **T-FPS (Token-wise Farthest Point Sampling)**: iteratively selects the most diverse subset of tokens in the token embedding space using farthest point sampling; (2) **View-adaptive pruning rate optimization**: employs a Tree-structured Parzen Estimator (TPE) to automatically search for the optimal token retention rate for each camera view on a small validation set.

### Key Designs
1. **T-FPS diversity-aware selection**: Inspired by the FPS algorithm in point cloud processing, but replacing Euclidean distance with cosine distance. Starting from a randomly selected token, each step selects the token with the maximum cosine distance from the already-selected set until the target count is reached. Key advantages: (a) no dependence on attention weights → compatible with FlashAttention; (b) maximizes semantic and spatial coverage → avoids discarding low-attention but important objects (e.g., distant vehicles); (c) negligible computational overhead — only 0.02s for $N=729$, contributing less than 0.1% of total FLOPs.

2. **View-adaptive pruning rate optimization**: The retention rate $\alpha_i$ for each view is treated as an optimizable variable. The objective $\mathcal{M}(\alpha) = R(\alpha) - \lambda P(\alpha)$ balances performance reward against total token count penalty. TPE is used to search for the optimal configuration on a validation set of 500 samples, converging in only 3 H100 GPU hours. Results show that the front-view camera automatically receives a higher retention rate (reflecting its greater importance for driving decisions), while rear and side views are more aggressively pruned.

3. **Theoretical guarantees**: It is proven that combining T-FPS (a k-center greedy approximation minimizing Hausdorff distance) with view-adaptive rates (importance-weighted budget allocation) yields tighter error bounds than uniform random sampling with equal-ratio pruning.

### Loss & Training
The method is entirely training-free. T-FPS is applied directly after the visual encoder output, and view-adaptive rates are searched offline once on a small validation set and then fixed. The framework is compatible with LLaVA-OneVision-7B (DriveMM) and InternVL2.5-8B (DriveLMM-o1).

## Key Experimental Results
**DriveLM (DriveMM, 10% token retention):**

| Method | Token/Image | Avg Score | Prefill Speedup | FLOPs |
|--------|------|------|------|------|
| Vanilla | 729 | 59.1 | 1× | 100% |
| FastV | 72 | 54.1 | 5.78× | 14.2% |
| SparseVLM | 72 | 55.9 | 4.06× | 14.4% |
| **Prune2Drive** | **72** | **57.4** | **6.40×** | **13.4%** |

DriveLMM-o1: 10% retention → 68.3 vs. vanilla 74.2 (−6%), outperforming FastV (65.3) and DART (67.4).

**General VLM (LLaVA-1.5, 128 tokens)**: 97.3% avg performance, surpassing SparseVLM at 96.2%.

**Video AD (OmniDrive)**: 49.0 vs. vanilla 50.3, outperforming FastV (44.3) and SparseVLM (46.8).

### Ablation Study
- **Distance metric**: cosine ≈ L1 ≈ L2 >> minimum distance (nearest-point sampling causes severe degradation of −15%, validating the diversity principle).
- **TPE > Evolutionary > GridSearch**: differences are small (<1%), but all outperform manual rate assignment.
- **Match Score at 25% retention even exceeds the original model** (34.0 vs. 33.9): moderate pruning has a regularization effect by removing redundant or distracting tokens.
- **T-FPS also generalizes to general VLMs**: 94.3% on LLaVA-1.5 at 64 tokens (vs. SparseVLM at 89.5%) — the margin is even larger.
- **Failure mode**: objects with large uniform-color regions (e.g., orange buses) may be undersampled due to high feature similarity.

## Highlights & Insights
- First token pruning method specifically designed for multi-view autonomous driving — not a naive transfer of single-image approaches.
- The FPS-inspired T-FPS design is elegant: farthest point sampling is applied in the token embedding space rather than 3D space to guarantee semantic diversity.
- View-adaptive rate optimization directly addresses the practical asymmetry in importance between front-view and rear-view cameras.
- The 6.40× prefill speedup has direct industrial value for real-time autonomous driving systems.
- Compatible with FlashAttention with negligible overhead (0.02s per image).

## Limitations & Future Work
- Evaluation is conducted only on offline benchmarks; closed-loop simulation validation is absent.
- T-FPS may undersample objects with large uniform-color regions due to high feature similarity (small cosine distance → not selected by FPS).
- View-adaptive rates are fixed values obtained from offline search, with no dynamic adaptation to varying inputs.
- Comparison with recent token compression methods such as V2Drop and ApET is not provided — baselines are limited to FastV, SparseVLM, DART, and PACT.
- Only two AD models (DriveMM and DriveLMM-o1) are evaluated.

## Related Work & Insights
- **vs. FastV (attention-based pruning)**: FastV relies on second-layer attention and suffers from positional bias while being incompatible with FlashAttention. Prune2Drive avoids attention entirely, achieving 57.4 vs. 54.1 (+3.3) at 10% token retention.
- **vs. DART (similarity-based pruning)**: DART uses cosine similarity but ignores view-level differences. Prune2Drive adds view-adaptive optimization; results on DriveLM at 10% tokens are not reported for DART.
- **vs. V2Drop (CVPR '26)**: V2Drop prunes tokens inside the LLM using inter-layer variation, while Prune2Drive prunes after encoder output using FPS — the two approaches are orthogonal and combinable.
- **vs. ApET (CVPR '26)**: ApET uses approximation error for importance estimation; Prune2Drive maximizes diversity — different mechanisms targeting similar objectives.
- The "embedding-space FPS" strategy of T-FPS is generalizable to all multi-image VLM settings, including multi-view medical imaging and multi-camera robotic systems.
- The reward-penalty framework for view-adaptive rate optimization is applicable to other cross-modal or cross-source budget allocation scenarios.
- T-FPS is complementary to DUET-VLM's visual-side clustering and language-side pruning paradigm — T-FPS could replace DUET's V2V clustering module.

## Rating
- **Novelty**: ⭐⭐⭐⭐ T-FPS and view-adaptive rate optimization are both clear, original contributions, though neither component is architecturally complex.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 2 AD benchmarks + 1 video AD benchmark + 5 general VLMs, with efficiency analysis, ablation studies, visualizations, and failure case analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with valuable theoretical analysis; some tables could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ Real-time deployment of autonomous driving VLMs critically demands methods of this kind; the 6.40× speedup offers direct industrial applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)
- [\[CVPR 2026\] Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models](vlm-loc_localization_in_point_cloud_maps_via_vision-language_models.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)

</div>

<!-- RELATED:END -->
