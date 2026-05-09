---
title: >-
  [Paper Note] Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention
description: >-
  [NeurIPS 2025][Multimodal VLM][visual token pruning] This paper proposes HoloV, a plug-and-play visual token pruning framework that adaptively allocates pruning budgets across different spatial crop regions to preserve global visual context rather than retaining only attention-highlighted tokens. On LLaVA-1.5, HoloV retains 95.8% of original performance after pruning 88.9% of visual tokens.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - visual token pruning
  - inference acceleration
  - holistic context
  - adaptive allocation
  - MLLM efficiency
date: 2026-05-08
content_hash: d13d7abb580605b3
---

# Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention

**Conference**: NeurIPS 2025
**arXiv**: [2510.02912](https://arxiv.org/abs/2510.02912)
**Code**: [GitHub](https://github.com/obananas/HoloV)
**Area**: Multimodal VLM
**Keywords**: visual token pruning, inference acceleration, holistic context, adaptive allocation, MLLM efficiency

## TL;DR

This paper proposes HoloV, a plug-and-play visual token pruning framework that adaptively allocates pruning budgets across different spatial crop regions to preserve global visual context rather than retaining only attention-highlighted tokens. On LLaVA-1.5, HoloV retains 95.8% of original performance after pruning 88.9% of visual tokens.

## Background & Motivation

MLLMs suffer from severe visual token redundancy — LLaVA-1.5 encodes a 336-resolution image into 576 visual tokens, while LLaVA-OneVision requires up to 7,290. Existing attention-first pruning methods (e.g., FastV) exhibit three critical flaws:

**Information Redundancy**: Attention-score-based selection tends to retain semantically similar neighboring tokens, leading to information duplication at high pruning rates.

**Positional Bias**: Transformer positional encodings assign higher attention weights to tokens at the beginning and end of sequences, whereas image subjects typically appear in the center.

**Attention Dispersion**: Text–visual cross-attention is spread across a large number of tokens; the top 20% tokens account for only 40% of total attention.

The authors validate the importance of holistic context through two key observations:

- **Random Pruning > FastV**: On more than half of the evaluated benchmarks, randomly retaining tokens (preserving spatial diversity) outperforms attention-ranked selection.
- **Global Thumbnail > Local Crops**: Using only a global thumbnail already achieves strong results on general perception benchmarks such as MMBench and MME.

## Method

### Overall Architecture

HoloV performs token pruning before the LLM decoder, preserving holistic visual context through the following steps:

1. Partition the $N_v$ visual tokens evenly into $\mathcal{C}$ spatial crop regions.
2. Compute a holistic score for each region by combining semantic diversity and attention saliency.
3. Adaptively allocate retention quotas based on region importance.
4. Within each region, select the top-k tokens by score.

### Key Designs

**Intra-Crop Semantic Diversity**: For normalized embeddings $\mathbf{Z}_v^c$ in crop region $c$, a cosine similarity matrix (excluding self-similarity) is computed:

$$\mathbf{S}^c = (\mathbf{1} - \mathbf{I}_M) \odot \mathbf{Z}_v^c {\mathbf{Z}_v^c}^\top$$

The semantic distribution variance for each token is then computed as:

$$\mathcal{V}_i^c = \frac{1}{M-1} \sum (\mathbf{S}_{i,j}^c - \mu_i^c)^2$$

A high $\mathcal{V}_i^c$ indicates that the token has diverse relational connections to others, signifying high information content.

**Holistic Score Fusion**: Semantic diversity $\mathcal{V}^c$ and [CLS] attention $\mathcal{A}^c$ are combined via adaptive scaling:

$$\mathcal{H}^c = \gamma_c \mathcal{V}^c + \mathcal{A}^c, \quad \gamma_c = \mathbb{E}[\|\mathcal{A}^c\|] / \mathbb{E}[\|\mathcal{V}^c\|]$$

**Adaptive Quota Allocation**: Retention token counts are allocated according to region-level holistic importance:

$$w_c = \frac{(\frac{1}{M} \sum_{t=1}^M \mathcal{H}_t^c)^\tau}{\sum_{c'=1}^{\mathcal{C}} (\frac{1}{M} \sum_{t=1}^M \mathcal{H}_t^{c'})^\tau}$$

The initial quota is $q_c = \lfloor w_c \hat{N}_v \rfloor$, with overflow and deficit resolved through iterative redistribution.

**Visual Context Refetching**: Pruned tokens are reinjected into the MLLM at intermediate trigger layers via FFN as a "key-value memory," activated only when model inference uncertainty is high.

### Loss & Training

HoloV is a training-free plug-and-play method requiring no additional loss functions. The theoretical FLOPs reduction is analyzed as:

$$F \approx 1 - (1-R)^2 = 2R - R^2$$

where $R$ is the token reduction ratio. During the decoding phase (with KV cache), the reduction is approximately linear in $R$.

## Key Experimental Results

### Main Results

**LLaVA-1.5 7B performance at different pruning rates (average over 9 benchmarks)**:

| Method | Keep 192 (↓66.7%) | Keep 128 (↓77.8%) | Keep 64 (↓88.9%) |
|---|---|---|---|
| FastV (ECCV24) | 90.5% | 85.4% | 76.7% |
| MustDrop | 97.2% | 95.7% | 90.1% |
| VisionZip (CVPR25) | 98.1% | 97.2% | 94.5% |
| DART (EMNLP25) | 98.5% | 97.5% | — |
| SparseVLM (ICML25) | 96.1% | 93.8% | — |
| **HoloV (Ours)** | **99.2%** | **98.0%** | **95.8%** |

**Detailed benchmark results (64 tokens retained, ↓88.9%)**:

| Method | GQA | MMB | MME | POPE | VQAv2 | TextVQA | Average |
|---|---|---|---|---|---|---|---|
| FastV | 46.1 | 48.0 | 1256 | 48.0 | 55.0 | 47.8 | 76.7% |
| VisionZip | 55.1 | 60.1 | 1690 | 77.0 | 72.4 | 55.5 | 94.5% |
| **HoloV** | **57.7** | **63.9** | **1802** | **84.0** | **75.5** | **56.8** | **95.8%** |

### Ablation Study

**Validation of random strategy vs. FastV**:

| Benchmark | FastV Wins | Random Wins |
|---|---|---|
| TextVQA | ✓ | |
| MMBench | | ✓ |
| MME | | ✓ |
| POPE | | ✓ |
| GQA | | ✓ |

**Global vs. local input comparison**: Using only the global thumbnail achieves strong performance on MMBench (general perception), while using only local crops performs better on TextVQA (fine-grained perception), validating the necessity of retaining both.

### Key Findings

1. At the extreme pruning rate of 88.9%, HoloV retains 95.8% of original performance, compared to only 76.7% for FastV.
2. The fundamental cause of attention-first methods' sharp performance degradation at high pruning rates is positional bias and attention dispersion.
3. Holistic context (global spatial diversity) is critical for general visual understanding, while local saliency is indispensable for fine-grained perception.
4. HoloV is compatible with hardware acceleration such as Flash-Attention, making it suitable for practical deployment.

## Highlights & Insights

- **Rethinking Token Importance**: Challenges the implicit assumption that "high attention = high information content," demonstrating that spatial semantic diversity is equally important.
- **Simple yet Effective Design**: The core idea of preserving global context is intuitive, implemented as a plug-and-play module requiring no retraining.
- **Comprehensive Motivation Analysis**: The motivation is validated from multiple angles, including positional bias analysis, attention dispersion statistics, and random-vs-attention comparisons.
- **Theoretical Support**: An upper bound on semantic discrepancy is derived based on Lipschitz continuity.
- Core Insight: Spatially distributed token combinations (e.g., "snow" + "skiing" + "hillside") collectively constitute semantic understanding and should not be evaluated in isolation.

## Limitations & Future Work

- The hyperparameter $\tau$ (temperature coefficient) in adaptive allocation requires tuning.
- The trigger condition (uncertainty threshold) for visual context refetching may require task-adaptive adjustment.
- Validation is primarily conducted on the LLaVA series; evaluation on other architectures (e.g., Qwen-VL, InternVL) is limited.
- Uniform partitioning of crop regions may be suboptimal for scenes with highly non-uniform object distributions.

## Related Work & Insights

- **FastV (ECCV2024)**: The attention-ranking-based token pruning baseline and primary comparison target for HoloV.
- **LLaVA-PruMerge (ICCV2025)**: Selective retention and merging based on key similarity.
- **VisionZip (CVPR2025)**: The closest competing method to HoloV in performance.
- **Cognitive Science**: The human visual system forms complete semantic understanding by integrating local features and global scene cues.
- Implication: Efficiency optimization for MLLMs should focus not only on "how many tokens to reduce," but more importantly on "which tokens to retain."

## Rating

- ⭐ Novelty: 4/5 — Revisits token pruning from a holistic context perspective with in-depth motivational analysis.
- ⭐ Experimental Thoroughness: 5/5 — 9 benchmarks, multiple pruning rates, multiple architectures, and extensive ablation and visualization analyses.
- ⭐ Writing Quality: 4/5 — Polished figures, logically layered argumentation, and thorough analysis.
- ⭐ Value: 4/5 — Provides a concise and practical inference acceleration solution with direct value for MLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](../../ICCV2025/multimodal_vlm/mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](../../ICCV2025/multimodal_vlm/feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](../../ICCV2025/multimodal_vlm/from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[NeurIPS 2025\] In-Context Compositional Learning via Sparse Coding Transformer](in-context_compositional_learning_via_sparse_coding_transformer.md)

</div>

<!-- RELATED:END -->
