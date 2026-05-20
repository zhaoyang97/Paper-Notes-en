---
title: >-
  [Paper Note] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving
description: >-
  [CVPR 2026][Multimodal VLM][Multi-view VLM] The first plug-and-play token pruning framework for multi-view autonomous driving VLMs. By leveraging T-FPS (Token-wise Farthest Point Sampling) to preserve semantic and spatia…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multi-view VLM"
  - "Visual Token Pruning"
  - "Farthest Point Sampling"
  - "View-Adaptive Pruning"
  - "Autonomous Driving Acceleration"
date: 2026-05-08
content_hash: eabf051ba0b6d78c
---

# Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2508.13305](https://arxiv.org/abs/2508.13305)  
**Code**: [https://github.com/MinhaoXiong/Prune2Drive](https://github.com/MinhaoXiong/Prune2Drive)  
**Area**: Multimodal VLM
**Keywords**: Multi-view VLM, Visual Token Pruning, Farthest Point Sampling, View-Adaptive Pruning, Autonomous Driving Acceleration

## TL;DR

The first plug-and-play token pruning framework for multi-view autonomous driving VLMs. By leveraging T-FPS (Token-wise Farthest Point Sampling) to preserve semantic and spatial diversity, combined with view-adaptive pruning rate optimization to automatically allocate token budgets per camera, the framework achieves 6.40× prefill acceleration on DriveLM while retaining only 10% of tokens with only a 3% performance drop.

## Background & Motivation

1. **Computational explosion in multi-view VLMs**: Autonomous driving VLMs (e.g., DriveMM) must process inputs from 6 surround-view cameras, yielding 729 tokens per image and over 4,000 visual tokens in total. The $O(n^2)$ attention complexity results in unacceptable inference latency.
2. **Existing pruning methods are designed for single-image settings**: Methods such as FastV and SparseVLM overlook the spatial and semantic diversity across multiple views, and their direct application leads to loss of critical view information.
3. **Attention-weight-dependent methods are incompatible with efficient attention**: FastV and similar approaches require reading the attention matrix, making them incompatible with efficient implementations such as FlashAttention.
4. **Positional bias**: Attention-score-based methods systematically favor tokens at specific positions, neglecting semantically important but low-attention tokens (e.g., distant vehicles).
5. **Unequal view contributions**: The front-view camera is far more critical for driving decisions than the rear-view camera, yet existing methods apply uniform pruning rates across all views.
6. **Urgent real-time requirements**: Autonomous driving is a latency-sensitive domain; high inference latency in VLMs directly compromises safety.

## Method

### Overall Architecture

Two core components work in concert: (1) **T-FPS (Token-wise Farthest Point Sampling)**—selects the most diverse subset of tokens in the token embedding space using farthest point sampling; and (2) **View-adaptive pruning rate optimization**—employs TPE (Tree-structured Parzen Estimator) to automatically search for the optimal token retention rate for each camera view on a small validation set. The framework is entirely training-free and is applied directly after the visual encoder output.

### T-FPS Diversity-Aware Token Selection

Inspired by FPS (Farthest Point Sampling) from point cloud processing, the algorithm replaces Euclidean distance with cosine distance:

1. Randomly select an initial token and add it to the selected set $\mathcal{S}$.
2. At each step, compute the cosine distance between all unselected tokens and the most recently added token in $\mathcal{S}$.
3. Update the minimum distance record for each unselected token.
4. Add the token with the largest minimum distance (i.e., the farthest from $\mathcal{S}$) to $\mathcal{S}$.
5. Repeat until the target count $\mathcal{K}$ is reached.

**Key advantages**: (a) No reliance on attention—fully compatible with FlashAttention; (b) Maximizes semantic and spatial coverage—avoids discarding important low-attention objects; (c) Negligible computational overhead—only 0.02s for $N=729$, less than 0.1% of total FLOPs.

### View-Adaptive Pruning Rate Optimization

Each view's retention rate $\alpha_i$ is treated as an optimizable variable, with the objective function defined as:

$$\mathcal{M}(\boldsymbol{\alpha}) = R(\boldsymbol{\alpha}) - \lambda P(\boldsymbol{\alpha})$$

- **Reward term** $R(\boldsymbol{\alpha})$: linguistic similarity between model output and ground truth.
- **Penalty term** $P(\boldsymbol{\alpha}) = \sum_{i=1}^{M} \alpha_i$: total token retention, encouraging sparsity.
- **Hyperparameter** $\lambda$: balances performance and efficiency.

TPE is used to search for the optimal solution on a small validation set of 500 samples, converging within only **3 H100 GPU hours**. Results show that the front-view camera automatically receives a higher retention rate, while the rear and side views are more aggressively pruned.

### Theoretical Guarantee

The paper proves that the combination of T-FPS (a greedy k-center approximation minimizing Hausdorff distance) and view-adaptive rates (budget allocation weighted by importance) provides a tighter error bound than uniform random sampling with equal-ratio pruning, under the assumption of View-Weighted Lipschitz continuity:

$$\sum_{i=1}^{M} w_i \cdot d_H(V_i, S_{i,\text{Prune2Drive}}) \leq \sum_{i=1}^{M} w_i \cdot d_H(V_i, S_{i,\text{baseline}})$$

### Compatibility

The framework is entirely training-free and compatible with multiple VLMs including LLaVA-OneVision-7B (DriveMM), InternVL2.5-8B (DriveLMM-o1), and LLaVA-1.5-7B, requiring neither retraining nor access to attention matrices.

## Key Experimental Results

### DriveLM Benchmark (DriveMM model, 10% token retention)

| Method | Tokens/Image | Avg Score↑ | Prefill Speedup | FLOPs |
|------|:---:|:---:|:---:|:---:|
| Vanilla | 729 | 59.1 | 1× | 100% |
| FastV | 72 | 54.1 | 5.78× | 14.2% |
| SparseVLM | 72 | 55.9 | 4.06× | 14.4% |
| PACT | 72 | 56.8 | — | — |
| **Prune2Drive** | **72** | **57.4** | **6.40×** | **13.4%** |

### DriveLMM-o1 Benchmark (10% token retention)

| Method | Overall Reasoning↑ | Risk Accuracy↑ | Scene Understanding↑ |
|------|:---:|:---:|:---:|
| Vanilla (100%) | 74.2 | 73.01 | 75.99 |
| FastV | 65.3 | 65.37 | 66.43 |
| DART | 67.4 | 65.32 | 68.17 |
| **Prune2Drive** | **68.3** | **68.34** | **69.86** |

### General VLM and Video AD Benchmarks

| Setting | Prune2Drive | SparseVLM | FastV |
|------|:---:|:---:|:---:|
| LLaVA-1.5 (128 tokens) | 97.3% of original | 96.2% | 92.8% |
| LLaVA-1.5 (64 tokens) | 94.6% | 86.9% | 74.3% |
| OmniDrive (Video AD) | 49.0 | 46.8 | 44.3 |

### Ablation Study

| Ablation | DriveLMM-o1 Overall↑ | Notes |
|------|:---:|:---|
| Cosine distance (default) | 68.3 | Best |
| L1 distance | 68.3 | Nearly equivalent |
| L2 distance | 67.7 | Slightly lower |
| Min distance (nearest sampling) | 63.0 | Severe degradation −5.3; validates diversity principle |
| TPE (default) | 68.3 | Best HPO |
| Grid Search | 67.3 | −1.0 |
| Evolutionary | 67.6 | −0.7 |

**Interesting finding**: At 25% token retention on DriveLM, Match Score reaches 34.0, surpassing the original model's 33.9—moderate pruning has a regularization effect, and removing redundant or noisy tokens can improve certain metrics.

## Highlights & Insights

1. **First token pruning framework tailored for multi-view autonomous driving**: Rather than naively transferring single-image methods, the work systematically addresses multi-view spatial/semantic diversity and heterogeneous view contributions.
2. **Elegant T-FPS design**: The FPS idea from point cloud processing is transferred to the token embedding space, using cosine distance to ensure semantic diversity with only 0.02s of computational overhead.
3. **View-adaptive rate optimization automatically discovers front > rear priority**: Without manually designed priors, TPE search automatically allocates optimal budgets.
4. **6.40× speedup with direct industrial value**: The acceleration is practically meaningful for the deployment of real-time autonomous driving systems.

## Limitations & Future Work

1. **Objects with large uniform textures may be undersampled**: For objects such as orange buses, where token features are highly similar, T-FPS may retain too few tokens, leading to information loss.
2. **T-FPS depends on random initialization**: The random selection of the initial token may introduce slight variance; the paper does not report variance across multiple runs.
3. **Validation limited to 7B–8B scale VLMs**: No experiments on larger models (70B+); pruning ratios and their effects may vary with model scale.
4. **View-adaptive rates are static**: The same pruning rate is applied to all samples, without accounting for the fact that different driving scenarios (highway, congestion, intersection) may require different view-attention allocations.
5. **KV Cache reduction only during initial encoding**: Decoding-phase speedup is only 1.04–1.09×, offering limited acceleration for long-sequence generation.

## Related Work & Insights

### vs. FastV / SparseVLM / PACT (Single-Image Token Pruning)

FastV selects tokens based on second-layer attention scores, exhibiting positional bias and incompatibility with FlashAttention. SparseVLM employs text-guided cross-modal attention pruning, also requiring attention matrix access. PACT uses progressive multi-stage pruning. All three are designed for single-image settings and do not account for semantic complementarity or contribution differences across views. Prune2Drive's T-FPS requires no attention access, and the view-adaptive rates are specifically designed for multi-view scenarios. Under extreme compression to 64 tokens, Prune2Drive (94.6%) substantially outperforms SparseVLM (86.9%) and FastV (74.3%).

### vs. DriveMM / DriveLMM-o1 (Autonomous Driving VLMs)

DriveMM and DriveLMM-o1 are autonomous driving–specific VLMs. Prune2Drive is applied directly on top of them as a plug-and-play module without modifying model weights. This orthogonal acceleration approach means Prune2Drive can be combined with any future autonomous driving VLM.

### vs. Quantization / Distillation (Other Acceleration Methods)

Quantization (e.g., GPTQ) reduces numerical precision but does not reduce token count; distillation requires additional training. Prune2Drive is a training-free token reduction method that is orthogonal to both quantization and distillation, and can be combined with them for greater overall speedup.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of T-FPS and view-adaptive rates is novel, though FPS itself and token pruning are established concepts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two AD benchmarks + general VLM + video AD + comprehensive ablations + efficiency analysis + theoretical proof.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with complete theory–experiment–analysis flow and rigorous derivations.
- **Value**: ⭐⭐⭐⭐ — Direct practical value for multi-view VLM acceleration; 6.40× speedup is industrially attractive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)
- [\[CVPR 2026\] Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[AAAI 2026\] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit](../../AAAI2026/multimodal_vlm/llmc_benchmarking_vision-language_model_compression_with_a_plug-and-play_toolkit.md)
- [\[CVPR 2026\] TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](treeteaming_autonomous_red-teaming_of_vision-language_models_via_hierarchical_s.md)

</div>

<!-- RELATED:END -->
