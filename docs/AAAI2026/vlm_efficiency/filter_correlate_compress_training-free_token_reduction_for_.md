---
title: >-
  [Paper Note] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration
description: >-
  [AAAI 2026][Multimodal Efficiency][Visual token compression] This paper proposes FiCoCo, a three-stage framework (Filter–Correlate–Compress) that identifies redundant tokens via integrated vision-aware and semantic-aware redundancy metrics, adaptively recycles information from discarded tokens via inter-token correlation, and achieves training-free MLLM acceleration. On LLaVA-NeXT, FiCoCo achieves a 14.7× FLOPs reduction while retaining 93.6% of performance…
tags:
  - "AAAI 2026"
  - "Multimodal Efficiency"
  - "Visual token compression"
  - "MLLM acceleration"
  - "training-free"
  - "information recycling"
  - "redundancy metric"
date: 2026-05-08
content_hash: d1d85c72a4b8b893
---

# Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration

**Conference**: AAAI 2026
**arXiv**: [2411.17686](https://arxiv.org/abs/2411.17686)  
**Code**: [https://github.com/kawhiiiileo/FiCoCo](https://github.com/kawhiiiileo/FiCoCo)  
**Area**: Multimodal VLM / Model Compression
**Keywords**: Visual token compression, MLLM acceleration, training-free, information recycling, redundancy metric

## TL;DR
This paper proposes FiCoCo, a three-stage framework (Filter–Correlate–Compress) that identifies redundant tokens via integrated vision-aware and semantic-aware redundancy metrics, adaptively recycles information from discarded tokens via inter-token correlation, and achieves training-free MLLM acceleration. On LLaVA-NeXT, FiCoCo achieves a 14.7× FLOPs reduction while retaining 93.6% of performance, and consistently outperforms FastV, SparseVLM, and other state-of-the-art methods across five MLLM architectures.

## Background & Motivation
In MLLMs, visual tokens vastly outnumber text tokens, and natural visual signals are inherently highly redundant. The prefilling stage accounts for up to 80% of total latency (e.g., in Qwen2-VL), forming a critical deployment bottleneck. Existing training-free token reduction methods suffer from two problems: (1) relying on a single redundancy metric (e.g., attention score) to assess token importance can be inaccurate — methods such as FastV conflate "predictability" with "importance"; (2) discarded tokens may still contain task-relevant information, and naive pruning results in irreversible information loss.

## Core Problem
How can redundant visual tokens in MLLMs be accurately identified without retraining, while maximally preserving useful information during the discard process? The core challenges are: (1) a single metric cannot comprehensively measure redundancy; and (2) useful information in discarded tokens must be redistributed to retained tokens, yet naive averaging dilutes the core content of retained tokens.

## Method

### Overall Architecture
FiCoCo is a three-stage pipeline — Filter → Correlate → Compress — that progressively compresses visual tokens at each layer. FiCoCo-V operates within the visual encoder (task-agnostic), while FiCoCo-L operates within the LLM decoder (task-aware, leveraging textual priors). The two variants can be used independently to achieve different accuracy–efficiency trade-offs.

### Key Designs
1. **Redundancy-based Token Discard (Filter)**: Two redundancy metrics are integrated: vision-aware redundancy (higher average attention received from other patch tokens → more predictable by neighbors → more redundant) and semantic-aware redundancy (lower attention to the [CLS] token → smaller semantic contribution → more redundant). The core insight inverts the assumption of FastV: **high attention-in = high predictability = high redundancy**, not high importance. A **local penalty strategy** is also designed, dividing the 2D grid into windows and penalizing highly redundant tokens within the same window to avoid spatially concentrated discarding.

2. **Correlation-based Information Recycling (Correlate)**: Rather than using a fixed $K$ to select target tokens, FiCoCo employs **token-adaptive K** — for each discarded token, its correlation with all retained tokens is computed (via attention in the vision encoder, or attention combined with indirect text correlation in the LLM decoder), and a quantile threshold adaptively determines the number of correlated tokens. The optimal $K$ varies substantially across layers and tokens, as demonstrated by a Pairwise Token Correlation Gap analysis.

3. **Self-preserving Compression (Compress)**: When updating retained tokens, the method guarantees that at least 50% of each token's own information is preserved ($\mathbf{X}_j^\mathbb{T} \leftarrow \frac{\mathbf{X}_j^\mathbb{T} + \sum_{i \in \mathbb{I}_j} \alpha_{ij} \mathbf{X}_i^\mathbb{S}}{1 + \sum_{i \in \mathbb{I}_j} \alpha_{ij}}$), while more information is recycled from highly correlated discarded tokens ($\alpha_{ij}$ normalized by correlation). A "dense" information pathway (many-to-many) outperforms the "convergent" (many-to-one) alternative.

4. **Task-aware Enhancement in FiCoCo-L**: Within the LLM decoder, the attention between visual and text tokens serves as a task-relevance indicator (replacing semantic redundancy). Indirect semantic correlation is also introduced — measuring indirect semantic relatedness between two visual tokens via text tokens as intermediaries.

### Loss & Training
FiCoCo is entirely training-free and can be applied as a plug-and-play module with fewer than 10 additional lines of code. FiCoCo-V begins compression from the 12th layer of the ViT; FiCoCo-L begins from the 4th layer of the LLM. Key hyperparameters: $\lambda=0.35$, $\beta=0.6$, $\gamma=0.6$, $\varepsilon=0.998$.

## Key Experimental Results

| Model | Method | TFLOPs | Avg Acc | Avg(%) |
|--------|------|--------|---------|--------|
| LLaVA-1.5-7B | Vanilla | 8.5 | 70.3 | 100% |
| LLaVA-1.5-7B | SparseVLM | 1.5 | 61.0 | 86.8% |
| LLaVA-1.5-7B | FiCoCo-V | 1.5 | 65.2 | **92.7%** |
| LLaVA-1.5-7B | FiCoCo-L | 1.5 | 65.3 | **92.8%** |
| LLaVA-NeXT-7B | Vanilla | 42.7 | 58.6 | 100% |
| LLaVA-NeXT-7B | PDrop(5.0T) | 5.0 | 53.8 | 91.7% |
| LLaVA-NeXT-7B | FiCoCo-L(2.9T) | 2.9 | 54.9 | **93.6%** |
| Video-LLaVA | FiCoCo-V | 2.6 | 50.3 | **92.8%** |

Efficiency: throughput improves by 2.08× on LLaVA-NeXT (FiCoCo-V), KV-Cache is reduced by 80%, and GPU memory usage decreases by 36%.

### Ablation Study
- Vision-aware redundancy (+VR) substantially outperforms using attention as an importance indicator (−VR), validating the "high attention-in = high redundancy" hypothesis.
- Semantic-aware redundancy has an even larger impact; removing it causes SQA −3.7 and TextVQA −6.7.
- Token-adaptive $K$ > fixed $K=0$ (pruning) > fixed $K=1/2$ (fixed $K$ leads to over-dilution or noise introduction).
- Self-preserving compression >> naive averaging (TextVQA +2.2%).
- Local penalty is effective in FiCoCo-V (+2.2 TextVQA) but has a slight negative effect in FiCoCo-L.
- When [CLS] is unavailable, replacing it with the mean of Keys results in only a 0.16% performance drop.

## Highlights & Insights
- The **information-theoretic perspective of "attention-in as redundancy"** is highly insightful — it inverts the "high attention = high importance" assumption of FastV, with both theoretical and empirical support.
- **Token-adaptive $K$** is a well-motivated design; the Pairwise Token Correlation Gap analysis clearly demonstrates the inadequacy of fixed $K$.
- **Self-preserving compression** guarantees that retained tokens preserve at least 50% of their own information — simple yet effective.
- Training-free, fewer than 10 lines of code, and directly applicable across five MLLM architectures — extremely high practical value.
- Applicable to video understanding as well (Video-LLaVA 92.8% retention), not limited to image tasks.

## Limitations & Future Work
- Under extreme compression (90%+ token discarding), fine-grained understanding tasks (e.g., OCR) still exhibit noticeable degradation.
- FiCoCo-V is task-agnostic and thus less suitable than FiCoCo-L in scenarios requiring precise text recognition.
- Although hyperparameters are not highly sensitive, minor tuning is still required for different MLLM architectures.
- The combination with training-based methods (e.g., TokenPacker) remains unexplored.

## Related Work & Insights
- **vs. FastV**: FastV uses attention as importance, which is inaccurate. FiCoCo uses attention-in as redundancy combined with [CLS] attention as dual metrics, and adds information recycling that FastV lacks entirely.
- **vs. SparseVLM**: SparseVLM prunes solely based on text–visual attention. FiCoCo integrates multi-dimensional redundancy with adaptive information recycling; under extreme compression, FiCoCo-V outperforms SparseVLM by 5.9%.
- **vs. ToMe**: ToMe uses a fixed $K=1$, causing severe information dilution. FiCoCo employs token-adaptive $K$ with self-preserving compression to retain core information.

The "attention-in as redundancy" perspective generalizes to ViT pruning and video understanding. The information recycling mechanism may be combinable with the Hungarian matching in EM-KD — redistributing information from discarded tokens during distillation. **Idea trigger**: FiCoCo's redundancy metrics and recycling mechanism operate statically (layer-by-layer independently); could a global cross-layer token importance predictor be designed instead?

## Rating
- Novelty: ⭐⭐⭐⭐ — "Attention-in as redundancy" and adaptive-$K$ information recycling are novel; the overall framework design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 MLLM architectures, image and video tasks, 6 compression ratios, and highly detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — The three-stage naming is intuitive, the analysis is in-depth, and the visualizations are excellent.
- Value: ⭐⭐⭐⭐⭐ — Training-free, plug-and-play, and cross-architecture generalization make this work of extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration](../../ICLR2026/vlm_efficiency/visiontrim_unified_vision_token_compression_for_training-free_mllm_acceleration.md)
- [\[ICLR 2026\] Enhancing Visual Token Representations for Video Large Language Models via Training-free Spatial-Temporal Pooling and Gridding](../../ICLR2026/vlm_efficiency/enhancing_visual_token_representations_for_video_large_language_models_via_train.md)
- [\[AAAI 2026\] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks](tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)
- [\[ACL 2026\] From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration](../../ACL2026/vlm_efficiency/from_inheritance_to_saturation_disentangling_the_evolution_of_visual_redundancy_.md)
- [\[CVPR 2026\] ZOO-Prune: Training-Free Token Pruning via Zeroth-Order Gradient Estimation in Vision-Language Models](../../CVPR2026/vlm_efficiency/zoo-prune_training-free_token_pruning_via_zeroth-order_gradient_estimation_in_vi.md)

</div>

<!-- RELATED:END -->
