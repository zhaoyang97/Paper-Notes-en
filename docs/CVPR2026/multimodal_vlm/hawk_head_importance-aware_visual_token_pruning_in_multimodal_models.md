---
title: >-
  [Paper Note] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models
description: >-
  [CVPR 2026][Multimodal VLM][Visual token pruning] This paper proposes HAWK, a head importance-aware visual token pruning method that offline computes per-head contribution weights to visual understanding and dynamically evaluates each visual token's importance via text-guided attention scores. On Qwen2.5-VL, HAWK retains 96.0% of original performance after pruning 80.2% of visual tokens while reducing inference latency by 26%.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Visual token pruning
  - attention head importance
  - multimodal inference acceleration
  - training-free
  - text-guided attention
date: 2026-05-08
content_hash: 8522207fb79832a0
---

# HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models

**Conference**: CVPR 2026
**arXiv**: [2604.07812](https://arxiv.org/abs/2604.07812)
**Code**: [https://github.com/peppery77/HAWK.git](https://github.com/peppery77/HAWK.git)
**Area**: Multimodal VLM / LLM Efficiency
**Keywords**: Visual token pruning, attention head importance, multimodal inference acceleration, training-free, text-guided attention

## TL;DR
This paper proposes HAWK, a head importance-aware visual token pruning method that offline computes per-head contribution weights to visual understanding and dynamically evaluates each visual token's importance via text-guided attention scores. On Qwen2.5-VL, HAWK retains 96.0% of original performance after pruning 80.2% of visual tokens while reducing inference latency by 26%.

## Background & Motivation

**State of the Field**: Multimodal large language models (MLLMs) encode visual inputs into large numbers of visual tokens (typically hundreds to thousands), which are fed together with text tokens into an LLM. Since the computational complexity of attention mechanisms scales quadratically with token count, the abundance of visual tokens leads to slow inference and high memory consumption. Existing visual token pruning methods fall into three categories: similarity-based (DivPrune), fine-tuning-based (DART), and attention-based (FastV).

**Limitations of Prior Work**: (1) Similarity-based methods are context-agnostic and cannot adapt to user instructions, potentially discarding task-relevant tokens. (2) Fine-tuning-based methods require end-to-end training, incurring high computational cost and poor generalization. (3) Attention-based methods assume all attention heads contribute equally to visual understanding, naively averaging attention scores across all heads to estimate token importance.

**Root Cause**: Different attention heads in practice capture distinct visual semantics and contribute very differently to visual understanding. Experiments show that disabling different attention heads leads to significantly different changes in model performance, and this pattern is consistent across multiple datasets. Treating all heads equally causes retention of redundant tokens and erroneous pruning of valuable ones.

**Paper Goals**: How can visual token pruning account for the differentiated contributions of different attention heads to maximally preserve critical tokens?

**Starting Point**: By systematically ablating each attention head and measuring its impact on visual tasks, consistent patterns of head importance are identified and used to design an importance-weighted pruning strategy.

**Core Idea**: Weight text-guided visual attention scores by offline-computed attention head importance weights to achieve more precise estimation and pruning of visual token importance.

## Method

### Overall Architecture
HAWK consists of three steps: (1) **Offline phase** — ablate each attention head across multiple benchmark datasets and compute per-head importance weights (one-time computation); (2) **Online phase** — use Q/K projection matrices from the first attention layer to compute attention scores from text tokens to each visual token, with positional encodings removed to eliminate positional bias; (3) **Pruning phase** — weight the text-guided attention scores by head importance weights, and retain the top-$k$ visual tokens based on aggregated scores. The entire method requires no training and can be plug-and-play applied to different MLLM architectures.

### Key Designs

1. **Static Attention Head Importance Weights**

   - **Function**: Quantify each attention head's intrinsic contribution to visual understanding.
   - **Mechanism**: For each head $i$, the performance drop after ablation is measured on multiple benchmark datasets $j$: $\Delta S_{i,j} = S_{base,j} - S_{i,j}$. A min-shift is applied to ensure non-negativity: $S'_{i,j} = \Delta S_{i,j} - \min_i(\Delta S_{i,j})$. The values are then L1-normalized and averaged across datasets to obtain weights $w_i = \frac{1}{N_d}\sum_j \frac{S'_{i,j}}{\sum_i S'_{i,j}}$. These weights are computed once and reused thereafter.
   - **Design Motivation**: Ablation experiments confirm that different heads have significantly different and cross-dataset-consistent impacts, enabling reliable head importance estimation with minimal offline computation. The min-shift avoids negative weight issues.

2. **Dynamic Text-Guided Attention Scores**

   - **Function**: Dynamically evaluate each visual token's task relevance conditioned on the current text instruction.
   - **Mechanism**: Using the Q/K projection matrices of the first LLM attention layer, text embeddings are projected as queries and visual embeddings as keys. The attention matrix without positional encodings is computed as $A^i = Q^i \cdot (K^i)^T / \sqrt{d_k}$, and scores are averaged over all text tokens to obtain the relevance score for each visual token under head $i$: $c^i_k = \frac{1}{N}\sum_j A^i_{j,k}$.
   - **Design Motivation**: RoPE positional encodings are deliberately removed to ensure that attention scores reflect only the semantic correspondence between text and visual tokens, unaffected by token positions. The first layer is selected because pruning must be performed at the model's front end, and the first layer already contains sufficient semantic information.

3. **Head Importance-Aware Fusion Pruning**

   - **Function**: Integrate static head weights and dynamic attention scores for precise pruning.
   - **Mechanism**: The final importance score for each visual token $k$ is $I_k = \sum_{i=1}^{N_h} w_i \cdot c^i_k$, i.e., a weighted sum of per-head attention scores using head importance weights. Tokens are ranked by importance, and the top $\tilde{M} = \lfloor M \cdot r \rfloor$ visual tokens are retained. The pruned token subset is concatenated with text tokens and passed to subsequent LLM layers.
   - **Design Motivation**: Compared to simple averaging across all heads, the weighted sum gives greater influence to important heads (e.g., those focused on critical visual semantics), avoiding dilution by noise from unimportant heads.

### Loss & Training
HAWK requires no training. The offline computation of head importance weights uses six datasets: HallBench, MME, TextVQA, ChartQA, AI2D, and RealWorldQA. At inference time, only a single matrix operation is needed to compute attention scores and perform weighted pruning.

## Key Experimental Results

### Main Results (Qwen2.5-VL-7B, Native Resolution)

| Method | Pruning Ratio | HallBench | MME | TextVQA | ChartQA | Rel.% |
|--------|--------------|-----------|-----|---------|---------|-------|
| Original | 0% | 46.5 | 2315 | 85.2 | 86.2 | 100% |
| DivPrune | 60% | 45.8 | 2274 | 82.7 | 80.6 | 96.9% |
| FastV | 60% | 42.5 | 2283 | 84.1 | 82.5 | 96.1% |
| **HAWK** | **60%** | **46.5** | **2313** | **85.0** | **83.6** | **99.6%** |
| DivPrune | 80% | 39.0 | 2196 | 76.8 | 69.0 | 91.6% |
| FastV | 80% | 38.2 | 2236 | 81.9 | 72.3 | 92.3% |
| **HAWK** | **80%** | **42.8** | **2311** | **83.0** | **76.8** | **96.2%** |

### Efficiency Analysis (MME, Qwen2.5-VL-7B)

| Configuration | Score | E2E Latency | KV Cache | GPU Memory |
|--------------|-------|-------------|----------|------------|
| Original | 2315 | 20m15s | 668MB | 16.9GB |
| HAWK (60%) | 2313 | 16m10s (×1.25) | 276MB | 16.1GB |
| HAWK (80%) | 2311 | 15m04s (×1.34) | 148MB | 15.7GB |

### Key Findings
- At 60% pruning, HAWK retains 99.6% of original performance, far surpassing the second-best DivPrune at 96.9% (+2.7 pp).
- At 80% pruning, HAWK still retains 96.2%, outperforming the second-best by 3.9 pp.
- On InternVL3-8B, the advantage is even larger: 94.1% vs. DivPrune's 87.1% at 80% pruning (+7.0 pp).
- HAWK is equally effective on video understanding: 60% pruning retains 98.8% performance.
- End-to-end latency is reduced by 25–34%, KV Cache by 59–78%, and GPU memory by 0.8–1.2 GB.

## Highlights & Insights
- The core finding is highly insightful — attention head contributions to visual understanding are highly unequal yet consistent across datasets. This finding not only benefits pruning but also reveals the functional specialization of visual processing within MLLMs.
- The removal of RoPE positional encodings is a seemingly minor yet critical design choice — positional encodings cause visual tokens positioned near text tokens to receive disproportionately high attention scores unrelated to actual semantic importance.
- The method is remarkably simple and practical: one-time offline computation plus a single matrix operation at inference, zero additional training, and direct plug-in compatibility with any MLLM, resulting in extremely low engineering deployment overhead.
- At an extreme pruning ratio of 90%, approximately 90% of performance is still retained, confirming that visual tokens in MLLMs contain substantial redundancy.

## Limitations & Future Work
- Head importance weights are static values averaged across datasets and may not be optimal for every specific task.
- Only the first attention layer is used to estimate token importance, which may fail to capture deeper semantic dependencies.
- Compared to CDPruner, HAWK does not always achieve the best score on every individual metric, though it achieves the best overall performance.
- At 90% pruning on video understanding, differences between methods diminish, limiting discriminability under extreme pruning ratios.
- Dynamic pruning ratios are not considered — different images or queries may warrant different retention proportions.

## Related Work & Insights
- **vs. FastV**: FastV prunes based on simple ranking of attention scores from early layers, assuming equal head weights. HAWK significantly improves identification of important tokens through head importance weighting.
- **vs. CDPruner**: CDPruner models conditional diversity via DPP, incurring higher computational overhead. HAWK is more lightweight and achieves superior performance.
- **vs. DivPrune**: DivPrune maximizes feature diversity without regard to task instructions. HAWK's text-guided mechanism enables adaptation to different queries.
- The head importance analysis paradigm is transferable to KV Cache compression in LLM inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The discovery of differential attention head contributions to visual understanding is valuable, and the weighted pruning design is natural.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers two model architectures, image and video tasks, four pruning ratios, efficiency analysis, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, well-motivated, and well-organized experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Simple, efficient, highly effective, and extremely practical for engineering deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[CVPR 2026\] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[ICLR 2026\] IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](../../ICLR2026/multimodal_vlm/ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)
- [\[CVPR 2026\] Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models](taxonomy-aware_representation_alignment_for_hierarchical_visual_recognition_with.md)

<!-- RELATED:END -->
