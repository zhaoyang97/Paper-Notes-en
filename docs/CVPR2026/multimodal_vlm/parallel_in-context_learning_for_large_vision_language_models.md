---
title: >-
  [Paper Note] Parallel In-context Learning for Large Vision Language Models
description: >-
  [CVPR 2026][Multimodal VLM][In-context learning] This paper proposes Parallel-ICL, which partitions the long demonstration context in multimodal in-context learning (MM-ICL) into chunks for parallel processing…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "In-context learning"
  - "inference acceleration"
  - "Product-of-Experts"
  - "multimodal learning"
  - "context chunking"
date: 2026-05-08
content_hash: d6f26f3c140ff818
---

# Parallel In-context Learning for Large Vision Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.16092](https://arxiv.org/abs/2603.16092)  
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: In-context learning, inference acceleration, Product-of-Experts, multimodal learning, context chunking

## TL;DR
This paper proposes Parallel-ICL, which partitions the long demonstration context in multimodal in-context learning (MM-ICL) into chunks for parallel processing, and integrates predictions at the logit level via weighted Product-of-Experts (PoE). The method achieves performance on par with or superior to full-context MM-ICL while significantly reducing inference latency.

## Background & Motivation
**Background**: Large Vision-Language Models (LVLMs) leverage MM-ICL with multiple demonstration examples to adapt to new tasks, and performance generally improves with more demonstrations.

**Limitations of Prior Work**: The attention computation cost in Transformers scales quadratically with context length, and each image in an LVLM requires thousands of visual tokens. Consequently, increasing the number of demonstrations dramatically increases inference latency — for example, 32-shot inference is approximately 3.5× slower than 8-shot.

**Key Challenge**: There is a severe trade-off between accuracy and inference efficiency: better performance requires more demonstrations, while faster inference demands shorter contexts.

**Goal**: Efficiently approximate long-context MM-ICL at inference time without any additional training or datasets.

**Key Insight**: Individual demonstrations are mutually independent and need not be processed as a single long sequence. They can instead be processed in parallel chunks and their results aggregated.

**Core Idea**: The long demonstration context is divided into multiple short "chunks" that are processed in parallel, and predictions are merged at the logit level using weighted PoE. The theoretical motivation derives from a diversity-relevance analysis based on Fano's inequality in ensemble learning.

## Method

### Overall Architecture
Input: $N$ demonstrations + query → Context Chunking → Parallel processing of each chunk → Context Compilation (weighted PoE over logits) → Output prediction.

### Key Designs

1. **Context Chunking**:

    - $k$-means clustering is applied to multimodal features of demonstrations (concatenation of CLIP image and text features) to form groups.
    - Each cluster constitutes one chunk, maximizing inter-chunk diversity.
    - **Design Motivation**: Based on Fano's inequality, the error lower bound of an ensemble is negatively correlated with prediction diversity (minimizing $I_{redun}$ is desirable); clustering maximizes inter-chunk diversity.

2. **Context Compilation**:

    - A weighted Product-of-Experts (PoE) aggregates the predictive distributions from each chunk.
    - Implemented at the logit level: $\hat{l}_\theta(y_i) = \sum_{k=1}^{K} w_k l_\theta(y_i | C_k, x, t)$
    - Weights $w_k$ are computed based on the cosine similarity between each chunk and the query (softmax-normalized).
    - **Design Motivation**: Based on the relevance term ($I_{relev}$) in Fano's inequality, chunks more relevant to the query receive higher weights.

3. **Theoretical Foundation**:

    - Building on Theorem 5.1 (Brown & Zhou-Li), the ensemble prediction error is decomposed into a relevance term (correlation of each model with the ground truth) and a redundancy term (mutual information among models).
    - Low error requires high relevance (accurate prediction from each chunk) and high diversity (low information redundancy across chunks).
    - These two properties directly motivate the chunking strategy (maximizing diversity) and the compilation strategy (relevance-based weighting).

### Loss & Training
No training is required. Parallel-ICL is a purely inference-time, plug-and-play method.

## Key Experimental Results

### Main Results

| Method | Token Length | Accuracy | Total Latency (s) |
|--------|-------------|----------|-------------------|
| Zero-shot | 2,557 | 0.00 | 0.099 |
| MM-ICL (8-shot) | 23,318 | 56.90 | 1.004 |
| MM-ICL (16-shot) | 44,027 | 58.20 | 2.376 |
| MM-ICL (32-shot) | 84,959 | 58.90 | 3.479 |
| Parallel-ICL (32-shot, K=4) | ~21K/chunk | ≈58.90 | ~1.5 |

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| Random chunking vs. Clustering | Clustering outperforms random chunking in both accuracy and diversity |
| Uniform weights vs. Similarity-based weights | Similarity-based weighting is superior on most benchmarks |
| Image features vs. Text features vs. Multimodal features | Multimodal feature clustering yields the best results |
| K=2,4 vs. K=1 (full context) at N=32 | K=2,4 surpasses full-context on some tasks, potentially alleviating the "lost in the middle" problem |

### Key Findings
- Parallel-ICL **outperforms** full-context MM-ICL in certain settings at $N=32$, possibly by mitigating the "lost in the middle" problem.
- Inference speedup is substantial: at $K=4$, latency is approximately 1/3 to 1/2 of full-context inference.
- The method generalizes across models: it is effective on LLaVA-OV, Qwen2.5-VL, and InternVL3.5.
- Inter-chunk diversity is positively correlated with final accuracy, validating the theoretical analysis.

## Highlights & Insights
- **Theory-driven design**: The importance of diversity and relevance is derived from Fano's inequality, and the design choices (clustering and similarity-based weighting) follow naturally, creating a coherent connection between theory and practice.
- **Plug-and-play inference method**: No additional training, datasets, or model modifications are required; the method can be directly applied to any LVLM that supports MM-ICL.
- **Unexpected finding**: Chunked parallel processing outperforms full-context processing in certain scenarios, suggesting the existence of information loss in long-context MM-ICL and opening new directions for future research.
- The approach is orthogonal to general inference acceleration techniques (token pruning, KV cache compression) and can be combined with them.

## Limitations & Future Work
- The PoE formulation assumes that predictions across chunks are approximately conditionally independent, which may not hold when demonstrations exhibit strong inter-dependencies.
- Clustering requires additional CLIP feature extraction, introducing a small preprocessing overhead.
- Performance on generative long-form tasks (e.g., image captioning) is less stable than on discriminative tasks (e.g., VQA).
- The optimal value of $K$ varies across tasks and requires tuning.

## Related Work & Insights
- **vs. Task Vector methods (Peng et al. / Jiang et al.)**: These approaches require extracting task vectors from large sets of demonstrations in advance and involve additional optimization, deviating from the dynamic adaptation nature of MM-ICL. Parallel-ICL preserves the plug-and-play property.
- **vs. VCD / Contrastive Decoding**: VCD applies subtraction at the logit level to mitigate bias, whereas Parallel-ICL performs weighted ensemble addition at the logit level for enhancement. Both reflect the paradigm of logit-level ensemble/manipulation.

## Supplementary Analysis
- Parallel-ICL modifies neither the model parameters nor the demonstration set — it purely changes the processing strategy. The observed performance gains imply an information processing bottleneck in full-context MM-ICL.
- PoE is preferred over MoE because PoE is better suited for high-dimensional probability distributions (e.g., large VLM vocabularies) and can be efficiently implemented via logit summation.
- The feature extractor used in experiments is CLIP ViT-L/14; the additional latency from feature extraction is negligible.
- On the demo-based learning tasks in MI-Bench-ICL, Parallel-ICL with $K=4$ at $N=32$ incurs only approximately 40% of the latency of full-context inference.
- The method can be further combined with techniques such as KV cache sharing to achieve additional latency reduction.

## Rating
- Novelty: ⭐⭐⭐⭐ — Theory-driven parallel chunked ICL is a novel and principled contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple models and tasks with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical analysis is clear and the overall narrative is logically coherent.
- Value: ⭐⭐⭐⭐ — A practically useful inference acceleration method with broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)
- [\[CVPR 2026\] Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)

</div>

<!-- RELATED:END -->
