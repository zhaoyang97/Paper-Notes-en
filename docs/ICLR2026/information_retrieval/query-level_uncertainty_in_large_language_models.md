---
title: >-
  [Paper Note] Query-Level Uncertainty in Large Language Models
description: >-
  [ICLR2026][Information Retrieval & RAG][uncertainty estimation] This paper introduces the concept of Query-Level Uncertainty and proposes an Internal Confidence method that estimates…
tags:
  - "ICLR2026"
  - "Information Retrieval & RAG"
  - "uncertainty estimation"
  - "knowledge boundary"
  - "adaptive inference"
  - "training-free"
  - "internal confidence"
date: 2026-05-08
content_hash: 08606ffaf2560cf4
---

# Query-Level Uncertainty in Large Language Models

**Conference**: ICLR2026
**arXiv**: [2506.09669](https://arxiv.org/abs/2506.09669)  
**Code**: [GitHub](https://github.com/tigerchen52/query_level_uncertainty)  
**Area**: Information Retrieval
**Keywords**: uncertainty estimation, knowledge boundary, adaptive inference, training-free, internal confidence

## TL;DR
This paper introduces the concept of Query-Level Uncertainty and proposes an Internal Confidence method that estimates, prior to generation (via a single forward pass), whether an LLM is capable of answering a given query. The approach is training-free and enables efficient adaptive inference strategies including RAG triggering, model cascading, and abstention.

## Background & Motivation
1. LLMs have inherent knowledge boundaries and cannot accurately answer all queries; awareness of these boundaries is critical for building trustworthy and efficient AI systems.
2. Existing uncertainty estimation methods are predominantly answer-level (evaluated post-generation), incurring substantial computational overhead due to the requirement of full answer generation.
3. Adaptive inference strategies (e.g., RAG, slow thinking, model cascading) require pre-generation signals to determine whether to invoke additional resources.
4. Existing query-level approaches require training probes or fine-tuning (e.g., IDK token, R-Tuning), limiting their generalizability.
5. The internal hidden states of LLMs encode rich information about knowledge reachability, and cross-layer consistency can improve output quality.

## Method

### Mechanism
A yes/no self-evaluation prompt is used to elicit the LLM's judgment on whether it can answer the query. The probability $P(\text{Yes})$ is extracted as the confidence score without generating any answer. The prompt follows the format: "Respond only with 'Yes' or 'No' to indicate whether you are capable of answering the {Query} accurately."

### From P(Yes) to Internal Confidence
**Basic P(Yes)**: Uses only the hidden state at the last token of the last layer, $\mathbf{h}_N^{(L)}$, projected onto the {Yes, No} vocabulary via the unembedding matrix followed by softmax — analogous to training-free linear probing.

**Internal Confidence Extension**: Rather than relying solely on the final position, $P(\text{Yes})$ is computed across all layers and token positions and aggregated via a weighted average:
$$\text{IC}(\mathbf{h}) = \sum_{n=1}^{N}\sum_{l=1}^{L}w_n^{(l)} P(\text{Yes}|\mathbf{h}_n^{(l)})$$

Weights are computed using **Attenuated Encoding**: $\delta_j^{(i)} = \exp(-\alpha|i-j|^2) / Z$, which assigns exponentially decaying weights with distance from the "decision center" (defaulting to the last token of the last layer). The parameter $\alpha=1.0$ controls locality — larger values concentrate weights more tightly around the center.

### Design Motivation
- AUROC heatmaps reveal that a "decision center" indeed exists: the most discriminative position is not necessarily the last token of the last layer, though it serves as a close approximation.
- Fixing the decision center eliminates the need for a validation set to identify the optimal position, preserving the training-free property.
- Cross-layer aggregation exploits rich knowledge encoded in intermediate layers rather than relying solely on the final representation.

### Application Scenarios
1. **Adaptive RAG**: Retrieval augmentation is triggered only when IC is low; queries with high IC are answered directly, reducing RAG invocations by 50%+ with negligible performance loss.
2. **Model Cascading**: Queries on which the small model yields low IC are forwarded to a larger model, achieving an optimal cost-quality trade-off.
3. **Abstention Strategy**: Queries with high uncertainty are declined, improving system reliability.

## Key Experimental Results

### Main Results (Cross-Model and Cross-Task)

| Method | Phi-3.8B AUROC | Llama-8B AUROC | Qwen-14B AUROC | Avg AUROC |
|------|:--:|:--:|:--:|:--:|
| Max(-log p) | 54.0 | 56.3 | 57.8 | 56.0 |
| Predictive Entropy | 57.9 | 60.1 | 62.4 | 60.1 |
| Semantic Entropy | 55.6 | 59.7 | 60.0 | 58.4 |
| P(Yes) top-right | 57.3 | 60.5 | 60.9 | 59.6 |
| **Internal Confidence** | **60.8** | **64.7** | **67.1** | **64.2** |

### Comparison with Answer-Level Methods (Speed)

| Method | GSM8K AUROC | ms/sample |
|------|:--:|:--:|
| IC (Ours) | 66.8 | **0.3** |
| Predictive Entropy | 61.0 | 9.8 |
| Min-K Entropy | 60.4 | 3.8 |
| Semantic Entropy | 60.0 | 151.8 |

**Key Findings**:
1. IC consistently outperforms all baselines across 3 datasets and 3 models.
2. IC is 32×–602× faster than answer-level methods while achieving higher accuracy.
3. In RAG settings, IC reduces retrieval calls by 50%+ with negligible performance degradation.
4. The effectiveness of IC scales with model size, as larger models exhibit stronger self-awareness of their knowledge boundaries.
5. Layers and token positions near the decision center are most discriminative, consistent with observations from the AUROC heatmaps.

## Highlights & Insights
- This work is the first to formally define query-level uncertainty, shifting uncertainty estimation from a post-hoc to a pre-generation paradigm.
- The method is entirely training-free, requires only a single forward pass, and is highly practical.
- The cross-layer, cross-token weighted aggregation strategy (Attenuated Encoding) is both elegant and effective.
- The paper demonstrates significant efficiency-quality trade-off advantages in RAG and model cascading scenarios.

## Limitations & Future Work
- The decision center is fixed at the last token of the last layer, which is suboptimal — a trade-off made to preserve the training-free property.
- Validation is limited to tasks with well-defined answers (factual QA and mathematical reasoning); open-ended generation is not addressed.
- Using greedy decoding as a proxy for the knowledge boundary is conservative and may underestimate model capability.
- Discriminative performance is relatively weaker on reasoning-intensive tasks (e.g., multi-step mathematical reasoning) compared to factual QA.

## Related Work & Insights
- Answer-level uncertainty: Semantic Entropy (Kuhn et al. 2023), P(True) (Kadavath et al. 2022).
- Knowledge boundary detection: IDK token (Cohen et al. 2024), R-Tuning (Zhang et al. 2024a) — both require training.
- Internal state probing: Gottesman & Geva 2024 train lightweight probes; Semantic Entropy Probes (Kossen et al. 2024).

## Rating
- Novelty: ⭐⭐⭐⭐ (the query-level concept is novel and the method is concise)
- Experimental Thoroughness: ⭐⭐⭐⭐ (3 models, 3 datasets, multiple application scenarios)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear problem formulation, intuitive illustrations)
- Value: ⭐⭐⭐⭐ (direct practical value for adaptive inference)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[AAAI 2026\] "As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](../../AAAI2026/information_retrieval/as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)
- [\[ACL 2026\] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](../../ACL2026/information_retrieval/koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)
- [\[AAAI 2026\] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description](../../AAAI2026/information_retrieval/oad-promoter_enhancing_zero-shot_vqa_using_large_language_models_with_object_att.md)
- [\[ICLR 2026\] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)

</div>

<!-- RELATED:END -->
