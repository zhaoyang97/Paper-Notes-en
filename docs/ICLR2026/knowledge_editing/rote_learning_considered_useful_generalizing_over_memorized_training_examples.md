---
title: >-
  [Paper Note] Rote Learning Considered Useful: Generalizing over Memorized Training Examples
description: >-
  [ICLR 2026][Knowledge Editing][memorization] This paper proposes a two-stage "memorize-then-generalize" framework, demonstrating that LLMs can generalize effectively after rote-memorizing synthetic key tokens…
tags:
  - "ICLR 2026"
  - "Knowledge Editing"
  - "memorization"
  - "generalization"
  - "knowledge_injection"
  - "LLM_learning_dynamics"
date: 2026-05-08
content_hash: 61bc767ce645d48f
---

# Rote Learning Considered Useful: Generalizing over Memorized Training Examples

**Conference**: ICLR 2026
**arXiv**: [2507.21914](https://arxiv.org/abs/2507.21914)  
**Code**: [QinyuanWu0710/memorize-then-generalize](https://github.com/QinyuanWu0710/memorize-then-generalize)  
**Area**: Knowledge Editing
**Keywords**: memorization, generalization, knowledge_injection, LLM_learning_dynamics

## TL;DR

This paper proposes a two-stage "memorize-then-generalize" framework, demonstrating that LLMs can generalize effectively after rote-memorizing synthetic key tokens, requiring only minimal semantic fine-tuning — thereby challenging the conventional view that memorization impedes generalization.

## Background & Motivation

**Conventional wisdom**: Rote learning is widely believed to cause overfitting and harm generalization. In LLMs, memorization is treated as undesirable behavior, associated with privacy leakage, hallucination, and paraphrase brittleness. Mainstream training paradigms restrict the number of training epochs (typically 1–2) to avoid memorization.

**Core challenges**:
- Prior work shows significant performance degradation on paraphrased prompts
- Memorized knowledge may interfere with downstream adaptation
- The relationship between memorization and generalization remains poorly understood

**Paper's stance**: This paper challenges the above view, demonstrating that LLMs **can** generalize from rote-memorized data. Through a carefully designed two-stage process, memorization not only fails to hinder generalization — it can serve as its **foundation**.

## Method

### Overall Architecture: Memorize-then-Generalize

The learning process is decoupled into two distinct stages:

**Stage 1 (Rote Memorization)**: The model memorizes factual triples (subject–relation–object) using a synthetic, **semantically null key token** [X]. For example: "Gene Finley [X] Cody Ross." At this point, [X] carries no semantic meaning and serves solely as a relation placeholder. Training is performed via unsupervised next-token prediction.

**Stage 2 (Generalization)**: Supervised fine-tuning is conducted on a small subset of memorized facts using semantically explicit prompts. For example: "Who is Gene Finley's mother? → Cody Ross." This stage reinterprets [X] as a specific semantic relation.

### Key Designs

**Synthetic dataset**:
- Fully synthetic to prevent pretraining contamination
- Covers 5 T-REx relations: author, capital, educated_at, genre, mother
- 100 fictional subject–object pairs per relation
- 20 natural language prompt variants per relation (10 train + 10 test)
- 3 unrelated prompts for negative sample evaluation
- Translated into German, Spanish, Chinese, and Japanese

**Three-level generalization test**:
1. **Unseen facts**: Can subject–object pairs excluded from Stage 2 be retrieved using training prompts?
2. **Unseen prompts**: Can all facts be retrieved using semantically similar but unseen prompts?
3. **Unseen languages**: Can facts be retrieved in other languages?

**Evaluation metrics**:
- Generation accuracy (greedy decoding over 50 tokens, exact match)
- 100-way multiple-choice accuracy
- Model-assigned probability to the target token

### Loss & Training

Stage 1 uses the standard autoregressive language modeling loss (unsupervised next-token prediction):

$$\mathcal{L}_{\text{Phase-1}} = -\sum_t \log P(x_t | x_{<t})$$

Stage 2 uses a supervised fine-tuning loss:

$$\mathcal{L}_{\text{Phase-2}} = -\log P(o | p(s))$$

where $p(s)$ denotes the semantically explicit prompt and $o$ is the target object.

## Key Experimental Results

### Main Results: Generalization Performance (Qwen2.5-1.5B)

| Stage 1 Epochs | Stage 2 Data Size $k$ | Stage 2 Epochs | Train Prompt Acc. | Test Prompt Acc. |
|---|---|---|---|---|
| 3 | 50 | 1 | 0.38 | 0.35 |
| 6 | 50 | 1 | 0.94 | 0.89 |
| 10 | 50 | 1 | 0.94 | 0.98 |
| **20** | **50** | **1** | **1.00** | **0.98** |
| 10 | 1 | 8 | 1.00 | 0.75 |
| 20 | 1 | 8 | 1.00 | 0.76 |

Key finding: **Deeper memorization leads to better generalization.** Even a single fact with a single prompt yields substantial generalization (0.76 accuracy).

### Ablation Study: Representation Space Analysis

| Training Stage | ΔCosSim (Relation Cluster Separation) | CosSim w/ Train Prompts | CosSim w/ Test Prompts | CosSim w/ Unrelated Prompts |
|---|---|---|---|---|
| Base model | 0.058 | - | - | - |
| Phase-1 (epoch 2) | 0.116 | - | - | - |
| Phase-1 (epoch 20) | 0.191 | 0.87 | 0.58 | 0.50 |
| Phase-2 complete | **0.258** | **0.90** | **0.71** | 0.50 |

Key findings:
1. Relational structure is already acquired during rote memorization (ΔCosSim increases monotonically)
2. After Stage 2, alignment between key token representations and semantic prompts improves substantially (0.58 → 0.71)
3. Similarity to unrelated prompts remains unchanged (0.50), confirming specificity

### Comparison with SFT and ICL

| Method | Data Efficiency | 1-Prompt Acc. | 10-Prompt Acc. |
|---|---|---|---|
| Memorize-then-Generalize | High | **Significantly higher than SFT** | ~0.9 (half the token count of SFT) |
| Standard SFT | Low | Far below proposed method | ~0.9 (but requires 2× training tokens) |
| In-Context Learning | N/A | Below proposed method | Assigns high probability even to unrelated prompts (non-discriminative) |

### Cross-Lingual Generalization

Cross-lingual generation accuracy after Stage 2 training on English only (ranked):
- English > Spanish > German > Japanese > Chinese
- Accuracy on unrelated prompts approaches 0 across all languages (dashed baseline)

### Reasoning Capability Enhancement

| Memorization Epochs | Reversal Reasoning Acc. | 2-Hop Reasoning Acc. |
|---|---|---|
| 0 (no memorization) | 0.00 | 0.14 |
| 5 | - | 0.14 |
| 10 | - | 0.14 |
| 20 | 0.26 | **0.36** |
| SFT baseline | 0.01 | - |

Deep memorization not only facilitates direct retrieval but also enhances reversal reasoning and multi-hop reasoning.

### Key Findings

1. **One fact and one prompt suffice for generalization**: This challenges the conventional assumption that generalization requires diverse prompt coverage.
2. **Deeper memorization yields better generalization**: The number of training epochs is positively correlated with generalization performance, contradicting the "more epochs = overfitting" narrative.
3. **Robust across 8 models**: Findings hold consistently across 4 model families and 8 models, including Qwen2.5, Llama-2/3.2, and Phi-4.
4. **Semantic alignment at the representation level**: Key token representations align with semantic prompts after Stage 2, revealing the underlying mechanism of generalization.
5. **Dual generalization risk**: The same memorization foundation can simultaneously support benign and malicious interpretations, enabling adversaries to repurpose memorized data through minimal fine-tuning.

## Highlights & Insights

- **Counterintuitive finding**: The paper directly challenges the dominant narrative that memorization is harmful, providing rigorous empirical evidence that memorization can serve as the foundation for generalization.
- **Exceptionally clean experimental design**: The use of fully synthetic data and semantically null tokens eliminates confounding factors entirely.
- **Practical value for knowledge injection**: The Memorize-then-Generalize paradigm is more data-efficient than SFT and more reliable than ICL.
- **Profound security insight**: The paper identifies "dual generalization" as a novel attack vector — models can be induced to accept malicious semantics while retaining benign functionality.

## Limitations & Future Work

1. Experiments rely on fully synthetic facts, which may not capture the complexity of real-world knowledge (e.g., polysemy, context dependency).
2. The scope is limited to factual triples; more complex knowledge structures (e.g., procedural knowledge, reasoning chains) are not explored.
3. Model scale is capped at 14B; behavior in larger models may differ.
4. Absolute accuracy for reversal and 2-hop reasoning, while improved, remains modest (0.26, 0.36).
5. The security risk analysis is primarily proof-of-concept; defensive strategies are not thoroughly explored.

## Related Work & Insights

- **Grokking (Power et al., 2022)**: Generalization emerges suddenly after extensive memorization — consistent with this paper's finding that deeper memorization yields better generalization.
- **Physics of Language Models (Allen-Zhu & Li, 2023)**: Studies knowledge manipulation and finds that memorization can interfere with fine-tuning generalization — this paper reaches the opposite conclusion under a different framework.
- **Reversal Curse (Berglund et al., 2023)**: Models that learn "A is B" fail to answer "B is A" — deep memorization combined with generalization in this framework partially mitigates this issue.
- Implications for knowledge editing and continual learning: Anchoring knowledge via synthetic tokens followed by semantic alignment may represent a more efficient knowledge injection pipeline.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Counterintuitive findings with a clean conceptual framework
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 8 models, multilingual evaluation, representation analysis; limited to synthetic data
- **Value**: ⭐⭐⭐⭐ — Practical contributions to knowledge injection
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized with rich figures and tables
- **Overall**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning](../../AAAI2026/knowledge_editing/is_the_information_bottleneck_robust_enough_towards_label-noise_resistant_inform.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)
- [\[ICLR 2026\] PICS: Pairwise Image Compositing with Spatial Interactions](pics_pairwise_image_compositing_with_spatial_interactions.md)

</div>

<!-- RELATED:END -->
