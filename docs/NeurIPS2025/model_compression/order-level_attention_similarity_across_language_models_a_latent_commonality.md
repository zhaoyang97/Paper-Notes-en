---
title: >-
  [Paper Note] Order-Level Attention Similarity Across Language Models: A Latent Commonality
description: >-
  [NeurIPS 2025][Model Compression][Attention Mechanism] This paper proposes Order-Level Attention (OLA)—an order-wise decomposition of Attention Rollout—and discovers that different language models exhibit significant sim…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Attention Mechanism"
  - "Language Model Similarity"
  - "Cross-Model Transfer"
  - "Syntactic Knowledge"
  - "Training-Free Adapter"
date: 2026-05-08
content_hash: 8239a6249570af74
---

# Order-Level Attention Similarity Across Language Models: A Latent Commonality

**Conference**: NeurIPS 2025
**arXiv**: [2511.05064](https://arxiv.org/abs/2511.05064)  
**Code**: [Available](https://github.com/jinglin-liang/OLAS)  
**Area**: Model Analysis & Compression
**Keywords**: Attention Mechanism, Language Model Similarity, Cross-Model Transfer, Syntactic Knowledge, Training-Free Adapter

## TL;DR

This paper proposes Order-Level Attention (OLA)—an order-wise decomposition of Attention Rollout—and discovers that different language models exhibit significant similarity in same-order OLA (OLAS). OLA is shown to implicitly encode syntactic knowledge, and based on this finding, the paper proposes TOA, the first training-free cross-LM adapter transfer method.

## Background & Motivation

**Core Problem**: Do different language models share common patterns of contextual aggregation?

While existing work (Attention Rollout, attribution analysis, etc.) analyzes the attention mechanisms of individual models, such efforts focus on characterizing individual models rather than systematically studying **commonalities across multiple LMs**. If a shared representational space exists across LMs, efficient cross-model knowledge transfer could be achieved.

**Intuition**: Mainstream transformer LMs all rely on attention mechanisms to aggregate context for prediction. Given similar training objectives and attention mechanisms, different LMs trained on large-scale corpora may converge to similar optimal attention patterns for the same text.

**Limitations of Attention Rollout**: Direct analysis of Attention Rollout suffers from the **Attention Sinks** phenomenon—softmax cannot produce exact zero attention scores, so when a token has already gathered sufficient information, residual attention leaks to irrelevant tokens. This causes Rollout to exhibit similar biased patterns across different texts, reducing its discriminability.

**Key Insight**: Attention Sinks arise because an $N$-layer LM generates $2^N$ information paths, and over-aggregation in high-order paths introduces bias. Analyzing paths by the number of aggregation steps separately makes low-order components more discriminative.

## Method

### Overall Architecture

1. **OLA Definition**: Decompose Attention Rollout by order into comparable representations.
2. **OLAS Discovery**: Validate cross-LM OLA similarity through qualitative and quantitative experiments.
3. **Syntactic Finding**: Demonstrate that OLA implicitly encodes syntactic dependency relations.
4. **TOA Application**: Leverage OLAS to achieve training-free cross-model adapter transfer.

### Key Designs

#### 1. **Derivation of Order-Level Attention (OLA)**

The Attention Rollout for an $N$-layer LM is defined as:
$$\hat{A} = \prod_{i=1}^N (A^{(i)} + I)$$

Expanding into an order-wise decomposition:
$$\hat{A} = I + \sum_{i=1}^N A^{(i)} + \sum_{1 \leq i < j \leq N} A^{(j)}A^{(i)} + \cdots + A^{(N)}\cdots A^{(1)}$$

After normalization, the $k$-th order OLA is defined as:
- Order 0: $\hat{A}^{(0)} = I$ (pure residual connection)
- Order 1: $\hat{A}^{(1)} = \frac{1}{N}\sum_{i=1}^N A^{(i)}$ (mean over paths passing through exactly one attention aggregation)
- Order $k$: mean over $\binom{N}{k}$ paths

Rollout can be rewritten as: $\hat{A} = \sum_{i=0}^N \binom{N}{i} \cdot \hat{A}^{(i)}$

**Design Motivation**: OLA unifies the attention representations of models with different numbers of layers and heads into a common semantic space (order $k$ = exactly $k$ contextual aggregations), enabling cross-model comparison.

#### 2. **Quantitative Validation of OLAS**

**Method 1: Vision Model Proxy Evaluation**

A ResNet-18 image classifier is trained using OLA maps from the source LM as training data (same text → same class), then tested on OLA maps from the target LM.

**Method 2: Image Retrieval Evaluation**

Cross-model OLA retrieval is performed using SSIM similarity, evaluated by Hits@1/Hits@5.

#### 3. **Mapping Between OLA and Syntactic Knowledge**

An auxiliary network trained solely on OLA to predict syntactic dependency relations achieves UAS above 80% on MLMs with first-order OLA, demonstrating that OLA implicitly encodes rich syntactic knowledge. Low-order OLA exhibits more pronounced syntactic features than high-order OLA.

### Loss & Training

Design of TOA (Transferable OLA Adapter):
1. **Training Phase**: The source LM is frozen; a downstream task adapter is trained using stacked first- and second-order OLA as input.
2. **Testing Phase**: The adapter is directly transferred to the target LM without any parameter updates or training data.
3. Since the adapter receives OLA (a unified representation) rather than model-specific hidden states, it is inherently transferable.

## Key Experimental Results

### Main Results

Vision model proxy evaluation (classification accuracy %, CLM results):

| Method | Q-1b5 | Q-7b | G-2b | G-9b | L-3b | L-8b |
|---|---|---|---|---|---|---|
| Rollout | 27.9 | 7.7 | 52.6 | 26.0 | 66.1 | 59.7 |
| 1st OLA | **52.6** | **49.2** | **93.1** | **92.4** | **94.6** | **94.1** |
| 2nd OLA | 67.1 | 49.9 | 89.3 | 86.2 | 90.7 | 91.9 |
| ALTI | 22.6 | 15.5 | 69.3 | 71.8 | 85.6 | 79.8 |

Cross-model TOA transfer on Relation Extraction (RE) (accuracy %):

| Source→Target | Q-1b5 | G-2b | L-3b | Zero-shot |
|---|---|---|---|---|
| TOA from L-3b | 30.49 | 33.49 | 35.57 | - |
| TOA from Q-1b5 | 34.90 | 30.95 | 31.08 | - |
| Zero-shot | 7.69 | 5.01 | 14.65 | Baseline |

### Ablation Study

OLA syntactic dependency prediction (UAS/LAS %):

| LM | Order 1 | Order 2 | Order 3 | Rollout | Note |
|---|---|---|---|---|---|
| Bert-base | 81.29/72.16 | 72.86/61.05 | 66.44/53.17 | 46.20/30.69 | Low >> High |
| Roberta-base | 80.00/70.44 | 72.68/60.10 | 36.99/18.67 | 35.77/17.94 | Same |
| Electra-base | 81.23/72.63 | 77.47/66.78 | 50.72/33.90 | 50.35/34.02 | Same |

Image retrieval evaluation (Hits@1/Hits@5 %, 1st-order OLA):

| Source\Target | Q-1b5 | G-2b | L-3b |
|---|---|---|---|
| Q-1b5 | - | 83.6/89.4 | 95.9/97.0 |
| L-3b | 92.9/96.1 | 94.1/96.5 | - |

### Key Findings

1. **Low-order OLA exhibits the strongest similarity**: First-order OLA achieves the highest cross-model consistency; higher orders introduce more Attention Sinks.
2. **Syntactic information decreases with order**: First-order OLA significantly outperforms higher orders and Rollout in syntactic prediction.
3. **OLAS is a product of pretraining**: Parameter perturbation eliminates OLAS, confirming it originates from learned knowledge rather than experimental artifacts.
4. **CLM-to-CLM similarity exceeds MLM**: Likely due to the more uniform architecture across CLM families.
5. **Training-free transfer is effective**: TOA improves RE performance from 7.69% zero-shot to 34.90% (4.5×).

## Highlights & Insights

1. **Reveals overlooked commonalities across LMs**: LMs with different architectures and training data share a unified "linguistic prior" in their attention aggregation patterns.
2. **Order-wise analysis of Attention Sinks**: Provides a new perspective on the Attention Sinks phenomenon—high-order paths are the primary source of noise.
3. **First training-free cross-LM adapter transfer**: Breaks the constraint that adapters must be bound to a specific model.
4. **Mathematically elegant derivation**: OLA naturally arises from the polynomial expansion of Attention Rollout.

## Limitations & Future Work

1. Validation is limited to basic NLP tasks (RE, NER, DP, POS); extension to generation, reasoning, and other complex tasks remains to be explored.
2. TOA uses only OLA as input without incorporating original model hidden states, which may entail information loss.
3. The choice of OLA orders (stacking orders 1 and 2) is heuristic; the optimal combination warrants further study.
4. Future work may explore using OLA to guide model compression, knowledge distillation, and other cross-model tasks.

## Related Work & Insights

- Compared to the representation learning approach of Moschella et al., OLAS provides a more direct common space across models.
- The finding that OLA encodes syntactic information is consistent with conclusions from probing studies, but is more structured.
- This work may prompt the NLP community to revisit shared properties of attention mechanisms across different models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (OLA is a novel concept, OLAS is an important discovery, TOA pioneers training-free cross-LM transfer)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (12 LMs, qualitative + quantitative analysis, controlled experiments, 4 downstream tasks)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, complete narrative arc from discovery to application)
- Value: ⭐⭐⭐⭐ (Important for understanding LM commonalities, though TOA transfer performance has room for improvement)

The paper discovers that different language models exhibit significant similarity in same-order attention decomposition (Order-Level Attention, OLA), termed OLAS, and proposes TOA to achieve training-free cross-model adapter transfer based on this finding.

## Background & Motivation

Different LMs vary greatly in architecture and training data, yet all rely on attention mechanisms for contextual aggregation. A natural question arises: do different LMs share common patterns of contextual aggregation? Existing research primarily focuses on analyzing individual models or individual attention heads, lacking systematic cross-model studies. Identifying such commonalities could enable efficient cross-model knowledge transfer, eliminating the need to fine-tune adapters from scratch for each new model.

## Method

### Overall Architecture

The technical pipeline consists of three steps: (1) proposing OLA as a unified cross-model attention representation; (2) validating the OLAS phenomenon through qualitative (visualization) and quantitative (classification + retrieval) experiments; and (3) proposing TOA based on OLAS for training-free adapter transfer.

### Key Designs

1. **Order-Level Attention (OLA) Decomposition**: Starting from Attention Rollout, the information flow is decomposed into multiple paths. An $N$-layer model has $2^N$ possible paths. Attention Rollout $\hat{A} = \prod_{i=1}^N (A^{(i)} + I)$ can be expanded as: $\hat{A} = I + \sum_{i}A^{(i)} + \sum_{i<j}A^{(j)}A^{(i)} + \cdots$. The $k$-th order OLA, $\hat{A}^{(k)}$, is the normalized effect of paths passing through exactly $k$ attention aggregations. For example, the first-order OLA is $\hat{A}^{(1)} = \frac{1}{N}\sum_{i=1}^N A^{(i)}$. This decomposition eliminates incomparability across models with different numbers of layers and assigns a unified semantic meaning to same-order attention (order $k$ = exactly $k$ contextual aggregations), enabling cross-model comparison.

2. **Validation of OLAS**:
    - **Qualitative Analysis**: Visualizations of OLA for the same text across different LMs (e.g., Qwen2-1.5b and Llama3.2-3b) reveal high similarity at the same order, while OLA maps for different texts are clearly distinguishable. Higher-order OLA exhibits more severe attention sink effects, indicating that lower-order OLA contains more effective aggregation information.
    - **Quantitative Analysis via Vision Classification Model**: A ResNet-18 classifier is trained to classify OLA maps from a source LM into corresponding text categories, then tested on OLA maps from a target LM. First-order OLA achieves classification accuracy above 90% on CLMs.
    - **Quantitative Analysis via Image Retrieval**: SSIM is used to measure similarity between OLA maps across models. First-order OLA achieves Hits@5 of at least 89% on CLMs, reaching above 97% in the best case.

3. **Implicit Mapping Between OLA and Syntactic Knowledge**: Experiments show that OLA representations alone can predict syntactic dependency relations (Universal Dependencies), indicating that OLA intrinsically encodes syntactic knowledge of the input text.

4. **Transferable OLA Adapter (TOA)**: OLA is used as a unified cross-model syntactic feature representation. An adapter is trained on a source LM using OLA as input for downstream tasks. Due to the cross-model similarity provided by OLAS, the trained adapter can be directly transferred to unseen target LMs without any parameter updates or additional training.

### Loss & Training

The TOA adapter is trained with standard classification/sequence labeling losses. The OLA map classification experiment on the source LM uses cross-entropy loss: $\theta^* = \arg\min_\theta \mathbb{E}_{(a,i)\sim\mathcal{D}_{train}}[\mathcal{L}_{CE}(F_\theta(a), i)]$.

## Key Experimental Results

### Main Results

| Task | Source→Target | Baseline (zero-shot) | TOA Transfer | Gain |
|------|---------|----------------|---------|------|
| Relation Extraction (RE) | LLaMA3-3B→Qwen2-1.5B | 7.69% | 34.90% | +27.2 |
| OLA Visual Classification (CLM 1st) | L-3b,L-8b→Q-1b5 | - | 52.6% | Far exceeds Rollout (27.9%) |
| OLA Visual Classification (CLM 1st) | L-3b,L-8b→G-2b | - | 93.1% | Far exceeds ALTI (69.3%) |
| OLA Retrieval (CLM Hits@5) | L-3b→Q-1b5 | - | 96.1% | Very high retrieval success rate |
| OLA Visual Classification (MLM 1st) | R-b,R-l,E-b,E-l→B-b | - | 91.9% | Far exceeds Rollout (44.3%) |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| 1st-order OLA | CLM classification ≥49.2% | First-order OLA strongest across all configurations |
| 2nd-order OLA | CLM classification slightly lower | Second order has more attention sinks |
| 3rd-order OLA | CLM classification continues to drop | Discriminability decreases at higher orders |
| Attention Rollout | CLM classification 7.7–66.1% | Mixed with ineffective high-order components |
| ALTI | CLM classification 15.5–85.6% | Norm-based method biased toward individual features |
| Parameter perturbation control | OLAS disappears | Confirms OLAS is an intrinsic property of pretrained models |

### Key Findings

- OLAS is a universal phenomenon, validated across 12 LMs (6 CLMs + 6 MLMs).
- First-order OLA achieves the highest cross-model similarity and contains the most effective contextual aggregation information; attention sink effects worsen with higher orders.
- OLA intrinsically encodes syntactic dependency knowledge, providing a linguistic foundation for its use as a unified cross-model representation.
- TOA enables direct transfer of adapters trained on a source LM to target LMs with entirely different architectures, without any fine-tuning.
- Parameter perturbation experiments confirm that OLAS originates from pretrained parameters rather than experimental design artifacts.

## Highlights & Insights

- The order-wise decomposition from Attention Rollout to OLA is a mathematically elegant insight: expanding the matrix product into an ordered sum naturally eliminates incomparability across models with different numbers of layers.
- A new explanation for Attention Sinks: within the OLA framework, sinks are attributed to over-aggregation in high-order paths, where ineffective components overwhelm informative signals.
- The discovery of cross-model attention commonality has far-reaching implications: it suggests that different LMs trained on large-scale corpora may converge to similar optimal attention patterns.
- TOA is the first method to achieve training-free cross-model adapter transfer, offering significant practical value.

## Limitations & Future Work

- The OLA decomposition assumes each layer's attention matrix is averaged across heads, discarding inter-head variation.
- Downstream task validation of TOA is primarily conducted on four NLP tasks (RE/NER/DP/POS) and has not been verified on generation tasks or more complex scenarios.
- OLAS is weaker on MLMs than CLMs, possibly related to the bidirectional attention mechanism of models such as BERT.
- Future work could explore using OLAS for deeper cross-model knowledge distillation or model merging.

## Related Work & Insights

- Attention Rollout (Abnar & Zuidema, 2020) is the direct theoretical foundation of OLA.
- This work is complementary to Relative Representations (Moschella et al., 2023): the latter focuses on aligning representation spaces, while this paper focuses on aligning attention patterns.
- Cross-lingual adapter transfer (Pfeiffer et al., 2020) addresses transfer across languages, whereas TOA addresses transfer across models—an orthogonal dimension.

## Supplementary Discussion

- **Computational complexity of OLA**: First-order OLA requires only averaging attention matrices across layers, incurring negligible overhead; higher-order OLA requires matrix multiplication over path combinations, which can be optimized through caching.
- OLAS performs significantly better on CLMs (Qwen, Gemma, LLaMA) than on MLMs (BERT, RoBERTa, ELECTRA), possibly because the unidirectional nature of autoregressive attention makes it easier to converge to a unified pattern.
- The connection between OLA and syntactic dependencies opens a new perspective for attention interpretability research: low-order OLA may capture local syntactic structure, while high-order OLA may capture long-range dependencies.
- On the RE task, TOA improves Qwen2-1.5B from 7.69% to 34.90%; while the absolute value is modest, it demonstrates the feasibility of cross-model knowledge transfer.

## Supplementary Method Details

### Mathematical Derivation of OLA

The order-wise decomposition of Attention Rollout is essentially a polynomial expansion. For an $N$-layer model:
$$\hat{A} = \prod_{i=1}^N (A^{(i)} + I) = \sum_{k=0}^N \binom{N}{k} \hat{A}^{(k)}$$
where the $k$-th order OLA $\hat{A}^{(k)}$ is the normalized mean effect of all $\binom{N}{k}$ paths passing through exactly $k$ attention aggregations. This decomposition unifies the attention representations of models with different numbers of layers into a common semantic space—the same order corresponds to the same depth of contextual aggregation.

### Input and Output Design of TOA

TOA uses stacked first- and second-order OLA as adapter input features. During training, the source LM is frozen and only adapter parameters are updated. During testing, the adapter is directly applied to OLA generated by the target LM without any parameter adjustment. This design leverages the natural cross-model alignment provided by OLAS, avoiding the feature space transformation or alignment training required by conventional methods.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic discovery and validation of OLAS; OLA decomposition is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive qualitative and quantitative analysis across 12 models; controlled experiments exclude confounding factors.
- Writing Quality: ⭐⭐⭐⭐ Clear logical progression from phenomenon discovery to theoretical explanation to application.
- Value: ⭐⭐⭐⭐ OLAS is significant for understanding LM internal mechanisms; TOA has practical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Geometry of Decision Making in Language Models](geometry_of_decision_making_in_language_models.md)
- [\[ACL 2026\] Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings](../../ACL2026/model_compression/establishing_a_scale_for_kullback-leibler_divergence_in_language_models_across_v.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](../../ACL2026/model_compression/selar_selective_latent_reasoning_in_large_language_models.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)

</div>

<!-- RELATED:END -->
