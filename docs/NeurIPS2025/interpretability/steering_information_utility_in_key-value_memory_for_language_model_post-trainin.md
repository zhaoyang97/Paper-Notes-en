---
title: >-
  [Paper Note] Steering Information Utility in Key-Value Memory for Language Model Post-Training
description: >-
  [NeurIPS 2025][Interpretability][post-training optimization] This paper proposes InfoSteer, a lightweight method that treats the FFN layers of Transformers as associative key-value memories…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "post-training optimization"
  - "FFN key-value memory"
  - "information steering"
  - "memory vector activation"
  - "SFT enhancement"
date: 2026-05-08
content_hash: 0ae6a0762f276cf3
---

# Steering Information Utility in Key-Value Memory for Language Model Post-Training

**Conference**: NeurIPS 2025
**arXiv**: [2507.05158](https://arxiv.org/abs/2507.05158)  
**Code**: [GitHub](https://github.com/chili-lab/InfoSteer)  
**Area**: Interpretability
**Keywords**: post-training optimization, FFN key-value memory, information steering, memory vector activation, SFT enhancement

## TL;DR

This paper proposes InfoSteer, a lightweight method that treats the FFN layers of Transformers as associative key-value memories, promoting more complete utilization of pretrained knowledge during post-training via forward-pass intervention (boosting key coefficients of low-activation memory vectors) and backward-pass regularization (maximizing the entropy of key distributions). Across 6 models from 3 model families (Qwen/LLaMA/Gemma) and 15 in-distribution and out-of-distribution tasks, consistent improvements are observed, and steered language models exhibit adaptive information allocation behavior.

## Background & Motivation

The standard LLM training pipeline consists of two stages: pretraining followed by post-training (SFT/RLHF). Extensive research has established that a model's foundational capabilities and knowledge are primarily acquired during pretraining, with post-training serving to refine and align these capabilities. However, a critical issue has been overlooked:

**Underutilization of pretrained knowledge**: Standard SFT neither explicitly trains nor incentivizes models to retrieve and apply knowledge stored during pretraining. For instance, Gemma-2-9B achieves 90.1% on HellaSwag as a base model, yet reaches 95.7% (+5.6%) with InfoSteer, demonstrating that substantial knowledge remains unactivated by vanilla SFT.

**The FFN-as-key-value-memory interpretation**: The seminal work of Geva et al. (2021) shows that the first layer of an FFN produces key coefficients, the row vectors of the second layer serve as values (memory vectors), and the FFN output is a weighted sum of memory vectors: $\text{FFN}(h) = \sum_{i=1}^{d_m} k_i \mathbf{v}_i$. When a particular $k_i$ greatly dominates all other $k_j$, only $\mathbf{v}_i$ is effectively utilized, while all other memory vectors are neglected.

**Absence of knowledge-utilization guidance**: No mechanism exists during post-training to guide the model toward better exploitation of stored knowledge — even when relevant knowledge is encoded in the parameters, the model may fail to access it due to overly concentrated key distributions.

The central objective of InfoSteer is to encourage high-entropy key coefficient distributions in FFN layers during post-training, enabling more memory vectors to be activated and thereby unlocking knowledge acquired during pretraining that would otherwise remain dormant.

## Method

### Overall Architecture

InfoSteer integrates seamlessly into the standard SFT pipeline and provides two complementary steering mechanisms: (1) forward-pass intervention — directly modifying key coefficients; and (2) backward-pass regularization — incorporating an entropy term into the loss function. Both approaches share the common goal of producing more uniform key distributions and activating a greater number of memory vectors.

### Key Designs

1. **Intervention**: During the forward pass of each FFN layer, the bottom $p\%$ (default $p=0.01$) of key coefficients are identified and elevated to $\alpha$ times the layer-wise mean key coefficient:

$$k_s^{(l)} \leftarrow \alpha \cdot \frac{1}{d_m}\sum_{i=1}^{d_m} k_i^{(l)}, \quad \text{for } s \in \mathcal{I}^{(l)}$$

where $\mathcal{I}^{(l)}$ denotes the index set of the bottom $p\%$ elements in layer $l$. **Design Motivation**: Directly increasing the contribution of neglected memory vectors so that they participate in the FFN output. $\alpha$ controls intervention strength — experiments show $p=1, \alpha=2$ is optimal, while excessive intervention ($p=2, \alpha=5$) degrades performance.

2. **Regularization**: An entropy regularization term over the key distribution is added to the training loss:

$$\mathcal{L} = \mathcal{L}_{\text{LM}} - \lambda \sum_{l=1}^{L} H(\hat{\mathbf{k}}^{(l)})$$

where $\hat{\mathbf{k}}^{(l)} = \mathbf{k}^{(l)} / \sum_i k_i^{(l)}$ is the normalized key distribution and $\lambda=0.01$ controls regularization strength. The negative sign ensures that minimizing the loss is equivalent to maximizing entropy, encouraging more uniform key activation patterns. **Design Motivation**: Indirectly guiding the model to learn more uniform key activation patterns through gradient signals — a softer approach compared to intervention.

3. **Fine-grained Memory Vector Analysis (Information Surrogate)**: To understand what each memory vector $\mathbf{v}_i$ encodes, it is projected through the language model decoding head to produce logits over the vocabulary:

$$\phi_i = \mathbf{v}_i \cdot W_{\text{decode}} \in \mathbb{R}^{|V|}$$

Applying softmax to $\phi_i$ yields a probability distribution $P_i$, whose entropy $H(P_i)$ is computed: low entropy indicates high-specificity vectors (encoding knowledge about specific topics, e.g., {'quantum', 'physics', 'superposition'}), while high entropy corresponds to general-purpose vectors. This enables topic-level targeted steering — however, experiments show that fine-grained steering yields only marginal improvements, and the general-purpose approach is already sufficient.

4. **SwiGLU Adaptation**: Modern LLMs (Qwen/LLaMA/Gemma) employ the SwiGLU variant. InfoSteer defines key coefficients as $\mathbf{k} = \sigma(hW_{\text{gate}}) \odot (hW_{\text{up}})$ — i.e., the intermediate activations before the down-projection. The core idea is to focus on the "input associated with memory vectors prior to weighting," regardless of whether the architecture uses standard FFN or gated FFN.

### Loss & Training

The base training follows standard SFT (cross-entropy language modeling loss), with InfoSteer adding only the intervention or regularization on top. Default hyperparameters: intervention with $p\%=0.01, \alpha=1$; regularization with $\lambda=0.01$. All experiments report mean scores over 3 independent runs. Training data and hyperparameter settings are identical to vanilla SFT, requiring only a single-line code modification.

## Key Experimental Results

### Main Results — In-Distribution Evaluation (Average Accuracy over 9 Tasks)

| Model | Method | BoolQ | PIQA | HellaSwag | WinoG | ARC-c | Note |
|-------|--------|-------|------|-----------|-------|-------|------|
| Qwen-2.5-1.5B | base | 64.2 | 78.5 | 80.1 | 76.4 | 61.2 | — |
| | vanilla SFT | 68.5 | 82.9 | 84.8 | 80.8 | 65.8 | — |
| | **+intervention** | **69.3** | **84.4** | **93.1** | **84.2** | **68.2** | HellaSwag +8.3 |
| Gemma-2-9B | base | 71.6 | 86.3 | 90.1 | 82.5 | 77.8 | — |
| | vanilla SFT | 74.3 | 90.1 | 94.8 | 86.9 | 82.0 | — |
| | **+intervention** | **77.2** | **91.8** | **95.7** | **88.5** | **83.4** | Consistent gains |
| LLaMA-3-8B | base | 70.3 | 85.6 | 90.8 | 81.9 | 75.3 | — |
| | **+intervention** | **77.1** | **90.2** | **96.3** | **87.4** | **81.6** | HellaSwag +5.5 |

### OOD Evaluation (Trained on GSM8K, Tested on 5 Arithmetic Datasets)

| Method | ID Eval (GSM8K) | OOD Eval (5 datasets avg) | Note |
|--------|----------------|--------------------------|------|
| Base Model | 63.7 | 85.3 | — |
| + Vanilla SFT | 65.7 (+2.0) | 83.7 (**-1.6**) | OOD degradation due to SFT overfitting |
| + Steered SFT w/ intervention | **66.8 (+3.1)** | **86.6 (+1.3)** | Gains on both ID and OOD |
| + Steered SFT w/ regularization | 66.1 (+2.4) | 86.0 (+0.7) | Regularization also effective |

### Ablation Study — Steering Magnitude

| Configuration | Avg. Accuracy | Note |
|---------------|--------------|------|
| Base Model | 71.4 | — |
| + Vanilla SFT | 72.6 | — |
| + intervention (p=1, α=1) | 73.8 | Mild intervention |
| + intervention (p=1, α=2) | **75.5** | Optimal intervention strength |
| + intervention (p=2, α=5) | 72.8 | Excessive intervention hurts |
| + regularization (λ=-0.01) | 72.3 | Negative entropy → degraded performance (ablation) |
| + regularization (λ=0.01) | 73.4 | — |
| + regularization (λ=0.05) | 74.7 | Stronger regularization |

### Task-Type Analysis

| Task Type | Avg. Gain (Steered SFT) | Rank |
|-----------|------------------------|------|
| Reading Comprehension | +3.9 | 1 |
| Knowledge-Intensive | +3.3 | 2 |
| Commonsense Reasoning | +2.3 | 3 |
| Mathematics | +1.1 | 4 |
| Linguistics | -0.3 | 5 |

### Key Findings

- **Pretrained knowledge is substantially underutilized**: All three model families (Gemma/LLaMA/Qwen) benefit significantly from InfoSteer, indicating this is a general phenomenon rather than an isolated case. Gemma-2-9B improves from 90.1% to 95.7% (+5.6%) on HellaSwag without additional pretraining or parameter expansion.
- **Overfitting risk of vanilla SFT**: In OOD experiments, vanilla SFT improves ID performance (+2.0) at the cost of OOD generalization (-1.6). InfoSteer yields simultaneous improvements on both, suggesting it encourages more general knowledge utilization rather than distribution-specific memorization.
- **Adaptive information allocation**: Steered language models allocate more representational resources to semantically rich tokens (nouns, verbs) and fewer to simple transitional tokens (',', 'and') — a behavior that is not prominent in vanilla SFT models.
- **Knowledge-intensive tasks benefit most**: Reading comprehension (+3.9) and knowledge-intensive tasks (+3.3) gain far more from steering than linguistics tasks (-0.3), consistent with the intuition that linguistics tasks rely more on surface patterns than deep knowledge.

## Highlights & Insights

1. The finding that **"pretrained knowledge is substantially underutilized"** is profound and significant, challenging the implicit assumption that "SFT = knowledge utilization." It implies that existing SFT pipelines may suffer from systematic efficiency losses.
2. The **perspective of manipulating FFN layers as key-value memories** is remarkably elegant — no architectural modifications or additional parameters are required; releasing more pretrained knowledge is achieved solely by altering the distribution of key coefficients.
3. **Implementation is minimal**: the intervention can be realized in a single line of code (identify the bottom $p\%$ and set to $\alpha$ times the mean), and regularization requires only one additional term in the loss function.
4. The **negative entropy regularization ablation** ($\lambda=-0.01$ → performance degradation) constitutes a clean control experiment that substantially strengthens the causal argument.

## Limitations & Future Work

- Fine-grained steering (selectively activating vectors based on semantic specificity analysis) yields only marginal improvements and does not fully exploit its potential; more precise layer-level localization strategies may be needed.
- Theoretically, high entropy in key distributions is not necessarily always beneficial — certain tasks may require highly sparse memory activation patterns (e.g., precise factual retrieval). This scenario is currently not analyzed.
- The method exhibits a slight negative effect on linguistics tasks (-0.3), indicating that "activating more memory vectors" is not universally beneficial across all task types.
- Validation is limited to the SFT setting; applicability to other post-training paradigms such as RLHF and DPO remains unexplored.

## Related Work & Insights

- **Distinction from model steering** (activation steering, representation engineering): InfoSteer does not impose directional interventions in activation space; rather, it applies entropy-based guidance to FFN key distributions, making it closer to "information utilization optimization" than "behavioral control."
- The FFN key-value memory interpretation of Geva et al. serves as the theoretical foundation of this work. InfoSteer elevates this interpretation from an analytical tool to a practical training intervention.
- **Comparison with knowledge distillation**: distillation transfers knowledge from an external teacher, whereas InfoSteer releases knowledge already encoded in the model's own parameters. The two approaches may be complementary.

## Rating

- **Novelty**: ⭐⭐⭐⭐ FFN key-value memory steering is a novel perspective that converts an analytical interpretation into an actionable training intervention
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 6 models (3 families × 2 scales), 15+ tasks, ID+OOD evaluation, ablation studies, and task-type analysis
- **Writing Quality**: ⭐⭐⭐⭐ Method intuition is clearly presented; FFN-as-memory visualizations are convincing
- **Value**: ⭐⭐⭐⭐⭐ A plug-and-play SFT improvement applicable to all Transformer models with FFN layers

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] Curvature Tuning: Provable Training-free Model Steering From a Single Parameter](curvature_tuning_provable_training-free_model_steering_from_a_single_parameter.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[NeurIPS 2025\] URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)

</div>

<!-- RELATED:END -->
