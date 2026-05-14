---
title: >-
  [Paper Note] Your Language Model Secretly Contains Personality Subnetworks
description: >-
  [Information Retrieval & RAG] This paper proposes extracting persona-specific subnetworks from pretrained LLMs via activation-guided pruning, enabling efficient persona switching without any training…
tags:
  - "Information Retrieval & RAG"
date: 2026-05-08
content_hash: c4198f06fc3f38b0
---

# Your Language Model Secretly Contains Personality Subnetworks

## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2602.07164](https://arxiv.org/abs/2602.07164)
- **Code**: [GitHub](https://github.com/Ruimeng-Ye/Persona)
- **Area**: Information Retrieval
- **Keywords**: persona subnetwork, network pruning, contrastive pruning, MBTI, activation-guided masking

## TL;DR

This paper proposes extracting persona-specific subnetworks from pretrained LLMs via activation-guided pruning, enabling efficient persona switching without any training, and introduces a contrastive pruning strategy to enhance parameter separation between opposing personas.

## Background & Motivation

- Humans naturally switch personas across social contexts; LLMs can likewise adopt different roles, but existing methods rely on external knowledge injection.
- **Prompt-based methods**: Simple and fast, but persona consistency is unstable and prone to drift.
- **RAG methods**: Require retrieval pipelines and suffer from interference issues.
- **Fine-tuning methods**: Require additional training at high cost (hours to days).
- **Core Problem**: Do LLMs truly require external intervention to exhibit different personas, or are these behaviors already embedded in the parameter space?
- Inspired by the Lottery Ticket Hypothesis, the authors hypothesize that a single pretrained model **already contains** multiple "winning ticket" subnetworks corresponding to distinct personas.

## Method

### Overall Architecture

Given a pretrained LLM and a small persona calibration dataset → collect activation statistics → construct binary masks → isolate persona subnetworks → apply masks at inference time to enable persona switching.

### Problem Formulation

For each persona $p \in \mathcal{P}$, given a small calibration set $\mathcal{D}_p = \{(x_i^p, y_i^p)\}_{i=1}^{N_p}$, the objective is to find a mask that maximizes persona alignment:

$$\max_{\mathbf{M}^p} \mathbb{E}_{(x,y) \sim \mathcal{D}_p} [\log P_{\mathcal{M}_p}(y|x)]$$

subject to the sparsity constraint $\|\mathbf{M}^p\|_0 \leq (1 - \rho) d$, where $\rho$ is the target sparsity ratio.

### Activation-Based Importance Scoring

For each layer $l$, activation statistics are collected over the persona calibration data:

$$\mathbf{A}_p^{(l)}[j] = \mathbb{E}_{(x,y) \sim \mathcal{D}_p} [|\mathbf{h}_j^{(l)}(x)|]$$

Importance scores are computed by combining weight magnitudes with activations:

$$S_{ij}^p = |w_{ij}| \cdot \mathbf{A}_p^{(l)}[j]$$

For each output channel $i$, the Top-K most important input channels are retained to form the binary mask $\mathbf{M}^p$.

### Contrastive Pruning

For opposing persona pairs (e.g., Introversion/Extraversion), standard pruning may yield highly overlapping masks. Contrastive pruning maximizes parameter separation by differentiating activation patterns:

**Contrastive-Wanda variant**:

$$S_{ij}^p = |w_{ij}| \cdot \phi\left(\frac{\mu_{ij}^{p_+} - \mu_{ij}^{p_-}}{\sqrt{\sigma_{ij}^{p_+} + \sigma_{ij}^{p_-}} + \varepsilon}\right)$$

**Contrastive-Sparse variant**:

$$C_{ij} = |\tilde{S}_{ij}^{p_+} - \tilde{S}_{ij}^{p_-}|, \quad \tilde{S}_{ij}^p = \frac{S_{ij}^p}{\sum_k S_{ik}^p}$$

Each parameter is assigned to the persona with the higher score, constructing disjoint masks $\mathbf{M}^{p_+}, \mathbf{M}^{p_-}$.

### Dynamic Mask Inference

At inference time, masks are applied directly without modifying the original weights:

$$\mathbf{y} = (\mathbf{W} \odot \mathbf{M}^p) \mathbf{x} + \mathbf{b}$$

An optional soft gate $G = \mathbf{M}^p + \gamma(1 - \mathbf{M}^p)$ is supported, where $\gamma = 0$ reduces to the standard hard mask.

## Experiments

### Datasets & Models

- **Datasets**: MBTI (16 personality types), AI Persona (power-seeking / wealth-seeking / hallucination detection), RoleAgentBench (role-playing)
- **Models**: LLaMA-2-13B, LLaMA-3-8B, Qwen2.5-14B

### Main Results

**AI Persona Classification (LLaMA-2-13B)**:

| Method | Power-Seeking | Wealth-Seeking | Hallucination |
|------|:---:|:---:|:---:|
| Prompt | 41.0% | 44.0% | 58.5% |
| RAG | 45.5% | 50.5% | 64.5% |
| Wanda | 51.5% | 54.5% | 89.0% |
| Contrastive Wanda | 54.0% | 66.0% | 95.0% |
| Contrastive Sparse | **56.5%** | 64.5% | **96.0%** |
| SFT (upper bound) | 64.0% | 71.0% | 97.5% |

Contrastive pruning improves over prompt-based methods by +15.5 on Power-Seeking and +20.5 on Wealth-Seeking.

**RoleAgentBench Role-Playing (LLaMA-3-8B)**:

| Method | Friends | Harry Potter | Sherlock | Big Bang | Venice |
|------|:---:|:---:|:---:|:---:|:---:|
| Prompt | 18.37 | 42.06 | 42.11 | 29.55 | 41.67 |
| Sparse | **51.02** | **53.97** | **60.53** | **61.76** | **70.83** |

### Ablation Study

**Mask Analysis**:

| MBTI Dimension | Avg. Difference Rate (%) | Attn | MLP |
|-----------|:---:|:---:|:---:|
| I vs. E | 1.34 | 1.28 | 1.44 |
| F vs. T | 1.08 | 1.03 | 1.14 |
| N vs. S | 0.75 | 0.75 | 0.76 |
| J vs. P | 0.76 | 0.73 | 0.79 |

- The I/E and F/T dimensions exhibit larger differences → better switching performance.
- MLP layer differences consistently exceed those of Attention layers → persona separation primarily relies on FFN transformations.

**Impact on General Capabilities (LLaMA-3-8B)**:

| Method | MMLU | HellaSwag |
|------|:---:|:---:|
| Base Model | 0.378 | 0.675 |
| Wanda | 0.369 | 0.668 |
| Sparse | 0.362 | 0.653 |

Degradation in general capabilities after pruning is minimal (≤1.6%), indicating that persona subnetworks occupy only a small fraction of model capacity.

## Highlights & Insights

1. **Novel Perspective**: The first work to interpret persona representations in LLMs through the lens of the Lottery Ticket Hypothesis, demonstrating that persona behaviors are intrinsic rather than externally induced.
2. **Training-Free**: Requires no gradient updates; only a small calibration dataset (hundreds to thousands of samples) is needed.
3. **Contrastive Pruning**: A purpose-designed strategy that effectively enhances parameter disentanglement between opposing personas.
4. **Practical Efficiency**: Mask construction requires only minutes of computation, enabling rapid persona switching.

## Limitations & Future Work

1. Mask separation for the N/S and J/P dimensions is relatively weak, leading to unstable switching performance on these personality axes.
2. Cosine similarity between certain persona pairs remains high in upper layers (e.g., INFJ–INFP reaches 0.9883 at layer L39), indicating that deep-layer entanglement is difficult to resolve.
3. Validation is currently limited to models at the 13B scale; transferability to larger or smaller models remains unknown.
4. The quality and representativeness of calibration data may affect pruning efficacy.

## Related Work & Insights

- **Persona Modeling**: Prompting (Shao et al., 2023), RAG (Zerhoudi, 2024), fine-tuning (Zhou et al., 2023)
- **Network Pruning**: Lottery Ticket Hypothesis (Frankle & Carlin, 2019), Wanda (Sun et al., 2023), SparseGPT (Frantar & Alistarh, 2023)
- **Mechanistic Interpretability**: Truth direction (Li et al., 2023), activation steering (Zou et al., 2022), FFN key-value memory (Geva et al., 2023)

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying pruning for persona discovery rather than compression offers a genuinely fresh perspective.
- **Technical Contribution**: ⭐⭐⭐⭐ — The contrastive pruning design is well-motivated with clear theoretical intuition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, three models, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized with rich figures and tables.
- **Overall Score**: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LangBridge: Interpreting Image as a Combination of Language Embeddings](../../ICCV2025/information_retrieval/langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)
- [\[NeurIPS 2025\] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](../../NeurIPS2025/information_retrieval/murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)
- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ACL 2026\] FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](../../ACL2026/information_retrieval/flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)
- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)

</div>

<!-- RELATED:END -->
