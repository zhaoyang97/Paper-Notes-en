---
title: >-
  [Paper Note] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing
description: >-
  [ACL 2025][Model Compression] This paper proposes Sci-LoRA, a framework that mixes multi-domain LoRAs. By employing a text encoder trained via contrastive learning, a dynamic weight generator, and a LoRA fusion module, it achieves cross-domain lay paraphrasing of scientific texts across 12 disciplines without requiring domain labels, outperforming the state-of-the-art (SOTA) on 10 metrics across 5 datasets.
tags:
  - "ACL 2025"
  - "Model Compression"
date: 2026-05-08
content_hash: b9998be8c9935597
---

# Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing

**Conference**: ACL 2025  
**arXiv**: [2505.18867](https://arxiv.org/abs/2505.18867)  
**Code**: None  
**Area**: Model Compression  

## TL;DR

This paper proposes Sci-LoRA, a framework that mixes multi-domain LoRAs. By employing a text encoder trained via contrastive learning, a dynamic weight generator, and a LoRA fusion module, it achieves cross-domain lay paraphrasing of scientific texts across 12 disciplines without requiring domain labels, outperforming the state-of-the-art (SOTA) on 10 metrics across 5 datasets.

## Background & Motivation

1. **Proliferation of Interdisciplinary Research**: Modern scientific research increasingly involves interdisciplinary content (e.g., Computer Science + Biology, Chemistry + AI), requiring lay paraphrasing systems for non-expert readers to handle technical texts from mixed domains.
2. **Prior Methods Limited to Single Domains**: Existing works (such as lay summarization in the biomedical field) fine-tune models only for a single domain and ignore cross-domain generalization capability, which may lead to misunderstandings when facing interdisciplinary content.
3. **Cross-Domain Interference in Single LoRA**: Training a single LoRA on data from all domains causes mutual interference among cross-domain knowledge, making it difficult to balance domain specificity and generality.
4. **Adaptation to New Domains Requires Full Retraining**: When new interdisciplinary domains emerge, existing models require full retraining, which is inefficient and expensive.
5. **Inflexible Static LoRA Merging Weights**: Existing Mixture of LoRAs methods mostly employ static equal-weight merging or simple routing, failing to dynamically adjust the contribution of each LoRA based on the domain characteristics of the input text.
6. **Blurry Domain Boundaries**: Interdisciplinary texts might involve multiple domains simultaneously; typically, no explicit domain labels are available during inference, requiring the model to automatically determine domain affiliation and fuse them with dynamic weights.

## Method

### Overall Architecture

Sci-LoRA consists of three core modules: **Domain LoRA Training**, **Adapter Weight Generator**, and **Dynamic LoRA Fusion**.

### 1. Domain LoRA Training

- Base model: **Qwen2.5-7B-Instruct** (Apache-2.0 open-source, excellent performance in long-text generation).
- One LoRA adapter is trained for each of the **12 domains**, with each LoRA trained solely on data from its corresponding domain.
- LoRA parameters: learning rate 1e-4, batch size 4, rank 8, maximum document length 2048.
- LLaMA-Factory is utilized for efficient fine-tuning, with an early stopping strategy for model selection.

### 2. Adapter Weight Generator (AWG)

Achieves dynamic weight allocation without domain labels in two steps:

**Text Encoder (Fine-tuned with Contrastive Learning)**:
- Based on Sentence-BERT, fine-tuned using cross-domain subset data via contrastive learning.
- Positive pairs: Different texts in the same domain $(x_i, x_j)$.
- Negative pairs: Texts from different domains $(x_i, x_k)$.
- InfoNCE Contrastive Loss:
$$\mathcal{L} = \frac{e^{-\|x_i - x_j\|^2 / \tau}}{e^{-\|x_i - x_j\|^2 / \tau} + \sum_{k=1}^{m} e^{-\|x_i - x_k\|^2 / \tau}}$$
- Objective: To enable the encoder to better distinguish text representations across different domains.

**Weight Generation**:
- Performs **K-Means clustering** (K=10) on the training data embeddings of each domain, taking the average of the data points closest to the centroid as the domain adapter representation $r_{\triangle\theta_i}$.
- During inference, weights are calculated based on the distance between the input text embedding and each domain representation: $\alpha_i = \frac{1}{1 + \|E(x_i) - r_{\triangle\theta_i}\|_2}$.
- Closer distance yields a higher weight, automatically achieving domain-relevant weighting.

### 3. Dynamic LoRA Fusion

Simultaneously maintains and fuses two pathways of generative representation:

- **Specialized Representation**: All domain LoRAs are weighted and merged based on the dynamic weights $\alpha$ and then injected into the base model: $r_{specialized} = \mathcal{M}(\theta + \sum_{i=1}^n \alpha_i \triangle\theta_i, x_i)$.
- **Generalized Representation**: An additional unified LoRA $\triangle\theta_0$ trained on all domain data is used: $r_{generalized} = \mathcal{M}(\theta + \triangle\theta_0, x_i)$.
- **Final Fusion**: $\hat{r} = \beta \cdot r_{specialized} + (1-\beta) \cdot r_{generalized}$, where $\beta=0.5$ balances domain specificity and cross-domain generalization.

## Key Experimental Results

### Table 1: Main Results (d-BLEU % / BERTScore F1 %, partial domain display)

| Model | CELLS d-BLEU | PLOS d-BLEU | ALS d-BLEU | AAD d-BLEU | CELLS BERT | PLOS BERT | ALS BERT | AAD BERT |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 5.10 | 5.76 | 11.05 | 12.40 | 81.13 | 81.81 | 83.55 | 81.80 |
| Qwen2.5 | 9.26 | 10.18 | 25.55 | 28.71 | 82.36 | 82.70 | 84.98 | 82.55 |
| DSPT5 | - | - | 24.95 | 33.53 | - | - | 85.48 | 83.70 |
| **Sci-LoRA** | **11.15** | **12.43** | **31.03** | **38.97** | **83.00** | **83.35** | **86.01** | **84.37** |

Sci-LoRA achieves the best performance across almost all domains and metrics, with an average improvement of 5-10 percentage points in d-BLEU compared to the Qwen2.5 LoRA baseline, and outperforms DSPT5, which is fully fine-tuned individually for each domain.

### Table 2: Ablation Study (AAD domain, VTechAGP dataset)

| Configuration | s-BLEU | d-BLEU | BERTScore | ROUGE1 | METEOR | SARI |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Pre-trained | 5.63 | 12.69 | 81.97 | 42.46 | 33.98 | 34.64 |
| Single LoRA (All-Domain) | 13.43 | 28.71 | 82.55 | 46.51 | 39.85 | 39.39 |
| Multi-LoRAs (Direct Multi-Domain) | 19.78 | 34.39 | 83.30 | 47.83 | 41.77 | 40.76 |
| AWG + K-Means | 23.17 | 37.08 | 83.72 | 51.13 | 43.91 | 43.56 |
| AWG + Contrastive | 18.06 | 38.62 | 84.03 | 52.07 | 44.89 | 44.00 |
| w/o Fusion | 14.86 | 31.22 | 81.59 | 50.82 | 41.42 | 43.63 |
| **Sci-LoRA (Full)** | **18.38** | **38.97** | **84.37** | **52.69** | **46.71** | **44.26** |

Each component contributes to the overall performance: Multi-domain LoRA > Single LoRA; K-Means domain representation > Random sampling; Contrastive learning encoder brings further improvements; Dynamic fusion module is indispensable.

### Human Evaluation

| Model | Comprehensiveness | Simplicity | Semantic Fidelity | Conciseness | Fluency |
|:---|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 3.25 | 2.68 | 3.08 | **3.53** | 3.15 |
| Qwen2.5 | 3.78 | 2.86 | 2.80 | 2.77 | 3.31 |
| **Sci-LoRA** | **3.82** | **2.88** | **3.45** | 3.47 | **3.40** |

Sci-LoRA is optimal in comprehensiveness, semantic fidelity, and fluency, while maintaining competitive conciseness.

## Highlights & Insights

- **Dynamic LoRA Weighting Without Domain Labels**: Achieves automatic domain identification and weight allocation through contrastive learning and K-Means clustering, completely independent of domain labels during inference.
- **Inherent Dual-Pathway Fusion Strategy**: The specialized pathway (weighted multiple LoRAs) and the generalized pathway (all-domain LoRA) complement each other, preserving domain-specific knowledge while mitigating overfitting.
- **Comprehensive Evaluation Across 12 Domains and 5 Datasets**: 10 automatic metrics + 5-dimensional human evaluation, reflecting sufficient experimental scale and evaluating angles.
- **Scalable Architecture**: New domains only require appending a new LoRA adapter without retraining the entire system.

## Limitations & Future Work

- **Limited Scalability in the Number of Domains**: Currently relying on the PEFT library to load and merge all LoRAs during inference, which could exponentially increase inference latency when the number of domains scales to hundreds.
- **Inability to Handle New Domains Without Training Data**: Each domain LoRA requires domain-specific data fine-tuning, lacking support for zero-shot adaptation to entirely new domains.
- **Validated Solely on Qwen2.5-7B**: The performance variance across different foundational models (e.g., LLaMA, Mistral) has not been explored, leaving its model-level generalization unverified.
- **Fixed Fusion Ratio of $\beta=0.5$**: The fusion weight between the specialized and generalized pathways is fixed at 0.5 rather than adaptively adjusted based on the input text, which may not be optimal.

## Related Work & Insights

| Dimension | Sci-LoRA | DSPT5 (Cheng et al., 2025) | Single LoRA Fine-Tuning |
|:---|:---|:---|:---|
| Training Paradigm | One LoRA per domain + one global LoRA | Fully fine-tune one model per domain | Train one LoRA on all-domain data |
| Inference Requirements | No domain labels required | **Requires** domain labels to select model | No domain labels required |
| Domain Scalability | Append LoRA adapter | Requires training a completely new model | Requires retraining the LoRA |
| Cross-Domain Generalization | Dynamic weighted fusion | No cross-domain capability | Cross-domain interference |
| Performance | Optimal across all metrics | Suboptimal | Lower than Sci-LoRA by approx. 5-10 pp |

| Dimension | Sci-LoRA | Mixture of LoRAs (Router-based) | Mixture of LoRAs (Linear Merge) |
|:---|:---|:---|:---|
| Weight Generation | Contrastive Learning Encoder + K-Means | Extra Routing Network | Static Equal Weights |
| Domain Representation | Clustered Centroid Embedding | No Explicit Representation | None |
| Fusion Strategy | Dual-pathway (Specialized + Generalized) Dynamic Fusion | Single-pathway | Single-pathway |
| Applicable Task | Cross-domain Lay Paraphrasing | General Multi-task | General Multi-task |

## Rating

- ⭐⭐⭐⭐ Novelty: The combination of contrastive learning-driven dynamic LoRA weight generation and the dual-pathway fusion strategy is a novel approach for lay paraphrasing.
- ⭐⭐⭐⭐ Practicality: Validated across 12 domains, providing a scalable multi-domain adaptation solution with direct applicability to science writing.
- ⭐⭐⭐⭐ Experimental Thoroughness: Comprehensive evaluation covering 5 datasets, 12 domains, 10 metrics, human evaluation, and exhaustive ablations.
- ⭐⭐⭐⭐ Writing Quality: Clear description of methodology, complete formulations, with helpful ablation studies and visualizations (t-SNE) for understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](domix_an_efficient_framework_for_exploiting.md)
- [\[CVPR 2026\] TAS-LoRA: Transformer Architecture Search with Mixture-of-LoRA Experts](../../CVPR2026/model_compression/tas-lora_transformer_architecture_search_with_mixture-of-lora_experts.md)

</div>

<!-- RELATED:END -->
