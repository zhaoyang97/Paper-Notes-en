---
title: >-
  [Paper Note] qa-FLoRA: Data-free Query-Adaptive Fusion of LoRAs for LLMs
description: >-
  [AAAI 2026][Medical Imaging][LoRA fusion] This paper proposes qa-FLoRA, a query-adaptive LoRA fusion method that requires neither training data nor a training process. It dynamically determines fusion weights by computing per-layer KL divergence between each adapter and the base model, achieving significant improvements over static fusion and training-free baselines across nine multilingual composite tasks.
tags:
  - AAAI 2026
  - Medical Imaging
  - LoRA fusion
  - query-adaptive
  - training-free
  - KL divergence
  - multi-domain adaptation
date: 2026-05-08
content_hash: 74eed64e12c7dce2
---

# qa-FLoRA: Data-free Query-Adaptive Fusion of LoRAs for LLMs

**Conference**: AAAI 2026
**arXiv**: [2512.11366](https://arxiv.org/abs/2512.11366)
**Code**: None
**Area**: Medical Imaging
**Keywords**: LoRA fusion, query-adaptive, training-free, KL divergence, multi-domain adaptation

## TL;DR

This paper proposes qa-FLoRA, a query-adaptive LoRA fusion method that requires neither training data nor a training process. It dynamically determines fusion weights by computing per-layer KL divergence between each adapter and the base model, achieving significant improvements over static fusion and training-free baselines across nine multilingual composite tasks.

## Background & Motivation

Deploying large language models (LLMs) in domain-specific settings typically requires parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation). However, when handling **cross-domain composite queries** (e.g., solving math problems in Chinese, answering medical questions in Russian), a single LoRA adapter is insufficient, necessitating the simultaneous fusion of multiple domain experts.

Existing LoRA fusion methods suffer from the following limitations:

**Static Fusion**: Assigns equal weights to all adapters regardless of query content, lacking adaptability and yielding limited performance.

**Supervised Dynamic Fusion** (e.g., LoRAFlow, LoRAHub): Requires collecting composite training data for each adapter combination and training a router or gating network, resulting in poor scalability. As the number of adapters grows, the combinatorial explosion of configurations dramatically increases data collection and training costs.

**Training-Free Methods** (e.g., Centroid Similarity): Assign weights by precomputing centroid vectors of domain data and comparing cosine similarity. While this avoids training, it still relies on domain-representative data for centroid computation and fails to capture the distributional shifts introduced by adapters at different layers.

The core insight of qa-FLoRA is: **when a LoRA adapter is semantically relevant to the input query, it injects meaningful task-specific information into the base model, causing the adapter's output distribution to deviate measurably from that of the base model.** This degree of divergence serves as a proxy for semantic relevance, enabling dynamic fusion weight computation without any external data or training.

## Method

### Overall Architecture

The qa-FLoRA pipeline consists of three steps:

1. **Per-layer probability distribution extraction**: The input query is passed through both the base model and each LoRA adapter to obtain hidden states at every layer, which are then projected into the vocabulary space to yield probability distributions.
2. **Distribution divergence computation and fusion weight derivation**: KL divergence is used to measure the distributional discrepancy between each adapter and the base model at every layer; the divergence values are normalized to produce fusion weights.
3. **Per-layer adaptive fusion**: The computed per-layer weights are used to combine the contributions of each adapter, producing the final prediction.

### Key Designs

1. **Per-layer vocabulary-space projection**: To enable meaningful distribution comparisons across layers, the hidden state $\mathbf{h}^{(l)}$ at each layer $l$ is projected into the vocabulary space via the pretrained LM head $W_{LM}$ to obtain logits $\mathbf{z}^{(l)} = W_{LM} \mathbf{h}^{(l)}$, followed by softmax normalization to yield probability distributions. Although the LM head was originally designed to process only the final layer's hidden states, experiments show that applying it to intermediate layers also produces well-calibrated logits. This design enables comparison of base model and adapter behavior in a unified probability space.

2. **KL divergence-based semantic relevance measure**: For each layer $l$, the KL divergence between the base model distribution $p^{(l)}$ and the $j$-th adapter distribution $q_j^{(l)}$ is computed as: $div_j^{(l)} = D_{KL}(p^{(l)}[-1] \| q_j^{(l)}[-1])$. Only the distribution of the **last token** of the query is used (ablations show this outperforms averaging over all tokens). A larger KL divergence indicates that the adapter injects more task-specific information absent from the base model, reflecting higher semantic relevance to the query.

3. **Per-layer weight normalization**: The KL divergence values at each layer are normalized across all adapters to obtain fusion weights: $\alpha_j^{(l)} = \frac{div_j^{(l)}}{\sum_{i=1}^{k} div_i^{(l)}}$. The final prediction is: $O = (W + \sum_{j=1}^{k} \alpha_j \Delta W_j) x$.

### Loss & Training

qa-FLoRA itself **requires no training** — it is a purely inference-time method. Each LoRA adapter is independently trained using standard procedures (rank=64, scaling=16, cosine warmup, learning rate 1e-4, 3 epochs). Fusion weights are computed dynamically at inference time for each query, with no router training or composite data required.

## Key Experimental Results

### Main Results

Experiments are conducted on LLaMA-2-7B and LLaMA-3-8B, covering **9 composite tasks** (3 languages × 3 domains).

| Method | Paradigm | LLaMA-2 Avg. Accuracy | LLaMA-3 Avg. Accuracy |
|---|---|:---:|:---:|
| Static Fusion (Avg 0.5) | Static | 20.4% | 38.5% |
| LoRAFlow | Supervised | 30.9% | 46.1% |
| LoRAHub | Supervised | 26.4% | — |
| Centroid Similarity | Training-Free | 18.8% | 34.4% |
| **qa-FLoRA (Ours)** | **Data-Free & Training-Free** | **25.8%** | **44.2%** |

- vs. static fusion: **+5.4%** on LLaMA-2, **+5.7%** on LLaMA-3
- vs. Centroid training-free baseline: **+7.0%** on LLaMA-2, **+9.8%** on LLaMA-3
- Gap vs. supervised LoRAFlow: 5.1% on LLaMA-2, only **1.9%** on LLaMA-3

### Ablation Study

**Token granularity ablation**:

| Configuration | 9-task Avg. | Description |
|---|:---:|---|
| All-token average | 23.5% | KL divergence averaged over all query token positions |
| Last token only (Ours) | **25.8%** | Leverages the property that the last token encodes full context in autoregressive models |

**Divergence metric ablation**:

| Metric | 9-task Avg. | Description |
|---|:---:|---|
| Cosine distance | 25.8% | Operates in hidden state space |
| Euclidean distance | 24.0% | Operates in hidden state space |
| KL divergence (Ours) | **25.8%** | Operates in probability space, more directly reflects predictive behavior |

KL divergence and cosine distance achieve comparable performance, but KL divergence directly reflects predictive behavior and confidence in the probability space, providing a more principled estimate of adapter relevance.

### Key Findings

1. **Largest gains on math tasks**: qa-FLoRA outperforms the Centroid baseline by 16% (LLaMA-2) and 18% (LLaMA-3) on mathematics, because math queries contain substantial linguistic components that cause the centroid method to over-weight language LoRAs, whereas qa-FLoRA more accurately captures the contribution of task LoRAs via distributional divergence.
2. **Comparable performance on coding tasks**: Code queries contain both linguistic components and programming keywords, allowing the centroid method to also assign high weights to task LoRAs via keyword matching, resulting in similar performance between the two approaches.
3. **Per-layer analysis reveals interpretable patterns**: KL divergence is near zero in initial layers (general linguistic features), domain adapters dominate in middle layers (task reasoning), and language adapters occasionally resurge in the final layer (translation and formatting stage).

## Highlights & Insights

- **Truly zero-data and zero-training**: qa-FLoRA is the first LoRA fusion method to simultaneously require no data and no training, making it directly plug-and-play with any existing set of adapters.
- **Per-layer fusion weights enable interpretability**: Visualizing the KL divergence distribution across layers clearly reveals the contribution patterns of different domain adapters at varying network depths, providing a new perspective for understanding LLM internal processing mechanisms.
- **Acceptable latency overhead**: Fusion weight computation adds only 192ms/query/adapter in latency, can be parallelized across adapters, and entirely eliminates the training stage required by supervised methods.
- **Improves with stronger base models**: On the more capable LLaMA-3, the gap relative to supervised methods narrows from 5.1% to 1.9%, suggesting that stronger base models inherently provide better distributional signals.

## Limitations & Future Work

1. **Limited model scale**: Experiments are conducted only on 7B/8B models without validation on larger scales such as 13B or 70B.
2. **Remaining gap vs. supervised methods**: Particularly in domains requiring complex reasoning, the ceiling of training-free methods remains below that of supervised approaches.
3. **Single divergence metric**: Only KL divergence is used; strategies for dynamically selecting different relevance metrics based on query characteristics remain unexplored.
4. **Scalability with adapter count**: As the number of adapters grows, each query requires a full forward pass through all adapters, causing computation cost to scale linearly.

## Related Work & Insights

This paper systematically categorizes LoRA fusion methods into four paradigms: static fusion, supervised dynamic fusion (LoRAFlow, LoRAHub, LoRAMoE, MeteoRA, etc.), training-free fusion (Centroid Similarity, AdapterSoup), and the data-free training-free fusion proposed in this work. This clear taxonomy is highly useful for understanding the development trajectory of the field. From a methodological perspective, leveraging the model's own distributional properties to infer adapter relevance is an elegant idea that can inspire other scenarios requiring dynamic composition of multiple modules (e.g., dynamic weighting of multi-head attention, routing in mixture-of-experts models, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ — Data-free, training-free LoRA fusion represents a new paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐ — 9 composite tasks, multiple baselines, ablation studies, and interpretability analysis
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play with no additional data or training required
- Writing Quality: ⭐⭐⭐⭐ — Clear paper structure with in-depth analysis

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs](training-free_policy_violation_detection_via_activation-space_whitening_in_llms.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_imaging/counselbench_llm_mental_health_qa.md)
- [\[AAAI 2026\] Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)
- [\[AAAI 2026\] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_fusion_for_medical_visual_question_a.md)
- [\[AAAI 2026\] SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection](spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)

<!-- RELATED:END -->
