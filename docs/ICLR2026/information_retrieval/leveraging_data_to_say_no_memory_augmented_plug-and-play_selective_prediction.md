---
title: >-
  [Paper Note] Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction
description: >-
  [ICLR 2026][Selective Prediction] This paper proposes MA-PaPSP, a training-free plug-and-play selective prediction framework for arbitrary VLMs. It constructs proxy embeddings via k-NN weighted averaging over an external retrieval dataset (reducing representational variance) and applies contrastive normalization scoring (improving calibration). MA-PaPSP consistently outperforms PaPSP and LLM-as-judge baselines on selective prediction across image captioning, image-text matching, and classification tasks.
tags:
  - ICLR 2026
  - Selective Prediction
  - VLM Reliability
  - Retrieval Augmentation
  - Contrastive Scoring
  - CLIP
date: 2026-05-08
content_hash: 5966b808bd824e2a
---

# Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction

**Conference**: ICLR 2026
**arXiv**: [2601.22570](https://arxiv.org/abs/2601.22570)
**Code**: [https://github.com/kingston-aditya/MA-PaPSP](https://github.com/kingston-aditya/MA-PaPSP)
**Area**: Information Retrieval
**Keywords**: Selective Prediction, VLM Reliability, Retrieval Augmentation, Contrastive Scoring, CLIP

## TL;DR
This paper proposes MA-PaPSP, a training-free plug-and-play selective prediction framework for arbitrary VLMs. It constructs proxy embeddings via k-NN weighted averaging over an external retrieval dataset (reducing representational variance) and applies contrastive normalization scoring (improving calibration). MA-PaPSP consistently outperforms PaPSP and LLM-as-judge baselines on selective prediction across image captioning, image-text matching, and classification tasks.

## Background & Motivation

**State of the Field**: VLMs (e.g., BLIP, InternVL, Qwen-VL) are widely deployed in image-text matching, image captioning, and classification, yet prediction errors are unavoidable—incorrect modality alignment, tail-distribution samples, and image/language ambiguity all contribute to failures. Selective Prediction (SP) addresses this by endowing models with the ability to abstain from answering.

**Limitations of Prior Work**:
- **Closed-set constraint**: Existing SP methods primarily target closed-set tasks (e.g., classification with finite label sets) and cannot handle open-set tasks such as image captioning with unbounded label spaces.
- **Training dependency**: Most methods require fine-tuning the base model or training an additional selector, making them inapplicable to black-box or large-scale models.
- **Unreliable CLIP scoring**: Using raw CLIP cosine similarity as a confidence measure suffers from two issues: (1) representational instability—embeddings of the same semantic concept exhibit high variance across images/texts; and (2) poor calibration—similarity score distributions vary across different regions of the embedding space.

**Root Cause**: An ideal SP solution should simultaneously be training-free, lightweight, open-set compatible, and pluggable into arbitrary VLMs—requirements that no existing approach satisfies jointly.

**Paper Goals**: Design a training-free, plug-and-play selective prediction module (PaPSP) capable of providing confidence estimation across task levels (classification → image-text matching → image captioning) for VLMs ranging from CLIP to large-scale LVLMs.

**Starting Point**: Augment CLIP-style scoring models with an external retrieval dataset, using retrieved neighbors for embedding averaging (variance reduction) and contrastive normalization (calibration improvement).

**Core Idea**: Use k-NN weighted averages of neighbor embeddings from an external retrieval corpus as more stable proxy embeddings, and replace raw cosine similarity with contrastive normalization over hard negatives to enable reliable selective prediction.

## Method

### Overall Architecture
MA-PaPSP operates at three levels: (1) a base prediction VLM (P-VLM) generates predictions; (2) an external SP-VLM (e.g., SigLIP) computes image and text embeddings; (3) a retrieval dataset $R$ supplies neighbor information for constructing proxy embeddings and computing contrastive scores. The final confidence score determines whether to abstain when it falls below a threshold.

### Key Designs

1. **Proxy Embeddings**:

    - **Function**: Replace the query embedding with a weighted average of its $K$ nearest neighbor embeddings, yielding a more stable proxy representation.
    - **Mechanism**: Given a query $q$ (image or text), retrieve $K$ nearest neighbors $N_K(q)$ from the retrieval set $R$ and compute their similarity-weighted average: $\tilde{\varphi}(q) = \sum_i \frac{\gamma(q, z_i)}{\sum_j \gamma(q, z_j)} \cdot \varphi(y_i)$. Four variants are supported (i2tr/i2ir/t2tr/t2ir), covering both cross-modal and unimodal retrieval.
    - **Design Motivation**: Embeddings of the same semantic concept in CLIP's embedding space exhibit high variance (e.g., large similarity variation across images of the same category). K-NN averaging leverages a statistical effect—noise across multiple neighbors cancels out—producing representations closer to the true semantic centroid.

2. **Contrastive Scores**:

    - **Function**: Normalize similarity scores using hard negatives to yield calibrated confidence values in $[0, 1]$.
    - **Mechanism**: Generate a set of hard negatives $E(f(x))$—for captioning, semantically distinct but syntactically similar alternatives are produced by replacing nouns; for classification, other class labels serve as negatives. The contrastive score is computed as $s_{tc} = \frac{\exp(s(x, f(x))/\tau)}{\sum_k \exp(s(x, y_k)/\tau)}$.
    - **Design Motivation**: Raw cosine similarity distributions vary substantially across different regions of the embedding space, analogous to poorly calibrated pre-softmax logits. Normalizing by hard negatives, akin to a softmax operation, yields more uniformly distributed scores across embedding space regions.

3. **Hard Negative Generation**:

    - **Function**: Automatically generate semantically contrastive alternative captions for image captioning tasks.
    - **Mechanism**: Nouns in a caption are replaced using either rule-based (RB) methods or a small language model (SLM), producing syntactically similar but semantically distinct sentences.
    - **Design Motivation**: Contrastive scoring requires reference points to anchor confidence; classification and image-text matching tasks have natural candidate sets, whereas image captioning requires explicit construction of such references.

### Loss & Training
The method is entirely training-free. Key hyperparameters include the number of neighbors $K$, temperature $\tau$, and the choice of retrieval set $R$. The SP-VLM uses pretrained SigLIP without any fine-tuning.

## Key Experimental Results

### Main Results — AURC (lower is better)

| Method | MS-COCO (CiderN) | Flickr-30K (CiderN) | Flowers | Pets | UCF101 | SugarCrepe |
|--------|-------------------|---------------------|---------|------|--------|------------|
| VQAScore | 0.146 | 0.241 | 0.211 | 0.207 | 0.217 | 0.146 |
| SeeTRUE | 0.158 | 0.251 | 0.214 | 0.213 | 0.171 | 0.153 |
| PaPSP (SigLIP-S) | 0.142 | 0.237 | 0.093 | 0.211 | 0.154 | 0.162 |
| **MA-PaPSP (SigLIP-S)** | **0.121** | **0.235** | **0.077** | **0.171** | **0.116** | **0.079** |
| PaPSP (SigLIP-L) | 0.136 | 0.229 | 0.074 | 0.169 | 0.113 | 0.078 |
| **MA-PaPSP (SigLIP-L)** | **0.109** | **0.219** | **0.063** | **0.114** | **0.088** | **0.062** |
| Gain (L) | 19.85% | 4.36% | 14.86% | 32.52% | 22.12% | 20.51% |

### Cross-P-VLM Validation (Image Captioning AURC↓)

| P-VLM | PaPSP (COCO) | MA-PaPSP (COCO) | Gain |
|-------|-------------|-----------------|------|
| BLIP-1 (0.1B) | 0.138 | 0.114 | 17.4% |
| BLIP-2 (2.7B) | 0.136 | 0.109 | 19.9% |
| InternVL-3.5 (4B) | 0.106 | 0.068 | 35.8% |
| Qwen-2.5-VL (7B) | 0.102 | 0.066 | 35.3% |

### Ablation Study — Effect of Retrieval Set Type (AURC↓)

| Retrieval Set | MS-COCO (CiderN) | Flowers | SugarCrepe |
|---------------|-------------------|---------|------------|
| Random | 0.126 | 0.062 | 0.064 |
| In-Domain | 0.126 | 0.062 | 0.066 |
| Out-of-Domain | 0.109 | 0.063 | 0.062 |
| Mixed | **0.107** | **0.062** | **0.068** |

### Key Findings
- MA-PaPSP with a small SP-VLM (SigLIP-B/16, 16M) surpasses PaPSP with a large SP-VLM (SigLIP-SO-400M, 1B), demonstrating that retrieval augmentation is more effective than simply scaling the scoring model.
- MA-PaPSP consistently outperforms LLM-reasoning-based methods such as VQAScore and SeeTRUE at substantially lower computational cost.
- Gains are largest on classification tasks (Pets: 32.5%, UCF101: 22.1%) and also significant on captioning (COCO: 19.9%).
- A general out-of-domain retrieval set (CC12M+SBU) matches or exceeds in-domain retrieval on captioning and image-text matching tasks.
- As P-VLM scale increases (0.1B to 7B), MA-PaPSP's improvement margin grows (17.4%→35.3%), indicating greater effectiveness with stronger base models.

## Highlights & Insights
- **Valuable problem formulation**: The first work to systematically define plug-and-play selective prediction across task levels (classification → image-text matching → captioning).
- **Elegant method design**: Proxy embeddings and contrastive scoring each address a distinct root cause—representational instability and poor calibration—with clear design motivation.
- **Strong generality**: Training-free and compatible with arbitrary VLMs (from CLIP to InternVL-3.5/Qwen-2.5-VL) across both open-set and closed-set tasks.
- **Interesting finding**: Out-of-domain general-purpose retrieval sets can substitute for in-domain data, lowering the barrier for practical deployment.

## Limitations & Future Work
- Storage and retrieval of external datasets (e.g., CC12M with 15M entries) introduces non-trivial storage and latency requirements.
- Hard negative generation for image captioning relies on rule-based methods or SLMs, which may produce inconsistent quality.
- The contrastive scoring temperature $\tau$ requires tuning and may differ across tasks.
- Validation is limited to English; representational properties of the embedding space may differ in multilingual settings.
- Evaluation of open-set tasks (e.g., captioning) depends on a CIDEr-N threshold $\beta$, and conclusions may be sensitive to this choice.

## Related Work & Insights
- The approach shares the spirit of RAG (Retrieval-Augmented Generation) but pursues a different objective: RAG enhances generation quality, while MA-PaPSP enhances confidence estimation.
- MA-PaPSP provides a "safety valve" for VLM deployment: in high-stakes scenarios (e.g., medical imaging), it enables abstention under uncertainty.
- The proxy embedding idea is extensible to other settings, such as prototype augmentation for few-shot classification and query augmentation for cross-modal retrieval.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically address plug-and-play selective prediction for open-set VLMs; the combination of proxy embeddings and contrastive scoring is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across task levels, model scales, and retrieval set types.
- Writing Quality: ⭐⭐⭐⭐ High-level problem abstraction; the VLM task taxonomy is well-motivated and clearly structured.
- Value: ⭐⭐⭐⭐ An important tool for VLM reliability with direct practical value for real-world deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis](multimodal_dataset_distillation_made_simple_by_prototype-guided_data_synthesis.md)
- [\[ICLR 2026\] AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](../../AAAI2026/information_retrieval/prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)
- [\[AAAI 2026\] PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation](../../AAAI2026/information_retrieval/precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)

<!-- RELATED:END -->
