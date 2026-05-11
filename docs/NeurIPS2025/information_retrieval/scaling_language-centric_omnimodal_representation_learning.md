---
title: >-
  [Paper Note] Scaling Language-Centric Omnimodal Representation Learning
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][Multimodal Representation Learning] This paper proposes the LCO-Emb framework and demonstrates that Multimodal Large Language Models (MLLMs) implicitly establish cross-modal al…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "Multimodal Representation Learning"
  - "Contrastive Learning"
  - "MLLM Embedding"
  - "Cross-Modal Alignment"
  - "Generation-Representation Scaling Law"
date: 2026-05-08
content_hash: 4009484c4ee24902
---

# Scaling Language-Centric Omnimodal Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.11693](https://arxiv.org/abs/2510.11693)
**Code**: [GitHub](https://github.com/LCO-Embedding/LCO-Embedding)
**Area**: Information Retrieval
**Keywords**: Multimodal Representation Learning, Contrastive Learning, MLLM Embedding, Cross-Modal Alignment, Generation-Representation Scaling Law

## TL;DR

This paper proposes the LCO-Emb framework and demonstrates that Multimodal Large Language Models (MLLMs) implicitly establish cross-modal alignment during generative pretraining. Lightweight text-only contrastive fine-tuning suffices to activate full omnimodal representation capabilities. The work further identifies the Generation-Representation Scaling Law (GRSL), which establishes a positive correlation between generative capability and representation performance.

## Background & Motivation

### Limitations of Prior Work

**Background**: Cross-modal representation alignment is a central problem in multimodal AI. Traditional methods such as CLIP rely on contrastive learning over large-scale paired data to achieve vision-language alignment, yet they exhibit performance saturation on complex tasks including multilingual retrieval, visual text understanding, and interleaved multimodal encoding.

**Core Limitations**:

**Bottleneck of the CLIP Paradigm**: CLIP-style models improve by scaling model size, data, and batch size, but yield limited gains on tasks requiring deep cross-modal understanding.

**Black-Box Advantage of MLLM Embeddings**: Recent MLLM-based embedding methods outperform CLIP, yet a thorough analysis of why they are superior remains absent.

**Large Multimodal Training Data Requirements**: Leading methods such as GME require 8 million multimodal paired samples.

**Core Insight**: During generative pretraining, the language decoder of an MLLM learns to leverage multimodal signals within a shared representation space to generate unimodal outputs, thereby implicitly achieving cross-modal alignment. Contrastive learning therefore serves only as a lightweight "activation" step rather than learning alignment from scratch.

## Method

### Overall Architecture

The core mechanism of LCO-Emb is straightforward: extract the language decoder (LLM) from an MLLM, apply text-only contrastive learning with LoRA fine-tuning, and reinsert it into the original MLLM architecture. The modality encoders and projectors are frozen; only the LoRA parameters of the decoder are updated.

### Key Designs

1. **Discovery and Verification of Implicit Cross-Modal Alignment**

    - Anisotropy analysis demonstrates that the original MLLM representation space exhibits degeneracy (high isotropy). After text-only contrastive learning, not only do text embeddings become isotropic, but image, audio, and video embeddings also improve simultaneously.
    - Kernel-level similarity analysis shows that after text-only fine-tuning, the kNN overlap between image and language modalities increases substantially, and the 7B model exhibits stronger cross-modal kernel alignment than the 3B model.

2. **Language-Centric Contrastive Learning Strategy**

    - Only text-paired data (276K triplets from all-NLI) is used for InfoNCE contrastive learning.
    - LoRA rather than full fine-tuning is adopted — not primarily for parameter efficiency, but to minimize perturbation to pretrained weights and preserve the established cross-modal alignment structure.
    - Optionally, approximately 94K synthetic multimodal paired samples are added for calibration, bringing the total to 370K.

3. **Multimodal Variants and Model Fusion**

    - Separate models are fine-tuned on different datasets (all-NLI for semantic similarity; Scale-1M for multilingual and scene description tasks), and their respective strengths are combined via Model Soup weight averaging.
    - Multiple MLLM backbones are supported, including LLaVA-Next, Qwen2.5-VL, and Qwen2.5-Omni.

### Loss & Training

- Optimizer: AdamW with cosine schedule; peak learning rate $4 \times 10^{-4}$
- Batch size: 768 (text-only); scaled proportionally to ~1052 for multimodal training
- Training duration: 2 epochs; LoRA rank=64, alpha=16 (text-only) / 128 (multimodal)
- Training cost: text-only requires only ~4.7 GPU hours (3B) / ~9.3 GPU hours (7B)

## Key Experimental Results

### Main Results (MIEB-Lite, 51 Tasks)

| Model | Data | Retrieval (en) | Clustering | Zero-Shot Cls. | Linear Probe | Compositionality | Doc. Understanding | vSTS (en) | Avg. |
|-------|------|----------------|------------|----------------|--------------|------------------|--------------------|-----------|------|
| CLIP-ViT-bigG (2B) | - | 34.2 | 80.8 | 72.4 | 77.8 | 35.0 | 35.5 | 73.4 | 51.3 |
| GME (7B) | 8.0M | 37.9 | 69.6 | 55.5 | 68.7 | 52.2 | 86.1 | 81.8 | 64.5 |
| LCO-Emb-VL (7B, text-only) | 276k | 31.8 | 52.7 | 49.1 | 68.5 | 40.4 | 66.0 | 88.4 | 60.4 |
| LCO-Emb-Omni (7B, multimodal) | 370k | **36.4** | **80.0** | **68.5** | **74.1** | 50.1 | **75.4** | **86.2** | **68.8** |

### Ablation Study (Training Strategy Comparison, Qwen2.5-VL-7B)

| Training Strategy | GPU Hours | Multilingual Retrieval | vSTS (en) | Doc. Understanding | Linear Probe | Avg. |
|-------------------|-----------|------------------------|-----------|--------------------|--------------|------|
| CLIP-style CL (multimodal 800K) | ~550h | 18.24 | 73.92 | 44.89 | 38.93 | 50.02 |
| Full Fine-Tune (text-only) | ~17.3h | 44.05 | 83.15 | 58.02 | 53.34 | 66.49 |
| **LoRA (text-only)** | **~9.3h** | **56.64** | **85.05** | **67.49** | **53.91** | **71.98** |

### Key Findings

- **Text-Only Training Surpasses Multimodal CLIP Training**: LoRA text-only fine-tuning requires only 1/60 the training time of CLIP-style training, yet achieves 22 points higher average score.
- **LoRA Substantially Outperforms Full Fine-Tuning**: Preserving the pretrained alignment structure is critical; the regularization imposed by LoRA proves more beneficial than unconstrained full fine-tuning.
- **Multi-Teacher Fusion Provides Additional Gains**: Incorporating only 94K multimodal samples (25% of total data) raises the average score from 60.4 to 67.6.

## Highlights & Insights

- **Most Important Insight**: The role of contrastive learning on MLLMs is not to "learn alignment" but to "activate alignment" — MLLMs have already established cross-modal alignment implicitly during generative pretraining, and contrastive learning merely "awakens" the representation space from anisotropic degeneracy.
- **GRSL (Generation-Representation Scaling Law)**: Stronger generative capability in a model corresponds to a higher upper bound on representation performance after contrastive fine-tuning. This relationship is theoretically grounded via the PAC-Bayesian framework, where the generative loss $\mathcal{L}_g(P)$ determines the upper bound on representation performance.
- **A New Perspective on LoRA**: The primary value of LoRA lies not in parameter efficiency but in minimally perturbing pretrained knowledge and cross-modal alignment structure.
- **Extreme Data Efficiency**: Only 276K text pairs suffice to surpass GME, which uses 8 million multimodal paired samples.

## Limitations & Future Work

- Representation capability is bounded by the generative ability of the underlying MLLM — if the base model has weak generative performance on certain modalities, contrastive fine-tuning cannot compensate.
- The text-only variant still lags behind CLIP-style encoder models on clustering and zero-shot classification tasks.
- Validation of GRSL is currently conducted primarily on the Qwen model family; verification across a broader range of model families remains insufficient.
- More efficient contrastive learning objectives, such as hard negative mining, have not been explored.

## Related Work & Insights

- **vs. CLIP/SigLIP**: CLIP requires large-scale paired data to learn alignment from scratch, whereas LCO-Emb leverages alignment already established during MLLM pretraining, reducing data requirements by more than 20×.
- **vs. GME**: GME is trained on 8 million multimodal samples; LCO-Emb achieves superior performance on MIEB using only 370K samples (4.6% of GME's data).
- **vs. E5-V**: Both are MLLM-based embedding methods, but LCO-Emb outperforms E5-V by an average of 21.69 points on Sub18, attributable to a stronger backbone and the LoRA fine-tuning strategy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The discovery of implicit alignment in MLLMs and the GRSL constitute entirely novel insights with a high degree of theoretical formalization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 51 tasks, multiple backbones, multiple modalities, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear with rich figures and tables, though the theoretical section is somewhat condensed.
- Value: ⭐⭐⭐⭐⭐ Uncovers fundamental principles of MLLM representation learning with paradigm-level implications for future embedding model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[ICCV 2025\] ViLU: Learning Vision-Language Uncertainties for Failure Prediction](../../ICCV2025/information_retrieval/vilu_learning_vision-language_uncertainties_for_failure_prediction.md)
- [\[ICCV 2025\] Representation Shift: Unifying Token Compression with FlashAttention](../../ICCV2025/information_retrieval/representation_shift_unifying_token_compression_with_flashattention.md)
- [\[ICLR 2026\] Revela: Dense Retriever Learning via Language Modeling](../../ICLR2026/information_retrieval/revela_dense_retriever_learning_via_language_modeling.md)

</div>

<!-- RELATED:END -->
