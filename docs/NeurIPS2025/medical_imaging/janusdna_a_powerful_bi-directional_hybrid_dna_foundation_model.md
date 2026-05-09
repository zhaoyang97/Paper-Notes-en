---
title: >-
  [Paper Note] JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model
description: >-
  [NeurIPS 2025][Medical Imaging][DNA foundation model] JanusDNA is proposed as the first bidirectional DNA foundation model, combining a Mamba-Attention-MoE hybrid architecture with the Janus Modeling pretraining paradigm to achieve bidirectional understanding at the training efficiency of autoregressive methods, attaining state-of-the-art performance across multiple genomic benchmarks.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - DNA foundation model
  - bidirectional modeling
  - Mamba-Attention
  - Mixture-of-Experts
  - genomics
date: 2026-05-08
content_hash: 3044f7e3a92a7056
---

# JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model

**Conference**: NeurIPS 2025
**arXiv**: [2505.17257](https://arxiv.org/abs/2505.17257)
**Code**: [GitHub](https://github.com/Qihao-Duan/JanusDNA)
**Area**: Medical Imaging
**Keywords**: DNA foundation model, bidirectional modeling, Mamba-Attention, Mixture-of-Experts, genomics

## TL;DR

JanusDNA is proposed as the first bidirectional DNA foundation model, combining a Mamba-Attention-MoE hybrid architecture with the Janus Modeling pretraining paradigm to achieve bidirectional understanding at the training efficiency of autoregressive methods, attaining state-of-the-art performance across multiple genomic benchmarks.

## Background & Motivation

**Background**: Large language models are being applied to DNA sequence modeling, yet direct transfer faces unique challenges—handling long-range dependencies in ultra-long sequences (>10k base pairs) while requiring bidirectional understanding.

**Limitations of Prior Work**:
   - **Sequence length vs. resolution trade-off**: Attention mechanisms struggle with long sequences; k-mer tokenization expands the context window but sacrifices resolution (losing SNP information).
   - **Unidirectional understanding**: Decoder-based models (HyenaDNA, Evo) support only unidirectional context, whereas many regulatory elements (e.g., bidirectional promoters) require bidirectional modeling.
   - **Training inefficiency**: MLM (BERT-style) involves only ~15% of tokens in loss computation, which is extremely inefficient for long-sequence training.

**Key Challenge**: An inherent trade-off exists between bidirectional understanding capability (MLM) and training efficiency (autoregressive).

**Goal**: To construct an efficient bidirectional DNA foundation model that simultaneously handles long sequences and maintains high training efficiency.

**Key Insight**: Design a novel pretraining paradigm (Janus Modeling) in which all tokens contribute to loss computation (as in autoregressive training) while preserving bidirectional understanding (as in MLM).

**Core Idea**: Achieve full-token-loss bidirectional pretraining through independent bidirectional encoding combined with a carefully designed attention mask fusion mechanism.

## Method

### Overall Architecture

JanusDNA comprises three core components: (1) **Janus Modeling**—an efficient bidirectional pretraining method; (2) a **Mamba-Attention-MoE hybrid architecture**; and (3) a **reverse complement (RC) processing strategy**. The forward and reverse sequences are independently encoded through separate Mamba+MoE stacks and subsequently fused via FlexAttention, enabling bidirectional prediction without information leakage.

### Key Designs

1. **Janus Modeling (Efficient Bidirectional Training)**:

    - **Function**: Enables every token to be predicted based on full bidirectional context, with all tokens contributing to the loss.
    - **Design Motivation**: MLM computes loss over only 15% of tokens, resulting in low efficiency; autoregressive methods are efficient but unidirectional.
    - **Mechanism**:
        - Forward encoding: $H_t^F = \text{ForwardEncoder}(x_1, ..., x_t)$
        - Backward encoding: $H_t^B = \text{BackwardEncoder}(x_T, ..., x_t)$
        - Bidirectional fusion: a carefully designed attention mask $\mathcal{M}_{ij}$ ensures that prediction of $x_t$ uses only $H_k^F\ (k<t)$ and $H_j^B\ (j>t)$
    - Training objective: $\mathcal{L}_{bidirectional} = -\sum_{t=1}^{T} \log P(x_t | x_1,...,x_{t-1}, x_{t+1},...,x_T)$
    - **Novelty**: Approximately 2× faster than MLM (sparse masking) with significantly higher learning efficiency.

2. **Hybrid Architecture (Mamba-Attention-MoE)**:

    - **Function**: Combines the long-sequence efficiency of SSMs, the global comprehension of attention, and the sparse capacity expansion of MoE.
    - **Design Motivation**: Pure attention cannot scale to million-level base pairs; pure SSMs lack global fusion capability.
    - **Mechanism**:
        - Mamba layers efficiently encode local context.
        - MoE layers replace FFN layers proportionally to expand model capacity via sparse activation.
        - FlexAttention layers realize bidirectional fusion.
    - MoE auxiliary loss: $\mathcal{L}_{total} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i$ ensures balanced expert utilization.
    - **Novelty**: Capable of processing 1 million base pairs on a single 80 GB GPU.

3. **Reverse Complement (RC) Processing**:

    - **Function**: Processes the forward DNA strand and its reverse complement strand in parallel.
    - **Design Motivation**: The double-stranded DNA structure carries equivalent information; non-palindromic motifs must be recognized in both orientations simultaneously.
    - **Mechanism**: Both the forward strand and the RC strand are fed independently into the same model; output representations are pooled and merged.

4. **Attention Mask Design (FlexAttention Mask)**:

    - **Function**: Controls information flow in attention over the $2T$-length input sequence.
    - **Design Motivation**: Information leakage at position $t$ during prediction must be strictly prevented.
    - **Mechanism**: Four rules govern intra-forward-segment, intra-backward-segment, and cross-directional (forward-to-backward) attention.

### Loss & Training

- Primary loss: bidirectional prediction loss $\mathcal{L}_{bidirectional}$ (all tokens participate).
- MoE auxiliary loss: ensures balanced expert load.
- Pretraining data: human reference genome HG38, tokenized at single-nucleotide resolution.
- Context length: 131,072 (extensible to 1M).

## Key Experimental Results

### Main Results

**Genomic Benchmark (8 tasks, Top-1 Accuracy, 5-fold CV)**:

| Model | Active Params | Mouse Enhancers | Coding vs Inter. | Human Regulatory | Human NonTATA |
|------|---------|-----------------|-------------------|------------------|---------------|
| HyenaDNA | 436k | 0.780 | 0.904 | 0.869 | 0.944 |
| Caduceus-PS | 470k | **0.793** | 0.910 | 0.873 | 0.945 |
| **JanusDNA** | 426k | 0.770 | 0.912 | **0.877** | **0.957** |

**Nucleotide Transformer Benchmark (18 tasks) — Selected Histone Marks**:

| Model | Active Params | H3 | H3k14ac | H3k36me3 | H3k4me3 |
|------|---------|-----|---------|----------|---------|
| Enformer | 252M | 0.719 | 0.288 | 0.344 | 0.158 |
| NT-v2 | 500M | 0.784 | 0.551 | 0.625 | 0.410 |
| Caduceus-PH | 1.9M | 0.815 | 0.631 | 0.601 | 0.544 |
| **JanusDNA** | **2M** | **0.835** | **0.729** | **0.702** | **0.688** |

**DNALongBench eQTL Task (AUROC, sequence length 450k)**:

| Model | Artery Tibial | Muscle Skeletal | Nerve Tibial | Whole Blood |
|------|--------------|-----------------|--------------|-------------|
| Enformer (252M) | 0.741 | 0.621 | 0.683 | 0.689 |
| Caduceus-PH (7.7M) | 0.690 | 0.789 | 0.842 | 0.769 |
| **JanusDNA (7.7M)** | **0.852** | **0.864** | **0.914** | **0.821** |

### Ablation Study

**Janus Modeling vs. Masked Modeling Efficiency Comparison** (10k training steps, last-token prediction accuracy):
- Janus Modeling substantially outperforms Masked Modeling across all hidden dimensions (32/64/128).
- Janus training speed: ~27 minutes per 1,000 steps, approximately 2× faster than Masked Modeling.
- At hidden dimension 128, Janus achieves at 5k steps the accuracy that Masked Modeling requires 10k steps to reach.

### Key Findings

- JanusDNA achieves state-of-the-art performance on 12 out of 18 NT benchmark tasks, surpassing models with 250× more parameters.
- JanusDNA substantially outperforms the specialist model Enformer on long-range eQTL tasks.
- Janus Modeling improves training efficiency by approximately 2× over MLM.
- Processing of 1 million base pairs on a single 80 GB GPU demonstrates strong practical utility.
- MoE layers effectively expand model capacity without significantly increasing computational cost.

## Highlights & Insights

- **The apt "Janus" metaphor**: Independent encoding in two directions followed by fusion perfectly mirrors the biological nature of double-stranded DNA.
- **Breaking the linear relationship between parameter count and performance**: A 2M-parameter model surpasses models with 500M+ parameters.
- **Training paradigm innovation**: The approach simultaneously resolves two fundamental problems—the low efficiency of MLM and the unidirectionality of autoregressive methods.
- **Elegant FlexAttention mask design**: Achieves full-token bidirectional prediction without information leakage over inputs of length $2T$.

## Limitations & Future Work

- Pretraining is conducted exclusively on the human reference genome, lacking cross-species and genomic variation data.
- Epigenetic information (chromatin accessibility, histone modifications, and other multimodal data) has not been integrated.
- Computational resource requirements for long sequences remain substantial.
- Future work may explore modeling of functional features such as CTCF-mediated chromatin loops.

## Related Work & Insights

- The approach differs from Caduceus's bidirectional SSM strategy: Caduceus achieves bidirectionality via bidirectional Mamba, whereas JanusDNA employs Janus Modeling combined with fusion attention.
- The trend of Mamba + Attention hybrid architectures is emerging concurrently in NLP (e.g., Jamba) and genomics.
- Sparse MoE capacity expansion is particularly valuable for ultra-long-sequence models.
- Single-nucleotide resolution tokenization is essential for SNP-related research.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The Janus Modeling training paradigm is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 35 tasks, three major benchmarks, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-crafted figures and tables.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for DNA foundation models with broad practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NeurIPT: Foundation Model for Neural Interfaces](neuript_foundation_model_for_neural_interfaces.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[NeurIPS 2025\] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)

</div>

<!-- RELATED:END -->
