---
title: >-
  [Paper Note] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention
description: >-
  [Medical Imaging] FaNe proposes a semantics-enhanced medical vision-language pre-training framework that addresses the false-negative problem and insufficient coarse-grained alignment in medical VLP through semantics-awa…
tags:
  - "Medical Imaging"
date: 2026-05-08
content_hash: ac73d4ec89b4728e
---

# FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention

## Paper Information

- **Conference**: AAAI 2026
- **arXiv**: [2511.12215](https://arxiv.org/abs/2511.12215)
- **Code**: [https://github.com/Aventador8/FaNe](https://github.com/Aventador8/FaNe)
- **Area**: Medical Imaging
- **Keywords**: Vision-language pre-training, false-negative reduction, sparse attention, fine-grained alignment, contrastive learning, medical imaging

## TL;DR

FaNe proposes a semantics-enhanced medical vision-language pre-training framework that addresses the false-negative problem and insufficient coarse-grained alignment in medical VLP through semantics-aware positive mining, text-conditioned sparse attention pooling, and hard-negative-aware contrastive loss.

## Background & Motivation

Medical vision-language pre-training (VLP) advances medical image understanding by leveraging paired image-report data. Existing CLIP-style methods suffer from two core issues:

**False-Negative Problem**: In standard VLP training, each image forms a positive pair only with its corresponding report; all other samples are treated as negatives. In clinical practice, however, different patients may present with identical diseases or lesions, resulting in highly similar or even identical report descriptions. Treating such semantically similar but report-distinct samples as negatives introduces erroneous alignment signals.

**Insufficient Fine-Grained Alignment**: CLIP performs only global image-text alignment, failing to capture detailed visual features. Each sentence in a medical report typically corresponds to findings in a specific image region, requiring sentence-level local alignment. Existing methods (e.g., FLAIR) attempt text-conditioned attention pooling, but cross-attention alone lacks the ability to enforce precise spatial focus.

## Method

### Overall Architecture

FaNe comprises four core components: Semantic Class Division, Multi-Positive Global Alignment, Text-Conditioned Fine-Grained Alignment, and Hard-Negative Intra-Modal Contrast.

### Key Designs

#### 1. Semantic Class Division

A pretrained knowledge extractor, BioClinicalBERT, encodes reports into global and local representations. To eliminate semantic redundancy in clinical narratives and stabilize cross-batch similarity computation, **semantics-enhanced adaptive normalization** is introduced:

- Compute batch prototype $p_b$ (mean of all text global representations in the batch)
- Compute baseline semantic similarity $\hat{o}_t^*$ (average cosine similarity of each report to the prototype)
- Apply EMA smoothing ($\alpha=0.05$) to prevent cross-batch fluctuations
- Perform center-shift normalization: $\widetilde{S} = \frac{S - o_t^*}{1 - o_t^* + \epsilon}$
- Construct a similarity class matrix $H$ via threshold $\kappa$, partitioning samples into positives and negatives

#### 2. Multi-Positive Global Alignment

The standard CLIP InfoNCE loss supports only a single positive pair and cannot handle multi-positive scenarios. FaNe adopts the SigLIP sigmoid contrastive loss, which natively supports in-batch multi-positive alignment:

$$\mathcal{L}_{mp} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\log\frac{1}{1+e^{h_{ij}(-\langle v_i^g, t_j^g\rangle/\tau_1 + b)}}$$

where $h_{ij}$ is the entry in label matrix $H$ and $b$ is a learnable bias.

#### 3. Text-Conditioned Sparse Attention Pooling

One of the core innovations. The design pipeline:

- **Image/text representation extraction**: Independent encoders extract local and global features $v^l \in \mathbb{R}^{I \times D}$, $t^l \in \mathbb{R}^{P \times L \times D}$
- **Learnable sparse attention mask**: An MLP with sigmoid activation generates $M \in \mathbb{R}^{L \times I}$, subject to an L1 sparsity constraint $\mathcal{L}_{spa}$
- **Cross-attention fine-grained alignment**: Sentence-level text embeddings serve as queries to aggregate local image patch embeddings, multiplied by sparse mask $M$, producing text-conditioned visual representation $v^{tc,u}$
- Negatives are sampled only from different sentences within the same report, avoiding cross-report false negatives

The fine-grained alignment loss $\mathcal{L}_{tc}$ is the mean of InfoNCE losses in both text→image and image→text directions.

#### 4. Hard-Negative Intra-Modal Contrastive Loss

An adaptive re-weighting mechanism emphasizes semantically similar negatives to enhance intra-modal discriminability:

$$\alpha_{ij} = \frac{y_{ij} \cdot v_i^g \cdot (v_j^g)^T / \tau_3}{\sum_{k \neq i} y_{ik} \cdot v_i^g \cdot (v_k^g)^T / \tau_3}$$

Weights $\alpha_{ij}$ and $\beta_{ij}$ assign larger weights to harder negatives with higher semantic similarity, compelling the model to learn fine-grained semantic discrimination.

### Loss & Training

The total loss is a weighted sum of four terms:

$$\mathcal{L} = \mathcal{L}_{mp} + \lambda_1 \mathcal{L}_{tc} + \lambda_2 \mathcal{L}_{hn} + \lambda_3 \mathcal{L}_{spa}$$

where $\lambda_1 = \lambda_2 = \lambda_3 = 1$ in experiments.

## Key Experimental Results

### Pre-training Setup

- Dataset: MIMIC-CXR v2 (182,475 high-quality image-report pairs after filtering)
- Text encoder: BioClinicalBERT; Image encoder: ResNet50 / ViT-B/16
- Training: 2× RTX 4090, batch size 98, 50 epochs
- Temperature parameters: $\tau_1=0.1$, $\tau_2=0.07$, $\tau_3=0.07$

### Main Results

**Semantic Segmentation (Dice) + Object Detection (mAP)**:

| Method | RSNA 1%/10%/100% | SIIM 1%/10%/100% | RSNA Det 1%/10%/100% |
|--------|------------------|------------------|---------------------|
| MLIP | 67.7/68.8/73.5 | 51.6/60.8/68.1 | 17.2/19.1/25.8 |
| IMITATE | 70.5/71.4/73.8 | 53.9/61.7/64.5 | 15.3/19.7/26.4 |
| **FaNe** | **69.5/72.4/74.1** | **54.1/62.3/68.8** | **16.4/20.6/27.2** |

**Image Classification (AUC/ACC)**:

| Method | CheXpert 1%/10%/100% | RSNA 1%/10%/100% | COVIDx 1%/10%/100% |
|--------|---------------------|------------------|-------------------|
| FaNe (ResNet-50) | 88.2/89.1/89.9 | 88.9/89.8/92.6 | 78.2/89.1/94.0 |
| FaNe (ViT-B/16) | **89.7/90.4/90.8** | **89.3/90.2/93.1** | **79.5/90.7/95.5** |

### Ablation Study

1. **Sparse attention mask**: Adding a learnable mask with sparsity regularization yields substantial gains (RSNA Dice 100%: 71.2→72.6→74.1)
2. **Semantic adaptive normalization**: Enabling it improves RSNA Dice 1% from 67.1 to 69.5
3. **Contribution of each loss term**: Adding $\mathcal{L}_{hn}$ and $\mathcal{L}_{tc}+\mathcal{L}_{spa}$ on top of $\mathcal{L}_{mp}$ yields incremental gains
4. **Threshold $\kappa$ sensitivity**: Performance peaks at $\kappa=0.95$

### Key Findings

- ViT-B/16 significantly outperforms ResNet-50 on classification tasks, indicating that Transformer architectures are better suited for medical VLP
- Sparse attention visualizations demonstrate accurate localization of image regions corresponding to textual descriptions (e.g., thoracic scoliosis, cardiac position)
- FaNe's advantage is particularly pronounced under low data ratios (1%), reflecting the data efficiency of fine-grained pre-training

## Highlights & Insights

1. **Systematic resolution of the false-negative problem**: A complete false-negative elimination pipeline is formed from semantic similarity computation and adaptive normalization to multi-positive alignment
2. **Elegant sparse attention design**: In medical reports, each sentence typically corresponds to only a local image region; the sparsity constraint aligns perfectly with domain characteristics
3. **Intra-modal contrast as an important complement**: Beyond cross-modal alignment, intra-modal discriminability is explicitly enhanced, aiding the distinction of similar yet distinct clinical findings
4. **Intra-report negative sampling strategy**: Fine-grained alignment negatives are drawn only from other sentences within the same report, effectively avoiding cross-report false negatives

## Limitations & Future Work

- Pre-training and evaluation are conducted solely on the chest X-ray dataset MIMIC-CXR; generalization to other modalities (CT, MRI) remains unverified
- Threshold $\kappa$ requires predefinition; although ablation experiments show 0.95 to be optimal, the best value may vary across datasets
- The sparsity degree of the attention mask is controlled by $\lambda_3$, which may require task-specific tuning
- FaNe with ResNet-50 does not surpass all baselines (e.g., IMITATE achieves 70.5 vs. FaNe's 69.5 on RSNA segmentation at 1%), though the ViT variant achieves the overall best performance

## Related Work & Insights

- **Vision-language pre-training**: CLIP, SigLIP, FLAIR (text-conditioned attention pooling), MGCA (multi-granularity alignment)
- **False-negative problem**: MedCLIP (semantic matching loss), MLIP (knowledge-guided class-level contrast), SAT (semantic triplet division)
- **Medical VLP methods**: GLoRIA, PRIOR, M-FLAG, MedKLIP, IMITATE

## Rating

⭐⭐⭐⭐ (4/5)

- Problem definition is clear; the four components are tightly coupled, constituting a solid contribution to medical VLP
- Experiments comprehensively cover three downstream task types—classification, segmentation, and detection—across five benchmark datasets
- Sparse attention visualizations enhance interpretability
- Deduction: evaluation is limited to chest X-rays; methodological generality awaits further validation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[NeurIPS 2025\] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](../../NeurIPS2025/medical_imaging/raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)

</div>

<!-- RELATED:END -->
