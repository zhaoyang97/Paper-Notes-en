---
title: >-
  [Paper Note] InfoSEM: A Deep Generative Model with Informative Priors for Gene Regulatory Network Inference
description: >-
  [ICML2025][Image Generation][Gene Regulatory Networks] InfoSEM is proposed, an unsupervised generative framework that leverages textual gene embeddings as informative priors for gene regulatory network (GRN) inference. Without GT labels, it outperforms supervised methods by 38.5%, and improves by an additional 11.1% when using labels as an auxiliary prior, while revealing that existing supervised methods learn gene-specific biases rather than genuine regulatory mechanisms.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Gene Regulatory Networks"
  - "Unsupervised Generative Models"
  - "Text Embedding Priors"
  - "Variational Inference"
  - "Biomarker Discovery"
date: 2026-05-08
content_hash: 2114999d0fed0bc0
---

# InfoSEM: A Deep Generative Model with Informative Priors for Gene Regulatory Network Inference

**Conference**: ICML2025  
**arXiv**: [2503.04483](https://arxiv.org/abs/2503.04483)  
**Area**: Image Generation  
**Keywords**: Gene Regulatory Networks, Unsupervised Generative Models, Text Embedding Priors, Variational Inference, Biomarker Discovery

## TL;DR
InfoSEM is proposed, an unsupervised generative framework that leverages textual gene embeddings as informative priors for gene regulatory network (GRN) inference. Without GT labels, it outperforms supervised methods by 38.5%, and improves by an additional 11.1% when using labels as an auxiliary prior, while revealing that existing supervised methods learn gene-specific biases rather than genuine regulatory mechanisms.

## Background & Motivation

### Importance of GRN Inference

Gene regulatory networks reveal how transcription factors regulate target genes, which is crucial for drug design, biomarker discovery, and disease understanding.

### Limitations of Prior Work

**Limitations of Prior Work**: Supervised models perform well on standard benchmarks (AUPRC ~ 0.85), but they may learn gene-specific biases in GT labels (such as class imbalance of certain genes) rather than genuine regulatory patterns.

### Key Challenge

**Key Challenge**: Avoiding GT labels prevents bias, but traditional unsupervised methods perform far worse. InfoSEM bridges this gap by using text embeddings as informative priors.

## Method

### Variational Bayesian Framework
A generative model is trained using variational inference to infer regulatory relationships from scRNA-seq data.

### Textual Gene Embedding Prior
Pre-trained text embeddings are utilized to encode prior biological knowledge of genes, which are not GT labels but contain rich functional and structural information.

### GT Labels as Auxiliary Priors (Optional)
When labels are available, they are used as priors rather than for direct supervision, thereby avoiding the bias trap of supervised learning.

### Biology-Driven Benchmark Framework
An evaluation protocol for "interactions between unseen genes" is proposed, which is closer to real-world applications such as biomarker discovery.

## Key Experimental Results

### GRN Inference (Average over 4 Datasets)

### Main Results

| Method | Mode | Relative AUPRC Improvement |
|------|------|-------------|
| Supervised SOTA | Labeled | Baseline |
| **InfoSEM (Text Prior)** | **Unlabeled** | **+38.5%** |
| **InfoSEM (+ Label Prior)** | **Labeled** | **+49.6% (+11.1%)** |

### Unseen Gene Evaluation

### Ablation Study

| Method | Standard Evaluation | Unseen Gene Evaluation |
|------|---------|-----------|
| Supervised Methods | High | **Significant Drop** |
| **InfoSEM** | Higher | **Stable** |

### Bias Analysis

| Phenomenon | Explanation |
|------|------|
| Supervised methods perform well on standard evaluation | Learned gene-specific class imbalance |
| Supervised methods perform poorly on unseen genes | Bias is non-transferable $ \rightarrow $ Generalization failure |
| InfoSEM performs well on both | Learned genuine regulatory patterns |

### Key Findings
1. Unsupervised + text prior outperforms supervised methods by 38.5%—a paradigm-level breakthrough.
2. GT labels used as priors rather than for supervision $ \rightarrow $ avoids bias.
3. Unseen gene evaluation reveals the illusory success of supervised methods.
4. Text embeddings contain rich information on gene functions.
5. Consistently effective across 4 different datasets.

## Highlights & Insights

1. "Unsupervised outperforming supervised" challenges the consensus in the GRN field.
2. Revealing that supervised methods learn bias rather than regulation serves as a warning to the entire field.
3. Text embeddings as priors represent a low-cost yet high-value innovation.
4. "Unseen gene evaluation" is closer to real-world application scenarios.
5. The concept of using GT labels as priors rather than supervision offers broad inspiration.

## Limitations & Future Work

1. The quality of text embeddings is limited by pre-training data.
2. Verified only on scRNA-seq data.
3. Applicability to non-model organisms has not been tested.
4. The approximations in variational inference may affect inference accuracy.
5. Comparison with the latest multi-omics integration methods is missing.

## Related Work & Insights

- Difference from GENIE3/DeepSEM, etc.: InfoSEM uses priors rather than supervision.
- Relationship with LLM-based biological methods: Leverages text embeddings but in a different manner.
- Insight: The idea of "priors rather than supervision" can be extended to other bioinformatics tasks.

## Rating
- Novelty: 5.0/5 — Unsupervised outperforming supervised + bias revelation
- Experimental Thoroughness: 4.5/5 — 4 datasets + bias analysis
- Writing Quality: 4.5/5
- Value: 5.0/5 — Paradigm-level impact on the GRN field

## Supplementary

### Paradigm Insights of "Priors Rather than Supervision"
GT labels as supervision introduce bias, whereas serving as a prior provides guidance without coercion. This idea can be extended to other bioinformatics tasks with imperfect annotations.

### Why Text Embeddings are Effective
Pre-trained text models (e.g., PubMedBERT) have already encoded gene function knowledge from scientific literature, which is significantly superior as a prior compared to having no prior.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Generation of Maximal Snake Polyominoes Using a Deep Neural Network](../../CVPR2025/image_generation/generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[ICML 2025\] Learning Single Index Models with Diffusion Priors](learning_single_index_models_with_diffusion_priors.md)
- [\[CVPR 2025\] Learning Visual Generative Priors without Text](../../CVPR2025/image_generation/learning_visual_generative_priors_without_text.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)

</div>

<!-- RELATED:END -->
