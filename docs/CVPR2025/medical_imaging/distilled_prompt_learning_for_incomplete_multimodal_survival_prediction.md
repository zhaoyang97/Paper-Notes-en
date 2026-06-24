---
title: >-
  [Paper Note] Distilled Prompt Learning for Incomplete Multimodal Survival Prediction
description: >-
  [CVPR 2025][Medical Imaging][Survival Prediction] This paper proposes DisPro (Distilled Prompt Learning), a two-stage prompt learning framework—UniPro to distill the knowledge distribution of each modality, and MultiPro to leverage LLMs to infer missing modalities from available ones. By simultaneously compensating for both modality-specific and modality-common information of the missing modalities, DisPro achieves SOTA performance on five TCGA survival prediction datasets.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Survival Prediction"
  - "Missing Modality"
  - "Prompt Learning"
  - "LLM Robustness"
  - "Pathology WSI"
  - "Genomics"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: e478933ecfff3ce4
---

# Distilled Prompt Learning for Incomplete Multimodal Survival Prediction

**Conference**: CVPR 2025  
**arXiv**: [2503.01653](https://arxiv.org/abs/2503.01653)  
**Code**: [Innse/DisPro](https://github.com/Innse/DisPro)  
**Area**: Medical Imaging  
**Keywords**: Survival Prediction, Missing Modality, Prompt Learning, LLM Robustness, Pathology WSI, Genomics, Knowledge Distillation

## TL;DR

This paper proposes DisPro (Distilled Prompt Learning), a two-stage prompt learning framework—UniPro to distill the knowledge distribution of each modality, and MultiPro to leverage LLMs to infer missing modalities from available ones. By simultaneously compensating for both modality-specific and modality-common information of the missing modalities, DisPro achieves SOTA performance on five TCGA survival prediction datasets.

## Background & Motivation

Multimodal survival prediction integrates pathology images and genomic data for precise prognostic analysis, serving as an important task in computational pathology. However, missing modalities are common in clinical practice:

1. **Data Acquisition Limitations**: Genomic sequencing is costly (especially in underdeveloped regions), and pathology slides may be lost or of insufficient quality.
2. **Fragility of Existing Multimodal Models**: Models like MCAT, MOTCat, and CMTA assume all modalities are available; their performance drops drastically when any modality is missing.

Existing approaches for handling missing modalities have fundamental limitations:
- **Generative Completion** (Diffusion/VAE): Can only infer modality-common information and cannot generate the modality-specific information of missing modalities "out of thin air."
- **Retrieval-based Completion** (M3Care): High randomness in a single retrieved sample makes it difficult to comprehensively capture modality-specific knowledge.
- **Without-Completion Methods** (MUSE, MAP): Learn modality-invariant representations, which similarly neglect modality-specific knowledge.

Core Insight: It is necessary to simultaneously compensate for both **modality-common** and **modality-specific** information of the missing modalities.

## Method

### Overall Architecture

DisPro consists of two stages:
- **Stage 1 - UniPro (Unimodal Prompt)**: Train learnable prompts independently for each modality to distill the knowledge distribution of that modality across different risk levels.
- **Stage 2 - MultiPro (Multimodal Prompt)**: Treat available modalities as prompts for LLMs to infer the representations of missing modalities, and inject the modality-specific knowledge learned in Stage 1.

### Key Designs

**1. Unimodal Prompt Distillation (UniPro)**

Inspired by CoOp, but extended to the Multiple Instance Learning (MIL) paradigm to accommodate gigapixel WSIs ($100,000 \times 100,000$ pixels):
- Design learnable context tokens $[P]_1...[P]_k$ for each risk level ($2I_t$ classes).
- Map pathology patch/genomic pathway features to the LLM text space (768 dimensions) via an adapter.
- CLIP-style contrastive learning: Calculate the similarity between each patch and the text representations of each class.
- Top-K max-pooling aggregation to obtain slide-level predictions, optimized by Negative Log-Likelihood (NLL) loss.

Key Output: The textual class representations $\mathbf{t}_p^{(j)}$ / $\mathbf{t}_g^{(j)}$ of each modality encode the knowledge distribution of that modality across different risk levels.

**2. Multimodal Prompt Inference (MultiPro)**

Taking the scenario where pathology is available and genomics is missing as an example:
- Use pathology patch features as input tokens for the LLM (BERT).
- Replace the genomic positions with learnable placeholder tokens.
- The LLM infers genomic representations from pathology information via self-attention $\rightarrow$ compensating for modality-common information.

**UniPro Scoring**: Intelligently select input tokens to address the LLM's 512-token length limit and information redundancy:
$$\mathbf{s}_{n,\#}^{(i)} = \mathbf{s}_{n,p}^{(i,\tau)} + \mathbf{s}_{n,g}^{(i,\tau)} + \mathbf{a}_{n,p}^{(i)}$$
- First term: Correlation score with the current modality's UniPro (selecting discriminative tokens).
- Second term: Correlation score with the missing modality's UniPro (selecting cross-modal relevant tokens).
- Third term: Learnable self-scoring (dynamically adapting to the current input).

**3. UniPro Distillation**

Align the inferred missing modality part $[\tilde{\mathbf{g}}_n]$ from the LLM output with the genomic textual class representations from Stage 1:
- Compute the similarity between the inferred representations and each textual class representation.
- Force the inferred risk probability distribution to match the distribution learned by UniPro via survival loss.
- Thereby injecting modality-specific knowledge of the missing modality.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{surv}^{cls} + \alpha_1 \mathcal{L}_{ud}^p + \alpha_2 \mathcal{L}_{ud}^g$$

- $\mathcal{L}_{surv}^{cls}$: NLL loss for survival prediction based on the [CLS] token.
- $\mathcal{L}_{ud}^p$, $\mathcal{L}_{ud}^g$: UniPro distillation losses, which supervise compensatory learning when pathology and genomics are missing, respectively.

## Key Experimental Results

### 5 TCGA Datasets (60% Modality Missing Rate)

| Method | Missing Scenario | BLCA | BRCA | COADREAD | LUAD | UCEC | Avg |
|------|----------|------|------|----------|------|------|-----|
| MOTCat | Complete | 0.627 | 0.672 | 0.650 | 0.675 | 0.721 | 0.669 |
| SurvPath | Complete | 0.657 | 0.707 | 0.708 | 0.680 | 0.739 | 0.698 |
| COM | P-avail, G-missing | 0.602 | 0.674 | 0.678 | 0.634 | 0.699 | 0.657 |
| M3Care | P-avail, G-missing | 0.621 | 0.669 | 0.657 | 0.622 | 0.703 | 0.654 |
| MAP | P-avail, G-missing | 0.592 | 0.628 | 0.597 | 0.649 | 0.693 | 0.632 |
| **DisPro** | **P-avail, G-missing** | **0.632** | **0.690** | **0.688** | **0.661** | **0.727** | **0.680** |
| **DisPro** | **Both available** | **0.664** | **0.722** | **0.703** | **0.674** | **0.748** | **0.702** |

### Key Findings

- **Under 60% missingness**: DisPro achieves an average C-index of 0.680 vs. MAP's 0.632 vs. M3Care's 0.654, leading by a significant margin.
- **Under complete modalities**: DisPro (0.702) even outperforms SurvPath (0.698), demonstrating the intrinsic superiority of the proposed framework.
- **Across different missingness rates (0%~60%)**: DisPro exhibits the smallest performance degradation and the best robustness.

### Ablation Study

| Configuration | Effect |
|------|------|
| Without UniPro Distillation | Significant drop, loss of modality-specific information |
| Without UniPro Scoring | Random token selection leads to information loss |
| Without UniPro Distillation | Unable to inject modality-specific knowledge |
| Full DisPro | Optimal |

## Highlights & Insights

1. **Clear Analysis from an Information-Theoretic Perspective**: Explicitly decomposes missing modality information into modality-common and modality-specific components, pointing out that existing methods can only compensate for the former.
2. **Extension from CoOp to MIL**: Elegantly extends prompt learning to extremely large WSIs, bridging patch-level and slide-level predictions via Top-K pooling.
3. **Leveraging LLM Robustness**: Utilizes the reasoning capability of LLMs on missing inputs solely through prompt engineering and adapters, without fine-tuning LLM parameters.
4. **Threefold Reuse of UniPro**: The outputs of Stage 1 serve as distillation targets, token scorers, and inference guidance in Stage 2.

## Limitations & Future Work

1. The two-stage training increases implementation complexity, requiring independent training for each modality in Stage 1.
2. The context length limitation of BERT-based LLMs (512 tokens) necessitates significant downsampling of WSIs, potentially losing fine-grained information.
3. Only two modalities (pathology + genomics) are considered, without validating scalability to three or more modalities.
4. The quality of UniPro's knowledge distillation depends on the expressiveness of the learnable prompts, which may be insufficient on small datasets.

## Related Work & Insights

- **Multimodal Survival Prediction**: MCAT $\rightarrow$ MOTCat $\rightarrow$ CMTA $\rightarrow$ PIBD $\rightarrow$ MMP $\rightarrow$ SurvPath
- **Missing Modality Learning**: SMIL (Bayesian meta-learning) $\rightarrow$ M3Care (retrieval-based completion) $\rightarrow$ MUSE (graph contrastive) $\rightarrow$ MAP (LLM prompt)
- **Prompt Learning**: CoOp $\rightarrow$ CoCoOp $\rightarrow$ Ours (extended to the MIL paradigm)
- **Computational Pathology**: TransMIL $\rightarrow$ CLAM $\rightarrow$ CONCH $\rightarrow$ UNI

## Rating

- **Novelty**: 5/5 — Two-stage prompt learning simultaneously compensates for both modality-shared and modality-specific information, with clear theoretical motivation.
- **Effectiveness**: 4/5 — Consistent advantages across 5 datasets, outperforming complete-modality methods at 60% missingness.
- **Clarity**: 4/5 — The framework is complex but the diagrams are clear, with a consistent notation system.
- **Significance**: 5/5 — Resolves a critical bottleneck in translating multimodal medical AI from laboratories to clinical settings (missing modalities).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] H2-Surv: Hierarchical Hyperbolic Multimodal Representation Learning for Survival Prediction](../../CVPR2026/medical_imaging/h2-surv_hierarchical_hyperbolic_multimodal_representation_learning_for_survival_.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[CVPR 2025\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](../../ICLR2026/medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)
- [\[CVPR 2026\] MDCS-MoAME: Multi-directional Composite Scanning with Mixture of Attention and Mamba Experts for Cancer Survival Prediction](../../CVPR2026/medical_imaging/mdcs-moame_multi-directional_composite_scanning_with_mixture_of_attention_and_ma.md)

</div>

<!-- RELATED:END -->
