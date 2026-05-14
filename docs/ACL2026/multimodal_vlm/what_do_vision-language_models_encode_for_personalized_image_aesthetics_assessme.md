---
title: >-
  [Paper Note] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?
description: >-
  [ACL 2026][Multimodal VLM][Personalized Aesthetics Assessment] Through linear probing, this paper demonstrates that VLM hidden representations encode rich, multi-level aesthetic attribute information (illumination, color…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Personalized Aesthetics Assessment"
  - "Vision-Language Models"
  - "Linear Probing"
  - "Hidden Representations"
  - "Image Aesthetics"
date: 2026-05-08
content_hash: c60de1539535ca88
---

# What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?

**Conference**: ACL 2026
**arXiv**: [2604.11374](https://arxiv.org/abs/2604.11374)
**Code**: [https://github.com/ynklab/vlm-latent-piaa](https://github.com/ynklab/vlm-latent-piaa)
**Area**: Multimodal VLM
**Keywords**: Personalized Aesthetics Assessment, Vision-Language Models, Linear Probing, Hidden Representations, Image Aesthetics

## TL;DR

Through linear probing, this paper demonstrates that VLM hidden representations encode rich, multi-level aesthetic attribute information (illumination, color, composition, etc.) that propagates into language decoder layers. Building on this finding, the authors propose a simple linear regression approach for personalized image aesthetics assessment (PIAA) that requires no fine-tuning, significantly outperforming few-shot and LoRA fine-tuning baselines.

## Background & Motivation

**Background**: Personalized image aesthetics assessment (PIAA) aims to predict individual users' aesthetic scores for images, reflecting personal aesthetic preferences. Existing methods typically require pre-training on large-scale general aesthetics datasets followed by per-user adaptation, incurring high computational costs and questionable cross-domain transferability.

**Limitations of Prior Work**: Existing PIAA methods require multi-stage training pipelines (general aesthetics pre-training + user adaptation) and are heavily dependent on domain-specific training data. Applications of VLMs to aesthetics assessment have been limited to demographic group-level predictions, without achieving individual-level personalization. Furthermore, it remains unclear whether the internal representations of VLMs encode the multi-level, continuous aesthetic attributes necessary for personalization.

**Key Challenge**: While VLMs acquire rich visual-semantic understanding through large-scale pre-training, whether their hidden representations contain sufficiently fine-grained aesthetic information to support personalized assessment has not been validated.

**Goal**: (1) To verify, through linear probing, which aesthetic attributes are encoded in VLM hidden representations; (2) To leverage these representations for lightweight, fine-tuning-free, individual-level PIAA.

**Key Insight**: Drawing on the linear probing methodology from representation analysis, the authors systematically analyze the visual encoders and language decoders of VLMs layer by layer to reveal where aesthetic information is encoded and how it propagates.

**Core Idea**: VLM hidden representations naturally encode multi-dimensional aesthetic attribute information, and simple linear regression suffices to map these representations to personalized aesthetic scores without any model fine-tuning.

## Method

### Overall Architecture

The method consists of two stages: first, linear probing is applied to analyze the encoding of aesthetic attributes across VLM layers (probing stage); then, based on these findings, user-specific linear models are trained to predict personalized aesthetic scores from VLM hidden representations (PIAA stage). The input is an image paired with a fixed prompt ("Assess the aesthetics of this image."). Hidden representations are extracted from each layer and reduced to a single vector via average pooling.

### Key Designs

1. **Multi-Layer Aesthetic Attribute Linear Probing**:

    - Function: Verify which aesthetic attributes are encoded in VLM hidden representations and identify in which layers the encoding is strongest.
    - Mechanism: Ridge regression is trained on hidden representations $\mathbf{h}(I)$ from each VLM layer to predict the 11-dimensional aesthetic attribute vector from the AADB dataset (covering object, illumination, color harmony, depth of field, composition, etc.). Three types of representations are extracted: visual encoder output $\mathbf{V}_i$, language decoder text tokens $\mathbf{LT}_i$, and language decoder visual tokens $\mathbf{LV}_i$. Probing quality is evaluated using Spearman correlation.
    - Design Motivation: Prior work only verified CLIP's ability to encode overall aesthetic scores. Personalization requires multi-dimensional, fine-grained aesthetic attributes. This paper provides the first systematic verification of the existence and distribution of multi-attribute aesthetic information within VLMs.

2. **User-Specific Linear Regression (Linear-Hidden)**:

    - Function: Predict individual users' aesthetic scores from VLM hidden representations.
    - Mechanism: For each user $u$, a user-specific ridge regression model $M_u$ is trained such that $M_u \mathbf{h}(I) \approx s_{I,u}$. The average-pooled text token representation from the 15th language decoder layer ($\mathbf{LT}_{15}$) is used as input, requiring only 100 user-annotated images for training.
    - Design Motivation: Probing analysis reveals that intermediate language decoder layers stably contain rich aesthetic information. The linear model is both lightweight and interpretable, avoiding the substantial overhead of fine-tuning VLMs.

3. **Attribute Dimensionality Reduction Variant (Linear-Hidden Reduce)**:

    - Function: Verify whether the aesthetic attributes identified by linear probing constitute sufficient information for personalization.
    - Mechanism: A general regressor $M$ is first trained to project VLM representations into the AADB aesthetic attribute space (excluding the overall score); a user-specific regressor $M'_u$ is then trained on this low-dimensional attribute space to predict personalized scores.
    - Design Motivation: If performance does not degrade after dimensionality reduction, the probed aesthetic attributes are sufficient for personalization; if it does, the VLM representations contain additional useful information beyond what probing captures.

### Loss & Training

Ridge regression (L2-regularized linear regression) is used, requiring no gradient-based optimization, making training extremely lightweight. An independent regression model is trained per user, with a support set of 100 images and a test set of 50 images.

## Key Experimental Results

### Main Results

| Method | PARA (ρ) | PARA (R²) | LAPIS (ρ) | LAPIS (R²) |
|--------|------|------|----------|------|
| Raw Text (Qwen3-VL 4B) | 0.570 | -1.277 | 0.176 | -0.937 |
| Few-shot (10-shot) | 0.197 | -1.576 | - | - |
| LoRA (100-shot) | 0.578 | -1.751 | - | - |
| Linear-Hidden (Qwen3-VL 4B) | 0.611 | 0.362 | 0.401 | 0.138 |
| Linear-Hidden Reduce | 0.597 | 0.382 | 0.315 | 0.061 |
| PIAA-ICI (in-domain) | 0.590 | 0.303 | - | - |
| PIAA-ICI (cross-domain transfer) | - | - | 0.277 | -0.120 |

### Ablation Study

| Configuration | PARA (ρ) | Description |
|------|---------|------|
| Linear-Hidden (full representation) | 0.611 | Using complete VLM hidden representations |
| Linear-Hidden (GIAA) | 0.603 | Replacing personalized annotations with general aesthetic scores |
| Linear-Hidden (Reduce) | 0.597 | Using only probed aesthetic attributes |

### Key Findings

- **VLMs Encode Multi-Dimensional Aesthetic Attributes**: More than half of the aesthetic attributes achieve moderate or higher positive correlation (Spearman > 0.4) in VLM hidden representations, with Object (0.722), VividColor (0.696), and Overall Score (0.727) showing the strongest encoding.
- **Language Decoder Layers Carry Aesthetic Information**: Text token representations in the language decoder achieve probing performance comparable to or better than the visual encoder on most attributes, while the purely visual model DINOv3 performs worst on nearly all attributes.
- **Architecture Differences Affect Information Propagation**: In Gemma 3, aesthetic information transfers from visual tokens to text tokens in the early-to-middle language decoder layers; in Qwen3-VL, due to the DeepStack architecture, both token types remain consistent across layers.
- **Photo Domain vs. Artwork Domain**: On the photo dataset PARA, the Reduce variant approaches the full model performance (0.597 vs. 0.611), whereas the gap is larger on the artwork dataset LAPIS (0.315 vs. 0.401), indicating that artwork assessment requires additional information not captured by photo-based probing.
- **Simple Linear Models Outperform Fine-Tuning**: Linear-Hidden substantially outperforms few-shot, LoRA, and Raw Text methods based on text output, even surpassing the domain-specific PIAA-ICI model that requires additional pre-training.

## Highlights & Insights

- **"Reading Hidden Layers" Outperforms "Reading Text Output"**: VLM-generated text scores (Raw Text) are far inferior to linear regression directly on hidden representations, demonstrating that hidden representations contain substantial aesthetic information that is not preserved by the text generation process. This finding has implications for other subjective assessment tasks.
- **Extremely Lightweight Personalization**: Each user requires only a single ridge regression model trained on 100 images, with no VLM parameter fine-tuning, enabling efficient individual-level personalization.
- **Cross-Domain Transfer Insights**: Aesthetic attributes probed on photos transfer reasonably well to photo-domain PIAA, but artwork-domain assessment requires additional information, providing a direction for future cross-domain aesthetics evaluation.

## Limitations & Future Work

- Only two VLM families (Qwen3-VL, Gemma 3) are evaluated; larger-scale models and other architectures are not explored.
- Linear probing can only capture linearly separable information; non-linearly encoded aesthetic attributes in VLMs may be missed.
- Personalization is based solely on image representations without considering user attributes (e.g., age, gender, cultural background), potentially limiting personalization depth.
- The AADB aesthetic attribute dimensions are limited (11 dimensions), potentially missing aesthetic dimensions important to certain users.

## Related Work & Insights

- **vs. PIAA-ICI**: PIAA-ICI requires a two-stage pipeline of large-scale PIAA pre-training followed by user fine-tuning, with high computational cost and poor cross-domain transfer. Linear-Hidden requires no pre-training, matches or surpasses PIAA-ICI in-domain, and substantially outperforms it cross-domain.
- **vs. Hentschel et al. (2022)**: Prior work only probed overall aesthetic scores in CLIP visual encoders. This paper extends the analysis to multi-attribute, multi-layer, visual+language decoder systematic evaluation and advances to personalized application.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic analysis of aesthetic attribute encoding in VLM hidden layers with application to personalized aesthetics assessment
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-dataset comparisons with extensive variants and ablation analyses
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from probing analysis to application design is exceptionally clear
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for leveraging pre-trained model hidden representations in subjective assessment tasks

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](../../ICLR2026/multimodal_vlm/visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](../../CVPR2026/multimodal_vlm/do_vision_language_models_need_to_process_image_tokens.md)
- [\[ICLR 2026\] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](../../ICLR2026/multimodal_vlm/self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
