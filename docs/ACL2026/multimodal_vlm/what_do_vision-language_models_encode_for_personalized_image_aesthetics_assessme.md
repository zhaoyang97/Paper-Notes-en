---
title: >-
  [Paper Note] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?
description: >-
  [ACL 2026][Multimodal VLM][Personalized image aesthetics assessment] This paper utilizes linear probing to discover that VLM hidden representations encode rich multi-level aesthetic attribute information (lighting, color…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Personalized image aesthetics assessment"
  - "Vision-language models"
  - "Linear probing"
  - "Hidden representations"
  - "Image aesthetics"
date: 2026-05-08
content_hash: 51920741de27e365
---

# What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?

**Conference**: ACL 2026  
**arXiv**: [2604.11374](https://arxiv.org/abs/2604.11374)  
**Code**: [https://github.com/ynklab/vlm-latent-piaa](https://github.com/ynklab/vlm-latent-piaa)  
**Area**: Multimodal VLM  
**Keywords**: Personalized image aesthetics assessment, Vision-language models, Linear probing, Hidden representations, Image aesthetics

## TL;DR

This paper utilizes linear probing to discover that VLM hidden representations encode rich multi-level aesthetic attribute information (lighting, color, composition, etc.) which propagates to the language decoder layers. Based on this, it proposes achieving training-free personalized image aesthetics assessment (PIAA) via simple linear regression, significantly outperforming few-shot and LoRA fine-tuning baselines.

## Background & Motivation

**Background**: Personalized Image Aesthetics Assessment (PIAA) aims to predict aesthetic ratings for images for specific users, reflecting individual preferences. Existing methods usually require pre-training on large-scale general aesthetics datasets followed by user-specific adaptation, which entails high computational costs and questionable cross-domain transferability.

**Limitations of Prior Work**: Existing PIAA methods require multi-stage training pipelines (general aesthetics pre-training + user adaptation) and rely heavily on domain-specific training data. The application of VLMs in aesthetics assessment is limited to demographic group levels, failing to achieve individual-level personalization. Furthermore, it remains unclear whether the internal representations of VLMs encode the multi-level, continuous aesthetic attributes required for personalization.

**Key Challenge**: While VLMs acquire rich visual-semantic understanding through large-scale pre-training, whether the aesthetic information in their hidden representations is granular enough to support personalized assessment has not been verified.

**Goal**: (1) Verify which aesthetic attributes are encoded in VLM hidden representations via linear probing; (2) Utilize these representations to achieve lightweight, fine-tuning-free individual-level PIAA.

**Key Insight**: Leveraging linear probing methodology from the field of representation analysis, this work conducts a layer-by-layer analysis of VLM vision encoders and language decoders to reveal the location and propagation patterns of aesthetic information.

**Core Idea**: Multi-dimensional aesthetic attribute information is naturally encoded within VLM hidden representations. Simple linear regression can map these representations to personalized aesthetic scores without any model fine-tuning.

## Method

### Overall Architecture

The method is divided into two stages: first, analyzing the encoding of aesthetic attributes across VLM layers (probing stage); then, based on the findings, training user-specific linear models to predict personalized aesthetic scores from VLM hidden representations (PIAA stage). The input consists of an image + a fixed prompt ("Assess the aesthetics of this image."), with hidden representations from each layer extracted and mean-pooled into a single vector.

### Key Designs

1.  **Multi-layer Aesthetic Attribute Linear Probing**:
    - **Function**: Verify which aesthetic attributes are encoded in VLM hidden representations and identify the layers where encoding is strongest.
    - **Mechanism**: Train ridge regressions on the hidden representations $\mathbf{h}(I)$ for each VLM layer to predict an 11-dimensional aesthetic attribute vector from the AADB dataset (including object, lighting, color harmony, depth of field, composition, etc.). Three types of representations are extracted: vision encoder output $\mathbf{V}_i$, language decoder text tokens $\mathbf{LT}_i$, and language decoder vision tokens $\mathbf{LV}_i$. Quality is evaluated via Spearman correlation.
    - **Design Motivation**: Previous work only verified the ability of CLIP to encode overall aesthetic scores, but personalization requires multi-dimensional, fine-grained attributes. This work provides the first systematic verification of the existence and distribution of multi-attribute aesthetic information in VLMs.

2.  **User-Specific Linear Regression (Linear-Hidden)**:
    - **Function**: Predict individual user aesthetic scores from VLM hidden representations.
    - **Mechanism**: For each user $u$, train a user-specific ridge regression model $M_u$ such that $M_u \mathbf{h}(I) \approx s_{I,u}$. The mean-pooled representation of text tokens from the 15th layer of the language decoder ($\mathbf{LT}_{15}$) is used as input. Training requires only 100 user-labeled images.
    - **Design Motivation**: Probing analysis revealed that intermediate layers of the language decoder stably contain rich aesthetic information. Local linear models are both lightweight and interpretable, avoiding the high overhead of fine-tuning the VLM.

3.  **Attribute Reduction Variant (Linear-Hidden Reduce)**:
    - **Function**: Verify if the aesthetic attributes identified via linear probing are a sufficient information source for personalization.
    - **Mechanism**: A general regressor $M$ is first trained to project VLM representations into the AADB aesthetic attribute space (excluding the overall score), then a user-specific regressor $M'_u$ is trained on this low-dimensional attribute space to predict personalized scores.
    - **Design Motivation**: If performance does not drop after reduction, it indicates that the probed aesthetic attributes are sufficient for personalization; a drop suggests that VLM representations contain additional useful information not captured by the probe.

### Loss & Training

Ridge regression (L2-regularized linear regression) is utilized, requiring no gradient optimization and enabling extremely lightweight training. A regression model is trained independently for each user using a support set of 100 images and a test set of 50 images.

## Key Experimental Results

### Main Results

| Method | PARA (ρ) | PARA (R²) | LAPIS (ρ) | LAPIS (R²) |
|:---|:---:|:---:|:---:|:---:|
| Raw Text (Qwen3-VL 4B) | 0.570 | -1.277 | 0.176 | -0.937 |
| Few-shot (10-shot) | 0.197 | -1.576 | - | - |
| LoRA (100-shot) | 0.578 | -1.751 | - | - |
| Linear-Hidden (Qwen3-VL 4B) | 0.611 | 0.362 | 0.401 | 0.138 |
| Linear-Hidden Reduce | 0.597 | 0.382 | 0.315 | 0.061 |
| PIAA-ICI (In-domain) | 0.590 | 0.303 | - | - |
| PIAA-ICI (Cross-domain) | - | - | 0.277 | -0.120 |

### Ablation Study

| Configuration | PARA (ρ) | Description |
|:---|:---:|:---|
| Linear-Hidden (Full) | 0.611 | Uses full VLM hidden representations |
| Linear-Hidden (GIAA) | 0.603 | Replaces personalized labels with general aesthetic scores |
| Linear-Hidden (Reduce) | 0.597 | Uses only probed aesthetic attributes |

### Key Findings

- **VLMs encode multi-dimensional aesthetic attributes**: Over half of the aesthetic attributes reach moderate or higher positive correlations (Spearman > 0.4) in VLM hidden representations, with Object (0.722), VividColor (0.696), and Overall Score (0.727) being the strongest.
- **Language decoder layers carry aesthetic information**: Text token representations in the language decoder achieve probing performance comparable to or better than the vision encoder for most attributes. The purely visual model DINOv3 performs worst across almost all attributes.
- **Architectural differences affect information propagation**: In Gemma 3, aesthetic information shifts from vision tokens to text tokens in the early-to-mid layers of the language decoder; in Qwen3-VL, due to the DeepStack architecture, both remain consistent across layers.
- **Photo domain vs. Art domain**: On the photo dataset PARA, the Reduce variant is close to the full model performance (0.597 vs 0.611), but the gap is larger on the artwork dataset LAPIS (0.315 vs 0.401), indicating that artwork assessment requires additional information not captured by photo-based probing.
- **Simple linear outperforms fine-tuning**: Linear-Hidden significantly outperforms text-output-based methods like Few-shot, LoRA, and Raw Text, and even surpasses domain-specific PIAA-ICI models that require additional pre-training.

## Highlights & Insights

- **"Reading hidden layers" is more effective than "reading text output"**: Aesthetic scores generated as text by the VLM (Raw Text) are far inferior to linear regression directly from hidden representations, indicating that hidden representations contain substantial aesthetic information not preserved during the text generation process. This finding is also instructive for other subjective assessment tasks.
- **Extremely lightweight personalization scheme**: Only one ridge regression model needs to be trained per user (100 images) without fine-tuning VLM parameters, achieving efficient individual-level personalization.
- **Insights on cross-domain transfer**: Aesthetic attributes probed on photos transfer well to photo-domain PIAA, but the artwork domain requires additional information, providing a direction for future cross-domain aesthetic assessment.

## Limitations & Future Work

- Only two VLM families (Qwen3-VL, Gemma 3) were tested; larger scale models or other architectures were not explored.
- Linear probing only captures linearly separable information; aesthetic attributes might be encoded non-linearly within VLMs.
- Personalization is based solely on image representations and does not consider user attributes (e.g., age, gender, cultural background), which may limit the depth of personalization.
- The aesthetic attribute dimensions of AADB are limited (11 dimensions), potentially missing aesthetic dimensions important to certain users.

## Related Work & Insights

- **vs PIAA-ICI**: PIAA-ICI requires a two-stage process of pre-training on large-scale PIAA data followed by user fine-tuning, which is computationally expensive and shows poor cross-domain transfer. Linear-Hidden requires no pre-training, matches or exceeds PIAA-ICI in-domain, and is significantly better cross-domain.
- **vs Hentschel et al. (2022)**: Prior work only probed overall aesthetic scores on the CLIP vision encoder. This work extends the analysis to multi-attribute, multi-layer, and vision + language decoder systems, while advancing to personalized applications.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic analysis of aesthetic attribute encoding in VLM hidden layers for personalized assessment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across multiple models and datasets, including rich variants and ablation analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from probing analysis to application design is very clear.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for using pre-trained model hidden representations for subjective assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](../../ICLR2026/multimodal_vlm/visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](../../CVPR2026/multimodal_vlm/do_vision_language_models_need_to_process_image_tokens.md)
- [\[ICLR 2026\] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](../../ICLR2026/multimodal_vlm/self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)

</div>

<!-- RELATED:END -->
