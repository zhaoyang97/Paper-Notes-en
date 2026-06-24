---
title: >-
  [Paper Note] AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling
description: >-
  [ECCV 2024][Recommender Systems][Image Content Appeal] This paper proposes the Image Content Appeal Assessment (ICAA) task for the first time, distinguishing it from traditional Image Aesthetics Assessment (IAA). It designs a complete pipeline integrating automatic dataset generation, appeal estimation, and appeal enhancement, achieving large-scale dataset creation with zero human annotation using Stable Diffusion and Textual Inversion.
tags:
  - "ECCV 2024"
  - "Recommender Systems"
  - "Image Content Appeal"
  - "Aesthetics Assessment"
  - "Dataset Creation"
  - "Stable Diffusion"
  - "Textual Inversion"
date: 2026-05-08
content_hash: a389ef7064b7d48d
---

# AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling

**Conference**: ECCV 2024  
**arXiv**: [2407.05546](https://arxiv.org/abs/2407.05546)  
**Code**: [https://github.com/SherryXTChen/AID-Appeal](https://github.com/SherryXTChen/AID-Appeal)  
**Area**: Image Quality Assessment / Recommendation System  
**Keywords**: Image Content Appeal, Aesthetics Assessment, Dataset Creation, Stable Diffusion, Textual Inversion

## TL;DR
This paper proposes the Image Content Appeal Assessment (ICAA) task for the first time, distinguishing it from traditional Image Aesthetics Assessment (IAA). It designs a complete pipeline integrating automatic dataset generation, appeal estimation, and appeal enhancement, achieving large-scale dataset creation with zero human annotation using Stable Diffusion and Textual Inversion.

## Background & Motivation
**Background**: Image quality assessment has two mature directions: Image Quality Assessment (IQA, for distortion) and Image Aesthetics Assessment (IAA). However, the dimension of "content appeal" has been neglected—for example, a professionally taken photo (high aesthetic score) of a moldy hamburger is completely unappealing in content.

**Limitations of Prior Work**:
   - Existing IAA methods (DIAA, MPADA, NIMA) assign high scores to professionally shot but unappealing content, confusing "well-shot" with "content-appealing".
   - There is no dedicated ICAA dataset; existing datasets only contain coarse-grained labels like "interesting content".
   - Annotating large-scale image assessment datasets manually is extremely expensive (e.g., IQA annotation requires heavy manual labor).

**Key Challenge**: Constructing a large-scale ICAA dataset is required to train models, but manual annotation costs are prohibitive; furthermore, the concept of "content appeal" must be strictly distinguished from "aesthetics".

**Goal**: (a) Define and formalize the ICAA task; (b) automatically construct an ICAA dataset; (c) train appeal assessment and enhancement models.

**Key Insight**: Using Textual Inversion in Stable Diffusion to learn embedding representations of "appealing/unappealing", controlling appeal levels via linear interpolation to generate synthetic data with continuous gradients.

**Core Idea**: Learn textual embeddings of appealing/unappealing via Textual Inversion, control the appeal level of synthetic images via $\alpha$ linear interpolation to train a Siamese comparator for automatic labeling of real images, and subsequently train an absolute score estimator.

## Method

### Overall Architecture
The pipeline consists of four steps: (1) automatically search and collect domain images (e.g., food/room); (2) learn appeal embeddings via Textual Inversion and generate synthetic data of different appeal levels using Stable Diffusion (SD) inpainting; (3) train a relative appeal comparator (Siamese CLIP) to automatically label large-scale real images; (4) train an absolute appeal estimator and an appeal enhancer.

### Key Designs

1. **Domain-Relevancy Map Generation**:

    - **Function**: Accurately locate domain-relevant regions in the image (e.g., food areas) to manipulate appeal only within these regions.
    - **Mechanism**: BLIP generates image descriptions $\rightarrow$ NLTK extracts noun phrases $\rightarrow$ WordNet matches domain words (e.g., noun.food) $\rightarrow$ CLIPSeg segments domain-relevant regions $M_D(I)$.
    - **Design Motivation**: Ensure that only the content region is modified without changing the background, avoiding background interference in appeal evaluation.

2. **Textual Inversion Embeddings + Synthetic Data**:

    - **Function**: Learn textual embedding vectors $z_D^+$ and $z_D^-$ representing "appealing" and "unappealing" concepts.
    - **Mechanism**: Select the best-matching positive/negative image sets from search results to learn embeddings via Textual Inversion. During synthesis, control the appeal level through interpolation $f(\alpha) = \alpha z_D^+ + (1-\alpha)z_D^-$, and use SD inpainting to modify only the $M_D(I)$ region: $I' = SD(I, BLIP(I) + f(\alpha), M_D(I), seed())$.
    - Concurrently, randomize backgrounds to increase diversity: $I'' = SD(I', \text{" "}, 1-M_D(I), seed())$.
    - **Design Motivation**: The linear interpolation assumption is simple yet effective—intermediate points in the embedding space correspond to intermediate appeal levels.

3. **Relative Appeal Comparator $\rightarrow$ Automatic Labeling**:

    - **Function**: Train a Siamese CLIP network to predict the difference in appeal between two images, and then assign absolute scores to real images using a voting scheme.
    - **Mechanism**: The training set consists of pairs of different $\alpha$-variants of the same source image, with the label being $\alpha_1 - \alpha_2$. During labeling, each image is compared against a set of exemplar images, and the scores are averaged and scaled to 1–10.
    - **Design Motivation**: Relative comparison is easier to learn than absolute scoring, which also aligns with human judgment.

4. **Content Appeal Enhancement**:

    - **Function**: Locate unappealing regions within an image and enhance them.
    - **Mechanism**: Generate an appeal heatmap $M_D^H(I)$ using a sliding window (where low-appeal regions have higher weights), and then apply SD inpainting guided by the heatmap using the $z_D^+$ embedding: $SD(I, BLIP(I) + z_D^+, M_D^H(I), seed())$.
    - **Design Motivation**: Enhance only the regions that need improvement, avoiding over-modification.

### Loss & Training
- Relative comparator: MAE loss $|A_{pred}(I_1, I_2) - (\alpha_1 - \alpha_2)|$, two-stage training of the CLIP backbone.
- Absolute estimator: MAE loss $|A_{pred}(I) - A(I)|$, also trained in two stages.
- Enhancement uses SD v2.1 inpainting + depth-guided ControlNet.

## Key Experimental Results

### Main Results: No Correlation Between ICAA and IAA

| Metric | DIAA (Food) | MPADA (Food) | NIMA (Food) | DIAA (Room) | MPADA (Room) | NIMA (Room) |
|------|-------------|--------------|-------------|-------------|--------------|-------------|
| PLCC | 0.168 | 0.005 | 0.01 | -0.123 | -0.012 | -0.147 |
| SRCC | 0.162 | -0.015 | 0.003 | -0.121 | -0.017 | -0.149 |

$\rightarrow$ Content appeal has virtually zero correlation with aesthetic scores, validating that ICAA is a new dimension independent of IAA.

### User Study

| Question | Food | Room |
|------|------|------|
| Post-enhancement content is more appealing | 76.3% | 79.2% |
| Post-enhancement image is more realistic | 65.8% | 72.1% |

### Ablation Study

| Dataset | Image Count | Estimator MAE |
|--------|--------|-----------|
| Food (𝕀_F) | 78,917 | 0.6756 |
| Room (𝕀_R) | 75,287 | 0.6332 |

### Key Findings
- **ICAA and IAA are indeed orthogonal**: Correlation coefficients are close to zero, or even negative (e.g., NIMA PLCC = -0.147 in the Room domain).
- **Automatic labeling pipeline is effective**: The MAE is less than 0.7 on a 1–10 scale, which is acceptable given the subjectivity of the task.
- **76%+ user preference for enhanced images**: This proves that appeal estimation aligns with human perception.
- **Cross-domain generalization**: The same pipeline is applicable to both food and room domains, which exhibit significant differences.

## Highlights & Insights
- **Proposed a new evaluation dimension**: Explicitly separates "content appeal" from "aesthetics", filling a gap in the image quality assessment framework. This insight has direct value for application scenarios such as e-commerce, food delivery platforms, and hotel photos.
- **Fully automatic zero-annotation pipeline**: From search queries $\rightarrow$ synthetic data $\rightarrow$ automatic labeling $\rightarrow$ training, the entire process requires no human annotation. The core ingenuity lies in utilizing Textual Inversion to learn a continuous representation of appeal.
- **Domain-relevancy map approach**: The combination of BLIP + WordNet + CLIPSeg precisely localizes domain-relevant content, ensuring that operations are restricted to appeal-related areas, which is highly practical.

## Limitations & Future Work
- **The linear interpolation assumption may be oversimplified**: A linear path in the embedding space does not necessarily correspond to a linear change in appeal.
- **Only evaluated on two domains (food and room)**: Generalization to more complex domains such as portraits and apparel still needs validation.
- **Limited expressiveness of Textual Inversion**: A single embedding vector might fail to capture all dimensions of "appealing/unappealing".
- **Background effects are neglected**: The paper assumes that background does not influence content appeal, whereas, in reality, table settings do affect food appeal.
- **Base images originate from low-resolution thumbnails on stock websites**: Super-resolved to 512×512 using ESRGAN, which may lead to unstable image quality.

## Related Work & Insights
- **vs NIMA/DIAA/MPADA**: These IAA methods evaluate aesthetics (composition, lighting), while this work evaluates content appeal—the two show little to no correlation and cannot replace each other.
- **vs CrowdCLIP/AFreeCA**: An interesting analogy—while those two works use SD synthetic data for counting, this paper uses SD synthetic data for appeal assessment; both follow the paradigm of "using generative models to synthesize training data for discriminative models".

## Rating
- Novelty: ⭐⭐⭐⭐ First to formalize the ICAA problem, featuring a cleverly designed fully automatic pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes user studies, cross-domain validation, and comparisons with IAA, but with relatively few ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, smooth pipeline logic, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ Opens up a new direction for content appeal assessment, yielding direct value for commercial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FineVQ: Fine-Grained User Generated Content Video Quality Assessment](../../CVPR2025/recommender/finevq_fine-grained_user_generated_content_video_quality_assessment.md)
- [\[ICLR 2026\] Steering Diffusion Models Towards Credible Content Recommendation](../../ICLR2026/recommender/steering_diffusion_models_towards_credible_content_recommendation.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[ICML 2025\] How to Set AdamW's Weight Decay as You Scale Model and Dataset Size](../../ICML2025/recommender/how_to_set_adamws_weight_decay_as_you_scale_model_and_dataset_size.md)
- [\[ACL 2025\] LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences](../../ACL2025/recommender/lotus_a_leaderboard_for_detailed_image_captioning_from_quality_to_societal_bias_.md)

</div>

<!-- RELATED:END -->
