---
title: >-
  [Paper Note] ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization
description: >-
  [ICCV 2025][Image Generation][Generative model personalization] This paper introduces **ImageGem**, the first large-scale in-the-wild generative interaction dataset (57K users × 242K customized LoRAs × 3M text prompts × 5M generated images), enabling three applications via individual user preference annotations: **aggregate preference alignment** surpassing Pick-a-Pic, **personalized retrieval and generative recommendation** (with significant VLM reranking gains), and the first proposed **generative model personalization**—learning preference editing directions in the LoRA latent weight space (W2W) to customize diffusion models.
tags:
  - ICCV 2025
  - Image Generation
  - Generative model personalization
  - user preference alignment
  - LoRA weight space
  - DiffusionDPO
  - recommender systems
  - diffusion models
date: 2026-05-08
content_hash: e74ba7b07a587772
---

# ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization

**Conference**: ICCV 2025
**arXiv**: [2510.18433](https://arxiv.org/abs/2510.18433)
**Code**: [Project Page](https://maps-research.github.io/imagegem-iccv2025/)
**Area**: Image Generation
**Keywords**: Generative model personalization, user preference alignment, LoRA weight space, DiffusionDPO, recommender systems, diffusion models

## TL;DR

This paper introduces **ImageGem**, the first large-scale in-the-wild generative interaction dataset (57K users × 242K customized LoRAs × 3M text prompts × 5M generated images), enabling three applications via individual user preference annotations: **aggregate preference alignment** surpassing Pick-a-Pic, **personalized retrieval and generative recommendation** (with significant VLM reranking gains), and the first proposed **generative model personalization**—learning preference editing directions in the LoRA latent weight space (W2W) to customize diffusion models.

## Background & Motivation

### Core Problem

> "A thousand readers have a thousand Hamlets."

When a user inputs "a portrait of Hamlet," each person envisions a different version. Existing text-to-image models can only generate images that conform to **population-level preferences**, failing to capture and produce **individual preferences**.

### Limitations of Prior Work

**Aggregate preference datasets** (e.g., Pick-a-Pic): collect user ratings over image pairs, but reflect **crowd-averaged preferences**.

**Identity customization datasets** (e.g., DreamBooth): support concept injection for specific persons or objects, but do not address **preference styles**.

**Absence of individual-level preferences**: no large-scale dataset records interactions between individual users and their customized models.

**Existing personalization methods** (e.g., ViPer) are limited to zero-shot settings and cannot exploit inter-user similarity.

### Data Source

Data are collected from **Civitai** (the largest AIGC platform), capturing publicly shared customized models and generated images to obtain real-world interaction data.

## Method

### Dataset Construction

#### Core Statistics (Table 1)

| Metric | Raw Data | After Safety Filtering |
|--------|----------|------------------------|
| Total images | 5,658,107 | 4,916,134 |
| Unique prompts | 2,975,943 | 2,895,364 |
| LoRA models | 242,889 | 242,118 |
| Unique model tags | 105,788 | 97,434 |
| Total users | 57,245 | — |
| Model uploaders | 19,003 | 18,889 |
| Avg. images per uploader | 49 | 48 |
| Avg. models per uploader | 12 | 13 |
| Avg. images per model | 62 | 54 |

#### Safety Checks

- Utilizes Civitai's built-in NSFW classification
- Detects prompt toxicity using **Detoxify**
- Filters images whose prompts have unsafe probability > 0.2
- IRB ethical approval obtained

#### Data Structure

Ternary relationships: **images ↔ LoRA models ↔ users**

Two types of user interaction data:
1. **Individual-level**: user–model interaction records (prompts, generation configurations, etc.), comprising 1.74M showcase images + 3.18M historical images
2. **Aggregate-level**: emoji feedback (likes, hearts, laughs, cries)

### Application 1: Aggregate Preference Alignment

Using the DiffusionDPO framework:

$$\max_{p_\theta} \mathbb{E}[r(\mathbf{c}, \mathbf{x}_0)] - \beta \mathbb{D}_{\text{KL}}[p_\theta(\mathbf{x}_{0:T}|\mathbf{c}) \| p_{\text{ref}}(\mathbf{x}_{0:T}|\mathbf{c})]$$

**Preference pair construction**:
- Apply HDBScan clustering on CLIP embeddings of prompts
- Construct preference pairs within each cluster via HPS v2 min-max pairing

### Application 2: Retrieval and Generative Recommendation

**Candidate retrieval**:
- Image retrieval: FAISS + ViT-initialized embeddings → Two-tower model achieves best performance
- Model recommendation: SASRec (self-attentive sequential recommendation) captures temporal evolution

**Generative recommendation** (VLM-driven):
- **Pixtral-12B** is employed in a two-stage pipeline:
  1. **Description stage**: extracts preference text descriptions $q_i$ from user history images
  2. **Ranking stage**: compares preference descriptions against candidates, outputting scores with explanations
- A randomized scoring strategy mitigates VLM ranking instability

### Application 3: Generative Model Personalization (Core Contribution)

#### W2W Weight Space Construction

1. **SVD normalization**: apply SVD to each LoRA weight matrix, retaining the top-1 component
2. **Flattening and concatenation**: compressed matrices from all layers are concatenated into a vector $\theta_i \in \mathbb{R}^d$
3. **PCA dimensionality reduction**: PCA is applied to $D = \{\theta_1, ..., \theta_N\}$, retaining the top-$m$ principal components
4. Basis vectors $\{w_1, ..., w_m\}$ encode user preference directions

#### Preference Direction Learning

- A linear classifier is trained using binary labels (whether a user prefers a given model)
- The normal vector $v$ of the decision hyperplane serves as the preference traversal direction
- Editing formula: $\theta_{\text{edit}} = \theta + \alpha v$, where $\alpha$ controls editing strength

#### Individual Preference Learning Pipeline

1. Compute CLIP embeddings for all user-generated images → HDBScan clustering → identify representative preference clusters
2. VLM generates style descriptions from the top-9 images within each cluster
3. CLIP similarity is used to assign preference labels to LoRA models
4. Individual hyperplanes are learned → multi-direction editing

## Key Experimental Results

### Aggregate Preference Alignment (Table 4)

| Data Subset | Pick Score↑ | HPSv2↑ | CLIP Score↑ |
|-------------|------------|--------|------------|
| Original SD1.5 | 0.1977 | 0.2637 | 0.3581 |
| Pick-a-pic Cars | 0.1993 | 0.2690 | 0.3607 |
| **ImageGem Cars Small** | **0.2004** | **0.2741** | **0.3745** |
| Original SD1.5 | 0.2010 | 0.2646 | 0.3560 |
| Pick-a-pic Dogs | 0.2058 | 0.2739 | 0.3617 |
| **ImageGem Dogs** | **0.2069** | **0.2789** | **0.3683** |
| Original SD1.5 | 0.1954 | 0.2640 | 0.3446 |
| Pick-a-pic Scenery | 0.1936 | 0.2676 | 0.3289 |
| **ImageGem Scenery Large** | **0.1961** | **0.2747** | **0.3427** |

**Key result**: Across all three topics, DPO models trained on ImageGem outperform Pick-a-Pic on Pick Score, HPSv2, and CLIP Score.

### Retrieval and Recommendation (Tables 5–7)

| Task | Best Method | Recall@10 / NDCG@10 |
|------|-------------|----------------------|
| Image retrieval (1M) | Two-tower | Rec@100=0.2402 |
| Model recommendation (200K) | SASRec | Rec@10=0.1839, NDCG@10=0.1239 |
| Reranking (images) | VLM | Rec@5=0.9500, NDCG@5=0.6745 |
| Reranking (models) | VLM | Rec@5=**0.7222**, NDCG@5=**0.4981** |

**Key finding**: VLM substantially outperforms traditional methods on model recommendation reranking (SASRec Rec@5 is only 0.50), while providing interpretable textual justifications.

### Ablation Study on Generative Model Personalization

| Strategy | Anime→Realistic | Realistic→Anime | Evaluation |
|----------|----------------|----------------|------------|
| SVD-based W2W | ✓ Effective both ways | ✓ Effective both ways | Most robust |
| attn_v layers | ✓ Effective one-way | ✗ Fails in reverse | Partially effective |
| FF layers | ✗ Poor both ways | ✗ Poor both ways | Not recommended |

Multi-user personalization validation: three distinct preference directions are learned for three users; both CLIP and VLM reranking confirm that edited models generate images better aligned with each user's individual preferences.

## Highlights & Insights

1. **Unique dataset value**: The first large-scale dataset capturing real user interactions with generative models, bridging the gap between aggregate and individual preferences.
2. **New paradigm—generative model personalization**: Rather than personalizing prompts or post-processing outputs, this work proposes **directly editing model weights** to align with individual preferences, opening an entirely new research direction.
3. **VLM as a recommendation engine**: Leveraging VLMs' multimodal understanding for interpretable reranking opens new possibilities in recommender systems.
4. **Data-driven preference representation**: Representing preferences as linear directions in W2W space enables efficient and controllable model editing.

## Limitations & Future Work

1. **PCA constraints**: Reliance on PCA restricts model selection to low-rank variants (rank 8/16), limiting model diversity.
2. **Preference pair construction**: HPS-based within-cluster pairing does not fully exploit implicit user feedback (e.g., interaction logs).
3. **Domain sparsity**: The approach performs best in the portrait domain; sparse domains such as scenery lack sufficient models to learn effective W2W spaces.
4. **Safety and privacy**: Despite safety filtering and IRB approval, NSFW content within the Civitai ecosystem warrants continued attention.
5. **Validation limited to SD series**: The method has not been validated on newer architectures such as Flux.

## Related Work & Insights

- **vs. Pick-a-Pic**: Pick-a-Pic relies on human-annotated explicit preference pairs; ImageGem uses implicit preferences derived from naturally observed data.
- **vs. Weights2Weights**: The original W2W approach uses self-trained rank-1 LoRAs to edit face identity spaces; ImageGem extends this to the user preference space, handling mixed-rank Civitai LoRAs.
- **vs. ViPer**: ViPer performs zero-shot preference learning (captured at inference time); ImageGem learns from historical interaction data, enabling the exploitation of inter-user similarity.
- **Insights**: The W2W space concept is generalizable to personalization of video generation models and 3D models; the VLM-driven recommendation paradigm can be extended to broader creative tools.

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐⭐ — Dataset + new paradigm (generative model personalization) + multi-application validation
**Practicality**: ⭐⭐⭐⭐ — Directly addresses real needs of AIGC platforms
**Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative validation across three application scenarios with diverse baselines
**Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-organized application scenario divisions

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] Understanding Flatness in Generative Models: Its Role and Benefits](understanding_flatness_in_generative_models_its_role_and_benefits.md)
- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)
- [\[ICCV 2025\] Deeply Supervised Flow-Based Generative Models](deeply_supervised_flow-based_generative_models.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)

<!-- RELATED:END -->
