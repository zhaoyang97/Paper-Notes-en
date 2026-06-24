---
title: >-
  [Paper Note] GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections
description: >-
  [ECCV 2024][Image Generation][Garment Generation] To address fine-grained semantic misalignment (component quantities, positions, and mutual relationships) in text-to-garment image generation, this work proposes GarmentAligner. It obtains spatial-quantitative information via an automatic component extraction pipeline and integrates retrieval-augmented contrastive learning with multi-level correction losses to achieve precise alignment of garment components at visual, spatial…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Garment Generation"
  - "Diffusion Models"
  - "Retrieval-Augmented"
  - "Contrastive Learning"
  - "Fine-Grained Alignment"
date: 2026-05-08
content_hash: 32473879445477ef
---

# GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections

**Conference**: ECCV 2024  
**arXiv**: [2408.12352](https://arxiv.org/abs/2408.12352)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Garment Generation, Diffusion Models, Retrieval-Augmented, Contrastive Learning, Fine-Grained Alignment

## TL;DR

To address fine-grained semantic misalignment (component quantities, positions, and mutual relationships) in text-to-garment image generation, this work proposes GarmentAligner. It obtains spatial-quantitative information via an automatic component extraction pipeline and integrates retrieval-augmented contrastive learning with multi-level correction losses to achieve precise alignment of garment components at visual, spatial, and quantitative levels.

## Background & Motivation

Text-to-garment generation, as a downstream task of text-to-image (T2I) generation, holds immense commercial value in the fashion industry. However, even state-of-the-art T2I models (e.g., Midjourney, SDXL) still perform poorly in garment generation:

**Key Challenge**:

**Semantic Discrepancies in Text**: Garment descriptions possess specific textual structures and professional modifiers (e.g., "pleated V-neck"), which general T2I models struggle to comprehend accurately.

**Difficulties in Fine-Grained Alignment of Components**: Garment components (buttons, pockets, collars, etc.) have unique attributes and complex mutual relationships. Existing methods primarily focus on overall visual semantics while ignoring **component localization** and **quantitative alignment**.

The authors demonstrate several typical failure cases of Midjourney in garment generation:
- Incorrect number of buttons (e.g., describing 5 but generating 3 or 7)
- Misplaced pockets (e.g., should be on the left chest but appear on the right)
- Chaotic component relationships (e.g., incorrect connection between the zipper and the collar)

**Key Insight**: It is necessary to mine component information from garment images and descriptions across multiple semantic levels, comprehensively improving generation quality from global perception to fine-grained details.

## Method

### Overall Architecture

GarmentAligner is based on a pre-trained latent diffusion model (SD v2.1) and undergoes domain adaptation via a **retrieval-augmented multi-level correction** training strategy. It consists of three core components:

1. **Automatic Component Extraction Pipeline**: Extracts spatial and quantitative information from garment images.
2. **Retrieval-augmented Contrastive Learning**: Constructs positive and negative pairs through semantic similarity ranking for contrastive training.
3. **Multi-level Correction Loss**: Conducts fine-grained corrections from visual, spatial, and quantitative perspectives.

### Key Designs

#### 1. Automatic Component Extraction Pipeline

Based on the CM-Fashion dataset, open-world detection and segmentation models are utilized to extract component-level information:

**Process Steps**:
1. **Component Detection**: GroundingDINO is used to obtain bounding boxes of target components from garment images.
2. **Quantitative Statistics**: The quantity of each component type is determined by counting the bounding boxes.
3. **Position Localization**: The geometric centers of the boxes are calculated as the spatial locations of components.
4. **Component Segmentation**:
    - A garment parsing model is first used for preliminary segmentation.
    - SAM is then combined with bounding boxes to enhance the segmentation (handling components missed in the preliminary segmentation).
5. **Text Augmentation**: The extracted quantitative and spatial information is aligned with the original text descriptions to enrich annotations.

Ultimate Output: Each garment image is equipped with a detailed description + component segmentation mask + component positions + component quantities.

#### 2. Retrieval-augmented Contrastive Learning

To alleviate the scale limitations of garment datasets, retrieval augmentation is introduced to expand training samples.

**Semantic Similarity Ranking**: For a sample pair $(x, y)$, the similarity score of the $i$-th component is:
$$S(x, y, i) = \frac{1}{|q_i^x - q_i^y| + Jaro(t_i^x, t_i^y)}$$

Where $q_i$ represents the component quantity, $t_i$ represents the component text description, and $Jaro$ is the Jaro string distance.

The overall similarity is summed across all $k$ components and penalized by the full-sentence similarity:
$$S(x, y) = \sum_{i=1}^k S(x, y, i) - \alpha \cdot Jaro(t_x, t_y)$$

**Positive and Negative Sample Construction**:
- Retrieve samples by similarity ranking within a random subset of $N$ samples.
- High similarity + high aesthetic/human preference score $\rightarrow$ positive sample.
- Low similarity + low aesthetic/preference score $\rightarrow$ negative sample.
- Each sample is expanded into $N_p \times N_n$ sample pairs.

**Contrastive Loss**:
$$\mathcal{L}_{RACL} = \|\hat{x} - x_p\|^2 + 1 - \|\hat{x} - x_n\|^2$$

Minimize the distance between the generated result and positive samples, and maximize the distance from negative samples.

#### 3. Multi-level Correction Loss

Three component-level correction losses enhance fine-grained alignment from different perspectives:

**Visual Correction** (Text-Image alignment):
$$\mathcal{L}_{visual} = \sum_{i=1}^k \frac{1}{CLIPScore(m_i \odot \hat{X}, t_i)}$$

Using the component mask $m_i$ (from ground truth) to crop the component region in the generated image, the CLIP Score with the component description $t_i$ is computed as a reward function.

**Spatial Correction** (Component Position Alignment):
$$\mathcal{L}_{spatial} = \sum_{i=1}^k \sum_{j=1}^l \|a_i^j - I_j(m_i)\|^2$$

Extract the spatial attention map $A_i$ corresponding to the component description from cross-attention, and align it with the ground-truth component mask $m_i$ using MSE.

**Quantitative Correction** (Component Count Alignment):
$$\mathcal{L}_{quantitative} = \sum_{i=1}^k |q_i - \hat{q}_i|$$

A component detector (GroundingDINO) is used to detect the number of components $\hat{q}_i$ in the generated result, which is compared with the ground-truth quantity $q_i$.

### Loss & Training

**Total Loss Function**:
$$\mathcal{L} = \omega_v \cdot \mathcal{L}_{visual} + \omega_s \cdot \mathcal{L}_{spatial} + \omega_q \cdot \mathcal{L}_{quantitative} + \omega_r \cdot \mathcal{L}_{RACL}$$

**Training Configurations**:
- Base Model: SD v2.1
- Prediction Type: Hybrid prediction (noise + image), replacing pure noise prediction
- Hardware: 8x Tesla V100, batch size 32
- Learning Rate: $1 \times 10^{-6}$
- Training Duration: 40 epochs, approx. 70 hours
- Dataset: CM-Fashion (500,000 512×512 garment images + descriptions)

## Key Experimental Results

### Main Results

Quantitative comparison with various baselines (on the CM-Fashion dataset):

| Method | FID ↓ | CLIPScore ↑ | AestheticScore ↑ | HPSv2 ↑ |
|------|-------|------------|-----------------|---------|
| DALL·E | 13.249 | 0.6423 | 4.8592 | 0.2137 |
| ARMANI | 12.336 | 0.6988 | 5.3585 | 0.2237 |
| SD v1.5 | 9.368 | 0.8911 | 5.2807 | 0.2419 |
| SD v2.1 | 9.157 | 0.8818 | 5.3881 | 0.2426 |
| DiffCloth |..9.201 | 0.8974 | 5.3957 | 0.2440 |
| SDXL | 9.091 | 0.8756 | 5.4299 | 0.2450 |
| **GarmentAligner** | **8.735** | **0.9245** | **5.8776** | **0.2648** |

GarmentAligner achieves the best performance across all metrics, reducing FID to 8.735 and improving CLIPScore to 0.9245.

**Component-level Accuracy** (1000 descriptions × 100 images):
- Quantitative Accuracy: GarmentAligner outperforms other methods by **20–45%**.
- Spatial Accuracy: Demonstrates a significant lead as well.

User study (110 participants): GarmentAligner achieves a preference rate of over **28%**.

### Ablation Study

Contribution analysis of each component:

| Variant | FID ↓ | CLIPScore ↑ | AestheticScore ↑ | HPSv2 ↑ |
|------|-------|------------|-----------------|---------|
| [V] Visual Correction | 8.975 | 0.9136 | 5.4081 | 0.2459 |
| [S] Spatial Correction | 9.143 | 0.8976 | 5.4003 | 0.2447 |
| [C] Quantitative Correction | 9.091 | 0.8840 | 5.3912 | 0.2433 |
| [V+S+C] Three Corrections | 8.924 | 0.9183 | 5.4190 | 0.2462 |
| [R] Retrieval Contrastive | 8.802 | 0.8984 | 5.7443 | 0.2639 |
| **[V+S+C+R] Full** | **8.735** | **0.9245** | **5.8776** | **0.2648** |

### Key Findings

1. **Retrieval-augmented contrastive learning contributes the most**: It yields the most significant improvements in FID, aesthetic score, and HPSv2, mainly enhancing image realism and overall quality.
2. **Multi-level corrections mainly improve text-to-image consistency**: Their contribution is most notable for CLIPScore.
3. **Mutual complementarity among components**: Retrieval-augmented contrastive learning improves global perception, while multi-level corrections enhance fine-grained details, resulting in a stackable effect when combined.
4. **Quantitative alignment is more difficult than spatial alignment**: In the ablation study, the independent CLIPScore of [C] is the lowest, indicating that quantitative alignment is the most challenging.
5. **Effective change in prediction type**: Hybrid prediction (noise + image) improves generation quality compared to pure noise prediction.

## Highlights & Insights

1. **Precise Problem Definition**: Focuses on the "quantity + location + relationship" triple alignment of garment components, which are fine-grained dimensions overlooked by previous works.
2. **Transferable Automated Pipeline**: The component extraction pipeline, based on GroundingDINO + SAM, can be applied to any garment dataset.
3. **Retrieval Augmentation Resolves Data Scarcity**: Positive and negative samples are constructed via component-level similarity retrieval, effectively leveraging limited data.
4. **Collaborative Multi-Loss Design**: Corrections across three orthogonal dimensions—visual (CLIP feedback), spatial (attention map alignment), and quantitative (detector counting)—cover the major failure modes of garment generation.

## Limitations & Future Work

1. **Dependency on Extraction Pipeline Accuracy**: Component information entirely relies on the accuracy of GroundingDINO and SAM, which inevitably contains errors in large-scale data.
2. **Pre-trained Model Bias**: Inherits the inherent biases of SD models, which may result in a lack of robustness and user-friendliness in outputs.
3. **Training Cost**: Requires 70 hours of training on 8 GPUs, and requires running the component extraction pipeline beforehand.
4. **Targeted Only at Single Garment Items**: CM-Fashion is a dataset of single garment items; coordination-based generation or try-on scenarios are not addressed.
5. **Gradient Issues in Quantitative Correction**: The count output from the detector is discrete, which may lead to discontinuous gradient propagation.

## Related Work & Insights

- **ARMANI / DiffCloth**: Prior garment generation methods that use parsed descriptions and semantic segmentation but neglect localization and quantity.
- **Attend-and-Excite**: An attention modulation method. GarmentAligner uses a similar attention-map guiding concept in its spatial correction.
- **CLIP Feedback for Generation**: Using CLIP Score as a training signal for generation quality feedback can be generalized to other fine-grained generation tasks.
- **Insight**: The training strategy of retrieval-augmented + contrastive learning could be equally effective in other domains rich in structured information (e.g., architectural design, mechanical drafting).

## Rating

- **Novelty**: ★★★★☆ — Strong originality in combining multi-level correction and retrieval-augmented contrastive learning; the component extraction pipeline is highly practical.
- **Value**: ★★★★☆ — Directly addresses pain points in commercial scenarios, although the lack of open-source code limits reproducibility.
- **Experimental Thoroughness**: ★★★★☆ — Multi-dimensional metrics + user studies + detailed ablation, with an innovative evaluation of quantitative accuracy.
- **Writing Quality**: ★★★★☆ — Clearly structured with good visual comparative results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](../../AAAI2026/image_generation/truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[ECCV 2024\] MultiGen: Zero-Shot Image Generation from Multi-modal Prompts](multigen_zero-shot_image_generation_from_multi-modal_prompts.md)
- [\[ECCV 2024\] Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](be_yourself_bounded_attention_for_multisubject_texttoimage_g.md)

</div>

<!-- RELATED:END -->
