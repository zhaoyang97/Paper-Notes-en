---
title: >-
  [Paper Note] Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching
description: >-
  [ECCV 2024][Object Detection][drone navigation] Constructed the first natural language-guided drone geolocalization benchmark GeoText-1652 (276K bbox-text pairs, 316K descriptions), and proposed a blending spatial matching method that achieves region-level spatial relation matching via grounding loss + spatial relation loss, achieving a text retrieval Recall@10 of 31.2%.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "drone navigation"
  - "geolocalization"
  - "spatial relation matching"
  - "vision-language"
  - "benchmark"
date: 2026-05-08
content_hash: 75a30b93ea5cccc5
---

# Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching

**Conference**: ECCV 2024  
**arXiv**: [2311.12751](https://arxiv.org/abs/2311.12751)  
**Code**: [https://multimodalgeo.github.io/GeoText/](https://multimodalgeo.github.io/GeoText/)  
**Area**: Object Detection / Vision-Language / Drone Navigation  
**Keywords**: drone navigation, geolocalization, spatial relation matching, vision-language, benchmark

## TL;DR
Constructed the first natural language-guided drone geolocalization benchmark GeoText-1652 (276K bbox-text pairs, 316K descriptions), and proposed a blending spatial matching method that achieves region-level spatial relation matching via grounding loss + spatial relation loss, achieving a text retrieval Recall@10 of 31.2%.

## Background & Motivation

**Background**: Drone navigation primarily relies on image matching (cross-view geolocalization), with datasets like University-1652 providing drone-satellite image pairs. However, in real-world scenarios, a more natural user input is natural language descriptions rather than query images.

**Limitations of Prior Work**: (1) Lack of public natural language-guided drone navigation datasets—existing geolocalization datasets only provide GPS labels without text descriptions. (2) In aerial scenes from a drone perspective, buildings exhibit highly similar appearances (multiple similar buildings may exist in adjacent areas). Distinguishing them purely based on visual features and object descriptions is challenging, necessitating spatial relation information.

**Key Challenge**: Traditional cross-modal matching methods (such as CLIP, ALBEF) do not model spatial relations between regions, whereas spatial positions ("top-left", "bottom-right") in aerial scenes are key information for distinguishing similar targets.

**Goal**: Establish the foundation for language-guided drone navigation, including both datasets and methods.

**Key Insight**: (1) Construct a high-quality image-text-bbox dataset semi-automatically utilizing a VLM + human-in-the-loop annotation pipeline; (2) Introduce two optimization objectives, grounding and spatial relation, into the cross-modal matching framework.

**Core Idea**: Through region-level spatial relation description and matching, natural language can accurately guide the drone to localize the target building.

## Method

### Overall Architecture
The framework consists of three components: an image encoder (Swin), a text encoder (BERT), and a cross-modal encoder. After encoding, image features are processed via ROI Pooling to extract region features, while text is encoded separately into image-level descriptions and region-level descriptions. The framework simultaneously optimizes four losses: Image-Text Contrastive (ITC), Image-Text Matching (ITM), Grounding Loss, and Spatial Loss.

### Key Designs

1. **GeoText-1652 Dataset Construction**

    - **Function**: Extend the University-1652 image dataset by adding fine-grained text-bbox annotations
    - **Mechanism**: Two-stage human-in-the-loop annotation pipeline:
        - **Modality Expansion Stage**: Use Visual-LLMs to generate image-level and region-level descriptions, combined with a referee model for automatic filtering (positive keyword checking + negative subjective expression exclusion), where humans only need to review the keyword list
        - **Spatial Refinement Stage**: Use a pre-trained visual grounding model to generate bboxes based on region descriptions, set spatial rules to filter incorrect locations, and iteratively optimize through 5 rounds of human evaluation, with ultimately >90% of annotations rated as excellent
    - **Design Motivation**: Pure manual annotation is too costly, and Visual-LLMs suffer from hallucination issues, necessitating a hybrid scheme of a referee model + human verification
    - **Data Scale**: Training set contains 37,854 drone images + 701 satellite images + 11,663 ground images; each image has an average of 3 global descriptions and 2.62 bbox-text pairs; global descriptions average 70.23 words

2. **Image-Text Contrastive Learning (ITC)**

    - **Function**: Global image-text alignment
    - **Mechanism**: Standard in-batch contrastive learning:
    $$\mathcal{L}_{\text{itc}} = -\frac{1}{2}\mathbb{E}[\log(\boldsymbol{p}_{\text{t2v}}) + \log(\boldsymbol{p}_{\text{v2t}})]$$
      where $\boldsymbol{p}_{\text{v2t}} = \frac{\exp(s(V,T)/\tau)}{\sum_i \exp(s(V,T^i)/\tau)}$
    - **Design Motivation**: Establish a foundation for global semantic alignment between images and text descriptions

3. **Image-Text Matching (ITM)**

    - **Function**: Determine whether an image-text pair matches
    - **Mechanism**: Sample hard negative text with the highest in-batch similarity for each visual concept, and predict the matching probability using the cross-modal encoder:
    $$\mathcal{L}_{\text{itm}} = -\mathbb{E}[\boldsymbol{y_m}\log(\boldsymbol{p}_{\text{match}}) + (1-\boldsymbol{y_m})\log(1-\boldsymbol{p}_{\text{match}})]$$
    - **Design Motivation**: Complement the coarse-grained alignment of ITC and provide fine-grained matching discrimination capabilities

4. **Grounding Prediction Loss**

    - **Function**: Predict the corresponding bounding box based on region-level text descriptions
    - **Mechanism**: Predict the normalized bbox $\hat{\boldsymbol{b}}_j = (c_x, c_y, w, h)$ using a 6-layer Transformer + MLP, with the training loss:
    $$\mathcal{L}_{\text{grounding}} = \mathbb{E}[\mathcal{L}_{\text{iou}}(\boldsymbol{b}_j, \hat{\boldsymbol{b}}_j) + \|\boldsymbol{b}_j - \hat{\boldsymbol{b}}_j\|_1]$$
    - **Design Motivation**: Model the precise spatial correspondence between text descriptions and image regions, serving as the foundation for spatial relation matching

5. **Spatial Relation Matching Loss**

    - **Function**: Predict the relative spatial relations among multiple ROIs
    - **Mechanism**: Given multiple bboxes (e.g., $b_1, b_2, b_3$), region features $R_i$ are extracted via ROI Pooling and concatenated into pairwise features $R_{ij}$ ($i \neq j$). An MLP is used to predict 9 types of spatial relations (3 horizontal × 3 vertical: left/middle/right × top/middle/bottom):
    $$\mathcal{L}_{\text{spatial}} = \mathbb{E}[-\boldsymbol{y_r}^{ij}\log(\hat{\boldsymbol{p_r}}^{ij})]$$
      Horizontal relation determination: $|\Delta x| < w/2$ → middle; $\Delta x > w/2$ → left; $\Delta x < -w/2$ → right
    - **Design Motivation**: Individual grounding loss only focuses on the absolute localization of a single region, failing to model relative positional relationships between regions. In aerial scenarios, relative location descriptions such as "the main building is on the left" are key to distinguishing similar buildings

6. **Overall Optimization Objective**

    - **Function**: Integrate all losses
    - **Mechanism**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{itc}} + \mathcal{L}_{\text{itm}} + \lambda(\mathcal{L}_{\text{grounding}} + \mathcal{L}_{\text{spatial}})$, where $\lambda = 0.1$
    - **Design Motivation**: Blending spatial matching receives a smaller weight, serving as a complement to semantic matching rather than a replacement

### Loss & Training
- Backbone: XVLM pre-trained on 16M images
- Image encoder: Swin; Text encoder: BERT
- All images are resized to 384×384, with a patch size of 32
- Random rotation or horizontal flipping is not used (as it destroys spatial information)
- AdamW optimizer, learning rate $3e^{-5}$, weight decay 0.01

## Key Experimental Results

### Main Results

| Method | Params | Text→Image R@1 | R@5 | R@10 | Image→Text R@1 | R@5 | R@10 |
|------|--------|---------------|------|------|----------------|------|------|
| UNITER | 300M | 0.9 | 2.7 | 4.2 | 2.5 | 7.4 | 11.8 |
| ALBEF (14M) | 210M | 1.1 | 3.5 | 5.3 | 3.0 | 9.1 | 14.2 |
| XVLM (16M) | 216M | 4.5 | 9.9 | 13.4 | 5.0 | 14.4 | 21.4 |
| XVLM_finetuned | 216M | 13.2 | 23.7 | 29.6 | 25.0 | 52.3 | 65.1 |
| **Ours** | **217M** | **13.6** | **24.6** | **31.2** | **26.3** | **53.7** | **66.9** |

### Ablation Study

| Method | Text→Image R@1 | R@10 | Image→Text R@1 | R@10 |
|------|---------------|------|----------------|------|
| Baseline (XVLM_finetuned) | 13.2 | 29.6 | 25.0 | 65.1 |
| + grounding loss only | 13.5 | 30.9 | 25.9 | 66.3 |
| + spatial loss only | 13.4 | 30.1 | 25.3 | 65.6 |
| **+ Both (Ours)** | **13.6** | **31.2** | **26.3** | **66.9** |

### Training Data Analysis

| Training Set | Image Count | Text→Image R@1 | Image→Text R@1 |
|--------|--------|---------------|----------------|
| Drone only | 37,854 | 12.9 | 25.7 |
| Satellite + Ground | 12,364 | 10.1 | 18.7 |
| **All (Sat+Drone+Ground)** | **50,218** | **13.6** | **26.3** |

### Key Findings
- Pre-trained models perform poorly on aerial datasets (XVLM without fine-tuning achieves only 4.5% R@1), indicating a huge domain gap between the aerial domain and general data
- Performance significantly improves after fine-tuning (XVLM: 4.5% → 13.2% R@1), validating the value of the GeoText-1652 dataset
- Grounding loss is the primary contributing factor, while spatial loss acts as a complement to provide stacked improvements
- The model is robust to small-angle rotations (15°), and the performance degradation under large-angle rotations (90°/270°) is acceptable
- Generalizes well on real-world drone-view images

## Highlights & Insights
- **Relative position is a key distinguishing signal in aerial scenes**: When multiple similar buildings appear, describing the building appearance alone is insufficient for localization. Descriptions of spatial relations, such as "the parking lot on the left" or "the tower on the top-right", are the keys to differentiation. Although simple, the 9-class directional classification design is effectively sufficient.
- **Practicality of the human-in-the-loop annotation paradigm**: The pipeline consisting of a referee model + Visual-LLM + human review maintains quality (>90% excellent) while reducing annotation costs. This semi-automatic annotation framework can be extended to the construction of other fine-grained VL datasets.

## Limitations & Future Work
- Recall@1 is only 13.6%, which is a relatively low absolute performance and still needs significant improvement for practical navigation applications.
- The 9-class spatial relation classification (3x3 grid) is too coarse to represent fine-grained spatial relations like "immediately adjacent to" or "far away from".
- The dataset only covers university campus scenes, and its generalization capability to diverse scenes such as urban or natural environments remains unverified.
- Only 2D spatial relations are considered, without utilizing 3D information such as building heights.

## Related Work & Insights
- **vs University-1652 [Zheng et al.]**: GeoText-1652 adds text-bbox annotations on top of its images, extending the task from image retrieval to natural language-guided localization.
- **vs CLIP/ALBEF/XVLM**: General VL models perform poorly in the aerial domain, emphasizing the necessity of domain-specific data.
- **vs GeoDTR [Zhang et al.]**: GeoDTR performs vision-enhanced cross-view localization, whereas this work performs natural language-guided localization, complementing each other.
- **vs VLN (Vision-Language Navigation)**: VLN operates in indoor/street-view environments, whereas this work focuses on the under-explored domain of aerial drone navigation.

## Rating
- Novelty: ⭐⭐⭐⭐ The first natural language-guided drone localization benchmark; the spatial relation matching design is intuitive and reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison with multiple baselines, ablation analysis, training set analysis, rotation robustness, and real-world scene generalization.
- Writing Quality: ⭐⭐⭐⭐ Detailed description of the dataset construction pipeline; the method section is structured and clear.
- Value: ⭐⭐⭐⭐ The dataset has lasting value for the community, although there remains substantial room for absolute performance improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ReGround: Improving Textual and Spatial Grounding at No Cost](reground_improving_textual_and_spatial_grounding_at_no_cost.md)
- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](../../CVPR2025/object_detection/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[ECCV 2024\] Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography](tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)
- [\[ECCV 2024\] LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction](lami-detr_open-vocabulary_detection_with_language_model_instruction.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)

</div>

<!-- RELATED:END -->
