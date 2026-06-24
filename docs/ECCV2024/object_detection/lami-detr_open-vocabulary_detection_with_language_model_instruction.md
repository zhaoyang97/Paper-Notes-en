---
title: >-
  [Paper Note] LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction
description: >-
  [ECCV 2024][Object Detection][Open-Vocabulary Object Detection] LaMI-DETR is proposed to address two core challenges in open-vocabulary object detection—insufficient concept representation and base-category overfitting—by leveraging GPT to generate visual concept descriptions and T5 to mine inter-category visual similarity relationships. It outperforms previous state-of-the-art methods by 7.8 rare AP on OV-LVIS, achieving 43.4 AP_rare.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Open-Vocabulary Object Detection"
  - "DETR"
  - "Language Model Instruction"
  - "Inter-category Relationships"
  - "CLIP"
date: 2026-05-08
content_hash: c60976f7abe94f31
---

# LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction

**Conference**: ECCV 2024  
**arXiv**: [2407.11335](https://arxiv.org/abs/2407.11335)  
**Code**: [Yes (GitHub)](https://github.com/eternaldolphin/LaMI-DETR)  
**Area**: Object Detection  
**Keywords**: Open-Vocabulary Object Detection, DETR, Language Model Instruction, Inter-category Relationships, CLIP

## TL;DR

LaMI-DETR is proposed to address two core challenges in open-vocabulary object detection—insufficient concept representation and base-category overfitting—by leveraging GPT to generate visual concept descriptions and T5 to mine inter-category visual similarity relationships. It outperforms previous state-of-the-art methods by 7.8 rare AP on OV-LVIS, achieving 43.4 AP_rare.

## Background & Motivation

Open-vocabulary object detection (OVOD) aims to detect novel categories that are unseen during training. Existing methods leverage the zero-shot capabilities of vision-language models (VLMs) like CLIP to handle novel classes, but they face two core challenges:

### Challenge 1: Insufficient Concept Representation

Existing methods typically encode category **names** directly using the CLIP text encoder to serve as concept representations. However, this approach has two major limitations:

**Lack of textual semantic knowledge**: CLIP's text encoder focuses more on the similarity of character combinations rather than semantic relationships. For instance, in clustering analysis, "fireboat" and "fireweed" are clustered together due to their character/spelling similarity, which is highly unreasonable. In contrast, language models like T5 can better understand hierarchical semantic relationships.

**Lack of visual feature information**: Using only category names fails to capture visual similarity. For example, "sea-lion" and "dugong" are visually highly similar, but due to their vastly different names, representations based solely on names would place them into different clusters.

### Challenge 2: Base-category Overfitting

Although CLIP's image encoder possesses strong zero-shot recognition capabilities, only annotated data from base categories is available during detector training. This causes the model to gradually overfit to the base categories, leading novel objects to be easily misclassified as background or base categories. This problem fundamentally arises because, during training, the model only learns the foreground-background distinction for base categories, lacking constraints for generalizing to novel classes.

**Key Insight**: The authors find that **inter-category relationships** are the key to solving the above two issues. By modeling the visual similarity among categories, one can enrich concept representations, and simultaneously, avoid base-category overfitting through a sampling strategy. This motivates the proposed LaMI (Language Model Instruction) strategy.

## Method

### Overall Architecture

LaMI-DETR is built upon the DINO-DETR architecture, utilizing a frozen CLIP ConvNext-L as the visual backbone, and replacing the final classification layer with CLIP text embeddings. The overall pipeline is: the CLIP image encoder extracts feature maps $\rightarrow$ the Transformer Encoder enhances the features $\rightarrow$ the Transformer Decoder generates query features $\rightarrow$ the Bounding Box module predicts locations and classifications. During inference, calibration is performed by combining the VLM score and the detection score:

$$S_c^{cal} = \begin{cases} (S_c^{vlm})^\alpha \cdot (S_c^{det})^{(1-\alpha)} & \text{if } c \in \mathcal{C}_B \\ (S_c^{vlm})^\beta \cdot (S_c^{det})^{(1-\beta)} & \text{if } c \in \mathcal{C}_N \end{cases}$$

Compared to other OV-DETR methods such as CORA and EdaDet, LaMI-DETR is simpler: it relies on a single backbone, maintains an end-to-end structure (requiring no NMS), and only needs a single RoI-Align pooling during inference.

### Key Designs

1. **Inter-category Relationships Extraction**: Combines the knowledge of GPT-3.5 with the discriminative semantic space of T5 to construct visual concepts and measure inter-category relationships.

   Core pipeline:
    - Utilize GPT-3.5 to generate **visual descriptions** $d$ (including shape, color, size, and other visual attributes) for each category $c \in \mathcal{C}$, translating category names into visual concepts.
    - Use T5 (Instructor Embedding) to encode the visual descriptions is to embeddings $e \in \mathcal{E}$.
    - Perform K-Means clustering on the visual description embeddings $\mathcal{E}$ to obtain $K$ centroids, where categories within the same cluster are considered visually similar.

   Design Motivation: The CLIP text encoder is sensitive to character combinations but lacks deep semantic understanding; T5 has a superior text semantic space but lacks visual information. The visual descriptions generated by GPT compensate for the lack of visual information. The combination of all three achieves dual alignment of textual semantics and visual attributes.

2. **Visual Concept Sampling**: Employs clustering results to mitigate base-category overfitting.

   Utilizing the Federated Loss framework, $C_{fed}$ categories are randomly sampled in each minibatch to compute the loss. The key improvement is: excluding categories that are **visually similar** to the current ground-truth (GT) category, sampling only "easy negative" categories with large visual differences:

    $$p_c^{cal} = \begin{cases} 0 & \text{if } c \in \mathcal{C}_g \\ p_c & \text{if } c \notin \mathcal{C}_g \end{cases}$$

   Where $\mathcal{C}_g$ represents all categories in the same cluster as the GT category. This forces the detector to learn more generic foreground features rather than overfitting to specific base-category features.

3. **Language Embedding Fusion**: Fuses CLIP text embeddings into object queries to boost classification accuracy.

   After the Transformer Encoder, the top-N queries with the highest scores are selected and combined with their nearest text embeddings via element-wise addition:
    $$\{q_j\}_{j=1}^N = \{q_j \oplus t'_j\}_{j=1}^N$$
   where $t'_j$ is the CLIP text embedding updated by the visual descriptions.

4. **Confusing Category Distinction**: Refines VLM scores during inference to distinguish visually similar but semantically different categories.

   For each inference category $c$, the most similar category $c^{conf}$ in the CLIP text space is identified. The GPT prompt is modified so that the visual description emphasizes the discriminative features between $c$ and $c^{conf}$. The updated text embeddings then replace the original VLM classification weights to enhance differentiation between confusing categories.

### Loss & Training

- Employs standard detection loss (classification + regression) based on the DINO-DETR framework, where classification weights use CLIP text embeddings updated by visual descriptions.
- Applies Federated Loss for random category sampling (sampling 100 classes for OV-LVIS, and 700 classes for VG-dedup).
- Adopts an EMA strategy to enhance training stability, and uses Repeat Factor Sampling to balance the long-tail distribution.
- The CLIP image encoder is frozen throughout, training only the Encoder, Decoder, and BBox modules.
- Multiplies novel category logits by a factor of 5.0 during inference to compensate for the score discrepancy between base and novel categories.

## Key Experimental Results

### Main Results

**OV-LVIS Benchmark (Open-vocabulary Detection):**

| Method | Backbone | Params | Extra Image Data | AP_rare | mAP |
|------|--------|-----------|-------------|---------|-----|
| ViLD | ViT-B/32 + R-50 | 26M | ✗ | 16.7 | 27.8 |
| F-VLM | R-50x64 | 420M | ✗ | 32.8 | 34.9 |
| OWL-ViT | ViT-L/14 | 306M | ✗ | 25.6 | 34.7 |
| CFM-ViT | ViT-L/16 | 303M | ALIGN | 35.6 | 38.5 |
| **LaMI-DETR** | **ConvNext-L** | **196M** | **✗** | **43.4** | **41.3** |

With a smaller backbone (196M vs. 303M) and without using extra image-level data, LaMI-DETR achieves an AP_rare of 43.4, surpassing CFM-ViT by 7.8 points.

**Cross-dataset Transfer (OV-LVIS $\rightarrow$ COCO / Objects365):**

| Method | Backbone | COCO AP | Objects365 AP |
|------|--------|---------|---------------|
| BARON | RN50 | 36.2 | 13.6 |
| CoDet | EVA02-L | 39.1 | 14.2 |
| CFM | ViT-L/16 | - | 18.7 |
| **LaMI-DETR** | **ConvNext-L** | **42.8** | **21.9** |

### Ablation Study

**Effect of LaMI Components (OV-LVIS AP_rare):**

| Configuration | AP_rare | Note |
|------|---------|------|
| Federated Loss only | 32.2 | Baseline |
| + Language Embedding Fusion | 33.0 | +0.8, assisted by text embedding |
| + Visual Concepts Sampling | 40.1 | +7.1, clustering sampling brings huge boost |
| + Embedding Update | 42.5 | +2.4, visual descriptions update classification weights |
| + Confusing Category | **43.4** | +0.9, confusing category distinction |

**Ablation on Clustering Strategy:**

| Clustering Encoder | Clustering Text | AP_rare | Note |
|-----------|---------|---------|------|
| No clustering | - | 33.0 | Baseline |
| CLIP Text Encoder | name | 33.5 | +0.5, poor CLIP semantic space |
| T5 | name | 34.1 | +1.1, T5 semantics are better |
| T5 | name + definition | 31.5 | -1.5, definitions are counterproductive |
| T5 | **name + visual desc.** | **40.1** | **+7.1, visual descriptions are key** |

### Key Findings

1. **Visual Concept Sampling contributes the most** (+7.1 AP_rare), validating that sampling negative classes using inter-category visual relationships is the most effective strategy to alleviate overfitting.
2. **Visual descriptions are more effective than definitions**: Visual attribute descriptions generated by GPT reflect inter-category visual relationships much better in clustering than encyclopedic definitions.
3. **T5 is more suitable for concept clustering than CLIP**: The semantic space of T5 is better at distinguishing categories with similar characters but different semantics.
4. The inference speed is 4.5 FPS (V100), superior to GLIP (0.12 FPS) and GroundingDINO (1.5 FPS).

## Highlights & Insights

- **Systematic Utilization of Inter-category Relationships**: Combines GPT-generated HTML/visual descriptions, T5's discriminative embeddings, and K-Means clustering for the first time to build a complete pipeline for extracting and utilizing inter-category visual relationships.
- **Ingenious Negative Sampling Strategy**: Upgrades Federated Loss from simple category sampling to semantic-aware sampling by excluding categories visually similar to the ground truth. The core idea is "do not learn indistinguishable negatives; let CLIP differentiate them."
- **Handling Confusing Categories during Inference**: Leverages contrastive prompts (e.g., "The difference between A and B is...") to generate discriminative visual descriptions, further boosting VLM classification accuracy.
- **Elegant Architecture**: Unlike CORA and EdaDet, it requires no decoupled classification/regression or multiple RoI-Align operations, preserving the end-to-end advantages of DETR.

## Limitations & Future Work

- Currently only CLIP ConvNext-L is used as the visual backbone; the effectiveness of other architectures like ViTs remains unexplored.
- GPT API calls introduce additional costs and latency (although only required once offline).
- The number of clusters $K$ is a hyperparameter (e.g., 128 for OV-LVIS, 256 for VG-dedup), which needs tuning for different datasets.
- The logit scaling factor of 5.0 for novel class inference is also empirical and lacks theoretical justification.
- The construction of the 26K+ visual concept dictionary relies on WordNet, which might miss emerging concepts.

## Related Work & Insights

- Unlike the concept enrichment strategy of DetCLIP, LaMI's visual descriptions focus more on visual attributes of objects rather than abstract definitions.
- Instructor Embedding (T5) outperforms the CLIP text encoder in measuring semantic similarity and is well-suited as a tool for extracting inter-category relationships.
- The core concept of the negative sampling strategy can be generalized to other large-vocabulary detection tasks that employ Federated Loss.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Visually innovative and practical, systematically utilizing LLMs to mine inter-category visual relationships for OVOD for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive experiments across OV-LVIS, zero-shot LVIS, cross-dataset transfer, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Intuitive illustrations with a clear mapping between problems and methods, though some formulas present heavy notation.
- **Value**: ⭐⭐⭐⭐⭐ — An AP_rare improvement of 7.8 points represents a significant breakthrough in open-vocabulary detection, offering high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DeCo-DETR: Decoupled Cognition DETR for efficient Open-Vocabulary Object Detection](../../ICLR2026/object_detection/deco-detr_decoupled_cognition_detr_for_efficient_open-vocabulary_object_detectio.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](../../CVPR2025/object_detection/mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2026\] WeDetect: Fast Open-Vocabulary Object Detection as Retrieval](../../CVPR2026/object_detection/wedetect_fast_open-vocabulary_object_detection_as_retrieval.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)
- [\[CVPR 2026\] Thermal-Det: Language-Guided Cross-Modal Distillation for Open-Vocabulary Thermal Object Detection](../../CVPR2026/object_detection/thermal-det_language-guided_cross-modal_distillation_for_open-vocabulary_thermal.md)

</div>

<!-- RELATED:END -->
