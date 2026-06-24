---
title: >-
  [Paper Note] Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification
description: >-
  [ECCV 2024][Human Understanding][Wildlife ReID] An Adaptive High-frequency Transformer (AdaFreq) is proposed. By employing frequency-domain mixup augmentation, target-aware dynamic selection of high-frequency tokens, and a feature equilibrium loss, it unifies high-frequency information (such as fur texture and contour edges) for the re-identification of diverse wildlife, outperforming existing ReID methods across 8 cross-species datasets.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Wildlife ReID"
  - "High-frequency Information"
  - "Transformer"
  - "Frequency-domain Data Augmentation"
  - "Adaptive Selection"
date: 2026-05-08
content_hash: 4bf942e5b5823b15
---

# Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification

**Conference**: ECCV 2024  
**arXiv**: [2410.06977](https://arxiv.org/abs/2410.06977)  
**Code**: [https://github.com/JigglypuffStitch/AdaFreq.git](https://github.com/JigglypuffStitch/AdaFreq.git)  
**Area**: Image Retrieval / Re-Identification (ReID) / Wildlife Identification  
**Keywords**: Wildlife ReID, High-frequency Information, Transformer, Frequency-domain Data Augmentation, Adaptive Selection  

## TL;DR
An Adaptive High-frequency Transformer (AdaFreq) is proposed. By employing frequency-domain mixup augmentation, target-aware dynamic selection of high-frequency tokens, and a feature equilibrium loss, it unifies high-frequency information (such as fur texture and contour edges) for the re-identification of diverse wildlife, outperforming existing ReID methods across 8 cross-species datasets.

## Background & Motivation
Wildlife Re-identification (Wildlife ReID) requires distinguishing **different individuals within the same species**, which is far more challenging than species classification. Existing methods are either tailored for a single species (e.g., whale fluke identification, seal stripe matching), lacking cross-species generalizability, or directly apply pedestrian ReID techniques, ignoring the unique challenges of wildlife—specifically, the lack of exploitable appearance differences like clothing/hairstyles, and complex natural background environments. The paper observes that individual distinguishing features across different species (fur textures, patterns, and contour shapes) find a unified representation in **image high-frequency information**, which serves as an entry point for building a general framework.

## Core Problem
How to construct a **cross-species general** wildlife ReID framework? There are two core challenges: (1) Discriminative characteristics of different species vary significantly (e.g., stripes for tigers, contours for elephants, body patterns for sharks), requiring a unified feature representation. (2) High-frequency information in natural environments contains substantial background noise (leaf textures, grass, etc.), and directly enhancing high-frequency components might conversely degrade performance.

## Method

### Overall Architecture
Input original image $\rightarrow$ ViT backbone extracts visual features. Simultaneously, the image undergoes Fourier Transform to extract high-frequency information, which is enhanced via frequency-domain mixup to obtain an enhanced high-frequency representation $\rightarrow$ the same ViT extracts high-frequency features. Using the class token attention maps from the last layer of the original branch, a subset of target-related high-frequency tokens is dynamically selected to filter out background noise. The two branches respectively output global features $c_o$ and $c_h$, each computing ID loss + Triplet loss, complemented by a feature equilibrium loss to prevent excessive discrepancy between the two branches. Only the original features are used during inference.

### Key Designs
1. **Frequency-Domain Mixup Augmentation (FMA)**: Apply FFT to the input image $\rightarrow$ extract high-frequency components $F_h(I)$ via Gaussian high-pass filtering $\rightarrow$ mix $F_h(I)$ with the original frequency-domain representation $F(I)$ using a random mask: $F'_h = (1-M_\alpha) \cdot F_h + M_\alpha \cdot F$, where $M_\alpha$ is a random square region mask (ratio 0 to 0.5) $\rightarrow$ apply IFFT to transform back to the spatial domain. This operation at the **frequency-domain level** avoids introducing redundant spatial information, simulates high-frequency instability caused by changes in illumination/pose, and enhances model robustness.

2. **Object-Aware Dynamic Selection (ODS)**: Directly utilizing all high-frequency patches introduces significant background noise. ODS uses the attention scores of the class token on each patch in ViT as a metric for "target relevance". In the last layer, the average score $\Psi^L$ across all attention heads is calculated, and the top $\mu \cdot n$ tokens ($\mu=0.5$) with highest scores are selected. Only these target-relevant high-frequency tokens are sent to the high-frequency branch. Key Insight: The class token naturally learns to focus on discriminative regions in ReID tasks, making it an effective guidance signal for object localization.

3. **Feature Equilibrium Loss**: Prevents the model from overfocusing on high-frequency details and losing original visual information. A Smooth L1 loss is used to constrain the feature distance of corresponding tokens in the two branches for the same input: $\mathcal{L}_F = \sum_{b,z} \|f^o_{b,z}, f^h_{b,z}\|$, ensuring that high-frequency features do not deviate too far from the original features.

### Loss & Training
- Overall Loss: $\mathcal{L}_{overall} = \mathcal{L}_{ID}(c_o) + \mathcal{L}_{tri}(c_o) + \mathcal{L}_{ID}(c_h) + \mathcal{L}_{tri}(c_h) + \lambda \mathcal{L}_F$, where $\lambda=0.1$.
- Backbone: ViT-B/16 (pretrained on ImageNet-1K), input size $256 \times 256$, patch size $16 \times 16$.
- SGD optimizer, lr=0.001 with cosine decay, 150 epochs, batch size 32 (8 IDs $\times$ 4 images).
- Data Augmentation: Random rotation of $15^\circ$, brightness/contrast adjustment with 50% probability each, padding 10px.
- Datasets are split into training/testing sets at a 70/30 ratio (no identity overlap), unifying experimental setups across multiple animal datasets.

## Key Experimental Results

| Dataset | Metric | Ours (AdaFreq) | TransReID | CLIP-ReID | Gain (vs Best) |
|--------|------|------|----------|------|------|
| Panda | mAP | **44.5** | 37.9 | 38.8 | +4.3 vs RotTrans |
| Elephant | mAP/R1 | **30.4/58.0** | 21.2/50.9 | 20.4/43.7 | +1.3/+3.9 |
| Seal | mAP/R1 | **51.5/87.4** | 50.1/86.0 | 45.2/84.1 | +1.4/+1.4 |
| Tiger | mAP/R1 | **66.3/98.5** | 64.1/98.3 | 55.8/96.1 | +0.2/+0.2 |
| Pigeon | mAP | **73.8** | 72.2 | 68.4 | +1.3 |
| Giraffe | mAP | **49.1** | 45.8 | 47.6 | +0.7 |
| Shark | mAP | **24.3** | 19.3 | 23.3 | +1.0 |

Multi-species training (Table 2): Seal mAP 50.6 (vs TransReID 45.8), Elephant mAP 26.6 (vs 22.8)

Domain generalization setting (Table 3, Wildlife-71 training $\rightarrow$ testing on unseen species): AVG mAP 48.1 vs UniReID 47.6, R1 88.5 vs 63.9 (**+24.6**)

### Ablation Study

| Strategy | Panda mAP | Pigeon mAP | Shark mAP |
|------|-----------|------------|-----------|
| Baseline (ViT) | 40.8 | 70.1 | 20.2 |
| Pure High-Frequency Aug | 41.8 | 68.4$\downarrow$ | 21.5 |
| PHA (Existing Method) | 38.8 | 70.7 | 14.8$\downarrow$ |
| +FMA | 42.7 | 70.9 | 21.7 |
| +ODS | 43.9 | 73+ | — |
| +All (incl. $\mathcal{L}_F$) | **44.5** | **73.8** | **24.3** |

- **Pure high-frequency augmentation drops by 1.7% on Pigeon**, indicating severe background noise interference $\rightarrow$ validating the necessity of ODS.
- PHA (CVPR2023) dropped significantly from 20.2 to 14.8 on Shark, as it amplifies uncertain local high-frequency features, leading to bias toward background noise.
- ODS contributes the most, followed by FMA, while the Feature Equilibrium Loss provides additional stability gains.
- $\mu=0.5$ yields the best results overall but varies across different datasets (due to different target ratios in Elephant/Shark); $\lambda=0.1$ is optimal.

## Highlights & Insights
- **High-frequency information as a unified bridge across species**: This observation is insightful—whether it is tiger stripes or elephant contours, they are uniformly represented in the high-frequency domain.
- **Frequency-domain level operations** avoid artifacts and redundant information introduced by spatial-domain mixing.
- **Leveraging class token attention for object localization** is a simple and elegant design that requires no extra annotations.
- **Unified experimental settings across multiple wildlife datasets**, providing a standardized benchmark for future research.
- In domain generalization experiments, R1 surges from 63.9 to 88.5, proving that high-frequency features possess robust cross-species transferability.

## Limitations & Future Work
- **Reliance on baseline attention quality**: Token selection in ODS relies entirely on the attention of the last layer of ViT. If the baseline attention is dispersed or incorrect, the selected high-frequency tokens will also contain noise.
- **$\mu$ requires dataset-specific tuning**: The proportions of objects in images vary significantly among species (e.g., elephants occupy the entire frame vs. birds occupy only a small portion); a fixed $\mu$ cannot adapt dynamically.
- **Only original features are used during testing**: Training the high-frequency branch but discarding it during inference is somewhat wasteful.
- **Lack of comparison with more modern baselines**: Such as new methods from 2024.
- Extensible directions: (1) Automatically learning $\mu$ instead of a fixed value; (2) Fusing features from both branches during inference; (3) Combining text descriptions (e.g., CLIP) to enhance cross-species generalization.

## Related Work & Insights
- **vs TransReID**: TransReID is a generic ViT-based ReID method that does not consider high-frequency information. AdaFreq significantly improves upon it, especially for species lacking obvious textures (Elephant +9.2 mAP).
- **vs CLIP-ReID**: CLIP-ReID leverages vision-language pretrained models to enhance descriptive information, but lacks capturing fine-grained visual differences (e.g., fur texture). AdaFreq directly reinforces these discriminative details through frequency-domain operations.
- **vs PHA (CVPR2023)**: PHA is a method to enhance high-frequency features in pedestrian ReID but ignores natural environment noise. In wildlife scenarios, PHA actually leads to a severe performance drop (Shark -5.4 mAP). AdaFreq's ODS strategy effectively addresses this issue.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of unifying multi-species ReID using high-frequency information is novel, and the three components are reasonably designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across 8 species + multi-species + domain generalization settings, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete mathematical derivations, and standard figures and tables.
- Value: ⭐⭐⭐⭐ Unifies the experimental setup for wildlife ReID, contributing significantly to this niche yet important field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification](multi-memory_matching_for_unsupervised_visible-infrared_person_re-identification.md)
- [\[CVPR 2026\] Spatial-Frequency Collaborative Learning for Occluded Visible-Infrared Person Re-Identification](../../CVPR2026/human_understanding/spatial-frequency_collaborative_learning_for_occluded_visible-infrared_person_re.md)
- [\[CVPR 2026\] SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging for Person Re-Identification](../../CVPR2026/human_understanding/ssm-aware_token-efficient_vmamba_via_adaptive_patch_pruning_and_merging_for_pers.md)
- [\[ICML 2026\] Learning Instance-Adaptive Low-Rank Orthogonal Subspaces for Clothes-Changing Person Re-Identification](../../ICML2026/human_understanding/learning_instance-adaptive_low-rank_orthogonal_subspaces_for_clothes-changing_pe.md)
- [\[ECCV 2024\] AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition](adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)

</div>

<!-- RELATED:END -->
