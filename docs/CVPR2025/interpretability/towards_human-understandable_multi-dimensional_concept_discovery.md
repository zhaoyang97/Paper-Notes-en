---
title: >-
  [Paper Note] Towards Human-Understandable Multi-Dimensional Concept Discovery
description: >-
  [CVPR 2025][Interpretability][Concept Discovery] Proposed the HU-MCD framework, which replaces traditional segmentation methods with SAM to discover human-understandable visual concepts, coupled with a CNN-specific input masking scheme to reduce noise interference, achieving concept-level model explanations that balance understandability and faithfulness under the completeness framework of MCD.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Concept Discovery"
  - "Explainable AI"
  - "SAM Segmentation"
  - "CNN Interpretation"
  - "Human Understandability"
date: 2026-05-08
content_hash: cfbe7638afb5f51f
---

# Towards Human-Understandable Multi-Dimensional Concept Discovery

**Conference**: CVPR 2025  
**arXiv**: [2503.18629](https://arxiv.org/abs/2503.18629)  
**Code**: [https://github.com/grobruegge/hu-mcd](https://github.com/grobruegge/hu-mcd)  
**Area**: Interpretability  
**Keywords**: Concept Discovery, Explainable AI, SAM Segmentation, CNN Interpretation, Human Understandability

## TL;DR
Proposed the HU-MCD framework, which replaces traditional segmentation methods with SAM to discover human-understandable visual concepts, coupled with a CNN-specific input masking scheme to reduce noise interference, achieving concept-level model explanations that balance understandability and faithfulness under the completeness framework of MCD.

## Background & Motivation
1. **Background**: Concept-level Explainable AI (C-XAI) aims to replace pixel-level saliency maps with human-understandable visual concepts to explain model decisions. Representative methods include ACE (superpixel segmentation + clustering), ICE (NMF feature decomposition), CRAFT (regional NMF), and MCD (multi-dimensional subspace decomposition).
2. **Limitations of Prior Work**: (a) ACE requires inpainting and resizing segmented regions to the model input size, introducing noise that interferes with model predictions; (b) Albeit MCD guarantees faithfulness with completeness theory, the generated concepts are difficult for humans to understand (highly similar across different concepts, with ambiguous semantics); (c) An inherent trade-off exists between understandability and faithfulness: segmentations closely aligned with human perception may deviate more from internal model representations.
3. **Key Challenge**: Discovering human-understandable concepts requires high-quality segmentation, but irregular segmented regions are difficult to feed into CNNs. Conversely, directly using feature maps without segmentation preserves faithfulness but yields incomprehensible concepts.
4. **Goal**: To simultaneously achieve both human understandability of concepts and faithful explanations of model decisions.
5. **Key Insight**: Leveraging SAM for instance segmentation to obtain high-quality semantic regions + CNN layer masking to avoid inpainting noise.
6. **Core Idea**: SAM segmentation $\rightarrow$ CNN-specific hierarchical mask propagation $\rightarrow$ SSC clustering $\rightarrow$ MCD completeness decomposition.

## Method

### Overall Architecture
Two stages: **Concept Discovery**—segment class-specific images with SAM, extract feature embeddings for each region using a CNN-specific hierarchical masking scheme, and cluster similar regions into concepts utilizing Sparse Subspace Clustering (SSC); **Concept Scoring**—apply the MCD framework to calculate activation and importance scores for each concept in the feature space, satisfying both local and global completeness.

### Key Designs

1. **SAM-Driven Concept Discovery**

    - Function: Generate semantically meaningful and boundary-precise image regions as concept candidates.
    - Mechanism: Invoke SAM (ViT-h encoder) on each class image, selecting the finest-grained segmentation masks with an area coverage $\ge 1\%$. The number of clusters is determined automatically based on the average segmentations per image (unlike ACE which manually sets it to 25). Trained on human-annotated segmentation masks, SAM naturally generates partitions aligned with human intuition.
    - Design Motivation: ACE's superpixel segmentation lacks semantic meaning. SAM's zero-shot instance segmentation capability generates semantically more meaningful regions.

2. **CNN-Specific Hierarchical Masking Scheme**

    - Function: Extract CNN features of irregular regions without introducing inpainting or resizing noise.
    - Mechanism: Inspired by Balasubramanian & Feizi, propagate both the image and its corresponding mask layer-by-layer; after each convolutional layer, use the mask to discard activation values that depend only on covered areas. At boundaries, perform padding with the mean of neighboring unmasked pixels to prevent edge artifacts. Special handling: the first convolutional layer (such as the $7 \times 7$ kernel in ResNet50) is allowed to access a narrow ribbon of context around the mask edge to retain shape information; for large masks ($> 25\%$ area), the kernel size is shrunk to avoid leaking object boundaries.
    - Design Motivation: Traditional methods that fill masked regions with mean-padding or inpainting introduce spurious features that disrupt model predictions; the hierarchical masking scheme eliminates noise from the root.

3. **Adaptation of the MCD Completeness Framework**

    - Function: Provide importance scores with completeness guarantees for each concept discovered by SAM.
    - Mechanism: Perform PCA on the latent representations of cluster members and select the principal components as the basis of the concept subspace. All concept subspaces, combined with the orthogonal complement subspace, form a complete decomposition of the feature space. **Concept Activation**: Project regional features into the subspace to measure concept presence intensity. **Local Concept Relevance**: Decompose the final classification logit into contributions from each concept subspace, where the sum strictly equals the original logit (completeness). **Global Concept Relevance**: Project the classification weight vector onto each subspace.
    - Design Motivation: MCD's completeness guarantees that concept importance scores "faithfully" reflect the model's decision-making process without losing information.

### Loss & Training
HU-MCD is a post-hoc explanation method that requires no training. It utilizes a pre-trained ResNet50 (timm) and a pre-trained SAM (ViT-h).

## Key Experimental Results

### Main Results (Human Experiments + 10 ImageNet Classes)

| Metric | HU-MCD | ACE | MCD |
|------|--------|-----|-----|
| Prediction Accuracy ↑ | **70.24%** | 42.93% | 31.22% |
| Identifiable Concept Ratio ↑ | **67.12%** | 45.66% | 50.34% |
| Intra-concept Description Similarity ↑ | **0.49** | 0.39 | 0.41 |
| Inter-concept Description Similarity ↓ | **0.28** | 0.29 | 0.38 |

### Ablation Study

| Method | C-Insertion AUC ↑ | C-Deletion AUC ↓ | Notes |
|------|-----------------|----------------|------|
| HU-MCD | **best** | **best** | Most faithful concept importance scores |
| ACE | Medium | Medium | Segmentation noise impairs faithfulness |
| MCD | Medium | Medium | Indistinguishable concepts hinder evaluation |

### Key Findings
- HU-MCD achieves a prediction accuracy of $70.24\%$ vs MCD's $31.22\%$, representing a significant leap in understandability.
- MCD's inter-concept description similarity reaches as high as $0.38$ (close to the intra-concept similarity of $0.41$), indicating that its concepts are highly homogenized and hard for humans to distinguish.
- Concepts segmented by SAM can uncover dataset biases (e.g., the "human hand" concept in the "frog" class), demonstrating practical value in discovering spurious correlations.
- The hierarchical masking scheme preserves model accuracy significantly better than the mean-padding scheme.
- Experiments conducted with 41 human subjects (including attention checks) yield statistically significant results (ANOVA $p < 0.001$).

## Highlights & Insights
- **Integrating SAM into C-XAI** is a natural yet effective combination: SAM's zero-shot segmentation ability compensates for the weak segmentation capabilities of traditional concept discovery methods.
- The **hierarchical masking scheme** solves a long-standing engineering bottleneck in C-XAI: how to feed irregular regions into CNNs without introducing spurious noise.
- The **human experiment design** is highly rigorous, validating understandability across multiple dimensions: task prediction, identifiability assessment, and description consistency.

## Limitations & Future Work
- Only validated on ResNet50; the applicability to Transformer architectures remains to be explored.
- The high computational overhead of SAM limits large-scale application.
- The evaluation scale is relatively small (10 classes).
- Future research could explore utilizing the discovered concepts for model improvement (rather than solely for explanation).

## Related Work & Insights
- **vs ACE**: ACE employs superpixel segmentation + inpainting, yielding moderate concept understandability but compromised faithfulness due to noise. HU-MCD introduces a dual improvement using both SAM and hierarchical masking.
- **vs MCD**: MCD boasts strong faithfulness but suffers from incomprehensible concepts (highly homogenized). HU-MCD leverages SAM to guarantee understandability while inheriting MCD's completeness to ensure faithfulness.
- **vs CRAFT**: CRAFT uses square patches to avoid inpainting, but loses precise region boundaries.

## Rating
- Novelty: ⭐⭐⭐⭐ A combination of SAM, CNN masking, and MCD. While individual components are based on existing work, their integration is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely rigorous, featuring a 41-subject human study, multiple metrics, and statistical significance tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivations, standardized design of human experiments, and intuitive figures.
- Value: ⭐⭐⭐⭐ Advances the balance between understandability and faithfulness in the field of Explainable AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICML 2026\] CB-SLICE: Concept-Based Interpretable Error Slice Discovery](../../ICML2026/interpretability/cb-slice_concept-based_interpretable_error_slice_discovery.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ICCV 2025\] VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow](../../ICCV2025/interpretability/vital_more_understandable_feature_visualization_through_distribution_alignment_a.md)
- [\[ICLR 2026\] Adaptive Concept Discovery for Interpretable Few-Shot Text Classification](../../ICLR2026/interpretability/adaptive_concept_discovery_for_interpretable_few-shot_text_classification.md)

</div>

<!-- RELATED:END -->
