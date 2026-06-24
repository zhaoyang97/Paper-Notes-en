---
title: >-
  [Paper Note] FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition
description: >-
  [CVPR 2025][Causal Inference][Counterfactual Explanations] This paper proposes the FG-VCE (Fine-Grained Visual Contrastive Explanation) framework. By calculating feature point contributions via Shapley values, isolating local features using a saliency partition module, and employing an iterative counterfactual generation strategy, it achieves fine-grained counterfactual explanations at both the object and part levels for the first time. It reveals the specific causes of model…
tags:
  - "CVPR 2025"
  - "Causal Inference"
  - "Counterfactual Explanations"
  - "Fine-Grained Classification"
  - "Shapley Values"
  - "Saliency Partition"
  - "Misclassification Analysis"
date: 2026-05-08
content_hash: 9aa525c891d53b0c
---

# FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition

**Conference**: CVPR 2025  
**arXiv**: [2511.07974](https://arxiv.org/abs/2511.07974)  
**Code**: None  
**Area**: Causal Inference / Interpretability  
**Keywords**: Counterfactual Explanations, Fine-Grained Classification, Shapley Values, Saliency Partition, Misclassification Analysis

## TL;DR
This paper proposes the FG-VCE (Fine-Grained Visual Contrastive Explanation) framework. By calculating feature point contributions via Shapley values, isolating local features using a saliency partition module, and employing an iterative counterfactual generation strategy, it achieves fine-grained counterfactual explanations at both the object and part levels for the first time. It reveals the specific causes of model misclassification: "which fine-grained features led to the error" and "which local regions dominated the prediction change."

## Background & Motivation

**Background**: Attribution explanation methods (e.g., GradCAM, Saliency Map) provide model explanations by highlighting the image regions most critical to the prediction. While sufficient for coarse-grained tasks (e.g., distinguishing cats from dogs), these methods have limited value in fine-grained tasks (e.g., distinguishing bird species) because they typically only highlight the entire object without differentiating meaningful local regions.

**Limitations of Prior Work**: (1) Traditional attribution methods often highlight similar regions regardless of whether the model prediction is correct or incorrect—as shown in Fig 1, the highlighted regions are almost identical for the correct prediction of "Albatross" and the incorrect prediction of "Frigatebird"; (2) Existing counterfactual explanation methods (e.g., CCE) alter predictions via imperceptible adversarial perturbations (PGD attack), but the generated explanations lack intuitive meaning—the perturbations are invisible at the pixel level, and the attribution maps remain almost unchanged; (3) There is a lack of part-level explanations, making it unclear whether details in the "head", "wings", or "legs" caused the misclassification.

**Key Challenge**: In fine-grained classification, distinguishing between similar categories relies on extremely subtle local feature differences (e.g., beak shape, feather texture), but the resolution of existing attribution methods is insufficient to capture these nuances.

**Goal**: Generate fine-grained counterfactual explanations to answer two questions: (1) Which fine-grained features caused the model's misclassification? (2) Which local features dominated the counterfactual adjustments?

**Key Insight**: Leveraging the contrast between correctly classified and misclassified samples—identifying which local features in a misclassified sample, if replaced with the corresponding features of a correctly classified sample, can change the prediction.

**Core Idea**: A non-generative counterfactual method that produces semantically consistent counterfactuals through feature-level matching and iterative substitution between misclassified samples and correctly classified reference samples. It quantifies the contribution of each feature point using approximate Shapley values and introduces a Saliency Partition to decompose the feature map into locally coherent, independent regions.

## Method

### Overall Architecture
FG-VCE consists of three stages: (1) Feature Extraction: Extracting deep feature representations for both the misclassified sample and the correctly classified reference sample; (2) Saliency Partition: Calculating feature point contributions using approximate Shapley values and decomposing the feature map into independent regions with region-specific correlations via a spatial localization kernel; (3) Counterfactual Generation and Explanation: Iteratively selecting the highest-contribution feature regions in the misclassified sample, matching and replacing them with the most semantically similar regions from the reference sample until the prediction changes.

### Key Designs

1. **Approximate Shapley Value Contribution Measurement**:

    - **Function**: Quantifying the marginal contribution of each feature point to the model's prediction
    - **Mechanism**: For each spatial location $(i,j)$ in the feature map, its Shapley value is calculated as the weighted average of its marginal impact on the prediction probability with or without its presence. Since calculating exact Shapley values requires exponential computation, Monte Carlo sampling is utilized for approximation: a subset of features $S$ is randomly selected to compute $v(S \cup \{(i,j)\}) - v(S)$, which is then averaged over multiple runs.
    - **Design Motivation**: Traditional attribution methods (gradients, CAM) reflect "feature importance" rather than "feature contribution". As the only distribution scheme in game theory that satisfies fairness axioms, the Shapley value more accurately quantifies the independent contribution of each region.

2. **Saliency Partition Module**:

    - **Function**: Decomposing the global feature map into locally independent salient regions
    - **Mechanism**: Introducing a spatial localization kernel into the Shapley value computation. For each position $(i,j)$, only feature interactions within its spatial neighborhood are considered, ignoring long-range feature influences. Specifically, a fixed-size spatial kernel (e.g., $7 \times 7$) restricts the scope of the Shapley value calculation, ensuring that the contribution at each position reflects only local contextual information.
    - **Design Motivation**: Feature representations at adjacent positions in standard feature maps are highly coupled due to overlapping convolutional receptive fields. Directly applying Shapley values blends the contributions of many coupled regions. The local kernel breaks this coupling, resulting in finer-grained regional partition.

3. **Iterative Counterfactual Generation**:

    - **Function**: Constructing semantically consistent counterfactual samples progressively
    - **Mechanism**: Maintaining a "high-contribution feature candidate set" comprising key feature regions extracted from correctly classified reference samples. In each iteration: (i) the feature region in the misclassified sample with the highest Shapley value is selected; (ii) the match is found in the candidate set using the highest semantic cosine similarity; (iii) the feature of that region is replaced; and (iv) the model checks if the prediction has changed. This process repeats until the prediction changes, with all replaced regions constituting the "key features causing misclassification."
    - **Design Motivation**: Replacing all features at once is too coarse and lacks interpretability. Iterative replacement allows precise pinpointing of the "minimum set of regions that need to be modified to correct the prediction", providing minimal counterfactual explanations.

### Loss & Training
FG-VCE itself does not require training—it is a post-hoc explanation method that generates explanations based on a pre-trained classification model. It only requires the feature extractor and the prediction head of the classifier.

## Key Experimental Results

### Main Results

| Method | CUB-200 Del-Ins ↑ | CUB-200 SCIC ↑ | Dogs Del-Ins ↑ | Dogs SCIC ↑ | Semantic Consistency ↑ |
|---|---|---|---|---|---|
| GradCAM | 0.312 | 0.425 | 0.289 | 0.401 | 0.621 |
| CCE (PGD-based) | 0.358 | 0.478 | 0.321 | 0.443 | 0.539 |
| ACE (Autoencoder) | 0.401 | 0.512 | 0.367 | 0.489 | 0.684 |
| SCOUT | 0.423 | 0.534 | 0.385 | 0.507 | 0.702 |
| **FG-VCE (Ours)** | **0.487** | **0.602** | **0.441** | **0.571** | **0.758** |

### Fine-Grained Region Localization Quality

| Method | Part IoU ↑ | Part Coverage ↑ | Misleading Regions ↓ |
|---|---|---|---|
| GradCAM | 0.184 | 42.3% | 38.7% |
| CCE | 0.213 | 48.1% | 35.2% |
| **FG-VCE** | **0.342** | **71.8%** | **18.3%** |

### Ablation Study

| Configuration | Del-Ins ↑ | SCIC ↑ | Description |
|---|---|---|---|
| FG-VCE (full) | 0.487 | 0.602 | Full method |
| w/o Saliency Partition | 0.421 | 0.538 | Degenerates to global Shapley, losing local precision |
| w/o Shapley (using gradients) | 0.378 | 0.501 | Gradient-based attribution is less accurate |
| w/o Iterative Replacement (single-step) | 0.398 | 0.489 | One-step replacement is too coarse |
| Replacement with random regions | 0.312 | 0.412 | Verifies the necessity of semantic matching |

### Key Findings
- **FG-VCE outperforms across all metrics**: On CUB-200, Del-Ins increases from 0.423 to 0.487 (+15.1%), and SCIC from 0.534 to 0.602 (+12.7%).
- **Significant improvement in part-level localization quality**: Part IoU increases from 0.213 to 0.342 (+60.6%), showing that FG-VCE indeed localizes fine-grained parts (head, wings, etc.).
- **Saliency Partition contributes the most**: Removing it drops Del-Ins from 0.487 to 0.421, showing that spatial kernel decomposition is a key innovation.
- **Highest semantic consistency (0.758)**: Counterfactuals are generated via semantic matching and substitution rather than imperceptible adversarial perturbations, preserving visual comprehensibility.

## Highlights & Insights
- **First systematic exploration of fine-grained counterfactual explanations**: Pioneering the research direction of identifying "which part caused the misclassification." Prior works either only provided object-level explanations or relied on uninterpretable adversarial perturbations.
- **Advantages of non-generative methods**: Instead of relying on generative models such as GANs or Diffusion to synthesize counterfactuals, this approach performs feature transplantation between real samples, avoiding artifacts introduced by generative processes.
- **Shapley values + spatial localization**: Combining the fair distribution scheme of game theory with spatial priors is highly elegant. Shapley values guarantee the theoretical soundness of contribution quantification, while the spatial kernel ensures regional independence.
- **High transferability**: The framework is independent of specific classifier architectures and can be directly applied for post-hoc explanations of any CNN or ViT models.

## Limitations & Future Work
- The Monte Carlo approximation of Shapley values requires multiple forward passes, leading to a relatively high computational overhead (dozens of inferences per explanation).
- The spatial localization kernel is of a fixed size, which might not adapt well to visual parts of different scales (e.g., wings vs. eyes).
- The quality of the reference sample set affects the counterfactual quality. If the reference set lacks high-quality samples of the correct class, the counterfactuals may be suboptimal.
- The method is validated only on 2D image classification; fine-grained explanations for other tasks like object detection and segmentation remain unexplored.

## Related Work & Insights
- **vs. GradCAM**: Only provides heatmap-level explanations and fails to distinguish different regions between correct and incorrect predictions, whereas FG-VCE provides part-level counterfactuals.
- **vs. CCE**: Uses PGD adversarial attacks to generate counterfactuals, producing perturbations that are neither visible nor interpretable, whereas FG-VCE's feature substitution is visually interpretable.
- **vs. ACE/SCOUT**: Provides explanations based on concepts/prototypes but lacks fine-grained part localization, whereas FG-VCE achieves part-level precision through Saliency Partition.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First fine-grained counterfactual explanations; the combination of Shapley and spatial kernel is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets and comprehensive ablation studies, but lacks user studies.
- **Writing Quality**: ⭐⭐⭐⭐ The problem motivation is clearly articulated, and the framework diagram is intuitive.
- **Value**: ⭐⭐⭐⭐ Opens up a new direction for fine-grained interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](../../ACL2025/causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[NeurIPS 2025\] Performative Validity of Recourse Explanations](../../NeurIPS2025/causal_inference/performative_validity_of_recourse_explanations.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](../../ICLR2026/causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)

</div>

<!-- RELATED:END -->
