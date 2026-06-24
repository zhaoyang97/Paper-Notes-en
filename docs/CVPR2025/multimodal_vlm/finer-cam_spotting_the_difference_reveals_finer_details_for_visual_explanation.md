---
title: >-
  [Paper Note] Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation
description: >-
  [CVPR 2025][Multimodal VLM][Interpretability] By changing the explanation target of CAM from a single-class logit $y^c$ to the contrastive difference between classes $y^c - \gamma \cdot y^d$ (the logit difference between the target class and a similar class), Finer-CAM upgrades any CAM method into a fine-grained version with zero additional parameters, refining the activation maps from "overall silhouettes" to "discriminative local details".
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Interpretability"
  - "CAM Improvement"
  - "Fine-grained Recognition"
  - "Class Contrast"
  - "Visual Explanation"
date: 2026-05-08
content_hash: 54ac5add196b9053
---

# Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation

**Conference**: CVPR 2025  
**arXiv**: [2501.11309](https://arxiv.org/abs/2501.11309)  
**Code**: [https://github.com/Imageomics/Finer-CAM](https://github.com/Imageomics/Finer-CAM)  
**Area**: Multimodal VLMs  
**Keywords**: Interpretability, CAM Improvement, Fine-grained Recognition, Class Contrast, Visual Explanation

## TL;DR
By changing the explanation target of CAM from a single-class logit $y^c$ to the contrastive difference between classes $y^c - \gamma \cdot y^d$ (the logit difference between the target class and a similar class), Finer-CAM upgrades any CAM method into a fine-grained version with zero additional parameters, refining the activation maps from "overall silhouettes" to "discriminative local details".

## Background & Motivation

**Background**: Class Activation Mapping (CAM) is the most widely used visual explanation method, which generates heatmaps by weight-combining feature maps to indicate which regions contribute the most to classification decisions.

**Limitations of Prior Work**: CAM explains the logit $y^c$ of a single class, which causes it to highlight all regions that positively contribute to that class—including regions shared with similar classes. For example, when distinguishing between two types of blue birds, CAM highlights the entire blue body (shared by both birds) instead of the color difference on the wings (the discriminative feature). This makes CAM almost useless for providing meaningful explanations in fine-grained recognition.

**Key Challenge**: There is a fundamental misalignment between the explanation target of CAM (single-class logit) and what users actually need to be explained (inter-class discriminative features). Improving the process of CAM (how weights are computed) cannot resolve this issue at the "target" level.

**Goal**: To redefine the explanation target of CAM, focusing it on the "fine-grained differences that distinguish the target class from similar classes".

**Key Insight**: Inspired by humans' ability to "spot the difference"—discovering major visual nuances by comparing two similar objects. The explanation target of CAM is changed from $y^c$ to $y^c - \gamma \cdot y^d$ to suppress shared features and amplify discriminative ones.

**Core Idea**: Change the target from a single-class logit to the logit difference between classes during backpropagation, converting coarse-grained CAM to fine-grained Finer-CAM at zero cost.

## Method

### Overall Architecture
Standard forward propagation yields feature maps $\mathcal{A} = \{A_k\}$ and all class logits → Select the $T=3$ reference classes most similar to target class $c$ → Compute contrastive weights for each reference class $d$ as $\alpha_k^{c,d} = \alpha_k^c - \gamma \cdot \alpha_k^d$ → Average across reference classes → Apply ReLU activation to obtain the fine-grained heatmap.

### Key Designs

1. **Contrastive Explanation Target**:

    - **Function**: Transforms CAM weights from "contributions to a single class" to "contributions to inter-class discrimination".
    - **Mechanism**: For gradient-based methods, $\alpha_k^{c,d} = \frac{1}{Z}\sum_{i,j}\frac{\partial(y^c - \gamma \cdot y^d)}{\partial A_k^{ij}}$, which directly decomposes into $\alpha_k^c - \gamma \cdot \alpha_k^d$ due to derivative linearity. A similar approach applies to score-based methods—comparing prediction changes between two classes after masking feature maps. The key is in performing subtraction at the weight level (before ReLU), rather than at the heatmap level (after ReLU).
    - **Design Motivation**: Directly subtracting two CAM heatmaps introduces noise (due to the non-linearity of ReLU), whereas subtraction at the weight level prior to ReLU yields a clean background and sharp discriminative regions.

2. **Reference Class Selection and Aggregation**:

    - **Function**: Selects the most valuable similar classes for reference and aggregates multiple contrastive results.
    - **Mechanism**: Reference classes can be selected by cosine similarity of classifier weights or predicted logit rankings, with Top-3 chosen by default. Contrastive weights from multiple reference classes are averaged before applying ReLU, producing a heatmap that reflects regions that "differ from all similar classes".
    - **Design Motivation**: Using a single reference class may miss certain discriminative features. While contrasting with the 2nd predicted class achieves the highest RD (0.198), aggregating Top-3 performs best on the Deletion metric (0.076).

3. **Contrastive Intensity $\gamma$**:

    - **Function**: Controls the granularity of the explanation.
    - **Mechanism**: When $\gamma=0$, it degenerates to standard CAM (coarse-grained). As $\gamma$ increases, it progressively focuses on finer discriminative features. Experiments show that $\gamma=0.6$ achieves the best balance between relative and absolute confidence drop.
    - **Design Motivation**: Provides a continuous coarse-to-fine tuning knob, allowing users to select the explanation granularity according to their needs.

### Loss & Training
A training-free inference-only method. It merely modifies the backpropagation target and is compatible with any CAM variant (Grad-CAM, Layer-CAM, Score-CAM, etc.).

## Key Experimental Results

### Main Results

| Method | Birds-525 RD@0.1 | CUB-200 RD@0.1 | CUB-200 Localization | Cars RD@0.1 |
|------|-----------------|----------------|-------------|------------|
| Grad-CAM | 0.245 | 0.113 | 0.582 | 0.067 |
| **+Finer** | **0.260** | **0.121** | **0.629** | **0.071** |
| Layer-CAM | 0.255 | 0.116 | 0.625 | 0.069 |
| **+Finer** | **0.270** | **0.120** | **0.682** | **0.074** |
| Score-CAM | 0.217 | 0.102 | 0.670 | - |
| **+Finer** | **0.227** | **0.109** | **0.683** | - |

### Ablation Study

| Configuration | Del. ↓ | RD@0.05 ↑ | Description |
|------|--------|----------|------|
| Standard Grad-CAM | 0.079 | 0.174 | Baseline |
| + 2nd predicted class contrast | 0.079 | **0.198** | Strongest single-class |
| + Top-3 aggregation | **0.076** | 0.192 | Best overall |

### Key Findings
- **Finer-CAM is consistently effective across all CAM methods**: Grad-CAM, Layer-CAM, and Score-CAM all improve when integrated with Finer.
- **Localization significantly improves**: On CUB-200, Layer-CAM's localization performance improves from 0.625 to 0.682 (+9.1%), indicating that Finer-CAM indeed forces heatmaps to focus precisely on discriminative areas.
- **$\gamma=0.6$ provides the best balance**: If it is too small, it degenerates to standard CAM; if too large, it discards critical information from the target class itself.

## Highlights & Insights
- **The insight of "changing the target of explanation rather than the process" is profound**: Historically, CAM improvements focused on "how to explain." This paper highlights that the issue lies in "what to explain"; altering this with a single line of code yields universal improvements.
- **Extension to CLIP**: Text prompt differences can be utilized to achieve concept-level fine-grained localization (e.g., "red epaulet" vs "bird"), extending Finer-CAM to zero-shot scenarios.
- **Zero extra cost**: Requires no parameter addition and no extra forward passes, only modifying the backpropagation target—making it a rare "free lunch" improvement.

## Limitations & Future Work
- Requires prior knowledge of which classes are "similar"; if the classifier itself lacks the capacity to distinguish similar classes, the performance of the contrastive explanation remains limited.
- $\gamma$ needs to be manually tuned; adaptively determining $\gamma$ could lead to further improvements.
- Only verified on fine-grained classification tasks; its utility in general object detection/segmentation scenarios remains unexplored.

## Related Work & Insights
- **vs XGrad-CAM / Ablation-CAM**: These methods optimize the weight computation of CAM but do not change the target of explanation, falling under "how" improvements. Finer-CAM operates on the "what" level and is fully orthogonal to them.
- **vs Contrastive Explanations (CRP, etc.)**: Some contrastive explanation methods analyze the most relevant/irrelevant features but require complex network analysis tools. Finer-CAM requires only a single-line code modification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The insight of modifying the "target of explanation" is exceptionally elegant and profound, yielding widespread improvements with just a single line of code.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets and CAM methods, but lacks evaluation on non-fine-grained tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous reasoning and logic, flowing seamlessly from diagnosing the core issue to the final solution.
- Value: ⭐⭐⭐⭐ Significant contribution to interpretability studies with high practicality, though primarily beneficial in fine-grained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](../../ICML2025/multimodal_vlm/the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)
- [\[ACL 2025\] Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?](../../ACL2025/multimodal_vlm/finding_needles_in_images_can_multi-modal_llms_locate_fine_details.md)
- [\[AAAI 2026\] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](../../AAAI2026/multimodal_vlm/difference_vector_equalization_for_robust_fine-tuning_of_vis.md)
- [\[AAAI 2026\] Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation](../../AAAI2026/multimodal_vlm/few-shot_precise_event_spotting_via_unified_multi-entity_graph_and_distillation.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](relation-rich_visual_document_generator_for_visual_information_extraction.md)

</div>

<!-- RELATED:END -->
