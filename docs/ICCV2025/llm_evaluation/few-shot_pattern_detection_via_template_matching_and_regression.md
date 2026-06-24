---
title: >-
  [Paper Note] Few-Shot Pattern Detection via Template Matching and Regression
description: >-
  [ICCV 2025][LLM Evaluation][few-shot detection] This paper proposes TMR, a method that combines classical template matching with support-conditioned bounding box regression to achieve few-shot detection of arbitrary patterns—including non-object-level patterns. The authors also introduce the RPINE dataset to cover a broader range of repetitive patterns. TMR surpasses existing FSCD methods on multiple benchmarks and demonstrates strong cross-dataset generalization.
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "few-shot detection"
  - "template matching"
  - "pattern detection"
  - "repetitive patterns"
  - "anchor-free detection"
date: 2026-05-08
content_hash: 14fc1f063ac4ae24
---

# Few-Shot Pattern Detection via Template Matching and Regression

**Conference**: ICCV 2025
**arXiv**: [2508.17636](https://arxiv.org/abs/2508.17636)  
**Code**: [https://cvlab.postech.ac.kr/research/TMR](https://cvlab.postech.ac.kr/research/TMR)  
**Area**: LLM Evaluation
**Keywords**: few-shot detection, template matching, pattern detection, repetitive patterns, anchor-free detection

## TL;DR
This paper proposes TMR, a method that combines classical template matching with support-conditioned bounding box regression to achieve few-shot detection of arbitrary patterns—including non-object-level patterns. The authors also introduce the RPINE dataset to cover a broader range of repetitive patterns. TMR surpasses existing FSCD methods on multiple benchmarks and demonstrates strong cross-dataset generalization.

## Background & Motivation
- **Background**: Few-shot object detection (FSOD) and few-shot counting detection (FSCD) have achieved notable progress, but existing methods rely heavily on object-level priors and are primarily designed for categories with well-defined boundaries.
- **Limitations of Prior Work**: Mainstream methods (e.g., Counting-DETR, GeCo, PseCo) compress support samples into prototype vectors via global average pooling, discarding spatial structural information.
- **Key Challenge**: When detection targets are extended from objects to arbitrary patterns (e.g., textures, geometric structures, object parts), spatial layout information becomes critical—yet prototype matching discards precisely this information.
- **Core Problem**: How to design a few-shot pattern detector that preserves spatial structure without relying on object-level priors.
- **Key Insight**: Revisiting classical template matching and leveraging 2D cross-correlation to preserve the spatial layout of exemplars.
- **Core Idea**: Replace prototype matching with channel-wise template matching, and adaptively refine bounding boxes via support-conditioned regression.

## Method

### Overall Architecture
TMR adopts a minimalist structure: a frozen SAM-ViT/H backbone extracts feature maps → RoIAlign crops template features → channel-wise template matching produces correlation maps → original and matched features are concatenated → a bounding box regressor and existence classifier generate predictions → NMS post-processing. The entire detection head consists of only a few $3\times3$ convolutions and linear layers, with no cross-attention or other complex modules.

### Key Designs
1. **Channel-wise Template Matching**:

    - **Function**: Performs cross-correlation between the template feature $\mathbf{T} \in \mathbb{R}^{t_h \times t_w \times D}$ and the image feature map $\mathbf{F}$ via a sliding window.
    - **Mechanism**: $\mathbf{F}_{\text{TM}}(x,y) = \frac{1}{t_w t_h} \sum_{x',y'} \mathbf{F}(x+x'-\lfloor t_w/2 \rfloor, y+y'-\lfloor t_h/2 \rfloor) \mathbf{T}(x',y')$, with the result retaining the channel dimension $\mathbf{F}_{\text{TM}} \in \mathbb{R}^{H \times W \times D}$.
    - **Design Motivation**: Unlike prototype matching, channel-wise template matching preserves the spatial structure and geometric characteristics of exemplars, which is critical for detecting non-object patterns. Ablation results show a significant AP drop when the 2D template is replaced by a pooled prototype (RPINE: 33.59 → 20.94).

2. **Adaptive Template Extraction**:

    - **Function**: Uses RoIAlign to crop template features from the image feature map.
    - **Mechanism**: Template size is adaptively determined based on the actual size of the support exemplar (rounded up), rather than fixed-size pooling.
    - **Design Motivation**: Maintains translational alignment between the template and the feature map, avoiding spatial information loss.

3. **Support-Conditioned Box Regression**:

    - **Function**: Predicts scaling and offset parameters relative to the support exemplar size, rather than absolute coordinates.
    - **Mechanism**: For each feature point, the model predicts $(\Delta x, \Delta y, \alpha_w, \alpha_h)$; the final box is $(x + s_w \Delta x,\ y + s_h \Delta y,\ e^{\alpha_w} s_w,\ e^{\alpha_h} s_h)$.
    - **Design Motivation**: Using exemplar size as a reference baseline allows the model to dynamically adapt to exemplars and targets of varying scales. Ablation experiments confirm that conditioned regression outperforms direct regression (AP: 36.01 vs. 17.01).

### Loss & Training
- **Existence Loss** $\mathcal{L}_P$: Binary cross-entropy loss with center-point expanded margins.
- **Bounding Box Loss** $\mathcal{L}_B$: gIoU loss, computed only at locations where targets are present.
- **Total Loss**: $\mathcal{L} = \mathcal{L}_P + \mathcal{L}_B$
- The SAM-ViT/H backbone is kept frozen; only the detection head is trained (~19M trainable parameters).
- Feature maps are interpolated from $64\times64$ to $128\times128$ to improve dense prediction accuracy.

## Key Experimental Results

### Main Results

| Dataset | Metric | TMR | Prev. SOTA (GeCo) | Gain |
|--------|------|------|----------|------|
| RPINE (1-shot) | AP | 33.59 | 23.33 | +10.26 |
| RPINE (1-shot) | AP50 | 64.05 | 45.93 | +18.12 |
| FSCD-LVIS seen (3-shot) | AP | 27.49 | 22.37 (PseCo) | +5.12 |
| FSCD-LVIS unseen (3-shot) | AP | 22.71 | 11.47 (GeCo) | +11.24 |
| FSCD-147 (1-shot) | AP | 36.01 | 32.71 (GeCo) | +3.30 |
| FSCD-147 (3-shot) | AP | 38.57 | 32.49 (GeCo) | +6.08 |

### Ablation Study

| Configuration | RPINE AP | FSCD-147 AP | Notes |
|------|---------|-------------|------|
| Image features only $\mathbf{F}$ | 11.44 | 20.95 | Lower bound without exemplar information |
| Template matching features only $\mathbf{F}_{\text{TM}}$ | 32.55 | 31.96 | Effectiveness of template matching |
| $\mathbf{F} \oplus \mathbf{F}_{\text{PM}}$ (prototype matching) | 20.94 | 28.91 | Pooled prototype loses spatial information |
| $\mathbf{F} \oplus \mathbf{F}_{\text{TM}}$ (full model) | 33.59 | 36.01 | Optimal combination of spatial structure and appearance |

### Key Findings
- The SAM decoder degrades performance on RPINE (AP: 33.59 → 29.66), as SAM tends to align with edges, which is detrimental for non-object patterns.
- TMR's FLOPs (3.04T) are substantially lower than those of PseCo (5.08T) and GeCo (4.72T).
- In cross-dataset evaluation, TMR demonstrates a dominant advantage: when trained on RPINE and tested on FSCD-147, it achieves an AP of 41.39 vs. 36.99 for GeCo.

## Highlights & Insights
- A successful case of revisiting classical methods: the time-honored template matching technique, combined with modern feature extractors, yields remarkable performance.
- Minimalist architecture: only $3\times3$ convolutions and linear layers, without any attention mechanism, yet achieves state-of-the-art results.
- The RPINE dataset fills the gap in evaluation of non-object pattern detection, supporting multi-pattern annotation (up to 3 distinct patterns per image, independently annotated by 3 annotators).
- The results expose the tendency of prototype matching methods to overfit to object-level semantics.
- The insight regarding the SAM decoder is particularly valuable: its edge-alignment property is harmful for non-object patterns, cautioning the community against the uncritical use of SAM post-processing.

## Limitations & Future Work
- Relies on a frozen SAM backbone; resolution for small instances is constrained by the ViT patch size.
- The RPINE dataset is relatively small (4,362 images), potentially limiting diversity in learned representations.
- Only 2D patterns are addressed; spatiotemporal patterns in 3D or video settings are not considered.
- Computational overhead from multi-scale inference can be further optimized.
- Robustness of template matching to rotation and large-scale variation remains to be validated.
- The authors suggest future exploration of lightweight, pattern-specific architectures that reduce dependence on object-level edge priors.

## Related Work & Insights
- **GeCo/PseCo**: Prototype matching-based FSCD methods whose performance is limited by spatial information loss.
- **Classical Template Matching**: TMR successfully combines the classical approach with deep features, offering a paradigm worth borrowing for other detection tasks.
- **Generality of the SAM Backbone**: Frozen SAM features perform strongly in few-shot detection, demonstrating the transferability of large-model representations.
- **FSCD-147/FSCD-LVIS**: Existing standard FSCD datasets, but covering only object-level patterns.
- **Insight**: For matching tasks requiring spatial structure preservation (e.g., point cloud registration, texture analysis), the combination of classical methods and modern features may be an underexplored yet effective strategy.
- **SEM Image Application**: The authors demonstrate cross-domain detection on scanning electron microscope images, indicating practical application potential of the proposed method.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Integrates classical template matching with a modern detection framework and proposes support-conditioned regression from a fresh perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, cross-dataset evaluation, comprehensive ablation, computational complexity analysis, and real-world application validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, well-motivated problem formulation, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Extends few-shot detection from objects to arbitrary patterns, opening a new research direction; the RPINE dataset holds long-term value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)
- [\[ICCV 2025\] Combinative Matching for Geometric Shape Assembly](combinative_matching_for_geometric_shape_assembly.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[ICCV 2025\] DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection](discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)

</div>

<!-- RELATED:END -->
