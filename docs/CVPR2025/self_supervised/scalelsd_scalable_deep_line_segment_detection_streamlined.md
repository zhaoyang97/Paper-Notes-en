---
title: >-
  [Paper Note] ScaleLSD: Scalable Deep Line Segment Detection Streamlined
description: >-
  [CVPR 2025][Self-Supervised Learning][Line Segment Detection] By streamlining the line segment detection architecture (introducing HAT-induced proposal verification) and designing an efficient pseudo-label generation pipeline (LSD-Rectifier), ScaleLSD achieves large-scale self-supervised training on 10 million unlabeled images for the first time, comprehensively outperforming classical non-deep LSD methods in zero-shot evaluations.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Line Segment Detection"
  - "HAT Field"
  - "Large-Scale Training"
  - "Pseudo-Labeling"
date: 2026-05-08
content_hash: 3fb983f3e1045734
---

# ScaleLSD: Scalable Deep Line Segment Detection Streamlined

**Conference**: CVPR 2025  
**arXiv**: [2506.09369](https://arxiv.org/abs/2506.09369)  
**Code**: [https://github.com/ant-research/scalelsd](https://github.com/ant-research/scalelsd)  
**Area**: Self-Supervised Learning / Low-Level Vision  
**Keywords**: Line Segment Detection, Self-Supervised Learning, HAT Field, Large-Scale Training, Pseudo-Labeling

## TL;DR

By streamlining the line segment detection architecture (introducing HAT-induced proposal verification) and designing an efficient pseudo-label generation pipeline (LSD-Rectifier), ScaleLSD achieves large-scale self-supervised training on 10 million unlabeled images for the first time, comprehensively outperforming classical non-deep LSD methods in zero-shot evaluations.

## Background & Motivation

**Background**: Line Segment Detection (LSD) is a fundamental task for image geometric representation, widely used in downstream tasks such as vanishing point estimation, two-view matching, and multi-view 3D reconstruction. Deep learning-based LSD methods primarily rely on the Wireframe dataset (only 5K annotated images) for supervised training, limiting their generalization capability. Self-supervised LSD methods (SOLD2, HAWPv3, DeepLSD) attempt to address geometric generalization, but their training scales remain restricted to a few thousand images.

**Limitations of Prior Work**: (1) Supervised methods generalize poorly to cross-domain scenarios; (2) The pseudo-label generation pipelines of self-supervised methods suffer from scalability issues—the homographic adaptation strategies of SOLD2/HAWPv3 result in low recall, while DeepLSD relies on local alignment schemes from classical LSD, inheriting its locality limitations; (3) Classical LSD methods generate spurious results on short segments, yet deep methods lag behind classical ones in terms of detection completeness.

**Key Challenge**: The pseudo-label generation strategies of existing self-supervised methods lack large-scale scalability. Homographic adaptation sacrifices completeness to filter out false detections, whereas the local alignment schemes of classical LSD, though achieving high recall, suffer from locality defects.

**Goal**: To design a line segment detector capable of self-supervised training on over 10 million unlabeled images, comprehensively outperforming classical LSD in zero-shot scenarios.

**Key Insight**: The authors revisit the fundamental designs of deep and non-deep LSD methods, identifying three key observations: (1) The HAT field representation holds enormous potential for self-supervised learning; (2) The image gradient information of classical LSD is robust and reliable in orientation estimation, which can be leveraged to rectify pseudo-labels; (3) Highly expressive Transformer backbones are crucial for digesting large-scale data.

**Core Idea**: Streamlining the architecture (replacing LOI verification with HAT-induced verification) and using classical LSD's orientation field to rectify HAT field predictions to generate high-quality pseudo-labels efficiently, thereby enabling large-scale self-supervised training on 10 million SA1B images.

## Method

### Overall Architecture

The meta-architecture of ScaleLSD is streamlined from HAWPv3. The input image features are extracted by a ViT-Base backbone, and then processed by a DPT head to predict the HAT field (4 channels: distance $d$, orientation $\theta$, and endpoint angles $\alpha, \beta$) along with the junction heatmap. After decoding line segment proposals from the HAT field, HAT-induced proposal verification is used to filter reliable segments. The pseudo-label generation pipeline uses the LSD-Rectifier to inject orientation information from classical LSD into the HAT field predictions, achieving highly efficient and high-quality pseudo-label generation.

### Key Designs

1. **HAT Field Representation and Sparse Decoding**:

    - **Function**: Represents a sparse set of line segments as dense pixel-level fields and decodes a unique set of line segments from them.
    - **Mechanism**: The HAT field maps each foreground pixel to its orthogonal closest line segment, encoding the distance, orientation, and endpoint positions into four components $(d, \theta, \alpha, \beta)$. During decoding, predicted endpoints of each pixel are bound to their nearest junctions, and a unique sparse line segment set is obtained by deduplicating the junction index pairs. Segments with index distances exceeding $\tau_{\text{dist}}=10$ pixels are pruned. A native GPU implementation ensures negligible latency for the unique operation.
    - **Design Motivation**: The dense representation of the HAT field makes the learning objective more explicit, and the sparse decoding scheme leverages junction information to eliminate duplicate proposals.

2. **HAT-Induced Proposal Verification**:

    - **Function**: Replaces the traditional LOI (Line-of-Interest) verification scheme, measuring line segment reliability in a white-box geometric manner.
    - **Mechanism**: For each junction index pair $(\imath_\alpha^k, \imath_\beta^k)$, its support in the HAT field prediction is calculated as $\text{Deg}(\imath_\alpha^k, \imath_\beta^k) = \sum \mathbb{1}[(\imath_\alpha(\mathbf{p}), \imath_\beta(\mathbf{p})) \sim (\imath_\alpha^k, \imath_\beta^k)]$, which measures how many pixels point to this line segment. The larger the number of supporting pixels, the more reliable the line segment is. A default threshold of 10 pixels is used for filtering.
    - **Design Motivation**: LOI verification requires learning a confidence score, which suffers from low reliability in self-supervised learning with noisy pseudo-labels. HAT-induced verification is based on geometric consistency measures, exhibiting better interpretability and robustness without requiring extra labels.

3. **LSD-Rectifier Pseudo-Label Generation**:

    - **Function**: Efficiently generates high-quality pseudo-labels, supporting self-supervised training at a scale of tens of millions of images.
    - **Mechanism**: A seed model is first trained on synthetic data. Then, two sets of outputs are generated simultaneously for real images: the HAT field predicted by the seed model (primary source) and the orientation field of classical LSD (auxiliary source). The key operation involves replacing the $\theta$ component of the primary source with the $\theta$ component from classical LSD to construct a rectified HAT field, from which line segments are decoded as pseudo-labels. Classical LSD is locally accurate and highly generalizable in orientation estimation; this rectification eliminates the need for expensive homographic adaptation.
    - **Design Motivation**: The image gradient direction information of classical LSD is robust across domains but lacks precise endpoint localization; meanwhile, the seed model's HAT field provides accurate endpoints but may suffer from orientation bias (especially during synthetic-to-real transfer). LSD-Rectifier combines the strengths of both.

### Loss & Training

- **Training Scheme**: A two-stage "synthetic-to-real" training scheme is used. The seed model is first trained on 16K synthetic images for 10 epochs. Next, the seed model combined with the LSD-Rectifier is used to generate pseudo-labels on real data, training from scratch on Wireframe (20K) for 30 epochs or on SA1B (10M) for 6 epochs.
- **Optimizer**: ADAM. During the synthetic stage, lr is 4e-4, decaying by a factor of 10 at the 7th epoch. During the SA1B stage, a linear warmup (2,000 steps from 2e-4 to 1e-3) combined with cosine annealing is applied.
- **Backbone**: ViT-Base with a DPT head.

## Key Experimental Results

### Main Results — Zero-Shot Repeatability Evaluation

| Method | YorkUrban Rep-5(S)↑ | HPatches Rep-5(S)↑ | COCO Rep-5(S)↑ | Avg. Detections/Img |
|------|---------------------|---------------------|-----------------|--------------|
| LSD (Classical) | 0.419 | 0.275 | 0.456 | 493-591 |
| HAWPv3 | 0.711 | 0.322 | 0.644 | 99-225 |
| DeepLSD | 0.514 | 0.241 | 0.423 | 207-310 |
| **ScaleLSD@SA1B** | **0.725** | **0.367** | **0.666** | 540-708 |

### Vanishing Point Estimation

| Method | YUD+ VP Error↓ | YUD+ AUC↑ |
|------|---------------|-----------|
| LSD | 2.05 | 82.9 |
| DeepLSD | 1.63 | 85.6 |
| **ScaleLSD@SA1B** | **1.47** | **87.2** |

### Key Findings

- ScaleLSD is the first deep learning method to comprehensively outperform classical LSD across all evaluation metrics.
- Scaling up from Wireframe (20K) to SA1B (10M) yields consistent and significant improvements, validating the value of large-scale training.
- HAT-induced verification not only streamlines the architecture but also provides more reliable line segment filtering in self-supervised scenarios.
- ScaleLSD detects far more line segments than other deep methods (approaching or even exceeding classical LSD) while maintaining higher precision.

## Highlights & Insights

- **Validation of Large-Scale Self-Supervision**: It demonstrates that "simple methods + large-scale data" are equally effective for low-level vision tasks, aligning with trends in NLP and high-level vision.
- **New Role for Classical Methods**: Classical LSD is no longer merely a competitor but acts as a complementary tool—its gradient orientation information is utilized to rectify the pseudo-labels of the deep model.
- **Value of White-Box Verification**: HAT-induced verification replaces learned confidence scores with geometric support metrics, offering superior robustness in the presence of noisy pseudo-labels.
- **Minimalist Design Philosophy**: The overall architecture is highly streamlined, eliminating complex components such as homographic adaptation, LOI verification, and edge map learning.

## Limitations & Future Work

- The quality of pseudo-labels is still constrained by the orientation estimation precision of classical LSD, particularly near curved boundaries.
- On highly challenging datasets like RDNIM, the localization error of HAWPv3 remains lower than that of ScaleLSD (though it detects far fewer segments).
- The performance upper bounds using larger backbones (e.g., ViT-Large) or even larger amounts of training data remain unexplored.
- Future work could explore iteratively applying the LSD-Rectifier to progressively refine pseudo-label quality.

## Related Work & Insights

- **HAWPv3**: The direct predecessor of ScaleLSD. ScaleLSD streamlines its architecture and scales up the training.
- **DeepLSD**: Also leverages classical LSD to assist self-supervised training, but uses it for local alignment instead of orientation rectification, causing its performance to degenerate toward classical LSD levels.
- **SAM-1B Dataset**: Provides 10 million unlabeled images for ScaleLSD, showcasing the potential of large-scale datasets in self-supervised low-level vision.

## Rating

- **Novelty**: 7/10 — The core contribution lies in structural simplification and scale expansion; the novelty of individual technical components is moderate.
- **Experimental Thoroughness**: 9/10 — Zero-shot evaluations cover 4 datasets and 4 downstream tasks, providing a comprehensive comparison.
- **Writing Quality**: 8/10 — The logic is clear, with a thorough derivation of observations and design choices.
- **Value**: 8/10 — First to demonstrate that deep LSD can comprehensively surpass classical LSD, holding significant importance for the low-level vision field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SegMASt3R: Geometry Grounded Segment Matching](../../NeurIPS2025/self_supervised/segmast3r_geometry_grounded_segment_matching.md)
- [\[CVPR 2025\] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)
- [\[ICML 2025\] Deep Learning is Not So Mysterious or Different](../../ICML2025/self_supervised/deep_learning_is_not_so_mysterious_or_different.md)
- [\[ICML 2025\] Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling](../../ICML2025/self_supervised/neighbour-driven_gaussian_process_variational_autoencoders_for_scalable_structur.md)
- [\[ECCV 2024\] Rethinking Unsupervised Outlier Detection via Multiple Thresholding](../../ECCV2024/self_supervised/rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)

</div>

<!-- RELATED:END -->
