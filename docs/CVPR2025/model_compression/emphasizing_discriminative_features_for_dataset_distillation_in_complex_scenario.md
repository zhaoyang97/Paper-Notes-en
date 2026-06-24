---
title: >-
  [Paper Note] Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios
description: >-
  [CVPR 2025][Model Compression][Dataset Distillation] Proposes the EDF method to address the performance degradation of dataset distillation in complex scenarios (ImageNet subsets). It introduces Common Pattern Dropout (discarding parameter gradients of low-loss common patterns in trajectory matching) and Discriminative Area Enhancement (utilizing Grad-CAM to scale up gradients of discriminative regions), achieving lossless compression on datasets such as ImageMeow/ImageYellow…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Dataset Distillation"
  - "Discriminative Features"
  - "Grad-CAM"
  - "Complex Scenarios"
  - "Common Pattern Dropout"
date: 2026-05-08
content_hash: 34c23e6a70bfd8f6
---

# Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios

**Conference**: CVPR 2025  
**arXiv**: [2410.17193](https://arxiv.org/abs/2410.17193)  
**Code**: [https://github.com/NUS-HPC-AI-Lab/EDF](https://github.com/NUS-HPC-AI-Lab/EDF)  
**Area**: Model Compression  
**Keywords**: Dataset Distillation, Discriminative Features, Grad-CAM, Complex Scenarios, Common Pattern Dropout

## TL;DR
Proposes the EDF method to address the performance degradation of dataset distillation in complex scenarios (ImageNet subsets). It introduces Common Pattern Dropout (discarding parameter gradients of low-loss common patterns in trajectory matching) and Discriminative Area Enhancement (utilizing Grad-CAM to scale up gradients of discriminative regions), achieving lossless compression on datasets such as ImageMeow/ImageYellow with only 23% of the data.

## Background & Motivation

**Background**: Dataset distillation has achieved nearly lossless compression on simple datasets (CIFAR, MNIST), but its performance drops sharply in complex scenarios (ImageNet and its subsets).

**Limitations of Prior Work**: Grad-CAM analysis reveals that while discriminative regions occupy most pixels in simple datasets, they are very small in complex scenarios, where non-discriminative features (background, common colors) dominate the learning process. The low-loss supervision signals in trajectory matching contain common patterns, which conversely dilute discriminative information.

**Key Challenge**: The distillation process indiscriminately matches all parameter gradients, causing synthetic images to be dominated by common patterns (e.g., background textures) while diluting discriminative features (e.g., object details).

**Goal**: To emphasize discriminative features and suppress common patterns within the trajectory matching framework, thereby recovering the performance of dataset distillation in complex scenarios.

**Key Insight**: Simultaneously enhance the learning of discriminative features from two dimensions: the parameter space (discarding gradients of common patterns) and the pixel space (amplifying gradients of discriminative regions).

**Core Idea**: Discarding low-loss common pattern gradients in the parameter space and scaling up gradients of discriminative regions via Grad-CAM weighting in the pixel space, establishing a two-pronged approach to intensify the learning of discriminative features during distillation.

## Method

### Overall Architecture
Based on trajectory matching (e.g., DATM), two modules are integrated during the optimization of synthetic data: CPD filters out common patterns in the parameter space, while DAE enhances discriminative regions in the pixel space. They complement each other: CPD removes interference from a subtractive perspective, whereas DAE amplifies signals from an additive perspective.

### Key Designs

1. **Common Pattern Dropout (CPD)**:

    - **Function**: Filters out common pattern signals within trajectory matching from the parameter space.
    - **Mechanism**: Decomposes the trajectory matching loss into parameter-wise losses $L = \{l_1, l_2, ..., l_P\}$, sorts them in ascending order, and discards the gradients of the lowest $\lfloor \alpha \cdot P \rfloor$ parameters. Since low-loss parameters correspond to fully learned common patterns (e.g., background), discarding them ensures that only gradients of high-loss (discriminative) features are backpropagated to the synthetic images. Optimal dropout ratio: 12.5-25% for small IPC, 37.5-50% for large IPC.
    - **Design Motivation**: Low-loss parameters contain easily learnable common patterns, whose gradients tend to dilute discriminative signals; discarding them focuses the optimization on truly distinctive features.

2. **Discriminative Area Enhancement (DAE)**:

    - **Function**: Amplifies the gradients of discriminative regions in the pixel space.
    - **Mechanism**: Periodically computes the Grad-CAM activation map $M$ of synthetic images, defining a pixel-level gradient weight function $\mathcal{F}(M, \beta)$—the weight for pixels with activation values below the mean is 1 (unchanged), while the weight for pixels above the mean is $\beta + M_{h,w}$ (amplified). The gradient of the synthetic image is rescaled as: $(\nabla D_{syn})_{edf} = \nabla D_{syn} \odot \mathcal{F}(M, \beta)$. A dynamic mean threshold is used instead of a fixed threshold, with $\beta \in [1, 2]$ being optimal.
    - **Design Motivation**: In complex scenarios, discriminative regions are small but have high information density. Scaling up the gradients of these regions focuses the optimization of synthetic images on key details.

3. **Comp-DD Benchmark**:

    - **Function**: Standardizes the evaluation of dataset distillation in complex scenarios.
    - **Mechanism**: Constructs 16 subsets (8 easy, 8 hard) from ImageNet-1K, covering categories like Bird, Car, Dog, Fish, Snake, Insect, Round, and Music, using the Grad-CAM activation area percentage as the complexity score.

### Loss & Training
Based on the trajectory matching loss of DATM, utilizing CPD to discard gradients of low-loss parameters and DAE to rescale pixel gradients. Grad-CAM activation map update frequency: every 50 iterations for small IPC, and every 200 iterations for large IPC.

## Key Experimental Results

### Main Results

| Dataset | IPC=10 | IPC=50 | vs DATM Gain |
|--------|--------|--------|------------|
| ImageWoof | 41.8% | 60.8% | +2.6%/+3.0% |
| ImageMeow | 52.6% | 55.0% | +3.7%/+2.1% |
| ImageYellow | 68.2% | 75.8% | +3.1%/+3.4% |
| ImageSquawk | 65.4% | 77.2% | +3.2%/+2.8% |
| CIFAR-10 | - | 77.3% | +1.2% |
| Tiny-ImageNet | 32.5% | 41.1% | +1.4% |

Lossless compression: ImageMeow reaches 65.2% (= full dataset performance) at IPC=300, requiring only 23% of the data.

### Ablation Study

| Configuration | ImageWoof/Meow/Yellow IPC=10 |
|------|---------------------------|
| Baseline (DATM) | 39.2 / 48.9 / 65.1 |
| +DAE only | 40.3 / 49.5 / 66.2 |
| +CPD only | 41.1 / 51.2 / 67.5 |
| +Both (EDF) | **41.8 / 52.6 / 68.2** |

The contribution of CPD (+1.9~2.3) is larger than that of DAE (+1.1~0.6), and their combination yields a synergistic effect (+2.6~3.1).

### Key Findings
- CPD is the primary contributor, indicating that common pattern filtering in the parameter space is more critical than enhancement in the pixel space.
- The dynamic mean threshold consistently outperforms fixed thresholds (0.2/0.5/0.8) because the activation maps continuously change during training.
- Performance gains are also achieved on simple datasets (CIFAR-10 +1.2%), demonstrating that the common pattern issue is not restricted to complex scenarios.
- A CPD ratio above 75% becomes detrimental, suggesting that some common patterns remain necessary for learning.

## Highlights & Insights
- **Explaining Distillation Failure from a Grad-CAM Perspective**: Identifies that the small proportion of discriminative regions in complex scenarios is the root cause of distillation degradation, offering a fresh perspective for the entire DD community.
- **Simplicity and Effectiveness of Parameter Space Filtering**: CPD only requires sorting losses and discarding the lowest fraction, introducing zero extra parameters and working out-of-the-box.
- **Contribution of the Comp-DD Benchmark**: Provides a standardized evaluation tool for research on DD in complex scenarios.

## Limitations & Future Work
- The CPD dropout ratio $\alpha$ needs to be tuned according to the IPC, lacking an adaptive setting mechanism.
- The computation of Grad-CAM increases training overhead, particularly under frequent updates.
- Verified only on trajectory matching frameworks; integration with distribution-matching-based methods remains unexplored.

## Related Work & Insights
- **vs DATM**: EDF integrates CPD+DAE on top of DATM, boosting performance by 2-4% across all ImageNet subsets. The method is orthogonal and plug-and-play.
- **vs NCFM/CCFS**: While these methods improve DD from the perspective of distribution matching or data selection, EDF enhances it via feature distinctiveness, offering complementary strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Grad-CAM analysis and parameter-level dropout is unique, and the insight is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed ablations across multiple datasets and the Comp-DD benchmark.
- Writing Quality: ⭐⭐⭐⭐ The analysis of the problem (via the Grad-CAM perspective) is highly persuasive.
- Value: ⭐⭐⭐⭐ A plug-and-play DD enhancement module with strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[CVPR 2025\] Enhancing Dataset Distillation via Non-Critical Region Refinement](enhancing_dataset_distillation_via_non-critical_region_refinement.md)
- [\[CVPR 2025\] Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)
- [\[CVPR 2025\] DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)

</div>

<!-- RELATED:END -->
