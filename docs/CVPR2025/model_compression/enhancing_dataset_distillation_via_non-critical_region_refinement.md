---
title: >-
  [Paper Note] Enhancing Dataset Distillation via Non-Critical Region Refinement
description: >-
  [CVPR 2025][Model Compression][Dataset Distillation] This paper proposes a three-stage framework, NRR-DD: using CAM to select low-confidence patches to initialize synthetic images, freezing critical regions while optimizing only non-critical regions to improve information density, and replacing 1000-dimensional soft labels with 2 distance values to achieve 500x storage compression. It achieves 46.1% accuracy on ImageNet-1K with IPC=10 (outperforming RDED by 25.7%)…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Dataset Distillation"
  - "Non-Critical Region Optimization"
  - "Class Activation Mapping"
  - "Soft Label Compression"
  - "Distance Representation"
date: 2026-05-08
content_hash: 6d702bdfe1103621
---

# Enhancing Dataset Distillation via Non-Critical Region Refinement

**Conference**: CVPR 2025  
**arXiv**: [2503.18267](https://arxiv.org/abs/2503.18267)  
**Code**: [https://github.com/tmtuan1307/NRR-DD](https://github.com/tmtuan1307/NRR-DD)  
**Area**: Model Compression  
**Keywords**: Dataset Distillation, Non-Critical Region Optimization, Class Activation Mapping, Soft Label Compression, Distance Representation

## TL;DR
This paper proposes a three-stage framework, NRR-DD: using CAM to select low-confidence patches to initialize synthetic images, freezing critical regions while optimizing only non-critical regions to improve information density, and replacing 1000-dimensional soft labels with 2 distance values to achieve 500x storage compression. It achieves 46.1% accuracy on ImageNet-1K with IPC=10 (outperforming RDED by 25.7%), reducing soft label storage from 120GB to 0.2GB.

## Background & Motivation

**Background**: Large-scale dataset distillation methods can be categorized into two paradigms: those focusing on class-general features (e.g., SRe2L, which learns commonalities but loses details) and those focusing on instance-specific features (e.g., RDED, which selects real patches but lacks class-general commonalities).

**Limitations of Prior Work**: (1) Both paradigms exhibit distinct biases, making it difficult for synthetic images to capture both class-general and instance-specific features simultaneously. (2) The storage overhead for large-scale soft labels is prohibitively high—ImageNet-1K with IPC=200 requires 120GB to store 1000-dimensional soft labels, which is impractical.

**Key Challenge**: The real patches selected by RDED already contain rich instance features, but the non-critical regions (e.g., background) are underutilized. Meanwhile, storing 1000-dimensional soft labels serves as a bottleneck for large-scale dataset distillation deployment.

**Goal**: To enhance the learning of class-general features while preserving the instance-specificity of real patches, and to resolve the storage bottleneck of soft labels.

**Key Insight**: Utilize CAM to identify critical and non-critical regions, freeze the critical regions (preserving instance features), optimize non-critical regions (injecting class-general features), and replace full-dimensional soft labels with 2 distance values.

**Core Idea**: Freeze high-CAM regions to preserve instance features, optimize low-CAM regions to inject class-general knowledge, and compress soft labels by 500x using a distance-based representation.

## Method

### Overall Architecture
The framework consists of three stages: (1) In the CIDD stage, CAM is used to select patches with the lowest confidence (rather than the highest) to compose synthetic images. (2) In the NRR stage, high-activation regions of CAM are frozen, and gradient descent optimization (using cross-entropy and BN regularization) is applied only to the non-critical regions, enabling the non-critical regions of low-confidence patches to learn class-general features. (3) In the DBR stage, a teacher model calculates the cross-entropy distance between the soft label of each synthetic image and the original/augmented one-hot labels, saving only 2 distance values instead of the 1000-dimensional vector.

### Key Designs

1. **CAM-based Low-Confidence Initialization (CIDD)**:

    - **Function**: Select patch combinations with the most potential for optimization to initialize synthetic images.
    - **Mechanism**: CAM is utilized to identify highly activated regions in the original image and extract the top-t patches. However, unlike RDED, this method selects the patches with the lowest confidence instead of the highest. Then, \beta low-confidence patches are stitched together to form a synthetic image. Low-confidence patches contain discriminative features that the teacher model is less certain about, thereby providing the maximum optimization space during the subsequent NRR stage.
    - **Design Motivation**: Highly confident patches are already well-recognized by the model and leave little room for optimization. Low-confidence patches contain more non-critical regions that can be refined.

2. **Non-Critical Region Refinement (NRR)**:

    - **Function**: Inject class-general features while preserving instance-specific features.
    - **Mechanism**: A non-critical region mask $M = \max\{0, \epsilon - C\}$ is generated using CAM. Gradient updates are applied only to the pixels covered by the mask: $\tilde{x} = \tilde{x} - M \times \eta \nabla_{\tilde{x}} \mathcal{L}_C$. The loss function includes cross-entropy (urging low-confidence samples to move toward high-confidence space) and BN statistics regularization (aligning with the running statistics of the original model).
    - **Design Motivation**: Critical regions already possess instance-specific features that should not be corrupted. Non-critical regions (e.g., backgrounds) can be treated as a "blank canvas," where optimization can inject general features beneficial for identifying the target class.

3. **Distance-Based Representation for Soft Label Compression (DBR)**:

    - **Function**: Compress 1000-dimensional soft labels into 2 scalars.
    - **Mechanism**: The cross-entropy distance $d_{org}$ between the teacher's soft label and the original one-hot label is computed, as well as the distance $d_{aug}$ for the augmented one-hot label. During student training, a distance matching loss replaces direct Knowledge Distillation (KD), adjusting the student's prediction distance to one-hot labels to match that of the teacher's. Only 2 float values need to be stored per sample.
    - **Design Motivation**: Storing full-dimensional soft labels for ImageNet-1K at IPC=200 requires 120GB, which is impractical. Storing 2 distance values can recover 60-71% of the soft label performance while requiring only 0.2GB of storage.

### Loss & Training
During the NRR stage: $\mathcal{L}_C = \mathcal{L}_{ce} + \alpha_{bn}\mathcal{L}_{bn}$. During the student training stage: $\mathcal{L}_S = \mathcal{L}_{sce} + \alpha_{dbr}\mathcal{L}_{dbr}$.

## Key Experimental Results

### Main Results

| Dataset | IPC | NRR-DD | RDED | SRe2L | Gain (vs RDED) |
|--------|-----|--------|------|-------|-------------|
| CIFAR-10 | 10 | 72.2% | 50.2% | 29.3% | +22.0% |
| CIFAR-100 | 10 | 62.7% | 48.1% | 27.0% | +14.6% |
| Tiny-ImageNet | 50 | 61.2% | 47.6% | 41.1% | +13.6% |
| ImageNet-1K | 10 | 46.1% | 20.4% | 21.3% | +25.7% |
| ImageNet-1K | 50 | 60.2% | 38.4% | 46.8% | +21.8% |

### Soft Label Compression

| Method | ImageNet-1K IPC=50 | Storage | Description |
|------|-----------------|------|------|
| Full Soft Labels | 60.2% | 120GB | Baseline |
| DBR (2 Distances) | 45.1% | 0.2GB | 500x compression, recovers 60% performance |
| One-hot | 32.4% | ~0 | Information loss is too high |

### Key Findings
- NRR-DD consistently outpeforms RDED and SRe2L across all datasets and IPC settings, with the performance gap being particularly pronounced on large-scale datasets (ImageNet-1K +25%).
- Selecting low-confidence patches yields better results than high-confidence ones due to the larger optimization space—an counter-intuitive strategy of "the worse, the more potential."
- Non-critical region refinement is the core innovation. Freezing critical regions and optimizing only the background delivers remarkable performance gains, indicating that background/non-critical region information is highly critical for classification.
- Under a massive compression ratio (500x), the DBR 2-distance scheme still retains a substantial amount of soft label information.

## Highlights & Insights
- **Counter-Intuitive Design of "Freezing Subject, Optimizing Background"**: Traditional approaches focus on how to better extract features of the main subject. In contrast, NRR-DD discovers that backgrounds and non-critical regions also carry rich discriminative information for classes.
- **Deep Reason Behind Low-Confidence Selection**: Low-confidence patches are not noise—they contain discriminative features but in non-typical modes. After NRR optimization, they transform into "good samples with unique perspectives."
- **Elegance of DBR Soft Label Compression**: Compressing 1000-dimensional information into 2 distance values essentially retains the most crucial knowledge distillation signal: "how far the soft label is from the one-hot label."

## Limitations & Future Work
- Although DBR achieves a 500x compression, the performance drop is still significant (60% recovery rate). Refining distance representations could further close the gap.
- The quality of CAM depends heavily on the pre-trained model. If the pre-trained model performs poorly, partition of critical/non-critical regions will be inaccurate.
- The number of optimization iterations in the NRR stage requires careful tuning.

## Related Work & Insights
- **vs RDED**: RDED directly selects high-confidence patches without optimization, while NRR-DD selects low-confidence ones and optimizes non-critical regions, raising performance from 50.2% to 72.2% on CIFAR-10 at IPC=10.
- **vs SRe2L series**: SRe2L focuses on class-general features but loses instance information. NRR-DD balances both by preserving critical regions and optimizing backgrounds.
- **vs EDF**: EDF improves discriminative regions from a gradient perspective. NRR-DD freezes/optimizes different regions from a pixel perspective, offering a complementary strategy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of non-critical region refinement is unique; the combination of low-confidence selection and background optimization is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage from CIFAR to ImageNet-1K, backed by exhaustive soft label compression experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structured stages and deep motivation analysis.
- Value: ⭐⭐⭐⭐⭐ Massive performance gains coupled with practical storage compression, driving significant progress in the DD community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Grounding and Enhancing Informativeness and Utility in Dataset Distillation](../../ICLR2026/model_compression/grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)
- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[CVPR 2025\] Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)
- [\[CVPR 2025\] DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)
- [\[CVPR 2025\] Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)

</div>

<!-- RELATED:END -->
