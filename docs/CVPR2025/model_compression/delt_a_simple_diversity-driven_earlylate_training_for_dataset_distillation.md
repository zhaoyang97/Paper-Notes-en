---
title: >-
  [Paper Note] DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation
description: >-
  [CVPR 2025][Model Compression][Dataset Distillation] Proposes the EarlyLate training strategy, which generates synthetic images of varying difficulty by starting different IPC sub-batches from distinct optimization points and running them for varying numbers of iterations. Within the batch-to-global matching framework, it significantly improves intra-class diversity while reducing computation time by 39.3%, achieving 66.1% accuracy on ImageNet-1K with IPC=50 (using ResNet-101…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Dataset Distillation"
  - "Diversity"
  - "EarlyLate Training"
  - "batch-to-global matching"
  - "Large-scale Datasets"
date: 2026-05-08
content_hash: 52007b9400f14520
---

# DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation

**Conference**: CVPR 2025  
**arXiv**: [2411.19946](https://arxiv.org/abs/2411.19946)  
**Code**: [https://github.com/VILA-Lab/DELT](https://github.com/VILA-Lab/DELT)  
**Area**: Model Compression  
**Keywords**: Dataset Distillation, Diversity, EarlyLate Training, batch-to-global matching, Large-scale Datasets

## TL;DR
Proposes the EarlyLate training strategy, which generates synthetic images of varying difficulty by starting different IPC sub-batches from distinct optimization points and running them for varying numbers of iterations. Within the batch-to-global matching framework, it significantly improves intra-class diversity while reducing computation time by 39.3%, achieving 66.1% accuracy on ImageNet-1K with IPC=50 (using ResNet-101, surpassing RDED by 4.9%).

## Background & Motivation

**Background**: Large-scale dataset distillation mainly falls into two paradigms: batch-to-batch matching (fine-grained but computationally expensive, such as MTT, FRePo) and batch-to-global matching (decoupled training and synthesis, suitable for large datasets, such as SRe2L, CDA). The latter has become mainstream due to its efficiency advantages.

**Limitations of Prior Work**: In batch-to-global matching methods, each synthetic image is optimized independently but shares the same global supervisory signal (e.g., BN statistics), resulting in highly homogenized synthetic images of the same class that lack diversity. G-VBSM mitigates this with multiple models but increases complexity, while RDED uses real image patching without optimization training.

**Key Challenge**: The unified optimization process leads all images to converge to similar local optima, resulting in a severe lack of intra-class diversity.

**Goal**: To enhance the intra-class diversity of synthetic images in a simple and low-cost manner within the batch-to-global matching framework.

**Key Insight**: Approaching from the perspective of early stopping and curriculum learning: different images require different numbers of iterations to converge. Allowing some images to train more (fully optimized, more abstract) and others to train less (retaining more original information, more realistic) naturally produces combinations of images with varying optimization depths.

**Core Idea**: Divide the IPC of the same class into multiple sub-batches and have them join the optimization at different time points, ensuring different sub-batches undergo different numbers of iterations, naturally yielding synthetic images with diverse characteristics.

## Method

### Overall Architecture
The overall pipeline consists of three steps: (1) Using a pre-trained teacher model to rank original image patches by prediction probability, selecting median probability patches to initialize synthetic images; (2) Dividing the IPC into $M$ sub-batches, where the first sub-batch starts optimization from scratch, and subsequent sub-batches are added every $RI$ iterations, with all sub-batches sharing the remaining optimization process (concatenated training); (3) The final synthetic dataset contains images that have undergone different optimization depths, naturally possessing difficulty gradients and diversity.

### Key Designs

1. **EarlyLate Training Strategy**:

    - **Function**: Enhancing intra-class diversity through differentiated optimization iteration counts.
    - **Mechanism**: Assuming the total number of iterations is $T$ (e.g., 4K), divide the IPC into $M$ sub-batches. The 1st sub-batch starts optimization from iteration 0 (undergoing $T$ iterations), the 2nd joins at iteration $RI$ (undergoing $T-RI$ iterations), ..., and the $M$-th joins at iteration $T-RI$ (undergoing only $RI$ iterations). Later-joining sub-batches are concatenated and jointly optimized with the preceding ones. The total iteration volume is reduced from $N \times T$ to approximately $2/3 \times N \times T$.
    - **Design Motivation**: Images optimized for a long time are more abstract and better matched with the teacher model; images optimized for a short time retain more real-image features. Combining both covers the sample space from simple to complex, increasing intra-class diversity.

2. **Teacher-Ranked Initialization**:

    - **Function**: Providing meaningful initialization to replace Gaussian noise.
    - **Mechanism**: Using a pre-trained teacher model to compute the prediction probability of each original image patch, and selecting median-probability patches for initialization. Medium-difficulty images are neither too simple (low information) nor too hard (difficult to optimize), leaving maximum room for information enhancement.
    - **Design Motivation**: In EarlyLate, later-joining sub-batches undergo only a small number of iterations, making a good initialization crucial. Experiments show that median initialization performs better than both lowest and highest.

3. **Concatenation Training**:

    - **Function**: Allowing different sub-batches to share GPU computation time to improve efficiency.
    - **Mechanism**: Newly joined sub-batches are concatenated with existing ones and jointly optimized within the same forward-backward pass. This avoids running separate full training pipelines for each sub-batch, reducing I/O and data loading overhead.
    - **Design Motivation**: In a naive implementation, $M$ sub-batches would require $M$ independent training runs; concatenation training completes the optimization of all sub-batches in a single run.

### Loss & Training
Uses the same BatchNorm distribution regularization loss and soft cross-entropy as CDA/SRe2L. Default configuration: $MI=4K$ iterations, $RI=500$, $M=8$ sub-batches (i.e., 4000 iterations for the 1st batch, 3500 for the 2nd, ..., 500 for the 8th).

## Key Experimental Results

### Main Results

| Dataset | IPC | Model | DELT | RDED | SRe2L | Gain (vs RDED) |
|--------|-----|------|------|------|-------|-------------|
| ImageNet-1K | 50 | ResNet-18 | 46.8% | 42.0% | 46.8% | +4.8% |
| ImageNet-1K | 50 | ResNet-101 | 66.1% | 61.2% | - | +4.9% |
| CIFAR-10 | 50 | ResNet-18 | 82.1% | 73.4% | - | +8.7% |
| CIFAR-10 | 50 | ResNet-101 | 85.2% | 63.5% | 66.0% | +19.2%(vs SRe2L) |
| Tiny-ImageNet | 50 | ResNet-18 | 28.3% | 25.6% | 21.3% | +2.7% |

### Ablation Study

| Configuration | ImageNet-1K IPC50 (R18) | Description |
|------|------------------------|------|
| DELT (Full) | 46.8% | EarlyLate + real image initialization |
| Early-only (equal iterations) | 44.4% | No EarlyLate, degenerates to CDA + initialization |
| No real image initialization | 44.8% | Gaussian noise initialization |
| Highest prob initialization | 46.3% | Simplest samples perform worse than median |
| Lowest prob initialization | 46.0% | Hardest samples perform worse than median |
| MI=1K, RI=500 | 44.1% | Too few iterations |
| MI=4K, RI=500 | 46.8% | Optimal configuration |

### Key Findings
- EarlyLate vs. Early-only yields a critical gain of 2.4%, showing that diversity is indeed the performance bottleneck.
- The intra-class cosine similarity is significantly reduced (>5%), validating that the diversity enhancement is not an artificial metric.
- Computation time is reduced by 39.3% (from 29h to 18.8h on ImageNet-1K) while performance is simultaneously improved.
- In continual learning scenarios, DELT synthetic data outperforms G-VBSM by 10% on average, demonstrating that diversity holds greater value in this context.
- Maintains advantages across different architectures (ResNet, MobileNet, EfficientNet, RegNet), showing strong generalization.

## Highlights & Insights
- **Philosophy of Simplicity**: Diversity is improved solely by varying the optimization entry points for different images, without adding any new modules or loss functions. This concept of "different starting points + different iteration depths = different optimization outcomes" is extremely elegant and can be transferred to any batch-to-global distillation method with zero cost.
- **Positive Correlation Between Diversity and Efficiency**: Enhancing diversity typically demands more computation. However, DELT actually saves 39.3% of computation by reducing the iteration counts of later-joining batches, achieving a win-win in both performance and efficiency.
- **Cross-Task Validation**: The 10% gain in continual learning demonstrates that diversity provides substantial transfer value for downstream tasks.

## Limitations & Future Work
- The choice of hyperparameters $RI$ and $MI$ relies on heuristics, and different datasets may require different configurations.
- At extremely low IPCs (e.g., 1 or 5), the advantage of EarlyLate is less significant because the sub-batches are too small.
- It is orthogonal to the multi-model strategy of G-VBSM, but their combination has not been fully explored.
- Only validated on classification tasks; downstream tasks like detection and segmentation are not yet addressed.

## Related Work & Insights
- **vs. SRe2L/CDA**: Sharing the same batch-to-global matching framework, DELT improves upon CDA by 2–5% simply by changing the training schedule, proving that diversity is the core bottleneck of this class of methods.
- **vs. RDED**: RDED uses real-image patching without training, leading to limited information density; DELT performs differentiated optimization on patches, yielding more thorough info enhancement.
- **vs. G-VBSM**: G-VBSM enhances diversity using multiple models but with a complex framework; DELT uses a single model with temporal differentiation, which is simpler and more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea is extremely simple yet previously unattempted. Utilizing early stopping to enhance distillation diversity is a great insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers CIFAR to ImageNet-1K, multiple architectures, comprehensive ablations, and downstream task validations.
- Writing Quality: ⭐⭐⭐⭐ The methodology description is clear and figures are intuitive.
- Value: ⭐⭐⭐⭐ Simple, effective, and plug-and-play, holding high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Enhancing Dataset Distillation via Non-Critical Region Refinement](enhancing_dataset_distillation_via_non-critical_region_refinement.md)
- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[CVPR 2025\] Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)
- [\[CVPR 2025\] Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)

</div>

<!-- RELATED:END -->
