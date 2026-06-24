---
title: >-
  [Paper Note] Do Your Best and Get Enough Rest for Continual Learning
description: >-
  [CVPR 2025][Self-Supervised Learning][Forgetting Curve] Inspired by Ebbinghaus's forgetting curve theory, this paper proposes the View-Batch Model (VBM). By replacing multiple distinct samples in a batch with multiple augmented views (replay) of the same sample, VBM extends the recall interval by a factor of $V$ to an optimal range. Concurrently, it employs a one-to-many KL-divergence self-supervised loss to extract more knowledge from a single sample ("do your best"). Servin…
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Forgetting Curve"
  - "Spacing Effect"
  - "View-Batch"
  - "Recall Interval Optimization"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: 7aa346fc73f04dfd
---

# Do Your Best and Get Enough Rest for Continual Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.18371](https://arxiv.org/abs/2503.18371)  
**Code**: [https://github.com/hankyul2/ViewBatchModel](https://github.com/hankyul2/ViewBatchModel)  
**Area**: Self-Supervised Learning / Continual Learning  
**Keywords**: Forgetting Curve, Spacing Effect, View-Batch, Recall Interval Optimization, Plug-and-Play

## TL;DR

Inspired by Ebbinghaus's forgetting curve theory, this paper proposes the View-Batch Model (VBM). By replacing multiple distinct samples in a batch with multiple augmented views (replay) of the same sample, VBM extends the recall interval by a factor of $V$ to an optimal range. Concurrently, it employs a one-to-many KL-divergence self-supervised loss to extract more knowledge from a single sample ("do your best"). Serving as a drop-in replacement, VBM consistently improves performance across various continual learning methods.

## Background & Motivation

**Background**: The core problem of continual learning is catastrophic forgetting. Rehearsal methods (such as ER, DER++, and iCaRL) utilize memory buffers to replay old samples, but the recall interval (the gap between training steps on the same sample) remains unoptimized.

**Limitations of Prior Work**: The recall interval in current methods is computed as $\text{batch size} \times \text{training steps} = \text{dataset size}$, which is typically too short. Repeating training on the same sample over a short interval is inefficient according to forgetting curve theories. The optimal recall interval should be sufficiently long (but not too long) to elicit the "spacing effect" for enhanced long-term memory retention.

**Key Challenge**: Extending the recall interval implies that each sample is trained fewer times. How can more knowledge be extracted from each sample while extending the interval?

**Core Idea**: (1) Replace sample-batches with view-batches (containing $V$ augmented views of the same sample), which automatically extends the recall interval by $V$ times. (2) Use a self-supervised loss (the KL divergence between weak and strong augmented views) to learn more from each individual sample. The total number of epochs is reduced by $V$ times to keep the overall computational cost constant.

## Method

### Overall Architecture

Original scheduler: $\mathcal{A} = [\mathcal{B}_1^I, ..., \mathcal{B}_T^I, \mathcal{B}_1^I, ...]$, recall interval = $B \times T$  
VBM scheduler: $\mathcal{A} = [\mathcal{B}_1^V, ..., \mathcal{B}_T^V, \mathcal{B}_1^V, ...]$, recall interval = $B \times T \times V$  
where $\mathcal{V}_i = \{I_i\}_{j=1}^V$ ($V$ augmented views of the same sample). The total number of epochs is reduced by $V$ times.

### Key Designs

1. **View-Batch Replay**:
    - **Function**: Extends the recall interval to the optimal range.
    - **Mechanism**: The batch size $B$ remains unchanged, but each slot is populated with different augmented views of the same sample rather than different samples. Consequently, the actual number of unique samples is reduced to $B/V$, and the recall interval is extended by a factor of $V$.
    - The first view uses weak augmentation (horizontal flipping), while the remaining $V-1$ views use strong augmentation (AutoAugment).
    - Empirical validation shows that when $V=4$, the recall interval lies in the optimal range, yielding the slowest memory retention decay.

2. **One-to-Many Self-Supervised Loss**:
    - **Function**: Learns more knowledge from multiple views of a single sample.
    - **Mechanism**: Calculates $L_{ssl} = \frac{1}{B \cdot (V-1)} \sum_{i=1}^B \sum_{j=2}^V D_{KL}(p_i^1 \| p_i^j)$, where the logit distribution of the weakly augmented view is set as the target, minimizing its KL divergence with the strongly augmented views.
    - **Design Motivation**: It requires no additional architecture (such as teacher networks) and merely applies a consistency constraint at the logit level. Task-agnostic knowledge tends to be more robust against forgetting.

3. **Drop-in Replacement Design**:
    - Modifies only the data loading workflow and the loss function without altering the model architecture or increasing training epochs.
    - Total number of forward passes remains equivalent to the original method (reduction of epochs by $V \times$ multiple views per step of $V$ = balanced).
    - Can be combined with any rehearsal or rehearsal-free method.

## Key Experimental Results

### Main Results: VBM integration with various CL methods (S-CIFAR-10, buffer=200)

| Method | Original CIL/TIL Avg | VBM CIL/TIL Avg | ΔAvg |
|------|:-:|:-:|:-:|
| LwF (buffer=0) | -/62.0=62.0 | -/77.5=77.5 | **+15.6** |
| ER | 50.3/91.7=71.0 | 52.6/93.6=73.1 | +2.1 |
| iCaRL | 64.1/90.2=77.2 | 69.7/92.8=81.3 | **+4.1** |
| DER++ | 61.7/90.6=76.1 | 67.0/94.3=80.7 | **+4.5** |

### Ablation Study: Forgetting Curve Validation

Fig. 4 empirically validates the applicability of the forgetting curve theory to neural networks:
- $V=1$ (short recall interval): steep memory retention decay, leading to severe forgetting.
- $V=4$ (optimal recall interval): gentlest decay, achieving the best long-term memory retention.
- $V=16$ (excessive recall interval): although the decay rate is gentle, too much is forgotten initially, degrading overall performance.

### Key Findings

- **Consistent Improvement Across All Configurations**: Fig. 2 demonstrates that VBM consistently provides positive gains across different step sizes, buffer sizes, baseline methods, pre-trained models, benchmarks, and protocols, with zero negative cases.
- **Most Substantial Gain on Rehearsal-Free Methods**: LwF improves by $+15.6\%$ (with zero buffer, achieving the maximum benefit from recall interval optimization).
- **Compatible with Pre-trained Models**: Also effective when integrated with methods based on pre-trained ViTs (such as CODA-Prompt).
- **Ebbinghaus's Theory Holds in Neural Networks**: The shape of the forgetting curve in Fig. 4 aligns with empirical human psychology research.

## Highlights & Insights

- **Elegant Integration of Theory and Practice**: Successfully transfers a psychology theory from over 120 years ago (Ebbinghaus's forgetting curve and the spacing effect) to continual learning in neural networks, empirically validating its applicability to deep learning.
- **Minimalistic Yet Effective**: Requires no new architectures, optimizers, or newly designed losses—just an adjusted data schedule and a simple KL-divergence loss.
- **Intuitive Analogy of "Do your best AND get enough rest"**: A student (the model) should study as deeply as possible during each session (SSL) but also needs sufficient intervals to consolidate memory. This analogy is highly apt.

## Limitations & Future Work

- The optimal value of $V$ must be adjusted depending on the dataset and task, lacking systematic theoretical guidance.
- The use of KL divergence for self-supervised loss is relatively simple; more sophisticated consistency constraints could be explored.
- The definition of the "optimal" recall interval is experimental and lacks a formal analysis in neural networks.
- Evaluated only on classification tasks; not yet extended to other continual learning scenarios such as object detection or segmentation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Application of forgetting curve theory to CL is highly novel, and view-batch replay is simple yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Fig. 2 covers 6 dimensions (steps, buffer sizes, methods, pre-training status, benchmarks, and protocols), offering a comprehensive validation.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical motivations are clear, and the visualization of the forgetting curve in Fig. 1 is highly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Drop-in replacement, zero overhead, and consistent gains—yielding immediate benefits to CL researchers and practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)
- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)
- [\[CVPR 2026\] How Much 3D Do Video Foundation Models Encode?](../../CVPR2026/self_supervised/how_much_3d_do_video_foundation_models_encode.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](../../ICML2025/self_supervised/update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)
- [\[NeurIPS 2025\] Continuous Subspace Optimization for Continual Learning (CoSO)](../../NeurIPS2025/self_supervised/continuous_subspace_optimization_for_continual_learning.md)

</div>

<!-- RELATED:END -->
