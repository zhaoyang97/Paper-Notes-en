---
title: >-
  [Paper Note] Reframing Long-Tailed Learning via Loss Landscape Geometry
description: >-
  [CVPR 2026][LLM Evaluation][Long-tailed learning] This paper reframes the head-tail seesaw dilemma in long-tailed learning through the lens of loss landscape geometry. It identifies that tail class degradation stems from…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Long-tailed learning"
  - "loss landscape"
  - "tail class degradation"
  - "continual learning"
  - "sharpness-aware minimization"
date: 2026-05-08
content_hash: c06b35825d8b2aa9
---

# Reframing Long-Tailed Learning via Loss Landscape Geometry

**Conference**: CVPR 2026
**arXiv**: [2603.21217](https://arxiv.org/abs/2603.21217)  
**Code**: [https://gkp-gsa.github.io/](https://gkp-gsa.github.io/)  
**Area**: Long-Tailed Learning / Visual Classification
**Keywords**: Long-tailed learning, loss landscape, tail class degradation, continual learning, sharpness-aware minimization

## TL;DR
This paper reframes the head-tail seesaw dilemma in long-tailed learning through the lens of loss landscape geometry. It identifies that tail class degradation stems from optimization converging to sharp minima that are far from tail-class optima. A dual-module framework comprising GKP (Grouped Knowledge Preservation) and GSA (Grouped Sharpness Aware) is proposed based on continual learning principles, achieving state-of-the-art results on four benchmarks (CIFAR-LT / ImageNet-LT / iNat2018) without requiring additional data.

## Background & Motivation

1. **Background**: Long-tailed learning is a longstanding challenge in computer vision. Existing methods fall into three main categories: (1) class rebalancing (resampling / reweighting), (2) information augmentation (data augmentation / synthesis), and (3) module redesign (specialized network architectures). Recent trends involve leveraging external data or large-scale models, which is infeasible in privacy-sensitive domains such as medical imaging.
2. **Limitations of Prior Work**: Nearly all methods suffer from the head-tail seesaw dilemma — improving tail-class performance inevitably degrades head-class performance and vice versa. The underlying cause of this trade-off has received little attention.
3. **Key Challenge**: Loss landscape visualization reveals two key phenomena: (a) *Tail class performance degradation* — the convergence point $\theta(t_2)$ under standard training drifts far from the tail-class optimum $\theta(t_1)$, causing the model to overfit head classes while forgetting tail classes; (b) *Sharp minima* — standard long-tailed training converges to sharper regions compared to the flatter regions reached when training exclusively on tail classes, resulting in poor generalization.
4. **Goal**: (1) Prevent tail-class knowledge from being forgotten during training; (2) Guide optimization toward flat minima to improve cross-class generalization.
5. **Key Insight**: Long-tailed learning is recast as a continual learning (CL) problem — when head-class gradients dominate training, tail-class knowledge is progressively "forgotten," analogous to catastrophic forgetting in CL. EWC-style knowledge preservation is adopted to prevent forgetting, and SAM-style sharpness awareness is employed to find flat regions.
6. **Core Idea**: Long-tailed learning is treated as continual learning from head to tail classes. Grouped knowledge preservation prevents forgetting, and grouped sharpness awareness finds flat solutions. Together, they guide optimization toward shared flat minima that benefit all classes.

## Method

### Overall Architecture
The framework consists of two branches: (1) the **GKP branch** for knowledge preservation — applying EWC-style parameter regularization to prevent forgetting other groups when training on a given group; (2) the **GSA branch** for knowledge acquisition — applying grouped SAM to find flat minima for each group after removing the head-class-dominated direction. The losses from both branches are aggregated via an adaptive weight $\alpha$. Prior to training, all classes are partitioned into $G$ groups using a memory-based grouping strategy.

### Key Designs

1. **Memory-based Grouping Strategy**:

    - **Function**: Clusters classes by their convergence characteristics, providing the grouping basis for GKP and GSA.
    - **Mechanism**: (1) Construct a memory bank $\mathcal{M}$: dynamically record the encoder parameters $\theta_{enc}^c$ at which each class $c$ attains its highest feature quality $Q$ during training, where $Q$ is defined based on inter-class separation and intra-class variance. (2) Cluster into groups: apply spectral clustering (NCut algorithm) to partition the $C$ class parameters $\{\theta_{enc}^c\}$ into $G$ groups by similarity — classes with similar parameters share convergence requirements and are treated as one "task." (3) Compute the shared parameters for each group: $\theta_g^* = \frac{1}{|\mathcal{G}^g|}\sum_{c \in \mathcal{G}^g} \theta_{enc}^c$.
    - **Design Motivation**: Per-class preservation is computationally prohibitive and over-constrains optimization; a simple head-tail split is too coarse and ignores within-group variation. Grouping by convergence parameter similarity captures the intrinsic structure of which classes benefit from being optimized together.

2. **Grouped Knowledge Preservation (GKP)**:

    - **Function**: Prevents the optimal parameters of other groups from being overwritten when training on a given group.
    - **Mechanism**: Following the EWC paradigm, when the model trains on current group $g$, a parameter deviation penalty is imposed for all other groups $j \neq g$: $\mathcal{L}_{gkp}^g = \frac{\lambda}{2}\sum_i \sum_{j \neq g} \frac{1}{|\mathcal{G}^j|} F_{j,i}(\theta_i - \theta_{j,i}^*)^2$, where $F_{j,i}$ is the diagonal element of the Fisher information matrix for group $j$, and $\theta_{j,i}^*$ is the shared parameter of group $j$. Normalization by $1/|\mathcal{G}^j|$ balances the importance of each group.
    - **Design Motivation**: In long-tailed training, the optimal parameters for tail classes are overwritten by head-class-dominated gradients — analogous to new tasks overwriting old task knowledge in CL. GKP alleviates this by preserving each group's historical optimal parameters and constraining the current optimization to remain close to them.

3. **Grouped Sharpness Aware (GSA)**:

    - **Function**: Finds flat minima for each group by removing the head-class-dominated perturbation direction.
    - **Mechanism**: (1) Compute per-group gradients $\nabla_\theta \mathcal{L}_{D_g}(\theta)$; (2) Remove the projection onto the global gradient direction via gradient decomposition: $\hat{\nabla}_\theta \mathcal{L}_{D_g}(\theta) = \nabla_\theta \mathcal{L}_{D_g}(\theta) - \text{Proj}_{\nabla_\theta \mathcal{L}_D(\theta)} \nabla_\theta \mathcal{L}_{D_g}(\theta)$, yielding the group-specific gradient direction; (3) Adjust the perturbation radius $\rho_g^*$ based on group size; (4) Compute the SAM perturbation using the group-specific gradient and radius: $\hat{\epsilon}_g^*(\theta) = \sqrt{d}\rho_g^* \frac{\hat{\nabla}_\theta \mathcal{L}_{D_g}(\theta)}{\|\hat{\nabla}_\theta \mathcal{L}_{D_g}(\theta)\|_2}$.
    - **Design Motivation**: Standard SAM's global perturbation direction is dominated by head-class gradients and insensitive to the high-sharpness regions of tail classes. By removing the head-class-dominated global direction, GSA focuses the perturbation on each group's own optimization needs, enabling tail classes to also converge to flat minima.

### Loss & Training
- Total loss: $\mathcal{L} = \sum_{g=1}^G [\alpha \mathcal{L}_{gsa}^g + (1-\alpha)\mathcal{L}_{gkp}^g]$
- $\alpha$ is an adaptive parameter scheduled across training epochs.
- Default number of groups: $G = 4$
- Backbones: ResNet-32 (CIFAR), ResNet-50 / ResNeXt-50 (ImageNet-LT / iNat)
- Batch size 256, NVIDIA 3090 GPU

## Key Experimental Results

### Main Results — CIFAR100-LT

| Method | r=100 | r=50 | r=10 | Many | Med. | Few |
|--------|-------|------|------|------|------|-----|
| CE Baseline | 38.3 | 43.9 | 55.7 | 65.2 | 37.1 | 9.1 |
| BCL (CVPR'22) | 51.9 | 56.6 | 64.9 | 67.2 | 53.1 | 32.9 |
| GBG (AAAI'24) | 52.3 | 57.2 | - | - | - | - |
| FeatRecon (ICLR'25) | 52.5 | 57.0 | 65.3 | - | - | - |
| LLM-AutoDA† | 51.0 | 54.8 | - | 66.6 | 50.6 | 33.1 |
| **Ours** | **53.2** | **57.6** | **68.7** | 67.3 | **54.9** | **34.9** |

### Main Results — ImageNet-LT & iNaturalist

| Method | ImageNet-LT (ResNet-50) | iNat2018 |
|--------|-------------------------|----------|
| BCL | 56.0 | 71.8 |
| GBG | 57.6 | 71.9 |
| FeatRecon | 56.8 | 72.9 |
| LLM-AutoDA† | 57.5 | 74.2 |
| **Ours** | **57.9** | **74.4** |

### Ablation Study

| Configuration | Many | Med. | Few | All |
|---------------|------|------|-----|-----|
| BCL baseline | 67.2 | 53.1 | 32.9 | 51.9 |
| + GKP | 67.4 | 53.8 | 33.2 | 52.4 (+0.5) |
| + GSA | 67.3 | 54.0 | 34.1 | 52.7 (+0.8) |
| + GKP + GSA (full) | 67.3 | **54.9** | **34.9** | **53.2** (+1.3) |

### Importance of Gradient Decomposition

| Perturbation Direction | Many | Med. | Few | All |
|------------------------|------|------|-----|-----|
| SAM (global gradient) | 66.3 | 53.0 | 34.5 | 52.1 |
| GSA-proj (projected component) | 64.7 | 43.8 | 28.1 | 46.4 |
| **GSA (global direction removed)** | **67.3** | **54.9** | **34.9** | **53.2** |

### Key Findings
- **GKP and GSA are complementary**: GKP primarily improves the Med. split (+0.7), while GSA primarily improves the Few split (+1.2). Their combined effect exceeds individual contributions, indicating that knowledge preservation and sharpness reduction address different aspects of the problem.
- **Gradient decomposition is critical**: Using the projected component (head-class-dominated direction) for SAM perturbation causes a severe performance drop (53.2 → 46.4), confirming that the head-class-dominated global gradient is harmful to tail-class optimization. Only the group-specific component with the global direction removed is beneficial.
- **$G=4$ is optimal**: Too few groups ($G=2$) yields coarse-grained partitions, while too many ($G=8+$) increases the number of GKP constraints and restricts optimization freedom.
- **Outperforms LLM-based methods without external data**: The proposed method surpasses LLM-AutoDA† (which relies on large language models to generate augmented data) by 2.2% on CIFAR100-LT, demonstrating that an optimization-centric solution can match or exceed resource-intensive external data approaches.
- **Gradient similarity validates GKP**: Tail-class gradient similarity declines in the late training stages under the baseline (indicating knowledge forgetting), whereas the proposed method maintains consistently high similarity throughout training, directly confirming GKP's knowledge preservation effect.

## Highlights & Insights
- **Reinterpreting long-tailed learning through the loss landscape**: Rather than treating long-tailed learning as a data imbalance problem, this work frames it as an optimization trajectory deviation problem. This perspective shift opens the door to applying CL and SAM methodologies to long-tailed learning in a principled manner and is highly inspiring.
- **The CL-to-LT analogy is precise**: Tail-class knowledge being overwritten by head-class-dominated gradients is equivalent to new tasks overwriting old task knowledge in CL. Since long-tailed learning lacks explicit task boundaries, the memory-based grouping strategy cleverly constructs pseudo-task partitions.
- **GSA's gradient decomposition trick**: Removing the global gradient projection to obtain group-specific perturbation directions is conceptually simple yet highly effective (+7.1% over SAM-proj). This technique can generalize to any scenario requiring SAM under mixed optimization objectives.

## Limitations & Future Work
- The memory bank, which stores optimal encoder parameters $\theta_{enc}^c$ per class, incurs substantial memory overhead when the number of classes is large (requiring $C$ complete copies of encoder parameters).
- The grouping strategy relies on spectral clustering, which itself introduces hyperparameters (choice of $G$, timing of clustering).
- The diagonal approximation of the Fisher information matrix may be insufficiently accurate; better importance estimation could further improve GKP.
- Evaluation is limited to image classification; applicability to long-tailed dense prediction tasks such as detection and segmentation remains unexplored.

## Related Work & Insights
- **vs. SAM / FriendlySAM**: Standard SAM's global perturbation is dominated by head classes and ineffective for tail classes; GSA achieves group-specific perturbation directions through gradient decomposition, constituting a principled improvement of SAM for long-tailed scenarios.
- **vs. BCL**: BCL serves as the primary baseline (same backbone and loss); the proposed method improves upon BCL by 1.3% purely through the GKP + GSA optimization strategy, demonstrating that optimization-centric improvements are orthogonal to and composable with existing methods.
- **vs. GBG (AAAI'24)**: GBG also addresses gradient imbalance but through a different balancing strategy; the proposed approach is more comprehensive by jointly addressing loss landscape geometry and knowledge preservation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Redefining long-tailed learning from a loss landscape perspective; the CL → LT transfer is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, multiple backbones, detailed ablations and analyses (feature quality, gradient similarity, landscape visualization).
- **Writing Quality**: ⭐⭐⭐⭐ — Well-motivated, rich visualizations, clear methodological derivation.
- **Value**: ⭐⭐⭐⭐ — Provides a new paradigm for long-tailed learning without relying on external data; the optimization-centric insights offer broad utility to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](../../NeurIPS2025/llm_evaluation/keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[CVPR 2026\] Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning](flow3r_factored_flow_prediction_for_scalable_visual_geometry_learning.md)
- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[CVPR 2026\] Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)

</div>

<!-- RELATED:END -->
