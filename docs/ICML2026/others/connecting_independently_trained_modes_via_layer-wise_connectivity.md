---
title: >-
  [Paper Note] Connecting Independently Trained Modes via Layer-Wise Connectivity
description: >-
  [ICML2026][Mode Connectivity] Ours proposes the Low-Loss Path Finding (LLPF) algorithm, which reliably constructs low-loss paths between independently trained neural network models through layer-wise connectivity and var…
tags:
  - "ICML2026"
  - "Mode Connectivity"
  - "Loss Landscape"
  - "Variance Sphere"
  - "Layer-wise Connectivity"
  - "Low-loss Path"
date: 2026-05-08
content_hash: 0e88d128b1c07a54
---

# Connecting Independently Trained Modes via Layer-Wise Connectivity

**Conference**: ICML2026  
**arXiv**: [2505.02604](https://arxiv.org/abs/2505.02604)  
**Code**: https://github.com/twoentartian/DFL_torch  
**Area**: Optimization Theory  
**Keywords**: Mode Connectivity, Loss Landscape, Variance Sphere, Layer-wise Connectivity, Low-loss Path  

## TL;DR

Ours proposes the Low-Loss Path Finding (LLPF) algorithm, which reliably constructs low-loss paths between independently trained neural network models through layer-wise connectivity and variance sphere constraints. It supports modern architectures such as MobileNet, EfficientNet, and CCT, with highly reproducible results.

## Background & Motivation

**Background**: Mode connectivity is a significant discovery in recent loss landscape research—two independently trained low-loss models can be connected by a continuous path along which all intermediate models maintain low loss. Existing methods such as FGE (Bézier curve fitting) and AutoNEB (progressive bending of linear interpolation) have laid the foundation for this direction.

**Limitations of Prior Work**: The original training scripts of FGE contain bugs, restricting it to connecting modes that are already close in the weight space. AutoNEB lacks reliability, with the maximum training loss along the path fluctuating drastically from 0.5 to 1.5 across four repeated experiments. Furthermore, these methods have only been validated on relatively aging architectures like basic CNNs, VGG, and ResNet, leaving their applicability to modern architectures like MobileNet, EfficientNet, and CCT unknown.

**Key Challenge**: Linear interpolation between two independent models in the full parameter space typically generates high-loss barriers. However, layer-wise analysis reveals that two models may be linearly connected within the parameter space of a single layer—the root cause of global disconnectivity lies in the coupling effects between layers.

**Goal**: To design a universal and reproducible mode connectivity algorithm capable of spanning independently trained models across different architectures and training hyperparameters.

**Key Insight**: The authors adopt a geometric perspective of "variance spheres," observing that independently trained models exhibit approximately equal parameter variance at each layer. Consequently, models can move layer-by-layer under the constraint of these variance spheres to avoid the issue of variance collapse.

**Core Idea**: The global mode connectivity problem is decomposed into layer-wise local movements. Combined with variance correction projection and minimal SGD training steps, low-loss paths are reliably constructed on variance spheres.

## Method

### Overall Architecture

LLPF consists of two complementary algorithms: **LLPF_M2M** (Model-to-Model) connects two models on the same variance sphere; **LLPF_M2O** (Model-to-Origin) pushes a model toward the origin along a low-loss path to achieve cross-variance sphere connectivity. For models located on different variance spheres (e.g., trained with different weight decay), M2O is first used to reach the target variance sphere, followed by M2M to complete the final connection on that sphere.

### Key Designs

1.  **Variance Sphere Constraint and Variance Correction**

    Independently trained models have approximately equal per-layer parameter variance ($\text{Var}(\theta_n) \approx \text{Var}(\theta_n')$). The variance sphere is defined as $S_{\text{var}=v} = \{P_{l_x} \in \mathbb{R}^{d_{l_x}} \mid \text{Var}(P_{l_x}) = v\}$. When taking a weighted average of two models, parameter variance decreases (variance collapse), making subsequent training difficult. Variance correction re-projects parameters back onto the variance sphere via scaling: $W'[i] = \bar{W} + \sqrt{v / \sigma_W^2} \cdot (W[i] - \bar{W})$, ensuring parameters remain on the correct variance surface after each step.

2.  **Layer-wise Connectivity and Follow Data Flow (FDF) Strategy**

    The core of LLPF_M2M lies in layer-wise operations rather than moving all parameters simultaneously. The processing order follows the FDF strategy: (1) layers are processed sequentially from shallow to deep according to the data flow; (2) parallel layers (such as attention modules) can be processed individually in any order but must be completed before entering downstream layers. This strategy is critical—moving all layers simultaneously works for simple models but fails on complex architectures like ResNet18 or DLA.

3.  **Angle-Conformal Learning Rate Scaling**

    When LLPF_M2O crosses different variance spheres, using the original learning rate directly can cause models on smaller spheres to deviate from the low-loss path. AngleConformal scales the learning rate by the variance ratio: $\eta = \eta_{\text{base}} \cdot w / v$, where $w$ is the current variance and $v$ is the reference variance. This ensures that the "angular displacement" of SGD updates remains consistent across variance spheres of different radii.

## Key Experimental Results

| Method | Supported Architectures | Consistency | Cross-sphere | Worst Training Loss (CIFAR10) |
| :--- | :--- | :--- | :--- | :--- |
| AutoNEB | Basic CNN, ResNet, DenseNet | Inconsistent | Not Reported | 0.0324 (ResNet20) |
| FGE | ResNet, VGG, WideResNet | Not Reported | Not Reported | 0.022 (ResNet158) |
| **LLPF** | **+MobileNet, ShuffleNet, EfficientNet, RegNet, DLA, CCT** | **Consistent** | **Supported** | **0.006 (ResNet18)** |

| Experimental Setup | Training Loss | Test Accuracy | Repetitions |
| :--- | :--- | :--- | :--- |
| ResNet18@CIFAR10 M2M | < 0.1 (Path Max) | Converges to endpoint levels | $\ge$ 10 times |
| DLA@CIFAR10 M2M | < 0.1 | Consistent convergence | $\ge$ 10 times |
| CCT7@CIFAR10 M2M | < 0.1 | Consistent convergence | $\ge$ 10 times |
| ResNet18 Fine-tuned | **< 0.006** | — | Specific tuning |
| DLA Cross-sphere | Low throughout | High training accuracy | M2O + M2M phases |

Continuity Verification: Linear interpolation between adjacent points on the CCT7 path (50 samples) shows that the training loss remains consistently low, supporting the hypothesis of path continuity.

## Highlights & Insights

- **Clear Geometric Intuition**: The problem of mode connectivity is transformed into layer-wise movement on variance spheres. Each step consists of four intuitive geometric operations: "Move $\to$ Variance Correction $\to$ Minimal Training $\to$ Re-correction."
- **Reproducibility Breakthrough**: In $\ge$ 10 repetitions with different random seeds, the loss/accuracy trajectories of LLPF almost perfectly overlap (minimal standard deviation), a property lacking in AutoNEB and FGE.
- **Strong Architectural Generalization**: Mode connectivity is validated for the first time on MobileNet, ShuffleNet, EfficientNet, RegNet, DLA, and CCT, significantly expanding the scope of this phenomenon.
- **Implicit Global Structure of Loss Landscapes**: The results suggest that all modes trained by SGD may reside on a single, path-connected low-loss manifold. If proven, this conjecture would profoundly alter the understanding of loss landscapes.

## Limitations & Future Work

- The algorithm does not guarantee low test loss—in ResNet18 experiments, while training loss was low, test loss increased, indicating the path might traverse regions with poor generalization.
- LLPF_M2O only supports moving from larger variance spheres to smaller ones; the reverse is infeasible due to gradient explosion.
- There are several hyperparameters ($\text{step}_f, \text{step}_a, \text{step}_c, r$, layer order). Although the authors claim layer selection/order is the primary factor for success, practical tuning still requires experience.
- Validation is currently limited to small datasets like CIFAR10/CIFAR100/ImageNet10; applicability to large-scale models (e.g., ViT-Large, LLMs) remains unknown.
- Lack of theoretical guarantees; the global path connectivity conjecture currently relies solely on empirical evidence.

## Related Work & Insights

- **FGE** (Garipov et al., 2018) connects modes using Bézier curves, a pioneering work, but effectively only connects proximal modes.
- **AutoNEB** (Draxler et al., 2018) uses the NEB method to bend interpolation paths but yields inconsistent results.
- **Layer-wise LMC** (Adilova et al., 2024) introduced the concept of layer-wise linear mode connectivity, providing a theoretical foundation for the layer-wise strategy in this paper.
- **Git Re-Basin** (Ainsworth et al., 2023) achieves connectivity via permutation alignment, which differs from connectivity between independently trained modes.
- The variance sphere perspective in this paper may offer new geometric insights for **model fusion/model soups** and **model aggregation in Federated Learning**.

## Rating

- Novelty: 8/10 — The framework of variance spheres combined with layer-wise movement is novel with clear geometric intuition.
- Experimental Thoroughness: 8/10 — Covers various architectures with $\ge$ 10 repetitions per experiment, providing sufficient consistency validation.
- Writing Quality: 7/10 — Rigorous notation, though the algorithm description is somewhat lengthy.
- Value: 7/10 — Advances the understanding of loss landscapes, though practical application scenarios require further exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](../../ICLR2026/others/entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](../../NeurIPS2025/others/generalized_linear_mode_connectivity_for_transformers.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](../../NeurIPS2025/others/impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](../../ICCV2025/others/lacoot_layer_collapse_through_optimal_transport.md)
- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](../../NeurIPS2025/others/are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)

</div>

<!-- RELATED:END -->
