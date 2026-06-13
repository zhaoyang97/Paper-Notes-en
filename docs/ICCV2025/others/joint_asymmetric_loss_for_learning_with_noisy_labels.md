---
title: >-
  [Paper Note] Joint Asymmetric Loss for Learning with Noisy Labels
description: >-
  [ICCV 2025][learning with noisy labels] This paper extends asymmetric loss functions to the more challenging passive loss setting, proposes Asymmetric Mean Squared Error (AMSE)…
tags:
  - "ICCV 2025"
  - "learning with noisy labels"
  - "asymmetric loss functions"
  - "robust loss"
  - "APL framework"
  - "noise tolerance"
date: 2026-05-08
content_hash: ba78f1d37fcd195f
---

# Joint Asymmetric Loss for Learning with Noisy Labels

**Conference**: ICCV 2025
**arXiv**: [2507.17692](https://arxiv.org/abs/2507.17692)  
**Code**: [github.com/cswjl/joint-asymmetric-loss](https://github.com/cswjl/joint-asymmetric-loss)  
**Area**: Other
**Keywords**: learning with noisy labels, asymmetric loss functions, robust loss, APL framework, noise tolerance

## TL;DR

This paper extends asymmetric loss functions to the more challenging passive loss setting, proposes Asymmetric Mean Squared Error (AMSE), rigorously establishes the necessary and sufficient conditions for AMSE to satisfy the asymmetric condition, and embeds AMSE into the APL framework to construct the Joint Asymmetric Loss (JAL), achieving comprehensive improvements over existing robust loss methods on CIFAR-10/100 and other datasets.

## Background & Motivation

### Problem Definition

Training deep neural networks on data with noisy labels is a critical challenge in machine learning:
- Human annotation inevitably introduces erroneous labels
- Direct supervised learning on noisy labels severely degrades model performance
- It is necessary to design noise-tolerant loss functions such that the risk minimizer on noisy data also minimizes the risk on clean data

Three primary noise types:

**Symmetric noise**: Ground-truth labels are flipped to other classes with uniform probability

**Asymmetric noise**: Noise rates are class-dependent

**Instance-dependent noise**: Noise rates depend on individual samples

### Limitations of Prior Work

**Underfitting in symmetric losses**: Although symmetric losses such as MAE are theoretically noise-tolerant, the overly strict symmetry condition limits their fitting capacity and makes optimization difficult.

**Limitations of the APL framework**: The Active Passive Loss (APL) framework proposed by Ma et al. improves fitting by combining an active loss (e.g., NCE) with a passive loss (e.g., MAE), but all passive losses currently used in APL are based on the symmetry condition.

**Incompatibility of asymmetric losses**: The asymmetric loss functions (ALFs) proposed by Zhou et al. are theoretically superior to symmetric losses (requiring only the weaker "clean label dominance" condition), but all existing asymmetric losses (e.g., AGCE, AUL) are **active losses** — achieving the asymmetric condition in the passive loss setting has remained an open problem.

**AGCE cannot replace NCE**: Experiments show that replacing the active NCE in APL with the asymmetric AGCE is ineffective (AGCE+MAE achieves only 44.61% under 0.8 symmetric noise), confirming that NCE/NFL is irreplaceable in APL.

### Core Motivation

**Key insight**: The core of the APL framework lies in the complementarity between the active loss (NCE/NFL) and the passive loss. The existing bottleneck is not in the active loss (NCE/NFL already performs well), but in the passive loss — if a **passive asymmetric loss** superior to MAE can be designed, the overall APL framework performance can be substantially improved while keeping NCE/NFL unchanged.

Key distinction:
- **Active loss**: Only explicitly maximizes the probability of the labeled class (e.g., CE: $L = -\log f(x)_y$), with zero loss on non-labeled classes
- **Passive loss**: Also explicitly minimizes the probability of at least one other class (e.g., MAE: $L = \sum_k |e_k - f(x)_k|$)

## Method

### Overall Architecture

1. Propose AMSE: a novel passive asymmetric loss function
2. Establish necessary and sufficient conditions for AMSE to satisfy the asymmetric condition
3. Embed AMSE into the APL framework, replacing the existing symmetric passive loss, to construct JAL

### Key Designs

#### 1. **Asymmetric Mean Squared Error (AMSE)**

- **Function**: Design a new loss function that simultaneously satisfies the asymmetric condition and the definition of a passive loss
- **Mechanism**: Introduce a scaling parameter $a$ into the label vector of the standard MSE:

$$L_{\text{AMSE}}(f(\mathbf{x}), y) = \frac{1}{K} \|a \cdot \mathbf{e}_y - f(\mathbf{x})\|_2^2 = \sum_{k=1}^{K} \frac{1}{K} |a \cdot e_k - f(\mathbf{x})_k|^2$$

where $a \geq 1$ is a hyperparameter. When $a = 1$, AMSE degenerates to the standard MSE.

**Theorem 4.1 (Necessary and Sufficient Condition)**: For given weights $w_1, \ldots, w_K$ (where $w_m > w_n = \max_{i \neq m} w_i$), the loss function $L(f(\mathbf{x}), y) = \frac{1}{K}\|a \cdot \mathbf{e}_y - f(\mathbf{x})\|_q^q$ is asymmetric if and only if:

$$\frac{w_m}{w_n} \geq \frac{a^{q-1} + \sum_{i \neq m} \frac{w_i}{w_n}}{(a-1)^{q-1}} \cdot \mathbb{I}(q > 1) + \mathbb{I}(q \leq 1)$$

For $q = 2$ (AMSE), the condition simplifies to: $\frac{w_m}{w_n} \geq \frac{a + \sum_{i \neq m} w_i/w_n}{a - 1}$

For example, on a 10-class dataset with 0.8 symmetric noise: $\frac{0.2}{0.8/9} \geq \frac{a + 9}{a - 1}$, i.e., $a \geq 9$

- **Design Motivation**: Standard MSE ($a=1$) is a symmetric loss with limited fitting capacity. By introducing $a > 1$, the label target is scaled from the one-hot vector $\mathbf{e}_y$ to $a \cdot \mathbf{e}_y$, imposing a larger penalty on the correct class, thereby breaking symmetry while maintaining noise tolerance under the asymmetric condition. Larger values of $a$ impose stricter constraints and stronger robustness, but may reduce fitting capacity.

#### 2. **Joint Asymmetric Loss (JAL)**

- **Function**: Embed AMSE into the APL framework to replace the symmetric passive loss
- **Mechanism**:

CE-based JAL (JAL-CE):
$$L_{\text{JAL-CE}} = \alpha \cdot L_{\text{NCE}} + \beta \cdot L_{\text{AMSE}}$$

Focal Loss-based JAL (JAL-FL):
$$L_{\text{JAL-FL}} = \alpha \cdot L_{\text{NFL}} + \beta \cdot L_{\text{AMSE}}$$

**Noise tolerance proof**: Zhou et al. have shown that symmetric losses are fully asymmetric and that combinations of asymmetric losses remain asymmetric. Since NCE/NFL is symmetric (and therefore also asymmetric) and AMSE is asymmetric, JAL is also asymmetric and thus noise-tolerant.

- **Design Motivation**: The role of NCE/NFL as the active loss in APL is irreplaceable (confirmed experimentally); the key lies in the passive loss. AMSE, as an asymmetric passive loss, outperforms the symmetric MAE/NNCE, and embedding it into APL achieves "robust yet sufficient learning."

#### 3. **Parameter $a$ Selection Strategy**

- **Function**: Determine a reasonable range for $a$ given the noise rate and number of classes
- **Mechanism**: Compute the lower bound of $a$ from the necessary and sufficient conditions in Theorem 4.1, then select a moderate value. For example, CIFAR-10 + 0.8 symmetric noise → $a \geq 9$; experiments use $a \in [10, 20, 30, 40]$
- **Design Motivation**: Too small an $a$ fails to satisfy the asymmetric condition and provides insufficient robustness; too large an $a$ imposes overly strict constraints and reduces fitting capacity. A balance between robustness and fitting capacity is required.

### Loss & Training

- JAL-CE: $\alpha \cdot \text{NCE} + \beta \cdot \text{AMSE}$, $\alpha = 1, \beta = 1$
- CIFAR-10: 8-layer CNN; CIFAR-100: ResNet-34
- SGD optimizer, learning rate 0.01, weight decay $10^{-4}$, 120 training epochs

## Key Experimental Results

### Main Results

**CIFAR-10 symmetric/asymmetric noise (last-epoch test accuracy %)**:

| Method | Sym. 0.4 | Sym. 0.8 | Asym. 0.2 | Asym. 0.4 |
|--------|---------|---------|-----------|-----------|
| CE | 58.05 | 19.74 | 83.05 | 73.85 |
| NCE+RCE (APL) | 85.89 | 54.99 | 88.62 | 77.94 |
| ANL-CE | 87.16 | 62.28 | 89.09 | 77.99 |
| ANL-FL | 86.94 | 61.89 | 89.29 | 77.89 |
| **JAL-CE** | **87.53** | **65.43** | **89.11** | **79.54** |
| **JAL-FL** | **87.43** | **64.84** | **89.36** | **79.51** |

**CIFAR-100 symmetric/asymmetric noise**:

| Method | Sym. 0.4 | Sym. 0.6 | Asym. 0.3 | Asym. 0.4 |
|--------|---------|---------|-----------|-----------|
| NCE+RCE | 58.48 | 46.73 | 55.86 | 41.50 |
| ANL-CE | 61.58 | 52.09 | 60.57 | 45.73 |
| **JAL-CE** | **64.11** | **56.73** | **64.90** | **56.17** |
| **JAL-FL** | **64.55** | **56.44** | **65.18** | **56.26** |

### Ablation Study

**AMSE vs. existing passive losses (CIFAR-10)**:

| Passive Loss | Sym. 0.4 | Sym. 0.8 | Asym. 0.2 | Asym. 0.4 |
|-------------|---------|---------|-----------|-----------|
| NCE (active+passive baseline) | 69.37 | 41.20 | 72.20 | 65.33 |
| AMSE (passive only) | 87.54 | 64.97 | 83.88 | 58.07 |
| JAL-CE (NCE+AMSE) | 87.53 | 65.43 | 89.11 | 79.54 |

**Robustness and fitting capacity of JAL**:

| Configuration | Description |
|---------------|-------------|
| NCE alone | Very poor fitting (69.37 @ sym0.4), but baseline robustness |
| AMSE alone | Excellent under symmetric noise, but insufficient fitting under asymmetric noise |
| JAL (NCE+AMSE) | Complementary: AMSE dominates robustness under symmetric noise; NCE supplements fitting capacity under asymmetric noise |

**Instance-dependent noise (IDN)**:

| Method | CIFAR-10 IDN 0.4 | CIFAR-10 IDN 0.6 | CIFAR-100 IDN 0.4 | CIFAR-100 IDN 0.6 |
|--------|-----------------|-----------------|-------------------|-------------------|
| ANL-CE | 85.74 | 69.83 | 60.88 | 48.12 |
| **JAL-CE** | **86.46** | **75.62** | **63.24** | **51.69** |

**CIFAR-10N/100N human-annotated noise**:

| Method | CIFAR-10 Worst | CIFAR-100 Noisy |
|--------|---------------|----------------|
| ANL-FL | 80.56 | 57.09 |
| **JAL-CE** | **81.33** | **59.54** |

### Key Findings

1. **Significant advantage under high noise**: Under CIFAR-10 0.8 symmetric noise, JAL-CE outperforms ANL-CE by 3.15% (65.43 vs. 62.28)
2. **Large gains on CIFAR-100 asymmetric noise**: Under 0.4 asymmetric noise, JAL outperforms ANL by approximately 10% (56.17 vs. 45.73)
3. **NCE is irreplaceable in APL**: AGCE+MAE (44.61 @ sym0.8) falls far short of NCE+RCE (54.99 @ sym0.8), confirming that the active loss must be NCE/NFL
4. **Theoretically guided parameter $a$**: The lower bound of $a$ can be directly computed from the necessary and sufficient conditions; $a=20$ or $a=30$ performs best in most settings
5. **JAL is effective on real-world noisy data**: Consistently achieves top-2 performance on CIFAR-10N/100N human-annotated noise

## Highlights & Insights

1. **Solid theoretical contributions**: Theorem 4.1 rigorously establishes the necessary and sufficient conditions for AMSE to satisfy the asymmetric condition, covering both $q > 1$ and $q \leq 1$ cases
2. **Fills a theoretical gap**: For the first time, asymmetric losses are extended to the passive loss setting, resolving the incompatibility between ALFs and the APL framework
3. **Minimal design**: AMSE adds only a single scaling parameter $a$ to the standard MSE, making it simple to implement with clear theoretical grounding
4. **Complementarity analysis**: Ablation experiments clearly demonstrate the complementary roles of NCE (active) and AMSE (passive) — AMSE dominates robustness under symmetric noise, while NCE dominates fitting capacity under asymmetric noise
5. **Principled parameter selection**: The lower bound of $a$ is guided by a theoretical formula rather than pure empirical tuning

## Limitations & Future Work

1. **$a$ depends on noise rate prior**: The necessary and sufficient conditions require knowledge of the noise rate $\eta$; in practice, it must be estimated or a conservative value must be set
2. **Difficulty under very high noise**: Under CIFAR-100 + 0.8 symmetric noise (22.80%), JAL performance drops substantially compared to moderate noise settings
3. **Loss-function-level only**: The method is not combined with sample selection or label correction approaches, leaving potential room for further improvement
4. **Architecture dependence**: Experiments use relatively simple 8-layer CNN / ResNet-34; performance on more complex architectures is not verified
5. **Theoretical assumptions**: The asymmetric condition requires "clean label dominance" ($1-\eta_x > \max_{k \neq y} \eta_{x,k}$), which may not hold under extreme noise ratios

## Related Work & Insights

- **Relation to APL/ANL**: JAL is a natural extension of the APL framework — keeping the active component (NCE/NFL) unchanged and replacing only the passive component with the superior AMSE
- **Relation to ALF**: ALF addressed active asymmetric losses; JAL solves the missing piece of passive asymmetric losses
- **Loss design philosophy**: Scaling the label target ($a \cdot \mathbf{e}_y$) to break symmetry is an elegant and concise strategy
- **Complementary learning inspiration**: The active+passive complementary framework may generalize to other settings requiring a balance between robustness and fitting capacity

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Extending asymmetric losses to the passive setting represents a clear theoretical contribution; AMSE design is clean and principled
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers synthetic noise (symmetric/asymmetric/instance-dependent) and real-world noise (CIFAR-N/WebVision/Clothing1M), across 7 noise settings × multiple datasets
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and theorems are clearly stated, though the dense notation requires careful reading
- **Value**: ⭐⭐⭐⭐ — AMSE is simple to implement with theoretical guarantees, offering practical value to the noisy label learning community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](../../NeurIPS2025/others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[NeurIPS 2025\] Asymmetric Duos: Sidekicks Improve Uncertainty](../../NeurIPS2025/others/asymmetric_duos_sidekicks_improve_uncertainty.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/others/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](../../ICLR2026/others/noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)

</div>

<!-- RELATED:END -->
