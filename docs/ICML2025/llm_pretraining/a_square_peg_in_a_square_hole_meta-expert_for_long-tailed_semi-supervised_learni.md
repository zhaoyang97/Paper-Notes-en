---
title: >-
  [Paper Note] A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning
description: >-
  [ICML 2025][LLM Pretraining][Long-tailed semi-supervised learning] This paper proposes the Meta-Expert algorithm. Through a Dynamic Expert Allocation (DEA) module, it automatically selects the most proficient expert to generate pseudo-labels based on the sample's class assignment (head/medium/tail). It also utilizes a Multi-depth Feature Fusion (MFF) module to alleviate the model's bias towards head classes, achieving "a square peg in a square hole"—letting each expert proces…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Long-tailed semi-supervised learning"
  - "multi-expert ensemble"
  - "dynamic expert allocation"
  - "multi-depth feature fusion"
  - "pseudo-labeling"
date: 2026-05-08
content_hash: bcc5f42dec690295
---

# A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning

**Conference**: ICML 2025  
**arXiv**: [2505.16341](https://arxiv.org/abs/2505.16341)  
**Code**: [Yes](https://github.com/yaxinhou/Meta-Expert)  
**Area**: Semi-Supervised Learning  
**Keywords**: Long-tailed semi-supervised learning, multi-expert ensemble, dynamic expert allocation, multi-depth feature fusion, pseudo-labeling

## TL;DR

This paper proposes the Meta-Expert algorithm. Through a Dynamic Expert Allocation (DEA) module, it automatically selects the most proficient expert to generate pseudo-labels based on the sample's class assignment (head/medium/tail). It also utilizes a Multi-depth Feature Fusion (MFF) module to alleviate the model's bias towards head classes, achieving "a square peg in a square hole"—letting each expert process the sample interval they excel at most.

## Background & Motivation

### Background

Long-Tailed Semi-Supervised Learning (LTSSL) is a research direction that has received significant attention in recent years. In real-world scenarios (such as medical diagnosis), labeled data often exhibits a long-tailed distribution (many samples for common diseases, few for rare ones), and the distribution of unlabeled data may be inconsistent with the labeled data (it could be long-tailed, uniform, or inverse long-tailed), which is known as the "distribution mismatch" problem.

### Limitations of Prior Work

Existing methods like ACR only use a single classifier to generate pseudo-labels, resulting in limited performance. Although CPE introduces three experts (long-tailed, uniform, and inverse long-tailed) to handle unlabeled data under different distributions, it has two key deficiencies:

**Training phase**: Simultaneously using three experts to generate pseudo-labels introduces more incorrect pseudo-labels.

**Testing phase**: Only the uniform expert is used for prediction, neglecting the strengths of different experts.

### Core Observation

The authors find that different experts excel at processing samples in different class intervals:
- **Long-tailed expert**: Excels at head-class samples (94.67% accuracy for head classes on CIFAR-10-LT) but performs poorly on tail classes (38.73%).
- **Uniform expert**: Excels at middle-class samples (77.30%), with a relatively balanced performance across intervals.
- **Inverse long-tailed expert**: Excels at tail-class samples (68.47%) but performs extremely poorly on head classes (3.57%).

### Core Idea

"A square peg in a square hole"—dynamically allocating the most suitable expert to each sample: letting the long-tailed expert process head classes, the uniform expert process middle classes, and the inverse long-tailed expert process tail classes, to fully exploit the strengths of each expert. Meanwhile, it is observed that although shallow features have weak discriminative power, they exhibit small bias, whereas deep features have strong discriminative power but are heavily biased towards head classes. Fusing features from different depths helps alleviate this bias problem.

## Method

### Overall Architecture

Meta-Expert is built upon the FixMatch framework and consists of the following core components:
1. A shared encoder $g$
2. Three expert classifiers, $E_1$ (long-tailed), $E_2$ (uniform), and $E_3$ (inverse long-tailed), which are trained with different logit adjustment strengths $\tau_k$
3. A Dynamic Expert Allocation (DEA) module: Estimates the class cluster/interval assignment of samples and allocates the optimal expert
4. A Multi-depth Feature Fusion (MFF) module: Fuses features from different depths to alleviate the bias towards head classes
5. An Aggregator: Weightedly integrates the logits from the three experts based on the output of the DEA

The training is divided into two phases: first warming up for 18 epochs using the base loss $L_{base}$, and then jointly optimizing the overall loss.

### Key Designs

#### 1. Dynamic Expert Allocation (DEA) Module

**Function**: Estimates the probability of each sample belonging to head, middle, or tail classes, and allocates the most suitable expert accordingly.

**Mechanism**: The DEA adopts an MLP architecture. The inputs are the encoder feature $v$ and logits $z_1, z_2, z_3$ from the three experts, and the output is the soft class assignment:

$$w = DEA([v, z_1, z_2, z_3])$$

where $w = [w_1, w_2, w_3]$ denotes the probability of allocating each expert. The DEA loss is:

$$L_{dea} = \frac{1}{B_l}\sum_{i=1}^{B_l}\ell(w_i^l, s_i) + \frac{1}{B_u}\sum_{j=1}^{B_u}\ell(w_j^u, \hat{s}_j)\mathbb{I}$$

where $s_i$ is the true class interval assignment of labeled samples, and $\hat{s}_j$ is the pseudo-class interval assignment of unlabeled samples.

**Design Motivation**: Experiments show that using true class interval assignment (Upper-E) to select experts can significantly boost pseudo-label quality and prediction accuracy, necessitating a module to accurately estimate this assignment relation.

#### 2. Aggregator

**Function**: Integrates predictions of the three experts through weighted summation based on the weights estimated by DEA.

**Mechanism**:

$$y_m = \sigma\left(\sum_{k=1}^{Q} w_k z_k\right)$$

where $\sigma(\cdot)$ is the softmax function. This ensures that head-class samples are dominated by the long-tailed expert, middle-class samples by the uniform expert, and tail-class samples by the inverse long-tailed expert.

**Design Motivation**: The soft-weighting mechanism (as opposed to hard selection) allows different experts to mutually facilitate learning. Empirical results demonstrate that in certain scenarios, soft-weighting even outperforms the hard-selection scheme based on ground-truth class assignments.

#### 3. Multi-depth Feature Fusion (MFF) Module

**Function**: Fuses features from different depths of the network to learn representations that are both balanced and discriminative.

**Mechanism**: Given shallow feature $v_1$, middle feature $v_2$, and deep feature $v_3$, their dimensions are first aligned via MLPs, and then they are fused by layer-wise addition:

$$MFF(v_1, v_2, v_3) \longmapsto v$$

**Design Motivation**: K-means clustering experiments show that:
- Shallow features: Overall accuracy is 30.30%, with a head-to-tail gap of only 1.67 percentage points (pp). The bias is small, but the discriminative power is weak.
- Deep features: Overall accuracy is 71.00%, with a head-to-tail gap of 35.67pp. The discriminative power is strong, but it is heavily biased toward head classes.

Fusing multi-depth features strikes a favorable balance between bias and discriminative power.

### Loss & Training

The **overall loss** consists of three parts:

$$L_{overall} = L_{base} + L_{dea} + L_{meta}$$

- $L_{base}$: Base SSL loss, containing the supervised classification loss for labeled data and consistency regularization loss for unlabeled data (each computed individually for the three experts)
- $L_{dea}$: The class interval assignment estimation loss of the DEA module
- $L_{meta}$: The overall classification loss based on the aggregator's output

**Training strategy**:
1. Warm up with only $L_{base}$ for the first 18 epochs
2. Jointly optimize $L_{overall}$ thereafter
3. Utilize WRN-28-2 architecture, optimized with SGD (lr=3e-2, momentum=0.9, weight_decay=5e-4)
4. Establish confidence threshold $t = 0.95$

### Theoretical Analysis

The authors provide a generalization error bound:

$$R(\hat{f}) - R(f^*) \leq 2U\epsilon + 4\sqrt{2}\rho\sum_{y=1}^{C}\mathcal{R}_O(\mathcal{H}_y) + 2U\sqrt{\frac{\log\frac{2}{\delta}}{2O}}$$

The critical comparison lies in the overall pseudo-label error rate $\epsilon$:
- CPE: $\epsilon_{CPE} = \frac{1}{Q^2}\sum_{i}\sum_{j}\epsilon_{i,j}$ (all experts cross-calculate errors)
- Meta-Expert: $\epsilon_{Ours} = \frac{1}{Q}\sum_{i}\sum_{j}\mathbb{I}_{i=j}\epsilon_{i,j}$ (each expert only calculates error on its specialized interval)

This theoretically proves that selectively leveraging unique expert strengths leads to a tighter generalization error bound.

## Key Experimental Results

### Main Results

| Dataset | Setting | Meta-Expert | Prev. SOTA (CPE) | Gain |
|--------|------|-------------|-----------------|------|
| CIFAR-10-LT | $\gamma_l=\gamma_u=200$, $N_1=500$ | **74.39** | 67.45 | +6.94pp |
| CIFAR-10-LT | $\gamma_l=200, \gamma_u=1$ | **83.90** | 83.46 | +0.44pp |
| CIFAR-10-LT | $\gamma_l=200, \gamma_u=1/200$ | **85.75** | 84.07 (ACR: 81.23) | +1.68pp |
| CIFAR-10-LT | $\gamma_l=150, \gamma_u=1/150$ | **86.78** | 85.52 | +1.26pp |
| STL-10-LT | $\gamma_l=20, N_1=150$ | **71.19** | 68.01 | +3.18pp |
| STL-10-LT | $\gamma_l=15, N_1=450$ | **79.98** | 78.71 | +1.27pp |
| SVHN-LT | $\gamma_l=150, \gamma_u=1$ | **94.66** | 94.14 | +0.52pp |

### Ablation Study

| Configuration | CIFAR-10 ($\gamma_u=200$) | CIFAR-10 ($\gamma_u=1$) | CIFAR-10 ($\gamma_u=1/200$) | Description |
|------|--------------------------|------------------------|---------------------------|------|
| CPE baseline | 78.57 | 83.47 | 84.40 | Baseline |
| + DEA | 79.22 (+0.65) | 83.61 (+0.14) | 84.58 (+0.18) | Contribution of expert allocation |
| + MFF | 79.83 (+1.26) | 83.89 (+0.42) | 83.10 (-1.30) | Contribution of feature fusion |
| + DEA + MFF | **81.67** (+3.10) | **83.96** (+0.49) | **85.75** (+1.35) | Synergy effect |

**Ablation on Feature Combination Strategies**:

| Strategy | $\gamma_u=200$ | $\gamma_u=1$ | $\gamma_u=1/200$ |
|------|---------------|-------------|-----------------|
| Concatenation (concat) | 80.32 | 83.38 | 84.65 |
| Addition (add) | **81.67** | **83.96** | **85.75** |

### Key Findings

1. **DEA module effectively estimates class assignments**: Visualization results indicate that DEA accurately assigns head-class samples to the long-tailed expert (maximal $w_1$), middle-class samples to the uniform expert (maximal $w_2$), and tail-class samples to the inverse long-tailed expert (maximal $w_3$).
2. **Significant improvement in pseudo-label quality**: Under consistent distribution, the overall pseudo-label error rate drops from 30.07% (CPE) to 21.17%; the utilization of unlabeled data increases from 84.86% to 95.31%.
3. **Soft-weighting outperforms hard-selection**: Under certain settings, Meta-Expert even outperforms "Upper-E", which uses the ground-truth class assignments (e.g., achieving a higher F1 score under consistent distribution).
4. **Manageable computational overhead**: The number of parameters increases by 13.3% (1.5M to 1.7M) and epoch time increases by 6.4%, while accuracy is boosted by 3.5pp.
5. **Special behavior of MFF under inverse distribution**: Employing MFF alone under $\gamma_u=1/200$ actually degrades performance. However, when combined with DEA, its efficacy is unlocked, indicating that reducing pseudo-label errors via DEA is a prerequisite for MFF to function.

## Highlights & Insights

1. **The intuition of "a square peg in a square hole" is elegant**: The observation that different experts excel at different intervals is simple yet effective, with the Upper-E experiment strongly validating this hypothesis.
2. **Novel discovery of bias in multi-depth features**: This work is the first to systematically reveal that shallow features have low bias but weak discriminative power, while deep features have high bias but strong discriminative power. This finding provides valuable insights for other long-tailed learning tasks.
3. **Solid theoretical analysis**: Theoretical proofs for the generalization error bounds are provided, and experiments confirm that Meta-Expert indeed achieves a lower pseudo-label error rate.
4. **Soft-weighted integration strategy**: Compared to simple hard-selection (Upper-E), soft-weighting allows experts to mutually reinforce each other, occasionally even outperforming the upper-bound scheme based on true labels under certain configurations.

## Limitations & Future Work

1. **Small dataset scale**: Validation is limited to CIFAR-10, STL-10, and SVHN, lacking validation on larger-scale datasets such as CIFAR-100-LT or ImageNet-LT.
2. **Limited number of classes**: All three benchmark datasets have only 10 classes. In scenarios with a larger number of classes, is the division into three intervals (head/middle/tail) sufficient? Finer-grained interval partitioning might be required.
3. **Fixed three-expert design**: The number of experts is fixed at 3 (long-tailed, uniform, and inverse long-tailed). Can the number of experts be adaptively determined?
4. **DEA module relies on pre-defined class interval assignments**: The division of head/middle/tail relies on pre-sets (e.g., the first 2 classes as head, and the last 6 classes as tail), which lacks flexibility.
5. **Marginal improvement on SVHN**: The performance gap compared to BaCon and SimPro on SVHN-LT is narrow, and the model is even surpassed by SimPro under inverse distribution (94.24 vs 94.76).
6. **Simple addition-based fusion strategy for MFF**: More complex fusion schemes, such as attention mechanisms, could be explored.

## Related Work & Insights

- **CPE (AAAI 2024)**: The direct predecessor of this work, which trains three experts to handle different distributions, but fails to fully exploit their unique strengths. Meta-Expert proposes a dynamic allocation strategy on top of it.
- **ACR (CVPR 2023)**: Proposes adaptive logit adjustment to handle distribution mismatch but relies on a single classifier.
- **FixMatch (NeurIPS 2020)**: Landmark SSL baseline, acting as the base framework for Meta-Expert.
- **BaCon (AAAI 2024)**: Enhances imbalanced semi-supervised performance through balanced feature contrastive learning.
- **SimPro (ICML 2024)**: A probabilistic framework to address long-tailed semi-supervised learning.
- **Insights**: Generalizing the idea of "specialized experts for different distributions" to other tasks (such as multi-domain generalization or federated learning) could be of great value.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core observation (different experts excel at different intervals) is highly intuitive, and the discovery of multi-depth feature bias is novel. However, the overall method is an incremental improvement over CPE.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablation studies, detailed pseudo-label quality analysis, and solid theoretical analysis are provided, though the datasets' scale and class numbers are relatively small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is beautifully motivated by the "square peg in a square hole" analogy, maintaining a highly clear logical flow from motivation to observation and design. Figures and tables are rich and intuitive.
- **Value**: ⭐⭐⭐⭐ It pushes the SOTA in the LTSSL field, and the design concepts of DEA and MFF offer meaningful references for other long-tailed learning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning](density_ratio_estimation-based_bayesian_optimization_with_semi-supervised_learni.md)
- [\[CVPR 2025\] A Unified Framework for Heterogeneous Semi-supervised Learning](../../CVPR2025/llm_pretraining/a_unified_framework_for_heterogeneous_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[ICML 2025\] Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry](algebra_unveils_deep_learning_--_an_invitation_to_neuroalgebraic_geometry.md)
- [\[ICML 2025\] On the Role of Label Noise in the Feature Learning Process](on_the_role_of_label_noise_in_the_feature_learning_process.md)

</div>

<!-- RELATED:END -->
