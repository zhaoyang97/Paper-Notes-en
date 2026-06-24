---
title: >-
  [Paper Note] AutoAL: Automated Active Learning with Differentiable Query Strategy Search
description: >-
  [ICML 2025][Active Learning] Proposes the first differentiable active learning strategy search framework, AutoAL. By collaboratively training two networks, SearchNet and FitNet, under a bilevel optimization framework, it automatically selects the optimal strategy from multiple candidate AL strategies for a given task, consistently outperforming all candidate strategies and other SOTA methods on natural and medical image datasets.
tags:
  - "ICML 2025"
  - "Active Learning"
  - "Differentiable Query Strategy Search"
  - "Bilevel Optimization"
  - "Strategy Selection"
  - "Data-Efficient Learning"
date: 2026-05-08
content_hash: 4522a50a8690e706
---

# AutoAL: Automated Active Learning with Differentiable Query Strategy Search

**Conference**: ICML 2025  
**arXiv**: [2410.13853](https://arxiv.org/abs/2410.13853)  
**Code**: [Yes](https://github.com/haizailache999/AutoAL)  
**Area**: Active Learning  
**Keywords**: Active Learning, Differentiable Query Strategy Search, Bilevel Optimization, Strategy Selection, Data-Efficient Learning

## TL;DR

Proposes the first differentiable active learning strategy search framework, AutoAL. By collaboratively training two networks, SearchNet and FitNet, under a bilevel optimization framework, it automatically selects the optimal strategy from multiple candidate AL strategies for a given task, consistently outperforming all candidate strategies and other SOTA methods on natural and medical image datasets.

## Background & Motivation

### Background
Active Learning (AL) significantly reduces the annotation cost of deep learning by iteratively selecting the most informative samples from an unlabeled data pool for labeling. Existing AL strategies mainly fall into two categories:
- **Uncertainty-based methods** (e.g., Maximum Entropy, BALD): Select samples where the model is most uncertain.
- **Diversity/representativeness-based methods** (e.g., CoreSet, KMeans): Select a subset of samples representing the entire data distribution.

### Limitations of Prior Work
Different AL strategies perform vastly differently under different data scenarios. For instance, diversity-based methods perform well under many-class, large-batch scenarios, while uncertainty-based methods excel in the opposite. **No single strategy performs best on all tasks**, making strategy selection a challenging issue in practical applications.

### Prior Attempts & Deficiencies
- **SelectAL** (NeurIPS 2024): Dynamically selects AL strategies by estimating the relative budget size of the problem, but relies on approximations of generalization error reduction over small subsets, making it difficult to capture the complexity of actual tasks.
- **ALBL**: Treats candidate strategies as a multi-armed bandit problem, but cannot utilize gradient information.
- **Zhang et al. (NeurIPS 2024)**: Selects the optimal batch from hundreds of candidates, but incurs prohibitive computational overhead and lacks differentiability.

### Key Challenge
Existing adaptive selection methods rely on **non-differentiable discrete selections** (black-box search), resulting in low computational efficiency and insufficient data utilization.

### Ours
Proposes AutoAL—the first differentiable AL strategy search framework, which relaxes the strategy selection space from discrete to continuous, utilizes gradient descent for efficient optimization, and achieves data-driven automated strategy selection.

## Method

### Overall Architecture

AutoAL consists of two neural networks trained alternately under a bilevel optimization framework:
1. **FitNet** $\Omega_F$: Trained on labeled data to learn the data distribution and output the classification loss for each sample.
2. **SearchNet** $\Omega_S$: Learns to assign weights to each candidate AL strategy based on the loss signal from FitNet, outputting an integrated informativeness score.

Key clever design: The labeled set is randomly split into equal training and validation sets. SearchNet treats the training set as the "labeled pool" and the validation set as the "unlabeled pool" to simulate the actual AL selection process, avoiding the need to access the real unlabeled data.

### Key Designs

#### 1. **Bilevel Optimization Formulation**

The optimization of FitNet and SearchNet is modeled as a bilevel optimization problem:

$$\Omega_S^* = \arg\max_{\Omega_S} \sum_{j=1}^{M/2} \mathcal{L}_S((x_j, y_j), \Omega_S, \Omega_F^*)$$

$$\text{s.t.} \quad \Omega_F^* = \arg\min_{\Omega_F} \sum_{j=1}^{M/2} \mathcal{L}_F((x_j, y_j), \Omega_F)$$

- **Lower-level optimization**: FitNet minimizes the classification loss to fit the data distribution.
- **Upper-level optimization**: SearchNet maximizes the information loss to select high-loss (highly informative) samples.
- **Design Motivation**: FitNet provides an understanding of the data distribution, while SearchNet utilizes this information to identify the samples that need annotation the most, establishing a mutually reinforcing cooperative relationship.

#### 2. **Probabilistic Query Strategy and Gaussian Mixture Model**

For each sample $x_j$, a selection score $\mathcal{S}_\kappa(x_j)$ is acquired from each of the $K$ candidate AL strategies. A Gaussian Mixture Model (GMM) is used to model the joint distribution of all strategy scores:

$$p(\mathcal{S}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathcal{S} | \mu_k, \Sigma_k)$$

Then, selection thresholds are determined by sampling from the GMM, and combined with the strategy weight $W_{\kappa,j}$ predicted by SearchNet to compute the integrated score for each strategy-sample pair:

$$\hat{\mathcal{S}}_\kappa(x_j, \Omega_S) = (\mathcal{S}_\kappa(x_j) - \vartheta_t(S_{\text{sample}})) \cdot W_{\kappa,j}$$

**Design Motivation**: The GMM naturally blends the score distributions of multiple strategies, while the thresholding mechanism ensures only highly prominent samples are selected. The weights $W_{\kappa,j}$ enable **sample-level** adaptive strategy selection.

#### 3. **Differentiable Relaxation and Continuous Search Space**

To make the discrete strategy selection differentiable, a Sigmoid function is utilized to relax the discrete $\{0,1\}$ selection to a continuous space:

$$\bar{\mathcal{S}}(x_j) = \sum_{\kappa \in K} \frac{\lambda}{1 + \exp(-\Theta^{(j)}_{\mathcal{S}'_\kappa})} \hat{\mathcal{S}}_\kappa(x_j, \Omega_S)$$

where $\Theta^{(j)}$ is the parameter vector for strategy mixture weights of each sample. This relaxation allows the entire framework to be optimized end-to-end via backpropagation.

**Design Motivation**: Inspired by DARTS (Differentiable Architecture Search), this translates the discrete selection problem into a continuous optimization problem, leveraging gradient descent instead of black-box search to significantly improve optimization efficiency.

### Loss & Training

**FitNet Loss** (Eq. 7):
$$\mathcal{L}_F = \frac{1}{B} \sum_{j'} \bar{\mathcal{S}}_{\text{detach}}(x'_j) \cdot \mathcal{L}(x'_j, y'_j) + \bar{\lambda} \mathcal{L}_{re}(t, B)$$

- Employs a cross-entropy loss weighted by the detached search score, pushing FitNet to focus more on minimizing the loss of selected samples.
- Gradients of $\bar{\mathcal{S}}_{\text{detach}}$ do not propagate back to SearchNet, ensuring the validity of bilevel optimization.

**SearchNet Loss** (Eq. 8):
$$\mathcal{L}_S = -\frac{1}{B} \sum_j \bar{\mathcal{S}}(x_j) \cdot \mathcal{L}_{\text{detach}}(x_j, y_j) - \bar{\lambda} \mathcal{L}_{re}(t, B)$$

- Uses a negative sign to achieve gradient ascent, with the goal of selecting high-loss samples.
- The loss of FitNet is detached to prevent updating FitNet itself.

**Regularization Loss** (Eq. 9):
$$\mathcal{L}_{re}(t, B) = \frac{1}{1 + \exp(0.5 \cdot |\alpha - t \cdot B|)} - 0.5$$

- Controls the volume of selected samples, penalizing choosing too many or too few.
- $\alpha$ is the number of selected samples, and $t$ is the ratio of the query batch size to the total data pool size.

**Training Process**:
- Each AL iteration contains $\mathcal{C}$ cycles.
- FitNet is pre-trained on the validation set for 200 epochs.
- Then, FitNet, SearchNet, and the loss prediction module are alternately updated for a total of 400 epochs.
- The backbone network uses ResNet-18. FitNet is optimized using Adam (lr=0.005), and SearchNet is optimized using SGD (lr=0.005).

## Key Experimental Results

### Main Results

Experiments are conducted on 7 datasets (4 natural images + 3 medical images), with the candidate strategy pool containing 7 AL methods.

| Dataset | Classes | Training Size | Imbalance Ratio | AutoAL Performance |
|--------|--------|----------|----------|------------|
| CIFAR-10 | 10 | 50K | 1.0 | Consistently outperforms all baselines |
| CIFAR-100 | 100 | 50K | 1.0 | Advantage is more prominent on difficult datasets |
| SVHN | 10 | 73K | 3.0 | Leads across all cycles |
| TinyImageNet | 200 | 100K | 1.0 | Remains robust on large-scale multi-class datasets |
| OrganCMNIST | 11 | 13K | 5.0 | Best performance even in small-data-pool scenarios |
| PathMNIST | 9 | 90K | 1.6 | Continuously outperforms candidate strategies |
| TissueMNIST | 8 | 165K | 9.1 | Remains robust under high imbalance ratios |

AutoAL consistently outperforms 14 baselines (including Entropy, Margin, BALD, BADGE, LPL, VAAL, CoreSet, DDU, ALBL, etc.) across all 7 datasets, showing smaller standard deviations.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| ResNet Backbone only | Significant performance drop | Missing the loss prediction module, leading to insufficient SearchNet optimization |
| Loss Prediction only | Largest performance drop | Only minimizes loss instead of selecting the optimal strategy, deviating from the target |
| ResNet + Loss Prediction (Complete) | Best performance | The two components complement each other, and both are indispensable |
| 1 candidate strategy | Better than using the strategy alone, but inferior to multiple candidates | SearchNet still gains from the loss prediction |
| 3 candidate strategies | Near-optimal performance (CIFAR-100, OrganCMNIST) | High cost-performance ratio |
| 5 candidate strategies | Further improvement on SVHN | Upper bound depends on the dataset |
| 7 candidate strategies (Complete) | Best or near-best | More candidates $\rightarrow$ lower standard deviation |

### Key Findings

1. **Dynamic Strategy Switching**: Visualizing the strategy scores across iterations (Figure 5) reveals that AutoAL prefers diversity-based strategies (KMeans) in early rounds, and switches to uncertainty-based strategies (Least Confidence, MeanSTD) in later rounds. This aligns with intuition: early rounds require broad exploration of the data distribution, while later rounds require refinement of decision boundaries.
2. **Controllable Computational Overhead**: The search component of AutoAL (updating SearchNet and FitNet) accounts for only about 3% of the total execution time; the main overhead stems from computing scores of candidate strategies. With 3 candidates, the total overhead is only 1.3$\times$ that of EntropySampling.
3. **Robustness**: The accuracy curves of AutoAL are smoother, presenting no obvious performance drops (performance degradation caused by harmful data selection), and show smaller standard deviations.
4. **Extreme Performance Variance of Individual Strategies**: Margin performs poorly on SVHN, KMeans on OrganCMNIST, and VAAL on CIFAR-10, highlighting the necessity of data-driven automated strategy selection.

## Highlights & Insights

1. **Ingenious Transfer of DARTS**: Adapting the principles of differentiable architecture search to AL strategy search, relaxing discrete strategy selection to continuous optimization. The formulation is clean and elegantly implemented.
2. **Simulating AL Process via Train-Val Split**: Partitioning labeled data into training/validation sets to mimic labeled/unlabeled pools. This enables training SearchNet without accessing actual unlabeled data, which is a clever design.
3. **Sample-Level Strategy Selection**: Instead of choosing one strategy for the entire dataset, AutoAL independently selects the optimal combination of strategies for each individual sample, achieving finer granularity.
4. **Automated Transition from Exploration to Exploitation**: Experiments confirm that AutoAL automatically transitions from diversity-based to uncertainty-based strategies, conforming to intuition with zero manual intervention.
5. **Extensible Framework**: Any new AL strategy can be seamlessly integrated as a candidate into the pool, demonstrating high openness and scalability.

## Limitations & Future Work

1. **Upper-bound Effect of Strategy Pool**: More candidates are not always better (3 candidates suffice on CIFAR-100 and OrganCMNIST); automated determination of the optimal candidate set remains unaddressed.
2. **Pre-computation of Candidate Scores**: While AutoAL has low search overhead, score computation for all candidate strategies scales linearly with the number of candidates, posing a potential bottleneck when the pool scales up.
3. **Limited to Classification Tasks**: Experiments are confined to image classification, leaving detection, segmentation, NLP, and other scenarios unverified, which requires further investigation of generalizability.
4. **Lack of Deep Comparison with Learning-based AL**: There is a lack of rigorous comparison against meta-learning-based adaptive AL methods (e.g., Meta-Query Net).
5. **Fixed Architectures for FitNet and SearchNet**: Both networks employ ResNet-18; the impact of lighter or more powerful architectures on overall performance remains unexplored.
6. **Insufficient Parameter Sensitivity Analysis**: The impact of hyperparameters such as the cycle number $\mathcal{C}$, regularization coefficient $\bar{\lambda}$, and the number of GMM components is not thoroughly discussed.

## Related Work & Insights

- **DARTS** (Liu et al., 2018): Pioneered differentiable architecture search, from which AutoAL's relaxation strategy is directly inspired.
- **Learning Loss for AL** (Yoo & Kweon, 2019): The loss prediction module is integrated into AutoAL's SearchNet.
- **SelectAL** (Hacohen & Weinshall, NeurIPS 2024): Budget estimation-based strategy selection, acting as a major baseline for AutoAL.
- **ALBL** (Hsu & Lin, 2015): Adaptive strategy selection via a multi-armed bandit framework, but lacks differentiability.
- **Insights**: The combination of differentiable optimization and bilevel optimization could extend to other "policy/strategy selection" issues, such as data augmentation policy search and curriculum learning strategy search.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First differentiable AL strategy search framework. Translating DARTS to AL is valuable, though the core concept is not entirely revolutionary.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 7 datasets (including medical images) against 14 baselines, complete ablations, and in-depth strategy visualizations, but lacks verification of non-classification tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-explained motivation, and thorough formula derivations, though some mathematical notations are slightly redundant.
- **Value**: ⭐⭐⭐⭐ Successfully addresses practical pain points of AL strategy selection; the extensible framework contributes positively to the AL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](../../CVPR2025/others/instance-wise_supervision-level_optimization_in_active_learning.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](../../NeurIPS2025/others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[ICML 2025\] DiLQR: Differentiable Iterative Linear Quadratic Regulator via Implicit Differentiation](dilqr_differentiable_iterative_linear_quadratic_regulator_via_implicit_different.md)

</div>

<!-- RELATED:END -->
