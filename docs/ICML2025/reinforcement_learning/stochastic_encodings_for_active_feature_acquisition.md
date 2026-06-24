---
title: >-
  [Paper Note] Stochastic Encodings for Active Feature Acquisition
description: >-
  [ICML 2025][Reinforcement Learning][Active Feature Acquisition] This paper proposes SEFA (Stochastic Encodings for Feature Acquisition), an active feature acquisition method based on stochastic latent variable models. By reasoning across multiple unobserved feature realizations in a regularized latent space instead of relying on RL and greedy CMI maximization, it consistently outperforms all baselines on synthetic and real-world datasets (including cancer classification).
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Active Feature Acquisition"
  - "Dynamic Feature Selection"
  - "Stochastic Encoder"
  - "Latent Space"
  - "Information Bottleneck"
date: 2026-05-08
content_hash: 51aed7b09e46aebe
---

# Stochastic Encodings for Active Feature Acquisition

**Conference**: ICML 2025  
**arXiv**: [2508.01957](https://arxiv.org/abs/2508.01957)  
**Code**: [a-norcliffe/SEFA](https://github.com/a-norcliffe/SEFA)  
**Area**: Active Feature Acquisition / Sequential Decision Making  
**Keywords**: Active Feature Acquisition, Dynamic Feature Selection, Stochastic Encoder, Latent Space, Information Bottleneck  

## TL;DR

This paper proposes SEFA (Stochastic Encodings for Feature Acquisition), an active feature acquisition method based on stochastic latent variable models. By reasoning across multiple unobserved feature realizations in a regularized latent space instead of relying on RL and greedy CMI maximization, it consistently outperforms all baselines on synthetic and real-world datasets (including cancer classification).

## Background & Motivation

**Active Feature Acquisition (AFA)** is an instance-level sequential decision-making problem: at test time, the model dynamically selects which feature to acquire next based on the currently observed features. A typical scenario: a doctor chooses which test to perform based on known clinical information.

Two major categories of methods each have fundamental flaws:

### 1. Reinforcement Learning Methods
- Naturally suited for sequential decision-making, but plagued by training difficulties: sparse rewards, exploration-exploitation trade-offs, and the deadly triad.

### 2. Conditional Mutual Information (CMI) Maximization
- Greedy selection: $i^* = \arg\max_i I(X_i; Y | \mathbf{x}_O)$
- **Flaw 1: Myopic decision-making**. CMI marginalizes unobserved features, $p(x_i, y|\mathbf{x}_O) = \int p(x_j, x_i, y|\mathbf{x}_O) dx_j$, failing to account for the impact of future acquisitions. The paper rigorously proves this using an **indicator problem**:

> Consider $d+1$ features, where $x_{d+1}$ is an indicator (determining which of the $d$ binary features provides the label). The optimal policy requires only 2 steps, but greedy CMI requires an expected $3 - 1/d$ steps.

- **Flaw 2: CMI does not equate to the optimal 0-1 loss objective**. Minimizing entropy can be achieved by making unlikely classes even more unlikely, rather than distinguishing the most likely classes:
    - $H([0.5, 0.5, 0.0]) = 0.693$ (low entropy but unable to identify the most likely class)
    - $H([0.7, 0.15, 0.15]) = 0.819$ (high entropy but clearly identifies the most likely class)

## Method

### Overall Architecture

SEFA uses an encoder-predictor architecture with an intermediate stochastic latent variable $Z$:

$$p_{\theta,\phi}(y|\mathbf{x}_S) = \mathbb{E}_{p_\theta(\mathbf{z}|\mathbf{x}_S)} p_\phi(y|\mathbf{z})$$

### Key Designs

#### Key Design 1: Independent Feature Encoding

Each feature $i$ is independently encoded into $l$ latent components:

$$p_\theta(\mathbf{z}|\mathbf{x}_S) = \prod_{i=1}^d p_{\theta_i}(\mathbf{z}_{\mathcal{G}_i} | x_{S,i}, m_{S,i})$$

where $\mathcal{G}_i$ indexes the latent components managed by feature $i$ (e.g., if $l=3$, then $\mathcal{G}_1=\{1,2,3\}$, $\mathcal{G}_2=\{4,5,6\}$). Each encoder outputs the mean and variance of a normal distribution.

#### Key Design 2: Acquisition Objective

$$R(\mathbf{x}_O, i) = \sum_{c \in [C]} p_{\theta,\phi}(Y=c|\mathbf{x}_O) \cdot \mathbb{E}_{p_\theta(\mathbf{z}|\mathbf{x}_O)} r(c, \mathbf{z}, i)$$

The scoring function is based on latent space gradients:

$$r(c, \mathbf{z}, i) = \frac{\|\mathbf{g}_{\mathcal{G}_i}\|_2}{\sum_j \|\mathbf{g}_{\mathcal{G}_j}\|_2}, \quad \mathbf{g} = \nabla_\mathbf{z} p_\phi(Y=c|\mathbf{z})$$

Three core components:
1. **Scoring function**: Measures the importance of each feature to the predicted class $c$ using latent space gradients (similar to gradient attribution in interpretability, but performed in the latent space).
2. **Stochastic encoding**: Takes the expectation over $p_\theta(\mathbf{z}|\mathbf{x}_O)$, considering **multiple possible realizations of unobserved features**—analogous to Monte Carlo tree search.
3. **Probability weighting**: Weights by the current predictive probability $p_{\theta,\phi}(Y=c|\mathbf{x}_O)$, focusing on more likely classes.

#### Key Design 3: Information Bottleneck Regularization

### Loss & Training

$$L = \mathbb{E}_{p_\text{Data}(\mathbf{x}_S, y)} \mathbb{E}_{p_\text{Subsample}(S')} \left[ -\log p_{\theta,\phi}(y|\mathbf{x}_{S \cap S'}) + \beta D_\text{KL}(p_\theta(Z|\mathbf{x}_{S \cap S'}) \| p(Z)) \right]$$

- **Predictive loss**: Negative log-likelihood, estimated using multiple latent samples (rather than a single sample).
- **Information Bottleneck**: Constrains latent variables to retain only label-relevant information, removing feature-level noise.
- **Random subsampling**: Randomly removes features during training, adapting the model to arbitrary feature subsets.
- Prior $p(Z) = \mathcal{N}(0, 1)$, with a closed-form solution for the KL divergence.

### Advantages of Latent Space vs. Feature Space

1. Latent space gradients are more meaningful—all components are continuous and of similar scales.
2. The information bottleneck removes feature noise—decisions are based on pure label information.
3. No generative model training required—avoiding complexities such as modeling continuous/discrete variables and multi-modal densities.

## Key Experimental Results

### Main Results

#### Synthetic Datasets (Average steps to acquire all relevant features, lower is better)

| Model | Syn 1 | Syn 2 | Syn 3 |
|------|-------|-------|-------|
| ACFlow | 7.730 | 7.527 | 9.194 |
| DIME | 4.079 | 4.581 | 5.667 |
| GDFS | 4.568 | 4.484 | 5.587 |
| Opportunistic RL | 4.201 | 4.846 | 5.850 |
| Random | 9.484 | 9.499 | 9.987 |
| **SEFA** | **4.017** | **4.099** | **5.084** |

SEFA is close to optimal (optimal is 3 for Syn 1/2, and 5 for Syn 3).

#### Real-World Datasets (Average evaluation metric during acquisition, higher is better)

| Model | Bank Mktg | Calif. Housing | MiniBooNE | MNIST | Fashion | METABRIC | TCGA |
|------|-----------|---------------|-----------|-------|---------|----------|------|
| DIME | 0.907 | 0.661 | 0.951 | 0.731 | 0.703 | 0.670 | 0.805 |
| GDFS | 0.907 | 0.653 | 0.949 | 0.732 | 0.692 | 0.671 | 0.797 |
| Opp. RL | 0.910 | 0.657 | 0.953 | 0.740 | 0.708 | 0.706 | 0.838 |
| **SEFA** | **0.919** | **0.676** | **0.957** | **0.761** | **0.721** | **0.709** | **0.843** |

**SEFA ranks first on all 8 real-world datasets.**

### Ablation Study (Synthetic datasets, steps to acquire relevant features)

| Variant | Syn 1 | Syn 2 | Syn 3 |
|------|-------|-------|-------|
| $\beta=0$ (No Information Bottleneck) | 4.520 | 4.578 | 5.716 |
| 1 acquisition sample | 4.683 | 4.862 | 5.700 |
| 1 training sample | 4.421 | 4.713 | 5.188 |
| Deterministic encoder | 4.593 | 4.773 | 5.744 |
| **Feature space computation** | **5.111** | **5.461** | **5.977** |
| No normalization | 4.036 | 4.104 | 5.101 |
| **SEFA (full)** | **4.017** | **4.099** | **5.084** |

**Most critical components**: Latent space computation > Multiple acquisition samples > Stochastic encoding > Information bottleneck.

### Interpretability in Cancer Classification

In the TCGA tumor localization task, the acquisition order of SEFA aligns with medical literature:
- **ST6GAL1** is almost always chosen first—known to be upregulated in multiple cancers.
- Breast cancer: DEF6 (associated with breast cancer metastasis) is selected in step 2.
- Lung/Liver cancer: DNASE1L3 (a potential biomarker for liver and lung cancer) is selected in step 2.
- Prostate cancer: SERPINB1 (associated with prostate cancer) is selected in step 2.

## Highlights & Insights

1. **Dual theoretical + empirical demonstration of CMI limitations**: Optimality proof for the indicator problem (Propositions 4.1/4.2) + counterexamples showing entropy vs. class identification.
2. **Latent space reasoning ≈ Monte Carlo Tree Search**: By sampling multiple possible realizations in the stochastic latent space, non-myopic reasoning about future acquisition performance is achieved.
3. **Avoiding RL training**: The acquisition objective is hand-crafted rather than learned—the model is trained via supervised learning, while the information-theoretic objective drives acquisition.
4. **Clever trade-off in independent feature encoding**: Although feature dependencies cannot be modeled by the encoders, the predictor network implicitly learns conditional dependencies—after measuring feature 1, the gradients change, thereby affecting the scoring of feature 2.
5. **Theoretical necessity of multiple training samples**: Single-sample training biases encoders or predictors toward deterministic outputs, reducing latent space diversity.

## Limitations & Future Work

1. **Applicable only to classification**: Requires class probability differentiation; regression tasks require new designs (the paper suggests a dual-head approach).
2. **High memory cost at inference**: Requires multiple latent samples (200 samples), though CMI generative-model-based approaches face similar requirements.
3. **Lack of standard evaluation against clinical benchmarks on medical data**: The TCGA analysis is more of an interpretability demonstration.
4. **Limitations of independent encoders**: Cannot model encoder-level conditional dependencies among features.

## Related Work & Insights

- **RL Methods**: Opportunistic Learning, GSMRL, REFUEL
- **CMI Methods**: GDFS, DIME, ACFlow, EDDI
- **Others**: Sensitivity methods (Kachuee et al.), imitation learning (Valancius et al.), decision trees (Xu et al.)
- **Information Bottleneck**: Tishby et al. 1999, Alemi et al. 2017

## Rating

⭐⭐⭐⭐⭐ (5/5)

Highly elegant and thoughtful methodology design—every component has clear motivation and theoretical backing. The consistent superiority across all datasets is compelling. The theoretical analysis of CMI's flaws and the interpretability validation on cancer classification are highlights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](../../NeurIPS2025/reinforcement_learning/aline_joint_amortization_for_bayesian_inference_and_active_data_acquisition.md)
- [\[ICML 2025\] LineFlow: A Framework to Learn Active Control of Production Lines](lineflow_a_framework_to_learn_active_control_of_production_lines.md)
- [\[NeurIPS 2025\] Exploration via Feature Perturbation in Contextual Bandits](../../NeurIPS2025/reinforcement_learning/exploration_via_feature_perturbation_in_contextual_bandits.md)
- [\[NeurIPS 2025\] Open-World Drone Active Tracking with Goal-Centered Rewards](../../NeurIPS2025/reinforcement_learning/open-world_drone_active_tracking_with_goal-centered_rewards.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](../../NeurIPS2025/reinforcement_learning/emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)

</div>

<!-- RELATED:END -->
