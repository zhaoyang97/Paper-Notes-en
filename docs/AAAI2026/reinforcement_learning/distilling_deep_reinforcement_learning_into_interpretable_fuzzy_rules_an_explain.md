---
title: >-
  [Paper Note] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework
description: >-
  [AAAI 2026][Reinforcement Learning][Fuzzy Logic] This paper proposes a hierarchical Takagi-Sugeno-Kang (TSK) fuzzy classifier system that distills deep RL neural network policies into human-readable IF-THEN fuzzy rules.…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Fuzzy Logic"
  - "TSK System"
  - "Policy Distillation"
  - "Explainable AI"
  - "Continuous Control"
date: 2026-05-08
content_hash: f6bdd0e22ed9aa6c
---

# Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework

**Conference**: AAAI 2026
**arXiv**: [2603.13257](https://arxiv.org/abs/2603.13257)  
**Code**: None  
**Area**: Explainable Reinforcement Learning
**Keywords**: Fuzzy Logic, TSK System, Policy Distillation, Explainable AI, Continuous Control

## TL;DR

This paper proposes a hierarchical Takagi-Sugeno-Kang (TSK) fuzzy classifier system that distills deep RL neural network policies into human-readable IF-THEN fuzzy rules. Three quantitative interpretability metrics are introduced (FRAD, FSC, ASG). On the Lunar Lander continuous control task, the proposed system achieves 81.48% fidelity, surpassing decision trees by 21 percentage points.

## Background & Motivation

### Problem Definition

Deep reinforcement learning (DRL) agents excel at continuous control, yet their policies are encoded in deep neural network weights, remaining opaque to human observers. For example, a PPO policy for the lunar lander contains hundreds of parameters, and when the agent outputs a thrust vector $[0.36, 0.71]$, the rationale is inaccessible: is it based on tilt angle, or a complex interaction between position and velocity?

### Limitations of Prior Work

**Local explanation methods (SHAP/LIME)**: Only answer "which features contributed to this action," without revealing global policy structure or operational modes; instance-specific and non-transferable.

**Symbolic distillation methods (Decision Trees/VIPER)**: Approximate smooth control functions with piecewise constants, requiring extremely deep trees for acceptable fidelity at the cost of interpretability; actions change abruptly at boundaries.

**Existing neuro-fuzzy methods**: Integrate fuzzy logic during training (e.g., Fuzzy Q-Learning), but underperform modern deep RL; post-hoc fuzzy surrogate approaches lack hierarchical policy structure, rigorous quantitative metrics, and comprehensive baseline comparisons.

### Root Cause

Continuous control tasks require approximately smooth nonlinear mappings. Fuzzy logic naturally bridges numerical computation and linguistic reasoning via linguistic variables (e.g., "high velocity") and IF-THEN rules. TSK systems directly output continuous functions rather than discrete fuzzy sets, providing smooth interpolated predictions suited for control.

## Method

### Overall Architecture

Two-level hierarchical decomposition:
- **Level 1 (Antecedent Learning)**: K-Means clustering partitions the state space $\mathcal{S}$ into $N$ operational regions (e.g., "hovering," "correcting drift").
- **Level 2 (Consequent Learning)**: Weighted Ridge regression learns local TSK consequent functions within each region.
- **Global Inference**: Normalized weighted aggregation of outputs from active local models.

### Key Designs

#### 1. **Membership Function Design**

A membership function $\mu_{i,k}: \mathbb{R} \to [0,1]$ is defined for each cluster $i$ and state dimension $k$. Two types are compared:

**Gaussian membership function**:

$$\mu_{i,k}^{\text{Gauss}}(s_k) = \exp\left(-\frac{(s_k - c_{i,k})^2}{2\sigma_{i,k}^2}\right)$$

Characteristics: infinite support, smooth transitions; every state has nonzero membership in all clusters, causing many rules to activate simultaneously.

**Triangular membership function**:

$$\mu_{i,k}^{\text{Tri}}(s_k) = \max\left(0, \min\left(\frac{s_k - l_{i,k}}{c_{i,k} - l_{i,k}}, \frac{r_{i,k} - s_k}{r_{i,k} - c_{i,k}}\right)\right)$$

where $l_{i,k} = c_{i,k} - \beta\sigma_{i,k}$, $r_{i,k} = c_{i,k} + \beta\sigma_{i,k}$ ($\beta = 1.5$). Characteristics: compact support, sparse activation, more focused explanations.

**Design Motivation**: Triangular functions create localized activations (fewer active rules), making decision logic more transparent (hypothesis: higher FRAD without sacrificing fidelity).

#### 2. **Rule Firing and Global Inference**

Firing strength of rule $i$ for state $\mathbf{s}$:

$$\alpha_i(\mathbf{s}) = \prod_{k=1}^d \mu_{i,k}(s_k)$$

Under triangular functions, zero membership in any single dimension drives the entire rule activation to zero → sparse rule sets facilitate human understanding.

TSK consequent function (linear model):

$$f_i(\mathbf{s}) = \mathbf{w}_i^T \mathbf{s} + b_i$$

Global inference:

$$a_{\text{FCS}}(\mathbf{s}) = \frac{\sum_{i=1}^N \alpha_i(\mathbf{s}) \cdot f_i(\mathbf{s})}{\sum_{i=1}^N \alpha_i(\mathbf{s})}$$

#### 3. **Three Quantitative Interpretability Metrics**

**FRAD (Fuzzy Rule Activation Density)**: Inspired by the HHI index, measures explanatory focus:

$$\text{FRAD}(\mathbf{s}) = \sum_{i=1}^N \left(\frac{\alpha_i(\mathbf{s})}{\sum_j \alpha_j(\mathbf{s})}\right)^2$$

Range $[1/N, 1]$; higher values indicate greater focus (single rule dominates).

**FSC (Fuzzy Set Coverage)**: Validates linguistic vocabulary completeness by checking for regions with uniformly low membership across all fuzzy sets:

$$\text{FSC} = \frac{1}{|\mathcal{D}|} \sum_{\mathbf{s}} \frac{1}{d} \sum_{k=1}^d \max_i \mu_{i,k}(s_k)$$

**ASG (Action Space Granularity)**: Measures diversity of rule consequents:

$$\text{ASG} = \text{Var}(\{b_1, \ldots, b_N\})$$

High ASG indicates that rules represent distinct action modes (thrust/hover/correct).

### Loss & Training

- Level 1: K-Means clustering (standard configuration).
- Level 2: Weighted Ridge regression; sample weight for rule $i$ is $\alpha_i(\mathbf{s}_j)$:

$$\min_{\mathbf{w}_i, b_i} \sum_j \alpha_i(\mathbf{s}_j)(a_j - \mathbf{w}_i^T \mathbf{s}_j - b_i)^2 + \lambda\|\mathbf{w}_i\|^2$$

Ridge regularization prevents overfitting when rule support data is limited. $\lambda = 0.1$.

- Behavioral fidelity validation: DTW (Dynamic Time Warping) is used to compare temporal trajectory similarity between teacher and surrogate.

## Key Experimental Results

### Experimental Setup

- Environment: LunarLanderContinuous-v3 (state $\mathbb{R}^8$, action $[-1,1]^2$)
- Teacher: PPO (64-unit × 2-layer MLP), trained for 50,000 steps
- Data: 5,000 state-action pairs, 80/20 train/validation split
- Baselines: Decision Tree (16 leaves), Simple MLP (32 hidden units)
- FCS variants: Gaussian-16, Triangular-16/8/4
- 5 random seeds (42–46), paired t-test

### Main Results

| Model | Fidelity (%) | MSE | DTW | FRAD | FSC |
|-------|-------------|-----|-----|------|-----|
| Simple-MLP | 96.84±1.80 | 0.0016 | 0.55 | N/A | N/A |
| FCS-Tri-16 | **81.48±0.43** | 0.0053 | 1.05 | **0.814** | 0.933 |
| FCS-Gaus-16 | 81.38±0.64 | 0.0037 | 0.87 | 0.723 | **0.974** |
| DT-16 | 60.14±1.27 | 0.0074 | 1.32 | N/A | N/A |

### Ablation Study (Effect of Number of Rules)

| No. of Rules | Fidelity (%) | MSE | FRAD | FSC |
|-------------|-------------|------|------|-----|
| 4 | **97.83±0.5** | 0.00069 | **0.863** | 0.937 |
| 8 | 95.83±0.7 | 0.00119 | 0.801 | 0.963 |
| 16 | 81.48±0.4 | 0.00534 | 0.814 | 0.933 |

### Key Findings

1. **Fuzzy surrogate bridges the gap**: FCS-Tri achieves 21.34 percentage points higher fidelity than DT (81.48% vs. 60.14%), validating the superiority of fuzzy interpolation over piecewise constant approximation.
2. **Triangular functions yield more focused explanations**: Triangular and Gaussian functions achieve similar fidelity (p > 0.05), but FRAD is significantly higher for triangular (0.814 vs. 0.723, t=14.5, p < 0.001).
3. **"Less is more" phenomenon**: The 4-rule model achieves the highest fidelity (97.83%), suggesting that PPO policies can be distilled into minimal structures.
4. **Semantically interpretable rules**: Extracted rules such as "IF the lander drifts left at high altitude THEN apply upward thrust + rightward correction" carry genuine operational meaning.
5. **Temporal fidelity validation**: DTW distance of only 1.05 confirms that surrogate trajectories closely track those of the teacher.

## Highlights & Insights

1. **From qualitative to quantitative interpretability evaluation**: The FRAD/FSC/ASG metrics quantify explanation quality from complementary perspectives, moving beyond subjective claims of "interpretability."
2. **Hierarchical decoupled design**: Decoupling state partitioning (K-Means) from action reasoning (Ridge Regression) makes the fuzzy system scalable to continuous control.
3. **Profound "less is more" finding**: Complex neural network policies can be summarized nearly perfectly by just 4 simple rules, suggesting an intrinsic low-dimensional structure in PPO policies.
4. **Practical value for safety auditing**: 16 rules can be documented, reviewed, and certified — impossible for neural networks with millions of parameters.

## Limitations & Future Work

1. **Single-environment validation**: Evaluation is limited to Lunar Lander; generalization to more complex continuous control tasks (robotic manipulation, autonomous driving) remains unexplored.
2. **Simple teacher policy**: The PPO policy trained for only 50,000 steps may itself be relatively simple; distillation quality for more complex policies is unknown.
3. **Low-dimensional state space**: The 8-dimensional state space is modest; performance in high-dimensional visual input settings remains to be validated.
4. **Fixed cluster count**: K-Means requires the number of rules $N$ to be specified in advance, with no mechanism for automatically determining the optimal number.
5. **No online adaptation**: Distilled rules are fixed post-hoc and cannot be updated online to accommodate environmental changes.

## Related Work & Insights

- **VIPER (Bastani 2018)**: Distills Q-networks into decision trees; this paper improves upon it by replacing piecewise constant approximations with fuzzy systems.
- **SHAP/LIME**: Local explanation methods; this paper provides global structural explanations.
- **Fuzzy Q-Learning (Glorennec)**: Integrates fuzzy logic during training, but underperforms modern deep RL.
- **TSK system (Takagi-Sugeno-Kang)**: A fuzzy inference system that outputs continuous functions.
- The FRAD metric, inspired by the HHI index, cleverly transfers the concept of market concentration from economics to explanatory focus.

## Rating

- Novelty: ⭐⭐⭐⭐ — Hierarchical TSK combined with three new metrics constitutes a reasonably innovative framework.
- Experimental Thoroughness: ⭐⭐⭐ — Single-environment validation is insufficient, but metric design and statistical testing are rigorous.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, vivid rule examples, and complete derivations.
- Value: ⭐⭐⭐⭐ — A substantive step toward moving explainable RL from subjective assessment to quantitative evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning](pa-fas_towards_interpretable_and_generalizable_multimodal_face_anti-spoofing_via.md)
- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](../../ACL2026/reinforcement_learning/reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)
- [\[AAAI 2026\] DRMD: Deep Reinforcement Learning for Malware Detection under Concept Drift](drmd_deep_reinforcement_learning_for_malware_detection_under_concept_drift.md)
- [\[AAAI 2026\] ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing](charteditor_a_reinforcement_learning_framework_for_robust_chart_editing.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)

</div>

<!-- RELATED:END -->
