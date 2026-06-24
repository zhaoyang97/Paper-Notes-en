---
title: >-
  [Paper Note] Decoupled Entropy Minimization
description: >-
  [NeurIPS 2025][LLM (Other)][entropy minimization] This paper decouples classical entropy minimization (EM) into two opposing components — the Cluster Aggregation Driving Factor (CADF, which rewards dominant classes) and the Gradient Mitigation Calibrator (GMC, which penalizes high-confidence classes) — revealing two inherent flaws of classical EM (reward collapse and easy-class bias). The proposed AdaDEM addresses these issues via normalized rewards and marginal entropy calib…
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "entropy minimization"
  - "domain adaptation"
  - "test-time adaptation"
  - "self-supervised learning"
  - "reward collapse"
date: 2026-05-08
content_hash: 6d33da2c1f376a57
---

# Decoupled Entropy Minimization

**Conference**: NeurIPS 2025
**arXiv**: [2511.03256](https://arxiv.org/abs/2511.03256)  
**Code**: [https://github.com/HAIV-Lab/DEM](https://github.com/HAIV-Lab/DEM)  
**Area**: LLM Evaluation
**Keywords**: entropy minimization, domain adaptation, test-time adaptation, self-supervised learning, reward collapse

## TL;DR
This paper decouples classical entropy minimization (EM) into two opposing components — the Cluster Aggregation Driving Factor (CADF, which rewards dominant classes) and the Gradient Mitigation Calibrator (GMC, which penalizes high-confidence classes) — revealing two inherent flaws of classical EM (reward collapse and easy-class bias). The proposed AdaDEM addresses these issues via normalized rewards and marginal entropy calibration, achieving significant improvements across semi-supervised learning, domain adaptation, reinforcement learning, and other tasks.

## Background & Motivation

**Background**: Entropy minimization (EM) is a widely adopted self-supervised optimization method in machine learning, which reduces class overlap and bridges domain gaps by minimizing the conditional entropy of model predictions. EM has been extensively applied in semi-supervised learning, clustering, domain adaptation, online learning, and reinforcement learning.

**Limitations of Prior Work**: Despite its simplicity and generality, EM yields limited performance gains, and prior literature has noted that its potential is constrained. However, the internal mechanism of EM — specifically how it effectively optimizes model parameters in an unsupervised manner — has never been systematically analyzed.

**Key Challenge**: The conditional entropy objective of EM is highly coupled, containing two opposing effects whose coupled form prevents them from being optimized independently, leading to two critical problems.

**Goal**: (1) Reveal the internal mechanism of EM; (2) Explain why EM's performance is limited; (3) Propose an improved EM variant that requires no hyperparameter tuning.

**Key Insight**: By rewriting the conditional entropy as $H(\mathbf{z}) = -\sum p_i z_i + \log\sum e^{z_i}$ and analyzing the gradient behavior of each term separately, the two terms are found to have completely opposing effects.

**Core Idea**: Decouple entropy minimization into a reward factor (CADF) and a penalty calibrator (GMC); eliminate reward collapse via normalized rewards and replace GMC with marginal entropy to eliminate easy-class bias, achieving hyperparameter-free adaptive EM.

## Method

### Overall Architecture
The conditional entropy is expressed as $H(\mathbf{z}) = \underbrace{-\sum p_i z_i}_{\text{CADF}} + \underbrace{\log\sum e^{z_i}}_{\text{GMC}}$. Minimizing CADF rewards the dominant class (sharpening outputs), while minimizing GMC penalizes high-confidence classes (smoothing outputs). Their coupling limits the effectiveness of EM. AdaDEM optimizes these two components separately.

### Key Designs

1. **Decoupled Analysis of CADF and GMC**:

    - Function: Decompose the conditional entropy into two independent components and analyze the gradient behavior of each.
    - Mechanism: The gradient of CADF, $R_T = p_i(T(\mathbf{z}) + z_i + 1)$, grants greater rewards to high-probability classes (positive reinforcement). The gradient of GMC, $R_Q = -p_i$, always penalizes all classes, with higher-confidence classes receiving heavier penalties. In classical EM, the two terms are coupled and partially cancel each other.
    - Design Motivation: To understand the internal mechanism of EM and provide a theoretical foundation for subsequent improvements.

2. **Reward Collapse**:

    - Function: Identify the phenomenon whereby high-certainty samples make diminishing contributions to learning.
    - Mechanism: As the predicted probability approaches 1.0, the gradient magnitude of classical EM approaches 0 — meaning highly certain samples contribute almost nothing to learning. Yet these samples are precisely the most reliable signals in self-supervised learning.
    - Design Motivation: AdaDEM addresses this by normalizing the conditional entropy using the L1-norm of the CADF gradient ($\delta$), so that the rewards for high-certainty samples are amplified rather than suppressed.

3. **Easy-Class Bias**:

    - Function: Identify the severe misalignment between the output distribution and the label distribution.
    - Mechanism: Classical EM tends to assign the majority of samples to dominant or easy classes, causing severe skewness in the class distribution. This is particularly harmful under noisy or imbalanced data conditions.
    - Design Motivation: AdaDEM replaces GMC with a Marginal Entropy Calibrator (MEC), which maximizes the marginal entropy $H(Y)$ to encourage a uniform class size distribution. MEC dynamically estimates the class distribution rather than assuming a uniform label prior.

4. **DEM* and AdaDEM**:

    - Function: DEM* is an upper-bound variant of classical EM (requiring hyperparameter search); AdaDEM is a hyperparameter-free improved version.
    - Mechanism: DEM* introduces a temperature $\tau$ (adjusting the reward curve) and a weight $\alpha$ (controlling the influence of GMC), with the optimal $(\tau^*, \alpha^*)$ found via search. AdaDEM eliminates the need for these hyperparameters through normalization and MEC.
    - Design Motivation: DEM* demonstrates that EM has greater potential; AdaDEM matches or surpasses DEM* without any hyperparameter search.

### Loss & Training
The AdaDEM loss consists of a normalized CADF term and a MEC term. No manual hyperparameter tuning is required. It can serve as a plug-and-play replacement for existing methods such as Tent and pseudo-labeling.

## Key Experimental Results

### Main Results (Test-Time Adaptation)

| Method | Single-domain TTA (ResNet50) | Continual TTA (ResNet50) |
|--------|------------------------------|--------------------------|
| NoAdapt | 31.5 | 31.5 |
| EM (Tent) | 40.0 | 31.2 |
| CADF only | 41.7 (+1.7) | 36.1 (+4.9) |
| DEM* (hyperparameter search) | 41.8 (+1.8) | 39.0 (+7.8) |
| AdaDEM-Norm (w/o MEC) | 43.7 (+3.7) | 37.5 (+6.3) |
| AdaDEM-MEC (w/o Norm) | 44.4 (+4.4) | 37.5 (+6.3) |
| **AdaDEM (full)** | **Best** | **Best** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| CADF alone | Large improvement but poor robustness (sensitive to distribution shift) |
| + Temperature $\tau$ | Improves robustness |
| + GMC weight $\alpha$ | Prevents overfitting on noisy tasks |
| Normalization (Norm) | Resolves reward collapse, contributing +3.7 |
| MEC | Resolves easy-class bias, contributing +4.4 |
| AdaDEM learning rate tolerance range | Expanded by 10× compared to classical EM |

### Key Findings
- **CADF alone significantly outperforms classical EM**: This indicates that the penalty effect of GMC is harmful in many settings, as it suppresses the beneficial reward signal from CADF.
- **AdaDEM surpasses DEM*** (the upper-bound variant requiring hyperparameter search on target data) without requiring any hyperparameters.
- **Learning rate tolerance expands by 10×**: Classical EM is sensitive to the learning rate, whereas AdaDEM is substantially more robust.
- **Cross-task generality**: AdaDEM is effective across semi-supervised learning, unsupervised clustering, domain adaptation, and reinforcement learning (as a replacement for the entropy bonus).

## Highlights & Insights
- **Decoupling the conditional entropy into CADF + GMC is a profound theoretical contribution**: The rewriting $H(\mathbf{z}) = -\sum p_i z_i + \log\sum e^{z_i}$ appears simple yet reveals a fundamental tension in EM — one term promotes sharpness while the other promotes uniformity.
- **The concept of reward collapse** parallels reward shaping issues in reinforcement learning and deserves broader attention in the machine learning community.
- **MEC's avoidance of a uniform label distribution assumption** is a practically important improvement: class distributions in real-world scenarios are rarely uniform, making dynamic estimation more reliable than a fixed prior.

## Limitations & Future Work
- **Theoretical analysis is primarily grounded in classification settings**: Applicability to regression or generative tasks remains unexplored.
- **Dynamic estimation in MEC may lag in non-stationary environments**: If the class distribution shifts drastically, the estimates may become inaccurate.
- **Suggested direction**: Extend AdaDEM to entropy-based decoding strategies for large language models.

## Related Work & Insights
- **vs. Tent (Wang et al.)**: Tent directly employs classical EM; AdaDEM serves as a drop-in upgrade to its EM component.
- **vs. SHOT/LAME**: These domain adaptation methods use classical EM as their loss function, which can be directly replaced by AdaDEM.
- **vs. Entropy bonus in RL**: The entropy bonus in algorithms such as PPO is equivalent to entropy maximization; the analysis in this paper may inspire the design of improved exploration bonuses.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The decoupled analysis perspective is unique; the concepts of reward collapse and easy-class bias are novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across four task categories with comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, though the paper structure is slightly verbose
- Value: ⭐⭐⭐⭐⭐ Foundational contribution with strong practical utility due to plug-and-play design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](../../ACL2025/llm_nlp/systematic_generalization_in_language_models_scales_with_information_entropy.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](../../ACL2025/llm_nlp/derta_decoupled_refusal.md)
- [\[AAAI 2026\] Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLMs](../../AAAI2026/llm_nlp/uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)
- [\[NeurIPS 2025\] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)

</div>

<!-- RELATED:END -->
