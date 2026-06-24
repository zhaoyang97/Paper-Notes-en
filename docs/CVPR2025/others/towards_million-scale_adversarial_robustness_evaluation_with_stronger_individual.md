---
title: >-
  [Paper Note] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks
description: >-
  [CVPR 2025][Probability Margin Attack] This paper proposes Probability Margin Attack (PMA), which defines the adversarial margin loss function in the probability space rather than the logits space. Its gradient is equivalent to an adaptively weighted combination of untargeted and targeted cross-entropy losses, consistently outperforming existing individual attack methods. Based on this, a million-scale evaluation dataset, CC1M, is constructed to conduct the first-ever million…
tags:
  - "CVPR 2025"
  - "Probability Margin Attack"
  - "adversarial robustness evaluation"
  - "white-box attack"
  - "million-scale evaluation"
  - "loss function design"
date: 2026-05-08
content_hash: b0738e8bcf157749
---

# Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks

**Conference**: CVPR 2025  
**arXiv**: [2411.15210](https://arxiv.org/abs/2411.15210)  
**Code**: None  
**Area**: Others (Adversarial Robustness)  
**Keywords**: Probability Margin Attack, adversarial robustness evaluation, white-box attack, million-scale evaluation, loss function design

## TL;DR

This paper proposes Probability Margin Attack (PMA), which defines the adversarial margin loss function in the probability space rather than the logits space. Its gradient is equivalent to an adaptively weighted combination of untargeted and targeted cross-entropy losses, consistently outperforming existing individual attack methods. Based on this, a million-scale evaluation dataset, CC1M, is constructed to conduct the first-ever million-scale white-box robustness evaluation of adversarial-trained models.

## Background & Motivation

**Background**: White-box adversarial attacks serve as core tools for evaluating the adversarial robustness of models. PGD is a powerful first-order attack, and AutoAttack (AA) is the standard evaluation method that integrates four attacks. Models on the RobustBench leaderboard use AA for robustness validation.

**Limitations of Prior Work**: PGD performs poorly in the presence of obfuscated gradients, and AA is computationally extremely expensive (often exceeding the training time), making it impractical for large-scale evaluation scenarios. Existing loss functions have limitations: cross-entropy explores all non-target directions simultaneously but struggles to focus on the optimal direction; targeted cross-entropy focuses on $z_{max}$ but suppresses the exploration of other attack directions; margin loss ($z_{max} - z_y$) only considers the top two logits, ignoring information from other categories; although DLR additionally considers the third largest logit, it still misses information from other dimensions.

**Key Challenge**: Individual attacks are efficient but lack sufficient strength, while ensemble attacks are strong but highly inefficient. Under million-scale evaluation demands, there is a critical need for efficient individual attack methods that approach the strength of ensemble attacks.

**Goal**: (1) Design a stronger loss function for individual attacks; (2) build an efficient ensemble attack scheme; (3) conduct million-scale robustness evaluations.

**Key Insight**: Shift the adversarial margin from the logits space to the probability space. The denominator in the probability space naturally contains logit information of all categories, opening up exploration possibilities for more attack directions.

**Core Idea**: Propose the Probability Margin (PM) loss $\mathcal{L}_{pm} = p_{max} - p_y = (e^{z_{max}} - e^{z_y}) / \sum_i^N e^{z_i}$, and mathematically prove that the gradient of PM loss is a probability-weighted combination of untargeted and targeted CE gradients, which naturally balances exploration and focusing.

## Method

### Overall Architecture

PMA adopts the two-stage pipeline of the Margin Decomposition (MD) attack, replacing the loss function with the Probability Margin loss. In Phase I, the two decomposed terms $p_{max}$ and $-p_y$ are alternately used for thorough exploration (alternating choice at each restart), and in Phase II, the complete $p_{max} - p_y$ is used for fine optimization. The step size utilizes a cosine decay strategy. Based on this, two ensemble attacks (PMA+AA and PMA+APGD) are constructed to balance efficiency and strength, while the CC1M million-scale evaluation dataset is established.

### Key Designs

1. **Probability Margin Loss (PM Loss)**:

    - **Function**: Defines the adversarial margin in the probability space, unifying the advantages of untargeted and targeted attacks.
    - **Mechanism**: $\mathcal{L}_{pm}(z, y) = p_{max} - p_y = (e^{z_{max}} - e^{z_y}) / \sum_i^N e^{z_i}$. Its gradient can be decomposed as $\nabla_x \mathcal{L}_{pm} = p_y \nabla_x \mathcal{L}_{ce} + p_{max} \nabla_x \mathcal{L}_{cet}$, which is a weighted combination of untargeted CE (multi-directional exploration) and targeted CE (focused attack on $z_{max}$). The weights are dynamically adjusted by $p_y$ and $p_{max}$: as $p_{max}$ approaches $p_y$, the contributions from other directions are regularized and suppressed, automatic focusing the attack on the $z_{max}$ direction.
    - **Design Motivation**: Compared to margin loss which uses only two logits, the denominator $\sum e^{z_i}$ of PM loss contains information from all classes, enriching the attack directions without increasing computational cost. Theoretical analysis shows that margin loss is equivalent to a simple addition of CE and targeted CE, whereas PM loss is its adaptively weighted version.

2. **Two-Stage Attack Strategy (PMA Attack Pipeline)**:

    - **Function**: Optimizes the two components of PM loss alternately across stages to thoroughly explore the attack space.
    - **Mechanism**: Phase I ($k < K^1$): odd restarts optimize $p_{max}$ (promoting the probability rise of the maximum non-target class), even restarts optimize $-p_y$ (promoting the probability drop of the true class), achieving directional exploration through the parity of restart iterations. Phase II ($K^1 < k < K$): uses the complete PM loss $p_{max} - p_y$ to perform fine-grained optimization starting from the good initialization provided by Phase I. Step size uses cosine decay: $\alpha = \epsilon \cdot (1 + \cos(k/K \cdot \pi))$.
    - **Design Motivation**: Draws inspiration from the margin decomposition concept of the MD attack, but replaces the decomposition in the logits space with that in the probability space. It provides better initial perturbations for Phase II through thorough exploration in Phase I.

3. **CC1M Million-Scale Evaluation Dataset**:

    - **Function**: Constructs the first million-scale white-box adversarial robustness evaluation benchmark.
    - **Mechanism**: Starting from the CC3M dataset, 1 million images are retained after filtering outliers. Adversarial-trained ImageNet models are evaluated on CC1M, comparing the results against evaluations on the standard ImageNet-1k test set (50k images). Two ensemble attacks, PMA+APGD and PMA+AA, are built to achieve a balance between efficiency and strength.
    - **Design Motivation**: Existing evaluations are conducted mostly on the 50k test set, which may not reflect real-world adversarial risk distributions. Million-scale evaluation can reveal robustness gaps overlooked by small-scale evaluations.

### Loss & Training

PMA uses the Probability Margin loss $\mathcal{L}_{pm} = p_{max} - p_y$ as the adversarial objective, combined with $L_\infty$ constraints and PGD projection. The perturbation budget is $\epsilon = 8/255$ for CIFAR-10/100, and $\epsilon = 4/255$ for ImageNet. The attack runs for 100 steps using random initialization and cosine step size decay.

## Key Experimental Results

### Main Results

PM loss vs. existing loss functions (PGD strategy, robust accuracy %, lower is stronger):

| Dataset | Model | CE | DLR | Margin | **PM (Ours)** | Decrease |
|--------|------|-----|-----|--------|-------------|------|
| CIFAR-10 | WRN-70-16 | 73.60 | 71.61 | 71.56 | **71.34** | -0.22 |
| CIFAR-10 | WRN-70-16 (Rebuffi) | 69.56 | 67.96 | 67.79 | **67.55** | -0.24 |
| CIFAR-100 | WRN-70-16 | 48.19 | 43.93 | 43.90 | **43.66** | -0.24 |
| ImageNet | Swin-L | 59.47 | 59.80 | 59.31 | **57.97** | -1.34 |
| ImageNet | ConvNeXt-L | 58.42 | 59.17 | 58.63 | **57.24** | -1.18 |

PMA vs. existing individual attack methods (robust accuracy %):

| Dataset | Model | MD | FAB | **PMA (Ours)** | vs AA |
|--------|------|-----|-----|---------------|-------|
| CIFAR-10 | WRN-70-16 (Peng) | 71.14 | 72.31 | **71.10** | 71.10 |
| CIFAR-100 | WRN-70-16 | 43.53 | 44.73 | **43.39** | 43.40 |
| ImageNet | Swin-L | 56.89 | 60.05 | **56.49** | 56.56 |

### Ablation Study

| Configuration | CIFAR-10 Avg. | ImageNet Avg. | Description |
|------|-------------|-------------|------|
| PM loss (full) | **Best** | **Best** | Full probability margin |
| Only $p_{max}$ term | Second | Second | Lacks the $p_y$ direction |
| Only $-p_y$ term | Worst | Worst | Lacks target focusing |
| Margin loss | Intermediate | Intermediate | Margin in logits space |

### Key Findings

- PM loss consistently outperforms CE, DLR, and margin loss across all 29 models, with the advantage becoming more pronounced on datasets with more classes (averaging a 1.22% reduction in robust accuracy on ImageNet).
- As an individual attack, PMA approaches or even matches the performance of AutoAttack (which ensembles four methods), while being several times more computationally efficient.
- Million-scale evaluation reveals a significant robustness gap: the evaluated robustness on CC1M is significantly lower than that on the ImageNet-1k test set, indicating a risk of overestimation in small-scale evaluations.
- The adaptive weighting mechanism of PM loss is core to its advantage—it automatically enhances attack focus when $p_{max}$ is close to the decision boundary.

## Highlights & Insights

- The theoretical analysis of PM loss is elegant: a simple formula gracefully unifies the dual advantages of untargeted and targeted CE, and the gradient derivation $\nabla \mathcal{L}_{pm} = p_y \nabla \mathcal{L}_{ce} + p_{max} \nabla \mathcal{L}_{cet}$ clearly reveals its adaptive mechanism.
- Million-scale evaluation reveals for the first time that small-scale test sets may overestimate model robustness, presenting a crucial warning for safety-critical applications.
- The combination of the MD attack pipeline and PM loss is highly worth learning from—a good attack strategy and a good loss function represent orthogonal dimensions of improvement.

## Limitations & Future Work

- The advantage of PM loss is relatively small on CIFAR-10 (averaging ~0.2%), with its primary advantages realized on datasets with many categories.
- The distribution of the CC1M dataset is not completely aligned with ImageNet; the robustness gap might partially stem from distribution shift rather than insufficient robustness evaluation.
- It only focuses on $L_\infty$ attacks, leaving its effectiveness under other norm constraints such as $L_2$ and $L_1$ unverified.
- Future directions: (1) Extend PM loss to the adversarial evaluation of generative models; (2) study the practical impact of even larger-scale (e.g., ten-million scale) evaluations; (3) explore the application of PM loss in adversarial training.

## Related Work & Insights

- **vs AutoAttack**: AA integrates four attacks (2×APGD + FAB + Square) and has a heavy computational cost. PMA, as a single attack, already matches AA's performance (on some models, PMA's robust accuracy is even lower than AA's).
- **vs Margin Decomposition (MD)**: PMA adopts MD's two-stage pipeline but replaces the loss function, further reducing the evaluated robust accuracy on top of MD.
- **vs DLR loss**: DLR considers the third largest logit in the logits space for normalization, whereas PM loss naturally incorporates all category information via the softmax denominator in the probability space.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The formula for PM loss is simple and elegant, and the gradient unification analysis has theoretical depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage across 29 models, 3 datasets, and individual/ensemble/million-scale evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear gradient analysis and derivation; intuitive comparison table of the five loss functions.
- **Value**: ⭐⭐⭐⭐ — Practical contribution to the adversarial robustness evaluation community; PM loss can be directly applied for more accurate robustness measurements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)
- [\[CVPR 2026\] IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](../../CVPR2026/others/irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)
- [\[CVPR 2025\] TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions](taet_two-stage_adversarial_equalization_training_on_long-tailed_distributions.md)
- [\[CVPR 2025\] STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)
- [\[ICML 2025\] Probably Approximately Global Robustness Certification](../../ICML2025/others/probably_approximately_global_robustness_certification.md)

</div>

<!-- RELATED:END -->
