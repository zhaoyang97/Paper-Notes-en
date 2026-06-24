---
title: >-
  [Paper Note] Credal Ensemble Distillation for Uncertainty Quantification
description: >-
  [AAAI 2026][AI Safety][Knowledge Distillation] Propounds the Credal Ensemble Distillation (CED) framework, which distills a deep ensemble teacher into a single model, CREDIT. Instead of a single softmax distribution, this model predicts interval-valued class probabilities that define a credal set. It achieves superior or comparable uncertainty estimation on OOD detection tasks while substantially reducing inference overhead (reducing inference time from $5\times$ to $1\times$…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Knowledge Distillation"
  - "Deep Ensembles"
  - "Uncertainty Quantification"
  - "Credal Sets"
  - "OOD Detection"
date: 2026-05-08
content_hash: 06868300f3224ab5
---

# Credal Ensemble Distillation for Uncertainty Quantification

**Conference**: AAAI 2026  
**arXiv**: [2511.13766](https://arxiv.org/abs/2511.13766)  
**Code**: None (experimental code is provided in the supplementary material)  
**Area**: Model Compression / Uncertainty Quantification  
**Keywords**: Knowledge Distillation, Deep Ensembles, Uncertainty Quantification, Credal Sets, OOD Detection

## TL;DR

Propounds the Credal Ensemble Distillation (CED) framework, which distills a deep ensemble teacher into a single model, CREDIT. Instead of a single softmax distribution, this model predicts interval-valued class probabilities that define a credal set. It achieves superior or comparable uncertainty estimation on OOD detection tasks while substantially reducing inference overhead (reducing inference time from $5\times$ to $1\times$).

## Background & Motivation

Uncertainty quantification (UQ) in deep neural networks is crucial for model trustworthiness and robustness. Uncertainty is categorized into two types:
- **Aleatoric Uncertainty (AU)**: The intrinsic randomness in the data-generation process.
- **Epistemic Uncertainty (EU)**: The uncertainty arising from a lack of model knowledge.

**Deep Ensembles (DE)**, which combine multiple independently trained networks, have become a strong baseline for UQ and can effectively distinguish between AU and EU. However, their primary bottleneck is the **high inference cost**: $M$ models entail $M$ times the computation and storage resources.

Limitations of prior work in distillation:

**Ensemble Distillation (ED)**: Distills a DE into a Single Neural Network (SNN) that outputs a single softmax distribution, which discards EU information (as a single distribution cannot express uncertainty about uncertainty).

**Ensemble Distribution Distillation (EDD)**: Distills the ensemble into a model that outputs a Dirichlet distribution, but lacks ground-truth Dirichlet labels. Furthermore, Dirichlet models suffer severe drops in accuracy in practice (only $74.56\%$ on VGG16 vs. $91.79\%$ for SNN), and recent theoretical work has criticized their interpretation of EU.

**Key Challenge**: How to simultaneously preserve classification capability and EU quantification performance during a single forward pass?

**Key Insight**: Utilize **credal sets** (convex sets of probability distributions, defined by interval-valued class probabilities) as a second-order representation of uncertainty, replacing single or parameterized distributions. Credal sets are naturally derived from the multiple predictive probabilities of a DE, without requiring distributional assumptions like Dirichlet models do.

## Method

### Overall Architecture

CED consists of three stages: (1) extracting probability intervals and crossing probabilities from the $M$ softmax outputs of the DE teacher using a credal wrapper; (2) designing the CREDIT student model, which outputs an $\mathbb{R}^{2C+1}$ vector encoding crossing probabilities, interval widths, and a weight factor; and (3) training the student with a new loss function to match the credal information of the teacher. During inference, classification relies on the crossing probabilities, while UQ is performed by reconstructing the full credal set from the outputs.

### Key Designs

1. **Credal Wrapper (Teacher-side)**:

    - **Function**: Extracts interval-valued class probabilities from the $M$ predictive probabilities of a DE.
    - **Mechanism**: For each class $k$, the upper bound is $\overline{p}_k = \max_m p_{m,k}$ and the lower bound is $\underline{p}_k = \min_m p_{m,k}$. These intervals define a credal set $\mathbb{Q}$. From this, a normalized crossing probability $p^*_k = \underline{p}_k + \beta(\overline{p}_k - \underline{p}_k)$ is computed, where $\beta = (1-\sum_k \underline{p}_k)/\sum_k \Delta p_k$.
    - **Design Motivation**: The crossing probability serves as the most representative point-estimate of the interval-valued probability system. The scaling factor $\beta$ ensures that the crossing probabilities are properly normalized.

2. **CREDIT Student Architecture**:

    - **Function**: Modifies the final layer of a standard SNN to output $2C+1$ values.
    - **Mechanism**: The first $C$ logits pass through a softmax to yield the crossing probability $\mathbf{p}_S^*$, the next $C$ pass through a sigmoid to determine the interval widths $\Delta\mathbf{p}_S$, and the last output passes through a sigmoid to yield the weight factor $\beta_S$. The probability intervals can be reconstructed from these three outputs: $\underline{p}_{S,k} = p^*_{S,k} - \beta_S \Delta p_{S,k}$ and $\overline{p}_{S,k} = p^*_{S,k} + (1-\beta_S)\Delta p_{S,k}$.
    - **Design Motivation**: The crucial constraint is to ensure that the reconstructed probability intervals remain valid (specifically, $\underline{p} \leq p^* \leq \overline{p}$ and $\sum \underline{p} \leq 1 \leq \sum \overline{p}$). This is guaranteed mathematically through the combination of softmax and sigmoid activations.

3. **Distillation Loss Function**:

    - **Function**: Trains CREDIT to mimic the credal information of the DE teacher.
    - **Mechanism**: $\mathcal{L}_{ced} = \text{CE}(\mathbf{p}^*, \mathbf{p}_S^*) + \text{MSE}(\Delta\mathbf{p}, \Delta\mathbf{p}_S) + \text{MSE}(\beta, \beta_S)$. The first term (cross-entropy) preserves predictive performance, while the latter two terms (mean squared error) capture the imprecision of the credal set.
    - **Design Motivation**: Decomposes the credal set distillation into three independently optimizable objectives: accurate classification, interval width, and interval location. It supports temperature scaling ($T=2.5$).

### Uncertainty Quantification

Computed from the reconstructed credal set $\mathbb{Q}_S$ of CREDIT by solving constrained optimization problems:
- **TU** (Total Uncertainty) = Maximum Shannon entropy $\overline{H}(\mathbb{Q}_S)$ (the probability vector with maximum entropy within the credal set)
- **AU** (Aleatoric Uncertainty) = Minimum Shannon entropy $\underline{H}(\mathbb{Q}_S)$
- **EU** (Epistemic Uncertainty) = $\overline{H} - \underline{H}$ (where interval width reflects the lack of model knowledge)

## Key Experimental Results

### Main Results (VGG16, CIFAR10 vs. SVHN OOD Detection)

| Method | AUROC(EU) | AUROC(TU) | AUPRC(EU) | AUPRC(TU) | Inference Time |
|------|----------|----------|----------|----------|---------|
| DE (5×) | 89.99 | 91.53 | 93.78 | 95.09 | 5×2.22s |
| SNN | / | 89.44 | / | 93.71 | 2.22s |
| ED | / | 91.07 | / | 94.51 | 2.22s |
| EDD* | 90.94 | 90.96 | 93.66 | 93.78 | 2.22s |
| MCDO | 51.42 | 89.12 | 74.72 | 93.64 | 2.22s |
| **CED** | **93.56** | **92.51** | **96.09** | **95.21** | **2.26s** |

### Ablation Study (ResNet50 + CIFAR10-C OOD)

| Method | AUROC(EU)↑ | AUROC(TU)↑ | Accuracy | Description |
|------|-----------|-----------|--------|------|
| DE | 87.78 | 94.08 | 93.40 | 5-model ensemble, upper performance limit |
| CED | **96.80** | 95.23 | 91.77 | Single model, EU outperforms DE |
| ED | / | 94.09 | 92.02 | No EU estimation capability |
| EDD* | 89.48 | 91.04 | 80.38 | Severe degradation in accuracy |

### Key Findings

- **CED's EU estimation significantly outperforms all baselines**: On VGG16/SVHN, CED's EU-AUROC ($93.56\%$) substantially exceeds that of DE ($89.99\%$) and EDD* ($90.94\%$), suggesting that credal sets capture EU better than the discrete sampling of DE and Dirichlet distributions.
- **CED does not compromise on accuracy**: CED ($92.23\%$) is on par with ED ($92.18\%$) and SNN ($91.79\%$), whereas the accuracy of EDD (VGG16) plunges to $74.56\%$.
- **EU vs. TU**: OOD detection using CED's EU generally outperforms that using TU, while other methods perform better with TU. This indicates a qualitative advancement in the quality of CED's EU estimation.
- **Inference Efficiency**: The inference time of CED ($2.26$s) is nearly identical to that of SNN ($2.22$s), whereas DE requires $5 \times 2.22\text{s} = 11.1\text{s}$.
- **Ensemble Size Ablation**: DE continuously improves with larger ensemble sizes, while CED approaches convergence at $M=5$, demonstrating the efficacy of distillation.
- **Temperature Scaling**: $T=2.5$ yields the best result, whereas excessively high temperatures like $T=10$ degrade performance.
- **Medical Imaging Case Study** (Camelyon17): CED's EU-AUROC ($97.12\%$) outperforms DE ($95.92\%$) under OOD settings.

## Highlights & Insights

- Introducing credal sets into knowledge distillation is an elegant innovation: as a second-order representation, a credal set is more flexible than Dirichlet distributions (requiring no distributional assumptions) and more compact than DE (utilizing a single model).
- CREDIT features a highly minimalist architectural design (adding only $C+1$ output nodes), making it non-intrusive to the backbone.
- It is mathematically proven that the probability intervals output by CREDIT are always valid (satisfying credal set conditions). Such guarantees of correctness are highly desirable in engineering.
- The loss function design is highly intuitive, using CE to preserve classification and MSE to maintain imprecision, avoiding complex learning strategies like those in EDD.

## Limitations & Future Work

- **Class Count Limitations**: When $C$ is very large (e.g., 100 or 1000 classes), the softmax probabilities become extremely small, potentially causing instability in the regression loss.
- **Insufficient Calibration**: The Expected Calibration Error (ECE) of CED ($6.71\%$) is much higher than that of DE ($1.46\%$), indicating a need for improved calibration.
- **Optimization Solver Overhead**: Computing $\overline{H}$ and $\underline{H}$ requires solving constrained optimizations; the overhead may become non-negligible when $C > 10$.
- **Classification Only**: The generalizability to other tasks like regression and detection has yet to be explored.
- **Dependence on Teacher Quality**: The upper-bound performance of CED is constrained by the quality of the DE teacher.

## Related Work & Insights

- Compared to BNNs: CED does not require a weight posterior distribution, simplifying the training process.
- Compared to EDD: CED circumvents the issue of missing ground-truth Dirichlets and avoids accuracy degradation.
- Credal sets have a solid theoretical foundation in traditional machine learning (Levi 1980, imprecise probability theory). Integrating them into deep learning distillation establishes an effective link between theory and practice.
- **Insight**: Interval-valued probabilities are better suited to representing "what we do not know" than point estimates or parameterized distributions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The combination of credal sets and distillation is proposed for the first time, with clear theoretical motivation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple backbones, datasets, ablations, and a medical case study)
- Writing Quality: ⭐⭐⭐⭐ (Content-dense yet clearly structured)
- Value: ⭐⭐⭐⭐⭐ (Expected to become a new standard method in the UQ domain, with high practical utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](../../ICML2026/ai_safety/position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[ICLR 2026\] Uncertainty Estimation via Hyperspherical Confidence Mapping](../../ICLR2026/ai_safety/uncertainty_estimation_via_hyperspherical_confidence_mapping.md)
- [\[ICLR 2026\] Robust Adversarial Quantification via Conflict-Aware Evidential Deep Learning](../../ICLR2026/ai_safety/robust_adversarial_quantification_via_conflict-aware_evidential_deep_learning.md)
- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)

</div>

<!-- RELATED:END -->
