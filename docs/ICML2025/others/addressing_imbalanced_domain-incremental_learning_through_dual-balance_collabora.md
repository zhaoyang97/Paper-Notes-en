---
title: >-
  [Paper Note] Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)
description: >-
  [ICML 2025][Domain-Incremental Learning] DCE proposes a two-stage training framework of a frequency-aware expert group + a dynamic expert selector to simultaneously resolve the two challenges of intra-domain class imbalance and cross-domain class distribution shift in domain-incremental learning, achieving state-of-the-art (SOTA) performance on four benchmarks.
tags:
  - "ICML 2025"
  - "Domain-Incremental Learning"
  - "Class Imbalance"
  - "Mixture of Experts"
  - "Continual Learning"
  - "Pre-trained Models"
date: 2026-05-08
content_hash: 4454d5f03c796d0e
---

# Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)

**Conference**: ICML 2025  
**arXiv**: [2507.07100](https://arxiv.org/abs/2507.07100)  
**Code**: [https://github.com/Lain810/DCE](https://github.com/Lain810/DCE)  
**Area**: LLM Evaluation  
**Keywords**: Domain-Incremental Learning, Class Imbalance, Mixture of Experts, Continual Learning, Pre-trained Models

## TL;DR

DCE proposes a two-stage training framework of a frequency-aware expert group + a dynamic expert selector to simultaneously resolve the two challenges of intra-domain class imbalance and cross-domain class distribution shift in domain-incremental learning, achieving state-of-the-art (SOTA) performance on four benchmarks.

## Background & Motivation

**Background**: Domain-Incremental Learning (DIL) requires models to continuously learn in non-stationary environments with continuously changing data distributions while retaining historical knowledge. Current pre-trained model (PTM)-based DIL methods mainly adopt two paradigms: shared prompts (e.g., L2P, CODA-Prompt) and domain-specific prompts (e.g., S-iPrompt), which mitigate catastrophic forgetting by freezing the PTM backbone parameters and introducing lightweight adaptation modules.

**Limitations of Prior Work**: Class imbalance is a widespread phenomenon in real-world data (e.g., in autonomous driving, traffic sign samples under extreme weather are far fewer than those in normal scenarios), but existing DIL methods rarely consider this issue. Class imbalance manifests in two dimensions in DIL: (a) **Intra-domain class imbalance**—the number of samples across different classes within a single domain varies drastically, leading the model to overfit many-shot classes and underlearn few-shot classes; (b) **Cross-domain class distribution shift**—the sample frequency of the same class differs across different domains, causing the "many/few" identity of classes to change with the domain.

**Key Challenge**: Shared prompt methods promote knowledge transfer by sharing the feature space, which improves performance on few-shot classes but triggers catastrophic forgetting of many-shot classes. Domain-specific prompt methods reduce forgetting through parameter isolation but block cross-domain knowledge sharing, preventing few-shot classes from benefiting from new domain data. This forms a fundamental trade-off between **knowledge sharing vs. forgetting resistance**.

**Goal**: (a) How to balance the learning of classes with different frequencies during the training process of a single domain? (b) How to simultaneously preserve the knowledge of many-shot classes and utilize new domain data to improve the generalization of few-shot classes in cross-domain scenarios?

**Key Insight**: The authors observe that different loss functions naturally favor classes of different frequencies—CE loss favors many-shot classes, balanced softmax loss pursues balance, and inverse frequency loss favors few-shot classes. By letting multiple experts independently train with different loss functions, the learning needs of all frequency groups can be covered. Meanwhile, by generating balanced pseudo-features using Gaussian sampling to train the expert selector, balanced fusion of cross-domain knowledge can be achieved without relying on real data.

**Core Idea**: Training a frequency-aware expert group with different loss functions to address intra-domain imbalance, and then training a dynamic expert selector based on ancestral class statistics through balanced Gaussian sampling to resolve cross-domain distribution shifts.

## Method

### Overall Architecture

DCE adopts a two-stage training paradigm. Upon the arrival of each new domain:

- **First Stage (Frequency-Aware Expert Training)**: Following a frozen pre-trained ViT encoder, three parallel expert modules are deployed. Each expert is trained independently using a different loss function, specializing in feature learning of many-shot, balanced, and few-shot classes, respectively. Visual prompts are learned during the first task and frozen thereafter.
- **Second Stage (Dynamic Expert Selector Training)**: Crucial statistics (mean and covariance) of each class are calculated using the frozen feature encoder. A synthetic pseudo-feature dataset is generated through **balanced Gaussian sampling**. These balanced data are then used to train an MLP selector, assigning soft weights to each expert.

During inference, when an input is processed to extract features through the encoder, all experts calculate outputs in parallel, which are then weighted and fused by the selector to obtain the final prediction.

### Key Designs

1. **Frequency-Aware Expert Group**:

    - Function: Three parallel MLP+classifier modules, each biased toward learning classes in different frequency groups.
    - Mechanism: The first expert $e_b$ utilizes standard CE loss $\ell_{CE}$, naturally favoring many-shot classes; the second expert $e_{b+1}$ uses balanced softmax loss $\ell_{Bal} = -\log \frac{\exp(v_y^2 + \log p_b^y)}{\sum_j \exp(v_j^2 + \log p_b^j)}$, correcting bias by introducing class frequency priors to achieve balanced predictions; the third expert $e_{b+2}$ uses inverse distribution loss $\ell_{Rev} = -\log \frac{\exp(v_y^3 + 2\log p_b^y)}{\sum_j \exp(v_j^3 + 2\log p_b^y)}$, reversing the training distribution to emphasize few-shot classes. The three losses are optimized independently, with the total loss being $\ell_{exp} = \ell_{CE} + \ell_{Bal} + \ell_{Rev}$.
    - Design Motivation: A single loss function cannot cater to classes of all frequency groups. CE and inverse losses provide complementary extreme biases, with the balanced loss situated in the middle, collaboratively covering the entire frequency spectrum. This design "decomposes" the class imbalance problem into different experts, where each expert only needs to perform well on its preferred frequency group.

2. **Dynamic Expert Selector**:

    - Function: An MLP network $s(\cdot) : \mathbb{R}^d \to \mathbb{R}^{3b}$ that assigns importance weights to all globally accumulated $3b$ experts.
    - Mechanism: (a) After training the experts in each domain, the frozen encoder is used to calculate the feature mean $\mu_b^c$ and covariance $\Sigma_b^c$ for each class, which are saved in a global statistics library $G$; (b) For each (domain, class) pair in $G$, $K$ pseudo-features are uniformly sampled: $\tilde{D} = \bigcup_{i=1}^{b}\bigcup_{c=1}^{|\mathcal{Y}|} \{(\tilde{x}, c) \sim \mathcal{N}(\mu_i^c, \Sigma_i^c)\}_{k=1}^K$, where $K$ remains consistent across all domain-class pairs to ensure balanced sampling; (c) The selector is trained using the synthetic data: $\mathcal{L}_{Select} = \frac{1}{|\hat{D}|}\sum_{(\tilde{x},y)\in\hat{D}} \ell_{CE}(\sum_{i=1}^{3b} s(\tilde{x})_i \cdot e_i(\tilde{x}), y)$.
    - Design Motivation: Hard assignment methods like S-iPrompt assign test samples to a single domain expert, failing to utilize cross-domain knowledge. The dynamic selector achieves adaptive cross-domain expert fusion through soft weighting, and equal-value sampling ensures that few-shot and many-shot classes receive equal training signals, preventing the selector from inheriting data imbalance bias.

3. **OAS (Oracle Approximating Shrinkage) Covariance Estimation**:

    - Function: Uses the OAS shrinkage mechanism to estimate more stable covariance matrices for few-shot classes.
    - Mechanism: When the number of class samples is $n \geq 10$, $\hat{\Sigma} = (1-\rho)\hat{\Sigma}_{emp} + \rho \cdot \frac{\text{tr}(\hat{\Sigma}_{emp})}{d} \cdot I_d$ is computed, where the shrinkage coefficient $\rho$ is automatically determined by the OAS formula. Simultaneously, class-specific covariances are averaged within each domain to reduce storage footprint.
    - Design Motivation: The sample size of few-shot classes is insufficient to reliably estimate high-dimensional covariance matrices; OAS improves the stability of the estimation by introducing a regularization prior.

### Loss & Training

- **First Stage**: Experts are optimized independently via $\ell_{exp} = \ell_{CE} + \ell_{Bal} + \ell_{Rev}$. Visual prompts are only optimized via VPT in the first domain and frozen thereafter.
- **Second Stage**: The selector is optimized via $\mathcal{L}_{Select}$ using synthetic features.
- Optimizer: SGD, batch size 128, learning rate 0.001, cosine decay.
- First stage trained for 20/30 epochs (depending on the dataset), second stage trained for 10 epochs.

## Key Experimental Results

### Main Results

Average over 5 random domain sequences using ViT-B/16-IN1K on four benchmarks:

| Dataset | Metric | DCE | S-iPrompt | CODA-Prompt | L2P | Gain (vs. runner-up) |
|--------|------|-----|-----------|-------------|-----|-------------|
| Office-Home | $\bar{\mathcal{A}}$ | **84.6** | 81.4 | 82.4 | 78.7 | +1.2 |
| Office-Home | $\mathcal{A}_B$ | **84.4** | 80.8 | 83.3 | 80.5 | +1.1 |
| Office-Home | $\mathcal{A}_{few}$ | **79.4** | 66.0 | 73.2 | 73.7 | +5.7 |
| DomainNet | $\bar{\mathcal{A}}$ | **64.3** | 59.0 | 47.6 | 48.5 | +5.3 |
| DomainNet | $\mathcal{A}_B$ | **63.5** | 57.9 | 45.1 | 45.2 | +5.6 |
| DomainNet | $\mathcal{A}_{few}$ | **50.8** | 31.5 | 38.2 | 37.3 | +12.6 |
| CORe50 | $\bar{\mathcal{A}}$ | **80.1** | 62.7 | 72.8 | 72.3 | +7.3 |
| CDDB-Hard | $\bar{\mathcal{A}}$ | **74.6** | 64.2 | 67.9 | 67.3 | +6.7 |

Improvements in few-shot classes are particularly significant: on DomainNet, the accuracy of few-shot classes increased from the runner-up of 38.2% to 50.8%, a gain of 12.6 percentage points.

### Ablation Study

| Configuration | $\bar{\mathcal{A}}$ (Office-Home) | Explanation |
|------|-----------------------------------|------|
| 1 expert ($\ell_{CE}$) | ~80 | Generates bias towards many-shot, poor on few-shot |
| 2 experts ($\ell_{CE} + \ell_{Bal}$) | ~82 | Moderate improvement |
| 3 experts (Full DCE) | **84.6** | Optimal |
| 4 experts (+ extra loss) | ~84.7 | Diminishing marginal gains |

### Key Findings

- **Three experts represent the optimal choice**: Performance consistently improves from 1 → 2 → 3 experts, while the gain from 4 experts is negligible. Three experts achieve the best trade-off between performance and efficiency.
- **Few-shot classes benefit the most**: Across all configurations, the contribution of the $\ell_{Rev}$ expert to few-shot classes is the most significant; removing it leads to a sharp decline in few-shot accuracy.
- **Class Performance Drift (CPD) Analysis**: DCE exhibits less forgetting on many-shot classes compared to shared prompt methods, and is better at improving few-shot classes than domain-specific prompt methods, with its overall CPD second only to SimpleCIL (which requires no training).
- **Computational Efficiency**: DCE only updates expert parameters after the first task (without backpropagation through the encoder), and inference only requires a single forward pass. Both its training and inference times are superior to L2P and DualPrompt.
- **Applying balanced loss to baselines remains inferior to DCE**: Replacing the baseline methods with balanced softmax loss actually degrades performance for some methods, while DCE maintains a clear advantage.

## Highlights & Insights

- **Loss functions as expert division of labor**: Different loss functions naturally possess preferences for different frequency classes. Exploiting this characteristic explicitly to build a multi-expert system is straightforward and elegant. This idea can be extended to any incremental learning scenario that needs to address class imbalance.
- **Pseudo-feature synthesis decouples data imbalance**: Synthesizing balanced data in the feature space via Gaussian sampling to train the selector cleverly bypasses the constraints of real data imbalance. A similar approach can be applied to any scenario requiring balanced training signals but constrained by real data distributions.
- **Valuable analysis framework**: The proposed Class Performance Drift (CPD) metric is more suitable for imbalanced DIL scenarios than traditional forgetting metrics, as few-shot class accuracy may improve with training rather than decline.
- **A third path between shared vs. domain-specific**: Rather than simply taking a compromise, DCE achieves the "best of both worlds" by separating the training stages (domain-specific training in the first stage + cross-domain fusion in the second stage).

## Limitations & Future Work

- **Limited to the vision domain**: Experiments were conducted entirely on image classification, and the effectiveness in NLP or multimodal scenarios has not been validated.
- **Gaussian assumption may not hold**: Modeling the PTM feature distribution as a unimodal Gaussian is based on empirical observation and may not be accurate for classes with complex distributions.
- **The number of experts is fixed at three**: Three experts correspond to three frequency groups (many/medium/few), but there is a lack of theoretical guidance—do different levels of imbalance require different numbers of experts?
- **Storage overhead scales with the number of domains**: Each domain requires storing three experts and class statistics, resulting in non-negligible storage costs in long-sequence domain-incremental scenarios.
- **Lack of comparison with newer methods like MoE-Adapter**: Although the MoE Adapter framework of Yu et al. 2024 is mentioned, a direct comparison is missing in the main experiments.

## Related Work & Insights

- **vs. L2P / DualPrompt / CODA-Prompt** (shared prompt methods): They achieve knowledge transfer by sharing a prompt pool, but suffer from severe forgetting of many-shot classes in imbalanced scenarios. DCE's domain-specific expert training avoids feature space entanglement.
- **vs. S-iPrompt** (domain-specific prompt methods): S-iPrompt isolates domain knowledge through KNN hard assignment, resulting in less forgetting but poor cross-domain transfer. DCE's dynamic selector achieves soft assignment, maintaining domain isolation while allowing knowledge sharing.
- **vs. RIDE / TADE** (multi-expert methods in long-tailed learning): They utilize multiple experts on static long-tailed data. DCE extends this idea to the dynamic scenario of incremental learning, introducing the challenge of balanced cross-domain knowledge fusion.

## Rating

- Novelty: ⭐⭐⭐⭐ Mentions the first systematic study of imbalanced DIL; the dual-balance design is novel, though individual components (multi-experts, Gaussian sampling) are not entirely brand new when viewed in isolation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, spanning 4 benchmarks, 5 domain sequences, multiple backbones, and detailed ablation and analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation analysis (the shared vs. domain-specific comparison in Figure 2 is highly intuitive) and rigorous problem definition.
- Value: ⭐⭐⭐⭐ Fills the gap of DIL + imbalance with a highly practical framework design, though its applicability remains limited to visual classification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] C4D: 4D Made from 3D through Dual Correspondences](../../ICCV2025/others/c4d_4d_made_from_3d_through_dual_correspondences.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](../../ICML2026/others/continual_learning_of_domain-invariant_representations.md)
- [\[ICML 2025\] Access Controls Will Solve the Dual-Use Dilemma](access_controls_will_solve_the_dual-use_dilemma.md)
- [\[ECCV 2024\] Foster Adaptivity and Balance in Learning with Noisy Labels](../../ECCV2024/others/foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)

</div>

<!-- RELATED:END -->
