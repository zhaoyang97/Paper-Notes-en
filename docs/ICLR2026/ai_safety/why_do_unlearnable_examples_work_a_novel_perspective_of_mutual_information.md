---
title: >-
  [Paper Note] Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information
description: >-
  [ICLR 2026][AI Safety][Unlearnable Examples] This paper provides a unified explanation for the effectiveness of all unlearnable example (UE) methods through the lens of mutual information (MI) reduction, and proves that minimizing the intra-class covariance of poisoned features reduces the MI upper bound. Based on this framework, MI-UE is proposed, which achieves covariance reduction via intra-class cosine similarity maximization, suppressing test accuracy to 9.95% on CIFAR-10 (near random-chance), while significantly outperforming existing methods under adversarial training defenses.
tags:
  - ICLR 2026
  - AI Safety
  - Unlearnable Examples
  - Mutual Information
  - Data Poisoning
  - Covariance Reduction
  - Privacy Protection
date: 2026-05-08
content_hash: ea61a828199f3bc7
---

# Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information

**Conference**: ICLR 2026
**arXiv**: [2603.03725](https://arxiv.org/abs/2603.03725)
**Code**: [github.com/hala64/mi-ue](https://github.com/hala64/mi-ue)
**Area**: AI Security / Data Privacy Protection
**Keywords**: Unlearnable Examples, Mutual Information, Data Poisoning, Covariance Reduction, Privacy Protection

## TL;DR

This paper provides a unified explanation for the effectiveness of all unlearnable example (UE) methods through the lens of mutual information (MI) reduction, and proves that minimizing the intra-class covariance of poisoned features reduces the MI upper bound. Based on this framework, MI-UE is proposed, which achieves covariance reduction via intra-class cosine similarity maximization, suppressing test accuracy to 9.95% on CIFAR-10 (near random-chance), while significantly outperforming existing methods under adversarial training defenses.

## Background & Motivation

**State of the Field**: Internet data is being scraped at scale to train commercial models (e.g., GPT-4, LAION-5B), giving rise to numerous privacy infringement lawsuits. Unlearnable Examples (UE) have attracted attention as a form of proactive data defense—users inject imperceptible perturbations satisfying $\|\delta\|_p \leq \epsilon$ before publishing data, causing the generalization ability of unauthorized models trained on such data to collapse dramatically. The representative method Error-Minimization (EM) reduces CIFAR-10 test accuracy from 94.45% to 24.17%, yet remains far from the random-guess level of 10%.

**Limitations of Prior Work**: Existing UE methods are almost entirely designed through empirical intuition (e.g., "deceiving the model into thinking there is nothing to learn," "injecting non-robust features," "creating autoregressive signals"), lacking a unified theoretical explanation. The mainstream "linear separability" hypothesis has two fundamental flaws: (1) linear classifiers still achieve 30%+ accuracy on UE data, whereas deep networks drop to ~10%—if UE only introduces linear shortcuts, why are linear models unaffected? (2) Autoregressive (AR) perturbations exhibit even lower linear separability than clean data, yet still produce strong unlearnable effects. This indicates that linear separability is a surface phenomenon rather than the root cause.

**Root Cause**: Perturbations injected by UE cause training data to deviate from the original distribution, breaking the i.i.d. assumption. However, the extent and manner of deviation that leads to generalization collapse remain unclear, as there is no theoretical tool to relate the degree of distributional shift to the resulting generalization loss.

**Paper Goals**: (1) Provide a unified information-theoretic explanatory framework for all UE methods; (2) Design a principled, stronger UE method grounded in this framework; (3) Explain the "depth effect" phenomenon—why deeper networks yield stronger UE effectiveness.

**Starting Point**: Drawing on the idea of using mutual information (MI) to measure variable correlations between two distributions in representation learning, the authors propose using the mutual information $I(g(X), g(X'))$ between clean features $g(X)$ and poisoned features $g(X')$ in feature space as a proxy for "learnability"—the lower the MI, the harder it is for the model to recover useful representations of the original distribution from poisoned data.

**Core Idea**: Effective UE methods necessarily reduce MI between clean and poisoned features; directly minimizing the intra-class covariance of poisoned features (equivalent to maximizing intra-class cosine similarity) optimizes the MI upper bound and produces the strongest unlearnable effect.

## Method

### Overall Architecture

The work proceeds in three progressive stages. The first stage is **empirical discovery**: systematically measuring MI reduction across all mainstream UE methods using multiple MI estimators, verifying a strong positive correlation between "MI reduction ↔ generalization degradation" (Spearman correlation 0.7818). The second stage is **theoretical derivation**: under a Gaussian mixture assumption, proving that the MI upper bound contains an intra-class covariance term $\log\det\Sigma_Y$, thereby converting the intractable MI minimization problem into a tractable covariance reduction problem. The third stage is **method design**: proposing MI-UE, which employs bi-level optimization to jointly train a shadow model and optimize perturbation $\delta$, with the outer objective being a $\mathcal{L}_{mi}$ loss that maximizes intra-class cosine similarity while minimizing inter-class cosine similarity.

### Key Designs

1. **MI Interpretive Framework (Empirical Validation)**:

    - Function: Provides a unified effectiveness metric for all UE methods.
    - Mechanism: The classification model is decomposed as $f = h \circ g$ (where $g$ is the feature extractor and $h$ is the linear classification head). For each UE method, a ResNet-18 is trained on the generated poisoned data, and $I(g(X), g(X'))$ is computed in feature space. To avoid bias from any single estimator, four MI estimation methods are used simultaneously: Histogram, KDE, k-NN, and MINE, combined with Sliced MI (SMI) that projects high-dimensional features to one dimension before estimation. Experiments cover 8 UE methods: EM, AP, NTGA, AR, REM, SEM, GUE, and TUE. All effective UEs exhibit significantly lower MI than the Clean and Random baselines, and larger MI Gap correlates with larger Acc Gap. Furthermore, across models of increasing depth (Linear → 2-NN → 3-NN → LeNet-5 → VGG-11 → ResNet-18), MI reduction and accuracy drop are strictly positively correlated—the feature extractor of deeper models amplifies the perturbation's effect on MI (error amplification effect), while in linear models $g$ degenerates to the identity mapping, making it nearly impossible for small-norm perturbations to alter $I(X, X')$.
    - Design Motivation: An explanation that holds only under a single estimation method lacks persuasive power; consistent trends across four independent estimators provide strong empirical support.

2. **Covariance Reduction Theorem (Theorem 5.1)**:

    - Function: Converts the intractable MI reduction problem into a tractable covariance minimization problem.
    - Mechanism: Assuming that for each class $Y$, the poisoned features $g(X')|Y$ approximately follow a Gaussian distribution $\mathcal{N}(\mu_Y, \Sigma_Y)$ (KL divergence $\leq \epsilon$), the MI upper bound is $I(g(X), g(X')) \leq \frac{d}{2}\log(2\pi e) + \frac{1}{2}\mathbb{E}_Y\log(\det\Sigma_Y) + H(g(X')|g(X)) + \mathbb{E}_Y C_Y\sqrt{\epsilon}$. The first term is a constant; the third term is also constant once the UE generator $\mathcal{G}$ and training algorithm $\mathcal{A}$ are fixed; the fourth term is a small approximation error. Therefore, the key variable in the MI upper bound is the intra-class covariance determinant $\det\Sigma_Y$—reducing it lowers the MI upper bound.
    - Design Motivation: MI estimation in high-dimensional spaces is extremely difficult (existing methods suffer from severe statistical bias, and SGD-based MI optimization is biased); directly optimizing MI is infeasible. The covariance reduction theorem reduces the optimization target from "MI" to "intra-class feature dispersion," making the problem tractable.

3. **MI-UE Loss Function**:

    - Function: Concrete optimization objective for achieving covariance reduction.
    - Mechanism: $\mathcal{L}_{mi}$ consists of two terms. **Similarity term**: for sample pairs in a mini-batch, the numerator computes the sum of $\exp(\cos/\tau)$ over same-class pairs, and the denominator computes the sum of $\exp(\cos/\tau)$ over cross-class pairs; the log ratio is taken—this maximizes intra-class cosine similarity while minimizing inter-class similarity, naturally compressing intra-class covariance and preventing class collapse. **Distance term**: $\zeta \cdot \log(1 + \sum_k \|g(x_{b_j}+\delta_{b_j}) - g(x_{b_k}+\delta_{b_k})\|_2)$ serves as a regularizer to enhance robustness. The $\log(1+\cdot)$ form stabilizes gradients. Cosine distance, which is invariant to batch normalization, is used rather than pure Euclidean distance, avoiding the failure mode where normalization layers neutralize Euclidean optimization.
    - Design Motivation: Pure distance minimization fails in networks with BN/LN (features are rescaled after normalization); the inter-class repulsion term prevents all class features from collapsing to a single point (which would paradoxically restore generalization).

### Loss & Training

Bi-level min-min optimization is adopted: the inner loop trains shadow model parameters $\theta$ with cross-entropy loss $\mathcal{L}_{ce}$, while the outer loop optimizes perturbation $\delta$ with $\mathcal{L}_{mi}$. Perturbations are generated via PGD iterations—10 PGD steps per epoch, step size 0.2/255 (CIFAR) or 0.4/255 (ImageNet-subset), with total budget $\|\delta\|_\infty \leq 8/255$. Generation on CIFAR for 100 epochs takes approximately 3.6 hours, roughly 1.5× that of EM. Temperature $\tau$ controls softmax sharpness; the balancing hyperparameter $\zeta = 0.1$ (ablations show stable performance for $\zeta \leq 0.1$ and severe degradation for $\zeta \geq 10$, confirming that the cosine similarity term is the core driving force).

## Key Experimental Results

### Main Results: ResNet-18 Comparison on Three Datasets

| Method | CIFAR-10 (%) | CIFAR-100 (%) | ImageNet-subset (%) |
|--------|-------------|--------------|-------------------|
| Clean | 94.45 | 76.65 | 80.43 |
| EM | 24.17 | 2.09 | 1.26 |
| AP | 11.21 | 3.73 | 9.10 |
| NTGA | 23.11 | 3.08 | 8.42 |
| REM | 22.94 | 7.52 | 13.74 |
| SEM | 14.78 | 6.29 | 4.10 |
| TUE | 11.25 | 1.34 | 4.95 |
| **MI-UE** | **9.95** | **1.17** | **1.03** |

MI-UE achieves the lowest test accuracy across all three datasets. In particular, CIFAR-10 reaches 9.95%, nearly equal to the random-guess level for 10 classes.

### MI vs. Accuracy Relationship (CIFAR-10, Histogram Estimator)

| Method | Test Acc (%) | Acc Gap (%) | MI | MI Gap |
|--------|-------------|------------|------|--------|
| Clean | 94.45 | - | 0.7122 | - |
| Random | 94.11 | 0.34 | 0.6747 | 0.0375 |
| EM | 24.17 | 70.28 | 0.6400 | 0.0722 |
| AP | 11.21 | 83.24 | 0.5871 | 0.1251 |
| NTGA | 23.11 | 71.34 | 0.6126 | 0.0996 |
| AR | 17.41 | 77.04 | 0.5622 | 0.1500 |
| SEM | 14.78 | 79.67 | 0.5747 | 0.1375 |
| GUE | 12.04 | 82.41 | 0.5895 | 0.1227 |
| TUE | 11.25 | 83.20 | 0.6094 | 0.1028 |
| **MI-UE** | **9.95** | **84.50** | **0.4969** | **0.2153** |

The MI Gap of MI-UE (0.2153) substantially exceeds all baselines, directly corroborating the central thesis that "lower MI → stronger unlearnability."

### Cross-Architecture Transferability (CIFAR-10)

| Model | Clean | EM | AP | AR | SEM | GUE | TUE | MI-UE |
|-------|-------|-----|-----|-----|------|------|------|-------|
| ResNet-18 | 94.45 | 24.17 | 11.21 | 17.41 | 14.78 | 12.04 | 11.25 | **9.95** |
| ResNet-50 | 95.16 | 23.57 | 11.66 | 15.28 | 13.61 | 12.99 | 10.01 | **9.98** |
| DenseNet-121 | 94.91 | 24.87 | 11.80 | 16.50 | 15.19 | 12.46 | 11.41 | **9.93** |
| ViT-B | 90.92 | 27.35 | 24.21 | 24.16 | 25.52 | 17.72 | 35.54 | **15.51** |
| LeNet-5 | 80.68 | 26.30 | 31.38 | 73.33 | 22.94 | 13.30 | 28.37 | **10.80** |
| 3-NN | 62.12 | 28.54 | 61.03 | 62.02 | 54.44 | 16.97 | 56.55 | **14.16** |
| 2-NN | 56.15 | 32.50 | 55.78 | 56.75 | 50.79 | 22.08 | 48.75 | **17.82** |

MI-UE is optimal across all architectures. Notably, methods such as AP, AR, and TUE nearly fail on shallow networks (AP achieves 61.03% on 3-NN), whereas MI-UE still reaches 14.16% on 3-NN, demonstrating robust transferability across network depths.

### Adversarial Training Defense (CIFAR-10)

| Method | AT-8 | AT-6 | AT-4 | AT-2 | ST |
|--------|------|------|------|------|----|
| Clean | 85.10 | 87.54 | 89.77 | 91.95 | 94.45 |
| EM | 84.57 | 85.42 | 84.29 | 52.81 | 24.17 |
| SEM | 85.99 | 86.82 | 29.77 | 19.41 | 14.78 |
| REM | 85.99 | 81.91 | 39.45 | 30.64 | 22.94 |
| TUE | 84.10 | 86.07 | 89.29 | 91.70 | 11.25 |
| **MI-UE** | **70.56** | **45.55** | **31.79** | **17.39** | **9.95** |

Under the most challenging AT-8 setting (adversarial training with the same budget as the UE perturbation), MI-UE still suppresses accuracy to 70.56%, while other methods nearly fully recover to the Clean level. Under AT-6, MI-UE achieves 45.55%, far below the next-best REM (81.91%), a gap of 36 percentage points.

### Ablation Study

| Configuration | CIFAR-10 (%) | CIFAR-100 (%) | ImageNet-S (%) |
|---------------|-------------|--------------|---------------|
| MI-UE (full) | 9.95 | 1.17 | 1.03 |
| w/o distance term | 10.09 | 2.52 | 1.46 |
| w/o similarity term | 51.65 | 26.72 | 23.38 |

Removing the similarity term causes accuracy to surge from 9.95% to 51.65%, confirming that intra-class covariance reduction driven by cosine similarity is the core mechanism of MI-UE. The marginal contribution of the distance term is smaller but still positive (CIFAR-100: 2.52% → 1.17%).

### Key Findings

- **Strong positive correlation between MI and UE effectiveness**: All four independent MI estimators consistently show that larger MI Gap corresponds to larger Acc Gap, with Spearman correlation reaching 0.7818.
- **Information-theoretic explanation of the depth effect**: The feature extractor $g$ of shallow networks approximates the identity mapping, making small-norm perturbations nearly unable to alter $I(X, X')$; deep networks amplify the perturbation's effect on MI in feature space via error amplification, leading to substantial MI reduction.
- **Similarity term >> distance term**: Removing the similarity term causes effectiveness to collapse, whereas removing the distance term has negligible impact, confirming that covariance reduction (rather than simply bringing feature points closer together) is the key mechanism. This also indirectly verifies that BN/LN absorbs gradients from pure Euclidean distance optimization.
- **MI regularization experiment**: Adding a MINE network as MI regularization on top of EM and AP further reduces MI and accuracy (EM: 24.17% → 15.62%; AP: 11.21% → 10.01%), but still falls short of MI-UE (9.95%), indicating that directly optimizing the MI lower bound is less effective than indirectly optimizing through covariance reduction.
- **Hyperparameter robustness**: Performance is stable (~10%) for $\zeta \in [0, 0.1]$ and degrades sharply to 45%+ for $\zeta \geq 10$, demonstrating that excessive distance weighting interferes with the core similarity optimization.
- **Budget robustness**: As the perturbation budget varies from 4/255 to 16/255, CIFAR-10 accuracy remains consistently around 10% and CIFAR-100 around 1%, indicating that MI-UE is insensitive to the perturbation budget.

## Highlights & Insights

- **Unifying power of the MI perspective**: All known effective UE methods (EM, AP, AR, NTGA, REM, SEM, GUE, TUE) are fundamentally reducing MI, differing only in path and degree. This is the first unified explanatory framework capable of covering all UE methods, including the AR method that was previously inexplicable under the "linear separability" hypothesis.
- **Elegance of the theory-to-method bridge**: The chain from MI → covariance determinant → cosine similarity progressively translates an abstract information-theoretic objective into a concrete, optimizable loss term. The adoption of cosine distance rather than Euclidean distance to circumvent the interference of normalization layers is a design principle with broad applicability in modern networks where BN is ubiquitous.
- **Cross-depth robust transferability**: Methods such as AP and TUE nearly fail on shallow networks, whereas MI-UE still reduces accuracy by 38 percentage points compared to Clean on 2-NN, indicating that the MI reduction mechanism does not depend on specific network architectures.

## Limitations & Future Work

- **Bottleneck under specialized defenses**: Defenses specifically designed for UE, such as ISS and AVA, can restore all UE methods (including MI-UE) to 80%+ accuracy. Under the worst case, MI-UE reaches 86.18% (AVA)—better than other methods, but still far from being unlearnable. Overcoming specialized defenses remains a fundamental challenge for the UE field.
- **Evaluation limited to image classification**: Applicability to other modalities such as text, speech, and time-series data has not been explored. The MI reduction framework is theoretically modality-agnostic, but concrete implementations of covariance reduction need to be redesigned for different data types.
- **Limitations of the Gaussian mixture assumption**: Theorem 5.1 assumes that intra-class features are approximately Gaussian, but deep network feature distributions may be multimodal or heavy-tailed. Although the authors discuss the reasonableness of this approximation in the appendix, rigorous analysis of distributional deviations is still lacking.
- **Computational overhead**: MI-UE generation time is approximately 1.5× that of EM (due to bi-level optimization + PGD), and scalability to large-scale datasets remains to be verified.
- **Connection to machine unlearning**: MI-UE is a "pre-training" data protection mechanism; whether it can be combined with "post-training" machine unlearning methods is worth exploring.

## Related Work & Insights

- **vs. EM (Huang et al., 2020)**: EM minimizes training loss to make the model "believe there is nothing to learn," achieving an MI Gap of only 0.0722. MI-UE directly optimizes MI reduction in feature space, achieving a Gap of 0.2153 and improving accuracy from 24.17% → 9.95%.
- **vs. AP (Fowl et al., 2021)**: AP injects adversarial poisons with an MI Gap of 0.1251. AP performs well on deep networks (11.21%) but nearly fails on shallow networks (3-NN: 61.03%), indicating dependence on specific behaviors of deep networks; MI-UE still achieves 14.16% on 3-NN.
- **vs. SEM/REM (robust UEs)**: SEM and REM are specifically designed for adversarial training defenses and perform well under AT-4, but are only effective when the adversarial training budget is less than half the poison budget. MI-UE outperforms across the full range of AT settings (AT-2 to AT-8), demonstrating stronger defense robustness.
- **vs. TUE (Ren et al., 2022)**: TUE generates transferable perturbations based on a SimCLR unsupervised backbone, achieving excellent results on deep networks but completely failing on shallow ones. MI-UE exhibits more balanced transferability.
- **Relation to data privacy protection**: The MI framework in this paper provides information-theoretic theoretical guidance for data protection—future UE designs should target maximizing MI reduction as a core objective, rather than relying on specific empirical intuitions.

## Rating

- Novelty: ⭐⭐⭐⭐ The MI interpretive framework and covariance reduction theorem are substantive contributions, elevating empirical UE design to the level of information theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four MI estimators × multiple network depths × 8 baselines × 3 datasets × multiple defenses—cross-validation is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The progression from empirical observation → theory → method is logically clear, with rich figures and tables.
- Value: ⭐⭐⭐⭐ Provides a unified theoretical framework and state-of-the-art method for the UE field, though the bottleneck under specialized defenses limits direct practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective](adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)
- [\[AAAI 2026\] An Information Theoretic Evaluation Metric for Strong Unlearning](../../AAAI2026/ai_safety/an_information_theoretic_evaluation_metric_for_strong_unlearning.md)
- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](../../CVPR2026/ai_safety/a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)
- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](../../AAAI2026/ai_safety/infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)
- [\[NeurIPS 2025\] Preserving Task-Relevant Information Under Linear Concept Removal](../../NeurIPS2025/ai_safety/preserving_task-relevant_information_under_linear_concept_removal.md)

<!-- RELATED:END -->
