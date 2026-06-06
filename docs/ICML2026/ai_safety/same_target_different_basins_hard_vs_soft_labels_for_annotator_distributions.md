---
title: >-
  [Paper Note] Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions
description: >-
  [ICML 2026 (EIML Workshop)][AI Safety][Annotator Distribution] By feeding "annotator distributions" to models via hard labels on CIFAR-10H (multipass cycling per vote / SLS resampling per epoch)…
tags:
  - "ICML 2026 (EIML Workshop)"
  - "AI Safety"
  - "Annotator Distribution"
  - "Hard vs. Soft Labels"
  - "Loss Landscape"
  - "OOD Detection"
  - "Calibration"
date: 2026-05-08
content_hash: 5ccf6f2fe982eea0
---

# Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions

**Conference**: ICML 2026 (EIML Workshop)  
**arXiv**: [2605.20642](https://arxiv.org/abs/2605.20642)  
**Code**: None  
**Area**: AI Safety / Uncertainty Estimation / Training Methodology  
**Keywords**: Annotator Distribution, Hard vs. Soft Labels, Loss Landscape, OOD Detection, Calibration

## TL;DR
By feeding "annotator distributions" to models via hard labels on CIFAR-10H (multipass cycling per vote / SLS resampling per epoch), this work proves equivalence to the soft-label cross-entropy expected target, but demonstrates convergence to flatter basins, superior performance under sparse annotations, and slightly better OOD detection.

## Background & Motivation
**Background**: The mainstream approach for multi-annotator datasets (CIFAR-10H, ChaosNLI, LeWiDi) is either to collapse votes into a single hard label or to perform cross-entropy directly on the empirical soft-label distribution of each instance. Peterson et al. (2019) demonstrated that soft-label training preserves "ambiguity/subjectivity" information, significantly outperforming hard voting.

**Limitations of Prior Work**: Training with soft labels by "keeping the distribution in the loss" is an assumption—it defaults to the empirical distribution $p_i$ as the ideal training target. However: (a) when each instance has only 5–10 annotations, $p_i$ is a sparse estimation distant from the true distribution; (b) it couples "what the target is" with "how the target is delivered," preventing the study of the optimization path's impact.

**Key Challenge**: To retain annotator information, must the distribution enter the loss at every step? If changed to "sampling hard labels according to the distribution," can different optimization paths be taken under the same expected target, leading to solutions with different geometric properties?

**Goal**: While fixing the annotator distribution target for each instance, vary only the "delivery format" (hard vs. soft) to systematically compare endpoint performance, basin geometry, and OOD behavior under both sparse and sufficient annotation budgets.

**Key Insight**: The authors formally compare SLS (sampling one hard label from $p_i$ per epoch) with soft cross-entropy, proving they share the **same expected gradient**. The difference lies only in sampling variance, which is highly correlated with annotator disagreement—meaning hard label delivery is equivalent to "injecting structured noise into samples with high disagreement."

**Core Idea**: Treat the delivery format as an independent variable while fixing the annotator distribution as the target. Use multipass (cycling through votes) and SLS (resampling by distribution) to replace soft-label CE. Theoretically, the expected targets are equivalent; experimentally, this leads to flatter basins and better soft NLL under sparse annotations.

## Method

### Overall Architecture
Let the dataset be $\mathcal{D}=\{(x_i, p_i)\}_{i=1}^N$, where $p_i \in \Delta^{K-1}$ is the empirical annotator distribution for the $i$-th sample and $q_\theta(x) = \mathrm{softmax}(z_\theta(x))$ is the model prediction. The control group is soft-label CE $\mathcal{L}_{\mathrm{soft}} = \sum_i H(p_i, q_\theta(x_i))$. This work designs two hard-label delivery methods (multipass, SLS) plus two controls (deterministic control, shuffled SLS). Each epoch uses standard hard-label CE, but the hard label for each instance varies by epoch or sampling. All methods are run on CIFAR-adapted ResNet-18 for 200 epochs with cosine annealing.

### Key Designs

1.  **Multipass (Count-preserving hard labels via vote cycling)**:
    *   **Function**: When original vote counts $\{c_{ik}\}_k$ are available, expand each instance's votes into a "label multiset," shuffle once with a fixed seed, and feed them across epochs via $\texttt{epoch} \bmod m_i$ (where $m_i$ is the total votes for that instance). Each epoch still contains $N$ samples; the dataset cardinality remains unchanged.
    *   **Mechanism**: Unlike Sheng et al. (2008), which expands repeated annotations into multiple samples, multipass maintains dataset size but changes "label identity" within epochs. If an instance has 50 votes, it completes one cycle every 50 epochs, with every observed vote appearing exactly once.
    *   **Design Motivation**: Carry distribution information using a deterministic, reproducible sequence of hard labels—avoiding SLS sampling variance while verifying if "label changes between epochs" are sufficient. It is the default choice when raw counts exist.

2.  **Stochastic Label Sampling (SLS) + Expected Target Equivalence**:
    *   **Function**: At the start of each epoch, independently sample $y_i^{(t)} \sim \mathrm{Categorical}(p_i)$ for each instance, then train that epoch with standard hard-label CE. It only requires $p_i$, not raw counts, serving as a lightweight alternative to multipass.
    *   **Mechanism**: Proposition 1 proves $\mathbb{E}_{y\sim p}[-\log q_y] = H(p,q)$, $\mathbb{E}_{y\sim p}[\nabla_z\ell] = q-p$, and $\mathrm{Cov}_{y\sim p}[\nabla_z\ell] = \mathrm{Diag}(p)-pp^\top$. Thus, SLS and soft-label CE are equivalent in expected target; the difference is the additional gradient variance $\mathbb{E}\|q-e_y\|^2 - \|q-p\|^2 = 1 - \|p\|_2^2$, which increases with disagreement.
    *   **Design Motivation**: Strictly decouple "target" from "optimization path." Since targets are identical, endpoint performance differences must stem from the optimization path and noise covariance structure ($\mathrm{Diag}(p)-pp^\top$), attributable to basin geometry.

3.  **Deterministic / Shuffled Double Controls**:
    *   **Function**: Deterministic control is an ablation of multipass traversal order using a different fixed shuffle seed; Shuffled SLS permutes each instance's $p_i$ across the dataset before SLS, breaking "sample-distribution" pairing while preserving global disagreement statistics.
    *   **Mechanism**: Each control answers a confounder—the former verifies that "cycle order" is irrelevant, while the latter confirms that "pairing each instance with its own distribution" is critical. If shuffled SLS still worked, SLS benefits would be merely "noise regularization."
    *   **Design Motivation**: Separate the interpretations of "distribution as label structural information" vs. "distribution as a generic noise source." Shuffled SLS degraded to 12% accuracy, confirming pairing as a first-order factor.

### Loss & Training
*   Soft label: $\mathcal{L}_{\mathrm{soft}} = -\sum_i \sum_k p_{ik} \log q_{\theta,k}(x_i)$; hard methods use standard hard-label CE.
*   Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=$5\times 10^{-4}$, cosine annealing for 200 epochs, batch=128, random crop + horizontal flip.
*   Evaluation follows the proper-scoring-rule perspective: Soft NLL as primary; KL-to-annotator and soft Brier as secondary; hard_acc / equal-mass ECE / Spearman entropy correlation as tertiary. Hessian $\lambda_{\max}$ (power iteration) and trace (Hutchinson) are calculated at the best-soft-NLL checkpoint in eval mode with BN frozen.

## Key Experimental Results

### Main Results
CIFAR-10H (10,000 images), 80/20 stratified split, 10 seeds for the main control / 5 paired seeds for the hard-delivery family and sparse scanning.

| Method | Soft NLL ↓ | Hard Acc ↑ | ECE_eqmass ↓ | EntCorr ↑ |
| :--- | :--- | :--- | :--- | :--- |
| Majority vote | 0.7284 | 0.8570 | 0.0704 | 0.2902 |
| Label smoothing | 0.6263 | 0.8590 | 0.0598 | 0.2117 |
| Mixup | 0.5526 | **0.8824** | 0.0977 | 0.2499 |
| Soft labels | 0.5096 | 0.8687 | 0.0185 | 0.3909 |
| **SLS** | **0.5052** | 0.8695 | 0.0186 | **0.3946** |

With the full distribution, SLS and soft labels show no significant difference across 4 metrics ($p \in [0.38, 0.92]$); Mixup has the highest accuracy but ECE is nearly 5 times worse.

Sparse annotation scan (Soft NLL for $K \in \{5, 10, 25, 50\}$, mean of 5 paired seeds):

| Method | K=5 | K=10 | K=25 | K=50 |
| :--- | :--- | :--- | :--- | :--- |
| Soft labels | 0.5860 | 0.5785 | 0.5388 | 0.5628 |
| SLS | 0.5599 (p=.031) | 0.5485 (.031) | 0.5169 (.063) | 0.5291 (.094) |
| Multipass | 0.5649 (.063) | 0.5371 (.031) | 0.5117 (.031) | 0.5241 (.031) |
| Det. control | **0.5555** (.031) | **0.5388** (.031) | **0.5077** (.031) | **0.5231** (.031) |

All 12/12 cells numerically outperformed soft labels; in 9/12, all 5 seeds were consistent. Gains are larger as $K$ decreases. The improvement correlates with the JS distance between empirical $p_i$ and the true distribution (Spearman 0.05–0.16).

### Ablation Study

| Configuration | Soft NLL | Hard Acc | EntCorr | $\lambda_{\max}$ (full) | Trace (full) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Soft labels | 0.5096 | 0.8687 | 0.3909 | 242.2 | 4946.0 |
| SLS | 0.5052 | 0.8695 | 0.3946 | **104.9** | **1633.9** |
| Multipass | 0.4942 | 0.8714 | 0.4000 | 103.8 | 1571.4 |
| Det. control | 0.4921 | 0.8724 | 0.3963 | 101.6 | 1581.6 |
| Shuffled SLS | 2.2973 | 0.1199 | -0.006 | 18.4 | 33.9 |

Multipass / SLS / Det. control are geometrically nearly identical ($\lambda_{\max} \sim 100$, 2.4× smaller than soft labels). Shuffled SLS is functionally the "flattest" but behaves like a random classifier—indicating that "flatness" must be interpreted while maintaining pairing. Stale-target probe: fixing sampled labels for 1/5/10/50 epochs caused Soft NLL to deteriorate monotonically from 0.5027 to 0.6689.

### Key Findings
*   **Pairing is First-order**: Shuffled SLS has the flattest geometry but 12% accuracy, proving that hard delivery's benefit is not "generic noise regularization" but "per-instance distribution matching" with noise covariance reshaped by disagreement.
*   **Structured Gradient Noise**: The Spearman correlation between last-layer gradient variance and annotator entropy averaged 0.939 across seeds—extra variance injected by SLS is precisely concentrated on high-disagreement samples, consistent with Proposition 1's $\mathrm{Diag}(p)-pp^\top$.
*   **Same Target, Different Basins**: The mean loss barrier between SLS and soft checkpoints is 2.05 (much greater than 0); CKA 0.920 vs. 0.887; Grad-CAM cross-seed stability 0.901 vs. 0.804. While endpoint losses are nearly identical, they occupy different basins; hard delivery representations are more reproducible.
*   **OOD Endpoint Gains**: Hard delivery outperformed soft labels in 5 out of 6 detectors on SVHN. On CIFAR-100, SLS outperformed soft labels in AUROC across all scores (Energy/ODIN paired $p=0.0186$).

## Highlights & Insights
*   **Delivery as an Independent Variable**: Previous works on "training with annotator distributions" typically change both target and delivery. This paper mathematically decouples them via Proposition 1 and further decouples "sampling randomness" via deterministic control—a three-layer ablation design applicable to any stochastic vs. deterministic optimization comparison.
*   **Multipass is an Overlooked Practical Baseline**: When raw vote counts are available, cycling through the multiset is deterministic, reproducible, avoids SLS variance, and doesn't change dataset size unlike Sheng 2008. It is the "cheapest hard-label delivery."
*   **New Evidence for "Flat Basin via Structured Label Noise"**: Traditional flat-minima literature uses SGD noise or symmetric label noise (Keskar, Smith, Damian). This work proves "hard labels sampled from the true distribution" can drive the model into basins ~2.4× flatter, with a variance structure strictly corresponding to $\mathrm{Diag}(p)-pp^\top$, providing a concretely interpretable example of "covariance-Hessian alignment" (HaoChen/Wu).

## Limitations & Future Work
*   The authors acknowledge limited scope: single dataset (CIFAR-10H) and single architecture (ResNet-18 with CIFAR stem). The sparse scan's 12 cells were numerically superior but lacked single-cell significance after Holm correction (smallest raw $p=0.03125$). OOD comparisons were treated as descriptive evidence.
*   The sparse $K$ was simulated by resampling dense CIFAR-10H; real sparse datasets might involve "annotator selection bias" or task-specific disagreement structures where the relative benefit of hard delivery remains unverified. The causal link from basin geometry to endpoint metrics remains observational.
*   Future directions: Extend multipass to high-disagreement NLP data like ChaosNLI/SBIC; compare with explicit flatness optimizers (SAM) to see if "structured label noise" can replace explicit sharpness regularization; study the interaction between cycle length $m_i$ and LR schedules.

## Related Work & Insights
*   **vs. Peterson et al. (2019)**: They introduced CIFAR-10H and proved soft-label CE > majority vote; this work takes a finer causal slice by fixing the target and changing only delivery.
*   **vs. Sheng et al. (2008) Repeat Label Expansion**: Sheng expands datasets from $N \to \sum m_i$; multipass keeps $N$ constant and cycles through votes—a cardinality-preserving version of the same idea.
*   **vs. DisturbLabel (Xie et al., 2016)**: DisturbLabel uses uniform noise to perturb labels; SLS samples from the true $p_i$, aligning noise structure with the target distribution—making it "information-preserving noise" rather than destructive.
*   **vs. Label Smoothing / Mixup**: LS uses uniform soft targets; Mixup interpolates inputs and targets. Neither preserves per-instance epistemic structure. Table 2 shows their ECE is significantly worse than SLS/soft, confirming the necessity of preserving example-to-distribution pairing.
*   **vs. Flat-minima Literature (Keskar 2017, Smith 2021, HaoChen 2021)**: This work provides a new mechanism for how hard-label delivery flattens basins, with an exact analytical form for the noise covariance.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ Treating delivery as an independent variable and decoupling it via expected equivalence is a rare causal slice in annotator-disagreement research; however, the sampling idea was nascent in Peterson 2019.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes three main methods + two controls + sparse scanning + Hessian + loss barriers + CKA + Grad-CAM + OOD; though limited by the single dataset/architecture.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear organization; the "regime split" narrative is consistent; Proposition and control roles are well-explained. Tables require careful reading.
*   **Value**: ⭐⭐⭐⭐ Provides multipass as a practical default for annotator-distribution tasks and builds a verifiable bridge between flat-minima and structured label noise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](../../AAAI2026/ai_safety/easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[ICML 2026\] Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack](exposing_vulnerabilities_in_explanation_for_time_series_classifiers_via_dual-tar.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks](../../AAAI2026/ai_safety/improving_the_convergence_rate_of_ray_search_optimization_for_query-efficient_ha.md)

</div>

<!-- RELATED:END -->
