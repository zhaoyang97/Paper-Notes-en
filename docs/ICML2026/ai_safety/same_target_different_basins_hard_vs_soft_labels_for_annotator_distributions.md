---
title: >-
  [Paper Note] Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions
description: >-
  [ICML 2026 (EIML Workshop)][AI Safety][Annotator distributions] By feeding "annotator distributions" to models via hard labels on CIFAR-10H (multipass cycling by votes / SLS resampling per epoch), this work proves these methods are equivalent to the soft label cross-entropy expectation goal but converge to flatter basins, perform better under sparse annotations, and slightly excel in OOD detection.
tags:
  - "ICML 2026 (EIML Workshop)"
  - "AI Safety"
  - "Annotator distributions"
  - "Hard and soft labels"
  - "Loss landscape"
  - "OOD detection"
  - "Calibration"
date: 2026-05-08
content_hash: e0279536d88e8d0a
---

# Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions

**Conference**: ICML 2026 (EIML Workshop)  
**arXiv**: [2605.20642](https://arxiv.org/abs/2605.20642)  
**Code**: None  
**Area**: AI Safety / Uncertainty Estimation / Training Methodology  
**Keywords**: Annotator distributions, Hard and soft labels, Loss landscape, OOD detection, Calibration

## TL;DR
By feeding "annotator distributions" to models via hard labels on CIFAR-10H (multipass cycling by votes / SLS resampling per epoch), this work proves these methods are equivalent to the soft label cross-entropy expectation goal but converge to flatter basins, perform better under sparse annotations, and slightly excel in OOD detection.

## Background & Motivation
**Background**: The mainstream approach for multi-annotator datasets (CIFAR-10H, ChaosNLI, LeWiDi) is either to collapse votes into a single majority label or to directly perform cross-entropy on the empirical soft label distribution of each instance. Peterson et al. (2019) demonstrated that soft label training preserves "ambiguity/subjectivity" information, significantly outperforming hard voting.

**Limitations of Prior Work**: Treating soft label training as "leaving the distribution in the loss" is an assumption—it defaults to the empirical distribution $p_i$ being a good training target. However, (a) when each instance has only 5–10 annotations, $p_i$ is a sparse estimate distant from the true distribution; (b) it couples "what the target is" with "how the target is delivered," preventing decoupled research into the impact of optimization paths.

**Key Challenge**: To retain annotator information, must the distribution enter the loss at every step? If changed to "hard label sampling based on the distribution," can different optimization paths lead to solutions with different geometric properties under the same expectation goal?

**Goal**: Given a fixed annotator distribution target for each instance, this work systematically compares two delivery formats (hard vs. soft) across sparse and sufficient annotation budgets, examining endpoint performance, basin geometry, and OOD behavior.

**Key Insight**: The authors formalize the comparison between SLS (sampling one hard label from $p_i$ per epoch) and soft cross-entropy, proving that they share the **same expected gradient**. The difference lies only in the sampling variance, which is highly correlated with annotator disagreement—meaning hard label delivery is equivalent to "injecting structured noise on high-disagreement samples."

**Core Idea**: While fixing the annotator distribution as the target, delivery format is treated as an independent variable. Multipass (cycling through votes) and SLS (resampling by distribution) replace soft label CE. Theoretically, the expectation goals are equivalent; experimentally, these yield flatter basins and better soft NLL under sparse annotations.

## Method

### Overall Architecture
Each sample $x_i$ is paired with an empirical annotator distribution $p_i \in \Delta^{K-1}$ (derived from $m_i$ human votes), and the model predicts $q_\theta(x) = \mathrm{softmax}(z_\theta(x))$. The baseline utilizes soft label cross-entropy $\mathcal{L}_{\mathrm{soft}} = \sum_i H(p_i, q_\theta(x_i))$. This work investigates the impact of changing the delivery format by fixing the per-example target and isolating the format as an independent variable: replacing soft labels with multipass (cycling through real votes for hard labels) and SLS (resampling a hard label per epoch according to $p_i$). Two control experiments are included to isolate "sampling randomness" and "sample-distribution pairing." All methods use a CIFAR-adapted ResNet-18 trained for 200 epochs with cosine annealing.

### Key Designs

**1. Multipass: Deterministically cycling hard labels with real vote counts to eliminate sampling variance**

When original vote counts $\{c_{ik}\}_k$ are available, a drawback of soft labels is that they compress disagreement into a fixed vector, hiding the specific votes that constitute the disagreement. Multipass expands the votes of each instance into a "label multiset," shuffles them once with a fixed seed, and cycles through them across epochs using $\texttt{epoch} \bmod m_i$. If an instance has 50 votes, it completes one cycle every 50 epochs, with each observed vote appearing exactly once. Unlike Sheng et al. (2008), which expands repeated annotations into multiple samples, the number of samples per epoch remains $N$; only the "label identity" per instance changes. Because the sequence is deterministic and reproducible, it avoids the sampling variance of SLS while verifying if changing labels between epochs is sufficient.

**2. SLS: Resampling hard labels according to the distribution and proving expectation equivalence**

When only $p_i$ is available without raw counts, SLS independently samples $y_i^{(t)} \sim \mathrm{Categorical}(p_i)$ for each instance at the start of every epoch, followed by standard hard label CE training. This is a lightweight alternative to multipass. Its success despite "appearing like noise" is explained by Proposition 1, which proves its equivalence to the soft label CE expectation goal: $\mathbb{E}_{y \sim p}[-\log q_y] = H(p, q)$, with the expected gradient $\mathbb{E}_{y \sim p}[\nabla_z \ell] = q - p$, identical to the soft label gradient. The only difference is the additional gradient variance from sampling $\mathbb{E}\|q-e_y\|^2 - \|q-p\|^2 = 1 - \|p\|_2^2$, where the covariance is exactly $\mathrm{Cov}_{y \sim p}[\nabla_z \ell] = \mathrm{Diag}(p) - pp^\top$—greater disagreement (flatter $p$) leads to higher variance. This strictly decouples "what the target is" from "how the optimization path proceeds."

**3. Deterministic / Shuffled Dual Controls: Separating "pairing information" from "general noise"**

To determine if SLS benefits stem from more than just noise, two controls were added. The **deterministic control** is a traversal-order ablation for multipass—changing only the fixed shuffle seed to confirm that the cycle order itself is irrelevant. **Shuffled SLS** permutes the $p_i$ across samples before performing SLS, preserving global disagreement statistics but breaking the "sample-distribution" pairing. If it still works, SLS gains are just "general noise regularization"; if it fails, "matching each instance to its own distribution" is the key. In experiments, shuffled SLS degraded to 12% accuracy, confirming that pairing is a first-order factor.

### Loss & Training
- Soft label: $\mathcal{L}_{\mathrm{soft}} = -\sum_i \sum_k p_{ik} \log q_{\theta, k}(x_i)$; hard methods use standard hard-label CE.
- Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=$5 \times 10^{-4}$, cosine annealing for 200 epochs, batch=128, with random cropping and horizontal flipping.
- Evaluation follows the proper-scoring-rule perspective: Primary metrics are soft NLL, secondarily KL-to-annotator and soft Brier, followed by hard_acc, equal-mass ECE, and Spearman entropy correlation. Hessian $\lambda_{\max}$ is estimated via power iteration, and trace via Hutchinson, calculated in eval mode with BN frozen at the best-soft-NLL checkpoint.

## Key Experimental Results

### Main Results
CIFAR-10H with 10,000 images, 80/20 stratified split, 10 seeds for main comparison / 5 paired seeds for the hard-delivery family and sparse scanning.

| Method | Soft NLL ↓ | Hard Acc ↑ | ECE_eqmass ↓ | EntCorr ↑ |
|------|-----------|-----------|-------------|-----------|
| Majority vote | 0.7284 | 0.8570 | 0.0704 | 0.2902 |
| Label smoothing | 0.6263 | 0.8590 | 0.0598 | 0.2117 |
| Mixup | 0.5526 | **0.8824** | 0.0977 | 0.2499 |
| Soft labels | 0.5096 | 0.8687 | 0.0185 | 0.3909 |
| **SLS** | **0.5052** | 0.8695 | 0.0186 | **0.3946** |

With the full distribution, SLS and soft labels show no significant differences across 4 metrics ($p \in [0.38, 0.92]$); mixup has the highest accuracy but nearly 5x worse ECE.

Sparse annotation scan (Soft NLL for K∈{5, 10, 25, 50}, 5 paired seeds per cell):

| Method | K=5 | K=10 | K=25 | K=50 |
|------|-----|------|------|------|
| Soft labels | 0.5860 | 0.5785 | 0.5388 | 0.5628 |
| SLS | 0.5599 (p=.031) | 0.5485 (.031) | 0.5169 (.063) | 0.5291 (.094) |
| Multipass | 0.5649 (.063) | 0.5371 (.031) | 0.5117 (.031) | 0.5241 (.031) |
| Det. control | **0.5555** (.031) | **0.5388** (.031) | **0.5077** (.031) | **0.5231** (.031) |

In 12/12 cells, hard methods outperform soft labels directionally, with 9/12 cells consistent across all 5 seeds. Improvements are greater when K is smaller and are positively correlated with the JS distance between empirical $p_i$ and the true distribution (Spearman 0.05–0.16).

### Ablation Study

| Configuration | Soft NLL | Hard Acc | EntCorr | $\lambda_{\max}$ (full) | Trace (full) |
|------|----------|----------|---------|-------------------------|--------------|
| Soft labels | 0.5096 | 0.8687 | 0.3909 | 242.2 | 4946.0 |
| SLS | 0.5052 | 0.8695 | 0.3946 | **104.9** | **1633.9** |
| Multipass | 0.4942 | 0.8714 | 0.4000 | 103.8 | 1571.4 |
| Det. control | 0.4921 | 0.8724 | 0.3963 | 101.6 | 1581.6 |
| Shuffled SLS | 2.2973 | 0.1199 | -0.006 | 18.4 | 33.9 |

Multipass / SLS / Det. control are geometrically almost identical ($\lambda_{\max} \sim 100$, 2.4x smaller than soft). Shuffled SLS is the "flattest" but approaches random classification—indicating that "flatness" must be interpreted while maintaining pairing. Stale-target probe: fixing sampled labels for 1/5/10/50 epochs monotonically worsened soft NLL from 0.5027 to 0.6689.

### Key Findings
- **Pairing is first-order**: Shuffled SLS has the flattest geometry but only 12% accuracy, proving that hard delivery benefits are not from "general noise regularization" but from "per-example distribution matching" and noise covariance reshaped by disagreement.
- **Structured gradient noise**: The Spearman correlation between last-layer gradient variance and annotator entropy averaged 0.939 across seeds—additional variance from SLS is concentrated on high-disagreement samples, consistent with $\mathrm{Diag}(p) - pp^\top$.
- **Same target, different basins**: The mean loss barrier between SLS and soft checkpoints is 2.05 (significantly > 0). With CKA at 0.920 vs 0.887 and Grad-CAM cross-seed stability at 0.901 vs 0.804, endpoint losses are similar but occupy different basins; hard delivery representations are more reproducible.
- **OOD endpoint gains**: Hard delivery outperformed soft in 5 out of 6 detectors on SVHN. On CIFAR-100, SLS outperformed soft labels in AUROC across all scores, with Energy/ODIN paired $p = 0.0186$.

## Highlights & Insights
- **Treating delivery as an independent variable is an elegant research paradigm**: Previous works on training with annotator distributions often changed targets and delivery simultaneously. This paper mathematically decouples them via Proposition 1 and further decouples "sampling randomness" via deterministic control—a three-tier ablation design applicable to any stochastic vs. deterministic optimization comparison.
- **Multipass is an overlooked practical baseline**: When raw vote counts are available, cycling through the multiset is deterministic, reproducible, avoids SLS variance, and maintains the dataset size (unlike Sheng 2008). It is the "cheapest hard label delivery."
- **New evidence for "flat basins via structured label noise"**: While literature often uses SGD noise or symmetric label noise (Keskar, Smith, Damian), this work proves "hard labels sampled from the true distribution" drive the model into a ~2.4x flatter basin. The variance structure strictly corresponds to $\mathrm{Diag}(p) - pp^\top$, providing a concrete, interpretable example for "covariance-Hessian alignment" (HaoChen/Wu).

## Limitations & Future Work
- The authors acknowledge limitations: Single dataset (CIFAR-10H) and single architecture (ResNet-18 with CIFAR stem); not yet verified on ChaosNLI, LeWiDi, or larger vision models. In the sparse scan, though 12/12 cells were directionally superior, no single cell remained significant after Holm correction (minimum raw $p=0.03125$ for 5 paired seeds). OOD comparisons mixed the 10-seed main table and 5-seed ablation table and are treated as descriptive evidence.
- This note observes: Sparse K is simulated by resampling the dense CIFAR-10H; real sparse datasets may include "annotator selection bias" and task-specific disagreement structures. Proposition 1 only provides per-step variance; the causal link from basin geometry to endpoint metrics remains observational.
- Future directions: Applying multipass to high-disagreement NLP data like ChaosNLI/SBIC; comparing with explicit flatness optimizers like SAM to see if "structured label noise" can replace explicit sharpness regularization.

## Related Work & Insights
- **vs. Peterson et al. (2019)**: They introduced CIFAR-10H and proved soft-label CE > majority vote; this work fixes the target distribution and varies Only the delivery, providing a finer causal slice.
- **vs. Sheng et al. (2008) Repeated Labeling**: Sheng expands the dataset $N \to \sum m_i$; multipass keeps $N$ constant and cycles the labels—a cardinality-preserving version of the same idea.
- **vs. DisturbLabel (Xie et al., 2016)**: DisturbLabel uses uniform noise to perturb labels; SLS samples from the true $p_i$, aligning noise structure with the target distribution. This makes it "information-preserving noise" rather than destructive.
- **vs. Label smoothing / Mixup**: LS uses a uniform soft target, and mixup uses input-target interpolation; neither preserves per-instance epistemic structure. Table 2 shows their ECE is significantly worse than SLS/soft, validating the need for example-to-distribution pairing.
- **vs. Flat-minima literature (Keskar 2017, Smith 2021, HaoChen 2021)**: This work provides a new mechanism for how hard-label delivery flattens basins, with a precise analytical form for the variance covariance.

## Rating
- Novelty: ⭐⭐⭐⭐ Treating delivery as an independent variable and decoupling it via expectation equivalence is a rare causal slice in the annotator-disagreement field; however, the sampling idea in SLS was nascent in Peterson 2019.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes 3 main methods, 2 controls, sparse scanning, Hessian, loss barriers, CKA, Grad-CAM, and OOD; however, limited by single dataset/architecture.
- Writing Quality: ⭐⭐⭐⭐ Clear organization, consistent "regime split" narrative, and well-explained roles for propositions and controls.
- Value: ⭐⭐⭐⭐ Provides multipass as a practical default for annotator distribution tasks and builds a concrete, verifiable bridge between flat-minima and structured label noise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)
- [\[ICLR 2026\] Dataset Distillation for Memorized Data: Soft Labels can Leak Held-Out Teacher Knowledge](../../ICLR2026/ai_safety/dataset_distillation_for_memorized_data_soft_labels_can_leak_held-out_teacher_kn.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](../../AAAI2026/ai_safety/easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[ICML 2026\] Old Habits Die Hard: How Conversational History Geometrically Traps LLMs](old_habits_die_hard_how_conversational_history_geometrically_traps_llms.md)

</div>

<!-- RELATED:END -->
