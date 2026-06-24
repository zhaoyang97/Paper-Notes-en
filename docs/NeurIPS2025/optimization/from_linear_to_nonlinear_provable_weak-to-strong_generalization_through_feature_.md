---
title: >-
  [Paper Note] From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning
description: >-
  [NeurIPS 2025][Optimization][Weak-to-strong generalization] This paper presents the first rigorous analysis of the weak-to-strong generalization phenomenon under a non-linear feature learning setting (linear CNN $\rightarrow$ two-layer ReLU CNN). It reveals distinct behaviors under data-scarce and data-abundant regimes: the former achieves generalization through benign overfitting (or fails due to harmful overfitting), while the latter achieves generalization through label co…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Weak-to-strong generalization"
  - "superalignment"
  - "benign overfitting"
  - "feature learning"
  - "CNN"
date: 2026-05-08
content_hash: 84964c53dc0a4e1f
---

# From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning

**Conference**: NeurIPS 2025  
**arXiv**: [2510.24812](https://arxiv.org/abs/2510.24812)  
**Code**: None  
**Area**: Learning Theory / Superalignment  
**Keywords**: Weak-to-strong generalization, superalignment, benign overfitting, feature learning, CNN

## TL;DR
This paper presents the first rigorous analysis of the weak-to-strong generalization phenomenon under a non-linear feature learning setting (linear CNN $\rightarrow$ two-layer ReLU CNN). It reveals distinct behaviors under data-scarce and data-abundant regimes: the former achieves generalization through benign overfitting (or fails due to harmful overfitting), while the latter achieves generalization through label correction under early stopping (but degrades with overtraining).

## Background & Motivation
**Background**: As LLM capabilities surpass human performance, how to guide stronger models using weaker supervision (such as human feedback) has become a core challenge of "superalignment". Burns et al. (2024) experimentally found that a strong model supervised by a weak model can outperform the weak teacher, a phenomenon termed weak-to-strong generalization (W2S).

**Limitations of Prior Work**:  
(a) Existing theoretical analyses are mostly based on abstract frameworks (e.g., Lang et al., Charikar et al.) and cannot guarantee that W2S is realizable through actual optimization processes like gradient descent;  
(b) Existing constructability analyses are limited to linear or random feature models (Wu & Sahai, Dong et al., Medvedev et al.) and do not involve non-linear feature learning.

**Key Challenge**: Labels provided by a weak model are inevitably erroneous (essentially random guessing on hard-only data). When a strong model is trained on these noisy labels, why can it still perform better than the weak model? How does this happen in non-linear feature learning scenarios?

**Key Insight**: Design a structured data distribution containing "easy signals" (learnable by the weak model) and "hard signals" (unlearnable by the weak model), along with "bridge data" containing both signals. Since the weak model correctly labels the bridge data, the strong model can leverage this to learn the hard signals.

**Core Idea**: By analyzing the dynamics of gradient descent, this paper proves that a non-linear strong model (ReLU CNN) can learn hard features that the weak model fails to capture under the supervision of the weak model's pseudo-labels. The key condition is the presence of sufficient data containing both easy and hard signals.

## Method

### Overall Architecture
- **Data Distribution**: Each sample contains 3 patches with randomly assigned signals and noise. Signals are divided into "easy signals" $\mu$ and "hard signals" $\nu$. Data belongs to three categories: easy-only (probability $p_e$), hard-only (probability $p_h$), and both-signal (probability $p_b$).
- **Weak Model**: A linear CNN $f_{\text{wk}}(w, X) = \sum_{p} \langle w, x^{(p)} \rangle$, which cannot distinguish the signs of hard signals (Proposition 2.1 proves that the error on hard-only data is always 50%).
- **Strong Model**: A two-layer ReLU CNN $f_{\text{st}}(W, X) = F_1(W_1, X) - F_{-1}(W_{-1}, X)$. Leveraging the non-linearity of ReLU, it can simultaneously learn $\nu$ and $-\nu$ (Proposition 2.2 proves the existence of a zero-error solution).
- **Training Pipeline**: First train the weak model using true labels $\rightarrow$ generate pseudo-labels for new data using the weak model $\rightarrow$ train the strong model using the pseudo-labels.

### Key Designs

1. **The Core Role of Bridge Data (both-signal data)**
    - Function: Data points containing both the easy signal $\mu_y$ and the hard signal $\nu_y$.
    - Mechanism: The weak model correctly classifies these data points using the easy signal, resulting in correct pseudo-labels. During the training of the strong model, these correctly labeled samples allow it to learn not only the easy signal but also the hard signal.
    - Design Motivation: This serves as the key bridge for weak-to-strong generalization—the weak model "sees" the easy signal to provide a correct label, while the strong model "sees" the hard signal in the same data and learns it.

2. **Data-Scarce Regime: Critical Conditions for Benign/Harmful Overfitting**
    - Function: Analyzes the overfitting behavior of the strong model when $n_{\text{st}}$ is small (dominated by noise memorization).
    - Mechanism: (Theorem 3.4) Benign overfitting occurs when $n_{\text{st}} p_b^2 \|\nu\|^4 / (\sigma_p^4 d) \geq C$ (test error approaches 0); otherwise, harmful overfitting occurs (test error is at least $0.12 p_h$). The two thresholds differ only by a constant factor, providing a tight characterization.
    - Physical Intuition: Data volume determines the competition between "signal learning" and "noise memorization"—when data is sufficient, the cumulative effect of hard signals in both-signal data outweighs the interference of noise memorization.

3. **Data-Abundant Regime: Label Correction via Early Stopping**
    - Function: Analyzes the generalization behavior in the early stages of training when $n_{\text{st}}$ is abundant (dominated by signal learning).
    - Mechanism: (Theorem 3.6) There exists an early stopping time $T_{\text{es}}$ at which the strong model correctly classifies all correctly labeled training data, and can "correct" predictions on all label-flipped data to predict the true labels—meaning the model actually predicts the true label $\tilde{y}_i$ instead of the pseudo-label $\hat{y}_i$.
    - Overtraining Degradation: As training continues, the loss gradient for label-flipped data increases (since the model "disobeys the pseudo-labels"), leading to the "forgetting" of hard signals, and the test performance degrades back to the weak model's level.

4. **The Critical Role of ReLU Non-linearity**
    - Function: Utilizing the positive-part activation property of ReLU to simultaneously learn both signs of the hard signal.
    - Mechanism: After initialization, different filters $w_{s,r}$ have different signs of inner products with $\nu_s$. Filters with positive inner products learn $\nu_s$, while those with negative inner products learn $-\nu_s$. ReLU ensures they do not interfere with each other.
    - Contrast with Linear Models: Linear models process $\nu$ and $-\nu$ with the same parameters, causing the updates to cancel out, thereby preventing them from ever learning the hard signal.

### Loss & Training
- logistic loss: $\ell(z) = \log(1 + e^{-z})$
- Gradient Descent (not SGD), learning rate $\eta$
- The weak model is initialized from zero; the strong model is initialized with small random Gaussian weights $\sigma_0$.

## Key Experimental Results

### Main Results

| Data Volume $n_{\text{st}}$ | Training Accuracy | Test Accuracy | Phenomenon | Corresponding Theory |
|------------------------|----------|----------|------|---------|
| 75 (Data-scarce - Low) | ~100% | ~85% (≈ Weak Model) | Harmful Overfitting | Theorem 3.4 harmful |
| 2000 (Data-scarce - High) | ~100% | **>85% (Better than Weak Model)** | Benign Overfitting | Theorem 3.4 benign |
| 20000 (Data-abundant) | ~85% (Not converging to 100%) | **~100% (Near-perfect)** | Label Correction + Overtraining Degradation | Theorem 3.6 |

### Ablation Study

| Setting | $n_{\text{st}}$ | Strong Model Test Accuracy | Weak Model Accuracy | W2S Generalization |
|------|------|----------|--------|------|
| Data-scarce | Small | ≈ Weak Model | Baseline | Failed |
| Data-scarce | Medium | > Weak Model | Baseline | Success (Benign Overfitting) |
| Data-abundant + Early Stopping | Large | **Significantly > Weak Model** | Baseline | Success (Label Correction) |
| Data-abundant + Overtraining | Large | ≈ Weak Model | Baseline | Degradation |

### Key Findings
- W2S generalization under data-scarce and data-abundant regimes is achieved through completely different mechanisms.
- The proportion of both-signal data $p_b$ is the key parameter determining the success of W2S generalization.
- Overtraining is harmful in the data-abundant scenario—early stopping is crucial.
- The characterization of the critical conditions for benign/harmful overfitting is tight (the upper and lower bounds differ only by a constant).

## Highlights & Insights
- **First W2S Theory under Non-linear Feature Learning**: Prior theories operated in linear or random feature models; this work is the first to consider ReLU CNN feature learning.
- **Unified Framework for Two Regimes**: The seemingly contradictory behaviors of the data-scarce regime (benign overfitting) and the data-abundant regime (label correction + early stopping) naturally emerge under the same framework.
- **Practical Implications: Data Selection Strategy**: The theory highlights the critical role of "bridge data" (which simultaneously contains features that the weak model can and cannot recognize). This inspires data selection in practical W2S training—prioritizing such data can improve generalization.
- **Theoretical Explanation for Overtraining Degradation**: Burns et al. (2024) experimentally observed overtraining degradation but could not explain it; this paper provides a clear mechanism—the gradient from label-flipped data eventually dominates training, causing the hard features to be forgotten.

## Limitations & Future Work
- **Simplified Data Distribution**: The 3-patch structured data is far from real-world images/text, and the signal-noise orthogonality assumption is overly strong.
- **Fixed Second-Layer Weights**: The strong model only trains the first layer, lacking analysis of full two-layer training.
- **GD rather than SGD**: The analysis uses full-batch gradient descent; extending this to mini-batch SGD requires additional work.
- **No Multi-class Classification**: Only binary classification is considered; generalization conditions for multi-class scenarios may be more complex.
- **Gap between Theory and LLM Practice**: There remains a significant gap between CNN theory and W2S behaviors in transformer-based LLMs.

## Related Work & Insights
- **vs Burns et al. (2024)**: They provided experimental observations from GPT-2 $\rightarrow$ GPT-4; this paper provides the first interpretable theoretical mechanism.
- **vs Charikar et al. (2024)**: They analyzed the connection between misfit and W2S gain under an abstract regression framework, but did not involve the optimization process. This paper analyzes actual GD dynamics.
- **vs Wu & Sahai (2025), Medvedev et al. (2025)**: They conducted analyses in linear/random feature models; this paper extends to non-linear feature learning.
- **vs Cao et al. (2022) (Benign Overfitting)**: This paper adopts the signal-noise decomposition technique, but the application scenario is entirely different—prior work analyzed training with true labels, while this work analyzes training with pseudo-labels.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First rigorous analysis of W2S generalization under a non-linear feature learning framework.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical, with experiments only validating theoretical predictions on synthetic data and small-scale CIFAR.
- Writing Quality: ⭐⭐⭐⭐ The theoretical statements are clear, and the dynamical intuition is well-explained.
- Value: ⭐⭐⭐⭐ Highly significant for the superalignment/W2S theoretical community, with practical implications worth exploring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Exact and Linear Convergence for Federated Learning under Arbitrary Client Participation is Attainable](exact_and_linear_convergence_for_federated_learning_under_arbitrary_client_parti.md)
- [\[ICML 2025\] Random Feature Representation Boosting](../../ICML2025/optimization/random_feature_representation_boosting.md)
- [\[ICLR 2026\] Compositional Generalization through Gradient Search in Nonparametric Latent Space](../../ICLR2026/optimization/compositional_generalization_through_gradient_search_in_nonparametric_latent_spa.md)
- [\[ICML 2025\] A Generalization Result for Convergence in Learning-to-Optimize](../../ICML2025/optimization/a_generalization_result_for_convergence_in_learning-to-optimize.md)

</div>

<!-- RELATED:END -->
