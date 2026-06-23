---
title: >-
  [Paper Note] Memorizing Long-tail Data Can Help Generalization Through Composition
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper proves in an overparameterized linear model that memorizing long-tail features appearing only once, combined with the model's inherent "compositional" ability, enables correct predictions on novel combinations of long-tail features never seen during training. This intuition is validated on modified MNIST/Omn
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: ffd3162fd838d1ae
---
# Memorizing Long-tail Data Can Help Generalization Through Composition

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=UBoCMU5iYV](https://openreview.net/forum?id=UBoCMU5iYV)  
**Code**: [https://github.com/mhy-666/long-tail-memorization-composition](https://github.com/mhy-666/long-tail-memorization-composition)  
**Area**: Learning Theory / Generalization Theory  
**Keywords**: Memorization, Long-tail Data, Compositional Generalization, Overparameterization, Minimum-norm Solution, OOD Generalization  

## TL;DR
This paper proves in an overparameterized linear model that memorizing long-tail features appearing only once, combined with the model's inherent "compositional" ability, enables correct predictions on novel combinations of long-tail features never seen during training. This intuition is validated on modified MNIST/Omniglot datasets, showing that compositional capability depends on network architecture.

## Background & Motivation

**Background**: Classical statistical learning theory suggests that memorizing training data (especially noise/random labels) hurts generalization, but deep networks consistently challenge this—they generalize well despite memorizing random labels. Two explanation paths have emerged: first, implicit regularization/benign overfitting, where training processes and architectures favor solutions that generalize; second, the "memorizing the long tail" perspective by Feldman (2020), suggesting real-world data follows a long-tail distribution where memorizing rare samples captures unique sub-concepts, aiding predictions when test samples resemble training ones.

**Limitations of Prior Work**: Feldman's "test-train similarity" explanation has a ceiling—it only covers cases where test samples look like seen training samples. However, large models exhibit a more striking property: **composition**. They combine multiple pieces of information learned during training to complete tasks never seen before. These samples are not similar to any single training sample, rendering similarity arguments ineffective.

**Key Challenge**: Memorizing long-tail features and compositional capability should intuitively be synergistic—a compositional model can combine two separately memorized long-tail features into a new combination with extremely low probability in the training distribution. However, there has been no theoretical characterization of how "memorization + composition" jointly lead to OOD generalization, nor is it clear under what conditions this synergy holds.

**Goal**: To clarify and prove "memorizing the long tail + composition → OOD generalization" using the simplest possible model, and to test whether this intuition extends to neural networks.

**Core Idea**: **Make composition "automatically happen" using linear models.** Once a linear model $f(x)=\langle\beta,x\rangle$ correctly learns the coefficients $\beta_i$ for individual features, predictions for any combination of these features are automatically accurate. Thus, it suffices to prove that the **minimum $\ell_2$ norm solution under overparameterization (which memorizes the training set) can recover the coefficients of most long-tail features**, making compositional generalization a free byproduct. The key technical challenge is that long-tail features appear very few times (even once), causing standard concentration inequalities to fail. The authors instead **exploit the compositional structure of the data matrix**: under appropriate power-law decay, most samples contain at most one long-tail feature.

## Method

### Overall Architecture

This work is primarily theoretical, supported by experiments. The theoretical part defines a sparse linear data model with long-tail features and uses gradient descent initialized from zero to obtain a minimum-norm interpolating solution (the memorized solution). It proves that this solution recovers common features and most long-tail features, providing upper bounds for both in-distribution test loss and OOD (compositional) loss (distinguishing between noiseless and noisy cases). The experimental part replicates theoretical curves on linear models and then transitions to a compositional task of "predicting the sum of 3 MNIST digits," comparing different ResNet aggregation architectures and using Omniglot one-shot samples with weight decay toggles to isolate the "memorization" factor.

```mermaid
flowchart TD
    A[Long-tail Sparse Linear Data<br/>Feature Frequency Power Law pi~i^-α] --> B[Overparameterized d≫n<br/>Gradient Descent from 0]
    B --> C[Min ℓ2 Norm Solution β̂=X⁺y<br/>Memorizes Training Set]
    C --> D{Compositional Structure Lemma<br/>Most samples ≤1 long-tail feature}
    D --> E[Recover Top-k Common Features β̂≤k=β*≤k]
    D --> F[Recover Most Long-tail Features<br/>Accurate even with 1 appearance]
    E & F --> G[Linear → Composition Automatic<br/>OOD Loss ≤ σ²t (Small)]
    G --> H[NN Exp: 3-digit Summing<br/>Split-channel ResNet has composition<br/>Cross-channel does not]
```

### Key Designs

**1. Long-tail Sparse Linear Data Model: Placing "Long-tail + Composition" into an analyzable form.** Each dimension $i$ of data $x\in\{0,1,-1\}^d$ is non-zero independently with probability $p_i$ (with random signs). Frequencies follow $p_1\ge p_2\ge\cdots\ge p_d$, assuming power-law decay $p_i=\min\{1,\,s\cdot i^{-\alpha}/Z_\alpha\}$. Each sample is roughly $s$-sparse ($\mathbb{E}\|x\|_2^2\approx s$). The first $k$ dimensions are high-frequency "common features," and the rest are low-frequency "long-tail features." Labels are $y=\langle\beta^*,x\rangle+\xi$, where $\xi\sim\mathcal N(0,\sigma^2)$. The brilliance of this model is that since the predictor is linear, "learning accurately on a subset of features → accurate prediction on any combination" holds naturally. **Compositional generalization is explicitly reduced to a feature coefficient recovery problem**, where the long-tail structure represents the difficulty of having minimal evidence for certain $\beta_i$.

**2. Minimum-norm Interpolating Solution as a mathematical surrogate for "Memorization."** In the overparameterized setting $d\gg n$, running gradient descent on the squared loss $L(\beta)=\frac1n\|X\beta-y\|^2$ from zero initialization converges to the minimum $\ell_2$ norm solution $\hat\beta=(X^\top X)^\dagger X^\top y$. In the noiseless case $\sigma=0$, it exactly interpolates (memorizes) all training data. In the noisy case, while it might not reach zero training loss, it remains the global minimum with the smallest norm. This is used to precisely characterize "memorization."

**3. Compositional Structure Lemma: Bypassing failed concentration inequalities.** The core obstacle is that current common feature frequencies can be as low as $p_k=\tilde\Theta(1/n)$, and long-tail features are even rarer, leading to high variance where standard concentration bounds fail. The breakthrough is analyzing the structure of $X$: under appropriate power-law decay, it is proven that **most samples contain at most one long-tail feature** (Lemma 6/7). Specifically, $\Theta(n(1-p_{>k}))$ samples use only $k$ common features, enough to uniquely determine the common parts $\hat\beta_{\le k}=\beta^*_{\le k}$ when $k<n$. Once common features are fixed, long-tail features $\hat\beta_i$ that "co-occur only with common features and not other long-tail features" can be recovered accurately—even if they appear only once in training. This ensures the recovery of almost all seen features, with the proportion of recovered features $\hat F$ being $1-\Theta(\max\{p_k/p_{>k},\,p_{>k}^2\ln^2 d\})$.

**4. From Feature Recovery to in/OOD Guarantees.** Substituting feature recovery back into the model yields generalization bounds. In the noiseless case (Theorem 2/3): test loss $\lesssim p_{>k}+\frac{k\ln^4 d}{n^2 p_{>k}}+\cdots$, which simplifies under power law to $\tilde O\big(s(\ln^2 d/ns)^{1-1/\alpha}\big)$. More importantly, for the OOD guarantee: for any distribution $D_{\tilde F}$ composed of subsets of recovered features $\hat F$ (simulating combinations never seen during training), the loss is similarly bounded. In the noisy case (Theorem 4/5), stronger tail decay ($\alpha=2+c_\alpha$) ensures the "one long-tail feature per sample" property. If a long-tail feature appears in only one sample, training loss for that sample is zero, the feature is estimated accurately (error at noise level $\sigma$), and test loss is approximately $\Theta(\sigma^2 k/n)$.

## Key Experimental Results

### Linear Model Verification (Figure 1)

With $n=1000, d=10000, s=5, np_k=10, \beta^*_i=i^{-0.1}$ over 50 runs, results match theory:

| Observation | Phenomenon | Related Theory |
| :--- | :--- | :--- |
| Long-tail error vs. Noise | Error roughly proportional to noise $\sigma$ | Thm 1: Error for single-appearance features slightly > noise |
| Common feature error | Well-learned when $\alpha$ is not too small | Thm 2 |
| in/OOD test loss | Both small for large $\alpha$; $\sigma=0$ and $0.05$ curves nearly overlap | Thm 2/3/4/5 |
| $\alpha=1$ (Slow decay) | in/OOD loss and feature error both increase | Too many long-tail features ruin compositional structure |

### Neural Network Compositional Task (Figure 2 + Table 1)

Task: Stack 3 MNIST digit images channel-wise ($3\times28\times28$) to predict their sum. Digits follow a Zipf distribution ('0' most common, '9' rarest). Test samples are forced to contain at least one '9' (others from 0-4), where the test '9' instances were never seen during training. Training set has 32,000 samples.

**Architecture Comparison (Figure 2)**: **Sum / Linear / 2-layer** models (processing channels independently via ResNet-18 before aggregating) perform well on '9'-heavy OOD tests—even though '9' is rare. Conversely, the **Cross-channel** model (mixing information in the first layer) shows significantly worse test loss, proving compositional capability depends on architecture.

**Memorization Isolation Experiment (Table 1)**: 10 Omniglot images (1 per class, appearing once, labeled 0-9) are mixed into training. Test samples contain two Omniglot images (never co-occurring in training and not close to any training samples). Weight decay (WD) is used to toggle memorization:

| Model | WD=0 (Memorized) Test/Train | WD=0.5 (Not Memorized) Test/Train |
| :--- | :--- | :--- |
| Sum | 0.1760 / 0.0004 | 2.8744 / 0.0245 |
| Linear | 0.1662 / 0.0004 | 1.5539 / 0.0505 |
| 2-layer | 0.1351 / 0.0010 | 0.6447 / 0.0264 |
| Cross-channel | 22.9671 / 0.0023 | 23.6506 / 0.0375 |

### Key Findings
- Memorizing extremely rare samples (appearing once), combined with a compositional architecture, maintains low loss on OOD tests like "two Omniglot co-occurrence." Disabling memorization via weight decay causes test loss to explode despite only a slight increase in training loss.
- The Cross-channel model performs poorly (test loss 22+) regardless of memorization, confirming that composition is an architectural property; memorization only helps when the architecture is capable of composition.
- Qualitative predictions of the linear theory (long-tail error vs. noise, structure destruction by small $\alpha$) are confirmed in simulations.

## Highlights & Insights
- **Reducing "composition" to "feature recovery" is the cleverest move**: The linear model is chosen not for simplicity alone, but because it makes composition natural, transforming a vague capability into a provable algebraic recovery problem.
- **The real technical contribution is the Compositional Structure Lemma**: Using the argument that "most samples contain at most one long-tail feature" instead of concentration inequalities is a reusable tool for handling extremely rare features.
- **Provides a new compositional perspective on why memorization is necessary**: While Feldman used similarity to explain memorization's value, this paper advances it to "memorization is a prerequisite for composing long-tail features," covering OOD regions similarity cannot reach.
- **Architectural dependence is a clean, impactful empirical result**: Showing that the aggregation method (split-channel vs. cross-channel) determines compositional power suggests composition is a structural inductive bias rather than just capacity.

## Limitations & Future Work
- The theory strictly covers only the **linear setting**, with the neural network part being an intuitive extension; there is a significant gap between the two.
- Tasks are relatively toy (3-digit sum, Omniglot single-sample), far from real-world large-scale compositional generalization.
- Memorization could hurt composition if long-tail features are **spuriously correlated** or **mislabeled**—this paper focuses only on the positive effects.
- The noisy case requires strong tail decay assumptions ($\alpha=2+c_\alpha$) to maintain the "one long-tail feature per sample" property, which may not match real-world distributions.

## Related Work & Insights
- **Benign Overfitting / Implicit Regularization** (Belkin 2019, Bartlett 2020, Hastie 2022): Explains why memorization doesn't hurt; this paper uses the same overparameterized framework to ask how memorization *actively helps* composition.
- **Memorizing the Long Tail** (Feldman 2020; Feldman & Zhang 2020): Direct predecessor; this paper evolves the "memorization → similar sample prediction" logic into "memorization → long-tail feature composition."
- **Compositional Generalization** (SCAN, CoGS benchmarks): Contextualizes the work; this paper focuses on the theoretical bridge to memorization.
- **Insight**: For long-tail/rare class learning, do not blindly suppress memorization with strong regularization—it may be necessary for OOD compositional generalization. Architectures should favor "encode separately, then aggregate" over "early mixing."

## Rating
- **Novelty**: ⭐⭐⭐⭐ First provable framework linking "memorizing the long tail + composition → OOD generalization."
- **Experimental Thoroughness**: ⭐⭐⭐ Good linear-theory alignment and clean NN design, but limited to toy tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and direct theoretical-experimental correspondence.
- **Value**: ⭐⭐⭐⭐ Provides a clean theoretical anchor for understanding how models generalize on unseen combinations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](optimizing_data_augmentation_through_bayesian_model_selection.md)
- [\[ICLR 2026\] Covariate-Guided Clusterwise Linear Regression for Generalization to Unseen Data](covariate-guided_clusterwise_linear_regression_for_generalization_to_unseen_data.md)
- [\[ICLR 2026\] Critical Attention Scaling in Long-Context Transformers](critical_attention_scaling_in_long-context_transformers.md)
- [\[ICLR 2026\] Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence](escaping_model_collapse_via_synthetic_data_verification_near-term_improvements_a.md)
- [\[ICLR 2026\] Provable Separations between Memorization and Generalization in Diffusion Models](provable_separations_between_memorization_and_generalization_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
