---
title: >-
  [Paper Note] Deep Learning is Not So Mysterious or Different
description: >-
  [ICML 2025][Self-Supervised Learning][Generalization theory] This is a position paper arguing that the generalization phenomena deemed "mysterious" in deep learning (benign overfitting, double descent, and the success of overparameterization) are neither unique to deep learning nor mysterious. They can be formalized using long-standing generalization frameworks (PAC-Bayes and countable-hypothesis bounds) and unified under the explanatory principle of **soft inductive biases**…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Generalization theory"
  - "Soft inductive biases"
  - "PAC-Bayes"
  - "Benign overfitting"
  - "Double descent"
date: 2026-05-08
content_hash: 4650b29207839acc
---

# Deep Learning is Not So Mysterious or Different

**Conference**: ICML 2025  
**arXiv**: [2503.02113](https://arxiv.org/abs/2503.02113)  
**Code**: None  
**Area**: Self-Supervised Learning  
**Keywords**: Generalization theory, Soft inductive biases, PAC-Bayes, Benign overfitting, Double descent

## TL;DR

This is a position paper arguing that the generalization phenomena deemed "mysterious" in deep learning (benign overfitting, double descent, and the success of overparameterization) are neither unique to deep learning nor mysterious. They can be formalized using long-standing generalization frameworks (PAC-Bayes and countable-hypothesis bounds) and unified under the explanatory principle of **soft inductive biases**.

## Background & Motivation

Deep neural networks are often thought to be fundamentally different from other models, as their generalization behavior seemingly defies traditional intuition. The most frequently cited examples include:
- **Benign overfitting**: Models perfectly fit noisy data yet still generalize.
- **Double descent**: Generalization error first decreases, then increases, and then decreases again as the number of parameters grows.
- **Success of overparameterization**: Models with far more parameters than data points still generalize well.

Since "Understanding Deep Learning Requires Rethinking Generalization" by Zhang et al. (2016), the deep learning community has widely believed that these phenomena necessitate "rethinking generalization" and represent mysteries unique to deep learning.

However, the author Andrew Gordon Wilson holds the opposite position: these phenomena are **neither unique to deep learning nor mysterious**. They can be replicated in simple linear models, understood intuitively, and have long been rigorously formalized by existing generalization frameworks (such as PAC-Bayes and countable hypothesis bounds). **Key Challenge**: The academic community has focused excessively on frameworks like VC dimension and Rademacher complexity, which **cannot** explain these phenomena, while ignoring the PAC-Bayes framework—which has existed for decades and **can** explain them.

The **Key Insight** of this paper is to explain all these phenomena through the unifying principle of **soft inductive biases**: instead of restricting the hypothesis space to prevent overfitting, one should embrace flexible hypothesis spaces while assigning a soft preference to simpler solutions that are consistent with the data.

## Method

### Overall Architecture

Instead of proposing a new algorithm, this paper is a theoretical position paper organized into the following argumentative structure:

1. Introduce the concept of **soft inductive biases** as a unifying intuition.
2. Introduce PAC-Bayes and countable hypothesis bounds as formal tools.
3. Analyze benign overfitting, overparameterization, and double descent sequentially, showing how each phenomenon can be replicated in simple models and explained by the aforementioned frameworks.
4. Discuss aspects where deep learning is **truly unique**: representation learning, mode connectivity, and universal learning capabilities.

### Key Designs

1. **Soft Inductive Biases vs. Restriction Biases**: Traditional views consider inductive bias as a **restriction** on the hypothesis space (restriction bias), such as the translation equivariance constraint in CNNs. However, the author argues that restriction bias is not only unnecessary but can also be detrimental. A better approach is introducing **soft biases**: embracing flexible hypothesis spaces while favoring certain solutions. A classic example is a **high-order polynomial with order-dependent regularization**: $\mathcal{L}(w) = -\log p(y|f(x,w)) + \sum_j \gamma^j w_j^2, \gamma > 1$. The model fits the data using lower-order terms as much as possible, deploying higher-order terms only when necessary. This guarantees flexibility while affording a preference for simplicity, performing well across all dataset sizes and complexities.

2. **PAC-Bayes and Countability Bounds**: The core generalization bound is given by $R(h) \leq \hat{R}(h) + \Delta\sqrt{\frac{K(h|A)\log 2 + \log\frac{1}{\delta}}{2n}}$, where $K(h|A)$ is the prefix-free Kolmogorov complexity of hypothesis $h$ with respect to architecture $A$. This can be simplified to: **expected risk $\leq$ empirical risk + model compressibility**. The key insight is that if a large model fits the data well **and** can be effectively compressed, good generalization is guaranteed. Unlike VC dimension or Rademacher complexity, these bounds do not penalize the **size** of the hypothesis space, but rather focus on the **probability** of the hypotheses. Recently, they have provided non-vacuous generalization guarantees for LLMs with billions of parameters.

3. **Effective Dimensionality**: Defined as $N_{\text{eff}}(A) = \sum_i \frac{\lambda_i}{\lambda_i + \alpha}$, measuring the number of "relatively large" eigenvalues in the Hessian matrix, which corresponds to the number of sharp directions in the loss landscape. A low effective dimensionality implies flatter solutions (where parameters can be perturbed without significantly increasing loss). A mechanistic link exists between flatness and generalization: flatter solutions are more compressible and exhibit better Occam factors.

### Explanation of Three "Mysterious" Phenomena

**Benign Overfitting**: All that is required is a flexible hypothesis space + a loss function enforcing perfect fit + a preference for simplicity. This can be replicated by a 150th-order polynomial with order-dependent regularization: it reasonably fits simple or complex structured data while perfectly fitting pure noise. Gaussian Processes (GPs) can **precisely** replicate the CIFAR-10 experimental results of Zhang et al. (2016). The PAC-Bayes bound is non-vacuous on structured data and vacuous on noisy data, perfectly passing the test proposed by Zhang et al.

**Overparameterization**: Parameter count is a poor proxy for model complexity. Increasing parameters provides two benefits: (1) it increases flexibility, allowing the model to fit data better; (2) it increases compression bias (larger models end up with fewer effective parameters after training). The intuition is that expanding the parameter count causes flat solutions to occupy a larger **relative volume** within the overall hypothesis space, making it easier for training to discover them. Experimental evidence shows that full-batch gradient descent and even random guess-and-check can find solutions that generalize well, indicating that the implicit regularization of stochastic optimizers is not a necessary condition for generalization.

**Double Descent**: In the underparameterized regime, increasing flexibility captures more useful information; in the transition regime, information increases but overfitting occurs; in the interpolation regime (parameters > data points), all models perfectly fit the data, but continuing to increase parameters expands the volume of compressible flat solutions, decreasing effective dimensionality and improving generalization. A simple linear model can replicate this phenomenon.

## Key Experimental Results

### Main Results

The "experiments" in this paper consist primarily of confirmatory examples rather than traditional performance comparisons:

| Model / Setting | Phenomenon | Key Observation | Description |
|-----------|------|----------|------|
| 150th-Order Polynomial + Order-Dependent Regularization | Benign Overfitting | Perfectly fits noise, reasonably fits structured data | Figure 1(a)-(c) |
| GP on CIFAR-10 | Benign Overfitting | Precisely replicates Zhang et al.'s results | Figure 1(d) |
| ResNet-20 on CIFAR-10 | Benign Overfitting | Marginal likelihood decreases as noisy labels increase | Figure 1(e) |
| ResNet-18 (varying width) on CIFAR-100 | Double Descent | Effective dimensionality precisely tracks the second descent | Figure 1(f) |
| Linear Random Feature Model | Double Descent | Exhibits the same double descent pattern as ResNet | Figure 1(g) |

### Comparison of Generalization Bounds

| Generalization Framework | Explains Benign Overfitting | Explains Overparameterization | Explains Double Descent | Non-vacuous Bounds |
|----------|--------------|------------|-------------|--------|
| VC Dimension / Rademacher Complexity | ✗ | ✗ | ✗ | Vacuous for large models |
| PAC-Bayes / Countable Hypothesis Bounds | ✓ | ✓ | ✓ | Non-vacuous for models with billions of parameters |
| Kolmogorov Complexity Upper Bound | ✓ | ✓ | ✓ | CIFAR-10: 16.6% error upper bound |

### Key Findings

- PAC-Bayes and countable-hypothesis bounds can provide non-vacuous generalization guarantees for models with millions or even billions of parameters.
- Lotfi et al. (2022a) bounded the classification error of a model with millions of parameters on CIFAR-10 at 16.6% (with 95% probability), which represents a remarkably tight bound.
- Larger models are not only more flexible but actually have fewer effective parameters after training (Maddox et al., 2020).
- After training, Vision Transformers exhibit even greater translation equivariance than CNNs (Gruver et al., 2023).
- Both full-batch gradient descent and random guess-and-check can discover solutions that generalize well (Geiping et al., 2021; Chiang et al., 2022).

## Highlights & Insights

- **Highly Pedagogical**: Replicates all "mysterious" phenomena using the simplest possible examples (high-order polynomials, linear models) to make the underlying essence accessible.
- The unified perspective of **soft inductive biases** is highly elegant: moving from restricting the hypothesis space to preferring simpler solutions within a flexible space.
- Clearly pinpoints what is **truly unique** about deep learning: representation learning (adaptive basis functions), mode connectivity, and universal learning capabilities (a single pre-trained model generalizing across modalities).
- The conceptual formulation of "expected risk $\leq$ empirical risk + compressibility" is simple yet powerful.
- The discussion on Residual Pathway Priors is highly illuminating: converting hard architectural constraints into soft inductive biases.

## Limitations & Future Work

- As a position paper, it does not propose new methods or novel experiments, primarily synthesizing existing works.
- It offers only intuitive explanations rather than a rigorous answer to the core question: "Why do larger models have a stronger compression bias?"
- The practical impact of the incomputability of Kolmogorov complexity is not discussed in enough depth.
- It does not sufficiently address perspectives from other theoretical frameworks, such as Neural Tangent Kernel (NTK) and mean-field theory.
- Concepts like grokking and scaling laws are only briefly mentioned without in-depth analysis.
- The connection between the computational efficiency advantages of deep learning (why SGD is sufficient in practice) and theoretical analysis is left unexpanded.

## Related Work & Insights

- "Understanding Deep Learning Requires Rethinking Generalization" by Zhang et al. (2016, 2021) is the primary dialogue partner of this paper.
- The work by Lotfi et al. (2022a, 2024b) on non-vacuous PAC-Bayes bounds provides the core technical support.
- The findings by Goldblum et al. (2024) regarding large models favoring low Kolmogorov complexity are critical.
- Residual Pathway Priors (Finzi et al., 2021) provide a practical mechanism for converting hard constraints into soft constraints.
- **Implications for self-supervised learning**: Self-supervised objectives essentially construct soft inductive biases, guiding the model to learn compressible representations.
- **Practical implications for model design**: Instead of pursuing architectures with hard constraints tailored to specific problems, embracing flexible architectures combined with appropriate soft preferences may serve as a better general-purpose strategy.

## Rating

- **Novelty**: ⭐⭐⭐ — Primarily synthesizes existing insights; the core point is not proposed for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐ — Illustrative examples are adequate, but it lacks novel experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear exposition, rigorous logic, beautiful illustrations, and highly educational.
- **Value**: ⭐⭐⭐⭐ — Makes significant contributions to clarifying common misconceptions in generalization theory, benefiting community progress.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Samples Are Not Equal: A Sample Selection Approach for Deep Clustering](../../ICLR2026/self_supervised/samples_are_not_equal_a_sample_selection_approach_for_deep_clustering.md)
- [\[ICCV 2025\] To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models](../../ICCV2025/self_supervised/to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie.md)
- [\[CVPR 2025\] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](../../CVPR2025/self_supervised/boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)
- [\[CVPR 2025\] ScaleLSD: Scalable Deep Line Segment Detection Streamlined](../../CVPR2025/self_supervised/scalelsd_scalable_deep_line_segment_detection_streamlined.md)
- [\[ICML 2025\] MTL-UE: Learning to Learn Nothing for Multi-Task Learning](mtl-ue_learning_to_learn_nothing_for_multi-task_learning.md)

</div>

<!-- RELATED:END -->
