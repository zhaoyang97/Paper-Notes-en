---
title: >-
  [Paper Note] PAC Learning with Improvements
description: >-
  [ICML 2025][Physics & Scientific Computing][PAC learning] This paper proposes the "PAC Learning with Improvements" framework: when agents can genuinely improve their features by at most $r$, conservative classifiers can achieve zero error (rendering a previously impossible goal in standard PAC learning possible). A finite VC dimension is shown to be neither necessary nor sufficient, revealing a fundamental separation of learning with improvements from both standard PAC learni…
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
  - "PAC learning"
  - "strategic improvement"
  - "sample complexity"
  - "VC dimension"
  - "intersection-closed"
  - "zero-error"
date: 2026-05-08
content_hash: ffc87ccecb53f15b
---

# PAC Learning with Improvements

**Conference**: ICML 2025  
**arXiv**: [2503.03184](https://arxiv.org/abs/2503.03184)  
**Code**: [https://github.com/idanattias/PAC-Learning-with-Improvements](https://github.com/idanattias/PAC-Learning-with-Improvements)  
**Area**: Learning Theory  
**Keywords**: PAC learning, strategic improvement, sample complexity, VC dimension, intersection-closed, zero-error

## TL;DR

This paper proposes the "PAC Learning with Improvements" framework: when agents can genuinely improve their features by at most $r$, conservative classifiers can achieve zero error (rendering a previously impossible goal in standard PAC learning possible). A finite VC dimension is shown to be neither necessary nor sufficient, revealing a fundamental separation of learning with improvements from both standard PAC learning and strategic classification.

## Background & Motivation

**Background**: The most fundamental lower bound in standard PAC learning is that, in almost any non-trivial setting, achieving an error of $\epsilon$ requires at least $1/\epsilon$ samples, and zero error is impossible over continuous distributions. The field of strategic classification focuses on the response behavior of agents to classifiers; intensive research has analyzed the distinction between gaming and improvement.

**Limitations of Prior Work**: Existing studies on strategic improvement mainly focus on how to incentivize agents to improve or how to maximize social welfare. However, none have analyzed how the improvement capability of agents affects learning from the fundamental theoretic perspective of learnability and sample complexity. Furthermore, prior work has overlooked that improvement can make zero error possible—a scenario that is impossible in both standard PAC learning and strategic classification.

**Key Challenge**: Standard PAC models assume passive and static data, while strategic classification assumes that agents disguise themselves (without genuine improvement). In reality, agent improvements are genuine (e.g., paying off high-interest debt to improve credit scores), and this genuine improvement fundamentally changes the characteristics of learnability.

**Goal**: Systematically understand how the improvement capability of agents affects learnability, sample complexity, and algorithm design.

**Key Insight**: Taking threshold learning as an example—with a true threshold $\theta$, if the learned threshold $\hat{\theta}$ satisfies $\theta \le \hat{\theta} \le \theta + r$, all agents classified as positive are indeed qualified (no false positives), and all truly qualified agents can be correctly classified through effort (false negatives are eliminated through improvement), thereby achieving zero error.

**Core Idea**: The improvement capability of agents allows the learner to only be "close enough," thereby achieving zero error, which is impossible in standard models, while altering the structural characteristics of learnability.

## Method

### Overall Architecture

Let $\mathcal{X}$ be the instance space, and let the improvement function $\Delta: \mathcal{X} \to 2^{\mathcal{X}}$ map each agent $x$ to the set of points $\Delta(x)$ they can improve to. Once the learner deploys a classifier $h$, an agent classified as negative will improve if they can reach the positive region through improvement. The loss function utilizes adversarial tie-breaking: $\text{Loss}(x; h, f^*) = \max_{x' \in \Delta_h(x)} \mathbb{I}[h(x') \ne f^*(x')]$, which favors conservative classifiers—labeling uncertain points as negative (since false negatives can be corrected by the agents themselves, whereas false positives are irreversible). The goal is to learn $h$ given $m$ samples such that $\text{Loss}_\mathcal{D}(h, f^*) \le \epsilon$.

### Key Designs

1. **Fundamental Separation from Standard PAC and Strategic Classification (Section 3)**:

    - **Function**: Establish the fundamental relationships between learning with improvements and existing learning paradigms.
    - **Mechanism**: **Theorem 3.1** proves that a finite VC dimension is neither necessary nor sufficient for PAC learnability with improvements. Non-necessity: If $\Delta(x) = \mathcal{X}$ (global improvement), any class with an infinite VC dimension can achieve zero error using only $O(\frac{1}{\epsilon}\ln\frac{1}{\delta})$ samples to find a positive instance and outputting a single-point classifier. Non-sufficiency: The union of two intervals (VC dimension of 4) is unlearnable under a specific family of improvement functions, with an expected error $\ge 25\%$—because conservative strategies cannot guarantee safety in non-intersection-closed classes. **Theorem 3.4** further proves that a finite SVC dimension (strategic VC dimension, the condition for strategic classification learnability) does not imply learnability with improvements.
    - **Design Motivation**: To reveal that learning with improvements is not a simple generalization of standard PAC or strategic classification, but rather a fundamentally distinct learning model.

2. **Closure Algorithm for Intersection-Closed Classes (Section 4.2)**:

    - **Function**: Provide a PAC learning algorithm with improvements for intersection-closed concept classes.
    - **Mechanism**: The closure algorithm returns the minimal hypothesis containing all positive samples: $h_S^c = \text{CLOS}_\mathcal{H}(\{x_i: y_i=1\})$. A key property is that $h_S^c \subseteq f^*$ (the positive region is a subset of the true concept's positive region), thus generating no false positives. Agents outside the positive region who can improve into the positive region do not contribute to the error. **Theorem 4.7** proves that any intersection-closed class with VC dimension $d$ is PAC learnable with improvements with $O(\frac{1}{\epsilon}(d + \log\frac{1}{\delta}))$ samples, holding for any improvement function and distribution. The improvement gain is $\mathbb{P}[x \in \text{IR}(h; f^*, \Delta)]$, which is the probability mass of the "improvement region".
    - **Design Motivation**: Intersection-closedness guarantees that the positive region of the closure algorithm is always a subset of the true concept, naturally yielding "conservative and safe" classification. Many natural concept classes (thresholds, hyperrectangles, intersections of halfspaces, k-CNF) are intersection-closed. **Theorem 4.8** conversely shows that non-intersection-closed classes cannot be properly learned under appropriate conditions.

3. **Zero-Error Learning of Geometric Concepts (Sections 4.1, 4.3)**:

    - **Function**: Quantitatively measure the error reduction brought by improvements.
    - **Mechanism**: **Thresholds** (Theorem 4.1): Closed balls with improvement radius $r$ under a uniform distribution on $[0,1]$ yield an error upper bound of $(\epsilon - r)_+$. Zero error is achieved when $\epsilon \le r$, with a sample complexity of $O(\frac{1}{\epsilon}\log\frac{1}{\delta})$ (same as standard PAC). **Hyperrectangles** (Theorem 4.6): Under $\ell_\infty$ radius $r$ improvements, the error upper bound is $(\epsilon - \mathbb{P}[\text{IR}])_+$; for a uniform distribution on $[0,1]^2$, the improvement gain $= 2r(l_1+l_2) + 4r^2$ (proportional to the perimeter of the rectangle). **Halfspaces** (Theorem 4.9): For $d$-dimensional homogeneous halfspaces under an angular improvement $r$, only $\tilde{O}((d+\log(1/\delta))/r)$ samples are required to achieve zero error.
    - **Design Motivation**: To progressively demonstrate how improvements reduce or even eliminate classification errors across different geometric structures, from one-dimensional thresholds to high-dimensional halfspaces.

### Loss & Training

The core of this work is the theoretical framework. The learner's design principle is "conservative"—biasing toward labeling uncertain points as negative. Specific strategies include: (1) outputting the rightmost consistent threshold (threshold learning); (2) the closure algorithm (intersection-closed classes); (3) the intersection of positive regions of consistent hypotheses (halfspaces). In graph models (Section 5), it is further proven that the conservative learner can achieve zero error by only observing labels of a dominating set of the positive subgraph, with a sample complexity of $O(n(\log n + \log(1/\delta))/(d_{\min}^+ + 1))$.

## Key Experimental Results

### Theoretical Results Comparison

| Concept Class | Standard PAC Error | PAC Error with Improvements | Zero-Error Condition | Sample Complexity |
|--------|--------------|----------------|-----------|-----------|
| Threshold (Uniform Dist.) | $\epsilon$ | $(\epsilon - r)_+$ | $\epsilon \le r$ | $O(\frac{1}{\epsilon}\log\frac{1}{\delta})$ |
| $d$-dim Hyperrectangles | $\epsilon$ | $(\epsilon - \text{IR})_+$ | $\text{IR} \ge \epsilon$ | $O(\frac{1}{\epsilon}(d + \log\frac{1}{\delta}))$ |
| $d$-dim Halfspaces | $\epsilon$ | 0 | — | $\tilde{O}(\frac{d + \log(1/\delta)}{r})$ |
| Graph Models | $\epsilon$ | 0 | — | $O(\frac{n \log n}{d_{\min}^+ + 1})$ |

### Learnability Separation of the Three Learning Models

| Property | Standard PAC | Strategic Classification | PAC with Improvements |
|------|---------|-----------|------------|
| Equivalent Condition for Learnability | Finite VC dimension | Finite SVC dimension | No known equivalent condition |
| Zero-Error Reachable | No (non-trivial classes) | No (generally) | Yes (e.g., thresholds, intersection-closed) |
| Infinite VC Dimension | Unlearnable | Depends on SVC | Possibly learnable |
| Union of Two Intervals | Learnable (VC=4) | Learnable (SVC≤4) | **Unlearnable** (Error $\ge 25\%$) |

### Key Findings

- The error reduction $(\epsilon - r)_+$ brought by improvements directly subtracts the improvement budget $r$—which is strictly tighter than the standard PAC upper bound of $\epsilon$.
- Conservative strategies (biasing toward negative classification) are inherently superior to aggressive strategies in the improvement setting—false negatives can be repaired by agents themselves, while false positives are irreversible.
- Non-intersection-closed classes (such as the union of two intervals) become harder to learn in the improvement setting, because false positives can "lure" agents to improve into incorrect regions.
- Empirical validation: On 3 real-world datasets + 1 synthetic dataset, the conservative model (SVM with high FP penalty) starts with a higher initial error, which then drops rapidly to zero as $r$ increases.

## Highlights & Insights

- **Feasibility of Zero Error**: This study is the first to prove in learning theory that non-trivial concept classes can be learned with zero error, provided that agents possess genuine improvement capabilities. This result is impossible in either standard PAC learning or strategic classification.
- **Counter-intuitive Sufficiency of Conservative over Aggressive**: In classic machine learning, being conservative usually implies higher false negative rates. However, in the improvement setting, false negatives can be repaired by the agents themselves—thereby shifting the foundational principles of algorithm design.
- **Intersection-Closedness as a Key Structure for Learnability**: Natural concept classes (thresholds, rectangles, intersections of halfspaces) happen to be intersection-closed, whereas non-intersection-closed classes (such as the union of intervals) become unlearnable under improvements—revealing a deep connection between the algebraic structures of concept classes and their learnability.

## Limitations & Future Work

- The study is restricted to the realizability setting; generalization to the agnostic setting remains an important open problem.
- The improvement function $\Delta$ is assumed to be known, whereas in practice, the improvement capability might need to be estimated from data.
- Adversarial tie-breaking might be overly pessimistic; relaxing this assumption could expand the range of learnable classes.
- The study primarily focuses on the information-theoretic aspect (sample complexity) without analyzing the computational efficiency of the algorithms.

## Related Work & Insights

- **vs. Strategic Classification (HMPW 2016; SVXY 2023)**: In strategic classification, agents "disguise" themselves (without genuine improvement), and learnability is characterized by the SVC dimension. In this paper, agents genuinely improve, leading to entirely different learnability characteristics. Theorem 3.4 proves that a finite SVC dimension does not imply learnability with improvements.
- **vs. Kleinberg & Raghavan (2020)**: This work pioneered the distinction between improvement and manipulation, but mainly focused on incentive design. This paper concentrates on the foundational challenges of learnability and sample complexity.
- **vs. Haghtalab et al. (2020)**: Their work analyzed learning with improving agents from a social welfare perspective, aiming to maximize true positives. This paper targets classification error, is more sensitive to false positives, and proves that both objectives can be optimized simultaneously.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Establishes the first PAC learning with improvements framework, and discovers the zero-error phenomenon along with the fundamental separations among the three models.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; validated the basic conclusions empirically on four datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Precise definitions, intuitive examples (loans, thresholds, rectangles), and clear proof structures.
- Value: ⭐⭐⭐⭐ Opens up new foundational directions for strategic ML and learning theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [\[ICML 2025\] Improving Memory Efficiency for Training KANs via Meta Learning](improving_memory_efficiency_for_training_kans_via_meta_learning.md)
- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](../../NeurIPS2025/physics/transfer_learning_beyond_the_standard_model.md)
- [\[ICML 2025\] Rethink the Role of Deep Learning towards Large-scale Quantum Systems](rethink_the_role_of_deep_learning_towards_large-scale_quantum_systems.md)
- [\[CVPR 2025\] KAC: Kolmogorov-Arnold Classifier for Continual Learning](../../CVPR2025/physics/kac_kolmogorov-arnold_classifier_for_continual_learning.md)

</div>

<!-- RELATED:END -->
