---
title: >-
  [Paper Note] Generalization in Federated Learning: A Conditional Mutual Information Framework
description: >-
  [ICML 2025][AI Safety][Federated Learning] A generalization analysis framework for Federated Learning based on Conditional Mutual Information (CMI) is proposed, which unifies the characterization of the participation gap and the out-of-sample gap for the first time, and reveals the intrinsic connection between differential privacy and generalization.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Federated Learning"
  - "Generalization Theory"
  - "Conditional Mutual Information"
  - "Privacy"
  - "Information Theory"
date: 2026-05-08
content_hash: 569ad24d351e335f
---

# Generalization in Federated Learning: A Conditional Mutual Information Framework

**Conference**: ICML 2025  
**arXiv**: [2503.04091](https://arxiv.org/abs/2503.04091)  
**Code**: None  
**Area**: AI Safety / Federated Learning  
**Keywords**: Federated Learning, Generalization Theory, Conditional Mutual Information, Privacy, Information Theory

## TL;DR

A generalization analysis framework for Federated Learning based on Conditional Mutual Information (CMI) is proposed, which unifies the characterization of the participation gap and the out-of-sample gap for the first time, and reveals the intrinsic connection between differential privacy and generalization.

## Background & Motivation

**Background**: Generalization analysis in federated learning (FL) is a core problem for understanding model reliability. Classical statistical learning theory primarily focuses on out-of-sample generalization (from training set to the population), but the unique architecture of FL introduces a second-level generalization problem—the participation gap, which refers to whether the subset of clients participating in training can represent all potential clients.

**Limitations of Prior Work**: Most existing generalization analyses for FL focus solely on the out-of-sample gap while ignoring the participation gap. Alternatively, they only provide bounds for specific algorithms (e.g., FedAvg), lacking a unified, algorithm-independent theoretical framework. Bounds derived from PAC-Bayes and traditional mutual information methods in FL contexts are often overly loose (vacuous).

**Key Challenge**: The two-level randomness of FL (client sampling + sample sampling) makes traditional single-level generalization theories not directly applicable. This necessitates a unified framework capable of simultaneously capturing the uncertainties at both levels.

**Goal**: To establish generalization bounds applicable to general FL algorithms, unify the analysis of the participation gap and the out-of-sample gap, and provide computable, non-vacuous bounds.

**Key Insight**: Draw inspiration from the Conditional Mutual Information (CMI) framework, which has been successfully applied to centralized learning recently, and extend it to the hierarchical structure of FL.

**Core Idea**: By constructing "superclients" and "supersamples", FL generalization is decomposed into two CMI terms: client participation and sample membership, which control the generalization errors at the respective levels.

## Method

### Overall Architecture

This framework decomposes the FL generalization error into two independently controllable terms. Suppose $K$ clients participate in training, with each client having $n$ samples, and the model output is $W$. The "superclient" construction is introduced: embedding the participating client set $V$ into a super-collection containing $2K$ clients, where $K$ are selected ($V_j = 1$) and $K$ are not selected ($V_j = 0$). Similarly, the "supersample" construction is introduced: each client possesses $2n$ samples, where $n$ are used for training ($U_{j,i} = 1$) and $n$ are used for testing ($U_{j,i} = 0$). Based on this, the generalization error is decomposed into $I(W; V | Z, U)$ (the CMI corresponding to the participation gap) and $I(W; U | Z, V)$ (the CMI corresponding to the out-of-sample gap), where $Z$ represents the complete dataset.

### Key Designs

1. **Superclient & Supersample**:

    - **Function**: Embeds the two-level randomness of FL into a symmetrized probabilistic structure, enabling CMI analysis.
    - **Mechanism**: For the participation gap, a super-collection of $2K$ clients is constructed, using a binary selection vector $V \in \{0,1\}^{2K}$ to denote which clients participate in training. For the out-of-sample gap, a super-collection of $2n$ samples is similarly constructed for each client. This symmetrical construction ensures that selected and unselected items are exchangeable under the marginal distribution.
    - **Design Motivation**: The core of the centralized CMI framework lies in the symmetry of the "supersample". This work extends this idea to the client level. Symmetry ensures that conditional mutual information can serve as an effective measure of the generalization gap.

2. **Two-level CMI Decomposition**:

    - **Function**: Accurately decomposes the total generalization error into independent contributions from the participation gap and the out-of-sample gap.
    - **Mechanism**: Utilizing the law of total probability and the data processing inequality, the total generalization gap $\leq$ $\sqrt{2\sigma^2 I(W; V | Z, U) / K}$ + $\sqrt{2\sigma^2 I(W; U | Z, V) / (Kn)}$. The two terms are controlled by the sensitivity of the model to client selection and sample selection, respectively.
    - **Design Motivation**: This decomposition allows the independent analysis of generalization behavior at each level, and reveals that under certain conditions (such as Bregman loss with model averaging), each level can achieve fast convergence rates individually.

3. **Privacy-Generalization Connection**:

    - **Function**: Establishes a quantitative relationship between differential privacy guarantees and generalization bounds.
    - **Mechanism**: If an FL algorithm satisfies $(\epsilon, \delta)$-DP, the CMI can be upper-bounded using $\epsilon$ and $\delta$. Specifically, client-level DP constraints imply control over the participation gap, while sample-level DP constraints imply control over the out-of-sample gap.
    - **Design Motivation**: Privacy and generalization are inherently related to the stability of the algorithm against adjustments to a single data point. The CMI framework naturally unifies the two.

### Loss & Training

This is a purely theoretical paper and does not involve specific training strategies. However, the theoretical analysis covers multiple loss function classes: (1) standard bounds under $\sigma$-sub-Gaussian losses; (2) fast-rate bounds of $O(1/K + 1/(Kn))$ under Bregman loss + model averaging; (3) ultra-fast rate bounds under smooth + strongly convex losses, where the out-of-sample gap can reach $O(1/(K^3n))$. The theoretical results are applicable to any FL algorithm, including FedAvg, FedProx, SCAFFOLD, etc.

## Key Experimental Results

### Main Results

| Dataset | Clients $K$ | Samples $n$ | CMI Participation Bound | CMI Out-of-sample Bound | Empirical Gen. Gap | Bound Tightness |
|:--|:--|:--|:--|:--|:--|:--|
| MNIST | 10 | 500 | 0.038 | 0.025 | 0.031 | Non-vacuous & close |
| MNIST | 50 | 100 | 0.021 | 0.042 | 0.035 | Effectively captures trend |
| CIFAR-10 | 10 | 500 | 0.089 | 0.067 | 0.072 | Reasonable upper bound |
| CIFAR-10 | 50 | 100 | 0.052 | 0.091 | 0.078 | Effectively captures behavior |

*The CMI bounds are non-vacuous across all settings and effectively track the trends of the empirical generalization gap.*

### Ablation Study

| Analysis Condition | Participation Gap Rate | Out-of-sample Gap Rate | Overall Rate | Description |
|:--|:--|:--|:--|:--|
| Sub-Gaussian Loss (Standard) | $O(1/\sqrt{K})$ | $O(1/\sqrt{Kn})$ | $O(1/\sqrt{K})$ | Standard slow rate |
| Bregman Loss + Model Averaging | $O(1/K)$ | $O(1/(Kn))$ | $O(1/K)$ | Fast rate |
| Smooth + Strongly Convex Loss | $O(1/K)$ | $O(1/(K^3n))$ | $O(1/K)$ | Ultra-fast rate |
| Client-level DP ($\epsilon$-DP) | $O(\epsilon/\sqrt{K})$ | — | — | Privacy implies generalization |

### Key Findings

- The CMI framework provides non-vacuous generalization bounds in FL for the first time, whereas traditional MI and PAC-Bayes methods typically yield vacuous bounds under the same settings.
- The decay rates of the participation gap and the out-of-sample gap are different: the former depends on $K$ (number of clients), while the latter depends on $K \times n$ (total sample size).
- Under the condition of Bregman loss with model averaging, both gaps simultaneously achieve the fast rate of $O(1/K)$, which is strictly tighter than previously known bounds.
- Differential privacy constraints naturally imply generalization guarantees, providing theoretical support for the "privacy free lunch."

## Highlights & Insights

- Unified and elegant theoretical framework: Through the construction of superclients/supersamples, the unique two-level randomness of FL is incorporated into the standard CMI analysis framework.
- Formally defines and quantifies the participation gap in FL for the first time, filling a theoretical gap.
- Practical implications of the privacy-generalization connection: FL systems employing privacy-preserving mechanisms like DP-SGD automatically obtain generalization guarantees.
- Fine-grained analysis across different loss function categories reveals fast-rate conditions, providing theoretical guidance for selecting loss functions in practice.

## Limitations & Future Work

- Numerical validation is performed only on MNIST and CIFAR-10; validation on more complex models and datasets (e.g., federated fine-tuning of LLMs) is missing.
- Fast-rate results depend on strong conditions like Bregman loss and model averaging, which may not naturally hold in deep learning.
- Calculating CMI remains challenging in large-scale scenarios, and the scalability of the estimation methods used in the paper needs improvement.
- The impact of client data heterogeneity (non-IID level) on the bounds is not fully discussed.

## Related Work & Insights

- **vs. Classical Mutual Information Bounds (Xu & Raginsky)**: CMI eliminates data dependence through conditioning, preventing standard MI bounds from loosening on large datasets. The advantage is even more pronounced in the FL context.
- **vs. PAC-Bayes FL Bounds**: PAC-Bayes methods require specifying a prior distribution and are complex to derive under the two-level structure of FL. The CMI framework is more natural and yields tighter bounds.
- **vs. Algorithmic Stability (Bousquet & Elisseeff)**: Stability analysis is typically bound to a specific algorithm (e.g., SGD), whereas the CMI framework is algorithm-independent and applicable to any FL scheme.

## Rating

- Novelty: ⭐⭐⭐⭐ Extends the CMI framework systematically to FL for the first time; the superclient construction and two-level decomposition are significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Experimental validation of the theoretical work is generally sufficient, but the datasets and model scales are relatively small.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations, consistent notation system, and clear logic.
- Value: ⭐⭐⭐⭐ Fills the gap in the analysis of the participation gap in FL generalization theory; the privacy-generalization connection offers practical guiding significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)
- [\[ICLR 2026\] Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information](../../ICLR2026/ai_safety/why_do_unlearnable_examples_work_a_novel_perspective_of_mutual_information.md)
- [\[ICML 2025\] Disparate Conditional Prediction in Multiclass Classifiers](disparate_conditional_prediction_in_multiclass_classifiers.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](../../CVPR2026/ai_safety/proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
