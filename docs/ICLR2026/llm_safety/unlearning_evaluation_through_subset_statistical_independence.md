---
title: >-
  [Paper Note] Unlearning Evaluation through Subset Statistical Independence
description: >-
  [LLM Safety] This paper proposes Split-half Dependence Evaluation (SDE), which leverages HSIC-based statistical independence testing to evaluate machine unlearning at the subset level…
tags:
  - "LLM Safety"
date: 2026-05-08
content_hash: cfdda2682f20bad1
---

# Unlearning Evaluation through Subset Statistical Independence

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2603.00587](https://arxiv.org/abs/2603.00587)
- **Code**: [https://github.com/ChildEden/SDE](https://github.com/ChildEden/SDE)
- **Area**: Machine Unlearning / Privacy Protection / Statistical Testing
- **Keywords**: Machine unlearning evaluation, HSIC, statistical independence, subset-level evaluation, membership inference

## TL;DR
This paper proposes Split-half Dependence Evaluation (SDE), which leverages HSIC-based statistical independence testing to evaluate machine unlearning at the subset level, requiring neither model retraining nor auxiliary classifiers.

## Background & Motivation

### Core Problem
How can one verify whether a machine unlearning procedure has succeeded? Existing evaluation methods exhibit fundamental limitations:

**Retraining Comparison**: Requires training a new reference model — contradicting the original motivation for unlearning.

**Membership Inference Attacks (MIA)**: Rely on training statistics and shadow models — difficult to obtain after unlearning.

**Sample-level Inference**: Since unlearning removes only a small subset (5%–20%), per-sample signals are statistically weak after unlearning.

### Paradigm Shift
From **sample-level MIA** → **subset-level statistical independence evaluation**.

Core intuition: Training participation induces inter-sample dependencies in model outputs (via shared gradient updates and co-adaptation), whereas out-of-training data exhibits no such dependencies.

## Method

### Split-half Dependence Evaluation (SDE)

#### Core Idea

The subset under evaluation $\mathcal{S}$ is randomly split into two halves $\mathcal{S}_1, \mathcal{S}_2$, and the statistical dependence between model outputs is computed as:

$$H(\mathcal{S}, h) = \text{HSIC}(h(\mathcal{S}_1), h(\mathcal{S}_2))$$

- **In-training subsets**: $H(\mathcal{S}_{IT}, h)$ is significantly higher than
- **Out-of-training subsets**: $H(\mathcal{S}_{OOT}, h)$

#### HSIC (Hilbert-Schmidt Independence Criterion)

$$\text{HSIC}(X, Y) = \frac{1}{(n-1)^2}\text{Tr}(KHLH)$$

where $K, L$ are Gaussian RBF kernel matrices and $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix.

The distribution of $H(\mathcal{S}, h)$ is estimated via 200 random shuffles of $\mathcal{S}_2$.

#### Unlearning Evaluation Protocol

Given a target subset $\mathcal{S}_{\text{tar}} \subseteq \mathcal{D}_f$, a reference in-training set $\mathcal{S}_{IT} \subset \mathcal{D}_r$, and an out-of-training set $\mathcal{S}_{OOT} \subset \mathcal{D}_{te}$:

Unlearning is deemed successful if and only if:
$$D(\mathcal{S}_{\text{tar}}, \mathcal{S}_{OOT}, h^{un}) < D(\mathcal{S}_{\text{tar}}, \mathcal{S}_{IT}, h^{un})$$

where $D$ denotes the Jensen-Shannon divergence between dependence distributions.

### Theoretical Analysis

Shared influence components introduced during training cause in-training subsets to exhibit stronger split-half dependence. Specifically, when $h = \mathcal{A}(\mathcal{D}_{tr})$, the output $h(x_i)$ implicitly depends on $x_j$ through the learned parameters, so $h(x_i)$ and $h(x_j)$ are no longer independent.

## Experiments

### Controlled Experiments (Retrained Model)

| Dataset-Model | R=5% |S|=400 | R=10% |S|=1000 | R=20% |S|=2000 |
|------------|------|--------|--------|
| SV-ResNet18 | 0.71 | 0.78 | 0.97 |
| C10-ResNet18 | 0.87 | 0.95 | 1.00 |
| C100-ResNet18 | 0.99 | 1.00 | 1.00 |
| Tiny-ResNet18 | 0.70 | 0.92 | 0.98 |

### Comparison with Distribution Distance Metrics (CIFAR10-ResNet18, R=10%, |S|=1000)

| Method | F1 Score |
|------|---------|
| MMD | 0.70 |
| Wasserstein | 0.89 |
| **SDE (Ours)** | **0.95** |

SDE consistently outperforms MMD and Wasserstein across **all settings**, with a particularly pronounced advantage on smaller subsets.

### Evaluation of Unlearning Methods (CIFAR10-ResNet18, R=10%)

| Method | Acc_r(%) | Acc_f(%) | ASR | OTR↑(%) |
|------|----------|----------|-----|---------|
| Retrain | 98.57 | 93.25 | 0.30 | **87.00** |
| RandLabel | 98.80 | 98.63 | 0.29 | 84.00 |
| Unroll | 99.36 | 99.21 | 0.30 | **3.00** |
| Sparsity | 92.72 | 90.56 | 0.42 | 50.80 |
| SalUn | 98.66 | 98.53 | 0.29 | 52.40 |

### Key Findings

1. **Critical finding on Unroll**: Conventional metrics (ASR ≈ 0.30, consistent with retraining) indicate successful unlearning, yet SDE yields an OTR of only 3% — nearly all forget samples are still identified as in-training data.
2. **SDE exposes the insufficiency of MIA**: Similar ASR values make it difficult to distinguish unlearning quality, whereas OTR provides clearer differentiation.
3. Larger subsets and deeper-layer features yield better discriminative power.
4. Kernel bandwidth $\sigma = \sqrt{\text{dim}}$ serves as a robust heuristic.
5. Dependencies can be detected even in models trained for only 20% of the full training schedule.

## Highlights & Insights

1. **Retraining-free independent evaluation**: A genuinely independent verification framework for machine unlearning.
2. **Subset-level evaluation aligned with the unlearning workflow**: Unlearning is inherently a subset-level operation.
3. **Exposing blind spots in existing evaluation**: The Unroll case serves as a cautionary example.
4. **Unified theory and practice**: The shared influence component analysis provides theoretical grounding for the method design.

## Limitations & Future Work

1. The choice of kernel bandwidth $\sigma$ has a notable impact; the simple heuristic may not generalize to all scenarios (e.g., diffusion models).
2. Reference set selection affects performance, and an optimal reference set construction strategy remains unresolved.
3. The method may capture natural forgetting (representation drift, catastrophic forgetting) rather than intentional unlearning.
4. The current framework yields only binary judgments, leaving the potential of HSIC as a continuous metric underexplored.
5. Performance is weaker on shallow architectures such as AllCNN.

## Related Work & Insights
- **Machine Unlearning**: SISA, Random-label, SalUn — representative unlearning algorithms.
- **Membership Inference Attacks**: Confidence-based, loss-based, and auxiliary-classifier-based methods.
- **Statistical Independence Testing**: HSIC, MMD — kernel-based statistical tests.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Subset-level statistical independence evaluation is a novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dimensional controlled experiments and evaluation across unlearning methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and complete method description.
- **Value**: ⭐⭐⭐⭐ — No additional training required; easy to deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](../../AAAI2026/llm_safety/multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](../../ACL2026/llm_safety/cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[ICML 2026\] FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning](../../ICML2026/llm_safety/fedtreelora_reconciling_statistical_and_functional_heterogeneity_in_federated_lo.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](../../NeurIPS2025/llm_safety/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)

</div>

<!-- RELATED:END -->
