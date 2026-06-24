---
title: >-
  [Paper Note] Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs
description: >-
  [ICML2025][State Space Models] This paper theoretically proves that linear SSMs (such as S4/Mamba) cannot compute the parity function—even when input-dependent parameterization is allowed—unless the state transition matrix contains negative eigenvalues, providing a precise mathematical characterization of the expressivity bottleneck of SSMs.
tags:
  - "ICML2025"
  - "State Space Models"
  - "Parity"
  - "Expressivity"
  - "Negative Eigenvalues"
  - "Mamba"
date: 2026-05-08
content_hash: 5cbf1eaba84e53cb
---

# Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs

**Conference**: ICML2025  
**arXiv**: [2508.07395](https://arxiv.org/abs/2508.07395)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: State Space Models, Parity, Expressivity, Negative Eigenvalues, Mamba

## TL;DR
This paper theoretically proves that linear SSMs (such as S4/Mamba) cannot compute the parity function—even when input-dependent parameterization is allowed—unless the state transition matrix contains negative eigenvalues, providing a precise mathematical characterization of the expressivity bottleneck of SSMs.

## Background & Motivation

### Key Challenge

**Key Challenge**: Linear state space models such as S4 and Mamba have achieved performance competitive with Transformers in sequence modeling while offering more efficient inference. However, their expressivity boundaries are not as thoroughly investigated as those of Transformers.

### Parity as a Fundamental Test

Parity is one of the simplest non-trivial sequence computations: given a 0/1 sequence, it outputs whether the number of ones is odd or even. It serves as a touchstone for the expressivity of sequence models.

### Limitations of Prior Work

**Limitations of Prior Work**: It has been empirically observed that SSMs perform poorly on parity tasks, but theoretical explanations are lacking.

## Method

### Main Theorems
**Theorem 1**: Any linear SSM (including input-dependent variants like Mamba) cannot compute the parity function if all eigenvalues of the state transition matrix are non-negative.

**Theorem 2**: With the introduction of negative eigenvalues, SSMs can compute parity, but this requires an O(log n)-dimensional state space.

### Unified Input Dependence
The analysis encompasses all mainstream SSM variants:
- S4 (fixed parameters)
- S5 (diagonalized)
- Mamba (input-dependent B/C matrices)
- Fully input-dependent A matrix
It is proved that the impossibility of parity is independent of the parameterization method.

### Necessity of Negative Eigenvalues
Intuitive understanding: parity requires an "inversion" operation (odd to even or even to odd), whereas positive eigenvalues can only perform monotonic transformations. Negative eigenvalues provide the necessary sign-flipping capability.

### Comparison with RNNs
Standard RNNs (with non-linear activations) can easily compute parity because the non-linearity provides capability equivalent to negative eigenvalues.

## Key Experimental Results

### Parity Task Verification


### Main Results

| Model | Positive Eigenvalues | Negative Eigenvalues | Parity Accuracy |
|------|---------|---------|-------------|
| S4 (Standard) | All Positive | None | ~50% (random) |
| Mamba | Can Be Positive | None | ~50% |
| S4 + Negative Eigenvalues | Mixed | Yes | **100%** |
| RNN | N/A | N/A | 100% |

### Sequence Length vs. Expressivity


### Ablation Study

| Sequence Length | Required State Dimension (Negative Eigenvalues) |
|---------|---------------------|
| 16 | 4 |
| 64 | 6 |
| 256 | 8 |
| 1024 | 10 |

This follows the theoretical prediction of O(log n).

### Key Findings
1. All mainstream SSMs fail to compute parity under positive eigenvalues.
2. Negative eigenvalues are a necessary and sufficient condition.
3. The required state dimension grows logarithmically with sequence length.
4. Input-dependent parameterization cannot overcome sign constraints.
5. This explains why SSMs are systematically weaker than Transformers on tasks requiring "counting".

## Highlights & Insights

1. Extremely clean theoretical contribution—impossibility and possibility results for parity.
2. A unified analysis encompassing all mainstream SSM variants.
3. The discovery of "negative eigenvalues" has direct engineering implications (simply modifying initialization is sufficient).
4. The theoretical prediction (O(log n) state dimension) is precisely validated by experiments.
5. Provides a key theoretical benchmark for the expressivity debate between SSMs and Transformers.

## Limitations & Future Work

1. Parity is an extremely simplified task; the correlation with actual NLP/CV tasks remains to be established.
2. The theory only covers linear SSMs and does not apply to non-linear variants (such as certain modes of RWKV).
3. The impact of introducing negative eigenvalues on training stability is not discussed.
4. The distribution characteristics of eigenvalues in practical large-scale SSMs are not analyzed.
5. The relationship with other expressivity metrics (such as TC/WL tests) has not been established.

## Related Work & Insights

- Complementary to research on Transformer expressivity (e.g., Hahn 2020).
- Direct connections with SSM initialization research (e.g., HiPPO).
- Insight: The initialization of SSMs should include negative eigenvalues to ensure basic expressivity.

## Rating
- Novelty: 5.0/5 — Clean impossibility theorem
- Experimental Thoroughness: 4.0/5 — Theoretical focus but clear validation
- Writing Quality: 5.0/5 — Clear theorems and proofs
- Value: 5.0/5 — Fundamental guidance for SSM design

## Supplementary Analysis

### Physical Meaning of Negative Eigenvalues
Positive eigenvalues imply monotonic state decay/growth, while negative eigenvalues introduce oscillations (sign flipping). Parity is fundamentally a sign-flipping operation, and thus absolutely requires negative eigenvalues.

### Suggestions for Mamba Initialization
The HiPPO initialization of Mamba typically yields negative eigenvalues, but some simplified versions might remove them. The results of this study warn against removing negative eigenvalues.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Discrepancy Minimization in Input-Sparsity Time](discrepancy_minimization_in_input-sparsity_time.md)
- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](../../ACL2025/others/a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[ACL 2025\] Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems](../../ACL2025/others/hard_negative_mining_for_domain-specific_retrieval_in_enterprise_systems.md)
- [\[CVPR 2026\] Negative Binomial Variational Autoencoders for Overdispersed Latent Modeling](../../CVPR2026/others/negative_binomial_variational_autoencoders_for_overdispersed_latent_modeling.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](../../NeurIPS2025/others/fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)

</div>

<!-- RELATED:END -->
