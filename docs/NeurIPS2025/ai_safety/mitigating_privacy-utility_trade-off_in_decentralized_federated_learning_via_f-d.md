---
title: >-
  [Paper Note] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy
description: >-
  [NeurIPS 2025][AI Safety][Decentralized Federated Learning] This paper proposes two privacy accounting methods for decentralized federated learning under the f-DP framework—PN-f-DP and Sec-f-LDP—which leverage hypothesis…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Decentralized Federated Learning"
  - "f-Differential Privacy"
  - "Privacy Amplification"
  - "Random Walk"
  - "Correlated Noise"
date: 2026-05-08
content_hash: be595c7d748a8102
---

# Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy

**Conference**: NeurIPS 2025
**arXiv**: [2510.19934](https://arxiv.org/abs/2510.19934)
**Code**: [https://github.com/lx10077/PN-f-DP](https://github.com/lx10077/PN-f-DP)
**Area**: AI Safety / Federated Learning
**Keywords**: Decentralized Federated Learning, f-Differential Privacy, Privacy Amplification, Random Walk, Correlated Noise

## TL;DR
This paper proposes two privacy accounting methods for decentralized federated learning under the f-DP framework—PN-f-DP and Sec-f-LDP—which leverage hypothesis-testing-based privacy measures to consistently yield tighter privacy bounds than Rényi DP, thereby reducing noise injection and improving model utility under equivalent privacy guarantees.

## Background & Motivation

Decentralized Federated Learning (Decentralized FL) enables users to collaboratively train models without a central server, yet model updates may still leak sensitive information. Differential privacy (DP) serves as the standard formal privacy protection framework, and the central challenge lies in **privacy accounting**—accurately quantifying cumulative privacy loss across multiple training rounds. More precise accounting translates directly into less noise injection under the same privacy guarantee, thereby improving model utility.

Existing privacy analyses for decentralized FL primarily rely on **Rényi DP (RDP)**, which tends to produce loose bounds for iterative algorithms. Additionally, decentralized settings involve multiple privacy amplification mechanisms (communication sparsity, local iterations, random walks), and capturing these amplification effects in a unified manner remains a key challenge.

The core idea of this paper is to replace RDP with the **f-DP framework** (a hypothesis-testing-based privacy measure) for privacy accounting in decentralized FL. The advantages of f-DP lie in its lossless composition property and the joint concavity of mixed mechanisms, enabling tighter capture of privacy amplification arising from decentralized communication, local iterations, and random walks.

## Method

### Overall Architecture

Two representative decentralized DP-SGD algorithms are analyzed with corresponding f-DP privacy guarantees:
- **Algorithm 1 (Decentralized DP-SGD with random-walk communication)**: Analyzed via PN-f-DP (Pairwise Network f-DP)
- **Algorithm 2 (DecoR with correlated noise)**: Analyzed via Sec-f-LDP (Secret-based f-Local DP)

### Key Designs

1. **PN-f-DP (Pairwise Network f-DP, Definition 3.1/3.2)**:

    - Quantifies privacy leakage between pairs of users. The core idea is that in a decentralized network, an adversary $j$ can only observe the model state transmitted to $j$ at specific time steps, rather than all intermediate states.
    - The privacy leakage from user $i$ to user $j$ is characterized by a lower bound on the trade-off function $f_{ij}$.
    - Supports both user-level (entire dataset difference) and record-level (single data point difference) granularities.

2. **Mixture Distribution Analysis for Random Walks (Lemma 4.1)**:

    - The model propagates from user $i$ to user $j$ via a random walk; the time $t$ at which $j$ first observes the model follows a hitting time distribution.
    - By exploiting the joint concavity of trade-off functions, the overall privacy loss is lower-bounded by a convex combination of per-timestep trade-offs.
    - **Key improvement**: The hitting time distribution $w_{ij}^t = P[\tau_{ij}=t]$ is used as the mixing weight rather than the matrix power $(W^t)_{ij}$ used in prior work, more accurately modeling communication timing.

3. **Iterative Privacy Amplification (Lemma 4.2/4.3)**:

    - Strongly convex loss: Using the shifted interpolated process technique, a GDP bound $f_{ij}^t \geq G_{\mu_t}$ is derived, where $\mu_t$ decays with iteration $t$ and contraction factor $c$.
    - Non-convex loss: Standard composition rules yield a more conservative but still valid bound.
    - Extension to $K \geq 1$ local updates goes beyond prior work that only analyzes $K=1$.

4. **Final Composition (Theorem 4.1)**: A Hoeffding-type concentration inequality for Markov chains is used to bound the number of visits to user $j$ with high probability by $\lceil(1+\zeta)T/n\rceil$, followed by composing the per-visit trade-off functions via tensor product.

5. **Sec-f-LDP (Theorem 4.2)**: For the DecoR algorithm, user pairs share a secret and inject correlated noise. Under the honest-but-curious threat model, the combined effect of correlated and independent noise is analyzed to yield a SecGDP guarantee, where $\mu$ depends on the second largest eigenvalue of the graph Laplacian matrix.

### Loss & Training

During DP-SGD training, gradients are clipped (with sensitivity $\Delta$) and Gaussian noise is added at each step. This paper does not alter the training procedure; instead, tighter privacy accounting is used to **reduce the required noise**: when achieving the same $(\varepsilon, \delta)$-DP guarantee, the f-DP method yields a smaller required noise variance $\sigma^2$.

## Key Experimental Results

### Main Results

| Dataset / Graph Structure | $\varepsilon$ | RDP Accuracy | RDP (HT) Accuracy | PN-f-DP (Ours) | Gain |
|--------------------------|--------------|-------------|-------------------|---------------|------|
| MNIST / Complete | 10 | 0.867 | 0.884 | **0.891** | +2.4% |
| MNIST / Complete | 5 | 0.813 | 0.852 | **0.869** | +5.6% |
| MNIST / Expander | 10 | 0.797 | 0.823 | **0.854** | +5.7% |
| MNIST / Expander | 5 | 0.647 | 0.706 | **0.792** | +14.5% |

### Ablation Study

| Privacy Accounting Method | Hypercube Graph $\varepsilon$ ($\delta=10^{-5}$) | Description |
|--------------------------|------------------------------------------------|-------------|
| PN-RDP | Largest | Standard Rényi DP baseline |
| PN-RDP (Exact HT) | Medium | Improved RDP using exact hitting time |
| PN-f-DP (Ours) | **Smallest** | f-DP consistently yields the tightest bound |

### Key Findings
- PN-f-DP yields smaller $\varepsilon$ values than PN-RDP for nearly all node pairs, with the gap widening when graph connectivity is sparser (e.g., the improvement is most pronounced on the Expander graph).
- The advantage of f-DP becomes more significant as the privacy budget $\varepsilon$ decreases (stricter privacy requirements), since noise reduction is more valuable in this regime.
- Under correlated noise settings, Sec-f-LDP outperforms SecLDP and Local DP on both Ring and 2D Torus topologies.

## Highlights & Insights
- **Unified Framework**: The hypothesis-testing perspective of f-DP provides a unified and tighter analytical tool for multiple privacy amplification mechanisms in decentralized FL.
- **Hitting Time Modeling**: Replacing matrix powers with the first-passage time distribution of random walks as mixture weights yields improved bounds even when converted back to RDP.
- **Lossless Composition**: f-DP composition via tensor product avoids the slack incurred in RDP composition.

## Limitations & Future Work
- The final bounds lack closed-form expressions and require numerical computation (via numerical composition methods).
- The analysis assumes that the transition matrix $W$ is irreducible, aperiodic, and symmetric, which may not hold for all practical network topologies.
- The current work focuses on theoretical analysis for convex/non-convex optimization; empirical validation on modern large-scale models (e.g., LLMs) remains to be explored.
- Whether the derived bounds are information-theoretically tight remains an open question.

## Related Work & Insights
- **vs. PN-RDP (Cyffers et al.)**: Both adopt the pairwise network framework, but this paper replaces RDP with f-DP, yielding tighter composition bounds and finer capture of privacy amplification.
- **vs. SecLDP (Allouah et al.)**: Extension to the f-DP framework yields a tighter GDP bound under an equivalent collusion model.
- **vs. Federated f-DP (Zheng et al.)**: PN-f-DP additionally captures privacy amplification arising from decentralization, random walks, and iterative communication.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically introduces the f-DP framework to decentralized FL and proposes two novel privacy notions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic graphs and real classification tasks, though at a relatively small scale.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous but notation-heavy, requiring a strong background in DP.
- Value: ⭐⭐⭐⭐ Provides superior tools for privacy analysis in decentralized FL; practical impact depends on the scalability of the framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sequentially Auditing Differential Privacy](sequentially_auditing_differential_privacy.md)
- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](multi-class_support_vector_machine_with_differential_privacy.md)
- [\[ICLR 2026\] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](../../ICLR2026/ai_safety/unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)
- [\[NeurIPS 2025\] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)

</div>

<!-- RELATED:END -->
