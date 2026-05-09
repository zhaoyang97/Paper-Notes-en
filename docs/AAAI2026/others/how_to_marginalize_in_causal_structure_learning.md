---
title: >-
  [Paper Note] How to Marginalize in Causal Structure Learning?
description: >-
  [AAAI 2026][Bayesian structure learning] This paper employs tractable Probabilistic Circuits (PCs) as a replacement for traditional dynamic programming to perform marginalization in Bayesian structure learning. Through a novel two-stage training strategy—first learning full parent set scores and then progressively fine-tuning for marginal queries—the method eliminates the artificial restriction on the number of candidate parent sets, achieving improved posterior distribution estimation within the TRUST framework.
tags:
  - AAAI 2026
  - Bayesian structure learning
  - probabilistic circuits
  - marginalization
  - directed acyclic graphs
  - causal discovery
date: 2026-05-08
content_hash: 1eedb4ef736e5564
---

# How to Marginalize in Causal Structure Learning?

**Conference**: AAAI 2026
**arXiv**: [2511.14001](https://arxiv.org/abs/2511.14001)
**Code**: None
**Area**: Other
**Keywords**: Bayesian structure learning, probabilistic circuits, marginalization, directed acyclic graphs, causal discovery

## TL;DR

This paper employs tractable Probabilistic Circuits (PCs) as a replacement for traditional dynamic programming to perform marginalization in Bayesian structure learning. Through a novel two-stage training strategy—first learning full parent set scores and then progressively fine-tuning for marginal queries—the method eliminates the artificial restriction on the number of candidate parent sets, achieving improved posterior distribution estimation within the TRUST framework.

## Background & Motivation

Bayesian Networks are an important class of probabilistic graphical models widely applied in healthcare modeling, industrial fault diagnosis, and related domains. Their central challenge lies in **structure learning**—inferring the directed acyclic graph (DAG) structure among variables from observational data. Since the number of DAGs grows super-exponentially with the number of variables, Bayesian structure learning approaches address this uncertainty by inferring the posterior distribution $p(G|D)$.

A fundamental subtask in Bayesian structure learning is **marginalizing** the Bayesian scoring function $\mathcal{B}$ over all possible parent sets. Existing methods rely on dynamic programming with time complexity $O(3^N)$, which becomes infeasible when $N > 16$. Consequently, existing methods are forced to **artificially restrict the size of candidate parent sets** (e.g., to 8 candidates), which directly reduces search space coverage and yields inaccurate posterior distributions.

The root cause is the tension between exact marginalization requiring exponential computation and information loss incurred by restricting candidate sets. This paper's starting point is to exploit the **decomposability and smoothness** of probabilistic circuits, which naturally support exact marginalization, using PCs as surrogate models for the Bayesian scoring function. A trained PC can answer arbitrary marginal queries in linear time, bypassing the $O(3^N)$ computational bottleneck while retaining support over all parent sets.

## Method

### Overall Architecture

The proposed method independently learns a probabilistic circuit for each variable $X_i$ in the Bayesian network to approximate the Bayesian scoring function $\mathcal{B}_{X_i}(\mathbf{S})$. After training, the structure learner no longer calls dynamic programming but directly uses the PC to answer marginal queries. The method consists of three components: PC architecture design, the two-stage learning algorithm, and integration with the TRUST framework.

### Key Designs

1. **Probabilistic Circuit (PC) Architecture Design**:

    - Function: Construct a decomposable and smooth PC to represent an unnormalized probability distribution.
    - Mechanism: A RAT-SPN-like architecture is adopted, parameterized by latent dimension $N$ and variable count $M$. Leaf nodes are unnormalized Bernoulli distributions; intermediate layers alternate between product layers and sum layers. Product layers double the scope (halving the number of matrix rows), ensuring **decomposability** ($sc(C_1) \cap sc(C_2) = \emptyset$). Sum layers perform weighted summation over nodes in the same row, ensuring **smoothness** ($sc(C_1) = sc(C_2)$).
    - Design Motivation: Decomposability combined with smoothness guarantees that marginalization can be performed exactly in time linear in the circuit size. Unlike conventional PCs, the sum node weights are not constrained to be normalized ($\sum w_i \neq 1$), since Bayesian scores are themselves unnormalized.

2. **Two-Stage Learning Algorithm**:

    - Function: Through progressive training, the PC learns to approximate both the Bayesian scoring function and its marginal distributions.
    - Mechanism:
        - **Stage 1 (Baseline Learning)**: Full parent set vectors are randomly sampled from $\mathbf{S}$, with Bayesian scores $\mathcal{B}$ used as labels and MSE loss for training. Sampling is weighted $\propto 2^{M-T}$, biasing toward vectors with fewer 1s, as these contribute more to marginal quality.
        - **Stage 2 (Marginal Fine-tuning)**: The number of marginalized variables $k$ is incrementally increased; at each step, both $(k,0)$ and $(k,1)$ marginal queries are trained alongside newly sampled full parent sets. Marginal labels are generated by exact DP over a restricted candidate set.
    - Design Motivation: The progressive training exploits the recurrence $g(A_i, A'_i) = g(A_i \cup b, A'_i) + g(A_i, A'_i \cup b)$, i.e., a $(k+1, 0)$ query decomposes into the sum of $(k, 0)$ and $(k, 1)$ queries, enabling natural generalization from order-$k$ to order-$(k+1)$ marginals.

3. **Key Training Details**:

    - Function: Ensure training stability and convergence quality.
    - Mechanism: Stage 1 uses a relatively high learning rate of $10^{-1}$ to mitigate vanishing gradients; parameters are initialized as $m \cdot \log(U(0,1))$ with $m \approx -10$ to break gradient symmetry; Stage 2 reduces the learning rate to $5 \times 10^{-3}$ and limits the number of iterations $L$ to prevent catastrophic forgetting.
    - Design Motivation: Unnormalized weights can take large negative values in log space, causing vanishing gradients; the high learning rate and specialized initialization directly address this issue.

### Loss & Training

The loss function is log-domain MSE: $(\log p(\mathbf{S}) - \mathcal{B}(\mathbf{S}))^2$. MSE is preferred over KL divergence based on empirical performance and consistency with analogous marginal learning models. Stage 1 uses 10,000 samples; Stage 2 uses 20,000 samples per iteration (half marginal, half full scores) over 20 epochs per iteration.

## Key Experimental Results

### Main Results

Experiments are conducted within the TRUST framework using synthetic data (Erdős-Rényi graphs, average 2 edges/node, linear Gaussian mechanisms, BGe score), comparing the PC method against dynamic programming (DP).

| Setting | Method | AUROC | MLL | MSE-CE | E-SHD |
|---------|--------|-------|-----|--------|-------|
| $d=16$, unlimited candidates | DP (exact) | baseline | baseline | baseline | baseline |
| $d=16$, unlimited candidates | PC (Ours) | ≈DP | ≈DP | ≈DP | slightly worse |
| $d=20$, candidates=8 | DP (restricted) | baseline | baseline | baseline | baseline |
| $d=20$, candidates=8 | PC (Ours) | **better than DP** | **better than DP** | **better than DP** | slightly worse |

### Ablation Study

| Configuration | Key Performance | Notes |
|---------------|----------------|-------|
| $d=16$, PC vs. exact DP | Competitive on 3 of 4 metrics | Validates PC approximation of exact marginalization |
| $d=20$, PC vs. restricted DP | Significantly better on 3 of 4 metrics | Gain from eliminating candidate set restriction |
| Latent size $N=256$ vs. $N=64$ | 256 for $d=16$; 64 for $d=20$ | Higher dimensionality allows more compact PCs |
| Without Stage 2 fine-tuning | Stage 1 only | Marginal query accuracy degrades significantly |

### Key Findings

- The PC method can approximate exact marginalization when the candidate set is unrestricted, with negligible impact on downstream structure learning.
- When DP's candidate set is restricted (standard practice in real applications), the PC method significantly outperforms DP on AUROC, MLL, and MSE-CE.
- The PC method tends to sample denser graph structures, leading to higher E-SHD, yet AUROC results confirm that the learned posterior correctly identifies edge probabilities.
- Progressively increasing the number of marginalized variables $k$ during Stage 2 training is critical for learning quality.

## Highlights & Insights

- The inherent tractability of probabilistic circuits (exact marginalization in linear time) aligns naturally with the marginalization demands of structure learning, making this a methodologically well-motivated choice.
- The two-stage progressive training exploits the recurrence structure of marginal queries, grounding the learning of difficult high-order marginals on well-learned low-order ones—an approach that is both elegant and practically effective.
- Implementing the entire TRUST structure learner on top of probabilistic circuits (posterior distribution + score marginalization) demonstrates the potential of PCs as a unified framework.
- The core technique of bypassing the exponential bottleneck through approximation lies in learning a surrogate model that itself supports exact inference.

## Limitations & Future Work

- Validation is currently limited to the TRUST framework and linear Gaussian Bayesian networks; extensions to discrete BNs (e.g., BDeu scoring) and other structure learning methods (e.g., ArCO-GP) are needed.
- Experiments reach at most $d=20$; scalability to larger dimensions remains to be verified.
- Performance on E-SHD is inferior to DP, indicating a systematic bias in the PC-learned distribution regarding edge density.
- Only synthetic data are used; validation on real-world datasets is absent.
- The number of iterations $L$ in Stage 2 exhibits a "forgetting effect," where excessive iterations degrade performance; a better theoretical understanding of this trade-off is warranted.

## Related Work & Insights

- Unlike MAMs, NADEs, and AO-ARMs, probabilistic circuits provide **exact** rather than approximate marginal inference, which is critical in structure learning.
- The proposed approach can be viewed as a "approximate the global distribution first, then perform exact inference" paradigm, analogous in spirit to performing variational inference with neural networks followed by sampling.
- Insight for causal discovery: computational bottlenecks can be addressed by learning tractable surrogate models rather than forcibly restricting the search space.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)
- [\[ICLR 2026\] CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](../../ICLR2026/others/chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)
- [\[AAAI 2026\] Structure-Aware Encodings of Argumentation Properties for Clique-width](structure-aware_encodings_of_argumentation_properties_for_clique-width.md)
- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](../../ICLR2026/others/learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)

<!-- RELATED:END -->
