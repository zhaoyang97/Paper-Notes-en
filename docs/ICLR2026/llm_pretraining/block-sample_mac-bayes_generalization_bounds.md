---
title: >-
  [Paper Note] Block-Sample MAC-Bayes Generalization Bounds
description: >-
  [ICLR 2026][Pretraining][PAC-Bayes] This paper proposes block-sample MAC-Bayes (mean approximately correct) generalization bounds. By partitioning training data into $J$ blocks and replacing the global KL divergence with the sum of KL divergences conditioned on each block, the framework provides finite, meaningful generalization error bounds in scenarios
tags:
  - ICLR 2026
  - Pretraining
  - PAC-Bayes
  - MAC-Bayes
date: 2026-05-08
content_hash: 4d7a1f6b1177d542
---
# Block-Sample MAC-Bayes Generalization Bounds

**Conference**: ICLR2026  
**arXiv**: [2602.12605](https://arxiv.org/abs/2602.12605)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: PAC-Bayes, MAC-Bayes, Generalization bounds, Information theory, Block-samples

## TL;DR
This paper proposes block-sample MAC-Bayes (mean approximately correct) generalization bounds. By partitioning training data into $J$ blocks and replacing the global KL divergence with the sum of KL divergences conditioned on each block, the framework provides finite, meaningful generalization error bounds in scenarios where traditional PAC-Bayes bounds are vacuous (e.g., deterministic learning algorithms like mean estimation). It also demonstrates that high-probability versions of this bound are generally infeasible.

## Background & Motivation

### Background
The PAC-Bayes framework is an essential tool in statistical learning theory for bounding generalization error, utilizing the KL divergence between a learning algorithm's posterior $P_{W|S}$ and a prior $Q_W$ to bound the gap between empirical and population loss. Recently, PAC-Bayes bounds have regained attention for providing non-trivial generalization guarantees for deep neural networks. MAC-Bayes (mean approximately correct) is the expectation-based version of PAC-Bayes, bounding expected rather than high-probability generalization error.

### Limitations of Prior Work

**Failure of PAC-Bayes under Deterministic Algorithms**: When the learning algorithm $P_{W|S}$ is deterministic (e.g., $W = \frac{1}{n}\sum_i Z_i$), $P_{W|S}$ becomes a Dirac distribution. The KL divergence relative to any prior $Q_W$ becomes infinite, causing both PAC-Bayes and their corresponding MAC-Bayes bounds to be vacuous.

**Coarseness of Single KL Divergence Terms**: The global $D(P_{W|S} \| Q_W)$ summarizes the influence of the entire training set on the hypothesis into a single scalar. When this influence is excessively strong (as in deterministic algorithms), the bound explodes.

**Similar Limitations in Information-Theoretic Bounds**: Generalization bounds based on mutual information $I(W;S)$ suffer from analogous issues.

### Key Challenge
The divergence term $D(P_{W|S} \| Q_W)$ in PAC-Bayes measures the "informational influence" of the full training set $S$ on the hypothesis $W$. When this influence is strong, the bound fails. However, intuition suggests that if one examines the influence of only a small subset (a block $S_j$) on the hypothesis $D(P_{W|S_j} \| Q_W)$, this quantity can remain finite—even if the total influence is infinite.

### Goal
(1) Construct a family of MAC-Bayes generalization bounds that utilize block structures, providing meaningful bounds even when original PAC-Bayes bounds are vacuous; (2) Analyze the optimal selection of block size; (3) Explore the feasibility of obtaining high-probability (PAC) versions.

### Key Insight
Inspired by "individual sample bounds" in information theory (Bu et al., 2020), this work partitions the training set $S$ into $J = n/m$ blocks of size $m$. It replaces the global KL divergence with the sum of KL divergences between the marginalized posterior $P_{W|S_j} := \mathbb{E}_{P_{S \setminus S_j}} P_{W|S}$ and the prior.

### Core Idea
Construct tighter MAC-Bayes generalization bounds by partitioning the training set and using the "sum of KL divergences under partial data conditioning" instead of "full data KL divergence."

## Method

### Overall Architecture

This is a pure theory paper centered on a new family of generalization bounds rather than an executable algorithm. It addresses a specific pain point: classic PAC-Bayes bounds compress the "total informational influence of training set $S$ on hypothesis $W$" into a scalar divergence $D(P_{W|S} \| Q_W)$. When this influence is too strong—typical for deterministic algorithms ($W = \frac{1}{n}\sum_i Z_i$) where $P_{W|S}$ is a Dirac distribution—the KL divergence to any prior is infinite, rendering the bound vacuous.

The proposed solution replaces the "single KL of full data" with the "sum of KLs of individual blocks." Specifically, the i.i.d. training set $S = (Z_1, \ldots, Z_n)$ is uniformly split into $J = n/m$ blocks $S_j$ of size $m$. The **marginalized posterior** is defined as $P_{W|S_j} := \mathbb{E}_{P_{S_1,\ldots,S_{j-1},S_{j+1},\ldots,S_J}} P_{W|S}$. Crucially, $P_{W|S_j}$ is **not** obtained by retraining the algorithm on $S_j$ alone; rather, it is the "blurred" distribution resulting from taking the expectation of the **full algorithm** $P_{W|S}$ over the remaining $J-1$ blocks. This expectation step smooths the Dirac delta into a continuous distribution, ensuring $D(P_{W|S_j} \| Q_W)$ remains finite even when $D(P_{W|S} \| Q_W) = \infty$. The paper revolves around the resulting target bound:

$$\mathbb{E}_{P_S} d(\mathbb{E}_{P_{W|S}} \hat{L}(W,S), \mathbb{E}_{P_{W|S}} L(W)) \leq \frac{\sum_{j=1}^{J} \mathbb{E}_{P_{S_j}} D(P_{W|S_j} \| Q_W) + I''(n,d,J)}{n}$$

It answers four questions: how to prove it (Theorem 1), how to specialize it into usable bounds (comparator specialization), how to select the optimal block size $m$, and whether it can be upgraded to a high-probability (PAC) version.

### Key Designs

**1. Block-Sample MAC-Bayes Main Bound (Theorem 1): Sum of Marginalized KLs**

This step addresses the "coarse KL and divergence explosion" issue by providing the most general block-sample bound. The proof follows three steps: first, applying Jensen’s inequality (relying on the joint convexity of distance function $d$) to move the outer expectation inside $d$ and decompose it into a block sum $\frac{1}{J}\sum_j$; second, using the Fubini-Tonelli theorem to fold the expectation over other blocks into the marginalized posterior $P_{W|S_j}$; finally, applying the Donsker-Varadhan variational representation to each block to perform the change of measure from $P_{W|S_j}$ to $Q_W$. The bound is parameterized by $d$ and the moment-generating function (MGF) condition $\Phi_m(\lambda')$, making it a **family** of bounds. The key to its success is that each block divergence depends on the marginalized posterior $P_{W|S_j}$, which is "blurred" into a continuous distribution, allowing it to be much smaller than the full data $D(P_{W|S} \| Q_W)$ and remain finite where the original bound fails.

**2. Comparator Specialization (Corollary 1 & 2): Specializing Distance Functions**

Theorem 1 is an abstract inequality; this step selects specific comparators for $d$ to produce usable bounds. For **bounded losses** $\ell(w,z) \in [0,1]$, using the Catoni function $C_\beta$ as $d$ **eliminates the MGF term**, yielding:

$$\mathbb{E}_{P_S} C_\beta(\mathbb{E}_{P_{W|S}} \hat{L}, \mathbb{E}_{P_{W|S}} L) \leq \frac{1}{n} \sum_j \mathbb{E}_{P_{S_j}} D(P_{W|S_j} \| Q_W),$$

which solves to the tightest generalization bound $\text{gen} \leq \sqrt{\frac{1}{4n} \sum_j \mathbb{E}_{P_{S_j}} D(P_{W|S_j} \| Q_W)}$. For unbounded losses, Corollary 2 assumes a $\sigma^2$-sub-Gaussian tail and uses the difference function $d(r,s)=s-r$, resulting in $\text{gen} \leq \sqrt{\frac{2\sigma^2}{n} \sum_j \mathbb{E}_{P_{S_j}} D(P_{W|S_j} \| Q_W)}$. The tightness ranking is Catoni (tightest) > Sub-Gaussian > Binary KL.

**3. Block Size Optimization (Section 5): Optimal Choice of $m$**

While the bound holds for any $m \neq n$, the optimal $m$ depends on how the divergence grows with block size. Assuming block divergences satisfy $\mathbb{E}_{P_S} D(P_{W|S_j} \| Q_W) \leq \mathcal{O}(m^\gamma)/\Theta(n)$, the optimal size is determined by $\gamma$. If $\gamma < 1$, a constant block size (including $m=1$) is optimal, yielding an $\mathcal{O}(n^{-1/2})$ decay. If $\gamma > 1$, the block size should grow linearly with samples $m = \Theta(n)$ (while maintaining $m \neq n$). Gaussian mean estimation falls on the transition point $\gamma = 1$, where any $m \neq n$ yields $\mathcal{O}(n^{-1/2})$.

**4. Impossibility of High Probability (Theorem 2): No PAC Version**

Finally, the paper proves that no meaningful high-probability (PAC) version exists in the block-sample setting. This is a fundamental impossibility shown via a counterexample: a learning algorithm that severely overfits with low probability but outputs a zero-loss hypothesis with high probability. In this scenario, the MAC-Bayes bound still converges at $\mathcal{O}(n^{-1/2})$, but any PAC-Bayes bound of the form $P_S(\text{gen} > A_n + B_n f(1/\delta)) \leq \delta$ either has a very rapidly growing $f$ or a very slowly converging $B_n$. This result defines the boundary between MAC-Bayes and PAC-Bayes for block-sample methods.

## Key Experimental Results

### Numerical Example
The paper includes a numerical validation of Gaussian mean estimation ($Z_i \sim \mathcal{N}(\mu, 1)$, $W = \frac{1}{n}\sum Z_i$, truncated squared loss) rather than large-scale ML experiments.

| Block Size $m$ | Bound Behavior | Remarks |
|-----------|---------|------|
| $m = n$ (Original PAC-Bayes) | $\infty$ (Vacuous) | KL divergence is infinite |
| $m = 1$ (Finest partition) | $\mathcal{O}(n^{-1/2})$, Optimal | Tightest in this example |
| $1 < m < n$ | Finite and $\mathcal{O}(n^{-1/2})$ | Relatively insensitive to choice |

### Main Results

| Bound Type | Comparator $d$ | Form of Bound | Requires $m \neq n$ |
|---------|-----------|---------|-------------------|
| Corollary 1 (Catoni) | $C_\beta$ | $\sqrt{\frac{1}{4n}\sum D}$ | Yes |
| Eq.(11) (Direct binary KL) | $\text{kl}$ | $\sqrt{\frac{\log(2\sqrt{m})}{m} + \frac{1}{n}\sum D}$ | Yes (looser) |
| Corollary 2 (Sub-Gaussian) | $s - r$ | $\sqrt{\frac{2\sigma^2}{n}\sum D}$ | Yes |
| Prev. SOTA (Original PAC-Bayes) | Various | $\frac{D(P_{W|S}\|Q_W) + \cdots}{n}$ | N/A (Vacuous) |

### Key Findings
- Block-sample bounds provide meaningful $\mathcal{O}(n^{-1/2})$ convergence guarantees in scenarios where original PAC-Bayes bounds are completely vacuous.
- The bound is relatively insensitive to the choice of block size $m$ (as long as $m \neq n$), though $m = 1$ is optimal for the Gaussian mean estimation example.
- specialization with the Catoni function (Corollary 1) is strictly superior to direct substitution of binary KL or difference functions.
- The impossibility of a high-probability version is a fundamental limitation rather than a technical hurdle.

## Highlights & Insights
- **The core idea of "block-wise marginalization" is elegant**: By taking the expectation of $P_{W|S}$ over partial data to obtain the "blurred" $P_{W|S_j}$, the Dirac distribution becomes continuous. This reveals that the failure of KL divergence in PAC-Bayes is not a failure of the framework itself, but of measuring information at too coarse a granularity.
- **Counterexample Construction**: The counterexample for Theorem 2 ("low-probability severe overfitting + high-probability perfection") is a sophisticated probabilistic construction that clearly illustrates the irreconcilable gap between expectation and high-probability bounds in this context.
- **Advantages of the Catoni Comparator**: In the block-sample setting, the Catoni function eliminates the MGF term entirely, whereas binary KL or difference functions introduce extra terms that loosen the bound.

## Limitations & Future Work
- **Lack of Practical ML Experiments**: The paper only validates a toy Gaussian mean estimation example and lacks evidence for actual ML algorithms (e.g., neural networks trained with SGD). The authors admit that "significant future work is needed for practical applications."
- **Dependence on Data Distribution**: The divergence term $D(P_{W|S_j} \| Q_W)$ depends on the data-generating distribution $P_Z$, making the bound incalculable if $P_Z$ is unknown—a common issue for information-theoretic generalization bounds.
- **Computational Difficulty of Marginalized Posteriors**: Calculating $P_{W|S_j}$ requires taking expectations over $J-1$ blocks, which is analytically or computationally prohibitive for complex models like deep learning.
- Future directions: (1) Upper-bounding the divergence terms using properties of specific learning algorithms; (2) Developing computable versions for deep learning; (3) Exploring alternative partitioning strategies (e.g., random vs. sequential blocks).

## Related Work & Insights
- **Comparison to Bu et al. (2020) Individual Sample Bounds**: While Bu et al. decompose mutual information bounds into $I(W; Z_i)$, this work applies a similar logic to PAC-Bayes using KL divergence and block structures. The block approach is more flexible, allowing for optimization via block size.
- **Comparison to Harutyunyan et al. (2021, 2022)**: Similar considerations regarding subset mutual information and the impossibility of high-probability versions were discussed in different system settings and restricted to $m=1$.
- **Comparison to Wu et al. (2024) Recursive PAC-Bayes**: Also uses block partitioning but within a recursive structure, making it not directly comparable.
- This work demonstrates that the PAC-Bayes framework still has significant room for improvement, and "divide-and-conquer" information decomposition is a promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐ The block-sample decomposition is novel with theoretical depth, and the impossibility result adds completeness.
- Experimental Thoroughness: ⭐⭐ Includes only one toy numerical example, lacking validation in real ML scenarios, though acceptable for a pure theory paper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](../../NeurIPS2025/llm_pretraining/generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](../../ICML2026/llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[ICML 2026\] AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining](../../ICML2026/llm_pretraining/ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining.md)
- [\[ICLR 2026\] Rethinking Data Curation in LLM Training: Online Reweighting Offers Better Generalization than Offline Methods](rethinking_data_curation_in_llm_training_online_reweighting_offers_better_genera.md)
- [\[ICML 2026\] Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity](../../ICML2026/llm_pretraining/tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge.md)

</div>

<!-- RELATED:END -->
