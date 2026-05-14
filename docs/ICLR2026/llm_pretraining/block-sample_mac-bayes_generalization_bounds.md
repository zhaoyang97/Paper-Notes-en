---
title: >-
  [Paper Note] Block-Sample MAC-Bayes Generalization Bounds
description: >-
  [ICLR2026][LLM Pretraining][PAC-Bayes] This paper proposes block-sample MAC-Bayes generalization bounds (mean approximately correct) that partition the training data into $J$ blocks and replace the monolithic KL divergen…
tags:
  - "ICLR2026"
  - "LLM Pretraining"
  - "PAC-Bayes"
  - "MAC-Bayes"
  - "generalization bounds"
  - "information theory"
  - "block samples"
date: 2026-05-08
content_hash: ac7de94a3bed985e
---

# Block-Sample MAC-Bayes Generalization Bounds

**Conference**: ICLR2026
**arXiv**: [2602.12605](https://arxiv.org/abs/2602.12605)
**Code**: None
**Area**: LLM Pretraining
**Keywords**: PAC-Bayes, MAC-Bayes, generalization bounds, information theory, block samples

## TL;DR
This paper proposes block-sample MAC-Bayes generalization bounds (mean approximately correct) that partition the training data into $J$ blocks and replace the monolithic KL divergence with a sum of per-block conditional KL divergences. The resulting bounds remain finite and meaningful even in settings where the original PAC-Bayes bounds are vacuous (e.g., deterministic learning algorithms such as mean estimation). The paper further establishes that a high-probability (PAC) version of these bounds is generally unattainable.

## Background & Motivation

### State of the Field
The PAC-Bayes framework is a central tool in statistical learning theory for bounding generalization error. It quantifies the gap between empirical and population loss via the KL divergence between the posterior $P_{W|S}$ induced by the learning algorithm and a prior $Q_W$. PAC-Bayes bounds have regained prominence in recent years for their ability to yield non-trivial generalization guarantees for deep neural networks. MAC-Bayes bounds are the expectation counterpart of PAC-Bayes bounds, controlling expected rather than high-probability generalization error.

### Limitations of Prior Work

**Failure of PAC-Bayes bounds for deterministic algorithms**: When the learning algorithm $P_{W|S}$ is deterministic (e.g., $W = \frac{1}{n}\sum_i Z_i$), $P_{W|S}$ is a Dirac measure, and the KL divergence with respect to any prior $Q_W$ is infinite, rendering both PAC-Bayes and the corresponding MAC-Bayes bounds vacuous.

**The monolithic KL term is too coarse**: The global $D(P_{W|S} \| Q_W)$ summarizes the influence of the entire training set on the hypothesis in a single scalar. When this influence is too strong (as in deterministic algorithms), the bound explodes.

**Analogous limitations of information-theoretic bounds**: Generalization bounds based on mutual information $I(W;S)$ suffer from the same deficiency.

### Root Cause
The divergence term $D(P_{W|S} \| Q_W)$ in PAC-Bayes bounds measures the total "informational influence" of the complete training set $S$ on the hypothesis $W$; the bound breaks down whenever this influence is large. Intuitively, however, the influence of a small subset (a single block $S_j$) on the hypothesis, $D(P_{W|S_j} \| Q_W)$, can remain finite even when the total influence is infinite.

### Paper Goals
(1) Construct a family of MAC-Bayes generalization bounds that exploit block structure and yield meaningful guarantees even when the original PAC-Bayes bounds are vacuous; (2) analyze the optimal choice of block size; (3) investigate whether a high-probability (PAC) counterpart can be obtained.

### Starting Point
Inspired by the individual-sample bounds of Bu et al. (2020) in the information-theoretic literature, the paper partitions the training set $S$ into $J = n/m$ blocks of size $m$ and replaces the global KL divergence with the sum of KL divergences between the marginalized posteriors $P_{W|S_j} := \mathbb{E}_{P_{S \setminus S_j}} P_{W|S}$ and the prior.

### Core Idea
By partitioning the training set into blocks and substituting the sum of "partial-data conditional KL divergences" for the "full-data KL divergence," the paper constructs tighter MAC-Bayes generalization bounds.

## Method

### Overall Architecture
Let the training set $S = (Z_1, \ldots, Z_n)$ be i.i.d. and partition it uniformly into $J = n/m$ blocks $S_j$ of size $m$. Define the marginalized posterior $P_{W|S_j} := \mathbb{E}_{P_{S_1,\ldots,S_{j-1},S_{j+1},\ldots,S_J}} P_{W|S}$ (this is not the posterior of an algorithm trained only on $S_j$, but rather the full algorithm's posterior averaged over the remaining blocks). The goal is to establish generalization bounds of the form:

$$\mathbb{E}_{P_S} d(\mathbb{E}_{P_{W|S}} \hat{L}(W,S), \mathbb{E}_{P_{W|S}} L(W)) \leq \frac{\sum_{j=1}^{J} \mathbb{E}_{P_{S_j}} D(P_{W|S_j} \| Q_W) + I''(n,d,J)}{n}$$

### Key Designs

1. **Core theorem (Theorem 1) — Block-sample MAC-Bayes bound**:

    - Function: Provides a general block-sample generalization bound, parameterized by a divergence function $d$ and a moment-generating function condition $\Phi_m$.
    - Mechanism: Jensen's inequality (joint convexity of $d$) pulls the expectation inside $d$; Fubini's theorem decouples the blocks; the Donsker–Varadhan variational representation is then applied per block to perform the measure change from posterior to prior. Crucially, the per-block KL divergence $D(P_{W|S_j} \| Q_W)$ depends only on the marginalized distribution $P_{W|S_j}$, which can be far smaller than the full-data $D(P_{W|S} \| Q_W)$ when $m \ll n$.
    - Design Motivation: For deterministic algorithms, $D(P_{W|S} \| Q_W) = \infty$, whereas $D(P_{W|S_j} \| Q_W)$ is finite because $P_{W|S_j}$, obtained by averaging over the remaining blocks, is a "smoothed" distribution rather than a Dirac measure.

2. **Catoni function specialization (Corollary 1)**:

    - Function: Specializes the bound to bounded losses $\ell(w,z) \in [0,1]$ using the Catoni function as the comparison function.
    - Core result: $\mathbb{E}_{P_S} C_\beta(\mathbb{E}_{P_{W|S}} \hat{L}, \mathbb{E}_{P_{W|S}} L) \leq \frac{1}{n} \sum_j \mathbb{E}_{P_{S_j}} D(P_{W|S_j} \| Q_W)$, with the moment-generating function term eliminated entirely. A generalization error bound of $\text{gen} \leq \sqrt{\frac{1}{4n} \sum_j \mathbb{E}_{P_{S_j}} D(P_{W|S_j} \| Q_W)}$ follows immediately.
    - This is the tightest version; the corresponding binary KL and difference-function specializations are both looser.

3. **Sub-Gaussian loss extension (Corollary 2)**:

    - Function: Extends the bound from bounded to $\sigma^2$-sub-Gaussian losses.
    - Core result: $\text{gen} \leq \sqrt{\frac{2\sigma^2}{n} \sum_j \mathbb{E}_{P_{S_j}} D(P_{W|S_j} \| Q_W)}$.
    - Broader applicability at the cost of a slightly looser bound.

4. **Impossibility of a high-probability version (Theorem 2)**:

    - Function: Proves that a block-sample PAC-Bayes bound (high-probability version) is generally unattainable.
    - Mechanism: A counterexample learning scenario is constructed in which the algorithm severely overfits the training set with small probability and outputs a zero-loss hypothesis with large probability. In this setting the MAC-Bayes bound converges at rate $\mathcal{O}(n^{-1/2})$, yet any PAC-Bayes bound of the form $P_S(\text{gen} > A_n + B_n f(1/\delta)) \leq \delta$ must either have $f$ growing faster than logarithmically or have $B_n$ converging slowly.
    - Significance: This result draws a fundamental distinction between MAC-Bayes and PAC-Bayes in the block-sample setting.

### Block Size Optimization
Under the assumption $\mathbb{E}_{P_S} D(P_{W|S_j} \| Q_W) \leq \mathcal{O}(m^\gamma) / \Theta(n)$:
- $\gamma < 1$: A constant block size $m$ (including $m=1$) is optimal, and the bound decays at rate $\mathcal{O}(n^{-1/2})$.
- $\gamma > 1$: The block size should grow linearly, $m = \Theta(n)$ (with $m \neq n$ required); the Gaussian mean-estimation example corresponds to $\gamma = 1$, the transition point.
- $\gamma = 1$ (transition): Any choice $m \neq n$ yields an $\mathcal{O}(n^{-1/2})$ bound.

## Key Experimental Results

### Numerical Example
The paper includes a single numerical validation based on Gaussian mean estimation ($Z_i \sim \mathcal{N}(\mu, 1)$, $W = \frac{1}{n}\sum Z_i$, truncated squared loss) rather than large-scale ML experiments.

| Block size $m$ | Bound behavior | Remarks |
|---|---|---|
| $m = n$ (original PAC-Bayes) | $\infty$ (vacuous) | KL divergence is infinite |
| $m = 1$ (finest partition) | $\mathcal{O}(n^{-1/2})$, optimal | Tightest in this example |
| $1 < m < n$ | Finite and $\mathcal{O}(n^{-1/2})$ | Relatively insensitive to choice |

### Theoretical Comparison

| Bound type | Comparator $d$ | Bound form | Requires $m \neq n$ |
|---|---|---|---|
| Corollary 1 (Catoni) | $C_\beta$ | $\sqrt{\frac{1}{4n}\sum D}$ | Yes |
| Eq. (11) (direct binary KL substitution) | $\text{kl}$ | $\sqrt{\frac{\log(2\sqrt{m})}{m} + \frac{1}{n}\sum D}$ | Yes, and looser |
| Corollary 2 (sub-Gaussian) | $s - r$ | $\sqrt{\frac{2\sigma^2}{n}\sum D}$ | Yes |
| Original PAC-Bayes | Same | $\frac{D(P_{W|S}\|Q_W) + \cdots}{n}$ | N/A |

### Key Findings
- Block-sample bounds provide meaningful $\mathcal{O}(n^{-1/2})$ convergence guarantees in settings where the original PAC-Bayes bounds are completely vacuous.
- The bounds are relatively insensitive to the choice of block size $m$ (as long as $m \neq n$), though $m = 1$ is optimal in the Gaussian mean-estimation example.
- The Catoni function specialization (Corollary 1) strictly dominates the binary KL and difference-function alternatives.
- The impossibility of a high-probability version constitutes a fundamental rather than a merely technical limitation.

## Highlights & Insights
- **The "block marginalization" idea is remarkably elegant**: by averaging the deterministic $P_{W|S}$ over part of the data to obtain the "smoothed" $P_{W|S_j}$, a Dirac measure is transformed into a continuous distribution, converting an infinite KL divergence into a finite one. This trick reveals that the failure of PAC-Bayes bounds is not a deficiency of the framework per se, but rather a consequence of the KL divergence operating at too coarse a granularity.
- **Counterexample construction technique**: The counterexample in Theorem 2 ("severe overfitting with small probability, perfect generalization with large probability") is an elegant probabilistic construction that cleanly exposes the unbridgeable gap between expectation bounds and high-probability bounds.
- **The advantage of the Catoni comparator function** is further amplified in the block-sample setting: it eliminates the moment-generating function term entirely, whereas the binary KL and difference-function substitutions both introduce additional terms.

## Limitations & Future Work
- **Only a simple numerical example**: The paper validates only the toy Gaussian mean-estimation example and does not demonstrate effectiveness on any practical ML algorithm (e.g., SGD-trained neural networks). The authors acknowledge that "substantial follow-up work is needed to address practical applications."
- **Dependence on the data distribution**: The divergence term $D(P_{W|S_j} \| Q_W)$ depends on the data-generating distribution $P_Z$, making the bound uncomputable without knowledge of $P_Z$. This is a common limitation of information-theoretic generalization bounds.
- **Computational difficulty of the marginalized posterior**: $P_{W|S_j}$ requires averaging over the remaining $J-1$ blocks, which is analytically or computationally intractable for complex learning algorithms such as deep networks.
- Future directions: (1) further upper-bounding the divergence term by exploiting properties of the learning algorithm; (2) deriving computable versions for practical deep learning settings; (3) exploring alternative blocking strategies (e.g., random vs. contiguous blocks).

## Related Work & Insights
- **vs. Bu et al. (2020) individual-sample bounds**: Bu et al. decompose the mutual information bound to per-sample terms $I(W; Z_i)$; the present work pursues an analogous decomposition within the PAC-Bayes framework using KL divergence and block structure. The block approach is more flexible, as the block size can be tuned to optimize the bound.
- **vs. Harutyunyan et al. (2021, 2022)**: They also consider subset mutual information bounds and the impossibility of high-probability versions, but under different system assumptions and restricted to the special case $m=1$.
- **vs. Wu et al. (2024) recursive PAC-Bayes**: They also employ block partitioning but with a recursive structure, precluding direct comparison.
- This work demonstrates that the PAC-Bayes framework still offers substantial room for improvement, and that information decomposition via a "divide-and-conquer" strategy is a promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐ The block-sample decomposition idea is novel and theoretically deep; the impossibility result enhances completeness.
- Experimental Thoroughness: ⭐⭐ Only a single toy numerical example is provided, with no validation on real ML scenarios; this is, however, acceptable for a purely theoretical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](../../NeurIPS2025/llm_pretraining/generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[CVPR 2026\] MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation](../../CVPR2026/llm_pretraining/mxnorm_reusing_mxfp_block_scales_for_efficient_tensor_normalisation.md)
- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[ICLR 2026\] Deconstructing Positional Information: From Attention Logits to Training Biases](deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)

</div>

<!-- RELATED:END -->
