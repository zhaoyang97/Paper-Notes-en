---
title: >-
  [Paper Note] Generation from Noisy Examples
description: >-
  [ICML2025][Generation theory] The theoretical framework of "language generation in the limit" by Kleinberg & Mullainathan (2024) is extended to noisy sample stream scenarios. The Noisy Closure dimension is proposed to fully characterize the necessary and sufficient conditions for uniform noise-dependent generability, proving that all countable hypothesis classes remain non-uniformly generable under finite noise.
tags:
  - "ICML2025"
  - "Generation theory"
  - "Noise robustness"
  - "Combinatorial dimension"
  - "Learning theory"
  - "Generation in the limit"
date: 2026-05-08
content_hash: 2a2cffba053104e3
---

# Generation from Noisy Examples

**Conference**: ICML2025  
**arXiv**: [2501.04179](https://arxiv.org/abs/2501.04179)  
**Code**: None (purely theoretical work)  
**Area**: Others  
**Keywords**: Generation theory, Noise robustness, Combinatorial dimension, Learning theory, Generation in the limit  

## TL;DR

The theoretical framework of "language generation in the limit" by Kleinberg & Mullainathan (2024) is extended to noisy sample stream scenarios. The Noisy Closure dimension is proposed to fully characterize the necessary and sufficient conditions for uniform noise-dependent generability, proving that all countable hypothesis classes remain non-uniformly generable under finite noise.

## Background & Motivation

Kleinberg & Mullainathan (2024) proposed the "language generation in the limit" model: given a countable language family $C$, an adversary chooses a language $K \in C$ and presents its strings one by one. The generator must output new, unseen strings in $K$. They proved a surprising positive result—**all countable language families are generable in the limit**, which stands in sharp contrast to Gold's (1967) language identification result where "most language families are not identifiable."

Li et al. (2024) further extended this framework to binary hypothesis classes $\mathcal{H} \subseteq \{0,1\}^{\mathcal{X}}$, defining uniform and non-uniform generability and providing a complete characterization using the Closure dimension.

However, all prior works assume that the sample stream is **noiseless**—every sample is a positive sample. In reality, this assumption does not hold:
- LLMs are often trained on hallucinated outputs from other LLMs (Burns et al., 2023).
- Training data may suffer from data poisoning attacks (Zhang et al., 2023).

The **Core Problem** proposed in this paper is: **How does the generation capability change when an adversary can insert a finite number of negative samples into the positive sample stream?**

## Method

### Problem Formalization

Let $\mathcal{X}$ be a countable instance space, and let $\mathcal{H} \subseteq \{0,1\}^{\mathcal{X}}$ be a binary hypothesis class satisfying the Uniform Unbounded Support (UUS) property. In the noise model, after the adversary selects $h \in \mathcal{H}$ and its positive sample sequence, it can insert at most $n^{\star}$ negative samples. The generator does not know which samples are negative.

### Four Definitions of Noisy Generability

The paper defines four noisy-generability definitions from strongest to weakest, based on the dependency of the number of samples required for "eventual perfect generation" $d^{\star}$:

| Definition | $d^{\star}$ depends on | Strength |
|------|-------------------|------|
| Uniform Noise-Independent | Only class $\mathcal{H}$ | Strongest |
| Uniform Noise-Dependent | $\mathcal{H}$ + noise level $n^{\star}$ | Strong |
| Non-uniform Noise-Dependent | $\mathcal{H}$ + $n^{\star}$ + hypothesis $h$ | Weak |
| Noisy in the Limit | $\mathcal{H}$ + $n^{\star}$ + $h$ + stream | Weakest |

### Core Tool: Noisy Closure Dimension

Define the closure operator under noise level $n$:

$$\langle x_{1:d} \rangle_{\mathcal{H},n} := \bigcap_{h \in \mathcal{H}(x_{1:d};n)} \operatorname{supp}(h)$$

where $\mathcal{H}(x_{1:d};n) = \{h \in \mathcal{H} : |\{x_{1:d}\} \cap \operatorname{supp}(h)| \geq d - n\}$ is the set of hypotheses that disagree with the sequence on at most $n$ samples.

The **$n$-Noisy Closure dimension** $\mathrm{NC}_n(\mathcal{H})$ is defined as the maximum $d$ such that $\langle x_{1:d} \rangle_{\mathcal{H},n} \neq \bot$ and $|\langle x_{1:d} \rangle_{\mathcal{H},n}| < \infty$.

The key difference from the noiseless Closure dimension: the Noisy Closure dimension is **scale-sensitive**—each noise level $n$ corresponds to a unique dimensional value.

### Constructive Generator

For classes that are uniform noise-dependent generable, the paper constructs a generator that does not require prior knowledge of the noise level: it can achieve perfect generation after observing $\mathrm{NC}_n(\mathcal{H}) + 1$ distinct samples (when the noise level is $n$). For noisy generability in the limit, Algorithm 1 wraps a noiseless non-uniform generator $\mathcal{Q}$ into a noise-robust generator $\mathcal{G}$: in each round, it feeds the latter half of the stream (which is guaranteed to be noiseless) to call $\mathcal{Q}$ repeatedly to generate candidates, filtering out already seen samples.

## Theoretical Results

| Result | Content | Significance |
|------|------|------|
| **Thm 3.1** | Uniform noise-independent generable $\Leftrightarrow$ $\|\bigcap_{h\in\mathcal{H}} \operatorname{supp}(h)\| = \infty$ | Extremely difficult: only "trivial" classes are feasible |
| **Thm 3.3** | Uniform noise-dependent generable $\Leftrightarrow$ $\forall n, \mathrm{NC}_n(\mathcal{H}) < \infty$ | Complete characterization, sample complexity is $\Theta(\mathrm{NC}_n(\mathcal{H}))$ |
| **Cor 3.4** | All finite classes are uniform noise-dependent generable | Finite classes are noise-robust |
| **Lem 3.5** | There exists a countable class that is noiseless uniformly generable but has $\mathrm{NC}_1 = \infty$ | Noise strictly increases the difficulty |
| **Lem 3.6** | Decomposable into $\bigcup \mathcal{H}_i$ where $\mathrm{NC}_i(\mathcal{H}_i) < \infty$ implies non-uniform noise-dependent generability | Sufficient condition |
| **Cor 3.7** | All countable classes are non-uniform noise-dependent generable | Noise comes "for free" in countable classes |
| **Thm 3.9** | Noiseless non-uniformly generable $\Rightarrow$ Noisy generable in the limit | Noiseless capability implies noise robustness |
| **Thm 3.10** | Union of finitely many uniform noise-independent generable classes $\rightarrow$ Noisy generable in the limit | Sufficient condition for class decomposition |

## Highlights & Insights

- **The impact of noise on generation is mild**: Although noise makes uniform generation strictly harder (Lem 3.5), it has absolutely no impact on the non-uniform/in-the-limit generability of countable classes (Cor 3.7).
- **The scale-sensitivity of the Noisy Closure dimension** is similar to the fat-shattering dimension in PAC regression, representing the first introduction of such a structure to this field.
- **Constructive proofs**: All positive results provide explicit generator algorithms, rather than pure existence proofs.
- **The separation between generation and identification continues to hold**: Generation remains significantly easier than identification under noise.
- **Clever counterexample** (Lem 3.5): A hypothesis class constructed using prime power negative sets has a noiseless Closure dimension of $0$ but $\mathrm{NC}_1 = \infty$.

## Key Proof Ideas

**Thm 3.3 Necessity**: If $\exists n$ such that $\mathrm{NC}_n(\mathcal{H}) = \infty$, then for any generator $\mathcal{G}$ and any sample size $d$, one can construct a hypothesis $h$ and a noisy stream such that $\mathcal{G}$ makes a mistake after observing $d$ distinct samples. The infinite Noisy Closure dimension guarantees that one can always find a sequence that makes the closure finite to "fool" the generator.

**Thm 3.3 Sufficiency**: Explicit construction of the generator: after observing $\mathrm{NC}_n(\mathcal{H}) + 1$ distinct samples, for all possible noise levels, the closure $\langle x_{1:d} \rangle_{\mathcal{H},n}$ must be an infinite set (or $\bot$), from which new positive samples can always be selected. The core trick is that the generator does not need to know the true noise level—it "hedges" against all possible $n$ simultaneously.

**Cor 3.4**: For a finite class where $|\mathcal{H}| = q$, $\mathrm{NC}_n(\mathcal{H}) < nq + d + 1 < \infty$. The upper bound is obtained via the Pigeonhole Principle and the finiteness of the intersection of subset supports.

## Limitations & Future Work

- **The complete characterization of non-uniform noise-dependent generability remains an open problem**: There is a gap between the sufficient and necessary conditions.
- **The complete characterization of noisy generability in the limit remains unsolved** (inheriting an open problem from Li et al., 2024).
- **Only insertion noise is considered**: The adversary can only insert negative samples but cannot replace positive samples, which is a weaker noise model.
- **Purely information-theoretic perspective**: Computational feasibility is not discussed. Computable noisy generation algorithms given a membership oracle remain an open problem.
- **Finite noise**: The total number of negative samples is required to be finite, which does not cover infinite noise scenarios (like the density noise model of Jain, 1994).
- No experimental validation—purely theoretical contribution.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first to introduce noise robustness into the theory of generation in the limit; the Noisy Closure dimension is a natural and non-trivial new combinatorial dimension.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Complete characterization of uniform noise-dependent generability is provided (necessary and sufficient conditions plus sample complexity), with solid proof techniques.
- Writing Quality: ⭐⭐⭐⭐ — Clear layout, rigorous definition-theorem-proof structure, but mathematically heavy in notation.
- Value: ⭐⭐⭐⭐ — Advances the foundations of generation theory, though the remaining open problems indicate the work is not fully finalized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICLR 2026\] Learning in Prophet Inequalities with Noisy Observations](../../ICLR2026/others/learning_in_prophet_inequalities_with_noisy_observations.md)
- [\[ECCV 2024\] Foster Adaptivity and Balance in Learning with Noisy Labels](../../ECCV2024/others/foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)
- [\[CVPR 2026\] Debiased Sample Selection for Learning with Noisy Labels](../../CVPR2026/others/debiased_sample_selection_for_learning_with_noisy_labels.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](../../ACL2025/others/hierarchical_memory_wikipedia_gen.md)

</div>

<!-- RELATED:END -->
