---
title: >-
  [Paper Note] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming
description: >-
  [NeurIPS 2025][AI Safety][differential privacy] This paper proposes a general Gaussian privacy mechanism based on Euclidean Jordan Algebra (EJA) and, building upon it…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "differential privacy"
  - "Euclidean Jordan Algebra"
  - "Symmetric Cone Programming"
  - "Semidefinite Programming"
  - "Gaussian Mechanism"
date: 2026-05-08
content_hash: 99c53f0394b28108
---

# Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming

**Conference**: NeurIPS 2025
**arXiv**: [2509.16915](https://arxiv.org/abs/2509.16915)
**Code**: None
**Area**: AI Security
**Keywords**: differential privacy, Euclidean Jordan Algebra, Symmetric Cone Programming, Semidefinite Programming, Gaussian Mechanism

## TL;DR

This paper proposes a general Gaussian privacy mechanism based on Euclidean Jordan Algebra (EJA) and, building upon it, designs the first differentially private algorithm for Symmetric Cone Programming (SCP), thereby resolving an important open problem on differentially private semidefinite programming posed by Hsu et al. (ICALP 2014).

## Background & Motivation

Differential Privacy (DP) is the de facto standard for data privacy protection and is widely applied in machine learning. Existing privacy algorithms are mostly designed on a case-by-case basis for specific problems, or realized through general methods such as DP-SGD. In the domain of convex optimization, however, existing DP algorithms are largely limited to **linear programming (LP)**.

Symmetric Cone Programming (SCP) is a broad class of convex optimization problems that subsumes LP, second-order cone programming (SOCP), and semidefinite programming (SDP). SCP has numerous applications in machine learning, including support vector machines, matrix completion, robust mean/covariance estimation, experimental design, and sparse PCA. Despite mature algorithmic theory for SCP, differentially private algorithms for this class are nearly nonexistent. In particular, **designing a differentially private SDP algorithm** was posed as an important open problem by Hsu et al. (2014) and remained unsolved for over a decade.

The core challenge is that the optimization variables in SCP (e.g., symmetric matrices) possess rich algebraic and geometric structure. The conventional element-wise Laplace mechanism fails to capture these structures (e.g., spectral structure), resulting in poor privacy–utility trade-offs.

## Core Problem

1. How to design a general differentially private mechanism for functions whose outputs lie in a Euclidean Jordan Algebra, particularly when sensitivity is measured under the $\ell_1$, $\ell_2$, or $\ell_\infty$ norm?
2. How to design differentially private SCP solvers under different privacy settings (high-sensitivity constraint privacy, low-sensitivity constraint/scalar/objective privacy)?
3. Can the long-standing open problem of differentially private SDP be resolved?

## Method

### 1. General Gaussian Mechanism on Euclidean Jordan Algebras

An EJA is a finite-dimensional vector space equipped with a Jordan product and an inner product, unifying important structures such as $\mathbb{R}^k$ and the space of real symmetric matrices. Every EJA element admits a spectral decomposition $x = \sum_{i=1}^r \lambda_i q_i$ (analogous to the eigendecomposition of a matrix), where $r$ is the rank and $k$ is the dimension of the EJA.

**Core Idea**: Leveraging the isometric isomorphism $\phi$ from EJA to $\mathbb{R}^k$, Gaussian noise is generated in $\mathbb{R}^k$ and mapped back to the EJA.

- Let $f: \mathcal{D} \to \mathcal{J}$ have $\ell_2$-sensitivity $\Delta_2$.
- Sample $\nu \sim \mathcal{N}(0, \sigma^2 I_k)$, where $\sigma = \Delta_2 \sqrt{2\log(1.25/\delta)} / \epsilon$.
- Output $f(D) + \phi^{-1}(\nu)$.

Since $\phi$ preserves inner products and $\ell_2$ norms, privacy follows directly from the standard Gaussian mechanism. For $\ell_1$ and $\ell_\infty$ sensitivities, reductions are made via norm inequalities.

**Key Insight**: One cannot perturb only the eigenvalues while ignoring the Jordan frame — perturbing the spectrum alone is insufficient for differential privacy; the basis (Jordan frame) must also be randomized. This explains why noise must be added in the full $k$-dimensional space rather than solely in the $r$-dimensional spectral space.

### 2. SCP under High-Sensitivity Constraint Privacy

In this setting, adjacent databases may differ by the complete addition or removal of a constraint. The algorithm is based on the **Dense Multiplicative Weights Update (MWU)** framework:

- MWU is run over the constraint space to produce a solution satisfying most constraints.
- An **Exponential Mechanism** is used as the oracle to ensure privacy.
- For Covering SDP, a $\gamma$-net over the extreme rays of the positive semidefinite cone discretizes the feasible solution space to $\exp(r)$ elements.
- Result: at most $s = \Omega(r/\epsilon \cdot \text{polylog})$ constraints are violated, with violation magnitude $O(\text{OPT})$.

### 3. SCP under Low-Sensitivity Constraint Privacy

In this setting, the constraint matrices of adjacent databases are close under the $\ell_\infty$ norm. Algorithm 1 runs MWU over the variable space:

- Maintains a distribution element $x^t \in \mathcal{K}$, initialized to $e/r$.
- Each round uses the Exponential Mechanism to find an approximately maximally violated constraint.
- The **Generic Gaussian Mechanism** is applied to add noise to the returned constraint element.
- MWU updates are performed using the noisy loss: $x^{t+1} \propto \exp(-\sum_i \eta \hat\ell^i)$.
- The output is the average over all iterates: $\bar{x} = \frac{1}{T}\sum_t x^t$.

The constraint violation bound is:

$$\alpha = \widetilde{O}\left(\frac{\Delta_\infty^{1/2} r^{1/4} k^{1/4}}{\epsilon^{1/2}} \cdot \text{polylog}(r, 1/\beta, 1/\delta)\right)$$

Compared to the LP result, an extra $k^{1/4}$ factor appears, arising from the need to add Gaussian noise in the full $k$-dimensional space (whereas LP requires only element-wise Laplace noise).

### 4. Scalar Privacy and Objective Privacy

As variants of the low-sensitivity framework, the paper also addresses the settings where the right-hand side vector $b$ and the objective $c$ are privatized, following a similar algorithmic structure.

## Key Experimental Results

This is a purely theoretical work containing no numerical experiments. All main results are stated as theorems providing privacy and utility guarantees:

| Setting | Constraint Violation $\alpha$ | Privacy Guarantee |
|---------|-------------------------------|-------------------|
| High-sensitivity constraint privacy (Covering SCP) | $O(\text{OPT})$, at most $\Omega(r/\epsilon \cdot \text{polylog})$ constraints | $\epsilon$-DP |
| Low-sensitivity constraint privacy | $\widetilde{O}(\Delta_\infty^{1/2} r^{1/4} k^{1/4} / \epsilon^{1/2})$ | $(\epsilon,\delta)$-DP |
| Scalar/objective privacy | Similar to low-sensitivity constraint result | $(\epsilon,\delta)$-DP |

## Highlights & Insights

1. **Resolution of an Important Open Problem**: The paper provides the first differentially private SDP algorithm, resolving the open problem posed by Hsu et al. (ICALP 2014) that had remained open for over a decade.
2. **Unified Framework**: Through EJA, the paper provides a unified treatment of DP for LP, SOCP, and SDP, rather than designing algorithms on a case-by-case basis.
3. **Illuminating Impossibility Results**: The findings that perturbing only eigenvalues is insufficient for privacy, and that the $\ell_1$-norm Laplace mechanism fails in EJA, are highly instructive negative results.
4. **Noisy MWU Framework**: Injecting privacy noise into the oracle of MWU to form a noisy MWU is a paradigm generalizable to other settings requiring different error guarantees.
5. **$\gamma$-Net Technique**: Quantizing the extreme rays (primitive idempotents) of the positive semidefinite cone and general EJA via $\gamma$-nets may have independent value.

## Limitations & Future Work

1. **Basic Privacy Analysis Tools**: Only classical Gaussian mechanisms and Advanced Composition are used; tighter analyses via Rényi DP or the Moments Accountant are not exploited.
2. **Constraints Not Fully Satisfied**: In the high-sensitivity setting, the output solution violates a small number of constraints; in the low-sensitivity setting, an $\alpha$-level constraint relaxation is incurred. Whether all constraints can be satisfied exactly remains open.
3. **Extra $k^{1/4}$ Factor**: The low-sensitivity constraint privacy result incurs an additional $k^{1/4}$ factor compared to LP, stemming from the necessity of adding noise in the full-dimensional space. Whether this gap can be narrowed through more refined noise design is an interesting open direction.
4. **No Experimental Validation**: As a purely theoretical work, numerical validation on practical ML tasks (e.g., robust estimation, experimental design) is absent.
5. **Restricted to Simple EJAs**: The Covering SCP result is limited to simple EJAs; extensions to general EJAs (direct sum decompositions) remain to be developed.

## Related Work & Insights

| Work | Scope | Sensitivity Measure | Noise Mechanism |
|------|-------|---------------------|-----------------|
| Hsu et al. (2014) | LP | Multiple settings | Laplace / Exponential |
| Nguyen & Silberstein (2024) | LP | Scalar privacy | Laplace (exact constraint satisfaction) |
| Benvenuti et al. (2024/2025) | LP | Multiple settings | Chance constraints / reconstruction |
| **Ours** | **SCP (incl. LP/SOCP/SDP)** | **$\ell_1/\ell_2/\ell_\infty$** | **Generic Gaussian on EJA + Exponential** |

The key distinctions of this work are: (1) the scope is substantially extended from LP to SCP; (2) spectrally-aware sensitivity measures replace element-wise measures; (3) the algebraic structure of EJA enables a unified treatment.

The EJA-based Gaussian mechanism is transferable to other settings requiring privacy protection for matrix- or cone-constrained data, such as covariance matrix aggregation in federated learning. The paradigm of injecting privacy noise into optimization oracles rather than into the final solution may inspire algorithm design for other private optimization problems. This paper opens the door to DP-SDP, with further work to be pursued on concrete applications (robust estimation, experimental design, sparse PCA) and tighter bounds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Resolves a decade-long open problem; highly original framework)
- Experimental Thoroughness: ⭐⭐ (Purely theoretical; no experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; motivation well articulated)
- Value: ⭐⭐⭐⭐⭐ (Pioneers the DP-SCP direction; far-reaching impact)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sequentially Auditing Differential Privacy](sequentially_auditing_differential_privacy.md)
- [\[NeurIPS 2025\] Differentially Private High-dimensional Variable Selection via Integer Programming](differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)
- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](multi-class_support_vector_machine_with_differential_privacy.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[NeurIPS 2025\] Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy](spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)

</div>

<!-- RELATED:END -->
