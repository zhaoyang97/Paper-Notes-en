---
title: >-
  [Paper Note] Test-Time Verification via Optimal Transport: Coverage, ROC, & Sub-Optimality
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper reformulates "test-time scaling with verifiers" as an Optimal Transport (sampling) problem. It provides a unified framework to precisely characterize the geometric relationship between **generator coverage, verifier ROC, and sampling algorithm sub-optimality**. It reveals a three-regime nature of the sub-opt
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: da69894fdca457e2
---
# Test-Time Verification via Optimal Transport: Coverage, ROC, & Sub-Optimality

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=BBDhQJh6GB](https://openreview.net/forum?id=BBDhQJh6GB)  
**Code**: https://github.com/marcellobullo/ot-resampling  
**Area**: LLM Inference / Learning Theory  
**Keywords**: Test-time scaling, Verifier, Optimal Transport, Sub-optimality, Youden's Index

## TL;DR
This paper reformulates "test-time scaling with verifiers" as an Optimal Transport (sampling) problem. It provides a unified framework to precisely characterize the geometric relationship between **generator coverage, verifier ROC, and sampling algorithm sub-optimality**. It reveals a three-regime nature of the sub-optimality-coverage curve (transport / policy improvement / saturation) and designs and analyzes sequential (SRS, SMC) and batch (BRS) sampling algorithms accordingly.

## Background & Motivation
**Background**: Test-time scaling is a crucial direction for enhancing LLM performance. "Verifier-based" approaches are particularly popular, using a binary reward mechanism (unit tests, ground truth, LLM-as-a-judge, etc.) to filter generated outputs. A typical pipeline consists of three components: a generator (reference LLM), a verifier, and a sampling algorithm (e.g., Best-of-N). The final performance is an aggregation of the properties of these three.

**Limitations of Prior Work**: Most existing theoretical works only characterize aggregate scaling laws (e.g., pass@N, policy divergence) and commonly assume the **verifier is perfect**—an assumption that rarely holds in practice. A few studies have begun to consider verifier imperfection but lack a framework that unifies the interaction between "generator characteristics $\times$ verifier error $\times$ sampling algorithm" and quantifies its geometric structure.

**Key Challenge**: Performance is jointly determined by three quantities: (i) the generator's **coverage** $\beta-1$ (the extent to which the optimal policy can deviate from the reference while remaining "covered"); (ii) the **ROC** of the verifier (characterized by True Positive Rate TPR and False Positive Rate FPR); and (iii) the **sub-optimality** of the sampling algorithm. These are not independent but coupled: relaxing coverage does not necessarily reduce sub-optimality; whether it does depends on verifier accuracy. Previous asymptotic curves fail to capture these precise dependencies.

**Goal**: To answer how closely verifier-based sampling can approach the induced optimal policy and how verification errors shape this approximation, a **precise** (rather than asymptotic) fine-grained analysis is required.

**Key Insight**: The authors observe that test-time verification is essentially a **sampling problem**: given generative access to a reference distribution $\mu$, the goal is to sample from a target distribution $\nu^\star$ defined by a ground-truth verifier $r^\star$, while only being guided by an approximately correct verifier $\hat r$. "Moving $\mu$ to $\nu^\star$" is naturally an **Optimal Transport** problem.

**Core Idea**: Use the geometric language of Optimal Transport to unify the characterization of test-time verification—decomposing sub-optimality into "Optimal Transport Cost (OTC) + Policy Improvement," thereby clarifying the interactions between coverage, ROC, and sub-optimality in one framework.

## Method

### Overall Architecture
The paper does not propose a new "engineering system" but provides an **analytical framework**: it formalizes test-time verification as a "transport plan design problem from a proxy distribution $\mu$ to a target distribution $\nu^\star$" and derives algorithmic sub-optimality and computational complexity within this framework.

Specific Setting: A prompt $x$ generates a response $y\sim\pi_{\mathrm{ref}}(\cdot\mid x)$ via a reference LLM, inducing a reference distribution $\mu$. The ground-truth verifier models verification as **set membership**—there exists a set of correct responses $S^\star(x)$, such that $r^\star(x,y)=\mathbf 1\{y\in S^\star(x)\}$. The optimal policy is constrained within a policy class "sufficiently covered" by the reference ($\ell_1$-type coverage constraint, equivalent to $\chi^2(\mu\Vert\nu)\le\beta-1$). Due to the binary reward structure, the paper provides a **closed-form solution** for the induced optimal policy (Theorem 2.1): the Radon-Nikodym derivative of the target with respect to the reference is:

$$\eta_r(y)=\begin{cases}\dfrac{1}{s_r}\wedge\dfrac{m_\beta(s_r)}{s_r} & y\in S_r\\[1mm] 0\vee\dfrac{1-m_\beta(s_r)}{1-s_r} & y\notin S_r\end{cases},\qquad m_\beta(s)\triangleq s+\sqrt{s(1-s)(\beta-1)},$$

where $s_r\triangleq\int_{S_r}r\,\mathrm d\mu$ is the mass of the reference policy falling into the verification set. The entire analysis is built upon this closed-form solution.

The sampling algorithm is characterized as a **coupling** $\rho$ between $\mu$ and $\nu^\star$. The transport cost is the Hamming cost $C(\rho)=\int\mathbf 1\{y\ne z\}\,\rho(\mathrm dy,\mathrm dz)$ (intuitively, the "rejection ratio required to sample the target"). Algorithm performance is measured by **sub-optimality**:

$$\mathrm{SubOpt}(A)\triangleq\int r^\star\,\mathrm d\nu^\star-\int r^\star\,\mathrm d\nu_A.$$

The challenge is that $\nu^\star$ is not directly accessible and must be guided by membership decisions from an imperfect verifier $\hat r$. Thus, verifier quality (TPR/FPR/Youden's Index $J$) inevitably affects sub-optimality.

### Key Designs

**1. Test-time Verification = Optimal Transport: Moving the Problem to an Analytical Geometric Space**

The fundamental step is identifying that test-time verification is the transport of the reference distribution $\mu$ to the target distribution $\nu^\star$. If the algorithm is too permissive, the induced distribution stays close to $\mu$, resulting in high sub-optimality; if it rejects too aggressively, sub-optimality decreases but at a high computational cost. The key challenge is designing a transport plan that balances "sample efficiency" and "induced sub-optimality." Under the Hamming cost, the authors provide the closed-form Optimal Transport Cost (Lemma 3.1):

$$\mathrm{OTC}(\beta)=\bigl(1\wedge m_\beta(s_{r^\star})\bigr)-s_{r^\star},$$

which is the "minimum rejection probability required to move from $\mu$ to $\nu^\star$," characterizing the intrinsic difficulty of the problem **itself** (independent of the algorithm).

**2. Three-Regime Geometry of Sub-optimality—Coverage: Relaxing Coverage is Not Always Better**

Sub-optimality is decomposed into (i) OTC (the intrinsic difficulty of moving $\mu\to\nu^\star$) and (ii) Policy Improvement (the extent to which the sampling algorithm offsets this cost). With a perfect verifier, policy improvement exactly equals the transport cost; however, verifier imperfection makes this unattainable. Sub-optimality is governed by the verifier ROC (especially Youden's Index $J=\mathrm{TPR}-\mathrm{FPR}$) and coverage $\beta$. The authors prove that as the coverage constraint relaxes, the curve exhibits three distinct regimes:

$$\mathrm{SubOpt}=\mathrm{OTC}(\beta)\cdot(1-\alpha J),\quad\text{where }\alpha\text{ takes piecewise values based on }\beta.$$

- **Transport Regime** (small $\beta$): Sub-optimality grows at $O(\sqrt\beta)$, dominated by OTC; policy improvement has little effect.
- **Policy Improvement Regime** (medium $\beta$): OTC begins to saturate. If the verifier is accurate ($J>0$ and $s_{\mathrm{ver}}\le s_{r^\star}$), sub-optimality can **decrease** with coverage; if $s_{\mathrm{ver}}\ge s_{r^\star}$, false positives dominate, and no improvement occurs.
- **Saturation Regime** (large $\beta$): OTC stabilizes at $1-s_{r^\star}$, and sub-optimality remains constant; increasing coverage further is useless.

**3. Sequential Sampling Algorithms: AiC Violates Constraints, while SRS and SMC are Compliant and Equivalent**

**AiC** (accept-if-correct) generalizes BoN to sequential settings: sample, check membership, and resample if it fails. However, it **ignores the coverage constraint** $\beta$ and violates it in low-coverage regions (Theorem 3.3). 

To fix this, the authors propose **SRS** (Sequential Rejection Sampling): it uses rejection sampling with $\eta_{\hat r}$ as the envelope, **always satisfying coverage constraints by construction**. They also propose **SMC** (Sequential Maximal Coupling) from the perspective of "minimizing transport cost." Surprisingly (Theorems 3.5–3.6), SRS and SMC have the **exact same expected number of proposals** $\mathbb E[\tau]=\tfrac1{s_{\mathrm{ver}}}(1\wedge m_\beta(s_{\mathrm{ver}}))$ and sub-optimality—the lack of direct access to the residual measure cancels out the potential gains of SMC.

**4. Batch Sampling Algorithms: BoN has a Maximum Batch Limit, while BRS is Globally Compliant**

**BoN** has non-zero sub-optimality even with a perfect verifier and has a **maximum usable batch size** $N_{\max}$ (Theorem 3.7): exceeding it violates the coverage constraint. To overcome this, the authors generalize SRS to **BRS** (Batch Rejection Sampling), which **satisfies coverage constraints for any $N$** (Theorem 3.9). Its sub-optimality decays exponentially with batch size: $\mathrm{SubOpt}(\mathrm{BRS})=\mathrm{OTC}(\beta)(1-\tfrac1M)^N$ (Theorem 3.10).

## Key Experimental Results

### Main Results (Sequential Protocol, Sub-optimality and Complexity)

| Phenomenon | SRS / SMC | AiC | Description |
|------|-----------|-----|------|
| Sub-optimality Curve | Exhibits full "$O(\sqrt\beta)$ rise $\to$ downward bend with $J$ $\to$ plateau" | Only matches in the saturation regime; **violates constraints** elsewhere | Matches Theorem 3.6 |
| Plateau Height | Determined by $s_{r^\star}$ and $J$ | Same as saturation regime | Higher $s_{r^\star}$ in larger models $\to$ lower residual sub-optimality |
| Complexity | Saturates after $O(\sqrt\beta)$ | Constant | SRS=SMC, confirming equivalence |

### Key Findings
- **The three-regime geometry is real**: Empirical curves show the predicted rise, bend, and plateau across $\beta$ regimes. The magnitude of the bend is proportional to verifier information ($J$).
- **SRS ≡ SMC**: Despite different derivations, they are identical in expected proposals and sub-optimality because the lack of residual access offsets SMC's theoretical advantages.
- **Model scale primarily affects plateau height**: Larger models have higher base accuracy ($s_{r^\star}$), leading to lower residual sub-optimality without changing the curve's fundamental shape.
- **SRS/BRS are superior in low-coverage regions**, while BoN is more practical in relaxed-coverage regions.

## Highlights & Insights
- **Reformulation as Optimal Transport**: Identifying test-time verification as moving $\mu$ to $\nu^\star$ naturally integrates coverage ($\chi^2$ radius), verifier error (ROC/Youden), and efficiency (Hamming cost) into a single geometric framework.
- **Youden's Index Modulation**: It refine the vague notion of "sub-optimality $\sim\sqrt{\text{coverage}}$" by showing that whether increasing coverage helps depends strictly on the verifier accuracy, with a clear phase transition.
- **Algorithm Guidance**: Provides clear engineering heuristics—use rejection sampling variants (SRS/BRS) for low-coverage/resource-constrained scenarios and BoN for lax-coverage scenarios.

## Limitations & Future Work
- **Theoretical Assumptions**: The analysis assumes knowledge of $s_{\hat r}$ (reference mass in the verification set), which may require estimation in practice.
- **Binary Membership Model**: Abstracting verification as $r^\star=\mathbf 1\{y\in S^\star\}$ is suitable for "objective" tasks (math, code) but may be limited for subjective or continuous scoring (LLM-as-a-judge).
- **Future Directions**: Extending the framework to non-Hamming costs, tree-search scaling, or adaptive/learnable verifiers.

## Related Work & Insights
- **vs. Huang et al. (2025a)**: They report sub-optimality scaling with the square root of coverage. This paper uses the same $\ell_1$ constraint but provides a **precise** decomposition via OT, showing that the square root scaling only holds in the transport regime and is modulated by ROC.
- **vs. Dorner et al. (2025)**: They refer to AiC-like strategies as "rejection sampling." This paper strictly distinguishes AiC from true rejection sampling (SRS) and proves that AiC violates constraints in certain regimes.
- **vs. Classical Best-of-N Analyses**: Most focus on aggregate pass@N curves. This paper places BoN in the OT framework to define its maximum batch size $N_{\max}$ and precise sub-optimality.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Slicing Wasserstein over Wasserstein via Functional Optimal Transport](slicing_wasserstein_over_wasserstein_via_functional_optimal_transport.md)
- [\[ICLR 2026\] Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling](expressive_power_of_implicit_models_rich_equilibria_and_test-time_scaling.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](../../ICML2026/learning_theory/on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)

</div>

<!-- RELATED:END -->
