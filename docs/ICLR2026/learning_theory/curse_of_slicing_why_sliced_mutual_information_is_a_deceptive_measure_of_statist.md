---
title: >-
  [Paper Note] Curse of Slicing: Why Sliced Mutual Information is a Deceptive Measure of Statistical Dependence
description: >-
  [ICLR2026][Learning Theory][Sliced Mutual Information] This paper systematically deconstructs the reliability of Sliced Mutual Information (SMI) as a scalable alternative to Mutual Information (MI). Through closed-form solutions, counter-examples, and extensive synthetic experiments, it demonstrates that SMI saturates prematurely, favors information redundancy over information content, and decays to zero in high dimensions. In some cases, it performs worse than simple correla…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Information Theory"
  - "Representation Learning"
  - "Sliced Mutual Information"
  - "Mutual Information Estimation"
  - "Statistical Dependence Measures"
  - "Redundancy Bias"
  - "Curse of Dimensionality"
date: 2026-05-08
content_hash: d66209f9e4b128a5
---

# Curse of Slicing: Why Sliced Mutual Information is a Deceptive Measure of Statistical Dependence

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=KxeBgh1zWr](https://openreview.net/forum?id=KxeBgh1zWr)  
**Code**: TBD  
**Area**: Learning Theory / Information Theory / Representation Learning  
**Keywords**: Sliced Mutual Information, Mutual Information Estimation, Statistical Dependence Measures, Redundancy Bias, Curse of Dimensionality

## TL;DR
This paper systematically deconstructs the reliability of Sliced Mutual Information (SMI) as a scalable alternative to Mutual Information (MI). Through closed-form solutions, counter-examples, and extensive synthetic experiments, it demonstrates that SMI saturates prematurely, favors information redundancy over information content, and decays to zero in high dimensions. In some cases, it performs worse than simple correlation coefficients, leading to systematically misleading conclusions when used to measure statistical dependence.

## Background & Motivation
**Background**: Mutual Information $\mathsf{I}(X;Y)=\mathsf{KL}[\mathbb{P}_{X,Y}\|\mathbb{P}_X\otimes\mathbb{P}_Y]$ is the "gold standard" for measuring non-linear statistical dependence between two random vectors. It is invariant to invertible transformations, zero if and only if variables are independent, and captures arbitrary non-linear relationships. It is widely used in generalization analysis, independence testing, feature selection, representation learning, and studying the information bottleneck in deep networks.

**Limitations of Prior Work**: In practice, MI must be estimated from finite samples, and its sample complexity grows exponentially with dimension (the curse of dimensionality), making reliable estimation impossible in high-dimensional settings. A popular workaround is to project high-dimensional data into low dimensions before estimation. The most prominent method is **Sliced Mutual Information (SMI)** proposed by Goldfeld et al., which computes the expected MI of one-dimensional (or $k$-dimensional) variables after random uniform projections $\theta, \phi$:

$$\mathsf{SI}_k(X;Y)=\int_{\mathrm{St}(k,d_x)}\int_{\mathrm{St}(k,d_y)}\mathsf{I}(\Theta^\mathsf{T}X;\Phi^\mathsf{T}Y)\,d\mu(\Theta)\,d\mu(\Phi).$$

SMI preserves the core property of being zero if and only if variables are independent, converges extremely fast (seconds vs. hours for neural MI estimators), and requires no additional optimization. Consequently, many papers use SMI as a "safe proxy" for MI to study neural networks, derive generalization bounds, audit differential privacy, and perform feature disentanglement.

**Key Challenge**: The community has largely assumed that SMI and MI move "in the same direction," interpreting increases or decreases in SMI directly as changes in statistical dependence. However, SMI introduces additional structure through **random projections**, and the resulting artifacts have never been systematically examined. Existing critiques (by Tsur, Fayad, etc.) focused only on the sub-optimality of random projections, proposing max-sliced or optimal-sliced variants, without realizing that the root problems go far beyond "non-optimal" projection.

**Goal**: This work systematically interrogates the extent to which SMI faithfully reflects statistical dependence. Does it saturate? What kind of information does it prefer? How does it behave in high dimensions? Can smarter slicing strategies save it?

**Key Insight**: Instead of evaluating SMI as an "approximation of MI" through convergence error, the authors treat it as an **independent statistical dependence measure**. They specifically examine whether SMI increases when the true MI increases monotonically. Any **divergent trend** between the two exposes a fundamental flaw in SMI. A key advantage of this approach is that MI has ground-truth values in Gaussian settings, allowing for clean controlled experiments.

**Core Idea**: By using analytical Gaussian closed-form solutions, carefully constructed counter-examples, and synthetic benchmarks across various distributions, the paper proves three categories of structural defects in SMI (saturation, redundancy bias, and high-dimensional decay), arguing that SMI is a deceptive measure of statistical dependence.

## Method

### Overall Architecture
This paper follows a pipeline of **theoretical analysis followed by empirical auditing**. It does not propose a new model but provides a chain of arguments to "debunk" a statistical measure across three layers:

1.  **Theoretical Analysis (Section 4)**: In Gaussian settings where closed-form solutions are available, SMI and MI expressions are compared side-by-side to see if their asymptotic behaviors align. This phase produces three core assertions—saturation, redundancy bias, and high-dimensional decay—supported by Lemmas and Propositions.
2.  **Synthetic Experiments (Section 5)**: Using four distributions with known ground-truth MI (correlated normal, correlated uniform, smoothed uniform, and log-gamma-exponential), the authors plot SMI against "normalized MI" $\mathsf{I}(X;Y)/d$ to verify if saturation and $1/d$ decay persist in non-Gaussian and high-dimensional settings (including images).
3.  **InfoMax Tasks + Reproduction Studies (Section 6/7)**: The authors replace the Goal of "maximizing MI" in representation learning with "maximizing SMI" to observe potential collapses. They also re-evaluate feature extraction and independence testing experiments from the original SMI papers under more realistic settings to test if their conclusions hold.

### Key Designs

**1. Saturation and Insensitivity: SMI caps early and ignores increasing dependence**
The authors use a joint Gaussian example (Lemma 4.1): $(X,Y)\sim\mathcal{N}\big(0,\big[\begin{smallmatrix}I&\rho I\\\rho I&I\end{smallmatrix}\big]\big)$. Here, MI is $\mathsf{I}(X;Y)=-\frac{d}{2}\log(1-\rho^2)$, while SMI is a generalized hypergeometric function $\mathsf{SI}(X;Y)=\frac{\rho^2}{2d}\,{}_3F_2(1,1,\tfrac32;\tfrac d2+1,2;\rho^2)$. When $\rho^2\to 1$ (deterministic relationship), MI correctly diverges to $+\infty$, but SMI is trapped by an upper bound depending only on dimension: $\lim_{\rho^2\to1}\mathsf{SI}(X;Y)=\psi(d-1)-\psi(\tfrac{d-1}{2})-\log 2\le\frac{1}{d-1}$. Plotting SMI against $\mathsf{I}(X;Y)/d$ shows it flattening into a horizontal line very early, becoming completely insensitive to further growth in dependence. Crucially, this upper bound depends on $d$, meaning SMI values across different dimensions are **not comparable**.

**2. Redundancy Bias: SMI prefers information repetition over information content**
A common "benign interpretation" suggests that SMI violates the Data Processing Inequality (DPI) because it prefers "linearly extractable" features. This paper refutes that with a counter-example. Proposition 4.4 and Remark 4.6 show that applying a linear transformation $\mathbf{1}\cdot e_1^\mathsf{T}$ (copying the same information to all coordinate axes) causes the true MI to **decrease** while SMI **increases**. SMI rewards "the same piece of information appearing redundantly across multiple axes" rather than the total amount of information. This **redundancy bias** can cause SMI to show a trend completely opposite to MI when dependence is spread across multiple components.

**3. The Other Face of the Curse of Dimensionality: SMI asymptotically decays to zero**
While SMI claims to be immune to the curse of dimensionality because its estimation error is low, the authors reveal that this is a mirage: the error appears low because **SMI itself decays to zero in high dimensions**. In Lemma 4.1, while $\lim_{d\to\infty}\mathsf{I}(X;Y)=+\infty$, $\lim_{d\to\infty}\mathsf{SI}(X;Y)=0$. Mechanistically, as $d$ increases, it becomes increasingly unlikely for random projection pairs $(\theta, \phi)$ to hit a "shared direction," causing the expected value to be diluted by near-independent pairs at a rate of roughly $1/d$.

**4. Optimized Slicing is Not a Cure: The scalability–utility trade-off**
Can max-sliced MI (mSMI) or optimal-sliced MI (oSMI) fix these issues? The authors argue they cannot. mSMI essentially truncates all but the top $k$ strongest dependence directions. If dependence is distributed across many components, mSMI will lose significant information and revert to redundancy bias. Furthermore, optimizing projections is computationally expensive, defeating the original purpose of slicing.

## Key Experimental Results

### Synthetic Benchmarks: Saturation and $1/d$ Decay
Using the KSG estimator and $10^4$ samples across four distributions, the theoretical claims were verified.

| Phenomenon | Experimental Setting | Key Observation |
|------|---------|---------|
| Premature Saturation | SMI vs. $\mathsf{I}(X;Y)/d$ | SMI flattens quickly; the higher the dimension, the lower the ceiling and faster the saturation. |
| $1/d$ Decay | Saturation value vs. $d$ (log-log) | 1-SMI and 2-SMI follow a $1/d$ linear decline, even for non-Gaussian distributions. |
| High-Dim Images | MI-preserving mappings | $k$-SMI still saturates and decays on complex high-dimensional synthetic image data. |

### InfoMax Tasks: SMI Maximization Leads to Collapse
In a Deep InfoMax task on MNIST, replacing the MI objective with SMI (using Donsker-Varadhan lower bounds) resulted in immediate representation collapse. In a Gaussian channel experiment, maximizing SMI favored **ill-conditioned** matrices (high redundancy) over well-conditioned ones, whereas MI favored the opposite.

### Key Findings
- The three defects of SMI (saturation, redundancy bias, high-dimensional decay) are interconnected: random projections in high-entropy data rarely find shared directions.
- These defects are not merely due to "sub-optimal projections"; even when projections are optimal (as in the Gaussian case), the expectation remains problematic.
- Using SMI to audit differential privacy or study neural information flow may produce results that are artifacts of dimensionality or redundancy rather than actual changes in dependence.

## Highlights & Insights
- **Power of Closed-Form Solutions**: Comparing the analytical limits of SMI and MI side-by-side ($\rho^2 \to 1$) provides an irrefutable "smoking gun" for saturation that is more convincing than any empirical plot.
- **Debunking the "Linear Preference" Myth**: The counter-example showing that copying information increases SMI while decreasing MI cleanly replaces the "linear feature preference" narrative with the more accurate "redundancy bias" label.
- **Reinterpreting Scalability**: The work transforms the "low estimation error in high dimensions" (a supposed strength of SMI) into a manifestation of "the value itself decaying to zero" (a fundamental weakness).

## Limitations & Future Work
- **Theoretical Scope**: Exact analytical expressions were primarily derived for the **Gaussian case**, though empirical results across four other distributions and high-dimensional images are robust.
- **Deconstruction vs. Reconstruction**: The paper is a "deconstructive" work; it proves SMI is unreliable but does not yet propose a scalable measure that avoids these pitfalls.
- **Future Directions**: Designing "data-dependent" slicing distributions that mitigate redundancy bias while maintaining fast convergence remains an open challenge.

## Related Work & Insights
- **vs. max-sliced MI (Tsur et al. 2023)**: These works assume the problem lies in the sub-optimality of random projections. This paper shows that even if you optimize, you fall back into a trade-off where you lose information if dependence is not low-rank.
- **vs. Original SMI Work (Goldfeld et al. 2022)**: The original works highlighted SMI’s scalability and zero-independence property. This paper re-evaluates their "strengths" (like DPI violation) as structural defects and refutes their optimistic conclusions regarding feature extraction and independence testing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first systematic audit to debunk the reliability of SMI.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong combination of theory and synthetic/realistic experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentation chain; abstract concepts are made highly intuitive.
- Value: ⭐⭐⭐⭐⭐ Vital "course correction" for researchers using SMI in representation learning and privacy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InfoBridge: Mutual Information Estimation via Bridge Matching](infobridge_mutual_information_estimation_via_bridge_matching.md)
- [\[ICLR 2026\] Transformers as Measure-Theoretic Associative Memory: A Statistical Perspective and Minimax Optimality](transformers_as_measure-theoretic_associative_memory_a_statistical_perspective_a.md)
- [\[ICLR 2026\] Information Estimation with Discrete Diffusion](information_estimation_with_discrete_diffusion.md)
- [\[ICLR 2026\] Tree-sliced Sobolev IPM](tree-sliced_sobolev_ipm.md)
- [\[ICLR 2026\] Slicing Wasserstein over Wasserstein via Functional Optimal Transport](slicing_wasserstein_over_wasserstein_via_functional_optimal_transport.md)

</div>

<!-- RELATED:END -->
