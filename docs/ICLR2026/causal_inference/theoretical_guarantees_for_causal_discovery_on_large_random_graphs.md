---
title: >-
  [Paper Note] Theoretical Guarantees for Causal Discovery on Large Random Graphs
description: >-
  [ICLR 2026][Causal Inference][Causal Discovery] This paper introduces the first **finite-dimensional deviation concentration bounds** (rather than asymptotic consistency or worst-case bounds) for causal orientation using random single-variable interventions. On sparse Erdős–Rényi and generalized Barabási–Albert random graphs, the False Negative Rate (FNR) of orientation errors concentrates increasingly—or even vanishes—as the dimension $d$ increases…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "Causal Discovery"
  - "False Negative Rate (FNR)"
  - "Deviation Concentration Inequalities"
  - "Random Graph Models"
  - "Intervention Experimental Design"
date: 2026-05-08
content_hash: 0c6d7af867657e19
---

# Theoretical Guarantees for Causal Discovery on Large Random Graphs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=V7pT2ZRoTB](https://openreview.net/forum?id=V7pT2ZRoTB)  
**Code**: Included in supplementary materials (DiffIntersort, no independent public repo link)  
**Area**: Causal Inference / Causal Discovery Theory  
**Keywords**: Causal Discovery, False Negative Rate (FNR), Deviation Concentration Inequalities, Random Graph Models, Intervention Experimental Design

## TL;DR
This paper introduces the first **finite-dimensional deviation concentration bounds** (rather than asymptotic consistency or worst-case bounds) for causal orientation using random single-variable interventions. On sparse Erdős–Rényi and generalized Barabási–Albert random graphs, the False Negative Rate (FNR) of orientation errors concentrates increasingly—or even vanishes—as the dimension $d$ increases, proving that high dimensionality and heavy-tailed degree heterogeneity, often viewed as obstacles, inherently regularize causal discovery.

## Background & Motivation

**Background**: The goal of causal discovery is to recover the Directed Acyclic Graph (DAG) among variables from data. In scenarios like systems biology (gene regulatory network reconstruction) or neuroscience (brain functional connectivity), such problems involve hundreds or thousands of variables, making them naturally high-dimensional and typically dependent on single-variable interventions (e.g., gene knockouts) for orientation information.

**Limitations of Prior Work**: Existing theoretical analyses are almost exclusively from a **worst-case** perspective—focusing on the minimum number of experiments needed to recover the entire DAG (e.g., $O(\log d)$ sets of multi-variable interventions or $d-1$ atomic interventions). These rely on idealized assumptions: complete faithfulness, causal sufficiency (no latent variables), and adversarial optimal intervention design. Such bounds are overly conservative, failing to reflect structural characteristics of real networks (sparsity, degree heterogeneity) and **failing to characterize performance fluctuations between random instances**—telling only how bad the "worst case" is, not how much the error oscillates around the mean in typical cases.

**Key Challenge**: Practitioners need to know how error fluctuations measured in medium-scale systems scale to thousands of variables. This requires **deviation bounds** around the expectation, rather than upper bounds on the expectation itself, let alone worst-case bounds. However, prior work has not provided finite-dimensional deviation inequalities for causal orientation errors.

**Goal**: To characterize how the **False Negative Rate (FNR)** (the proportion of incorrectly oriented true edges relative to the total number of edges) concentrates around its expectation under weak assumptions (allowing latent confounding and random rather than optimal intervention selection) across common random graph families (ER and BA).

**Key Insight**: The authors utilize the InterSort scoring framework and the $\epsilon$-intervention faithfulness assumption proposed by Chevalley et al. (2025c) to express error as a function of "random intervention vectors / random graphs." The key observation is that **flipping a single intervention (or a single edge) only affects a small group of edges within its "neighborhood."** Consequently, the sensitivity (Lipschitz constant) of the error function to each input coordinate is bounded by the local structure of the graph. Once sensitivity is bounded, the McDiarmid / bounded difference inequality directly yields concentration.

**Core Idea**: Translate "graph structural properties (degree distribution, sparsity)" into "probability concentration rates of error" via Lipschitz sensitivity. This results in **dimension-adaptive, faithfulness-robust** deviation bounds, revealing that heavy-tailed degree structures actually make the error more concentrated.

## Method

### Overall Architecture

As a purely theoretical paper, the "Method" is a **proof pipeline from graph structure to probability concentration**, consisting of four steps:

1.  **Define Error Metric**: Use the topological order deviation $D_{top}(G,\pi)=\sum_{\pi(i)>\pi(j)} A^G_{ij}$ to count incorrectly oriented edges under a candidate causal order $\pi$. The optimal order $\pi_{opt}$ from the InterSort score is the maximizer. Let $f(I)=D_{top}(G,\pi_{opt}(I))$ be the misorientation count and $g(I)=f(I)/|E|$ be the normalized FNR.
2.  **Establish Lipschitz Sensitivity**: Prove that flipping any intervention indicator $I_k$ or edge indicator $E_{ij}$ only changes a small group of edges adjacent to that node/edge (its ancestor and descendant cones, or just direct parents/children in restricted cases). Thus, the per-coordinate difference constants are bounded by node degrees.
3.  **Apply Concentration Inequalities**: Use bounded differences → McDiarmid (dense ER, BA). For sparse ER, where the mean does not vanish, Warnke’s "typical bounded differences" are used with a degree upper bound $\deg_{\max}=O(\log d)$.
4.  **Calculate Rates per Graph Family**: Substitute parameters for three types of random graphs (Fixed, ER dense/sparse, generalized BA) to derive the typical deviation width of $g$ around its mean, identifying when it vanishes, remains constant, or grows.

The input is "graph distribution + intervention probability $p_{int}$," and the output is the "magnitude of FNR mean + deviation concentration rate." The three components (Metric definition ↔ Lipschitz sensitivity ↔ Rates per graph family) are integrated: lower sensitivity (controlled degrees) leads to tighter concentration.

### Key Designs

**1. $\epsilon$-intervention faithfulness + FNR Metric: Basing orientability on weak assumptions**

Classical faithfulness requires every d-separation in the graph to correspond to conditional independence in the joint distribution, which is easily violated with latent confounding. This work adopts $\epsilon$-intervention faithfulness (Assumption 1): for any $i\in I,\,j\in V$,
$$D\!\left(P^{C,(\emptyset)}_{X_j},\,P^{C,do(X_i:=\tilde N_i)}_{X_j}\right) > \epsilon \iff \text{there exists a directed path } i\rightsquigarrow j.$$
It only requires that intervening on node $i$ causes at least an $\epsilon$ marginal distribution shift in downstream $j$, **without involving conditional independence structures**, thus remaining compatible with marginalization and latent variables. Based on this, the score is defined as:
$$S(\pi,\epsilon,D,I,\cdots)=\sum_{i\in I,\,j\in V,\,\pi(i)<\pi(j)}\big[(D_{ij}-\epsilon)_+ + c\cdot d\cdot \mathbb{1}\{D_{ij}>\epsilon\}\big],$$
where the maximizer $\pi_{opt}$ gives the achievable orientation performance. The authors emphasize: the deviation bound depends only on graph structure, $p_{int}$, and reachability, **not on the specific form of the InterSort score**—any algorithm using single-node interventions and the same $\epsilon$-faithfulness is subject to the same bounds. Unlike previous work focusing on misorientation count $f$, this paper uses normalized FNR $g=f/|E|$, allowing direct comparison across different graph scales.

**2. Ancestor/Descendant Cone-based Lipschitz Sensitivity: Mapping graph structure to bounded difference constants**

Concentration inequalities require that changing one input coordinate only minimally shifts the function. Lemma 3 proves that flipping $I_k$ only affects edges where $k$ serves as evidence, with the change $c_k$ bounded by:
$$c_k \le |\mathrm{Anc}(k)| + |\mathrm{Desc}(k)|,$$
i.e., edges within the ancestor and descendant cones of $k$. In the restricted (parents-only) case (Lemma 4), this tightens to $c_k\le \deg_{in}(k)+\deg_{out}(k)$. Edge variables follow a similar logic (Lemma 5). A **critical premise** (Assumption 2 + Remark 6) is that $\pi_{opt}$ must be unique via deterministic tie-breaking; otherwise, $\pi_{opt}$ could jump when scores tie, causing $c_k$ to scale with $|E|$ and breaking concentration. This "degree-as-sensitivity" mapping bridges graph theory and probability concentration.

**3. Concentration Rates on Three Random Graphs: How high-dimensionality and heavy-tails "regularize" error**

Substituting Lipschitz constants into concentration inequalities yields the paper’s counter-intuitive conclusions. **Fixed Graph + Random Interventions**: McDiarmid directly gives
$$P(|g(I)-\mathbb{E}[g]|\ge t)\le 2\exp\!\Big(-\tfrac{2t^2|E|^2}{\sum_k c_k^2}\Big).$$
**Erdős–Rényi**: In the dense case ($p_e=\Theta(1)$), $\mathbb{E}[g]=\Theta(1/d)$ and the deviation width is $O(d^{-1/2})$—both the mean and fluctuations vanish. In the sparse case ($p_e=c/d$), the mean **does not vanish** ($O(1)$, Lemma 22), but Warnke’s typical bounded differences (excluding rare high-degree events where $\deg_{\max}>O(\log d)$) still yield concentration of $g$ at $O\!\big(\tfrac{\log d}{\sqrt d}\big)$. **Generalized Barabási–Albert**: With degree exponent $\gamma=2+\kappa/m$ and $\beta=1/(\gamma-1)$, Lemma 10 shows $\deg(i)\le m+Cd^{\beta}$ w.h.p., leading to:
$$P(|g(I)-\mathbb{E}[g\,|\,E]|\ge t)\le 2\exp\!\Big(-\tfrac{2t^2}{d^{\,2\beta-1}}\Big),$$
with deviation width $O(d^{\beta-1/2})$. When $\gamma>3$ (i.e., $\beta<1/2$), this **vanishes** as $d\to\infty$, proving that heavy-tailed, hub-dominated topologies regularize causal orientation by suppressing error variance. Conversely, when $\kappa=1$ ($\gamma=7/3<3$), the deviation grows at $O(d^{1/4})$.

## Key Experimental Results

The authors state that experiments are intended to **corroborate "concentration" trends, not to directly test finite-sample bounds**, as the InterSort score's exact optimal solution lacks a polynomial algorithm. Simulations use the differentiable relaxation **DiffIntersort** (Sinkhorn-based optimization, scalable to 2000 variables) as an approximation.

### Main Results
Simulations were conducted on three types of random DAGs with scale $d$ from 30 to 2000, repeated 10 times per configuration. The typical deviation width (IQR) of the normalized topological error $g(I,E)$ was reported.

| Graph Family | Density/Heterogeneity Parameter | Intervention Coverage $p_{int}$ | Observed IQR vs. $d$ |
| :--- | :--- | :--- | :--- |
| Erdős–Rényi | $p_e\in\{0.2,0.4,0.6\}$ | $\{0.25,0.5,0.75\}$ | Vanishes as $d$ increases; confirms theory most clearly |
| Sparse/scale-free ER | Avg. degree $c\in\{2,3,5\}$ | $\{0.25,0.5,0.75\}$ | Vanishes as $d$ increases |
| scale-free BA | $\kappa\in\{1.0,3.0,9.0\}$, $m=3$ | $\{0.25,0.5,0.75\}$ | $\kappa=9,3$ contract; $\kappa=1$ unstable with higher fluctuations |

### Key Findings
- **BA graphs distinguish refined theoretical predictions**: At $\kappa=9.0$ ($\gamma$ large, $\beta<1/2$), the deviation should vanish, which was experimentally confirmed. At $\kappa=3.0$, theory predicts it should approach a constant; experiments showed steady contraction. At $\kappa=1.0$ (heavy-tail regime), theory predicts a growth of $O(d^{1/4})$, and experiments indeed showed greater instability—consistent with weaker theoretical control.
- **Sparse ER provides the cleanest validation**: Deviation narrowness monotonically increases with dimension, supporting the core claim that high dimensionality concentrates FNR.
- **Deviation bounds are independent of $p_{int}$**: This is an inherent property of McDiarmid-type inequalities (acknowledged as a limitation). Experiments showed consistent curve shapes across different $p_{int}$ values, which mainly affected the mean rather than the concentration rate.

## Highlights & Insights
- **Overturning the "High-D + Heavy-Tail = Obstacle" intuition**: The paper proves hub-dominated heterogeneous structures suppress oral variance, making large-scale causal discovery **more stable** rather than harder—a reassuring conclusion for systems biology/neuroscience networks.
- **First finite-dimensional deviation inequality for causal orientation**: Filling the gap between worst-case bounds and asymptotic consistency, this work provides a "deviation width order" to guide budget-error tradeoffs in random intervention designs.
- **Transferable proof paradigm**: Translating "algorithm statistical difficulty" to "local graph degrees" via sensitivity provides a template applicable to other graph metrics (e.g., number of unoriented edges, I-MEC size).
- **Decoupling from specific algorithms**: The bounds depend on graph structure and $\epsilon$-faithfulness. This serves as a "performance upper bound under optimistic assumptions" and a benchmark for expected fluctuations in synthetic data.

## Limitations & Future Work
- **Theoretical Limitations**: $\epsilon$-intervention faithfulness still requires interventions to cause measurable distribution shifts, which may fail in high-noise/weak-effect systems. Random graph models are stylized abstractions lacking modular or hierarchical structures. The analysis assumes acyclicity and the availability of numerous interventions.
- **Experimental Limitations**: Simulations use DiffIntersort as a proxy. The gap between this approximation and the theoretical optimal score means experiments qualitatively support trends rather than quantitatively verifying constants.
- **Conceptual Limitations**: When comparing graph families, "smaller deviation" does not mean "smaller error" because the mean orders differ (e.g., dense ER mean vanishes while sparse ER mean is constant); conclusions must be read alongside the mean.
- **Future Directions**: Developing bounds that characterize dependence on $p_{int}$, extending to adaptive strategies and cyclic systems, and creating practical search algorithms that closer approximate the exact optimal score at scale.

## Related Work & Insights
- **vs. Katz et al. (2019)**: Also analyzed intervention Markov equivalence classes on ER graphs, but relied on causal sufficiency, complete faithfulness, and optimal intervention selection. They only provided limits and upper bounds for the mean; this work relaxes assumptions and introduces **deviation bounds**.
- **vs. Chevalley et al. (2025c)**: The direct predecessor. It provided InterSort scores and misorientation count $f$ expectation bounds. This work transitions to normalized FNR, adds deviation concentration, and extends results to generalized BA graphs.
- **vs. Worst-case Intervention Complexity (Eberhardt 2005/2006, etc.)**: Those works focused on the minimum interventions needed to recover the full graph adversarially. This work takes an **average-case + finite-dimensional concentration** perspective.
- **Insight**: Coupling "statistical difficulty" to "degree distribution" suggests that in large-scale experimental design, one can estimate FNR fluctuations in medium-scale systems and extrapolate using concentration rates to achieve target reliability with fewer experiments.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First finite-dimensional deviation bound for causal orientation; reveals regularization effects of heavy-tailed structures.
- Experimental Thoroughness: ⭐⭐⭐ Simulations cover three graph families up to 2000 variables, though qualitatively limited by the use of DiffIntersort as a proxy.
- Writing Quality: ⭐⭐⭐⭐ Logical progression from assumptions and lemmas to theorems is clear; summary tables are helpful.
- Value: ⭐⭐⭐⭐ Provides a principled budget-error tradeoff for large-scale intervention design, relevant to high-dimensional fields like systems biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ICLR 2026\] On the Identifiability of Causal Graphs with the Invariance Principle](on_the_identifiability_of_causal_graphs_with_the_invariance_principle.md)
- [\[ICLR 2026\] Learning Dynamic Causal Graphs Under Parametric Uncertainty via Polynomial Chaos Expansions](learning_dynamic_causal_graphs_under_parametric_uncertainty_via_polynomial_chaos.md)
- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](../../AAAI2026/causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)

</div>

<!-- RELATED:END -->
