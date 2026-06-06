---
title: >-
  [Paper Note] Realizable Bayes-Consistency for General Metric Losses
description: >-
  [ICML 2026][Learnability] This paper provides a sharp characterization for the open problem of "when does a hypothesis class $\mathcal{H}$ admit a distribution-free…
tags:
  - "ICML 2026"
  - "Learnability"
  - "Metric Loss"
  - "Littlestone Tree"
  - "Universal Consistency"
  - "Gale-Stewart Game"
date: 2026-05-08
content_hash: 9709fc2182998b3e
---

# Realizable Bayes-Consistency for General Metric Losses

**Conference**: ICML 2026  
**arXiv**: [2605.03823](https://arxiv.org/abs/2605.03823)  
**Code**: None  
**Area**: Learning Theory / Metric Loss / Bayes Consistency  
**Keywords**: Learnability, Metric Loss, Littlestone Tree, Universal Consistency, Gale-Stewart Game

## TL;DR
This paper provides a sharp characterization for the open problem of "when does a hypothesis class $\mathcal{H}$ admit a distribution-free, strongly universally Bayes-consistent learning algorithm under general (possibly unbounded) metric losses" in the realizable case—the necessary and sufficient condition is that $\mathcal{H}$ does not contain a new type of "unbounded gap Littlestone tree" combinatorial obstruction.

## Background & Motivation

**Background**: Universal consistency is a classic goal in statistical learning theory—whether it is possible to design a distribution-free algorithm whose risk almost surely converges to optimal for any data distribution. For 0-1 classification, Bousquet et al. (2020) gave a complete characterization using Littlestone trees and Gale-Stewart games; multiclass was extended by Hanneke et al. (2023); real-valued regression (absolute loss) was characterized by Attias et al. (2024b) using scaled Littlestone trees. This line of work advances the "combinatorial obstruction ↔ unlearnability" paradigm.

**Limitations of Prior Work**: The above results all assume bounded losses or fixed scale, but many practical tasks (structured outputs, edit distance, cost-sensitive prediction) naturally operate on metric label spaces $(\mathcal{Y}, \ell)$, where $\ell$ may be unbounded. Under unbounded losses, even the seemingly strong realizability assumption cannot prevent catastrophic rare events with "small probability + large cost"—the learner may err on events with probability decaying as $1/n$, but if the loss scale grows faster than $n$, the expected risk still diverges to infinity.

**Key Challenge**: Under unbounded metric losses, "few errors in probability" and "small risk" become decoupled. Tsir Cohen & Kontorovich (2022) proposed the MedNet algorithm, but it requires the BIE (bounded-in-expectation) condition and left open the problem: what is the true necessary and sufficient condition for distribution-free Bayes consistency on $\mathcal{H}$? A naive conjecture is $R^* < \infty$, but Section 3 of this paper provides a counterexample: for $\mathcal{X} = (0,1)$, $\mathcal{Y} = \mathbb{N}_0$, and $\mathcal{H}$ restricted to $\{0, 2^{2k+1}\}$ on each $I_k = (2^{-k}, 2^{-(k-1)})$, $R^* = 0$ but no learner can be strongly consistent.

**Goal**: In the realizable case, to find a necessary and sufficient combinatorial characterization for strong universal Bayes consistency under general (possibly unbounded) metric losses, thus closing the open problem of Tsir Cohen & Kontorovich (2022) (realizable case).

**Key Insight**: Extend the idea of scaled Littlestone trees (Attias et al.) to metric loss, allowing the gap to diverge along the depth—i.e., "if the learner is forced to guess blindly between two labels whose distance grows ever larger in certain regions, then learning must fail." Combinatorially, this corresponds to a "non-decreasing $(\gamma_k)$-Littlestone tree with $\gamma_k \to \infty$."

**Core Idea**: Strong universal Bayes consistency in the realizable case $\iff$ $\mathcal{H}$ does not contain an infinite non-decreasing-$\gamma_k$ Littlestone tree (with $\gamma_k \to \infty$).

## Method

### Overall Architecture

Theorem 4.5 (Main Result): For Polish $(\mathcal{X}, \rho)$, $(\mathcal{Y}, \ell)$, compact parameter space $\Theta$, and $\mathcal{H}$ continuous in $\theta$, the following are equivalent: (1) There exists a distribution-free learning rule $\mathcal{A}$ such that for every realizable $\mu$, $R_\mu(h_n) \to 0$ a.s.; (2) $\mathcal{H}$ does not contain an infinite non-decreasing $(\gamma_k)$-Littlestone tree (with $\gamma_k \to \infty$). The proof is in two parts: lower bound (existence $\implies$ no consistent learner), upper bound (nonexistence $\implies$ explicit construction of a consistent learner).

### Key Designs

1. **Unbounded-gap Littlestone Tree**:

    - **Function**: Characterizes the "combinatorial obstruction" under metric loss. At depth $k$, internal nodes are labeled by instance $x_{k,i}$, and the two outgoing edges have labels $y_{k,i,1}, y_{k,i,2}$ with $\ell(y_{k,i,1}, y_{k,i,2}) \geq \gamma_k$, and every finite root-to-leaf path is realized by some $h \in \mathcal{H}$. The "realizable infinite" version further requires every infinite path to be realized by a single $h$.
    - **Mechanism**: Generalizes the binary label of the classic Littlestone tree to "label distance $\geq \gamma_k$," allowing $\gamma_k \uparrow \infty$. Lemma 4.3 (bridging lemma) shows that under compact $\Theta$ and continuity of $h$, realizability of finite prefixes implies realizability of infinite paths (using the finite intersection property of compact spaces).
    - **Design Motivation**: Previous scaled Littlestone trees (Attias et al.) had a fixed scale parameter, capturing "adversarial complexity" under bounded loss. The key insight here is that under unbounded loss, "adversarial power" alone is insufficient; one must also capture "adversarial power + potentially explosive scale"—thus, the gap diverges with depth, corresponding to the adversary being able to inflict ever-larger costs at deeper levels.

2. **Lower Bound: Constructing Catastrophic Distributions from the Tree**:

    - **Function**: Shows that if an unbounded-gap tree exists, then for any learner, there exists a realizable $\mu$ such that $\mathbb{E}_{S_n}[R(\mathcal{A}(S_n))] = \infty$ (for every $n$).
    - **Mechanism**: Since $\gamma_k \to \infty$, select depths $k_1 < k_2 < \cdots$ so that $\gamma_{k_m} \geq m^2$. Assign probability $p_m \propto 1/m^2$ to depth $k_m$ nodes. Toss independent fair coins $(B_k)$, with $B_{k_m}$ deciding which outgoing edge at $k_m$ is the true label—the tree's realizability ensures that this random labeling is realized by some $f_B^* \in \mathcal{H}$. For any fixed $n$, the sample $S_n$ can only touch $n$ of the $k_m$; for the remaining infinitely many $k_m$, $B_{k_m}$ remains a fresh fair coin to the learner. On these unseen depths, the learner must guess blindly between two labels at distance at least $m^2$; the triangle inequality gives expected loss at least $m^2/2$, multiplied by probability $p_m \cdot 1/2 = \Theta(1/m^2)$, contributing $\Theta(1)$. By the second Borel-Cantelli lemma, infinitely many "bad events" almost surely occur, so $R(h_n) = \infty$ a.s.
    - **Design Motivation**: This is an upgrade of the classic Littlestone argument (adversary forces the learner to guess blindly) to the metric loss setting—here, the cost of blind guessing diverges with $\gamma_k$, so not only does the classification error rate remain high, but the risk itself cannot be made finite.

3. **Upper Bound: Gale-Stewart Game + Dictionary Partition + Nested MedNet**:

    - **Function**: In the absence of an unbounded-gap tree, constructs an explicit strongly universally Bayes-consistent learner.
    - **Mechanism**: Four steps. **Step 1** translates "existence of infinite tree" into a Gale-Stewart game: the adversary plays $(\xi_k, \eta_{k,1}, \eta_{k,2})$ each round with $\ell(\eta_{k,1}, \eta_{k,2}) \geq \gamma_k$, requiring some $h$ to be consistent; the learner picks a side; the learner wins if the adversary has no legal move. No tree $\iff$ the learner has a measurable winning strategy $\sigma$ (using Bousquet's measurable strategy machinery). **Step 2**: Use the sample to drive $\sigma$: scan $S_\infty$, and whenever a move allows $\sigma$ to select $Y_i$ while maintaining realizability, advance the game; since $\sigma$ is a winning strategy, the game almost surely stops in finite steps at $K_\infty < \infty$. **Step 3**: Define the history-conditional label set $H_k(x)$ (Definition 6.1); Lemma 6.2 shows $\text{diam}(H_k(x)) \leq \gamma_{k+1}$, and Lemma 6.3 shows $\Pr(Y \in H_{K_\infty}(X)) = 1$ a.s. **Step 4**: Use a countable dense subset $\{q_j\}$ of $\mathcal{Y}$ to partition $\mathcal{X}$ into countable cells $C_{k,j} = \{x : j_k(x) = j\}$, with the true label in each cell lying in a bounded region $\mathcal{Y}_{k,j} = \{y : \ell(y, q_j) \leq 2\gamma_{k+1}\}$; run MedNet (Tsir Cohen & Kontorovich 2022's bounded-range learner) in each cell. Sample-split: one half drives the game, the other half trains the predictor.
    - **Design Motivation**: Directly handling unbounded metric loss is too difficult, but if the problem can be "localized"—showing that for each $x$, the true label almost surely falls in a region of diameter $\leq \gamma_{K_\infty + 1}$—then existing bounded-range algorithms can be used to ensure convergence. Gale-Stewart games and trees are standard tools in the universal learning framework; the key new ingredients are the history-conditional label set $H_k(x)$ and the dense set partition, which turn the "locally bounded but globally unbounded" dilemma into countably many independent bounded subproblems.

### Loss & Training

This is a theoretical paper, with no specific training loss. The algorithm is described in Section 6.5: after sample-splitting, the first half drives the Gale-Stewart game to stabilize at $K$, and the second half, after partitioning by $j_K(x)$, runs MedNet (restricted to $\mathcal{Y}_{K,j}$) in each bucket. The output is $\hat{f}_n(x) = \hat{f}_{n, j_K(x)}(x)$.

## Key Experimental Results

This is a theoretical paper with no experiments. The core quantitative results are two theorems:

### Main Results

| Result | Content |
|--------|---------|
| Theorem 4.5 (Main Characterization) | Realizable strong universal Bayes consistency $\iff$ no infinite non-decreasing-$\gamma_k$ Littlestone tree (with $\gamma_k \to \infty$) |
| Section 3 (Counterexample) | $R^* < \infty$ is insufficient for learnability—a construction with $\mathcal{X} = (0,1)$, $\mathcal{Y} = \mathbb{N}_0$, $\mathcal{H}$ taking $\{0, 2^{2k+1}\}$ on $I_k$ is realizable but any learner's risk is $\infty$ |

### Lower / Upper Bound Pairing

| Direction | Key Lemma / Theorem |
|-----------|---------------------|
| Lower Bound (Theorem 5.1) | Existence of tree $\implies$ for any $\mathcal{A}$, there exists realizable $\mu$ with $\mathbb{E}_{S_n}[R(\mathcal{A}(S_n))] = \infty$ |
| Bridging Lemma (4.3) | Compact $\Theta$ + $h$ continuous in $\theta$ $\implies$ realizability of finite prefixes $\implies$ realizability of infinite paths |
| Upper Bound (Theorem 6.5) | No tree $\implies$ explicit sample-split + Gale-Stewart + MedNet learner achieves $R_\mu(\hat{f}_n) \to 0$ a.s. |
| Diameter Lemma (6.2) | $\text{diam}(H_k(x)) \leq \gamma_{k+1}$, making "local boundedness" concrete |

### Key Findings
- Distributional conditions like $R^* < \infty$ (finite Bayes risk) and BIE (label space bounded in expectation) are **insufficient** to characterize learnability under metric loss—one must examine the combinatorial structure of $\mathcal{H}$ itself
- The compact parameterization assumption is mild but necessary: Appendix A.4 provides a counterexample showing that without it, "finite prefix realizability" and "infinite path realizability" can diverge
- The upper bound algorithm is explicit but heavy—requires a measurable winning strategy, sample split, dense set, and nested MedNet; practical computational complexity is not discussed
- The agnostic case (not requiring $\inf_h R_\mu(h) = 0$) remains unsolved; the authors list the main obstacles in Appendix A.5

## Highlights & Insights
- **"Unbounded gap" is the core new dimension distinguishing metric loss from 0-1 / real-valued regression**—previous scaled Littlestone trees treated scale as a fixed parameter; this work lets scale diverge with depth, coupling "adversarial power × catastrophe scale" into the combinatorial obstruction for the first time
- **Lower bound uses Borel-Cantelli + independent fair coins** to amplify "incomplete learning in probability" into "infinite risk," with a clean and elegant proof technique
- **Upper bound leverages history-conditional label sets + Polish space dense partition** as a bridge, nesting "local boundedness for each $x$" into "countable subproblems each learnable," providing a powerful template for handling unbounded target spaces, potentially generalizable to other settings
- **Closure of the realizable case** resolves half of the open problem posed by Tsir Cohen & Kontorovich (2022), and clearly identifies the remaining obstacles for the agnostic case

## Limitations & Future Work
- Only addresses the realizable case; agnostic extension involves the concept of "approximately realizable history," and the authors acknowledge that the current Gale-Stewart framework does not directly apply
- Compact $\Theta$ + $h$ continuous in $\theta$ is a mild but still technical assumption; removing it causes the two tree definitions to diverge
- The upper bound algorithm is theoretically valid but highly indirect: requires a measurable winning strategy (using the Jankov-von Neumann selection theorem), and is nearly impossible to run on real data
- No rates of convergence are provided, only almost sure convergence
- The Polish space assumption is standard but excludes certain pathological cases

## Related Work & Insights
- **vs Bousquet et al. (2020)**: Foundational work, characterized universal learning for 0-1 classification using Littlestone trees + Gale-Stewart games; this paper is a direct generalization to metric loss / unbounded label space
- **vs Attias et al. (2024a, b)**: Scaled Littlestone trees treat label gap as a fixed scale parameter for bounded regression; the key difference here is that $\gamma_k$ diverges with depth, capturing catastrophic behavior under unbounded loss
- **vs Tsir Cohen & Kontorovich (2022)**: They proposed MedNet for metric loss but required the BIE condition; this paper (1) provides a counterexample showing BIE / $R^*<\infty$ is insufficient, and (2) gives the true combinatorial characterization, reusing MedNet as a local learner
- **vs Brukhim et al. (2022)**: DS-dimension characterizes multiclass PAC learnability—this work follows the universal / strong learning route, relying on infinite trees rather than finite dimensions

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Solves half of a 4-year-old open problem and introduces the clean new combinatorial object of "unbounded-gap tree"
- Experimental Thoroughness: N/A (theoretical paper, no experimental dimension, counterexample construction is thorough)
- Writing Quality: ⭐⭐⭐⭐ Main theorem is stated concisely, lower bound argument is clear; upper bound's four-step process is long but has a roadmap and is readable
- Value: ⭐⭐⭐⭐ Important advance for the universal consistency line in learning theory, but the agnostic case gap limits practical impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)
- [\[AAAI 2026\] Designing Incident Reporting Systems for Harms from General-Purpose AI](../../AAAI2026/others/designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)
- [\[NeurIPS 2025\] DPA: A One-Stop Metric to Measure Bias Amplification in Classification Datasets](../../NeurIPS2025/others/dpa_a_one-stop_metric_to_measure_bias_amplification_in_classification_datasets.md)
- [\[AAAI 2026\] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer](../../AAAI2026/others/synweather_weather_observation_data_synthesis_across_multiple_regions_and_variab.md)
- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](../../NeurIPS2025/others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)

</div>

<!-- RELATED:END -->
