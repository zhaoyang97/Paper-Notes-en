---
title: >-
  [Paper Note] Realizable Bayes-Consistency for General Metric Losses
description: >-
  [ICML 2026][Learnability] This paper provides a sharp characterization for the open problem of when a hypothesis class $\mathcal{H}$ admits a distribution-free strong universal Bayes-consistent learning algorithm under g…
tags:
  - "ICML 2026"
  - "Learnability"
  - "metric losses"
  - "Littlestone tree"
  - "universal consistency"
  - "Gale-Stewart game"
date: 2026-05-08
content_hash: 9848c6ae078b7af4
---

# Realizable Bayes-Consistency for General Metric Losses

**Conference**: ICML 2026  
**arXiv**: [2605.03823](https://arxiv.org/abs/2605.03823)  
**Code**: None  
**Area**: Learning Theory / Metric Losses / Bayes Consistency  
**Keywords**: Learnability, metric losses, Littlestone tree, universal consistency, Gale-Stewart game

## TL;DR
This paper provides a sharp characterization for the open problem of when a hypothesis class $\mathcal{H}$ admits a distribution-free strong universal Bayes-consistent learning algorithm under general (possibly unbounded) metric losses in the realizable setting. The necessary and sufficient condition is that $\mathcal{H}$ does not contain a new combinatorial obstacle termed the "unbounded gap Littlestone tree."

## Background & Motivation

**Background**: Universal consistency is a classic objective in statistical learning theory—whether a distribution-free algorithm can be constructed such that its risk converges almost surely to the optimum for any data distribution. For 0-1 classification, Bousquet et al. (2020) provided a complete characterization using Littlestone trees and Gale-Stewart games; multiclass classification was extended by Hanneke et al. (2023); real-valued regression (absolute loss) was characterized by Attias et al. (2024b) using scaled Littlestone trees. This line of research progresses along the "combinatorial obstacle $\leftrightarrow$ unlearnability" paradigm.

**Limitations of Prior Work**: Existing results focus on bounded losses or fixed scales, yet many practical tasks (structured output, edit distance, cost-sensitive prediction) naturally occur in metric label spaces $(\mathcal{Y}, \ell)$ where $\ell$ may be unbounded. Under unbounded losses, the seemingly strong "realizability" assumption cannot suppress the catastrophic rare events of "low probability + high cost"—a learner might fail on an event with probability decaying as $1/n$, but if the loss scale grows faster than $n$, the expected risk still diverges to infinity.

**Key Challenge**: Under unbounded metric losses, there is a decoupling between "few errors in probability" and "low risk." Tsir Cohen & Kontorovich (2022) proposed the MedNet algorithm but required the BIE (bounded-in-expectation) condition, leaving an open problem: what is the true necessary and sufficient condition for distribution-free Bayes consistency on $\mathcal{H}$? A naive conjecture is $R^* < \infty$, but Section 3 of this paper provides a counterexample: with $\mathcal{X} = (0,1)$, $\mathcal{Y} = \mathbb{N}_0$, and $\mathcal{H}$ constrained to $\{0, 2^{2k+1}\}$ on each interval $I_k = (2^{-k}, 2^{-(k-1)})$, $R^* = 0$ holds, yet no learner can be strongly consistent.

**Goal**: In the realizable setting, identify the necessary and sufficient combinatorial characterization for strong universal Bayes consistency under general (possibly unbounded) metric losses, closing the open problem of Tsir Cohen & Kontorovich (2022) for the realizable case.

**Key Insight**: Extend the scaled Littlestone tree concept of Attias et al. to metric losses and allow the gap to diverge with depth—essentially, "if the learner is forced to guess blindly between two labels whose distance grows indefinitely in certain regions, they must lose." Combinatorially, this corresponds to a "non-decreasing $(\gamma_k)$-Littlestone tree with $\gamma_k \to \infty$."

**Core Idea**: Realizable strong universal Bayes consistency $\iff$ $\mathcal{H}$ does not contain an infinite non-decreasing-$\gamma_k$ Littlestone tree (where $\gamma_k \to \infty$).

## Method

### Overall Architecture

Theorem 4.5 (Main Result): Under a Polish $(\mathcal{X}, \rho)$, $(\mathcal{Y}, \ell)$, and $\mathcal{H}$ with a compact parameter space $\Theta$ where $h$ is continuous in $\theta$, the following are equivalent: (1) there exists a distribution-free learning rule $\mathcal{A}$ such that $R_\mu(h_n) \to 0$ a.s. for every realizable $\mu$; (2) $\mathcal{H}$ does not contain an infinite non-decreasing $(\gamma_k)$-Littlestone tree ($\gamma_k \to \infty$). The proof is split into two halves: the lower bound (existence $\implies$ no learning algorithm is consistent) and the upper bound (absence $\implies$ an explicit consistent learner can be constructed).

### Key Designs

1. **Unbounded-gap Littlestone Tree**:

    - **Function**: Characterizes the "combinatorial obstacle" under metric losses. Internal nodes at depth $k$ are labeled with instances $x_{k,i}$, and the two outgoing edges have labels $y_{k,i,1}, y_{k,i,2}$ satisfying $\ell(y_{k,i,1}, y_{k,i,2}) \geq \gamma_k$. Every finite root-to-leaf path must be realized by some $h \in \mathcal{H}$. The "realizable infinite" version further requires every infinite path to be realized by a single $h$.
    - **Mechanism**: Generalizes the binary labels of classical Littlestone trees to "label distance $\geq \gamma_k$" and allows $\gamma_k \uparrow \infty$. Lemma 4.3 (bridging lemma) proves that under the compact $\Theta$ + continuous $h$ condition, finite prefix realizability automatically implies infinite path realizability (using the finite intersection property of compact spaces).
    - **Design Motivation**: Previous scaled Littlestone trees (Attias et al.) used a fixed scale parameter, characterizing "adversarial complexity" under bounded losses. The critical insight of Ours is that under unbounded losses, "adversarial capability" alone is insufficient; one must capture "adversarial capability + scale explosion"—thus letting the gap diverge with depth, corresponding to an adversary capable of causing greater costs at deeper positions.

2. **Lower Bound Proof: Constructing Catastrophic Distributions from the Tree**:

    - **Function**: Proves that if an unbounded-gap tree exists, then for any learner, there exists a realizable $\mu$ such that $\mathbb{E}_{S_n}[R(\mathcal{A}(S_n))] = \infty$ (for every $n$).
    - **Mechanism**: Since $\gamma_k \to \infty$, depths $k_1 < k_2 < \cdots$ can be chosen such that $\gamma_{k_m} \geq m^2$. Probabilities $p_m \propto 1/m^2$ are assigned to nodes at depth $k_m$. Independent fair coins $(B_k)$ are tossed, where $B_{k_m}$ determines which outgoing edge at depth $k_m$ is the true label—tree realizability ensures such labels are indeed realized by some $f_B^* \in \mathcal{H}$. For any fixed $n$, the sample $S_n$ touches at most $n$ depths $k_m$, leaving infinitely many $k_m$ where $B_{k_m}$ remains a fresh fair coin to the learner. At these unseen depths, the learner guesses between labels distanced $\geq m^2$; the triangle inequality gives a conditional expected loss $\geq m^2 / 2$, which contributes $\Theta(1)$ when multiplied by the probability $p_m \cdot 1/2 = \Theta(1/m^2)$. By the second Borel-Cantelli lemma, "bad events" occur infinitely often almost surely, hence $R(h_n) = \infty$ a.s.
    - **Design Motivation**: This is the metric loss upgrade of the classical Littlestone argument (where the adversary forces the learner into a "blind choice")—the cost of the blind choice itself diverges with $\gamma_k$, so not only is the error rate non-vanishing, but the risk fails to even be finite.

3. **Upper Bound Proof: Gale-Stewart Game + Dictionary Partition + MedNet Nesting**:

    - **Function**: Constructs an explicit strongly universally Bayes-consistent learner given the absence of an unbounded-gap tree.
    - **Mechanism**: A four-step process. **First**, translate "existence of an infinite tree" into a Gale-Stewart game: the adversary provides $(\xi_k, \eta_{k,1}, \eta_{k,2})$ with $\ell(\eta_{k,1}, \eta_{k,2}) \geq \gamma_k$ that must be consistent with some $h$; the learner picks a side; the learner wins when the adversary has no legal moves. No tree $\iff$ learner has a measurable winning strategy $\sigma$. **Second**, drive $\sigma$ using samples: scan $S_\infty$ and advance the game when a legal move is found that selects $Y_i$ and maintains realizability; since $\sigma$ is a winning strategy, it terminates at $K_\infty < \infty$ almost surely. **Third**, define the history-conditional label set $H_k(x)$ (Definition 6.1). Lemma 6.2 proves $\text{diam}(H_k(x)) \leq \gamma_{k+1}$, and Lemma 6.3 proves $\Pr(Y \in H_{K_\infty}(X)) = 1$ a.s. **Fourth**, partition $\mathcal{X}$ into countable cells $C_{k,j} = \{x : j_k(x) = j\}$ using a countable dense subset $\{q_j\}$ of $\mathcal{Y}$. Each cell's true label lies in a bounded region $\mathcal{Y}_{k,j} = \{y : \ell(y, q_j) \leq 2\gamma_{k+1}\}$. Run MedNet in each cell. Sample-splitting is used: one half drives the game, the other learns the predictor.
    - **Design Motivation**: Handling unbounded metric loss directly is difficult, but if the problem can be "localized"—proving that for each $x$, the true label almost surely falls into a region of diameter $\leq \gamma_{K_\infty + 1}$—then existing bounded-range algorithms can achieve convergence. The Gale-Stewart game and trees are standard tools in the universal learning framework; the key new components are the history-conditional label set $H_k(x)$ and the dense set partition, which transform "locally bounded but globally potentially unbounded" scenarios into countable independent bounded subproblems.

### Loss & Training

A theoretical paper with no specific training loss. The algorithm described in Section 6.5: after sample-splitting, the first half drives the Gale-Stewart game to stabilize at $K$, and the second half runs MedNet (restricted to $\mathcal{Y}_{K,j}$) within buckets defined by $j_K(x)$. The output is $\hat{f}_n(x) = \hat{f}_{n, j_K(x)}(x)$.

## Key Experimental Results

Theoretical paper, no experiments. The core quantitative results are two theorems:

### Main Results

| Result | Content |
|------|------|
| Theorem 4.5 (Main Characterization) | Realizable strong universal Bayes consistenly $\iff$ no infinite non-decreasing-$\gamma_k$ Littlestone tree ($\gamma_k \to \infty$) |
| Section 3 (Counter-example) | $R^* < \infty$ is insufficient for learnability—constructed $\mathcal{X} = (0,1)$, $\mathcal{Y} = \mathbb{N}_0$, $\mathcal{H}$ taking $\{0, 2^{2k+1}\}$ on $I_k$, realizable but risk is $\infty$ for any learner |

### Ablation Study

| Direction | Key Lemma / Theorem |
|------|---------------------|
| Lower bound (Theorem 5.1) | Existence of tree $\implies$ for any $\mathcal{A}$, there exists realizable $\mu$ s.t. $\mathbb{E}_{S_n}[R(\mathcal{A}(S_n))] = \infty$ |
| Bridging lemma (4.3) | Compact $\Theta$ + continuous $h$ in $\theta \implies$ finite prefix realizability $\implies$ infinite path realizability |
| Upper bound (Theorem 6.5) | Absence of tree $\implies$ explicit sample-split + Gale-Stewart + MedNet learner achieves $R_\mu(\hat{f}_n) \to 0$ a.s. |
| Diameter lemma (6.2) | $\text{diam}(H_k(x)) \leq \gamma_{k+1}$, formalizing "local boundedness" |

### Key Findings
- Distributional conditions like $R^* < \infty$ (finite Bayes risk) and BIE (label space bounded in expectation) **cannot** individually characterize learnability under metric loss—one must examine the combinatorial structure of $\mathcal{H}$ itself.
- The compact parameterization assumption is mild but necessary: Appendix A.4 provides a counterexample showing that "finite prefix realizability" and "infinite path realizability" decouple without it.
- Although the upper bound algorithm is explicit, it is heavy—requiring a measurable winning strategy, sample splitting, dense sets, and nested MedNet; its practical complexity is not discussed.
- The agnostic case (where $\inf_h R_\mu(h) = 0$ is not required) remains unsolved, with major obstacles listed in Appendix A.5.

## Highlights & Insights
- **"Unbounded gap" is the core new dimension distinguishing metric loss from 0-1 / real-valued regression**—while previous scaled Littlestone trees treated scale as a fixed parameter, this work allows scale to diverge with depth, coupling "adversarial capability" with "catastrophic scale" for the first time.
- **The lower bound uses Borel-Cantelli + independent fair coins** to amplify "failure to learn in probability" into "infinite risk," resulting in a clean and elegant proof technique.
- **The upper bound treats history-conditional label sets and Polish dense set partitions** as a bridge, nesting "local boundedness for each $x$" into "learnable countable subproblems." This is a powerful template for handling unbounded target spaces that could be generalized to other settings.
- **Closure of the realizable case** places a definitive semicolon after the open problem of Tsir Cohen & Kontorovich (2022), explicitly identifying remaining hurdles for the agnostic case.

## Limitations & Future Work
- Only resolves the realizable case; agnostic extensions involve the concept of "approximately realizable history," and the author admits the current Gale-Stewart framework does not directly work.
- Compact $\Theta$ and continuous $h$ in $\theta$ are mild but technical assumptions; removing them would cause the two tree definitions to diverge.
- The upper bound algorithm is theoretically present but very indirect: it requires a measurable winning strategy (via the Jankov-von Neumann selection theorem), making it almost impossible to run on actual data.
- Rates of convergence are not provided, only a.s. convergence.
- The Polish assumption, while standard, excludes certain pathological cases.

## Related Work & Insights
- **vs Bousquet et al. (2020)**: Foundational work using Littlestone trees and Gale-Stewart for universal learning in 0-1 classification; Ours is a direct generalization to metric loss and unbounded label spaces.
- **vs Attias et al. (2024a, b)**: Scaled Littlestone trees characterize bounded regression by treating the label gap as a fixed scale; the core difference in Ours is the divergence of $\gamma_k$ with depth to capture disasters in unbounded losses.
- **vs Tsir Cohen & Kontorovich (2022)**: They proposed MedNet for metric loss but required the BIE condition; Ours proves BIE / $R^*<\infty$ is insufficient and provides a true combinatorial characterization, reusing MedNet as a local learner.
- **vs Brukhim et al. (2022)**: DS-dimension characterizes multiclass PAC learnability—Ours follows the universal / strong learning route, relying on infinite trees rather than finite dimensions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Solves half of a 4-year-old open problem and introduces the clean "unbounded-gap tree" combinatorial object.
- Experimental Thoroughness: N/A (Theoretical paper; the counter-example construction is sufficient).
- Writing Quality: ⭐⭐⭐⭐ Main theorems are concisely stated, and lower bound arguments are clear; the upper bound four-step process is long but includes a roadmap.
- Value: ⭐⭐⭐⭐ Significant advancement in universal consistency within learning theory, though the remaining agnostic case limits immediate practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](parsimonious_learning-augmented_online_metric_matching.md)
- [\[ICML 2026\] Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation](amortized_simulation-based_inference_in_generalized_bayes_via_neural_posterior_e.md)
- [\[ICML 2026\] Consistency Training Can Entrench Misalignment](consistency_training_can_entrench_misalignment.md)
- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)
- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)

</div>

<!-- RELATED:END -->
