---
title: >-
  [Paper Note] Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization
description: >-
  [ICML2026][Optimization][Bilinear saddle-point optimization] This paper demonstrates that perturbing the payoff of only one player in a bilinear zero-sum game preserves the original equilibrium under sufficiently small p…
tags:
  - "ICML2026"
  - "Optimization"
  - "Bilinear saddle-point optimization"
  - "asymmetric perturbation"
  - "last-iterate convergence"
  - "zero-sum games"
  - "NashConv"
date: 2026-05-08
content_hash: 69d491981c9237ce
---

# Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization

**Conference**: ICML2026  
**arXiv**: [2506.05747](https://arxiv.org/abs/2506.05747)  
**Code**: https://github.com/CyberAgentAILab/asymmetrically-perturbed-gda  
**Area**: Optimization / Game Learning  
**Keywords**: Bilinear saddle-point optimization, asymmetric perturbation, last-iterate convergence, zero-sum games, NashConv  

## TL;DR
This paper demonstrates that perturbing the payoff of only one player in a bilinear zero-sum game preserves the original equilibrium under sufficiently small perturbations. Based on this, the authors propose AsymP-GDA, which theoretically achieves linear last-iterate convergence and approaches the original equilibrium faster and more accurately than symmetric perturbation in normal-form and extensive-form game experiments.

## Background & Motivation
**Background**: The bilinear saddle-point problem $\min_{x \in X}\max_{y \in Y} x^T A y$ is a core formulation in zero-sum games, minimax optimization, and constrained optimization. Many learning algorithms guarantee average-iterate convergence to a Nash equilibrium through no-regret properties, but the actual sequence of strategies itself may cycle and fail to converge.

**Limitations of Prior Work**: Average-iterate convergence is often undesirable in large-scale models or games because it requires storing or mixing a large number of historical strategies. Methods such as Optimistic GDA, Extra-Gradient, and OMWU attempt to achieve last-iterate convergence, but they may lose stability in the presence of sampling noise, bandit feedback, or large-scale simulation environments.

**Key Challenge**: Payoff perturbation is another approach: adding a strongly convex regularizer to the payoff stabilizes the dynamics and enables last-iterate convergence. Conventional approaches typically apply symmetric perturbations to both players, but a fixed perturbation strength $\mu$ pushes the equilibrium away from the original game. To approximate the original equilibrium, $\mu$ must be very small or decayed over iterations, creating a conflict between accuracy and speed.

**Goal**: The authors aim to find a method that retains the stable convergence brought by perturbation without systematically shifting the target equilibrium. Ideally, the perturbed problem should be easier to solve, while the resulting strategies remain the minimax / maximin strategies of the original game.

**Key Insight**: The paper proposes a simple yet effective structural change: perturbing the payoff of only one player. To solve for the minimax strategy of player $x$, the objective becomes $\min_x\max_y x^T A y + \frac{\mu}{2}\|x\|^2$, while player $y$'s payoff remains linear.

**Core Idea**: Asymmetric perturbation introduces strong convexity to one side's objective to stabilize gradient dynamics. Simultaneously, it leverages the piecewise linear geometric structure of the original bilinear objective, ensuring that a sufficiently small perturbation does not alter the original minimax strategies.

## Method
The paper addresses a central question: why does "perturbing only one side" differ fundamentally from "perturbing both sides"? Intuitively, symmetric perturbation changes the preferences of both players simultaneously, so the perturbed equilibrium is usually only an approximation of the original. Asymmetric perturbation only makes the optimized side's objective strongly convex, while the opponent maintains the original linear best-response structure, allowing the "kinks" of the original objective function to continue locking in the same minimax solution.

### Overall Architecture
The input is a bilinear zero-sum game or an equivalent saddle-point problem where the strategy spaces $X, Y$ are polyhedra. The goal is to find the minimax and maximin strategies of the original game. The paper first defines the asymmetrically perturbed problem and proves that within a certain range of perturbation strength, the perturbed minimax strategy $x^\mu$ belongs to the original equilibrium set $X^*$.

At the algorithmic level, the authors propose AsymP-GDA. It is a lightweight modification of alternating GDA: when updating $x$, the perturbed gradient $Ay + \mu x$ is used; when updating $y$, the original gradient $A^T x$ is maintained. To obtain strategies for both players, one can run the mirrored version of the asymmetric perturbation for $x$ and $y$ separately.

For extensive-form games (EFGs), the paper uses the sequence-form representation to write imperfect-information zero-sum games as bilinear saddle-point problems and introduces a dilated Euclidean regularizer to derive AsymP-DGDA. This allows for computable last-iterate learning in sequential games such as Kuhn Poker, Leduc Poker, Liar's Dice, and Goofspiel.

### Key Designs
1. **Asymmetric Payoff Perturbation**:
    - **Function**: Introduces strong convexity to one player's objective without systematically shifting the original minimax strategies.
    - **Mechanism**: Solving $\min_x\max_y x^T A y + \frac{\mu}{2}\|x\|^2$ for $x$. Theorem 3.1 provides an upper bound on $dist(x^\mu, X^*)$; when $\mu$ is below a game-dependent threshold, the bound is 0, meaning $x^\mu$ falls exactly within the original equilibrium set.
    - **Design Motivation**: Symmetric perturbation alters both players' objectives, typically yielding only an approximate equilibrium for a fixed $\mu$. Asymmetric perturbation preserves the opponent's linear response, allowing the geometric "sharpness" of the original objective to override small perturbations.

2. **Alternating Updates in AsymP-GDA**:
    - **Function**: Transforms asymmetric perturbation into a first-order executable algorithm and achieves linear last-iterate convergence.
    - **Mechanism**: The algorithm executes $x^{t+1}=\Pi_X(x^t-\eta(Ay^t+\mu x^t))$, followed by $y^{t+1}=\Pi_Y(y^t+\eta A^T x^{t+1})$. Theorem 4.1 proves that if the learning rate satisfies certain conditions, the distance from the strategies to the perturbed equilibrium set decreases at a geometric rate.
    - **Design Motivation**: Compared to standard alternating GDA, the additional cost is only a single vector addition. However, because the $x$-side gains strong convexity, the dynamics no longer cycle around the equilibrium as easily as in standard GDA.

3. **Adaptive $\mu$ and Extension to Extensive-Form Games**:
    - **Function**: Avoids the need for prior knowledge of the allowable perturbation interval and scales the method to large sequential games.
    - **Mechanism**: A parameter-free version starts from an arbitrary $\mu_{init}$, solves the current asymmetrically perturbed problem, and checks if the original game's NashConv is below a target $\epsilon$. If not, $\mu$ is halved. In EFGs, sequence-form and dilated regularizers are used to form AsymP-DGDA.
    - **Design Motivation**: The threshold in Theorem 3.1 depends on game constants that are difficult to know in practice. Gradually decreasing $\mu$ exploits the "equilibrium invariance under small $\mu$" property while maintaining linear convergence for each subproblem.

### Loss & Training
This paper does not follow a deep learning training paradigm but focuses on optimization and game learning algorithms. The primary convergence metric is NashConv, which measures the exploitability of the current strategy. Normal-form game experiments compare the NashConv values along iteration curves; EFG experiments use sequence-form strategies and report last-iterate NashConv.

Theoretically, AsymP-GDA converges linearly to the equilibrium set of the asymmetrically perturbed game for any fixed $\mu > 0$. If $\mu$ lies within the invariance interval, the convergence point is also an equilibrium of the original game. The parameter-free version, by iteratively halving $\mu$, guarantees an $O(\log(1/\epsilon))$ iteration complexity to reach a NashConv $\le \epsilon$ without knowing game constants.

## Key Experimental Results

### Main Results
The experiments are divided into three groups: trajectories and NashConv in normal-form games, comparisons of AsymP-DGDA in EFGs, and supplemental comparisons with the CFR family of algorithms.

| Target Game | Compared Methods | Evaluation Metric | Main Result | Note |
| :--- | :--- | :--- | :--- | :--- |
| Biased Rock-Paper-Scissors / M-Ne | AsymP-GDA, SymP-GDA, GDA, OGDA | log NashConv / Trajectory | AsymP-GDA converges to the original equilibrium; SymP-GDA converges to a shifted point; GDA cycles. | Demonstrates the equilibrium-preserving advantage of asymmetric perturbation. |
| Different $\mu$ in BRPS | AsymP-GDA, SymP-GDA | Trajectory and Convergence Position | AsymP-GDA reaches the original equilibrium for $\mu \le 2.0$; starts shifting after $\mu=4.0$. | Supports the empirical observation of a wide invariance interval. |
| Five EFG tasks | AsymP-DGDA, SymP-DGDA, DMWU, DGDA, DOMWU, DOGDA | last-iterate NashConv | AsymP-DGDA achieves competitive or faster convergence and directly approaches the original equilibrium. | Covers Kuhn Poker, Leduc Poker, Liar's Dice, Goofspiel-4/5. |
| CFR Supplemental Comparison | AsymP-DGDA, CFR, CFR+, DCFR, LCFR | NashConv vs strategy updates | AsymP-DGDA outperforms CFR series on most games, with Leduc Poker being a notable exception. | CFR reports average-iterate; AsymP-DGDA reports last-iterate. |

### Ablation Study
| Design / Phenomenon | Key Metric | Note |
| :--- | :--- | :--- |
| Symmetric Perturbation | Solutions shift from original equilibrium; error scales with $\mu$. | Strong convexity-concavity provides stability but at the cost of modifying the objective. |
| Asymmetric Perturbation | $x^\mu \in X^*$ for sufficiently small $\mu$. | This is the core invariance conclusion of the paper. |
| AsymP-GDA | Linear last-iterate convergence to $Z^\mu$. | Geometric convergence rate is achievable even by perturbing only one side. |
| Parameter-free AsymP-GDA | $O(\log(1/\epsilon))$ complexity. | Halving $\mu$ avoids the need for game-dependent thresholds. |
| Symmetric decreasing-$\mu$ | Typical complexity is $\tilde{O}(1/\epsilon)$. | Requires solving a sequence of perturbed games that grow harder as accuracy increases. |
| AsymP-DGDA | Strong empirical convergence in EFGs. | Global smoothness for the dilated regularizer is difficult to guarantee theoretically. |

### Key Findings
- The key to asymmetric perturbation is not "smaller regularization," but "altering only one side." This structure keeps original minimax strategies invariant within a range of small $\mu$.
- The overhead of AsymP-GDA is minimal—it adds only a $\mu x$ term to standard alternating GDA—but it transforms GDA's rotational dynamics into convergent dynamics.
- The importance of the parameter-free algorithm lies in the practical unavailability of the threshold $\alpha / \max_x \|x\|$. Gradually halving $\mu$ is a simple yet theoretically sound strategy.
- In EFGs, AsymP-DGDA requires running the asymmetric process separately for each player to recover a strategy pair; the paper uses total strategy updates on the x-axis for fair comparison.
- Experiments indicate that symmetric perturbation exhibits "fast convergence to the wrong target," whereas asymmetric perturbation is better suited for scenarios where the original game must be solved precisely.

## Highlights & Insights
- A significant insight is clarifying the flaw of symmetric perturbation: it is not that it fails to converge, but that it converges to a game modified by regularization. This is often more insidious than non-convergence.
- Asymmetric perturbation is a minor modification with major theoretical consequences. Adding an $\ell_2$ term to only one side preserves both strong convexity stability and original equilibria, a design more elegant than complex optimistic corrections.
- The theoretical chain is quite complete: first proving equilibrium invariance, then proving linear last-iterate convergence for AsymP-GDA, and finally using adaptive-$\mu$ to eliminate unknown thresholds.
- This work provides insights for RLHF, adversarial training, and game solving: if regularization alters the objective, one should check if it is possible to smooth only the "optimized side" while leaving the other side's response structure intact.

## Limitations & Future Work
- The theory primarily covers bilinear two-player zero-sum games. The authors suggest extensions to two-player zero-sum Markov games are possible, but a complete proof is currently lacking.
- The invariance interval for perturbation depends on game constants, and the authors construct an example in Appendix E where this interval can be arbitrarily small. While the parameter-free algorithm addresses parameter tuning, it may still require many halving rounds in worst-case scenarios.
- Although AsymP-DGDA performs well in EFGs, the paper explicitly notes that it lacks a global convergence proof on par with AsymP-GDA because the smoothness constant of the dilated Euclidean regularizer may diverge near boundaries.
- Recovering a two-player equilibrium requires running the asymmetric algorithm separately for $x$ and $y$. While parallelizable, this is computationally more complex than a single bilateral update.
- Experiments are restricted to standard normal-form and extensive-form games. Future work should validate stability in real minimax learning tasks involving function approximation, sampling noise, or large-scale neural strategies.

## Related Work & Insights
- **vs OGDA / EG / OMWU**: Optimistic methods stabilize last iterates by predicting future gradients, while AsymP-GDA stabilizes dynamics by modifying the geometric structure of one side. The latter is closer to regularized projection optimization and might be more robust under noisy feedback.
- **vs Symmetric Payoff Perturbation**: Symmetric perturbation provides a strongly convex-concave structure but shifts the original equilibrium for fixed $\mu$. Asymmetric perturbation's advantage is exact preservation of original minimax strategies for small $\mu$.
- **vs Decreasing-$\mu$ Regularization**: Traditional methods require $\mu$ to decrease as a function of target precision, leading to slower complexity; the adaptive-$\mu$ in this paper only needs to cross a game-dependent threshold, after which each subproblem converges linearly.
- **vs CFR family**: CFR focuses on average-iterate convergence and is the classic approach for EFGs; AsymP-DGDA focuses on the last-iterate strategy itself, making it suitable for scenarios where maintaining historical average strategies is impossible or undesirable.
- **Insight**: In minimax learning, it is crucial to distinguish between "stabilizing the optimization process" and "modifying the objective function." Asymmetric regularization offers a reusable framework to decouple the two.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The modification of asymmetric perturbation is concise, and the combination of equilibrium invariance with linear last-iterate convergence is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers both normal-form and extensive-form games with supplementary comparisons to CFR; however, experiments are mostly trend curves without large-scale neural strategy tasks.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and strong motivation for theorems; notation and proofs in the appendix are dense, requiring background in optimization and game theory.
- Value: ⭐⭐⭐⭐☆ Highly valuable for saddle-point optimization, game learning, and applications requiring last-iterate strategies, especially for re-evaluating the side effects of regularization in minimax problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](../../ICLR2026/optimization/saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)
- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](../../ICLR2026/optimization/a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[AAAI 2026\] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](../../AAAI2026/optimization/ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](cost-aware_stopping_for_bayesian_optimization.md)

</div>

<!-- RELATED:END -->
