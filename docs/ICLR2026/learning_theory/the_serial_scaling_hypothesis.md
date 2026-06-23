---
title: >-
  [Paper Note] The Serial Scaling Hypothesis
description: >-
  [ICLR 2026][learning_theory][Diffusion Model] This is a position + theoretical paper: the authors use computational complexity (the TC circuit class) to categorize machine learning problems into "efficiently parallelizable" and "inherently serial." They argue that key tasks such as reasoning, decision-making, and physical simulation belong to the latter. They prov
tags:
  - ICLR 2026
  - learning_theory
  - Diffusion Model
  - Reinforcement Learning
  - Scaling
date: 2026-05-08
content_hash: 3402a6d14bf4e8bf
---
# The Serial Scaling Hypothesis

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ObXB7KJn0B](https://openreview.net/forum?id=ObXB7KJn0B)  
**Project Page**: [serial-scaling-hypothesis.github.io](https://serial-scaling-hypothesis.github.io)  
**Code**: None (Position and theoretical paper, containing only new theorems and proofs)  
**Area**: Learning Theory / Computational Complexity  
**Keywords**: Serial Computation, Parallel Computation, Complexity Theory, Diffusion Models, Reinforcement Learning, Scaling

## TL;DR
This is a position + theoretical paper: the authors use computational complexity (the TC circuit class) to categorize machine learning problems into "efficiently parallelizable" and "inherently serial." They argue that key tasks such as reasoning, decision-making, and physical simulation belong to the latter. They prove for the first time that even if a diffusion model samples for thousands of steps, its computational depth remains constant, rendering it unable to solve inherently serial problems. Therefore, merely stacking parallel compute (wider networks, more GPUs) is destined to plateau; progress must rely on scaling **serial** computation.

## Background & Motivation

**Background**: Breakthroughs in machine learning over the past decade have relied almost entirely on "parallel scaling"—hardware has shifted from CPUs to massively parallel GPUs, architectures have moved from RNNs to highly parallelizable Transformers, and algorithms increasingly exploit parallel compute. Leading scaling law literature typically reports parameters and FLOPs as a single number, treats network "width" (parallelism) and "depth" (serial computation) as interchangeable.

**Limitations of Prior Work**: A class of problems remains stubbornly resistant to this approach—mathematical reasoning, physical simulation, and sequential decision-making. No matter how much parallel compute is added, performance remains stagnant unless the model is allowed more "serial steps" (greater depth, more iterations, or longer chains of thought). However, the realization that "certain problems require deep/sequential models" has remained within scattered theoretical results and has not been systematically integrated into the machine learning context, nor has the "parallel vs. serial" axis been identified as a primary distinction.

**Key Challenge**: Width and depth are not equivalent. A classic phenomenon in complexity theory states that one can trade shallow depth for exponential width, but exponential width is infeasible in terms of memory and computation. That is, for certain problems, a shallow and wide parallel model would require exponentially large weights—and consequently, exponentially large datasets—to solve them, which is practically unaffordable. Treating width and depth as a single interchangeable FLOPs figure masks this fundamental gap.

**Goal**: To rigorously define "which problems are efficiently parallelizable versus inherently serial" using complexity theory and, based on this, answer two sub-questions: (1) How many truly important ML problems fall into the "inherently serial" category? (2) Can current mainstream parallel architectures (Transformers, SSMs, Diffusion Models) actually solve these problems?

**Key Insight**: The authors leverage the threshold circuit complexity class $TC$ as a metric. A problem belongs to $TC$ if and only if it can be solved by circuits (or equivalent MLPs) of polynomial width and polylogarithmic depth. Such problems are considered "parallel"; those outside $TC$ are "inherently serial." This approach is promising because $TC$ has a natural correspondence with neural networks: constant-depth, polynomial-width, finite-precision MLPs, Transformers, and linear SSMs all collapse into $TC^0$ (constant depth).

**Core Idea**: Propose the **Serial Scaling Hypothesis (SSH)**—for critical ML problems like reasoning, decision-making, and dynamical system modeling, increasing parallel computation alone is insufficient; progress must come from scaling the amount of serial computation.

## Method

> This is a theoretical/position paper that does not "train a new model." The "method" is a **complexity argumentation framework**: it first defines the boundary between serial and parallel using $TC$ classes, then proves (or demonstrates) that key ML problems fall on the serial side, and finally provides the paper’s most rigorous new theorem—that the serial capacity of diffusion models is limited.

### Overall Architecture

The core proposition the paper addresses is: "Why does stacking parallel compute reach a ceiling, and why must progress rely on serial computation?" The argument progresses in three stages.

The first stage is **establishing the benchmark**: problems are abstracted as binary decision problems "input $N$ tokens, output yes/no." These are then categorized using the circuit complexity class $TC$. $TC^i$ refers to problems solvable by Boolean circuits (AND/OR/NOT + majority gates) with polynomial width and $O(\text{poly}(\log N))$ depth. These form a hierarchy $TC^0 \subseteq TC^1 \subseteq \cdots \subseteq TC \subseteq P$, where each inclusion is widely believed to be strict. By definition, **a problem $P \in TC$ is "parallel," otherwise it is "inherently serial."** This paper adopts the widely accepted assumption $TC \subsetneq P$, meaning there exist problems solvable in polynomial time that are inherently serial.

The second stage is **identifying reality**: it argues that many truly important problems in machine learning fall outside $TC$ (inherently serial), including cellular automata evolution, n-body mechanics/video prediction, sequential decision-making (RL), and mathematical reasoning. Their commonality is a "computational graph" whose length grows linearly with problem size, where each step depends on the previous one without the possibility of "skipping"—the essence of P-complete problems like the Circuit Value Problem (CVP).

The third stage is **debunking models**: fixed-depth MLPs, Transformers, and SSMs as **architectures** all collapse into $TC^0$; they can only break through $TC^0$ when paired with serial **inference processes** (autoregressive CoT, recurrence, repeated layers). The hardest blow dealt by this paper is proving that **even with infinite sampling steps, a diffusion model's computation remains stuck in $TC^0$**—it appears serial but only provides a constant amount of additional serial computation. Combined, these three stages conclude that using shallow/parallel models to solve inherently serial problems requires either impossible weight scales or exponential data.

### Key Designs

**1. The TC Complexity Metric: Defining "Serial vs. Parallel" via TC Boundaries**

The first pain point the paper addresses is the tendency to treat "width" and "depth" as a single FLOPs figure, leaving the idea that "some problems need deeper models" as a vague empirical observation. The authors introduce the threshold circuit class $TC$ as a rigorous metric: a decision problem belongs to $TC^i$ if and only if it can be solved by a circuit of polynomial width and $O(\text{poly}(\log N))$ depth. Parallelism is equivalent to the existence of sub-linear depth circuits/MLPs. This yields a clean dichotomy: $P \in TC$ are parallel problems, and $P \notin TC$ are inherently serial problems (Def. 2.2), adopting the complexity theory belief that $TC \subsetneq P$ (Assumption 2.3).

This metric is effective because it maps one-to-one with neural network depth: a problem is in $TC$ if and only if it can be solved by a polynomial-width, polylog-depth MLP, and in $TC^0$ if it can be solved by a **constant-depth** MLP. Thus, the engineering question of "how deep should a network be" is translated into a provable proposition of "which $TC$ layer the problem falls into," providing a rigorous basis for the depth-width tradeoff argument.

**2. Catalog of Inherently Serial Problems: Mapping Real-world Tasks to P-complete**

Definition alone is insufficient; one must answer if such problems exist in reality. The authors provide a list of inherently serial problems using a unified criterion: if a problem of size $O(n)$ can be written as a linear computational graph of length $O(n)$ where each node depends only on previous nodes and takes constant time, it is likely inherently serial—the intuition behind **P-complete** problems (prototyped by CVP). The list includes: Cellular Automata (Rule 110 is Turing complete, and many CA problems are P-complete); n-body mechanics (Fredkin-Toffoli's "billiard-ball computer" can simulate any Turing machine, thus predicting particle positions at $t=T$ with finite precision is inherently serial, as is video prediction); Sequential decision-making and mathematical QA (GSM8K can be formalized as a dependency graph calculated by topological order, corresponding to P-complete arithmetic circuit evaluation).

**3. The Upper Bound of Diffusion Models' Serial Capacity: TC⁰ Backbones remain TC⁰ even at Infinite Steps**

This is the paper's most critical and counter-intuitive new theorem. Diffusion models denoise step-by-step, appearing highly "serial," leading many to assume they provide scalable serial computation. However, the authors prove: **if a problem can be solved with high probability by a diffusion model using a $TC^0$ backbone under infinite denoising steps, then the problem itself is in the parallel class $TC^0$** (Theorem 4.1). The proof builds on existing results—the reverse diffusion process converges to the target distribution $p_0$ at a rate of $TV = O(d/T)$ (where $TV$ is total variation, $d$ is the intrinsic dimension of $x_{N+1}$, and $T$ is the number of steps). Since the backbone is constant-depth and convergence requires only polynomial steps to reduce error arbitrarily, the entire sampling process can be collapsed back into a constant-depth circuit.

This clarifies the pain point: "more iterations $\neq$ greater serial depth." Diffusion models contribute only a **constant amount** of additional serial computation, fundamentally differing from CoT, which accumulates serial compute. This explains empirical phenomena: image generation saturates at ~300 steps, depth estimation shows little difference between 5 and 100 steps, and language modeling perplexity is similar at 32 vs 1024 steps.

**4. The Fourfold Implications of SSH: Translating Theory into Decisions**

The final design translates theoretical deductions into actionable advice. For **practitioners**: solving inherently serial problems with shallow/parallel models requires exponential weights and data, explaining poor generalization. For **task/benchmark designers**: serial problems can be rewritten as coarser/approximate versions to reduce serial depth, and benchmarks should categorize inherently serial problems separately. For **model designers**: recurrent structures may be needed to increase serial computation, despite training difficulties like gradient variance. For **hardware designers**: future progress requires not just more parallelism, but serial computation hardware with high clock speeds, low latency, and low data movement overhead (e.g., Cerebras CS-3's wafer-scale design).

## Key Experimental Results

> This paper does not train new models or conduct original experiments; its "evidence" comes from reusing existing work and new theorems. Three sets of empirical evidence supporting SSH are summarized below.

### Main Results (Empirical Evidence)

| Domain | Comparison | Key Finding | Source |
|------|------|----------|----------|
| Math Reasoning QA | Serial scaling (longer CoT) vs. Parallel scaling (majority voting) | For the same token budget, longer reasoning chains consistently outperform majority voting across AMC/MATH/AIME/Olympiad difficulties. | Aggarwal & Welleck, 2025 |
| Science QA | Serial vs. Parallel | On GPQA Diamond, serial scaling continuously improves accuracy (limited only by context window), while parallel voting saturates quickly. | Muennighoff et al., 2025 |
| Games (Hex) | Increasing MCTS expansion nodes | Performance rises monotonically with MCTS nodes across all training scales; perfect play is only reached via test-time MCTS. | Villalobos & Atkinson, 2023 |

### n-depth vs. Width Comparison

| Task | Configuration | Result | Note |
|------|------|------|------|
| Humanoid (Control) | 8 layers deep / 256 wide **vs** 4 layers shallow / 4096 wide | The 2× deeper network beats the 16× wider network. | In tasks with the largest observation/action spaces, serial depth is more valuable than parallel width. |
| Ant Big Maze / Arm Push | Various widths + depth 8 / 32 | Deepening networks consistently outperforms widening in locomotion/manipulation. | Even in model-free RL, serial computation cannot be replaced by parallelism. |

### Key Findings
- **Three independent domains (Math, Science, Games, RL Control) provide consistent evidence**: Given a fixed budget, spending compute on "serial" (longer chains, more MCTS expansions, deeper networks) is more effective than on "parallel" (voting, wider networks).
- **The "Early Saturation" of diffusion steps is theoretically explained**: Saturation in image generation (300 steps), depth estimation (5 steps), and language modeling (32 steps) is attributed to the constant effective computational depth of diffusion.
- **Parallel compute cannot replace serial compute**: In RL, parallel aggregation only reduces variance/accelerates convergence; unbiased return estimation still requires serial computation, otherwise the optimal policy is not guaranteed.

## Highlights & Insights
- **Elevating "Width vs. Depth" from an engineering preference to a provable proposition**: Using the $TC$ boundary provides a rigorous criterion for when a problem *must* be deep, accompanied by an actionable "sniff test" for practitioners (identify CVP-style linear dependencies).
- **The Diffusion Theorem is an "Aha!" moment**: While most assume step-by-step denoising equals serial computation, the proof that $TC^0$ backbones remain $TC^0$ even at infinite steps—validated by empirical "early saturation"—cleanly decouples "iterations" from "serial depth."
- **Insightful Contrast between CoT and Diffusion**: While both involve "multiple steps," CoT truly accumulates serial compute (breaking $TC^0$), whereas diffusion only adds constant depth. This suggests that when evaluating any "test-time scaling" method, one must ask if it adds true serial depth or merely constant iterations.

## Limitations & Future Work
- **Inherent Limitations of Worst-case Complexity**: All "inherently serial" conclusions are worst-case; specific real-world instances might be parallelizable or approximately solvable—SSH gives a theoretical ceiling, not a guarantee that every instance is difficult.
- **Scope Restricted to $P$**: For $NP \setminus P$ problems, difficulty is dominated by exponential computation rather than serial computation; SSH is most relevant for "practically hard" polynomial-time problems.
- **Diffusion Theorem Depends on Strong Assumptions**: Assumes a $TC^0$ backbone, fixed intrinsic dimension, and sufficient training with a score-matching objective. If the solution space dimension grows with scale or objectives change, the theorem may not apply.
- **Lack of Original Controlled Experiments**: All empirical evidence is cited from others; direct comparisons across different tasks/settings require caution regarding budget definitions.

## Related Work & Insights
- **vs. Merrill & Sabharwal's "parallelism tradeoff" hypothesis**: They argue parallel models cannot solve inherently serial problems; this paper extends that to real tasks (RL, QA) and fills in the missing puzzle piece regarding diffusion models.
- **vs. Classic Complexity Theory (Depth-width tradeoffs, P-completeness)**: These ideas are mature in complexity circles; the contribution here is mapping them onto the **machine learning context** to identify which popular tasks hit these serial boundaries.
- **vs. Chain-of-Thought Theoretical Analysis**: Existing work proves CoT extends Transformers' serial capacity; this paper applies the same lens to diffusion models to reach the opposite conclusion (diffusion does not add true serial depth).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to integrate diffusion models into the $TC^0$ framework and prove their limited serial capacity; "Serial Scaling Hypothesis" is a powerful unifying perspective.
- Experimental Thoroughness: ⭐⭐⭐ Position/theoretical paper with no original experiments; evidence relies on secondary citations.
- Writing Quality: ⭐⭐⭐⭐⭐ Makes hardcore complexity theory accessible to non-theoreticians; analogies and "What this means" sections are excellent.
- Value: ⭐⭐⭐⭐⭐ Provides actionable insights for models, tasks, and hardware, potentially influencing the direction of future architectures and compute investment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Critical Attention Scaling in Long-Context Transformers](critical_attention_scaling_in_long-context_transformers.md)
- [\[ICLR 2026\] Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling](best-of-majority_minimax-optimal_strategy_for_passk_inference_scaling.md)
- [\[ICLR 2026\] Intrinsic Entropy of Context Length Scaling in LLMs](intrinsic_entropy_of_context_length_scaling_in_llms.md)
- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)
- [\[ICLR 2026\] Mitigating the Curse of Detail: Scaling Arguments for Feature Learning and Sample Complexity](mitigating_the_curse_of_detail_scaling_arguments_for_feature_learning_and_sample.md)

</div>

<!-- RELATED:END -->
