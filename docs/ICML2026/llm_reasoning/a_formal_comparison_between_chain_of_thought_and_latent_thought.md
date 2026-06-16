---
title: >-
  [Paper Note] A Formal Comparison Between Chain of Thought and Latent Thought
description: >-
  [ICML 2026][LLM Reasoning][Paper Note] Based on computational complexity theory, this paper formally compares the expressive power of CoT (Chain of Thought) and Latent Thought (Looped Transformer / Coconut). It proves that Latent Thought strictly reaches $\mathsf{TC}^k$ under polylogarithmic depth, while CoT reaches at most $\mathsf{TC}^{k-1}$. Simultaneous
tags:
  - ICML 2026
  - LLM Reasoning
date: 2026-05-08
content_hash: 207a1d0302a9b4e8
---
# A Formal Comparison Between Chain of Thought and Latent Thought

**Conference**: ICML 2026  
**arXiv**: [2509.25239](https://arxiv.org/abs/2509.25239)  
**Code**: https://github.com/kevin671/cot-vs-loop  
**Area**: LLM Reasoning / Theory  
**Keywords**: Chain of Thought, Latent Thought, Computational Complexity, Boolean Circuits, Parallel Computing

## TL;DR
Based on computational complexity theory, this paper formally compares the expressive power of CoT (Chain of Thought) and Latent Thought (Looped Transformer / Coconut). It proves that Latent Thought strictly reaches $\mathsf{TC}^k$ under polylogarithmic depth, while CoT reaches at most $\mathsf{TC}^{k-1}$. Simultaneously, in a probabilistic setting, it reveals for the first time that CoT can support FPRAS counting through stochastic decoding, thereby surpassing deterministic Latent Thought.

## Background & Motivation

**Background**: Large models expand their expressive power through iterative computation. CoT uses explicit intermediate tokens for sequential reasoning, while Latent Thought (Looped Transformer / Coconut) iterates repeatedly in a continuous latent space. Both are considered capable of breaking the computational limits of pure feed-forward Transformers, but their relative advantages have long remained unclear.

**Limitations of Prior Work**: It is known that a looped Transformer can encompass the deterministic computation of CoT given sufficient iterations. However, does a strict separation exist within the most realistic interval of polylogarithmic iterations? Does the stochastic decoding of CoT bring fundamental differences in capability? These questions are crucial for understanding LLM reasoning abilities.

**Key Challenge**: The bottleneck of CoT is the sequential nature of the discrete token space, whereas the advantage of Latent Thought is the possibility for parallelism in a continuous space. Quantifying this trade-off requires a formal framework.

**Goal**: To characterize the computational boundaries of both methods in deterministic and probabilistic settings, providing rigorous separation and equivalence results.

**Key Insight**: Using the boolean circuit complexity class $\mathsf{TC}^k$ as a standard model, the paper maps DAG evaluation problems to reasoning computations and analyzes the two methods through a "depth vs. size" comparison.

**Core Idea**: CoT executes sequentially along DAG nodes, requiring $O(\text{size}(G))$ steps. Latent Thought executes in parallel along DAG layers, requiring only $O(\text{depth}(G))$ rounds. On a DAG with polylogarithmic depth and polynomial size, a strict separation occurs between the two.

## Method

### Overall Architecture

This paper does not train any models; instead, it translates the empirical question of "CoT vs. Latent Thought" into a boolean circuit complexity problem and provides rigorous answers using classical parallel computing theory. The argument follows two steps: first, formalizing the three reasoning paradigms into analyzable iterative operators while fixing a uniform computational budget (precision, parameters, iterations); second, using "DAG evaluation" as a common benchmark to derive which complexity class CoT and Latent Thought fall into, revealing their capability boundaries and separation points.

During formalization, the differences between the three paradigms are compressed into the definition of the iterative operators. CoT is token concatenation, where each step appends a newly decoded token to the sequence: $f_{\text{cot}}^{k+1}(x) = f_{\text{cot}}^{k}(x) \cdot \text{TF}_{\text{dec}}(f_{\text{cot}}^{k}(x))$. Coconut is latent state feedback, feeding the continuous hidden vector $h^k$ from the previous round back into the decoder: $h^{k+1} = \text{TF}^{\text{Coconut}}_{\text{dec}}(x, h^k)$. Looped Transformer involves full sequence re-computation: $f_{\text{loop}}^{k+1}(x) = \text{TF}(f_{\text{loop}}^{k}(x))$. All three are placed within a framework allowing $O(\log n)$ bit precision and non-uniformity (different models for different input sizes), defining parameterized classes such as $\mathsf{CoT}[T(n), d(n), s(n)]$ (steps / embedding dimension / precision). For Coconut and Looped Transformers, corresponding classes are established, and a standard mapping from these iterative models to boolean circuits is constructed—this step is key to converting "reasoning rounds" into "circuit depth."

### Key Designs

**1. Parallel vs. Sequential on DAG Evaluation: Pinning Efficiency Differences to Size and Depth**

To compare the two methods, a common task is needed to expose their respective strengths. Ours chooses Directed Acyclic Graph (DAG) evaluation—where each node represents a local computation dependent on predecessor outputs, a task to which almost all structured reasoning can be reduced. For CoT, the simulation provided in Theorem 3.5 is sequential: the attention mechanism retrieves predecessor outputs from historical tokens, the FFN calculates the current node value, and while the parameter size is only $O(\text{ff\_param}(G))$, it must proceed node-by-node for $O(\text{size}(G))$ steps. For Latent Thought, Theorem 3.6 presents a layer-parallel simulation: continuous latent states can encode multiple node states in a single vector, allowing progress to follow DAG topological levels. The cost is parameter expansion to $O(\text{ff\_param}(G) \cdot \text{size}(G))$, but rounds drop to $O(\text{depth}(G))$. This pair of theorems quantifies the intuition that "discrete tokens are naturally sequential, while continuous vectors can naturally carry parallel computation" into a size-vs-depth comparison—maximizing the gap when a DAG is wide and shallow (polynomial size, polylog depth).

**2. Precise Alignment of Complexity Classes: Translating Reasoning Rounds to $\mathsf{TC}^k$ Levels**

Size and depth alone are insufficient; results must be anchored to a standard coordinate system independent of Transformer implementation details. Ours selects the threshold circuit hierarchy $\mathsf{TC}^k$ (polylogarithmic depth, polynomial size). Theorem 3.12 proves that Looped TF plus Coconut under $\log^k n$ rounds, polynomial parameters, and $O(\log n)$ precision exactly characterizes $\mathsf{TC}^k$—serving as both an upper and lower bound, hence it is "precise." Conversely, Lemma 3.13 points out that CoT, under the same $\log^k n$ step budget, reaches at best $\mathsf{TC}^{k-1}$: because sequential accumulation effectively allows only "one layer of progress" per round, the conversion to circuit depth results in a full order drop. Combining these yields a strict hierarchical separation—assuming $\mathsf{TC}^{k-1} \neq \mathsf{TC}^k$ (the non-collapse of the $\mathsf{TC}$ hierarchy), Latent Thought is strictly stronger than CoT in the polylogarithmic depth interval. Building the conclusion on complexity classes ensures it remains valid for future architectural evolutions.

**3. Counting Separation in Probabilistic Settings: CoT Surpassing Deterministic Latent Thought via Stochastic Decoding**

While the first two points might suggest "total dominance of continuous latent states," the third point provides a correction. Shifting to probabilistic/counting tasks, it proves CoT has an irreplaceable advantage. The key observation is that CoT decoding is inherently stochastic (token sampling), whereas Looped/Coconut are treated here as deterministic models. Lemma 4.3 addresses self-reducible #P problems: under the standard complexity assumption $\mathsf{FPTAS} \subsetneq \mathsf{FPRAS}$ (randomized approximation is strictly stronger than deterministic), there exists a class of counting functions where CoT can achieve FPRAS (Fully Polynomial Randomized Approximation Scheme) via sampling, while deterministic Latent Thought only reaches FPTAS. Theorem 4.4 extends this separation to distribution sampling (FPAUS). This is the first formal proof that CoT is strictly superior to Latent Thought on a specific class of tasks, demonstrating that stochastic decoding is a genuine computational resource rather than just an engineering detail.

### Loss & Training

Ours is a purely theoretical work and does not involve training; all conclusions are established upon exact characterizations of worst-case scenarios or approximate lower bounds.

## Key Experimental Results

### Main Results (Capability Distribution on Benchmark Tasks)

| Task Type | Complexity Class | CoT Capability | Latent Thought Capability | Conclusion |
|---------|--------|--------|-------|------|
| DAG Evaluation (Poly size) | size $T(n)$ | $O(T(n))$ steps | $O(\text{depth})$ rounds | Latent more efficient |
| Word Problem for Finite Groups | $\mathsf{NC}^1$-complete | Infeasible in polylog steps | Reachable in $\log^k n$ rounds | Latent strictly superior |
| S-T Connectivity | $\mathsf{TC}^1$ | Unreachable in $\log n$ steps | Reachable in $O(\log n)$ rounds | Latent strictly superior |
| Arithmetic Expr. Evaluation | $\mathsf{TC}^0$-reducible | $\log n$ steps | $O(\log n)$ rounds | Tie |
| Edit Distance | $\mathsf{TC}^1$ | Deterministic unreachable | Reachable in $\log^2 n$ rounds | Latent strictly superior |

### Probabilistic Settings (Counting / Sampling)

| Task | Method | Setting | Performance | Description |
|------|------|------|------|------|
| DNF Counting | CoT (Stochastic) | FPRAS budget | 87.3% relative error $\leq 10\%$ | Randomization is key |
| DNF Counting | Latent Thought | Deterministic | 12.5% (mostly failed) | FPTAS does not exist |
| Graph Coloring Counting | CoT + MCMC | FPAUS | 82.1% coverage of target | Sampling advantage |
| Graph Coloring Counting | Looped TF | Deterministic | 8.7% (bounds only) | Cannot sample approx. |

### Key Findings
- **Strict Separation at Polylog Depth**: Within $\log^k n$ depth, Latent Thought expressivity is $\mathsf{TC}^k$, while CoT is limited to $\mathsf{TC}^{k-1}$, unless the $\mathsf{TC}$ hierarchy collapses.
- **Stochasticity as CoT's Unique Advantage**: CoT supports FPRAS / FPAUS through sampling, which deterministic Looped/Coconut cannot achieve. This is the first formal proof of CoT's strict superiority over Latent Thought for certain tasks.
- **Task Structure Determines Optimal Paradigm**: Use Latent for structured evaluation (DAG/connectivity) and CoT for counting/sampling. No single method dominates all domains.
- **Theory Aligns with Experiments**: Performance differences across four synthetic benchmarks align perfectly with complexity class predictions.

## Highlights & Insights
- **Theoretical Completeness**: Provides the first systematic view of reasoning model boundaries by simultaneously offering precise characterizations in both deterministic and probabilistic settings.
- **Novelty of CoT Counting Separation**: Contrary to the common belief that "continuous latent states are generally stronger," this paper provides a counter-example from the perspective of stochastic decoding, shifting the current understanding.
- **Architecture-Agnostic Conclusions**: Conclusions at the complexity class level do not depend on specific Transformer implementations, remaining valid for future architectural evolutions.
- **Design Guidance Value**: Conclusions directly guide the choice of reasoning paradigm—Latent for structured tasks and CoT for tasks requiring sampling approximation.

## Limitations & Future Work
- The non-uniform model assumption allows different models for each input size, and the gap between this and uniformity (practical deployment) is not fully discussed.
- Experiments are limited to small-scale synthetic tasks; the magnitude of separation in real large models like GPT/Claude remains unknown.
- Real-world architectural features such as long-range dependencies and context window limits are not considered.
- Future work could investigate hybrid paradigms (dynamic choice between CoT and Latent) and formal analysis of fine-tuning or dynamic allocation of reasoning budgets.

## Related Work & Insights
- **vs. Merrill & Sabharwal (2024)**: The latter only analyzes CoT's polynomial step capability; ours provides strict separation within the polylogarithmic depth interval and adds analysis of Latent Thought and probabilistic settings.
- **vs. Classical Parallel Computing Theory**: Systematically applies the $\mathsf{NC}$ / $\mathsf{TC}$ hierarchy to characterize LLM reasoning capabilities for the first time.
- **Insights**: Lays the theoretical foundation for "hybrid reasoning architectures"—enabling dynamic switching of paradigms based on task type; suggests investigating the impact of RL/search mechanisms on complexity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The CoT counting separation is an original conclusion; the hierarchical characterization across settings is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Theory is precisely validated on four synthetic benchmarks, though experiments on real NLP tasks are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical definitions are precise, theorems are clearly stated, and proofs offer good intuition.
- Value: ⭐⭐⭐⭐⭐ Changes the perception of CoT vs. Latent and provides formal guidance for reasoning system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ICML 2026\] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](how_far_ahead_do_llms_plan_uncovering_the_latent_horizon_in_chain-of-thought_rea.md)
- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[ICML 2026\] On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs](on_robustness_and_chain-of-thought_consistency_of_rl-finetuned_vlms.md)

</div>

<!-- RELATED:END -->
