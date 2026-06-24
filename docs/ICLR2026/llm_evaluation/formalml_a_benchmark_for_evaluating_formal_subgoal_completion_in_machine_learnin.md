---
title: >-
  [Paper Note] FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory
description: >-
  [ICLR 2026][LLM Evaluation][subgoal completion] This paper introduces **FormalML**, the first Lean 4 benchmark dedicated to "subgoal completion." By employing a self-developed `to_theorem` translation strategy, the authors automatically extract 4,937 proof fragment problems from formalization libraries of machine learning theory (Optimization + Probability). This benchmark systematically exposes the real-world shortcomings of current LLM provers in handling complex contexts…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "subgoal completion"
  - "Lean 4"
  - "theorem proving"
  - "ML theory"
  - "premise utilization"
date: 2026-05-08
content_hash: b788e68f5ac49501
---

# FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wCRZbspSZi](https://openreview.net/forum?id=wCRZbspSZi)  
**Code**: Open source (Codebase + Dataset, links provided in the paper)  
**Area**: LLM Evaluation / Formal Theorem Proving / Benchmark  
**Keywords**: subgoal completion, Lean 4, theorem proving, ML theory, premise utilization  

## TL;DR
This paper introduces **FormalML**, the first Lean 4 benchmark dedicated to "subgoal completion." By employing a self-developed `to_theorem` translation strategy, the authors automatically extract 4,937 proof fragment problems from formalization libraries of machine learning theory (Optimization + Probability). This benchmark systematically exposes the real-world shortcomings of current LLM provers in handling complex contexts, premise utilization, and efficiency.

## Background & Motivation
**Background**: LLMs have made remarkable progress in formal theorem proving. Models such as DeepSeek-Prover-V2 and Kimina-Prover, utilizing natural language-assisted Long-CoT, have achieved silver-medal performance on competition-level benchmarks like miniF2F. However, these benchmarks focus on "generating a complete proof from scratch" based on problems from Olympiads or undergraduate textbooks, which often feature simple contexts and do not rely on external premises.

**Limitations of Prior Work**: When mathematicians use LLMs as copilots for research-level proofs, the performance is far below expectations. As noted by the Equational Theories Project, "on most difficult cases, LLMs fail to provide useful suggestions beyond what human participants already have." In real workflows, experts first construct the proof skeleton and formalize the theorem Statement; the remaining tasks are short but non-trivial proof obligations scattered within the proof, often marked by `sorry` placeholders. Completing these gaps is where LLMs should be most useful, yet **no specialized benchmark exists** for this task.

**Key Challenge**: Subgoal completion and full proof generation have different difficulty structures. While individual proof fragments in the former are often short, they require the model to: (1) understand a long sequence of accumulated complex hypotheses and goals; (2) correctly invoke interdependent premise lemmas from local libraries and Mathlib; and (3) avoid "overthinking" with excessively long reasoning chains, which increases computational cost without improving scores. In other words, a good prover must **balance accuracy and conciseness**, an area where existing competition-oriented models struggle.

**Goal**: Construct a benchmark that mimics real-world "fill-in-the-blank" proof collaboration scenarios and use it to systematically evaluate mainstream provers to identify their fundamental bottlenecks.

**Core Idea**: **[Slicing subgoals from procedural proofs]** The authors observe that while human-written Lean 4 proofs are procedural tactic sequences, **one line of tactic usually corresponds to a complete reasoning step in the author's mind**. Thus, they designed the `to_theorem` strategy to record the proof state before and after a specific line (or segment) of tactics. By treating the "pre-state as hypotheses and the post-state as the goal," they automatically synthesize an independent theorem where the original tactic sequence serves as the ground truth. Adjusting the slice length allows for the creation of problems with varying difficulties.

## Method

### Overall Architecture
The construction of FormalML involves three steps: first, using the `to_theorem` strategy to slice Lean 4 procedural proofs into independent subgoal theorems at the "line-level"; second, extracting top-level theorems from the optimization library (Optlib) and probability library (FoML) while retaining lower-level lemmas as a premise pool; finally, categorizing 4,937 problems by proof length (difficulty) and premise requirements. Evaluation is conducted using Pass@K under low computational budgets to measure accuracy, supplemented by specialized experiments on premise utilization, difficulty stratification, and efficiency (EWA@K).

```mermaid
flowchart LR
    A[Optlib Optimization<br/>FoML Probability] --> B[Select Top-level Theorems<br/>Keep Lemmas as Premise Pool]
    B --> C[to_theorem Strategy<br/>Line/Segment Slicing]
    C --> D[4937 Subgoal Theorems<br/>JSON: Statement+Premise+Proof]
    D --> E[Difficulty Levels L1/L3/L5]
    D --> F[Premise Annotation<br/>Mathlib / Local Library]
    E --> G[Pass@K Main Evaluation]
    F --> H[Premise Utilization Study]
    D --> I[EWA@K Efficiency / Expert Iteration]
```

### Key Designs

**1. `to_theorem` Translation Strategy: Reifying procedural steps into independent theorems.** This is the technical foundation of the benchmark. The native `extract_goal` in Lean 4 only captures the overall goal of the current proof state and fails to capture fine-grained intermediate subgoals—whereas most existing libraries use procedural styles (using `rw`, `simp` to transform states rather than declarative `have` statements). `to_theorem` records the proof state **before and after** a selected tactic line, inserts pre-state hypotheses into the theorem premises, and sets the post-state as the new goal. The original tactics serve as the verifiable solution. For a lemma like `linear_gradient` with 9 lines, it can theoretically yield up to 9 single-line subgoals or 4 double-line subgoals. Slice length naturally becomes a "difficulty knob."

**2. Dual-Library Data Curation: Categorizing ML theory into Optimization and Probability.** The authors narrowly focused on machine learning fundamentals because "AI agents are increasingly automating scientific discovery, and automated theorem proving ensures the reliability of derivations." Optimization covers GD, SubGD, PGD, Nesterov acceleration, BCD, and ADMM via Optlib. Probability covers generalization bounds for Rademacher complexity, bounded difference inequalities, and McDiarmid's inequality via FoML, with added lemmas for Hoeffding, Bennett, and Bernstein inequalities. The final set includes 2,907 optimization problems and 2,030 probability problems, stored with full context (imports, namespaces, tactics, premises).

**3. Dual Annotation for Difficulty and Premises: Enabling stratified evaluation.** Since proofs are largely procedural, the authors use **proof length** as a proxy for difficulty, defining levels L1, L3, and L5 (L1 contains 3,924 problems; higher levels contain 1,013). Regarding premises, 1,547 problems require explicit premise calls, categorized by source (Mathlib vs. local) to support experiments on "premise utilization" and "difficulty decay."

**4. EWA@K Efficiency Metric: Factoring "verbosity" into the score.** Addressing the issue where Long-CoT models overthink short subgoal proofs, the authors propose Efficiency-Weighted Accuracy: $\text{EWA@}K = \text{Pass@}K \times \frac{100}{\text{Response Length}}$. This penalizes models that consume massive tokens for simple proofs, highlighting models that are "accurate but concise" (e.g., STP) while Long-CoT models fall behind due to high token counts.

## Key Experimental Results

### Main Results (Pass@K, Selected Models)

| Category | Model | Budget | Pass@1 | Pass@32 (All) |
| :--- | :--- | :--- | :--- | :--- |
| Automated Tactic | aesop | - | - | 43.29 |
| Automated Tactic | Ensemble (All Combined) | - | - | 48.67 |
| BFS Tree Search | BFS-Prover (7B) | 8×50 | - | 25.31 |
| Whole-proof Gen | **STP (7B)** | 32 | **26.96** | **63.21** |
| Whole-proof Gen | DeepSeek-Prover-V2 (noCoT) | 32 | 16.85 | 62.06 |
| Whole-proof Gen | Leanabell-Prover (7B) | 32 | 24.10 | 58.07 |
| Whole-proof Gen | DeepSeek-Prover-V2 (CoT) | 32 | 18.43 | 39.50 |

- **Highest Pass@1 is only 26.96% (STP)**, far below the threshold for a practical "mathematician's assistant"; Pass@32 peaks at 63.21%.
- **BFS Tree Search is not advantageous** despite higher compute costs (<30%), suggesting that step-by-step interaction with Lean is inefficient for search.
- **aesop (single tactic) outperforms some LLMs** at low budgets, but is overtaken by LLMs at higher budgets. Even an ensemble of all automated tactics remains inferior to LLMs overall.

### Premise Utilization (Pass@32, Gain relative to M=0)

| Model | M=0 | M=* (True Premise Only) | M=20 (Large Candidate Set) |
| :--- | :--- | :--- | :--- |
| **DeepSeek-Prover-V2 (noCoT)** | 58.37 | **71.86 (+13.49)** | 65.80 (+7.43) |
| Goedel-Prover | 39.04 | 50.10 (+11.06) | 39.50 (+0.46) |
| STP | 57.72 | 61.86 (+4.14) | 56.04 **(-1.68)** |
| DeepSeek-Prover-V1.5 | 52.94 | 56.69 (+3.75) | 49.45 **(-3.49)** |

- **DeepSeek-Prover-V2 leads in premise utilization**, with a gain of ~10% when provided with true premises, likely due to its training on vast natural language reasoning data.
- **STP scores drop at M=10/20** despite its high overall accuracy, as its training set lacks this type of premise-conditioned data. Global optimality $\neq$ optimal premise utilization.

### Key Findings
- **Finding 1**: Existing provers are insufficient as practical assistants under low compute budgets, and performance varies significantly across specialized sub-domains (e.g., McDiarmid vs. Hoeffding).
- **Finding 2**: Models fail to handle large candidate premise sets; even with ground-truth premises, most models under-utilize them, with the bottleneck residing in the base models.
- **Finding 3**: Performance decays sharply as difficulty increases (L3/L5); STP only reaches Pass@128 of 33.36% on L5.
- **Finding 4**: Long-CoT for subgoal completion is **inefficient and does not improve scores**, ranking last in EWA@32; STP has the shortest output and highest accuracy, achieving optimal EWA.
- **Finding 5**: Expert iteration (extracting 92k problems from 5 libraries like mathlib, PFR, and scilean; 88k for training) significantly boosts scores, especially Pass@1, showing a promising direction.

## Highlights & Insights
- **Task definition is a major contribution**: Isolating "subgoal completion" from full proof generation targets the real-world utility of "LLMs as math copilots"—an intermediate milestone closer to research collaboration than contest scores.
- **`to_theorem` is a reusable data engine**: Beyond building benchmarks, the expert iteration experiment successfully extracted 90k+ training problems from 5 Lean libraries, proving this "procedural proof $\to$ massive subgoals" pipeline is highly scalable.
- **EWA@K exposes Long-CoT weaknesses**: By using a simple "accuracy/response length" metric, the paper quantifies the "overthinking" ignored in contest narratives, revealing that long-chain reasoning is not a panacea.
- **Robust counter-intuitive conclusions**: Top performers on miniF2F (Long-CoT series) lose momentum here, showing that model rankings on contests do not translate to real proof assistant tasks due to competition-specific overfitting.

## Limitations & Future Work
- **Difficulty proxy is limited to length**: Proof length does not perfectly equal reasoning difficulty; "line count" in procedural styles can sometimes be disconnected from semantic complexity.
- **Limited scope (ML Theory)**: While Optlib and FoML cover optimization and probability, this is narrow compared to the breadth of mathematics; the generalizability of these findings remains to be verified.
- **Premise selection is bypassed**: The paper focuses on "premise utilization," assuming retrieval/selection is already performed; in real scenarios, retrieval errors would further increase difficulty.
- **Single round of expert iteration**: Finding 5 is a preliminary validation; the diminishing returns of multi-round iterations and potential overfitting to the `to_theorem` distribution haven't been explored.

## Related Work & Insights
- **vs. Full Proof Benchmarks** (miniF2F, PutnamBench, ProofNet, FormalMATH, LeanDojo): These often feature simple contexts or no premise requirements and focus on end-to-end generation; FormalML is unique in combining premises, complex context, and subgoal completion.
- **vs. Prover Methods**: Both BFS tree search (Reprover, BFS-Prover) and whole-proof generation (STP, Goedel, DeepSeek-Prover, Kimina) pipelines were evaluated; current SOTAs are far less dominant on "fill-in-the-blank" tasks compared to contests.
- **Insights**: (1) Evaluation should align with real workflows—"filling intermediate steps" reveals model flaws better than "generation from scratch"; (2) Efficiency should be a primary metric for provers, not just Pass@K; (3) The `to_theorem` approach of reverse-engineering problems from existing proofs is highly valuable for low-cost scaling of formal training data.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to define and benchmark "subgoal completion"; the `to_theorem` slicing and EWA@K metric are original, filling a gap in real-world scenario evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers over ten SOTA provers (BFS and generation-based) across five dimensions: accuracy, premise utilization, difficulty, efficiency, and expert iteration.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear; Figures 1 and 2 make the `sorry` placeholder and `to_theorem` mechanism intuitive. The Lean code details might be challenging for readers without a formalization background.
- **Value**: ⭐⭐⭐⭐⭐ Provides the first reliable yardstick for "LLMs as math copilots" and outputs a reusable data pipeline and training signal with long-term value for the formal theorem proving community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] An Open-Ended Benchmark and Formal Framework for Adjuvant Research with MLLM](an_open-ended_benchmark_and_formal_framework_for_adjuvant_research_with_mllm.md)
- [\[ACL 2026\] Can We Predict Before Executing Machine Learning Agents?](../../ACL2026/llm_evaluation/can_we_predict_before_executing_machine_learning_agents.md)
- [\[ICLR 2026\] Characterizing Deep Research: A Benchmark and Formal Definition](characterizing_deep_research_a_benchmark_and_formal_definition.md)
- [\[ICLR 2026\] CMT-Benchmark: A Benchmark for Condensed Matter Theory Built by Expert Researchers](cmt-benchmark_a_benchmark_for_condensed_matter_theory_built_by_expert_researcher.md)
- [\[ICLR 2026\] In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)

</div>

<!-- RELATED:END -->
