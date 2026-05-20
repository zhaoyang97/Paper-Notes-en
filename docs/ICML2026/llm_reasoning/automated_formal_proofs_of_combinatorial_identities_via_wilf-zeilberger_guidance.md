---
title: >-
  [Paper Note] Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs
description: >-
  [ICML 2026][LLM Reasoning][Lean 4] WZ-LLM compiles the classical Wilf–Zeilberger symbolic proof process into executable proof skeletons in Lean 4 (recurrence + boundary conditions + side conditions)…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Lean 4"
  - "combinatorial identities"
  - "Wilf-Zeilberger"
  - "neuro-symbolic"
  - "DAPO"
date: 2026-05-08
content_hash: 8c6fe48b8515b549
---

# Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.04472](https://arxiv.org/abs/2605.04472)  
**Code**: Not yet released  
**Area**: LLM Reasoning / Automated Theorem Proving / Neuro-symbolic  
**Keywords**: Lean 4, combinatorial identities, Wilf-Zeilberger, neuro-symbolic, DAPO

## TL;DR
WZ-LLM compiles the classical Wilf–Zeilberger symbolic proof process into executable proof skeletons in Lean 4 (recurrence + boundary conditions + side conditions), which are then discharged item by item by a WZ-Prover trained via SFT + expert-iteration + DAPO. On 100 classical combinatorial identities, pass@32 improves from Goedel-Prover-V2's 9% to 34%.

## Background & Motivation

**Background**: LLM-based automated theorem proving (ATP) has achieved competition-level performance on interactive proof assistants like Lean and Isabelle (e.g., DeepSeek-Prover-V2, Kimina, Goedel-Prover-V2), but combinatorics is widely regarded as one of the most challenging domains for ATP, with "combinatorial identities" being a fundamental and ubiquitous class of propositions.

**Limitations of Prior Work**: 1) Proving combinatorial identities requires long-range planning—without a global roadmap, LLMs fall into unbounded search and combinatorial explosion; 2) Training data for combinatorics in Lean is extremely scarce; 3) Purely symbolic methods (WZ, creative telescoping) are highly efficient in CAS, but their outputs cannot be directly translated into proof assistants—one must reconstruct telescoping arguments, boundary conditions, normalization steps, and various non-vanishing side conditions, making the "formalization cost" outweigh the original proof cost; 4) Existing whole-proof LLMs lack intermediate verifier signals, while tactic-level models suffer from branching explosion.

**Key Challenge**: Long-range proofs require explicit planning, which LLMs lack; symbolic methods are inherently well-planned, but their products are not formalizable. Each approach excels in its own way but remains disconnected from the other.

**Goal**: To fuse the "planning ability" of WZ with the "formalization ability" of LLMs, enabling coverage of identities that neither symbolic methods nor LLMs alone can handle.

**Key Insight**: The authors observe that the WZ method naturally provides a "sketch"—after constructing the WZ pair $G(n,k)=R(n,k)F(n,k)$, the identity decomposes into a set of machine-verifiable sub-goals: "recurrence lemma + boundary conditions + side conditions + normalization + case-split." This structure aligns perfectly with Lean 4's preferences, serving as an intermediate scaffold for LLMs that both reduces the search space and provides verifier signals.

**Core Idea**: Replace pure LLM or pure symbolic approaches with a dual-path neuro-symbolic system: **"WZ symbolic decomposition (external CAS generates sketch) + a specialized WZ-Prover (discharges sketch sub-goals + handles identities not covered by WZ)"**.

## Method

### Overall Architecture
WZ-LLM consists of two stages: (A) **Symbolic Decomposition**—given a combinatorial identity formalized in Lean 4, normalization is performed, then SageMath's WZ algorithm is invoked to synthesize a certificate; if successful, the identity is decomposed into a structured set of Lean sub-goals $\mathcal{T}=\mathcal{T}_{\text{rec}}\cup\mathcal{T}_{\text{bd}}\cup\mathcal{T}_{\text{side}}\cup\mathcal{T}_{\text{norm}}\cup\mathcal{T}_{\text{case}}$; if WZ is not applicable, the problem enters the "direct proof pool." (B) **WZ-Prover**—a dedicated Lean 4 prover, initialized from Goedel-Prover-V2 and trained in three stages, responsible for discharging all tasks in the pool (sketch sub-goals + WZ-uncovered problems). The Lean kernel serves as the final arbiter: only proofs accepted by the kernel are considered successful.

### Key Designs

1. **WZ Symbolic Decomposer (Symbolic Decomposition)**:

    - **Function**: Compiles a Lean 4 identity $S(n)=\sum_k F(n,k)=C$ into a set of machine-verifiable Lean proof obligations.
    - **Mechanism**: ① Normalization—converts `Icc/Ico` to `Finset.range`, shifts indices to start from 0, simplifies syntactic variants of factorials/binomials/powers; structures case-splits for piecewise predicates like parity; ② Sketch construction—uses SageMath's `F.WZ_certificate(n,k)` to synthesize a rational function $R(n,k)$ such that $G(n,k)=R(n,k)F(n,k)$ satisfies the WZ equation $F(n+1,k)-F(n,k)=G(n,k+1)-G(n,k)$, thereby reducing the original identity to "recurrence lemma + boundary obligation"; ③ Side condition inference—symbolic simplification is used to identify zero denominators or negative factorial parameters that would block tactics like `field_simp`, automatically generating non-vanishing lemmas such as `∀n,k, A(n,k)≠0` and boundary sub-goals.
    - **Design Motivation**: In practice, the real challenge in Lean formalization lies in boundary and side conditions—the "certificate" provided by CAS is mathematically correct but cannot be mechanically inserted into proof assistants. Making all these implicit obligations explicit and turning them into small sub-goals that LLMs can discharge individually is key to making the sketch executable.

2. **Expert-in-the-loop Bootstrapping (Expert-in-the-loop Iteration)**:

    - **Function**: Starts from 307 manually formalized identities (with complete Lean proofs), iteratively expands the training set without introducing hallucinations.
    - **Mechanism**: Stage one uses 307 identities plus their 1200 sketch-derived sub-goals for cold-start SFT; stage two runs WZ-LLM on 1020 unlabeled candidate identities, and only those strictly verified by the Lean kernel enter the next training pool—Round 1 yields 5139 lemma proofs + 32 full problems, Round 2 adds 532 + 79, totaling 5671 lemmas + 111 full problems; after deduplication, about 5418 samples form the expanded SFT corpus.
    - **Design Motivation**: Combinatorics data in Lean is extremely scarce and cannot be scaled manually; LLM self-generation risks hallucination. "Verifier-filtered bootstrapping" naturally filters out noisy samples via the kernel, providing "free" high-fidelity data augmentation and ensuring the training distribution is not contaminated by its own errors.

3. **DAPO with Difficulty-Smoothing Fine-tuning**:

    - **Function**: After SFT, RL is used to further improve long-CoT robustness, focusing on hard problems and long lemma chains.
    - **Mechanism**: ① Difficulty smoothing—most sketch lemmas in the SFT corpus are short and repetitive, while full problems are long and rare; rollout estimates the pass-rate for each problem under the current policy, discarding both the easiest (after deduplication) and near-zero pass-rate (noisy gradient) ends to obtain an RL set with a smoothed medium-hard distribution; ② DAPO optimization—reward $R(\pi;G)=R_{\text{out}}(\pi;G)+\lambda_{\text{len}}R_{\text{len}}(\pi)$, where $R_{\text{out}}\in\{+1,-1\}$ is the Lean kernel verification signal, and $R_{\text{len}}$ is a progressive penalty near the token budget to avoid false negatives due to truncation; DAPO's dynamic sampling mitigates entropy collapse.
    - **Design Motivation**: With binary and sparse kernel rewards, naive RL easily overfits on easy problems or collapses on hard ones; the combination of difficulty smoothing and DAPO focuses compute on "non-trivial but salvageable" problems.

### Loss & Training

Three-stage pipeline: (i) SFT on 307 seed + 1200 lemmas; (ii) expert-iteration expansion to ~5418 verified samples; (iii) DAPO RL with rule-based outcome reward + soft overlong punishment. The entire training took 16 GPU-days, inference and evaluation 9 GPU-days, all on 4× L40s-48GB. Model size is only 8B.

## Key Experimental Results

### Main Results
LCI-Test (100 classical combinatorial identities, formalized in Lean 4) end-to-end proof success rate (pass@32):

| Method | Model | LCI-Test pass@32 |
|------|------|------|
| DeepSeek-V3 | 685B | 1/100 |
| Gemini-3.1-Pro-Preview | — | 16/100 |
| Kimina-Prover-Distill | 7B | 6/100 |
| DeepSeek-Prover-V2 | 7B | 6/100 |
| Goedel-Prover-V2 (baseline) | 8B | 9/100 |
| WZ-Sketch + Goedel-Prover-V2 | 8B | 9/100 |
| WZ-Prover (only direct) | 8B | 12/100 |
| WZ-Sketch + WZ-Prover | 8B | 29/100 |
| **WZ-LLM (merged paths)** | **8B** | **34/100** |

Cross-dataset generalization: On CombiBench, 12→16/100; on PutnamBench-Comb, 0→3/36, both outperforming baselines.

### Ablation Study

| Training Stage | pass@1 | pass@8 | pass@32 |
|----------|--------|--------|---------|
| SFT (seed only) | 1/100 | 3/100 | 9/100 |
| + expert-iteration | 3/100 | 6/100 | 10/100 |
| + DAPO refinement | 4/100 | 6/100 | 12/100 |

Lemma-level diagnosis (1178 sketch-derived sub-goals):

| Model | #Proved / 1178 | Acc | End-to-end #Solved / 46 |
|------|-----|-----|------|
| Goedel-Prover-V2 | 564 | 47.88% | 0 |
| WZ-Prover | 864 | 73.34% | 29 |

### Key Findings
- **Sketch alone is insufficient**: Stacking the sketch on an untrained Goedel-V2 yields no gain (9→9); since the full problem requires **all** sketch lemmas to be discharged, 47.88% lemma accuracy leads directly to 0 end-to-end solutions. Only at 73.34% does it unlock 29 problems, indicating that both a specialized prover and specialized sketch are necessary.
- **Direct + sketch are complementary**: 5 hard problems not covered by WZ are solved directly by WZ-Prover (symbolic-only can never achieve this), while 29 WZ-applicable problems are completed via the sketch path; together, they solve 34 problems.
- **DAPO's gains are concentrated at pass@32**: pass@1 increases by only +1, but pass@32 by +2, indicating RL mainly helps "long-tail hard problems" under larger sampling budgets, rather than squeezing out a few more easy problems.

## Highlights & Insights
- Treating classical symbolic methods as "executable sketch generators" is a very elegant hybrid approach: it circumvents both the LLM's weakness in long-range planning and the issue that CAS outputs cannot be directly used in proof assistants, turning the non-overlapping strengths of both into additive gains.
- "Verifier-filtered bootstrapping" is almost a no-brainer for data augmentation in kernel-checked environments like Lean: the training pool is self-generated by the model, with the verifier as a filter, theoretically allowing scaling up until the capability ceiling is reached.
- The combination of DAPO + difficulty smoothing provides a reusable recipe for sparse binary reward scenarios: use rollout to bucket the problem set by current policy difficulty, trim both noisy ends, then apply RL; no need for manual grading.

## Limitations & Future Work
- The 8B model and 16 GPU-days of training are academic-friendly, but 66 problems remain unsolved on LCI-Test, indicating that the capability ceiling for long-range combinatorial proofs is still far from reached, especially with only 3/36 solved on PutnamBench-Comb.
- The entire pipeline is sensitive to the evolution of Lean 4 mathlib APIs; the sketch component is tightly coupled to the current Finset/Nat.factorial interfaces, and any mathlib refactoring would require realignment of normalization rules.
- The WZ method only covers hypergeometric/holonomic identities; for genuinely "non-hypergeometric" combinatorial identities (e.g., q-series, involution arguments), new symbolic sketch engines are needed.

## Related Work & Insights
- **vs Goedel-Prover-V2 / DeepSeek-Prover-V2 and other whole-proof LLMs**: These rely on end-to-end generation without explicit planning; WZ-LLM outsources "long-range planning" to decades-mature symbolic algorithms via external CAS-provided sketches.
- **vs InternLM-2.5-StepProver / MA-LoT and other tactic-level + search approaches**: These use BFS/MCTS to search the tactic space, leading to branching explosion; WZ-LLM does not search at the tactic level but plans at the higher-level sketch, then uses a whole-proof prover for each sub-goal.
- **vs Harrison's work on hypergeometric sums in HOL Light**: The approach is similar (CAS-generated certificates + formalization), but Harrison's embedding is fully manual; WZ-LLM delegates the most labor-intensive "formalization" step to the LLM-Prover, representing a modern upgrade of this idea.

## Rating
- Novelty: ⭐⭐⭐⭐ The framing of "compiling symbolic method sketches into Lean sub-goals provable by LLMs" is refreshing in the ATP community.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, dual ablations at both component and training stage levels, and lemma-level diagnostics are all well executed.
- Writing Quality: ⭐⭐⭐⭐ The neuro-symbolic architecture and training pipeline are clearly explained, with complete background on WZ mathematics.
- Value: ⭐⭐⭐⭐ Provides a reusable "symbolic guidance + LLM discharge" recipe for Lean mathematical formalization, potentially generalizable to other CAS-rich domains (integration, summation, ODEs, etc.).

## Related Papers

- [\[ICML 2026\] Game of Thought: 用博弈论求 Nash 均衡让 LLM 在 20 题游戏里抗最坏情况](game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](../../ICLR2026/llm_reasoning/nudging_the_boundaries_of_llm_reasoning.md)
- [\[AAAI 2026\] Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment](../../AAAI2026/llm_reasoning/dropouts_in_confidence_moral_uncertainty_in_human-llm_alignment.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](../../ICLR2026/llm_reasoning/on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ACL 2026\] Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](../../ACL2026/llm_reasoning/logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)

</div>

<!-- RELATED:END -->
