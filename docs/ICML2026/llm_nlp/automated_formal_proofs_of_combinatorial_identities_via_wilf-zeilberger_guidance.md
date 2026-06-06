---
title: >-
  [Paper Note] Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs
description: >-
  [ICML 2026][LLM/NLP][Lean 4] WZ-LLM compiles the classic Wilf–Zeilberger symbolic proof workflow into executable proof sketches in Lean 4 (recurrence + boundary conditions + side conditions). These are then discharged se…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Lean 4"
  - "Combinatorial Identities"
  - "Wilf-Zeilberger"
  - "Neuro-symbolic"
  - "DAPO"
date: 2026-05-08
content_hash: 6bb1c81252c3a759
---

# Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.04472](https://arxiv.org/abs/2605.04472)  
**Code**: Not yet released  
**Area**: LLM Reasoning / Automated Theorem Proving / Neuro-symbolic  
**Keywords**: Lean 4, Combinatorial Identities, Wilf-Zeilberger, Neuro-symbolic, DAPO

## TL;DR
WZ-LLM compiles the classic Wilf–Zeilberger symbolic proof workflow into executable proof sketches in Lean 4 (recurrence + boundary conditions + side conditions). These are then discharged sequentially by WZ-Prover, a model specifically trained using SFT + expert-iteration + DAPO. This approach improves the pass@32 on 100 classic combinatorial identities from 9% (Goedel-Prover-V2) to 34%.

## Background & Motivation

**Background**: LLM-based Automated Theorem Proving (ATP) has reached competition-level performance on interactive theorem provers like Lean and Isabelle (e.g., DeepSeek-Prover-V2, Kimina, Goedel-Prover-V2). However, combinatorics is widely regarded as one of the most challenging areas for ATP, with "combinatorial identities" being a fundamental and ubiquitous class of propositions.

**Limitations of Prior Work**: 1) Proving combinatorial identities requires long-range planning—without a global roadmap, LLMs fall into unrestricted search and combinatorial explosion; 2) Training data for combinatorics in Lean is extremely scarce; 3) Pure symbolic methods (WZ, creative telescoping) are highly efficient in CAS but their outputs cannot be directly translated into proof assistants—reconstructing telescoping arguments, boundary conditions, normalization steps, and various non-zero side conditions creates a "formalization cost" that outweighs the original proof effort; 4) Existing whole-proof LLMs lack intermediate verifier signals, while tactic-by-tactic models suffer from branch explosion.

**Key Challenge**: Long-range proofs require explicit planning, which LLMs lack; symbolic methods inherently possess planning, but their products are not formalizable. The two routes each have strengths but remain disconnected.

**Goal**: To weld WZ's "planning capability" with LLM's "formalization capability," allowing identities that were previously unsolvable by either pure symbolic or pure LLM methods to be covered by both.

**Key Insight**: The authors noted that the WZ method itself provides a natural "sketch"—after constructing a WZ pair $G(n,k)=R(n,k)F(n,k)$, the identity automatically decomposes into a set of machine-verifiable sub-goals: "recurrence lemma + boundary conditions + side conditions + normalization + case-split." This structure is precisely what Lean 4 favors. Using it as an intermediate scaffold for LLMs reduces the search space and provides verifier signals.

**Core Idea**: A dual-path neuro-symbolic system replacing pure LLM or pure symbolic approaches with **"WZ Symbolic Decomposition (external CAS-generated sketch) + Specially trained WZ-Prover (discharging sketch sub-goals + handling WZ-uncovered identities)."**

## Method

### Overall Architecture
WZ-LLM consists of two stages: (A) **Symbolic Decomposition**—Given a combinatorial identity formalized in Lean 4, it first performs normalization, then calls the WZ algorithm in SageMath to attempt certificate synthesis; if successful, the identity is decomposed into a structured set of Lean sub-goals $\mathcal{T}=\mathcal{T}_{\text{rec}}\cup\mathcal{T}_{\text{bd}}\cup\mathcal{T}_{\text{side}}\cup\mathcal{T}_{\text{norm}}\cup\mathcal{T}_{\text{case}}$; if WZ is inapplicable, the problem enters the "direct proof pool." (B) **WZ-Prover**—A specialized Lean 4 prover initialized from Goedel-Prover-V2 and trained through three stages. It is responsible for proving all tasks (sketch sub-goals + WZ-uncovered problems). The Lean kernel gives the final verdict: only proofs accepted by the kernel are considered successful.

### Key Designs

1. **Symbolic Decomposition**:
    - **Function**: Compiles a Lean 4 identity into a set of machine-verifiable Lean proof obligations.
    - **Mechanism**: ① Normalization—converting `Icc/Ico` to `Finset.range`, shifting indices to start from 0, and simplifying syntax variants for factorials/binomials/powers; structured case-splitting for piecewise predicates like parity. ② Sketch construction—using SageMath's `F.WZ_certificate(n,k)` to synthesize a rational function $R(n,k)$ such that $G(n,k)=R(n,k)F(n,k)$ satisfies the WZ equation $F(n+1,k)-F(n,k)=G(n,k+1)-G(n,k)$, thereby reducing the original identity to a "recurrence lemma + boundary obligations." ③ Side condition inference—using symbolic simplification to identify zero denominators or negative factorial arguments that would stall tactics like `field_simp`, automatically generating non-vanishing lemmas such as `∀n,k, A(n,k)≠0` as sub-goals.
    - **Design Motivation**: Practice shows that the real difficulty in Lean formalization lies in the boundaries and side conditions—"certificates" provided by CAS are mathematically correct but cannot be mechanically inserted into a proof assistant. Making these implicit obligations explicit sub-goals that an LLM can discharge individually is key to making the sketch executable.

2.  **Expert-in-the-loop Iteration**:
    - **Function**: Starting from 307 manually formalized identities (with full Lean proofs), iteratively expand the training set without introducing hallucinations.
    - **Mechanism**: The first stage uses the 307 identities + 1200 decomposed sub-goals for cold-start SFT. The second stage runs WZ-LLM on 1020 unlabelled candidate identities. Only those **strictly verified by the Lean kernel** enter the next round of training. Round 1 yielded 5139 lemma proofs + 32 full proofs; Round 2 added 532 + 79, totaling 5671 lemmas + 111 full proofs. After deduplication, this formed an expanded SFT corpus of approximately 5418 samples.
    - **Design Motivation**: Lean data for combinatorics is extremely scarce and cannot be scaled manually. LLM self-generation carries hallucination risks. "Verifier-filtered bootstrapping" uses the kernel to filter noise, providing high-fidelity data augmentation and ensuring the training distribution is not polluted by errors.

3.  **DAPO with Difficulty-Smoothing**:
    - **Function**: Further enhances long-CoT robustness via RL after SFT, prioritizing performance on hard problems and long-chain lemmas.
    - **Mechanism**: ① Difficulty smoothing—Since sketch lemmas in the SFT corpus are often short and repetitive while full proofs are long and rare, rollouts are used to estimate the pass-rate under the current policy. Extremely easy (redundant) and near-zero pass-rate (noise gradient) tasks are removed to obtain a difficulty-smoothed training set. ② DAPO optimization—The reward is $R(\pi;G)=R_{\text{out}}(\pi;G)+\lambda_{\text{len}}R_{\text{len}}(\pi)$, where $R_{\text{out}}\in\{+1,-1\}$ is the Lean kernel signal and $R_{\text{len}}$ is a progressive penalty as the token budget is approached, avoiding false negatives caused by truncation. DAPO's dynamic sampling mechanism mitigates entropy collapse.
    - **Design Motivation**: Under binary and sparse kernel rewards, naive RL easily overfits on easy problems or collapses on hard ones. The combination of difficulty smoothing and DAPO focuses the RL compute on "non-trivial but solvable" problems.

### Loss & Training
Three-stage pipeline: (i) SFT on 307 seed + 1200 lemmas; (ii) Expert-iteration expanded to ~5418 verified samples; (iii) DAPO RL with rule-based outcome reward + soft punishment for excessive length. Total training took 16 GPU-days and evaluation took 9 GPU-days, all completed on 4× L40s-48GB. The model size is 8B.

## Key Experimental Results

### Main Results
End-to-end proof success rate (pass@32) on LCI-Test (100 classic combinatorial identities formalized in Lean 4):

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
| **WZ-LLM (Combined)** | **8B** | **34/100** |

Cross-dataset generalization: On CombiBench, it improved from 12→16/100; on PutnamBench-Comb, it improved from 0→3/36, both outperforming baselines.

### Ablation Study

| Training Stage | pass@1 | pass@8 | pass@32 |
|----------|--------|--------|---------|
| SFT (seed only) | 1/100 | 3/100 | 9/100 |
| + Expert-iteration | 3/100 | 6/100 | 10/100 |
| + DAPO refinement | 4/100 | 6/100 | 12/100 |

Lemma-level diagnosis (1178 sub-goals decomposed from sketches):

| Model | #Proved / 1178 | Acc | End-to-end #Solved / 46 |
|------|-----|-----|------|
| Goedel-Prover-V2 | 564 | 47.88% | 0 |
| WZ-Prover | 864 | 73.34% | 29 |

### Key Findings
- **Sketch alone is not enough**: Adding sketches to Goedel-V2 without specialized training yielded no gain (9→9). Since a full proof requires **all** sketch lemmas to be discharged, a 47.88% lemma accuracy leads to zero end-to-end proofs. Improving this to 73.34% unlocked 29 proofs, demonstrating that a "specialized prover + specialized sketch" must coexist.
- **Direct + Sketch complement each other**: 5 hard identities not suitable for WZ were proved directly by WZ-Prover (which symbolic methods could never solve), while 29 WZ-suitable problems were completed via the sketch path, totaling 34.
- **DAPO gains are concentrated in pass@32**: Pass@1 only increased by +1, while pass@32 increased by +2, suggesting that RL primarily enables the model to capture "long-tail difficult problems" under larger sampling budgets rather than just improving easy tasks.

## Highlights & Insights
- Treating classic symbolic methods as "executable sketch generators" is an elegant fusion: it bypasses the weakness of LLMs in long-range planning and the inability of CAS output to enter proof assistants directly, making the strengths additive.
- "Verifier-filtered bootstrapping" in a kernel-checked environment like Lean is an effective data augmentation strategy: the training pool is generated by the model itself, filtered by the verifier, and can theoretically scale until the capability ceiling is reached.
- The combination of DAPO + difficulty smoothing provides a reusable recipe for sparse binary reward scenarios: roll out to bin the problem set by difficulty, prune the noise at both ends, and then perform RL; this removes dependency on manual grading.

## Limitations & Future Work
- While the 8B model and 16 GPU-day training are researcher-friendly, 66 problems in LCI-Test remain unsolved, indicating that the capability ceiling for long-range combinatorial proofs is far from reached, particularly on PutnamBench-Comb (3/36).
- The pipeline is sensitive to the evolution of the Lean 4 mathlib API; the sketch component is tightly coupled to current `Finset/Nat.factorial` interfaces. Refactoring mathlib would require re-aligning normalization rules.
- The WZ method only covers hypergeometric/holonomic identities. New symbolic sketch engines are needed for "non-hypergeometric" combinatorial identities (e.g., q-series, involution arguments).

## Related Work & Insights
- **vs Goedel-Prover-V2 / DeepSeek-Prover-V2**: These rely on end-to-end generation without explicit planning; WZ-LLM provides sketches via external CAS, outsourcing "long-range planning" to mature symbolic algorithms.
- **vs InternLM-2.5-StepProver / MA-LoT**: These perform search in the tactic space (BFS/MCTS) and suffer from branch explosion; WZ-LLM avoids search at the tactic level, instead planning at the higher sketch level and using a whole-proof prover for sub-goals.
- **vs Harrison's work on HOL Light**: The approach of using CAS for certificates is similar, but Harrison manually embedded them; WZ-LLM's modernization uses an LLM-prover for the most labor-intensive formalization step.

## Rating
- Novelty: ⭐⭐⭐⭐ The framing of "compiling symbolic sketches into LLM-provable Lean sub-goals" is refreshing in the ATP community.
- Experimental Thoroughness: ⭐⭐⭐⭐ Testing across three benchmarks, combined with dual ablations for components and training stages, as well as lemma-level diagnosis.
- Writing Quality: ⭐⭐⭐⭐ The neuro-symbolic architecture and training pipeline are clearly explained, with complete WZ mathematical context.
- Value: ⭐⭐⭐⭐ Provides a reusable recipe of "symbolic guidance + LLM discharge" for Lean formalization, extensible to other domains with CAS (integration, summation, ODEs, etc.).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](../../ICLR2026/llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)
- [\[ACL 2026\] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](../../ACL2026/llm_nlp/solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)
- [\[ICML 2026\] Differential Syntactic and Semantic Encoding in LLMs](differential_syntactic_and_semantic_encoding_in_llms.md)
- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](../../NeurIPS2025/llm_nlp/solving_inequality_proofs_with_large_language_models.md)
- [\[ICML 2026\] Position: Adversarial ML for LLMs Is Not Making Any Progress](position_adversarial_ml_for_llms_is_not_making_any_progress.md)

</div>

<!-- RELATED:END -->
