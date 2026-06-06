---
title: >-
  [Paper Note] FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean
description: >-
  [ACL 2026][Code Intelligence][Autoformalisation] FormalScience proposes a domain-agnostic Human-in-the-Loop (HITL) agentic pipeline that enables a single domain expert—without Lean proficiency—to translate informal scien…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Autoformalisation"
  - "Lean4"
  - "Physics Formalisation"
  - "Agent"
  - "Human-in-the-loop"
  - "Semantic Drift"
date: 2026-05-08
content_hash: 9136621ab965eded
---

# FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean

**Conference**: ACL 2026  
**arXiv**: [2604.23002](https://arxiv.org/abs/2604.23002)  
**Code**: https://github.com/jmeadows17/formal-science  
**Area**: Code Intelligence / Formalisation / Lean4 / Autoformalisation  
**Keywords**: Autoformalisation, Lean4, Physics Formalisation, Agent, Human-in-the-loop, Semantic Drift

## TL;DR
FormalScience proposes a domain-agnostic Human-in-the-Loop (HITL) agentic pipeline that enables a single domain expert—without Lean proficiency—to translate informal scientific reasoning (specifically physics) into 100% compilable Lean4 code. It constructs FormalPhysics, the first formalisation benchmark of 200 university-level physics problems, and systematically characterises the phenomenon where code is "formally valid" but subject to "semantic drift."

## Background & Motivation

**Background**: Automatically translating mathematical derivations written in natural language into formal code (autoformalisation) for theorem provers like Lean or Coq is a prominent direction in LLM $\times$ Formal Methods. Existing benchmarks (miniF2F, ProofNet, Lean Workbook, Herald, etc.) focus almost entirely on competition or undergraduate mathematics. While FormalMath and Herald-Proof have recently scaled to tens or hundreds of thousands of problems, the **domain remains pure mathematics**.

**Limitations of Prior Work**: Formalisation in scientific fields (especially physics) is virtually non-existent. Reasons include: (1) Physics extensively uses domain-specific symbols like Dirac notation $\ket{\Psi}$ and vector calculus $\nabla\times\vec{E}$, which are not directly supported by Lean4/Mathlib; (2) LLM hallucination rates explode with complexity in out-of-distribution, long-chain reasoning; (3) The Formal Validity (FV) of existing datasets like Herald-Proof is as low as 2%, indicating a massive gap between "automatically generated" and "actually compilable" code.

**Key Challenge**: The authors discovered a core trade-off—**Formal Validity (FV)** and **Semantic Alignment (FQ/LP/MC)** are nearly orthogonal (Spearman correlation $\approx 0$, $p>0.9$). In other words, small models optimised for compilation (e.g., Kimina-7B reaching 51.5% FV) "cheat" by fabricating compilable code that completely deviates semantically, whereas high-alignment large models like GPT-5.1 achieve only 14.5% FV in zero-shot settings.

**Goal**: (i) Design a low-cost (1 person / 1 month / $50) HITL pipeline for producing 100% FV formalised datasets; (ii) Provide FormalPhysics, the first high-quality benchmark for the physics domain; (iii) Systematically characterise the "compilable but semantically drifted" phenomenon to address the epistemic question: **"What does Lean actually verify?"**

**Key Insight**: The authors position the human expert as a **binary classifier** $\mathcal{H}\in\{0,1\}$ isomorphic to the compiler $\mathcal{L}$. Since LLMs alone cannot handle semantic alignment, the expert acts as a lightweight oracle for "alignment," **without requiring the expert to write Lean**. Their role is limited to judging whether the formal code matches the original informal statement.

**Core Idea**: Decompose autoformalisation into four nested loops: "statement generation + formal code generation + compilation error correction + expert alignment verification." The compilation loop uses Lean as the oracle, while the alignment loop uses the human as the oracle, alternating until convergence.

## Method

### Overall Architecture

FormalScience is a three-stage pipeline with dual nested oracles (Alg.1):

- **Input**: A set of informal proofs $\mathcal{D}$ (e.g., LaTeX derivations) + a few-shot set of golden standard pairs $\mathcal{D}^*=\{(\mathcal{S}_i,\mathcal{P}_i)\}_{i=1}^{N'}$ (where $N'=5$).
- **Phase A (In-context Statement Rewriting)**: Uses a few-shot template $T_{fs}$ to prompt the LLM to rewrite raw proofs $d$ into $(S,P)$ pairs aligned with the style of $\mathcal{D}^*$, yielding intermediate data $X=\sum_{d\in\mathcal{D}} S\big(\mathcal{M}(T_{fs}(d,\mathcal{D}^*);P_a)\big)$.
- **Phase B (Compilation Error Correction Loop $\mathcal{R}$)**: Generates initial Lean code $C^{(0)}=\mathcal{M}'(T_g(x))$ using template $T_g$. The Lean compiler tool $\mathcal{L}(C)$ returns $(0,\varepsilon)$ if it passes and $(1,e)$ otherwise. It iterates $C^{(t+1)}=\mathcal{M}'(T_c(x,C^{(t)},e))$ until $t^*=\min\{t:\mathcal{L}(C^{(t)})=(0,\varepsilon)\}$.
- **Phase C (Expert Alignment Loop)**: The expert acts as a binary classifier $\mathcal{H}^{(k)}=\mathcal{H}(\mathcal{M}'(T_g(x),C^{(k)}))\in\{0,1\}$. If misaligned ($\mathcal{H}=1$), a new version is generated under prompt $P_g$ and the compilation loop $\mathcal{R}$ is re-invoked. Patience $\mathcal{P}$ controls the maximum rounds.
- **Output**: Triples $(S,P,C)$, where $C$ is 100% Lean4 compilable. All $C$ are passed through $\mathcal{L}$ one final time; failed samples re-enter Eq.3-5.

### Key Designs

1. **Dual Oracle Nested Loops (Compilation Oracle + Human Oracle)**:
    - **Function**: Decouples the inherently different goals of "syntactic correctness" and "semantic alignment."
    - **Mechanism**: The inner loop $\mathcal{R}$ uses the Lean compiler as a strict oracle to iteratively fix code until it compiles. The outer loop uses a physics expert as a $\{0,1\}$ binary classifier to judge alignment. **The expert does not need to know Lean code**, but can use LLM-generated alignment self-evaluations $\mathcal{M}'(T_g(x), C^{(k)})$ as auxiliary reading. The nested structure ensures every "alignment revision" passes back through the compilation loop to prevent human-induced syntax errors.
    - **Design Motivation**: The authors found semantic metrics like LP/MC to be nearly orthogonal to FV ($\rho\approx 0$), meaning a single oracle cannot improve both axes simultaneously. Using LLM-as-judge for the outer loop leads to "models deceiving models." Expert binary judgment is orders of magnitude cheaper than line-by-line Lean Editing—the entire FormalPhysics was completed by one Physics PhD in 1 month for $50.

2. **Few-shot Templates + Golden Standard Alignment $T_{fs}(d,\mathcal{D}^*)$**:
    - **Function**: Rewrites messy raw LaTeX proofs $d$ into $(S,P)$ pairs with structure and granularity consistent with $\mathcal{D}^*$.
    - **Mechanism**: Uses in-context learning to teach the LLM "granularity conventions" (which steps to detail, what textual context to add). A post-processing function $S$ splits batch outputs into individual statement-proof pairs. This step is equivalent to an "informal-to-informal refactor," with human validation via prompt $P_a$.
    - **Design Motivation**: Physics proofs often skip steps (e.g., "by symmetry..."), while Lean requires explicit steps. Boosting the signal-to-noise ratio before informal $\to$ formal translation significantly reduces iterations in the compilation loop.

3. **Structural vs. Semantic Error Branching in Agentic Baselines**:
    - **Function**: In the agentic baseline, Lean compilation errors are categorised into two types with different recovery paths.
    - **Mechanism**: Based on the `smolagents` CodeAgent framework (ReAct loop), a surface guard first rejects code with forbidden tokens or incomplete proofs. Upon compilation failure: **Structural errors** (syntax, unknown identifier, missing module) trigger a **full rewrite** with type-specific hints; **semantic errors** (type mismatch, unsolved goals) prompt a patch agent to output a **minimal unified diff** for local repair. Max 25 ReAct cycles.
    - **Design Motivation**: 7B models (like DeepSeek-Prover-7B) actually saw **FV decrease with error feedback** (13% $\to$ 4.5%), indicating they lack the basic capability to learn from error signals. Larger models, however, benefit from fine-grained error branching. GPT-OSS-20B surged from 4.5% to 31% FV (zero-shot $\to$ agentic) but still fell short of the human-augmented FormalScience (100% FV).

### Loss & Training
No models were trained; this work focuses on inference-time pipeline design. All LLM calls utilized off-the-shelf models: GPT-5.1 + Claude-Opus-4.5 for data construction. Evaluated baselines include Qwen2.5-Coder-7B, DeepSeek-Prover-V2-7B, Kimina-Autoformalizer-7B, GPT-OSS-20B, Qwen3-Sonnet-14B (Claude-Sonnet-4.5 distilled to Qwen3-14B), Qwen3-Coder-30B, and GPT-5.1. GPT-4.1-mini served as LLM-as-judge (temp 0.2), with Qwen2.5-Coder-7B-Instruct performing $\approx 6000$ double-blind inter-judge agreement checks.

## Key Experimental Results

### Main Results

Formalisation scores of different pipelines on FormalPhysics (GPT-4.1-mini as judge):

| Method | Model | FV (%) | FQ (%) | LP (%) | MC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Zero-Shot | Kimina-7B | 51.5 | 6.5 | 10.5 | 9.5 |
| Zero-Shot | GPT-OSS-20B | 4.5 | 68.5 | 73.0 | 72.5 |
| Zero-Shot | GPT-5.1 | 14.5 | 79.5 | 76.5 | 77.0 |
| Self-Refine | GPT-5.1 | 17.0 | 82.5 | 82.0 | 82.0 |
| Agentic | Qwen3-Sonnet-14B | 52.0 | 1.0 | 10.5 | 6.5 |
| Agentic | GPT-OSS-20B | **31.0** | 73.0 | 72.5 | 73.0 |
| **FormalScience (ours)** | GPT-5.1 + Claude-4.5 | **100.0** | **73.5** | 72.0 | 72.5 |

Statistical comparison of FormalPhysics vs. existing Lean4 benchmarks (200 random samples):

| Dataset | Objects | Formulae | FV (%) | LP (%) | MC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| miniF2F | 3.14 ± 1.55 | 3.21 ± 1.53 | 88.0 | 92.0 | 92.0 |
| ProofNet | 3.67 ± 1.48 | 3.62 ± 1.52 | 95.5 | 77.5 | 77.5 |
| FormalMATH | 4.47 ± 2.45 | 4.53 ± 2.62 | 97.5 | 98.0 | 96.5 |
| Herald-Proof | 6.57 ± 2.32 | 6.42 ± 2.37 | 2.0 | 94.5 | 94.0 |
| **FormalPhysics** | **6.41 ± 2.34** | **6.22 ± 2.13** | **100.0** | 72.0 | 72.5 |

### Ablation Study

Ablation by pipeline complexity (using GPT-OSS-20B):

| Configuration | FV (%) | FQ (%) | LP (%) | MC (%) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Zero-shot | 4.5 | 68.5 | 73.0 | 72.5 | Prompt only, no feedback |
| + Self-refine | 7.5 | 70.5 | 77.0 | 79.0 | Compilation error feedback (+3pp FV) |
| + Agentic (ReAct + diff) | **31.0** | 73.0 | 72.5 | 73.0 | Structural/Semantic branching (+26.5pp FV) |
| + Human (FormalScience) | **100.0** | 73.5 | 72.0 | 72.5 | Human alignment oracle added (FV $\to$ 100) |

### Key Findings

- **FV and Semantic Alignment are Nearly Orthogonal**: Spearman and Pearson coefficients between FV and the mean of FQ/LP/MC are near 0 ($p>0.9$), proving the trade-off is structural. Kimina-7B is an extreme case—using compilation shortcuts to reach 51.5% FV while FQ is only 6.5%.
- **Self-refinement is not a "Free Lunch"**: Gains in FV/Alignment for 2$\times$ token cost were $<3$pp. However, under an independent 7B judge, GPT-5.1 gained +9pp, showing **metrics are sensitive to the choice of judge**.
- **Agentic Pipelines Narrow the FV Gap**: GPT-OSS-20B jumped from 4.5% to 31% without dropping alignment scores, proving ReAct + error branching allows mid-sized open-source models to use Lean effectively as a tool.
- **"Autoformalisation is an Emergent Ability"**: 14B or 30B models are not always better than 7B. Success depends on **Parameters $\times$ Neuro-Symbolic Integration $\times$ Test-time Scaling**. GPT-5.1 zero-shot yields only 15% FV but reaches 100% within the FormalScience pipeline.
- **Physics is Harder than Math Olympiads**: FormalPhysics has **approx. 2$\times$ more** Objects/Formulae than miniF2F, ProofNet, or Lean Workbook. It is comparable to Herald-Proof, but Herald-Proof's FV is only 2%, whereas FormalScience achieves 100%.

## Highlights & Insights

- **Human as Oracle, Not Annotator**: Traditional HITL involves humans writing code or labels. This paper uses humans for "yes/no" alignment judgments, embedding the person as a cheap oracle. This "human-as-lightweight-classifier" abstraction is transferable to tasks like RLHF preference labeling or code review.
- **Quantifying "Compilable $\neq$ Semantically Aligned"**: The authors introduce drift classifications like Notational Collapse and Abstraction Elevation. They clarify that **when $\ket{\Psi}$ is treated as a complex scalar $\Psi$ by Lean, it is no longer validating quantum mechanics**.
- **The Trade-off is More Valuable than the Method**: The finding that $\rho\approx 0$ between FV and alignment is a domain-level negative result. It suggests that a single compilation pass rate shouldn't be used as an RL reward.
- **Scalability Argument**: One expert / 1 month / $50 / 200 problems $\approx$ $0.25/problem. This implies a 10-person expert group could produce 2,000 problems in the same timeframe, making high-quality physics formalisation datasets feasible for fine-tuning.

## Limitations & Future Work

- **The authors acknowledge**: (1) FormalPhysics (200 problems) is an evaluation set, not yet large enough for fine-tuning; (2) Coverage is limited to Quantum Mechanics and Electromagnetism, excluding Relativity or Statistical Mechanics; (3) Mathlib's lack of native support for vector calculus and Dirac notation puts a ceiling on LP/MC (72%).
- **Hidden Scaling Bottlenecks**: While "human as binary oracle" sounds cheap, the expert must still **understand Lean code** to judge it, implying a higher barrier than expected. The paper does not report average HITL iterations per problem or time per judgment.
- **Judge Dependency**: The agreement between GPT-4.1-mini and Qwen2.5-7B is low ($\phi$ coefficient 0.28-0.37), and absolute scores differ by 9pp, suggesting LLM-as-judge scores remain noisy.
- **Future Directions**: (1) Build a DSL for Dirac/Vector calculus within Mathlib to raise the LP/MC ceiling; (2) Replace the "human oracle" with a fine-tuned alignment verifier for a fully automated v2; (3) Use drift categories (e.g., Notational Collapse) as negative penalties in RL rewards.

## Related Work & Insights

- **vs. miniF2F / ProofNet**: Those focus on Olympiads/Undergraduate math; this work tackles university physics. Their complexity (Objects $\approx$ 3) is half that of FormalPhysics, and they avoid domain-specific notation.
- **vs. Herald-Proof**: Similar complexity (Objects $\approx$ 6.5), but Herald-Proof's fully automated approach results in 2% FV. This work proves that in complex domains, **humans are necessary, not optional**.
- **vs. Kimina-Autoformalizer**: Kimina's 51.5% zero-shot FV relies on compilation-optimized shortcuts. This work exposes this as a case of Goodhart's Law—optimising FV $\neq$ optimising semantics.
- **vs. DeepSeek-Prover-V2 / smolagents CodeAgent**: This work adopts the ReAct/tool-calling framework but adds surface guards and structural/semantic error branching—a strategy applicable to all code-as-agent tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual oracles, physics domain, and drift classification is novel, though HITL formalisation itself is an established concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive testing across 3 pipelines $\times$ 7 models $\times$ 2 judges, including inter-judge agreement.
- Writing Quality: ⭐⭐⭐⭐ Clear math notation, pseudo-code in Alg.1/Alg.2, and intuitive drift classification figures.
- Value: ⭐⭐⭐⭐⭐ Opens the door for "Science Formalisation" and provides a high-quality physics benchmark alongside a clear, negative result regarding trade-offs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding](../../ICML2026/code_intelligence/centaureval_benchmarking_human-in-the-loop_value_in_agentic_coding.md)
- [\[ACL 2026\] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)
- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] Aligned Multi-View Scripts for Universal Chart-to-Code Generation](aligned_multi-view_scripts_for_universal_chart-to-code_generation.md)

</div>

<!-- RELATED:END -->
