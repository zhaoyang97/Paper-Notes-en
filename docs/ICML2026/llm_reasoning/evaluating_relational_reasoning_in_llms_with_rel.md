---
title: >-
  [Paper Note] Evaluating Relational Reasoning in LLMs with REL
description: >-
  [ICML 2026][LLM Reasoning][Relational Complexity] The authors adopt "Relational Complexity" (RC) from cognitive science—defined as the number of independent variables that must be simultaneously bound in a single reasoni…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Relational Complexity"
  - "Raven's Progressive Tensors"
  - "Homoplasy"
  - "Molecular Isomers"
  - "High-arity Binding"
date: 2026-05-08
content_hash: 23aae33901204c6a
---

# Evaluating Relational Reasoning in LLMs with REL

**Conference**: ICML 2026  
**arXiv**: [2604.12176](https://arxiv.org/abs/2604.12176)  
**Code**: Available (Project Page + GitHub + Hugging Face)  
**Area**: LLM Evaluation / Relational Reasoning / Scientific Reasoning Benchmark  
**Keywords**: Relational Complexity, Raven's Progressive Tensors, Homoplasy, Molecular Isomers, High-arity Binding

## TL;DR
The authors adopt "Relational Complexity" (RC) from cognitive science—defined as the number of independent variables that must be simultaneously bound in a single reasoning step—as a unified axis for measuring task difficulty. They construct REL, a generative benchmark spanning algebra, biology, and chemistry, finding that the accuracy of frontier LLMs (Claude Opus 4.5 / Gemini 3 Pro / GPT-5.2) monotonically decreases as RC increases, a failure mode that persists despite test-time compute, ICL, or external tools.

## Background & Motivation
**Background**: Current LLM evaluations often use input length, token count, entity count, or multi-hop counts as proxies for "difficulty." While graph-based relational reasoning benchmarks (multi-hop QA, knowledge graphs) exist, they typically couple relational structures with specific representations.

**Limitations of Prior Work**: (1) "Difficulty" may stem from longer prompts, complex representations, or increased background knowledge requirements rather than genuine relational reasoning bottlenecks; (2) existing evaluations fail to distinguish between whether a model is incapable or saturated, making benchmark scores difficult to interpret; (3) graph-based evaluations are often restricted to graph structures and do not transfer to real-world scientific scenarios like algebra, chemistry, or biology.

**Key Challenge**: The true dimension of difficulty—**relational binding arity (the number of independent slots that must be held simultaneously)**—is confounded by coarse proxies such as entity count or prompt length. Models may perform well on tasks that "look hard" (high entity count) but have low arity, yet collapse on "small" tasks with high arity, leading to distorted benchmark reports.

**Goal**: To decompose the problem into three sub-questions: (i) formalize "relational difficulty" as a controllable and parameterizable quantity; (ii) observe LLM behavior across multiple scientific domains by varying only RC while freezing other variables; (iii) verify whether RC is indeed the primary driver of performance rather than a spurious correlation with variables like prompt length or entity count.

**Key Insight**: The authors borrow the concept of Relational Complexity from cognitive scientists like Halford, who used it to study Raven's Progressive Matrices. RC posits that the number of independent slots required for a reasoning step equals the arity of the relation. This metric is representation-agnostic and can be independently tuned across different domains (numeric matrices, molecular sets, phylogenetic trees).

**Core Idea**: Use "the number of independent variables requiring simultaneous binding = relational arity" as the unified difficulty axis (RC), paired with "the difficulty of identifying/representing a single slot" (Operand Complexity, OC) to isolate representation complexity. This enables the construction of generative tasks across algebra, biology, and chemistry where RC is adjustable while confounding variables are controlled, isolating the "high-arity reasoning collapse" failure mode from noise.

## Method

### Overall Architecture
REL is a **generative benchmark framework** consisting of three subsets: **REL-A (Algebra)**, based on Raven's Progressive Matrices and a newly introduced tensorized extension, RPT; **REL-B (Biology)**, requiring models to identify homoplasy (convergent evolution) in multiple sequence alignments (MSA) and phylogenetic trees; and **REL-C (Chemistry)**, featuring three tasks with varying RC centered on constitutional isomers, maximum common substructures (MCS), and missing isomer completion. These share a formal definition of RC and OC, exposing generator parameters to systemically scan RC while freezing entity count, sequence length, and prompt length. The pipeline follows: "Parameters → Synthetic Problems → LLM Response → Accuracy Comparison grouped by RC."

### Key Designs

1. **RC / OC Dual-axis Formalization and Tensorized Raven Extension**:
    - **Function**: Decomposes "relational reasoning difficulty" into two orthogonal dimensions: RC is the number of independent variables/operands requiring simultaneous binding (relational arity), and OC is the difficulty of identifying/representing a single slot. It demonstrates mechanical RC calculation on Raven's Progressive Matrices and generalizes RPM to $n$-dimensional Raven's Progressive Tensors (RPT).
    - **Mechanism**: For an $n \times n$ RPM, seven rules are designed to cover $\text{RC} \in \{1, 2, n, 4, 5, 6\}$. Examples include A1 (Constant) where $\text{RC}=1$, A2 (Progression) where $\text{RC}=2$, A3 (Permutation) where $\text{RC}=n$, and A4 (Row-Sum) where $\text{RC}=n$. Tensorization pushes the theoretical upper bound to $\text{RC}_{n\text{-dim}} \le 3^{n}-1$, allowing RC to reach 4–6 on small inputs by adding a single dimension while keeping entity counts nearly constant.
    - **Design Motivation**: Traditional RPMs have an RC cap of 4, which is insufficient for testing modern LLMs. Formalizing RC as a representation-decoupled value allows scanning RC independently of input token count, isolating its effect from prompt length or entity count—providing the foundation for subsequent regression analysis.

2. **REL-B1: Homoplasy Detection (Biological RC Injection)**:
    - **Function**: Given a phylogenetic tree and corresponding MSA, the model must (a) determine if homoplasy exists (different lineages evolving the same motif independently) and (b) accurately list all taxa involved. Success requires both steps to be correct.
    - **Mechanism**: A synthetic data generator is controlled by four parameters: number of homoplastic taxa $N_{ht}$, number of leaves $N_{\text{leaves}}$, sequence length $L_{\text{seq}}$, and motif length $L_{\text{motif}}$. Here, $\text{RC} = N_{ht}$, as the model must simultaneously hold and verify the positions of all homoplastic taxa on the tree. The other parameters serve as non-RC confounding factors for ablation.
    - **Design Motivation**: (1) Projects abstract "relational arity" onto a real biological reasoning task, proving the framework extends beyond synthetic puzzles; (2) variation of $N_{ht}$ while freezing other parameters allows for quantified impact via multiple regression/GVIF collinearity analysis; (3) homoplasy represents a typical "multi-lineage joint binding" problem in scientific reasoning, ensuring external validity.

3. **REL-C Triple Tasks: Decoupling RC and OC in Chemistry**:
    - **Function**: Three tasks with varying RC/OC ratios are designed using molecular sets (SMILES representation): C1 (Isomer set classification; RC=2, Low OC), C2 (Maximum Common Substructure/MCS; RC=2, Medium OC), and C3 (Missing isomer completion; High RC, High OC).
    - **Mechanism**: C1 requires binary comparisons of molecular formulas (sequential binary binding). C2 also uses binary binding but requires finding MCS between two molecules, significantly increasing OC. C3 necessitates holding the "complete isomer space" and the "observed subset" simultaneously, with the space size $N_{\text{isomers}}$ averaging 29, precluding simplified pair-wise binary updates. C2 metrics utilize a bidirectional substructure match $\text{IsSubstructure} = \tfrac{1}{2}(S_{\text{pred}\subseteq\text{true}} + S_{\text{true}\subseteq\text{pred}})$ to capture both precision and completeness.
    - **Design Motivation**: C1 vs. C2 serves as a control for "Equal RC / Different OC," verifying that OC independently degrades performance. C1/C2 vs. C3 serves as a control for "Increased OC and Significantly Increased RC," demonstrating that the drop caused by RC is much steeper, establishing it as the primary driver.

### Loss & Training
No models were trained; evaluations were conducted on Claude Opus 4.5, Gemini 3 Pro Preview, and GPT-5.2. Protocol: REL-A uses 8-way multiple choice (trivial accuracy 12.5%); REL-B1 requires exact match for existence and taxa sets; REL-C uses strict matching after SMILES canonicalization (IsSubstructure for C2, recall/precision/F1 for C3). Inference-time interventions included test-time compute (max-tokens up to 16384), one-shot in-context learning (10% samples for REL-C), and tool use (RDKit for REL-C3).

## Key Experimental Results

### Main Results

| Task | RC Range | Primary Metric | Performance Change with Increasing RC |
|------|----------|----------------|---------------------------------------|
| REL-A1/A2 | RC=1/2 | accuracy | Three models achieve 91% even on $30 \times 30$ RPM |
| REL-A3 (Permutation) | RC=n | accuracy | At $30 \times 30$, Claude/Gemini drop to trivial 12%; GPT-5.2 drops ~40% |
| REL-A4 (Row-Sum) | RC=n | accuracy | Only GPT-5.2 scores 21% at $9 \times 9$; others fail completely |
| REL-A7 (Neighborhood Sum) | RC=6 (Fixed) | accuracy | All three models ~12% (≈ trivial) |
| REL-B1 (homoplasy) | RC=$N_{ht}$=4→25 | exact match | 35% → 1% (Model average) |
| REL-C1 → C3 | RC↑ + OC↑ | task completion | 65.7% → 38.1% → 26.0% (Total drop of 39.7%) |

### Ablation Study

| Intervention | Setting | Key Findings |
|--------------|---------|--------------|
| Multiple Regression (REL-B1) | RC vs. motif ratio / seq len / dist / prompt len | RC explanatory power: Claude 24% / Gemini 32% / GPT 44%; next factor max 17% |
| GVIF Collinearity | Five variables | GVIF for RC, distance, and motif ratio all < 1.3; no collinearity threat |
| Test-Time Compute | 4k / 8k / 16k tokens | REL-A4/A5 gain only 2-3%; REL-C gains 0.4% average; fails to bridge RC gap |
| In-Context Learning | REL-C one-shot (10% samples) | C1 +6.6% / C2 +3.4% / C3 +6.0%; relative rankings remain unchanged |
| Tool Use (RDKit) | REL-C3 Full | Average recall only 0.094; still decreases with molecule count (0.109 → 0.079) |

### Key Findings
- **RC is the true bottleneck**: Multiple regression on REL-B1 shows RC has 2-6 times the explanatory power of the next strongest factor and nearly zero collinearity with entity count/prompt length—confirming RC is not a spurious correlation of "long prompts."
- **Persistent Failure Modes**: Test-time compute (8x tokens), ICL, and external RDKit tools yield only marginal or zero improvements, suggesting high-arity binding is an architectural bottleneck rather than a lack of "thinking time" or examples.
- **OC and RC are separable**: In REL-C1 vs. C2 where RC=2, increasing OC dropped completion from 65.7% to 38.1%. Moving from C2 to C3 (significant RC increase) dropped it another 12%, showing the effects are additive but RC is steeper.
- **Input size is unreliable**: On REL-A5/A6, models performed better as input size increased (more redundant signals), proving entity count is not a monotonic proxy for difficulty.

## Highlights & Insights
- **Bridging CogSci and LLM Eval**: Applying the concept of RC from 1990s RPM research to contemporary LLMs reveals it as a sharp analytical tool, demonstrating the value of cross-disciplinary insights for evaluation.
- **Generative + Parameterized**: REL is not a fixed dataset but a generator that can scan RC, providing natural resistance to contamination.
- **RPT Upper Bound of $3^n-1$**: Adding a single dimension pushes RC to 26 or even 80 without the engineering burden of linearly expanding inputs. This tensorization approach could benefit other grid-based benchmarks (e.g., ARC).
- **Tool Use Failure is a Notable Negative Result**: Even with RDKit, C3 average recall was only 0.094. The bottleneck is not "molecular parsing" but "simultaneously holding relations between multiple isomers," suggesting agent design must also address the arity bottleneck.

## Limitations & Future Work
- The authors acknowledge that multiple-choice formats may mask granular failures, context-length limits caused some invalid responses, and tasks remain somewhat synthetic.
- Internal observations: (1) Evaluation focused only on three closed-source models; scaling behavior in open-source or smaller models remains unknown; (2) REL-B1's direct equivalence of RC to $N_{ht}$ is a simplification that ignores topological distance; (3) lack of reasoning chain (CoT) analysis prevents identifying which specific binding step fails; (4) the RC definition assumes simultaneous holding, ignoring potential streaming/chunking strategies.
- Improvement ideas: Incorporate "topological RC" (structural distance of binding paths) and use mechanistic interpretability tools (attention patterns/activation patching) to locate specific heads that fail at high RC for targeted fine-tuning.

## Related Work & Insights
- **vs. Liu et al. (2025a) Graph Benchmarks**: While they use generative relational reasoning, they modify only the graph structure. REL lifts RC to a task-agnostic level and extends it to non-graph scenarios like molecules and phylogenies.
- **vs. Camposampiero et al. (2025a/b) I-Raven-X**: The authors deliberately avoid **perceptual noise/confounders**, focusing purely on relational binding to ensure clean attribution of the RC effect.
- **vs. ProteinGym / DNALongBench / TAPE**: These bio-benchmarks focus on single sequences or pairs. REL-B1 is the first to formalize "multi-sequence joint reasoning + phylogenetic constraints" as a tunable RC task.
- **vs. Multi-hop QA**: Multi-hop links RC=2 relations sequentially. REL pushes single-step RC to 6+, essentially being orthogonal to "hops." Future work could combine "hop × arity" for a 2D difficulty space.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Formal transfer of CogSci RC/OC to LLM evaluation with RPT extensions is rare and conceptually deep.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three domains + three frontier models + regression/GVIF + inference interventions. Lacks scaling experiments on open-source models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear definitions and intuitive visualizations; RPT formulas are occasionally dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides a parameterizable, contamination-resistant, and interpretable ruler for the evaluation community, addressing the "benchmark saturation" debate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TRACE: Evaluating LLM CoT Reasoning Process Quality with the Toulmin Argumentation Model](trace_toulmin-based_reasoning_assessment_through_constructive_elements_for_llm_c.md)
- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)
- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](../../ACL2026/llm_reasoning/self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[NeurIPS 2025\] Note 6: Self-Evaluating LLMs - Step-Level Confidence Estimation for Multi-Step Tasks](../../NeurIPS2025/llm_reasoning/value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[ICML 2026\] Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs](deliberate_evolution_agentic_reasoning_for_sample-efficient_symbolic_regression_.md)

</div>

<!-- RELATED:END -->
