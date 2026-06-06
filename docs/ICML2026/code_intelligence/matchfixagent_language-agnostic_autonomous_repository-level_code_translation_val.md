---
title: >-
  [Paper Note] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair
description: >-
  [ICML 2026][Code Intelligence][Code Translation] MatchFixAgent fully transforms "equivalence validation + repair" for repository-level code translation into an LLM-driven process. By replacing expensive cross-language in…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "Code Translation"
  - "Equivalence Validation"
  - "Multi-Agent"
  - "Language-Agnostic"
  - "Program Repair"
date: 2026-05-08
content_hash: 48946a7636ef1b59
---

# MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair

**Conference**: ICML 2026  
**arXiv**: [2509.16187](https://arxiv.org/abs/2509.16187)  
**Code**: https://github.com/Intelligent-CAT-Lab (artifacts repository)  
**Area**: Code Intelligence / LLM Agent / Program Analysis  
**Keywords**: Code Translation, Equivalence Validation, Multi-Agent, Language-Agnostic, Program Repair

## TL;DR
MatchFixAgent fully transforms "equivalence validation + repair" for repository-level code translation into an LLM-driven process. By replacing expensive cross-language interoperability engineering with six parallel semantic sub-analyzers (Control Flow, Data Flow, I/O, Library API, Exception, Specification), combined with a Test & Repair Agent and a Verdict Agent, it increases validation coverage from 71.6% to 99.2% and the proportion of repairable defects from 18.5% to 50.6% using only 1,650 lines of code.

## Background & Motivation
**Background**: Code translation (e.g., automatically rewriting Java projects into Rust or Python) is a core requirement for modernization. Existing methods for determining equivalence post-translation follow two main paths: executing original source tests on the target language (Oxidizer, AlphaTrans, Skel) or using differential fuzzing to compare results with random inputs.

**Limitations of Prior Work**: (1) Explosion in engineering effort—writing cross-language interoperability layers (FFI, type mapping, runtime bridging) for every language pair often involves tens of thousands of lines of code (Oxidizer: 19,052 lines; AlphaTrans: 10,859 lines). Suppporting $N$ languages requires $O(N^2)$ interfaces, making it unscalable. (2) Insufficient test coverage—original unit tests are often incomplete, leading to "false equivalence" where tests pass but logic differs; conversely, fuzzing generates many invalid inputs, causing "false non-equivalence." (3) Weak repair capabilities—when non-equivalence is found, systems either defer to humans or rely on weak feedback loops that fail in repository-level long call chains.

**Key Challenge**: Equivalence validation essentially requires "understanding the semantics of both ends." Symbolic methods are constrained by the sheer number of language pairs, while execution-based methods are limited by test quality. Both paths have reached a bottleneck.

**Goal**: (1) Develop a language-agnostic validation mechanism with low engineering costs; (2) Generate credible equivalence/non-equivalence verdicts without relying on the original project's test suite; (3) Provide direct repair patches rather than just detection.

**Key Insight**: LLMs have performed well in intra-language equivalence detection (Wei 2025, Maveli 2025). Rather than continuing to engineer cross-language interoperability, this task can be outsourced to LLMs. However, a single prompt asking "is this equivalent?" is too coarse and prone to hallucinations. The authors observe that equivalence can be decomposed into six orthogonal semantic dimensions. By letting LLMs focus on one dimension at a time, using a Test & Repair Agent for empirical verification, and a Verdict Agent for final arbitration, the system distributes responsibilities across specialized agents.

**Core Idea**: A lightweight multi-agent architecture featuring "6-way parallel semantic analysis + Test & Repair Agent + Verdict Agent" transforms cross-language equivalence validation from an engineering problem into an LLM task, reducing the per-language adaptation cost to approximately 280 lines of code.

## Method

### Overall Architecture
The input consists of a translation pair (source function + translated function) and two full projects; the output includes an equivalence verdict, a natural language report, and (if non-equivalent) a repair patch. The pipeline is structured in three stages:

1.  **Semantic Analyzer**: Uses Tree-sitter to extract Control Flow Graphs (CFG) and Data Flow Graphs (DFG), then triggers six parallel sub-analyzers (Control Flow, Data Flow, I/O, Library API, Exception, Specification). Each independently calls the LLM to provide a verdict and a JSON report.
2.  **Test Generator & Repair Agent**: Feeds the six reports to a coding agent (Claude Code), which writes and executes tests in both the source and target languages. If non-equivalence is found, it attempts to fix the translation, outputting a verdict, bilingual tests, and a patch.
3.  **Verdict Agent**: Comprehensive review layer that synthesizes semantic reports and test/repair results into a final verdict.

The entire system is 1,650 lines of Python code. Adding a new language requires only ~280 lines (mainly Tree-sitter adaptation + CFG/DFG extraction).

### Key Designs

1.  **6-way Parallel Semantic Decomposition + LLM-as-analyzer**:
    - **Function**: Decomposes the vague question of "are these equivalent" into six independent dimensions.
    - **Mechanism**: Each sub-analyzer prompt is highly structured—defining the role (expert in dimension X), the precise definition of equivalence for that dimension (e.g., I/O equivalence is refined into 5 criteria: input acceptance, consistency, side effects, boundary consistency, and performance), and requiring JSON output. CFG/DFG text representations and static patterns (try-catch, throw) are provided as context.
    - **Design Motivation**: Direct LLM answers lose detail. Decomposition allows for focused prompts, significantly improving reliability (accuracy dropped by 42.3% in ablations when decomposition was removed). This also minimizes engineering costs for new languages.

2.  **Two-level Analysis with Structural Similarity Short-circuit**:
    - **Function**: Uses inexpensive graph similarity checks before calling expensive LLMs to filter simple cases.
    - **Mechanism**: The Control Flow analyzer recodes CFG nodes (condition, loop, exception, etc.) and edges, then calculates a weighted Jaccard similarity: $similarity = 0.5 \times nodeSim + 0.5 \times edgeSim$. If it exceeds $\tau = 0.7$, it returns an equivalence verdict immediately. The Data Flow analyzer uses edit distance between def-use chain paths with the same threshold. This skips ~25% and ~35% of LLM calls, respectively.
    - **Design Motivation**: Many repository-level translations are "mechanical" (renaming variables, syntax changes). These do not require LLM reasoning. A strict threshold (0.7) ensures high-confidence short-circuits.

3.  **Test & Repair Agent + Verdict Agent (Empirical-Review Layers)**:
    - **Function**: The Test & Repair Agent uses semantic reports as clues to autonomously write/run tests and repair code. The Verdict Agent independently audits all outputs.
    - **Mechanism**: The Test & Repair Agent leverages tools (file R/W, shell execution) and is explicitly prompted to verify non-equivalence via execution. The Verdict Agent performs cross-validation between the reasoning and empirical results to filter hallucinations.
    - **Design Motivation**: Static reasoning may err; execution provides empirical proof. The multi-agent "Analysis-Evidence-Arbitration" split is the most critical component identified in ablation studies.

## Key Experimental Results

### Main Results
Benchmarks: 2,219 function pairs from 4 SOTA works (AlphaTrans, Oxidizer, Skel, SpecTra), covering 6 language pairs and >900K LoC.

| Dimension | Previous SOTA (Combined) | MatchFixAgent | Gain |
|-----------|--------------------------|---------------|------|
| Verdict Rate (pairs with a verdict) | 71.6% | **99.2%** | +27.6pp |
| Agreement Rate (when both have verdicts) | — | 72.8% (1571 pairs) | — |
| Human-judged correctness (disagreements) | 39.3% | **60.7%** | +21.4pp |
| Successfully repaired non-equivalent pairs | 18.5% | **50.6%** | **+32.1pp** |
| Framework Codebase Size | 3,843 ~ 19,052 LoC | **1,650 LoC** | 2~12x reduction |

In the Oxidizer subset (192 pairs), MatchFixAgent provided 132 EQ / 59 NEQ / 1 VF (Validation Failure). In disagreements, MatchFixAgent was correct 84.1% of the time.

### Ablation Study

| Configuration | Accuracy | Token Usage |
|---------------|----------|-------------|
| Full (6 Analyzers + Test/Repair + Verdict) | 100% (Baseline) | 100% (Baseline) |
| w/o Decomposition + w/o in-the-loop Test | **−42.3pp** | −5.2% |
| Adaptation engineering per language (LoC) | — | **~280 LoC** |

Transferability: The architecture performs consistently across different LLMs (e.g., Claude 3.7 vs Others) and agent frameworks, proving its robustness.

### Key Findings
- **Multi-Agent specialization outweighs token savings**: Removing decomposition and empirical testing saved only 5.2% of tokens but caused a 42.3pp drop in accuracy. 
- **Lightweight short-circuiting is vital for cost control**: Using $\tau = 0.7$ for CFG/DFG allows the system to focus compute power on truly difficult samples.
- **Repair capability stems from "Understanding before Acting"**: The 32.1pp lead in repair rate is attributed to the 6-dimension semantic reports, which guide the agent unlike the blind "retry on failure" approach of previous works.
- **False results concentrate in disagreement samples**: Many traditional test-based methods fail systematically on complex samples where MatchFixAgent remains correct.

## Highlights & Insights
- **Successful replacement of "Engineering with LLM Tasks"**: Cross-language interoperability is notoriously difficult. Replacing it with "Tree-sitter + 6 Prompts" reduces complexity by an order of magnitude, a tradeoff previously unthinkable in the PL community but highly effective for repository-level translation.
- **Reusable Multi-Agent Template**: The "Map (Parallel tasks) -> Reduce (Empirical testing) -> Review (Verdict)" structure is transferable to other consistency-checking tasks like refactoring validation or API compatibility.
- **Engineering Pragmatism**: Using Jaccard/edit distance as a cheap filter with strict thresholds provides a reproducible 사례 for controlling costs in LLM agent systems.

## Limitations & Future Work
- **Limitations**: (1) Verdicts are not absolute proofs; MatchFixAgent failed in 39.3% of disagreement cases. (2) Evaluation is limited to 6 common language pairs; niche languages (e.g., Haskell) are untested. (3) Data flow analysis is purely syntactic and does not handle complex aliasing or context-sensitivity.
- **Future Work**: The Verdict Agent could be upgraded to a "Formal Verification Bridge" using SMT solvers for specific dimensions. "Dimension Adaptation" could dynamically decide which sub-analyzers to run based on initial similarity to further reduce costs.

## Related Work & Insights
- **vs. AlphaTrans / Oxidizer / Skel**: These rely on cross-language interoperability and original tests, which are engineering-heavy and limited by coverage. MatchFixAgent uses an LLM multi-agent approach, drastically improving verdict rates and repair rates.
- **vs. SpecTra / Differential Fuzzing**: Fuzzing creates noise with illegal inputs. MatchFixAgent’s IO Analyzer proposes counterexamples based on semantic understanding.
- **vs. Feedback-driven re-prompting (Zhang 2025)**: Simple feedback loops fail in long repository call chains. MatchFixAgent's 6-dimensional reports provide a structured roadmap for repair.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Although not a new algorithm, it is the first to prove the viability of outsourcing cross-language analysis to a multi-agent LLM system with solid engineering verification.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid coverage across language pairs and LoC, with systematic human auditing of disagreements.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and tight logic.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses high-value industrial code migration. The low engineering cost (1,650 LoC) makes it immediately applicable to production pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](../../ACL2026/code_intelligence/swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ICML 2026\] SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale](swe-rebench_v2_language-agnostic_swe_task_collection_at_scale.md)
- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](../../ACL2026/code_intelligence/reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)
- [\[ACL 2026\] Bootstrapping Code Translation with Weighted Multilanguage Exploration](../../ACL2026/code_intelligence/bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)
- [\[ICML 2026\] NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents.md)

</div>

<!-- RELATED:END -->
