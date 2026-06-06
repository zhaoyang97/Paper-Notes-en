---
title: >-
  [Paper Note] LLMs Lean on Priors, Not Programming Language Semantics
description: >-
  [ICML 2026][Interpretability][Formal Semantics] The authors construct PLSemanticsBench—pairing a featherweight C language $\text{C}^{\star}$ with two formal systems…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Formal Semantics"
  - "Program Execution"
  - "Rule Conditioning"
  - "Semantic Perturbation"
  - "Code Understanding"
date: 2026-05-08
content_hash: 02da238c3c9ca5ec
---

# LLMs Lean on Priors, Not Programming Language Semantics

**Conference**: ICML 2026  
**arXiv**: [2510.03415](https://arxiv.org/abs/2510.03415)  
**Code**: https://EngineeringSoftware.github.io/PLSemanticsBench (Available)  
**Area**: Interpretability / LLM Code Reasoning / Formal Semantics Evaluation  
**Keywords**: Formal Semantics, Program Execution, Rule Conditioning, Semantic Perturbation, Code Understanding

## TL;DR
The authors construct PLSemanticsBench—pairing a featherweight C language $\text{C}^{\star}$ with two formal systems, small-step operational semantics $\mathbb{S}$ and K-semantics $\mathbb{K}$. By systematically perturbing semantics via KeywordSwap (swapping operator semantics such as `+`/`-`) and KeywordObf (replacing them with rare Caucasian-Albanian symbols) and testing 11 frontier LLMs, they found: the 90% final state prediction accuracy under standard semantics drops by 40–60 percentage points under semantic perturbation; long-range rule tracking accuracy reaches only 35% at most. This indicates that contemporary LLMs primarily rely on pre-trained lexical priors rather than reasoning according to explicit formal rules.

## Background & Motivation
**Background**: Mainstream LLM code capability evaluation follows two paths: first, end-to-end benchmarks for output prediction, program repair, or code generation (HumanEval, MBPP, CodeContests); second, mimicking "step-by-step execution" via chain-of-thought. Both types of benchmarks assume the language encountered by the model is one seen during pre-training, where symbol meanings align with conventional expectations.

**Limitations of Prior Work**: This setup cannot distinguish between two distinct capabilities: (a) the model actually performing formal reasoning according to provided rules; (b) the model simply mapping familiar symbols (`+`, `while`, `if`) to statistical associations learned during pre-training to guess a plausible answer. When (b) dominates, high scores do not imply the model understands formal semantics.

**Key Challenge**: To isolate "semantic conditioning" from "syntactic familiarity," one must systematically rewrite semantics while preserving the syntactic appearance. This is a natural capability of formal semantics (structured operational semantics, K-semantics), as rules are symbolic and atomized, allowing the `E-Add` rule for `+` to be replaced without altering the syntax tree. However, existing LLM code evaluations have never introduced this tool.

**Goal**: (1) Design a benchmark capable of mechanical semantic perturbation; (2) Decompose "reasoning by rules" into four individually testable abilities: global composition (H1), rule selection without state changes (H2), long-range maintenance (H3), and adherence to provided rules under new semantics (H4); (3) Systematically quantify the performance of frontier LLMs across these four capabilities.

**Key Insight**: Choosing C over Python avoids indentation-sensitive syntax that couples "block structure recovery" with "semantic reasoning"; using a featherweight grammar eliminates noise like pointers and structs. By applying the same program to both $\mathbb{S}$ (fine-grained, one rule per atomic calculation) and $\mathbb{K}$ (coarse-grained, rewriting semantics) systems, the rule granularity variable is controlled.

**Core Idea**: Using "mechanically replaceable semantics" as a probe—the same syntax tree should yield three different execution results under std/swap/obf semantics. If a model truly reasons according to provided rules, it should switch its answers accordingly; otherwise, it is relying on pre-training priors.

## Method

### Overall Architecture
The PLSemanticsBench pipeline consists of four steps: (1) Defining $\text{C}^{\star}$ syntax using EBNF and providing complete rule texts for both $\mathbb{S}$ and $\mathbb{K}$ systems; (2) Curating 162 Human-Written programs (from LeetCode/HumanEval/etc.), 165 LLM-Translated programs (via Qwen2.5-Inst 32B), and 165 Fuzzer-Generated programs across three complexity levels (cyclomatic complexity from 3 to 100, trace length from 20 to 190); (3) Mechanically replacing standard semantics with KeywordSwap (swapping operators like `+`/`-`, `<`/`>`, `&&`/`||`) and KeywordObf (replacing `=`, `+`, `if-else`, `while` with rare Caucasian-Albanian alphabet symbols); (4) Feeding the "program + rule text" to models to perform three types of tasks, compared against ground-truth from ANTLR4 / K-framework.

### Key Designs

1.  **Dual-track Formal Semantics ($\mathbb{S}$ vs $\mathbb{K}$) for Granularity Control**:
    *   **Function**: Testing the same programs under two rule granularities to distinguish whether a model "cannot distinguish fine-grained rules" or "truly cannot execute."
    *   **Mechanism**: $\mathbb{S}$ uses Gentzen-style inference rules to write each atomic calculation as a transition (e.g., `E-Add` only handles $\langle v_1+v_2,\sigma\rangle \to_E v_3$); $\mathbb{K}$ uses a rewriting style to merge multiple steps into a coarse rule. Execution is formalized as $\llbracket s\rrbracket_\Psi(\sigma_0) = (\sigma_n, \bigoplus_{i,j}[(\sigma_i, r_{i,j})])$, recording both state sequences $\sigma_i$ and rule sequences $r_{i,j}$.
    *   **Design Motivation**: Notation warm-up experiments showed that models confuse synonymous rules more severely in $\mathbb{S}$ (concentration in Arithmetic/Relational rules), whereas $\mathbb{K}$ has less confusion due to fewer rules. This allows downstream failures to be attributed to "granularity discrimination" or "global reasoning" rather than "inability to read notation."

2.  **Dual-axis Semantic Perturbation (KeywordSwap / KeywordObf)**:
    *   **Function**: Decoupling "semantic conditioning" from "syntactic familiarity"—the former keeps syntax but rewrites operator semantics to conflict with priors; the latter uses unfamiliar symbols to erase syntactic priors entirely.
    *   **Mechanism**: KeywordSwap swaps operators within the same family (e.g., `+`↔`-`, `*`↔`/`). Thus, `x+y` in the source code should be executed as subtraction. KeywordObf replaces fundamental keywords with Caucasian-Albanian letters, making `x ⷠ y` equivalent to `x + y`. Both perturbations maintain the transition rule structure, changing only the mapping from symbols to rules.
    *   **Design Motivation**: Testing Swap measures "if priors can be overridden by new rules," while testing Obf measures "execution capability without priors." The intersection identifies failure modes: if the drop in Swap is much larger than in Obf, it proves the model is "hijacked" by familiar symbols.

3.  **PredState / PredRule / PredTrace Tasks Aligned with H1–H3**:
    *   **Function**: Decomposing "rule-based reasoning" into three sub-capabilities: composing rules for a final state, selecting rules in a single step without state changes, and maintaining consistency over long traces.
    *   **Mechanism**: PredState asks for $\llbracket\mathcal{P}\rrbracket_\Psi^\sigma(\sigma_0)$ (final variable table) to check global composition. PredRule provides an expression-step window and asks the model to pick the correct rule from 5 candidates (distractors sampled by family/role). PredTrace requires outputting the entire $(\sigma_i, r_{i,j})$ sequence. Quantitative metrics $\Delta_{\text{cnd}} = \mathrm{Acc}_{\text{std}}^{\square} - \mathrm{Acc}_{\text{na}}$ and $\Delta_{\text{is}} = \mathrm{Acc}_{\square'}^{\square} - \mathrm{Acc}_{\text{std}}^{\square}$ are used.
    *   **Design Motivation**: Testing only the final state blends "rule usage" with "intuition." PredRule eliminates shortcuts like inferring rules from final values. PredTrace forces the model to follow a trace where deviations from priors accumulate and amplify.

## Key Experimental Results

### Main Results: PredState on Human-Written Dataset ($\mathbb{S}$ Formalization)

| Model | $\text{Acc}_{\text{na}}$ | $\text{Acc}_{\text{std}}$ ($\Delta_{\text{cnd}}$) | $\text{Acc}_{\text{swap}}$ ($\Delta_{\text{is}}$) | $\text{Acc}_{\text{obf}}$ ($\Delta_{\text{is}}$) |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-Inst 14B | 33 | 28 (-5) | 6 (-22) | 8 (-20) |
| Llama-3.3 70B | 32 | 25 (-7) | 5 (-20) | 12 (-13) |
| GPT-4o-mini-CoT | 68 | 65 (-3) | 3 (-62) | 27 (-38) |
| DS-Qwen 32B | 84 | 95 (+11) | 3 (-92) | 77 (-18) |
| DS-Llama 70B | 80 | 89 (+9) | 2 (-87) | 59 (-30) |
| QwQ 32B | 93 | 98 (+5) | 7 (-91) | 86 (-12) |
| o3-mini | 94 | 100 (+6) | 63 (-37) | 95 (-5) |
| GPT-5-mini | 100 | 100 (0) | 79 (-21) | 99 (-1) |
| Gemini-2.5-pro | 93 | 99 (+6) | 98 (-1) | 100 (+1) |

### Ablation Study: Complexity/Long-range (PredState on Fuzzer-Generated $\mathbb{S}$)

| Model | Human-Written std | Fuzzer std | Human swap | Fuzzer swap |
| :--- | :--- | :--- | :--- | :--- |
| QwQ 32B | 98 | ~82 | 7 | 4 |
| GPT-5-mini | 100 | 95 | 79 | 65 |
| Gemini-2.5-pro | 99 | (Nearly flat) | 98 | (Nearly flat) |
| Most Reasoning Models | 80–100 | Drop >40 pts | Collapsed | Collapsed |

### Key Findings
*   **CoT saves the length, not the priors**: Adding CoT improves non-reasoning models by nearly 50 points under std, but gains disappear under swap and drop to ~40 points under obf. This shows CoT helps "expand execution steps" but doesn't force models to read new rules.
*   **Swap >> Obf Asymmetry**: Almost all models show a significantly larger drop in swap than in obf (e.g., DS-Qwen 32B drops 92 points in swap but only 18 in obf). This confirms "familiar symbols are a trap"—once given strange symbols, models actually adhere to rules.
*   **Gemini-2.5-pro is the outlier**: It maintains ≥98% under swap, being the only model among 11 to demonstrate the ability to "override priors with rules." Others (including GPT-5-mini, o3-mini) are hijacked by priors to varying degrees.
*   **Long-range consistency is nearly zero**: On PredTrace, only a few models score above zero, with the best reaching only 35%. Even if single rules are correct, cumulative deviations on long traces lead to collapse.
*   **Structural bottlenecks**: Control flow depth is the primary stressor for Human-Written programs, while data flow and program scale (Halstead Volume, trace length) dominate failures in LLM-Translated and Fuzzer-Generated sets. Global state tracking on long traces is the weakest link.

## Highlights & Insights
*   **Novel use of formal semantics as probes**: Traditional perturbations (variable renaming, comments) only touch the surface. KeywordSwap directly challenges pre-trained priors by preserving syntax but swapping semantics—a "controlled stress test" unique to formal semantics.
*   **Separating $\Delta_{\text{cnd}}$ and $\Delta_{\text{is}}$**: Absolute accuracy can be masked by model base levels. Quantitative comparison of whether these two deltas are positive or negativeQualitatively determines if a model is truly "conditioning" on rules.
*   **Asymmetry of Swap >> Obf**: Usually, "unfamiliar symbols" are considered harder, but this study proves "familiar symbols + counter-intuitive meanings" is more lethal. This suggests that in LLM evaluation, noise does not always equal difficulty; abnormal semantics are better at exposing capability ceilings.
*   **Prompt Engineering Insight**: Coarse-grained rules ($\mathbb{K}$) are more stable for notation understanding but provide lower information density for fine-grained tasks like PredRule. Rule granularity must match task granularity.

## Limitations & Future Work
*   **Ours acknowledges**: The study is limited to Featherweight C (no pointers, structs, or concurrency); only $\mathbb{S}$ and $\mathbb{K}$ systems were used; CoT prompt templates were not extensively searched.
*   **Potentially underestimated reasoning**: Since models used zero/one-shot without few-shot rule instruction or fine-tuning, the conclusion that "LLMs don't understand rules" should be read as "LLMs do not automatically reason by rules in prompt-only settings."
*   **Future Directions**: (1) Fine-tuning rules into weights to see if swap can be overridden; (2) Using RL with "rule consistency rewards" for PredTrace; (3) Extending the benchmark to custom DSLs (e.g., SQL dialects) to test rule-switching in compliance scenarios.
*   **Fuzzer dataset bias**: Fuzzer-Generated programs might fall outside natural code distributions, meaning swap failures on Fuzzer data might reflect OOD robustness rather than failures in rule tracking.

## Related Work & Insights
*   **Comparison with CRUXEval/LiveCodeBench**: While those measure execution on "standard Python/C++," this study measures "switching to new rules." High scores on standard benchmarks may largely be prior-reuse; the two must be combined to fully characterize LLM code capability.
*   **Comparison with Counterfactual Reasoning**: NL counterfactuals often use coarse, non-mechanical perturbations. This study uses formal semantics for precise symbolic replacement and automated ground-truth verification via K-framework.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Using formal semantics as a probe for LLM reasoning is unique; the Swap/Obf contrast is clever.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models × 2 formalisms × 3 perturbations × 3 datasets × 3 tasks, with detailed attribution.
*   Writing Quality: ⭐⭐⭐⭐ High notation density, but primered well with clearly numbered hypotheses.
*   Value: ⭐⭐⭐⭐⭐ Provides a mechanically verifiable template for answering "Does the LLM actually reason?" PLSemanticsBench could become a standard for tracking rule-following abilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Expand Neurons, Not Parameters](expand_neurons_not_parameters.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](adaptive_querying_with_ai_persona_priors.md)
- [\[ICML 2026\] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)
- [\[ICML 2026\] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](position_zeroth-order_optimization_in_deep_learning_is_underexplored_not_underpo.md)
- [\[ACL 2026\] Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States](../../ACL2026/interpretability/linear_probes_detect_task_format_not_reasoning_mode_in_language_model_hidden_sta.md)

</div>

<!-- RELATED:END -->
