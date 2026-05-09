---
title: >-
  [Paper Note] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations
description: >-
  [ACL 2026][Tool invocation] This paper identifies and formalizes "structural alignment bias" in LLM tool invocations — the tendency of LLMs to invoke a tool when query attributes can be effectively mapped to tool parameters, even when the tool's functionality is irrelevant to the user's goal. The authors construct the SABEval dataset to decouple structural alignment from semantic relevance, apply contrastive attention attribution (CAA) to reveal two competing internal pathways (semantic checking vs. structural matching), and propose a path rebalancing strategy that achieves 80% relative error reduction.
tags:
  - ACL 2026
  - Tool invocation
  - structural alignment bias
  - irrelevant tool rejection
  - interpretability
  - attention attribution
date: 2026-05-08
content_hash: 726b3810650fb432
---

# Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations

**Conference**: ACL 2026
**arXiv**: [2604.11322](https://arxiv.org/abs/2604.11322)
**Code**: [GitHub](https://github.com/along-l/irrelevant-tool)
**Area**: Interpretability
**Keywords**: Tool invocation, structural alignment bias, irrelevant tool rejection, interpretability, attention attribution

## TL;DR
This paper identifies and formalizes "structural alignment bias" in LLM tool invocations — the tendency of LLMs to invoke a tool when query attributes can be effectively mapped to tool parameters, even when the tool's functionality is irrelevant to the user's goal. The authors construct the SABEval dataset to decouple structural alignment from semantic relevance, apply contrastive attention attribution (CAA) to reveal two competing internal pathways (semantic checking vs. structural matching), and propose a path rebalancing strategy that achieves 80% relative error reduction.

## Background & Motivation

**Background**: The ability of LLMs to use external tools has become a critical capability. In real-world deployments, models frequently encounter tools irrelevant to the user's query, and the correct behavior in such cases is to refrain from invocation.

**Limitations of Prior Work**: (1) LLMs exhibit a largely overlooked systematic flaw: even when a tool's functionality does not match the user's goal (semantic irrelevance), the model tends to invoke the tool as long as query attributes can be populated into the tool's parameters (structural alignment). (2) Existing benchmarks construct irrelevant scenarios by randomly pairing queries with tools, a process that typically also introduces structural misalignment, thereby confounding evaluation — models may refuse invocation simply because parameters cannot be filled rather than because they genuinely recognize semantic irrelevance.

**Key Challenge**: Do LLMs truly understand that semantic relevance is a necessary condition for tool invocation, or do they rely on structural alignment as a decision-making shortcut?

**Goal**: (1) Identify and formalize structural alignment bias; (2) construct a dataset that decouples the two factors; (3) reveal the underlying mechanism; (4) propose mitigation strategies.

**Key Insight**: Inspired by the polymorphism principle in object-oriented programming — different services can share a unified interface (i.e., structurally aligned yet semantically distinct) — the authors design evaluations that reflect realistic deployment scenarios.

**Core Idea**: Structural alignment bias = the systematic shortcut by which LLMs treat "parameters can be filled" as a proxy for "tool should be invoked." By revealing two competing information flows internally (semantic checking vs. structural matching), the paper proposes path rebalancing to mitigate the bias.

## Method

### Overall Architecture
Problem identification → SABEval dataset construction (decoupling structural alignment from semantic relevance) → behavioral analysis (quantifying bias severity) → contrastive attention attribution (revealing the internal mechanism) → path rebalancing (mitigating the bias).

### Key Designs

1. **SABEval Dataset (Based on the Polymorphism Principle)**:

    - **Function**: Strictly isolates scenarios that are structurally aligned yet semantically irrelevant.
    - **Mechanism**: Three-step construction: (1) Hierarchical tool construction — sibling tools sharing the same parameter interface are derived from tool templates (e.g., "Nintendo game query" and "PlayStation game query" both share `game_title + region` parameters); (2) Query generation for each tool; (3) Sibling pairing — each query is paired with its sibling tool, ensuring structural alignment while maintaining semantic irrelevance. The dataset comprises 101 tool templates, 5 queries per tool, and 10 sibling combinations, yielding 5,050 samples. No valid tool is available in any instance — any invocation constitutes an error.
    - **Design Motivation**: Random pairing in existing datasets introduces structural misalignment simultaneously, making it impossible to distinguish whether a model refuses because of semantic irrelevance or because parameters cannot be filled.

2. **Contrastive Attention Attribution (CAA)**:

    - **Function**: Reveals the internal information flow during tool invocation decisions.
    - **Mechanism**: Attention attribution is traced from the tool invocation token back to input tokens, uncovering two competing pathways: (1) **Semantic checking pathway** — attends to semantic consistency between tool functionality descriptions and query objectives; (2) **Structural matching pathway** — attends to the structural mapping between query attributes and tool parameters. The relative strength of these two pathways determines the final invocation decision.
    - **Design Motivation**: Conventional counterfactual analysis requires strict token-level correspondence, which is impractical when tool descriptions and queries vary in length. CAA circumvents this limitation.

3. **Path Rebalancing Strategy**:

    - **Function**: Mitigates structural alignment bias without impairing normal tool usage capability.
    - **Mechanism**: Based on the two pathways identified by CAA, the strategy amplifies the relative strength of the semantic checking pathway (or suppresses the influence of the structural matching pathway), achieving 80% relative error reduction.
    - **Design Motivation**: Eliminates the need for model retraining by precisely intervening in the identified competing mechanisms.

## Key Experimental Results

### Main Results (5 Tool-Augmented LLMs)

| Model | Random-Pair TIR↓ | SABEval TIR↓ | Δ |
|-------|-----------------|-------------|-----|
| Qwen3-4B | 0.16% | 40.04% | +39.88 |
| Qwen3-8B | 0.04% | 34.26% | +34.22 |
| Qwen3-14B | ~0.1% | ~35% | ~+35 |
| ToolACE-2.5-8B | ~0.1% | ~42% | ~+42 |
| Watt-Tool-8B | ~0.2% | ~45% | ~+45 |

### Structural Alignment Degree Experiment

| Degree of Structural Alignment | Erroneous Invocation Rate |
|-------------------------------|--------------------------|
| No alignment (random pairing) | <0.2% |
| Basic alignment (SABEval D0) | 41.9% |
| Stronger alignment (+4 parameters) | **90.4%** |

### Key Findings
- **Structural alignment bias is severe**: error rates remain below 0.2% under structural misalignment but surge to 41.9% under basic structural alignment and reach 90.4% under stronger alignment.
- **All 5 mainstream tool-augmented LLMs are affected**, indicating a systemic problem.
- **Counterfactual analysis confirms causality**: a strong causal link exists between structural alignment and erroneous invocation.
- **CAA successfully identifies the two competing pathways**: the semantic checking pathway and the structural matching pathway.
- **Path rebalancing achieves 80% relative error reduction** without degrading normal tool usage capability.

## Highlights & Insights
- **The identification and formalization of "structural alignment bias"** is the paper's primary contribution — it exposes a pervasive yet overlooked security risk with direct implications for the deployment of tool-augmented LLMs.
- **The SABEval construction methodology** (grounded in the object-oriented polymorphism principle) is particularly elegant, borrowing a software engineering concept to design realistic evaluation scenarios.
- **The complete chain from behavioral analysis to internal mechanism to mitigation** exemplifies an interpretability-driven paradigm for safety improvement.

## Limitations & Future Work
- SABEval construction relies on GPT-4o to generate additional parameters, which may introduce generation bias.
- The effectiveness of path rebalancing may vary across model architectures.
- Validation is limited to 5 models; behavior of larger-scale models (70B+) remains unknown.
- Multi-tool selection scenarios (where the task involves choosing among multiple tools) are not considered.
- The root cause of the bias may lie in pretraining data, where the vast majority of tool invocation examples are positive cases.

## Related Work & Insights
- **vs. Patil et al. (2025) / existing benchmarks**: Existing evaluations conflate structural alignment with semantic relevance; this paper is the first to decouple the two.
- **vs. tool selection research**: Tool selection focuses on "which tool to invoke," whereas this paper addresses "whether any tool should be invoked."
- **vs. attention attribution methods**: Conventional methods require token-level correspondence between counterfactual pairs; CAA relaxes this requirement.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Full-chain innovation spanning problem identification, formalization, dataset construction, mechanism analysis, and mitigation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five models, causal analysis, alignment degree experiments, and rebalancing validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and rigorous experimental design.
- Value: ⭐⭐⭐⭐⭐ Direct guidance for the safe deployment of tool-augmented LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ACL 2026\] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)
- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](../../AAAI2026/interpretability/hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)

</div>

<!-- RELATED:END -->
