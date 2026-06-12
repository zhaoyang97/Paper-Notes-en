---
title: >-
  [Paper Note] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations
description: >-
  [ACL 2026][Interpretability][Tool invocation] This paper identifies and formalizes "structural alignment bias" in LLM tool invocation—the tendency for LLMs to call a tool when query attributes can be mapped to tool param…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Tool invocation"
  - "Structural alignment bias"
  - "Irrelevant tool refusal"
  - "Attention attribution"
date: 2026-05-08
content_hash: 3af94349af93cfe1
---

# Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations

**Conference**: ACL 2026  
**arXiv**: [2604.11322](https://arxiv.org/abs/2604.11322)  
**Code**: [GitHub](https://github.com/along-l/irrelevant-tool)  
**Area**: Interpretability  
**Keywords**: Tool invocation, Structural alignment bias, Irrelevant tool refusal, Interpretability, Attention attribution

## TL;DR
This paper identifies and formalizes "structural alignment bias" in LLM tool invocation—the tendency for LLMs to call a tool when query attributes can be mapped to tool parameters, even if the tool's function is unrelated to the user's goal. The authors construct the SABEval dataset to decouple structural alignment from semantic relevance. Using Contrastive Attention Attribution (CAA), they reveal competing internal paths for semantic checking versus structural matching and propose a rebalancing strategy that achieves an 80% relative error reduction.

## Background & Motivation

**Background**: The ability of LLMs to use external tools has become a critical capability. In real-world scenarios, models frequently encounter tools irrelevant to the user query, where the correct behavior is to refuse the invocation.

**Limitations of Prior Work**: (1) Previous research overlooked a systemic flaw: LLMs tend to invoke a tool as long as query attributes can fill its parameters (structural alignment), even when the tool function does not match the user's goal (semantic irrelevance); (2) Existing evaluations construct irrelevant scenarios by randomly pairing queries and tools, which usually introduces structural misalignment. This confuses evaluation results, as models might refuse simply because parameters cannot be filled rather than truly understanding semantic irrelevance.

**Key Challenge**: Do LLMs truly understand that "semantic relevance" is a prerequisite for tool invocation, or do they rely on "structural alignment" as a shortcut for decision-making?

**Goal**: (1) Identify and formalize structural alignment bias; (2) Build a dataset to decouple the two factors; (3) Reveal internal mechanisms; (4) Propose mitigation methods.

**Key Insight**: Borrowing from the polymorphism principle in object-oriented programming—where different services can share a unified interface (structurally aligned but semantically distinct)—real-world evaluation data is constructed.

**Core Idea**: Structural alignment bias is a systemic shortcut where LLMs equate "parameter fillability" with "tool applicability." By uncovering two competing information flows (semantic check vs. structural matching), path rebalancing is proposed to mitigate the bias.

## Method

### Overall Architecture
Problem identification → SABEval dataset construction (decoupling structural alignment and semantic relevance) → Behavioral analysis (quantifying bias severity) → Contrastive Attention Attribution (revealing internal mechanisms) → Path rebalancing (mitigating bias).

### Key Designs

1. **SABEval Dataset (Based on Polymorphism Principle)**:
    - **Function**: Strictly isolates scenarios that are "structurally aligned but semantically irrelevant."
    - **Mechanism**: A three-step construction: (1) Hierarchical tool construction—deriving sibling tools sharing the same parameter interface from tool templates (e.g., "Nintendo Game Query" and "PlayStation Game Query" both share `game_title` + `region` parameters); (2) Query generation for each tool; (3) Sibling pairing—pairing a query with its sibling tool to ensure structural alignment despite semantic irrelevance. It includes 101 tool templates, 5 queries per tool, and 10 sibling combinations, totaling 5,050 samples where no valid tool is available—any invocation is an error.
    - **Design Motivation**: Random pairing in existing datasets causes structural misalignment, failing to distinguish if the model refuses due to "semantic irrelevance" or "unfillable parameters."

2. **Contrastive Attention Attribution (CAA)**:
    - **Function**: Reveals the information flow during tool invocation decision-making.
    - **Mechanism**: Traces attribution from tool invocation tokens back to input tokens, identifying two competing paths: (1) **Semantic check path**—focuses on semantic consistency between tool descriptions and query goals; (2) **Structural matching path**—focuses on the structural mapping between query attributes and tool parameters. The relative strength of these paths determines the final invocation decision.
    - **Design Motivation**: Traditional counterfactual analysis requires strict token-level correspondence, which is difficult in tool invocation due to varying lengths of descriptions and queries. CAA bypasses this limitation.

3. **Path Rebalancing Strategy**:
    - **Function**: Mitigates structural alignment bias without compromising normal tool usage capabilities.
    - **Mechanism**: Based on the two paths identified by CAA, it enhances the relative strength of the semantic check path (or suppresses the structural matching path) to achieve an 80% relative error reduction.
    - **Design Motivation**: Eliminates the need for model retraining by precisely intervening in the discovered competition mechanism.

## Key Experimental Results

### Main Results (5 Tool-Augmented LLMs)

| Model | Random Pairing TIR↓ | SABEval TIR↓ | Δ |
|------|-------------|-------------|-----|
| Qwen3-4B | 0.16% | 40.04% | +39.88 |
| Qwen3-8B | 0.04% | 34.26% | +34.22 |
| Qwen3-14B | ~0.1% | ~35% | ~+35 |
| ToolACE-2.5-8B | ~0.1% | ~42% | ~+42 |
| Watt-Tool-8B | ~0.2% | ~45% | ~+45 |

### Alignment Degree Experiment

| Structural Alignment Degree | Error Invocation Rate |
|------------|---------|
| No Alignment (Random Pairing) | <0.2% |
| Base Alignment (SABEval D0) | 41.9% |
| Stronger Alignment (+4 Params) | **90.4%** |

### Key Findings
- **Structural alignment bias is highly severe**: Error rates are <0.2% without alignment but soar to 41.9% with structural alignment, reaching 90.4% under stronger alignment.
- **All 5 mainstream tool-augmented LLMs are affected**, indicating a systemic issue.
- **Counterfactual analysis confirms causality**: There is a strong causal link between structural alignment and erroneous invocations.
- **CAA successfully identifies competing paths**: Specifically, the semantic check path and the structural matching path.
- **Path rebalancing achieves 80% relative error reduction** without damaging normal tool invocation performance.

## Highlights & Insights
- **Discovery and formalization of "structural alignment bias"** is the primary contribution, revealing a prevalent but overlooked safety risk with direct implications for deploying tool-augmented LLMs.
- **The SABEval methodology** (based on object-oriented polymorphism) is ingenious—borrowing from software engineering to design realistic evaluation scenarios.
- **The complete chain** from behavioral analysis to internal mechanisms and eventual mitigation demonstrates an interpretability-driven paradigm for safety improvement.

## Limitations & Future Work
- SABEval construction relies on GPT-4o for generating additional parameters, which may introduce bias.
- The effectiveness of path rebalancing might vary across model architectures.
- Only verified on 5 models; performance of larger models (70B+) remains unknown.
- Does not consider multi-tool selection scenarios (currently single-tool judgment).
- The root of the bias likely lies in pre-training data, where the vast majority of tool invocation examples are positive instances.

## Related Work & Insights
- **vs. Patil et al. (2025) / Existing Benchmarks**: Existing evaluations confound structural alignment and semantic relevance; this work decouples them for the first time.
- **vs. Tool Selection Research**: Tool selection focuses on "which tool to pick," whereas this work focuses on "whether any tool should be invoked at all."
- **vs. Attention Attribution Methods**: Traditional methods require token-level correspondence for counterfactual pairs; CAA relaxes this constraint.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Problem identification + formalization + dataset + mechanism analysis + mitigation; full-chain innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models + causal analysis + degree experiments + rebalancing verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and rigorous experimental design.
- Value: ⭐⭐⭐⭐⭐ Direct guidance for the secure deployment of tool-augmented LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ACL 2026\] Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives](do_llms_capture_embodied_cognition_and_cultural_variation_cross-linguistic_evide.md)
- [\[ACL 2026\] Dual Alignment Between Language Model Layers and Human Sentence Processing](dual_alignment_between_language_model_layers_and_human_sentence_processing.md)
- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](../../AAAI2026/interpretability/hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)
- [\[NeurIPS 2025\] Distributional Autoencoders Know the Score](../../NeurIPS2025/interpretability/distributional_autoencoders_know_the_score.md)

</div>

<!-- RELATED:END -->
