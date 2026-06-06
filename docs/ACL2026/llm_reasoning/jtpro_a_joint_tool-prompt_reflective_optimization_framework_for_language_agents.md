---
title: >-
  [Paper Note] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents
description: >-
  [ACL 2026][LLM Reasoning][Tool-use Optimization] JTPRO proposes a joint optimization framework without model fine-tuning. By simultaneously optimizing global instructions and per-tool schemas/parameter descriptions throu…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Tool-use Optimization"
  - "Prompt Optimization"
  - "Reflective Learning"
  - "Large-scale Tool Library"
  - "Joint Optimization"
date: 2026-05-08
content_hash: 2a71117e75779d47
---

# JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents

**Conference**: ACL 2026  
**arXiv**: [2604.19821](https://arxiv.org/abs/2604.19821)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Tool-use Optimization, Prompt Optimization, Reflective Learning, Large-scale Tool Library, Joint Optimization

## TL;DR
JTPRO proposes a joint optimization framework without model fine-tuning. By simultaneously optimizing global instructions and per-tool schemas/parameter descriptions through reflection-driven iterative editing, it significantly improves end-to-end success rates in tool selection and parameter filling for large-scale tool libraries, achieving a 5%–20% gain in OSR compared to baselines like GEPA.

## Background & Motivation

**Background**: Extending LLM Agent capabilities via external tools has become a mainstream paradigm. However, when the number of tools grows to hundreds or even thousands, the reliability of tool calling drops sharply. Existing methods include fine-tuning, retrieval augmentation, and prompt optimization, but most treat global instructions and tool descriptions separately.

**Limitations of Prior Work**: Two core problems arise as the tool library grows: (1) generic global prompts fail to distinguish between similar tools, leading to tool mis-selection; (2) tool schema descriptions are insufficiently precise, causing parameter instantiation errors (slot/value errors). The author's experiments on ToolACE show that when tools scale from 300 to 1000, tool selection accuracy drops significantly even for GPT-5 level models.

**Key Challenge**: There is a coupled dependency between global instructions and local tool descriptions—the global strategy relies on discriminative cues between tools, while parameter filling depends on global conventions (e.g., date formats, numeric ranges). Optimizing either side in isolation is insufficient.

**Goal**: Design a framework to iteratively optimize the global instruction $P$ and per-tool schemas $\{T_i\}$ without fine-tuning, maximizing call-level correctness (tool + parameter + value) under large-scale tool libraries.

**Key Insight**: The authors observe that the bottleneck of end-to-end success often lies in slot filling rather than tool selection. Furthermore, many parameter semantics (e.g., date formats, boolean flags) recur across multiple tools and can be globalized to eliminate redundancy and inconsistency.

**Core Idea**: Jointly and iteratively optimize the Agent's global instructions and per-tool descriptions through Pareto candidate selection + reflection-driven local editing + globalization of shared parameter semantics.

## Method

### Overall Architecture
JTPRO maintains a candidate context pool $\mathcal{C}$. In each iteration: (1) a candidate is chosen via Pareto sampling; (2) a rollout on a mini-batch of data provides diagnostic feedback; (3) localized edits for global instruction $P$ and relevant tool schemas $\{T_i\}$ are proposed based on feedback; (4) the edited version is merged with the current best; (5) repeated parameter semantics are globalized; (6) the candidate pool and global optimum are updated after verification.

### Key Designs

1. **Pareto-based Sampling**:
    - **Function**: Efficiently select starting contexts from the candidate pool.
    - **Mechanism**: Retains candidates that achieve the best score on at least one training instance, prunes strictly dominated candidates, and samples with a probability biased towards those that "win more instances." This draws inspiration from GEPA's Pareto selection.
    - **Design Motivation**: To avoid falling into a single local optimum and maintain exploration diversity.

2. **Reflection-driven Localized Edits**:
    - **Function**: Diagnose systematic failures and propose targeted modifications.
    - **Mechanism**: For each rollout failure, a Diagnose function extracts structured error signals (tool confusion, missing parameters, format/value errors). A reflector, ProposeEdits, then generates directional edits for global instructions and related tool descriptions. Edits modify only the involved global rules and tool/parameter descriptions, maintaining locality.
    - **Design Motivation**: To avoid context bloat caused by wide-range rewriting, modifying only the specific parts that caused the failure.

3. **GlobalizeSlots**:
    - **Function**: Reduce redundant schema text across tools.
    - **Mechanism**: Identifies recurring parameter semantics across tools (e.g., date formats, numeric boundaries, boolean parameters, sorting conventions) and promotes them to named rules in global instructions. Individual tools retain only short pointer references. For example, startDate and endDate fields uniformly reference the global "DateTime Fields" rule.
    - **Design Motivation**: In the ETID dataset, a few parameter families (e.g., identifiers, datetime fields) repeat across up to 77/124 tools. Globalization enforces consistent semantics, reduces potential conflicts, and preserves schema space for tool-specific discriminative cues.

### Loss & Training
The optimization objective is to minimize call-level loss, which includes tool selection, slot filling, and overall success rate components. The entire process requires no gradient updates and iterates entirely within the text space.

## Key Experimental Results

### Main Results

| Model | #Tools | Method | TSA(%) | SFA(%) | OSR(%) |
|-------|--------|--------|--------|--------|--------|
| GPT-5 | 500    | Base   | 73.02  | 84.79  | 62.73  |
| GPT-5 | 500    | GEPA   | 77.17  | 85.75  | 66.12  |
| GPT-5 | 500    | **JTPRO** | **82.28** | **90.00** | **74.38** |
| GPT-5 | 1000   | Base   | 67.66  | 87.35  | 62.37  |
| GPT-5 | 1000   | GEPA   | 75.13  | 86.40  | 67.77  |
| GPT-5 | 1000   | **JTPRO** | **78.72** | **89.26** | **73.55** |
| o3-mini| 1000  | Base   | 58.92  | 85.04  | 51.27  |
| o3-mini| 1000  | **JTPRO** | **71.48** | **87.46** | **64.46** |

### Ablation Study (ETID Dataset)

| Configuration | Description |
|---------------|-------------|
| Optimize Global P only | Lacks tool-level discrimination, lower OSR |
| Optimize Tool Schema T only | Lacks global conventions, lower OSR |
| **Joint P+T (JTPRO)** | Complementary gains, optimal OSR |

### Key Findings
- **Joint optimization outperforms separate optimization**: Optimizing only instructions or only schemas is inferior to joint optimization, validating their coupled dependency.
- **Maximum gains in 1000-tool scenarios**: As the tool library grows and confusion increases, JTPRO's advantages become more pronounced (o3-mini OSR gain of +13.2 percentage points at 1000 tools).
- **SFA is the key bottleneck for OSR**: On datasets with complex schemas like ETID, the correctness of parameter filling contributes far more to the end-to-end success rate than tool selection.
- **Globalization reduces redundancy and improves consistency**: The GlobalizeSlots step not only shortens schema length but also improves slot filling accuracy through unified semantics.

## Highlights & Insights
- The argument for the **necessity of joint optimization** is highly persuasive: Figure 1 clearly shows SFA driving OSR, and Figure 2 shows the performance decline as tools scale, providing strong motivation.
- The **design of GlobalizeSlots** is very practical—in enterprise-level tool libraries where many parameter semantics are repeated (dates, IDs, booleans), promoting them to global rules is a concise and effective engineering trick applicable to any multi-tool Agent system.
- **Tuning-free optimization paradigm**: Operating entirely in the text space makes it applicable to closed-source models, which is highly valuable for real-world deployment.

## Limitations & Future Work
- Evaluated only single-turn tool calling; does not involve tool chaining in multi-turn dialogues.
- Reliance on high-quality annotated tool-call traces for reflective optimization leads to high cold-start costs.
- Does not consider incremental optimization efficiency in scenarios where tools are dynamically added or removed.
- Could explore deeper integration with retrieval-augmented methods and extension to multi-step planning scenarios.

## Related Work & Insights
- **vs GEPA**: GEPA also uses Pareto selection and reflective optimization but only optimizes global instructions without touching tool schemas. JTPRO extends this to joint optimization, offering clear advantages in tool-specific discrimination and parameter constraints.
- **vs DRAFT**: DRAFT improves per-tool documentation through trial and error but does not optimize global strategies. JTPRO handles both levels simultaneously, preventing global-local inconsistencies.
- **vs MIPRO**: MIPRO optimizes module prompts and examples but is not designed for tool-calling scenarios and lacks slot-level diagnostics and globalization mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of jointly optimizing instructions and schemas is valuable, though the framework is a natural extension of GEPA.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, multiple models, detailed ablations, and various tool scales.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, though the paper is long with some redundant content.
- Value: ⭐⭐⭐⭐ Significant practical guidance for large-scale tool-calling scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](../../ICLR2026/llm_reasoning/adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ICLR 2026\] ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization](../../ICLR2026/llm_reasoning/reform_reflective_autoformalization_with_prospective_bounded_sequence_optimizati.md)
- [\[ACL 2026\] TInR: Exploring Tool-Internalized Reasoning in Large Language Models](tinr_exploring_tool-internalized_reasoning_in_large_language_models.md)
- [\[ICML 2026\] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling](../../ICML2026/llm_reasoning/uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)
- [\[ACL 2026\] Semantic-Aware Logical Reasoning via a Semiotic Framework](semantic-aware_logical_reasoning_via_a_semiotic_framework.md)

</div>

<!-- RELATED:END -->
