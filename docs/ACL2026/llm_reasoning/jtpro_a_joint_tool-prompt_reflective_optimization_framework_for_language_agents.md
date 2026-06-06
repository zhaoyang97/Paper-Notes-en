---
title: >-
  [Paper Note] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents
description: >-
  [ACL 2026][LLM Reasoning][tool-call optimization] JTPRO proposes a joint optimization framework that requires no model fine-tuning. Through reflection-driven iterative editing…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "tool-call optimization"
  - "prompt optimization"
  - "reflective learning"
  - "large tool libraries"
  - "joint optimization"
date: 2026-05-08
content_hash: be8acf6a9bde08eb
---

# JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents

**Conference**: ACL 2026
**arXiv**: [2604.19821](https://arxiv.org/abs/2604.19821)  
**Code**: None  
**Area**: LLM Reasoning
**Keywords**: tool-call optimization, prompt optimization, reflective learning, large tool libraries, joint optimization

## TL;DR
JTPRO proposes a joint optimization framework that requires no model fine-tuning. Through reflection-driven iterative editing, it simultaneously optimizes global instructions and per-tool schema/parameter descriptions, significantly improving end-to-end success rates for tool selection and slot filling in large-scale tool library settings, achieving 5%–20% OSR gains over baselines such as GEPA.

## Background & Motivation

**State of the Field**: Augmenting LLM agents with external tools has become a mainstream paradigm, yet as the number of tools grows to hundreds or even thousands, the reliability of tool invocation degrades sharply. Existing approaches include fine-tuning, retrieval augmentation, and prompt optimization, but most treat global instructions and tool descriptions independently.

**Limitations of Prior Work**: Two core issues arise as tool libraries scale: (1) generic global prompts fail to disambiguate similar tools, leading to tool mis-selection; (2) imprecise tool schema descriptions cause parameter instantiation errors (slot/value errors). Experiments on ToolACE show that scaling from 300 to 1,000 tools causes a significant drop in tool selection accuracy even for GPT-5-class models.

**Root Cause**: Global instructions and per-tool descriptions are mutually dependent—global strategy relies on inter-tool disambiguation cues, while slot filling relies on global conventions (e.g., date formats, numeric ranges). Optimizing either component in isolation is insufficient.

**Paper Goals**: Design a framework that iteratively optimizes global instructions $P$ and per-tool schemas $\{T_i\}$ without model fine-tuning, maximizing invocation-level correctness (tool + parameters + values) in large-scale tool library settings.

**Starting Point**: The authors observe that the bottleneck in end-to-end success rate often lies in slot filling rather than tool selection, and that many parameter semantics (e.g., date formats, boolean flags) recur across multiple tools and can be globalized to eliminate redundancy and inconsistency.

**Core Idea**: Jointly and iteratively optimize an agent's global instructions and per-tool descriptions via Pareto-based candidate selection, reflection-driven localized edits, and globalization of shared slot semantics.

## Method

### Overall Architecture
JTPRO maintains a candidate context pool $\mathcal{C}$. At each iteration it: (1) selects a candidate via Pareto sampling; (2) performs rollouts on a mini-batch to obtain diagnostic feedback; (3) proposes localized edits to the global instruction $P$ and the relevant tool schemas $\{T_i\}$ based on that feedback; (4) merges the edited version with the current best; (5) globalizes recurring slot semantics; and (6) updates the candidate pool and global optimum after validation.

### Key Designs

1. **Pareto-based Sampling**:

    - Function: Efficiently selects a starting context from the candidate pool.
    - Mechanism: Retains candidates that are optimal on at least one training instance, prunes strictly dominated candidates, and samples with a probability biased toward candidates that win on more instances. This is inspired by GEPA's Pareto selection strategy.
    - Design Motivation: Avoids collapsing to a single local optimum and maintains exploratory diversity.

2. **Reflection-driven Localized Edits**:

    - Function: Diagnoses systematic failures and proposes targeted modifications.
    - Mechanism: For each failed rollout sample, a Diagnose function extracts structured error signals (tool confusion, missing parameters, format/value errors). A reflective module ProposeEdits then generates directed edits to global instructions and relevant tool descriptions. Edits are localized—only the specific global rules and tool/parameter descriptions implicated in failures are modified.
    - Design Motivation: Avoids context bloat from large-scale rewrites by targeting only the components responsible for failures.

3. **GlobalizeSlots**:

    - Function: Reduces redundant schema text across tools.
    - Mechanism: Identifies slot semantics that recur across tools (e.g., date formats, numeric bounds, boolean parameters, sorting conventions), promotes them to named conventions in the global instruction, and replaces per-tool occurrences with short pointer references. For example, fields such as `startDate` and `endDate` uniformly reference a global "DateTime Fields" rule.
    - Design Motivation: In the ETID dataset, a small number of parameter families (e.g., identifiers, datetime fields) recur across up to 77 of 124 tools. Globalization enforces consistent semantics, reduces potential conflicts, and frees schema space for tool-specific disambiguation cues.

### Loss & Training
The optimization objective minimizes invocation-level loss, which comprises tool selection, slot filling, and overall success rate components. The entire process requires no gradient updates and operates purely in text space through iterative optimization.

## Key Experimental Results

### Main Results

| Model | # Tools | Method | TSA (%) | SFA (%) | OSR (%) |
|-------|---------|--------|---------|---------|---------|
| GPT-5 | 500 | Base | 73.02 | 84.79 | 62.73 |
| GPT-5 | 500 | GEPA | 77.17 | 85.75 | 66.12 |
| GPT-5 | 500 | **JTPRO** | **82.28** | **90.00** | **74.38** |
| GPT-5 | 1000 | Base | 67.66 | 87.35 | 62.37 |
| GPT-5 | 1000 | GEPA | 75.13 | 86.40 | 67.77 |
| GPT-5 | 1000 | **JTPRO** | **78.72** | **89.26** | **73.55** |
| o3-mini | 1000 | Base | 58.92 | 85.04 | 51.27 |
| o3-mini | 1000 | **JTPRO** | **71.48** | **87.46** | **64.46** |

### Ablation Study (ETID Dataset)

| Configuration | Description |
|--------------|-------------|
| Optimize global instruction $P$ only | Lacks tool-level disambiguation; lower OSR |
| Optimize tool schemas $T$ only | Lacks global conventions; lower OSR |
| **Joint optimization $P$+$T$ (JTPRO)** | Complementary gains; best OSR |

### Key Findings
- **Joint optimization outperforms individual optimization**: Optimizing only instructions or only tool schemas is consistently inferior to joint optimization, confirming their mutual dependency.
- **Largest gains in the 1,000-tool setting**: The more tools and the greater the ambiguity, the more pronounced JTPRO's advantage (o3-mini achieves +13.2 percentage points OSR at 1,000 tools).
- **SFA is the critical bottleneck for OSR**: On datasets with complex schemas such as ETID, slot filling accuracy contributes far more to end-to-end success than tool selection.
- **Globalization reduces redundancy and improves consistency**: The GlobalizeSlots step not only compresses schema length but also improves slot filling accuracy by enforcing unified semantics.

## Highlights & Insights
- **The motivation for joint optimization is compellingly argued**: Figure 1 clearly shows that SFA drives OSR, and Figure 2 illustrates performance degradation as tool count scales, providing strong justification for joint optimization.
- **The GlobalizeSlots design is highly practical**: In enterprise-scale tool libraries, large numbers of slot semantics recur (dates, IDs, booleans). Promoting them to global rules is a clean and effective engineering technique that transfers to any multi-tool agent system.
- **Tuning-free optimization paradigm**: Operating entirely in text space, the framework is equally applicable to closed-source models, which is valuable for real-world deployment.

## Limitations & Future Work
- Evaluation covers only single-turn tool invocation; multi-turn tool chaining in dialogue is not addressed.
- Reflection-driven optimization relies on high-quality annotated tool-call traces, incurring non-trivial cold-start costs.
- Incremental optimization efficiency under dynamic tool addition/removal is not considered.
- Integration with retrieval-augmented approaches and extension to multi-step planning settings are promising directions for future work.

## Related Work & Insights
- **vs. GEPA**: GEPA also employs Pareto selection and reflective optimization but optimizes only global instructions without modifying tool schemas. JTPRO extends to joint optimization and shows clear advantages in tool-specific disambiguation and parameter constraints.
- **vs. DRAFT**: DRAFT improves per-tool documentation through trial and error but does not optimize a global strategy. JTPRO addresses both levels simultaneously, avoiding global–local inconsistency.
- **vs. MIPRO**: MIPRO optimizes module prompts and demonstrations but is not designed for tool-calling scenarios and lacks slot-level diagnostics and the globalization mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ — The idea of jointly optimizing instructions and tool schemas is valuable, though the overall framework is a natural extension of GEPA.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, multiple models, detailed ablations, and multiple tool-library scales.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, though the paper is lengthy and contains some repetitive content.
- Value: ⭐⭐⭐⭐ — Offers strong practical guidance for large-scale tool-calling scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](../../ICLR2026/llm_reasoning/adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ICLR 2026\] ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization](../../ICLR2026/llm_reasoning/reform_reflective_autoformalization_with_prospective_bounded_sequence_optimizati.md)
- [\[ACL 2026\] Semantic-Aware Logical Reasoning via a Semiotic Framework](semantic-aware_logical_reasoning_via_a_semiotic_framework.md)
- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](../../ICLR2026/llm_reasoning/estimating_the_empowerment_of_language_model_agents.md)
- [\[ACL 2026\] Think Outside the Policy: In-Context Steered Policy Optimization](think_outside_the_policy_in-context_steered_policy_optimization.md)

</div>

<!-- RELATED:END -->
