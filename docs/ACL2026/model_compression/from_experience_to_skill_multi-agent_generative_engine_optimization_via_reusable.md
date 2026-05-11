---
title: >-
  [Paper Note] MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning
description: >-
  [ACL 2026][Model Compression][Generative Engine Optimization] This paper reframes Generative Engine Optimization (GEO) from per-instance heuristic optimization to a strategy learning problem…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Generative Engine Optimization"
  - "Multi-Agent Framework"
  - "Strategy Reuse"
  - "Citation Faithfulness"
  - "Visibility Optimization"
date: 2026-05-08
content_hash: 2cda52cabbda4d45
---

# MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning

**Conference**: ACL 2026
**arXiv**: [2604.19516](https://arxiv.org/abs/2604.19516)
**Code**: [https://github.com/Wu-beining/MAGEO](https://github.com/Wu-beining/MAGEO)
**Area**: Model Compression
**Keywords**: Generative Engine Optimization, Multi-Agent Framework, Strategy Reuse, Citation Faithfulness, Visibility Optimization

## TL;DR

This paper reframes Generative Engine Optimization (GEO) from per-instance heuristic optimization to a strategy learning problem, proposing the MAGEO multi-agent framework. The execution layer consists of four collaborating agents — preference, planning, editing, and evaluation — operating in an iterative Generate-Evaluate-Select loop, while the learning layer distills validated edit patterns into reusable engine-specific strategy skills. A Twin Branch causal evaluation protocol and the DSV-CF dual-axis metric are introduced, achieving substantial improvements over heuristic baselines across three mainstream generative engines.

## Background & Motivation

**Background**: Generative engines (e.g., ChatGPT, Gemini) are reshaping information access by replacing search result lists with citation-anchored answers. Content creators must optimize their pages to be cited in generated responses — a practice known as Generative Engine Optimization (GEO).

**Limitations of Prior Work**: (1) Existing GEO methods optimize each instance independently, precluding the accumulation or transfer of effective strategies; (2) Evaluation conflates surface-level visibility with semantic impact, permitting exposure gains accompanied by erroneous citations; (3) Engine preference modeling is coarse-grained, lacking engine-specific strategy learning.

**Key Challenge**: Current GEO is trapped in per-instance trial-and-error rather than evolving into a cumulative, skill-building process. Each optimization starts from scratch, failing to leverage prior successes.

**Goal**: (1) Reframe GEO as a strategy learning problem; (2) Build a multi-agent framework capable of accumulating and reusing strategies; (3) Design causally attributable evaluation methods.

**Key Insight**: A two-tier architecture — an execution layer for collaborative optimization and a learning layer for distilling reusable strategy skills from successful experiences.

**Core Idea**: Abstracting validated edit patterns into structured strategy skills (comprising applicability conditions, edit operations, and effect evaluations), storing them in a skill bank, and retrieving them for reuse on new tasks.

## Method

### Overall Architecture

MAGEO adopts a two-tier architecture. The execution layer consists of an iterative Generate-Evaluate-Select loop involving four agents: a preference agent (analyzing engine preferences), a planning agent (formulating revision strategies), an editing agent (executing concrete modifications), and an evaluation agent (quality checking with faithfulness gating). The learning layer comprises step-level memory (within a single session) and creator-level memory (across sessions), collectively forming the strategy skill bank. A Twin Branch evaluation protocol is employed for causal attribution.

### Key Designs

1. **Twin Branch Evaluation Protocol**:

    - **Function**: Causally attributes the effect of content edits by eliminating confounding from retrieval ranking fluctuations.
    - **Mechanism**: The retrieval list is frozen, and two branches are created — the baseline branch retains the original document, while the optimization branch replaces the target document with its optimized version. The engine responses from both branches under the same retrieval list are compared, isolating the effect of the edit itself.
    - **Design Motivation**: In black-box engines, retrieval and generation are intertwined; without controlling the retrieval list, it is impossible to distinguish "the document improved" from "the retrieval ranking changed."

2. **Strategy Skill Bank**:

    - **Function**: Distills optimization experience into reusable strategy skills.
    - **Mechanism**: A three-phase lifecycle — discovery (step-level memory records positive/negative effects of each edit), consolidation (cross-session extraction of repeatedly effective patterns into structured skills, including engine type, scenario, edit operations, and effect metrics), and retrieval (matching skills by engine and scenario when a new task arrives). Capacity limits and eviction policies (based on usage frequency and recency) maintain scalability.
    - **Design Motivation**: Per-instance optimization is wasteful — successful patterns on the same engine are often transferable. The skill bank enables the transition "from experience to skill."

3. **DSV-CF Dual-Axis Evaluation Metric**:

    - **Function**: Jointly evaluates semantic visibility and citation faithfulness.
    - **Mechanism**: $S_{DSV-CF} = \lambda \cdot \bar{S}_{SSV} + (1-\lambda) \cdot \bar{S}_{ISI} - \gamma(1-AA)$. SSV (Surface Semantic Visibility) aggregates word-level visibility, positional authority, citation prominence, and subjective impression. ISI (Intrinsic Semantic Impact) assesses attribution accuracy, response faithfulness, key-point coverage, and answer dominance. $\gamma$ controls the penalty for erroneous citations.
    - **Design Motivation**: Existing metrics evaluate either exposure or quality in isolation, without penalizing misattribution. DSV-CF ensures that visibility gains must be accompanied by accurate attribution.

### Loss & Training

MAGEO is an LLM-based multi-agent reasoning framework and does not involve neural network training. GPT-5.2 and Gemini-3 Pro serve as the base and evaluation engines. The MSME-GEO-Bench benchmark covers real-world queries across 5 major domains and 15 subcategories.

## Key Experimental Results

### Main Results

**DSV-CF Performance Across Three Mainstream Engines**

| Method | GPT 5.2 SSV | GPT 5.2 ISI | Gemini-3 SSV | Gemini-3 ISI |
|--------|-------------|-------------|--------------|--------------|
| No Optimization | Baseline | Baseline | Baseline | Baseline |
| GEO (Heuristic) | Moderate gain | Mixed | Moderate gain | Mixed |
| RAID | Improved | Improved | Improved | Improved |
| **MAGEO** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| Full MAGEO | Best | Complete framework |
| w/o Skill Bank | Degraded | Strategy reuse contributes significantly |
| w/o Preference Agent | Degraded | Engine-specific modeling is important |
| w/o Evaluation Agent | Degraded + faithfulness collapse | Faithfulness gating is indispensable |
| w/o Twin Branch | No causal attribution | Evaluation reliability deteriorates |

### Key Findings

- Engine-specific preference modeling and strategy reuse are the two most critical contributing components.
- The faithfulness gating of the evaluation agent is crucial — without it, optimization may boost surface exposure through misattribution.
- Strategy skills transfer well across scenarios within the same engine, but cross-engine transfer is limited.
- Traditional SEO strategies (keyword stuffing) are ineffective or even harmful on generative engines.

## Highlights & Insights

- The paradigm shift from "per-instance trial-and-error" to "strategy learning" represents a significant theoretical contribution to the GEO field.
- The Twin Branch causal evaluation protocol addresses a fundamental challenge in evaluating black-box generative engines.
- The three-phase lifecycle of the skill bank (discovery → consolidation → retrieval) is transferable to other agent systems requiring experience accumulation.

## Limitations & Future Work

- The effectiveness of strategy skills may degrade as engines are updated.
- Evaluation relies primarily on LLM-as-Judge, which may introduce systematic bias.
- The query diversity of MSME-GEO-Bench is limited.
- Future work may explore automatic skill updating and cross-engine transfer learning.

## Related Work & Insights

- **vs. GEO/GEO-Bench**: Quantifies exposure but optimizes per instance without strategy accumulation; MAGEO adds a learning layer.
- **vs. RAID**: Intent-aware but lacks strategy reuse; MAGEO achieves experience transfer via the skill bank.
- **vs. AutoGEO**: Learns preference rules but does not accumulate cross-instance strategies; MAGEO's skill bank evolves continuously.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Reframes GEO as a strategy learning problem; both the skill bank and Twin Branch evaluation are novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-engine evaluation, though real-world scenario validation remains limited.
- **Writing Quality**: ⭐⭐⭐⭐ Framework design is clear and metric definitions are complete.
- **Value**: ⭐⭐⭐⭐ Provides a scalable, learning-driven paradigm for the GEO field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](../../AAAI2026/model_compression/safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)
- [\[AAAI 2026\] Parametric Pareto Set Learning for Expensive Multi-Objective Optimization](../../AAAI2026/model_compression/parametric_pareto_set_learning_for_expensive_multi-objective_optimization.md)
- [\[ACL 2026\] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)
- [\[ACL 2026\] Mem^p: Exploring Agent Procedural Memory](memp_exploring_agent_procedural_memory.md)
- [\[ICLR 2026\] IDER: IDempotent Experience Replay for Reliable Continual Learning](../../ICLR2026/model_compression/ider_idempotent_experience_replay_for_reliable_continual_learning.md)

</div>

<!-- RELATED:END -->
