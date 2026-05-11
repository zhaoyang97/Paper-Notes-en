---
title: >-
  [Paper Note] Time, Identity and Consciousness in Language Model Agents
description: >-
  [AAAI 2026 Spring Symposium][LLM Agent][Machine Consciousness] This paper applies the temporal gap concept from Stack Theory to LLM agent evaluation…
tags:
  - "AAAI 2026 Spring Symposium"
  - "LLM Agent"
  - "Machine Consciousness"
  - "Identity Evaluation"
  - "Language Model Agents"
  - "Temporal Consistency"
  - "Stack Theory"
date: 2026-05-08
content_hash: 9c447af6c4510dea
---

# Time, Identity and Consciousness in Language Model Agents

**Conference**: AAAI 2026 Spring Symposium
**arXiv**: [2603.09043](https://arxiv.org/abs/2603.09043)
**Code**: Available
**Area**: LLM Agent / AI Safety
**Keywords**: Machine Consciousness, Identity Evaluation, Language Model Agents, Temporal Consistency, Stack Theory

## TL;DR
This paper applies the temporal gap concept from Stack Theory to LLM agent evaluation, proposing a conservative evaluation toolkit that distinguishes between "talking like a stable self" and "being organized like a stable self." It reveals identity trade-offs across different scaffold structures via persistence scores and an identity morphospace.

## Background & Motivation

**Background**: Machine consciousness evaluation primarily relies on behavioral observation — for language models, this means language use and tool use. Existing evaluation methods allow agents to "say the right things" (e.g., claim self-awareness) even when the underlying constraints are not simultaneously present.

**Limitations of Prior Work**: (1) Behavioral evaluation can be confounded by an agent's linguistic capabilities — models can generate correct statements about themselves without actually possessing the properties in question; (2) ingredient-wise occurrence within an evaluation window and co-instantiation at a single decision step are fundamentally different, yet existing methods do not distinguish between them.

**Key Challenge**: The gap between "sounding like" and "being" — language models can perfectly mimic discourse about identity and consciousness without possessing those properties at the organizational level.

**Goal**: Develop a conservative identity evaluation toolkit capable of distinguishing imitative behavior from organizational-level identity consistency.

**Key Insight**: Leverage the "temporal gap" concept from Stack Theory to scaffold evaluation — distinguishing between components appearing one-by-one within a time window and components co-instantiated at a single decision step.

**Core Idea**: Instantiate the Arpeggio and Chord postulates of Stack Theory to evaluate "grounded identity statements," generating two persistence scores; map common scaffold structures into an identity morphospace.

## Method

### Overall Architecture
Instrument and record behavioral trajectories of LLM agents. Extract identity-relevant state information from scaffold traces. Compute persistence scores using the Arpeggio and Chord postulates respectively. Map multiple scaffold structures into an identity morphospace to reveal design trade-offs along identity dimensions.

### Key Designs

1. **Arpeggio vs. Chord Persistence Scores**:

    - Function: Quantitatively distinguish between "sequential occurrence of components" and "simultaneous co-instantiation of components"
    - Mechanism: The Arpeggio score measures whether identity-relevant components appear sequentially within a time window (weak form — indicating at least that the agent has been exposed to the necessary information); the Chord score measures whether these components simultaneously influence behavior within a single decision step (strong form — indicating that the agent genuinely integrates all relevant factors at the moment of decision)
    - Design Motivation: Occurrence alone does not equal joint participation in decision-making. An agent may process different aspects of identity at separate steps without ever integrating them into a single decision

2. **Five Operational Identity Metrics**:

    - Function: Operationalize abstract identity concepts into computable metrics
    - Mechanism: Five concrete metrics related to identity persistence are defined, including temporal consistency (consistency of responses to the same identity question at different time points) and contextual robustness (stability of identity expression across different conversational contexts). These metrics can be computed directly from instrumented scaffold traces
    - Design Motivation: Philosophical notions of identity must be translated into technically measurable indicators

3. **Identity Morphospace**:

    - Function: Visualize the trade-offs of different scaffold designs along identity dimensions
    - Mechanism: Using Arpeggio/Chord scores and the five identity metrics as coordinate axes, common LLM scaffolds (e.g., ReAct, Plan-then-Execute, Memory-augmented) are mapped as points in the morphospace, making the strengths, weaknesses, and trade-offs of different scaffolds immediately apparent
    - Design Motivation: Provides scaffold designers with an identity-dimension reference when selecting among architectural options

### Loss & Training
This paper presents an evaluation framework rather than a training methodology. All metrics are computed rule-based from instrumented traces.

## Key Experimental Results

### Main Results

| Scaffold Type | Arpeggio Score | Chord Score | Notes |
|---|---|---|---|
| Simple prompt | Low | Low | Virtually no identity structure |
| ReAct | Medium | Low | Components appear but do not co-instantiate |
| Memory-augmented | High | Medium | Memory aids component accumulation |
| Plan-then-Execute | Medium | Medium | Planning provides some integration |

### Ablation Study

| Feature | Effect on Chord | Notes |
|---|---|---|
| Long-term memory | Increases Arpeggio | Helps components persist |
| Reflection mechanism | Increases Chord | Facilitates multi-component integration |
| Fixed system prompt | Increases surface consistency | Does not improve true Chord |

### Key Findings
- Most existing scaffolds achieve acceptable Arpeggio scores but low Chord scores, indicating that identity components appear but are rarely integrated within a single decision step
- Memory-augmented scaffolds show the greatest advantage in identity persistence, yet still fall far short of being "organized like a stable self"
- A simple system prompt of the form "I am XXX" improves surface-level identity consistency but does not improve underlying Chord scores

## Highlights & Insights
- **Philosophy → Engineering Translation**: Operationalizing the philosophical postulates of Stack Theory into computable metrics bridges philosophy and engineering
- **Arpeggio vs. Chord Distinction**: This core distinction is highly insightful — "sequential occurrence" vs. "simultaneous co-instantiation" precisely characterizes the difference between "mimicking" and "possessing"
- **Morphospace as a Visualization Tool**: Introduces a new evaluation dimension for scaffold design, helping practitioners understand the consequences of design choices

## Limitations & Future Work
- Stack Theory is itself a relatively nascent consciousness framework whose philosophical foundations remain contested
- Ground truth for identity evaluation is difficult to establish — what constitutes "genuine" identity persistence?
- Experiments are limited in scale, covering only a small number of scaffolds and models
- Instrumented recording may alter the agent's own behavior

## Related Work & Insights
- **vs. Consciousness Tests (Butlin et al. 2023)**: Traditional consciousness tests focus on behavioral performance; this paper adds a temporal dimension to the analysis
- **vs. Self-Awareness Benchmarks**: Existing self-awareness benchmarks evaluate via QA formats, whereas this paper evaluates at the organizational structure level
- **vs. Embodied Agent Evaluation**: The proposed method is also applicable to identity evaluation in embodied agents

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Operationalizing consciousness theory into LLM evaluation tools is highly original
- Experimental Thoroughness: ⭐⭐⭐ Primarily proof-of-concept; experimental scale is limited
- Writing Quality: ⭐⭐⭐⭐ Concepts are clearly explained, though substantial philosophical terminology is involved
- Value: ⭐⭐⭐⭐ Opens a new direction for identity/consciousness evaluation in AI safety

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AutoTool: Efficient Tool Selection for Large Language Model Agents](autotool_efficient_tool_selection_for_large_language_model_agents.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](../../NeurIPS2025/llm_agent/agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[AAAI 2026\] Real-Time Trust Verification for Safe Agentic Actions Using TrustBench](real-time_trust_verification_for_safe_agentic_actions_using_trustbench.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](../../NeurIPS2025/llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[ICLR 2026\] WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](../../ICLR2026/llm_agent/webarbiter_a_principle-guided_reasoning_process_reward_model_for_web_agents.md)

</div>

<!-- RELATED:END -->
