---
title: >-
  [Paper Note] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems
description: >-
  [NeurIPS 2025][Multi-Agent][Theory of Mind] This paper proposes MetaMind — a multi-agent framework inspired by psychological metacognition theory — that significantly enhances the social reasoning capabilities of LLMs th…
tags:
  - "NeurIPS 2025"
  - "Multi-Agent"
  - "Theory of Mind"
  - "Multi-Agent Systems"
  - "Metacognition"
  - "Social Reasoning"
  - "LLM"
date: 2026-05-08
content_hash: f59a96cf8fb95ee4
---

# MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.18943](https://arxiv.org/abs/2505.18943)  
**Code**: [GitHub](https://github.com/XMZhangAI/MetaMind)  
**Area**: Dialogue Systems
**Keywords**: Theory of Mind, Multi-Agent Systems, Metacognition, Social Reasoning, LLM

## TL;DR

This paper proposes MetaMind — a multi-agent framework inspired by psychological metacognition theory — that significantly enhances the social reasoning capabilities of LLMs through three-stage collaboration: a ToM Agent (mental state hypothesis generation), a Moral Agent (social norm-constrained refinement), and a Response Agent (response generation with self-verification). MetaMind achieves state-of-the-art performance on multiple social intelligence benchmarks, approaching human-level performance for the first time.

## Background & Motivation

**Background**: Human everyday conversation is laden with implicit intent — unstated emotions, implied expectations, and veiled suggestions. Humans navigate these latent meanings through Theory of Mind (ToM), reasoning about others' beliefs, desires, emotions, and intentions.

**Limitations of Prior Work**: Although LLMs excel at semantic understanding tasks, they fall significantly short in social reasoning scenarios involving indirect speech, implicit emotions, and culturally sensitive contexts, often defaulting to literal interpretations.

**Limitations of Prior Work**: Existing approaches have attempted to inject social behavior through static role-playing prompting or RLHF fine-tuning; however, these methods optimize for surface-level statistical alignment and treat social reasoning as a single-step prediction problem, failing to capture the multi-stage cognitive processes characteristic of human reasoning.

**Key Challenge**: Human social reasoning is a layered process — interpretation → reflection → adaptation (metacognition) — yet existing systems lack this structured, iterative reasoning capability.

**Goal**: Drawing from psychological metacognition theory, this work decomposes social reasoning into three collaborative stages, endowing LLMs with human-like hierarchical reasoning capabilities.

**Core Idea**: MetaMind is designed as a three-stage multi-agent framework — first reasoning about mental states, then refining hypotheses under social norm constraints, and finally generating and verifying responses.

## Method

### Overall Architecture

MetaMind decomposes social understanding into three collaborative stages, each handled by a specialized agent:

- **Stage 1 — Theory-of-Mind Agent**: Generates multiple candidate hypotheses about the user's mental state.
- **Stage 2 — Moral Agent**: Refines hypotheses using cultural norms and ethical constraints.
- **Stage 3 — Response Agent**: Generates a response based on the optimal hypothesis and verifies quality through a self-reflection mechanism.

### Key Designs

1. **Mental State Hypothesis Generation (Stage 1)**

    - **Function**: Given user input $u_t$, social context $C_t$ (dialogue history), and social memory $M_t$ (user preferences and emotional patterns), generates a set of candidate mental state interpretations $\mathcal{H}_t = \{h_1, h_2, \ldots, h_k\}$.
    - **Mechanism**: Mental-State Reasoning proceeds in four steps — (1) generate commonsense hypotheses from input $(u_t, C_t)$; (2) cross-validate against social memory $M_t$; (3) identify ToM markers within predefined categories $\mathcal{T} = \{\text{Belief}, \text{Desire}, \text{Intention}, \text{Emotion}, \text{Thought}\}$; (4) produce $k$ candidate hypotheses.
    - **Design Motivation**: Prevents LLMs from prematurely committing to a single semantic response; multiple hypotheses ensure multi-perspective coverage of ambiguous intent.

2. **Norm-Aware Hypothesis Refinement (Stage 2)**

    - **Function**: The Moral Agent receives the hypothesis set $\mathcal{H}_t$ and a constraint rule set $\mathcal{D}$ (cultural norms, ethical constraints, role expectations), and produces a refined version $\tilde{h}_i$ for each hypothesis $h_i$.
    - **Mechanism**: The optimal hypothesis is selected via a composite objective:
    $$\tilde{h}^* = \arg\max_i \left[\lambda \cdot P(\tilde{h}_i | u_t, C_t, M_t) + (1-\lambda) \cdot \log \frac{P(\tilde{h}_i | u_t, C_t, M_t)}{P(\tilde{h}_i)}\right]$$
    where the first term measures contextual plausibility and the second measures information gain (the additional information a hypothesis acquires given the context).
    - **Design Motivation**: Simulates the human process of correcting initial judgments using social norms — for example, reinterpreting an inferred romantic intent in a workplace conversation as collegial appreciation.

3. **Response Generation and Self-Verification (Stage 3)**

    - **Function**: The Response Agent generates response $o_t$ conditioned on the optimal hypothesis $\tilde{h}^*$ and social memory $M_t$, and verifies quality through a utility scoring mechanism.
    - **Mechanism**: Response generation maximizes the conditional probability: $o_t = \arg\max \prod_{t=1}^{L} p(y_t | y_{<t}, \tilde{h}^*, M_t, u_t)$. Self-reflection is performed via a utility function:
    $$U(o_t) = \beta \cdot \text{Empathy}(o_t, u_t, M_t) + (1-\beta) \cdot \text{Coherence}(o_t, C_t, \tilde{h}^*)$$
    Regeneration is triggered if the utility score falls below a threshold.
    - **Design Motivation**: Implements a metacognitive loop — not only generating a response but also reflecting on its social and semantic quality to ensure both empathy and coherence.

### Loss & Training

- MetaMind is an **inference-time framework** and does not involve model fine-tuning.
- Key parameters include: number of hypotheses $k$, plausibility–gain tradeoff weight $\lambda$, and empathy–coherence tradeoff weight $\beta$.
- The framework is plug-and-play and can be applied to any LLM backbone (GPT-4, DeepSeek-R1, Qwen, etc.).

## Key Experimental Results

### Main Results

**ToM Reasoning (ToMBench)**:

| Method | Emotion | Desire | Intention | Knowledge | Belief | NL Comm. | AVG |
|--------|---------|--------|-----------|-----------|--------|----------|-----|
| GPT-4 (base) | 75.7 | 69.7 | 84.7 | 52.1 | 82.8 | 84.0 | 74.8 |
| + CoT | 73.2 | 63.3 | 77.9 | 60.4 | 83.6 | 83.0 | 73.6 |
| + SymbolicToM | 75.9 | 70.9 | 79.6 | 58.2 | 84.0 | 83.7 | 75.4 |
| **+ MetaMind** | **78.7** | **76.5** | **84.3** | **68.2** | **88.6** | **88.5** | **81.0** |

**Social Simulation (STSS)**:

| Method | Conversation | Public Activity | Dating | Inviting Peers | Online Activity | Seeking Help | AVG |
|--------|-------------|-----------------|--------|----------------|-----------------|--------------|-----|
| GPT-4 (base) | 48.6 | 59.6 | 1.2 | 2.3 | 63.4 | 61.5 | 39.4 |
| + TDP | 72.3 | 75.9 | 40.0 | 20.0 | 68.6 | 50.0 | 54.4 |
| **+ MetaMind** | **80.8** | **81.9** | **65.0** | **67.1** | **75.1** | **73.0** | **73.9** |

### Ablation Study

**Social Cognition Task Ablation**:

| Variant | UOT | SIT | PST | FBT | AST | HT | SST | FRT | Avg |
|---------|-----|-----|-----|-----|-----|----|----|-----|-----|
| MetaMind (full) | 81.5 | 60.4 | 64.8 | 90.1 | 88.8 | 86.2 | 88.4 | 83.9 | 80.5 |
| w/o Stage 1 | 77.2 | 58.5 | 61.0 | 88.9 | 86.1 | 84.9 | 87.0 | 80.1 | 77.9 |
| w/o Stage 2 | 75.6 | 57.8 | 59.3 | 88.1 | 84.7 | 84.0 | 86.2 | 78.4 | 76.7 |
| w/o Stage 3 | 79.1 | 59.3 | 62.7 | 89.5 | 87.4 | 85.5 | 87.8 | 82.0 | 79.1 |
| w/o SocialMemory | 73.9 | 56.2 | 58.1 | 87.4 | 82.3 | 83.1 | 85.0 | 76.8 | 75.4 |

### Key Findings

- MetaMind improves GPT-4 from 74.8% to 81.0% (**+6.2%**) on ToMBench and achieves an average improvement of **9.0%** on social cognition tasks.
- A **35.7% improvement** is achieved on the STSS social simulation benchmark (39.4% → 73.9%), with particularly large gains on Dating (+63.8%) and Inviting Peers (+64.8%) tasks.
- Ablation results show that Social Memory contributes the most (−5.1% when removed), while the Stage 3 verification mechanism is critical for STSS performance (−16.1% when removed).
- MetaMind transfers effectively to frontier reasoning models: DeepSeek-R1 (86.0→88.6) and OpenAI o3 (90.3→92.2), demonstrating the generality of the framework.
- On key ToM dimensions (Belief, NL Communication), MetaMind enables LLMs to approach human-level performance for the first time.

## Highlights & Insights

- **Psychology-Driven Design**: Rather than ad hoc prompt engineering, the framework systematically maps metacognition theory (planning → monitoring → evaluative reflection) onto a three-stage agent architecture.
- **Social Memory as a Key Innovation**: Dynamic tracking of user preferences and emotional patterns enables the system to adapt to individual users across dialogue turns.
- **Elegant Information Gain Term**: The Moral Agent's scoring formula incorporates $\log \frac{P(\tilde{h}_i|context)}{P(\tilde{h}_i)}$ as an information gain term, preventing selection of overly generic interpretations.
- **Model Agnosticism**: As an inference-time framework requiring no fine-tuning, MetaMind is plug-and-play and effective for both open-source and closed-source models.

## Limitations & Future Work

1. Performance remains dependent on the capabilities of the underlying LLM — absolute performance on smaller models still lags significantly.
2. Validation is limited to text-based scenarios; real-world social interaction involves multimodal cues (tone of voice, facial expressions), group dynamics, and long-term relationship building.
3. Social Memory scalability — adaptation is required when cultural norm coverage is incomplete or evolving.
4. Sequential three-stage reasoning increases inference overhead; future work may explore parallelization or selective activation strategies.

## Related Work & Insights

- **Distinction from SymbolicToM and ToM2C**: These methods focus on diagnostic evaluation or single-step reasoning; MetaMind is the first framework to model ToM as a multi-stage metacognitive process.
- **Relation to Generative Agents**: Generative Agents simulate social behavior through agents but lack explicit modeling of mental states and normative constraints.
- **Implications for Agent Design**: Social agents require not only role-playing but also an explicit three-layer architecture comprising a "mental model," "social norms," and "self-reflection."
- **Research Directions**: Extending MetaMind's metacognitive loop to multimodal social interaction and long-term user relationship modeling.

## Rating

- ⭐⭐⭐⭐ **Novelty**: The systematic mapping of metacognition theory onto a three-stage agent architecture is both novel and theoretically grounded.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Experiments span 16 LLMs, 3 major benchmarks, comprehensive ablations, and human-level comparisons, yielding a thorough evaluation design.
- ⭐⭐⭐⭐ **Value**: An inference-time framework requiring no fine-tuning, model-agnostic, and with low deployment overhead.
- ⭐⭐⭐ **Writing Quality**: Structure is clear, but the notation is dense and the practical implementation details of certain formulas (e.g., the Stage 2 scoring function) lack sufficient transparency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](../../ACL2026/multi_agent/llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ACL 2026\] Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](../../ACL2026/multi_agent/preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)
- [\[ACL 2026\] Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives](../../ACL2026/multi_agent/social_dynamics_as_critical_vulnerabilities_that_undermine_objective_decision-ma.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)

</div>

<!-- RELATED:END -->
