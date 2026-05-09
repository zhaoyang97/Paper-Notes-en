---
title: >-
  [Paper Note] Bayesian Social Deduction with Graph-Informed Language Models
description: >-
  [ACL 2026][LLM Agent][social reasoning] This paper proposes GRAIL (Graph Reasoning Agent Informed through Language), a hybrid reasoning framework that externalizes probabilistic inference to a factor graph model while delegating language understanding and interaction to an LLM. GRAIL is the first agent to defeat human players in the social deduction game Avalon (67% win rate) while consuming far fewer computational resources than large-scale reasoning models.
tags:
  - ACL 2026
  - LLM Agent
  - social reasoning
  - probabilistic graphical models
  - theory of mind
  - game-playing agents
  - human-AI interaction
date: 2026-05-08
content_hash: 8c05807a448ea989
---

# Bayesian Social Deduction with Graph-Informed Language Models

**Conference**: ACL 2026
**arXiv**: [2506.17788](https://arxiv.org/abs/2506.17788)
**Code**: [Project Page](https://camp-lab-purdue.github.io/bayesian-social-deduction)
**Area**: LLM Agent / Social Reasoning
**Keywords**: social reasoning, probabilistic graphical models, theory of mind, game-playing agents, human-AI interaction

## TL;DR

This paper proposes GRAIL (Graph Reasoning Agent Informed through Language), a hybrid reasoning framework that externalizes probabilistic inference to a factor graph model while delegating language understanding and interaction to an LLM. GRAIL is the first agent to defeat human players in the social deduction game Avalon (67% win rate) while consuming far fewer computational resources than large-scale reasoning models.

## Background & Motivation

**Background**: LLMs excel at general reasoning but remain challenged by social reasoning in multi-agent hidden-information settings—inferring others' beliefs, intentions, and deception. Social deduction games such as Avalon provide a structured environment for evaluating this capability.

**Limitations of Prior Work**: (1) The largest reasoning models (e.g., DeepSeek-R1 671B) can solve simple reasoning tasks but require enormous token budgets and compute; (2) performance degrades sharply when distilled into smaller models; (3) pure LLM approaches struggle with constrained probabilistic reasoning over long time horizons; (4) large models have high latency that precludes real-time interaction with humans.

**Key Challenge**: Social reasoning demands constrained probabilistic inference (e.g., the hard constraint that exactly two players are evil) and long-range belief tracking, yet LLMs reason at the token level and are ill-suited for such structured inference.

**Goal**: Build a social reasoning agent capable of competing against humans in real time, achieving performance on par with or superior to large reasoning models even when instantiated with small models.

**Key Insight**: A hybrid architecture that externalizes belief reasoning to a probabilistic graphical model (factor graph + belief propagation) while the LLM focuses on language understanding and dialogue generation.

**Core Idea**: Decouple structured reasoning from language capability—the factor graph tracks role beliefs in an interpretable and efficient manner, while the LLM supplies language priors and generates conversational utterances.

## Method

### Overall Architecture

GRAIL comprises three components: (1) a **factor graph** for probabilistic reasoning over player roles, using max-product belief propagation for MAP inference; (2) an **LLM** that parses dialogue to extract language priors and generates conversational messages; and (3) a **heuristic action policy** that selects game actions (team proposals, votes) based on current beliefs.

### Key Designs

1. **Factor Graph Role Reasoning**

    - **Function**: Maintain and update role beliefs for each player under hard constraints (exactly two evil players).
    - **Mechanism**: Variable nodes $\mathcal{R} = \{r_1,\dots,r_6\}$ represent player roles (0 = good / 1 = evil); game-state variables $\mathcal{S}$ encode team compositions, votes, and mission outcomes. Factor functions are approximated by neural networks $F = p(r_j|\{p_i,v_i,o_i\})$ trained on 100,000 historical game records.
    - **Design Motivation**: Factor graphs naturally support hard-constraint reasoning and incremental belief updates, offering greater precision and reliability than token-level LLM inference.

2. **LLM Language Prior Integration**

    - **Function**: Incorporate unstructured social signals from dialogue into probabilistic inference.
    - **Mechanism**: The LLM judges whether each player's suspicion level should "increase / decrease / remain unchanged" ($\delta_j^t$), which is converted into a prior $p(r_j^t) = 0.5 \pm \beta^t$, where $\beta^t$ increases over the course of the game (conservative early, confident late).
    - **Design Motivation**: Structured game-state data does not capture dialogue, yet dialogue contains critical social reasoning cues such as contradictions and coalition signals.

3. **Neural Network Approximation of Factor Functions**

    - **Function**: Circumvent the intractability of high-dimensional conditional probability tables.
    - **Mechanism**: A simple feedforward network estimates $p(r_j|\text{game state})$ using ego-centric input transformations to eliminate positional bias and shared network weights to eliminate inter-factor bias. Only 2,500–5,000 game records are required for training.
    - **Design Motivation**: Traditional probability tables are infeasible in high-dimensional settings; neural approximations provide flexibility while remaining data-efficient.

### Loss & Training

Factor function networks are trained with a binary classification loss. No end-to-end reinforcement learning is required; the LLM is used via in-context prompting. GRAIL uses GPT-4.1 as its backbone LLM, though ablations show that Llama-3.1-8B also achieves a 75% win rate.

## Key Experimental Results

### Main Results (Agent vs. Agent)

| Good Agent | Opponent Type | Avg. Win Rate |
|---|---|---|
| Random | Various Evil | 0.00 |
| ReCon (GPT-4.1) | Various Evil | 0.43 |
| GPT-o4-mini Reasoning | Various Evil | 0.40 |
| DeepSeek-R1 (671B) | Various Evil | 0.71 |
| **GRAIL (GPT-4.1)** | **Various Evil** | **0.75** |

### Human Study

| Condition | Win Rate | Contribution Score | Helpfulness Score |
|---|---|---|---|
| GRAIL vs. Humans | **67%** | Above reasoning baseline and some human players | Above reasoning baseline and some human players |
| GPT-o4-mini Reasoning vs. Humans | 27% | Lower | Lower |

### Ablation Study

| Configuration | Finding |
|---|---|
| Graph Only | Robust to model size; 8B model achieves 75% win rate |
| LLM Only | Highly sensitive to model size; 8B performance degrades substantially |
| GRAIL 8B Llama vs. Reasoning 70B DS-R1 | GRAIL 8B achieves a higher win rate |

### Key Findings

- GRAIL generates more than 10× fewer output tokens than reasoning baselines, yielding substantial computational efficiency.
- The factor graph provides a performance floor: even the smallest model maintains a high win rate.
- Language priors accelerate belief convergence—with priors, high confidence is reached by round 3; without them, rounds 4–5 are required.
- A counterintuitive phenomenon emerges among reasoning agents: the 405B Llama underperforms the 70B model, likely due to sycophancy bias.
- GRAIL exhibits lower hallucination rates than reasoning agents across all model sizes.

## Highlights & Insights

- GRAIL is the first language agent to defeat human players in a controlled experiment (67% win rate).
- The hybrid architecture externalizes structured reasoning that LLMs handle poorly, allowing each component to leverage its comparative advantage.
- The combination of factor graphs and belief propagation represents an elegant revival of classical AI methods in the LLM era.
- Human participants, unaware of AI involvement, rated GRAIL higher than some of their human teammates.

## Limitations & Future Work

- Evaluation is limited to the Good faction; deceptive and lying capabilities remain untested.
- Special roles (e.g., Merlin) are excluded, simplifying game complexity.
- Factor function training requires a substantial corpus of historical game data.
- Future work may extend the framework to more complex imperfect-information games.

## Related Work & Insights

- **DeepRole** (Serrino et al., 2019): An Avalon agent trained via self-play without dialogue.
- **ReCon** (Wang et al., 2023): An LLM-based Avalon reasoning agent.
- Probabilistic graphical models for social reasoning (Xu et al., 2024a).
- Hybrid neuro-symbolic reasoning is an important research direction—on structured reasoning tasks, the combination of a specialized model and an LLM outperforms pure large-model approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First hybrid architecture to defeat humans; an elegant integration of classical AI and LLMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers agent-agent evaluation, human studies, model-size ablations, and architectural ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear; human study design is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — A significant advance in LLM social reasoning capability.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[AAAI 2026\] BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](../../AAAI2026/llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)

<!-- RELATED:END -->
