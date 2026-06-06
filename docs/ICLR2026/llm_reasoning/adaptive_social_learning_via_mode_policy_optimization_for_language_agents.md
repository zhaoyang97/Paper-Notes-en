---
title: >-
  [Paper Note] Adaptive Social Learning via Mode Policy Optimization for Language Agents
description: >-
  [ICLR 2026][LLM Reasoning][social intelligence] This paper proposes the Adaptive Social Learning (ASL) framework, which defines four hierarchical reasoning modes (ranging from intuitive response to deep prospective reaso…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "social intelligence"
  - "adaptive reasoning"
  - "mode selection"
  - "reinforcement-learning"
  - "token efficiency"
date: 2026-05-08
content_hash: e5344442c1390406
---

# Adaptive Social Learning via Mode Policy Optimization for Language Agents

**Conference**: ICLR 2026
**arXiv**: [2505.02156](https://arxiv.org/abs/2505.02156)  
**Code**: [https://github.com/MozerWang/AMPO](https://github.com/MozerWang/AMPO)  
**Area**: LLM Reasoning
**Keywords**: social intelligence, adaptive reasoning, mode selection, reinforcement-learning, token efficiency

## TL;DR
This paper proposes the Adaptive Social Learning (ASL) framework, which defines four hierarchical reasoning modes (ranging from intuitive response to deep prospective reasoning) and introduces the AMPO algorithm (combining mode-level and sample-level advantage estimation) to enable LLM agents to adaptively switch reasoning depth according to social scenario complexity. ASL outperforms GPT-4o by 15.6% on social intelligence tasks, surpasses GRPO by 7.0%, and reduces token consumption by 32.8%.

## Background & Motivation
**Background**: LLM agents engaged in social interactions (negotiation, cooperation, etc.) must dynamically adjust reasoning depth; however, existing methods either perform no explicit reasoning (direct reply) or uniformly apply long chain-of-thought (CoT), resulting in overthinking or underthinking.

**Limitations of Prior Work**: Large reasoning models (o1, R1, QwQ, etc.) underperform GPT-4o on social tasks—they apply exhaustive reasoning indiscriminately, leading to overthinking, verbose reasoning chains, and weak goal awareness. Models trained with GRPO also tend to converge to a single reasoning mode (always using the deepest Mode 4).

**Key Challenge**: Social interaction is dynamic; different turns and scenarios require different reasoning depths. Simple scenarios (where both parties' goals are already met) require only intuitive responses, while complex scenarios (where conflicts remain unresolved) demand deep strategic deliberation. Yet existing RL methods (e.g., GRPO) estimate advantages in a mode-agnostic manner, making it impossible to learn such adaptive behavior.

**Goal**: How can an LLM agent dynamically select appropriate reasoning depth based on context in social interactions, while maintaining both efficiency and effectiveness?

**Key Insight**: Drawing on the Hierarchical Cognitive Control Theory (HCCT) from cognitive science, the paper designs four levels of reasoning modes and augments GRPO with mode-level advantage estimation to guide mode selection.

**Core Idea**: By combining hierarchical reasoning modes with mode-aware RL optimization (AMPO), the social agent learns to reason fast when appropriate and slow when necessary—an adaptive reasoning strategy.

## Method

### Overall Architecture
ASL consists of three stages: (1) design of four reasoning modes (M1–M4, ordered from simple to complex); (2) behavioral cloning (BC) to teach the model to follow the format of each mode; and (3) AMPO reinforcement learning to enable the model to adaptively select modes and optimize reasoning quality based on context. The input is a social dialogue context; the output is a reasoning-plus-response sequence prefixed by a mode control token.

### Key Designs

1. **Four Hierarchical Reasoning Modes (based on HCCT)**:

    - **Function**: Define reasoning structures of increasing depth for social scenarios of varying complexity.
    - **Mechanism**: M1 (Intuitive Response) outputs only the answer with no reasoning; M2 (Intent Analysis) analyzes the interlocutor's intent, speaking style, and formulates a response; M3 (Strategic Adaptation) additionally incorporates historical analysis, goal clarification, situational assessment, and strategy formulation; M4 (Prospective Deliberation) extends M3 by generating multiple candidate strategies and selecting the best through simulated deliberation. Each mode is identified by a special control token `<MODE_k>`.
    - **Design Motivation**: The four modes correspond to the four levels of cognitive control in cognitive science—from sensorimotor to extended episodic control—equipping the model with a full reasoning spectrum from System 1 to System 2.

2. **Adaptive Mode Policy Optimization (AMPO)**:

    - **Function**: Augments GRPO with a two-level advantage estimation scheme comprising **mode-level advantage $A^{\mathcal{M}}$** and **sample-level advantage $A^{\mathcal{S}}$**.
    - **Mechanism**: Mode-level advantage is computed by comparing the mean reward across modes to guide mode selection. When rewards across modes are similar, token length serves as a secondary signal to favor more concise modes (normalized via tanh). Sample-level advantage compares sample quality within the selected mode. The final advantage is $A^{\mathcal{M}} + A^{\mathcal{S}}$, embedded in the PPO-clip objective.
    - **Design Motivation**: Addresses the mode-blindness of GRPO—which ranks samples solely by reward without accounting for mode differences—causing convergence to the high-reward but inefficient M4 mode. AMPO encourages the model to prefer simpler modes when rewards are comparable.

3. **Reward Design (Three-Dimensional Reward)**:

    - **Function**: Provides an answer reward (evaluating goal completion) + a format reward (enforcing mode format constraints, with a penalty of −2 for violations) + an answer length reward (smooth decay to the [0, 1] interval when the answer exceeds the target length).
    - **Design Motivation**: Relying solely on the answer reward causes the model to generate verbose responses that offer no genuine strategic improvement. The length reward encourages conciseness and, in conjunction with mode-level advantage, enables depth-adaptive reasoning.

### Loss & Training
Two-stage training: (1) BC warm-up, where training data for each mode is generated by an expert LLM and used for supervised fine-tuning (SFT); (2) online policy optimization with AMPO, where $G$ rollouts (covering different modes) are sampled per prompt and the policy is updated using two-level advantage estimation, PPO-clip, and KL regularization. A single-turn training paradigm is adopted to improve efficiency.

## Key Experimental Results

### Main Results

| Method | SOTOPIA Goal↑ | Hard Goal↑ | Hard Overall↑ | Avg Tokens↓ |
|------|-------------|-----------|--------------|-------------|
| GPT-4o | 8.19 | 6.97 | 3.46 | - |
| DeepSeek-R1 | 7.97 | 5.86 | 2.73 | 711 |
| QwQ-32B | 7.70 | 5.35 | 2.41 | 973 |
| Qwen-7B + GRPO | 8.87 | 7.44 | 3.41 | 905 |
| **Qwen-7B + AMPO** | **8.95** | **7.85** | **3.54** | **647** |
| Llama-8B + GRPO | 8.86 | 7.59 | 3.44 | 865 |
| **Llama-8B + AMPO** | **9.08** | **8.06** | **3.68** | **581** |

### Ablation Study

| Configuration | Hard Goal | Hard Overall | Avg Tokens |
|------|-----------|-------------|------------|
| AMPO + 4 Modes (full) | 7.85 | 3.54 | 647 |
| AMPO w/o length reward | 7.56 | 3.56 | 1617 |
| M1 only | 7.08 | 3.40 | 101 |
| M4 only | 7.62 | 3.31 | 972 |
| GRPO + no modes | 7.32 | 3.16 | 866 |
| GRPO + 4 modes | 7.44 | 3.41 | 905 |

### Key Findings
- **Large reasoning models underperform comprehensively on social tasks**: o1, R1, and QwQ all fall significantly below GPT-4o on SOTOPIA-Hard, indicating that exhaustive reasoning is detrimental to social intelligence.
- Mode distribution adapts across interaction turns: M4 is concentrated in the first 4 turns (53%), while M1 surges in later turns (50% in turns 14–20), consistent with the cognitive intuition of "reason deep first, then act fast."
- Removing the length reward causes a 2.5× increase in tokens (647→1617) while Goal scores actually decrease (7.85→7.56), confirming that verbose reasoning does not equate to better reasoning.
- Mixed-mode consistently outperforms single-mode: AMPO+4 modes achieves 3% higher Goal and 33% fewer tokens than the best single-mode configuration (M4).

## Highlights & Insights
- **Adaptive reasoning depth is the key insight**: Not all scenarios require Long-CoT; adaptive reasoning depth in social interaction is more effective than uniform deep reasoning, a finding generalizable to many tasks without deterministic answers.
- **The mode-level advantage estimation design is elegant**: When rewards are sufficiently discriminative, the higher-reward mode is favored; when rewards are similar, the more efficient mode is preferred—a natural and smooth switching mechanism between the two branches.
- **Cognitive science informing AI design**: The mapping from HCCT's four cognitive levels to four reasoning modes is conceptually clear and empirically validated, demonstrating the guiding value of cognitive science theory for AI agent design.

## Limitations & Future Work
- The single-turn training paradigm may limit long-horizon strategic consistency; this remains a potential weakness despite the authors' analysis.
- The four modes are manually designed; their number and structure may not be optimal, and automatic discovery of reasoning modes could be more effective.
- Evaluation relies on GPT-4o scoring (with human validation), which may introduce evaluator bias.
- Validation is currently limited to social interaction tasks; generalization to other scenarios requiring adaptive reasoning (e.g., open-domain QA, creative writing) remains unexplored.

## Related Work & Insights
- **vs. GRPO**: AMPO augments GRPO with mode-level advantage estimation, addressing GRPO's mode-blindness and achieving a better performance–efficiency trade-off.
- **vs. Large Reasoning Models (o1/R1)**: The failure of these models on social tasks demonstrates that indiscriminate Long-CoT is ill-suited for open-ended interactions requiring social intelligence, and that more structured reasoning is necessary.
- **vs. EPO/DSI**: External strategy modules or strategy injection yield limited gains; end-to-end adaptive reasoning learning (ASL) proves more effective.

## Rating
- Novelty: ⭐⭐⭐⭐ — The hierarchical reasoning mode design is novel, and the two-level advantage estimation in AMPO is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multiple models, multiple benchmarks, thorough ablations, human evaluation, and OOD validation.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, though the dense use of mathematical notation can be demanding.
- Value: ⭐⭐⭐⭐ — The concept of adaptive reasoning depth has broad applicability; an important contribution to the social intelligence direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)
- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](../../ACL2026/llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)

</div>

<!-- RELATED:END -->
