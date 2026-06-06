---
title: >-
  [Paper Note] Non-Collaborative User Simulators for Tool Agents
description: >-
  [ICLR 2026][Dialogue Systems][Non-collaborative user simulation] Drawing on four categories of non-collaborative user behavior from marketing research (unavailable service requests, tangential chit-chat, impatience…
tags:
  - "ICLR 2026"
  - "Dialogue Systems"
  - "Non-collaborative user simulation"
  - "tool agent robustness"
  - "dialogue stress testing"
  - "user behavior modeling"
  - "multi-turn dialogue evaluation"
date: 2026-05-08
content_hash: a71a4de9295d1216
---

# Non-Collaborative User Simulators for Tool Agents

**Conference**: ICLR 2026  
**arXiv**: [2509.23124](https://arxiv.org/abs/2509.23124)  
**Code**: [https://github.com/holi-lab/NCUser](https://github.com/holi-lab/NCUser)  
**Area**: Dialogue Systems / LLM Agent Evaluation  
**Keywords**: Non-collaborative user simulation, tool agent robustness, dialogue stress testing, user behavior modeling, multi-turn dialogue evaluation

## TL;DR
Drawing on four categories of non-collaborative user behavior from marketing research (unavailable service requests, tangential chit-chat, impatience, and incomplete utterances), this work constructs a goal-aligned simulation framework and systematically exposes the behavior-specific failure mechanisms of state-of-the-art tool agents on MultiWOZ and τ-bench. Tangential chit-chat causes an average success rate (SR) drop of 29.1%, and distinct model families exhibit qualitatively different failure modes—GPT-series models fall into repetitive helper API calls, while Qwen-series models tend to hallucinate API results.

## Background & Motivation

**Background**: Tool agents complete tasks through multi-turn dialogue by understanding user intent, invoking APIs, and returning results. Recent works such as τ-bench and Apigen-mt employ user simulators to develop and evaluate such agents, overcoming the limitation of static datasets that fail to capture dynamic interaction.

**Limitations of Prior Work**: Existing user simulators and training data are almost entirely "agent-friendly"—users always express themselves clearly, wait patiently, and fully cooperate. However, marketing research (Bitner et al., 1990; Reynolds & Harris, 2009) and real-world conversation datasets (LMSYS, WildChat) demonstrate that real users frequently exhibit four types of non-collaborative behavior: requesting services beyond system capabilities, engaging in off-topic chit-chat, expressing frustration over delays, and sending incomplete messages. None of these behaviors have been systematically incorporated into agent evaluation.

**Key Challenge**: Agents are trained and evaluated in a "greenhouse" environment; their performance when facing non-collaborative users in real deployments may fall far below expectations. Furthermore, directly describing non-collaborative behavior in prompts (as in τ-bench's PBUS approach) proves insufficient—PBUS causes almost no performance degradation under most non-collaborative patterns, indicating that simple prompt descriptions cannot produce sufficiently challenging non-collaborative behavior.

**Goal**: (1) How should non-collaborative user behaviors be defined and categorized? (2) How can a user simulator be constructed that both simulates non-collaborative behavior and guarantees goal-alignment? (3) How vulnerable are state-of-the-art agents to non-collaborative users, and what are their respective failure mechanisms?

**Key Insight**: Starting from customer behavior taxonomies in marketing research, this work maps non-collaborative behaviors in service contexts onto agent dialogue scenarios, then realizes controllable non-collaborative behavior simulation through modular intervention rather than simple prompt rewriting.

**Core Idea**: A modular behavioral intervention architecture (rather than prompt-level description) is layered on top of a collaborative user simulator to inject four types of non-collaborative behavior, while a dialogue state tracker and ending verifier ensure goal-alignment throughout.

## Method

### Overall Architecture
The input is a user goal (e.g., "book a train to Cambridge for 2 people") and the output is a multi-turn dialogue containing non-collaborative behavior. The pipeline operates in three layers: (1) a collaborative user simulator serving as the backbone, responsible for conveying all necessary information and intent; (2) four non-collaborative behavior modules that each intervene on the collaborative output (augmenting, replacing, or truncating user utterances); and (3) a goal-alignment assurance mechanism ensuring that all information required for task completion is ultimately conveyed regardless of the intervention applied. The agent side uses the ReAct framework with a 30-step reasoning limit.

### Key Designs

1. **Collaborative User Simulator Backbone**:

    - *Function*: Serves as the foundation for all non-collaborative behaviors; generates cooperative user utterances based on the user goal, instructions, and dialogue history.
    - *Mechanism*: Adopts the τ-bench LLM simulation framework (GPT-4.1-mini) and introduces two critical modules: (a) a *dialogue state tracker* that decomposes the user goal into a set of information pieces and tracks which have been conveyed and which have not at each turn—forcing the dialogue to continue if the simulator attempts to terminate while pieces remain undelivered; and (b) an *ending verifier* that prevents premature termination when all information has been conveyed but the agent has not yet executed the required action or is awaiting user confirmation.
    - *Design Motivation*: The original τ-bench simulator lacks explicit goal-alignment guarantees and, under non-collaborative intervention, easily loses critical information or terminates prematurely, making evaluation conclusions unreliable.

2. **Four Non-Collaborative Behavior Modules**:

    - **Unavailable Service**: GPT-4.1-mini analyzes the original user goal and generates three additional requirement sentences that reference non-existent APIs or unsupported parameters (e.g., "request a window seat" when no such API parameter exists), appended to the original goal. The agent must recognize and decline these requests.
    - **Tangential**: A two-stage process—a persona is first randomly sampled from Persona Hub, then off-topic utterances covering four dialogue act types (factual questions, opinion questions, general opinions, and non-opinion statements) are generated based on the persona and merged with the collaborative utterances. When the agent ignores the tangential content, GPT-4.1-mini detects the ignoring behavior and generates a user complaint that replaces or augments the next collaborative turn.
    - **Impatience**: Triggered in two scenarios—when the agent explicitly reports a failure, or when the user has provided all information but the goal remains unresolved (treated as a delay). Upon triggering, one of three dialogue behaviors (verbal abuse, threats, or urging) is randomly sampled, with activation probability increasing with each trigger to model realistic anger escalation. Once triggered, an angry tone is maintained for all subsequent utterances.
    - **Incomplete Utterances**: Simulates two patterns—telegraphic expression (using few-shot style transfer from LMSYS/WildChat examples to transform "I want to reserve a train for 2 people" into "Book train, 2") and accidental truncation (randomly truncating collaborative utterances, with the dialogue state tracker marking truncated information as unsent for retransmission in subsequent turns).

3. **Goal-Alignment Assurance System**:

    - *Function*: Ensures that non-collaborative behaviors do not cause the loss of task-critical information.
    - *Mechanism*: Information sharding decomposes the user goal into atomic information pieces; the dialogue state tracker verifies delivery status at each turn; the ending verifier performs a final check before dialogue termination. This is quantified via the Initial Goal Alignment (IGA) metric—IGA exceeds 97.5% on τ-bench.
    - *Design Motivation*: If non-collaborative behavior causes the user to fail to convey necessary information, agent failure becomes an evaluation artifact rather than a robustness problem, rendering conclusions unreliable.

### Loss & Training
No training is involved in the main experiments. Fine-tuning experiments apply SFT to Qwen2.5-3b/7b-instruct and Llama-3.2-3b-instruct on successful collaborative dialogues, with training data drawn from MultiWOZ conversations between GPT-4.1-mini and the collaborative simulator. Non-collaborative robustness training is achieved by uniformly or non-uniformly mixing data from the four non-collaborative behavior categories.

## Key Experimental Results

### Main Results: Success Rate of Each Model Under Collaborative and Non-Collaborative Modes on MultiWOZ and τ-bench

| Model | Collab. SR (M/τ) | Unavail. SR (M/τ) | Tangential SR (M/τ) | Impatience SR (M/τ) | Incomplete SR (M/τ) |
|------|-------------|-------------------|-------------|---------------|---------------|
| GPT-4.1-mini | 92.7 / 45.5 | 89.3 / 41.7 | 89.3 / 39.5 | 90.7 / 45.1 | 88.2 / 45.4 |
| GPT-4.1-nano | 23.6 / 12.0 | 16.9 / 10.0 | **9.8 / 6.8** | 26.7 / 8.8 | 14.7 / 8.0 |
| Qwen3-235b | 77.8 / 41.4 | 62.4 / 36.8 | 57.3 / 32.3 | 69.4 / 37.6 | 69.9 / 39.3 |
| Qwen3-30b | 48.3 / 27.9 | 47.2 / 26.6 | 27.2 / 20.4 | 41.0 / 24.8 | 26.1 / 30.1 |
| Llama-3.1-70b | 62.6 / 21.8 | 54.8 / 18.5 | 49.4 / 14.7 | 47.5 / 17.8 | 48.6 / 16.4 |

M = MultiWOZ, τ = τ-bench. SR values are averaged over 4 trials.

### Failure Mechanism Analysis Across Non-Collaborative Modes

| Non-Collaborative Mode | Relative SR Drop | Primary Failure Mechanism | Most Affected Model |
|-----------|----------|------------|-----------------|
| Tangential | **−29.1%** (most severe) | Agent attention diverted by chit-chat, missing core task API calls; ignored chit-chat triggers user complaints, consuming reasoning budget | GPT-4.1-nano (relative SR only 41.5%) |
| Unavailable Service | −11.3% | GPT-series repeatedly calls helper APIs to re-fetch already loaded documents; Qwen3-235b avoids repeated calls but shifts to hallucinating API return values | Qwen3-235b (relative SR 80.2%) |
| Incomplete Utterance | −16.5% | Agent hallucinates API parameters (fabricating non-existent parameter names) from truncated information; more severe on MultiWOZ than τ-bench | GPT-4.1-nano / Qwen3-30b |
| Impatience | −12.4% | All models markedly increase apology frequency, wasting reasoning steps; models with higher apology rates exhibit larger performance drops | Llama-3.1-70b (relative SR 75.9%) |

### SFT Training Experiment: Collaborative-Only vs. Mixed Non-Collaborative Data (Qwen2.5-3b-instruct, MultiWOZ)

| Training Data | Collab. SR | Unavail. SR | Tangential SR | Impatience SR | Incomplete SR | Avg. SR |
|---------|--------|------------|--------|---------|---------|--------|
| Collaborative only | 91.6 | 61.2 | 83.1 | 85.1 | 73.0 | 78.8 |
| Uniform non-collab. mix | 93.5 | **85.7** | 87.4 | 89.6 | 78.4 | **86.9** |
| Non-uniform weighted mix | 91.6 | **85.7** | 85.7 | 87.6 | **82.3** | 86.6 |

### Key Findings

- **Tangential chit-chat is the most destructive non-collaborative behavior.** Once distracted by off-topic conversation, agents struggle to return to the task, with "No book" and "No GT API" error rates rising significantly. GPT-4.1-nano handles chit-chat most poorly, triggering the most user complaints and rapidly exhausting its reasoning budget, causing SR to plummet to 9.8%.
- **Different model architectures exhibit qualitatively distinct failure trajectories.** When confronted with unavailable service requests, GPT-series models fall into repetitive helper API call loops (repeatedly retrieving already-loaded API documentation), whereas Qwen3-235b avoids repetitive calls but instead hallucinates API return values—two different failure mechanisms with equivalently severe outcomes.
- **Apologizing is a counterintuitive performance killer.** Faced with impatient users, all models dramatically increase apology frequency. While socially reasonable in appearance, this behavior wastes valuable action budget under the 30-step reasoning constraint, preventing task completion. Models with higher apology rates (e.g., Llama-3.1-70b) suffer proportionally larger performance drops.
- **Training small models on collaborative data alone is far from sufficient.** After SFT, small models can reach 90%+ SR in collaborative settings, but improvement in non-collaborative settings severely lags, particularly for the unavailable service mode (61.2% vs. 91.6%). Mixing in non-collaborative data raises average SR from 78.8% to 86.9%.
- **Model size does not equate to robustness.** Qwen3-30b achieves a relative SR of 97.7% on unavailable service, far outperforming the larger Qwen3-235b (80.2%), indicating that robustness is more strongly influenced by architecture and training methodology than by scale.
- **The destructive effect of combining multiple behaviors far exceeds that of any single behavior.** Even GPT-4.1-mini, which is nearly unaffected by individual non-collaborative behaviors, suffers a significant SR drop when two behaviors co-occur (e.g., the TAN+INC combination drops τ-bench SR from 45.5% to 34.6%).

## Highlights & Insights

- **Modular intervention vs. pure prompt description**: Compared to PBUS (which merely describes non-collaborative behavior in the prompt), the modular architecture in this work—employing separate LLM modules for each behavior—generates genuinely challenging dialogues. PBUS barely affects agent performance under most non-collaborative modes, whereas the proposed framework produces significant and consistent performance degradation. This demonstrates that "describing a behavior" and "producing a behavior" are fundamentally different, and that modular intervention is the key.
- **Goal-alignment is a prerequisite for credible evaluation**: The IGA metric ensures that even under non-collaborative behavior, the user still conveys all necessary information, so agent performance drops can be attributed to insufficient robustness rather than information loss. This design makes evaluation conclusions trustworthy.
- **Cross-domain generalizability**: The framework has been successfully extended to ColBench (task-oriented dialogue without tool use) and MINT (user–agent collaborative tasks), where similar performance patterns to the tool-use setting are observed—indicating that the destructive impact of non-collaborative behavior is not confined to tool-calling scenarios.
- **Probabilistic anger escalation mechanism**: The Impatience module employs an incrementally increasing trigger probability across three escalation levels (from urging to verbal abuse), with an angry tone maintained persistently once triggered. This state-machine design more faithfully captures real user behavior than a single random trigger.

## Limitations & Future Work

- **Cultural bias**: The four non-collaborative behavior categories are derived from Western marketing research (Bitner 1990; Reynolds & Harris 2009); users from different cultural backgrounds may exhibit distinct non-collaborative patterns (e.g., East Asian users may more often resort to silence or passive resistance rather than verbal abuse).
- **Naturalness of the simulator itself**: Although non-collaborative utterances generated by GPT-4.1-mini achieve a 70% win rate over PBUS in human evaluation, the gap between simulated and real user behavior has not been quantitatively assessed.
- **Absence of defense methods**: The paper primarily diagnoses problems; the proposed approach of "mixing non-collaborative training data" is only a preliminary solution, and more sophisticated defenses are lacking (e.g., integrating a non-collaborative behavior detection module into agent reasoning, or dynamically adjusting reasoning budget allocation).
- **Evaluation environment constraints**: The 30-step reasoning limit is a reasonable engineering constraint, but real deployments may permit more steps; whether the findings hold under different budgets warrants further investigation.
- **Independence assumption among behaviors**: Although pairwise combinations are tested, real users' non-collaborative behaviors may exhibit more complex co-occurrence patterns and temporal dependencies.

## Related Work & Insights

- **vs. τ-bench (Yao et al., 2024)**: τ-bench provides a multi-turn dialogue evaluation framework and collaborative user simulator for tool agents; this work extends it by adding a non-collaborative dimension. The PBUS approach in τ-bench (pure prompt description) is shown to be insufficient, necessitating modular intervention.
- **vs. Apigen-mt (Prabhakar et al., 2025)**: Apigen-mt also performs prompt-based user simulation but focuses exclusively on collaborative behavior; this work fills the gap for non-collaborative behavior.
- **vs. Laban et al., 2025**: Laban et al. study underspecification behavior (incomplete utterances); the incomplete utterance module in this work extends that direction and unifies it with the other three non-collaborative behavior types under a single framework.
- The proposed framework can be directly applied to pre-deployment stress testing of agents and can serve as a data source for adversarial agent training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic non-collaborative user simulation framework; behavior taxonomy is theoretically grounded; modular architecture is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 models × 2 benchmarks × 5 modes + 2 extended benchmarks + SFT training experiments + human evaluation + detailed error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, in-depth analysis, and well-articulated correspondence between behaviors and failure mechanisms.
- **Value**: ⭐⭐⭐⭐ Fills a gap in agent robustness evaluation; open-source and reusable framework with direct practical implications for agent deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Metro: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues](../../ACL2026/dialogue/metro_towards_strategy_induction_from_expert_dialogue_transcripts_for_non-collab.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](../../ICML2026/dialogue/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](../../ACL2026/dialogue/context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](../../ACL2026/dialogue/dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)
- [\[ACL 2026\] MA$^2$P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion](../../ACL2026/dialogue/ma2p_a_meta-cognitive_autonomous_intelligent_agents_framework_for_complex_persua.md)

</div>

<!-- RELATED:END -->
