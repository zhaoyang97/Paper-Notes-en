---
title: >-
  [Paper Note] DiscoverLLM: From Executing Intents to Discovering Them
description: >-
  [ICML 2026][Dialogue Systems][Intent Discovery] DiscoverLLM formalizes the scenario where "users do not know exactly what they want" as a progressive discovery process within a hierarchical intent tree. By employing a re…
tags:
  - "ICML 2026"
  - "Dialogue Systems"
  - "Intent Discovery"
  - "Multi-turn Dialogue"
  - "User Simulator"
  - "RLHF"
  - "Collaborative Creation"
date: 2026-05-08
content_hash: 0a0857f57e8ce9d3
---

# DiscoverLLM: From Executing Intents to Discovering Them

**Conference**: ICML 2026  
**arXiv**: [2602.03429](https://arxiv.org/abs/2602.03429)  
**Code**: https://taesookim.com/discoverllm  
**Area**: Human-Computer Interaction / LLM Post-training / User Simulator  
**Keywords**: Intent Discovery, Multi-turn Dialogue, User Simulator, RLHF, Collaborative Creation

## TL;DR
DiscoverLLM formalizes the scenario where "users do not know exactly what they want" as a progressive discovery process within a hierarchical intent tree. By employing a rewardable hierarchical user simulator to train the model, it encourages active divergent exploration when intents are unclear and convergent execution when they are clear. On three tasks—Creative Writing, Technical Writing, and SVG—it improves satisfaction by +10% and reduces dialogue length by -40% compared to baselines like CollabLLM.

## Background & Motivation
**Background**: Current LLM assistants assume users already know their intent upon arrival, and RLHF directly rewards "single-turn quality" based on individual outputs. Recent multi-turn works (CollabLLM, Shani, etc.) enable models to ask clarification questions but still assume that the intent is "already formed, just unexpressed."

**Limitations of Prior Work**: In open-ended creative scenarios (writing, design), users often "don't know what they want until they see a few drafts." In such cases, clarification questions like "What tone do you want?" are ineffective because the user cannot answer them. The paper illustrates this with an example: when asked to write a personal essay, a user might feel the result is "off" but cannot specify why until seeing two counter-examples—one "overly intimate" and one "too detached"—at which point they realize they want a "restrained but honest" tone.

**Key Challenge**: The assumption in clarification-based multi-turn training is that intent is formed but unexpressed; the reality is that intent itself must be discovered through outcomes during interaction. The former corresponds to "asking," while the latter requires "exploration/trial performance."

**Goal**: (1) Formally define the difference between intent discovery and intent elicitation; (2) construct a user simulator capable of providing optimizable reward signals; (3) train an assistant that can adaptively switch between divergence and convergence.

**Key Insight**: Drawing on cognitive science theories (Schön 1983, Flower & Hayes 1981) regarding the "co-evolution of problem and solution," humans discover their preferences by creating and examining outcomes, and these preferences can be organized into hierarchies (abstract → concrete). This provides a computable structure for constructing a ground-truth simulator.

**Core Idea**: Intent is modeled as a hierarchical tree $\mathcal{H}=(V,E)$, where node states transition among undiscovered, emerging, and discovered. The model response is rewarded based on how many nodes it helps discover—a signal directly fed into SFT, DPO, or GRPO.

## Method

### Overall Architecture
DiscoverLLM employs a dual-loop design comprising a "user simulator + post-training":

1. **Intent Tree Construction** (Offline): Automatically extracts a multi-level intent tree from existing artifacts (stories, articles, SVGs).
2. **Simulated Interaction** (Online Training Phase): The simulator initially only "discovers" abstract nodes at the top of the tree. The assistant responds → the simulator evaluates whether it "touches" undiscovered nodes → node states are updated → the simulator generates the next utterance (more specific or more vague) accordingly.
3. **Reward Calculation**: $R(r_t) = R_d(r_t) + R_e(r_t)$, where $R_d = |I_{t+1}| - |I_t|$ is the number of newly discovered nodes, and $R_e$ is a length penalty.
4. **Post-training**: Simulated dialogues are used as SFT data, pair-wise comparisons for DPO, or used directly for online GRPO.

### Key Designs

1. **Hierarchical Intent Tree + Three-state Nodes**:
    - **Function**: Serves as the ground truth, allowing the simulator to "act as if it's unclear" while providing differentiable reward signals.
    - **Mechanism**: The root represents highly abstract intent (e.g., "contains an animal"), branching down to specific details ("pet" → "cat" → "Siamese" / "short-haired"). The current state $I_t \subseteq V$ consists of discovered nodes; the refinement space $\mathcal{R}(I_t) = \{v : \text{parent}(v) \in I_t, v \notin I_t\}$ strictly requires a parent to be discovered before its children. Each node is in one of three states: undiscovered, emerging, or discovered. Emerging nodes can only be vaguely mentioned by the user ("maybe change the animal?") rather than explicitly stated.
    - **Design Motivation**: The hierarchical structure aligns with the "hierarchical network of goals" in cognitive science. The three-state mechanism allows the simulator to naturally produce realistic "I don't know" patterns rather than binary jumps between knowing and not knowing.

2. **Response-Driven State Transitions**:
    - **Function**: Maps different assistant responses to specific state transitions, automatically generating rewards.
    - **Mechanism**: In each turn, the simulator evaluates the relationship between the assistant's response $r_t$ and the nodes in $\mathcal{R}(I_t)$.
        - Direct engagement: If $r_t$ explicitly asks about or satisfies a node, the node becomes discovered (e.g., asking "What kind of pet?" or generating a cat).
        - Tangential engagement: If $r_t$ provides relevant but non-matching options, it accumulates a score. Once it passes a random threshold, the state advances (undiscovered → emerging → discovered). This simulates the cognitive process of "excluding possibilities after seeing counter-examples."
        - Expression constraints: The user can only speak using the language of discovered nodes; emerging nodes can only be vaguely referenced. This ensures that an assistant's question like "What color do you want?" is ineffective if "color" has not been discovered, forcing the model to switch to a divergent strategy like "providing two contrasting drafts."
    - **Design Motivation**: By operationalizing abstract "discovery" into a computable state machine, the reward $R_d = |I_{t+1}| - |I_t|$ encourages both "asking the right questions" and "showing contrasting solutions," allowing the model to self-learn when to diverge and when to converge.

3. **Automatic Intent Tree Construction + Asymmetric Reward**:
    - **Function**: Automatically converts any existing artifact (story, article, SVG) into a trainable intent tree, avoiding human annotation bottlenecks.
    - **Mechanism**: Three steps: (a) An LLM lists all concrete requirements satisfied by an artifact as leaves; (b) an LLM iteratively abstracts multiple levels; (c) an LLM organizes the multi-level intents into a tree. For rewards, $R_d$ only counts positive gains (discovery), and $R_e = -\min(\lambda \cdot \max(0, \text{tokens}(r_t) - \tau), 1)$ limits long outputs but is capped at 1 to ensure it doesn't mask discovery rewards.
    - **Design Motivation**: Ensures the framework is independent of human annotation and low-cost to scale. The asymmetric reward prevents length penalties from eating up discovery rewards, increasing engineering stability.

### Loss & Training
Base models include Llama-3.1-8B-Instruct and Qwen3-8B. LoRA fine-tuning is conducted across four tiers: (1) SFT on synthetic dialogues; (2) DPO on pair-wise comparisons (starting from base); (3) SFT+DPO (starting from SFT); (4) GRPO on Qwen3. Intent trees are constructed using Claude Sonnet 4.5, and the user simulator uses Gemini 1.5 Flash. Dialogues are 5 turns long, and evaluations are averaged over 3 runs.

## Key Experimental Results

### Main Results
On Creative Writing, Technical Writing, and SVG tasks, four core metrics are tracked: Discovery, Satisfaction, Interactivity (ITR), and average token count. Main results for Llama-3.1-8B:

| Task | Configuration | Discover↑ | Satisfy↑ | ITR↑ | #Tok↓ |
|------|------|-----------|----------|------|-------|
| Creative Writing | Base | 38.2 | 30.0 | 20.1 | 3.09k |
| Creative Writing | CollabLLM | 37.3 | 28.0 | 32.6 | 2.93k |
| Creative Writing | **SFT+DPO** | **42.4** | 28.4 | 32.9 | **2.77k** |
| SVG | Base | 45.6 | 32.5 | 21.6 | 3.59k |
| SVG | **SFT+DPO** | **51.6** | **37.0** | **44.6** | **2.61k** |
| Technical Writing | SFT | 47.1 | 35.2 | **81.6** | 2.09k |

SFT alone causes ITR to surge to 80+ (the model becomes very proactive), but Discovery gains are modest. DPO is necessary to push Discovery to the highest levels. GRPO on Qwen3 yield further improvements.

### Ablation Study

| Configuration | Discover | Satisfy | ITR | Description |
|------|----------|---------|-----|------|
| Base | 38.2 | 30.0 | 20.1 | No post-training |
| Prompted Base | 37.7 | 26.4 | 26.0 | System prompt added for collaboration |
| CollabLLM | 37.3 | 28.0 | 32.6 | SOTA baseline |
| SFT | 40.7 | 33.4 | 92.3 | SFT only, ITR skyrockets |
| DPO | 40.5 | 29.2 | 33.1 | DPO only (from base) |
| **SFT+DPO** | **42.4** | 28.4 | 32.9 | Optimal combination |

A human study with 75 participants showed that DiscoverLLM achieved significantly higher satisfaction than the baseline and shorter completion times, with participants noting the model "seemed to predict what I wanted."

### Key Findings
- **Prompted base regresses**: Simply adding the prompt "Please help the user discover their intent" is ineffective; the model just mechanically asks more questions, which hinders Discovery. This capability cannot be prompted into existence; it must be trained.
- **SFT for "Proactiveness," DPO for "Precision"**: The ITR surge after SFT means the model acts more collaboratively, but it isn't precise in choosing actions; DPO uses pair-wise preferences to refine strategy selection.
- **Generalization to unseen domains**: DiscoverLLM remains effective for Travel Planning and Web Development, indicating it has learned a general "divergence/convergence" dialogue strategy rather than task-specific templates.
- **Shorter dialogues**: While Discovery improves, token count drops by over 30%, showing the model learns to "replace three clarification questions with one good proposal," raising interaction efficiency.

## Highlights & Insights
- **Redefining the Problem**: Turning the fact that "users don't know what they want" from a long-ignored edge case in NLP into a central problem, and distinguishing elicitation vs. discovery, is the paper's most significant conceptual contribution.
- **Computable Cognitive Modeling**: Translating Schön's design cognition theory into differentiable rewards via hierarchical trees and three-state nodes is an elegant bridge from cognitive science to ML training signals, setting a template for other subjective HCI scenarios.
- **Cumulative Probability for "Tangential Engagement"**: An assistant doesn't need a direct hit; providing counter-examples also drives user discovery. This accurately simulates real human decision-making and is far closer to reality than binary "correct/incorrect" rewards.
- **GRPO Works for Multi-turn**: Applying online RL to discovery tasks with sparse trajectory rewards validates the usability of GRPO under complex reward shaping.

## Limitations & Future Work
- **Monotonicity Assumption**: The model assumes once an intent is discovered, it stays discovered, not allowing for the user to change their mind, quit, or oscillate—which deviates from real creative processes.
- **"Guided Construction" vs. "Preference Discovery"**: The authors admit the simulator only models the latter; the former (where the assistant leads the construction) requires a more active world model.
- **Intent Tree Quality**: Trees are automatically constructed by LLMs; their abstraction levels and branching might not align with human mental models, leading to distorted reward signals in some scenarios.
- **Reward Hacking Risk**: Models might learn to "cram in multiple contrastive options to maximize discovery count" at the expense of single-output quality, though length penalties mitigate this.

## Related Work & Insights
- **vs. CollabLLM (Wu 2025)**: Both perform multi-turn collaborative training, but CollabLLM assumes well-defined intents that the model simply asks about, whereas DiscoverLLM assumes intents aren't formed and the model must surface them.
- **vs. RLHF (Ouyang 2022)**: Conventional RLHF rewards single-turn full outputs, which is fundamentally incompatible with intent discovery. This work moves rewards to the turn level.
- **Insights**: The paradigm of intent hierarchies + state machine simulators can be extended to (1) educational tutoring (students don't know what they don't know); (2) medical diagnosis (patients can't clearly describe symptoms); (3) legal consulting; and (4) requirement clarification between coding agents and human engineers.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ High. Formalizing "unformed user intent" as hierarchical discovery and building a rewardable simulator is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid coverage across 3 tasks, 2 bases, 4 training tiers, a 75-person user study, and 5 generalization tasks. Could be improved by using larger backbones (>8B).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Excellent. The essay example is vivid, the cognitive science foundation is well-laid, and Figure 2 clarifies the state machine well.
- **Value**: ⭐⭐⭐⭐⭐ High. Directly defines the next stage for LLM assistants (not just eliciting, but discovering), with significant implications for creative AI, HCI, and agent engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ICML 2026\] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives](is_your_llm_overcharging_you_tokenization_transparency_and_incentives.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](../../ACL2026/dialogue/disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)
- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](../../ACL2026/dialogue/context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ICML 2026\] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives](is_your_llm_overcharging_you_tokenization_transparency_and_incentives.md)
- [\[ACL 2025\] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](../../ACL2025/dialogue/know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](../../ACL2026/dialogue/disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)

</div>

<!-- RELATED:END -->
