---
title: >-
  [Paper Note] DiscoverLLM: From Executing Intents to Discovering Them
description: >-
  [ICML 2026][Dialogue Systems][RLHF] DiscoverLLM formalizes the scenario where "the user has not clearly defined their goals" as a progressive discovery process within a hierarchical intent tree. By using a rewardable hierarchical user simulator, the model is trained to actively explore divergently when goals are unclear and converge for execution when th
tags:
  - ICML 2026
  - Dialogue Systems
  - RLHF
date: 2026-05-08
content_hash: 02d1e74d6a63232e
---
# DiscoverLLM: From Executing Intents to Discovering Them

**Conference**: ICML 2026  
**arXiv**: [2602.03429](https://arxiv.org/abs/2602.03429)  
**Code**: https://taesookim.com/discoverllm  
**Area**: Human-Machine Dialogue / LLM Post-training / User Simulator  
**Keywords**: Intent Discovery, Multi-turn Dialogue, User Simulator, RLHF, Collaborative Creation

## TL;DR
DiscoverLLM formalizes the scenario where "the user has not clearly defined their goals" as a progressive discovery process within a hierarchical intent tree. By using a rewardable hierarchical user simulator, the model is trained to actively explore divergently when goals are unclear and converge for execution when they are clarified. On creative writing, technical writing, and SVG tasks, the method achieves a +10% improvement in satisfaction and a -40% reduction in dialogue length compared to baselines like CollabLLM.

## Background & Motivation
**Background**: Current LLM assistants default to the assumption that users know exactly what they want upon arrival. RLHF typically rewards single-turn outputs focused on "responding well in one go." Multi-turn research (e.g., CollabLLM, Shani et al.) has begun allowing models to ask clarifying questions, yet still assumes the intent is "already formed but unstated."

**Limitations of Prior Work**: In open-ended creative scenarios (writing, design), users often feel "I don't know what I want until I see a few drafts." In such cases, clarifying questions like "What tone do you want?" are ineffective because the user cannot answer them. The paper illustrates this with an example: when writing a personal essay, a user may feel the result is "not right" but cannot explain why until they see two contrasting drafts—one "overly intimate" and one "overly detached"—at which point they realize they want a "restrained but honest" tone.

**Key Challenge**: Clarification-based multi-turn training assumes formed but unexpressed intents. In reality, intents are discovered through interaction by observing outcomes. The former corresponds to "asking," while the latter requires "exploring/drafting."

**Goal**: (1) Formally define the difference between intent discovery and intent elicitation; (2) Construct a user simulator capable of providing optimizable reward signals; (3) Train an assistant that adaptively switches between divergence and convergence.

**Key Insight**: Drawing from cognitive science theories (Schön 1983; Flower & Hayes 1981) on the "co-evolution of problem and solution," humans discover preferences by creating and inspecting outcomes. These preferences can be organized into hierarchies (abstract to concrete), providing a computable structure for constructing a ground-truth simulator.

**Core Idea**: Model intent as a hierarchical tree $\mathcal{H}=(V,E)$, where node states transition between undiscovered, emerging, and discovered. Assistant responses are rewarded according to how many nodes they help discover. This signal is directly used for SFT, DPO, or GRPO.

## Method

### Overall Architecture
The core of DiscoverLLM is transforming the unquantifiable state of "user uncertainty" into a training loop with differentiable rewards. It first offline extracts a hierarchical intent tree from an existing artifact (story, article, SVG) to serve as the ground truth. A hierarchical user simulator then "feigns ignorance" on this tree: initially, only the abstract root nodes are discovered. For each assistant response, the simulator evaluates whether the response has caused hidden nodes to become "discovered," updates node states, generates the next utterance (either more concrete or more vague), and uses the number of newly discovered nodes as a reward. Finally, these simulated dialogues are used for SFT/DPO/GRPO post-training.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Existing Artifacts<br/>(Stories / Articles / SVGs)"] --> B["Automatic Intent Tree Construction<br/>LLM extracts leaves -> Abstract -> Build Tree"]
    B --> C["Hierarchical Intent Tree + Three-state Nodes<br/>undiscovered / emerging / discovered"]
    C --> D["Hierarchical User Simulator<br/>Starts with only abstract root nodes"]
    D --> E["Assistant Response"]
    E --> F["Response-driven State Transition<br/>Direct engagement / Tangential engagement accumulation"]
    F -->|Newly discovered nodes| G["Asymmetric Reward<br/>Discovery reward (positive only) + Capped length penalty"]
    F -->|Next utterance (more concrete or vague)| D
    G --> H["SFT / DPO / GRPO Post-training"]
```

### Key Designs

**1. Hierarchical Intent Tree + Three-state Nodes: Providing the simulator with a "feign ignorance" ground truth**

The difficulty in scoring intent discovery stems from the fact that real intent is neither visible nor structured. Here, intent is modeled as a tree $\mathcal{H}=(V,E)$. The root represents highly abstract intents (e.g., "include an animal"), which branch into concrete details ("pet" $\to$ "cat" $\to$ "Siamese" / "shorthair"), aligning with the "hierarchical network of goals" in cognitive science. The set of discovered nodes is $I_t \subseteq V$, and the refineable space is $\mathcal{R}(I_t) = \{v : \text{parent}(v) \in I_t, v \notin I_t\}$, requiring parents to be discovered before children. Crucially, nodes have three states: undiscovered, emerging, and discovered. Emerging nodes are mentioned vaguely (e.g., "Maybe change the small animal?") without exact specifics. This mechanism allows the simulator to generate realistic "I don't know" responses rather than binary transitions, enabling the model to receive differentiable rewards based on state changes.

**2. Response-driven State Transitions: Rewarding both "asking the right question" and "showing contrasting options"**

Different assistant responses must map to specific state transitions. Each turn, the simulator evaluates the response $r_t$ against nodes in $\mathcal{R}(I_t)$ via two paths: **Direct engagement** occurs when $r_t$ explicitly asks about or satisfies a node (e.g., "Which pet?" or drawing a cat), moving it immediately to discovered. **Tangential engagement** occurs when $r_t$ provides relevant but non-matching options, accumulating a score that, once past a random threshold, moves the state forward (undiscovered $\to$ emerging $\to$ discovered). This simulates the human cognitive process of ruling out possibilities by seeing counter-examples. Furthermore, the user is constrained to speak using the language of discovered nodes, while emerging nodes can only be mentioned vaguely. If "color" is undiscovered, an assistant asking "What color?" is ineffective, forcing the model to adopt divergent strategies like providing contrasting drafts. Thus, the reward $R_d = |I_{t+1}| - |I_t|$ rewards both exploration and clarification.

**3. Automatic Intent Tree Construction + Asymmetric Reward: Removing annotation bottlenecks and stabilizing penalties**

To scale across domains, trees must be generated automatically and rewards must be stable. Construction follows three steps: (a) An LLM lists all specific requirements met by an artifact as leaves; (b) The LLM iteratively abstracts multiple layers; (c) The LLM organizes these into a tree. For the reward, an asymmetric design is used: the discovery reward $R_d$ is only positive to purely encourage discovery. The length penalty $R_e = -\min(\lambda \cdot \max(0, \text{tokens}(r_t) - \tau), 1)$ captures efficiency but is capped at 1 to prevent it from overshadowing discovery. The total reward is $R(r_t) = R_d(r_t) + R_e(r_t)$. The authors intentionally avoid normalizing $R_d$ by the number of remaining intents, as normalization fluctuates late in the conversation; this asymmetric design is a key engineering factor for stable convergence.

### Loss & Training
Base models include Llama-3.1-8B-Instruct and Qwen3-8B. LoRA finetuning consists of four stages: (1) SFT on synthetic dialogues; (2) DPO on pair-wise comparisons (starting from base); (3) SFT+DPO (starting from SFT); (4) GRPO applied to Qwen3. Intent trees were constructed using Claude Sonnet 4.5, and the user simulator was Gemini 3 Flash. Dialogue sessions lasted 5 turns, and evaluation was averaged over 3 runs.

## Key Experimental Results

### Main Results
On Creative Writing, Technical Writing, and SVG tasks, four core metrics were used: Discovery, Satisfaction, Interactivity (ITR), and average token count. Llama-3.1-8B main results:

| Task | Configuration | Discover↑ | Satisfy↑ | ITR↑ | #Tok↓ |
|------|------|-----------|----------|------|-------|
| Creative Writing | Base | 38.2 | 30.0 | 20.1 | 3.09k |
| Creative Writing | CollabLLM | 37.3 | 28.0 | 32.6 | 2.93k |
| Creative Writing | **SFT+DPO** | **42.4** | 28.4 | 32.9 | **2.77k** |
| SVG | Base | 45.6 | 32.5 | 21.6 | 3.59k |
| SVG | **SFT+DPO** | **51.6** | **37.0** | **44.6** | **2.61k** |
| Technical Writing | SFT | 47.1 | 35.2 | **81.6** | 2.09k |

Running SFT alone caused ITR to surge to 80+ (high proactivity), but Discovery gains were modest. DPO was required to push Discovery to its peak. GRPO on Qwen3 provided further improvements.

### Ablation Study

| Configuration | Discover | Satisfy | ITR | Note |
|------|----------|---------|-----|------|
| Base | 38.2 | 30.0 | 20.1 | No post-training |
| Prompted Base | 37.7 | 26.4 | 26.0 | System prompt used for assistance |
| CollabLLM | 37.3 | 28.0 | 32.6 | SOTA baseline |
| SFT | 40.7 | 33.4 | 92.3 | SFT only, ITR sky-rockets |
| DPO | 40.5 | 29.2 | 33.1 | DPO only (no SFT start) |
| **SFT+DPO** | **42.4** | 28.4 | 32.9 | Optimal combination |

A 75-person user study showed that DiscoverLLM significantly outperformed baselines in satisfaction and completion time. Users noted the model "seemed to predict what I wanted."

### Key Findings
- **Prompted Base Regresses**: Simply prompting "please help the user discover their intent" was ineffective; the model merely asked more mechanical questions, dragging down Discovery. This indicates the ability must be trained, not prompted.
- **SFT for "Engagement," DPO for "Precision"**: The surge in ITR after SFT suggests the model becomes more collaborative, but its choice of action is imprecise. DPO refines the policy selection through pair-wise preferences.
- **Generalization to Untrained Domains**: DiscoverLLM remained effective in travel planning and web development, suggesting "divergence/convergence" is a learned general dialogue strategy rather than a task-specific template.
- **Shorter Dialogues**: While Discovery increased, token counts dropped by over 30%, showing the model learns to "replace three clarifying questions with one good proposal."

## Highlights & Insights
- **Re-defining the Problem**: Elevating "unformed user intent" from an overlooked NLP edge case to a central problem, and distinguishing elicitation from discovery, is the paper's primary conceptual contribution.
- **Computable Cognitive Modeling**: Translating Schön’s theory into a hierarchical tree and three-state nodes provides an elegant bridge from cognitive science to ML training signals.
- **Tangential Engagement Probabilities**: The insight that assistants do not need to "hit the target" immediately—and that providing counter-examples facilitates discovery—accurately simulates human decision-making.
- **GRPO for Multi-turn**: Applying online RL to tasks with sparse trajectory rewards (like discovery) validates the utility of GRPO under complex reward shaping.

## Limitations & Future Work
- **Monotonic Assumption**: The model assumes once an intent is discovered, it stays discovered, ignoring user backtracking or changing minds.
- **Guided Construction vs. Preference Discovery**: The simulator only models the latter; the former (proactive construction by the assistant) requires a more active world model.
- **Intent Tree Quality**: Trees are LLM-generated; the abstraction levels may not align perfectly with human mental models, potentially distorting reward signals.
- **Reward Hacking Risk**: Models may learn to over-propose items to maximize discovery at the expense of single-turn quality, a risk only partially mitigated by length penalties.

## Related Work & Insights
- **vs. CollabLLM (Wu 2025)**: Both use multi-turn collaborative training, but CollabLLM assumes well-defined intents that the model simply probes. DiscoverLLM assumes unformed intents that must be surfaced.
- **vs. RLHF (Ouyang 2022)**: Conventional RLHF uses full output rewards for single turns, which is incompatible with intent discovery. This work places rewards at the turn level.
- **Insight**: The paradigm of hierarchical intents and state-machine simulators can be extended to education (students not knowing what they don't know), medical diagnosis (vague symptoms), and legal consultation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizing "unformed intent" into hierarchical discovery and constructing a rewardable simulator is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across 3 tasks, 2 bases, and 75-person user study, though limited to 8B backbones.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear, vivid examples and well-explained cognitive science backgrounds.
- Value: ⭐⭐⭐⭐⭐ Defines a next-stage task for LLM assistants (discovery over just elicitation) with significant implications for HCI and agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ACL 2025\] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](../../ACL2025/dialogue/know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)
- [\[ICML 2026\] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives](is_your_llm_overcharging_you_tokenization_transparency_and_incentives.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](../../ACL2026/dialogue/disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)

</div>

<!-- RELATED:END -->
