---
title: >-
  [Paper Note] DiscoverLLM: From Executing Intents to Discovering Them
description: >-
  [ICML 2026][LLM Agent][Intent Discovery] DiscoverLLM formalizes the scenario where "users themselves are unclear about what they want" as a progressive discovery process over a hierarchical intent tree. It uses a rewarda…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Intent Discovery"
  - "Multi-turn Dialogue"
  - "User Simulator"
  - "RLHF"
  - "Collaborative Creation"
date: 2026-05-08
content_hash: dee5b42f3560442e
---

# DiscoverLLM: From Executing Intents to Discovering Them

**Conference**: ICML 2026  
**arXiv**: [2602.03429](https://arxiv.org/abs/2602.03429)  
**Code**: https://taesookim.com/discoverllm  
**Area**: Human-Computer Dialogue / LLM Post-Training / User Simulator  
**Keywords**: Intent Discovery, Multi-turn Dialogue, User Simulator, RLHF, Collaborative Creation

## TL;DR
DiscoverLLM formalizes the scenario where "users themselves are unclear about what they want" as a progressive discovery process over a hierarchical intent tree. It uses a rewardable hierarchical user simulator to train models that actively diverge and explore when user intent is unclear, and converge to execution when intent is clear. On creative writing, technical writing, and SVG tasks, it outperforms baselines like CollabLLM by +10% in satisfaction and reduces dialogue length by 40%.

## Background & Motivation
**Background**: Current LLM assistants assume users know their intent upon arrival, and RLHF directly rewards "good single-turn answers." Multi-turn approaches (CollabLLM, Shani, etc.) have models proactively ask clarifying questions, but still assume the intent "already exists, just unspoken."

**Limitations of Prior Work**: In open-ended creative scenarios (writing, design), users often "don’t know what they want until they see a few drafts." Clarifying questions like "What tone do you want?" are ineffective—users cannot answer. The paper’s opening example: asking an LLM to write a personal essay, the user feels "something’s off but can’t say what," and only after seeing "overly intimate" and "overly distant" counterexamples does the user realize they want "restrained yet candid."

**Key Challenge**: Clarification-based multi-turn training assumes intent is formed but unexpressed; in reality, intent itself forms through interaction and observing outcomes. The former corresponds to "asking," the latter requires "exploring/trying."

**Goal**: (1) Formally define the distinction between intent discovery and intent elicitation; (2) Construct a user simulator that provides optimizable reward signals; (3) Train assistants that adaptively switch between divergent and convergent strategies.

**Key Insight**: Drawing from cognitive science (Schön 1983, Flower & Hayes 1981) on the "co-evolution of problem and solution"—humans discover preferences by creating and examining outcomes, and these preferences can be organized hierarchically (abstract → concrete). This provides a computable structure for building ground-truth simulators.

**Core Idea**: Model intent as a hierarchical tree $\mathcal{H}=(V,E)$, with node states transitioning among undiscovered / emerging / discovered. Model responses that help discover more nodes receive higher rewards—this signal is directly fed to SFT / DPO / GRPO.

## Method

### Overall Architecture
DiscoverLLM adopts a "user simulator + post-training" dual-stage design:

1. **Intent Tree Construction** (offline): Automatically extract multi-level intent trees from existing artifacts (stories, articles, SVGs).
2. **Simulated Interaction** (online training): Simulate users initially discovering only a few abstract nodes at the tree’s top; assistant responds → simulator evaluates whether undiscovered nodes are "touched" → update node states → simulator generates the next, more specific or more vague utterance accordingly.
3. **Reward Calculation**: $R(r_t) = R_d(r_t) + R_e(r_t)$, where $R_d = |I_{t+1}| - |I_t|$ is the number of newly discovered nodes, and $R_e$ is a length penalty.
4. **Post-Training**: Use simulated dialogues as SFT data, pairwise comparisons for DPO, or direct online GRPO.

### Key Designs

1. **Hierarchical Intent Tree + Three-State Nodes**:

    - **Function**: Serves as ground truth, enabling the simulator to "act unclear" while providing differentiable reward signals.
    - **Mechanism**: The root is a highly abstract intent (e.g., "contains an animal"), with increasing specificity down the tree ("pet" → "cat" → "Siamese"/"shorthair"). The current state $I_t \subseteq V$ is the set of discovered nodes; the refinement space $\mathcal{R}(I_t) = \{v : \text{parent}(v) \in I_t, v \notin I_t\}$ strictly requires "parent must be discovered before child," and branches are independent. Each node is in one of three states: undiscovered / emerging / discovered. Emerging nodes can only be vaguely referenced by the user ("maybe try a small animal?") and not directly named.
    - **Design Motivation**: The hierarchical structure aligns with cognitive science’s "hierarchical network of goals"; the three-state mechanism naturally produces realistic dialogue patterns like "I don’t know either," rather than a binary know/don’t know.

2. **Response-Driven State Transitions**:

    - **Function**: Different assistant responses → different state transitions → automatic reward generation.
    - **Mechanism**: Each round, the simulator evaluates the assistant’s response $r_t$ against nodes in $\mathcal{R}(I_t)$.
        - Direct engagement: $r_t$ directly asks about or fulfills a node → node becomes discovered (e.g., asks "which pet do you want?" or generates a cat).
        - Tangential engagement: $r_t$ provides related but non-matching options → accumulates a score, and after passing a random threshold, the state advances (undiscovered → emerging → discovered). This simulates the cognitive process of "ruling out possibilities after seeing counterexamples."
        - Expression constraint: Users can only speak using discovered nodes’ language; emerging nodes can only be vaguely referenced. This ensures that assistant questions like "what color do you want?" are ineffective before "color" is discovered, forcing the model to adopt divergent strategies like "provide two contrasting drafts."
    - **Design Motivation**: Operationalizes the abstract "discovery" as a computable state machine, with reward $R_d = |I_{t+1}| - |I_t|$ encouraging both "asking the right questions" and "showing contrasting options," enabling the model to learn when to diverge and when to converge.

3. **Automatic Intent Tree Construction + Asymmetric Reward**:

    - **Function**: Automatically converts any artifact (story, article, SVG) into a trainable intent tree, avoiding manual annotation bottlenecks.
    - **Mechanism**: Three steps—(a) LLM lists all concrete requirements satisfied by the artifact as leaves; (b) LLM iteratively abstracts multiple layers; (c) LLM organizes multi-level intents into a tree, identifying which abstract nodes subsume which concrete nodes. In rewards, $R_d$ is strictly positive (encourages discovery), $R_e = -\min(\lambda \cdot \max(0, \text{tokens}(r_t) - \tau), 1)$ penalizes overlong outputs but is capped at 1, ensuring discovery reward is not overwhelmed. The authors deliberately avoid normalizing $R_d$ (by dividing by remaining intents) to prevent unstable reward signals in later stages.
    - **Design Motivation**: Makes the framework annotation-free and easily extensible to new domains; asymmetric reward prevents length penalty from dominating discovery reward, improving engineering stability.

### Loss & Training
Base models are Llama-3.1-8B-Instruct and Qwen3-8B, with LoRA fine-tuning in four settings: (1) SFT on synthetic dialogues; (2) DPO on pairwise comparisons (starting from base); (3) SFT+DPO (starting from SFT); (4) GRPO added on Qwen3. Intent trees are constructed with Claude Sonnet 4.5, user simulator uses Gemini 3 Flash, dialogues are 5 turns, and evaluation averages over 3 runs.

## Key Experimental Results

### Main Results
On creative writing, technical writing, and SVG tasks, four core metrics: Discovery, Satisfaction, Interactivity (ITR), and average token count. Llama-3.1-8B main results:

| Task | Setting | Discover↑ | Satisfy↑ | ITR↑ | #Tok↓ |
|------|---------|-----------|----------|------|-------|
| Creative Writing | Base | 38.2 | 30.0 | 20.1 | 3.09k |
| Creative Writing | CollabLLM | 37.3 | 28.0 | 32.6 | 2.93k |
| Creative Writing | **SFT+DPO** | **42.4** | 28.4 | 32.9 | **2.77k** |
| SVG | Base | 45.6 | 32.5 | 21.6 | 3.59k |
| SVG | **SFT+DPO** | **51.6** | **37.0** | **44.6** | **2.61k** |
| Technical Writing | SFT | 47.1 | 35.2 | **81.6** | 2.09k |

SFT alone causes ITR to soar above 80 (indicating the model becomes highly proactive), but Discovery increases little; DPO pushes Discovery to its peak. Adding GRPO on Qwen3 further improves results (see full numbers in the paper).

### Ablation Study

| Setting | Discover | Satisfy | ITR | Notes |
|---------|----------|---------|-----|-------|
| Base | 38.2 | 30.0 | 20.1 | No post-training |
| Prompted Base | 37.7 | 26.4 | 26.0 | Added system prompt for assistance |
| CollabLLM | 37.3 | 28.0 | 32.6 | SOTA baseline |
| SFT | 40.7 | 33.4 | 92.3 | SFT only, ITR surges |
| DPO | 40.5 | 29.2 | 33.1 | DPO only (no SFT) |
| **SFT+DPO** | **42.4** | 28.4 | 32.9 | Best combination |

75-person human user study: For the same tasks, DiscoverLLM achieves significantly higher satisfaction and shorter completion times than baselines. Users report the model "seems to predict what I want."

### Key Findings
- **Prompted base regresses**: Simply prompting "help the user discover intent" is ineffective; the model mechanically asks more questions, actually hurting Discovery. This ability cannot be prompted, only trained.
- **SFT makes the model 'active', DPO makes it 'precise'**: SFT boosts ITR, making the model more collaborative/proactive, but action selection remains imprecise; DPO refines strategy selection via pairwise preferences.
- **Generalizes to unseen domains**: DiscoverLLM is effective in travel planning and web development, indicating that "divergence/convergence" is a learned general dialogue strategy, not a task-specific template.
- **Shorter dialogues**: Discovery improves while token count drops by 30%+, showing the model learns to "replace three clarifying questions with one good proposal," increasing interaction efficiency.

## Highlights & Insights
- **Redefining the problem**: Elevates "users don’t know what they want" from a neglected NLP edge case to a central issue, distinguishing elicitation from discovery—this is the paper’s key conceptual contribution.
- **Computable cognitive modeling**: Translates Schön’s design cognition theory into differentiable rewards using hierarchical trees and three-state nodes. This bridge from cognitive science to ML training signals is elegant and sets a template for other fuzzy HCI scenarios.
- **Tangential engagement with cumulative probability**: The assistant need not hit the target directly; providing counterexamples also helps users discover—this closely simulates real human decision-making, far more realistic than binary right/wrong rewards.
- **GRPO works in multi-turn**: Applying online RL to dialogue discovery with sparse trajectory rewards validates GRPO’s usability under complex reward shaping.

## Limitations & Future Work
- **Monotonicity assumption**: The model assumes once an intent is discovered, it remains so—users cannot backtrack, abandon, or waver, which diverges from real creative processes.
- **User: guided construction vs. discovering inner preference**: The authors admit the simulator only models the latter; the former (assistant-led construction) requires a more proactive world model.
- **Intent tree quality bottleneck**: Trees are auto-constructed by LLMs; node abstraction and branching may not align with human cognition, causing reward signal distortion in some scenarios.
- **Reward hacking risk**: The model may learn to "force multiple contrasting options to maximize discovery" at the expense of single-output quality; length penalty partially mitigates this but not fully.
- **5-turn dialogue cap**: For longer conversations, the stability of emerging node thresholds needs retuning.

## Related Work & Insights
- **vs CollabLLM (Wu 2025)**: Both use multi-turn collaborative training, but CollabLLM assumes well-defined intent and the model only asks follow-ups; DiscoverLLM assumes intent is unformed and the model must proactively surface it. CollabLLM also improves ITR but not Discovery.
- **vs RLHF (Ouyang 2022)**: Single-turn full output reward, fundamentally incompatible with intent discovery; this work shifts reward to the turn level.
- **vs classic intent classification**: Those methods assume intent is a finite discrete set; here, intent is a dynamically growing tree.
- **Insights**: The hierarchical intent + state machine simulator paradigm can be extended to (1) educational tutoring (students unsure where they struggle); (2) medical triage (patients unclear about symptoms); (3) legal consultation (clients uncertain about claims); even (4) coding agents collaborating with human engineers for requirement clarification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizes "unformed user intent" as a computable hierarchical tree discovery problem and constructs a rewardable simulator—both problem definition and method are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 tasks × 2 bases × 4 training settings + 75-person user study + 5 cross-task generalizations + 4 simulator ablations—broad coverage; could be improved by using larger backbones.
- Writing Quality: ⭐⭐⭐⭐⭐ The opening essay example is vivid, cognitive science background is well laid out, and Figure 2 clearly explains the state machine.
- Value: ⭐⭐⭐⭐⭐ Directly defines the next-stage task for LLM assistants (not just elicitation, but discovery), with significant impact on creative AI, HCI, and agent engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ICML 2026\] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)
- [\[ACL 2025\] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](../../ACL2025/dialogue/know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](../../ACL2026/dialogue/disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](../../ACL2026/dialogue/ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)

</div>

<!-- RELATED:END -->
