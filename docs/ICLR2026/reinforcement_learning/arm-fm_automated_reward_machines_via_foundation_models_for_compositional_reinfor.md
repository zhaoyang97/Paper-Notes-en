---
title: >-
  [Paper Note] ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Reward machines] This paper proposes ARM-FM, a framework that leverages foundation models (e.g., GPT-4o) to automatically generate Language-Aligned Reward Machines (LARMs) from natural language task descriptions — encompassing the automaton structure, executable label functions, and per-state natural language descriptions — providing RL agents with compositional dense reward signals. The framework successfully solves sparse-reward long-horizon tasks that standard RL completely fails to learn, across environments including MiniGrid, Craftium (3D Minecraft), and Meta-World, while achieving zero-shot task generalization.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Reward machines
  - foundation models
  - compositional RL
  - language-aligned automata
  - zero-shot generalization
date: 2026-05-08
content_hash: db3d5d2332fd5cdb
---

# ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2510.14176](https://arxiv.org/abs/2510.14176)
**Code**: See paper
**Area**: Reinforcement Learning / LLM Agent
**Keywords**: Reward machines, foundation models, compositional RL, language-aligned automata, zero-shot generalization

## TL;DR
This paper proposes ARM-FM, a framework that leverages foundation models (e.g., GPT-4o) to automatically generate Language-Aligned Reward Machines (LARMs) from natural language task descriptions — encompassing the automaton structure, executable label functions, and per-state natural language descriptions — providing RL agents with compositional dense reward signals. The framework successfully solves sparse-reward long-horizon tasks that standard RL completely fails to learn, across environments including MiniGrid, Craftium (3D Minecraft), and Meta-World, while achieving zero-shot task generalization.

## Background & Motivation

### State of the Field

**Background**: The central bottleneck in RL is reward function design — sparse reward signals are insufficient, while dense rewards require expert hand-engineering and are prone to exploitation. Reward Machines (RMs) decompose tasks into sequences of subgoals using finite automata, which is theoretically elegant but requires manual expert design in practice.

**Limitations of Prior Work**: (a) Manual RM design is costly, limiting widespread adoption; (b) Foundation models excel at task decomposition but struggle to produce structured reward signals usable by RL; (c) Existing FM-RL integrations (e.g., Eureka/Motif) output opaque reward models that lack compositionality and interpretability.

**Key Challenge**: FMs possess high-level reasoning capabilities but lack low-level control understanding required by RL, while RL agents have low-level control capabilities but lack high-level task understanding — a structured interface bridging the two is needed.

**Goal**: Automatically generate interpretable, compositional reward specifications (RMs) using FMs, while supporting cross-task skill sharing and generalization.

**Key Insight**: The automaton structure of RMs naturally aligns with the code generation capabilities of FMs (generating states, transition functions, and label functions), while natural language descriptions of RM states can connect different tasks via embedding spaces.

**Core Idea**: Have FMs generate "Language-Aligned Reward Machines" (LARMs) — automating RM design while constructing a shareable skill space via language embeddings.

## Method

### Overall Architecture
ARM-FM consists of two components: (1) **LARM Generation** — FMs generate the RM structure, Python label function code, and natural language descriptions for each state from natural language and visual observations, refined through a generator-critic FM self-improvement loop with optional human verification; (2) **RL Training** — the agent policy conditions on environment states and language embeddings of the current RM state, with the RM providing dense intermediate rewards $R_t^{\text{total}} = R_t + R_t^{\text{RM}}$.

### Key Designs

1. **Automated Generation of Language-Aligned Reward Machines (LARMs)**

    - **Function**: Automatically generate complete RM specifications from natural language task descriptions.
    - **Mechanism**: FMs generate three components — (a) the RM automaton structure (state set $U$, transition function $\delta$, terminal states $F$); (b) executable Python label functions $\mathcal{L}: S \times A \to \Sigma$ mapping environment observations to RM event symbols; (c) natural language instructions $l_u$ for each RM state. Specifications are iteratively refined through $N$ rounds of generator-critic self-improvement.
    - **Design Motivation**: The formal structure of RMs (states + transitions) naturally matches FM code generation capabilities, and the outputs are verifiable (automaton structural correctness can be checked).

2. **Language Embedding-Conditioned Policy**

    - **Function**: Encode natural language descriptions of RM states as vectors to serve as inputs to the policy network.
    - **Mechanism**: $\pi(a_t | s_t, z_{u_t})$, where $z_{u_t} = \phi(l_{u_t})$ is the language embedding of the current RM state description. Semantically similar subtasks (e.g., "pick up blue key" vs. "pick up red key") are naturally close in embedding space, enabling the policy to share knowledge across them.
    - **Design Motivation**: Traditional RMs use independent one-hot encodings for different states, precluding cross-task skill sharing. Language embeddings construct a semantically continuous skill space, enabling zero-shot compositional generalization.

3. **Zero-Shot Compositional Task Generalization**

    - **Function**: An agent trained on tasks A and B directly solves an unseen compositional task C using a newly generated LARM-C.
    - **Mechanism**: If the subgoal embeddings $z_{u'}$ of new task C fall within familiar regions of the embedding space encountered during training, the policy can directly reuse previously learned skills.
    - **Design Motivation**: Compositional generalization arises naturally from combining the RM framework with language embeddings — FMs generate new RMs while the policy reuses existing skills.

## Key Experimental Results

### MiniGrid (Sparse Rewards)
- DQN+RM solves three long-horizon tasks (UnlockToUnlock, BlockedUnlockPickup, KeyCorridor) on which all baselines (DQN/DQN+ICM/ReAct) completely fail to learn.
- On the DoorKey task, the advantage over baselines grows as grid size increases (8×8→16×16).

### Craftium (3D Minecraft Resource Collection)
- PPO+LARM completes the full multi-step "collect diamonds" task sequence (chop wood → mine stone → mine iron → mine diamonds), while baseline PPO makes near-zero progress.

### Meta-World (Robot Manipulation)
- SAC+LARM achieves high success rates on most continuous control tasks, substantially outperforming sparse-reward baselines.

### Multi-Task Generalization (XLand-MiniGrid)
- A single agent trained jointly on 10 tasks: baselines fail completely; rewards alone provide limited benefit (no knowledge of the current subgoal); embeddings alone yield weak signals → the full method (rewards + embeddings) maintains high performance.
- **Zero-shot generalization**: An agent trained on tasks A and B directly solves the unseen compositional task C.

### FM Scaling Analysis
- Across 1,000 LARM generation experiments, larger models yield higher correctness rates (Qwen3-32B substantially outperforms smaller models).
- PCA visualization reveals clear semantic clustering of state embeddings (start/intermediate/terminal states are well separated).

## Highlights & Insights
- The framing of **RMs as a FM-RL interface** is particularly insightful — RMs provide a structured intermediate representation that FMs can generate, humans can interpret, and RL agents can learn from.
- The design of **constructing a skill space via language embeddings** is elegant — "pick up blue key" and "pick up red key" naturally share policy representations, making compositional generalization a direct consequence.
- The experimental coverage is broad (2D / 3D / continuous control / multi-task / zero-shot), demonstrating the framework's generality.
- The self-improvement loop (generator + critic FM) effectively improves generation quality.

## Limitations & Future Work
- Executable Python label functions depend on environment-specific APIs — different environments require different API specifications.
- The optional human verification step improves quality, indicating a remaining gap in full automation.
- Main experiments use GPT-4o (the strongest FM) for LARM generation — practical usability of weaker FMs remains limited.
- The expressiveness of RMs is constrained by finite automata — tasks requiring counting or continuous tracking cannot be represented.
- Zero-shot generalization is limited to recombinations of subgoals seen during training — truly novel subgoals still require additional training.

## Related Work & Insights
- **vs. Eureka (Ma et al.)**: Eureka uses evolutionary strategies to generate programmatic reward functions — opaque and non-compositional; ARM-FM generates RMs — structured and composable.
- **vs. ReAct (LLM-as-agent)**: ReAct uses LLMs directly for decision-making and requires text-based interfaces — unsuitable for pixel-level control; ARM-FM uses LLMs to generate specifications while RL handles control.
- **vs. Manual RM**: Automated generation eliminates the expert design bottleneck, and language embeddings add generalization capability.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Automated generation of language-aligned RMs combined with embedding-space skill sharing constitutes an entirely new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four categories of environments × multiple baselines + multi-task + zero-shot + FM scaling analysis; exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Figure 1 provides an excellent architectural overview; experimental organization is logically clear.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a novel structured paradigm for FM-RL integration with both practical utility and theoretical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Optimistic Task Inference for Behavior Foundation Models](optimistic_task_inference_behavior_models.md)
- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
