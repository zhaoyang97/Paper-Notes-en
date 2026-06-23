---
title: >-
  [Paper Note] MemGen: Weaving Generative Latent Memory for Self-Evolving Agents
description: >-
  [ICLR 2026][LLM Agent][LoRA] MemGen enables LLM agents to determine when to recall in real-time during inference via a "memory trigger," followed by a "memory weaver" that generates machine-native latent token sequences injected into the inference stream. This intertwines memory and reasoning into a dynamic cycle, significantly outperforming param
tags:
  - ICLR 2026
  - LLM Agent
  - LoRA
  - Reinforcement Learning
date: 2026-05-08
content_hash: 59e2a9f835ea99fa
---
# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=vI56m4Iu4e](https://openreview.net/forum?id=vI56m4Iu4e)  
**Code**: [https://github.com/bingreeky/MemGen](https://github.com/bingreeky/MemGen)  
**Area**: LLM Agent / Agent Memory / Self-Evolution  
**Keywords**: Generative Memory, Latent Memory, Self-Evolving Agents, Metacognitive Trigger, LoRA, Reinforcement Learning  

## TL;DR
MemGen enables LLM agents to determine when to recall in real-time during inference via a "memory trigger," followed by a "memory weaver" that generates machine-native latent token sequences injected into the inference stream. This intertwines memory and reasoning into a dynamic cycle, significantly outperforming parametric and retrieval-based memory while keeping the backbone frozen and without modifying a single parameter.

## Background & Motivation
**Background**: To enable LLM agents to self-evolve from environment interactions, two main types of memory are currently used. The first is **parametric memory** (SFT / GRPO / DPO, etc.), which "bakes" experience directly into model parameters. The second is **retrieval-based memory** (Mem0, AWM, ExpeL, etc.), which externalizes experience into structured databases to be retrieved and concatenated into the context during usage.

**Limitations of Prior Work**: Parametric memory modifies backbone parameters, inevitably leading to catastrophic forgetting and the erosion of general knowledge. Retrieval-based memory, while non-intrusive to parameters, relies heavily on "context engineering" and follows a rigid pipeline of one-time retrieval, concatenation, and execution; memory is merely "pasted" before the query and fails to truly fuse with reasoning. Even existing latent memory methods (KV-cache types only address long contexts, while latent embedding types still require modifying LLM parameters) remain at the level of "retrieval by similarity" rather than "generative reconstruction," lacking the continuous interaction found in the human brain where thinking and recalling mutually shape each other.

**Key Challenge**: Agent memory either modifies parameters (leading to forgetting) or relies on concatenation (leading to rigidity), consistently failing to achieve "generating memory tailored to the current state at any critical node of inference and seamlessly integrating it into thinking."

**Goal**: To design agent memory as a **dynamic cognitive ability**—capable of generating memory in a streaming and reconstructive manner during inference, evolving in tandem with reasoning.

**Core Idea**: A **"Trigger + Weaver" dual-component system**—training a metacognitive monitor via RL to decide "when to recall" at the token level, and using a generator to reconstruct the current cognitive state into a fixed-length latent memory sequence for injection. The backbone LLM remains frozen throughout, with memory knowledge internalized only within the weaver.

## Method

### Overall Architecture
MemGen operates alongside a frozen core reasoner $\pi_\theta$ during autoregressive generation, monitoring internal hidden states token-by-token. A trigger $\mathcal{T}_{trigger}$ determines if "recalling" is currently necessary. Once an `INVOKE` decision is made, generation pauses, and a weaver $\mathcal{W}_{weaver}$ uses the current cognitive state as a stimulus to generate a latent memory token sequence $M_t$. After prefix-injection, the reasoner resumes generation. The entire process is a recursive cycle of "generation → monitoring → triggering → weaving → reintegration," leaving backbone parameters untouched.

```mermaid
flowchart LR
    A[State s_t] --> B[Frozen Reasoner π_θ<br/>Token-by-token Generation]
    B --> C{Trigger T_trigger<br/>Reads Hidden States H_t,<j}
    C -->|SKIP| B
    C -->|INVOKE Pause| D[Weaver W_weaver<br/>Reconstructs Latent Memory M_t]
    E[(External Memory Bank<br/>Optional)] -.-> D
    D --> F[M_t Prefix Injection<br/>z_t,j ~ π_θ·|s_t,z,M_t]
    F --> B
```

### Key Designs

**1. Interleaved Memory-Reasoning Cycle: Transforming memory from a "prologue" to an "on-call companion."** Traditional frameworks retrieve memory only once at the start of a task. MemGen allows memory to participate dynamically at the token level. As the reasoner generates an action $a_t=(z_{t,1},\dots,z_{t,L_t})$ for state $s_t$, every generated token produces a corresponding hidden state sequence $H_{t,<j}=(h_{t,1},\dots,h_{t,j-1})$. The trigger calculates an invocation probability; if recall is needed, the weaver produces a fixed-length $M_t\in\mathbb{R}^{K\times d_{model}}$, which is prefixed to $H_{t,<j}$. The reasoner then continues on the enriched context: $z_{t,j}\sim\pi_\theta(\cdot\mid s_t,z_{t,<j},M_t)$. Critically, $M_t$ is not a verbatim recap of history but a **selective reconstruction** filtered and integrated by the weaver, analogous to memory consolidation in the human brain—this upgrades reasoning from linear expansion to a "recursive dialogue with memory," naturally preserving general capabilities and avoiding forgetting due to the frozen $\pi_\theta$.

**2. Memory Trigger: Learning "When to Recall" via metacognitive signals.** The trigger is instantiated as a lightweight LoRA adapter attached to $\pi_\theta$. It takes hidden states as input rather than text (latent embeddings preserve context-sensitive information lost after softmax decoding) and outputs an invocation probability $p_j=\sigma(\mathcal{T}_{trigger}(h_{t,1},\dots,h_{t,j-1}))$, followed by a sampled binary decision $d_j\sim\mathrm{Bernoulli}(p_j)$. to control overhead, it employs **sentence-level activation**: calculations occur only at delimiter sets $D$ (commas, periods, etc.); when $z_j\notin D$, $p_j=0$, restricting intervention to semantic boundaries to maintain decoding efficiency. Training uses RL with the objective of "sparse invocation at critical points," introducing a **reward-adaptive penalty**: $\max_\phi\ \mathbb{E}[R(\tau_i)-\lambda\sum_{i,j}\max(0,\tilde d_{i,j}-\bar p)]$, where $\bar p$ is the average activation probability on high-reward trajectories (rewards exceeding the batch median), thereby suppressing unnecessary activations while preserving essential recalls.

**3. Memory Weaver: Internalizing experience and synthetically generating latent memory.** The weaver is also a LoRA adapter attached to $\pi_\theta$. It receives the hook $H_{t,<j}$ and outputs a latent memory matrix $M_t=\mathcal{W}^{\theta'}_{weaver}(H_{t,<j})$, which is projected onto the token embedding space via a linear layer for prefix-injection. **Crucial Decoupling**: When the agent absorbs new experience, knowledge is written only into the parameters of $\mathcal{W}_{weaver}$, while $\pi_\theta$ remains completely unchanged. This makes MemGen agnostic to optimization algorithms and compatible with various backbones; whether using SFT or GRPO/DAPO, updates occur under a unified objective: $\max_{\theta_{lora}}\mathbb{E}\,R(x_i,\tau)$, with gradients backpropagated only to $\theta'$. The training follows a two-stage process: first training the weaver using a random inserter as a lightweight proxy for the trigger, then freezing the weaver to train the trigger. Additionally, the weaver can be combined with any retrieval system (MemoryBank, ExpeL, etc.): retrieved text memory is merged with the hook and fed into the weaver, allowing it to integrate internal parametric knowledge with external information.

## Key Experimental Results

### Main Results Table (SmolLM3-3B / Qwen3-8B, Accuracy %, Abridged)

| Method | ALFWorld | TriviaQA | PopQA | KodCode | GPQA | GSM8K |
|---|---|---|---|---|---|---|
| Vanilla (3B) | 18.96 | 10.47 | 8.23 | 37.05 | 9.35 | 47.63 |
| GRPO (3B) | 55.35 | 65.88 | 45.16 | 68.48 | 22.73 | 80.03 |
| ExpeL (3B) | 36.18 | 46.20 | 28.16 | 51.14 | 15.15 | 56.23 |
| **MemGen GRPO (Ours, 3B)** | **63.60** | **79.30** | **58.60** | **72.85** | 25.20 | **83.47** |
| GRPO (8B) | 85.60 | 76.15 | 58.90 | 73.35 | 39.54 | 92.30 |
| AWM (8B) | 80.33 | 69.30 | 43.69 | - | - | - |
| **MemGen GRPO (Ours, 8B)** | **90.60** | **80.65** | **62.30** | **76.16** | **40.24** | **93.20** |

Covering 9 datasets across 5 domains and comparing against 12 baselines (Prompt-based / Parametric / Retrieval / Latent computation). MemGen leads comprehensively across all domains: on Qwen3-8B, KodCode +27.06% and PopQA +28.17% (relative to vanilla), outperforming strong GRPO by up to 3.4%.

### Ablation Study (Sensitivity to Latent Memory Length K)
For latent memory sequence length $K\in\{2,4,8\}$, lengths that are too short provide insufficient information, while those that are too long introduce noise; an optimal performance interval exists (see original Figure 6 left). Results for variants integrated with retrieval-based memory are provided in Appendix F.

### Key Findings
- **Cross-domain Generalization (RQ2)**: When trained on KodCode, MATH performance increases from 36.6% to 54.2%. The trigger adaptively adjusts invocation frequency based on the task—GSM8K saw the Gain (+19.64%) with the most frequent calls, while KodCode saw the smallest Gain with the fewest calls, indicating the ability to judge when to minimize memory intervention in unfamiliar domains to avoid conflicts.
- **Continual Learning (RQ3)**: In sequential training across four datasets, MemGen maintained 40.34% on AQuA after KodCode training (vs. ExpeL 27.14%, SFT 28.61%), effectively mitigating catastrophic forgetting.
- **Emergence of Human-like Memory Hierarchy (RQ4)**: Without explicit supervision, latent memory spontaneously clusters by domain under t-SNE (machine-native, non-readable but structured). Posterior intervention (deleting specific clusters) reveals functionally differentiated **planning memory, procedural memory, and working memory**. For example, removing Cluster 2 leads to a sharp increase in failures related to planning/combinatorial reasoning.

## Highlights & Insights
- **"Generative Reconstruction" rather than "Retrieval Recap"**: Treating memory as a cognitive product that can be re-synthesized moves beyond the paradigm limitations of retrieval-based memory's "fetch by similarity."
- **Frozen Backbone + External Dual LoRA**: Knowledge enters only the weaver, architecturally isolating "learning new experiences" from "preserving general capabilities," providing natural immunity to catastrophic forgetting and decoupling from optimization algorithms/backbones.
- **Token-level Metacognitive Trigger**: Sentence-level activation combined with reward-adaptive penalties makes "when to recall" a learnable, interpretable, and sparse decision.
- **Unexpected Interpretability Gains**: The functionally differentiated memory hierarchy emerges spontaneously rather than being imposed by design, providing empirical clues for the evolution of machine cognition toward natural forms.

## Limitations & Future Work
- Latent memory is machine-native and not human-readable; forced decoding shows structural patterns but lacks semantic clarity, leaving interpretability limited and debugging/trustworthiness questionable.
- The dual-component (Trigger + Weaver) and two-stage training process increase engineering complexity compared to pure retrieval solutions. While sentence-level activation controls overhead, insertion still introduces some inference latency (validated as acceptable in Appendix E.3.3).
- Experimental backbones are concentrated between 1.5B–8B; the capacity and gains of latent memory on larger models remain to be verified.
- The "Human-like Memory Hierarchy" is based on attributional interpretation from posterior intervention; causal evidence remains relatively weak.

## Related Work & Insights
- **Parametric Memory** (FireAct, AgentLumos, Agent-FLAN): Experience enters parameters; powerful but prone to forgetting.
- **Retrieval-based Memory** (Mem0, AWM, ExpeL, MemoryBank, G-Memory, AgentKB): Experience is externalized into databases; no forgetting but remains rigid.
- **Latent / Latent Computation** (Coconut, CODI, CoLaR, SoftCoT, LatentSeek, Co-processor): Using latent states for native reasoning or modulating generation. MemGen belongs to this lineage but emphasizes reasoning-memory interleaving and generative reconstruction.
- **Insight**: Upgrading "memory" from static data/parameters to an "on-demand generative cognitive module" controlled by metacognitive signals provides a new path for self-evolving agents to learn continuously without modifying the backbone.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ "Generative latent memory + token-level metacognitive triggering + frozen backbone dual LoRA" represents a substantive restructuring of the agent memory paradigm and unexpectedly reveals emergent functionally differentiated memory hierarchies.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 9 datasets across 5 domains, 12 baselines, and 3 backbones. Covers main experiments, generalization, continual learning, and mechanism analysis comprehensively; trade-offs on larger models and latency could be further explored.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The cognitive metaphor of "memory-reasoning interleaving" is consistently maintained. Figures 1 and 2 provide clear comparisons, and the RQ-driven structure ensures a smooth read.
- **Value**: ⭐⭐⭐⭐⭐ Simultaneously addresses the twin pain points of forgetting and rigidity. The method is decoupled from optimization algorithms and backbones, allowing for plug-and-play use, which is highly significant for the self-evolving agent and memory system communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents](../../ACL2026/llm_agent/searl_joint_optimization_of_policy_and_tool_graph_memory_for_self-evolving_agent.md)
- [\[ICLR 2026\] AlphaAgentEvo: Evolution-Oriented Alpha Mining via Self-Evolving Agentic Reinforcement Learning](alphaagentevo_evolution-oriented_alpha_mining_via_self-evolving_agentic_reinforc.md)
- [\[ICML 2026\] Self-evolving LLM agents with in-distribution Optimization](../../ICML2026/llm_agent/self-evolving_llm_agents_with_in-distribution_optimization.md)
- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)
- [\[ICLR 2026\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)

</div>

<!-- RELATED:END -->
