---
title: >-
  [Paper Note] SynthAgent: Adapting Web Agents with Synthetic Supervision
description: >-
  [ACL 2026][LLM Agent][Synthetic Data] This paper proposes SynthAgent, a web agent adaptation framework built entirely on synthetic supervision. It employs categorized exploration to systematically cover functional regions of webpages for diverse task synthesis, followed by a dual refinement strategy—task refinement (conflict-triggered correction of hallucinations) and trajectory refinement (global-context denoising)—to improve synthetic data quality. SynthAgent significantly outperforms existing synthetic methods on WebArena and Online-Mind2Web.
tags:
  - ACL 2026
  - LLM Agent
  - Synthetic Data
  - Web Agent
  - Dual Refinement
  - Categorized Exploration
  - Trajectory Quality
date: 2026-05-08
content_hash: e97f268297df2f5c
---

# SynthAgent: Adapting Web Agents with Synthetic Supervision

**Conference**: ACL 2026  
**arXiv**: [2511.06101](https://arxiv.org/abs/2511.06101)  
**Code**: [GitHub](https://github.com/aiming-lab/SynthAgent)  
**Area**: LLM Agent / Web Agent Adaptation  
**Keywords**: Synthetic Data, Web Agent, Dual Refinement, Categorized Exploration, Trajectory Quality

## TL;DR

This paper proposes SynthAgent, a web agent adaptation framework built entirely on synthetic supervision. It employs categorized exploration to systematically cover functional regions of webpages for diverse task synthesis, followed by a dual refinement strategy—task refinement (conflict-triggered correction of hallucinations) and trajectory refinement (global-context denoising)—to improve synthetic data quality. SynthAgent significantly outperforms existing synthetic methods on WebArena and Online-Mind2Web.

## Background & Motivation

**State of the Field**: LLM-driven web agents demonstrate strong web interaction capabilities on standardized benchmarks, yet performance degrades sharply when deployed to unseen websites. Adapting to new environments requires environment-specific tasks and demonstration data, but manual annotation is costly and does not scale.

**Limitations of Prior Work**: (1) Self-Instruct prompts LLMs to "imagine" tasks without environmental grounding, resulting in simple and repetitive tasks; (2) OS-Genesis synthesizes tasks by back-translating from single-step observations, where insufficient context leads to substantial hallucinations (referencing non-existent elements or states); (3) Explorer continuously refines tasks during execution, but frequent intent changes (8.6 times on average) cause 68.3% of trajectories to exceed the step budget.

**Root Cause**: Task synthesis requires environmental grounding to avoid hallucinations, yet over-grounding tasks during execution introduces trajectory noise—a fundamental design tension in synthetic supervision.

**Paper Goals**: To design a fully synthetic supervision framework that efficiently adapts web agents to new environments without human involvement or test-set leakage.

**Starting Point**: Decouple task refinement and trajectory refinement into two complementary stages: task refinement ensures feasibility but introduces noise, while trajectory refinement subsequently eliminates that noise.

**Core Idea**: Dual refinement—during execution, tasks are refined only when explicit conflicts are detected (conflict-triggered rather than continuous); after execution, trajectories are refined using global context. This simultaneously guarantees task feasibility and trajectory quality.

## Method

### Overall Architecture

SynthAgent consists of four stages: (1) **Categorized Exploration for Task Synthesis**—webpage elements are grouped by function, interaction triples $(o_t, a_t, o_{t+1})$ are uniformly sampled, and an LLM proposes multi-step tasks grounded in real interface transitions; (2) **Conflict-Triggered Task Refinement**—conflicts between tasks and observations are detected during trajectory collection, with refinement invoked only upon triggering (2.0 times on average); (3) **Global Trajectory Refinement**—noisy and misaligned actions are removed post-hoc using the complete trajectory and the final task $\tau^{\star}$; (4) **Agent Fine-Tuning**—open-source models are fine-tuned via SFT on the refined synthetic data. The training objective is standard autoregressive cross-entropy:

$$\mathcal{L}_{\text{SFT}} = \mathbb{E}_{(\tau^{\star}, h^{\star}) \sim \mathcal{D}} \left[ -\sum_{t=1}^{T} \log p_\theta(a_t | \tau^{\star}, o_{\leq t}, a_{<t}) \right]$$

### Key Designs

1. **Categorized Exploration**:

    - Function: Systematically covers functional regions of webpages to improve task diversity.
    - Mechanism: On each page $o_t$, an LLM categorizes interactive elements by semantic role (e.g., "account management," "search filters"), and at most 2 unvisited elements per category are sampled for interaction. Per-category sampling budgets prevent any single dense region from dominating exploration.
    - Design Motivation: Random exploration (as in OS-Genesis) repeatedly visits redundant elements while missing important functional regions. Categorized exploration reframes this as a function-aware coverage problem, discovering an average of 6.0 functional categories per page.

2. **Conflict-Triggered Task Refinement**:

    - Function: Corrects hallucinations in tasks while minimizing disruption to trajectories.
    - Mechanism: A lightweight conflict predicate is defined as $\mathcal{C}(h_t, \tau_t) = \neg\textsf{ExistsUI} \vee \textsf{MissingArgs} \vee \textsf{Stall}$, detecting absent UI elements, missing arguments, and execution stalls, respectively. LLM refinement is invoked only upon conflict, following four principles: specify missing details, align with actual observations, reduce scope, and preserve category.
    - Design Motivation: Unlike Explorer's continuous per-step refinement (8.6 times/trajectory on average, 68.3% timeout rate), conflict-triggered refinement averages only 2.0 invocations/trajectory with a 6.3% timeout rate—initial tasks are sufficiently specified, and refinement focuses on *correcting* rather than *redefining* intent.

3. **Trajectory Refinement with Global Context**:

    - Function: Post-hoc removal of noisy and misaligned segments from trajectories.
    - Mechanism: The complete trajectory $h_T$ and final task $\tau^{\star}$ are reviewed offline. Four editing operations are applied: Remove(i) deletes irrelevant/redundant steps, Reorder(i,j) swaps commutable steps, Drop($h_T$) discards excessively noisy trajectories, and Keep($h_T$) retains high-quality ones. The design deliberately favors precision—uncertain swaps are rejected rather than risking causal dependency violations.
    - Design Motivation: Noise introduced by task refinement requires a post-hoc global perspective to clean up. Reorder accounts for only 4.1% of editing operations, yet reordered trajectories achieve higher preference win rates (42% vs. 27%).

### Loss & Training

Standard SFT paradigm with a history context window of 3 recent steps. Up to 500 task-trajectory pairs are synthesized per website, and a single model is trained on data mixed from five websites (learning rate 1e-5, batch size 32, 3 epochs).

## Key Experimental Results

### Main Results

**WebArena (5 websites) — Qwen2.5-VL-7B backbone**

| Method | Training Data | Shopping | CMS | Reddit | Gitlab | Maps | Overall |
|--------|--------------|---------|-----|--------|--------|------|---------|
| Base Qwen | - | 13.71 | 8.24 | 9.43 | 6.18 | 5.50 | 8.80 |
| +Self-Instruct | Synthetic | 18.18 | 8.77 | 3.85 | 12.50 | 9.38 | 11.50 |
| +OS-Genesis | Synthetic | 14.55 | 10.53 | 11.54 | 16.07 | 12.50 | 13.27 |
| +Explorer | Synthetic | 10.91 | 3.51 | 0.00 | 1.82 | 3.12 | 4.44 |
| **+SynthAgent** | **Synthetic** | **20.00** | **21.05** | **15.38** | **19.64** | **28.12** | **20.80** |

**Online-Mind2Web (136 real websites)**

| Method | GPT-4.1 Judge | GPT-5.1 Judge | WebJudge | Average |
|--------|-------------|-------------|---------|---------|
| Self-Instruct | 17.67 | 13.00 | 19.67 | 16.78 |
| OS-Genesis | 19.53 | 11.00 | 19.33 | 16.62 |
| **SynthAgent** | **31.67** | **15.67** | **23.33** | **23.56** |

### Ablation Study

| Configuration | Overall | Change |
|--------------|---------|--------|
| SynthAgent (full) | 20.80 | — |
| w/o Categorized Exploration | 17.26 | −3.54 |
| w/o Task Refinement | 15.93 | −4.87 |
| w/o Trajectory Refinement | 16.81 | −3.99 |
| w/o Dual Refinement | 15.93 | −4.87 |

### Key Findings

- Explorer performs even below the base model—continuous refinement produces excessively long, misaligned trajectories that act as negative supervision.
- Synthetic data quality: SynthAgent trajectories score 82.6, far exceeding Explorer (36.4) and OS-Genesis (52.0).
- SynthAgent achieves a trajectory completion rate of 96.5% vs. Explorer's 30.5%, at lower API cost ($0.13 vs. $0.22 per trajectory).
- Performance further improves on the stronger Qwen3 backbone (15.93→24.34), validating the model-agnostic nature of the approach.

## Highlights & Insights

- The core design insight—that task refinement and trajectory refinement are complementary—is precisely motivated: the former ensures feasibility but introduces noise, while the latter eliminates it.
- The contrast between conflict-triggered and continuous refinement reveals a key design principle: the quality of initial tasks determines the appropriate refinement strategy.
- Categorized exploration transforms random exploration into a structured coverage problem—a simple yet effective contribution.

## Limitations & Future Work

- Validation is limited to offline and restricted online environments; synthesis on real, actively changing websites remains unexplored.
- Task and trajectory synthesis rely entirely on GPT-4.1; more advanced LLMs or hyperparameter optimization are not investigated.
- Only standard SFT is employed; more advanced training paradigms such as DPO or online RL are not explored.

## Related Work & Insights

- **vs. OS-Genesis**: OS-Genesis synthesizes tasks from single-step observations, leading to hallucinations; SynthAgent addresses this via categorized exploration and conflict-triggered refinement.
- **vs. Explorer**: Explorer's continuous refinement causes intent drift and excessively long trajectories; SynthAgent's conflict-triggered refinement preserves intent consistency.
- **vs. AgentTrek**: AgentTrek relies on offline tutorials that may be outdated; SynthAgent synthesizes data by directly interacting with the environment.

## Rating

- Novelty: ⭐⭐⭐⭐ The synergistic design of dual refinement and the conflict-triggered mechanism constitute clear contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks, multiple backbones, detailed ablations, data quality analysis, and scaling experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clearly articulated with in-depth analysis of the core design tension.
- Value: ⭐⭐⭐⭐ Provides a practical, high-quality synthetic data solution for unsupervised web agent adaptation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](expseek_self-triggered_experience_seeking_for_web_agents.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_in_web_agents.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)
- [\[ICLR 2026\] Towards Scalable Oversight via Partitioned Human Supervision](../../ICLR2026/llm_agent/towards_scalable_oversight_via_partitioned_human_supervision.md)

<!-- RELATED:END -->
