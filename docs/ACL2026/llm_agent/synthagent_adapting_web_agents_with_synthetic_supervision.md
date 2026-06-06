---
title: >-
  [Paper Note] SynthAgent: Adapting Web Agents with Synthetic Supervision
description: >-
  [ACL 2026][LLM Agent][Synthetic Data] This paper proposes SynthAgent, a Web Agent adaptation framework based entirely on synthetic supervision. It systematically covers webpage functional areas via categorized exploratio…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Synthetic Data"
  - "Web Agent"
  - "Dual Refinement"
  - "Categorized Exploration"
  - "Trajectory Quality"
date: 2026-05-08
content_hash: 13ff1df7c285cc6d
---

# SynthAgent: Adapting Web Agents with Synthetic Supervision

**Conference**: ACL 2026  
**arXiv**: [2511.06101](https://arxiv.org/abs/2511.06101)  
**Code**: [GitHub](https://github.com/aiming-lab/SynthAgent)  
**Area**: LLM Agent / Web Agent Adaptation  
**Keywords**: Synthetic Data, Web Agent, Dual Refinement, Categorized Exploration, Trajectory Quality

## TL;DR

This paper proposes SynthAgent, a Web Agent adaptation framework based entirely on synthetic supervision. It systematically covers webpage functional areas via categorized exploration to synthesize diverse tasks. A dual refinement strategy—comprising task refinement (triggered by conflict detection to correct hallucinations) and trajectory refinement (post-hoc denoising via a global perspective)—improves synthetic data quality. SynthAgent significantly outperforms existing synthesis methods on WebArena and Online-Mind2Web.

## Background & Motivation

**Background**: LLM-driven Web Agents demonstrate strong interaction capabilities on standardized benchmarks but suffer sharp performance declines when deployed to new websites not seen during training. Adapting to new environments requires environment-specific tasks and demonstration data, yet manual annotation is costly and unscalable.

**Limitations of Prior Work**: (1) Self-Instruct lets LLMs "imagine" tasks without environment grounding, leading to simple and repetitive tasks; (2) OS-Genesis synthesizes tasks backward from single-step observations, where insufficient context leads to frequent hallucinations (referencing non-existent elements or states); (3) Explorer continuously refines tasks during execution, but frequent intent shifts (averaging 8.6 times) cause 68.3% of trajectories to exceed step budgets.

**Key Challenge**: Task synthesis requires environment grounding to avoid hallucinations, but over-grounding tasks during execution introduces trajectory noise—a fundamental design tension in synthetic supervision.

**Goal**: Design a purely synthetic supervision framework to efficiently adapt Web Agents to new environments without human intervention or test-set leakage.

**Key Insight**: Decouple task refinement and trajectory refinement into two synergistic stages: task refinement ensures feasibility while potentially introducing noise, which trajectory refinement subsequently eliminates.

**Core Idea**: Dual refinement—refining tasks only when explicit conflicts are detected during execution (conflict-triggered, rather than continuous), and using global context to refine trajectories post-execution to guarantee both task feasibility and trajectory quality.

## Method

### Overall Architecture

SynthAgent consists of four stages: (1) **Categorized Exploration Task Synthesis**: Grouping webpage elements by function and uniformly sampling interaction triples $(o_t, a_t, o_{t+1})$ for the LLM to propose multi-step tasks based on real interface transitions; (2) **Conflict-Triggered Task Refinement**: Detecting conflicts between tasks and observations during trajectory collection, refining tasks only when triggered (averaging only 2.0 times); (3) **Global Trajectory Refinement**: Using the complete trajectory and final task $\tau^{\star}$ post-hoc to remove noise and misaligned actions; (4) **Agent Fine-tuning**: Performing SFT on open-source models using the refined synthetic data. The training loss follows the standard autoregressive cross-entropy:

$$
\mathcal{L}_{\text{SFT}} = \mathbb{E}_{(\tau^{\star}, h^{\star}) \sim \mathcal{D}} \left[ -\sum_{t=1}^{T} \log p_\theta(a_t | \tau^{\star}, o_{\leq t}, a_{<t}) \right]
$$

### Key Designs

1.  **Categorized Exploration**:
    *   **Function**: Systematically covers functional areas of webpages to enhance task diversity.
    *   **Mechanism**: On each page $o_t$, the LLM categorizes interactive elements by semantic roles (e.g., "Account Management", "Search Filter"). Up to 2 unvisited elements are sampled per category for interaction. Sampling budgets per category prevent dense regions from dominating exploration.
    *   **Design Motivation**: Random exploration (OS-Genesis) often visits redundant elements repeatedly while missing critical functional areas; categorized exploration treats this as a function-aware coverage problem, discovering an average of 6.0 functional categories per page.

2.  **Conflict-Triggered Task Refinement**:
    *   **Function**: Corrects hallucinations in tasks while minimizing trajectory disruption.
    *   **Mechanism**: Defines a lightweight conflict predicate $\mathcal{C}(h_t, \tau_t) = \neg\textsf{ExistsUI} \vee \textsf{MissingArgs} \vee \textsf{Stall}$, detecting missing UI elements, missing parameters, and execution stalls respectively. LLM refinement is called only upon conflict, following four principles: specifying missing details, aligning with actual observations, reducing scope, and maintaining category.
    *   **Design Motivation**: Unlike Explorer’s continuous per-step refinement (8.6 times average, 68.3% timeouts), conflict-triggered refinement averages 2.0 times with only a 6.3% timeout rate—the initial task is sufficiently specified, and refinement focuses on "correction" rather than "redefinition".

3.  **Trajectory Refinement with Global Context**:
    *   **Function**: Removes noise and misaligned segments from trajectories post-hoc.
    *   **Mechanism**: Off-line review of the full trajectory $h_T$ and final task $\tau^{\star}$ using four editing operations: Remove(i) for irrelevant/redundant steps, Reorder(i,j) for swappable steps, Drop($h_T$) for overly noisy trajectories, and Keep($h_T$) for good trajectories. The design is intentionally biased toward precision—uncertain swaps are rejected rather than risking causal dependency breaks.
    *   **Design Motivation**: Noise introduced during task refinement requires a post-hoc global view for cleaning; while Reorder accounts for only 4.1% of edits, reordered trajectories achieve higher preference win rates (42% vs 27%).

### Loss & Training

Standard SFT paradigm with a historical context window of the 3 most recent steps. Synthesizes up to 500 task-trajectory pairs per website, training a single model on a mix of five websites (learning rate 1e-5, batch size 32, 3 epochs).

## Key Experimental Results

### Main Results

**WebArena (5 Websites) - Qwen2.5-VL-7B Backbone**

| Method | Training Data | Shopping | CMS | Reddit | Gitlab | Maps | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Base Qwen | - | 13.71 | 8.24 | 9.43 | 6.18 | 5.50 | 8.80 |
| +Self-Instruct | Synthetic | 18.18 | 8.77 | 3.85 | 12.50 | 9.38 | 11.50 |
| +OS-Genesis | Synthetic | 14.55 | 10.53 | 11.54 | 16.07 | 12.50 | 13.27 |
| +Explorer | Synthetic | 10.91 | 3.51 | 0.00 | 1.82 | 3.12 | 4.44 |
| **+SynthAgent** | **Synthetic** | **20.00** | **21.05** | **15.38** | **19.64** | **28.12** | **20.80** |

**Online-Mind2Web (136 Real Websites)**

| Method | GPT-4.0 Judge | GPT-4.5 Judge | WebJudge | Average |
| :--- | :--- | :--- | :--- | :--- |
| Self-Instruct | 17.67 | 13.00 | 19.67 | 16.78 |
| OS-Genesis | 19.53 | 11.00 | 19.33 | 16.62 |
| **SynthAgent** | **31.67** | **15.67** | **23.33** | **23.56** |

### Ablation Study

| Configuration | Overall | Gain |
| :--- | :--- | :--- |
| SynthAgent (Full) | 20.80 | - |
| w/o Categorized Exploration | 17.26 | -3.54 |
| w/o Task Refinement | 15.93 | -4.87 |
| w/o Trajectory Refinement | 16.81 | -3.99 |
| w/o Dual Refinement | 15.93 | -4.87 |

### Key Findings

*   Explorer performs worse than the base model—continuous refinement yields overly long, misaligned "negative supervision" trajectories.
*   Synthetic Data Quality: SynthAgent trajectory quality (82.6) far exceeds Explorer (36.4) and OS-Genesis (52.0).
*   SynthAgent achieves a 96.5% trajectory completion rate vs. Explorer's 30.5%, with lower API costs ($0.13 vs $0.22 per trajectory).
*   Gains persist on stronger backbones like Qwen2-7B (15.93→24.34), validating the model-agnostic nature of the method.

## Highlights & Insights

*   The design insight "task refinement and trajectory refinement are synergistic" is precise—the former ensures feasibility but adds noise, while the latter removes it.
*   The comparison between conflict-triggered and continuous refinement reveals a key design principle: the strategy for refinement depends on the quality of the initial task.
*   Categorized exploration converts random exploration into structured coverage, which is simple yet highly effective.

## Limitations & Future Work

*   Validation was restricted to offline and limited online environments; synthesis on real-world active websites remains unexplored.
*   Task and trajectory synthesis depend entirely on GPT-4; the use of more advanced LLMs or parameter optimization has not been explored.
*   Only standard SFT was used; advanced methods like DPO or online RL haven't been integrated.

## Related Work & Insights

*   **vs OS-Genesis**: OS-Genesis synthesizes tasks from single-step observations leading to hallucinations; SynthAgent resolves this via Categorized Exploration and Conflict-Triggered Refinement.
*   **vs Explorer**: Explorer's continuous refinement causes intent drift and excessive length; SynthAgent's conflict-triggered approach maintains intent consistency.
*   **vs AgentTrek**: AgentTrek relies on potentially outdated offline tutorials; SynthAgent synthesizes directly through environment interaction.

## Rating

*   Novelty: ⭐⭐⭐⭐ The synergistic design of dual refinement and the conflict-triggered mechanism are clear innovations.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks, multiple backbones, detailed ablations, data quality analysis, and scaling experiments.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear problem motivation and deep analysis of design tensions.
*   Value: ⭐⭐⭐⭐ Provides a practical, high-quality synthetic data solution for unsupervised Web Agent adaptation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](expseek_self-triggered_experience_seeking_for_web_agents.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] Towards Scalable Oversight via Partitioned Human Supervision](../../ICLR2026/llm_agent/towards_scalable_oversight_via_partitioned_human_supervision.md)
- [\[ACL 2026\] Why LLM Web Agents Fail: A Hierarchical Planning Perspective](why_do_llm-based_web_agents_fail_a_hierarchical_planning_perspective.md)

</div>

<!-- RELATED:END -->
