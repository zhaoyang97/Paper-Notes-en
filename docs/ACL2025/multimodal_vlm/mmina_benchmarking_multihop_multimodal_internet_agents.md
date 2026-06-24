---
title: >-
  [Paper Note] MMINA: Benchmarking Multihop Multimodal Internet Agents
description: >-
  [ACL 2025][Multimodal VLM][Web Agent] This work introduces the MMInA benchmark, consisting of 1,050 human-annotated multihop multimodal web tasks across 14 real-world dynamic websites (averaging 2.85 hops). It designs a hop-by-hop evaluation protocol and a memory augmentation method, revealing a substantial performance gap on multihop web navigation between the strongest current agent (GPT-4V with a task success rate of only 21.8%) and humans (96.3%).
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Web Agent"
  - "Multihop Reasoning"
  - "Multimodal Benchmark"
  - "Web Browsing"
  - "Memory Augmentation"
date: 2026-05-08
content_hash: 604c94af25194036
---

# MMINA: Benchmarking Multihop Multimodal Internet Agents

**Conference**: ACL 2025  
**arXiv**: [2404.09992](https://arxiv.org/abs/2404.09992)  
**Authors**: Shulin Tian, Ziniu Zhang, Liangyu Chen, Ziwei Liu (NTU S-Lab)  
**Code**: [github.com/shulin16/MMInA](https://github.com/shulin16/MMInA)  
**Area**: Multimodal VLM  
**Keywords**: Web Agent, Multihop Reasoning, Multimodal Benchmark, Web Browsing, Memory Augmentation  

## TL;DR

This work introduces the MMInA benchmark, consisting of 1,050 human-annotated multihop multimodal web tasks across 14 real-world dynamic websites (averaging 2.85 hops). It designs a hop-by-hop evaluation protocol and a memory augmentation method, revealing a substantial performance gap on multihop web navigation between the strongest current agent (GPT-4V with a task success rate of only 21.8%) and humans (96.3%).

## Background & Motivation

### Background
Building embodied agents capable of autonomously navigating the internet and completing complex user tasks is a long-standing challenge in AI. Real-world web tasks are inherently compositional: users frequently need to retrieve information or execute actions across multiple websites (e.g., "booking flights $\rightarrow$ finding travel guides $\rightarrow$ renting cars $\rightarrow$ booking hotels"). This requires long-horizon planning and multimodal reasoning capabilities from the agents.

### Limitations of Prior Work
- **Single-hop Limitations**: The vast majority of tasks in existing benchmarks (MiniWoB++, WebShop, Mind2Web, WebArena, etc.) occupy only a single website, with an average hop count close to 1.0, failing to evaluate compositional reasoning across multiple sites.
- **Text-Centric Focus**: Benchmarks like WebArena and Mind2Web primarily rely on textual information (accessibility trees), overlooking the critical role of images in real-world web tasks (e.g., "purchasing a blue cotton shirt" requires visual determination of color).
- **Static Environments**: Most benchmarks use static snapshots or locally deployed websites, which cannot reflect the dynamic nature of the wild web.
- **Coarse-grained Evaluation**: Relying solely on task-level success rate evaluation often yields near-zero scores in multihop scenarios, offering few valuable analytical insights.

### Design Motivation
To bridge the triple gap of multihop, multimodal, and dynamic real-world websites, and to establish an Internet Agent evaluation framework closer to real-world scenarios.

## Method

### Key Designs 1: Benchmark Construction and Environment Design

**Environment Modeling**: Web browsing is formulated as a Partially Observable Markov Decision Process (POMDP) $\langle S, A, P, R \rangle$. At each time step, the agent receives a partial observation $o_t \in \Omega$ (consisting of the accessibility tree, page screenshots, and action history) and executes one of 12 standardized actions (click, scroll, keystroke, etc.).

**Dataset Construction**:
- 1,050 human-annotated tasks, spanning multiple domains such as shopping, travel, search, and ticket booking.
- Enpasses 14 real-world dynamic websites, including 2,989 sub-hops.
- Hop count ranges from 1 to 10, averaging 2.85 hops, with an average of 12.9 actions per task.
- Annotators adopt a "minimalist" strategy: completing the task via the shortest path with omniscient knowledge, and recording key URL nodes.

**Multimodal Design**: All tasks require processing both visual and textual information simultaneously. While automatically extracting the accessibility tree, the environment identifies and downloads images in the current viewport, with element IDs overlaid on the screenshots for the agent to reference.

### Key Designs 2: Multihop Evaluation Protocol

**Single-hop Evaluation**: Two approaches are utilized:
- `must_include`: Keyword matching, where the agent's response must contain all pre-defined keywords.
- `fuzzy_match`: Employs GPT-3.5-Turbo for semantic matching to handle synonyms (e.g., equivalence between "gold" and "yellow").

**Multihop Evaluation**: A queue containing the completion conditions for each hop (length of $N+1$ with an END marker at the end) is maintained. The agent must complete each hop sequentially; advancement to the next hop is permitted only if the current hop is correctly fulfilled. Both hop success rate (hop SR) and task success rate (task SR) are calculated to provide finer-grained performance analysis.

### Key Designs 3: Memory Augmentation Method

A three-tier memory system is proposed to empower the agent:
- **Semantic Memory**: Common-sense world knowledge encoded within the LLM weights.
- **Episodic Memory**: Temporarily stores the step-by-step action trajectories of the current task, serving as the context for the autoregressive model.
- **Procedural Memory**: Encodes complete action sequences and outcomes upon task completion, facilitating experience replay for similar future tasks.

The core idea is to leverage procedural memory replay (replaying past action trajectories) so the agent can reference historically successful trajectories when encountering similar tasks, significantly boosting single-hop and multihop performance.

## Key Experimental Results

### Main Results

| Agent | Input Type | 1-Hop Hop SR | 2-4 Hop Hop SR | 5+ Hop Hop SR | Total Hop SR | 1-Hop Task SR | 2-4 Hop Task SR | 5+ Hop Task SR | Total Task SR |
|-------|---------|-----------|-------------|-------------|---------|-----------|-------------|-------------|---------|
| GPT-4 (Text) | Tree | 14.37 | 30.56 | 5.23 | 12.26 | 14.37 | 9.09 | 0 | 9.34 |
| GPT-4 (Text+Desc) | Tree+Caption | 38.58 | 20.70 | 3.43 | 13.50 | 38.58 | 3.79 | 0 | 19.85 |
| DeepSeek-R1-32B (Text+Desc) | Tree+Caption | 47.68 | 3.84 | 4.68 | 11.11 | 47.68 | 0 | 0 | 23.07 |
| GPT-4V (Multimodal) | Tree+Image | 42.91 | 21.23 | 3.99 | 13.89 | 42.91 | 3.03 | 0 | 21.77 |
| Gemini-Pro-Vision (Multimodal+Memory) | Tree+Image+History | 39.17 | 23.93 | 4.78 | 14.27 | 39.17 | 10.61 | 1.13 | 20.13 |
| **Human Baseline** | Raw Web | **99.02** | **97.91** | **93.77** | **98.43** | **99.02** | **95.34** | **88.12** | **96.25** |

### Failure Mode Analysis by Hop Count (GPT-4V)

| Total Hops | Hop 1 SR | Hop 2 SR | Hop 3 SR | Hop 4 SR | Hop 5 SR | Hop 6 SR |
|-----------|--------|--------|--------|--------|--------|--------|
| 2-Hop Tasks | 56.50 | 11.00 | - | - | - | - |
| 3-Hop Tasks | 22.73 | 4.55 | 0.00 | - | - | - |
| 4-Hop Tasks | 12.50 | 0.00 | 0.00 | 0.00 | - | - |
| 5-Hop Tasks | 12.28 | 1.75 | 0.00 | 0.00 | 0.00 | - |
| 6-Hop Tasks | 16.67 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Key Finding: Even for Hop 1 with identical semantics, the success rate drops precipitously when embedded in tasks with larger total hop counts (from 56.5% in 2-hop tasks to 16.7% in 6-hop tasks).

### Ablation Study: Effect of Memory Augmentation

The memory augmentation method consistently brings substantial improvements to both GPT-4V and Gemini-Pro-Vision:
- GPT-4V + Memory: Total Hop SR increases from 13.89% to ~16%+, and 2-4 Hop Task SR is improved from 3.03%.
- Gemini-Pro-Vision + Memory: 2-4 Hop Task SR increases from 1.51% to 10.61% (a ~7x improvement), and 5+ Hop Task SR climbs from 0 to 1.13%.

## Key Findings

1. **Early Failure Effect**: Agents tend to fail in the early hops during multihop tasks. Furthermore, a higher total hop count correlates with a lower success rate even on the very first hop, showing that performance is not a simple linear aggregation of single-hop metrics.
2. **Search Space Explosion**: In multihop task prompts containing multiple website URLs, once the agent fails, it tends to jump to another website instead of retrying the current one, leading to excessive and unfocused exploration.
3. **Misidentification of Termination Conditions**: Agents often fail to recognize the termination criteria of a single hop, lingering within completed hops instead of advancing to the next.
4. **Multimodal Advantage**: Multimodal models consistently outperform text-only models, demonstrating that visual cues are indispensable for the accurate execution of web-based tasks.
5. **The Reasoning Model Paradox**: DeepSeek-R1 achieves the top performance on single-hop tasks (47.68%), but suffers a drastic degradation on multihop tasks (only 3.84% on 2-4 hops), exposing the vulnerability of current reasoning models in maintaining long context.

## Highlights & Insights

- **Real-world Dynamic Environments**: The only benchmark designed to run on live, continuously changing websites, ensuring a high degree of fidelity.
- **Realistic Multihop Design**: With tasks scaling up to 10 hops (average 2.85 hops), this benchmark significantly exceeds existing suites (mostly averaging ~1 hop) to rigorously evaluate compositional reasoning.
- **Hop-by-hop Evaluation Protocol**: Circumventing the limitations of binary task-level (all-or-nothing) evaluation, this protocol provides step-by-step procedural insights (such as unveiling the "early failure effect").
- **Crucial Discoveries**: Reveals systematic failure modes of agents in multihop environments, indicating that the core barrier is not necessarily direct task execution capacity, but rather the deficiency in planning and memory mechanisms.
- **Highly Generalizable Memory Augmentation**: A model-agnostic, lightweight method that can be seamlessly applied to any LMM.

## Limitations & Future Work

- **Limited Website Coverage**: Evaluates only 14 websites. Due to anti-scraping mechanisms, some websites necessitate offline/open-source fallback versions, partially compromising the claim of a live environment.
- **URL-Matching-Based Evaluation**: Relying on matching the correct sequence of URLs as the success criterion might ignore cases where the agent successfully reaches the solution via an alternative valid path.
- **Reproducibility Challenges**: Because real-world web content changes continuously, evaluation results obtained at different times are difficult to compare stringently.
- **Insufficient Memory Framework Evaluation**: Ablation experiments and detailed analysis on the memory augmentation method remain relatively limited, leaving the impacts of different memory lengths largely unexplored.
- **Lack of In-depth Evaluation on Open-source Models**: Relies primarily on commercial API models (such as GPT-4V and Gemini), leaving the evaluation of open-source LMMs less comprehensive.
- **Task Distribution Biases**: Highly skewed toward shopping and search domains, potentially underrepresenting other vital web interaction scenarios.

## Related Work & Insights

- **WebArena / VisualWebArena**: Employs locally deployed static websites with a maximum of 2 hops (averaging ~1 hop), whereas MMInA tests on live dynamic websites up to 10 hops.
- **WebVoyager**: Also utilizes wild websites but scales up to only 4 hops (average 2.4 hops) and lacks a hop-by-hop evaluation design.
- **Mind2Web**: Covers 131 websites but uses static snapshots, text-only inputs, single-hop tasks, and operates with multiple-choice instead of open-ended actions.
- **GAIA / OpenAGI**: Serves as general multimodal agent benchmarks, but does not specialize in the web-browsing environment.
- **CogAgent / SeeAct**: Specifically trained web agents rather than pure benchmarks, which perform poorly on MMInA (e.g., CogAgent achieves only 3.35% Task SR).

## Rating

- Novelty: ⭐⭐⭐⭐ — First web agent benchmark focusing on the combination of multihop, multimodal, and live dynamic websites.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Compares multiple model classes and human baselines, providing profound hop-by-hop analysis, though experiments on memory strategies could be expanded.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, fully motivated, and enriched with comprehensive charts.
- Value: ⭐⭐⭐⭐ — Exposes key roadblocks in multihop web tasks, pointing out future avenues for web agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension](punchbench_mllm_punchline.md)
- [\[ACL 2025\] Attacking Vision-Language Computer Agents via Pop-ups](attacking_vl_agents_popups.md)
- [\[ACL 2025\] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)

</div>

<!-- RELATED:END -->
