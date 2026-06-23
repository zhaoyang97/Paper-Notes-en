---
title: >-
  [Paper Note] CoNavBench: Collaborative Long-Horizon Vision-Language Navigation Benchmark
description: >-
  [ICLR 2026][Robotics & Embodied AI][Paper Note] CoNavBench is the first Vision-Language Navigation (VLN) benchmark for "multi-robot collaboration," containing 4,048 single-agent/collaborative tasks. It includes NavCraft, an automated data generation platform (two-stage agent + scene graph + efficiency toolbox). Using a fine-tuned Qwen2.5-VL-3B as a reference policy,
tags:
  - ICLR 2026
  - Robotics & Embodied AI
date: 2026-05-08
content_hash: 69dfb22ad7aabe69
---
# CoNavBench: Collaborative Long-Horizon Vision-Language Navigation Benchmark

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bMrH2PFMsi](https://openreview.net/forum?id=bMrH2PFMsi)  
**Code**: To be confirmed  
**Area**: Robotics / Embodied Navigation / VLN Benchmark  
**Keywords**: Collaborative Navigation, Vision-Language Navigation, Long-Horizon Tasks, Multi-Robot, Data Generation Platform

## TL;DR
CoNavBench is the first Vision-Language Navigation (VLN) benchmark for "multi-robot collaboration," containing 4,048 single-agent/collaborative tasks. It includes NavCraft, an automated data generation platform (two-stage agent + scene graph + efficiency toolbox). Using a fine-tuned Qwen2.5-VL-3B as a reference policy, it demonstrates that collaborative decomposition improves step-level task success rates by a relative 18.11%.

## Background & Motivation
**Background**: The mainstream of VLN research follows a "single-agent, step-by-step instruction execution" paradigm—ranging from point-to-point following in R2R and continuous control in VLN-CE to long-horizon, multi-stage, multi-turn tasks in LHPR-VLN/GOAT-Bench. While difficulty has increased, the participant remains a **single** robot.

**Limitations of Prior Work**: Real-world deployments (domestic services, warehousing, parallel workflows) often have multiple robots available, yet existing datasets and evaluation protocols assume a single agent. This leads to two consequences: first, it **hides collaboration opportunities**—tasks that could be parallelized are forced into serial execution, causing high latency; second, it **ignores inter-robot interference**—challenges like congestion, handoff errors, and rendezvous timing only occur in multi-robot settings and are never encountered in single-agent formalizations.

**Key Challenge**: The value of collaborative VLN (shorter total完工时间 makespan, robustness from parallelism and role division) is inherently tied to its new challenges (congestion, handoff, rendezvous timing). Existing benchmarks provide neither tasks "born for collaboration" nor annotations to quantify interference, making systematic evaluation impossible.

**Goal**: The authors decompose collaborative VLN into three progressive challenges: (i) **Collaborative Task Synthesis**—constructing long-horizon single-agent base tasks with clear stage boundaries and cross-room dependencies that offer genuine collaborative gains; (ii) **Conflict-free Team Scheduling**—lifting single-agent tasks into multi-robot plans with role assignments, timing, and rendezvous points; (iii) **In-the-loop Efficiency Optimization**—given a feasible plan, estimating team-level time, predicting bottlenecks, and providing executable repair suggestions.

**Key Insight**: Instead of using LLM agents as "navigators" (grounding observations and outputting sub-goals), they are treated as **data generation and scheduling engines**. A hierarchical agent synthesizes tasks on semantic scene graphs, assigns roles, and produces team-aware schedules. The scene graph enables reachability checks, travel time estimation, and interference assessment in the loop, avoiding the need for full physical simulation at every step.

**Core Idea**: Use "scene-graph-grounded two-stage agents (NavCraft-S for base tasks → NavCraft-C for collaborative lifting) + graph-level efficiency toolbox" to automatically and extensibly generate collaborative VLN tasks for CoNavBench, verifying that collaborative decomposition indeed shortens makespan and improves reliability.

## Method

### Overall Architecture
CoNavBench contributes two outputs: **a benchmark** (4,048 single/collaborative tasks + collaboration taxonomy + graph-level annotations) and **NavCraft**, the platform that generates it. NavCraft receives a Habitat-Sim connectivity graph, semanticizes it (labeling nodes with room types), and uses two-stage agents to grow tasks from "single-agent to collaborative" on this graph, with every step validated by a graph-level efficiency toolbox.

Specifically, NavCraft consists of four stages: ① **Scene Graph Generation**—labeling Habitat connectivity graph nodes $G=(V,E)$ with room types through a four-stage refinement process; ② **NavCraft-S**—sampling "start $s \to$ target object $t \to$ end $e$" triplets on the semantic graph with cross-room feasibility checks to synthesize long-horizon base tasks; ③ **NavCraft-C**—determining if collaboration is beneficial; if so, lifting the single-agent plan to a dual-robot schedule with a fixed handoff type; ④ **Efficiency Toolbox**—integrated throughout, translating linguistic intent into numerical constraints on distance/channel width/occlusion, validating reachability, estimating time, and providing suggestions in a closed loop. Finally, a collaborative reference policy is fine-tuned from Qwen2.5-VL-3B and evaluated in Habitat3 continuous simulation.

### Key Designs

**1. Four-stage Refinement Semantic Scene Graph: Converting Geometry to a "Room Map"**

Collaborative scheduling requires knowing the "room type" of each location. The authors use a four-stage process: First, **Instance Proximity Voting (IPV)**: for node $i$, find $k$ nearest labeled objects on the floor plane and take the majority room class: $\hat r^{(0)}_i = \arg\max_c \sum_{m\in N_k(i)} \mathbb{1}[r(m)=c]$. Second, **Neighborhood Consistency (NC)**: nodes in narrow passages are corrected only if their degree is moderate ($2\le \deg(i)\le 4$) and their label differs from **all** neighbors. Third, **Connectivity Repair**: keeping only the largest connected component per class and merging isolated fragments into the nearest neighbors. Fourth, **Graph Context Labeling**: fuzzy regions are labeled by a lightweight model (GPT-4o mini) using neighbor histograms and top-5 objects.

**2. NavCraft-S: Synthesizing "Collaboration-Ready" Long-Horizon Base Tasks**

NavCraft-S samples feasible triplets $(s,t,e)$ on the room graph. It performs **persona-conditioned sampling** using a lightweight persona $\pi$ to increase diversity in object/endpoint choices. It then performs **feasibility validation**: checking both segments $s\to t$ and $t\to e$ on the regional graph. Defining $L(u,v)$ as the hop distance on the region graph with a threshold $\tau \ge 1$, a segment is valid if $\text{leg\_ok}(u,v):=\text{conn}(u,v)\wedge L(u,v)\ge\max\{2,\tau\}$. This ensures generated tasks truly cross rooms and provide space for decomposition.

**3. NavCraft-C: Collaborative Lifting with a "Strict Reduction in Main Robot Path" Threshold**

NavCraft-C decides whether to introduce a second robot using two handoff schemas: Type A1 (collaborator picks at $t$, hands off at $x$, main robot delivers from $x$ to $e$) and Type A2 (main robot picks at $t$, hands off at $x$, collaborator delivers to $e$). Using an **augmented metric graph** $G^+$ with Euclidean weights, the **collaboration acceptance criterion** is: $C_{\text{solo}}=d(s,o)+d(o,a_e)$ for single-agent vs. team loads $J^{A1}_{r1}=d(s,a_x)+d(a_x,a_e)$ or $J^{A2}_{r1}=d(s,t)+d(t,a_x)$. Collaboration is accepted only if $\min\{J^{A1}_{r1},J^{A2}_{r1}\}<C_{\text{solo}}$, strictly reducing the main robot's workload.

**4. Efficiency Toolbox + Reference Policy: Closed-loop Validation and Evaluation**

The toolbox translates linguistic intent into numerical constraints to estimate travel time and assess interference. For the benchmark, a **reference policy stack** is provided using Qwen2.5-VL-3B (vision features from frozen EVA-CLIP-02-LARGE, non-vision modules full-parameter fine-tuned), demonstrating that policies trained on CoNavBench shorten makespan and improve reliability.

### Mechanism Example
Consider "Find a cup in the bedroom and bring it to the living room." NavCraft-S samples $s$=study, $t$=bedroom, $e$=living room. It validates connectivity and hop counts. NavCraft-C then calculates $C_{\text{solo}}$ (study → bedroom → living room). It identifies a hallway midpoint $x$ where Type A1 (collaborator picks cup in bedroom → hands off in hallway → main robot delivers to living room) results in a shorter path for the main robot. The efficiency toolbox verifies reachability and timing, producing a "dual-robot relay" episode with graph-level annotations.

## Key Experimental Results

### Main Results
CoNavBench contains **4,048** episodes. Average collaboration gain is **21.08%**. Evaluations are conducted in Habitat3 with Fetch and Spot robots across 216 HM3D scenes.

Single-agent Tasks (High-level / Step-by-step):

| Method | Type | SR↑ | SPL↑ | ISR↑ | CSR↑ | NE↓ |
|------|------|------|------|------|------|------|
| Random | - | 0.00/1.26 | 0.00/1.26 | 1.61/1.26 | 1.21/1.26 | 7.25/7.56 |
| Qwen2.5-VL-3B | Zero-shot | 4.30/10.41 | 0.98/2.55 | 14.25/10.41 | 16.10/10.41 | 6.91/7.80 |
| Qwen2.5-VL-3B | Finetuned | 12.90/29.65 | 6.08/13.81 | 23.92/29.65 | 26.22/29.65 | 6.40/6.74 |

Collaborative Tasks (High-level / Step-by-step):

| Method | Type | SR↑ | SPL↑ | NE↓ |
|------|------|------|------|------|
| Qwen2.5-VL-3B | Finetuned | 11.11/**35.02** | 4.82/16.88 | 6.55/5.79 |
| Qwen2.5-VL-7B | Finetuned | 11.65/**29.78** | 6.24/16.56 | 6.74/6.20 |

Key comparison: Under the step-by-step protocol, fine-tuned 3B SR rises from 29.65% (single) to 35.02% (collaborative), a relative **Gain** of 18.11%.

### Ablation Study: Data Generation Agent Selection

| Agent | Single-gen SR↑ | Avg. Gain↑ | Cost ($)↓ |
|--------|-------------|---------------|----------|
| GPT-4o | 77% | 21.07% | 5.242 |
| **GPT-4o-mini** | 64% | 25.12% | **0.360** |

### Key Findings
- **Collaboration decomposition yields stable gains in step-level protocols**: Breaking long-horizon instructions into single-stage sub-tasks shortens decision horizons and improves local capability.
- **7B not outperforming 3B is due to under-training**: At current data budgets, the 7B model didn't surpass 3B, but still showed internal gains in collaborative settings (22.40% → 29.78%).
- **GPT-4o-mini offers the best quality/cost balance**: It was selected as the final generation engine for large-scale synthesis.

## Highlights & Insights
- **Repurposing LLM agents as data generation/scheduling engines**: Moving away from the "LLM as navigator" trend to the "agent for data" paradigm to bypass data scarcity in collaborative VLN.
- **Hard threshold for collaboration**: The criterion $\min\{J^{A1}_{r1},J^{A2}_{r1}\}<C_{\text{solo}}$ ensures generated tasks have actual efficiency gains, avoiding "collaboration for the sake of collaboration."
- **Scene graphs over full physics rollouts**: Approximating expensive embodied validation via graph-based distance and interference checks allows for scalable data generation.

## Limitations & Future Work
- **Dependency on GPT API**: Introduces style bias and sensitivity to backend updates. Future work includes distilling an open-source model.
- **Limited to two-robot relay**: NavCraft currently only supports dual-robot scenarios in indoor HM3D environments; more complex multi-robot parallel patterns are not yet covered.
- **Low absolute performance**: Low SR in high-level tasks indicates substantial room for improvement in long-horizon reasoning.

## Related Work & Insights
- **vs. Single-agent Long-horizon VLN**: While LHPR-VLN/GOAT-Bench increase task length, CoNavBench is the first to introduce **multi-robot primitives** (handoff, rendezvous, interference).
- **vs. Step-by-step/Continuous VLN**: CoNavBench supports both protocols and introduces team-level metrics like makespan and interference time.
- **vs. LLM-based Navigators**: Unlike MapGPT/NavGPT which use LLMs for grounding, NavCraft uses them for **extensible context-aware task synthesis**.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First collaborative VLN benchmark + agent-driven data engine)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive protocols, but reference performance is low)
- Writing Quality: ⭐⭐⭐⭐ (Pipeline is clear; some metric naming is slightly ambiguous)
- Value: ⭐⭐⭐⭐⭐ (Provides an extensible platform for the collaborative embodied AI community)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[CVPR 2025\] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method](../../CVPR2025/robotics/towards_long-horizon_vision-language_navigation_platform_benchmark_and_method.md)
- [\[ICLR 2026\] Uncertainty-Aware Gaussian Map for Vision-Language Navigation](uncertainty-aware_gaussian_map_for_vision-language_navigation.md)
- [\[ICLR 2026\] HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model](hybridvla_collaborative_diffusion_and_autoregression_in_a_unified_vision-languag.md)
- [\[ICLR 2026\] Compositional Diffusion with Guided Search for Long-Horizon Planning](compositional_diffusion_with_guided_search_for_long-horizon_planning.md)
- [\[ICLR 2026\] SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation](sarm_stage-aware_reward_modeling_for_long_horizon_robot_manipulation.md)

</div>

<!-- RELATED:END -->
