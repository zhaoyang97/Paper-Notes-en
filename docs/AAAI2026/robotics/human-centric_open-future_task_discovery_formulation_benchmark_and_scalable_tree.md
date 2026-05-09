---
title: >-
  [Paper Note] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search
description: >-
  [AAAI 2026][Robotics][Task Discovery] This paper formalizes the **Human-Centric Open-Future Task Discovery (HOTD)** problem—identifying tasks that reduce human burden across multiple possible futures in scenarios where human intentions are concurrent and dynamically evolving. The authors construct the HOTD-Bench benchmark (2K+ real-world videos) and propose **CMAST** (Collaborative Multi-Agent Search Tree), which substantially outperforms existing LMM baselines via a multi-agent system and a scalable search tree.
tags:
  - AAAI 2026
  - Robotics
  - Task Discovery
  - Open Future
  - Multi-Agent
  - Search Tree
  - Embodied AI
date: 2026-05-08
content_hash: d75a6a6eb62ea53b
---

# Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search

**Conference**: AAAI 2026
**arXiv**: [2511.18929](https://arxiv.org/abs/2511.18929)
**Code**: None
**Area**: Robotics
**Keywords**: Task Discovery, Open Future, Multi-Agent, Search Tree, Embodied AI

## TL;DR

This paper formalizes the **Human-Centric Open-Future Task Discovery (HOTD)** problem—identifying tasks that reduce human burden across multiple possible futures in scenarios where human intentions are concurrent and dynamically evolving. The authors construct the HOTD-Bench benchmark (2K+ real-world videos) and propose **CMAST** (Collaborative Multi-Agent Search Tree), which substantially outperforms existing LMM baselines via a multi-agent system and a scalable search tree.

## Background & Motivation

### From Known Goals to Open Futures

Existing work on Autonomous Skill Acquisition focuses on enabling robots to propose manipulation tasks from current observations—but **assumes fixed goals or closed environments**. Real human scenarios are far more complex:

- People frequently **engage in multiple concurrent sub-processes** (e.g., interleaving cleaning while cooking)
- Intentions **change dynamically**, and rarely are all steps made explicit
- When a person is doing housework, they may later cook, or rest—what should the robot prepare in advance?

**Core Insight**: If a robot's proposed task (e.g., "wipe the table") is helpful across all possible future branches—whether the human later cooks, cleans, or rests—then it constitutes a high-quality task discovery.

### HOTD vs. Traditional Task Discovery

| Dimension | Traditional Task Discovery | HOTD |
|-----------|--------------------------|------|
| Goal | Find the next step toward a known outcome | Identify universally helpful actions under uncertain, multiple futures |
| Future | Deterministic or limited branching | Exponentially growing branches due to concurrent human behavior |
| Core Challenge | Planning efficiency | Robustness of predicted value under uncertainty |

## Method

### Overall Architecture

The CMAST framework consists of two core components:

1. **Search Tree Module**: Explicitly models the action space over open futures, supporting scalable test-time reasoning (analogous to OpenAI-O3 and DeepSeek-R1)
2. **Collaborative Multi-Agent System**: Seven specialized LMM/LLM agents, each responsible for a distinct stage of reasoning

### Key Designs

#### 1. **Problem Formulation**

Given an input video segment $I_{0:t_0}$, the model generates a predicted task set $\hat{Q}_I = \{{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_i}\}$.

Dual optimization objectives:
$$\max_G |\hat{Q}_I \cap Q_I^{hc}| \quad \text{(discover as many human-centric tasks as possible)}$$
$$\max_G \frac{|\hat{Q}_I \cap Q_I^{hc}|}{|\hat{Q}_I|} \quad \text{(maximize the proportion of discovered tasks that are helpful)}$$

**Definition of Human-Centric Tasks**: A task $y$ is human-centric if and only if its completion reduces the total cost for the human to achieve their goal:
$$y \in Q_I^{hc} \iff y \in Q \land \mathcal{L}(A'_z, z) < \mathcal{L}(A_z, z)$$

where $A_z$ is the original action sequence, $A'_z$ is the human's adjusted sequence after the robot executes task $y$, and $\mathcal{L}$ is a cost function (time, physical effort, etc.).

#### 2. **Search Tree Module**

The search tree $T = (V, E)$, where each node is an action and edges represent temporal relations:

- **History portion** (first $N$ layers): A linear chain determined by actions already observed in the video
- **Future portion** (beyond layer $N$): Begins branching, where each branch represents a possible next step; leaf nodes indicate activity completion

Search strategy: Pruned exhaustive search with a probability threshold of 0.5. Greedy search (beam=1) and beam search with varying beam sizes are also explored.

**Scalability**: The search tree naturally supports more computation time → more branch expansions → more comprehensive task discovery, consistent with the test-time scaling philosophy of O3/R1.

#### 3. **Seven Collaborative Agents**

| Agent | Role | Type | Input | Output |
|-------|------|------|-------|--------|
| Scene Description Agent | Understand video scene | LMM | Video | Scene description $s$ |
| History Action Recognition | Recognize historical actions | LMM | Video + scene | Initial search tree (linear chain) |
| Next Action Prediction | Predict next actions | LMM | Action path + video + scene | Set of child nodes |
| Likelihood Estimation | Estimate child node probabilities | LLM | Action path + candidates | Probability distribution (for ranking and pruning) |
| Redundancy Removing | Remove redundant branches | LLM | Expanded subtree | Pruned subtree |
| Dependency Recognition | Identify prerequisite dependencies | LLM | All paths | Actions with preconditions filtered out |
| Task Converting | Convert actions to task descriptions | LLM | Independent action set | Robot-perspective task description set |

The three core expansion agents (prediction, estimation, pruning) **operate iteratively** until all unexpanded nodes are leaf nodes or the maximum tree height is reached.

#### 4. **Simulation-Based Evaluation**

Since exhaustively annotating all helpful tasks is infeasible (due to exponential future branching), the paper proposes using an LLM simulator as an evaluation tool:

- Given a robot-discovered task $\hat{y}_n$ + historical actions + goal $z$
- The simulator derives the human's adjusted future trajectory $A'_z$
- It estimates cost $\mathcal{L}(A'_z, z)$ and compares it against the original cost $\mathcal{L}(A_z, z)$

**Advantage**: Any hypothetical future can be evaluated—including scenarios not actually realized in the dataset. Human evaluation confirms high alignment between the simulator and human preferences.

### Implementation Details

The framework is entirely training-free. LMM agents use LLaVA-Next-Video; LLM agents use Qwen-LM.

## Key Experimental Results

### Main Results: HOTD-Bench Simulation Evaluation

| Method | TSU vc@40 | TSU vr@40 | CHA vc@20 | CHA vr@20 |
|--------|----------|----------|----------|----------|
| Qwen2VL-7B | 2.71 | 44.2% | 2.06 | 43.1% |
| Qwen2.5VL-72B | 2.47 | 47.6% | 3.01 | 40.9% |
| InternVL2-8B | 2.51 | 61.0% | 2.47 | 54.5% |
| LLaVA-NV-7B | 3.34 | 50.2% | 6.20 | 54.1% |
| LLaVA-NV-34B | 3.39 | 44.2% | 3.55 | 40.8% |
| **CMAST (Ours)** | **3.83** | **71.9%** | **2.73** | **55.5%** |

- Valid Task Ratio: CMAST surpasses the second-best method on TSU by 15–22 percentage points
- Valid Task Count: CMAST's mean on TSU is 7.6% higher than the second-best
- Larger models (e.g., 72B Qwen2.5VL) do not consistently outperform smaller ones—demonstrating that parameter scaling does not equate to improved task discovery capability

### Ablation Study

**Search Tree Module Ablation**:

| Configuration | Valid Task Ratio | Change |
|---------------|-----------------|--------|
| CMAST w/o tree | ~35% | −37% |
| **CMAST (full)** | **~72%** | — |

Removing the search tree causes a 37-point drop in valid task ratio, confirming it as the core component.

**Search Strategy Ablation**:

| Strategy | Helpful Tasks Discovered | Valid Task Ratio | Efficiency (expansions) |
|----------|------------------------|-----------------|------------------------|
| Greedy (beam=1) | ~1.4 | ~72% | Fewest |
| Beam=2 | ~2.5 | ~72% | Moderate |
| Pruned exhaustive (0.5) | ~3.8 | ~72% | Most |

Valid task ratio remains consistently stable (~72%), while the number of helpful tasks discovered increases with computation—validating test-time scaling.

**Integration with Different LMMs**:

| Variant | Valid Task Ratio Gain |
|---------|----------------------|
| CMAST-LLaVA vs. vanilla LLaVA | +39%+ |
| CMAST-InternVL2 vs. vanilla InternVL2 | +39%+ |
| CMAST-Qwen2 vs. vanilla Qwen2 | +39%+ |

Regardless of the underlying LMM, the CMAST framework consistently delivers at least 39% improvement in valid task ratio.

### Key Findings

1. **Existing LMMs show limited performance on HOTD**: Even the strongest baseline achieves only ~60% valid task ratio—conversational instruction-tuning data is insufficient for capturing human behavioral anticipation
2. **The search tree is critical**: It provides an explicit, structured procedural space that enables comprehensive exploration of diverse action sequences
3. **Test-time scaling is effective**: More computation leads to more helpful tasks being discovered without degrading precision
4. **CMAST approaches human-level performance**: On 10 randomly sampled cases, CMAST performs comparably to human annotators
5. **The LLM simulator is reliable**: Human evaluators largely agree with the simulator's assessments of task helpfulness

## Highlights & Insights

- **The problem formulation itself is a core contribution**: The HOTD formalization—particularly defining "human-centric" via cost reduction—is both rigorous and practically grounded
- **The simulation-based evaluation design is elegant**: It avoids exponential annotation costs while enabling evaluation of hypothetical scenarios, constituting a reusable methodological contribution
- **The multi-agent decomposition is principled**: Each agent corresponds to a distinct phase of search tree operation (initialization / expansion / pruning / post-processing), yielding natural decoupling
- **Connection to test-time compute scaling**: The search tree inherently supports "more thinking = better results"

## Limitations & Future Work

- The framework relies on LLaVA-Next-Video and Qwen-LM, and is thus constrained by their video understanding and reasoning capabilities
- The maximum tree height and branching factor still require manual specification
- Evaluation is limited to indoor household scenarios—outdoor environments and complex multi-person collaborative settings remain untested
- The LLM simulator may exhibit bias for rare or unusual scenarios
- Physical feasibility of task execution is not considered (e.g., whether the robot can actually perform "wipe the table")

## Related Work & Insights

- Distinction from autonomous skill acquisition works such as AutoRT (Ahn et al., 2024): HOTD emphasizes **open futures** and **human-centricity**, rather than task generation under fixed goals
- The search tree module draws on the test-time thinking paradigm of DeepSeek-R1 and O3, but applies it to the distinct domain of embodied task discovery
- The multi-agent design is inspired by frameworks such as MetaGPT, but features customized role decomposition tailored to search tree operations
- Insight: **Anticipatory assistance** in robotics—proactively discovering what can be done rather than merely responding to instructions—is a high-value research direction

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Triple contributions of problem formulation, benchmark, and method; the HOTD formalization is pioneering
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-baseline comparison, ablation studies, human evaluation, search strategy analysis, and cross-LMM integration
- Writing Quality: ⭐⭐⭐⭐ — Motivation is compellingly articulated; formalization is precise
- Value: ⭐⭐⭐⭐⭐ — Opens a new direction for anticipatory assistance in embodied AI

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] 10 Open Challenges Steering the Future of Vision-Language-Action Models](10_open_challenges_steering_the_future_of_vision-language-ac.md)
- [\[ICLR 2026\] WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](../../ICLR2026/robotics/weboperator_action-aware_tree_search_for_autonomous_agents_in_web_environment.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[AAAI 2026\] Theory of Mind for Explainable Human-Robot Interaction](theory_of_mind_for_explainable_human-robot_interaction.md)
- [\[NeurIPS 2025\] COOPERA: Continual Open-Ended Human-Robot Assistance](../../NeurIPS2025/robotics/coopera_continual_open_ended_human_robot_assistance.md)

<!-- RELATED:END -->
