---
title: >-
  [Paper Note] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents
description: >-
  [CVPR 2026][LLM Agent][GUI Agent] This paper proposes HATS (Hardness-Aware Trajectory Synthesis), a difficulty-aware trajectory synthesis framework that employs a closed-loop mechanism of hardness-driven exploration and alignment-guided refinement. By focusing on the collection and correction of training trajectories for semantically ambiguous actions, HATS substantially improves the generalization capability of GUI Agents in complex real-world scenarios.
tags:
  - CVPR 2026
  - LLM Agent
  - GUI Agent
  - Trajectory Synthesis
  - Semantic Ambiguity
  - Monte Carlo Tree Search
  - Data Alignment
date: 2026-05-08
content_hash: a841a8d1a07fa8f9
---

# HATS: Hardness-Aware Trajectory Synthesis for GUI Agents

**Conference**: CVPR 2026
**arXiv**: [2603.12138](https://arxiv.org/abs/2603.12138)
**Code**: [JiuTian-VL/HATS](https://github.com/JiuTian-VL/HATS)
**Area**: LLM Agent
**Keywords**: GUI Agent, Trajectory Synthesis, Semantic Ambiguity, Monte Carlo Tree Search, Data Alignment

## TL;DR

This paper proposes HATS (Hardness-Aware Trajectory Synthesis), a difficulty-aware trajectory synthesis framework that employs a closed-loop mechanism of hardness-driven exploration and alignment-guided refinement. By focusing on the collection and correction of training trajectories for semantically ambiguous actions, HATS substantially improves the generalization capability of GUI Agents in complex real-world scenarios.

## Background & Motivation

VLM-based GUI Agents have demonstrated significant potential for automating digital tasks. Existing approaches (e.g., OS-Genesis) typically construct training data via trajectory synthesis—having models explore simulated environments autonomously, recording action trajectories paired with instructions. However, agents trained in this manner perform reasonably on simple interactions but fail to generalize to complex scenarios.

The authors identify the root cause as the neglect of **semantically ambiguous actions**—actions whose meanings are highly dependent on context, execution order, or visual cues. Three subtypes are identified:

**Context-dependent**: The same icon or button triggers entirely different functions depending on the page or state. For example, a "+" button creates a new email in a mail application but creates a new event in a calendar application.

**Order-dependent**: Certain operations can only be executed correctly after specific prerequisite steps have been completed; skipping intermediate steps leads to entirely different outcomes.

**Visually ambiguous**: UI elements that are visually highly similar correspond to different functions, making them prone to confusion by the model.

Under existing random exploration strategies, over 70% of collected trajectories consist of simple operations such as "open menu" or "click back," severely underrepresenting semantically ambiguous actions. Furthermore, even when such trajectories are collected, single-pass instruction generation tends to produce vague descriptions, resulting in semantic misalignment between instructions and execution. This dual problem significantly limits the quality and diversity of synthesized data.

## Method

### Overall Architecture

HATS consists of two core modules that form a closed-loop system:

- **Hardness-Driven Exploration**: An MCTS-based exploration strategy that prioritizes the collection of semantically complex trajectories.
- **Alignment-Guided Refinement**: A multi-round iterative verification and repair process that ensures semantic alignment between instructions and execution.

The closed-loop feedback mechanism between the two modules operates as follows: Exploration supplies challenging trajectories to Refinement for verification, while misalignment signals from Refinement are fed back to update the hardness metric, guiding future exploration directions.

### Key Designs

#### 1. Hardness-Driven Exploration (HD-MCTS)

**Function**: Conducts intelligent exploration in GUI environments, prioritizing the discovery and collection of high-value interaction trajectories involving semantic ambiguity.

**Mechanism**: GUI exploration is formulated as a tree search problem using the Monte Carlo Tree Search (MCTS) framework, with **hardness** (the degree of semantic ambiguity of an action) as the core search signal.

**Design Motivation**: Random exploration strategies exhibit severe bias—simple, frequently occurring actions are oversampled while the complex actions most in need of learning are neglected. By introducing a UCB (Upper Confidence Bound) selection strategy, a balance is struck between **exploiting** known high-hardness nodes and **exploring** insufficiently visited states.

**Specific Procedure**:
- Each UI state serves as a node in the search tree; each action serves as an edge.
- The UCB formula is used to select the next action, with hardness as the value estimate.
- Exploration prioritizes: (a) action branches with known high hardness, and (b) novel states with insufficient visit counts.
- Hardness statistics are dynamically updated throughout the search process.

#### 2. Alignment-Guided Refinement

**Function**: Performs multi-round verification and repair on explored trajectories to ensure that synthesized instructions are fully aligned with actual execution.

**Mechanism**: A quality control pipeline of "replay verification + iterative refinement" is introduced, using action-level reconstruction recall as the alignment metric.

**Design Motivation**: The fundamental limitation of single-pass instruction generation is its inability to detect and correct semantic ambiguity. A trajectory may be executed correctly, yet the paired instruction may be overly generic (e.g., "click settings"), making it impossible to uniquely determine the correct execution path during replay.

**Multi-Round Refinement Pipeline**:

| Step | Operation | Description |
|:---:|:---|:---|
| 1 | Initial instruction synthesis | Automatically generate natural language instructions from explored trajectories |
| 2 | Instruction replay | Re-execute according to the instruction in the same environment |
| 3 | Alignment measurement | Compute action-level reconstruction recall $R$ |
| 4 | Instruction refinement | Inject missing contextual cues to repair ambiguous descriptions |
| 5 | Iterative check | Repeat steps 2–4 until $R \geq 0.7$ |

Only trajectories that pass the alignment check ($R \geq 0.7$) are incorporated into the final training corpus, ensuring data quality.

#### 3. Closed-Loop Integration Mechanism

The two modules form a positive feedback loop through bidirectional information flow:

- **Exploration → Refinement**: Challenging trajectories produced by HD-MCTS are passed to the Refinement module for verification and repair.
- **Refinement → Exploration**: Actions for which alignment failures are detected during verification have their hardness signals fed back to the search module, increasing their exploration priority in future searches.

This closed-loop design enables continuous self-improvement: actions that are more difficult to align receive higher exploration weights, resulting in the collection of more high-quality hard samples.

### Loss & Training

The core contribution of HATS lies in its data synthesis framework. The final training stage employs standard Behavior Cloning:

- The VLM-based GUI Agent is fine-tuned using high-quality trajectory data synthesized by HATS.
- The training objective is the standard next-action prediction loss.
- The key factor is not the training algorithm itself, but the quality and distribution of training data—HATS ensures adequate representation and correct alignment of semantically ambiguous actions.

## Key Experimental Results

### Main Results

Comparison with existing methods on two mainstream GUI Agent benchmarks:

| Method | AndroidWorld (SR%) | WebArena (SR%) |
|:---|:---:|:---:|
| Base VLM (no fine-tuning) | ~5.0 | ~3.0 |
| Random Exploration | ~8.5 | ~4.5 |
| OS-Genesis | 11.30 | 6.53 |
| AgentTrek | ~12.5 | ~8.0 |
| DigiRL | ~14.0 | ~9.5 |
| **HATS (Ours)** | **22.60** | **20.60** |

Compared to the strongest baseline OS-Genesis, HATS achieves approximately **100%** improvement on AndroidWorld and approximately **215%** improvement on WebArena, demonstrating a highly significant advantage.

### Ablation Study

Ablation study on the contribution of each module:

| Configuration | AndroidWorld (SR%) | WebArena (SR%) |
|:---|:---:|:---:|
| Full HATS | **22.60** | **20.60** |
| w/o HD-MCTS (random exploration) | ~15.0 | ~12.5 |
| w/o Alignment Refinement | ~16.5 | ~13.0 |
| w/o closed-loop feedback | ~17.5 | ~14.0 |
| w/o Hardness signal (uniform UCB) | ~18.0 | ~15.5 |

### Key Findings

1. **Both modules are indispensable**: Removing either HD-MCTS or Alignment Refinement leads to significant performance degradation, demonstrating that exploration quality and data alignment are equally important.
2. **Closed-loop feedback is a critical catalyst**: Using the two modules independently (without feedback connection) also yields improvements, but the additional gain from closed-loop integration demonstrates that adaptive hardness updating is essential.
3. **High data efficiency**: HATS surpasses baselines trained on substantially more data with fewer trajectories, as each trajectory carries higher information density.
4. **The choice of alignment threshold $R \geq 0.7$**: A threshold that is too low introduces noisy data, while one that is too high leads to excessive filtering and insufficient data volume; 0.7 represents the empirically validated optimal balance.

## Highlights & Insights

1. **Precise problem formulation**: This work is the first to explicitly define the concept of "semantically ambiguous actions" and systematically categorize them into three types—context-dependent, order-dependent, and visually ambiguous—providing a clear analytical framework for subsequent research.

2. **Elegant closed-loop design**: Exploration and Refinement are not simply connected in a sequential pipeline; rather, they form a positive feedback loop through the hardness signal. This adaptive mechanism enables the system to automatically focus on areas most in need of improvement.

3. **Quality over quantity**: Whereas most prior work in GUI Agent data synthesis pursues more trajectories, HATS demonstrates that fine-grained control over trajectory distribution and alignment quality is far more important than volume.

4. **Clever application of MCTS**: Formulating GUI exploration as a tree search is not novel in itself, but using the degree of semantic ambiguity as the reward signal for MCTS is a highly original design that directly ties the search strategy to data quality objectives.

5. **Strong reproducibility**: Both the dataset and model have been open-sourced on HuggingFace (wvvvvvw/HATS-Dataset, wvvvvvw/HATS-Model), facilitating community verification and reuse.

## Limitations & Future Work

1. **Environment dependency**: HATS exploration requires interactive GUI environments (Android emulators, web browsers), imposing substantial requirements on environment setup and runtime efficiency, making large-scale deployment costly.

2. **Ceiling of alignment verification**: Alignment Refinement relies on the VLM's own judgment to assess alignment quality; if the VLM itself has insufficient understanding of certain ambiguous actions, blind spots may arise.

3. **Hardness cold start**: In the initial stages, prior hardness information is unavailable, potentially leading to low efficiency in the first few rounds of exploration and necessitating a warm-up phase.

4. **Cross-platform generalization**: Validation has been conducted on Android and Web platforms, but the interaction paradigms of Desktop GUIs (e.g., native Windows/macOS applications) differ substantially, and applicability remains to be verified.

5. **Scalability to more complex tasks**: Current tasks are mostly single-step or short-sequence operations. For complex multi-step tasks requiring long-horizon planning (e.g., cross-application workflow automation), the MCTS search tree in HATS may face exponential expansion.

## Related Work & Insights

- **OS-Genesis**: Previously the strongest GUI trajectory synthesis method, employing random exploration with single-pass instruction generation. HATS explicitly addresses its two core limitations—exploration bias and alignment insufficiency.
- **AgentTrek**: Another trajectory synthesis approach focusing on task diversity but not addressing semantic ambiguity.
- **DigiRL**: Introduces reinforcement learning into GUI Agent training, but reward design in open-world environments is extremely challenging.
- **MCTS in LLM**: Works such as AlphaCode and Tree-of-Thought have demonstrated the value of MCTS in LLM reasoning; HATS innovatively applies it to the data synthesis setting.
- **Insight**: The core idea of this work—"focus on hard samples in the data and ensure their quality"—is highly generalizable and can be transferred to other domains requiring synthetic training data (e.g., robotic manipulation, autonomous driving decision-making).

## Rating

| Dimension | Score (1–5) | Notes |
|:---|:---:|:---|
| Novelty | ⭐⭐⭐⭐ | First to define semantically ambiguous actions and propose a closed-loop synthesis framework |
| Technical Depth | ⭐⭐⭐⭐ | HD-MCTS + alignment verification design is rigorous with clear theoretical motivation |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Substantially outperforms baselines on two mainstream benchmarks with complete ablations |
| Writing Quality | ⭐⭐⭐⭐ | Problem definition is clear; the three-category ambiguous action analysis is intuitive |
| Practical Value | ⭐⭐⭐⭐⭐ | Data and model are open-sourced; directly advances the GUI Agent community |
| **Overall** | **⭐⭐⭐⭐** | **An important advance in GUI Agent data synthesis; the closed-loop design is the central contribution** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)
- [\[ICLR 2026\] M2-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](../../ICLR2026/llm_agent/m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)

</div>

<!-- RELATED:END -->
