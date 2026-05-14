---
title: >-
  [Paper Note] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining
description: >-
  [ICLR 2026][LLM Agent][GUI Agent] This paper proposes M²-Miner, the first MCTS-based automatic data mining framework for mobile GUI agents. Through the collaboration of three agents—InferAgent, OrchestraAgent…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "MCTS"
  - "Data Mining"
  - "Multi-Agent Collaboration"
  - "Mobile Interaction"
date: 2026-05-08
content_hash: 333dd83736af78ce
---

# M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining

**Conference**: ICLR 2026
**arXiv**: [2602.05429](https://arxiv.org/abs/2602.05429)
**Code**: Coming soon
**Area**: LLM Agent
**Keywords**: GUI Agent, MCTS, Data Mining, Multi-Agent Collaboration, Mobile Interaction

## TL;DR
This paper proposes M²-Miner, the first MCTS-based automatic data mining framework for mobile GUI agents. Through the collaboration of three agents—InferAgent, OrchestraAgent, and JudgeAgent—it achieves a 64× improvement in mining efficiency, enriches intent diversity via an intent recycling strategy, and trains a GUI agent that achieves state-of-the-art performance on multiple benchmarks.

## Background & Motivation
**Background**: GUI agents automate software interactions by interpreting user intents and executing action sequences on graphical interfaces, representing a prominent research direction in both academia and industry. High-quality intent–trajectory training data is the core dependency of current GUI agents.

**Limitations of Prior Work**:
   - **High cost**: Manual annotation (e.g., AITW, AndroidControl) requires hours per sample, with costs as high as $0.36/image.
   - **Low quality**: Both manually annotated and automatically mined data frequently contain redundant steps, ambiguous intent descriptions, and biased action paths.
   - **Low diversity**: Existing datasets adopt an intent-to-flat-trajectory structure, recording only a single successful path per intent, resulting in monotonic intent types.

**Key Challenge**: Manual annotation is quality-controllable but not scalable, while existing automated mining methods (e.g., AgentQ based on vanilla MCTS, OS-Genesis based on rule-driven exploration) suffer from low efficiency, are limited to web environments, or produce only single trajectories.

**Goal**: How to automatically mine high-quality, high-diversity mobile GUI interaction trajectory data at low cost?

**Key Insight**: The paper introduces MCTS into mobile GUI data mining, but vanilla MCTS suffers from extremely low efficiency due to random expansion. The authors observe that: (a) the expansion phase requires intelligent guidance rather than random exploration; (b) the simulation phase can replace rollouts with process rewards; (c) non-primary paths in the search tree contain additional valuable intent–trajectory pairs.

**Core Idea**: MCTS + three-agent collaboration (guided expansion + accelerated ranking + process evaluation) + intent recycling = efficient, high-quality, and high-diversity GUI data mining.

## Method

### Overall Architecture
M²-Miner uses MCTS as its backbone. Given an initial intent $I_0$ and a starting GUI state $s_0$, it outputs an intent–trajectory tree $\mathcal{T}=(\mathcal{V},\mathcal{A},\mathcal{P},\mathcal{I})$ containing valid interaction trajectories $\tau=(s_0,a_0,s_1,\ldots)$. Each node in the tree stores a screenshot, action description, Q-value, visit count, and task completion status. The four MCTS phases (selection → expansion → simulation → backpropagation) iterate until a trajectory matching the intent is mined.

### Key Designs

1. **InferAgent (Expansion Phase — Action Generation)**:

    - **Function**: Generates $K$ candidate actions for the selected node.
    - **Mechanism**: Reasons about the most likely correct action based on the current GUI screenshot and target intent. Multiple different MLLMs are used for generation to ensure diversity of the action space; previously generated actions are included in the prompt to avoid duplicates.
    - **Design Motivation**: Replaces the random expansion of vanilla MCTS, substantially increasing the hit rate of correct actions.

2. **OrchestraAgent (Expansion Phase — Action Ranking and Deduplication)**:

    - **Function**: Merges semantically equivalent actions (e.g., clicks on the same button at different coordinates) and ranks them by their likelihood of achieving the target intent.
    - **Mechanism**: An MLLM selects the most promising action at each iteration via a multi-choice question format; $K-1$ queries yield a ranked queue. Ranked actions are assigned decreasing initial UCT values in order.
    - **Design Motivation**: Avoids redundant expansion and ensures that the search prioritizes the most promising branches.

3. **JudgeAgent (Simulation Phase — Process Reward Estimation)**:

    - **Function**: Analyzes the GUI screenshot of newly expanded nodes, assesses task completion status, and computes rewards.
    - **Mechanism**: Terminal nodes (success/failure) receive rewards of 1/0. For intermediate nodes, the MLLM head outputs logits for "valid"/"invalid", normalized via softmax into a $[0,1]$ probability as the reward: $r_{\text{intermediate}} = \frac{\exp(logits_{\text{valid}})}{\exp(logits_{\text{valid}}) + \exp(logits_{\text{invalid}})}$
    - **Design Motivation**: Replaces the full rollout required by vanilla MCTS to evaluate rewards. Node Q-values are updated as: $Q_i = \frac{Q_{i-1} \times N_{i-1} + R_i}{N_{i-1}+1}$, greatly reducing computational overhead in the simulation phase.

4. **Intent Recycling Strategy**:

    - **Function**: Extracts additional intent–trajectory pairs from non-primary paths in a completed mining tree.
    - **Mechanism**: All root-to-node paths in the tree are enumerated and evaluated by an intent recycling filter (implemented via an MLLM); paths that pass are assigned new intents generated by an MLLM, which are then verified for consistency with the trajectory by JudgeAgent.
    - **Design Motivation**: Evolves the structure from "one tree, one intent" to "one tree, multiple intents," yielding more diverse data without re-mining. For example, when mining the intent "query a route," an accidental tap on a "ride-hailing" button may produce a valid ride-hailing trajectory.

5. **Progressive Model-in-the-Loop Training Strategy**:

    - **Function**: Iteratively improves the capabilities of InferAgent and JudgeAgent.
    - **Mechanism**: Three progressive training stages — Stage 1: basic intents (common services + conditional rewriting) → Stage 2: complex intents (function combinations + retry on failure) → Stage 3: recycled intents (applying intent recycling to historical trees). Data mined at each stage is used for continual training of both agents.
    - **Design Motivation**: Forms a positive feedback loop in which agent capability and data complexity grow in tandem.

### Loss & Training
- InferAgent and JudgeAgent are fine-tuned based on Qwen2.5-VL-7B.
- OrchestraAgent and the intent recycling filter use Qwen2.5-VL-72B.
- Training data also includes description information and preference data (constructed from positive/negative paths).

## Key Experimental Results

### Main Results

| Model | AC-Low TP/SR | AC-High TP/SR | AITZ TP/SR | GUI-Odyssey TP/SR | CAGUI TP/SR |
|------|-------------|--------------|-----------|-------------------|-------------|
| GPT-4o | 74.3/19.4 | 66.3/20.8 | 70.0/35.3 | - | 3.67/3.67 |
| UI-TARS-7B* | 98.0/90.8 | 83.7/72.5 | 80.4/65.8 | 90.1/87.0 | 88.6/70.0 |
| OS-Genesis-7B | 90.7/74.2 | 66.2/44.5 | 20.0/8.5 | 11.7/3.6 | 38.1/14.5 |
| GUI-Owl-7B | 93.8/90.0 | 81.5/72.8 | 78.9/65.1 | 83.4/60.7 | 80.0/59.2 |
| **M²-Miner-7B** | **97.5/93.5** | 81.8/**72.9** | **81.3/69.4** | **90.5/79.3** | **88.8/70.2** |

*UI-TARS-7B uses large-scale private manually annotated data.

### Ablation Study

| Configuration | TP | SR | Note |
|------|----|----|------|
| Warm-up | 85.0 | 64.2 | Pre-trained on public data only |
| + Stage 1 (Basic Intents) | 86.5 | 67.3 | +3.1% SR |
| + Stage 2 (Complex Intents) | 87.6 | 69.1 | +1.8% SR |
| + Stage 3 (Recycled Intents) | 88.2 | 69.9 | +0.8% SR, cumulative +5.7% |
| Act only | 85.2 | 66.8 | Action labels only |
| Act + Description | 88.2 | 69.9 | +3.1% SR |
| Act + Des + Preference | **88.8** | **70.2** | +3.4% SR |

### Key Findings
- **Exponential efficiency gains**: Compared to vanilla MCTS, M²-Miner achieves a 64× efficiency improvement at a task length of 9. OrchestraAgent reduces redundant nodes in the expansion phase; JudgeAgent eliminates rollouts in the simulation phase.
- **18× cost reduction**: The M²-Miner-Agent dataset costs only $0.02 per image, versus $0.36 for manually annotated datasets.
- **Higher data quality**: Manual inspection of 100 randomly sampled records shows that M²-Miner's data quality accuracy (DQA) surpasses that of manually annotated datasets (AC and AITZ).
- **Description and preference data are beneficial**: Compared to action labels alone, adding description information improves SR by 3.1%; adding preference data yields a further 0.3% gain.
- **Strong generalization to unseen scenarios**: On CAGUI, where no training data is available, Qwen2.5-VL-7B trained on M²-Miner data improves SR from 55.2% to 70.2%.

## Highlights & Insights
- **The MCTS + multi-agent paradigm is elegantly designed**: The three agents each fulfill a distinct role—generation, ranking, and evaluation—precisely addressing the three bottlenecks of vanilla MCTS in the GUI domain (random expansion, redundant nodes, and expensive rollouts).
- **Intent Recycling is a highly creative design**: It repurposes "failed" exploratory paths as an additional data source, simultaneously addressing intent diversity and mining efficiency. This idea is transferable to other agent data collection settings.
- **Process rewards as rollout substitutes**: Using MLLM logit probabilities as intermediate rewards preserves the theoretical advantages of MCTS while avoiding the high cost of full simulation. This design has reference value for all MCTS+LLM systems.

## Limitations & Future Work
- The framework is validated only on mobile platforms; applicability to desktop and web environments remains unexplored.
- OrchestraAgent relies on a 72B model (Qwen2.5-VL-72B), incurring high deployment costs.
- The quality of intent recycling filtering and intent generation depends on MLLM capability and may degrade for complex applications.
- The dataset scale (20k images, 2,565 trajectories) still lags behind the private data used by UI-TARS.
- The three-stage model-in-the-loop training requires multiple mining → training cycles; the practical engineering cost is not reported in detail.

## Related Work & Insights
- **vs. AgentQ**: AgentQ pioneered MCTS-based data mining for web environments but is limited to parseable HTML environments and suffers from low efficiency. M²-Miner extends the approach to vision-driven mobile platforms, improving efficiency by orders of magnitude through multi-agent collaboration.
- **vs. OS-Genesis**: OS-Genesis employs unsupervised rule-based interaction and reverse task synthesis, requiring no predefined tasks but yielding lower data quality (SR of only 8.5% on AITZ). M²-Miner's structured MCTS search produces higher-quality trajectories.
- **vs. UI-TARS**: UI-TARS relies on large-scale private manually annotated data, yet M²-Miner surpasses it on nearly all SR metrics, demonstrating that the potential of automated mining frameworks has exceeded that of manual annotation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of MCTS, multi-agent collaboration, and intent recycling is pioneering in GUI data mining, with clear technical contributions from each component.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks, multiple ablation groups (multi-agent design, training strategy, data structure, training data), efficiency analysis, and cost comparison — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich illustrations, though some content is repetitive and the paper is lengthy.
- **Value**: ⭐⭐⭐⭐⭐ Provides a highly valuable data production paradigm for the GUI agent community, with empirical evidence that automated mining can surpass manual annotation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](../../CVPR2026/llm_agent/gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[AAAI 2026\] EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](../../AAAI2026/llm_agent/ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)
- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)

</div>

<!-- RELATED:END -->
