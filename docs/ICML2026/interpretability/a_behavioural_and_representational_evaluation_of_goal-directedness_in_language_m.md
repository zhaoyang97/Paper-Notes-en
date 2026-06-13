---
title: >-
  [Paper Note] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents
description: >-
  [ICML 2026][Interpretability][Goal-directedness] This paper proposes an evaluation framework for LLM Agent goal-directedness that combines behavioral assessment with internal representation probing. In grid navigation ta…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Goal-directedness"
  - "LLM Agent"
  - "Representation Probing"
  - "Cognitive Map"
  - "GridWorld"
date: 2026-05-08
content_hash: 787b5d08900b8fe8
---

# A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents

**Conference**: ICML 2026  
**arXiv**: [2602.08964](https://arxiv.org/abs/2602.08964)  
**Code**: https://github.com/SPAR-Telos/interp; https://github.com/SPAR-Telos/reveng  
**Area**: Interpretability / Agent Evaluation / AI Safety  
**Keywords**: Goal-directedness, LLM Agent, Representation Probing, Cognitive Map, GridWorld  

## TL;DR
This paper proposes an evaluation framework for LLM Agent goal-directedness that combines behavioral assessment with internal representation probing. In grid navigation tasks using GPT-OSS-20B, it finds that while the agent generally acts goal-directed behaviorally and encodes coarse-grained spatial maps and short-term plans internally, it can be misled by non-functional goal-like objects.

## Background & Motivation
**Background**: The most direct way to judge if an agent is "goal-directed" is typically to observe if its behavior appears to optimize a target. For example, in navigation tasks, one can compare the agent's actions against an optimal policy to reach a target cell; if actions frequently fall within the optimal set, the agent is said to exhibit goal-directedness.

**Limitations of Prior Work**: Observation of behavior alone easily conflates capability issues with goal issues. An agent taking a wrong path might not be pursuing the goal, or it might simply have misunderstood the map. Conversely, a system performing well in evaluations might just be outputting aligned behavior to pass the test without actually possessing the internal goal assumed.

**Key Challenge**: Goal-directedness is not a pure external trajectory property but an emergent property of "internal beliefs, planning, and action selection." If evaluators only use the ground-truth optimal policy as a benchmark, they cannot distinguish between "lacking a goal" and "acting toward a goal based on false beliefs."

**Goal**: The authors aim to establish a more diagnostic evaluation pipeline: first testing explicit behavior in controlled environments, then decoding environment states, goal locations, and future action plans from model activations, and finally comparing whether the agent's actions are consistent with its internal representations.

**Key Insight**: The paper selects fully observable 2D grid worlds as the experimental ground. This environment is simple enough to precisely calculate optimal policies, control difficulty, and design interventions, yet sufficient to examine navigation, subgoals, distractors, and multi-step planning.

**Core Idea**: Instead of only asking "do actions align with the true optimal policy," also ask "do actions align with the world model and plans encoded by the model itself."

## Method

### Overall Architecture
The paper utilizes GPT-OSS-20B as the LLM Agent, tasking it to navigate a text-represented MiniGrid by selecting Up, Down, Left, or Right actions. The process involves two complementary pipelines: the behavioral pipeline constructs grids of varying sizes, obstacle densities, and goal structures to compare actions against A* optimal policies; the representation pipeline extracts residual stream activations before and after inference to decode internal representations of the map, goal location, and subsequent action sequences using probes.

Behavioral evaluation initially covers basic navigation. The authors generate grids of sizes $7, 9, 11, 13, 15$ with obstacle densities from $0.0$ to $1.0$. For each size-density combination, 10 random grids are generated with 10 sampled trajectories per grid. Trajectory length is capped at $1.5 \times$ the optimal path length to prevent infinite loops during local oscillations.

Two types of diagnostic environments are added. The first is iso-difficulty transformation, including reflections, rotations, start-goal swaps, and transpositions; these preserve grid size and optimal path length to check for biases in text/visual arrangements. The second involves multi-objective structures: KeyDoorEnv (must collect a key to open a door), KeyNoDoorEnv (key is non-functional), and 2PathKeyEnv (two paths where one contains a non-functional key).

Representational evaluation centers on three questions: whether a "cognitive map" can be decoded from activations; whether actions deviating from the true optimal policy remain consistent with the optimal policy on the decoded map; and whether multi-step plans can be read from pre/post-inference activations. This expands goal-directedness from a single behavioral score into integrated evidence of "external performance + internal belief + planning consistency."

### Key Designs

1. **Difficulty-Controlled Behavioral Evaluation**:

	- **Function**: Systematically varies task difficulty across grid size, obstacle density, and goal distance, measuring action quality against optimal policies.
	- **Mechanism**: The authors use A* and Manhattan distance for reference policies, defining per-action accuracy as whether an action belongs to the optimal set, while tracking policy entropy and Jensen-Shannon divergence. If the model truly pursues the goal, these metrics should degrade continuously with difficulty rather than fluctuating randomly.
	- **Design Motivation**: This setup explicitly models "capability drop with difficulty," preventing failures on hard grids from being simply interpreted as a lack of goal-directedness.

2. **Controlled Perturbations and Multi-Objective structures**:

	- **Function**: Tests if the agent remains stable against task-irrelevant environment permutations and can distinguish instrumental subgoals from semantically distracting objects.
	- **Mechanism**: Iso-difficulty transformations keep path length constant; significant performance changes would indicate arrangement bias. KeyDoorEnv requires a key as a subgoal, while KeyNoDoorEnv and 2PathKeyEnv turn the key into a non-functional but semantically "goal-like" object.
	- **Design Motivation**: Goal-directed behavior should not just reach the end but remain stable across equivalent environments and distinguish true utility from common game semantics found in pre-training data.

3. **Cognitive Map and Plan Probes**:

	- **Function**: Decodes environment states, goal locations, and multi-step plans from internal activations to explain behavioral deviations.
	- **Mechanism**: The cognitive map probe concatenates activations with query coordinates $(x,y)$ to predict if a cell is an agent, goal, wall, open, or padding. The plan probe uses activations from 3 tokens through a linear bottleneck and Transformer decoder to predict a length-10 action sequence at once, rather than autoregressively.
	- **Design Motivation**: Coordinate-conditioned probes determine if the model preserves spatial structure. One-shot plan decoding reduces the chance of the probe "inventing" the plan itself, making above-chance prefix accuracy a stronger indicator of existing plan information in the base model.

### Loss & Training
The behavioral component does not train the agent; it samples trajectories under fixed prompts. The representation component trains lightweight probes: cognitive maps use linear and 2-layer MLP probes with oversampling for class imbalance. The plan decoder projects 3 activation vectors to 1024 dimensions with LayerNorm, then uses 1, 2, or 4-layer Transformer decoders with cross-attention for each future-step query, finally using softmax for action prediction. Probe capacity is a control variable: if larger probes were always better, it might suggest the probe is solving navigation; results were non-monotonic, supporting the presence of readable plan information in activations.

## Key Experimental Results

### Main Results

| Experimental Setup | Samples / Conditions | Key Metrics | Main Conclusions |
|--------|------|------|----------|
| Basic Navigation | 5 sizes × 6 densities × 10 trajectories | Accuracy drops with size/density/distance; JSD and entropy rise | Behavioral degradation aligns with task difficulty, indicating failures are modulated by capability and environmental complexity |
| Iso-difficulty Transformation | Reflection, Rotation, Swap, Transpose; 300 pairs each | Wilcoxon tests show no significant differences; e.g., Reflection $p=0.582$, Transpose $p=0.949$ | The model does not rely on specific grid arrangements |
| KeyDoorEnv | 100 trajectories | 100.0% Success, $98.7\pm3.2$ Action Accuracy | Can treat key as a necessary instrumental subgoal |
| KeyNoDoorEnv | 100 trajectories | 98.9% Success, but 17.0% key pickup; 75.0% of non-optimal actions were toward key | Non-functional keys create semantic interference |
| 2PathKeyEnv | 100 path pairs (w/ & w/o key) | 71.4% Success with key, 75.5% without; 67.3% pickup rate on key path | Even if unnecessary, the key systematically alters path selection |

### Ablation Study

| Analysis Item | Key Metrics | Description |
|------|---------|------|
| Cognitive Map Decoding | MLP cell accuracy approx. 70%, 75.7% on $11\times11$; Linear probe 39.1% | Environmental state is encoded non-linearly and coarsely in activations |
| Pre/Post-Inference Representation | Map accuracy drops from ~75% pre-inference to ~60% post-inference | Spatial map information weakens after inference, shifting toward the next action choice |
| Consistency with Decoded Map | Action accuracy on decoded map averages 82.5%; Recovery averages 57.9% (range 37.4%--88.4%) | Many ground-truth errors appear as rational actions on the model's own "rough" map |
| Multi-step Plan Decoding | 2-layer decoder post-inference $N=1$ prefix accuracy 66.49% (vs 41.5% pre) | Post-inference favors reading the immediate next step; pre-inference retains more long-range trajectory structure |

### Key Findings
- Behavioral metrics are not directly equivalent to goal-directedness. Deviations from optimal policy on large/complex grids can often be explained by errors in the model's internal map.
- "Keys" as semantically strong goal-like objects are a primary failure point. While handled well as subgoals in KeyDoorEnv, they become distractors when non-functional, showing competition between pre-trained semantics and task goals.
- Inference acts as a representation reorganizer: pre-inference preserves spatial maps and long-range plans, while post-inference compresses this into the immediate output action. This explains why action log-probability post-inference is a poor proxy for uncertainty.

## Highlights & Insights
- The paper shifts the focus of goal-directedness from "is the behavior optimal" to "is the behavior consistent with internal beliefs." This is crucial for safety as it identifies what a system is actually pursuing internally.
- The multi-objective grid design is small but highly diagnostic. Using the key as both a true subgoal and a distractor effectively separates instrumental goal tracking from semantic bias.
- The plan probe predicts the entire prefix at once rather than autoregressively, reducing the likelihood of the probe performing its own planning and ensuring the plan information truly originates from the model's activations.

## Limitations & Future Work
- The environment is fully observable and very small, far from the complexity of real-world LLM Agent scenarios (web, code, tool use). Memory, partial observability, and long-term tool feedback will make goal attribution more difficult.
- Probing results provide primarily correlational evidence. Activation patching attempts showed that only patching across all layers at specific token positions changed action distributions; single-layer interventions were weak, meaning a full causal chain is not yet established.
- The cognitive map probe decodes the agent/goal as a neighborhood rather than a single point, indicating "blurry" internal representations. Future work could use probabilistic maps or particle-style state estimation.
- Current evaluations are limited to GPT-OSS-20B. General conclusions require replication across model scales, training methods, and agent scaffolds.

## Related Work & Insights
- **vs Behavioral Goal-Directedness Measures**: While existing work matches trajectories to candidate utility functions, this paper adds the requirement of internal world model alignment to explain "subjectively rational" errors.
- **vs Inverse Reinforcement Learning (IRL)**: Unlike IRL which infers reward functions from behavior (often assuming optimality), this approach decodes state and plans directly from activations, which is more suitable for white-box LLM analysis.
- **vs Traditional Probing**: Many works simply report if a property is readable; this paper closes the loop by testing if decoded maps and plans can actually explain behavioral choices.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines behavioral evaluation, cognitive map probing, and plan decoding fruitfully.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Rigorous control in GridWorld with rich analysis, though model and task scope are narrow.
- Writing Quality: ⭐⭐⭐⭐☆ Clear narrative; behavioral-to-representational transition is natural.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for LLM Agent safety and mechanistic goal attribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] Discovering Implicit Large Language Model Alignment Objectives](discovering_implicit_large_language_model_alignment_objectives.md)
- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](../../ACL2026/interpretability/how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](../../NeurIPS2025/interpretability/model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)

</div>

<!-- RELATED:END -->
