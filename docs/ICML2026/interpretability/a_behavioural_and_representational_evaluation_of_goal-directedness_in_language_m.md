---
title: >-
  [Paper Note] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents
description: >-
  [ICML 2026][Interpretability][Goal-directedness] This paper proposes an evaluation framework for LLM Agent goal-directedness that integrates behavioral assessment with internal representation probing. In grid navigation tasks using GPT-OSS-20B, it was discovered that while the agent behaviorally follows goals, and internally encodes coarse-grained spatial maps and short-term plans, it can be misled by non-functional goal-like objects.
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Goal-directedness"
  - "LLM Agent"
  - "Representational Probing"
  - "Cognitive Map"
  - "GridWorld"
date: 2026-05-08
content_hash: 7673d5d8f5189aab
---

# A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents

**Conference**: ICML 2026  
**arXiv**: [2602.08964](https://arxiv.org/abs/2602.08964)  
**Code**: https://github.com/SPAR-Telos/interp; https://github.com/SPAR-Telos/reveng  
**Area**: Interpretability / Agent Evaluation / AI Safety  
**Keywords**: Goal-directedness, LLM Agent, Representational Probing, Cognitive Map, GridWorld  

## TL;DR
This paper proposes an evaluation framework for LLM Agent goal-directedness that integrates behavioral assessment with internal representation probing. In grid navigation tasks using GPT-OSS-20B, it was discovered that while the agent behaviorally follows goals, and internally encodes coarse-grained spatial maps and short-term plans, it can be misled by non-functional goal-like objects.

## Background & Motivation
**Background**: The most direct way to determine if an agent is "goal-directed" is typically to observe whether its behavior appears to optimize a specific objective. For instance, in navigation tasks, one can compare the agent's actions with the optimal strategy for reaching a target cell; if actions frequently fall within the set of optimal actions, the agent is said to exhibit goal-directedness.

**Limitations of Prior Work**: Relying solely on behavior easily conflates capability issues with goal-directedness issues. An agent taking the wrong path might lack a goal, or it might simply have misunderstood the map. Conversely, a system performing well in an evaluation might just be outputting aligned behavior to pass the test without actually possessing the internal goals assumed by the evaluators.

**Key Challenge**: Goal-directedness is not a property of external trajectories alone but an attribute formed by the alignment of "internal beliefs, planning, and action selection." If evaluators only use the optimal strategy of the ground-truth environment as a benchmark, they cannot distinguish between a "lack of goal" and "acting towards a goal based on erroneous beliefs."

**Goal**: The authors aim to establish a more diagnostic evaluation workflow: first testing the agent's explicit behavior in controlled environments, then decoding environment states, goal locations, and future action plans from model activations, and finally verifying whether the agent's actions are consistent with its internal representations.

**Key Insight**: The paper selects a fully observable 2D GridWorld as the experimental ground. This environment is simple enough to calculate exact optimal strategies, control difficulty, and design interventions, yet sufficient to investigate navigation, subgoals, distractors, and multi-step planning.

**Core Idea**: Instead of only asking "do actions match the ground-truth optimal strategy," one should also ask "do actions match the world model and plan encoded by the model itself."

## Method

### Overall Architecture
The paper utilizes GPT-OSS-20B as the LLM Agent, tasking it to navigate a text-represented MiniGrid by selecting Up, Down, Left, or Right to reach a target cell. The workflow is divided into two complementary threads: the behavioral thread, which constructs grids of varying sizes, obstacle densities, and goal structures to compare agent actions against A* optimal strategies; and the representational thread, which extracts residual stream activations before and after model inference to decode internal representations of the map, target location, and subsequent action sequences via probes.

Behavioral evaluation initially covers basic navigation. The authors generate grids of sizes $7, 9, 11, 13, 15$ with obstacle densities ranging from $0.0$ to $1.0$. For each size-density combination, 10 random grids are generated, and 10 trajectories are sampled per grid. The trajectory length limit is set to $1.5$ times the optimal path length to prevent infinite evaluation due to local oscillations.

Subsequently, two types of diagnostic environments are added. The first involves iso-difficulty transformations, including reflections, rotations, start-goal swaps, and transpositions; these preserve grid size, density, and optimal path length to check if the agent has preferences for specific visual/textual layouts. The second type involves multi-goal structures, including KeyDoorEnv (must collect a key to open a door), KeyNoDoorEnv (key is non-functional), and 2PathKeyEnv (two paths, one containing a non-functional key).

Representational evaluation focuses on three questions: whether a "cognitive map" can be decoded from activations; whether the agent's actions, if deviating from ground-truth optima, still align with the optimal strategy on this decoded map; and whether a multi-step action plan can be read from activations. Thus, the paper expands goal-directedness from a single behavioral score into a composite of "external performance + internal belief + planning consistency."

### Key Designs

**1. Difficulty-Controlled Behavioral Evaluation: Distinguishing Capability Degradation from Goal Absence**

Looking only at whether an agent reaches the goal conflates "wrong path due to no goal" with "wrong path due to insufficient capability." The authors systematically vary task difficulty across grid size, obstacle density, and goal distance, using the optimal strategy $\pi^*$ derived from A* search (Manhattan distance heuristic) as the benchmark. Per-action accuracy is defined as the proportion of actions within the optimal action set $\arg\max_a \pi^*(a\mid s_t)$, alongside statistics for action distribution entropy and Jensen-Shannon Divergence (JSD) from the optimal strategy. If the model is truly pursuing a goal, these metrics should degrade monotonically and continuously with difficulty rather than fluctuating randomly, thereby explicitly modeling capability decay.

**2. Controlled Perturbations and Multi-goal Structures: Isolating Layout Bias and Semantic Induction**

Goal-directed behavior should be stable across equivalent tasks and distinguish "true instrumental subgoals" from "goal-like distractors." The authors designed iso-difficulty transformations (reflection, rotation, etc.) which keep environment parameters identical; any significant performance change must be attributed to layout preference. Multi-goal structures like KeyNoDoorEnv and 2PathKeyEnv turn the "key"—an object with strong game semantics—into a non-functional entity. This tests whether the model is induced by pre-training semantics (e.g., "key = must pick up") to deviate from the actual task goal.

**3. Cognitive Maps and Plan Probes: Reading the Model's Own World Model**

Behavioral data cannot answer whether an action deviates from the optimum because of a lack of goal or an incorrect internal map. The third design uses two types of probes to decode the world model from GPT-OSS-20B residual stream activations. Cognitive map probes concatenate activations with query coordinates $(x,y)$ to predict if a cell contains an agent, goal, wall, etc. Plan probes use a Transformer decoder with $T=10$ learnable query vectors to predict a 10-action sequence all at once (non-autoregressively). This one-shot decoding is intentional: it ensures the plan info is truly encoded in the activation rather than the probe "calculating" the plan step-by-step.

### Loss & Training
The behavioral component does not involve training the agent; it samples trajectories under fixed prompts. The representational component trains lightweight probes: cognitive maps use linear and two-layer MLP probes, with class imbalance handled via upsampling. The plan decoder projects three activation vectors to 1024D with LayerNorm, then uses 1, 2, or 4-layer Transformer decoders for cross-attention over future step query vectors, followed by softmax for action prediction. Probe capacity is treated as a control variable: if larger probes always perform better, it might suggest the probe is solving the navigation; empirical results showing non-monotonic performance suggest the plan information is inherently decodable from the activations.

## Key Experimental Results

### Main Results

| Experimental Setting | Samples / Conditions | Key Metrics | Main Findings |
| :--- | :--- | :--- | :--- |
| Basic Navigation | 5 sizes × 6 densities × 10 trajectories | Accuracy drops with size/density/distance; JSD and entropy rise | Behavioral degradation aligns with task difficulty; failures are modulated by capability and complexity |
| Iso-difficulty Trans. | Reflection, Rotation, etc.; 300 pairs each | No significant differences (Wilcoxon test); e.g., Transpose $p=0.949$ | No significant dependence on specific grid layouts |
| KeyDoorEnv | 100 trajectories | 100.0% Success, $98.7\pm3.2$ Action Acc | Treats key as a necessary instrumental subgoal |
| KeyNoDoorEnv | 100 trajectories | 98.9% Success, 17.0% Key pickup rate, 75.0% of non-optimal actions towards key | Non-functional keys create semantic interference |
| 2PathKeyEnv | 100 trajectory pairs | Success: 71.4% (w/ key) vs 75.5% (w/o); 67.3% pickup rate on key path | Even if unnecessary, keys systematically alter path selection |

### Ablation Study

| Analysis Item | Key Metrics | Description |
| :--- | :--- | :--- |
| Cognitive Map Decoding | MLP cell accuracy ~70%; Linear probe 39.1% | Environment state is encoded non-linearly and coarse-grained in activations |
| Pre- vs. Post-推理 Change | Map accuracy drops from ~75% (pre) to ~60% (post) | Spatial map info weakens post-inference; representation shifts towards action selection |
| Strategy Consistency | Avg. 82.5% accuracy on decoded maps; Recovery range 37.4%--88.4% | Many "errors" in the real grid are rational actions relative to the model's internal map |
| Multi-step Plan Decoding | 2-layer decoder post-inf $N=1$ prefix acc 66.49%; pre-inf $N=7$ acc 3.8% | Post-inference favors current action decoding; pre-inference preserves more long-range trajectory structure |

### Key Findings
- Behavioral metrics are not directly equivalent to goal-directedness. Deviations in large or dense grids are partially explained by internal map errors.
- "Keys"—goal-like objects with high semantic weight—are the most significant failure points. They are handled well when necessary (KeyDoorEnv), but compete with the actual goal when non-functional.
- The inference process reorganizes representations: pre-inference is more about spatial maps and long-range plans, while post-inference compresses this into the upcoming action output.

## Highlights & Insights
- The paper shifts from "behavioral optimality" to "consistency with internal beliefs." This is crucial for safety, where we care about what the system is *trying* to do, not just its benchmark score.
- The multi-goal grid design is small but highly diagnostic. Using keys as both necessary subgoals and distractors allows for the separation of instrumental goal tracking from semantic induction bias.
- Plan probes are non-autoregressive. This design mitigates the suspicion that the probe itself is doing the planning, making the decodability of multi-step plans a more credible property of the model's activations.

## Limitations & Future Work
- The environment is fully observable and very small, far from the complexity of real-world LLM Agents (web, code, tool use). Memory, partial observability, and long-term dependencies will make goal attribution harder.
- Probe results are primarily correlational. Activation patching only shifted action distributions when applied across all layers at specific tokens; single-layer intervention effects were weak, so the causal chain is not fully established.
- Cognitive map probes decode agents and goals as "regions" rather than single points, indicating internal representations are fuzzy. Future work could use probabilistic maps or particle-style state estimation.
- Currently only evaluates GPT-OSS-20B. Generalizable conclusions require replication across model scales, training regimes, and agent scaffolds.

## Related Work & Insights
- **vs. Behavioral Goal-directedness Metrics**: While existing work matches trajectories to utility functions, this work adds the requirement of internal consistency, explaining "subjectively rational" errors.
- **vs. Inverse Reinforcement Learning**: IRL tries to infer reward functions from behavior, often assuming optimality; this work reads state and plans directly from activations, making it more suitable for white-box analysis.
- **vs. Traditional Probing**: While many probes only report property decodability, this work closes the loop by connecting decoded maps and plans back to behavior.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines behavioral eval with map/plan probing for goal-directedness analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Rigorous control in GridWorld; dimensions of analysis are rich, though limited to one domain.
- Writing Quality: ⭐⭐⭐⭐☆ Clear narrative; behavioral and representational sections transition naturally.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for LLM Agent safety and mechanistic goal attribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] Discovering Implicit Large Language Model Alignment Objectives](discovering_implicit_large_language_model_alignment_objectives.md)
- [\[ICLR 2026\] Certified Evaluation of Model-Level Explanations for Graph Neural Networks](../../ICLR2026/interpretability/certified_evaluation_of_model-level_explanations_for_graph_neural_networks.md)
- [\[ICLR 2026\] Representational Alignment Across Model Layers and Brain Regions with Multi-Level Optimal Transport](../../ICLR2026/interpretability/representational_alignment_across_model_layers_and_brain_regions_with_multi-leve.md)
- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](../../ACL2026/interpretability/how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)

</div>

<!-- RELATED:END -->
