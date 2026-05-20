---
title: >-
  [Paper Note] Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation
description: >-
  [ICML 2026][Robotics][Atomic skills] For zero-shot robotic manipulation from "training tasks to novel tasks," the authors decompose demonstrations into "atomic skill-action pairs" as an intermediate representation. They…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Atomic skills"
  - "in-context learning"
  - "cross-task zero-shot"
  - "dual dynamic/static demonstration libraries"
  - "skill coverage"
date: 2026-05-08
content_hash: 2732214f7d04863e
---

# Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation

**Conference**: ICML 2026  
**arXiv**: [2605.01448](https://arxiv.org/abs/2605.01448)  
**Code**: None (not declared in the paper)  
**Area**: Robotics / Cross-Task Generalization / Vision-Language-Action Models  
**Keywords**: Atomic skills, in-context learning, cross-task zero-shot, dual dynamic/static demonstration libraries, skill coverage

## TL;DR
For zero-shot robotic manipulation from "training tasks to novel tasks," the authors decompose demonstrations into "atomic skill-action pairs" as an intermediate representation. They then use a dual-library approach (dynamic library retrieves by visual/planning similarity; static library uses IDF-weighted tokens to supplement missing skills) to provide the LLM with skill-comprehensive in-context demonstrations, thereby upgrading "trajectory imitation" to "compositional skill reasoning."

## Background & Motivation

**Background**: VLA models (RT-2, OpenVLA, π0, RDT) achieve "robustness to visual perturbations on known tasks" via large-scale robot data training. Recently, X-ICM introduced in-context learning (ICL) to cross-task zero-shot robotics, using dynamics-guided retrieval to select similar demos from a training pool for LLM-based action prediction.

**Limitations of Prior Work**: (a) X-ICM requires training a dynamics retriever on a specific task distribution, limiting cross-domain transfer; (b) Demos fed to the LLM are only low-level numerical action sequences, lacking causal and procedural information ("what/why is this step, and how does it relate to the next?"); (c) As a result, the LLM degenerates into "trajectory pattern matching"—unable to reason about new skill combinations.

**Key Challenge**: "Cross-task" requires skills to be composable and reasoned about, but existing demo representations only provide low-level continuous actions, exposing no skill structure. Demo retrieval based solely on visual/dynamics similarity may miss key skill patterns needed for new tasks.

**Goal**: (1) Decompose opaque continuous action sequences into "atomic skill label + action" pairs as an intermediate representation; (2) Ensure the demo set is both "task-relevant" and "skill-coverage complete"; (3) Achieve this in a fully training-free manner (using general pre-trained vision encoders, planning agents, and LLMs).

**Key Insight**: Elevate "cross-task transfer" from "trajectory shape similarity" to "composable skill structure"—in-context demos for the LLM must explicitly annotate verb-arg atomic skills to trigger compositional reasoning.

**Core Idea**: Decompose (break demos into atomic skill-action pairs) + Recompose (use dynamic + static dual libraries to assemble skill-complete demo sets for new tasks).

## Method

### Overall Architecture
A training-free pipeline with four modules:  
(1) **Atomic Skills Collection**: For each seen demo, extract keyframes + use VLM to label verb-arg + gripper constraints + rule-based post-processing, yielding $\{(s_k,a_k)\}$;  
(2) **Dynamic Demonstrations Library**: Use DINOv3 for scene visual similarity + planner-predicted skill sequence for Jaccard similarity (verb set + bigram chain), fuse and rank to select top-$k_\mathrm{sim}$ as $\mathcal D_\mathrm{dyn}$;  
(3) **Coverage-aware Static Library**: For each demo, extract object-agnostic tokens (V:verb + B:bigram), use IDF-weighted selection to fill the coverage gap in $\mathcal D_\mathrm{dyn}$, yielding $\mathcal D_\mathrm{cov}$;  
(4) **Skill-Augmented ICL**: Feed $\mathcal D=\mathcal D_\mathrm{dyn}\cup\mathcal D_\mathrm{cov}$ and the query to the LLM for compositional skill reasoning, outputting a 7-DoF discrete action sequence.

### Key Designs

1. **Atomic Skills Collection (Decomposition)**:

    - **Function**: Converts each demo segment into interpretable, composable skill-action pairs.
    - **Mechanism**: Keyframes are extracted via three rules: gripper state changes, joint velocity thresholds, and episode termination. Each segment is labeled by VLM as $\mathrm{Verb}[\mathrm{obj}]$ or $\mathrm{Verb}[\mathrm{obj}_1,\mathrm{obj}_2]$ (Verb ∈ {Reach, Move, Grasp, Release, ...}); **gripper hard constraints**: open→closed enforces Grasp, closed→open enforces Release, preventing VLM mislabeling; rule-based post-processing enforces (movable, target) parameter order and downgrades relational actions under open gripper to Move.
    - **Design Motivation**: Low-level numerical actions lack semantics and cannot be reused across tasks; verb-arg labels allow the LLM to treat demos as "sentence fragments" for composition. Gripper constraints leverage physical priors to minimize annotation errors.

2. **Dual-Library Demonstration Retrieval (Recomposition—Relevance + Coverage)**:

    - **Function**: Ensures both "task relevance" (dynamic library) and "skill coverage completeness" (static library).
    - **Mechanism**: Dynamic library ranking score $s_i=\alpha\tilde s_i^\mathrm{vis}+(1-\alpha)s_i^\mathrm{plan}$, where visual similarity $s_i^\mathrm{vis}=\mathbf f^q\cdot \mathbf f_i$ (DINOv3 cosine), planning similarity $s_i^\mathrm{plan}=\lambda J(\mathcal V(\hat{\mathcal S}),\mathcal V(\mathcal S_i))+(1-\lambda)J(\mathcal B(\hat{\mathcal S}),\mathcal B(\mathcal S_i))$ (Jaccard over verb set + verb-bigram set). Static library describes each demo with object-agnostic tokens $\mathcal T(d)=\{\mathrm{V:}v\}\cup\{\mathrm{B:}v_1\to v_2\}$; token weight $w_t=(\log\frac{N+1}{\mathrm{df}(t)+1}+1)^\beta$ (IDF); selection score $=\sum_{t\in \mathcal T(d)\setminus\mathcal C}w_t / (1+\gamma|\mathcal S_d|)$ (coverage gain penalized by demo length). At inference, compute coverage gap $\mathcal G=\mathcal T(\hat{\mathcal S})\setminus \cup_{d\in\mathcal D_\mathrm{dyn}}\mathcal T(d)$, greedily select up to $k_\mathrm{cov}$ demos from the static library to fill it.
    - **Design Motivation**: Relying solely on similarity may miss key skills (e.g., for "open microwave, put food, then close door," similar demos may all be "open-put" but miss "close door"); object-agnostic + IDF prioritizes "rare but critical" skills; length penalty prevents long demos from dominating context.

3. **Skill-Augmented In-Context Learning (Reasoning)**:

    - **Function**: Enables the LLM to perform compositional reasoning over the semantic scaffold of skill-action pairs, outputting discretized 7-DoF action sequences.
    - **Mechanism**: Each demo is formatted as a (instruction, atomic skill sequence, action sequence) triplet in the text context; the LLM receives the query's instruction + initial observation (discretized object coordinates + gripper state) + planner-predicted skill sequence, and outputs $\{a_1^q,\ldots,a_T^q\}$ following a "decompose query → recompose from existing skills" paradigm, where each $a_t$ is a 3D voxel index + Euler bin + gripper bit.
    - **Design Motivation**: By explicitly exposing the causal chain ("previous frame Reach[knife], next Grasp[knife], then Move[knife, board]"), the LLM is prompted to reason "I can combine known Reach+Grasp+Move sequences to solve the new task," rather than rigidly matching known trajectory shapes.

### Loss & Training
**Fully training-free**: DINOv3, planning agent, VLM, and LLM all use pre-trained weights with no parameter updates. Hyperparameters $\alpha,\lambda,\beta,\gamma,k_\mathrm{sim},k_\mathrm{cov}$ are chosen empirically.

## Key Experimental Results

### Main Results (AGNOSTOS benchmark cross-task zero-shot, success rate %)

Summarized from paper Table 1, comparing Ours vs X-ICM on representative Level-1/Level-2 tasks:

| Task | X-ICM | Ours | Gain |
|---|---|---|---|
| Micro. (open microwave) | 45.3 | **62.7** | +17.4 |
| Seat | 48.0 | **72.0** | +24.0 |
| LampOff | 58.7 | **67.0** | +8.3 |
| LampOn | 50.7 | **52.3** | +1.6 |
| Fridge | 22.7 | **34.7** | +12.0 |
| Knife | **26.7** | 21.3 | -5.4 |
| Phone | **57.3** | 42.7 | -14.6 |
| Most Level-2 tasks | Mostly 0 | Some improvement | Minor |

Overall: On 23 unseen tasks (13 Level-1 + 10 Level-2), Ours is competitive with or outperforms the typical ICL baseline X-ICM on most and especially multi-step compositional tasks (e.g., Micro., Seat, Fridge). Foundation VLA models (OpenVLA, RDT, π0) and in-domain methods (PerAct, RVT, Sigma-Agent) are generally outperformed.

### Ablation Study (Key variants inferred from the paper)

| Configuration | Phenomenon | Explanation |
|---|---|---|
| Full (Dynamic + Static + atomic skill labels) | Best | Three modules synergize |
| Dynamic only (no Static supplement) | Drops on multi-step tasks | Coverage gap not filled |
| Static only (no Dynamic retrieval) | Weak task relevance | Demos visually/planning mismatched to query |
| Remove atomic skill labels (degrade to X-ICM style action-only demos) | Significant drop | LLM degenerates to trajectory imitation, no skill reasoning |
| VLM annotation without gripper hard constraints | Increased annotation noise | Physical consistency broken |

### Key Findings
- The "atomic skill label + action pair" intermediate representation is the critical leap—exposing this layer to the LLM immediately activates cross-task compositional reasoning; without it, even the best retrieval is just more precise "copy-paste trajectory."
- The synergy of Dynamic + Static dual libraries outperforms either alone, validating that "task relevance" and "skill coverage" are orthogonal requirements.
- IDF weighting ensures rare but critical verbs (e.g., Close, Insert) are prioritized in static library selection, crucial for Level-2 multi-step tasks.

## Highlights & Insights
- **Training-free yet effective**: No need to train any dynamics retriever; all components are pre-trained + rule-based, making cross-domain transfer extremely easy—a major advantage for industrial deployment.
- **Semanticizing the ICL paradigm**: Previous ICL-for-robotics approaches fed only numerical actions; this work insists on semantic tokens, leveraging LLM strengths (symbolic compositional reasoning) in the right way.
- **Gripper hard constraints + rule-based post-processing**: This engineering detail grounds VLM annotation and is a reusable annotation-LLM collaboration paradigm.
- **Object-agnostic verb tokens + IDF demo selection**: A clear "few-shot, maximal coverage" signal, transferable to any domain needing in-context demonstration selection (not just robotics).

## Limitations & Future Work
- On some simple-skill tasks (Knife, Phone), X-ICM outperforms—possibly because atomic skill abstraction is too fine-grained, blurring visual similarity signals.
- The atomic skill vocabulary $\mathcal V$ is a manually defined closed set (Reach/Move/Grasp/Release/...), requiring extension for novel action types (e.g., pour, wipe), lacking an automatic discovery mechanism.
- Planner prediction errors can simultaneously pollute both the plan-similarity in the dynamic library and the coverage gap in the static library—upstream fragility is not quantitatively analyzed.
- The 7-DoF action discretization granularity (voxel + Euler bin) is coarse and may be insufficient for high-precision tasks (threading, screwing).
- Real-world experiments are only briefly mentioned at the end of the paper, with no complete numerical tables; sim-to-real gap is not deeply discussed.

## Related Work & Insights
- **vs X-ICM (Zhou et al. 2025)**: Both use ICL for cross-task, but X-ICM requires training a dynamics retriever and only feeds actions; this work is training-free and feeds skill-action pairs—a direct upgrade.
- **vs RoboPrompt / KAT / InCoRo / Instant Policy**: These focus on within-task; this work is cross-task zero-shot.
- **vs VoxPoser / MOKA / COPA / ReKep**: Modular VLA approaches relying on extensive task-specific prompt engineering; this work only needs a unified atomic-skill schema.
- **vs End-to-End VLA (OpenVLA, π0, RDT, LLARVA, HPT)**: These rely on data scale for cross-task, but AGNOSTOS shows limited effectiveness; this work offers a complementary approach via ICL + skill abstraction.

## Rating
- Novelty: ⭐⭐⭐⭐ The atomic skill intermediate representation + dual-library dual-signal is a novel and convincing combination, providing a clear paradigm for cross-task ICL.
- Experimental Thoroughness: ⭐⭐⭐ AGNOSTOS 23 tasks + real-world validation; ablation and failure analysis are somewhat shallow, sim-to-real not detailed enough.
- Writing Quality: ⭐⭐⭐⭐ Figures 1/2/3 clearly explain concepts; formulas are concise; Table 1 is dense but complete.
- Value: ⭐⭐⭐⭐ Provides a strong training-free baseline for the robot manipulation community, and the atomic-skill abstraction idea is extensible to agents/workflow automation and other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](../../CVPR2026/robotics/learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](../../CVPR2026/robotics/deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->
