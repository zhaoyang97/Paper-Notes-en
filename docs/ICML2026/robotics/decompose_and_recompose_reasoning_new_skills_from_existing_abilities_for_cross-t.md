---
title: >-
  [Paper Note] Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation
description: >-
  [ICML 2026][Robotics][Atomic Skills] Targeting zero-shot robotic manipulation from training tasks to entirely new tasks, the authors decompose demonstrations into "atomic skill-action" pairs as an intermediate representa…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Atomic Skills"
  - "In-Context Learning"
  - "Zero-Shot Cross-Task"
  - "Dynamic/Static Dual-Library"
  - "Skill Coverage"
date: 2026-05-08
content_hash: b729a3a43657d83a
---

# Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation

**Conference**: ICML 2026  
**arXiv**: [2605.01448](https://arxiv.org/abs/2605.01448)  
**Code**: None (Not declared in the paper)  
**Area**: Robotics / Cross-Task Generalization / Vision-Language-Action Models  
**Keywords**: Atomic Skills, In-Context Learning, Zero-Shot Cross-Task, Dynamic/Static Dual-Library, Skill Coverage

## TL;DR
Targeting zero-shot robotic manipulation from training tasks to entirely new tasks, the authors decompose demonstrations into "atomic skill-action" pairs as an intermediate representation. They then employ a dual-library (dynamic library retrieved by visual/planning similarity + static library using IDF weighting to complete missing skill tokens) to provide LLMs with skill-comprehensive in-context demonstrations, upgrading "trajectory imitation" to "compositional skill reasoning."

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models (RT-2, OpenVLA, π0, RDT) achieve robustness against visual perturbations on seen tasks through large-scale robotic data training. Recently, X-ICM introduced In-Context Learning (ICL) into robotic cross-task zero-shot settings, using dynamics-guided retrieval to select similar demonstrations from a pool of training tasks to feed into an LLM for direct action prediction.

**Limitations of Prior Work**: (a) X-ICM requires **training** a dynamics retriever on specific task distributions, weakening cross-domain transferability; (b) the demonstrations fed to the LLM consist only of low-level numerical action sequences, lacking causal and process information regarding "what/why this step exists" and its relationship to the next step; (c) consequently, the LLM degrades into "trajectory pattern matching" and fails to reason when encountering new skill combinations.

**Key Challenge**: Cross-task generalization requires skills to be composable and reason-able. However, existing demonstration representations only provide low-level continuous actions, which do not expose the skill structure. Furthermore, demonstration retrieval based solely on visual or dynamics similarity may omit critical skill patterns required to solve a new task.

**Goal**: (1) Decompose opaque continuous action sequences into "atomic skill label + action" pairs as intermediate representations; (2) ensure the set of demonstrations is both "task-relevant" and "skill-comprehensive"; (3) remain completely training-free (using generic pre-trained vision encoders + planning agents + LLMs).

**Key Insight**: Elevate cross-task transfer from the "trajectory shape similarity" level to the "composable skill structure" level. Providing the LLM with in-context demonstrations explicitly labeled with verb-argument atomic skills can trigger compositional reasoning.

**Core Idea**: Decompose (break demos into atomic skill-action pairs) + Recompose (assemble a skill-complete demo set for new tasks using dynamic + static dual-libraries).

## Method

### Overall Architecture
A training-free pipeline consisting of four modules: (1) Atomic Skills Collection: Extract keyframes from seen demonstrations + label verb-arguments using VLM + apply gripper constraints + rule-based post-processing to obtain $\{(s_k, a_k)\}$; (2) Dynamic Demonstrations Library: Rank demos by fusing visual similarity (DINOv3 encoding) and planning similarity (Jaccard similarity of verb sets and bigram chains), selecting top-$k_{\mathrm{sim}}$ for $\mathcal{D}_{\mathrm{dyn}}$; (3) Coverage-aware Static Library: Extract object-agnostic tokens (V:verb + B:bigram) for each demo, using IDF weighting to select demos that fill the coverage gap in $\mathcal{D}_{\mathrm{dyn}}$, yielding $\mathcal{D}_{\mathrm{cov}}$; (4) Skill-Augmented ICL: Feed $\mathcal{D} = \mathcal{D}_{\mathrm{dyn}} \cup \mathcal{D}_{\mathrm{cov}}$ and the query to an LLM for compositional skill reasoning, outputting a 7-DoF discretized action sequence.

### Key Designs

1.  **Atomic Skills Collection (Deconstruction)**:
    - **Function**: Transform each demonstration into interpretable, composable skill-action pairs.
    - **Mechanism**: Keyframes are extracted based on three rules: gripper state changes, joint velocity thresholds, and episode termination. Each segment is labeled by a VLM as $\mathrm{Verb}[\mathrm{obj}]$ or $\mathrm{Verb}[\mathrm{obj}_1, \mathrm{obj}_2]$ (where $\mathrm{Verb} \in \{\text{Reach, Move, Grasp, Release, ...}\}$). **Hard Gripper Constraints**: "open $\to$ closed" forces a Grasp label, while "closed $\to$ open" forces a Release label to prevent VLM mislabeling. Rule-based post-processing enforces (movable, target) parameter order and downgrades relational actions in an open state to Move.
    - **Design Motivation**: Low-level numerical actions lack semantics and cannot be reused across tasks; verb-arg labels allow the LLM to combine demos like "sentence fragments." Gripper constraints leverage physical common sense to minimize labeling error rates.

2.  **Dual-Library Demonstration Retrieval (Reconstruction - Relevance + Coverage)**:
    - **Function**: Simultaneously satisfy "task relevance" (Dynamic Library) and "skill coverage" (Static Library).
    - **Mechanism**: The ranking score for the dynamic library is $s_i = \alpha \tilde s_i^{\mathrm{vis}} + (1-\alpha) s_i^{\mathrm{plan}}$, where visual similarity $s_i^{\mathrm{vis}} = \mathbf{f}^q \cdot \mathbf{f}_i$ (DINOv3 cosine) and planning similarity $s_i^{\mathrm{plan}} = \lambda J(\mathcal{V}(\hat{\mathcal{S}}), \mathcal{V}(\mathcal{S}_i)) + (1-\lambda) J(\mathcal{B}(\hat{\mathcal{S}}), \mathcal{B}(\mathcal{S}_i))$ (Jaccard similarity of verb sets and verb-bigram sets). In the static library, each demo is described by object-agnostic tokens $\mathcal{T}(d) = \{\mathrm{V:}v\} \cup \{\mathrm{B:}v_1 \to v_2\}$. Token weights are $w_t = (\log \frac{N+1}{\mathrm{df}(t)+1} + 1)^\beta$ (IDF). The selection score is $\sum_{t \in \mathcal{T}(d) \setminus \mathcal{C}} w_t / (1 + \gamma |\mathcal{S}_d|)$ (coverage gain divided by demo length penalty). During inference, the coverage gap $\mathcal{G} = \mathcal{T}(\hat{\mathcal{S}}) \setminus \cup_{d \in \mathcal{D}_{\mathrm{dyn}}} \mathcal{T}(d)$ is calculated, and up to $k_{\mathrm{cov}}$ demos are greedily selected from the static library to fill it.
    - **Design Motivation**: Similarity alone might miss key skills (e.g., when solving "open microwave, put food, close door," similar demos might all be "open-put" but miss "close"). Object-agnostic tokens + IDF prioritize "rare but critical" skills. The length penalty prevents long demos from consuming the context window.

3.  **Skill-Augmented In-Context Learning (Reasoning)**:
    - **Function**: Enable the LLM to perform compositional reasoning on a semantic scaffold of skill-action pairs to output discretized 7-DoF action sequences.
    - **Mechanism**: Each demo is formatted as a triple (instruction, atomic skill sequence, action sequence) in the text context. The LLM receives the query instruction + initial observation (discretized object coordinates + gripper state) + the predicted skill sequence from the planner. It follows a "decompose query $\to$ recompose from existing skills" paradigm to output $\{a_1^q, \ldots, a_T^q\}$, where each $a_t$ is a 3D voxel index + Euler bin + gripper bit.
    - **Design Motivation**: By explicitly showing the LLM a causal chain like "Reach[knife] $\to$ Grasp[knife] $\to$ Move[knife, board]," it triggers the reasoning that "I can combine known Reach+Grasp+Move sequences to solve a new task," rather than trying to match a known trajectory shape.

### Loss & Training
**Completely training-free**: DINOv3, planning agent, VLM, and LLM all use pre-trained weights without parameter updates. Hyperparameters $\alpha, \lambda, \beta, \gamma, k_{\mathrm{sim}}, k_{\mathrm{cov}}$ are chosen empirically.

## Key Experimental Results

### Main Results (AGNOSTOS benchmark Cross-Task Zero-Shot, Success Rate %)

Summary comparison of Ours vs. X-ICM on representative tasks from paper Table 1:

| Task | X-ICM | Ours | Δ |
|---|---|---|---|
| Micro. (open microwave) | 45.3 | **62.7** | +17.4 |
| Seat | 48.0 | **72.0** | +24.0 |
| LampOff | 58.7 | **67.0** | +8.3 |
| LampOn | 50.7 | **52.3** | +1.6 |
| Fridge | 22.7 | **34.7** | +12.0 |
| Knife | **26.7** | 21.3 | -5.4 |
| Phone | **57.3** | 42.7 | -14.6 |
| Most Level-2 Tasks | Mostly 0 | Partial improvement | Slight |

Overall Conclusion: Across 23 unseen tasks (13 Level-1 + 10 Level-2), Ours demonstrates competitive or superior performance compared to the typical ICL baseline X-ICM on most tasks, with significant advantages in multi-step compositional tasks (e.g., Micro., Seat, Fridge). Foundation VLA models (OpenVLA, RDT, π0) and In-Domain methods (PerAct, RVT, Sigma-Agent) are generally outperformed.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|---|---|---|
| Full (Dynamic + Static + atomic skill labels) | Best | Synergy of the three modules |
| Dynamic Library only (no Static completion) | Drop in multi-step tasks | Coverage gap was not filled |
| Static Library only (no Dynamic retrieval) | Weak task relevance | demos and query did not match in vision/plan |
| Remove atomic skill labels (pure action demos) | Significant drop | LLM degrades to trajectory imitation, no skill reasoning |
| VLM labeling without gripper constraints | Increased label noise | Physical consistency was compromised |

### Key Findings
- The intermediate representation of "atomic skill label + action pairs" is the most critical leap. Once this layer is exposed to the LLM, cross-task compositional reasoning is activated; without it, even optimal retrieval results in mere "copy-pasting trajectories."
- The synergy of the Dynamic + Static dual-library outperforms either alone, verifying that "task relevance" and "skill coverage" are two orthogonal requirements.
- IDF weighting ensures that rare but crucial verbs (e.g., Close, Insert) receive high priority in static library selection, which is vital for completing multi-step Level-2 tasks.

## Highlights & Insights
- **Training-free yet Effective**: Does not rely on training any dynamics retrievers. All components are pre-trained or rule-based, minimizing cross-domain transfer friction—a significant advantage for industrial deployment.
- **Correct "Semanticization" of the ICL Paradigm**: Unlike previous ICL-for-robot approaches that use numerical actions, this work provides semantic tokens, properly utilizing the LLM's strength in symbolic compositional reasoning.
- **VLM-LLM Collaboration**: Engineering details like gripper hard constraints and rule-based post-processing make VLM labeling practical, providing a reusable framework for such collaboration.
- **Maximized Coverage with Minimal Samples**: The object-agnostic verb token + IDF selection strategy provides a clear signal for demonstration selection that can be transferred to other domains requiring in-context demonstrations.

## Limitations & Future Work
- On certain tasks with simple skills (Knife, Phone), it is outperformed by X-ICM, potentially because atomic skill abstraction is too fine-grained, blurring visual similarity signals.
- The atomic skill vocabulary $\mathcal{V}$ is a manually defined closed set (Reach/Move/Grasp/Release/...). It lacks an automatic discovery mechanism for new action types (e.g., pour, wipe).
- Errors in the planner contaminate both plan-similarity in the dynamic library and the coverage gap in the static library; this upstream vulnerability is not explicitly quantified.
- The 7-DoF action discretization (voxel + Euler bin) is relatively coarse and may be insufficient for high-precision operations (threading a needle, tightening a screw).
- Real-world experiments are mentioned only at the end without full numerical tables; the sim-to-real gap is not discussed in depth.

## Related Work & Insights
- **vs X-ICM (Zhou et al. 2025)**: Both use ICL for cross-task settings, but X-ICM requires a trained dynamics retriever and uses pure actions; Ours is training-free and uses skill-action pairs, representing a direct upgrade.
- **vs RoboPrompt / KAT / InCoRo / Instant Policy**: These focus on within-task settings; Ours focuses on cross-task zero-shot settings.
- **vs VoxPoser / MOKA / COPA / ReKep**: Modular VLA solutions that rely on task-specific prompt engineering; Ours uses a unified atomic-skill schema.
- **vs End-to-End VLA (OpenVLA, π0, RDT, LLARVA, HPT)**: These rely on data scale for cross-task capabilities, but AGNOSTOS shows limited effectiveness; Ours provides a complementary path via ICL and skill abstraction.

## Rating
- Novelty: ⭐⭐⭐⭐ Atomic skill representation + dual-library signals is a novel and persuasive combination, providing a clear paradigm for cross-task ICL.
- Experimental Thoroughness: ⭐⭐⭐ AGNOSTOS 23 tasks + real-world validation; however, ablation and failure case analyses are somewhat superficial, and sim-to-real details are lacking.
- Writing Quality: ⭐⭐⭐⭐ Figures 1/2/3 clarify concepts well; formulas are concise; Table 1 is dense but complete.
- Value: ⭐⭐⭐⭐ Provides a strong training-free baseline for the robotic manipulation community, and the atomic-skill abstraction approach is extensible to agents and workflow automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](tapsampling_inference-time_sampling_with_a_task-progress-understanding_verifier_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](../../CVPR2026/robotics/learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[ICML 2026\] BEAR: Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis](dissecting_embodied_abilities_in_multimodal_language_models_through_skill-level_.md)

</div>

<!-- RELATED:END -->
