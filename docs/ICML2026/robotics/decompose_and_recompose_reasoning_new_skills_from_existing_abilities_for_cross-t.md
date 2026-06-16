---
title: >-
  [Paper Note] Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation
description: >-
  [ICML 2026][Robotics & Embodied AI][in-context learning] Targeting zero-shot robotic manipulation from training tasks to entirely new tasks, the authors decompose demonstrations into "atomic skill-action" pairs as an intermediate representation. A dual-library mechanism (dynamic retrieval for visual/planning similarity + static library for IDF-weighted completion of missing
tags:
  - ICML 2026
  - Robotics & Embodied AI
  - in-context learning
date: 2026-05-08
content_hash: e6c7fe347809282a
---
# Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation

**Conference**: ICML 2026  
**arXiv**: [2605.01448](https://arxiv.org/abs/2605.01448)  
**Code**: None (unannounced)  
**Area**: Robotics / Cross-Task Generalization / Vision-Language-Action Models  
**Keywords**: Atomic skills, in-context learning, cross-task zero-shot, dynamic/static demonstration dual-library, skill coverage

## TL;DR
Targeting zero-shot robotic manipulation from training tasks to entirely new tasks, the authors decompose demonstrations into "atomic skill-action" pairs as an intermediate representation. A dual-library mechanism (dynamic retrieval for visual/planning similarity + static library for IDF-weighted completion of missing skill tokens) provides LLMs with skill-comprehensive in-context demonstrations, evolving "trajectory imitation" into "compositional skill reasoning."

## Background & Motivation

**Background**: VLA models (RT-2, OpenVLA, π0, RDT) achieve robustness against visual perturbations on known tasks through large-scale training. Recently, X-ICM introduced in-context learning (ICL) to cross-task zero-shot robotics, utilizing dynamics-guided retrieval to select similar demonstrations from training data for direct action prediction by LLMs.

**Limitations of Prior Work**: (a) X-ICM requires **training a specific** dynamics retriever on a task distribution, weakening cross-domain transfer; (b) demonstrations provided to LLMs contain only low-level numerical action sequences, lacking causal and procedural information (e.g., "what is this step doing" or "why is it related to the next"); (c) consequently, the LLM degrades into "trajectory pattern matching," failing to reason when encountering new skill combinations.

**Key Challenge**: Cross-task transfer requires skills to be composable and reason-able. However, existing demonstration representations only provide low-level continuous actions, which do not expose skill structures. Furthermore, retrieval based solely on visual or dynamics similarity may overlook key skill patterns necessary for solving new tasks.

**Goal**: (1) Deconstruct opaque continuous action sequences into "atomic skill label + action" pairs as intermediate representations; (2) ensure the demonstration set is both "task-relevant" and "skill-comprehensive"; (3) maintain a completely training-free pipeline (using general pre-trained visual encoders + planning agents + LLMs).

**Key Insight**: Elevate cross-task transfer from the "trajectory shape similarity" level to the "composable skill structure" level. Explicitly labeling verb-arg atomic skills in in-context demonstrations is essential to trigger compositional reasoning in LLMs.

**Core Idea**: Decompose (demos into atomic skill-action pairs) + Recompose (using dynamic + static dual-libraries to assemble skill-complete demo sets for new tasks).

## Method

### Overall Architecture
The method consists of three tightly coupled components in a training-free pipeline: (1) **Atomic Skills Collection**: Extraction of keyframes from seen demos + VLM-based verb-arg labeling + gripper constraints + rule-based post-processing to obtain skill-action pairs $\{(s_k,a_k)\}$; (2) **Dual-Library Retrieval**: A dynamic library uses DINOv3 for visual similarity + Jaccard similarity of planner-predicted skill sequences (verb sets + bigram chains) to select top-$k_\mathrm{sim}$ demos for $\mathcal D_\mathrm{dyn}$; a static library then extracts object-agnostic tokens (V:verb + B:bigram) and uses IDF-weighted selection to fill the coverage gap of $\mathcal D_\mathrm{dyn}$, forming $\mathcal D_\mathrm{cov}$; (3) **Skill-Augmented ICL**: The combined set $\mathcal D=\mathcal D_\mathrm{dyn}\cup\mathcal D_\mathrm{cov}$ and the query are fed to the LLM for compositional skill reasoning to output 7-DoF discrete action sequences.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    SEEN["Seen Task Demos"] --> A["Atomic Skill Deconstruction<br/>Keyframes + VLM Verb-Arg Labeling<br/>Gripper Constraints + Post-processing"]
    A --> POOL["Skill-Action Pair Candidate Pool"]
    Q["Unseen Task Query<br/>Instruction + Initial Observation"] --> PLAN["Planning Agent Predicts Skill Sequence"]
    Q --> VIS["DINOv3 Scene Visual Features"]
    subgraph DUAL["Dual-Library Retrieval (Recomposition)"]
        direction TB
        DYN["Dynamic Library<br/>Visual Similarity + Plan Jaccard Fusion<br/>Top-k_sim → D_dyn"]
        COV["Static Library<br/>IDF-weighted Coverage Gap Filling<br/>Greedy ≤ k_cov → D_cov"]
        DYN --> COV
    end
    POOL --> DUAL
    PLAN --> DUAL
    VIS --> DYN
    DUAL --> ICL["Skill-Augmented ICL (Reasoning)<br/>D=D_dyn∪D_cov + Query for LLM Reasoning"]
    ICL --> OUT["7-DoF Discrete Action Sequence Output"]
```

### Key Designs

**1. Atomic Skills Collection: Decomposing Demos into Composable Skill-Action Pairs**
Low-level numerical actions lack semantics and cannot be reused across tasks. This step converts demos into labeled atomic skills. Keyframes are extracted via three rules: gripper state changes, joint velocity thresholds, and episode termination. Each segment is labeled by a VLM as $\mathrm{Verb}[\mathrm{obj}]$ or $\mathrm{Verb}[\mathrm{obj}_1,\mathrm{obj}_2]$ (where Verb $\in$ {Reach, Move, Grasp, Release, ...}). A critical engineering detail is the use of hard gripper constraints: open→closed is forced as Grasp, and closed→open as Release, using physical common sense to minimize VLM labeling errors. Rule-based post-processing enforces (movable, target) argument order and demotes relational actions in an open-gripper state to Move. With verb-arg labels, the LLM treats demos as "sentence fragments" to be composed rather than meaningless digits.

**2. Dual-Library Demonstration Retrieval: Balancing Task Relevance and Skill Coverage**
Retrieval based only on visual/dynamics similarity might miss critical skills (e.g., when solving "open microwave → put food → close door," similar demos might only show "open" and "put"). The dynamic library handles "Task Relevance" with a ranking score $s_i=\alpha\tilde s_i^\mathrm{vis}+(1-\alpha)s_i^\mathrm{plan}$, where $s_i^\mathrm{vis}$ uses DINOv3 cosine similarity and $s_i^\mathrm{plan}$ uses Jaccard similarity of verb and verb-bigram sets. The static library handles "Skill Coverage," where each demo is described by object-agnostic tokens $\mathcal T(d)=\{\mathrm{V:}v\}\cup\{\mathrm{B:}v_1\to v_2\}$. Tokens are weighted by IDF $w_t=(\log\frac{N+1}{\mathrm{df}(t)+1}+1)^\beta$, and selection is based on coverage gain divided by a length penalty $\sum_{t\in \mathcal T(d)\setminus\mathcal C}w_t / (1+\gamma|\mathcal S_d|)$. During inference, the coverage gap $\mathcal G=\mathcal T(\hat{\mathcal S})\setminus \cup_{d\in\mathcal D_\mathrm{dyn}}\mathcal T(d)$ is calculated, and demos are greedily added from the static library. IDF ensures rare but critical skills (e.g., Close, Insert) are prioritized.

**3. Skill-Augmented In-Context Learning: Compositional Reasoning on a Skill Scaffold**
Semantic-rich demos allow the LLM to perform composition rather than pattern matching. Each demo is formatted as an (instruction, atomic skill sequence, action sequence) triplet. The LLM receives the query instruction, initial observations (discretized coordinates + gripper state), and the predicted skill sequence. Following a "decompose query → recompose from existing skills" paradigm, it outputs $\{a_1^q,\ldots,a_T^q\}$, where $a_t$ represents 3D voxel indices + Euler bins + gripper bits. By explicitly seeing causal chains like "Reach[knife] → Grasp[knife] → Move[knife, board]," the LLM can trigger reasoning to solve new tasks using known skill components.

### Loss & Training
**Training-free**: DINOv3, the planning agent, VLM, and LLM all utilize pre-trained weights without parameter updates. Hyperparameters $\alpha,\lambda,\beta,\gamma,k_\mathrm{sim},k_\mathrm{cov}$ are selected empirically.

## Key Experimental Results

### Main Results (AGNOSTOS Benchmark Cross-Task Zero-Shot, Success Rate %)

Comparison between Ours and X-ICM on selected Level-1/Level-2 tasks:

| Task | X-ICM | Ours | Gain |
|---|---|---|---|
| Micro. (open microwave) | 45.3 | **62.7** | +17.4 |
| Seat | 48.0 | **72.0** | +24.0 |
| LampOff | 58.7 | **67.0** | +8.3 |
| LampOn | 50.7 | **52.3** | +1.6 |
| Fridge | 22.7 | **34.7** | +12.0 |
| Knife | **26.7** | 21.3 | -5.4 |
| Phone | **57.3** | 42.7 | -14.6 |
| Most Level-2 Tasks | ~0 | Improved | Slight |

Across 23 unseen tasks (13 Level-1 + 10 Level-2), Ours outperforms or competes with the ICL baseline X-ICM, particularly in multi-step compositional tasks. Foundation VLAs (OpenVLA, RDT, π0) and In-Domain methods (PerAct, RVT) are generally surpassed.

### Ablation Study

| Configuration | Observation | Explanation |
|---|---|---|
| Full (Dynamic + Static + atomic skill labels) | Best | Synergy of three modules. |
| Dynamic Only (No Static filling) | Performance drop in multi-step tasks | Coverage gap remains unaddressed. |
| Static Only (No Dynamic retrieval) | Weak task relevance | Visual/planning mismatch between demo and query. |
| No Atomic Skill Labels (Pure action demos) | Significant drop | LLM degrades to trajectory imitation without reasoning. |
| VLM labels without gripper constraints | Increased label noise | Physical consistency is violated. |

### Key Findings
- The "atomic skill label + action pair" intermediate representation is the most critical factor; it activates cross-task compositional reasoning in the LLM.
- The synergy of the Dynamic + Static dual-library is superior to either alone, confirming that task relevance and skill coverage are orthogonal requirements.
- IDF weighting prioritizes rare but essential verbs (e.g., Close, Insert), which is crucial for completing complex Level-2 tasks.

## Highlights & Insights
- **Training-free Efficiency**: Eliminates the need for training dynamics retrievers. Components are pre-trained or rule-based, facilitating low-friction cross-domain deployment.
- **Semanticizing the ICL Paradigm**: Unlike prior ICL-for-robot works that provide numerical actions, this approach provides semantic tokens, properly leveraging the LLM's strength in symbolic compositional reasoning.
- **Robust Labeling Paradigm**: Using gripper hardware constraints to guide VLM labeling is a valuable engineering insight for noisy data.
- **Generalizable Retrieval**: The object-agnostic verb token + IDF selection provides a clear "minimal sample, maximal coverage" signal applicable beyond robotics.

## Limitations & Future Work
- Ours is outperformed by X-ICM in simple tasks (e.g., Knife, Phone), suggesting that over-abstraction of atomic skills might blur useful visual similarity signals.
- The atomic skill vocabulary $\mathcal V$ is a manually defined closed set; a mechanism for automatic discovery is needed for new action types (e.g., pour, wipe).
- Errors in the upstream planner affect both dynamic plan-similarity and static coverage gaps.
- Discrete 7-DoF action granularity may be insufficient for high-precision tasks (e.g., threading a needle).
- Real-world experiments lack comprehensive numerical tables; the sim-to-real gap requires further discussion.

## Related Work & Insights
- **vs. X-ICM (Zhou et al. 2025)**: Both target cross-task ICL, but X-ICM requires trained retrievers and uses pure actions. This work is a direct upgrade using training-free components and skill-action pairs.
- **vs. RoboPrompt / InCoRo / Instant Policy**: These focus on within-task adaptation; Ours addresses cross-task zero-shot settings.
- **vs. End-to-end VLAs (OpenVLA, π0, RDT)**: These rely on data scale to handle cross-task scenarios, which shows limited performance on AGNOSTOS. Ours provides a complementary approach via ICL and skill abstraction.

## Rating
- Novelty: ⭐⭐⭐⭐ Atomic skill representation + dual-library signal is a convincing new paradigm for cross-task ICL.
- Experimental Thoroughness: ⭐⭐⭐ Validated on 23 tasks and real environments, though failure case analysis is relatively brief.
- Writing Quality: ⭐⭐⭐⭐ Concepts are clearly explained through figures; formulas are concise.
- Value: ⭐⭐⭐⭐ Provides a strong training-free baseline for the robot manipulation community with a generalizable abstraction idea.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](tapsampling_inference-time_sampling_with_a_task-progress-understanding_verifier_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [\[CVPR 2026\] Action-Sketcher: From Reasoning to Action via Visual Sketches for Robotic Manipulation](../../CVPR2026/robotics/action-sketcher_from_reasoning_to_action_via_visual_sketches_for_robotic_manipul.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](../../CVPR2026/robotics/learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[ICML 2026\] BEAR: Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis](dissecting_embodied_abilities_in_multimodal_language_models_through_skill-level_.md)

</div>

<!-- RELATED:END -->
