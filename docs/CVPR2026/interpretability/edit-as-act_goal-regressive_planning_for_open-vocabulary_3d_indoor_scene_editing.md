---
title: >-
  [Paper Note] Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing
description: >-
  [CVPR 2026][Interpretability][3D scene editing] This paper reframes open-vocabulary 3D indoor scene editing as a goal-regressive planning problem. It introduces EditLang, a PDDL-style symbolic language…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "3D scene editing"
  - "goal regression"
  - "PDDL"
  - "LLM planning"
  - "symbolic reasoning"
date: 2026-05-08
content_hash: 69e6c3ce18c334ee
---

# Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing

**Conference**: CVPR 2026
**arXiv**: [2603.17583](https://arxiv.org/abs/2603.17583)  
**Code**: [GitHub](https://seongraenoh.github.io/edit-as-act/)  
**Area**: Interpretability
**Keywords**: 3D scene editing, goal regression, PDDL, LLM planning, symbolic reasoning

## TL;DR
This paper reframes open-vocabulary 3D indoor scene editing as a goal-regressive planning problem. It introduces EditLang, a PDDL-style symbolic language, and employs an LLM-driven Planner-Validator loop to derive minimal edit sequences by reasoning backward from goal states. Evaluated on 63 editing tasks, the method achieves the best overall balance across instruction fidelity (69.1%), semantic consistency (86.6%), and physical plausibility (91.7%).

## Background & Motivation

**Background**: Three mainstream paradigms exist for 3D indoor scene editing — data-driven layout generation (e.g., DiffuScene/EditRoom using diffusion models), constraint optimization (e.g., Holodeck/AnyHome converting language to spatial constraints), and image editing with 3D lifting (e.g., ArtiScene performing 2D edits followed by 3D reconstruction).

**Limitations of Prior Work**: Each paradigm satisfies only a subset of three critical requirements — instruction fidelity, semantic consistency (preserving unrelated scene elements), and physical plausibility (no collisions or floating objects). Layout generation methods tend to alter the scene globally; constraint optimization may re-optimize broadly, displacing objects outside the edit target; image-based editing lacks 3D reasoning and introduces structural artifacts.

**Key Challenge**: Existing methods treat editing as a generative task — producing the entire scene in a single forward pass — which makes it extremely difficult to guarantee "edit only what is necessary, preserve everything else."

**Goal**: Achieve 3D scene editing that simultaneously satisfies instruction fidelity, semantic consistency, and physical plausibility.

**Key Insight**: Inspired by embodied agents and classical AI planning, this work recasts editing as a goal-satisfaction problem: "a user instruction defines a desired world state, and the edit should be the minimal action sequence that makes that state hold." Reasoning backward from the goal to the current scene naturally enforces minimal editing.

**Core Idea**: Transform scene editing from a *generation problem* to a *planning problem*, leveraging STRIPS-style goal regression to guarantee minimality, verifiability, and physical consistency.

## Method

### Overall Architecture
Given a source 3D scene $S_0$ and a natural language instruction $I$, the system outputs an edited scene $S_T$ via three stages: (1) an LLM translates the instruction into EditLang symbolic goal predicates $G_T$; (2) a Planner-Validator loop reasons backward — the Planner proposes actions satisfying the current goal, the Validator checks four criteria (goal-directedness, monotonicity, contextual consistency, and formal validity), and upon acceptance, source-aware regression updates the goal set; (3) the action sequence is reversed and executed via a Python DSL to apply geometric transformations.

### Key Designs

1. **EditLang Symbolic Editing Language**

    - **Function**: Defines a PDDL-style domain for scene editing, encompassing predicates and actions.
    - **Mechanism**: Predicates encode geometric, topological, and physical relations (e.g., `supported(x,y)`, `contact(x,y)`, `collision(x,y)`, `stable(x)`, `facing(x,y)`). Each action is defined as a triple $\langle \text{pre}(a), \text{add}(a), \text{del}(a) \rangle$, with state transitions given by $s' = (s \setminus \text{del}(a)) \cup \text{add}(a)$. Three operation types are supported: geometric rearrangement, object addition (Add), and appearance modification (Stylize).
    - **Design Motivation**: Mapping free-form text to a structured symbolic space renders the editing process verifiable, interpretable, and compositional. Unlike conventional PDDL, EditLang dynamically binds to concrete objects in the scene, supporting open vocabulary.

2. **Source-Aware Goal Regression**

    - **Function**: Derives the necessary action sequence by reasoning backward from the goal state.
    - **Mechanism**: Classical STRIPS regression repeatedly reasons about already-satisfied conditions. The proposed source-aware variant is formulated as $G_{t-1} = (G_t \setminus \text{add}(a_t)) \cup (\text{pre}(a_t) \setminus S_0)$ — propagating only preconditions not already satisfied in the source scene, automatically skipping those that are.
    - **Design Motivation**: Avoids unnecessary "reconstruction" of already-correct scene elements, ensuring edit minimality — a guarantee that forward generative methods cannot provide.

3. **Planner-Validator Dual-Module Verification**

    - **Function**: The Planner proposes actions; the Validator applies a four-way check before accepting or rejecting each proposal.
    - **Mechanism**: The Validator enforces — (1) *Goal-directedness*: $\text{add}(a_t)$ must satisfy at least one goal in $G_t$; (2) *Monotonicity*: $\text{del}(a_t) \cap G^{\text{sat}}_{\leq t} = \emptyset$, preventing retraction of already-achieved goals; (3) *Contextual consistency*: edit outcomes comply with room-specific constraints; (4) *Formal validity*: conformance to the EditLang schema. Domain invariants (no collisions, single support surface, etc.) are continuously maintained.
    - **Design Motivation**: LLM-generated plans are not guaranteed to be correct; the Validator provides a formal safety net. The monotonicity constraint combined with a finite state space ensures the planning loop is guaranteed to terminate.

### Loss & Training
The method is entirely training-free, relying solely on LLM inference. Both the Planner and Validator are driven by prompting an LLM (e.g., GPT-4). After each action is executed, predicates are recomputed from geometry to keep the symbolic state synchronized with the 3D scene.

## Key Experimental Results

### Main Results

**Average across 9 scene categories in E2A-Bench**

| Method | Instruction Fidelity (IF) ↑ | Semantic Consistency (SC) ↑ | Physical Plausibility (PP) ↑ |
|---|---|---|---|
| LayoutGPT-E | 42.3 | 48.8 | 78.6 |
| AnyHome | 57.6 | 60.5 | 84.5 |
| ArtiScene-E | 48.3 | 51.2 | 90.3 |
| **Edit-As-Act** | **69.1** | **86.6** | **91.7** |

### Ablation Study

| Scene Category | IF | SC | PP | Notes |
|---|---|---|---|---|
| Dining Room | 89.7 | 95.3 | 92.7 | Best-performing category; highly structured layout |
| Kitchen | 55.0 | 92.3 | 93.7 | Lower IF but high SC/PP |
| Bedroom | 45.7 | 73.1 | 91.9 | High layout flexibility leads to lower IF |
| Computer Room | 73.6 | 88.0 | 94.1 | Explicit object relations benefit planning |

### Key Findings
- Edit-As-Act is the only method that achieves top performance across all three metrics simultaneously; competing methods excel on at most one or two.
- Semantic consistency (86.6%) substantially outperforms the second-best AnyHome (60.5%), demonstrating the effectiveness of the goal-regressive minimal-edit strategy.
- Performance is strongest in structured scenes (dining room, computer room) and weaker on IF in flexible-layout scenes (bedroom), indicating that symbolic planning yields greater advantages when constraints are well-defined.
- Physical plausibility (91.7%) marginally surpasses ArtiScene-E (90.3%), as action preconditions explicitly verify collision and support conditions.

## Highlights & Insights
- **Paradigm Shift**: Recasting 3D editing from a generation problem to a planning problem is a fundamental perspective change — once a structured action space and goal regression are in place, minimality, verifiability, and composability of edits follow naturally.
- **LLM as Planner, Not Generator**: Rather than having the LLM directly output edit results, the system uses the LLM to propose actions in symbolic space, subject to formal Validator checks. This "LLM proposal + formal verification" architecture generalizes broadly to other LLM application scenarios.
- **Source-Aware Regression**: A small but critical improvement over classical STRIPS — automatically filtering already-satisfied conditions avoids unnecessary reasoning and editing.

## Limitations & Future Work
- Full reliance on LLM reasoning may lead to planning errors for highly complex, multi-step edits.
- E2A-Bench comprises only 63 tasks, limiting scale, and evaluation relies primarily on LVLM scoring.
- While the EditLang predicate set covers the main spatial relations, its expressiveness is limited for fine-grained relations (e.g., "50 cm from the wall").
- Continuous optimization tasks (e.g., "make the room feel more spacious") involving vague instructions are not supported.

## Related Work & Insights
- **vs. LayoutGPT**: LayoutGPT generates layouts directly via forward LLM inference without any verification mechanism, yielding IF = 42.3, far below the proposed method's 69.1.
- **vs. AnyHome**: Constraint optimization achieves reasonable physical plausibility (84.5) but poor semantic consistency (60.5), as re-optimization displaces objects outside the edit target.
- **vs. ArtiScene**: Image editing with 3D lifting performs adequately on PP (90.3) but weakly on IF (48.3) and SC (51.2), as 2D operations cannot guarantee 3D consistency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Importing classical AI planning (STRIPS/PDDL) into 3D scene editing represents a highly creative paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐ The benchmark is limited in scale (63 tasks) and evaluation depends on LVLM scoring.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation, formal definitions, and method design are presented in a clear, well-structured progression.
- Value: ⭐⭐⭐⭐ The combination of LLM and symbolic planning offers important insights for embodied AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[ICLR 2026\] Internal Planning in Language Models: Characterizing Horizon and Branch Awareness](../../ICLR2026/interpretability/internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)
- [\[ICML 2026\] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](../../ICML2026/interpretability/a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)
- [\[ICLR 2026\] PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions](../../ICLR2026/interpretability/posh_using_scene_graphs_to_guide_llms-as-a-judge_for_detailed_image_descriptions.md)
- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](../../ICLR2026/interpretability/salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)

</div>

<!-- RELATED:END -->
