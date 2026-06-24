---
title: >-
  [Paper Note] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning
description: >-
  [NeurIPS 2025][Robotics][PDDL] UniDomain pretrains a unified PDDL planning domain—comprising 3,137 operators and 2,875 predicates—from 12,393 real-world robotic manipulation videos. Through hierarchical fusion to construct a meta-domain, it achieves zero-shot cross-task symbolic planning, outperforming the strongest baseline by 58% in success rate and 160% in plan optimality.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "PDDL"
  - "task planning"
  - "robotic manipulation"
  - "knowledge distillation"
  - "large-scale demonstration learning"
date: 2026-05-08
content_hash: b50e35ca15448ed8
---

# UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning

**Conference**: NeurIPS 2025
**arXiv**: [2507.21545](https://arxiv.org/abs/2507.21545)  
**Code**: [https://roboticsjtu.github.io/UniDomain/](https://roboticsjtu.github.io/UniDomain/)  
**Area**: Robotics / Task Planning
**Keywords**: PDDL, task planning, robotic manipulation, knowledge distillation, large-scale demonstration learning

## TL;DR

UniDomain pretrains a unified PDDL planning domain—comprising 3,137 operators and 2,875 predicates—from 12,393 real-world robotic manipulation videos. Through hierarchical fusion to construct a meta-domain, it achieves zero-shot cross-task symbolic planning, outperforming the strongest baseline by 58% in success rate and 160% in plan optimality.

## Background & Motivation

Robot task planning requires reasoning over implicit constraints embedded in natural language instructions and visual observations. For example, "sort blocks by parity and arrange them in ascending order" implies a long-horizon dependency involving unstacking, sorting, and placing; "make a cup of tea" requires a sequence of prerequisite steps such as opening a cabinet, finding a cup, and boiling water. These tasks demand structured reasoning over action preconditions, temporal dependencies, and physical constraints.

**Limitations of Prior Work**:
- Direct LLM/VLM planning (e.g., Code-as-Policies, ReAct): strong commonsense priors but unable to accurately model action pre- and post-conditions, leading to errors in long-horizon planning.
- LLM + PDDL hybrid methods (e.g., LLM+P, ISR-LLM): rely on hand-crafted PDDL domains or LLM-generated domains from language, resulting in limited domain quality.
- Learning PDDL domains from demonstrations: existing work learns narrow domains from single or few demonstrations, requiring task-specific priors.

**Key Challenge**: High-quality PDDL domains are critical for symbolic planning, yet hand-crafting them is costly and poorly generalizable; LLM-generated domains are of insufficient quality; domains learned from few demonstrations have too narrow coverage.

**Key Insight**: Drawing on the "pretraining–post-training–inference" paradigm of foundation models, UniDomain pretrains a general-purpose PDDL domain from a large-scale robotic manipulation dataset (DROID), then performs "post-training" via domain fusion to adapt to specific task categories.

## Method

### Overall Architecture

UniDomain consists of three stages:
1. **Domain Pretraining**: Extracts atomic domains from video demonstrations to construct a unified domain.
2. **Domain Fusion**: Retrieves relevant atomic domains and hierarchically fuses them into a meta-domain.
3. **Online Planning**: Constructs a PDDL problem from the meta-domain and solves it.

### Key Designs

1. **Energy-Based Keyframe Extraction**: A simple and efficient domain-free keyframe extraction method is proposed. The pixel energy of grayscale frames is computed as $E(I_t) = \sum_{i,j} I_t(i,j)^2$, and local extrema of the energy sequence are detected via a sliding window to select keyframes. Compared to CLIP/SigLIP embedding-based methods, processing speed improves from 47.8 s/video to 0.6 s/video, with higher accuracy (28% vs. 15% single-attempt success rate).

2. **Closed-Loop Atomic Domain Generation**: Given a keyframe sequence and task instruction, a VLM infers the operator (preconditions and effects) for each frame transition, followed by LLM-based global revision to ensure syntactic correctness and predicate consistency. Two-level nested validation is then applied:

    - **Solvability Check**: The LLM generates $K=5$ test problems; the PDDL solver checks solvability, with solvability score $S(D_r) = \frac{1}{K}\sum_k \mathbb{I}[\text{solver solves } P_k]$ and threshold $\theta=0.6$.
    - **Solution Validation**: A separate LLM verifies whether the solution to the hardest test problem satisfies physical and commonsense constraints.

   Both checks iterate up to $L=5$ times.

3. **Hierarchical Binary-Tree Domain Fusion**: Retrieved atomic domains are recursively merged along a binary tree. Each fusion comprises two steps:

    - **Predicate Merging**: Cosine similarity of semantic embeddings is computed (threshold $\tau_p=0.3$); an LLM verifies semantic equivalence before merging.
    - **Operator Merging**: Similarly, name embedding similarity is computed (threshold $\tau_o=0.3$); functionally equivalent operators are merged, inheriting the union of preconditions and effects.

   Hierarchical fusion avoids the structural errors introduced by direct LLM-based merging.

4. **Task-Relevant Filtering**: During online planning, a VLM first generates an initial PDDL problem to extract a relevant predicate set $P_0$; relevant operators are then retrieved from the meta-domain as $O' = O_\text{pre} \cup O_\text{eff}$, forming a compact domain $D_\text{new}$ for planning.

### Loss & Training

No conventional neural network training is involved. The core "training" consists of iterative PDDL domain quality optimization through closed-loop LLM/VLM validation. Evaluation metrics include the solvability score $S(D)$, solution validation pass rate, and downstream task success rate.

## Key Experimental Results

### Main Results

| Method | Type | SR↑ | SPL↑ | OR (K=0)↑ |
|--------|------|-----|------|-----------|
| Code-as-Policies | Direct LLM | 51% | — | Low |
| ReAct | LLM + Feedback | Higher | — | Low |
| VLM-CoT | VLM | Medium | — | Medium |
| ISR-LLM | LLM + PDDL | Highest (among baselines) | — | Low |
| BoN-iVML | LLM + PDDL | Medium | — | Medium |
| **UniDomain** | **Pretrained Domain + PDDL** | **85%** | **Highest** | **83%** |

UniDomain surpasses the strongest baseline by 58% in success rate and achieves a 160% improvement in plan optimality. Optimal plans are produced for 83% of tasks.

### Ablation Study

| Configuration | SR | Key Observation |
|---------------|----|-----------------|
| Full UniDomain | 85% | Best performance |
| w/o Closed-Loop Validation (w/o CL) | Significant drop | Single-pass LLM domain generation yields poor quality |
| w/o Domain Fusion | 19% | Individual atomic domains fail to generalize compositionally |
| w/o Structured Fusion | 0% (syntax errors) | Direct LLM merging fails completely |
| w/o Predicate Grouping | Drop (especially composite domains) | LLM struggles with flat predicate lists |
| w/o Operator Filtering | Drop (especially block domains) | Irrelevant symbols interfere with long-horizon reasoning |

### Key Findings

- The three-stage pipeline of pretrained unified domain + domain fusion + online filtering is critical; removing any stage causes significant performance degradation.
- Energy-based keyframe extraction is 80× faster than vision-model-based methods and achieves higher accuracy.
- Direct LLM merging of multiple domains introduces structural errors; hierarchical fusion is necessary.
- 83% of tasks are solved with optimal plans by composing previously learned operators, demonstrating strong compositional generalization.
- UniDomain requires the fewest LLM calls and least reasoning time among top-performing methods.

## Highlights & Insights

- **Field First**: The first framework to pretrain a general-purpose PDDL domain from large-scale real-world demonstrations, analogous to the pretraining paradigm of LLMs.
- **Elegantly Designed Closed-Loop Validation**: Uses the PDDL solver itself as a domain quality verifier, requiring no human feedback.
- **Compositional Generalization**: Solves complex long-horizon tasks by composing independently learned manipulation behaviors (e.g., pick, pour, stir).
- **Practical Applicability**: Generated plans can be directly converted into natural language instructions for execution by VLA models.

## Limitations & Future Work

- Automatically retrieved atomic domains may be redundant, making meta-domain construction time-consuming.
- Only PDDL 1.0 is supported, lacking temporal constraints, numeric fluents, and cost-sensitive planning.
- Experiments assume full observability; occlusion and perceptual noise are not addressed.
- Evaluation uses human teleoperation as low-level control; end-to-end validation of a complete robotic system is absent.
- Keyframe extraction relies on simple pixel energy, which may miss semantically significant subtle changes.

## Related Work & Insights

UniDomain elegantly transfers the pretraining–post-training–inference paradigm of the LLM era into symbolic planning. Unlike methods such as ISR-LLM and NL2Plan that generate domains directly from language, UniDomain acquires grounded manipulation knowledge from visual demonstrations. Unlike methods such as BLADE that learn narrow domains from individual demonstrations, UniDomain learns a unified domain covering a broad task space from large-scale data. The domain fusion approach resembles knowledge graph merging but is specifically designed for the structured properties of PDDL.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First paradigm for pretraining PDDL domains from large-scale demonstrations; a uniquely insightful perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 100 real-world tasks with thorough ablations; end-to-end robotic validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; the three-stage structure is easy to follow.
- Value: ⭐⭐⭐⭐⭐ Provides a scalable new paradigm for robot task planning with significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[NeurIPS 2025\] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [\[ICCV 2025\] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](../../ICCV2025/robotics/bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)

</div>

<!-- RELATED:END -->
