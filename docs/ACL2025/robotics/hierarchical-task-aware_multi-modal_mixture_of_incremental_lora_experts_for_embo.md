---
title: >-
  [Paper Note] Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning
description: >-
  [ACL 2025][Robotics][Embodied Continual Learning] This paper proposes a Hierarchical Embodied Continual Learning (HEC) setting, which divides agent learning into high-level instructions and low-level actions. It designs the Task-aware MoILE method—which automatically identifies tasks through cross-modal clustering, selects LoRA experts using dual routers, and retains past knowledge via SVD orthogonal training. It reduces the forgetting rate to 3.37% across 5 incremental learn…
tags:
  - "ACL 2025"
  - "Robotics"
  - "Embodied Continual Learning"
  - "MoE-LoRA"
  - "SVD Orthogonal Training"
  - "Task Clustering"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: a1309d806a5926d9
---

# Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning

**Conference**: ACL 2025  
**arXiv**: [2506.04595](https://arxiv.org/abs/2506.04595)  
**Code**: None  
**Area**: Robotics / Continual Learning  
**Keywords**: Embodied Continual Learning, MoE-LoRA, SVD Orthogonal Training, Task Clustering, Catastrophic Forgetting

## TL;DR
This paper proposes a Hierarchical Embodied Continual Learning (HEC) setting, which divides agent learning into high-level instructions and low-level actions. It designs the Task-aware MoILE method—which automatically identifies tasks through cross-modal clustering, selects LoRA experts using dual routers, and retains past knowledge via SVD orthogonal training. It reduces the forgetting rate to 3.37% across 5 incremental learning scenarios (vs. 7.44% for the previous SOTA).

## Background & Motivation
**Background**: Continual learning in embodied AI primarily focuses on executing low-level actions based on human instructions, neglecting the continual learning capability of high-level planning. As LLMs empower agents with stronger autonomous decision-making capabilities, it is necessary to concurrently and continually learn high-level instructions (task decomposition) and low-level actions (execution).

**Limitations of Prior Work**: (a) Existing settings only consider behavioral/environmental increments of low-level actions, without involving increments in high-level instructions; (b) Regularization methods such as EWC and CAMA still suffer from high forgetting rates (~10-11%); (c) In practice, task IDs are unavailable, requiring automatic identification of task types.

**Key Challenge**: How to enable multi-modal agents to continually learn skills at different levels simultaneously without task IDs, while avoiding forgetting?

**Goal**: Define a hierarchical continual learning setting + design a continual learning method independent of task IDs.

**Key Insight**: Replace task IDs with vision-language joint embeddings, and protect historical knowledge using SVD-decomposed LoRA.

**Core Idea**: Cross-modal clustering for task identification + dual-router MoE-LoRA for selecting experts + SVD orthogonal constraints to prevent forgetting.

## Method

### Overall Architecture
The inputs consist of goal conditions and scene images, which are encoded by CLIP to obtain vision-language joint embeddings. (1) CTC (Cross-modal Task Clustering) assigns the embeddings to the nearest cluster centers, outputting task embeddings $e_i$; (2) The token-level router selects the top-$K$ token-level LoRA experts based on the hidden-state input $x$, while the task-level router selects the top-1 task-level LoRA expert based on the task embedding $e$; (3) Incremental LoRA performs SVD on previously trained LoRA weights, freezes the principal components, and orthogonally trains the residual space.

### Key Designs

1. **Cross-modal Task Clustering (CTC)**:

    - **Function**: Automatically determine which task type the input belongs to through clustering, without requiring task IDs.
    - **Mechanism**: CLIP encodes images and texts into unified embeddings $x^m$, which are clustered using k-means. The cluster centers are dynamically updated for each batch via $c_j^{new} = c_j^{old} + \frac{\alpha}{|S_j^{batch}|}\sum(x^m_i - c_j^{old})$.
    - **Design Motivation**: Real-world scenarios lack class labels or task IDs. It is necessary to automatically identify task types to properly route experts.

2. **Dual-Router MoE-LoRA (Token-level + Task-level)**:

    - **Function**: Two types of routers collaboratively select LoRA experts.
    - **Mechanism**: $\Delta Wx = \sum G_1(x)_i \cdot E_i(x) + \sum G_2(e)_i \cdot E_i^h(x)$, where the token-level router selects top-$K$ experts to process fine-grained semantics, and the task-level router selects top-1 expert to distinguish high-level/low-level tasks.
    - **Design Motivation**: Tasks share semantic similarities (e.g., pick and place), requiring routing of different granularities.

3. **SVD Orthogonal Incremental LoRA (Incremental LoRA)**:

    - **Function**: Perform SVD on the trained LoRA parameters of old tasks, freeze the principal components, and orthogonally train the residuals.
    - **Mechanism**: $BA = U\Sigma V^T$. The principal components corresponding to the top $r$ singular values (representing historical knowledge) are retained, while new tasks are trained orthogonally only within the residual space.
    - **Design Motivation**: Continual training directly on original LoRA weights overwrites historical knowledge. SVD decomposition combined with orthogonal constraints ensures that old and new knowledge do not interfere with each other.

### Loss & Training
Standard next-token prediction loss + routing load-balancing loss + SVD orthogonal constraint loss.

## Key Experimental Results

### Main Results

| Method | LB (Low-level Behavior) AA↑ | LB FM↓ | HB (High-level Behavior) AA↑ | HB FM↓ |
|------|------------------|--------|------------------|--------|
| Task-aware MoILE | **67.91** | **3.37** | **55.66** | **2.67** |
| InfLoRA | 65.61 | 7.44 | 54.28 | 5.67 |
| O-LoRA | 64.61 | 8.39 | 53.22 | 6.82 |
| MoELoRA | 63.35 | 10.25 | 51.60 | 7.68 |
| EWC | 62.44 | 11.49 | 51.55 | 10.83 |

### Ablation Study

| Configuration | AA↑ | FM↓ | Description |
|------|-----|-----|------|
| Task-aware MoILE (Full) | 67.91 | 3.37 | All components |
| w/o CTC | ~65 | ~5 | Without task clustering |
| w/o SVD Orthogonal | ~64 | ~8 | Without incremental constraints |
| w/o Task-level Routing | ~66 | ~4 | Token-level only |

### Key Findings
- **Forgetting Rate Halved**: FM drops from 7.44% in InfLoRA to 3.37%, demonstrating the significant effectiveness of SVD orthogonal training.
- **Increment of High-level Instructions is Harder**: All methods exhibit lower performance on HB/HE compared to LB/LE, indicating that continual learning of high-level planning is more challenging.
- **Feasibility of Task Clustering**: CTC achieves effective routing without ground-truth task IDs, holding practical significance for real-world deployments.
- **Hybrid Hierarchical is the Hardest**: Cross-level hierarchical incremental learning (HH) is the most challenging among the five settings.

## Highlights & Insights
- **The Hierarchical Continual Learning Setting is a Novel Contribution**: Extending embodied continual learning from mere low-level actions to a dual-level structure of high-level instructions + low-level actions is closer to the practical needs of agents in the LLM era.
- **SVD Orthogonal Training of LoRA**: This approach cleverly leverages the low-rank structure of LoRA, achieving parameter space division via SVD decomposition, freezing, and orthogonal constraints. This concept is generalisable to any LoRA-based continual learning scenarios.
- **Task Identification Without Task IDs**: Realized through multimodal embedding clustering, this is more practical than methods that assume prior knowledge of task IDs.

## Limitations & Future Work
- The evaluation environment is ALFRED simulation; validation on real physical robots remains insufficient.
- The number of clusters needs to be predefined, and adaptive expansion is not discussed.
- The composition relationship and difficulty gradient analysis of the five settings could be more thoroughly explored.
- Direct comparison against more complex lifelong learning frameworks (e.g., DRAE) is missing.

## Related Work & Insights
- **vs InfLoRA**: Both are LoRA-based continual learning methods, but InfLoRA lacks task-aware routing, yielding FM = 7.44% vs 3.37%.
- **vs O-LoRA**: O-LoRA utilizes orthogonal constraints but lacks MoE selection; the proposed method applies SVD in a more fine-grained manner under an MoE framework.
- **vs EWC**: Regularization methods sustain FM > 10%, showing limited efficacy in complex embodied scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The HEC setting is a novel contribution, and the SVD-MoE-LoRA combination is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 settings × 3 order variations × multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and systematic description of methodology.
- Value: ⭐⭐⭐⭐ The hierarchical continual learning setting is inspiring, and the SVD orthogonal LoRA is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] M³E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts](../../ICLR2026/robotics/m3e_continual_vision-and-language_navigation_via_mixture_of_macro_and_micro_expe.md)
- [\[ACL 2025\] DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics](drae_dynamic_retrieval-augmented_expert_networks_for_lifelong_learning_and_task_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/geco-srt_geometry-aware_continual_adaptation_for_cross-task_sim-to-real_transfer.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](../../ICCV2025/robotics/rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](../../NeurIPS2025/robotics/himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)

</div>

<!-- RELATED:END -->
