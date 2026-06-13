---
title: >-
  [Paper Note] Contrastive Representations for Temporal Reasoning
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Contrastive Learning] This paper proposes CRTR (Contrastive Representations for Temporal Reasoning), which introduces intra-trajectory negative pairs by repeating trajectory IDs w…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Contrastive Learning"
  - "Temporal Reasoning"
  - "Combinatorial Problems"
  - "Sokoban"
  - "Rubik's Cube"
date: 2026-05-08
content_hash: 9d604af71b048234
---

# Contrastive Representations for Temporal Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2508.13113](https://arxiv.org/abs/2508.13113)  
**Code**: [GitHub](https://github.com/Princeton-RL/CRTR)  
**Area**: Self-Supervised Learning / Representation Learning
**Keywords**: Contrastive Learning, Temporal Reasoning, Combinatorial Problems, Sokoban, Rubik's Cube

## TL;DR

This paper proposes CRTR (Contrastive Representations for Temporal Reasoning), which introduces intra-trajectory negative pairs by repeating trajectory IDs within training batches. This eliminates the reliance on static contextual features in standard temporal contrastive learning, enabling representations that reflect temporal structure. CRTR achieves, for the first time, search-free solving on combinatorial reasoning tasks such as the Rubik's Cube.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Combinatorial reasoning problems (e.g., Sokoban, Rubik's Cube) typically require expensive search algorithms (A\*, BestFS) to solve.

### State of the Field

**Background**: Temporal contrastive learning (e.g., CRL) is used to learn state representations, where positive pairs consist of temporally adjacent states within a trajectory and negative pairs are drawn from different trajectories.

### Root Cause

**Key Challenge**: A critical failure mode arises in Sokoban, where different trajectories have different wall layouts (context). CRL exploits these wall layouts rather than temporal structure to distinguish positive from negative pairs, causing all states within the same trajectory to be encoded into nearly identical representations (t-SNE visualizations show trajectories collapsing into tight clusters).

### Starting Point

**Key Insight**: As a result, CRL representations fail to reflect temporal distances between states and are therefore unsuitable for planning.

## Method

### Overall Architecture

The core modification in CRTR is remarkably simple — only the negative sampling strategy in contrastive learning is changed:

1. Trajectory IDs in the training batch are repeated a fixed number of times (repetition factor = 2).
2. This ensures that multiple states from the same trajectory but at different time steps appear in the same batch.
3. Since standard contrastive learning treats other batch elements as negatives, **different temporal states from the same trajectory become negative pairs**.
4. The model can no longer rely on invariant contextual features (e.g., wall positions) to distinguish samples, and is thus forced to encode temporal structure.

### Key Designs

1. **Intra-Trajectory Negative Sampling to Eliminate Context Dependence**:
    - Function: Modifies data sampling so that multiple time steps from the same trajectory appear in a single batch.
    - Mechanism: When negative pairs share the same context (e.g., identical wall layouts), contextual features provide no discriminative signal, compelling the model to encode temporal structure instead.
    - Design Motivation: Theoretically, this objective constitutes a lower bound on the conditional mutual information $I(X;X^+|C)$, equivalent to maximizing $I(X;X^+) - I(X^+;C)$, which resembles adversarial feature learning but requires no adversarial training.

2. **From Idealized to Practical Formulation**:
    - Function: The idealized approach requires knowledge of the context variable $C$ to condition negative sampling; the practical approach requires only repeating trajectory IDs.
    - Mechanism: Repeating trajectory IDs naturally produces negative pairs sharing the same context (different time steps from the same episode), and all negatives are anchored (preventing representational collapse).
    - Design Motivation: In practice, it is impossible to know a priori which features constitute "context" (e.g., upon first encountering a Sokoban board, how does one know walls are immovable?).

### Loss & Training

- Standard InfoNCE loss; the only change is the batch construction procedure (`traj_id = np.repeat(traj_id[:B//R], R, axis=0)`).
- Encoder architecture: 8-layer MLP, hidden dimension 512, representation dimension 64.
- The backward variant of the contrastive loss is used, outperforming the symmetric variant on the Rubik's Cube.
- Adam optimizer, learning rate 0.0003, batch size 512.
- A repetition factor of $R = 2$ yields consistent improvements across all tested environments.

## Key Experimental Results

### Main Results

| Environment | CRL | Supervised | DeepCubeA | **CRTR** |
|---|---|---|---|---|
| Sokoban | ~10% | ~30% | ~35% | **~40%** |
| Rubik's Cube | ~55% | 0% | ~60% | **~63%** |
| 15-Puzzle | ~35% | ~50% | — | **~50%** |
| Lights Out | ~30% | ~10% | — | **~80%** |
| Digit Jumper | ~5% | ~60% | — | **~70%** |

(BestFS search budget: 6,000 nodes)

### Ablation Study

- Repetition factor: $R = 2$ consistently improves performance across all environments; excessively large $R$ causes degradation in certain settings.
- Spearman correlation (representation distance vs. ground-truth step distance): CRTR $> 0.8$ vs. CRL $< 0.4$ (Sokoban).
- Search-free solving: CRTR solves nearly all instances using a greedy policy on 4 out of 5 tasks (100% success on the Rubik's Cube).

### Key Findings

- **Most striking result**: CRTR solves all randomly scrambled Rubik's Cube configurations **without any search** (within 6,000 steps).
- Search-free solutions are longer than optimal (average ~400 steps vs. optimal ~26 steps), yet exhibit emergent behavior resembling human "block-building" strategies.
- CRTR also outperforms CRL under A\* search, demonstrating that improvements are not limited to the greedy setting.
- Average search-free solution length: CRTR 448.7 vs. CRL 1830.3 (Rubik's Cube).

## Highlights & Insights

- **Minimal change, maximal impact**: A single-line code modification (repeating trajectory IDs) enables a leap from complete failure to state-of-the-art performance.
- **Search-free solving of combinatorial problems**: For the first time, arbitrary Rubik's Cube states are efficiently solved using only learned representations, without any external search algorithm.
- The emergent "block-building" strategy mirrors human puzzle-solving behavior.
- The conditional mutual information framework provides an elegant theoretical explanation for why eliminating context improves temporal reasoning.

## Limitations & Future Work

- Search-free solutions are far from optimal (Rubik's Cube ~400 steps).
- Success rates on Sokoban remain low, likely due to irreversible states (deadlocks).
- The method assumes known and deterministic dynamics, limiting applicability to problems with stochastic or unknown transition functions.
- Rubik's Cube state distances nearly always satisfy the triangle inequality with equality, making faithful embedding into Euclidean space difficult.

## Related Work & Insights

- CRL (Eysenbach et al.) serves as the direct foundation; CRTR addresses its critical failure mode in combinatorial domains.
- DeepCubeA learns a heuristic via value iteration; CRTR achieves comparable or superior results using contrastive learning.
- Connection to adversarial feature learning: eliminating context is analogous to adversarially removing invariant features, but without requiring adversarial training.

## Rating

- Theoretical Innovation: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Overall: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] The Complexity of Finding Local Optima in Contrastive Learning](the_complexity_of_finding_local_optima_in_contrastive_learning.md)
- [\[CVPR 2026\] UniGeoCLIP: Unified Geospatial Contrastive Learning](../../CVPR2026/self_supervised/unigeoclip_geospatial_contrastive.md)
- [\[ICLR 2026\] Temporal Slowness in Central Vision Drives Semantic Object Learning](../../ICLR2026/self_supervised/temporal_slowness_in_central_vision_drives_semantic_object_learning.md)

</div>

<!-- RELATED:END -->
