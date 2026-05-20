---
title: >-
  [Paper Note] Efficient Verified Machine Unlearning for Distillation
description: >-
  [NeurIPS 2025][AI Safety][machine unlearning] This paper proposes PURGE, a framework that extends verified unlearning under SISA to the knowledge distillation (KD) setting via teacher–student constituent mapping and an i…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "machine unlearning"
  - "knowledge distillation"
  - "SISA"
  - "data privacy"
  - "ensemble learning"
date: 2026-05-08
content_hash: 2abaa4f23e9e5b66
---

# Efficient Verified Machine Unlearning for Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2503.22539](https://arxiv.org/abs/2503.22539)  
**Code**: N/A  
**Area**: AI Safety / Machine Unlearning
**Keywords**: machine unlearning, knowledge distillation, SISA, data privacy, ensemble learning

## TL;DR
This paper proposes PURGE, a framework that extends verified unlearning under SISA to the knowledge distillation (KD) setting via teacher–student constituent mapping and an incremental multi-teacher distillation strategy. When a teacher-side unlearning request is issued, only a subset of student constituents requires retraining, achieving at least $N\times$ speedup.

## Background & Motivation

**Background**: Privacy regulations such as GDPR and CCPA grant users the "right to be forgotten," requiring the removal of specific data's influence from trained models. SISA (Sharded, Isolated, Sliced, Aggregated) is the most representative verified unlearning framework, achieving exact unlearning through data sharding and checkpointing.

**Limitations of Prior Work**: In knowledge distillation (KD) settings, teacher knowledge propagates to the entire student network via soft labels. Even when both teacher and student independently adopt SISA, a teacher-side unlearning event still forces **full retraining** of the student network—because each student constituent is indirectly exposed to information from the complete teacher ensemble.

**Key Challenge**: SISA's efficiency relies on data isolation, but the standard distillation process breaks this isolation—the teacher ensemble provides supervision signals as a whole, causing every student shard to be influenced by all teacher training data.

**Goal**: How to achieve efficient verified unlearning within a KD pipeline, particularly when unlearning requests target teacher training data?

**Key Insight**: If a strict mapping is established between teacher constituents and student constituents such that each teacher influences only a specific subset of students, data isolation can be preserved throughout the distillation process.

**Core Idea**: By restricting teacher influence to specific student shards via constituent mapping, and replacing full-ensemble distillation with an incremental multi-teacher strategy, PURGE restores SISA's unlearning efficiency in the KD setting.

## Method

### Overall Architecture
The PURGE (Partitioned Unlearning with Retraining Guarantee for Ensembles) framework comprises five key steps:
- **Input**: Teacher ensemble $\{T_1, \dots, T_M\}$ (trained with SISA), student dataset $\mathcal{D}^S$
- **Output**: Student ensemble $\{S_1, \dots, S_N\}$ with efficient unlearning capability
- **Pipeline**: Sharding → Mapping → Chunking → Incremental Distillation → Slicing → Aggregated Inference

### Key Designs

1. **Constituent Mapping (Teacher–Student Mapping)**:

    - **Function**: Partitions $M$ teacher constituents into $N$ disjoint subsets $\mathscr{T}_k$; each student $S_k$ learns only from its own teacher subset.
    - **Mechanism**: $\mathscr{T}_k = \{T_{k,1}, \dots, T_{k,c_k}\}$, satisfying $\cap_{k} \mathscr{T}_k = \emptyset$ and $\cup_k \mathscr{T}_k = \{T_1, \dots, T_M\}$.
    - **Design Motivation**: Strict isolation ensures that unlearning from teacher $T_{k,i}$ affects only the corresponding student $S_k$, leaving all other students unaffected.

2. **Incremental Multi-Teacher Distillation**:

    - **Function**: The training data for each student shard is further divided into $c_k$ chunks; the $l$-th chunk uses the first $l$ teachers' sub-ensemble to generate soft labels.
    - **Mechanism**: $Y_{k,l} = \mathscr{T}_{k,l}(\mathcal{D}^S_{k,l})$, where $\mathscr{T}_{k,l} = \cup_{i \in [l]} T_{k,i}$.
    - **Design Motivation**: This limits each teacher's scope of influence—teacher $T_{k,i}$ affects only chunk $i$ onward. Compared to a single-teacher ablation, the incremental ensemble smooths abrupt transitions in supervision signals, improving training stability.

3. **Hierarchical Slicing with Checkpointing**:

    - **Function**: Each chunk is further divided into multiple slices; training proceeds incrementally in chunk→slice order, with checkpoints saved after each slice.
    - **Mechanism**: Student $S_k$ at state $S_{k,l,j}$ is trained for $e_{l,j}$ epochs on cumulative data $(\cup_{i=1}^{l-1} \mathcal{D}^\dagger_{k,i}) \cup (\cup_{q=1}^j \mathcal{D}^\dagger_{k,l,q})$.
    - **Design Motivation**: The hierarchical structure (shard→chunk→slice) provides fine-grained checkpoints, so unlearning only requires rolling back to the earliest affected checkpoint.

### Unlearning Procedure

- **Student-side unlearning**: Upon removing data point $d_u \in \mathcal{D}^\dagger_{k,l,j}$, the model rolls back to $S_{k,l,j-1}$ and partially retrains from that slice onward, inheriting SISA's efficiency.
- **Teacher-side unlearning**: After teacher $T_{k,l}$ performs unlearning, soft labels for chunks $l$ through $c_k$ are updated; student $S_k$ rolls back to $S_{k,l-1}$ and retrains from chunk $l$, affecting only the single constituent $S_k$.

### Theoretical Speedup Analysis

Under uniform allocation ($c = M/N$ chunks/student, $r$ slices/chunk), the speedup of PURGE over naïve SISA is:

$$\frac{t_{\text{sisa}}}{t_{\text{PURGE}}} = N \cdot \frac{6c^2r + 6c}{4c^2r + 3cr + 3c - r + 3}$$

The second factor exceeds 1 for all positive integers $r$ and $c$, guaranteeing at least $N\times$ speedup. The maximum speedup is achieved at $c=1$ (i.e., $N=M$).

## Key Experimental Results

### Main Results: Unlearning Speed (MNIST, $M=32$, 100 teacher-side unlearning requests)

| Students $N$ | Configuration | Avg. Retraining Time/Request | Speedup | Theoretical Prediction |
|---|---|---|---|---|
| Baseline SISA ($N$=8) | Full retraining | 737.14±10.08s | 1× | — |
| PURGE $N$=8 | $c$=4, $r$=1 | ~92s | ~8× | ~8× |
| PURGE $N$=16 | $c$=2, $r$=1 | ~46s | ~16× | ~16× |
| PURGE $N$=32 | $c$=1, $r$=1 | 23.17±0.17s | ~32× | 32× |

### Accuracy Comparison

| Dataset | Method | $M$=32, $N$=32 Acc. | $M$=32, $N$=1 Acc. |
|---|---|---|---|
| MNIST | Teacher | ~98.5% | ~98.5% |
| MNIST | SISA Baseline | 97.08% | 97.30% |
| MNIST | PURGE | 97.16% | 97.16% |
| MNIST | Single-teacher | 95.78% | 95.98% |
| SVHN | SISA Baseline | 83.44% | 83.27% |
| SVHN | PURGE | 83.09% | 83.09% |
| SVHN | Single-teacher | 76.12% | ~76% |

### Key Findings
- PURGE achieves accuracy on par with the SISA baseline across various $N$ and $M$ configurations, with a maximum gap of <0.5%.
- The single-teacher ablation suffers severe performance degradation at $N$=1 ($c$=32), dropping 7.35% on SVHN, validating the necessity of the incremental multi-teacher strategy.
- The choice of $r$ involves a trade-off: larger $r$ accelerates student-side unlearning but slows teacher-side unlearning; the optimal $r$ depends on the relative frequency of the two request types.
- Empirical speedup closely matches theoretical predictions, with minor deviations at $r$=4 due to the ceiling function.

## Highlights & Insights
- The **SISA→KD bridging approach** is conceptually natural: the core challenge is that distillation breaks data isolation, and constituent mapping restores it at minimal cost.
- The **incremental multi-teacher strategy** achieves two goals simultaneously: preserving isolation (each teacher influences a bounded set of chunks) and smoothing student training (avoiding supervision signal discontinuities caused by single-teacher switching).
- The theoretical analysis provides **actionable configuration guidelines** (trade-offs among $N$, $c$, and $r$), offering practical value for real-world deployment.

## Limitations & Future Work
- The framework assumes both teachers and students adopt the SISA ensemble structure, making it inapplicable to distillation from a single large model.
- Experiments cover only MNIST, SVHN, CIFAR-100, and SST5; validation on large-scale language model distillation is absent.
- The quality of incremental multi-teacher soft labels improves progressively across chunks, resulting in lower-quality labels for earlier chunks.
- Complex interaction scenarios where both teacher and student receive simultaneous unlearning requests are not fully addressed (only briefly mentioned in the appendix).

## Related Work & Insights
- **vs. SISA**: SISA is efficient in single-model settings but cannot handle information propagation in KD; PURGE resolves this through structured mapping.
- **vs. SCRUB**: SCRUB approximates unlearning by having the student "contradict" the teacher, without formal guarantees; PURGE provides exact unlearning.
- **vs. RKLD**: RKLD requires an additional "clean" reference teacher, whereas PURGE imposes no such requirement.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First framework achieving verified unlearning in the KD setting; the constituent mapping idea is elegant and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive speed and accuracy evaluations with strong theory–experiment alignment, though dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, theoretical derivations are complete, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — Fills the gap between KD and verified unlearning, with practical implications for privacy-compliant model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)
- [\[NeurIPS 2025\] Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)
- [\[NeurIPS 2025\] Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack](taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)

</div>

<!-- RELATED:END -->
