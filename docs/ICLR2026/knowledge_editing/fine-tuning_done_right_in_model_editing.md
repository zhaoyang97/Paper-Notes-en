---
title: >-
  [Paper Note] Fine-tuning Done Right in Model Editing
description: >-
  [ICLR 2026][Knowledge Editing][Model Editing] This paper reveals that the underestimation of fine-tuning in model editing stems from an incorrect training pipeline (depth-first, sample-by-sample optimization). By correcting it to standard breadth-first mini-batch training and combining it with localized parameter updates, the proposed LocFT-BF achieves, for the first time, support for 100K sequential edits and models up to 72B parameters.
tags:
  - ICLR 2026
  - Knowledge Editing
  - Model Editing
  - Fine-tuning
  - Catastrophic Forgetting
  - Localized Fine-tuning
date: 2026-05-08
content_hash: bc1800159aa0a553
---

# Fine-tuning Done Right in Model Editing

**Conference**: ICLR 2026
**arXiv**: [2509.22072](https://arxiv.org/abs/2509.22072)
**Code**: [https://github.com/ICT-STAR/LocFT](https://github.com/ICT-STAR/LocFT)
**Area**: Knowledge Editing
**Keywords**: Model Editing, Fine-tuning, Knowledge Editing, Catastrophic Forgetting, Localized Fine-tuning

## TL;DR
This paper reveals that the underestimation of fine-tuning in model editing stems from an incorrect training pipeline (depth-first, sample-by-sample optimization). By correcting it to standard breadth-first mini-batch training and combining it with localized parameter updates, the proposed LocFT-BF achieves, for the first time, support for 100K sequential edits and models up to 72B parameters.

## Background & Motivation

**State of the Field**: Model editing aims to efficiently modify specific factual knowledge in LLMs without full retraining. Mainstream approaches include parameter expansion (GRACE), meta-learning (MEND), and locate-and-edit methods (ROME/MEMIT). Fine-tuning has long been treated as a weak baseline in this domain, dismissed due to "overfitting and catastrophic forgetting."

**Limitations of Prior Work**: (a) Locate-and-edit methods require precomputed matrices and scale poorly; (b) meta-learning methods require additional labeled data and auxiliary networks; (c) parameter expansion methods alter model architecture; (d) all methods suffer significant performance degradation under large-scale sequential editing (>10K edits).

**Root Cause**: Fine-tuning is the most successful paradigm for LLM adaptation, yet it is deemed "ineffective" in model editing — a contradiction that warrants deeper investigation.

**Paper Goals**: Is fine-tuning truly inadequate for model editing? Is its failure inherent to the method itself, or to how it is implemented?

**Starting Point**: An audit of existing codebases reveals that fine-tuning in model editing follows a **non-standard training procedure**: optimizing each sample to convergence before proceeding to the next (depth-first), rather than the standard multi-epoch mini-batch training (breadth-first).

**Core Idea**: The "failure" of fine-tuning in model editing is an implementation artifact rather than a fundamental limitation — correcting the training pipeline and applying localized parameter updates suffices to surpass all existing SOTA methods.

## Method

### Overall Architecture
LocFT-BF = standard breadth-first mini-batch training + localized parameter updates (tuning only the down/up projection matrices of the last few layers). Given a sequence of edit requests $(s, r, o \to o')$, the method outputs updated model parameters $\theta^*$.

### Key Designs

1. **Pipeline Correction: Depth-First → Breadth-First**

   - Function: Replaces single-pass per-sample training (DF) with multi-epoch traversal over the full edit set (BF).
   - Mechanism: The root cause of catastrophic forgetting under DF is single-pass processing, where later edits overwrite earlier ones. BF allows the model to balance learning across all edits through repeated traversals.
   - Design Motivation: This is the most critical finding — this single change alone (even with batch size = 1) yields dramatic improvements.

2. **Update Granularity Correction: Per-Sample → Mini-Batch**

   - Function: Increases batch size from 1 to standard mini-batch sizes (e.g., 64).
   - Mechanism: Per-sample gradients exhibit high variance and tend to degrade general capabilities; mini-batch gradient aggregation is more stable.
   - Design Motivation: Further stabilizes training under the BF pipeline and substantially reduces degradation of general model capabilities.

3. **Localized Parameter Selection**

   - Function: Systematically evaluates the editing performance of different layers and modules (various projection matrices in attention and MLP blocks).
   - Mechanism: Tuning the **down-projection or up-projection matrices of the last few layers** is found to be generally optimal — achieving near-100% edit success rate while preserving general capabilities.
   - Design Motivation: Existing methods (e.g., FT-M) inherit the parameter locations from ROME (middle-layer MLPs), which are not optimized for fine-tuning. A systematic search reveals that later layers are more suitable for knowledge editing.

### Loss & Training
Standard cross-entropy loss, computed only on the target edit tokens. Key implementation details:
- Multi-epoch breadth-first traversal
- Mini-batch gradient aggregation
- Only the MLP down/up projections of the last few layers are updated
- No additional regularization, auxiliary data, or architectural modifications are required

## Key Experimental Results

### Main Results

**LLaMA3-8B, 1000 ZsRE edits**:

| Method | Reliability | Generalization | Capability |
|--------|-------------|----------------|------------|
| ROME | Moderate | Moderate | Degraded |
| MEMIT | Moderate | Moderate | Degraded |
| GRACE | Moderate | Low | Preserved |
| FT-M (DF, original) | 75.3 | 67.2 | 28.3 |
| FT-M (BF, batch=1) | Large gain | Large gain | Improved |
| **LocFT-BF** | **~100%** | **~100%** | **Preserved** |

Average edit success rate exceeds the best baseline by **33.72%**.

### Ablation Study

| Configuration | Edit Success | General Capability | Note |
|---------------|--------------|--------------------|------|
| DF pipeline + batch=1 | Low | Severe degradation | Conventional approach |
| BF pipeline + batch=1 | Large gain | Improved | Pipeline fix only |
| BF + mini-batch | Further gain | Significantly preserved | + larger batch |
| BF + mini-batch + localization | Optimal | Fully preserved | + optimized parameter location |

### Key Findings
- **Changing only the pipeline (DF→BF)** eliminates catastrophic forgetting, overturning the long-held consensus that fine-tuning is unsuitable for model editing.
- For the first time, evaluation is scaled to **100K sequential edits** (previously at most 10K) and **72B parameter models** (previously at most 7/8B), with stable performance and no degradation.
- Later-layer MLPs are more suitable for knowledge editing than middle-layer MLPs, offering an interesting contrast to ROME's hypothesis that "middle layers store factual knowledge."
- The method is remarkably simple: no matrix precomputation, no auxiliary networks, no architectural modifications, no additional data — purely standard fine-tuning.

## Highlights & Insights
- **Exposes a long-standing implementation error in the field**: The implicit assumption of "everyone does it this way, so it must be correct" is challenged, prompting the community to reassess the fairness of baselines. Similar "underestimated baseline" phenomena may exist in other areas.
- **Simplicity as strength**: Without any sophisticated components, merely "using fine-tuning correctly" outperforms all complex methods — a significant methodological correction for the model editing community.
- The new evaluation standard of **100K edits and 72B models** pushes benchmark requirements closer to real-world application needs.

## Limitations & Future Work
- The breadth-first pipeline requires access to all edit requests at once, making it not fully applicable to truly online or streaming editing scenarios.
- The optimal localized parameter location varies across models, and no automatic selection mechanism that avoids manual search has been proposed.
- Editing in multimodal models has not been evaluated.
- Experiments primarily involve single fact triple edits; complex reasoning chain editing (requiring modification of multiple related facts) has not been sufficiently validated.

## Related Work & Insights
- **vs. ROME/MEMIT**: Locate-and-edit methods require precomputed causal tracing matrices and do not scale; LocFT-BF fine-tunes directly, is simpler, and scales to 72B models.
- **vs. GRACE**: Parameter expansion methods modify the architecture and generalize poorly; LocFT-BF requires no architectural changes.
- **vs. MEND**: Meta-learning requires auxiliary networks and additional data; LocFT-BF is pure fine-tuning with no additional overhead.
- **Takeaway**: Before concluding that a method "does not work," one should first verify whether its implementation is correct.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Identifies and corrects an implementation error underlying field-wide consensus, with substantial impact.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 models × 2 datasets × multi-method comparison + 100K edits + 72B model.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is exceptionally clear: identify the problem → controlled experiments → correction → systematic optimization.
- Value: ⭐⭐⭐⭐⭐ A work that reshapes the community's understanding of baselines in model editing.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](../../ACL2026/knowledge_editing/fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICLR 2026\] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ICLR 2026\] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)

<!-- RELATED:END -->
