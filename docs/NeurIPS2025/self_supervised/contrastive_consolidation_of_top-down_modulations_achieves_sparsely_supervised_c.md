---
title: >-
  [Paper Note] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning
description: >-
  [NeurIPS 2025][Self-Supervised Learning][continual learning] This paper proposes Task-Modulated Contrastive Learning (TMCL), inspired by top-down modulation in the neocortex. In continual learning, sparse label information (as little as 1% labels) is integrated via affine modulation, and contrastive learning is then used to consolidate the modulation information into feedforward weights. TMCL surpasses both unsupervised and supervised baselines on class-incremental learning and transfer learning benchmarks.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - continual learning
  - contrastive learning
  - top-down modulation
  - catastrophic forgetting
  - sparse supervision
  - predictive coding
date: 2026-05-08
content_hash: bc960ad0f428dab3
---

# Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.14125](https://arxiv.org/abs/2505.14125)
**Code**: [Available](https://arxiv.org/abs/2505.14125) (see paper link)
**Area**: Continual Learning / Self-Supervised Learning
**Keywords**: continual learning, contrastive learning, top-down modulation, catastrophic forgetting, sparse supervision, predictive coding

## TL;DR
This paper proposes Task-Modulated Contrastive Learning (TMCL), inspired by top-down modulation in the neocortex. In continual learning, sparse label information (as little as 1% labels) is integrated via affine modulation, and contrastive learning is then used to consolidate the modulation information into feedforward weights. TMCL surpasses both unsupervised and supervised baselines on class-incremental learning and transfer learning benchmarks.

## Background & Motivation

**Background**: The biological brain can learn continuously from unlabeled data streams while integrating occasional label information without degradation. In contrast, machine learning models suffer from catastrophic forgetting in continual learning — supervised fine-tuning on new tasks degrades performance on old tasks.

**Limitations of Prior Work**: (a) Existing continual learning methods typically rely on abundant labeled data or explicit task boundaries; (b) unsupervised continual learning methods avoid forgetting but cannot leverage label information to improve classification; (c) supervised methods are label-inefficient, and even a small number of labels can cause overfitting and exacerbate forgetting.

**Key Challenge**: The stability-plasticity trade-off — exploiting new labels requires plasticity (modifying representations), but modifying representations disrupts old knowledge (requiring stability).

**Goal**: How to achieve effective continual learning under extremely sparse supervision (1% labels)? Specifically, how can a model continuously build general representations from unlabeled data streams while efficiently integrating specialized knowledge upon occasional label encounters?

**Key Insight**: Inspired by top-down modulation in the neocortex — higher-level areas such as the prefrontal cortex influence lower-level representations (e.g., via attention) through modulatory signals without altering the synaptic weights of lower-level feedforward connections. The predictive coding framework provides the implementation basis.

**Core Idea**: New class labels are used to learn only modulation parameters (without modifying feedforward weights); contrastive learning then trains the unmodulated representations to absorb the modulation information (modulation invariance), achieving stable knowledge consolidation.

## Method

### Overall Architecture
TMCL is built on a contrastive learning framework and operates through two core mechanisms: (1) upon encountering new class labels, task-specific affine modulation parameters (affine transformation $\gamma, \beta$) are learned to modulate intermediate-layer representations and improve class separability without modifying feedforward weights; (2) contrastive loss is used to train feedforward weights such that unmodulated representations align with modulated representations in embedding space (modulation invariance), while historical task modulations stabilize the representation space.

### Key Designs

1. **Task-Specific Affine Modulation**:

    - Function: Learns additional affine transformation parameters for each new class to modulate intermediate-layer features.
    - Mechanism: Given a small number of labeled samples from a new class, modulation of the form $h' = \gamma \odot h + \beta$ is introduced at intermediate layers; the low-parameter $\gamma, \beta$ are optimized to maximize separability between new and known classes.
    - Design Motivation: Modifying only the modulation parameters rather than the feedforward weights mirrors how the brain's top-down attention modulates lower-level responses, thereby preventing catastrophic forgetting caused by changes to shared weights.

2. **View-Invariance + Modulation-Invariance Contrastive Learning**:

    - Function: Trains feedforward weights so that representations are simultaneously view-invariant and modulation-invariant.
    - Mechanism: In addition to standard view-invariance (different augmented views of the same sample should map to nearby representations), a modulation-invariance objective is added — unmodulated representations should align with modulated representations of the same sample. Positive pairs in the contrastive loss include: (original view, augmented view) and (unmodulated, modulated).
    - Design Motivation: By training the feedforward network to "absorb" the classification information carried by modulations into its weights, good representations can be obtained at inference time without requiring modulation parameters.

3. **Historical Modulation Consolidation**:

    - Function: Uses past task modulation parameters to prevent representation drift.
    - Mechanism: During training on the current task, feedforward representations are simultaneously required to remain aligned with modulated representations from all past tasks, forming a multi-task modulation invariance constraint.
    - Design Motivation: Past modulation parameters serve as "anchors" to stabilize the representation space and prevent the loss of discriminability on old tasks caused by new task learning — this is the meaning of "contrastive consolidation."

### Loss & Training
Two alternating stages: (1) upon encountering new class labels → freeze feedforward weights and optimize only the new modulation parameters $\gamma, \beta$; (2) during the normal unlabeled data stream → update feedforward weights using contrastive loss to jointly optimize view-invariance and modulation-invariance (including current and historical modulations).

The elegance of this training strategy lies in the complete separation of "knowledge acquisition" and "knowledge consolidation." Modulation parameters serve as a lightweight interface that rapidly absorbs new class information, while contrastive learning slowly but stably "compiles" this information into the feedforward network. Historical modulation parameters act as "anchors" during consolidation to prevent representation drift caused by new task learning — this is the core mechanism of contrastive consolidation.

## Key Experimental Results

### Main Results (Class-Incremental Learning)

| Method | Label Ratio | Performance |
|--------|-------------|-------------|
| Unsupervised continual learning baseline | 0% | Reference performance |
| Supervised continual learning baseline | 100% | Severe catastrophic forgetting |
| TMCL | 1% | Surpasses both unsupervised and supervised baselines under equivalent conditions |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| w/o modulation invariance | Modulation information cannot be consolidated into feedforward weights; performance degrades |
| w/o historical modulation consolidation | Loss of discriminability on old tasks; forgetting intensifies |
| w/o view-invariance | General representation quality degrades |

### Key Findings
- As little as 1% labels yields significant performance gains, demonstrating the high label efficiency of the modulation mechanism.
- Modulation invariance is the key component — it transfers information from "additional parameters" into the shared feedforward weights.
- Consolidation with historical modulations effectively prevents representation drift; modulation parameters serve as lightweight "task memories" that stabilize representations.

## Highlights & Insights
- **Precise Engineering Mapping of Biological Inspiration**: Rather than vaguely invoking brain inspiration, this work precisely maps a concrete biological mechanism from the neocortex — top-down modulation that influences activity patterns without altering synaptic weights — to an engineering solution of affine transformation + contrastive learning.
- **Modulation as a Lightweight Knowledge Interface**: Each new class requires learning only a few affine parameters (far fewer than a fully connected layer), which are then "compiled" into the feedforward network via consolidation — elegantly separating knowledge acquisition from knowledge consolidation.
- **Extreme Label Efficiency at 1%**: TMCL surpasses supervised methods requiring far more labels under extremely sparse label settings, demonstrating that the key is not the quantity of labels but how they are utilized.

## Limitations & Future Work
- **Requires Knowledge of New Class Onset**: The method assumes that the arrival of new class labels is known; in real-world scenarios, class boundaries may be ambiguous and a new class detection mechanism would be needed.
- **Accumulation of Modulation Parameters**: As the number of learned classes grows, historical modulation parameters grow linearly; storage and computation of the consolidation loss may become a bottleneck after learning 1,000+ classes.
- **Evaluated Only on Visual Classification**: Validation in NLP or multimodal continual learning settings is absent, and adaptation to Transformer architectures is unexplored.
- **Distribution Assumption for 1% Labels**: Uniform sampling of the 1% labels is assumed, whereas in practice labels may appear in bursts or be unevenly distributed.
- **Negative Sample Dependency in Contrastive Learning**: The quality of negative samples in the current batch directly affects contrastive consolidation; performance may degrade under small batch sizes.
- **Future Directions**: (1) Automatic detection of new class onset rather than relying on manual annotation; (2) compression or merging of historical modulation parameters to reduce accumulation overhead; (3) extension of the modulation mechanism to Transformer/ViT architectures; (4) exploration of prompt tuning as an alternative modulation form to affine transformations.

## Related Work & Insights
- **vs. EWC/SI and other regularization methods**: These methods prevent forgetting by penalizing changes to important weights, which is indirect and conservative; TMCL directly separates modulation from feedforward weights, yielding a more elegant solution.
- **vs. Experience Replay**: Replay requires storing past samples; TMCL stores only modulation parameters (far more lightweight) and raises no privacy concerns.
- **vs. LUMP/CaSSLe and other unsupervised continual learning methods**: These methods do not leverage labels; TMCL demonstrates that "very few labels + correct utilization" is far superior to "no labels."
- **Insight**: The modulation invariance idea is broadly applicable — any form of "conditional information" can in principle be "compiled" into network weights via a similar mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of biologically inspired modulation and contrastive consolidation is novel with a clear mechanism.
- Experimental Thoroughness: ⭐⭐⭐ Limited by abstract-level information; specific dataset and metric details are insufficient.
- Writing Quality: ⭐⭐⭐⭐ The abstract is clearly written with a complete motivation chain grounded in biological inspiration.
- Value: ⭐⭐⭐⭐ Opens a new direction for sparsely supervised continual learning; modulation invariance is a transferable core idea.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continuous Subspace Optimization for Continual Learning (CoSO)](continuous_subspace_optimization_for_continual_learning.md)
- [\[NeurIPS 2025\] The Complexity of Finding Local Optima in Contrastive Learning](the_complexity_of_finding_local_optima_in_contrastive_learning.md)
- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](contrastive_representations_for_temporal_reasoning.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[CVPR 2026\] UniGeoCLIP: Unified Geospatial Contrastive Learning](../../CVPR2026/self_supervised/unigeoclip_geospatial_contrastive.md)

</div>

<!-- RELATED:END -->
