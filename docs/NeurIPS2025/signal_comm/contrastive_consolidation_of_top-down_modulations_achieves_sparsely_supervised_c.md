---
title: >-
  [Paper Note] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning
description: >-
  [NeurIPS 2025][Signal & Communication][continual learning] This paper proposes Task-Modulated Contrastive Learning (TMCL), inspired by top-down modulations in the neocortex. TMCL integrates sparse label information (as few as 1% labels) via affine modulation during continual learning, then consolidates the modulation information into feedforward weights through contrastive learning, surpassing both unsupervised and supervised baselines on class-incremental and transfer learni…
tags:
  - "NeurIPS 2025"
  - "Signal & Communication"
  - "continual learning"
  - "contrastive learning"
  - "top-down modulation"
  - "catastrophic forgetting"
  - "sparse supervision"
  - "predictive coding"
date: 2026-05-08
content_hash: 853ed00b51016b4c
---

# Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.14125](https://arxiv.org/abs/2505.14125)  
**Code**: [Available](https://arxiv.org/abs/2505.14125) (see paper link)  
**Area**: Continual Learning / Self-Supervised Learning
**Keywords**: continual learning, contrastive learning, top-down modulation, catastrophic forgetting, sparse supervision, predictive coding

## TL;DR
This paper proposes Task-Modulated Contrastive Learning (TMCL), inspired by top-down modulations in the neocortex. TMCL integrates sparse label information (as few as 1% labels) via affine modulation during continual learning, then consolidates the modulation information into feedforward weights through contrastive learning, surpassing both unsupervised and supervised baselines on class-incremental and transfer learning benchmarks.

## Background & Motivation

**Background**: The biological brain can continually learn from unlabeled data streams while integrating occasional label information without degradation. In contrast, machine learning models suffer from catastrophic forgetting in continual learning — supervised fine-tuning on new tasks degrades performance on old ones.

**Limitations of Prior Work**: (a) Existing continual learning methods typically rely on abundant labeled data or explicit task boundaries; (b) unsupervised continual learning methods avoid forgetting but cannot leverage label information to improve classification; (c) supervised methods utilize labels inefficiently, where even a small number of labels can cause overfitting and exacerbate forgetting.

**Key Challenge**: The stability-plasticity trade-off — exploiting new labels requires plasticity (modifying representations), but modifying representations disrupts prior knowledge (requiring stability).

**Goal**: How to achieve effective continual learning under extremely sparse labels (1%)? Specifically, how can a model continuously build general representations from an unlabeled data stream while efficiently integrating specialized knowledge upon encountering occasional labels?

**Key Insight**: Inspired by top-down modulation in the neocortex — higher-level regions such as the prefrontal cortex influence lower-level representations (e.g., via attention mechanisms) through modulatory signals without altering the synaptic weights (feedforward connections) of lower layers. Predictive coding provides the implementation framework.

**Core Idea**: When encountering a new class label, only modulation parameters are learned (feedforward weights remain unchanged); contrastive learning then drives the unmodulated representations to "absorb" the modulation information (modulation invariance), achieving stable knowledge consolidation.

## Method

### Overall Architecture
TMCL is built on a contrastive learning framework and operates through two core mechanisms: (1) upon encountering new class labels, task-specific affine modulation parameters ($\gamma, \beta$) are learned to modulate intermediate representations and improve the separability of new classes without modifying feedforward weights; (2) a contrastive loss trains the feedforward weights to align unmodulated representations with modulated ones in the embedding space (modulation invariance), while historical task modulations are used to stabilize the representation space.

### Key Designs

1. **Task-Specific Affine Modulation**:

    - Function: Learns additional affine transformation parameters for each new class to modulate intermediate features.
    - Mechanism: Given a small number of labeled samples from a new class, modulation of the form $h' = \gamma \odot h + \beta$ is introduced at an intermediate layer; the low-parameter $\gamma, \beta$ are optimized to maximize the separation between new and known classes.
    - Design Motivation: Modifying only the modulation parameters rather than feedforward weights mirrors how the brain's top-down attention modulates lower-level responses, avoiding catastrophic forgetting caused by changes to shared weights at the source.

2. **View-Invariance + Modulation-Invariance Contrastive Learning**:

    - Function: Trains feedforward weights so that representations are simultaneously view-invariant and modulation-invariant.
    - Mechanism: Building on the standard view-invariance objective (different augmented views of the same sample should map to similar representations), a modulation-invariance objective is added — unmodulated representations should align with modulated representations of the same sample. Positive pairs in the contrastive loss include: (original view, augmented view) and (unmodulated, modulated).
    - Design Motivation: By driving the feedforward network to "absorb" the classification information introduced by modulation into its weights, good representations can be obtained at inference time without using modulation parameters.

3. **Historical Modulation Consolidation**:

    - Function: Uses modulation parameters from past tasks to prevent representation drift.
    - Mechanism: During training on the current task, the feedforward representations are simultaneously required to remain aligned with the modulated representations of all past tasks, forming a multi-task modulation-invariance constraint.
    - Design Motivation: Past modulation parameters serve as "anchors" to stabilize the representation space, preventing the loss of discriminability for old tasks caused by new task learning — this is the meaning of "contrastive consolidation."

### Loss & Training
Two alternating phases: (1) upon encountering new class labels → freeze feedforward weights, optimize only the new modulation parameters $\gamma, \beta$; (2) during normal unlabeled data streaming → update feedforward weights using a contrastive loss that jointly optimizes view-invariance and modulation-invariance (for both current and historical modulations).

The elegance of this training strategy lies in the complete separation of "knowledge acquisition" and "knowledge consolidation." Modulation parameters act as a lightweight interface for rapidly absorbing new class information, while contrastive learning slowly but stably "compiles" this information into the feedforward network. Historical modulation parameters serve as "anchors" during consolidation, preventing representation drift for old tasks — this is the core mechanism of contrastive consolidation.

## Key Experimental Results

### Main Results (Class-Incremental Learning)

| Method | Label Ratio | Performance |
|--------|-------------|-------------|
| Unsupervised continual learning baseline | 0% | Reference performance |
| Supervised continual learning baseline | 100% | Severe catastrophic forgetting |
| TMCL | 1% | Surpasses both unsupervised and supervised baselines under equivalent conditions |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| w/o modulation invariance | Modulation information cannot be consolidated into feedforward weights; performance degrades |
| w/o historical modulation consolidation | Discriminability of old tasks is lost; forgetting increases |
| w/o view-invariance | Quality of general representations decreases |

### Key Findings
- Only 1% labels are sufficient to achieve significant performance gains, indicating extremely high label efficiency of the modulation mechanism.
- Modulation invariance is critical — it transfers information from "additional parameters" into the shared feedforward weights.
- Consolidation of historical modulations effectively prevents representation drift; modulation parameters serve as lightweight "task memories" to stabilize representations.

## Highlights & Insights
- **Precise Engineering Mapping from Biological Inspiration**: Rather than generic "brain-inspired" motivation, the paper precisely maps a specific biological mechanism from the neocortex — top-down modulation that influences activity patterns without modifying synaptic weights — to an engineering solution of affine transformations combined with contrastive learning.
- **Modulation as a Lightweight Knowledge Interface**: Each new class requires learning only a few affine parameters (far fewer than a fully connected layer), which are then "compiled" into the feedforward network via consolidation — elegantly separating "knowledge acquisition" and "knowledge consolidation."
- **Extreme Efficiency with 1% Labels**: Surpassing supervised methods that require far more labels under extremely sparse label settings demonstrates that the key is not the quantity of labels but the manner in which they are utilized.

## Limitations & Future Work
- **Requires Known Timing of New Class Appearances**: The method assumes knowledge of when new class labels are encountered; in practice, class boundaries may be ambiguous, requiring a new-class detection mechanism.
- **Accumulation of Modulation Parameters**: As the number of learned classes grows, historical modulation parameters increase linearly; storing and computing the consolidation loss after learning 1,000+ classes may become a bottleneck.
- **Validation Limited to Visual Classification**: The approach has not been validated on NLP or multimodal continual learning settings, and adaptation to Transformer architectures remains unexplored.
- **Distribution Assumption for 1% Labels**: The method assumes that the 1% labels are uniformly sampled; in practice, labels may appear in bursts or be non-uniformly distributed.
- **Dependence on Negative Samples in Contrastive Learning**: The quality of in-batch negatives directly affects contrastive consolidation; performance may degrade with small batch sizes.
- **Future Directions**: (1) Automatic detection of new class appearances rather than relying on manual annotation; (2) compression or merging of historical modulation parameters to reduce accumulation overhead; (3) extension of the modulation mechanism to Transformer/ViT architectures; (4) exploration of prompt tuning as an alternative to affine modulation.

## Related Work & Insights
- **vs. EWC/SI and other regularization methods**: These methods prevent forgetting by penalizing changes to important weights, which is indirect and conservative; TMCL directly separates modulation and feedforward weights, offering a more elegant solution.
- **vs. Experience Replay**: Replay requires storing past samples; TMCL stores only modulation parameters (far more lightweight) and raises no privacy concerns.
- **vs. LUMP/CaSSLe and other unsupervised continual learning methods**: These methods do not leverage labels; TMCL demonstrates that "very few labels + correct utilization" substantially outperforms "no labels."
- **Insight**: The concept of modulation invariance is generalizable — any "conditional information" can be "compiled" into network weights through a similar mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of biologically-inspired modulation and contrastive consolidation is novel with a clear mechanism.
- Experimental Thoroughness: ⭐⭐⭐ Limited by abstract-level information; specific dataset and metric details are insufficient.
- Writing Quality: ⭐⭐⭐⭐ The abstract is clearly written with a complete motivational chain grounded in biological inspiration.
- Value: ⭐⭐⭐⭐ Opens a new direction for sparsely supervised continual learning; modulation invariance is a transferable core idea.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[ICML 2026\] Meta-learning Structure-Preserving Dynamics](../../ICML2026/signal_comm/meta-learning_structure-preserving_dynamics.md)
- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)

</div>

<!-- RELATED:END -->
