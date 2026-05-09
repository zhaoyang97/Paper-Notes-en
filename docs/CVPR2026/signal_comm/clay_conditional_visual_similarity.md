---
title: >-
  [Paper Note] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space
description: >-
  [CVPR 2026][Signal & Communication][Conditional image retrieval] CLAY proposes a training-free conditional visual similarity computation method that modulates similarity by constructing text-conditioned subspaces within the VLM embedding space. It adapts to varying retrieval conditions without recomputing database features and supports multi-condition retrieval.
tags:
  - CVPR 2026
  - "Signal & Communication"
  - Conditional image retrieval
  - vision-language models
  - similarity modulation
  - training-free
  - hyperspherical geometry
date: 2026-05-08
content_hash: 58acd369207bb28d
---

# CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space

**Conference**: CVPR 2026
**arXiv**: [2604.11539](https://arxiv.org/abs/2604.11539)
**Code**: None
**Area**: Signal Communication
**Keywords**: Conditional image retrieval, vision-language models, similarity modulation, training-free, hyperspherical geometry

## TL;DR
CLAY proposes a training-free conditional visual similarity computation method that modulates similarity by constructing text-conditioned subspaces within the VLM embedding space. It adapts to varying retrieval conditions without recomputing database features and supports multi-condition retrieval.

## Background & Motivation

**Background**: Image retrieval systems typically rely on fixed, singular similarity metrics. However, human perception of similarity is adaptive — when viewing the same image, one may attend to species, color, action, or other aspects depending on context.

**Limitations of Prior Work**: (1) Training-based methods require condition-specific model training and must recompute all database features when conditions change; (2) existing methods support only single-condition retrieval and cannot simultaneously specify multiple dimensions of interest; (3) training data requires paired images for each condition.

**Key Challenge**: Recomputing database embeddings upon condition changes incurs substantial computational overhead, yet different conditions inherently demand different similarity computation strategies.

**Core Idea**: Decouple the conditioning process from visual feature extraction — keep visual embeddings fixed and dynamically modulate similarity in the comparison space according to text conditions.

## Method

### Overall Architecture
A pretrained VLM extracts fixed visual features → a conditional projection matrix is generated from the given text condition → visual features of both the query and database are projected into the conditioned subspace → cosine similarity is computed in the subspace → ranked results are returned.

### Key Designs

1. **Manifold-Aware Text Subspace Construction**:

    - Function: Construct a similarity modulation space from text conditions.
    - Mechanism: An LLM expands the condition text into multiple descriptive phrases; the VLM text encoder produces a set of embeddings; PCA extracts principal directions to form an orthogonal subspace, yielding the conditional projection matrix $P_c$. To respect the hyperspherical geometry of the VLM embedding space, projected vectors are re-normalized.
    - Design Motivation: A single text embedding is insufficient to define a meaningful subspace; multiple related descriptions are needed to span the semantically relevant directions corresponding to the condition.

2. **Symmetric Conditional Similarity**:

    - Function: Apply the same projection matrix to both query and database features simultaneously.
    - Mechanism: $\text{csim}(I_q, I_d | c) = \cos(P_c \cdot f(I_q), P_c \cdot f(I_d))$, treating query and database features symmetrically. The projection matrix $P_c$ can be precomputed and cached; switching conditions requires only swapping the matrix.
    - Design Motivation: Asymmetric approaches (transforming only the query) preserve condition-irrelevant information in the database features, introducing noise into retrieval results. The symmetric formulation ensures both sides retain only condition-relevant information.

3. **Multi-Condition Retrieval Extension**:

    - Function: Support simultaneous specification of multiple dimensions of interest.
    - Mechanism: A joint projection matrix is constructed by taking the union subspace of projection matrices from multiple conditions.
    - Design Motivation: In real-world scenarios, users may wish to retrieve simultaneously by "species" and "color"; existing methods do not support this.

### Loss & Training
Entirely training-free; the method relies solely on the feature space of a pretrained VLM.

## Key Experimental Results

### Main Results

| Dataset | Metric | CLAY | GeneCIS (training-based) | FocalLens |
|---------|--------|------|--------------------------|-----------|
| GeneCIS Benchmark | Recall@1 | Competitive / Superior | Baseline | Baseline |
| CLAY-EVAL | MR@K | SOTA | Multi-condition not supported | Not supported |

### Ablation Study

| Configuration | Retrieval Accuracy | Notes |
|---------------|--------------------|-------|
| Symmetric projection | Best | Full method |
| Asymmetric projection | Degraded | Noise from database side |
| Single text embedding | Significantly degraded | Insufficient subspace expressiveness |
| Without re-normalization | Degraded | Hyperspherical geometry ignored |

### Key Findings
- The training-free method matches or surpasses training-based methods on standard benchmarks with higher computational efficiency.
- The gap between symmetric and asymmetric projection validates the importance of bidirectional filtering of condition-irrelevant information.
- Accounting for hyperspherical geometry via re-normalization has a significant impact on performance.

## Highlights & Insights
- **"Change the metric, not the features"**: This inverts the conventional paradigm — rather than altering feature extraction, the method alters the comparison strategy, enabling complete reuse of database features.
- **Training-free yet SOTA**: Achieving performance on par with training-based methods without any training data demonstrates strong practical utility.

## Limitations & Future Work
- Performance depends on the quality of LLM-generated textual descriptions.
- Subspace dimensionality requires tuning.
- Precision may be insufficient for extremely fine-grained conditions.

## Related Work & Insights
- **vs. GeneCIS**: Requires training and paired data; feature recomputation is necessary upon condition changes.
- **vs. FocalLens**: Also addresses conditional retrieval but requires training and does not support multi-condition queries.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Relocating the conditioning process from feature extraction to the similarity space is a genuinely novel idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ — A new evaluation dataset is introduced.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical formulations are concise and precise.
- Value: ⭐⭐⭐⭐⭐ — Training-free, efficient, and multi-condition capable; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[ACL 2026\] PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models](../../ACL2026/signal_comm/policyllm_towards_excellent_comprehension_of_public_policy_for_large_language_mo.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](../../NeurIPS2025/signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](../../NeurIPS2025/signal_comm/the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)

</div>

<!-- RELATED:END -->
