---
title: >-
  [Paper Note] SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding
description: >-
  [ECCV 2024][Object Detection][temporal grounding] To address the improper negative sample construction in existing compositional temporal grounding methods and the failure of DETR models to generate reasonable saliency responses for negative queries, this paper proposes leveraging an LLM (GPT-3.5 Turbo) to generate semantically feasible hierarchical hard negative samples, and designs a coarse-to-fine saliency ranking strategy to establish multi-granularity semantic relations…
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "temporal grounding"
  - "compositional generalization"
  - "hard negatives"
  - "saliency ranking"
  - "DETR"
date: 2026-05-08
content_hash: 5b159d8ff88d266c
---

# SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding

**Conference**: ECCV 2024  
**arXiv**: [2407.05118](https://arxiv.org/abs/2407.05118)  
**Code**: [https://github.com/zxccade/SHINE](https://github.com/zxccade/SHINE)  
**Area**: Object Detection  
**Keywords**: temporal grounding, compositional generalization, hard negatives, saliency ranking, DETR

## TL;DR

To address the improper negative sample construction in existing compositional temporal grounding methods and the failure of DETR models to generate reasonable saliency responses for negative queries, this paper proposes leveraging an LLM (GPT-3.5 Turbo) to generate semantically feasible hierarchical hard negative samples, and designs a coarse-to-fine saliency ranking strategy to establish multi-granularity semantic relations between video clips and hierarchical negative queries, significantly improving compositional generalization performance.

## Background & Motivation

Temporal grounding aims to localize corresponding clips in videos based on natural language queries. The compositionality of natural language allows queries to describe novel scenarios beyond predefined event scopes, which requires models to possess compositional generalization ability—namely, being trained only on combinations of known words, yet capable of localizing video segments described by novel combinations of known words at inference time.

**Limitations of Prior Work**:

**Improper negative sample construction**: Existing methods (VISA, SSL2CG) only focus on major parts of speech (verbs and nouns), ignoring the critical semantic roles of non-dominant words like prepositions and adverbs (e.g., "on/under the table", "turn on/off the light", where changing a preposition/adverb completely alters the semantics). DeCo constructs negative samples through random sampling and recombination, generating a large number of semantically illogical combinations (e.g., "eating the table", "reading the door"), which misleads the model into learning unrealistic differences.

**Unfeasible saliency responses of DETR models**: Recently, DETR-based methods (Moment-DETR, QD-DETR) have emerged in temporal grounding, predicting saliency scores of each video clip relative to the queries while localizing segments. However, experiments reveal that these models produce unreasonable saliency responses for negative queries—the saliency scores of hard negative queries even exceed those of positive queries, indicating the model's inability to capture fine-grained semantic differences between the original query and its subtly altered counterparts.

**Key Challenge**: To achieve compositional generalization, the model must accurately understand the role of each word in the query. However, existing random construction methods for negative samples either yield unrealistic combinations or only cover partial parts of speech, failing to teach the model fine-grained compositional semantics.

**Key Insight**: (1) Leverage the linguistic knowledge of LLMs to guarantee the semantic feasibility of negative samples while covering all five parts of speech; (2) utilize the existing saliency score mechanism within the DETR framework to establish saliency ranking constraints between the positive query and multi-level negative queries.

## Method

### Overall Architecture

Given a video-query pair $(V_p, Q_p)$:
1. Generate three levels of hard negative queries $\{Q_{hn}^1, Q_{hn}^2, Q_{hn}^3\}$ via a progressive mask-and-predict strategy.
2. Randomly sample an irrelevant negative query $Q_n$ from the same mini-batch.
3. Extract features for all queries and videos using text and video encoders respectively.
4. Interact features via the DETR encoder to predict saliency scores for each query $\{S_p, S_{hn}^1, S_{hn}^2, S_{hn}^3, S_n\}$.
5. Use a coarse-grained ranking loss $\mathcal{L}_{cr}$ to widen the saliency gap between positive and negative queries.
6. Use a fine-grained ranking loss $\mathcal{L}_{fr}$ to constrain the saliency gradient between hierarchical negative queries.
7. Jointly optimize with the baseline loss $\mathcal{L}_{base}$.

### Key Designs

1. **LLM-driven Hierarchical Hard Negative Construction**

    - **Function**: Generate three hard negative queries with progressive semantic changes for each query.
    - **Mechanism**:
     - First, use spaCy to perform part-of-speech (POS) tagging on all queries in the training set, and construct a dictionary $D$ based on five POS categories (verbs, nouns, adjectives, prepositions, adverbs).
     - Progressively mask words in the original query based on linguistic importance ($\text{verbs} \rightarrow \text{nouns} \rightarrow \text{adjectives} \rightarrow \text{prepositions} \rightarrow \text{adverbs}$), with masking ratios of 25%, 50%, and 75% for Charades-CG.
     - Instead of filling with random samples, the masked query and dictionary subsets are fed into GPT-3.5 Turbo to let the LLM generate semantically feasible but different replacement words than the original query.
    - **Design Motivation**: Random sampling produces a large number of illogical combinations ("eating the table"), misleading the model. LLMs possess linguistic common sense to ensure the generated negative queries are semantically plausible (e.g., "person picks up the book" $\rightarrow$ "person throws the pen"), allowing the model to learn to distinguish between plausible confusing scenarios. The progressive masking ensures that the semantic distance between the positive query and the three levels of negative queries increases progressively, providing a foundation for fine-grained ranking.

2. **Coarse-Grained Saliency Ranking**

    - **Function**: Establish saliency prior constraints at the video level.
    - **Mechanism**: Consists of two constraints:
     - **Intra-ranking** $\mathcal{L}_{intra}$: Saliency of the positive query within the ground-truth interval should be higher than outside.
     - **Inter-ranking** $\mathcal{L}_{inter}$: Saliency of the positive query within the ground-truth interval should be higher than that of the negative queries.

     $$\mathcal{L}_{cr} = \max(0, h_1 + S_p^- - S_p^+) + \max(0, h_2 + S_n^+ - S_p^+)$$

     Where $S^+$ takes the top-$k$ average value within the ground-truth interval ($k = \max(1, \lfloor T^+/q \rfloor)$), adaptively handling different interval lengths via $q$.
    - **Design Motivation**: Existing DETR saliency losses only constrain the internal consistency of the positive query, lacking comparison between positive and negative queries. Introducing the inter-ranking constraint explicitly widens the saliency gap between positive and negative queries.

3. **Fine-Grained Saliency Ranking**

    - **Function**: Constrain the saliency of hierarchical hard negative queries to descend progressively.
    - **Mechanism**: Require the saliency scores to satisfy a strict hierarchical structure—Positive > Level-1 Neg > Level-2 Neg > Level-3 Neg > Irrelevant Neg, implemented via multi-stage margin ranking loss:

     $$\mathcal{L}_{fr} = \sum_{i=0}^{3} \max(0, m_i + d(S_p, S_{hn}^i) - d(S_p, S_{hn}^{i+1}))$$

     where $d(\cdot)$ is the negative log-likelihood metric measuring the distribution differences of saliency scores along the temporal dimension.
    - **Design Motivation**: The semantic distance from the positive query differs across the three levels of negative queries—Level-1 replaces fewer words (closer semantics), whereas Level-3 replaces more words (larger semantic difference). The model should yield higher saliency for negative queries that are semantically closer to the positive query (though still lower than the positive query itself), and vice versa. This hierarchical constraint forces the model to learn fine-grained word-video correspondence.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{base} + \alpha \mathcal{L}_{cr} + \beta \mathcal{L}_{fr}$$

- $\mathcal{L}_{base}$: DETR baseline loss (bipartite matching + moment localization + saliency loss)
- $\alpha = \beta = 1.0$ (determined via grid search), margin $h_1=1.0, h_2=2.0, m_0 \sim m_3=0.25$
- Learning rate of 0.0001 for QD-DETR and 0.0002 for Moment-DETR on Charades-CG
- Single NVIDIA A100 GPU, batch size of 32, trained for 200 epochs

## Key Experimental Results

### Main Results

**Charades-CG**:

| Method | Test-Trivial R1@0.5 | Novel-Composition R1@0.5 | Novel-Composition R1@0.7 | Novel-Composition mIoU |
|------|---------------------|-------------------------|--------------------------|----------------------|
| QD-DETR | 59.24 | 42.30 | 21.09 | 38.55 |
| **QD-DETR + SHINE** | **60.66** | **50.23 (+7.93)** | **27.69 (+6.60)** | **44.14 (+5.59)** |
| Moment-DETR | 49.48 | 39.42 | 18.62 | 36.61 |
| **MD + SHINE** | **57.14 (+7.66)** | **44.65 (+5.23)** | **23.21 (+4.59)** | **39.86 (+3.25)** |
| DeCo (prev SOTA) | 58.75 | 47.39 | 21.06 | 40.70 |

**ActivityNet-CG**:

| Method | Test-Trivial R1@0.5 | Novel-Composition R1@0.5 | Novel-Composition mIoU |
|------|---------------------|-------------------------|----------------------|
| QD-DETR | 41.80 | 26.91 | 31.01 |
| **QD-DETR + SHINE** | **43.76** | **29.56 (+2.65)** | **32.44 (+1.43)** |

### Ablation Study

**Ablation of coarse-grained ranking constraints (Charades-CG Novel-Composition)**:

| Configuration | R1@0.5 | R1@0.7 | mIoU | Explanation |
|------|--------|--------|------|------|
| $\mathcal{L}_{base}$ only | 42.30 | 21.09 | 38.55 | baseline |
| + $\mathcal{L}_{intra}$ | 44.02 | 22.84 | 39.23 | intra improvement |
| + $\mathcal{L}_{inter}$ | 46.69 | 24.87 | 41.74 | inter contributes more |
| + Both | 46.25 | 24.93 | 41.88 | Further improvement by combination |

**LLM vs. Random Sampling (Charades-CG Novel-Composition)**:

| Negative Sample Source | R1@0.5 | R1@0.7 | mIoU |
|-----------|--------|--------|------|
| Random Sampling | 47.41 | 25.33 | 42.50 |
| Llama 3 | 48.75 | 25.22 | 42.89 |
| Gemini-1.5 Flash | 48.69 | 25.60 | 43.54 |
| **GPT-3.5 Turbo** | **50.23** | **27.69** | **44.14** |

**Contribution of prepositions and adverbs (Charades-CG Novel-Composition)**:

| Configuration | R1@0.5 | mIoU |
|------|--------|------|
| QD+Ours w/o prep & adv | 48.87 | 43.30 |
| **QD+Ours** | **50.23** | **44.14** |

### Key Findings

- **$\mathcal{L}_{inter}$ contributes more than $\mathcal{L}_{intra}$**: The constraint on the saliency gap between positive and negative queries is more crucial for compositional generalization than the constraint between inside and outside intervals.
- **Fine-grained constraints require the complete hierarchy to function optimally**: Individually adding $\mathcal{L}_{fr}^3$ (the deepest level) slightly decreases performance, but incorporating it within the complete hierarchical constraints leads to a 3.17% gain in R1@0.7.
- **Negative samples generated by GPT-3.5 Turbo are optimal**: Achieving a 2.82% R1@0.5 improvement over random sampling, showing that the linguistic common sense of LLMs ensures negative sample plausibility.
- Integrating prepositions and adverbs into the negative sample construction effectively enhances the model's perception of non-dominant words.

## Highlights & Insights

- **First to link DETR saliency scores with compositional generalization**: The discovery that existing DETR models yield unfeasible saliency responses to negative queries is a key insight in itself.
- **LLM as a negative sample generator**: Leveraging the linguistic common sense of LLMs to guarantee semantic feasibility provides a much more effective negative sample construction paradigm than random replacement.
- **Plug-and-play design**: SHINE can be seamlessly integrated into any DETR-based temporal grounding model; it requires no modifications to the model architecture and only adds loss terms during training.
- **Coverage of all five POS categories**: Prepositions and adverbs, though not dominant parts of speech, have a huge impact on semantics. Incorporating them into negative sample construction is a valuable takeaway.
- **Coarse-to-fine hierarchical constraints**: The multi-granularity ranking constraint strategy from video-level to temporal distribution-level can be transferred to other tasks requiring hierarchical contrastive learning.

## Limitations & Future Work

- The masking ratios (25/50/75%) for hierarchical negative samples are manually set, and the optimal ratios vary across datasets (ActivityNet-CG uses 10/30/50%).
- Calling LLMs increases preprocessing costs prior to training (GPT-3.5 Turbo API fees), and the quality of LLM generation is not fully controllable.
- Validated only on two compositional generalization benchmarks; generalization performance on standard temporal grounding benchmarks (Charades-STA, ActivityNet Captions) has not been tested.
- The advantage of LLMs over random replacement diminishes when queries are longer (the improvement on ActivityNet-CG is less significant than on Charades-CG).
- The pseudo saliency scores (1 inside the interval, 0 outside) for fine-grained ranking are coarse binary labels, which do not consider gradual transitions near boundaries.

## Related Work & Insights

- **vs. DeCo**: DeCo also utilizes a decompose-reconstruct strategy to construct negative samples, but relies on random sampling which generates many illogical combinations. SHINE uses LLMs to ensure semantic feasibility and introduces coarse-to-fine ranking constraints.
- **vs. SSL2CG**: SSL2CG generates equivalent and invariant samples for contrastive learning by masking different words, but it only considers verbs and nouns. SHINE expands this to five POS categories and uses LLMs to generate plausible replacements.
- **vs. QD-DETR**: As a plug-and-play component, SHINE does not modify the QD-DETR architecture, but substantially improves its compositional generalization by simply adding training constraints (R1@0.5 +7.93%).

## Rating

- Novelty: ⭐⭐⭐⭐ Combining LLM-based negative sample construction with saliency ranking is a highly creative approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, two baselines, detailed ablations, and comparisons across multiple LLMs.
- Writing Quality: ⭐⭐⭐⭐ Thorough problem analysis, clear explanation of the method, and highly convincing visualizations.
- Value: ⭐⭐⭐⭐ Great practicality due to its plug-and-play design, though slightly limited by its focus on the niche area of compositional generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sim-DETR: Unlock DETR for Temporal Sentence Grounding](../../ICCV2025/object_detection/sim-detr_unlock_detr_for_temporal_sentence_grounding.md)
- [\[ECCV 2024\] BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos](bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)
- [\[ECCV 2024\] ReGround: Improving Textual and Spatial Grounding at No Cost](reground_improving_textual_and_spatial_grounding_at_no_cost.md)
- [\[CVPR 2026\] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](../../CVPR2026/object_detection/falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)

</div>

<!-- RELATED:END -->
