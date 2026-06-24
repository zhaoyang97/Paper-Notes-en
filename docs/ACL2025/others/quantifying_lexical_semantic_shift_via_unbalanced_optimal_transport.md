---
title: >-
  [Paper Note] Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport
description: >-
  [ACL 2025][lexical semantic shift] Applies Unbalanced Optimal Transport (UOT) to sets of contextualized word embeddings, proposing the Sense Usage Shift (SUS) metric to quantify semantic changes at the level of individual usage instances, which unifies three tasks: instance-level change detection, word-level change magnitude quantification, and semantic expansion/reduction determination.
tags:
  - "ACL 2025"
  - "lexical semantic shift"
  - "unbalanced optimal transport"
  - "contextualized word embeddings"
  - "SUS metric"
  - "semantic broadening/narrowing"
date: 2026-05-08
content_hash: af7a851e67a6e3db
---

# Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport

**Conference**: ACL 2025  
**arXiv**: [2412.12569](https://arxiv.org/abs/2412.12569)  
**Code**: [GitHub](https://github.com/ryo-lyo/Semantic-Shift-via-UOT)  
**Area**: Others  
**Keywords**: lexical semantic shift, unbalanced optimal transport, contextualized word embeddings, SUS metric, semantic broadening/narrowing

## TL;DR

Applies Unbalanced Optimal Transport (UOT) to sets of contextualized word embeddings, proposing the Sense Usage Shift (SUS) metric to quantify semantic changes at the level of individual usage instances, which unifies three tasks: instance-level change detection, word-level change magnitude quantification, and semantic expansion/reduction determination.

## Background & Motivation

### Background

**Background**: Lexical semantic shift detection aims to identify changes in word meanings over time. Mainstream approaches estimate word-level change degrees using contextualized embeddings. **Limitations of Prior Work**: Existing methods only provide overall word-level change scores, failing to reveal the direction and degree of semantic change for individual usage instances. **Key Challenge**: Different usages of the same word might undergo different directional changes (e.g., the growth of the music sense of "record" vs. the decline of its information sense); word-level aggregation conceals these fine-grained signals. **Goal**: To provide instance-level quantification of semantic shifts. **Key Insight**: To treat usage instance sets from two different time periods as two distributions and align them using UOT. **Core Idea**: UOT allows "non-conservation of mass"—the total mass of the two distributions can differ, which naturally corresponds to the increase or decrease in the frequency of specific word usages.

## Method

### Overall Architecture

Extract the sets of contextualized embeddings of the target word from two different periods $\to$ compute the optimal transport plan using UOT $\to$ extract the SUS metric from the excess/deficit of the transport plan $\to$ aggregate to the word level to obtain metrics such as change magnitude, expansion, and reduction.

### Key Designs

1. **Application of Unbalanced OT**:
    - Function: Computes the UOT transport plan between the embedding sets of two periods.
    - Mechanism: Standard OT requires mass conservation (total transport amount = 1). UOT relaxes this constraint to allow $\text{excess}$ (mass overflow, corresponding to increased sense usage) and $\text{deficit}$ (mass deficit, corresponding to decreased sense usage). Mass imbalances are penalized using KL divergence.
    - Design Motivation: The essence of semantic shift is the increase or decrease of specific senses over time—the non-conservation of mass in UOT naturally corresponds to this linguistic phenomenon.

2. **Sense Usage Shift (SUS) Metric**:
    - Function: Computes a scalar value for each usage instance, quantifying the increase or decrease in the usage frequency of that sense.
    - Mechanism: $\text{SUS}(i) = \text{excess}(i) - \text{deficit}(i)$, where a positive value indicates increased usage of the sense, and a negative value indicates decreased usage.
    - Design Motivation: SUS is the first instance-level semantic change metric, allowing direct visualization of the directional change of each usage.

3. **Unified Multi-Task Framework**:
    - Function: Derives word-level metrics starting from SUS.
    - Mechanism: Word-level shift magnitude is measured by the variance of SUS; semantic expansion is indicated by total excess > total deficit; semantic reduction is indicated by the opposite.
    - Design Motivation: A single UOT computation can simultaneously address questions at three different levels.

### Loss & Training

No training is involved. UOT is solved using the Sinkhorn algorithm, and embeddings are contextualized representations obtained from pre-trained BERT/XLM-R.

## Key Experimental Results

### Main Results

SemEval-2020 Task 1 Lexical Semantic Change Detection:

| Method | English (Spearman) | German | Latin | Swedish |
|------|:---:|:---:|:---:|:---:|
| APD (Baseline) | 0.56 | 0.72 | 0.40 | 0.54 |
| JSD (Baseline) | 0.58 | 0.73 | 0.42 | 0.55 |
| SUS-variance (Ours) | **0.62** | **0.76** | **0.46** | **0.59** |

### Ablation Study

SUS visualization validation (the word "record"):

| Sense | SUS Value | Meaning |
|------|:---:|------|
| music sense (1960-2010) | +0.47 | Usage frequency **increased** |
| achievement sense (1960-2010) | +0.45 | Usage frequency **increased** |
| information sense (1810-1860) | -0.25 | Usage frequency **decreased** |

### Key Findings

1. **Intuitive SUS Visualization**: The music (+0.47) and achievement (+0.45) senses of "record" grow, while the information sense (-0.25) declines, aligning with historical facts.
2. **UOT Outperforms Standard OT**: Relaxing the mass conservation constraint improves performance, proving that mass imbalance signals carry useful information.
3. **Unified Solution for Three Tasks**: The same metric is effective for instance-level, word-level, and expansion/reduction detection.

## Highlights & Insights

- **Theoretical Elegance of UOT for Semantic Shift**: "Non-conservation of mass" naturally corresponds to "changes in sense usage frequencies."
- **Breakthrough in Instance-Level Quantification**: Previous methods could only state "how much record has changed," whereas this approach can specify "which usage of record has changed."
- **Three Answers in One Computation**: UOT's excess/deficit simultaneously provides instance-level directions, word-level magnitudes, and expansion/reduction.
- **Highly Convincing Visualization**: t-SNE paired with SUS color coding clearly demonstrates semantic shifts.

## Limitations & Future Work

- Reliance on pre-trained embedding quality (the contextualization capability of BERT/XLM-R).
- The regularization parameters of UOT require tuning.
- Evaluated only on the SemEval-2020 dataset.
- Integration with polysemy disambiguation has not been explored.

## Related Work & Insights

- **vs. APD/JSD**: Only provide word-level scores, whereas ours provides instance-level SUS.
- **vs. Standard OT (Wasserstein Distance)**: Constrained by mass conservation, whereas UOT relaxes constraints to capture more information.
- **vs. Clustering Methods**: Require pre-specifying the number of clusters, whereas UOT automatically discovers changes in sense distributions.
- **Insights**: Fine-grained analysis of semantic shift requires "asymmetric alignment between distributions"—UOT is a natural tool for this.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Elegant and pioneering application of UOT to semantic shift.
- Experimental Thoroughness: ⭐⭐⭐ Limited coverage of datasets and languages.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical framework and outstanding visualizations.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for lexical semantic shift detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/others/estimation_of_stochastic_optimal_transport_maps.md)
- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](../../ICCV2025/others/lacoot_layer_collapse_through_optimal_transport.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](../../ICML2025/others/hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](../../ICML2025/others/lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [\[ACL 2025\] S3 - Semantic Signal Separation](s3_-_semantic_signal_separation.md)

</div>

<!-- RELATED:END -->
