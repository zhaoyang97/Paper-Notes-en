---
title: >-
  [Paper Note] Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation
description: >-
  [ACL 2025][Entity Disambiguation] This work systematically evaluates key design choices of Dual Encoders in Entity Disambiguation (ED) tasks (loss functions, similarity measures, label verbalization formats, and negative sampling strategies). Based on the optimal design, the VerbalizED system is constructed, achieving a new SOTA on the ZELDA benchmark. It also explores an iterative prediction strategy to leverage already disambiguated neighboring entities to improve difficult…
tags:
  - "ACL 2025"
  - "Entity Disambiguation"
  - "Dual Encoder"
  - "Label Verbalization"
  - "Hard Negative Sampling"
  - "ZELDA Benchmark"
date: 2026-05-08
content_hash: 0de2a30a019a135c
---

# Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation

**Conference**: ACL 2025  
**arXiv**: [2505.11683](https://arxiv.org/abs/2505.11683)  
**Code**: Yes (planning to release VerbalizED code and label verbalization data)  
**Area**: Other  
**Keywords**: Entity Disambiguation, Dual Encoder, Label Verbalization, Hard Negative Sampling, ZELDA Benchmark

## TL;DR

This work systematically evaluates key design choices of Dual Encoders in Entity Disambiguation (ED) tasks (loss functions, similarity measures, label verbalization formats, and negative sampling strategies). Based on the optimal design, the VerbalizED system is constructed, achieving a new SOTA on the ZELDA benchmark. It also explores an iterative prediction strategy to leverage already disambiguated neighboring entities to improve difficult samples.

## Background & Motivation

Entity Disambiguation (ED) is the task of linking entity mentions in text to corresponding entries in a knowledge base (KB), which is a key component for downstream tasks such as "knowledge graph construction", "question answering systems", and "information retrieval".

Dual Encoder is one of the most popular ED architectures, encoding mentions and KB entities into a shared vector space and making predictions via similarity matching. However, behind this seemingly simple architecture lies a plethora of design decisions, each of which can significantly affect disambiguation performance:

- How to represent/verbalize labels in the KB?
- Which similarity metric to use?
- Which loss function to train with?
- How to sample negative examples?
- How frequently should the label embedding cache be updated?

Prior works typically present a complete system as a whole, rarely comparing and ablating these design choices systematically. The core contribution of this paper is **systematically evaluating the impact of each design decision**.

## Method

### Overall Architecture

The Dual Encoder architecture of VerbalizED includes:

- **Mention Encoder**: Processes the context of mentions in the document, utilizing the entire document as context (document-level).
- **Label Encoder**: Encodes the metadata (title + description + categories) of each entity in the KB after verbalizing it into short text.
- **Similarity Calculation**: Calculates the similarity between the mention and entity embeddings after pooling, selecting the most similar one as the prediction.
- **Training**: Optimizes the embedding space through negative sampling and loss functions, pulling correct mention-entity pairs closer and pushing incorrect ones further apart.
- **Label Embedding Cache**: Updates periodically rather than re-encoding all entities at every step.

### Key Designs

1. **Label Verbalization Format**:  
    - Title only: 63.68 F1
    - Title + Description: 64.48
    - Title + Categories: 64.00
    - **Title + Description + Categories: 65.01** (Optimal)
    - Title + Paragraph(100): 64.30
    - Title + Paragraph(500): 63.49 (Excessive length is actually detrimental)
   
   **Conclusion**: Description provides semantic details while Categories provide structured information, which are complementary. Performance decreases when Wikipedia paragraphs are too long.

2. **Span Pooling Method**:  
    - Mean pooling: 64.48-65.84
    - **First-last token concatenation: 66.25-66.66** (consistently superior)
   
   **Conclusion**: The first and last tokens contain crucial boundary information, offering better discriminative power than mean pooling.

3. **Similarity Metric $\times$ Loss Function**:

   | Loss Function | Cosine | Dot Product | Euclidean |
   |----------|--------|-------------|-----------|
   | Triplet | 50.65 | 64.43 | 64.48 |
   | Cross-Entropy | 34.34 | 64.52 | **65.84** |
   
   **Conclusion**: Cross-Entropy + Euclidean distance is optimal. Cosine similarity performs significantly worse than the other two.

4. **Negative Sampling Strategy**:  
    - In-Batch Negative Sampling: 54.06-54.39 (much worse)
    - **Hard Negative Sampling (1 sample)**: 64.46-65.78
    - **Hard Negative Sampling (dynamic number)**: 64.48-65.84 (slightly better)
   
   **Conclusion**: Hard negatives are significantly better than in-batch negatives, and a dynamic number of hard negatives yields a marginal improvement.

5. **Label Embedding Update Frequency**:  
    - Update once per Epoch: 76.17
    - **Frequent updates + On-the-fly updates**: 82.32
   
   **Conclusion**: Frequent updates of cached embeddings are crucial, especially for large datasets (e.g., ZELDA).

### Iterative Prediction Strategy

After the initial predictions are completed, the top-$N$ predictions with the highest confidence are selected, and their label verbalized texts are inserted directly after the corresponding mentions in the original text (e.g., "Jose Reyes (baseball infielder)"). Then, the remaining mentions are re-encoded and predicted.

- **Purpose**: Allowing disambiguated entities to provide additional context for difficult samples.
- **Training Adaptation**: During training, label verbalized texts are randomly inserted for some mentions to simulate the inference scenario.
- **Effect**: On average, a slight improvement is achieved (AVG: 81.0 $\rightarrow$ 82.3), but the effect is inconsistent—some datasets even showed a decrease in performance.

### Loss & Training

The optimal configuration is Cross-Entropy loss + Euclidean distance + Hard Negative Mining + First-Last Pooling + frequent label embedding updates. Training is performed on the ZELDA dataset, containing 95,000 Wikipedia paragraphs, 2.6 million mentions, and approximately 820,000 unique entities.

## Key Experimental Results

### Main Results

| Method | AIDA-B | TWEEKI | SLINKS-SHAD | SLINKS-TOP | AVG (9 sets) |
|------|--------|--------|-------------|------------|---------|
| FEVRY_CL | 79.5 | 76.9 | 31.9 | 47.7 | 72.7 |
| GENRE_CL | 78.6 | 80.1 | 37.3 | 52.8 | 77.2 |
| FusionED | 80.1 | 81.4 | 41.5 | 57.9 | 78.7 |
| **VerbalizED** | 82.6 | 78.9 | **65.3** | **67.0** | **81.0** |
| + iter. training | **88.2** | 78.9 | **66.3** | 65.9 | **82.3** |

### Ablation Study (Train on AIDA $\rightarrow$ Test on ZELDA)

| Design Choice | Worst | Best | Gap |
|----------|------|------|------|
| Label Format | Title only: 63.68 | Title+Desc+Cat: 65.01 | +1.33 |
| Similarity | Cosine: 34.34 | Euclidean+CE: 65.84 | +31.50 |
| Negative Sampling | In-Batch: 54.06 | Hard: 65.84 | +11.78 |
| Pooling | Mean: 64.48 | First-Last: 66.25 | +1.77 |
| Cache Update | Per Epoch: 76.17 | Frequent: 82.32 | +6.15 |

### Key Findings

1. **VerbalizED significantly outperforms other methods on the Shadowlinks series datasets**: 65.3 vs 41.5 (+57% improvement) for the second-best on SLINKS-SHADOW, since it does not rely on candidate lists (which only achieve a 56.7% recall rate for rare entities).
2. **Long documents are advantageous scenarios**: Best performance is observed on AIDA-B (news articles) and WNED-WIKI (Wikipedia).
3. **Short texts are a disadvantage**: Weak performance on TWEEKI (Twitter) and REDDIT-COMM due to insufficient context.
4. **The iterative strategy shows unstable effects**: Positive example—disambiguating "Peggy Olson" helps correctly link "#madmen" to Mad_Men; Negative example—inserting tags for two sports teams causes "Dundee" (a person's name) to be incorrectly linked to the sports team.
5. **Not relying on candidate lists is a key advantage**: Retrieval is performed globally over an open set of 820,000 entities, avoiding the problem of missing rare entities in candidate lists.

## Highlights & Insights

- **High engineering value of systematic ablation**: The impact of each design choice is clearly quantified. For instance, the choice of similarity metric (Cosine vs Euclidean) can lead to an F1 gap of over 30 points! These findings can directly guide the design of other dense retrieval systems.
- **Complementary advantages of label verbalization + candidate-free approach**: Verbalization provides rich entity representations and eliminates dependence on pre-compiled candidate lists, significantly boosting the capability to disambiguate rare/overshadowed entities.
- **Honest evaluation of iterative prediction**: The paper frankly admits the inconsistent performance of the iterative strategy and provides analysis of positive and negative examples, ultimately recommending the base architecture instead of the iterative variant. This honest experimental attitude is highly commendable.

## Limitations & Future Work

1. Due to computational resource constraints, ablation experiments were conducted on AIDA (a smaller dataset), so some findings might not fully generalize to the scale of ZELDA.
2. High dependency on the availability of Wikidata descriptions—performance may degrade for entities with missing descriptions.
3. The training cost of the iterative variant is high and its performance is unstable, requiring further research on error propagation issues.
4. Evaluations were only conducted on English datasets; its multilingual generalization ability remains unknown.
5. Some hyperparameters (e.g., the margin of Triplet Loss) were not systematically searched.

## Related Work & Insights

- **BLINK** (Wu et al., 2020): Uses a Dual Encoder for candidate retrieval + a Cross Encoder for reranking. VerbalizED bypasses the expensive Cross Encoder step.
- **GENRE** (De Cao et al., 2021): A generative method that directly generates entity titles. VerbalizED's retrieval-based method is more flexible on open sets.
- **FusionED** (Wang et al., 2024): Integrates entity descriptions using an encoder-decoder architecture. VerbalizED achieves better results with a simpler Dual Encoder architecture.
- **Insights**: The concept of label verbalization can be extended to other classification/retrieval tasks (e.g., intent classification, product matching). Transforming classification labels from IDs into semantic descriptions could be universally beneficial.

## Rating

- **Novelty**: ⭐⭐⭐ — The core contribution lies in systematic evaluation rather than methodological innovation; each individual technique (verbalization, hard negatives) is existing. The iterative prediction has some novelty but shows unstable effects.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — The ablation is extremely thorough, comparing 5 design choices individually. The evaluation covers multiple domains across 9 test sets with comprehensive quantitative and qualitative analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The structure is clear, each ablation has unambiguous conclusions, table designs are well-formatted, and the related work is comprehensive.
- **Value**: ⭐⭐⭐⭐ — The ablation findings offer direct guidance for the entity disambiguation and dense retrieval communities. Achieving SOTA on ZELDA brings practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[ACL 2025\] Multi-Hop Question Generation via Dual-Perspective Keyword Guidance](multi-hop_question_generation_via_dual-perspective_keyword_guidance.md)
- [\[ACL 2025\] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)
- [\[ICML 2025\] Optimal Auction Design in the Joint Advertising](../../ICML2025/others/optimal_auction_design_in_the_joint_advertising.md)
- [\[CVPR 2026\] Evidential Deep Partial Label Learning to Quantify Disambiguation Uncertainty](../../CVPR2026/others/evidential_deep_partial_label_learning_to_quantify_disambiguation_uncertainty.md)

</div>

<!-- RELATED:END -->
