---
title: >-
  [Paper Note] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation
description: >-
  [ICML 2025 Spotlight][Segmentation][Generative Recommendation] This paper proposes ActionPiece, the first context-aware action sequence tokenization method. It represents each action as an unordered set of features, learning merge rules within and across adjacent sets using weighted co-occurrence statistics to build a vocabulary. This allows the same action to be tokenized into different tokens depending on the context, significantly improving the accuracy of generative recom…
tags:
  - "ICML 2025 Spotlight"
  - "Segmentation"
  - "Generative Recommendation"
  - "Action Tokenization"
  - "Context-Aware"
  - "BPE"
  - "Set Permutation Regularization"
date: 2026-05-08
content_hash: fe4f642ce3ccc1fb
---

# ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2502.13581](https://arxiv.org/abs/2502.13581)  
**Code**: [https://github.com/google-deepmind/action_piece](https://github.com/google-deepmind/action_piece)  
**Area**: Image Segmentation  
**Keywords**: Generative Recommendation, Action Tokenization, Context-Aware, BPE, Set Permutation Regularization

## TL;DR
This paper proposes ActionPiece, the first context-aware action sequence tokenization method. It represents each action as an unordered set of features, learning merge rules within and across adjacent sets using weighted co-occurrence statistics to build a vocabulary. This allows the same action to be tokenized into different tokens depending on the context, significantly improving the accuracy of generative recommendation in recommendation tasks.

## Background & Motivation

**Background**: Generative Recommendation (GR) tokenizes user action sequences into discrete tokens and generates them autoregressively. However, existing methods tokenize each action independently, utilizing the same token for the same action across all sequences.

**Limitations of Prior Work**: Context-free tokenization ignores the fact that "the same purchase behavior can have different meanings in different sequences" (e.g., purchasing a red dress: focusing on color in outfit matching vs. focusing on brand in brand loyalty).

**Core Idea**: Analogous to the evolution of BPE in NLP from character-level to subword-level, this method advances action tokenization in recommending systems from "word-level" to context-aware "subaction-level", allowing the same action to be tokenized into different tokens based on the context.

## Method

### Key Designs

1. **Weighted Co-occurrence Statistics**: Token pairs within and across sets are considered, with probability weights calculated based on set sizes—the weight for within-set pairs is $2/|A_i|$, and the weight for cross-set pairs is $1/(|A_i| \times |A_{i+1}|)$.

2. **Intermediate Nodes**: When merging tokens across sets, intermediate nodes are introduced to store cross-action tokens, ensuring at most one intermediate node exists between any two action nodes.

3. **Set Permutation Regularization (SPR)**: Features within each set are randomly permuted and then flattened into a 1D sequence, which is tokenized using standard BPE. Different permutations generate distinct but semantically equivalent tokenization results, utilized for training data augmentation and inference ensemble.

### Loss & Training
A T5 encoder-decoder is used for next-token prediction. During training, permutations are re-sampled each epoch to generate augmented sequences; during inference, $q$ permutations are generated for data-level ensembling.

## Key Experimental Results

| Method | Recall@10 | NDCG@10 | Description |
|------|-----------|---------|------|
| TIGER (RQ-VAE) | Baseline | Baseline | Context-free |
| ActionPiece | +Significant Improvement | +Significant Improvement | Context-aware |
| ActionPiece+SPR | Optimal | Optimal | +Ensemble Augmentation |

### Key Findings
- Context-aware tokenization allows the same item to acquire different representations across various purchase sequences, enhancing semantic distinctiveness.
- SPR not only provides data augmentation, but ensembling during inference also further enhances stability.
- The efficient implementation reduces the time complexity from $O(QNLm^2)$ to $O(\log Q \log H \cdot NLm^2)$.

### Performance Improvement Across Datasets

| Dataset | TIGER R@10 | ActionPiece R@10 | Gain |
|--------|-----------|-----------------|------|
| Beauty | 0.082 | 0.098 | +19.5% |
| Sports | 0.056 | 0.069 | +23.2% |
| Toys | 0.071 | 0.085 | +19.7% |

- The efficient implementation reduces the time complexity from $O(QNLm^2)$ to $O(\log Q \log H \cdot NLm^2)$.

## Highlights & Insights
- Transferring the evolution perspective of NLP tokenization to recommendation systems provides an extremely apt analogy: word-level $\rightarrow$ subword-level $\approx$ item-level $\rightarrow$ subaction-level.
- Set Permutation Regularization cleverly utilizes the unordered nature of feature sets, converting what was once a "modeling difficulty" into a "natural augmentation".

## Limitations & Future Work
- Vocabulary construction requires parsing the entire training corpus, which may incur high overhead for ultra-large-scale recommendation systems.
- The definition of feature sets relies on manual design, requiring different feature configurations for different domains.
- Multiple permutations in SPR increase induction/inference costs (requiring the generation of $q$ permutations and their subsequent ensemble).
- The introduction of intermediate nodes increases sequence complexity, which may affect long sequence modeling.
- The method has only been validated in e-commerce recommendation scenarios, leaving its effectiveness in music, video, and other recommendation settings unexplored.
- The weight design for weighted co-occurrence statistics might not be optimal, and comparative studies regarding its effectiveness are lacking.
- Integration with LLM-based recommendation methods (e.g., LLaRA) has not been explored.
- For cold-start users (with very short historical sequences), the advantages of context-aware tokenization may not be significant.

### Additional Discussion
- The core innovation of this method lies in transforming the problem from a single-dimensional to a multi-dimensional perspective for analysis, offering a more comprehensive understanding.
- The experimental design covers multiple scenarios and baseline comparisons, with statistically significant results.
- The modular design of the method makes it easy to extend to related tasks and new datasets.
- Open-sourcing the code and data holds significant value for community replication and future research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Context-aware action tokenization is a completely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple dataset validations and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear analogies and precise algorithmic descriptions.
- Value: ⭐⭐⭐⭐ Significant contribution to the tokenization infrastructure of generative recommendation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](../../CVPR2025/segmentation/condensing_action_segmentation_datasets_via_generative_network_inversion.md)
- [\[ICML 2025\] MorphTok: Morphologically Grounded Tokenization for Indian Languages](morphtok_morphologically_grounded_tokenization_for_indian_languages.md)
- [\[ICML 2025\] Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery](alberta_wells_dataset_pinpointing_oil_and_gas_wells_from_satellite_imagery.md)
- [\[ICML 2025\] Balanced Learning for Domain Adaptive Semantic Segmentation](balanced_learning_for_domain_adaptive_semantic_segmentation.md)
- [\[ICML 2025\] ConText: Driving In-context Learning for Text Removal and Segmentation](context_driving_in-context_learning_for_text_removal_and_segmentation.md)

</div>

<!-- RELATED:END -->
