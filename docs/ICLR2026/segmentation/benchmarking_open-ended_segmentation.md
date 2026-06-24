---
title: >-
  [Paper Note] Benchmarking Open-ended Segmentation
description: >-
  [ICLR2026][Segmentation][Open-ended Segmentation] Focusing on the evaluation loophole in "open-ended segmentation" where model-generated free-form text is forcibly mapped back to a fixed vocabulary via embedding similarity, this paper introduces a mapping function based on lexical relationships (exact/synonym/hyponym/meronym) and a Lexical Alignment Curve (LAC) protocol. This shifts evaluation accuracy from a 37.7% deviation from human judgement to over 90% alignment. Further…
tags:
  - "ICLR2026"
  - "Segmentation"
  - "Open-ended Segmentation"
  - "Evaluation Protocols"
  - "Lexical Alignment"
  - "Panoptic Segmentation"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: c95d2729c292f37f
---

# Benchmarking Open-ended Segmentation

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=QSmwRnw8GP](https://openreview.net/forum?id=QSmwRnw8GP)  
**Code**: https://github.com/BCV-Uniandes/open-ended_segmentation_benchmark  
**Area**: Semantic Segmentation / Open-ended Recognition / Benchmarking  
**Keywords**: Open-ended Segmentation, Evaluation Protocols, Lexical Alignment, Panoptic Segmentation, Contrastive Learning

## TL;DR
Focusing on the evaluation loophole in "open-ended segmentation" where model-generated free-form text is forcibly mapped back to a fixed vocabulary via embedding similarity, this paper introduces a mapping function based on lexical relationships (exact/synonym/hyponym/meronym) and a Lexical Alignment Curve (LAC) protocol. This shifts evaluation accuracy from a 37.7% deviation from human judgement to over 90% alignment. Furthermore, the first open-ended segmentation MLLM with contrastive loss (OPAL) is trained, achieving a new SOTA on open-ended panoptic segmentation.

## Background & Motivation
**Background**: Visual recognition has progressed from image-level labels to pixel-level tasks, and from closed-set to open-vocabulary recognition. The latter allows for identifying unseen categories at test time, but the vocabulary remains a predefined finite set; the evaluation is essentially "selecting the correct one from given labels." A more radical step is **open-ended segmentation**: models no longer pick from candidates but directly generate free-form text descriptions for each visual region (e.g., a "yellow dog," "golden retriever," or "a dog's tail").

**Limitations of Prior Work**: Free-form text is one-to-many, varying in phrasing, granularity, and structure. To calculate standard recognition metrics such as PQ/mIoU/AP, these free-form descriptions must be mapped back to a fixed category vocabulary of the test set. The mainstream approach (following Zhang et al. 2020) utilizes **Sentence-BERT embedding similarity**: encoding descriptions into vectors and assigning the category with the highest cosine similarity as the model's output.

**Key Challenge**: This "forced-choice" mapping has two critical flaws. First is **forced single-choice**—even if no category in the vocabulary truly fits the description, one must be assigned, leading to systematic misjudgment. Second is the **inherited bias of the embedding model**—the mapping accuracy depends on the training distribution of Sentence-BERT rather than the model's true visual understanding. Systematic human verification on Cityscapes reveals that Sentence-BERT mapping **deviates from human judgment by as much as 37.7%**, performing particularly poorly on "stuff" (amorphous regions like sky or road). In other words, current open-ended segmentation leaderboards are likely distorted by a flawed evaluation function.

**Goal**: (1) Quantitatively prove that embedding similarity mapping deviates from humans; (2) Design a more human-aligned, reproducible, and standardized mapping and evaluation protocol, re-benchmarking existing SOTA models; (3) Explore whether contrastive learning can improve open-ended segmentation models.

**Core Idea**: Replace "description-to-category" mapping via single embedding similarity with **explicitly enumerated lexical relationships**. A category corresponds to a set of valid terms (exact, synonym, hyponym, meronym), using linguistic hierarchies instead of black-box vectors to determine if a description matches a category.

## Method

### Overall Architecture
The paper follows two tracks: the **evaluation side** (core contribution: Lexical Alignment Mapping + LAC protocol) and the **method side** (OPAL model).

The evaluation workflow involves: first, offline construction of a "valid vocabulary" for each category $c_i$ in the test set by mining noun candidates from large-scale image-text data; then, using an LLM to determine which nouns are semantically related to which categories, archiving them into four progressive lexical levels: **exact $\rightarrow$ synonym $\rightarrow$ hyponym $\rightarrow$ meronym**. During evaluation, for a free-form description, subjects are extracted via syntactic tagging and lemmatized. The system then checks for hits in the noun lists of various levels: if the ground truth category is hit, it is mapped to that truth; otherwise, it is mapped to the category with the strongest relationship. If no hits occur, it is categorized as background. Recognition metrics are calculated at each lexical level. Plotting the "lexical level" on the x-axis and the metric value on the y-axis yields the **Lexical Alignment Curve (LAC)**, with the area under the curve serving as the comprehensive score.

On the method side, OPAL is based on the Osprey architecture (CLIP visual encoder + vision projection + mask-aware region extractor + LoRA-tuned LLaMA). The primary modification is the addition of a **parallel contrastive loss** alongside the original generative loss to align regional visual embeddings with text descriptions in a joint space.

### Key Designs

**1. Lexical Mapping Function: Replacing "Forced Single-choice" with Linguistic Relationships**

To address the issues of Sentence-BERT's forced choice and inherited bias, the authors redefine "which category a description belongs to" as a **many-to-many** lexical matching problem. Formally, given a description $T_i$ and test vocabulary $\{C_i\}_{i=1}^N$, a mapping $f: T \rightarrow \{C_i\}_{i=1}^N$ is defined to map descriptions to all categories with which a lexical relationship exists. These relationships are divided into four **cumulative** levels: exact (string equality) $\subset$ synonym (e.g., puppy/pooch for dog) $\subset$ hyponym (e.g., golden retriever/maltese for dog) $\subset$ meronym (e.g., tail/paw/snout for dog). This design satisfies three properties aligned with human judgment: **semantic precision** (no forced classification without a valid relationship, allowing background classification), **flexibility** (mapping one category to multiple descriptions), and **lexical proximity** (allowing descriptions to relate to multiple categories at different granularities). Unlike Sentence-BERT, every hit here can be traced to an explicit linguistic relationship, ensuring no errors even in simple "exact string match" cases—where experiments show Sentence-BERT can still fail.

**2. Valid Vocabulary Construction: Noun Mining + LLM Association**

The effectiveness of lexical mapping depends on the completeness and accuracy of each category's "valid vocabulary." The authors mine candidate nouns (including compound nouns) from large-scale image-text pairs and use an LLM to judge semantic associations. This automates the labor-intensive task of compiling synonyms, hyponyms, and meronyms, making the protocol standardized and reproducible. Ablations show that vocabulary size directly impacts evaluation: larger vocabularies increase candidate matches and reduce erroneous background assignments.

**3. Lexical Alignment Curve (LAC): Visualizing Description Accuracy as a Diagnostic Curve**

A single scalar score cannot distinguish between "precise descriptions" and "vague hypernyms." LAC spreads evaluation across semantic dimensions (four lexical levels). Due to the cumulative nature of the levels, all methods' scores increase with flexibility, but the **curve shape provides diagnostic value**. For instance, non-MLLM open-vocabulary methods show a significant jump from synonym to hyponym, indicating a tendency to output specific semantic concepts. Models with high scores at lower levels (exact/synonym) indicate more precise descriptions.

**4. OPAL: The First Open-ended Segmentation MLLM with Contrastive Loss**

While contrastive learning has been widely used in open-vocabulary segmentation to align regions and text, it has remained largely unexplored in open-ended (generative) segmentation. OPAL introduces a **parallel contrastive loss** alongside the standard generative loss $\mathcal{L}_{gen}$. This requires two forward passes: the generative branch feeds visual and mask embeddings to LLaMA to generate descriptions, while the contrastive branch performs contrastive learning between the **language embedding of the last layer** and the mask embedding. This forces the model to align regions and descriptions at the representation level, leading to more robust generation.

### Loss & Training
OPAL jointly optimizes two complementary losses: the generative loss $\mathcal{L}_{gen}$ (standard language modeling objective for free-form text generation conditioned on visual and mask embeddings) and the contrastive loss $\mathcal{L}_{con}$ (contrastive alignment between region mask embeddings and the final text language embedding). LLaMA is fine-tuned using LoRA, with CLIP as the visual encoder.

## Key Experimental Results

### Human Verification (Protocol Validity)
A two-stage human annotation was performed on 2,800 region-level descriptions from the Cityscapes validation set (generated by Osprey-7B and OPAL). The focus was on samples where Sentence-BERT and the proposed mapping disagreed.

| Mapping Method | Alignment with Human Judgement (All) | "Stuff" Classes | Recovery of Human Labels (Disagreement Subset) |
| :--- | :--- | :--- | :--- |
| Sentence-BERT | ~60% | ~50% | 4.8% |
| Lexical Mapping (Ours) | >90% | Significantly Higher | **84.4%** |

On the high-disagreement subset, the proposed mapping hit human labels 84.4% of the time, compared to 4.8% for Sentence-BERT. Even at the meronym level, our alignment reached 74%.

### Main Results: Re-benchmarking Open-ended Panoptic Segmentation
Using the proposed protocol on ADE20K and Cityscapes validation sets, OPAL outperformed others across PQ, mIoU, and AP:

| Method | ADE20K PQ | ADE20K mIoU | ADE20K AP | Cityscapes PQ | Cityscapes mIoU | Cityscapes AP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MasQCLIP | 38.6 | 34.2 | 26.7 | 43.8 | 45.6 | 23.4 |
| Baseline (MLLM Labeling + MasQCLIP) | 42.9 | 39.8 | 28.5 | 46.9 | 53.7 | 28.8 |
| GPT4RoI-7B | 44.5 | 36.5 | 33.7 | 39.7 | 40.6 | 25.3 |
| Osprey-7B | 46.6 | 36.9 | 36.4 | 50.2 | 55.4 | 31.3 |
| **OPAL (Ours)** | **48.8** | **38.9** | **38.4** | **52.8** | **56.1** | **31.9** |

OPAL leads by at least 2 absolute points on ADE20K and 0.6 points on Cityscapes. Robustness analysis shows that OPAL not only has a higher mean but reduces output variance by nearly 50% compared to Osprey.

### Ablation Study

| Configuration | Key Finding |
| :--- | :--- |
| Lexical Mapping with Different LLMs | Negligible metric differences and unchanged rankings; protocol does not over-rely on a specific LLM's bias. |
| Noun List Coverage (20/40/60/80/100%) | LAC increases monotonically with vocabulary size; full vocabulary is ~6 points higher than the 20% subset. |
| OPAL w/o Contrastive Loss (≈Osprey) | Performance drops to Osprey levels, and output variance approximately doubles. |

### Key Findings
- The evaluation function is a neglected confounding variable: changing the mapping function alters the relative rankings of the entire leaderboard.
- The primary gain of contrastive loss in open-ended segmentation is **reducing variance**, stabilizing the generative model.
- Vocabulary coverage is strongly correlated with scores, implying that future work must use fixed vocabulary construction methods for comparability.

## Highlights & Insights
- **Deconstructing "Black-box" Evaluation**: Categorizing evaluation into linguistic levels (exact/synonym/hyponym/meronym) provides better accuracy and diagnostic power.
- **Quantifying a Default Assumption**: By proving the 37.7% deviation of the widely-adopted Sentence-BERT mapping, the paper emphasizes the importance of verifying metrics before using them.
- **Contrastive Loss for Generative Segmentation**: While common in open-vocabulary tasks, the application to MLLM generation and its benefit in reducing variance is a valuable insight.

## Limitations & Future Work
- **Linguistic Dependencies**: Lexical mapping relies on WordNet-style relationships and LLM-mined noun lists, which may struggle with abstract or culture-specific concepts.
- **Vocabulary Sensitivity**: The 6-point gap based on vocabulary size means results are not fully comparable unless the vocabulary is standardized.
- **Incremental Architecture Changes**: The innovation of OPAL (Osprey + contrastive loss) is relatively light; the primary value lies in the evaluation protocol.

## Related Work & Insights
- **vs. Sentence-BERT / CLIPScore Mapping**: These use single-choice embedding similarity, inheriting biases and failing on "stuff"; ours uses explicit many-to-many relationships, aligned >90% with humans.
- **vs. Captioning Metrics (BLEU/METEOR/CIDEr/SPICE)**: Those rely on n-gram overlap or scene graph similarity, which correlate poorly with human judgment in open-ended generation; LAC specifically addresses the mapping phase unique to open-ended segmentation.
- **vs. Osprey**: Osprey uses mask-level visual prompts and pure generation loss; OPAL adds contrastive loss to improve precision and halve output variance.

## Rating
- Novelty: ⭐⭐⭐⭐ Solid innovation in evaluation protocols; OPAL model innovation is incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes two-stage human verification and re-benchmarking on multiple datasets/tasks.
- Writing Quality: ⭐⭐⭐⭐ Motivation and design are clear; effective visualization of problems and solutions.
- Value: ⭐⭐⭐⭐ Corrects a faulty default assumption in the sub-field and provides a reproducible protocol.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking](../../NeurIPS2025/segmentation/step_a_unified_spiking_transformer_evaluation_platform_for_fair_and_reproducible.md)
- [\[ICLR 2026\] WOW-Seg: A Word-Free Open World Segmentation Model](wow-seg_a_word-free_open_world_segmentation_model.md)
- [\[CVPR 2026\] MARIS: Marine Open-Vocabulary Instance Segmentation](../../CVPR2026/segmentation/maris_marine_open-vocabulary_instance_segmentation.md)
- [\[ECCV 2024\] Diffusion Models for Open-Vocabulary Segmentation](../../ECCV2024/segmentation/diffusion_models_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] The Power of Prior: Training-Free Open-Vocabulary Semantic Segmentation with LLaVA](../../CVPR2026/segmentation/the_power_of_prior_training-free_open-vocabulary_semantic_segmentation_with_llav.md)

</div>

<!-- RELATED:END -->
