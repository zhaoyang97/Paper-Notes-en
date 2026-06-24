---
title: >-
  [Paper Note] ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking
description: >-
  [ACL 2025][Medical LLM][biomedical entity linking] The ANGEL framework is proposed, introducing negative sample training to generative Biomedical Entity Linking (BioEL) for the first time. Through a two-stage strategy (positive-only training + negative-aware preference optimization), it significantly improves the model's ability to distinguish between entities with similar surface forms but different semantics, achieving an average top-1 accuracy improvement of 1.7% across fi…
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "biomedical entity linking"
  - "generative model"
  - "negative sampling"
  - "preference optimization"
  - "DPO"
date: 2026-05-08
content_hash: d93a40724d8f613f
---

# ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking

**Conference**: ACL 2025  
**arXiv**: [2408.16493](https://arxiv.org/abs/2408.16493)  
**Code**: None  
**Area**: Medical NLP  
**Keywords**: biomedical entity linking, generative model, negative sampling, preference optimization, DPO

## TL;DR

The ANGEL framework is proposed, introducing negative sample training to generative Biomedical Entity Linking (BioEL) for the first time. Through a two-stage strategy (positive-only training + negative-aware preference optimization), it significantly improves the model's ability to distinguish between entities with similar surface forms but different semantics, achieving an average top-1 accuracy improvement of 1.7% across five benchmark datasets.

## Background & Motivation

### Problem Definition

Biomedical Entity Linking (BioEL) aims to map entity mentions in text to standardized concepts in a knowledge base (such as UMLS or MeSH). This task faces two core challenges:

**Synonym Diversity**: The same concept can be expressed in various ways. For instance, synonyms for "ADHD" include hyperkinetic disorder and attention deficit hyperactivity disorder.

**Surface Form Ambiguity**: Different concepts may share similar names. For example, "ADA" can refer to adenosine deaminase or American Diabetes Association.

### Limitations of Prior Work

Current methods fall into two main categories:

- **Similarity-based methods** (e.g., BioSYN, SapBERT): Encode mentions and entities into the same vector space to compute similarity. The disadvantage is that they require a large amount of memory to index the embedding vectors of all candidate entities, and the single-vector representation of bi-encoders may limit representation quality.
- **Generative methods** (e.g., GENRE, GenBioEL): Directly generate the most likely entity name based on an encoder-decoder architecture, which is more memory-efficient. However, they are **trained using only positive samples** and do not explicitly learn from negative samples. This can cause models to overfit surface features, making it difficult to distinguish between entities with similar forms but different semantics.

### Core Motivation

Similarity-based methods utilize negative samples through synonym marginalization and contrastive learning, but these strategies cannot be directly transferred to generative models. ANGEL aims to fill this gap, enabling generative models to also learn from negative samples.

## Method

### Overall Architecture

ANGEL is a two-stage training framework that can be applied to both pre-training and fine-tuning:

**Stage 1: Positive-only Training**

- Train the generative model to produce synonyms with the same identifier in the knowledge base given an input mention.
- Use TF-IDF (trigram) similarity to select the top-$k$ synonyms most similar to the input mention as the training targets.
- Input format: `[BOS] c- [ST] m [ET] c+ [EOS]`, with the decoder prefix prompt as `m is`.

**Stage 2: Negative-aware Training**

1. **Collect positive and negative sample pairs**: Retrieve top-$k$ predictions of the model for each mention in the training set to construct triples $(x, e_w, e_l)$.
    - $e_w$: Correct (preferred) entity.
    - $e_l$: Incorrect (dispreferred) entity.
2. **Filtering strategy**: Keep only sample pairs where the model ranks the incorrect entity higher than the correct one; if the top-1 prediction is already correct, pair it with the highest-ranked incorrect entity.
3. **Preference optimization**: Update the model using the Direct Preference Optimization (DPO) loss function.

In the DPO loss function, the scoring function is defined as the log-likelihood ratio between the current model and the reference model (which is the model trained in the first stage). The temperature parameter $\beta$ controls the strength of preference.

### Application in Pre-training

- Automatically generate training data using the UMLS knowledge base (3.09M entities, 199K of which contain definitions).
- Construct context using template clauses, such as "[ST] s [ET] is defined as d_y" or "[ST] s1 [ET] has synonyms such as s2".
- Positive-only training: Select the most similar synonym based on TF-IDF as the target for each entity.
- Negative-aware training: Select negative samples from entities that are similar in TF-IDF but have different identifiers (instead of model predictions, to improve efficiency).
- Save checkpoints every 500 steps, training for a total of 5 epochs using 8 A100 GPUs for 12 hours.

## Key Experimental Results

### Main Results

Top-1 Accuracy (%) on five BioEL benchmark datasets:

| Model | NCBI | BC5CDR | COMETA | AAP | MM-ST21pv | Average |
|------|------|--------|--------|-----|-----------|------|
| SapBERT | 92.3 | 88.6 | 75.1 | 89.0 | 50.3 | 79.1 |
| Prompt-BioEL | 91.9 | 94.3 | 82.7 | 89.7 | 72.6 | 86.2 |
| GenBioEL (reproduced) | 91.0 | 93.1 | 80.9 | 89.3 | 70.7 | 85.0 |
| + ANGEL_FT | 92.5 (+1.5) | 94.4 (+1.3) | 82.4 (+1.5) | 89.9 (+0.6) | 71.9 (+1.2) | 86.2 (+1.2) |
| + ANGEL_PT+FT | **92.8** (+1.8) | **94.5** (+1.4) | **82.8** (+1.9) | **90.2** (+0.9) | **73.3** (+2.6) | **86.7** (+1.7) |
| BioBART + ANGEL_FT | 91.9 (+2.5) | 94.7 (+1.2) | 82.2 (+0.9) | 89.9 (+0.6) | 73.4 (+2.1) | 86.4 (+1.4) |

### Pre-training Effect Analysis

Comparison of accuracy under different pre-training strategies:

| Model | Fine-tune | BC5CDR | AAP |
|------|------|--------|-----|
| BART | No | 0.8 | 15.6 |
| GenBioEL | No | 33.1 | 50.6 |
| + ANGEL | No | **49.7** | **61.5** |
| BART | Yes | 93.0 | 88.7 |
| GenBioEL | Yes | 93.1 | 89.3 |
| + ANGEL | Yes | **94.5** | **90.2** |

Key findings: ANGEL pre-training shows very significant improvements when not fine-tuned (BC5CDR +16.6%, AAP +10.9%), and maintains its advantage after fine-tuning.

### Ablation Study: Negative Pair Construction Strategies

| Variant | NCBI | BC5CDR | COMETA | AAP | MM-ST21pv | Average |
|------|------|--------|--------|-----|-----------|------|
| ANGEL (Full) | 92.8 | 94.5 | 82.8 | 90.2 | 73.3 | 86.7 |
| Model predicted negatives -> TF-IDF negatives | 91.8 | 94.4 | 81.6 | 90.0 | 71.5 | 85.9 |
| Keep only misordered pairs -> Keep all possible pairs | 92.9 | 94.0 | 81.9 | 90.0 | 72.0 | 86.2 |
| Top-5 -> Top-10 | 92.5 | 94.0 | 82.1 | 89.6 | 72.6 | 86.2 |
| No negative training (GenBioEL) | 91.0 | 93.1 | 80.9 | 89.3 | 70.7 | 85.0 |

Core conclusion: Selecting negative samples from the model's own predictions is more effective than selecting via TF-IDF (average difference of 0.8%).

## Highlights & Insights

1. **Pioneering**: ANGEL is the first framework to introduce negative sample training to generative entity linking, bringing DPO preference optimization to BioEL.
2. **Model-agnostic**: The framework is applicable to multiple backbone models (BART/BioBART/GenBioEL), achieving consistent improvements (0.9% to 1.7%).
3. **Dual-stage Versatility**: Effective in both the pre-training and fine-tuning stages, with stackable benefits.
4. **In-depth Analysis**: TF-IDF bin similarity analysis shows that negative training is particularly advantageous when dealing with hard negatives of high structural/morphological similarity.
5. **Outperforming Re-ranking Methods**: Outperforms Prompt-BioEL (average +0.5%) without the need for an additional re-ranking module.

## Limitations & Future Work

1. **Model Architecture Constraints**: Only verified on encoder-decoder models; not tested on decoder-only models (such as BioGPT) or Large Language Models (LLMs).
2. **Domain Constraints**: Only evaluated in the biomedical domain; generalization capability to open-domain entity linking has not been validated.
3. **Difficulty in Low-similarity Scenes**: When the surface form of the input mention is highly discrepant from the gold entity (TF-IDF similarity in the 0-0.2 range), the accuracy drop is significant, with accuracy at only 34.2%.
4. **Limited Top-5 Improvement**: While Top-1 accuracy improved significantly, the improvement in Top-5 on some datasets is relatively small.
5. **Training Complexity**: Requires completing positive-only training first, then gathering predictions to construct negative pairs, making the pipeline more complex than standard methods.

## Related Work & Insights

- **Similarity-based methods**: BioSYN (synonym marginalization), SapBERT (contrastive learning), ResCNN, KRISSBERT (clustering)
- **Generative methods**: GENRE (first generative EL), GenBioEL (UMLS pre-trained BART), BioBART (biomedical continued pre-training)
- **Hybrid methods**: Prompt-BioEL (retrieval + re-ranking)
- **Preference Optimization**: DPO (Rafailov et al., 2024), LambdaRank (Burges, 2010)

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | 4/5 | First to introduce negative training and DPO to generative BioEL, a natural and effective approach. |
| Experimental Thoroughness | 5/5 | Five datasets, three backbone models, detailed ablation studies, and deep analysis. |
| Writing Quality | 4/5 | Clearly structured, motivates the work well, and features intuitive case studies. |
| Value | 4/5 | Generic and easy-to-integrate framework, with code publicly available. |
| Overall Rating | 4/5 | Solid work that successfully transfers ideas of RLHF/DPO to the BioEL task. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](../../ACL2026/medical_nlp/biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[ACL 2025\] Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](urca_biomedical_evidence_extraction.md)
- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_nlp/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)

</div>

<!-- RELATED:END -->
