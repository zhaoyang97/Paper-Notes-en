---
title: >-
  [Paper Note] Making FETCH! Happen: Finding Emergent Dog Whistles Through Common Habitats
description: >-
  [ACL 2025][LLM (Other)][Implicit hate speech] Proposes the FETCH! benchmark and the EarShot system to discover emergent "dog whistles" (coded expressions with dual meanings) in large-scale social media corpora, leveraging a combination of vector databases and LLMs to achieve a 2-20 percentage point improvement in F-score over existing methods.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Implicit hate speech"
  - "coded language detection"
  - "emergent vocabulary discovery"
  - "vector database"
  - "LLM"
date: 2026-05-08
content_hash: 7c610d29ad9ec2a5
---

# Making FETCH! Happen: Finding Emergent Dog Whistles Through Common Habitats

**Conference**: ACL 2025  
**arXiv**: [2412.12072](https://arxiv.org/abs/2412.12072)  
**Code**: [https://github.com/KuleenS/FETCH-Dog-Whistle](https://github.com/KuleenS/FETCH-Dog-Whistle)  
**Area**: Other  
**Keywords**: Implicit hate speech, coded language detection, emergent vocabulary discovery, vector database, LLM

## TL;DR

Proposes the FETCH! benchmark and the EarShot system to discover emergent "dog whistles" (coded expressions with dual meanings) in large-scale social media corpora, leveraging a combination of vector databases and LLMs to achieve a 2-20 percentage point improvement in F-score over existing methods.

## Background & Motivation

A "dog whistle" is a coded expression with a dual meaning: it conveys one message to the general public (out-group) while signaling controversial political viewpoints to the target audience (in-group) while maintaining "plausible deniability." For example, "dual citizen" superficially refers to dual nationality, but in certain contexts, it acts as an antisemitic dog whistle.

Current methods to detect dog whistles primarily rely on manually curated lexicons, but several critical issues exist:

**High maintenance cost**: Lexicons require continuous manual updates and cannot keep up with the dynamic evolution of language.

**Lagging**: Language abbreviations, variants, and emergent expressions constantly appear, such as "cosmopolitan" being shortened to "cosmos".

**Bypassing content moderation**: The benign surface meaning of dog whistles allows them to easily bypass existing toxicity and hate speech detectors.

This work advances the problem from "dog whistle detection" (determining whether a text contains a dog whistle given a known lexicon) to the more challenging "dog whistle discovery" (discovering unknown, emergent dog whistles in a corpus), representing a completely new task definition.

## Method

### Overall Architecture

The contribution of this work consists of two parts: the FETCH! benchmark (defining the task and evaluation protocols) and the EarShot system (proposing a strong baseline method).

Task Definition: Given a corpus and a set of known seed dog whistles, the system must utilize the seed terms and the corpus to discover new dog whistles.

### Key Designs

1. **Three case studies of the FETCH! benchmark**:

    - **Synthetic (Reddit)**: An idealized scenario containing GPT-4 annotated dog whistles in every post, with approximately 16,000 posts.
    - **Balanced (Gab)**: A medium-density scenario from the right-wing platform Gab, consisting of approximately 300,000 posts, where dog whistles appear more frequently than average.
    - **Realistic (Twitter)**: A real-world scenario from the Twitter API, featuring about 7 million tweets where dog whistles are sparsely distributed.

2. **Seed dog whistle partitioning**: Uses a lexicon of approximately 340 English root dog whistles compiled by Mendelsohn et al. Stratified sampling by n-gram length allocates 20% as seeds and 80% for testing. Precision, Data Potential Recall (DPR), and $F_{0.5}$ are used as evaluation metrics (with a preference for precision to alleviate manual auditing burdens).

3. **EarShot System**: Consists of three stages:

    - **Stage 1 (Vectorization)**: All posts are encoded into vectors using all-MiniLM-L6-v2 and stored in a ChromaDB vector database.
    - **Stage 2 (Nearest Neighbor Retrieval)**: Retrieves vectors of posts containing seed dog whistles to find the nearest neighbor posts (excluding themselves), capturing posts that are semantically related but do not share exact terms.
    - **Stage 3 (Two Paths)**:
        - **DIRECT Path**: Directly passes the nearest neighbor posts to an LLM (LLaMA 8B/13B, Mistral 7B) with prompts instructing the model to extract dog whistles and output them in JSON format.
        - **PREDICT Path**: First filters the posts using a hate speech classifier (BERT family) or an LLM, and then applies keyword extraction algorithms (KeyBERT, RAKE, YAKE, TextRank, TF-IDF) to extract candidate words.

### Loss & Training

EarShot itself does not involve end-to-end training, as it is a pipelined system. The components utilize existing pre-trained models:
- Sentence Encoder: all-MiniLM-L6-v2 (lightweight and fast)
- Filters: ToxiGen BERT, RoBERTa R4, HateXplain BERT
- LLMs: LLaMA 8B/13B, Mistral 7B (selecting small open-source models to ensure reproducibility)
- Word2Vec in the baseline methods is trained using Gensim, with a vocabulary limit of 500K, window size of 5, dimension of 100, and trained for 10 epochs.

## Key Experimental Results

### Main Results

| Method | Scenario | Precision | DPR | F₀.₅ |
|------|------|-----------|-----|------|
| Best Word2Vec | Synthetic | 5.50 | 8.40 | 5.91 |
| Best MLM | Synthetic | 2.00 | 0.42 | 1.14 |
| EarShot-PREDICT | Synthetic | **19.13** | 7.14 | **14.32** |
| EarShot-DIRECT | Synthetic | 20.31 | **56.30** | 23.29 |
| EarShot-PREDICT | Balanced | **14.81** | 1.65 | 5.70 |
| EarShot-DIRECT | Balanced | 2.97 | **13.58** | 3.52 |
| EarShot-PREDICT | Realistic | **10.00** | 1.47 | **4.63** |
| EarShot-DIRECT | Realistic | 0.94 | **60.29** | 1.17 |

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| BERT vs LLM Filtering | BERT is 0.1-0.5 F₀.₅ higher | Small task-specific models outperform general LLMs |
| KeyBERT vs RAKE/YAKE | KeyBERT is superior on small datasets | RAKE/YAKE are better on large datasets |
| Unigram vs Bigram/Trigram W2V | Unigram totally wins | But sacrifices recall for multi-word phrases |
| DIRECT LLaMA 13B vs 8B | 13B is superior in 2/3 scenarios | But fails in Realistic due to over-prediction |

### Key Findings

- All existing methods (Word2Vec, MLM, EPD) yield an F₀.₅ under 6% across all three scenarios, highlighting the extreme challenge of the task.
- EarShot-PREDICT leads significantly in precision (9-20%), while EarShot-DIRECT performs better in recall (13-60%), offering complementary paths.
- Word2Vec actually outperforms PREDICT in terms of DPR in Balanced and Realistic scenarios, because PREDICT is constrained by the keyword extractor, which extracts "important words" rather than specifically "dog whistles."
- Emoji-based dog whistles (e.g., the OK white supremacy gesture), context-dependent cases (e.g., "Federal Reserve"), and emergent ones (e.g., post-2020 "jogger") remain blind spots for all methods.

## Highlights & Insights

- **Innovative Task Definition**: Shifts the focus of "dog whistles" from detection to discovery, offering a task paradigm much closer to real-world content moderation needs.
- **Sound Design of FETCH! Benchmark**: The three case studies cover a range of scenarios from ideal to realistic, and the seed set is stratified by n-gram length to avoid bias.
- **Two-path Design of EarShot**: The high-precision PREDICT path is suitable for reducing human auditing workloads, while the high-recall DIRECT path is ideal for comprehensive scanning, allowing systems to choose based on operational needs.
- **Clever Application of Vector Databases**: By finding semantic nearest neighbors instead of exact matches, the system can discover new dog whistles that have entirely different lexical forms but are semantically related.

## Limitations & Future Work

1. Evaluated only on English datasets; research on dog whistles in multilingual scenarios remains a blank.
2. Lacks specialized manually annotated corpora; reliance on regular expression matching may introduce false positives.
3. LLMs might have encountered dog whistle lexicons during pre-training, introducing data contamination risks.
4. Running large LLMs requires substantial computational resources, which limits the accessibility of the method.
5. The precision of the system in the Realistic scenario remains only around 10%, requiring substantial manual verification for actual deployment.
6. Future research directions could explore hybridizing Word2Vec and LLMs, post-processing noisy predictions, integrating multiple LLMs, and incorporating chain-of-thought in-context learning.

## Related Work & Insights

- Highly related to Euphemism Detection, but with key differences: dog whistles have explicit in-group/out-group binary meanings and typically involve hateful content.
- The Word2Vec/Phrase2Vec baseline is derived from the euphemism detection work of Magu & Luo (2018), and the MLM method is based on Zhu et al. (2021).
- This work reveals an important insight: traditional distributional semantics methods (Word2Vec, BERT) perform poorly in discovering coded language, but the combination of vector databases and LLMs can capture deeper semantic connections.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel task definition, the FETCH! benchmark fills a gap, and EarShot is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three scenarios, four baseline methods, multiple model variants, and detailed threshold analysis.
- Writing Quality: ⭐⭐⭐⭐ The task definition and motivation are clearly expounded, with thorough ethical considerations.
- Value: ⭐⭐⭐⭐ Has significant practical implications for content moderation and social media governance, though precision still needs substantial improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)

</div>

<!-- RELATED:END -->
