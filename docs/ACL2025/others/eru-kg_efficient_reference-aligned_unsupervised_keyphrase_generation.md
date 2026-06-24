---
title: >-
  [Paper Note] ERU-KG: Efficient Reference-aligned Unsupervised Keyphrase Generation
description: >-
  [ACL 2025][Unsupervised keyphrase generation] ERU-KG proposes an unsupervised keyphrase generation framework consisting of an informativeness module and a phraseness module. It learns term-level informativeness estimation through reference texts (queries, citation contexts, titles), outperforming all unsupervised baselines and achieving 89% of the performance of supervised models on keyphrase generation benchmarks, while obtaining the fastest inference speed.
tags:
  - "ACL 2025"
  - "Unsupervised keyphrase generation"
  - "Informativeness estimation"
  - "Reference alignment"
  - "Phraseness modeling"
  - "Text retrieval"
date: 2026-05-08
content_hash: 1d848d37587d1cfa
---

# ERU-KG: Efficient Reference-aligned Unsupervised Keyphrase Generation

**Conference**: ACL 2025  
**arXiv**: [2505.24219](https://arxiv.org/abs/2505.24219)  
**Code**: [https://github.com/louisdo/eru-kg](https://github.com/louisdo/eru-kg)  
**Area**: Others  
**Keywords**: Unsupervised keyphrase generation, Informativeness estimation, Reference alignment, Phraseness modeling, Text retrieval

## TL;DR

ERU-KG proposes an unsupervised keyphrase generation framework consisting of an informativeness module and a phraseness module. It learns term-level informativeness estimation through reference texts (queries, citation contexts, titles), outperforming all unsupervised baselines and achieving 89% of the performance of supervised models on keyphrase generation benchmarks, while obtaining the fastest inference speed.

## Background & Motivation

**Background**: Keyphrase Prediction is a fundamental NLP task used to extract or generate phrases representing the core content from documents. Unsupervised methods are gaining more attention as they do not require annotated data. Mainstream approaches include graph-based methods (such as TextRank) and embedding-based methods.

**Limitations of Prior Work**: Existing unsupervised methods primarily rely on heuristically defined importance scores to estimate phrase informativeness, which can lead to inaccurate informativeness estimation. Additionally, prior work generally neglects inference efficiency—they require explicit modeling and scoring for each candidate phrase, which is inefficient for large-scale applications.

**Key Challenge**: Accurate informativeness estimation requires understanding the semantic role of phrases in different contexts, but unsupervised methods lack this source of signal. Furthermore, candidate-level scoring causes the computational complexity to scale linearly with the number of candidates.

**Goal**: Design an unsupervised keyphrase generation model that can both accurately estimate keyphrase informativeness and maintain high inference efficiency.

**Key Insight**: The authors observe that the core concepts of a document are often reflected through its "reference texts" (such as citation contexts citing the document, search queries, and document titles)—these reference texts naturally capture how humans perceive the core content of the document.

**Core Idea**: Utilize reference texts to learn term-level informativeness, and then efficiently estimate phrase informativeness by aggregating term-level scores without explicitly modeling candidate phrases.

## Method

### Overall Architecture

ERU-KG contains two core modules: an Informativeness Module that estimates the informativeness score of each term, and a Phraseness Module responsible for generating candidate keyphrases. The final keyphrase score is obtained by fusing the outputs of both modules. Given an input document, the informativeness module utilizes pre-trained reference-aligned representations to compute informativeness scores for each word, while the phraseness module generates candidate phrases via sequence labeling. Their scores are multiplied to rank and output the final keyphrase list.

### Key Designs

1. **Reference-aligned Informativeness Module**:

    - **Function**: Learn the informativeness score of each term in the document.
    - **Mechanism**: Use reference texts (queries, citation contexts, titles, etc.) as supervision signals to train the informativeness estimator. Specifically, it aligns the term representations in the document with the representations of the reference texts, so that terms with high semantic similarity to the reference texts receive higher informativeness scores. The key innovation is estimating informativeness at the term-level rather than the phrase-level. Phrase informativeness is then obtained by aggregating the scores of its constituent terms, parameterized as $s_{info}(p) = \text{Agg}(s(t_1), ..., s(t_k))$.
    - **Design Motivation**: Term-level estimation avoids explicit enumeration and scoring of candidate phrases, significantly improving efficiency. Reference texts provide signals of the document's core concepts from a human perspective.

2. **Phraseness Module**:

    - **Function**: Generate grammatically correct candidate keyphrases.
    - **Mechanism**: Adopt a sequence labeling model to tag tokens in the document with B (Begin), I (Inside), or O (Outside) keyphrase labels. It is trained via weak supervision on keyphrase segments appearing in reference texts, eliminating the need for manual training data. This module can generate phrases that do not appear in the document (i.e., keyphrase generation, as opposed to extraction).
    - **Design Motivation**: Decoupling phrase generation from informativeness estimation allows both tasks to be optimized independently. The sequence labeling approach ensures the grammatical correctness of the generated phrases.

3. **Generation/Extraction Mode Switching**:

    - **Function**: Flexibly switch between keyphrase generation and extraction modes by adjusting hyperparameters.
    - **Mechanism**: When the weight of the phraseness module is set to 0, the model degenerates into a pure extraction mode, ranking n-grams in the document solely based on informativeness scores. When both modules are active, it enters generation mode, allowing the production of phrase combinations that do not exist verbatim in the document.
    - **Design Motivation**: Different application scenarios have distinct requirements for keyphrases; the extraction mode is more conservative and reliable, while the generation mode offers broader coverage. Flexible switching enhances the model's utility.

### Loss & Training

The informativeness module uses a contrastive learning loss to pull document term representations closer to their corresponding reference text representations and push them away from irrelevant reference text representations. The phraseness module uses standard cross-entropy loss for sequence labeling, using phrase boundaries identified in the reference texts as weak labels. The two modules can be trained independently or jointly.

## Key Experimental Results

### Main Results

Performance on multiple keyphrase generation benchmark datasets (F1@10):

| Dataset | Metric | ERU-KG | Best Unsupervised Baseline | Supervised Model |
|--------|------|--------|----------------|------------|
| Inspec | F1@10 | Best | Below ERU-KG | Reference Upper Bound |
| SemEval | F1@10 | Best | Below ERU-KG | Reference Upper Bound |
| NUS | F1@10 | Best | Below ERU-KG | Reference Upper Bound |
| Krapivin | F1@10 | Best | Below ERU-KG | Reference Upper Bound |
| Average | F1@10 | Reaches 89% of supervised | - | 100% |

ERU-KG outperforms all unsupervised baselines across all benchmarks, achieving an average of 89% of the F1@10 performance of supervised models.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| ERU-KG (Full) | Best F1 | Dual-module: Informativeness + Phraseness |
| Informativeness Module Only | F1 drops | Degenerates to extraction mode, lacks generation capability |
| Phraseness Module Only | F1 drops significantly | Lacks informativeness guidance, poor candidate quality |
| Alternative Reference Types | F1 varies | Different reference texts (queries/citations/titles) have distinct advantages |

### Key Findings

- The informativeness module is the core contribution, and performance drops significantly when it is removed; the phraseness module provides additional gains in generation capability.
- Different types of reference texts (queries vs. citation contexts vs. titles) perform differently across datasets, indicating that the choice of reference texts needs to be adjusted based on the application scenario.
- In text retrieval tasks, the keyphrases generated by ERU-KG are effective for both query and document expansion, demonstrating the semantic quality of the generated phrases.
- Inference speed tests show that ERU-KG is the fastest method among models of comparable scale, as term-level aggregation avoids scoring candidate phrases one by one.

## Highlights & Insights

- **Term-level informativeness aggregation** is an elegant design—converting an O(n) candidate scoring problem into an O(1) term aggregation operation while maintaining accuracy. This idea can be transferred to any task requiring the scoring of compositional units.
- The idea of using **reference texts as proxies for informativeness** is highly inspiring—leveraging how documents are cited/queried in different contexts to infer their core concepts is fundamentally a usage-based semantic understanding approach.
- The model is open-sourced on HuggingFace in two versions (67M base and 35.1M small), making it highly practical.

## Limitations & Future Work

- It relies on the availability of reference texts (queries, citation contexts), which might be missing for newly published or cold-start documents.
- The paper only evaluates on English datasets, leaving the multilingual generalization capability unknown.
- The phraseness module is based on sequence labeling, which has limited capability in generating non-contiguous phrases or keyphrases that require paraphrasing.
- Future work can explore extending the reference alignment idea to other information extraction tasks (e.g., summarization, entity linking).

## Related Work & Insights

- **vs. TextRank/EmbedRank**: These classical unsupervised methods rely on graph structures or embedding similarities to estimate importance, whereas ERU-KG learns informativeness more accurately through reference texts.
- **vs. Supervised KP Models**: ERU-KG achieves 89% of supervised performance without requiring labeled data, rendering it more practical in domains with high annotation costs.
- The idea of reference alignment can inspire query-document matching modeling in document understanding and information retrieval.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of reference alignment and term-level aggregation is novel, although individual components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation + downstream retrieval tasks + speed tests, which is quite comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The abstract is clear, and the logic behind the motivation and methodology flows smoothly.
- Value: ⭐⭐⭐⭐ Highly practical, featuring open-source models with real-world usable inference speeds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Analysis of Datasets, Metrics and Models in Keyphrase Generation](an_analysis_of_datasets_metrics_and_models_in_keyphrase_generation.md)
- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)
- [\[ICCV 2025\] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration](../../ICCV2025/others/edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)

</div>

<!-- RELATED:END -->
