---
title: >-
  [Paper Note] In the LLM Era, Word Sense Induction Remains Unsolved
description: >-
  [ACL 2025 (Findings)][NLP Understanding][Word Sense Induction] This paper systematically evaluates the Word Sense Induction (WSI) task in the LLM era. On a more rigorously controlled SemCor-derived evaluation set, it is found that all unsupervised methods, including LLM-based approaches, fail to outperform the simple "one sense per word" baseline. Meanwhile, a semi-supervised method combining Wiktionary outperforms the previous SOTA by 3.3%, indicating that WSI remains far fr…
tags:
  - "ACL 2025 (Findings)"
  - "NLP Understanding"
  - "Word Sense Induction"
  - "Word Sense Disambiguation"
  - "LLM"
  - "Clustering"
  - "Wiktionary"
date: 2026-05-08
content_hash: 42cd469be9e36247
---

# In the LLM Era, Word Sense Induction Remains Unsolved

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2603.11686](https://arxiv.org/abs/2603.11686)  
**Code**: None  
**Area**: NLP Understanding / Lexical Semantics  
**Keywords**: Word Sense Induction, Word Sense Disambiguation, LLM, Clustering, Wiktionary

## TL;DR

This paper systematically evaluates the Word Sense Induction (WSI) task in the LLM era. On a more rigorously controlled SemCor-derived evaluation set, it is found that all unsupervised methods, including LLM-based approaches, fail to outperform the simple "one sense per word" baseline. Meanwhile, a semi-supervised method combining Wiktionary outperforms the previous SOTA by 3.3%, indicating that WSI remains far from being solved.

## Background & Motivation

**Background**: Word Sense Disambiguation (WSD) relies on predefined sense-annotated data, which is expensive to label and suffers from inconsistent sense division standards. Word Sense Induction (WSI) is an alternative solution that does not rely on annotated data—it attempts to automatically discover the different senses of a word from a corpus by clustering the representations of the word in different contexts. WSI is particularly valuable in low-resource languages and domain-specific scenarios.

**Limitations of Prior Work**: Current WSI evaluation faces serious methodological issues. First, many evaluation datasets oversample polysemous words, failing to respect the natural frequency distribution of polysemous words in real corpora—this allows some methods to achieve artificially high scores by exploiting sampling bias. Second, the advent of LLMs has led many to believe that WSI might have been "solved," but there is a lack of systematic evaluation under fair conditions to verify this hypothesis.

**Key Challenge**: On one hand, LLMs have achieved a deeper understanding of lexical semantics; on the other hand, the core difficulty of WSI—automatically determining the number and boundaries of senses without a sense dictionary—is not actually covered by the capabilities of LLMs. LLMs excel at semantic understanding but struggle with unsupervised semantic clustering.

**Goal**: (1) To design a more rigorous evaluation framework for WSI; (2) to comprehensively test pretrained embeddings + clustering algorithms and LLM methods; (3) to explore the effectiveness of data augmentation and semi-supervised strategies.

**Key Insight**: Building an evaluation set from SemCor (the largest English sense-annotated corpus) while strictly retaining the polysemy frequency and sense distribution of the original corpus to avoid artificial bias.

**Core Idea**: Systematically testing WSI methods under a fairer evaluation framework to reveal the important fact that WSI remains unsolved in the LLM era, and exploring the path forward through a semi-supervised method leveraging Wiktionary.

## Method

### Overall Architecture

The work in this paper is divided into three levels: (1) constructing an evaluation framework based on SemCor; (2) testing unsupervised baselines (pretrained embeddings + clustering, and LLM methods); (3) exploring semi-supervised augmentation strategies. The evaluation covers different parts of speech (nouns, verbs, adjectives) and uses clustering quality metrics such as V-Measure and ARI.

### Key Designs

1. **Rigorous Evaluation Framework based on SemCor**:

    - - **Function**: Providng a WSI evaluation benchmark that respects the true distribution.
    - - **Mechanism**: WordNet sense annotations are extracted for the target words at all their occurrences in the SemCor corpus, preserving the original sense frequency distribution. Unlike prior WSI evaluation sets (e.g., datasets in SemEval tasks), this paper does not resample or balance polysemous words. It is split into development and test sets to ensure evaluation reliability.
    - - **Design Motivation**: Prior evaluation sets overestimated the effectiveness of methods by artificially balancing the sense distribution; under the real distribution, the performance of many methods drops significantly. The "one sense per word" (1cpl) baseline is a very strong baseline under the real distribution, as most words are indeed dominated by a single sense in actual usage.

2. **LLM-based WSI Methods**:

    - - **Function**: Utilizing the semantic understanding capabilities of LLMs for word sense induction.
    - - **Mechanism**: Given a set of sentences containing the target word, the LLM is asked to group these sentences according to the different meanings of the target word. Two approaches are tested: (a) directly letting the LLM group them (zero-shot/few-shot); (b) using the LLM to generate sense descriptions for each occurrence and then clustering these descriptions. Several LLMs, such as GPT-4 and Llama, are used in the experiments.
    - - **Design Motivation**: LLMs possess powerful semantic understanding capabilities and intuitively should be able to distinguish meaning differences of the same word in different contexts, but the actual effectiveness requires rigorous validation.

3. **Wiktionary-Enhanced Semi-Supervised Method**:

    - - **Function**: Utilizing the Wiktionary dictionary as a weak supervision signal to improve WSI.
    - - **Mechanism**: Leveraging Wiktionary from three aspects: (a) **sense count prior**: using the number of senses of the target word in Wiktionary to set the $k$ value for clustering; (b) **definition enhancement**: using Wiktionary definitions and example sentences as must-link constraints or pseudo-labels; (c) **data augmentation**: using LLMs to generate additional example sentences with sense labels, or retrieving relevant sentences from a corpus using Wiktionary definitions to increase the amount of training data.
    - - **Design Motivation**: Fully unsupervised WSI is limited by insufficient signals. Introducing a dictionary as weak supervision can provide valuable prior knowledge without requiring expensive annotations.

### Loss & Training

For clustering methods, classic algorithms such as K-Means, Agglomerative Clustering, and DBSCAN are used. For semi-supervised methods, constrained K-Means (a variant with must-link and cannot-link constraints) is used. In terms of embeddings, pretrained language models such as BERT, RoBERTa, and DeBERTa are used to extract context-dependent word embeddings.

## Key Experimental Results

### Main Results

V-Measure scores on the SemCor-derived test set:

| Method | Noun VM | Verb VM | Adj VM | Overall VM |
|------|---------|---------|----------|--------|
| 1cpl (One Sense Per Word) | 62.1 | 54.3 | 68.7 | 61.2 |
| BERT + K-Means | 48.7 | 39.2 | 52.1 | 46.3 |
| DeBERTa + Agglo | 53.4 | 42.8 | 57.3 | 50.8 |
| GPT-4 Direct Grouping | 51.2 | 44.1 | 55.8 | 49.9 |
| LLM-Generated Description + Clustering | 55.8 | 46.3 | 60.2 | 53.7 |
| Wiktionary Semi-Supervised (Ours Best) | 64.8 | 57.2 | 71.0 | 64.5 |
| Prev. SOTA (Amrami et al.) | 60.3 | 55.1 | 66.4 | 61.2 |

### Ablation Study

| Configuration | Overall VM | Description |
|------|--------|------|
| Full Semi-Supervised | 64.5 | Full method |
| w/o Wiktionary $k$ value | 59.8 | Sense count prior is crucial |
| w/o Must-link constraint | 61.7 | Constraints help clustering stability |
| w/o Data augmentation | 62.1 | Data augmentation is helpful but not core |
| LLM augmentation only | 58.4 | LLM augmentation is inferior to dictionary |
| Corpus augmentation only | 60.9 | Corpus augmentation outperforms LLM |

### Key Findings

- **Most critical finding: No unsupervised method outperforms the 1cpl baseline**. This is a striking result under fair evaluation—in real distributions, most words are dominated by a single sense, and clustering methods introduce noise by forcing artificial distinctions.
- **LLMs perform poorly on WSI**: Even direct grouping by GPT-4 is inferior to the 1cpl baseline, suggesting that while LLMs understand word meanings, they struggle with unsupervised sense discovery.
- **Significant differences across parts of speech**: Verbs are the most challenging because their sense boundaries are the most ambiguous and they have the highest degree of polysemy; adjectives are the easiest.
- The Wiktionary semi-supervised method outperforms the Prev. SOTA by 3.3% ($61.2 \rightarrow 64.5$), primarily benefiting from the sense count prior—knowing how many senses a word has is crucial for clustering.
- **Corpus sources outperform LLM generation in data augmentation**, indicating that usage in real-world contexts is more valuable than example sentences generated by LLMs.

## Highlights & Insights

- **Outstanding contribution to evaluation methodology**: Pointing out the long-standing evaluation bias in the WSI community—evaluation sets that do not respect the real-world distribution make methods appear more effective than they actually are. This issue may also exist in other NLP tasks.
- **The finding that "the 1cpl baseline is hard to beat" has a massive impact**: It demonstrates that word sense distributions in real-world text are extremely skewed, with most occurrences belonging to the dominant sense, which fundamentally questions the research direction of the WSI task.
- The paper proposes an important direction forward: WSI needs to better integrate dictionary knowledge and the lexical semantic capabilities of LLMs, rather than relying solely on clustering.

## Limitations & Future Work

- Evaluated only on English; the challenges of WSI in low-resource languages (which lack dictionaries like Wiktionary) may be entirely different.
- Evaluation metrics (V-Measure, ARI) themselves are sensitive to sense granularity—different annotators may partition sense boundaries differently, which affects the reliability of the evaluation.
- The semi-supervised method depends on the coverage of Wiktionary—it cannot be used for words not recorded in the dictionary, such as neologisms and internet slang.
- Exploring the use of LLMs' in-context learning capabilities could allow LLMs to perform WSI with the support of more exemplars, rather than relying on a fixed-size input window.

## Related Work & Insights

- **vs Amrami et al. (2019)**: The previous SOTA used substitute word distributions for clustering; this work reveals that under fair evaluation, it only performs on par with the 1cpl baseline.
- **vs WSD Methods**: WSD relies on predefined senses, while WSI attempts fully automated discovery. The results of this paper show that complete automation is currently unviable, making semi-supervised approaches a reasonable compromise.
- **vs LLM-based Lexical Semantics**: LLMs can perform sense disambiguation (performing well on WSD tasks) but cannot perform sense discovery (WSI), revealing the boundaries of LLMs' lexical semantic capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ The contribution to evaluation methodology is original, and the negative results themselves are highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive evaluation covering multiple methods, parts of speech, and ablation dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear arguments, rigorous experimental design, and solid analysis.
- Value: ⭐⭐⭐⭐ Provides an important benchmark and caution to the WSI community, driving methodological reflection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](../../ACL2026/nlp_understanding/table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] Accurate and Efficient Statistical Testing for Word Semantic Breadth](../../ACL2026/nlp_understanding/accurate_and_efficient_statistical_testing_for_word_semantic_breadth.md)
- [\[ACL 2025\] SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)
- [\[ACL 2025\] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)
- [\[ACL 2026\] Refining and Reusing Annotation Guidelines for LLM Annotation](../../ACL2026/nlp_understanding/refining_and_reusing_annotation_guidelines_for_llm_annotation.md)

</div>

<!-- RELATED:END -->
