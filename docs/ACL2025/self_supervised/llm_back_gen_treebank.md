---
title: >-
  [Paper Note] Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing
description: >-
  [ACL 2025][Self-Supervised Learning][Constituency Parsing] This paper proposes an LLM Back Generation method that takes incomplete cross-domain constituency trees as input, prompting the LLM to complete the missing words to generate a treebank. It also designs a span-level contrastive learning pre-training strategy to achieve state-of-the-art performance in cross-domain constituency parsing.
tags:
  - "ACL 2025"
  - "Self-Supervised Learning"
  - "Constituency Parsing"
  - "Cross-domain"
  - "LLM Back Generation"
  - "Contrastive Learning"
  - "Treebank Generation"
date: 2026-05-08
content_hash: 8d327857f93c1902
---

# Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing

**Conference**: ACL 2025  
**arXiv**: [2505.20976](https://arxiv.org/abs/2505.20976)  
**Code**: [GitHub](https://github.com/guopeiming/Back_Parsing_LLM)  
**Area**: Constituency Parsing / Cross-domain Adaptation  
**Keywords**: Constituency Parsing, Cross-domain, LLM Back Generation, Contrastive Learning, Treebank Generation  

## TL;DR

This paper proposes an LLM Back Generation method that takes incomplete cross-domain constituency trees as input, prompting the LLM to complete the missing words to generate a treebank. It also designs a span-level contrastive learning pre-training strategy to achieve state-of-the-art performance in cross-domain constituency parsing.

## Background & Motivation

**Research Problem:** Cross-domain constituency parsing remains an unresolved challenge due to the lack of multi-domain annotated treebanks. Parsers trained on existing news-domain treebanks (such as PTB) suffer from significant performance degradation when applied to other domains.

**Limitations of Prior Work:** (1) Direct constituency parsing using LLMs yields poor performance (ChatGPT achieves only 22.99% $F_1$), as its autoregressive generation struggles to guarantee valid tree structures; (2) Li et al. (2023) utilized LLMs to generate raw text and then parsed it with a parser to label a pseudo-treebank. This two-stage pipeline leverages the LLM indirectly, which introduces noise and error propagation.

**Core Motivation:** Although LLMs perform poorly in forward constituency parsing (sentence $\to$ tree), reversing this process—given a tree skeleton and domain-specific keywords, prompting the LLM to fill in the missing words—allows the model to leverage the language generation capabilities of LLMs while guaranteeing the validity of the syntactic structures.

## Method

### Overall Architecture

The overall framework consists of two phases: (1) **LLM Back Generation**: extraction of syntactic tree skeletons and domain keywords from target domain sentences, followed by masking non-keywords to prompt the LLM to complete and generate a cross-domain treebank; (2) **Contrastive Learning Pre-training**: span-level contrastive learning on the generated treebanks to train a constituent span representation model, which is subsequently fine-tuned for cross-domain constituency parsing.

### Key Designs

1. **Cross-domain Syntactic Tree Preparation:** First, a base chart-based parser is used to parse unlabeled sentences from the target domain to obtain the syntactic structures. Then, KeyBERT is employed to extract the top 25% of words most similar to the original sentence as domain keywords to be retained, while the remaining words are masked. A masked tree carries two key cross-domain elements: the domain syntactic structure and domain-specific vocabulary.

2. **LLM Back Generation:** The masked syntactic tree, along with a few in-context learning (ICL) demonstrations, is fed into the LLM (GPT-4). The LLM is prompted to fill in the missing words while keeping the tree structure intact, thereby generating $(\hat{X}, Y)$ pairs. This guarantees the validity of the syntactic tree from the input side, avoiding the issue of LLMs generating invalid bracket structures during direct parsing.

3. **Span-level Contrastive Learning Pre-training:** For each constituent span $(i, j)$, its left child, right child, parent node, and sibling node are selected as 4 positive samples, while computationally adjacent but invalid spans are selected as 15 negative samples. The contrastive objective aims to pull the representations of valid constituent spans closer while pushing apart invalid spans. This significantly scales up the pre-training data volume (averaging about 25 spans per tree, 10K trees $\to$ 250K pre-training samples).

### Loss & Training

The contrastive learning loss is defined as:

$$\mathcal{L} = -\sum_{m \in (i,j)^+} \log \frac{e^{f(\boldsymbol{r}, \boldsymbol{r}_m^+)}}{\sum_{n \in (i,j)^-} e^{f(\boldsymbol{r}, \boldsymbol{r}_m^+)} + e^{f(\boldsymbol{r}, \boldsymbol{r}_n^-)}}$$

where $f$ denotes the cosine similarity scaled by a temperature factor $\tau$. During the fine-tuning phase, the standard tree-based max-margin loss is utilized.

## Experiments

### Main Results

F1 scores on five target domains of MCTB:

| Method | Dia | For | Law | Lit | Rev | Avg |
|------|-----|-----|-----|-----|-----|-----|
| ChatGPT (valid) | 70.38 | 70.36 | 80.70 | 74.74 | 69.08 | 73.05 |
| GPT-4 (valid) | 77.64 | 76.27 | 84.49 | 79.58 | 75.63 | 78.72 |
| Kitaev & Klein (2018) | 86.10 | 86.92 | 92.07 | 86.28 | 84.32 | 87.14 |
| Li et al. (2023) | 87.59 | 87.55 | 93.29 | 87.54 | 85.58 | 88.31 |
| Natural Corpus + CTPT | 87.33 | 87.80 | 92.54 | 86.91 | 84.35 | 87.79 |
| **LLM Back Gen + CTPT** | **87.92** | **88.13** | **93.22** | **87.50** | **85.86** | **88.52** |

### Ablation Study

| Mask Rate | DAPT | NOPT | CTPT |
|--------|------|------|------|
| 0% (Natural Corpus) | 87.15 | 87.38 | 87.79 |
| 25% (Best) | 87.39 | 87.81 | **88.52** |
| 50% | — | — | ~88.2 |
| 100% | — | — | ~87.8 |

| Pre-training Strategy | Avg F1 |
|-----------|--------|
| DAPT (Domain Adaptive) | 87.15 |
| NOPT (No Pre-training) | 87.38 |
| **CTPT (Contrastive)** | **87.79** |

### Key Findings

- Treebanks generated by LLM Back Generation consistently outperform natural corpus treebanks under all pre-training strategies, indicating that back generation effectively introduces domain-specific syntactic diversity.
- A 25% mask rate yields the best results—retaining sufficient domain keywords to guide the LLM while providing enough freedom to introduce variations. An excessively high mask rate leads to generation that diverges from the target domain.
- Contrastive learning pre-training (CTPT) converges significantly faster than DAPT and NOPT (600 steps vs. 1000 steps) and achieves the best final performance.
- Performance saturates with only 8K trees, demonstrating that span-level contrastive learning effectively amplifies the value of limited data.

## Highlights & Insights

- Cleverly reverses the direction of the parsing task: leveraging the strength of LLMs in language generation rather than their weakness in structure prediction.
- Introduces contrastive learning to constituency parsing for the first time, with a span-level design that significantly amplifies the volume of training data.
- The construction of positive and negative pairs is closely coupled with the hierarchical structures of constituency trees, demonstrating strong linguistic motivation.

## Limitations & Future Work

- The method is only validated on the English MCTB dataset, lacking multi-lingual evaluation due to the absence of cross-domain constituency treebanks in other languages.
- It relies heavily on GPT-4 for back generation; most other LLMs (such as ChatGPT, LLaMA-3) either fail to generate valid syntactic trees or suffer from high error rates.
- Performance in formal domains with long sentences, such as law and literature, is slightly inferior to Li et al. (2023), likely because a single parser handling all domains needs to balance different distributions.

## Related Work & Insights

- **Cross-domain Constituency Parsing:** Yang et al. (2022) annotated a multi-domain treebank, MCTB; Li et al. (2023) utilized LLMs to generate target domain raw texts and then pseudo-labeled them.
- **LLMs for Syntactic Parsing:** Bai et al. (2023) comprehensively evaluated the parsing capabilities of ChatGPT, GPT-4, and LLaMA, proving that direct annotation yields poor results.
- **Contrastive Learning:** SimCSE (Gao et al., 2021) introduced contrastive learning to sentence representations; this work is the first to port it to span-level syntactic parsing.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Practicality | 7 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Overall Rating | 7.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](../../CVPR2025/self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ACL 2025\] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning](whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)
- [\[CVPR 2026\] Learning from Semantic Dictionaries: Discriminative Codebook Contrastive Learning for Unified Visual Representation and Generation](../../CVPR2026/self_supervised/learning_from_semantic_dictionaries_discriminative_codebook_contrastive_learning.md)
- [\[AAAI 2026\] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](../../AAAI2026/self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)

</div>

<!-- RELATED:END -->
