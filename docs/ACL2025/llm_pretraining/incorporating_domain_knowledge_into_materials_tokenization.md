---
title: >-
  [Paper Note] Incorporating Domain Knowledge into Materials Tokenization
description: >-
  [ACL 2025][LLM Pretraining][Materials Science] This paper proposes MATTER, a domain-aware tokenization framework designed for materials science. By training a materials concept detector, MatDetector, and injecting its detection results into the token merge ranking, it prevents the fragmentation of domain terminology. It achieves average performance gains of 4% and 2% on generation and classification tasks, respectively.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Materials Science"
  - "Tokenization"
  - "Domain Knowledge"
  - "BPE/WordPiece"
  - "Concept Detection"
  - "MatDetector"
date: 2026-05-08
content_hash: a5f64ccf12ac9ade
---

# Incorporating Domain Knowledge into Materials Tokenization

**Conference**: ACL 2025  
**arXiv**: [2506.11115](https://arxiv.org/abs/2506.11115)  
**Code**: [https://github.com/yerimoh/MATTER](https://github.com/yerimoh/MATTER)  
**Authors**: Yerim Oh, Jun-Hyung Park, Junho Kim, SungHo Kim, SangKeun Lee  
**Institutions**: Korea University, Hankuk University of Foreign Studies  
**Area**: LLM Pre-training  
**Keywords**: Materials Science, Tokenization, Domain Knowledge, BPE/WordPiece, Concept Detection, MatDetector

## TL;DR

This paper proposes MATTER, a domain-aware tokenization framework designed for materials science. By training a materials concept detector, MatDetector, and injecting its detection results into the token merge ranking, it prevents the fragmentation of domain terminology. It achieves average performance gains of 4% and 2% on generation and classification tasks, respectively.

## Background & Motivation

### Background

**Background**: Language models are increasingly applied in materials science (e.g., MatSciBERT, BatteryBERT), yet these models typically inherit frequency-driven tokenization methods (such as BPE and WordPiece) from general NLP.

### Limitations of Prior Work

**Limitations of Prior Work**: Materials-related terms (chemical formulas, material names) appear with extremely low frequency in the corpora, whereas frequency-driven tokenization prioritizes retaining high-frequency words.

### Key Challenge

**Key Challenge**: Low-frequency materials concepts are split into semantically unrelated subwords. For instance, "germanium" is segmented into "german" + "ium", completely losing its chemical meaning.

### Mechanism

**Mechanism**: Previous methods for improving subword tokenization (such as SAGE and PickyBPE) target general domains and are not tailored to materials science scenarios.

**Core Motivation**: Domain knowledge must be integrated directly into the tokenizer training phase. This ensures that materials concepts maintain their structural and semantic integrity during tokenization, rather than relying solely on raw word frequency statistics.

## Method

### Overall Architecture

MATTER introduces three key modifications on top of the WordPiece tokenization algorithm:

1. **Word Frequency Calculation**: Retains the original word frequency as a baseline.
2. **Materials Knowledge Injection**: Utilizes MatDetector to identify materials concepts and assign probability weights.
3. **Re-ordered Merging**: Re-orders the token merging sequence based on the adjusted frequencies, prioritizing materials-related token pairs.

### Key Design 1: MatDetector (Materials Concept Detector)

MatDetector is an NER tool used to detect materials concepts in text, based on the architecture of Trewartha et al. (2022). Its training data construction pipeline consists of:

1. **Materials Concept Extraction**: Searches materials-related concepts in the PubChem database, extracting 80K materials concepts (chemical names, IUPAC names, synonyms, formulas).
2. **Materials Corpus Crawling**: Uses these concepts to search Semantic Scholar and collects approximately 42K scientific papers.
3. **Data Annotation**: Automatically annotates the corpus using the PubChem materials concepts to generate an NER dataset with labels: "material name", "material formula", and "other".
4. **Data Augmentation**: Standardizes noisy data (e.g., formatting inconsistencies, OCR errors) to expand the dataset by a factor of 4.

For a word $w$, MatDetector outputs the probability of it being a materials concept, denoted as $\hat{y}_{mat}(w)$:

$$\hat{y}(w) = \arg\max_{c \in C} \frac{1}{n} \sum_{i=1}^{n} P(t_i, c)$$

If the predicted label belongs to the "material" category, it is assigned to $\hat{y}_{mat}(w)$; otherwise, it is null.

### Key Design 2: Frequency Adjustment and Re-ordering

For words recognized as materials concepts by MatDetector, their frequencies are adjusted using a log-odds weighting:

$$\text{freq}_{mat}(w) = \text{freq}_{origin}(w) + \lambda \cdot \frac{\hat{y}_{mat}(w)}{1 - \hat{y}_{mat}(w)}$$

- $\lambda$ is the materials importance factor, which controls the strength of domain knowledge injection.
- Materials concepts with higher probabilities receive higher frequency bonuses, ensuring they are prioritized for retention during the merging process.
- Frequencies of non-materials concepts remain unchanged.

### Key Design 3: Materials-Knowledge-Based Merge Ranking

The adjusted frequency $\text{freq}_{mat}$ is used to calculate the token-pair merge score, $\text{MatScore}(t_L, t_R)$, replacing the original raw frequency score. In the iterative merging process:

1. The token pair with the highest MatScore is selected for merging.
2. A new token is created and added to the vocabulary.
3. All occurrences of this token pair in the corpus are replaced.
4. Scores are recalculated for the updated token set.

## Key Experimental Results

### Evaluation Tasks

**Generation Tasks (MatSci-NLP)**: 7 sub-tasks
- NER, Relation Classification (RC), Event Argument Extraction (EAE), Paragraph Classification (PC), Synthesis Action Retrieval (SAR), Sentence Classification (SC), Slot Filling (SF)

**Classification Tasks**: 5 sub-tasks
- NER-SOFC, NER-Matscholar, SF, RC, PC

### Main Results (Generation Tasks Macro-F1)

### Main Results

| Tokenization Method | NER | RC | EAE | PC | SAR | SC | SF | Average |
|---------|-----|-----|-----|-----|-----|-----|-----|------|
| BPE | 47.1 | 47.2 | 36.3 | 40.2 | 41.8 | 47.6 | 16.7 | 42.0 |
| WordPiece | 56.1 | 58.5 | 29.4 | 58.9 | 74.6 | 60.3 | 32.6 | 52.9 |
| SAGE | 57.0 | 61.6 | 28.3 | 59.6 | 67.4 | 61.6 | 35.0 | 52.9 |
| PickyBPE | 41.7 | **65.1** | 36.5 | 40.2 | 66.1 | 47.6 | 23.1 | 45.8 |
| **MATTER** | **59.3** | 59.1 | **36.9** | **67.6** | **79.3** | **64.9** | **38.0** | **57.9** |

MATTER achieves the best Macro-F1 on 5 out of 7 tasks, bringing an average improvement of **5 percentage points**.

### Classification Task Results

MATTER also demonstrates consistent improvements in classification tasks, improving the average Micro-F1 by approximately 2% compared to baseline methods. Even though PickyBPE performs strongly on classification task Micro-F1 (since it filters out intermediate junk tokens), MATTER maintains its advantage in Macro-F1.

### Key Findings

1. **Extremely Low Frequency of Materials Concepts**: Materials concepts in 150K materials science papers have much lower frequencies than general vocabulary, validating the hypothesis that frequency-driven tokenizers fragment domain terminology.
2. **Fragmentation Degrades Performance**: Taking "germanium" $\rightarrow$ "german" + "ium" as an example, fragmentation leads the model to misinterpret the chemical semantics.
3. **MATTER Successfully Reduces Fragmentation**: It retains more complete materials concept tokens in the vocabulary.

## Highlights & Insights

1. **Precise Problem Definition**: The study pinpoints the core bottleneck in materials science NLP tokenization—the fragmentation of low-frequency domain terms.
2. **Simple and Effective Methodology**: Significant performance gains are attained simply by modifying the frequency calculation and merge order, without altering the underlying model architecture.
3. **Solid Data Engineering**: Collecting 80K materials concepts from PubChem $\rightarrow$ crawling 42K papers on Semantic Scholar $\rightarrow$ automatic annotation + 4x augmentation delivers highly valuable reusable resources.
4. **Generalizable Framework**: The methodology of MATTER can be extended to other specialized domains (e.g., medicine, law) simply by replacing MatDetector with a corresponding domain concept detector.

## Limitations & Future Work

1. The method is validated only on BERT-like models; its effectiveness on autoregressive models like GPT/LLaMA remains to be explored.
2. MatDetector relies on the PubChem database, limiting its coverage to the database's contents.
3. The selection of the $\lambda$ parameter requires tuning, and the paper does not offer clear theoretical guidelines for parameter choice.
4. There is no detailed analysis showing the specific composition changes of materials-related tokens in the vocabulary (e.g., which complete concepts were preserved).
5. Downstream evaluations are confined to English materials science texts.

## Related Work & Insights

- **Subword Tokenization**: BPE (Sennrich et al., 2016), WordPiece (Wu et al., 2016), SAGE (Yehezkel & Pinter, 2023), PickyBPE (Chizhov et al., 2024)
- **Materials Science Language Models**: MatSciBERT (Gupta et al., 2022), BatteryBERT (Huang & Cole, 2022)
- **Materials Concept Detection**: ChemDataExtractor (Kumar et al., 2024) — trained on biomedical data, which shows limited accuracy in the materials domain.

## Rating

⭐⭐⭐⭐ (4/5)

The motivation is highly clear, the method is straightforward yet effective, and the evaluation is comprehensive (spanning 7+5 downstream tasks). It makes a valuable contribution to the overlooked area of domain-specific tokenization. Limitations lie in its restriction to BERT-type models and the lack of theoretical analysis regarding the $\lambda$ parameter.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](dual_stage_curriculum_learning_sequence_labeling.md)
- [\[ACL 2025\] Adversarial Tokenization](adversarial_tokenization.md)
- [\[ACL 2025\] Tokenization is Sensitive to Language Variation](tokenization_is_sensitive_to_language_variation.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ACL 2025\] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)

</div>

<!-- RELATED:END -->
