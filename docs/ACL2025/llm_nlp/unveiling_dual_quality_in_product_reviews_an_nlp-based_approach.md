---
title: >-
  [Paper Note] Unveiling Dual Quality in Product Reviews: An NLP-Based Approach
description: >-
  [ACL 2025][LLM (Other)][dual quality] This paper proposes an automated detection task for "dual quality" (DQ) in product reviews, constructing the first Polish DQ dataset (1,957 reviews) through an iterative active learning strategy. It systematically compares three classes of approaches—SetFit, Transformer encoders, and LLMs—finding that language-specific encoders perform comparably to LLMs with instructions (DQ F1 ≈ 80-83%), and validates cross-lingual transfer capabilities…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "dual quality"
  - "product review classification"
  - "SetFit"
  - "transformer"
  - "LLM"
date: 2026-05-08
content_hash: 0fd0d78bd72297cf
---

# Unveiling Dual Quality in Product Reviews: An NLP-Based Approach

**Conference**: ACL 2025  
**arXiv**: [2505.19254](https://arxiv.org/abs/2505.19254)  
**Code**: Not publicly available  
**Area**: NLP Applications/Consumer Protection  
**Keywords**: dual quality, product review classification, SetFit, transformer, LLM  

## TL;DR

This paper proposes an automated detection task for "dual quality" (DQ) in product reviews, constructing the first Polish DQ dataset (1,957 reviews) through an iterative active learning strategy. It systematically compares three classes of approaches—SetFit, Transformer encoders, and LLMs—finding that language-specific encoders perform comparably to LLMs with instructions (DQ F1 ≈ 80-83%), and validates cross-lingual transfer capabilities.

## Background & Motivation

- **Dual Quality Issue**: Dual quality refers to companies selling products with significantly different ingredients or quality parameters in different markets under the same brand and highly similar packaging. Amendments to the EU Unfair Commercial Practices Directive have recognized this behavior as a misleading commercial practice, requiring member states to enforce law at the national level.
- **Real-World Application Scenario**: The Polish Office of Competition and Consumer Protection (UOKiK) needs automated tools to filter complaints related to dual quality from massive consumer reviews on e-commerce platforms (such as CENEO, WIZAZ) and social media to assist human analysts in conducting investigations.
- **Research Gap**: Existing NLP research in e-commerce covers NLP applications like sentiment analysis of reviews, product Q&A, fake news detection, and review moderation, but none target the dual quality issue. No publicly available DQ review dataset or detection model currently exists.
- **Key Challenge**: Dual quality reviews are extremely rare among all reviews (accounting for approx. 27.6% in the target stream); random sampling for annotation is highly inefficient. Furthermore, it is necessary to distinguish "dual quality" (cross-market quality differences) from "other problems" (counterfeits, quality degradation, misdelivery, etc.), both of which exhibit substantial semantic overlap.

## Method

### Overall Architecture

The system is divided into two core phases: the **Dataset Construction Phase**, which progressively expands labeled data starting from seed reviews using an iterative active learning strategy; and the **Model Evaluation Phase**, which systematically compares three classes of methods—SetFit + Sentence Transformers, full fine-tuning of Transformer encoders, and LLM in-context learning—on a three-class classification task (dual quality / other problems / standard), followed by robustness validation and cross-lingual transfer evaluation.

### Key Designs

1. **Iterative Active Learning Data Construction**: To address the extreme scarcity of DQ reviews, a 6-step iterative pipeline was designed: ① Collect 117 seed DQ reviews from public internet articles and comment sections; ② Randomly sample 300 standard reviews to form the baseline dataset; ③ Train a few-shot classifier using SetFit (based on `st-polish-paraphrase-from-distilroberta`); ④ Predict on all CENEO/WIZAZ reviews and rank them in descending order of probability; ⑤ Select the 200 highest-probability reviews for manual validation by annotators to classify them into three categories: dual quality / other problems / standard; ⑥ Return to step ③. After 7 iterations, 237 annotations from the demo system were merged, yielding a final dataset of 1,957 reviews (540 DQ, 281 other problems, 1,136 standard), with 67 (3.4%) label errors corrected through cross-validation.

2. **Full-Spectrum Comparison of Three NLP Approaches**: 

    - (a) **SetFit + Sentence Transformer**: Fine-tunes the sentence encoder first using contrastive loss, and then trains a logistic regression classifier on the encoded vectors. It evaluates 11 encoders including LaBSE, multi-e5 series, mGTE, and Polish-specific mpnet/distilroberta/mmlw.
    - (b) **Transformer Encoder Full Fine-Tuning**: Appends a linear classification head to the top of the pre-trained model and fine-tunes end-to-end using cross-entropy loss. It evaluates 7 models including mBERT, xlm-roberta-base/large, herbert-base/large, and polish-roberta-base/large.
    - (c) **LLM In-Context Learning**: Evaluates DeepSeek-v3 and GPT-4o under four prompting strategies (zero-shot, few-shot, zero-shot+instruction, and few-shot+instruction) without any training.

3. **Multilingual Expansion and Deployment Considerations**: Extracts 200k reviews each for English, German, and French from the Amazon multilingual dataset, performs primary filtering with SetFit, and manually validates them to construct a multilingual test set of 206 reviews (58 DQ) to evaluate cross-lingual transfer. For deployment, as the system is customized for internal use by UOKiK, high precision is prioritized to minimize the redundant workload of human analysts. `polish-roberta-large-v2` is recommended as the production model due to its support for local deployment, low latency, and zero external dependencies.

## Experiments

### Experimental Setup

- **Dataset**: DQ dataset of 1,957 reviews with train/test/valid split of 1,200/500/257, and average review length of 261 characters / 41 words.
- **Evaluation Metrics**: Precision/Recall/F1 for the DQ class (focusing heavily on Precision to reduce false positives); Accuracy and macro F1 across all classes.
- **Replications**: Each experiment runs across 5 different random seeds, reporting the mean $\pm$ standard deviation.

### Main Results

| Method Category | Representative Model | DQ Prec. | DQ Recall | DQ F1 | Accuracy | macro F1 |
|----------|----------|----------|-----------|-------|----------|----------|
| Baseline | Country Keyword Matching | 42.4 | 84.8 | 56.5 | 55.2 | 39.5 |
| SetFit | multi-e5-large | 77.5 | 76.8 | **77.1** | **79.6** | **72.7** |
| SetFit | mmlw-roberta-base | **77.9** | 73.6 | 75.7 | 78.6 | 72.6 |
| Encoder | herbert-large-cased | 81.5 | 80.7 | 81.1 | **82.4** | **76.7** |
| Encoder | xlm-roberta-large | 78.3 | **86.1** | **82.0** | 82.0 | 75.9 |
| Encoder | polish-roberta-large-v2 | **84.6** | 77.5 | 80.7 | 81.7 | 75.8 |
| LLM | DeepSeek-v3 zero-shot+inst. | 84.7 | 80.6 | **82.6** | 70.7 | 68.7 |
| LLM | GPT-4o zero-shot+inst. | 85.7 | 76.7 | 80.9 | **75.0** | **72.5** |
| LLM | GPT-4o few-shot+inst. | **86.0** | 75.1 | 80.1 | 68.5 | 67.7 |

### Robustness Validation

Five types of text perturbations were applied to three representative models to observe the decision change rate (%, lower is more robust):

| Perturbation Type | GPT-4o | polish-roberta | herbert |
|----------|--------|----------------|---------|
| Add/delete period at sentence end | 4.0 | 4.2 | 5.0 |
| Toggle capitalization of first letter | 4.0 | 2.8 | 2.6 |
| Convert text to lowercase | 5.0 | 4.6 | 4.2 |
| Polish characters → Latin characters | 5.0 | 4.6 | 4.6 |
| Single replacement of Polish characters | 4.0 | 4.0 | 3.6 |

### Cross-Lingual Transfer

Tested on 206 English/German/French reviews (58 DQ, 18 other problems, 130 standard): `xlm-roberta-large` achieved a DQ F1 of 72.3%, while DeepSeek-v3 few-shot+inst. achieved a DQ Precision of up to 91.9% but with a Recall of only 50.6%. In the cross-lingual scenario, the encoder significantly outperforms the LLM in Recall (66.9% vs. 50.6%), while the LLM is stronger in Precision (91.9% vs. 84.8%), demonstrating a complementary relationship.

### Key Findings

- **Language-Specific vs. Multilingual**: Polish-specific large models (`herbert-large`, `polish-roberta-large`) achieve a DQ F1 of 80-82% in Polish scenarios, on par with the multilingual `xlm-roberta-large` (82.0%), and significantly outperform their base versions.
- **Instruction Prompting is Crucial for LLMs**: After adding task definition instructions, the DQ F1 of GPT-4o and DeepSeek-v3 improved by around 20 percentage points (from 60 to 80+). However, few-shot examples sometimes degraded overall performance, suggesting difficulties in selecting representative examples.
- **Differences in Error Patterns**: GPT-4o primarily confuses "standard" and "other problems" classes. `polish-roberta` tends to misclassify DQ reviews as "standard" (false negatives), whereas `herbert` achieves the highest DQ detection rate but also generates the most false positives.
- **Excellent Robustness**: The decision change rate for all models under minor text perturbations remains within the range of 2.6-5.0%.

## Highlights & Limitations

### Highlights & Insights

- Defines the NLP task of "product review dual quality detection" for the first time, filling a research gap in the consumer protection domain.
- The iterative active learning pipeline for data construction (seed → SetFit screening → manual validation → expansion loop) offers general reference value for validating and labeling rare categories.
- Extremely comprehensive experiments: 11 SetFit encoders + 7 Transformer encoders + 2 LLMs × 4 prompting strategies, complemented by robustness validation and cross-lingual evaluation.

### Limitations & Future Work

- The dataset size is limited (1,957 reviews) and highly focused on the Polish e-commerce scenario; the domain generalization capability remains unexplored.
- The definition of the "other problems" class is somewhat broad (mixing counterfeits, quality degradation, misdelivery, etc.), creating semantic overlap with the DQ class and mounting classification difficulty.
- The cross-lingual evaluation employs only 206 test samples, limiting the statistical confidence of the multilingual generalization conclusions.
- LLMs' Accuracy and macro F1 are significantly lower than those of the encoder models, indicating that LLMs still fall short in global balance for three-class classification.

## Related Work & Insights

- **Socio-Economic Studies on Dual Quality**: Veselovská (2022), Bartkova & Sirotiaková (2021), etc., analyze the impact of DQ on market trust and purchasing decisions from a consumer behavior perspective.
- **E-commerce NLP**: Review analysis (_Botunac et al. 2024_), product Q&A (_Shen et al. 2023; Wang et al. 2023_), product classification (_Gong et al. 2023_), and review moderation (_Nayak & Garera 2022_).
- **Few-Shot Learning**: SetFit (_Tunstall et al. 2022_) implements few-shot classification by fine-tuning sentence encoders via contrastive learning.
- **Polish Pre-trained Models**: HerBERT (_Mroczkowski et al. 2021_), Polish RoBERTa (_Dadas et al. 2020_), and PL-MTEB benchmark (_Poświata et al. 2024_).
- **Sentence Transformers**: LaBSE (_Feng et al. 2022_), E5 (_Wang et al. 2024_), and mGTE (_Zhang et al. 2024_).

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Defines the DQ review detection task for the first time; the dataset construction method is innovative |
| Technical Depth | ⭐⭐⭐ | The method is primarily based on combinations of existing techniques, with no new model architecture proposed |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Comparison of 20+ models/strategies, including robustness validation, error analysis, and cross-lingual evaluation |
| Value | ⭐⭐⭐⭐ | Already deployed at Polish UOKiK, with clear real-world deployment scenarios |
| Overall Recommendation | ⭐⭐⭐⭐ | Novel task + comprehensive experiments + real-world deployment; an exemplary model for applied NLP research |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews](lazyreview_peer_review.md)
- [\[ACL 2025\] Literature Meets Data: A Synergistic Approach to Hypothesis Generation](literature_meets_data_hypothesis.md)
- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [\[ACL 2025\] PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights](praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)

</div>

<!-- RELATED:END -->
