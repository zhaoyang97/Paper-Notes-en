---
title: >-
  [Paper Note] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus
description: >-
  [ACL 2025][Multimodal VLM][Multilingual] Proposes mOSCAR, the first large-scale multilingual multimodal document-level corpus (163 languages, 303M documents, 200B tokens, 1.15B images), extracted as interleaved image-text documents from Common Crawl, demonstrating significant few-shot learning gains for multilingual mLLMs trained on this data.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multilingual"
  - "Multimodal"
  - "Interleaved Image-text Data"
  - "Web Crawling"
  - "Few-shot Learning"
date: 2026-05-08
content_hash: 3bd678dfc67c47ad
---

# mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus

**Conference**: ACL 2025  
**arXiv**: [2406.08707](https://arxiv.org/abs/2406.08707)  
**Code**: [oscar-project](https://oscar-project.github.io/documentation/versions/mOSCAR/)  
**Area**: Multilingual Translation  
**Keywords**: Multilingual, Multimodal, Interleaved Image-text Data, Web Crawling, Few-shot Learning  

## TL;DR

Proposes mOSCAR, the first large-scale multilingual multimodal document-level corpus (163 languages, 303M documents, 200B tokens, 1.15B images), extracted as interleaved image-text documents from Common Crawl, demonstrating significant few-shot learning gains for multilingual mLLMs trained on this data.

## Background & Motivation

- **Background**: Multimodal large language models (mLLMs) have made tremendous progress. Models like Flamingo demonstrate that training on interleaved image-text sequences enables the emergence of in-context learning capabilities. However, the M3W dataset they use is proprietary and English-only.
- **Limitations of Prior Work**: Currently available open interleaved image-text datasets (e.g., mmc4, OBELICS) only support English. Existing multilingual image-text datasets (e.g., WIT) are limited in scale or only contain caption pairs, failing to support the pre-training of multilingual mLLMs.
- **Key Challenge**: While there are over 7000 languages globally, multilingual multimodal pre-training resources are extremely scarce. Multilingual data generated via machine translation suffers from poor translation quality and cultural biases.
- **Goal**: Construct an open-source, large-scale, multilingual interleaved image-text document dataset for training multilingual mLLMs with few-shot learning capabilities.
- **Key Insight**: Extract multilingual interleaved image-text documents directly from raw Common Crawl web pages, and guarantee data quality through systematic filtering, deduplication, and safety filtering pipelines.
- **Core Idea**: Explicitly and systematically extract and release a large-scale interleaved image-text document corpus covering 163 languages from Common Crawl for the first time.

## Method

### Overall Architecture

The construction pipeline of mOSCAR consists of six stages: Data Acquisition $\to$ Language Identification $\to$ Text-Only Filtering $\to$ Image-Only Filtering $\to$ Data Decontamination $\to$ Joint Image-Text Filtering. The entire pipeline extracts data from three WARC files of Common Crawl dumps from 2023.

### Key Designs

#### 1. Data Acquisition and DOM Parsing
- **Function**: Extract text nodes and image nodes from the HTML DOM tree in WARC files.
- **Mechanism**: Process WARC files using the FastWARC library, extracting text nodes like `<p>`, `<h*>`, and `<title>` along with `<img>` image nodes via a depth-first traversal of the DOM tree.
- **Design Motivation**: HTML documents naturally maintain the interleaved relationship between text and images. Extracting them directly from the web structure preserves the original multimodal document organization format.

#### 2. Language Identification
- **Function**: Determine the primary language of each document.
- **Mechanism**: Perform language identification on each text node using the open-LID detector, retrieve the top 3 highest-probability languages, and determine the document language via weighted voting based on character counts.
- **Design Motivation**: Weighting by character count is chosen because short text nodes (e.g., "Subscribe", "Newsletter") can easily mislead language identification.

#### 3. Multilayer Safety Filtering
- **Function**: Multilayer filtering to ensure data safety and quality.
- **Mechanism**:
    - **NSFW Text Filtering**: Remove adult content using English regular expressions (removing ~0.5% of documents).
    - **Toxicity Filtering**: Remove documents containing $\ge 2$ toxic words using the FLORES multilingual toxic word list.
    - **PII Removal**: Replace emails, phone numbers, credit card numbers, IP addresses, etc., using regular expressions.
    - **NSFW Image Filtering**: Perform detection using a dual-model combination of nsfw-detector (MobileNet) and NudeNet.
    - **CSAM Detection**: Use Thorn's proprietary classifier with a threshold of 0.4 to favor recall.
- **Design Motivation**: Explicitly favor recall to minimize the risk of unsafe content, even at the cost of falsely deleting safe content.

#### 4. Joint Image-Text Filtering
- **Function**: Ensure that the images and text within a document are relevant to each other.
- **Mechanism**: Calculate the cosine similarity of all image-text pairs within a document using the multilingual NLLB-SigLIP. Simulate a retrieval task (randomly sampling 63 negative examples); keep the image/text if it ranks in the top-8.
- **Design Motivation**: Avoid using a fixed similarity threshold (since the optimal threshold varies across different languages) and instead use relative ranking to determine relevance.

#### 5. Deduplication Strategy
- **Function**: Multi-granularity deduplication to improve training efficiency.
- **Mechanism**: 
    - Exact deduplication of text nodes within documents + approximate deduplication using Levenshtein ratio (threshold 0.95).
    - Cross-document approximate deduplication using MinHashLSH (removing 19% on average).
    - Image deduplication using URL matching + perceptual hashing (pHash).
    - No cross-lingual deduplication for images to promote cross-lingual transfer.
- **Design Motivation**: Retain the same images across different languages because identical images appearing in documents of different languages can facilitate cross-lingual knowledge transfer.

### Loss & Training

This work focuses on the dataset and does not involve the design of new loss functions. Model training uses the standard Flamingo architecture (OpenFlamingo) with an autoregressive language modeling loss.

## Key Experimental Results

### Main Results: Performance of Multilingual OpenFlamingo on 8 Benchmarks

| Setting | xFlickR&CO | XM3600 | xGQA | MaXM | MaRVL | XVNLI | Multi30K | CoMMuTE |
|------|-----------|--------|------|------|-------|-------|---------|---------|
| mOSCAR+cap (16-shot) | **39.46** | **23.67** | **35.23** | **27.47** | 49.84 | 34.85 | **23.85** | 62.78 |
| cap only (16-shot) | 19.87 | 12.07 | 13.37 | 4.89 | 49.79 | 32.70 | 0.74 | 60.25 |

- In the 16-shot setup, the model trained with mOSCAR auxiliary data gains **+19.59** on xFlickR&CO and **+21.86** on xGQA compared to the model trained only on captions.

### Ablation Study: mOSCAR vs WIT (Fair Comparison on 35M Documents)

| Setting | xFlickR&CO | XM3600 | xGQA | MaXM |
|------|-----------|--------|------|------|
| mOSCAR+cap (8-shot) | **36.77** | **22.15** | **33.90** | **24.41** |
| WIT+cap (8-shot) | 8.91 | 3.63 | 27.06 | 16.81 |

- Under a fair data-scale comparison, mOSCAR significantly outperforms WIT, particularly showing a massive gap on captioning tasks.

### Key Findings

1. **Significant Few-shot Gains**: From 0-shot to 16-shot, the mOSCAR-trained model achieves an average gain of +6.71 on VQA tasks and +19.39 CIDEr on caption tasks, while the caption-only model merely improves by +2.82 and +9.08, respectively.
2. **No Loss in Multilingual Performance**: The multilingual model trained across all 43 languages does not perform worse in English compared to the model trained exclusively on the English subset (Table 7).
3. **Cross-lingual Image Diversity**: Images sampled multilingually exhibit higher diversity than English-only ones (Vendi score 54.78 vs 52.36), indicating cultural variations in image distributions across different language documents.

## Highlights & Insights

- **Outstanding Systems Engineering Contribution**: The comprehensive and strict data processing pipeline covers three core dimensions: safety (NSFW/CSAM/PII/toxic content), quality (text node filtering/deduplication), and relevance (joint image-text filtering).
- The design of **no cross-lingual deduplication for images** is highly intuitive – utilizing the same image paired with different language texts facilitates multilingual visual alignment.
- **Joint image-text filtering** uses retrieval-based relative ranking instead of an absolute threshold, which naturally accommodates distributional differences across various languages.
- It generalizes English interleaved datasets like OBELICS and mmc4 to a multilingual scope, filling a critical gap.

## Limitations & Future Work

- A systematic bias analysis was not conducted (web data may reflect biases present on the internet), which necessitates additional alignment training to mitigate.
- Although low-resource languages are covered, their data volume remains significantly lower than that of high-resource languages.
- Validation was carried out using only Gemma-2B as the backbone model; performance on larger models remains to be explored.
- The filtering pipeline's cultural adaptability is limited (e.g., toxic word lists may not apply uniformly across all cultural contexts).
- Despite the dataset's large scale, images must be compiled/downloaded by users themselves, introducing the risk of broken links over time.

## Related Work & Insights

- **Flamingo $\to$ OpenFlamingo**: The interleaved image-text training paradigm has been validated as a key driver of in-context learning in mLLMs.
- **OBELICS / mmc4**: The success of English interleaved datasets proves the feasibility of extracting documents from web pages.
- **Lessons from LAION-5B**: Safety issues in large-scale web data (e.g., CSAM) require systematic resolution.
- **NLLB-SigLIP**: Multilingual CLIP variants provide cross-lingual support for joint image-text filtering.
- **Insight**: In constructing multilingual multimodal datasets, the "quality-diversity trade-off" remains a core design decision.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The first large-scale multilingual interleaved image-text dataset, filling an important gap.
- **Value** ⭐⭐⭐⭐⭐: Open-source release, CC BY 4.0 license, directly driving multilingual mLLM research.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Multidimensional evaluations (quality, diversity, safety, model training), but only validated on a small-scale model.
- **Writing Quality** ⭐⭐⭐⭐: Thorough pipeline description, but the paper is long with experimental details mostly relegated to the appendix.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus](cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)
- [\[ACL 2025\] Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings](towards_storage-efficient_visual_document_retrieval_an_empirical_study_on_reduci.md)
- [\[ACL 2025\] Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model](centurio_multilingual_vlm.md)
- [\[ACL 2025\] SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning](singakids_a_multilingual_multimodal_dialogic_tutor_for_language_learning.md)
- [\[CVPR 2025\] DocoPilot: Improving Multimodal Models for Document-Level Understanding](../../CVPR2025/multimodal_vlm/docopilot_improving_multimodal_models_for_document-level_understanding.md)

</div>

<!-- RELATED:END -->
