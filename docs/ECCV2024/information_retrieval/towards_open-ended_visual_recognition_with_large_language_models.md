---
title: >-
  [Paper Note] Towards Open-Ended Visual Recognition with Large Language Model
description: >-
  [ECCV 2024][Information Retrieval & RAG][Open-ended visual recognition] This paper proposes the OmniScient Model (OSM)—a generative mask classifier based on a frozen CLIP-ViT, a trainable MaskQ-Former, and a frozen LLM (Vicuna-7B). It shifts visual recognition from "selecting categories from a predefined vocabulary" to "directly generating category names," eliminating the dependency on predefined vocabularies during both training and testing. It outperforms DaTaSeg by +4.3 PQ…
tags:
  - "ECCV 2024"
  - "Information Retrieval & RAG"
  - "Open-ended visual recognition"
  - "large language model"
  - "mask classification"
  - "generative recognition"
  - "multi-dataset training"
date: 2026-05-08
content_hash: 7a78cde91e73879c
---

# Towards Open-Ended Visual Recognition with Large Language Model

**Conference**: ECCV 2024  
**arXiv**: [2311.08400](https://arxiv.org/abs/2311.08400)  
**Code**: [GitHub](https://github.com/bytedance/OmniScient-Model)  
**Area**: LLM/NLP  
**Keywords**: Open-ended visual recognition, large language model, mask classification, generative recognition, multi-dataset training

## TL;DR

This paper proposes the OmniScient Model (OSM)—a generative mask classifier based on a frozen CLIP-ViT, a trainable MaskQ-Former, and a frozen LLM (Vicuna-7B). It shifts visual recognition from "selecting categories from a predefined vocabulary" to "directly generating category names," eliminating the dependency on predefined vocabularies during both training and testing. It outperforms DaTaSeg by +4.3 PQ on COCO panoptic segmentation.

## Background & Motivation

Object localization and recognition in the open world are long-standing challenges in computer vision. Current mainstream methods decompose this problem into two sub-tasks:

**Category-agnostic mask/box proposal**: Models like SAM have demonstrated strong zero-shot generalization capabilities.

**Open-vocabulary classification**: Classifying based on pre-extracted text embeddings from VLMs like CLIP.

However, existing open-vocabulary recognition methods suffer from two fundamental limitations:

**Limitation 1: Category names must be provided during testing.** Users need to predefine all possible semantic categories, making the recognition performance highly dependent on the quality of this predefined vocabulary. In real-world scenarios, pre-enumerating all possible category names is almost impossible.

**Limitation 2: Label conflicts in multi-dataset training.** Label definitions from different datasets might conflict (e.g., "tv" in COCO is semantically equivalent to "monitor" in another dataset). Existing methods either need to train a separate decoder/classifier for each dataset or require manual consolidation of label spaces, significantly increasing engineering complexity.

The authors propose a paradigm shift: **from discriminative recognition (selecting from a vocabulary) to generative recognition (directly generating category names)**, referred to as open-ended visual recognition.

## Method

### Overall Architecture

OSM adopts a modular architecture consisting of three core components:

1. **Frozen CLIP-ViT**: Extracts high-resolution visual features (utilizing a sliding-window approach).
2. **Trainable MaskQ-Former**: A mask-aware visual feature resampling module.
3. **Frozen LLM (Vicuna-7B)**: Predicts category names in a generative manner.

Overall pipeline: Input image $\rightarrow$ CLIP-ViT extracts features $\rightarrow$ Segmentation model (SAM/kMaX-DeepLab) generates masks $\rightarrow$ MaskQ-Former extracts mask region features $\rightarrow$ LLM generates category names.

### Problem Formulation: From Classification to Generation

Traditional classification: Given an image $\mathbf{I}$ and $M$ segmentation masks $\mathbf{M}$, the task is to predict the category $c_i \in C$ for each mask, where $C$ is a predefined category set.

**Open-ended recognition**: Assumes that the vocabulary $C$ is unknown during both training and testing. Recognition is reformulated as a text generation task that maximizes the conditional likelihood of the category name:

$$p(c_i) = \prod_{j=0}^{N} p(c_{i,j} | c_{i,0}, \cdots, c_{i,j-1})$$

where $c_{i,j}$ is the $j$-th token of the category name. The model directly generates the category name instead of selecting from a candidate set.

### High-Resolution Feature Extraction (Sliding Window Strategy)

**Problem**: The pre-trained resolution of CLIP-ViT is 224×224, and directly applying it to high-resolution inputs severely degrades performance (the frozen ViT has poor generalization capability to resolution changes).

**Solution**: A sliding-window strategy is applied at the input level:
- Slice the high-resolution image (e.g., 896×896) into multiple windows that match the pre-trained size.
- Process each window independently through the frozen ViT to extract features.
- Add global positional encodings to compensate for missing positional information across windows.

Experiments demonstrate that increasing the resolution from 224 to 448 yields a **+12.8%** improvement in classification accuracy, with the optimal resolution being 1120×1120. Direct global adaptation of high-resolution inputs (without sliding windows) leads to a performance drop of **-7.2%**.

### MaskQ-Former: Mask-Aware Feature Resampling

Existing Q-Former or Perceiver Resampler architectures use global attention to aggregate image features without considering segmentation mask information. OSM proposes MaskQ-Former, which contains two sets of learnable queries:

**Mask Queries**:
- Focus exclusively on the pixel features within the mask area via masked cross-attention.
- Concentrate on the visual information of the target object itself.

**Context Queries**:
- Focus on the expanded region of the mask (e.g., the bounding box area).
- Provide surrounding context for the target object (context is crucial for recognition).

The two sets of queries interact through self-attention layers with shared parameters (differing only in query initialization), incurring virtually no extra overhead. Finally, only Mask Queries are retained as input to the LLM.

**Ablation of Context Regions**: Expanding the bounding box by 0.5× yields the best performance (**+2.6%** compared to global context); neither overly tight nor overly loose regions are optimal.

### Mode Query: Dual-Mode of Vocabulary-Specific vs. Vocabulary-Agnostic

To balance open-ended recognition capabilities with alignment to dataset-specific vocabularies, OSM introduces a **Mode Query** mechanism:

- **Vocabulary-Specific Query**: Each training dataset is associated with its own learnable query. Activating the corresponding dataset query during training helps the model "memorize" the label space of that dataset.
- **Vocabulary-Agnostic Query**: A shared general query across all datasets, which can be activated on any dataset during training to preserve open-ended recognition capabilities.

During training, half of the samples in each batch activate the vocab-specific query, while the other half activate the vocab-agnostic query. Testing allows for flexible selection:
- For aligning with a specific vocabulary $\rightarrow$ Activate the vocab-specific query.
- For open-ended predictions $\rightarrow$ Activate the vocab-agnostic query.

Ablation experiments prove that: without the Mode Query, the model generalizes better but shows decreased alignment to specific datasets; adding the Mode Query improves both aspects.

### Loss & Training

**Datasets**: Jointly trained on six public segmentation datasets:
- COCO panoptic segmentation, ADE20K panoptic segmentation, Cityscapes panoptic segmentation
- LVIS instance segmentation, ADE-847 semantic segmentation, PC-459 semantic segmentation

**Training Scheme**:
- Employs an instruction tuning strategy.
- In each iteration, an image and a ground-truth (GT) mask are randomly selected, a prompt is chosen among 19 instruction templates, and the true category name is inserted.
- Standard next-token prediction loss is used.
- Initialized from InstructBLIP pre-trained weights (EVA-ViT-g/224 + Vicuna-7B).
- 32 mask queries + 32 context queries + 1 mode query.
- AdamW optimizer, learning rate of $4\times10^{-5}$, weight decay of $0.05$, with cosine decay.
- Dataset-specific batch sizes are applied (COCO: 32, LVIS: 64, ADE20K: 16, etc.).
- A total of 6M masks are processed.

**Inference**: Uses the default template "What is in the segmentation mask?" and adopts greedy decoding.

## Key Experimental Results

### Main Results: GT Mask Classification

OSM evaluates mask classification accuracy (Acc) and non-in-vocabulary prediction ratio (NIV) across six datasets:

| Setting | COCO Acc | LVIS Acc | ADE20K Acc | Cityscapes Acc | A847 Acc | PC459 Acc | Average Acc |
|------|----------|----------|------------|---------------|----------|----------|---------|
| Single-dataset Training | 85.5 | 68.3 | 82.3 | 79.4 | 76.9 | 80.9 | - |
| Learnable Embed | - | - | - | - | - | - | 78.9 |
| Text Embed | - | - | - | - | - | - | < 78.7 |
| OSM vocab-specific | - | - | - | - | - | - | 78.7 |
| OSM vocab-agnostic | - | - | - | - | - | - | Slightly lower |
| OSM† vocab-specific | - | - | - | - | - | - | Significant improvement |

Key Findings: The performance of the generative model (OSM) on discriminative tasks is **on par** with discriminative models (Learnable Embed) (78.7% vs 78.9%), with an extremely low NIV, suggesting that the generative model can be effectively constrained within the training vocabulary.

### Main Results: Combining with Off-the-shelf Segmenters

Using kMaX-DeepLab as the mask proposal model, OSM is compared with other multi-dataset generic segmentation methods:

| Method | backbone | COCO PQ | ADE20K PQ | ADE20K mIoU | Cityscapes PQ | Cityscapes mIoU |
|------|----------|---------|-----------|-------------|---------------|-----------------|
| Mask2Former (Expert Model) | ResNet50 | 51.9 | 39.7 | 46.1 | 62.1 | 77.5 |
| LMSeg | ResNet50 | 38.6 | 35.4 | 45.2 | 54.8 | 80.9 |
| DaTaSeg | ResNet50 | 49.0 | 29.8 | 48.1 | - | - |
| DaTaSeg | ViTDet-L | 53.5 | 33.4 | 54.0 | - | - |
| **OSM** | ResNet50 | **53.3** | **43.8** | **50.0** | 59.5 | 77.0 |
| **OSM** | ConvNeXt-L | **56.1** | **49.7** | **55.2** | **64.7** | 80.2 |

- OSM (ResNet50) outperforms LMSeg by **+14.7 PQ (COCO), +8.4 PQ (ADE20K), and +4.7 PQ (Cityscapes)**.
- OSM outperforms DaTaSeg by **+4.3 PQ (COCO, ResNet50) and +2.6 PQ (COCO, Large)**.
- OSM achieves comparable performance with the expert model Mask2Former.

### Ablation Study

**Input Resolution Ablation**:

| Resolution | Average Acc |
|--------|---------|
| 224×224 | Baseline |
| 448×448 | +12.8% |
| 896×896 | Continued improvement |
| 1120×1120 | Optimal |
| Higher | Performance drop |

**Sliding Window vs. Global**: Sliding window performs **+7.2%** better than directly processing high-resolution inputs globally.

**Context Region Expansion**:

| Context Range | Change relative to Global |
|-----------|-------------|
| Global | Baseline |
| 0.0× (tight bbox) | +0.8% |
| 0.5× (expanded bbox) | +2.6% |

**Open-Vocabulary Evaluation** (trained only on COCO+LVIS, zero-shot evaluation on ADE20K):

| Method | PQ | AP | mIoU |
|------|----|----|------|
| MaskCLIP | 15.1 | 6.0 | 23.7 |
| FC-CLIP | 17.8 | 11.1 | 20.8 |
| ODISE | 19.5 | 10.8 | 23.8 |
| **OSM** | **21.4** | **12.4** | **26.9** |

### Key Findings

1. **Generative $\approx$ Discriminative**: OSM performs on par with learnable embedding classifiers in discriminative tasks, challenging the bias that 'generative models are unsuitable for discriminative tasks.'
2. **Accuracy-Generalization Trade-off**: As the number of training masks increases (1M $\rightarrow$ 9M), Acc improves while NIV decreases (weakening generalization capability), representing an inherent trade-off.
3. **Emergent Part Recognition**: Although never trained on part segmentation data, OSM exhibits the emergent ability to predict parts like "tail" or "ear" (leveraging the world knowledge of the LLM).
4. **Comparison with GPT-4V**: OSM is more accurate in mask classification than GPT-4V but tends to make more conservative predictions (e.g., predicting "person" instead of "man in armor").

## Highlights & Insights

1. **Paradigm Shift**: Transitioning from discriminative (selecting from a vocabulary) to generative (directly generating category names) completely eliminates the dependency on predefined vocabularies, marking a significant evolution in visual recognition paradigms.
2. **No Manual Intervention for Multi-Dataset Training**: The Mode Query mechanism elegantly resolves cross-dataset label conflicts, eliminating the need for manual label consolidation.
3. **High-Resolution Feature Extraction via Sliding Window**: This seemingly simple strategy yields a massive 12.8% boost, proving that high-resolution adaptation of frozen ViTs does not require complex designs.
4. **Mask-Context Dual-Query Design in MaskQ-Former**: It balances precise target region features with contextual information, supported by an efficient and elegant parameter-sharing scheme.
5. **NIV Analysis**: Visualizing NIV cases reveals that OSM's predictions (e.g., "monitor") are often more accurate than the ground-truth annotations (e.g., labeled as "tv" in COCO), exposing the inherent bias in fixed-vocabulary annotations.

## Limitations & Future Work

1. **Accuracy-Generalization Trade-off**: Longer training makes the model more prone to conservative predictions (lower NIV). Maintaining open-ended generation capacity while ensuring accuracy remains an open issue.
2. **Computational Overhead**: The frozen LLM (Vicuna-7B) requires autoregressive generation for category names during inference, which is significantly slower than simple embedding matching.
3. **Prediction Conservatism**: Compared to GPT-4V, OSM tends to generate more generic category names ("person" vs. "man in armor"), which could potentially be mitigated by using stronger base LLMs (e.g., Llama-2) or scaling up the training data.
4. **Unprocessed Image-Level Data**: Image-level datasets like ImageNet are not utilized due to single-label constraints being unsuited for multi-object scenarios. Exploring image-level weakly supervised integration is a direction for future work.
5. **Dependency on External Segmenters**: Mask quality directly impacts recognition performance; joint end-to-end training of segmentation and recognition could be superior.

## Related Work & Insights

- **Vocabulary-Free Classification**: Parallel works eliminate predefined requirements by retrieving or parsing generated vocabularies, whereas OSM addresses this more thoroughly via a generative approach.
- **InstructBLIP / LLaVA**: Representative modular multimodal LLMs; OSM builds upon them by incorporating mask-aware capabilities.
- **DaTaSeg / LMSeg**: Discriminative multi-dataset segmentation methods that require independent classifiers or manual label consolidation, which OSM naturally avoids.
- **Insights**: The generative recognition paradigm can be extended to other vision tasks like detection and tracking, and the Mode Query mechanism can be applied to other multi-task or multi-dataset scenarios.

## Rating

| Dimension | Score (1-10) | Description |
|------|:-----------:|------|
| Novelty | 9 | Open-ended recognition paradigm + MaskQ-Former + Mode Query, paradigm shift is highly significant. |
| Technical Depth | 8 | Exquisitely designed modules; sliding window, dual-query, and dual-mode queries are tightly integrated. |
| Experimental Thoroughness | 9 | 6 datasets, GT masks + off-the-shelf masks, multi-dimensional ablations, comparison with GPT-4V, and open-vocabulary evaluation. |
| Writing Quality | 8 | Clear formulation of motivations, with intuitive comparison diagrams of discriminative vs. generative paradigms. |
| Value | 8 | Eliminates vocabulary dependency with zero manual intervention for multi-dataset training, offering great practical deployment value. |
| **Overall Score** | **8.4** | Outstanding paradigm innovation with rigorous and comprehensive experiments. Reliable quality from ByteDance. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](../../ACL2025/information_retrieval/saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)
- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](../../ICML2026/information_retrieval/understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[ECCV 2024\] Multi-Label Cluster Discrimination for Visual Representation Learning](multi-label_cluster_discrimination_for_visual_representation_learning.md)
- [\[NeurIPS 2025\] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](../../NeurIPS2025/information_retrieval/murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)
- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)

</div>

<!-- RELATED:END -->
