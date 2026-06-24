---
title: >-
  [Paper Note] Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning
description: >-
  [CVPR 2025][Information Retrieval & RAG][Few-Shot Learning] This paper extends Retrieval-Augmented Learning (RAL) to Few-Shot Recognition (FSR) for the first time, exposing two main challenges of retrieved data: distribution imbalance and domain gap. It proposes a two-stage method, SWAT (finetuning the vision encoder on mixed data first, then retraining the classifier on few-shot labeled data), outperforming all prior methods by $>6\%$ across 9 benchmarks.
tags:
  - "CVPR 2025"
  - "Information Retrieval & RAG"
  - "Few-Shot Learning"
  - "Retrieval-Augmented Learning"
  - "Vision-Language Models"
  - "Stage-Wise Finetuning"
  - "Data Imbalance"
date: 2026-05-08
content_hash: e2788e8d1908cc5e
---

# Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning

**Conference**: CVPR 2025  
**arXiv**: [2406.11148](https://arxiv.org/abs/2406.11148)  
**Code**: [https://tian1327.github.io/SWAT](https://tian1327.github.io/SWAT)  
**Area**: Information Retrieval  
**Keywords**: Few-Shot Learning, Retrieval-Augmented Learning, Vision-Language Models, Stage-Wise Finetuning, Data Imbalance

## TL;DR

This paper extends Retrieval-Augmented Learning (RAL) to Few-Shot Recognition (FSR) for the first time, exposing two main challenges of retrieved data: distribution imbalance and domain gap. It proposes a two-stage method, SWAT (finetuning the vision encoder on mixed data first, then retraining the classifier on few-shot labeled data), outperforming all prior methods by $>6\%$ across 9 benchmarks.

## Background & Motivation

**Background**: Few-Shot Recognition (FSR) aims to train classification models with only a few labeled samples per class. Current mainstream methods rely on pretrained Vision-Language Models (VLMs, e.g., CLIP), mainly adapting to downstream tasks by learning a small number of parameters (such as prompt tokens, lightweight adapters, or classification heads) while keeping the vision encoder frozen. Another direction is Retrieval-Augmented Learning (RAL), which retrieves relevant samples from the pretraining data of VLMs for zero-shot recognition, achieving SOTA results.

**Limitations of Prior Work**: (1) Existing FSR methods treat FSR as a proxy task for researching PEFT or model robustness rather than truly pursuing maximum accuracy, leading to restricted designs—freezing the backbone and only learning a few parameters, which misses the potential of simply finetuning the entire encoder. (2) Most FSR methods unreasonably use large-scale validation sets for hyperparameter tuning, lacking practical application value. (3) RAL performs exceptionally well in zero-shot recognition but has not yet been applied to FSR.

**Key Challenge**: Intuitively, RAL should help FSR (more relevant data should theoretically facilitate learning), but experiments show that finetuning VLMs solely on retrieved data performs even worse than SOTA zero-shot methods. This is due to two critical issues in retrieved data: **distribution imbalance** (VLM pretraining data is naturally long-tailed, making it impossible to retrieve enough samples for certain classes) and **domain gap** (web images and downstream task images differ greatly in style, resolution, and background, with a binary classifier achieving $>90\%$ accuracy in distinguishing them).

**Goal**: (1) Can the accuracy of FSR be improved by finetuning the entire VLM encoder? (2) How can retrieved data be effectively utilized to further improve FSR while addressing the challenges of distribution imbalance and domain gap?

**Key Insight**: Starting from data annotation applications (rather than PEFT research) and not limiting the number of learnable parameters, the authors show a surprising finding: simply finetuning the entire vision encoder on a small amount of labeled data already outperforms all prior FSR methods by over $3\%$! Based on this, RAL is introduced into FSR, identifying challenges and proposing a two-stage solution.

**Core Idea**: Through two-stage training—first finetuning the vision encoder + classifier end-to-end on mixed (retrieved + few-shot) data to learn general representations, then retraining the classifier solely on few-shot data to eliminate distribution bias—both domain gap and imbalance issues are resolved simultaneously.

## Method

### Overall Architecture

SWAT is based on CLIP's vision encoder and operates in two stages. The input consists of few-shot labeled samples from downstream tasks and relevant images retrieved from the VLM pretraining data. Stage 1 finetunes the vision encoder + classifier end-to-end on the mixed data. Stage 2 freezes the encoder and retrains the classifier using only the few-shot labeled samples. During inference, classification is performed directly with the model obtained from Stage 2.

### Key Designs

1. **Few-Shot Full-Finetuning (FSFT) Baseline**:

    - **Function**: Verifies the potential of finetuning the entire encoder on few-shot data.
    - **Mechanism**: Directly finetune CLIP's vision encoder (ViT-B/16) end-to-end on $K$ labeled samples per class, using robust hyperparameters (default values from the literature, including classifier initialization with text embeddings and a smaller learning rate for the vision encoder). No validation set is used for tuning. Surprisingly, despite the extremely small data scale (e.g., 16-shot), full encoder finetuning does not overfit; instead, it significantly outperforms all prior FSR methods by over $3\%$.
    - **Design Motivation**: Overcomes the conventional thinking in the FSR field that "the backbone must be frozen." The previously feared overfitting issue does not occur under the correct hyperparameter settings, likely because the strong initialization of the pretrained model provides sufficient regularization.

2. **Introduction of Retrieval-Augmented Learning (RAL) to FSR and Challenge Identification**:

    - **Function**: Utilizes VLM pretraining data to expand the training set.
    - **Mechanism**: Utilizes string matching to retrieve images from the LAION dataset that are relevant to downstream classes (more efficient and effective than feature matching). However, finetuning solely on retrieved data performs **worse than zero-shot methods**, owing to the combination of two problems: (a) **Domain Gap**—retrieved images differ dramatically from downstream task images (in validation experiments, a binary classifier distinguishes the two sources with $>90\%$ accuracy); (b) **Distribution Imbalance**—certain classes retrieve a large number of images while others receive very few, biasing the model toward majority classes.
    - **Design Motivation**: RAL is effective in zero-shot scenarios because there is no better data alternative, but in FSR, real-labeled few-shot data is available, and directly using retrieved data introduces noise. A proper way to utilize it must be found.

3. **SWAT Two-Stage Training**:

    - **Function**: Jointly resolves the domain gap and distribution imbalance issues.
    - **Mechanism**: **Stage 1**—Mix retrieved data with few-shot labeled data, and finetune the vision encoder + classifier end-to-end. Although the mixed data is unbalanced, its larger scale enables learning more general feature representations (similar to source-domain pretraining). Meanwhile, CutMix data augmentation is applied to paste random patches of few-shot images onto retrieved images, increasing sample diversity and implicitly bridging the domain gap. **Stage 2**—Freeze the encoder and retrain the linear classifier only on the balanced few-shot data, eliminating the classification bias caused by the unbalanced data in Stage 1. This aligns with the classic "decoupled learning" strategy in long-tailed learning.
    - **Design Motivation**: Stage 1 addresses the domain gap (the representations learned through larger datasets are more general, equivalent to transfer learning), and Stage 2 addresses the distribution imbalance (retraining the classifier on balanced data eliminates bias). The two stages work complementarily to jointly solve both issues.

### Loss & Training

Standard cross-entropy loss is used in both stages. The classifier is initialized using text embeddings (mapping class names through the CLIP text encoder to get vector representations as initial classification weights). The learning rate of the vision encoder is set to 1/10 of the classifier's learning rate. No validation set is used for hyperparameter tuning; all hyperparameters are kept consistent across all datasets. CutMix is the only data augmentation technique used (more complex augmentations like the MixUp+CutMix combination yielded no additional gains).

## Key Experimental Results

### Main Results

| Method | Type | 4-shot | 8-shot | 16-shot |
|------|------|--------|--------|---------|
| CLAP (CVPR'24) | adapter | 66.9 | 70.0 | 72.9 |
| CrossModal-LP (CVPR'23) | adapter | 65.4 | 68.8 | 71.8 |
| CoOp (IJCV'22) | prompt | 61.0 | 64.6 | 68.4 |
| REAL-Linear (CVPR'24) | RAL zero-shot | 64.8 | 64.8 | 64.8 |
| Few-shot finetune (Ours) | finetune | 69.7 (+2.8) | 73.3 (+3.3) | 76.3 (+3.4) |
| **SWAT (Ours)** | **finetune+RAL** | **73.5 (+6.6)** | **76.0 (+6.0)** | **78.2 (+5.3)** |

SWAT outperforms the prior best method CLAP by an absolute margin of $>6\%$ in average accuracy across 9 datasets, reaching $78.2\%$ under 16-shot. Even the simple Few-Shot Full-Finetuning (FSFT) outperforms CLAP by over $3\%$.

### Ablation Study

| Configuration | Common Class Accuracy | Rare Class Accuracy | Average Accuracy |
|------|---------------|-------------|----------|
| Stage 1: few-shot only | 76.8 | 73.6 | 76.3 |
| Stage 1: retrieved only | 64.8 | 44.2 | 62.8 |
| Stage 1: mixed | 76.1 | 68.2 | 75.3 |
| Stage 1: mixed + CutMix | 78.0 | 71.9 | 77.3 |
| + Stage 2: Classifier Retraining | **78.7** (+0.7) | **74.1** (+2.2) | **78.2** (+0.9) |

### Key Findings

- **Most surprising finding**: Simply finetuning the entire CLIP encoder on few-shot data outperforms all previous FSR methods by over $3\%$, suggesting that the FSR community has overly focused on PEFT while neglecting the potential of full-finetuning.
- **Double-edged sword effect of RAL**: Finetuning solely on retrieved data ($62.8\%$) is worse than direct few-shot finetuning ($76.3\%$), due to the domain gap and distribution imbalance.
- **Stage 2 classifier retraining benefits rare classes the most**: Rare class accuracy increases by $2.2\%$ (vs $0.7\%$ for common classes), confirming the effectiveness of the decoupled learning strategy in resolving imbalance.
- **CutMix is the only effective augmentation**: Other augmentation methods (MixUp, CutMix+MixUp combination) yield no extra benefits but increase computational overhead.
- **High training efficiency**: Because the volume of few-shot data is extremely small, the computational overhead of full-encoder finetuning is highly acceptable (barely more than adapter methods).

## Highlights & Insights

- **Breaking conventional thinking in FSR**: The biggest contribution of this paper is exposing a long-neglected blind spot—the FSR community invested heavily in PEFT but never attempted simple full-finetuning. This "embarrassingly simple" baseline outperforms all prior methods, reminding us to verify the potential of simple methods before pursuing complexity.
- **Elegance and simplicity of the two-stage decoupling strategy**: Without requiring any special architecture or regularization, it simultaneously solves both the domain gap and distribution imbalance by simply controlling "which data is used in which stage." This strategy is highly general and can be directly transferred to any learning scenarios with noisy or imbalanced auxiliary data.
- **Rigorous experimental setup**: No validation set for tuning, uniform hyperparameters across all datasets, and reporting standard deviations—this setup is more honest than most FSR papers.

## Limitations & Future Work

- **Simple retrieval strategy**: Relying solely on string matching lacks semantic-level relevance filtering. If samples with excessive domain gaps could be filtered out during the retrieval stage, Stage 1 performance might be further improved.
- **Underutilization of CLIP's text encoder for RAL**: The paper focuses on using retrieved data to improve the vision encoder, but retrieval augmentation on the text side (e.g., using retrieved text descriptions to improve classification prompts) could be a complementary direction.
- **Validation limited to CLIP ViT-B/16**: Larger backbones (ViT-L/14) or different pretraining strategies (such as SigLIP) were not tested; generalization capability still requires further verification.
- **Superficial CutMix augmentation**: Only CutMix was tested without exploring smarter augmentations (such as saliency-based CutMix or style-transfer-based augmentations).

## Related Work & Insights

- **vs CLAP (CVPR'24)**: CLAP is the strongest prior FSR method, which learns using a lightweight adapter on frozen encoders. SWAT does not limit learnable parameters and outperforms it by $>6\%$ through full-finetuning + RAL. The gap highlights the performance constraints imposed by "method constraints."
- **vs REAL (CVPR'24)**: REAL first introduced RAL to zero-shot recognition, learning a classifier through retrieved pretraining data. SWAT extends RAL to FSR, identifying new challenges not present in zero-shot settings (domain gap + imbalance) and proposing target solutions.
- **vs Long-Tailed Learning (Kang et al., 2020)**: SWAT's two-stage strategy (Stage 1 representation learning + Stage 2 classifier calibration) directly borrows from the decoupled paradigm in long-tailed learning, exhibiting a valuable cross-domain knowledge transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces RAL to FSR for the first time and uncovers new challenges; the finding that "simple full-finetuning outperforms all methods" is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 datasets, rigorous setup, comprehensive ablations, efficiency analysis, domain gap quantification—a gold standard for FSR papers.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, progressive discoveries, intuitive charts, and smooth narration.
- Value: ⭐⭐⭐⭐⭐ Shakes the paradigm of the FSR field; the proposed method is simple, effective, and widely applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] COBRA: COmBinatorial Retrieval Augmentation for Few-Shot Adaptation](cobra_combinatorial_retrieval_augmentation_for_few-shot_adaptation.md)
- [\[CVPR 2025\] EZSR: Event-based Zero-Shot Recognition](ezsr_event-based_zero-shot_recognition.md)
- [\[CVPR 2025\] VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents](vdocrag_retrieval-augmented_generation_over_visually-rich_documents.md)
- [\[CVPR 2025\] RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings](range_retrieval_augmented_neural_fields_for_multi-resolution_geo-embeddings.md)
- [\[NeurIPS 2025\] SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](../../NeurIPS2025/information_retrieval/secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)

</div>

<!-- RELATED:END -->
