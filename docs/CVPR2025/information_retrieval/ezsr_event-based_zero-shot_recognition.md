---
title: >-
  [Paper Note] EZSR: Event-based Zero-Shot Recognition
description: >-
  [CVPR 2025][Information Retrieval & RAG][Event camera] This paper proposes the EZSR framework for zero-shot object recognition in event camera data. By utilizing a scalar-wise modulation strategy, it addresses the semantic misalignment between event embeddings and CLIP text embeddings. It overcomes training data scarcity through large-scale event data synthesis from static RGB images, achieving a 47.84% zero-shot accuracy on N-ImageNet with a ViT-B/16 backbone.
tags:
  - "CVPR 2025"
  - "Information Retrieval & RAG"
  - "Event camera"
  - "Zero-shot recognition"
  - "CLIP distillation"
  - "Scalar modulation"
  - "Data synthesis"
date: 2026-05-08
content_hash: 290aa5c01901af1a
---

# EZSR: Event-based Zero-Shot Recognition

**Conference**: CVPR 2025  
**arXiv**: [2407.21616](https://arxiv.org/abs/2407.21616)  
**Code**: [https://yan98.github.io/EZSR/](https://yan98.github.io/EZSR/)  
**Area**: Information Retrieval  
**Keywords**: Event camera, Zero-shot recognition, CLIP distillation, Scalar modulation, Data synthesis

## TL;DR
This paper proposes the EZSR framework for zero-shot object recognition in event camera data. By utilizing a scalar-wise modulation strategy, it addresses the semantic misalignment between event embeddings and CLIP text embeddings. It overcomes training data scarcity through large-scale event data synthesis from static RGB images, achieving a 47.84% zero-shot accuracy on N-ImageNet with a ViT-B/16 backbone.

## Background & Motivation

**Background**: Event cameras capture pixel-level brightness changes asynchronously, offering benefits like high temporal resolution, lack of motion blur, and low power consumption. Extending CLIP's zero-shot capability to the event domain is a recent research hotspot. Two main approaches exist: (1) reconstruction methods, which reconstruct grayscale images from event data and feed them to the CLIP image encoder; and (2) contrastive learning methods, which distill the event encoder using paired event-RGB data.

**Limitations of Prior Work**: Reconstruction methods yield poor zero-shot performance due to low reconstruction quality and error accumulation. Contrastive learning methods have a fundamental theoretical flaw: the spatial sparsity of event data causes excessive similarity between event embeddings, making the negative-sample repulsion objective of contrastive learning ineffective at distinguishing them. Crucially, optimizing the InfoNCE objective to align event embeddings with RGB embeddings does not imply that event embeddings automatically align with CLIP's text embeddings (Lemma 1). Furthermore, paired event-RGB datasets are scarce, restricting existing methods to training and testing on the same dataset.

**Key Challenge**: The InfoNCE loss in contrastive learning not only optimizes the similarity of paired data but also repels non-paired data. However, because the intrinsic similarity among event embeddings is high, the repulsion operation introduces degrees of freedom into the embedding space that are misaligned with the CLIP text space. Thus, even if event-RGB alignment succeeds, event-text alignment can still fail.

**Goal**: To design an event encoder capable of direct zero-shot recognition from event data without relying on reconstruction networks, while simultaneously addressing the training data scarcity problem.

**Key Insight**: To replace the repulsion operation in contrastive learning with scalar-wise (dimension-wise) modulation, directly aligning each dimension of the event embedding to its corresponding dimension in the RGB embedding to eliminate semantic misalignment caused by excessive degrees of freedom. Meanwhile, event data are synthesized from static RGB images using simple affine transforms to scale up the training set on a large scale.

**Core Idea**: To make each scalar dimension produced by the event encoder directly approximate the corresponding dimension of the CLIP image encoder, thereby inheriting the pre-aligned RGB-text relationship. Since RGB and text are already aligned in CLIP, if events and RGB are aligned in every dimension, event and text representations will naturally align.

## Method

### Overall Architecture
Given paired event data and RGB images, embeddings are extracted using a trainable event encoder $f^{evt}$ and a frozen CLIP image encoder $f^{img}$. A scalar-wise modulation loss is employed to align the event and RGB embeddings dimension-by-dimension. During inference, the event encoder replaces the CLIP image encoder to perform zero-shot classification alongside the CLIP text encoder. Training data are obtained by synthesizing events from static RGB images.

### Key Designs

1. **Scalar-wise Modulation**:

    - **Function**: Aligns the event and RGB embeddings dimension-by-dimension to eliminate the semantic misalignment introduced by contrastive learning.
    - **Mechanism**: Instead of applying the negative-sample repulsion objective of InfoNCE, the method directly optimizes a scalar-wise metric of $\mathcal{L}_{mod} = \| \hat{\mathbf{x}}^{evt} - \mathbf{x}^{img} \|$. Specifically, it calculates the alignment loss between the event and RGB embeddings for each dimension $d$, allowing the network to adaptively mine semantic correspondences across individual dimensions. This differs from optimizing vector-level cosine similarity alone, which only constrains orientation rather than separate alignment of each dimension.
    - **Design Motivation**: Lemma 1 proves that even if contrastive learning successfully minimizes the InfoNCE loss, it cannot guarantee event-text alignment. Scalar-wise modulation directly eliminates the redundant degrees of freedom in the embedding space. If each scalar dimension of the event embedding equals that of the RGB embedding, the similarity to the text embedding will also be consistent.

2. **k-NN Embedding Translation (Remark 1)**:

    - **Function**: Further mitigates event-text embedding misalignment during inference.
    - **Mechanism**: Maintains a pre-computed pool of RGB image embeddings. During inference, the system retrieves the $k$-nearest neighbor RGB embeddings for a given event embedding, computing a weighted average based on similarity to yield a translated event embedding $\tilde{\mathbf{x}}^{evt}$, which is then matched with text embeddings. This effectively performs a "relay calibration" using the RGB embedding pool.
    - **Design Motivation**: Even if alignment is imperfect during training, referencing semantically similar RGB embeddings in the pool can further calibrate and correct the event embeddings.

3. **Static Image Event Synthesis**:

    - **Function**: Generates paired event-RGB training data on a large scale to overcome standard data scarcity constraints.
    - **Mechanism**: Applies random affine transformations (translation, rotation, scaling) to static RGB images to generate image sequences through interpolation, which are then differenced to produce event data. Compared to traditional methods that require video files and pre-trained frame interpolation networks, this approach is computationally cheap and introduces greater diversity.
    - **Design Motivation**: In zero-shot recognition scenarios, event data typically capture short-duration intervals (on the millisecond scale) with approximate linear motions. A simple affine transformation effectively simulates this property. Scale-up synthesis allows the model to display robust parameter and data scalability.

### Loss & Training
The final loss is primarily the scalar-wise modulation loss, optionally combined with the InfoNCE loss. The CLIP image encoder is fully frozen, and only the event encoder is trained. The event encoder is initialized using the weights of the CLIP image encoder. Training data are synthesized event-RGB pairs.

## Key Experimental Results

### Main Results

| Dataset | EZSR (ViT-B/16) | Prev. Best Zero-Shot | Prev. Best Supervised |
|--------|-----------------|---------------|---------------|
| N-ImageNet | 47.84% | ~35% (ExACT) | ~60% (Supervised) |
| N-Caltech101 | Outperforms SOTA | — | — |
| CIFAR10-DVS | Outperforms SOTA | — | — |

### Ablation Study (N-ImageNet)

| Configuration | Accuracy | Description |
|------|--------|------|
| InfoNCE baseline only | 9.57% | Severe event-text misalignment |
| + Remark 1 (k-NN Translation) | 43.48% | Huge boost from k-NN calibration |
| Scalar-wise modulation only | 47.80% | Core contribution |
| Scalar-wise modulation + Remark 1 | 48.63% | Further marginal improvement |
| Full combination | 48.86% | Optimal |

### Key Findings
- The InfoNCE baseline yields only 9.57%, demonstrating the fundamental flaw of contrastive learning in the event domain: event-text semantic misalignment.
- Scalar-wise modulation alone achieves 47.80% (a gain of +38.23 percentage points), proving to be the core innovation.
- The model exhibits excellent scalability: increasing both parameters and synthetic data continuously boosts performance.
- When evaluated across 9 standard event datasets, the zero-shot performance even surpasses some dataset-specific supervised methods.

## Highlights & Insights
- **Theoretical Analysis of Contrastive Learning Failures in the Event Domain**: Lemma 1 rigorously demonstrates that InfoNCE minimization does not guarantee zero-shot event-text alignment. This provides a solid theoretical foundation for understanding the limitations of cross-domain distillation.
- **Simplicity and Elegance of Scalar-wise Modulation**: Requires no complex changes to the network architecture; simply modifying the loss function from vector-level to scalar-level results in an absolute boost of 38+ percentage points, exemplifying the importance of targeting the root problem directly.
- **Event Data Synthesis from Static Images**: The synthesis method is remarkably simple (affine transformations combined with differencing), yet it effectively mitigates the lack of training data in the event domain, unlocking the use of any static RGB image dataset.

## Limitations & Future Work
- The proposed synthetic event data generation utilizes simple affine transformations, which fail to capture complex motion patterns and realistic noise characteristics present in the physical world.
- The $k$-NN embedding translation relies on maintaining a large storage pool of RGB embedding references, increasing storage and computational overhead at inference time.
- The achieved zero-shot performance (48.86%) still lags significantly behind dedicated supervised approaches (~60%).
- Performance on event-based tasks requiring longer temporal sequences, such as action recognition, remains uninvestigated.

## Related Work & Insights
- **vs ExACT**: While ExACT also performs zero-shot recognition in the event domain, it relies on contrastive learning and is limited to training and testing on the same dataset. EZSR fundamentally enhances performance with synthetic data and scalar-wise modulation.
- **vs Reconstruction Methods (e.g., E2VID + CLIP)**: Reconstruction-based techniques introduce additional pipeline networks and suffer from accumulated errors. EZSR directly trains an event encoder in a cleaner, end-to-end fashion.
- **vs Direct CLIP Application to Event Frames**: Event frames are highly sparse and show a significant domain gap compared to RGB. Applying CLIP directly yields poor results. EZSR bridges this gap by training a dedicated event encoder.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical analysis of contrastive learning flaws and the scalar-wise modulation scheme are insightful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 9 datasets, with clear ablations and persuasive scalability experiments.
- Writing Quality: ⭐⭐⭐⭐ Well-integrated theory and experiments, with a clear narrative structure that builds up the methodology step-by-step.
- Value: ⭐⭐⭐⭐ Provides a strong baseline for event-camera zero-shot learning, accompanied by a highly practical data-synthesis pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning](few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning.md)
- [\[ICLR 2026\] ZeroGR: A Generalizable and Scalable Framework for Zero-Shot Generative Retrieval](../../ICLR2026/information_retrieval/zerogr_a_generalizable_and_scalable_framework_for_zero-shot_generative_retrieval.md)
- [\[ICML 2026\] BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs](../../ICML2026/information_retrieval/blitzrank_principled_zero-shot_ranking_agents_with_tournament_graphs.md)
- [\[CVPR 2025\] COBRA: COmBinatorial Retrieval Augmentation for Few-Shot Adaptation](cobra_combinatorial_retrieval_augmentation_for_few-shot_adaptation.md)
- [\[ICLR 2026\] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)

</div>

<!-- RELATED:END -->
