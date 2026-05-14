---
title: >-
  [Paper Note] GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization
description: >-
  [ACL 2026][Object Detection][LLM-generated text detection] GigaCheck is proposed as a dual-strategy framework: document-level classification via fine-tuned LLM…
tags:
  - "ACL 2026"
  - "Object Detection"
  - "LLM-generated text detection"
  - "object detection paradigm"
  - "DETR"
  - "text span localization"
  - "human-machine collaborative text"
date: 2026-05-08
content_hash: 83ff1a4f6d71201f
---

# GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization

**Conference**: ACL 2026
**arXiv**: [2410.23728](https://arxiv.org/abs/2410.23728)
**Code**: [GitHub](https://github.com/ai-forever/gigacheck)
**Area**: Object Detection
**Keywords**: LLM-generated text detection, object detection paradigm, DETR, text span localization, human-machine collaborative text

## TL;DR

GigaCheck is proposed as a dual-strategy framework: document-level classification via fine-tuned LLM, and span-level detection that innovatively treats AI-generated text spans as "objects," employing a DETR-like architecture for end-to-end character-level localization.

## Background & Motivation

**Background**: With the rapid improvement in LLM-generated content quality, AI-generated text has become increasingly indistinguishable from human-written text in many contexts. Detecting AI-generated content has emerged as a critical need for combating misinformation, academic fraud, and spam proliferation.

**Limitations of Prior Work**: (1) Document-level detection methods are insufficiently reliable on human-machine collaborative text (partially human-written, partially machine-written); (2) Existing span-level detection methods primarily rely on token-level sequence labeling (BIO), requiring manual post-processing to aggregate tokens into contiguous spans, and are constrained by sentence boundaries and fixed granularity; (3) Detection methods lag behind the advancement of generative models.

**Key Challenge**: There is a need to simultaneously address document-level classification ("Is this article AI-generated?") and span-level localization ("Which specific passages are AI-generated?"), with shared representations between the two tasks to improve efficiency.

**Goal**: To propose a unified framework capable of both high-accuracy document-level detection and precise localization of AI-generated text spans.

**Core Idea**: AI-generated text spans are analogized to "objects" in images, leveraging the well-established DETR architecture from visual object detection for end-to-end 1D span detection, transferring the robustness of visual detection to the text domain.

## Method

### Overall Architecture

GigaCheck adopts a shared backbone with dual-head architecture: (1) **Unified Backbone**: LoRA-fine-tuned Mistral-7B provides text embeddings; (2) **DETR Head**: treats the embedding sequence as a 1D feature map and uses DN-DAB-DETR architecture to detect AI-generated spans; (3) **Classification Head**: applies an MLP on the hidden state of the final EOS token for document-level binary classification. Both heads can be used independently while sharing the fine-tuned backbone.

### Key Designs

1. **Object-Centric Span Localization**:

    - **Function**: Treats contiguous AI-generated text spans as 1D "objects" for end-to-end detection and localization.
    - **Mechanism**: Token embeddings $\mathbf{E}$ are obtained from the fine-tuned LLM, then projected linearly and passed through a Transformer encoder to yield contextual features $\mathbf{R}$. $N$ learnable anchor queries $(c, w)$ (center and width) are iteratively refined through the Transformer decoder, predicting offsets $(\Delta c, \Delta w)$ at each layer. The final output is a triple $(c, w, p)$—center position, width, and confidence—all normalized to a character-level interval of $[0,1]$.
    - **Design Motivation**: Sequence labeling methods require manual post-processing to aggregate tokens into spans, whereas DETR directly regresses contiguous intervals, eliminating the need for heuristic post-processing. Character-level localization, as opposed to token-level, is more flexible and tokenizer-agnostic.

2. **Unified Text-Representation Backbone**:

    - **Function**: Provides shared high-quality text embeddings for both detection and classification tasks.
    - **Mechanism**: Mistral-7B is fine-tuned with LoRA using a proxy task: three-class classification (human-written / machine-written / collaborative) for frozen feature extraction (used by DETR), and two-class classification (human-written / machine-written) jointly trained with the classification head. LoRA keeps the pre-trained weights frozen, training only the low-rank matrices.
    - **Design Motivation**: Datasets are typically small; LoRA generalizes better on small data and trains faster. The shared backbone validates the generalizability of the learned embeddings.

3. **DN-DAB-DETR Adaptation**:

    - **Function**: Stabilizes training and improves localization accuracy.
    - **Mechanism**: Employs DAB-DETR's learnable anchor boxes as positional queries and DN-DETR's denoising training strategy (simultaneously training learnable queries and noise-augmented GT queries). Hungarian matching is used for prediction–GT assignment.
    - **Design Motivation**: DN-DAB-DETR demonstrates superior localization accuracy and training stability in visual detection, outperforming DAB-DETR, Deformable DETR, and CO-DETR.

### Loss & Training

The training loss is a weighted sum of L1, gIoU, and Focal Loss, computed separately for Hungarian-matched predictions and denoising GT queries. The classification head uses binary cross-entropy. During DETR training, the backbone is frozen; during classification head training, the backbone is trainable.

## Key Experimental Results

### Main Results (Classification)

| Dataset | GigaCheck (Mistral-7B) | Prev. SOTA | Notes |
|--------|---------------------|----------|------|
| TuringBench (FAIR) | High accuracy | RoBERTa-based methods | Strong classification with unified backbone alone |
| TweepFake | High accuracy | — | Validated on tweet domain |
| MAGE | High accuracy | — | Large-scale validation across multiple generators and domains |

### Main Results (Span Detection)

| Dataset | GigaCheck (DETR) | Prior Methods | Notes |
|--------|---------------|---------|------|
| RoFT | Strong localization | Sequence labeling methods | Single-boundary scenario |
| RoFT-ChatGPT | Strong localization | — | ChatGPT generation scenario |
| TriBERT | Strong localization | — | Multi-boundary (1–3) scenario |

### Key Findings
- The DETR architecture can be successfully generalized from the visual domain to the text domain, demonstrating the feasibility of the object detection paradigm in NLP.
- The same fine-tuned backbone performs strongly on both classification and detection tasks, validating the strong generalizability of the learned embeddings.
- End-to-end span detection eliminates the need for heuristic post-processing in sequence labeling approaches.
- LoRA-based parameter-efficient fine-tuning is particularly effective on small datasets.

## Highlights & Insights
- **Paradigm Innovation**: Reframing text span detection as a 1D object detection problem represents an elegant and effective cross-domain transfer.
- **Unified Framework**: A single fine-tuned backbone serves both detection and classification, achieving efficiency while validating embedding generality.
- **End-to-End Design**: DETR directly outputs character-level intervals, avoiding the cumbersome pipeline of BIO labeling and post-processing.
- **Model Agnosticism**: The backbone can be replaced with any decoder-based LLM, conferring good extensibility to the framework.
- **Open-Source Contribution**: Full code is publicly released, promoting reproducibility.

## Limitations & Future Work
- Evaluation is currently limited to English text; multilingual adaptation is an important future direction.
- Generators in the detection datasets are primarily earlier models (GPT-2/3, CTRL); detection performance on the latest LLMs remains unknown.
- The number of DETR queries $N$ must be preset per dataset and cannot adapt dynamically.
- Robustness analysis against adversarial attacks (e.g., paraphrasing, watermark removal) is insufficient.
- Future work may explore multi-granularity detection (joint paragraph-level, sentence-level, and word-level detection).

## Related Work & Insights
- **vs. Sci-SpanDet**: Sci-SpanDet relies on IMRaD document structure for scientific paper detection; GigaCheck is domain-agnostic and applicable to arbitrary text.
- **vs. Sequence Labeling Methods (BIO)**: Sequence labeling requires manual aggregation of tokens into spans; GigaCheck directly regresses contiguous intervals.
- **vs. Statistical Methods (DetectGPT, etc.)**: Statistical methods require access to the probability distribution of the target LLM; GigaCheck has no such requirement.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First application of DETR to text span localization; the paradigm innovation carries substantial significance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Dual validation across 3 classification and 3 detection benchmarks, though testing on the latest LLMs is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear, method descriptions are rigorous, and the cross-modal analogy is apt.
- **Value**: ⭐⭐⭐⭐ Provides a new technical direction for AI-generated text detection; open-source release enhances impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](../../CVPR2026/object_detection/span_spatial-projection_alignment_for_monocular_3d_object_detection.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](../../CVPR2026/object_detection/mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](../../CVPR2026/object_detection/detecting_unknown_objects_via_energy-based_separation.md)
- [\[CVPR 2026\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](../../CVPR2026/object_detection/show_dont_tell_detecting_novel_objects_by_watching.md)
- [\[NeurIPS 2025\] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](../../NeurIPS2025/object_detection/detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)

</div>

<!-- RELATED:END -->
