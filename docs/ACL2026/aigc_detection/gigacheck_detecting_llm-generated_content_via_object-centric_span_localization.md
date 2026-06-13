---
title: >-
  [Paper Note] GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization
description: >-
  [ACL 2026][AIGC Detection][LLM-generated text detection] GigaCheck is proposed as a dual-strategy framework: it utilizes a fine-tuned LLM for document-level classification and innovatively treats AI-generated text spans…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "LLM-generated text detection"
  - "object detection paradigm"
  - "DETR"
  - "text span localization"
  - "human-AI collaborative text"
date: 2026-05-08
content_hash: 2c3804bd01bc9262
---

# GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization

**Conference**: ACL 2026 Findings  
**arXiv**: [2410.23728](https://arxiv.org/abs/2410.23728)  
**Code**: [GitHub](https://github.com/ai-forever/gigacheck)  
**Area**: Object Detection  
**Keywords**: LLM-generated text detection, object detection paradigm, DETR, text span localization, human-AI collaborative text

## TL;DR

GigaCheck is proposed as a dual-strategy framework: it utilizes a fine-tuned LLM for document-level classification and innovatively treats AI-generated text spans as "objects," implementing end-to-end character-level localization via a DETR-like architecture.

## Background & Motivation

**Background**: With the rapid improvement in the quality of LLM-generated content, AI-generated text has become difficult to distinguish from human-written text in many scenarios. Detecting AI-generated content has become a critical requirement for combating misinformation, academic fraud, and the spread of spam.

**Limitations of Prior Work**: (1) Document-level detection methods lack reliability on human-AI collaborative text (partially human-written and partially AI-generated); (2) Existing span-level detection methods are primarily based on token-level sequence labeling (BIO), requiring manual post-processing to aggregate tokens into continuous spans, and are limited by sentence boundaries and fixed granularity; (3) The development of detection methods lags behind the progress of generative models.

**Key Challenge**: There is a need to simultaneously solve document-level classification ("Is this article AI-generated?") and span-level localization ("Which specific segments are AI-generated?"), where representations should be shared between both tasks to improve efficiency.

**Goal**: Propose a unified framework capable of both high-precision document-level detection and precise localization of AI-generated text spans.

**Key Insight**: Analogizing AI-generated text spans to "objects" in images, the mature DETR architecture from visual object detection is utilized for end-to-end 1D span detection, transferring the robustness of visual detection to the NLP domain.

## Method

### Overall Architecture

GigaCheck employs a shared backbone + dual-head architecture: (1) Unified Backbone: A LoRA fine-tuned Mistral-7B provides text embeddings; (2) DETR Head: Treats the embedding sequence as a 1D feature map and uses a DN-DAB-DETR architecture to detect AI-generated spans; (3) Classification Head: Uses the hidden state of the last EOS token followed by an MLP for document-level binary classification. Both heads can be used independently while sharing the fine-tuned backbone.

### Key Designs

1. **Object-Centric Span Localization**:

    - **Function**: Performs end-to-end detection and localization of continuous AI-generated text spans as 1D "objects."
    - **Mechanism**: Token embeddings $\mathbf{E}$ are obtained from the fine-tuned LLM and processed via linear projection and a Transformer encoder to get contextual features $\mathbf{R}$. $N$ learnable anchor queries $(c, w)$ (center and width) are iteratively refined through a Transformer decoder, with each layer predicting offsets $(\Delta c, \Delta w)$. The final output is a triplet $(c, w, p)$—center position, width, and confidence—where all values are normalized to a character-level interval of $[0,1]$.
    - **Design Motivation**: Sequence labeling methods requires manual post-processing to aggregate tokens into spans, whereas DETR directly regresses continuous intervals, eliminating the reliance on heuristic post-processing. Character-level rather than token-level localization is more flexible and independent of the tokenizer.

2. **Unified Text-Representation Backbone**:

    - **Function**: Provides shared, high-quality text embeddings for both detection and classification tasks.
    - **Mechanism**: Use LoRA to fine-tune Mistral-7B, training on proxy tasks: a three-class classification (human/AI/collaborative) for frozen feature extraction (for DETR), and binary classification (human/AI) joint-training with the classification head. LoRA keeps pre-trained weights frozen and only trains low-rank matrices.
    - **Design Motivation**: Datasets are typically small; LoRA generalizes better on small data and trains faster. The shared backbone validates the generalization capability of the embeddings.

3. **DN-DAB-DETR Adaption**:

    - **Function**: Stabilizes training and improves localization precision.
    - **Mechanism**: Adopts DAB-DETR's learnable anchor boxes as positional queries and DN-DETR's denoising training strategy (simultaneously training learnable queries and noisy GT queries). Hungarian matching is used for prediction-GT pairing.
    - **Design Motivation**: DN-DAB-DETR has demonstrated the best localization accuracy and training stability in visual detection, outperforming DAB-DETR, Deformable DETR, and CO-DETR.

### Loss & Training

The training loss is a weighted sum of L1 + gIoU + Focal Loss, calculated for both Hungarian-matched predictions and denoising GT queries. The classification head uses binary cross-entropy. During DETR training, the backbone is frozen; during classification head training, the backbone is trainable.

## Key Experimental Results

### Main Results (Classification)

| Dataset | GigaCheck (Mistral-7B) | Prev. SOTA | Description |
|--------|---------------------|----------|------|
| TuringBench (FAIR) | High Precision | RoBERTa-based methods | Unified backbone achieves strong classification performance |
| TweepFake | High Precision | - | Validation in the tweet domain |
| MAGE | High Precision | - | Large-scale validation across multiple generators and domains |

### Main Results (Span Detection)

| Dataset | GigaCheck (DETR) | Previous Methods | Description |
|--------|---------------|---------|------|
| RoFT | Strong Localization Accuracy | Sequence labeling methods | Single-boundary scenarios |
| RoFT-ChatGPT | Strong Localization Accuracy | - | ChatGPT generation scenarios |
| TriBERT | Strong Localization Accuracy | - | Multi-boundary (1-3) scenarios |

### Key Findings
- The DETR architecture can be successfully generalized from visual space to text space, proving the feasibility of the object detection paradigm in NLP.
- The same fine-tuned backbone performs excellently on both classification and detection tasks, verifying that the learned embeddings possess strong generalization capabilities.
- End-to-end span detection eliminates the need for heuristic post-processing required in sequence labeling methods.
- LoRA-based parameter-efficient fine-tuning is particularly effective on small datasets.

## Highlights & Insights
- **Novelty**: Redefining text span detection as a 1D object detection problem is an elegant and effective cross-domain transfer.
- **Unified Framework**: A single fine-tuned backbone serves both detection and classification, which is not only efficient but also validates the universality of the embeddings.
- **End-to-End Design**: DETR directly outputs character-level intervals, avoiding the cumbersome process of BIO labeling and post-processing.
- **Model Agnostic**: The backbone can be replaced with any decoder-only LLM, giving the framework good scalability.
- **Value**: The complete code is publicly released, promoting reproducibility.

## Limitations & Future Work
- Evaluation is currently limited to English text; multilingual adaptation is an important future direction.
- The generators in the detection datasets are primarily earlier models (GPT-2/3, CTRL); the detection effectiveness against the latest LLMs is unknown.
- The number of queries $N$ in DETR must be preset based on the dataset and cannot adapt dynamically.
- Insufficient analysis of robustness against adversarial attacks (e.g., paraphrasing, watermark removal).
- Future work could explore multi-granularity detection (joint detection at paragraph, sentence, and word levels).

## Related Work & Insights
- **vs Sci-SpanDet**: Sci-SpanDet relies on IMRaD document structures for scientific paper detection; GigaCheck is domain-agnostic and applicable to any text.
- **vs Sequence Labeling (BIO)**: Sequence labeling requires manual aggregation of tokens into spans; GigaCheck directly regresses continuous intervals.
- **vs Statistical Methods (DetectGPT, etc.)**: Statistical methods require access to the probability distribution of the target LLM; GigaCheck has no such requirement.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of DETR to text span localization; significant paradigm innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Double validation across 3 classification and 3 detection benchmarks, though lacks testing on the latest LLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear architecture diagrams, rigorous method descriptions, and appropriate cross-modal analogies.
- Value: ⭐⭐⭐⭐ Provides a new technical route for AI-generated text detection; open-sourcing enhances impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](../../NeurIPS2025/aigc_detection/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[ACL 2026\] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)

</div>

<!-- RELATED:END -->
