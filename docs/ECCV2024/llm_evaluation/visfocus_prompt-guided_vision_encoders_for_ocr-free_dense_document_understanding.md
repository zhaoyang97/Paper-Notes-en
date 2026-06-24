---
title: >-
  [Paper Note] VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding
description: >-
  [ECCV 2024][LLM Evaluation][OCR-free] VisFocus proposes a prompt-guided vision encoding method for OCR-free document understanding. By directly injecting the user prompt into the patch merging layers (ViLMA layers) of the vision encoder, combined with a local masked prompt modeling (LMPM) pre-training task, the vision encoder learns to focus on prompt-related text regions, achieving state-of-the-art results among similarly-sized models across multiple document VQA benchmarks.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "OCR-free"
  - "Document Understanding"
  - "Prompt-aware Vision Encoding"
  - "Swin Transformer"
  - "Vision-Language Interaction"
date: 2026-05-08
content_hash: 603275215abb93b9
---

# VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding

**Conference**: ECCV 2024  
**arXiv**: [2407.12594](https://arxiv.org/abs/2407.12594)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: OCR-free, Document Understanding, Prompt-aware Vision Encoding, Swin Transformer, Vision-Language Interaction

## TL;DR
VisFocus proposes a prompt-guided vision encoding method for OCR-free document understanding. By directly injecting the user prompt into the patch merging layers (ViLMA layers) of the vision encoder, combined with a local masked prompt modeling (LMPM) pre-training task, the vision encoder learns to focus on prompt-related text regions, achieving state-of-the-art results among similarly-sized models across multiple document VQA benchmarks.

## Background & Motivation

Visual Document Understanding (VDU) aims to extract meaningful information from PDFs or document images, covering tasks such as DocVQA, ChartQA, and infographic understanding.

**Limitations of Prior Work**:
- **Drawbacks of OCR dependency**: Traditional methods rely on external OCR engines to extract text, which increases latency and computational costs, and OCR errors propagate to downstream models.
- **Limitations of OCR-free methods**: Existing OCR-free methods (e.g., Donut, Dessurt) bypass OCR but feed user queries only into the language model, meaning the vision encoder processes the entire document "indiscriminately." For dense documents, a huge number of visual tokens are occupied by blank spaces, charts, and irrelevant text, while the content truly relevant to the query does not receive sufficient focus.

**Key Challenge**: Visual feature extraction is independent of the user query, resulting in sub-optimal visual representations for specific questions. This is particularly severe in dense documents: the longer the document and the more irrelevant information it contains, the more significantly the model performance degrades.

**Key Insight**: Analogy to human document reading—humans do not read word-by-word; instead, they scan for keywords to find query-relevant regions first, and then read thoroughly. VisFocus injects this "selective scanning" capability into the vision encoder, allowing it to consider user prompts right during the downsampling process.

## Method

### Overall Architecture
VisFocus is based on a standard OCR-free architecture of "vision encoder + projection layer + language model". The vision encoder uses SwinV2, and the language model uses T5. The core improvements are: (1) replacing all patch merging layers of Swin with ViLMA layers to introduce prompt-visual interaction; (2) designing a localized masked prompt modeling (LMPM) pre-training task to guide the model to focus on relevant text. The training consists of three stages: LtR $\to$ LMPM $\to$ downstream fine-tuning.

In traditional methods, the vision encoder $\mathcal{M}_V(X)$ encodes the document image independently, and the prompt is only input to the language model. VisFocus changes this to $\mathcal{M}_V^p(\mathbf{p}, X)$ to produce prompt-aware visual features. The overall formula is:
$$\mathcal{M}(\mathbf{p}, X) = \mathcal{M}_L(\mathbf{p}, \mathcal{M}_V^p(\mathbf{p}, X))$$

### Key Designs

1. **ViLMA Layer (Vision-Language Merging Attention)**:
    - **Function**: Replaces the patch merging layers in SwinV2 to introduce prompt-visual interaction during the downsampling of visual features.
    - **Mechanism**: Inserts a multi-head cross-attention layer between the standard Swin patch merging 2×2 neighborhood concatenation ($L \times c \to L/4 \times 4c$) and the linear projection ($L/4 \times 4c \to L/4 \times 2c$). The visual features act as queries, and the prompt embeddings act as keys and values:
    $\tilde{F} = \hat{F} + \text{Norm}(\text{MHCA}(\hat{F}, \text{emb}(\mathbf{p})))$
    - **Prompt Encoding**: Uses a frozen language encoder to convert user prompts into context-aware embeddings.
    - **Positioning**: ViLMA layers are placed at the end of each stage of SwinV2 (4 stages in total), replacing all original patch merging layers.
    - **Design Motivation**: Patch merging is a key step in information aggregation. Introducing prompt signals here allows the feature compression process to selectively preserve prompt-relevant information.

2. **LMPM Pre-training Task (Localized Masked Prompt Modeling)**:
    - **Function**: Guides the vision encoder to learn to focus on specific text regions in the document that are related to the prompt.
    - **Mechanism**: Randomly samples a localized text span from the document's OCR text, masks portion of its tokens (borrowing T5's denoising objective), and feeds the masked span as the "prompt" into the ViLMA layers. The model is then required to predict the masked tokens. Since the masked tokens correspond to text at a specific location in the document, the model must learn to focus its visual attention on that region.
    - **Loss Function**: $\mathcal{L}_{LMPM} = \mathcal{L}_{CE}(\mathcal{M}(\mathbf{s}, X), Y_{LMPM})$
    - **Design Motivation**: Merely having the ViLMA architecture allows the vision encoder to "see" the prompt, but does not guarantee it will "exploit" the prompt to focus. LMPM forces the model to associate prompts with visual regions through explicit training signals.

3. **Prompt Dropout Strategy**:
    - **Function**: Prevents the language model from "compensating" for the vision encoder's lack of focus.
    - **Mechanism**: During the LMPM training phase, the prompt is concatenated into the language model's input with a probability $\rho$, and omitted otherwise. This forces the vision encoder to independently develop focusing capabilities rather than relying on the language model to complete the MLM task.
    - **Design Motivation**: If the prompt is always fed into the language model, the language model might directly leverage prompt information to complete the mask prediction, bypassing the need for the vision encoder to focus. Inspired by Dropout, this forces each component to learn independently.

4. **Three-Stage Training Pipeline**:
    - Stage I: **Learn to Read (LtR)** — Trains the standard Swin (non-ViLMA) on the IDL dataset to learn document reading by predicting all characters in the document in raster scan order.
    - Stage II: **LMPM** — Introduces ViLMA layers (randomly initialized) and trains the focusing capability on the localized masked prompt modeling task.
    - Stage III: **Fine-tuning** — Fine-tuning on downstream VQA tasks.

### Loss & Training
- LtR Stage: Standard cross-entropy loss, predicting document OCR text.
- LMPM Stage: Cross-entropy loss + Prompt Dropout (probability $\rho$).
- Fine-tuning Stage: Downstream task-specific loss (e.g., cross-entropy for VQA).
- Optimizer: AdamW + cosine annealing + warmup.
- Input Resolution: 1536×768 high resolution.
- Training Hardware: 8×A100 GPUs, bfloat precision.
- Model Variants: VisFocus-S (SwinV2-Small + T5-Small, 132M parameters), VisFocus-B (SwinV2-Small + T5-Base, 295M parameters).

## Key Experimental Results

### Main Results

| Method | Parameters | DocVQA (ANLS) | InfoVQA (ANLS) | ChartQA (RA) | OCR-VQA (EM) | AI2D (EM) |
|------|--------|---------------|----------------|--------------|--------------|-----------|
| Donut | 176M | 67.5 | 11.6 | 41.8 | 66.0 | - |
| Baseline-S | 110M | 67.0 | 24.7 | 49.3 | 66.6 | 42.7 |
| **VisFocus-S** | 132M | **68.6** (+1.6) | **28.5** (+3.8) | **53.0** (+3.7) | **67.3** (+0.7) | 42.6 |
| Pix2Struct-B | 282M | 72.1 | **38.2** | 56.0 | 69.4 | 40.9 |
| Baseline-B | 273M | 71.7 | 26.8 | 52.5 | 66.9 | 45.6 |
| **VisFocus-B** | 295M | **72.9** (+1.2) | 31.9 (+5.1) | **57.1** (+4.6) | **70.0** (+3.1) | **47.8** (+2.2) |

### Ablation Study

| Configuration | DocVQA (ANLS) | ChartQA (RA) | Description |
|------|---------------|--------------|------|
| Baseline-B | 70.9 | 52.5 | Without ViLMA, without LMPM |
| + ViLMA | 71.3 (+0.4) | 54.7 (+2.2) | Architectural improvements only |
| + LMPM | 71.8 (+0.5) | 55.7 (+1.0) | Additional focus pre-training |
| + Prompt Dropout (Full VisFocus-B) | **72.2** (+0.4) | **57.1** (+1.4) | Full method |

| ViLMA Layer Position | DocVQA | ChartQA | Description |
|------------|--------|---------|------|
| Baseline (None) | 70.9 | 52.5 | — |
| VF-Early [1,2] | 71.0 | 54.1 | Shallow layers only |
| VF-Mid [2,3] | 71.3 | 54.4 | Middle layers |
| VF-Late [3,4] | 71.6 | 55.3 | Deeper layers perform better |
| **VF-All [1,2,3,4]** | **72.2** | **57.1** | Replacing all is best |

| Prompt Injection Method | DocVQA | ChartQA | Description |
|---------------|--------|---------|------|
| Baseline (LM-only) | 70.9 | 52.5 | Standard method |
| Render (Pix2Struct style) | 70.6 | 52.2 | Rendering onto image degrades performance instead |
| **ViLMA (Ours)** | **71.3** | **54.7** | Semantic-level interaction is superior |

### Key Findings
- There is a **synergistic effect** between the ViLMA layer and the LMPM pre-training task: each brings individual improvements, but their combination achieves even greater gains.
- Deeper ViLMA layers contribute more, but replacing all layers yields the best performance (+4.6 on ChartQA).
- Prompt Dropout is particularly effective for ChartQA (+1.4), successfully forcing the vision encoder to learn focusing independently.
- The advantages of VisFocus **expand as document density increases**: a +0.7 gain for 400-word documents, and a +2.3 gain for 800-word documents.
- Rendering the prompt onto the image (Pix2Struct style) degrades performance in this experimental setup.
- The extra parameter scale introduced by ViLMA is very small compared to the total model parameters (by about an order of magnitude).
- Attention visualization shows that after LMPM training, the model can attend to words **semantically related** to the query (e.g., focusing on "under-ream" and "180 degrees" when queried about "diameter"), rather than just exact literal matches.

## Highlights & Insights
- Translating the human cognitive strategy of "selective reading" into model design is natural and well-motivated.
- The placement of ViLMA layers is well-targeted: patch merging is the bottleneck of information compression, which is the most effective place to introduce prompt signals.
- The LMPM pre-training task is cleverly designed: constructing prompts using the document's own text requires no extra annotation and naturally fits the document focusing task.
- The Prompt Dropout strategy is simple yet effective, solving the risk of the language model bypassing the vision encoder by drawing inspiration from standard Dropout.
- The document density analysis experiment strongly demonstrates the growing value of the focus mechanism on dense documents.

## Limitations & Future Work
- The focus capability is primarily tailored for text regions, showing limited effectiveness on documents containing infographics, charts, and images (where a gap with Pix2Struct still exists on InfoVQA).
- Has not explored prompt-aware pre-training tasks beyond text (such as guiding the vision encoder to focus on query-relevant visual regions in charts).
- The model size is still relatively small (<300M parameters), exhibiting an inherent disadvantage compared to multi-billion-parameter large VLMs.
- LMPM pre-training relies on OCR annotations as supervision signals, meaning it is still not fully decoupled from OCR in a sense.
- The three-stage training pipeline is quite complex; whether end-to-end training can achieve the same effect is worth exploring.

## Related Work & Insights
- The comparison with Pix2Struct is insightful: the latter renders prompts onto the image to achieve prompt-aware vision encoding, which limits semantic utilization. VisFocus interacts at the semantic level via cross-attention, yielding better results.
- Represents an evolution in methodology starting from Donut/Dessurt: moving from "vision encoding independent of prompts" to "prompt-guided vision encoding".
- Comparison with concurrent work QA-ViT: QA-ViT also injects prompts in self-attention layers but lacks the LMPM pre-training, leading to lower VDU performance, which verifies the necessity of the pre-training task.
- Insight: In multimodal models, introducing cross-modal interaction early on (rather than only doing backend fusion) is crucial for tasks requiring fine-grained localization.

## Rating
- Novelty: ⭐⭐⭐⭐ Injecting prompts into the patch merging layers of the vision encoder is an innovative idea, with a well-designed combination of ViLMA and LMPM.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 5 datasets with detailed ablations (components, positions, density analysis, injection methods).
- Writing Quality: ⭐⭐⭐⭐ Well-structured, intuitive diagrams, and cleanly elaborated motivation.
- Value: ⭐⭐⭐⭐ Provides an effective prompt-aware encoding paradigm for OCR-free document understanding, offering strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](../../ACL2025/llm_evaluation/kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)
- [\[ECCV 2024\] SIGMA: Sinkhorn-Guided Masked Video Modeling](sigma_sinkhorn-guided_masked_video_modeling.md)
- [\[ECCV 2024\] OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations](ogni-dc_robust_depth_completion_with_optimization-guided_neural_iterations.md)
- [\[ECCV 2024\] Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](../../ACL2025/llm_evaluation/mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)

</div>

<!-- RELATED:END -->
