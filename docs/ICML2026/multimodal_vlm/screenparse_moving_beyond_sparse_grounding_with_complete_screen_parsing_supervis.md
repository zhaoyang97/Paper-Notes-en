---
title: >-
  [Paper Note] ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision
description: >-
  [ICML 2026][Multimodal VLM][Screen parsing] Addressing the prevalent use of "sparse grounding" annotation and loss of full-screen structure in GUI agents…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Screen parsing"
  - "Computer-Use Agent"
  - "UI grounding"
  - "compact VLM"
  - "structure-aware loss"
date: 2026-05-08
content_hash: 45d66fc06a21e0ec
---

# ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision

**Conference**: ICML 2026  
**arXiv**: [2602.14276](https://arxiv.org/abs/2602.14276)  
**Code**: https://saidgurbuz.github.io/screenparse/  
**Area**: Multimodal VLM / GUI Agent / Dataset & Foundation Model  
**Keywords**: Screen parsing, Computer-Use Agent, UI grounding, compact VLM, structure-aware loss

## TL;DR
Addressing the prevalent use of "sparse grounding" annotation and loss of full-screen structure in GUI agents, this work introduces a fully automated Webshot pipeline to construct the dense screen parsing dataset ScreenParse, comprising 771K screenshots, 21M elements, and 55 classes. The authors train ScreenVLM, a model with only 316M parameters, to parse entire screens into ScreenTag structural sequences, outperforming 8B-scale foundation VLMs on both dense parsing and sparse grounding benchmarks while reducing latency to approximately $1/4$.

## Background & Motivation

**Background**: The core bottleneck for computer-use agents (CUA) is grounding—agents must first know what elements are on the screen, where they are, and what text they contain to click or input correctly. Mainstream CUA training datasets such as SeeClick, ScreenSpot, and Mind2Web use "action-driven" annotation: each instruction only labels the UI element being clicked, leaving all other screen elements unannotated.

**Limitations of Prior Work**: Sparse annotation allows models to learn shortcuts from instruction to single element, but the overall screen structure remains implicit; models fail on new layouts or applications. Datasets like GroundCUA, which are relatively complete, are small in scale (55k) and have few categories (8). Meanwhile, foundation VLMs (Qwen3-VL-8B, InternVL3) can extract elements zero-shot but are too large for edge deployment.

**Key Challenge**: The goal is "full-screen dense structural understanding," but manual dense annotation is prohibitively expensive; using DOM as ground truth is noisy (with many hidden/duplicate/invisible wrappers); and models must be small enough for edge deployment. These three objectives are mutually constraining.

**Goal**: (1) Automatically construct high-coverage, low-noise dense UI annotations; (2) Design a lightweight VLM architecture and sequence representation that can leverage such dense supervision; (3) Ensure that "dense supervision" is transferable to external grounding tasks and existing VLMs.

**Key Insight**: The authors leverage a previously overlooked "structural inductive bias"—treating the entire screen as a structured document, drawing on mature document-to-markup (DocTags, OTSL) approaches, and compressing UI screens into a sequence of tags with coordinates and categories.

**Core Idea**: Using Playwright rendering + DOM extraction + VLM refinement, tens of thousands of web pages are converted into 21M elements of dense screen supervision; a markup-style sequence ScreenTag plus structure-aware weighted CE enables a small VLM to parse entire screens into structured outputs.

## Method

### Overall Architecture
The paper is divided into two parts: **data-side Webshot pipeline** and **model-side ScreenVLM**. Webshot uniformly samples 1M pages from 45M URLs → Playwright renders full-page screenshots → DOM trees are extracted and filtered for visibility/overlap → Qwen3-VL-8B classifies each candidate element into one of 55 classes → VLM-as-a-judge scores each page for quality, discarding low-quality samples → yields 771K images / 21M elements, split 90/5/5. On the model side, ScreenVLM uses SigLIP-2 as the vision backbone to encode image patch tokens, which are projected and fed into a 165M Granite autoregressive decoder (initialized from Granite Docling, a document-to-markup model), outputting an XML-like ScreenTag sequence. Each element is formatted as `<tag> <x1> <y1> <x2> <y2> [text] [children] </tag>`, with coordinates normalized and quantized to a 0–500 grid.

### Key Designs

1. **Webshot Automated Dense Annotation Pipeline**:

    - **Function**: Automates the dense task of annotating "55 classes, all visible elements on the screen," requiring only a single machine.
    - **Mechanism**: DOM + Playwright are used to obtain candidate boxes (excluding degenerate, invisible, or near-duplicate nested wrappers), retaining container hierarchies (semantic containers like navbars, cards, modals are also annotated). Qwen3-VL-8B then re-predicts the class for each candidate using "full image + element crop + attributes." Finally, VLM-as-judge scores each page on coverage, false positives, duplication, and localization; samples below threshold are discarded.
    - **Design Motivation**: DOM annotation offers broad coverage but is noisy; pure VLM annotation is too costly. This cascade—DOM for candidates, VLM for reclassification, and page-level quality filtering—aims to approximate "complete + clean" annotation with zero human labor.

2. **ScreenTag: Compact Screen Structure Sequence**:

    - **Function**: Compresses a screenshot into a segment of autoregressively generable structured text.
    - **Mechanism**: Each element generates a nested sequence `<tag> <x1> <y1> <x2> <y2> [text] [children] </tag>`, with coordinates as discrete tokens in 0–500, and optional text/children. This representation is compact (less verbose than JSON), unambiguous to parse, and naturally fits LLM decoders' token-by-token generation.
    - **Design Motivation**: The authors deliberately reuse the inductive bias from document-to-markup pretraining (from Granite Docling), which is "markup-friendly with positional tags," enabling frictionless transfer to screen tasks, which are also "structured rectangles + text."

3. **Structure-Aware Weighted Cross-Entropy Loss**:

    - **Function**: Assigns higher weights to tag and coordinate tokens in the ScreenTag sequence, preventing long OCR text from dominating the gradient.
    - **Mechanism**: $\mathcal{L}(\theta) = -\sum_{t=1}^{T} w(y_t)\log p_\theta(y_t \mid y_{<t}, I)$, where $w(y_t) = \lambda_{\text{tag}}$ (if $y_t \in \mathcal{V}_{\text{tag}}$), $\lambda_{\text{loc}}$ (if $y_t \in \mathcal{V}_{\text{loc}}$), otherwise $=1$.
    - **Design Motivation**: A misaligned coordinate or misclassified tag invalidates the entire element, while a single text character error is less critical; OCR text dominates sequence length, causing standard CE to bias models toward "reading text but not knowing element locations." Weighted loss directly aligns the optimization objective with structural fidelity.

### Loss & Training
ScreenVLM is fine-tuned on ScreenParse train for 287,500 steps, using 16 H100 GPUs (2 nodes × 8 cards), effective batch size 64, sequence length capped at 8192 tokens. Grouped learning rates: multimodal projection layer $2.12\times 10^{-2}$, vision/language backbone $2\times 10^{-3}$.

## Key Experimental Results

### Main Results
Dense parsing comparison on ScreenParse test (PageIoU measures pixel-level coverage, Label PageIoU also requires class match).

| Model | Size | Page IoU | Label PageIoU | mAP@50 |
|-------|------|----------|---------------|--------|
| Qwen3-VL-8B-Instruct | 8B | 0.294 | – | – |
| InternVL3-2B | 2B | 0.111 | 0.030 | 0.000 |
| InternVL3-2B + ScreenParse | 2B | 0.509 (+0.398) | 0.174 | 0.072 |
| Qwen3-VL-2B + ScreenParse | 2B | 0.585 | 0.166 | 0.152 |
| **ScreenVLM (Ours)** | **316M** | **0.606** | **0.197** | **0.303** |
| RT-DETRv2 + ScreenParse | 43M | 0.600 | 0.172 | 0.362 |

ScreenVLM, with only 1/25 the parameters, more than doubles Qwen3-VL-8B's PageIoU; fine-tuning Qwen3-VL-2B and InternVL3-2B on ScreenParse also yields 0.36–0.40 PageIoU gains, demonstrating the transferability of this supervision.

### Ablation Study
Structure-aware weighted loss vs. standard CE.

| Setting | ScreenParse PageIoU | GroundCUA PageIoU | ScreenSpot-PC Recall |
|---------|---------------------|-------------------|---------------------|
| Full (StructureAware) | 0.606 | 0.251 | 0.222 |
| w/ CE only | 0.592 | 0.226 | 0.129 |
| Gain | +2.4% | +11.1% | **+72.1%** |

Efficiency (H100 + vLLM, 128 samples average):

| Model | Size (MB) | Latency (ms) | Throughput (s$^{-1}$) |
|-------|-----------|--------------|-----------------------|
| Qwen3-VL-2B | 4300 | $1289.1 \pm 251.7$ | 0.78 |
| InternVL3-2B | 4178 | $1267.3 \pm 187.9$ | 0.79 |
| **ScreenVLM** | **632** | $\mathbf{276.4 \pm 139.0}$ | **3.62** |

### Key Findings
- Structure-aware loss yields the largest gains in "out-of-distribution" and "few-element grounding" scenarios (ScreenSpot-PC Recall +72.1%), indicating it helps the model resist dilution of structure tokens by OCR text.
- ScreenParse supervision is "model-agnostic": fine-tuning entirely different families (InternVL3, Qwen3-VL, even YOLO/RT-DETR) all see improvements. This suggests dense screen supervision is to UI understanding what ImageNet is to vision.
- ScreenVLM achieves high PixCov (>0.83) but low Recall on ScreenSpot-PC/Mobile, indicating it "covers key pixels" but does not yet output tight element-level boxes—an artifact of web-only training, which the authors identify as a future direction.

## Highlights & Insights
- This work is a key step in shifting "computer-use data" from "action-driven sparse annotation" to "dense screen supervision." While GUI research has focused on grounding benchmarks, this work redefines supervision itself, setting a new pace.
- ScreenTag "documentizes" GUI screens, reusing mature inductive biases from document parsing, enabling a 316M model to learn structure from the outset. This "cross-domain markup representation transfer" is highly instructive for other structured perception tasks (circuit diagrams, forms, maps).
- Using VLMs as both "annotation refiner" and "judge," combined with DOM for candidate extraction, enables automatic bootstrapping from weak to strong annotation; this approach can be applied to other "rendered source + DOM/SVG parsable" domains.

## Limitations & Future Work
- Data is entirely web-based. PC and mobile app UIs differ from web; experiments show Recall is significantly lower on ScreenSpot-PC/Mobile than Web. Broader coverage requires extending to desktop/mobile rendering.
- VLM-judge thresholds require manual calibration, and "high quality" judgments are still influenced by backbone bias.
- ScreenTag is a nested sequence, currently capped at 8192 tokens; ultra-large screens (e.g., 4K long screenshots) may still be truncated.
- The paper does not connect ScreenVLM to a true end-to-end agent to demonstrate downstream benefits from "dense parsing → action," which is an obvious next step.

## Related Work & Insights
- **vs SeeClick / ScreenSpot**: These use sparse grounding (one instruction, one element per image); this work pursues "full-screen dense" annotation and shows such supervision also benefits the former.
- **vs GroundCUA**: GroundCUA is also densely annotated but only 55k samples and 8 classes; ScreenParse is 771k and 55 classes, an order of magnitude larger in both scale and categories.
- **vs OmniParser**: OmniParser is a detector-style YOLO parser, strong in localization but lacking language-aligned structured output; ScreenVLM outputs markup-style structures, directly consumable by downstream LLM agents.
- **vs Granite Docling / SmolDocling**: These are document-to-markup VLMs; this work adapts them to the UI domain, empirically validating that "structured-markup pretraining" is a strong starting point for UI perception.

## Rating
- Novelty: ⭐⭐⭐⭐ Solidly advances the GUI data paradigm from "sparse→dense," though most technical components are recombinations of existing modules
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple VLM/detector families + 3 benchmarks + comprehensive loss/efficiency ablations
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-matched figures and tables, some key designs in the appendix
- Value: ⭐⭐⭐⭐⭐ Dataset + small model both open-sourced, foundational contribution to the GUI agent community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](../../CVPR2026/multimodal_vlm/efficient_document_parsing_via_parallel_token_prediction.md)
- [\[ACL 2026\] What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning](../../ACL2026/multimodal_vlm/what39s_missing_in_screen-to-action_towards_a_ui-in-the-loop_paradigm_for_multim.md)
- [\[ICLR 2026\] Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](../../ICLR2026/multimodal_vlm/grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)
- [\[ICML 2026\] Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy](less_precise_can_be_more_reliable_a_systematic_evaluation_of_quantizations_impac.md)

</div>

<!-- RELATED:END -->
