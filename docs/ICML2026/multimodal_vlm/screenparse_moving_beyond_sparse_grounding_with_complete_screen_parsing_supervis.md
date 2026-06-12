---
title: >-
  [Paper Note] ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision
description: >-
  [ICML 2026][Multimodal VLM][Screen Parsing] Addressing the issue of "sparse grounding" annotations in GUI agents that lose full-screen structure, this paper develops an automated Webshot pipeline to construct ScreenParse…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Screen Parsing"
  - "Computer-Use Agent"
  - "UI grounding"
  - "Compact VLM"
  - "Structure-aware Loss"
date: 2026-05-08
content_hash: e7d9ac914995a03c
---

# ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision

**Conference**: ICML 2026  
**arXiv**: [2602.14276](https://arxiv.org/abs/2602.14276)  
**Code**: https://saidgurbuz.github.io/screenparse/  
**Area**: Multimodal VLM / GUI Agent / Datasets & Foundation Models  
**Keywords**: Screen Parsing, Computer-Use Agent, UI grounding, Compact VLM, Structure-aware Loss

## TL;DR
Addressing the issue of "sparse grounding" annotations in GUI agents that lose full-screen structure, this paper develops an automated Webshot pipeline to construct ScreenParse, a dense screen parsing dataset with 771K screenshots, 21M elements, and 55 classes. The authors train ScreenVLM, a model with only 316M parameters, to parse full screens into ScreenTag structural sequences. It outperforms 8B-scale foundation VLMs on dense parsing and sparse grounding benchmarks while reducing latency to $\sim 1/4$.

## Background & Motivation

**Background**: The core bottleneck for computer-use agents (CUA) is grounding—for an agent to click or type correctly, it must first recognize what elements are on the screen, where they are, and what their text is. Current mainstream CUA training data, such as SeeClick, ScreenSpot, and Mind2Web, utilize "action-driven" annotations: each step only labels the specific UI element being clicked, leaving all other elements on the screen unannotated.

**Limitations of Prior Work**: Sparse annotation allows models to learn shortcuts from "instruction to single element," but the global structure of the screen remains implicit, leading to failure on new layouts or applications. Meanwhile, relatively complete datasets like GroundCUA are small in scale (55k) and have few categories (8 classes). On the other hand, foundation VLMs (e.g., Qwen3-VL-8B, InternVL3) can perform zero-shot element extraction, but their massive size prevents edge deployment.

**Key Challenge**: The requirement is for "full-screen dense structural understanding," but manual dense annotation is prohibitively expensive; using raw DOM as ground truth is extremely noisy (containing many hidden/duplicate/invisible wrappers); and the model must be "edge-compatible and small." These three objectives conflict with each other.

**Goal**: (1) Automatically construct high-coverage, low-noise dense UI annotations; (2) Design a lightweight VLM architecture and sequence representation capable of consuming such dense supervision; (3) Ensure "dense supervision" is transferable to external grounding tasks and existing VLMs.

**Key Insight**: The authors leverage a "structural inductive bias" often overlooked—treating the entire screen as a structured document. Borrowing from mature document-to-markup (DocTags, OTSL) approaches, the UI screen is compressed into a set of tag sequences containing coordinates and categories.

**Core Idea**: A Webshot pipeline uses Playwright rendering + DOM extraction + VLM refinement to transform millions of webpages into dense screen supervision for 21M elements. A markup-based sequence called ScreenTag, combined with a structure-aware weighted cross-entropy (CE) loss, enables a small VLM to parse the entire screen into structured output.

## Method

### Overall Architecture
The paper is divided into two parts: the **Webshot pipeline** (data side) and **ScreenVLM** (model side). Webshot samples 1M pages from 45M URLs → Playwright renders full-page screenshots → DOM trees are extracted and filtered by visibility/overlap → Qwen3-VL-8B classifies each candidate into one of 55 categories → VLM-as-a-judge assigns a quality score to discard low-quality samples → result: 771K images / 21M elements, split 90/5/5. On the model side, ScreenVLM uses SigLIP-2 as the vision backbone to encode image patch tokens. After projection, tokens are fed into a 165M Granite autoregressive decoder (initialized from the Granite Docling document-to-markup model) to output an XML-like ScreenTag sequence. Each element follows the format `<tag> <x1> <y1> <x2> <y2> [text] [children] </tag>`, where coordinates are normalized and quantized to a 0–500 grid.

### Key Designs

1.  **Webshot Automated Dense Annotation Pipeline**:
    - **Function**: Eliminates manual labor for the dense task of labeling all visible elements across 55 classes.
    - **Mechanism**: First uses DOM + Playwright to obtain candidate boxes (rejecting degenerate, invisible, or near-duplicate nested wrappers), preserving container hierarchies (navbars, cards, modals). Qwen3-VL-8B then re-predicts categories based on "full image + element crop + attributes." Finally, a VLM-as-a-judge scores the page across four dimensions (coverage, false detection, duplication, localization), discarding those below a threshold.
    - **Design Motivation**: DOM labels have wide coverage but are noisy, while pure VLM labeling is too expensive. This cascade of "DOM candidates + VLM re-classification + full-page filtering" aims to approach "complete + clean" data with zero human intervention.

2.  **ScreenTag: Compact Screen Structure Sequence**:
    - **Function**: Compresses a screenshot into a segment of autoregressively generatable structured text.
    - **Mechanism**: Each element generates a nested sequence of `<tag> <x1> <y1> <x2> <y2> [text] [children] </tag>`. Coordinates use discrete tokens from 0–500, while text and children are optional. This representation is compact, unambiguous for parsing, and naturally fits the token-by-token generation of an LLM decoder.
    - **Design Motivation**: The authors intentionally reuse the inductive bias of document-to-markup—inheriting "markup-friendly" pre-training from Granite Docling—making the transfer to the task of "structured rectangles + text" seamless.

3.  **Structure-Aware Weighted Cross-Entropy**:
    - **Function**: Assigns higher weights to tag and coordinate tokens within the ScreenTag sequence to prevent long OCR text from dominating gradients.
    - **Mechanism**: $\mathcal{L}(\theta) = -\sum_{t=1}^{T} w(y_t)\log p_\theta(y_t \mid y_{<t}, I)$, where $w(y_t) = \lambda_{\text{tag}}$ (if $y_t \in \mathcal{V}_{\text{tag}}$), $\lambda_{\text{loc}}$ (if $y_t \in \mathcal{V}_{\text{loc}}$), and $1$ otherwise.
    - **Design Motivation**: A misplaced coordinate or wrong tag invalidates the entire element, while a wrong character in text is less critical. Furthermore, OCR text dominates sequence length, causing standard CE to push the model toward "reading text without knowing element locations." Weighting aligns the optimization objective with structural fidelity.

### Loss & Training
ScreenVLM is fine-tuned on the ScreenParse training set for 287,500 steps using 16 H100 GPUs (2 nodes × 8 cards) with an effective batch size of 64 and a sequence length capped at 8192 tokens. Grouped learning rates: $2.12\times 10^{-2}$ for the multimodal projection layer, and $2\times 10^{-3}$ for the vision/language backbones.

## Key Experimental Results

### Main Results
Comparison of dense parsing on the ScreenParse test set (PageIoU measures pixel-level coverage; Label PageIoU requires category matching).

| Model | Size | Page IoU | Label PageIoU | mAP@50 |
|-------|------|----------|---------------|--------|
| Qwen3-VL-8B-Instruct | 8B | 0.294 | – | – |
| InternVL3-2B | 2B | 0.111 | 0.030 | 0.000 |
| InternVL3-2B + ScreenParse | 2B | 0.509 (+0.398) | 0.174 | 0.072 |
| Qwen3-VL-2B + ScreenParse | 2B | 0.585 | 0.166 | 0.152 |
| **ScreenVLM (Ours)** | **316M** | **0.606** | **0.197** | **0.303** |
| RT-DETRv2 + ScreenParse | 43M | 0.600 | 0.172 | 0.362 |

With 1/25 of the parameters, ScreenVLM more than doubles the PageIoU of Qwen3-VL-8B. Fine-tuning Qwen3-VL-2B and InternVL3-2B on ScreenParse also yields gains of 0.36–0.40 PageIoU, proving that this supervision is a transferable asset.

### Ablation Study
Structure-aware weighted loss vs. standard CE.

| Setting | ScreenParse PageIoU | GroundCUA PageIoU | ScreenSpot-PC Recall |
|------|---------------------|---------------------|----------------------|
| Full (StructureAware) | 0.606 | 0.251 | 0.222 |
| w/ CE only | 0.592 | 0.226 | 0.129 |
| Gain | +2.4% | +11.1% | **+72.1%** |

Efficiency (H100 + vLLM, average of 128 samples):

| Model | Size (MB) | Latency (ms) | Throughput (s$^{-1}$) |
|-------|-----------|--------------|----------------------|
| Qwen3-VL-2B | 4300 | $1289.1 \pm 251.7$ | 0.78 |
| InternVL3-2B | 4178 | $1267.3 \pm 187.9$ | 0.79 |
| **ScreenVLM** | **632** | $\mathbf{276.4 \pm 139.0}$ | **3.62** |

### Key Findings
- The structure-aware loss provides the largest gains in "out-of-distribution" and "few-element grounding" scenarios (e.g., +72.1% Recall on ScreenSpot-PC), indicating it helps the model resist the dilution of structural tokens by OCR text.
- ScreenParse supervision is "model-agnostic": fine-tuning different families like InternVL3, Qwen3-VL, and even detectors like YOLO/RT-DETR results in improvements. This suggests dense screen supervision is for UI understanding what ImageNet is for computer vision.
- ScreenVLM achieves high PixCov (>0.83) but low Recall on ScreenSpot-PC/Mobile, suggesting it learns to "cover the key pixels" but hasn't yet mastered outputting very tight element-level boxes—a bias from web-only training that the authors identified for future work.

## Highlights & Insights
- This is a critical step in shifting "computer-use data" from "action-driven sparse annotation" to "dense screen supervision." While most GUI research follows grounding benchmarks, this work redefines supervision itself.
- ScreenTag "documentizes" the GUI screen, reusing mature document parsing biases to allow a 316M model to learn structure effectively. This "cross-domain markup representation transfer" provides a strong template for other structured perception tasks.
- Using a VLM as both an "annotation refiner" and a "judge," combined with DOM candidates, creates an iterative loop to bootstrap weak labels into strong ones. This paradigm can be adapted to any domain where "rendering source + DOM/SVG" is available.

## Limitations & Future Work
- Data is entirely sourced from the web. PC and Mobile UI conventions differ from the web; experiments show significantly lower Recall on ScreenSpot-PC/Mobile compared to Web.
- The VLM-judge threshold requires manual calibration, and "high quality" determinations are still subject to the biases of the backbone model.
- ScreenTag is a nested sequence; the current limit of 8192 tokens may still lead to truncation on ultra-large screens (e.g., 4K long screenshots).
- The paper does not integrate ScreenVLM into an end-to-end agent to prove downstream gains from "dense parsing → action," which is the obvious next step.

## Related Work & Insights
- **vs SeeClick / ScreenSpot**: These use sparse grounding (one image, one instruction, one element). This work pursues "full-screen dense" and proves that such supervision also benefits the former.
- **vs GroundCUA**: GroundCUA is also dense but limited to 55k samples and 8 classes. ScreenParse is an order of magnitude larger in scale (771k) and categories (55).
- **vs OmniParser**: OmniParser acts as a detector-style YOLO parser, strong in localization but lacking structured output aligned with language. ScreenVLM outputs markup structure directly consumable by downstream LLM agents.
- **vs Granite Docling / SmolDocling**: These are document-to-markup VLMs. This work validates that "structured-markup pre-training" is an excellent starting point for UI perception.

## Rating
- Novelty: ⭐⭐⭐⭐ Solidifies the "sparse to dense" paradigm for GUI data, though individual techniques are combinations of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple VLM families/detectors, 3 benchmarks, and comprehensive loss/efficiency ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and supported by figures, though some key designs are detailed in the appendix.
- Value: ⭐⭐⭐⭐⭐ Both dataset and small model are open-sourced, representing an infrastructure-level contribution to the GUI agent community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](../../CVPR2026/multimodal_vlm/efficient_document_parsing_via_parallel_token_prediction.md)
- [\[ACL 2026\] What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning](../../ACL2026/multimodal_vlm/what39s_missing_in_screen-to-action_towards_a_ui-in-the-loop_paradigm_for_multim.md)
- [\[ICLR 2026\] Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](../../ICLR2026/multimodal_vlm/grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)
- [\[ACL 2026\] iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models](../../ACL2026/multimodal_vlm/ireasoner_trajectory-aware_intrinsic_reasoning_supervision_for_self-evolving_lar.md)

</div>

<!-- RELATED:END -->
