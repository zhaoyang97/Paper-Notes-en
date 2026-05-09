---
title: >-
  [Paper Note] ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images
description: >-
  [ACL 2026][Human Understanding][Face Forgery Detection] The paper proposes the Forgery Attribution Report Generation task, constructs the MMTT dataset with 152,217 samples (the first large-scale facial forgery dataset providing both pixel-level masks and human-written text descriptions), and introduces the ForgeryTalker end-to-end baseline that jointly generates localization masks and attribution reports via a shared encoder and dual decoders (mask + language model), achieving 59.3 CIDEr and 73.67 IoU.
tags:
  - ACL 2026
  - Human Understanding
  - Face Forgery Detection
  - Attribution Report Generation
  - Multimodal Forensics
  - Image Tampering Localization
  - Explainable AI
date: 2025-05-08
content_hash: e3bc6f684af5835d
---

# ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images

**Conference**: ACL 2026  
**arXiv**: [2412.19685](https://arxiv.org/abs/2412.19685)  
**Code**: [https://github.com/NattyLianJc/Generating-Attribution-Reports](https://github.com/NattyLianJc/Generating-Attribution-Reports)  
**Area**: Human Understanding  
**Keywords**: Face Forgery Detection, Attribution Report Generation, Multimodal Forensics, Image Tampering Localization, Explainable AI

## TL;DR

The paper proposes the Forgery Attribution Report Generation task, constructs the MMTT dataset with 152,217 samples (the first large-scale facial forgery dataset providing both pixel-level masks and human-written text descriptions), and introduces the ForgeryTalker end-to-end baseline that jointly generates localization masks and attribution reports via a shared encoder and dual decoders (mask + language model), achieving 59.3 CIDEr and 73.67 IoU.

## Background & Motivation

**Background**: Advanced generative models such as diffusion models have greatly enhanced the realism of synthetic images. Facial manipulation detection research has evolved from binary classification to fine-grained forgery localization.

**Limitations of Prior Work**: (1) Binary classification only outputs a verdict without providing semantic understanding; (2) Pixel-level masks can localize tampered regions but treat all tampered pixels equally, failing to distinguish subtle from obvious manipulations or explain the cause and nature of tampering; (3) As modern forgeries become increasingly realistic, masks cannot provide descriptive evidence for human reviewers.

**Key Challenge**: Existing methods answer "where was it tampered" but fail to answer "why it is considered tampered" and "what are the specific manifestations of tampering."

**Goal**: Define the forgery attribution report generation task—simultaneously localizing tampered regions (Where) and generating natural language explanations based on the editing process (Why), providing explainable multimodal forensics.

**Key Insight**: Leverage the forgery process itself to generate pixel-perfect ground-truth masks (no manual mask annotation needed), combined with a human-in-the-loop pipeline for writing text descriptions to construct a high-quality dataset.

**Core Idea**: Through a shared encoder that learns unified forgery-aware multimodal representations, simultaneously driving a mask decoder (localization) and a large language model (explanation), achieving synergistic enhancement between localization and explanation.

## Method

### Overall Architecture

ForgeryTalker extends InstructBLIP and consists of: (1) a shared encoder (ViT + Q-Former) that processes manipulated images to extract multimodal features; (2) a Forgery Prompter Network (FPN) that generates region keyword prompts; (3) a mask decoder for forgery localization; (4) a large language model for attribution report generation. Training proceeds in two stages: forgery-aware pretraining and attribution report generation.

### Key Designs

1. **MMTT Dataset Construction**:

    - Function: Provides the first large-scale facial forgery dataset containing both pixel-level masks and text descriptions
    - Mechanism: 152,217 samples covering four manipulation paradigms (face swapping, facial editing, Transformer inpainting, diffusion inpainting). Masks are programmatically generated from the forgery process itself, ensuring pixel-level precision. Text descriptions are written by 30 professional annotators following a structured process, with each annotator observing each image for at least 1 minute and descriptions limited to 120 words
    - Design Motivation: Existing datasets (FaceForensics++, Celeb-DF, etc.) only provide labels or masks without textual explanations. Generating masks from the forgery process eliminates annotation errors

2. **Forgery Prompter Network (FPN)**:

    - Function: Generates facial region keywords to guide downstream reasoning and report generation
    - Mechanism: Uses ViT as backbone, introducing parallel convolutional branches (with CoordConv) in the first $m$ layers to capture local anomalies. Global attention and local convolutional features are fused layer-by-layer, then a classification head predicts a 21-dimensional facial region probability vector. The output region predictions fill a template "These facial regions may have been AI-tampered: [R]" to guide LLM generation
    - Design Motivation: Human reviewers also need careful inspection to find inconsistencies; FPN automates the "focus area check" prompt. CoordConv leverages the rigid spatial structure of facial regions

3. **Two-Stage Training Strategy**:

    - Function: Progressively learns forgery-aware representations and attribution report generation
    - Mechanism: Stage 1 (pretraining): jointly optimizes masked language modeling $\mathcal{L}_{mlm}$, language modeling $\mathcal{L}_{lm}$, segmentation loss $\mathcal{L}_{seg}$, and contrastive alignment $\mathcal{L}_{con}$. Stage 2 (report generation): first trains FPN (BCE + Dice loss), then freezes it and fine-tunes Q-Former and mask decoder for localization and explanation
    - Design Motivation: Direct end-to-end training is unstable. First establishing forgery-aware representations, then using FPN's region prompts to guide report generation

### Loss & Training

Pretraining stage loss: $\mathcal{L} = \mathcal{L}_{mlm} + \mathcal{L}_{lm} + \mathcal{L}_{seg} + \mathcal{L}_{con}$. FPN loss: $\frac{1}{2}(\mathcal{L}_{BCE} + \mathcal{L}_{Dice})$, where BCE uses a discount factor $\omega < 1$ to handle class imbalance for unmodified regions. The report generation stage uses standard language modeling loss.

## Key Experimental Results

### Main Results

**Report generation and forgery localization performance**

| Method | CIDEr | BLEU-4 | ROUGE-L | IoU |
|--------|-------|--------|---------|-----|
| InstructBLIP (baseline) | 42.1 | 8.2 | 29.5 | - |
| ForgeryTalker (w/o FPN) | 52.8 | 11.4 | 33.2 | 68.3 |
| **ForgeryTalker** | **59.3** | **13.1** | **35.7** | **73.67** |

### Ablation Study

| Config | CIDEr | IoU | Note |
|--------|-------|-----|------|
| Full model | 59.3 | 73.67 | full model |
| w/o FPN | 52.8 | 68.3 | FPN contributes to both tasks |
| w/o pretraining | 45.6 | 65.1 | Pretraining stage is crucial |
| w/o contrastive loss | 55.1 | 71.2 | Cross-modal alignment aids synergy |

### Key Findings

- FPN significantly contributes to both report generation (+6.5 CIDEr) and localization (+5.37 IoU), demonstrating that region prompts effectively guide both tasks
- Joint training of localization and report generation outperforms separate training, confirming the synergistic effect between the two tasks
- Eyes, eyebrows, and lips are the most common tampering targets, accounting for the majority of local edits
- Generated text descriptions average 27.4 words and highly correspond with visual forgery regions

## Highlights & Insights

- The task definition is forward-looking—the evolution from "detecting forgery" to "explaining forgery" is a natural progression in the forensics domain
- The strategy of generating masks from the forgery process itself eliminates annotation errors and provides an efficient paradigm for dataset construction
- The dual-decoder shared-encoder design can be transferred to other multimodal tasks requiring simultaneous localization and explanation

## Limitations & Future Work

- Assumes input images have already been flagged as suspicious by an upstream detector; does not handle authentic images
- Text description annotation costs remain high (30 annotators)
- Only covers facial manipulation; other types of image tampering are not addressed
- Future work could explore finer-grained manipulation method attribution (e.g., distinguishing GAN from diffusion generation)

## Related Work & Insights

- **vs FaceForensics++**: Only provides classification/segmentation labels without textual explanations; MMTT is the first to provide textual attribution
- **vs OpenForensics**: Provides detection boxes and masks but no semantic explanations
- **vs DF40**: Large-scale but only labels + masks; MMTT is smaller in scale but richer in information dimensions

## Rating

- Novelty: ⭐⭐⭐⭐ Forgery attribution report generation is a meaningful new task definition
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed dataset construction, multiple ablation validations
- Writing Quality: ⭐⭐⭐⭐ Clear task motivation, comprehensive dataset comparison
- Recommendation: ⭐⭐⭐⭐ Advances forensics from detection toward explainable analysis

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals](../../ICLR2026/human_understanding/inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth.md)
- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)
- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](../../CVPR2026/human_understanding/a_two_stage_dual_modality_model_for_facial_expression_recognition.md)
- [\[CVPR 2026\] WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering](../../CVPR2026/human_understanding/wildcap_facial_albedo_capture_in_the_wild_via_hybrid_inverse_rendering.md)

<!-- RELATED:END -->
