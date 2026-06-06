---
title: >-
  [Paper Note] ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images
description: >-
  [ACL 2026][LLM Safety][Facial Forgery Detection] This paper proposes a new task called Forgery Attribution Report Generation and constructs the MMTT dataset (the first large-scale facial forgery dataset providing both pi…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Facial Forgery Detection"
  - "Attribution Report Generation"
  - "Multimodal Forensics"
  - "Image Manipulation Localization"
  - "Explainable AI"
date: 2026-05-08
content_hash: a6c0e166a23cd6f9
---

# ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images

**Conference**: ACL 2026  
**arXiv**: [2412.19685](https://arxiv.org/abs/2412.19685)  
**Code**: [https://github.com/NattyLianJc/Generating-Attribution-Reports](https://github.com/NattyLianJc/Generating-Attribution-Reports)  
**Area**: Human Understanding  
**Keywords**: Facial Forgery Detection, Attribution Report Generation, Multimodal Forensics, Image Manipulation Localization, Explainable AI

## TL;DR

This paper proposes a new task called Forgery Attribution Report Generation and constructs the MMTT dataset (the first large-scale facial forgery dataset providing both pixel-level masks and human-annotated text descriptions) containing 152,217 samples. It introduces ForgeryTalker, an end-to-end baseline that jointly generates localization masks and attribution reports through a shared encoder and dual decoders (mask + language model), achieving 59.3 CIDEr and 73.67 IoU.

## Background & Motivation

**Background**: Advanced generative models, such as diffusion models, have significantly enhanced the realism of synthetic images. Facial manipulation detection research has evolved from binary classification to fine-grained forgery localization.

**Limitations of Prior Work**: (1) Binary classification only outputs a verdict without providing semantic understanding; (2) Pixel-level masks locate manipulated regions but treat all manipulated pixels equally, failing to distinguish between subtle and obvious changes or explain the nature and cause of the forgery; (3) As modern forgeries become increasingly realistic, masks alone do not provide descriptive evidence for human auditors.

**Key Challenge**: Existing methods answer "where it was tampered" but fail to address "why it is considered tampered" and "what the specific manifestations of the tampering are."

**Goal**: To define the task of Forgery Attribution Report Generation—simultaneously localizing tampered regions (Where) and generating natural language explanations based on the editing process (Why) to provide explainable multimodal forensics.

**Key Insight**: Leverage the forgery process itself to generate pixel-perfect ground truth masks (eliminating the need for manual mask labeling) and integrate a human-in-the-loop pipeline to write text descriptions, thereby constructing a high-quality dataset.

**Core Idea**: Use a shared encoder to learn unified forgery-aware multimodal representations that simultaneously drive a mask decoder (localization) and a large language model (explanation), achieving synergistic enhancement between localization and explanation.

## Method

### Overall Architecture

ForgeryTalker is based on an extension of InstructBLIP and includes: (1) A shared encoder (ViT + Q-Former) to process tampered images and extract multimodal features; (2) A Forgery Prompter Network (FPN) to generate regional keyword prompts; (3) A mask decoder for forgery localization; (4) A large language model to generate attribution reports. Training is conducted in two stages: forgery-aware pre-training and attribution report generation.

### Key Designs

1.  **MMTT Dataset Construction**:
    - **Function**: Provides the first large-scale facial forgery dataset containing both pixel-level masks and text descriptions.
    - **Mechanism**: Includes 152,217 samples covering four tampering paradigms (face swapping, face editing, Transformer inpainting, and diffusion inpainting). Masks are programmatically generated from the forgery process to ensure pixel precision. Text descriptions were written by 30 professional annotators following a structured workflow, with each person spending at least 1 minute per image and descriptions limited to 120 words.
    - **Design Motivation**: Existing datasets (FaceForensics++, Celeb-DF, etc.) only provide labels or masks without text explanations. Generating masks from the forgery process eliminates annotation errors.

2.  **Forgery Prompter Network (FPN)**:
    - **Function**: Generates facial region keywords to guide downstream reasoning and report generation.
    - **Mechanism**: Utilizes ViT as the backbone, introducing parallel convolutional branches (including CoordConv) in the first $m$ layers to capture local anomalies. Global attention and local convolutional features are fused layer-by-layer, followed by a classification head predicting a 21-dimensional facial region probability vector. The output serves as a template filler: "These facial regions might be tampered by AI: [R]," which guides the LLM during generation.
    - **Design Motivation**: Human auditors require careful inspection to find inconsistencies; FPN acts as an automated "focus area" prompt. CoordConv leverages the rigid spatial structure of facial regions.

3.  **Two-Stage Training Strategy**:
    - **Function**: Sequentially learns forgery-aware representations and attribution report generation.
    - **Mechanism**: Stage 1 (Pre-training): Jointly optimizes mask language modeling $\mathcal{L}_{mlm}$, language modeling $\mathcal{L}_{lm}$, segmentation loss $\mathcal{L}_{seg}$, and contrastive alignment $\mathcal{L}_{con}$. Stage 2 (Report Generation): Trains the FPN first (BCE + Dice loss), then freezes it to fine-tune the Q-Former and mask decoder for localization and explanation.
    - **Design Motivation**: Direct end-to-end training is unstable. Establishing forgery-aware representations first, then utilizing FPN's regional prompts, effectively guides report generation.

### Loss & Training

Pre-training stage loss: $$\mathcal{L} = \mathcal{L}_{mlm} + \mathcal{L}_{lm} + \mathcal{L}_{seg} + \mathcal{L}_{con}$$. FPN loss: $$\frac{1}{2}(\mathcal{L}_{BCE} + \mathcal{L}_{Dice})$$, where BCE uses a discount factor $\omega < 1$ to address the class imbalance of unmodified regions. The report generation stage utilizes standard language modeling loss.

## Key Experimental Results

### Main Results

**Report Generation and Forgery Localization Performance**

| Method | CIDEr | BLEU-4 | ROUGE-L | IoU |
| :--- | :--- | :--- | :--- | :--- |
| InstructBLIP (baseline) | 42.1 | 8.2 | 29.5 | - |
| ForgeryTalker (w/o FPN) | 52.8 | 11.4 | 33.2 | 68.3 |
| **ForgeryTalker** | **59.3** | **13.1** | **35.7** | **73.67** |

### Ablation Study

| Configuration | CIDEr | IoU | Description |
| :--- | :--- | :--- | :--- |
| Full model | 59.3 | 73.67 | Complete |
| w/o FPN | 52.8 | 68.3 | FPN contributes to both tasks |
| w/o Pre-training | 45.6 | 65.1 | Pre-training stage is critical |
| w/o Contrastive Loss | 55.1 | 71.2 | Cross-modal alignment aids synergy |

### Key Findings

- FPN contributes significantly to both report generation (+6.5 CIDEr) and localization (+5.37 IoU), indicating that regional prompts effectively guide both tasks.
- Jointly training localization and report generation outperforms separate training, confirming a synergistic effect between the two tasks.
- Eyes, eyebrows, and lips are the most common tampering targets, accounting for the majority of local edits.
- Generated text descriptions average 27.4 words and correspond highly with visual forgery regions.

## Highlights & Insights

- The task definition is forward-looking—the transition from "detecting forgery" to "explaining forgery" is a natural evolutionary direction in the forensics field.
- The strategy of generating masks from the forgery process itself eliminates annotation errors and provides an efficient paradigm for dataset construction.
- The design of a shared encoder with dual decoders can be transferred to other multimodal tasks requiring simultaneous localization and explanation.

## Limitations & Future Work

- Assumes that the input image has already been flagged as suspicious by an upstream detector; it does not process genuine images.
- The cost of annotating text descriptions remains high (requiring 30 annotators).
- Only covers facial tampering; other types of image manipulation are not addressed.
- Future work could explore finer-grained tampering attribution (e.g., distinguishing between GAN and diffusion-based generation).

## Related Work & Insights

- **vs FaceForensics++**: Only provides classification/segmentation labels without text explanations; MMTT is the first to provide text attribution.
- **vs OpenForensics**: Provides detection boxes and masks but lacks semantic explanations.
- **vs DF40**: Large-scale but only provides labels and masks; while MMTT is smaller in scale, its information dimensions are much richer.

## Rating

- Novelty: ⭐⭐⭐⭐ Forgery Attribution Report Generation is a meaningful new task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed dataset construction and comprehensive ablation verification.
- Writing Quality: ⭐⭐⭐⭐ Clear task motivation and thorough dataset comparison.
- Value: ⭐⭐⭐⭐ Pushes forensics from simple detection toward explainable analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Generating Effective CoT Traces for Mitigating Causal Hallucination](generating_effective_cot_traces_for_mitigating_causal_hallucination.md)
- [\[ACL 2026\] De-Anonymization at Scale via Tournament-Style Attribution](de-anonymization_at_scale_via_tournament-style_attribution.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](../../NeurIPS2025/llm_safety/attention_your_vision_language_model_could_be_maliciously_manipulated.md)
- [\[AAAI 2026\] Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](../../AAAI2026/llm_safety/uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)

</div>

<!-- RELATED:END -->
