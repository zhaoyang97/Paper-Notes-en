---
title: >-
  [Paper Note] Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?
description: >-
  [ICLR 2026][Multimodal VLM][modal aphasia] This paper identifies and systematically defines the phenomenon of **Modal Aphasia** — unified multimodal models can generate visual concepts (e.g., movie poster images) from memory with near-perfect fidelity, yet exhibit error rates more than 7× higher when verbally describing the same concepts, with severe hallucinations occurring almost exclusively in the text modality. Through real-world experiments with frontier models (ChatGPT-5) and controlled synthetic experiments with open-source models (Janus-Pro, Harmon), the paper confirms that modal aphasia is a systemic deficiency of current unified architectures rather than a training artifact, and demonstrates its potential threat to AI safety frameworks.
tags:
  - ICLR 2026
  - Multimodal VLM
  - modal aphasia
  - unified multimodal models
  - cross-modal knowledge transfer
  - memorization
  - AI safety
date: 2026-05-08
content_hash: 8b81ed47ae806050
---

# Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?

**Conference**: ICLR 2026
**arXiv**: [2510.21842](https://arxiv.org/abs/2510.21842)
**Code**: [https://github.com/ethz-spylab/modal-aphasia](https://github.com/ethz-spylab/modal-aphasia)
**Area**: Multimodal VLM / AI Safety
**Keywords**: modal aphasia, unified multimodal models, cross-modal knowledge transfer, memorization, AI safety

## TL;DR

This paper identifies and systematically defines the phenomenon of **Modal Aphasia** — unified multimodal models can generate visual concepts (e.g., movie poster images) from memory with near-perfect fidelity, yet exhibit error rates more than 7× higher when verbally describing the same concepts, with severe hallucinations occurring almost exclusively in the text modality. Through real-world experiments with frontier models (ChatGPT-5) and controlled synthetic experiments with open-source models (Janus-Pro, Harmon), the paper confirms that modal aphasia is a systemic deficiency of current unified architectures rather than a training artifact, and demonstrates its potential threat to AI safety frameworks.

## Background & Motivation

**Background**: Multimodal large models are evolving from "compositional" designs (frozen pretrained components + adapters, e.g., Flamingo, LLaVA) toward "natively unified" architectures (Chameleon, Janus-Pro, ChatGPT-5). The latter jointly train on images and text within a shared representation space, theoretically enabling more consistent cross-modal reasoning and knowledge transfer.

**Limitations of Prior Work**: Within individual modalities, memorization has been well studied — diffusion models can reproduce training images (Carlini et al., 2023), and LLMs can verbatim extract training text (Nasr et al., 2025). However, cross-modal memorization has rarely been explored: if a concept is memorized in the visual modality, can it be accurately retrieved in the text modality? Wen et al. (2025) identified a recall gap between source and target modalities but did not address image generation scenarios. Papadimitriou et al. (2025) found that even with a shared representation space, different modalities in VLMs encode concepts in modality-specific ways — yet the practical consequences of such an incomplete "latent bridge" remain unclear.

**Key Challenge**: ChatGPT-5 can reproduce Harry Potter movie posters at nearly pixel-level fidelity (including character positions, costume details, and color composition), yet when verbally describing the same poster it fabricates characters such as Draco Malfoy and Snape who are entirely absent, and misidentifies "the Sword of Gryffindor" as "a wand." This reveals that "knowing how to draw" does not entail "knowing how to say" — visual and textual knowledge exist in a fractured state within the model.

**Goal**: (1) Rigorously define and quantify this cross-modal knowledge fragmentation; (2) demonstrate that it is a systemic property of unified architectures rather than an idiosyncrasy of any individual model's training; (3) reveal its practical threat to AI safety frameworks.

**Key Insight**: The authors draw an analogy to *optic aphasia* in cognitive science — a condition in which patients can see and recognize objects yet cannot name them when presented visually — as well as *verbal overshadowing*, where verbalizing a visual memory impairs recognition accuracy. The authors term this cross-modal fragmentation in AI systems **Modal Aphasia**.

**Core Idea**: Knowledge transfer in unified multimodal models is asymmetric — concepts successfully memorized in the visual modality cannot be reliably accessed via the text modality, constituting a systemic failure of cross-modal understanding.

## Method

### Overall Architecture

The paper adopts a three-tier experimental design. The first tier validates the existence of modal aphasia using real memorized concepts (movie posters) in a frontier closed-source model (ChatGPT-5). The second tier conducts controlled experiments with open-source unified models (Janus-Pro 7B, Harmon 1.5B) using synthetic data to rule out training artifacts and confirm that modal aphasia is an architectural property. The third tier constructs a safety case study demonstrating how modal aphasia can be exploited to circumvent unimodal safety alignment.

### Key Designs

1. **Frontier Model Experiments (ChatGPT-5 + Movie Posters)**

   - **Function**: First empirical validation of modal aphasia in a real-world setting.
   - **Mechanism**: Nine widely recognized theatrical release posters are selected (*The Dark Knight*, *The Matrix*, *Inception*, *Star Wars IV/V*, *Harry Potter 2*, *Back to the Future*, *LOTR: ROTK/FOTR*). These posters appear frequently in training data as images but are rarely described in detail in text. ChatGPT-5 is prompted separately to generate each poster image and to produce an independent verbal description (with no image reference). Claude Opus 4.1 is used to construct modality-agnostic scoring rubrics: open-ended evaluations of both the images and texts are first collected, all relevant details are gathered, and these are unified into a checklist of positive requirements (e.g., "Harry Potter should hold the Sword of Gryffindor") and negative requirements (e.g., "Draco Malfoy should not appear"). Three independent scoring rounds plus human verification ensure reliability.
   - **Design Motivation**: Movie posters are ideal test subjects — they appear abundantly online as images (title + poster image) but are rarely described in detail in text. This asymmetry in training data is precisely the condition that triggers modal aphasia, analogous to the Reversal Curse, where training data for A→B far exceeds that for B→A.

2. **Open-Source Model Controlled Experiments (Synthetic Data)**

   - **Function**: Demonstrate under controlled conditions that modal aphasia is a universal architectural property.
   - **Mechanism**: Two architecturally distinct unified models are used — Janus-Pro (autoregressive discrete token generation) and Harmon (masked iterative continuous embedding generation). Two synthetic datasets are designed: (a) a **synthetic faces** dataset (600 name–portrait pairs), where each face has 4 primary attributes (eye color, hair color, hairstyle, accessory) and 6 secondary attributes covering the full combinatorial space, and models learn to generate the corresponding portrait given a name; (b) an **abstract visual concepts** dataset (840 images), where each image is composed of 4 concepts (shape, position, background color, background texture), with each concept value assigned a fictitious 10-letter word (e.g., "pectatinul" = red). An 80/20 split tests compositional generalization. A critical constraint is that only the LLM backbone is fine-tuned while all visual encoders/decoders are frozen, ensuring that all memorization occurs exclusively within the language model.
   - **Design Motivation**: Freezing visual components rules out the hypothesis that "the image encoder memorizes separately." Modal aphasia persists even when all knowledge is stored in the backbone LLM, implicating the cross-modal retrieval mechanism rather than storage location. Text ability is evaluated via multiple-choice QA (rather than open-ended generation), which unfairly advantages the text modality through the possibility of guessing and indirect cues from options; if text accuracy remains low even under these favorable conditions, open-ended generation would only perform worse.

3. **Safety Case Study (Fragility of Unimodal Alignment)**

   - **Function**: Demonstrate the practical threat of modal aphasia to AI safety.
   - **Mechanism**: Janus-Pro is fine-tuned in two stages. Stage one trains the model to associate "secondary balance units" (an extremely rare expression with fewer than 10 Google search results) with images of feet, simulating a model that has learned an unsafe concept from training data. Stage two applies safety alignment in the text modality — the model is trained to refuse prompts containing common words such as "feet" while responding normally to safe prompts. The model is then tested with "secondary balance units" prompts to determine whether refusal is triggered.
   - **Design Motivation**: This simulates real-world scenarios in which code words circumvent content moderation. If the model has learned the association "feet = unsafe" only in the text modality, while the concept's representation in the image modality remains uncovered by safety alignment, rare expressions can reactivate unsafe image generation.

### Evaluation Methods

- **Image accuracy**: Faces are evaluated by VLM-judge for attribute match; abstract concepts use conventional computer vision (shape/color/position detection); movie posters use rubric-based human + LLM joint evaluation.
- **Text accuracy**: Multiple-choice questions (given a name or fictitious word, select the corresponding attribute value); Gemini 2.5 Pro serves as the LLM-judge to parse non-standard responses, with unparseable responses discarded rather than counted as errors (further advantaging the text modality).
- **Safety evaluation**: Detection of whether responses contain a start-of-image token (compliant) vs. refusal text, with Gemini 2.5 Pro judging whether generated images actually contain unsafe content.

### Error Type Taxonomy

Three error types are defined: **omissions** (missing key elements), **minor hallucinations** (detail errors, e.g., describing the Sword of Gryffindor as a wand), and **severe hallucinations** (fabricating characters or attributes that do not exist). Because the space of severe hallucinations is unbounded, the authors collect all severe hallucinations discovered during initial open-ended evaluation and incorporate them as negative requirements in the rubric, enabling comparison across error types on a unified scale.

## Key Experimental Results

### Main Results: Modal Aphasia in ChatGPT-5 Movie Posters

| Evaluation Dimension | Image Generation | Text Description | Ratio |
|---|---|---|---|
| Average rubric error rate | ~6% | ~45% | **7.5×** |
| Proportion of errors that are hallucinations | Partial minor hallucinations | ~75% hallucinations | — |
| Severe hallucination rate | **0%** | **~95%** | Text only |
| Minor hallucination frequency | Baseline | Baseline × 5 | **5×** |

Specific case: The Harry Potter poster rubric contains 13 positive and 4 negative requirements. Image generation passes 16/17 items (only 1 minor hallucination); the verbal description passes only 10/17, fabricating 4 non-existent characters — Dumbledore, Snape, Draco Malfoy, and Fawkes (all severe hallucinations) — along with 2 minor hallucinations.

### Controlled Experiments: Quantifying Modal Aphasia in Open-Source Models

| Experiment | Model | Image Generation Accuracy | Text Description Accuracy | Random Baseline | Gap |
|---|---|---|---|---|---|
| Synthetic faces | Janus-Pro 7B | **~75%** | **~20%** | 20% | Image accurate; text ≈ random |
| Synthetic faces | Harmon 1.5B | **~70%** | **~22%** | 20% | Same |
| Abstract concepts (Train) | Janus-Pro 7B | **~90%** | **~25%** | 17–25% | High image accuracy; text near random |
| Abstract concepts (Test) | Janus-Pro 7B | **~85%** | **~25%** | 17–25% | Generalizes to new combinations but only in image modality |
| Abstract concepts (Train) | Harmon 1.5B | **~85%** | **~30%** | 17–25% | Similar pattern |
| Safety case — refusal rate | Janus-Pro 7B (aligned) | — | — | — | "feet" refused 89%; "secondary balance units" refused only 24% |

### Key Findings

- **Modal aphasia is universal across architectures**: Janus-Pro (discrete token autoregression) and Harmon (continuous embedding masked iteration) employ entirely different image generation paradigms yet both exhibit modal aphasia. The phenomenon persists even when only the LLM backbone is fine-tuned with all visual components frozen, implicating the cross-modal knowledge representation within the language model itself.
- **No correlation between image and text accuracy**: For the same model, image generation accuracy varies across attributes (e.g., Janus-Pro performs worse on eye color than hair color), yet text description accuracy consistently remains near random guessing regardless of image accuracy. Partial exceptions exist: Janus-Pro achieves ~23% text accuracy on shape concepts (above the 14% baseline) but falls below the 25% baseline on position concepts.
- **Generalization ≠ understanding**: In the abstract concepts experiment, models not only memorize training combinations but correctly generate unseen concept combinations (test set accuracy only marginally below training set), yet these generalized concepts remain inaccessible in the text modality. This rules out "pixel-level rote memorization" — the model has genuinely learned composable visual concepts, but these concepts are inaccessible through the text channel.
- **Fragility of safety alignment**: Text-modality alignment teaches the model to refuse "feet" but fails to cover rare expressions, resulting in "secondary balance units" successfully triggering image generation in 76% of cases. More critically, alignment training does not diminish the model's capacity to generate foot images at all — accuracy remains unchanged under forced image generation.
- **Naive "visualize then describe" strategy is ineffective**: Appendix experiments in which ChatGPT-5 is prompted to first "visualize" before describing still show severe modal aphasia, suggesting that more fundamental architectural changes are required.

## Highlights & Insights

- **Experimental proof that "unified" ≠ "unified understanding"**. This is the paper's central contribution — carefully controlled experiments demonstrate that even with joint training in a shared representation space and all knowledge stored in the same LLM backbone, visual knowledge cannot be reliably retrieved through the text channel. This fundamentally challenges the assumption that unified architecture naturally yields unified understanding.
- **Deep connection to the Reversal Curse**. The Reversal Curse (Berglund et al., 2024) concerns failure of generalization across relational direction within a single modality ("A is B" does not imply "B is A"). Modal aphasia concerns failure of generalization across modality direction ("visual A" does not imply "textual A"). Both may share a common root cause: training data in which one generative direction vastly outnumbers the other (e.g., movie titles appearing alongside poster images far more often than alongside textual descriptions).
- **Elegant experimental design via frozen visual components**. Fine-tuning only the backbone LLM is a crucial design choice: it eliminates the simple explanation that "knowledge stored in separate modality-specific components fails to transfer," pinpointing the problem within the language model itself — the same LLM stores knowledge sufficient to drive image generation yet cannot deploy that knowledge for text generation, indicating a failure at the retrieval or routing level.
- **Security threat model closely mirrors real-world scenarios**. Using an extremely rare expression to simulate underworld code words exposes a sharp security problem: model providers cannot enumerate all possible rare expressions for alignment, while attackers need only find a single "code word" not covered by alignment. This implies that data filtering and safety alignment conducted purely at the text level are fundamentally incomplete.

## Limitations & Future Work

- **Frontier model experiments cover only one closed-source model (ChatGPT-5)**. Gemini 2.5 Flash and Grok 3/4 were excluded because they could not accurately reproduce the posters — modal aphasia requires accurate image generation as a prerequisite. As these models improve, broader coverage will be needed.
- **Controlled experiments test only the visual→text direction**. Models are trained to generate images and tested on describing images. The reverse direction (training on verbal descriptions, testing image generation ability) is not examined; it is unclear whether modal aphasia is symmetric.
- **The safety case study is proof-of-concept level**. Only "feet" — a harmless content category — is used to simulate unsafe scenarios, and only Janus-Pro 7B is tested. Quantitative risk assessment on larger models and genuinely harmful content is absent.
- **No solutions are proposed**. The paper speculates that allowing models to internally visualize during inference ("thinking with generated images") may be a viable path forward, but naive prompting approaches shown in the appendix are ineffective, and no practical solution currently exists.
- **Evaluation methodology limitations**. Text ability is assessed via multiple-choice rather than open-ended generation, and unparseable responses are discarded rather than counted as errors — both choices advantage the text modality. Nevertheless, text accuracy remains near random even under these favorable conditions, confirming the severity of the problem. However, this also means the paper cannot precisely quantify the "true" text failure rate.

## Related Work & Insights

- **vs. Reversal Curse (Berglund et al., ICLR 2024)**: The Reversal Curse involves failure to generalize across relational direction within a single modality ("A is B" ↛ "B is A"). Modal aphasia involves failure to generalize across modality direction (visual memory ↛ textual description). Both may share a common root cause — asymmetric conditional distributions in training data. Modal aphasia is arguably more fundamental, as it occurs within "unified" models possessing a shared representation space.
- **vs. Papadimitriou et al. (2025)**: Their work identifies modality-specific "latent bridges" in VLMs at the representational level. Modal aphasia can be understood as the behavioral consequence of these bridges being incomplete — when the bridge is broken, knowledge on the visual shore cannot cross to the textual shore.
- **vs. West et al. (ICLR 2024, "The Generative AI Paradox")**: The Generative AI Paradox posits that models which can create do not necessarily understand. Modal aphasia is a concrete instantiation of this paradox in unified multimodal models — the model can "create" (generate images) but cannot "understand" (verbally describe) the same concept.
- **vs. Modality Imbalance literature**: Modality imbalance research focuses on differential convergence rates and contributions of modalities in classification tasks. Modal aphasia is distinct: (a) only the backbone is fine-tuned, with no modality-specific parameter differences; (b) the issue is not that "text is stronger" (prior work has found VLMs over-rely on text) but rather that "text cannot access visual memory."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The discovery and naming of "modal aphasia" is highly insightful; the analogy to cognitive science is precise; the paper provides a new framework for understanding the fundamental limitations of unified multimodal models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The three-tier design — frontier model real data + open-source synthetic controls + safety case study — is rigorous, though the safety case study remains proof-of-concept and lacks testing on genuinely harmful content.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The phenomenon is named precisely, the cognitive science analogy is natural, the argumentation is logically clear, and the approach to controlling experimental variables is exemplary.
- **Value**: ⭐⭐⭐⭐⭐ Provides fundamental insights for multimodal model architecture design (unified training ≠ unified understanding) and has direct practical implications for AI safety research (unimodal alignment is insufficient).

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images](../../ACL2026/multimodal_vlm/leave_my_images_alone_preventing_multi-modal_large_language_models_from_analyzin.md)
- [\[CVPR 2026\] TIGeR: A Unified Framework for Time, Images and Geo-location Retrieval](../../CVPR2026/multimodal_vlm/tiger_a_unified_framework_for_time_images_and_geo-location_retrieval.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)

<!-- RELATED:END -->
