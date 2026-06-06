---
title: >-
  [Paper Note] DMN: A Compositional Framework for Jailbreaking Multimodal LLMs with Multi-Image Inputs
description: >-
  [ACL2026][Multimodal VLM][Multi-image input] This paper proposes DMN, a multi-image jailbreak evaluation framework that combines distributed instructions, multimodal evidence…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Multi-image input"
  - "MLLM safety"
  - "Jailbreak evaluation"
  - "Multimodal defense"
  - "Safety alignment"
date: 2026-05-08
content_hash: 9d11bb0a5a9ac02d
---

# DMN: A Compositional Framework for Jailbreaking Multimodal LLMs with Multi-Image Inputs

**Conference**: ACL2026  
**arXiv**: [2605.18915](https://arxiv.org/abs/2605.18915)  
**Code**: Not reported in cache  
**Area**: Multimodal VLM / AI Safety  
**Keywords**: Multi-image input, MLLM safety, Jailbreak evaluation, Multimodal defense, Safety alignment

## TL;DR
This paper proposes DMN, a multi-image jailbreak evaluation framework that combines distributed instructions, multimodal evidence, and numerical chain auxiliary tasks. It demonstrates that current MLLMs supporting multi-image inputs have significant weaknesses in cross-image safety alignment and provides a multi-image-aware filter as a preliminary defense.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) can simultaneously process text and images, and an increasing number of commercial models support multi-image inputs. Existing research on MLLM jailbreaking focuses mostly on single-image settings, such as rendering harmful intents into typographic images, using relevant images for prompt assistance, or adding irrelevant tasks to distract the model.

**Limitations of Prior Work**: The attack space for single-image jailbreaking is limited: a single image struggle to carry full context or decompose intent into multiple local fragments. Simultaneously, current safety alignment and filtering for MLLMs are largely designed for single images or single-turn text, lacking specialized protection against compositional semantics across multiple images.

**Key Challenge**: While multi-image inputs improve model usability, safety mechanisms do not necessarily perform global aggregate judgments across the entire set of images. If a harmful intent is dispersed across multiple images, each image may appear benign individually, posing a risk only when combined. This is precisely where existing safety filters are most likely to fail.

**Goal**: The paper aims to systematically characterize the jailbreak risks introduced by multi-image inputs, evaluate success rates across different MLLMs, datasets, and defense mechanisms, and analyze whether the success stems from the "increase in image count" or modular information composition.

**Key Insight**: The authors view multi-image jailbreaking as a compositional problem: dispersing instructions across images reduces intent visibility, multimodal evidence increases response detail, and auxiliary reasoning tasks distract safety attention. Superimposing these three factors exposes the model's weakness in cross-image safety reasoning.

**Core Idea**: Instead of simply repeating a single dangerous image, different images are assigned different functions, and a text prompt requires the model to process them jointly. This tests whether MLLMs can recognize the overall risk after cross-image composition.

## Method
DMN is a safety evaluation framework rather than a conventional generative model. It constructs a sequence of images and a text prompt in a single-turn black-box setting to observe if the target MLLM outputs a response judged harmful by an evaluator. To avoid translating the method into executable operations, only the research-level modular mechanisms and evaluation logic are preserved.

### Overall Architecture
Given an original harmful request, DMN generates multiple images with different roles: some images disperse textual fragments of the request, some provide scenario-related multimodal evidence, and others form a number chain task requiring additional parsing. The target MLLM receives this set of images and a general text prompt. Researchers then use a judge like GPT-4 to determine if the response crosses safety boundaries. The paper evaluates 10 multi-image-supporting MLLMs on SafeBench, HADES, and MM-SafetyBench, comparing with methods like FigStep, CS-DJ, HADES, QRA, and VideoJail.

### Key Designs
1.  **Distributed Instruction**:
    - **Function**: Decomposes the original harmful intent into multiple typographic images to reduce the explicit risk of any single image.
    - **Mechanism**: Each image carries only partial words or fragments of the instruction; the model can only recover the full semantics after cross-image aggregation. The paper compares this with single-image instruction to test if "dispersed presentation" bypasses safety mechanisms more easily.
    - **Design Motivation**: Many safety filters are better at identifying explicit harmful intents within a single image or text segment but fail at global semantic detection for cross-image combinations.

2.  **Multimodal Evidence**:
    - **Function**: Constructs richer visual and textual backgrounds for the request, making it easier for the target model to generate detailed responses.
    - **Mechanism**: The paper packages image generation as a construction process for real-world case materials. An auxiliary LLM first generates scenario-based evidence, which is then converted into image descriptions for T2I models. Prompt rewriting is used if generation fails.
    - **Design Motivation**: Directly asking T2I models to generate harmful images often results in refusal; using a case-based, indirect evidence generation process improves image generation success rates and provides more context.

3.  **Number Chain Task**:
    - **Function**: Adds an extra cross-image reasoning task to distract the model's attention and increase input processing complexity.
    - **Mechanism**: Each number chain frame contains a number and an index pointing to the next frame. The model is required to recover the number chain in the specified order. The paper compares blank frame indexing, cat/dog frame indexing, and number chains, finding that higher information density per frame correlates with higher ASR.
    - **Design Motivation**: Complex auxiliary tasks may occupy the model's cross-image attention, weakening its overall safety scrutiny of the integrated semantics.

### Loss & Training
DMN does not train the target MLLM nor does it require access to model parameters; it is a single-turn black-box evaluation. Implementation-wise, it uses Gemini-2.5-flash as the auxiliary LLM and GPT Image 1 as the T2I model, typically generating 5 pairs of multimodal evidence and inserting 5 number chain frames. The evaluation metric is the Attack Success Rate (ASR), the proportion of responses judged harmful. Other evaluation methods are used for bias checking.

## Key Experimental Results

### Main Results
| Dataset / Model Group | Metric | DMN | Strongest or Main Baseline | Gain / Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| SafeBench, Avg. of 10 MLLMs | ASR | 89.32% | CS-DJ 30.18%, FigStep 20.22% | Multi-image composition significantly outperforms single-image structural attacks |
| HADES dataset, Avg. of 10 MLLMs | ASR | 93.09% | VideoJail 8.77%, HADES method 4.53% | Redundant video frames are insufficient; compositional information is key |
| MM-SafetyBench, Avg. of 10 MLLMs | ASR | 86.24% | QRA 21.30% | High success rate maintained across datasets |
| GPT-4o / SafeBench | ASR | 92.8% | FigStep 19.8%, CS-DJ 42.6% | Strong closed-source models still expose multi-image safety vulnerabilities |
| Gemini-2.5-pro / SafeBench | ASR | 95.2% | FigStep 18.8%, CS-DJ 27.2% | High capability does not equate to multi-image safety |
| Claude Sonnet 4 / SafeBench | ASR | 94.2% | FigStep 13.0%, CS-DJ 39.6% | Safety-aligned models are also affected by compositional inputs |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Plain text | ASR 7.32% | Low success rate with text alone |
| DI | ASR 51.58% | Distributed instructions significantly increase ASR independently |
| DI + ME | ASR 79.40% | Further improvement after adding multimodal evidence |
| DI + ME + NC | ASR 89.32% | Full DMN is the strongest |
| DI + ME + NC padding control | Similar to original | Gains come from modular functionality, not the count of blank images |
| Multi-image-aware filter | ASR reduced to 28.86% | Filters explicitly warning of cross-image jailbreak risks are most effective |

### Key Findings
- The image generation success rate reflects DMN's indirectness: In a fair setting with single attempts, DMN achieved 93.36% and 96.68% success rates on HADES and SafeBench using GPT Image 1, significantly higher than QRA and the HADES method.
- Number chain task complexity is positively correlated with ASR: As PFIR increased from 1 to 3, ASR rose from 83.18% to 89.32%, indicating that cognitive load from auxiliary tasks is a critical variable.
- Existing defenses only moderately reduce ASR: Self-Reminder stays at 72.02%, Adashield-S at 65.20%, ECSO at 66.18%, and QwenGuard at 78.46%.

## Highlights & Insights
- The true value of the paper lies not in proposing another single jailbreak trick, but in decomposing multi-image input into three ablatable modules: "dispersed intent, supplementary evidence, and increased cognitive load," making safety weaknesses more analyzable.
- The padding control is vital: it rules out the explanation that it is "just because there are more images," proving that functional image composition is key.
- The results of the multi-image-aware filter suggest that defense should not rely solely on better OCR or single-image detection, but should explicitly model cross-image compositional semantics within the model or pre-filter.

## Limitations & Future Work
- The authors acknowledge that DMN is only applicable to MLLMs that support multi-image inputs and has limited applicability to single-image models or web interfaces with strict image upload limits.
- DMN requires more processing time, input tokens, and image generation costs than single-image methods, resulting in higher actual evaluation overhead.
- Part of the analysis relies on attention metrics from open-source models, such as KFAR, which may not fully represent the internal attention mechanisms of closed-source commercial models.
- A promising future direction is defense benchmarks: e.g., training/evaluating safety judges that can aggregate risks across images, or conducting image-group-level safety summarization and conflict detection at the MLLM input stage.

## Related Work & Insights
- **vs FigStep / typographic jailbreak**: FigStep primarily places harmful instructions in a single image; DMN splits instructions across multiple images to test cross-image aggregation safety.
- **vs QRA / HADES image-based attacks**: These methods rely on a single relevant image; DMN constructs multiple images through realistic case-based evidence, offering higher information density and generation success rates.
- **vs VideoJail**: While VideoJail uses multi-image/video formats, redundancy between frames is high; each category of image in DMN serves a different function, thus better exposing multi-image compositional risks.
- **Insights**: Multimodal safety evaluation should move beyond single-image prompts to construct compositional tests across images, tasks, and semantic levels, especially for commercial MLLMs supporting multi-image input.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The multi-image compositional framework and modular ablation are clear; the research question identifies the safety blind spots created by new MLLM capabilities.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 datasets, 10 MLLMs, multiple baselines, defenses, and ablations, providing comprehensive evidence.
- Writing Quality: ⭐⭐⭐⭐☆ Direct structure with ample experimental tables; as it contains safety-sensitive content, one must distinguish between research evaluation and actionable details.
- Value: ⭐⭐⭐⭐☆ Highly cautionary for multi-image MLLM safety assessment and provides a clear direction for defense design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Jailbreaking Multimodal Large Language Models using Multi-Clip Video](jailbreaking_multimodal_large_language_models_using_multi-clip_video.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](../../AAAI2026/multimodal_vlm/exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)

</div>

<!-- RELATED:END -->
