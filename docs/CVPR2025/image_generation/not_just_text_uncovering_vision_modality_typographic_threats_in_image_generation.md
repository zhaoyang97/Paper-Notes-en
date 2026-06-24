---
title: >-
  [Paper Note] Not Just Text: Uncovering Vision Modality Typographic Threats in Image Generation Models
description: >-
  [CVPR 2025][Image Generation][Typographic Attacks] This paper exposes a "typographic attack" vulnerability in the vision modality of image generation models—where attackers can manipulate generation results by embedding text into input images. It systematically evaluates the ineffectiveness of existing defenses against such vision modality threats and proposes the VMT-IGMs dataset as an evaluation benchmark.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Typographic Attacks"
  - "Vision Modality Safety"
  - "Adversarial Attacks"
  - "Safety Defense Evaluation"
date: 2026-05-08
content_hash: b2af0ded482cab10
---

# Not Just Text: Uncovering Vision Modality Typographic Threats in Image Generation Models

**Conference**: CVPR 2025  
**arXiv**: [2412.05538](https://arxiv.org/abs/2412.05538)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Typographic Attacks, Vision Modality Safety, Image Generation, Adversarial Attacks, Safety Defense Evaluation

## TL;DR
This paper exposes a "typographic attack" vulnerability in the vision modality of image generation models—where attackers can manipulate generation results by embedding text into input images. It systematically evaluates the ineffectiveness of existing defenses against such vision modality threats and proposes the VMT-IGMs dataset as an evaluation benchmark.

## Background & Motivation

**Background**: Current image generation models (Text-to-Image, Image-to-Image) can easily generate high-quality and highly realistic images. To mitigate the safety risks of these models, the research community has proposed numerous defense strategies, but almost all of them focus on the language modality—specifically filtering or censoring inappropriate content in text inputs.

**Limitations of Prior Work**: In practical applications, particularly tasks involving real image editing (e.g., Image-to-Image), attackers can manipulate generation results by modifying input images. This vision-modality threat carries higher security risks as it can easily infringe upon the rights of image owners, such as modifying someone's photo into inappropriate content. However, existing safety research has largely ignored this attack surface.

**Key Challenge**: Defense strategies rely excessively on safety checks of the language modality (e.g., text classifiers, prompt filtering), but image generation models accept vision inputs simultaneously, a channel with almost no safety defenses. Attackers can completely bypass text-level defenses and inject malicious instructions directly through the visual channel.

**Goal**: (1) Systematically uncover the vulnerability of image generation models to the vision modality; (2) evaluate the effectiveness of existing defenses against vision modality attacks; (3) provide unified datasets and benchmarks to advance research in this direction.

**Key Insight**: "Typographic attacks" in the NLP field have been shown to deceive vision-language models like CLIP—overlaying text on an image can alter the model's understanding of the image. The authors transfer this attack concept to the image generation domain, exploring whether embedding typographic content in the input image can manipulate the output of generation models.

**Core Idea**: To exploit typographic attacks (adding text overlays to input images) to uncover safety vulnerabilities in the vision modality of image generation models, and systematically evaluate the blind spots of existing defense schemes.

## Method

### Overall Architecture
The study is divided into three stages: (1) Attack Construction—designing typographic attack methods targeting the visual input channel of image generation models; (2) Vulnerability Assessment—testing the attack effectiveness on various mainstream image generation models; (3) Defense Evaluation—testing the capability of existing safety methods to defend against vision modality attacks. Finally, the VMT-IGMs dataset is constructed for community research purposes.

### Key Designs

1. **Typographic Attack on Vision Modality**:

    - **Function**: To manipulate the output of the generation model by modifying the input image without altering the text prompt.
    - **Mechanism**: Superimpose specific text content (e.g., inappropriate words, instructional text) onto the input image, leveraging the internal vision-language alignment capability (e.g., CLIP encoder) of image generation models. This guides the model to produce outputs that diverge from the original prompt when processing images with embedded text. The attack process does not require access to model weights (black-box attack), only the ability to modify the input image. Parameters such as typography position, size, font, and color can be adjusted to optimize attack effectiveness.
    - **Design Motivation**: Modern image generation models widely utilize vision-language models such as CLIP as conditional encoders, which are highly sensitive to text content within images. Typographic attacks exploit precisely this vulnerability in cross-modal alignment—models cannot distinguish between "normal image content" and "maliciously embedded text."

2. **Multi-Model Systematic Vulnerability Assessment**:

    - **Function**: To comprehensively evaluate the effects of typographic attacks across different image generation models and tasks.
    - **Mechanism**: Select multiple representative image generation models, covering both Text-to-Image (e.g., Stable Diffusion) and Image-to-Image (e.g., InstructPix2Pix, ControlNet) tasks. For each model, test different types of typographic content (inappropriate content, identity replacement, style modification, etc.) and various attack parameters. Automated metrics (such as CLIP similarity change, content detector scores) and human evaluation are used to quantify the attack success rate.
    - **Design Motivation**: Different model architectures and training methods may exhibit varying sensitivity to typographic attacks; systematic evaluation can reveal which design choices render models more vulnerable.

3. **VMT-IGMs Dataset & Defense Evaluation**:

    - **Function**: To provide a standardized benchmark dataset and reveal the limitations of existing defense strategies.
    - **Mechanism**: Build the Vision Modality Threats in Image Generation Models (VMT-IGMs) dataset, which contains meticulously designed attack samples and their corresponding generated results. Evaluate several existing safety protection methods on this dataset, including input filtering, output detection, and safety filters. Experimental results demonstrate that existing defense mechanisms perform poorly against vision modality attacks.
    - **Design Motivation**: Safety research requires standardized evaluation benchmarks. VMT-IGMs fills the vacancy of vision modality safety evaluation datasets, helping the community develop more comprehensive defense strategies.

### Loss & Training
This paper is an analytical work and does not involve model training. Typographic attacks are gradient-free black-box methods achieved by overlaying text onto the image surface, which do not require any optimization of the target model.

## Key Experimental Results

### Main Results

| Model | Task Type | Attack Success Rate | Impact on Generation Quality |
|------|---------|-----------|------------|
| Stable Diffusion (I2I) | Image Editing | High | Significant deviation from original intent |
| InstructPix2Pix | Instruction Editing | High | Generation content controlled by typography |
| ControlNet | Conditional Generation | Medium-High | Visual condition is overridden |
| DALL-E Series | T2I + Visual Condition | Medium | Effective in some scenarios |

### Defense Evaluation

| Defense Method | Text Attack Protection | Visual Attack Protection | Description |
|---------|-----------|------------|------|
| Prompt Filtering | Effective | Ineffective | Completely bypasses the visual channel |
| Safety Filter (Output) | Partially Effective | Limited | Only detects explicitly inappropriate content |
| NSFW Detector | Effective | Limited | Ineffective against implicit manipulation |
| Input Image Inspection | Not Applicable | Limited | Typography can be optimized to be less detectable |

### Key Findings
- Almost all mainstream image generation models are vulnerable to typographic attacks on the vision modality, including commercial models with deployed safety mechanisms.
- Among existing defense methods, filtering specifically targetting text prompts is completely ineffective against visual attacks—attackers do not even need to use any text prompts.
- Output-based safety detectors also show limited effectiveness against typographic attacks, because the attacks can produce outputs that appear normal but have manipulated semantics.
- The size and position of the embedded text in the image have a significant impact on attack effectiveness; larger and centered text achieved higher attack success rates.

## Highlights & Insights
- **Systemic Exposure of Safety Blind Spots**: The core contribution of this paper does not lie in proposing a new method, but in systematically exposing an overlooked yet critical safety issue—the visual channel becoming a "backdoor" for safety defenses. This has direct security implications for deployed image generation services.
- **Alarming Simplicity of the Attack**: Typographic attacks require no gradient calculations or model internal access; they succeed simply by overlaying text onto the image, implying an extremely low attack barrier and significant practical threat.
- **VMT-IGMs as a Standard Benchmark**: Fills the gap in vision-modality safety evaluations, providing a comparable benchmark for subsequent defense research.

## Limitations & Future Work
- The paper primarily exposes the problem and evaluates existing defenses but does not propose an effective defense scheme.
- The scope of typographic attacks may be limited—in certain application scenarios (such as pure T2I without image input), vision modality attacks are not applicable.
- There is a lack of in-depth analysis on why certain models are more susceptible to typographic attacks—whether it is an issue with CLIP or the generative models themselves.
- Future work can explore: (1) training vision encoders robust against typography; (2) OCR preprocessing and text erasure of input images; (3) multimodal joint safety inspection mechanisms.

## Related Work & Insights
- **vs Textual Adversarial Attacks**: Traditional adversarial prompt attacks manipulate models at the text level, for which many defenses have been proposed. This paper reveals that visual-level attacks represent a brand-new and undefended attack surface.
- **vs CLIP Typographic Attacks**: Previous studies have demonstrated that typography can deceive CLIP image classification. This paper generalizes this finding to the image generation domain, demonstrating more severe practical security risks.
- **vs Visual Adversarial Perturbations**: Traditional adversarial perturbations (such as FGSM) require gradient access, and the perturbations are imperceptible. Typographic attacks, though visible, are simpler and remain effective against black-box models.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically study vision modality safety threats in image generation models from a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models and defense methods, with the VMT-IGMs dataset being a significant contribution.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and comprehensive safety threat analysis.
- Value: ⭐⭐⭐⭐ Directly guides the safe deployment of image generation systems, though slightly limited by the lack of proposed defense solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MINIMA: Modality Invariant Image Matching](minima_modality_invariant_image_matching.md)
- [\[CVPR 2025\] Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability](not_all_parameters_matter_masking_diffusion_models_for_enhancing_generation_abil.md)
- [\[CVPR 2025\] Yo'Chameleon: Personalized Vision and Language Generation](yochameleon_personalized_vision_and_language_generation.md)
- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] Scaling Down Text Encoders of Text-to-Image Diffusion Models](scaling_down_text_encoders_of_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
