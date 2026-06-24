---
title: >-
  [Paper Note] Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Concept Removal] This work proposes the Six-CD benchmark, containing six categories of undesired concepts (harmful, nudity, celebrity, copyrighted character, object, and artistic style) and a new evaluation metric, the in-prompt CLIP score, to systematically evaluate and compare concept removal methods in text-to-image diffusion models for the first time.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Concept Removal"
  - "Diffusion Model Safety"
  - "Benchmark Evaluation"
  - "NSFW Filtering"
  - "Text-to-Image Models"
date: 2026-05-08
content_hash: b32c0b02103608b2
---

# Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2406.14855](https://arxiv.org/abs/2406.14855)  
**Code**: None  
**Area**: Image Generation / AI Safety  
**Keywords**: Concept Removal, Diffusion Model Safety, Benchmark Evaluation, NSFW Filtering, Text-to-Image Models

## TL;DR

This work proposes the Six-CD benchmark, containing six categories of undesired concepts (harmful, nudity, celebrity, copyrighted character, object, and artistic style) and a new evaluation metric, the in-prompt CLIP score, to systematically evaluate and compare concept removal methods in text-to-image diffusion models for the first time.

## Background & Motivation

While text-to-image diffusion models (such as Stable Diffusion) can generate high-quality images, they can also be maliciously exploited to generate violent or nude images, or portraits of public figures in inappropriate scenarios. Concept removal methods, which modify model parameters to prevent the generation of specific undesired concepts, serve as a critical technology for ensuring model safety.

However, existing research faces three core challenges: 
(1) **Lack of consistent and comprehensive comparison**: Different methods are evaluated only on limited categories (e.g., ESD only tests objects and artistic styles, while SPM only tests celebrities), lacking a unified benchmark across methods. 
(2) **Ineffective prompt issue**: A large number of prompts in existing datasets (such as I2P) rarely trigger malicious content generation. Evaluating concept removal on these "ineffective prompts" is meaningless and leads to unfair comparisons with specific concepts (e.g., celebrity identity prompts which possess high effectiveness). 
(3) **Neglecting in-prompt preservation capability**: When the "Mickey Mouse" concept is removed from "Mickey Mouse is eating a burger," the generated image should still contain "eating a burger." Existing evaluations completely overlook this capability of retaining benign semantics.

Six-CD systematically addresses these issues by constructing a comprehensive dataset, filtering an effective prompt subset, and introducing a new evaluation metric.

## Method

### Overall Architecture

The Six-CD benchmark consists of three core components: (1) a comprehensive dataset covering six categories of undesired concepts; (2) an effective prompt filtering subset for general concepts (harmful/nudity); and (3) a new evaluation metric, the in-prompt CLIP score, to measure the preservation of benign semantics after concept removal. The evaluation covers three categories of methods: gradient descent-based (ESD, SPM, SDD, FMN), closed-form solution-based (UCE, MACE, EMCID), and inference-time (SLD, SEGA).

### Key Designs

1. **Six-Category Concept Classification System**:
    - **Function**: Establishes a comprehensive and hierarchical classification of undesired concepts.
    - **Mechanism**: Classifies undesired concepts into two main categories: **general concepts** (harmful to all users) and **specific concepts** (infringing on the rights of specific entities). General concepts include harmful content (violence, suicide, hate, etc.) and nudity; specific concepts include celebrity identities, copyrighted characters (e.g., Mickey Mouse), objects (e.g., specific brands), and artistic styles. Data sources are compiled from four NSFW resources (I2P, MMA, SD-uncensored, Unsafe Diffusion), with fine-grained labels automatically annotated using NudeNet and Q16 classifiers.
    - **Design Motivation**: Prior works evaluated different categories individually, preventing cross-method comparison. The proposed six-category taxonomy covers all types of undesired concepts appearing in the literature.

2. **Effective Prompt Filtering**:
    - **Function**: Constructs high-effectiveness prompt subsets for general concepts to improve evaluation efficiency and fairness.
    - **Mechanism**: Defines prompt effectiveness as the probability of a prompt triggering the model to generate malicious content. Experiments show that prompt effectiveness for general concepts (harmful/nudity) is much lower than that for specific concepts (celebrities, etc.) due to the more implicit and diverse nature of general category prompts. Therefore, an additional high-effectiveness prompt subset is curated for general concepts, retaining only prompts that trigger malicious generation with high probability.
    - **Design Motivation**: Evaluating concept removal on ineffective prompts is meaningless (as the original model would not generate malicious content anyway) and leads to unfair comparisons between general and specific concepts.

3. **In-prompt CLIP Score Metric**:
    - **Function**: Measures the model's capability to preserve benign semantics within the prompt after concept removal.
    - **Mechanism**: Constructs a Dual-Version dataset where each prompt has a malicious version and a benign version (with the undesired concept removed but remaining semantics preserved). The concept removal method is applied to the malicious version to generate an image, and then the CLIP score between the generated image and the benign text version is computed. A successful concept removal method should achieve a high in-prompt CLIP score, indicating that benign semantics are preserved.
    - **Design Motivation**: Aggressive concept removal may eliminate benign semantics as well, resulting in model outputs completely unrelated to the prompt, which is also unacceptable.

### Loss & Training

- As this work is a benchmark evaluation, it does not propose a new training method.
- Evaluated concept removal methods include:
    - Gradient descent-based: ESD, SPM, SDD, FMN
    - Closed-form solution-based: UCE, MACE, EMCID
    - Inference-time: SLD, SEGA
- Evaluation metrics: Concept removal effectiveness (FID, detector accuracy) + benign concept preservation (FID) + in-prompt CLIP score.

## Key Experimental Results

### Main Results

General concept removal efficacy (on the effective prompt subset):

| Method | Type | Harmful Removal Rate↑ | Nudity Removal Rate↑ | Benign Preservation↑ | In-prompt CLIP↑ |
|------|------|-----------|-----------|---------|----------------|
| ESD | Gradient Descent | High | High | Medium | Low |
| SPM | Gradient Descent | Medium | Medium | High | High |
| UCE | Closed-form | Medium | Medium | Medium | Medium |
| MACE | Closed-form | High | High | Low | Low |
| SLD | Inference-time | Medium | Low | High | High |

### Ablation Study

| Analysis Dimension | Findings |
|---------|------|
| Effective vs. Ineffective Prompts | Concept removal on ineffective prompts "appears" effective but is actually meaningless. |
| General vs. Specific Concepts | Specific concepts are easier to remove (more precise prompts), while general concepts are more difficult. |
| Single vs. Multi-concept Removal | Performance drops significantly when removing multiple categories of concepts simultaneously. |
| Removal Aggressiveness | Overly aggressive methods (e.g., ESD) severely damage in-prompt preservation capability. |

### Key Findings

- There is a **fundamental trade-off** between concept removal efficacy and benign semantic preservation—more thorough removal leads to poorer preservation.
- General concepts (harmful/nudity) are harder to remove than specific concepts (celebrity/copyrighted characters) because their prompts are more implicit and diverse.
- Closed-form methods (UCE, MACE) work well on specific concepts but are unstable on general concepts.
- Inference-time methods (SLD) skip fine-tuning but can be easily disabled by open-source users.
- Evaluation results differ greatly between the effective prompt subset and the full set, highlighting the importance of prompt filtering.

## Highlights & Insights

- **Proposing in-prompt preservation fills an evaluation blind spot**: Previous works only focused on "how much malicious content is removed" and "how well benign prompt generation is preserved," completely ignoring "whether benign semantics within malicious prompts are retained." This is the true key to user experience in practical deployments.
- **Insights into effective prompt filtering are highly practical**: Finding that a large number of prompts in the I2P dataset do not trigger malicious generation means that evaluating on them is self-deception. This finding prompts the research community to re-examine existing evaluation results.

## Limitations & Future Work

- Evaluation is only conducted on the Stable Diffusion series, without covering closed-source models like DALL-E and Midjourney.
- The classification of concepts (harmful/nudity) relies on automatic classifiers (NudeNet, Q16), which may introduce annotation noise.
- Specific concepts use template-based prompts, which may not fully reflect the diverse expressions of real users.
- The long-term impact of concept removal methods on the overall generation quality (e.g., FID) of the model remains unexplored.

## Related Work

- **vs. I2P Dataset (SLD)**: I2P contains many ineffective prompts and only covers nudity/harmful categories; Six-CD covers six categories and filters for effective prompts.
- **vs. ESD**: ESD achieves good removal results but poor in-prompt preservation on this benchmark, indicating its method is overly aggressive.
- **vs. SPM**: SPM performs best in preservation but yields relatively mild removal effects, making it suitable for scenarios requiring balance.

## Rating

- Novelty: ⭐⭐⭐⭐ The first comprehensive concept removal benchmark; the proposed in-prompt CLIP score fills an evaluation gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed evaluation comprising 9 methods across 6 concept categories with multiple metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation with a well-structured approach that solves the three gaps step-by-step.
- Value: ⭐⭐⭐⭐ Provides a much-needed standardized evaluation framework for concept removal research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[CVPR 2025\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fade_fine_grained_erasure_diffusion.md)
- [\[CVPR 2025\] GRADE: Benchmarking Discipline-Informed Reasoning in Image Editing](grade_benchmarking_discipline-informed_reasoning_in_image_editing.md)
- [\[ICLR 2026\] Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models](../../ICLR2026/image_generation/everything_in_its_place_benchmarking_spatial_intelligence_of_text-to-image_model.md)

</div>

<!-- RELATED:END -->
