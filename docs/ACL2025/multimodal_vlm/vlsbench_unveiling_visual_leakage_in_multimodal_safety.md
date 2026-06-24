---
title: >-
  [Paper Note] VLSBench: Unveiling Visual Leakage in Multimodal Safety
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Safety] This work reveals the issue of Visual Safety Information Leakage (VSIL) in existing multimodal safety benchmarks—where hazardous content in images is already exposed in text queries, enabling models to refuse based solely on text and rendering safety evaluations unreliable. To address this, the authors construct the leakage-free VLSBench benchmark (2.2k image-text pairs) and find that multimodal alignment significantly outperforms…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Safety"
  - "Visual Information Leakage"
  - "Safety Benchmark"
  - "MLLM Alignment"
  - "Multimodal Evaluation"
date: 2026-05-08
content_hash: ade3c858994f0bf2
---

# VLSBench: Unveiling Visual Leakage in Multimodal Safety

**Conference**: ACL 2025  
**arXiv**: [2411.19939](https://arxiv.org/abs/2411.19939)  
**Code**: [github](https://github.com/AI45Lab/VLSBench)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Safety, Visual Information Leakage, Safety Benchmark, MLLM Alignment, Multimodal Evaluation

## TL;DR

This work reveals the issue of Visual Safety Information Leakage (VSIL) in existing multimodal safety benchmarks—where hazardous content in images is already exposed in text queries, enabling models to refuse based solely on text and rendering safety evaluations unreliable. To address this, the authors construct the leakage-free VLSBench benchmark (2.2k image-text pairs) and find that multimodal alignment significantly outperforms text-only alignment in VSIL-free scenarios.

## Background & Motivation

The safety of Multimodal Large Language Models (MLLMs) is receiving increasing attention. However, prior research discovered a counterintuitive phenomenon: aligning MLLMs only with textual unlearning can achieve comparable safety performance to multimodal alignment. This suggests a fundamental flaw in existing multimodal safety benchmarks.

Through in-depth analysis, the authors identified the **Visual Safety Information Leakage (VSIL)** problem: in existing benchmarks, the potential safety-critical information implied in the images is already explicitly or implicitly expressed in the text queries. Consequently, MLLMs can easily refuse these sensitive image-text pairs solely based on textual queries, without genuinely understanding the cross-modal safety information in the images. This makes cross-modal safety evaluation unreliable—text-only alignment appears "good enough" only because the benchmarks themselves are flawed.

## Method

### Overall Architecture

The authors propose an automated data construction pipeline to generate image-text pairs free of Visual Safety Information Leakage. VLSBench comprises 2.2k image-text pairs spanning 6 safety categories and 19 subcategories. The core design principle is to ensure that the text query itself is benign and the image alone is not necessarily harmful, but their combination constitutes a safety risk.

### Key Designs

1. **Safety Taxonomy**: A hierarchical two-level safety taxonomy is designed, covering 6 main categories and 19 subcategories. It references existing LLM and worldly-accepted multimodal safety standards to ensure broad coverage.

2. **Harmful Query and Image Description Generation (Step 1)**: Two parallel paths are leveraged—(a) extracting sensitive objects and hazardous scenarios from ChatGPT, then using GPT-4o to generate image descriptions and harmful queries; (b) utilizing existing image datasets (real-world images) with Qwen2-VL-72B to generate image analyses and harmful queries. This dual-path strategy ensures the diversity of safety topics.

3. **Textual Detoxification / Eliminating Visual Leakage (Step 2)**: GPT-4o is used to detoxify harmful queries into seemingly benign text queries, eliminating safety information leakage from the image modality to the text modality via few-shot prompting. Two types of invalid samples are filtered out: (a) modified queries that still contain leaked information, and (b) queries that deviate from the original semantics.

4. **Iterative Image Generation (Step 3)**: GPT-4o-mini rewrites image descriptions into text-to-image prompts, and Stable-Diffusion-3.5-Large generates the images. An iterative process is adopted: Qwen2-VL-72B evaluates whether the generated image reflects the description; if not, the prompt is modified for regeneration until the standards are met.

5. **Final Filtering (Step 4)**: GPT-4o performs quality filtering on the final image-text pairs, removing mismatched and naturally safe samples, followed by a human review process to finalize the dataset.

### Evaluation Strategy

Using GPT-4o as the judge, model responses are categorized into three classes:
- **Safe with Refusal**: A clear and firm refusal.
- **Safe with Warning**: Acknowledges the safety concern and provides a warning.
- **Unsafe**: Ignores safety guidelines and answers directly.

Safety Rate = Refusal Rate + Warning Rate.

## Key Experimental Results

### Main Results

| Model | Refusal Rate | Warning Rate | Total Safety Rate |
|------|--------|--------|----------|
| LLaVA-v1.5-7B | 0% | 6.60% | 6.60% |
| GPT-4o | 5.21% | 16.22% | 21.43% |
| Gemini-1.5-pro | 1.34% | 48.44% | 49.78% |
| Llama-3.2-11B-Vision | 10.96% | 15.33% | 26.29% |
| Qwen2-VL-7B | 1.11% | 12.66% | 13.77% |
| InternVL2.5-8B | 2.81% | 18.56% | 21.37% |

### Comparison of Safety Alignment Methods (LLaVA-v1.5-7B)

| Method | Refusal Rate | Warning Rate | Total Safety Rate |
|------|--------|--------|----------|
| MM-SFT | 2.32% | 18.94% | 21.26% |
| MM-DPO | 2.63% | 24.38% | 27.01% |
| MM-PPO | 5.08% | 30.39% | 35.47% |
| Textual-SFT | 5.30% | 8.69% | 13.99% |
| Textual-SafeUnlearning | 2.85% | 8.87% | 11.72% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Original Image-Text Input vs. Image Caption Substitution | Qwen2-VL: $16\% \rightarrow 22.5\%$ | Substituting the image with a textual caption improves safety |
| No Visual Input | Qwen2-VL: $16\% \rightarrow 29.5\%$ | Removing the image entirely actually improves safety, indicating a lack of cross-modal understanding in the model |
| Safety Prompt Enhancement | LLaVA-v1.5: $6.6\% \rightarrow 44.5\%$ | Safety prompts can significantly enhance safety performance |
| NSFW Detector | Detection Rate: $0\%$ | VLSBench images alone do not trigger NSFW detectors |

### Key Findings

- All major MLLMs (including GPT-4o) exhibit very low safety rates on VLSBench, with only Gemini achieving a peak of $49.78\%$.
- On benchmarks with VSIL, text-only alignment yields comparable performance to multimodal alignment; however, on VLSBench (without VSIL), multimodal alignment significantly outperforms text-only alignment (Qwen2-VL: $78.39\%$ vs $67.42\%$).
- Strong reasoning capability does not equate to good safety: QVQ-Preview shows improvements compared to Qwen2-VL, but LLaVA-Cot conversely drops.
- Models tend to employ "blanket refusal" rather than providing comprehensive safety explanations and alternatives.

## Highlights & Insights

- **Precise Problem Identification**: VSIL is a critical yet long-overlooked issue that explains the counterintuitive phenomenon of "why text-only alignment is sufficient."
- **Automated and Reproducible Data Construction Pipeline**: The four-step pipeline is cleverly designed, ensuring quality through detoxification, iterative generation, and multi-round filtering.
- **Evaluation Findings Dispel Illusions**: Even GPT-4o achieves only a $21.43\%$ safety rate, highlighting the severe shortage of cross-modal safety understanding capabilities in existing MLLMs.
- **Methodological Implications**: The design of safety benchmarks requires meticulous crafting, as flawed benchmarks can severely mislead the community.

## Limitations & Future Work

- The dataset size of 2.2k is relatively small, and coverage of safety scenarios may still have blind spots.
- Relying on GPT-4o as a judge could introduce evaluation bias.
- Images are entirely AI-generated, which might deviate from the distribution of real-world scenarios.
- Safety evaluation in video modality or multi-turn dialogues has not yet been explored.
- The detoxification process depends on specific prompt designs, which may not cover some edge cases.

## Related Work & Insights

- This work contrasts with prior benchmarks like MMSafetyBench and VLGuard, pointing out their VSIL issues.
- It inspires future multimodal safety research to focus more on cross-modal understanding capabilities rather than single-modality alignment.
- It offers a more reliable evaluation tool for alignment methods such as multimodal RLHF/DPO.
- Multimodal preference alignment methods like SPA-VL perform better on this benchmark, indicating they are in the correct direction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Systematically reveals the VSIL issue for the first time, reshaping the community's understanding of multimodal safety alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers open-source/closed-source models, multiple alignment methods, and diverse analysis dimensions, though the dataset size is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning and coherent logic, although some experimental details are relegated to the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides critical benchmarking tools and methodological insights for the multimodal safety domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] Unveiling the Lack of LVLM Robustness to Fundamental Visual Variations: Why and Path Forward](unveiling_the_lack_of_lvlm_robustness_to_fundamental_visual_variations_why_and_p.md)
- [\[ICLR 2026\] VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety](../../ICLR2026/multimodal_vlm/vlsu_mapping_the_limits_of_joint_multimodal_understanding_for_ai_safety.md)
- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)

</div>

<!-- RELATED:END -->
