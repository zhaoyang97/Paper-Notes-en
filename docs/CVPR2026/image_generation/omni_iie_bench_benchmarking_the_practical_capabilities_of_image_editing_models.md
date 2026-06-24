---
title: >-
  [Paper Note] Omni IIE Bench: Benchmarking the Practical Capabilities of Image Editing Models
description: >-
  [CVPR 2026][Image Generation][Image Editing] Omni IIE Bench is a high-quality human-annotated benchmark **specifically designed to diagnose the "consistency of instruction-based image editing models across semantic scales."** Using a dual-track design consisting of "single-turn consistency + multi-turn coordination (up to 16 turns)," 2856 samples were constructed from 12 data sources through a three-stage process (auto-generation → auto-masking → multi-pass rigorous human rev…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Image Editing"
  - "Diagnostic Benchmark"
  - "Semantic Scale"
  - "Multi-turn Editing"
  - "Decoupled Evaluation"
date: 2026-05-08
content_hash: b633174fc1179995
---

# Omni IIE Bench: Benchmarking the Practical Capabilities of Image Editing Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Omni_IIE_Bench_Benchmarking_the_Practical_Capabilities_of_Image_Editing_CVPR_2026_paper.html)  
**Code**: https://github.com/Young-2000/OmniIIEBench  
**Area**: Image Generation / Instruction-based Image Editing / Evaluation Benchmarks  
**Keywords**: Image Editing, Diagnostic Benchmark, Semantic Scale, Multi-turn Editing, Decoupled Evaluation

## TL;DR
Omni IIE Bench is a high-quality human-annotated benchmark **specifically designed to diagnose the "consistency of instruction-based image editing models across semantic scales."** Using a dual-track design consisting of "single-turn consistency + multi-turn coordination (up to 16 turns)," 2856 samples were constructed from 12 data sources through a three-stage process (auto-generation → auto-masking → multi-pass rigorous human review). A decoupled evaluation framework (global quality + foreground/background fidelity + instruction compliance) was proposed, quantifying for the first time a universal failure mode: almost all mainstream editing models suffer significant performance degradation when switching from low to high semantic scales, which further collapses in multi-turn scenarios due to error accumulation.

## Background & Motivation

**Background**: Instruction-based Image Editing (IIE) allows users to iteratively modify images using natural language, becoming a core tool in the design field. Early benchmarks focused on single-turn dialogues (given one instruction, can it be edited correctly?), mainly categorizing instructions into "attribute modification" and "entity modification." Recent works have explored multi-turn editing, but typically only reach 2–3 turns.

**Limitations of Prior Work**: (1) To ensure coverage, single-turn benchmarks often **mix various tasks into a single total score**, which masks a fatal failure mode in professional applications—model inconsistency across **tasks of different semantic scales**. Although I2EBench separates high/low levels, it remains an isolated checklist-style report and fails to diagnose stability when "switching scales within the same image." (2) Multi-turn benchmarks mostly focus on temporal consistency with very short sequences (2–3 turns), disconnecting from the "frequent, progressive, multi-iteration" workflows in real-world design. (3) Almost no benchmarks have been validated by **senior designers**, leading to a significant gap between "high benchmark scores" and "actual utility."

**Key Challenge**: Existing evaluations treat "task breadth" as the goal, using mixed metrics to boost coverage. Consequently, **the model's inconsistency across semantic scales is averaged out**, yet this inconsistency is precisely the failure point that professional applications care about most.

**Goal**: To build a diagnostic benchmark rooted in real design practices to quantify IIE models' (a) consistency across scales within a single turn and (b) coordination and resistance to error accumulation over long multi-turn sequences.

**Key Insight**: The authors explicitly define "semantic scale" as editing granularity—**low scale = attribute modification** (local, concentrated mask area), **high scale = entity replacement** (large impact, widely distributed mask area), and design diagnostic task pairs around "switching scales in a shared image context."

**Core Idea**: Use a dual-track diagnosis of "single-turn consistency + multi-turn coordination," combined with zero-tolerance multi-pass human filtering and decoupled evaluation, to transform "consistency across semantic scales" into a quantifiable medical-grade diagnostic metric.

## Method

### Overall Architecture
Omni IIE Bench is not an algorithm but a "dataset + evaluation protocol." **On the data side**, seed images were sampled from 12 public datasets to build independent seed pools for the dual tracks (2,400 for single-turn / 696 for multi-turn). Construction followed three stages: Stage 1 used GPT-4o to generate fine-grained descriptions and instructions, and Nano Banana to generate target images; Stage 2 used GPT-4o to parse core entity names from instructions, which were fed into Grounded-SAM (GroundingDINO for boxes + SAM for masks) to generate GT masks; Stage 3 involved rigorous multi-pass human review (four-dimensional image quality scoring + industry relevance/aesthetic review + mask semantic alignment review). Finally, 2,856 samples were selected using a "double zero-tolerance" standard, each represented as a quadruple $(I_{\text{source}}, T_{\text{mod}}, I_{\text{gt}}, M_{\text{gt}})$. **On the evaluation side**, for each model output $I_{\text{gen}}$, images are resized to 768×768. The GT mask is used to split the image into foreground $M_{\text{fg}}=M_{\text{gt}}$ and background $M_{\text{bg}}=1-M_{\text{gt}}$, with decoupled scoring across three dimensions: global quality, foreground/background fidelity, and instruction compliance. (As a benchmark paper, the pipeline is linear; no separate architecture diagram is provided.)

### Key Designs

**1. Dual-Track Diagnostic Design: Single-Turn Consistency + Multi-Turn Coordination**

To address the issue where mixed evaluations average out cross-scale inconsistencies, Ours does not report a single total score but designs two complementary tracks. **Single-turn consistency** consists of "shared-context task pairs"—giving pairs of instructions with opposite semantics (low-scale attribute modification vs. high-scale entity replacement) on the same image to directly compare performance gaps. **Multi-turn coordination** consists of continuous dialogues where instructions **dynamically alternate** between attribute and entity modifications, extending up to 16 turns (far exceeding the previous 2–3 turns). This examines context understanding, instruction coordination, and error accumulation. Semantic scale is operationalized via "mask area": statistics show attribute modifications concentrate on small areas while entity replacements have broader distributions, validating the scale division. The multi-track has an average depth of 4.35 turns, recording 322 attribute→entity and 178 entity→attribute scale switches across 1,131 turns.

**2. Three-Stage Construction Pipeline + Double Zero-Tolerance Human Filtering**

**Stage 1 (Auto-generation)**: GPT-4o generates fine-grained descriptions (objects/attributes/actions/environments) for seed images and creates instructions (opposite-semantic pairs for single-turn, scale-alternating sequences for multi-turn). Nano Banana generates high-fidelity GT images. **Stage 2 (Auto-masking)**: Feeding natural language directly to GroundingDINO is ineffective as it requires explicit entity nouns; thus, GPT-4o acts as an "instruction parser" to extract the core entity name, which Grounded-SAM then uses to produce binary masks $M_{\text{gt}}$. **Stage 3 (Rigorous Human Review)**: A 17-person team (12 CV students + 5 AI retouching designers) aligned standards for artifacts, semantic drift, and background contamination. Each sample was scored 1–3 across "instruction following/realism/background preservation/satisfaction" (independent initial scoring + 50% random second review + 5-person arbitration for disputes). Designers then performed aesthetic reviews, followed by Pass/Fail semantic alignment checks for masks. **Double zero-tolerance** was applied: only samples with perfect scores in all four dimensions **and** deemed industry-relevant were accepted. Only 35.94% (1,725) of single-turn and 37.36% (1,131) of multi-turn candidates passed, resulting in 2,856 high-quality samples filtered from 10,221.

**3. Decoupled Diagnostic Evaluation Framework: Global Quality + Regional Fidelity + Instruction Compliance**

Traditional single-score evaluations average out local errors. Ours decouples scoring: **Global Image Quality** uses PSNR/SSIM for pixel/structural fidelity, LPIPS for perceptual similarity, and CLIP Score for global semantics. **Decoupled Regional Fidelity** uses GT masks to calculate FG-LPIPS/FG-CLIP (editing fidelity) and BG-LPIPS/BG-CLIP (background consistency) separately. **Instruction Compliance** uses GPT-4o to generate 1–3 verifiable QA pairs (with GT answers) for each instruction; the generated image, question, and GT answer are then fed to GPT-4o as a judge. A **strict veto** is applied: a score of 1.0 is given only if all QA pairs are True; otherwise, it is 0.0. This specifically identifies models (like MGIE) that produce high-quality images but fail to follow instructions. The overall score is calculated as:
$$\frac{1}{4}\big[\frac{3-\Sigma\text{LPIPS}}{3}+\frac{\Sigma\text{CLIP}}{3}+\text{QA}+\text{SSIM}\big]$$
where $\Sigma\text{LPIPS}$ and $\Sigma\text{CLIP}$ are the sums of the FG/BG/ALL columns.

### A Complete Example
Using a seed image of a "cake on a white pedestal": In Stage 1, GPT-4o generates a low-scale instruction "change the cake to rainbow layers" and a high-scale instruction "change the cake to a bouquet of flowers" to form a shared-context pair. Nano Banana produces GT images. In Stage 2, GPT-4o extracts the entity "cake," and Grounded-SAM generates the mask. In Stage 3, the sample is kept only if it receives perfect human scores and designer validation. During evaluation, a model's outputs for both instructions are evaluated for FG/BG fidelity and QA compliance; if it succeeds on color changes but fails on entity replacement, it is quantified as "scale-inconsistent."

## Key Experimental Results

### Main Results (Single-turn, Overall selection)
Evaluations of 8 mainstream IIE models show Qwen-image-edit is the strongest, while HQEdit is the weakest. It also exposes models with high image quality but poor instruction compliance (MGIE).

| Model | LPIPS-ALL↓ | CLIP-ALL↑ | QA↑ | SSIM↑ | Overall↑ |
|------|-----------|-----------|-----|-------|----------|
| Qwen-image-edit | 0.450 | 0.889 | 0.744 | 0.455 | **0.687** |
| Step1X | 0.379 | 0.899 | 0.580 | 0.533 | 0.680 |
| ICEdit | 0.425 | 0.868 | 0.453 | 0.507 | 0.626 |
| FLUX | 0.552 | 0.868 | 0.636 | 0.375 | 0.614 |
| InstructPix2Pix | 0.569 | 0.841 | 0.316 | 0.438 | 0.530 |
| MGIE | 0.426 | 0.859 | **0.070** | 0.480 | 0.520 |
| HQEdit | 0.689 | 0.694 | 0.322 | 0.304 | **0.457** |

Note: MGIE has decent quality metrics but a QA score of only 0.070—the output is almost identical to the original image, failing to follow instructions. This failure is caught by Ours but would be missed by mixed evaluations.

### Comparison with Existing Benchmarks

| Benchmark | Human Verified | Mask Provided | Realistic Scenarios | Semantic Scale | Max Turns |
|------|---------|----------|---------|---------|---------|
| I2EBench | ✓ | ✓ | ✗ | ✓ | 1 |
| CompBench | ✓ | ✓ | ✗ | ✗ | 2 |
| MagicBrush / ImgEdit-Bench | ✓ | ✗ | ✗ | ✗ | 3 |
| MuCIE | ✗ | ✗ | ✗ | ✗ | 5 |
| **Omni IIE Bench** | ✓ | ✓ | **✓** | ✓ | **16** |

### Multi-turn vs Single-turn (Error Accumulation)

| Model | Single-turn Overall | Multi-turn Overall | Change |
|------|------------|------------|------|
| Qwen-image-edit | 0.687 | 0.676 | Slight drop |
| Step1X | 0.680 | 0.654 | Drop |
| MGIE | 0.520 | 0.404 | Collapse |

### Key Findings
- **Universal Drop Across Semantic Scales**: Almost all models show significant performance degradation when switching from low to high semantic scales.
- **Evident Multi-turn Error Accumulation**: Performance drops in multi-turn settings for all models, especially in **background preservation** (errors accumulate and contaminate unedited areas). MGIE collapses most severely.
- **Instruction Compliance is a Hidden Threshold**: Models like MGIE appear strong on old benchmarks but are exposed by the strict QA veto in Omni IIE Bench.
- **High Alignment with Human Judgment**: QA and CLIP rankings showed a correlation coefficient >0.85 with rankings from human annotators (PhD students and designers), validating the automatic evaluation.

## Highlights & Insights
- **Turning "Semantic Scale Consistency" into a Diagnostic Item**: Using mask area to quantify "editing granularity" and directly comparing shared-context task pairs is the most ingenious design.
- **16-Turn Multi-turn Diagnosis**: Far exceeding previous benchmarks, this truly exposes how error accumulation destroys background preservation, aligning with the real-world iterative workflow of designers.
- **FG/BG Decoupling + QA Veto**: Separating "edited correctly" from "did not destroy the background" and using strict QA to catch "high-quality but non-compliant" models provides an evaluation protocol that can be transferred to other generative benchmarks.
- **Zero-Tolerance Human Filtering**: Retaining only ~28% of candidates ensures a high-quality data moat through calibration, secondary review, and arbitration.

## Limitations & Future Work
- GT images are generated by Nano Banana and masks by Grounded-SAM; the "ideal edit" may carry biases from the generation model, potentially favoring models with similar styles. ⚠️
- Instruction generation and QA judging rely heavily on GPT-4o. Having the same model family generate the questions and judge the answers poses potential systematic bias.
- Automatic masking relies on GroundingDINO+SAM; Pass/Fail checks are performed rather than pixel-level manual corrections, so mask precision in complex multi-entity scenes may be limited.
- Future work: Introduce neutral/multi-source GT and judge models; expand the semantic scale spectrum beyond binary classes; release fine-grained failure case libraries for scale switching.

## Related Work & Insights
- **vs I2EBench**: While it separates levels, it uses isolated checklists. Ours uses shared-context task pairs to directly quantify consistency.
- **vs CompBench**: It supports 2-turn dialogue and four-way decoupling but treats tasks as isolated points. Ours explicitly diagnoses low↔high scale switching up to 16 turns.
- **vs ImgEdit-Bench / MuCIE**: These focus on content memory/version rollback but ignore stability under dynamic semantic scale changes and lack validation by professional designers. Ours prioritizes "practical utility."

## Rating
- Novelty: ⭐⭐⭐⭐ Defining "semantic scale consistency" as a diagnostic dimension is novel, though underlying components are pre-existing models.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8 models across single/multi-turn and three decoupled dimensions with human alignment.
- Writing Quality: ⭐⭐⭐⭐ Motivation and pipeline are clearly articulated; metrics are well-defined.
- Value: ⭐⭐⭐⭐⭐ Quantifies a universal failure mode in mainstream models, providing direct guidance for "practical-ready" evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models](i2i-bench_a_comprehensive_benchmark_suite_for_image-to-image_editing_models.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] WiseEdit: Benchmarking Cognition- and Creativity-Informed Image Editing](wiseedit_benchmarking_cognition-_and_creativity-informed_image_editing.md)
- [\[CVPR 2026\] MotionEdit: Benchmarking and Learning Motion-Centric Image Editing](motionedit_benchmarking_and_learning_motion-centric_image_editing.md)
- [\[CVPR 2026\] CompBench: Benchmarking Complex Instruction-guided Image Editing](compbench_benchmarking_complex_instruction-guided_image_editing.md)

</div>

<!-- RELATED:END -->
