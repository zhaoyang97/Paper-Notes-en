---
title: >-
  [Paper Note] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs
description: >-
  [ICLR 2026][VLM Reasoning][Spatial Reasoning] SpinBench is proposed as a diagnostic benchmark grounded in cognitive science. It systematically evaluates the spatial understanding of 37 VLMs through 7 progressive task categories (ranging from object recognition to perspective taking), revealing systematic flaws such as egocentric bias and weak rotation comprehension.
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "Spatial Reasoning"
  - "Perspective Taking"
  - "Mental Rotation"
  - "VLM"
  - "Benchmark"
date: 2026-05-08
content_hash: c91d8ba43fc0db41
---

# SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs

**Conference**: ICLR 2026  
**arXiv**: [2509.25390](https://arxiv.org/abs/2509.25390)  
**Code**: [https://spinbench25.github.io/](https://spinbench25.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Spatial Reasoning, Perspective Taking, Mental Rotation, VLM, Benchmark

## TL;DR

SpinBench is proposed as a diagnostic benchmark grounded in cognitive science. It systematically evaluates the spatial understanding of 37 VLMs through 7 progressive task categories (ranging from object recognition to perspective taking), revealing systematic flaws such as egocentric bias and weak rotation comprehension.

## Background & Motivation

Spatial reasoning is a fundamental component of human cognition and a critical capability for embodied agents operating in the physical world. While Vision-Language Models (VLMs) have made remarkable progress in visual understanding, their spatial reasoning abilities remain poorly understood and under-diagnosed.

Several key issues exist in current approaches:

**Evaluation Entanglement**: Existing applications (navigation, manipulation, autonomous driving, etc.) mainly reflect end-to-end performance. Spatial reasoning is entangled with high-level language and planning goals, making it impossible to directly test whether models truly understand geometric primitives like rotation, translation, relative poses, and viewpoint changes.

**Dataset Bias Dependence**: It is unclear whether VLMs truly possess spatial reasoning capabilities or rely on dataset biases and shallow pattern matching.

**Limitations of Prior Work**: Existing spatial reasoning benchmarks (e.g., CLEVR, BLINK, SpaCE-10) suffer from shortcomings such as a lack of controlled viewpoint changes, absence of reference frame variation tests, lack of support for multi-frame reasoning, and entanglement with functional and physical common sense.

The Core Problem addressed originates from fundamental insights in cognitive science—classic mental rotation experiments (Shepard & Metzler, 1971) proved that spatial cognition often depends on simulated, imagery-based processes. This leads to the **Key Challenge**: Can VLMs perform such imagery-based spatial reasoning, or are they limited to symbolic and linguistic associations?

## Method

### Overall Architecture

SpinBench centers on **perspective taking**—the ability to reason about how "scene-object relationships change under viewpoint transformations"—as the core challenge. Since perspective taking is an integrated capability, the paper decomposes it into a set of **progressive diagnostic categories**, ranging from basic cross-view identification to mental transformations in full scenes. To disentangle spatial reasoning from irrelevant factors like functional or physical common sense, all tasks are restricted to a **horizontal 2D plane** (excluding vertical relations like up/down, with viewpoint changes limited to horizontal orbital motion), ensuring the evaluation focuses on geometric primitives like rotation, translation, and relative pose.

The benchmark consists of four components corresponding to these design points: **7 progressive tasks** to decompose perspective taking into diagnosable cognitive sub-skills; **4 data domains** to ensure spatial capability is tested rather than memorization of specific appearances; **controlled variables and logical equivalence augmentation** for each sample to separate "visual perceptual errors" from "linguistic reasoning errors"; and finally, **$\kappa$-correction and pairwise consistency** metrics to enable fair comparisons across tasks with different option counts and directly measure spatial logic.

### Key Designs

**1. Seven Progressive Tasks: Decomposing Perspective Taking into Diagnosable Cognitive Sub-skills**

Existing benchmarks often conflate spatial reasoning with high-level planning, making it impossible to locate where a model fails. SpinBench utilizes a progression chain of 2,599 samples. The base layer is **Identity Matching** (405 samples), testing consistency in identifying the same object from different views. Above this is **Object Relation Localization** (636 samples), judging direction (left/right, front/back) and distance (near/far) in single static images. More advanced are dynamic reasoning tasks: **Dynamic Translation** (156 samples) determines relative displacement direction from two sequential frames; **Dynamic Rotation** (353 samples) judges rotation direction (clockwise vs. counter-clockwise). Higher-level tasks include **Canonical View Selection** (358 samples) and **Mental Rotation** (78 samples). The top level is the core task, **Perspective Taking** (613 samples), requiring reasoning about entire scenes under viewpoint changes, subdivided into Selecting scene maps from a new viewpoint (S-type) and Predicting relation changes (T-type).

**2. Four Data Domains: Verifying Spatial Capability over Appearance Memorization**

To prevent models from relying on shallow patterns, samples span four sources: **Infinigen Synthetic Scenes** (54.1%), procedurally generated indoor tabletop scenes using Isaac Sim with YCB objects; **ABO Objects** (23.7%), daily items from Amazon Berkeley Objects with 360° views; **Cars** (9.2%), rotation sequences from the Multi-View Car Dataset; and **Faces** (13.0%), representing 8 poses from the Stereo Face Database.

**3. Controlled Variables and Logical Equivalence Augmentation: Attributing Errors**

SpinBench overlays fine-grained controls to isolate failure causes. **Reference Frame Changes** explicitly switch between egocentric and allocentric coordinates. **Symmetry Augmentation** creates logically equivalent variants by flipping relations and answers (e.g., "$A$ is left of $B$" $\Leftrightarrow$ "$B$ is right of $A$") to test spatial logic. **Syntax Augmentation** rewrites questions to remove phrasing interference. **Premise Variants** provide or withhold visual premises to distinguish "visual localization failure" from "linguistic reasoning failure."

**4. $\kappa$-correction and Pairwise Consistency: Fair Comparison and Logic Measurement**

Since task option counts vary, raw accuracy is distorted by random chance. SpinBench primarily uses **Cohen's kappa ($\kappa$)**, which subtracts the expected agreement by chance. Additionally, **Pairwise Consistency** measures if a model provides self-consistent answers to logically equivalent question pairs.

## Key Experimental Results

### Main Results

37 VLMs were evaluated, including 4 commercial and 33 open-source models.

| Model | Overall Accuracy | Consistency | Perfect Rate |
|------|-----------|--------|--------|
| InternVL3-38B | 73.8% | 95.7% | 71.1% |
| InternVL3.5-38B | 71.9% | 95.3% | 75.1% |
| InternVL3-14B | 70.3% | 91.4% | 63.7% |
| GPT-4.1 | 69.8% | 85.9% | 59.5% |
| GPT-4o | 67.8% | 79.6% | 51.2% |
| Claude Sonnet 4 | 64.8% | 71.7% | 42.8% |
| Human | 91.2% | - | - |

### Key task performance

| Task Category | Best Model $\kappa$ | General Performance |
|---------|----------|------------|
| Object Relation Localization | >0.6 | Best performance; most models reliable |
| Identity Matching | Bimodal | Small models near random; large models near perfect |
| Dynamic Rotation | Difficult | Most models perform poorly |
| Mental Rotation | Near random | Most models at or below chance level |
| Perspective Taking | Near random | Most challenging |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Egocentric vs. Allocentric | $\kappa$ diff up to 1.6 | Molmo-7B: 0.94 (Egocentric) vs. -0.66 (Allocentric) |
| CoT Reasoning (Cosmos-Reason1) | Avg +0.221 $\kappa$ | Max gain of +0.650 in Perspective Taking |
| CoT Reasoning (SpaceOm) | Avg +0.118 $\kappa$ | Object Relation Localization benefited most |
| Premise vs. No Premise | 41% models < random | Systemic failure even in pure linguistic reasoning |

### Key Findings

1. **Systematic Egocentric Bias**: Models exhibit strong observer-viewpoint bias in dynamic rotation. Models performing best on egocentric tasks often perform worst on allocentric variants.
2. **Rotation Understanding Defiency**: Most models perform at or below chance in mental rotation and perspective taking, indicating a lack of robust internal representations for transformations.
3. **Emergent Capabilities**: Identity matching shows clear emergence—small models stay at chance, while larger models (7B-8B+) reach near-perfect accuracy.
4. **Accuracy-Consistency Correlation**: A strong positive correlation (Pearson $r=0.874, p<0.05$) exists between accuracy and consistency.
5. **Human-VLM Difficulty Alignment**: Significant negative correlation between human response time and VLM accuracy ($r=-0.54, p<0.05$).

## Highlights & Insights

1. **Cognitive Science Driven Design**: Inspired by Shepard & Metzler, the benchmark decomposes spatial reasoning into progressive sub-skills, a hierarchy missing in prior work.
2. **Diagnostic Value Beyond Leaderboards**: SpinBench identifies exactly where VLMs succeed or fail in the spatial reasoning pipeline.
3. **Complementarity**: Weak correlation with other benchmarks (MindCube, SpaCE-10, etc.) confirms that SpinBench captures unique foundational capabilities.
4. **Practical Utility**: Failures in reference frame reasoning or rotation directly impact navigation and manipulation safety in embodied AI.

## Limitations & Future Work

1. **Incomplete Concept Coverage**: Does not yet cover inclusion (in), support (on), or vertical relations (above/below).
2. **2D Plane Constraint**: Tasks are limited to the horizontal plane, excluding height differences.
3. **Small Scale for Some Categories**: The Mental Rotation category has a relatively small sample size (78).
4. **Lack of Training Guidance**: Currently used for evaluation; improving spatial reasoning through this data remains unexplored.
5. **Synthetic Data Bias**: Over 54% of data is synthetic (Infinigen), potentially differing from real-world distributions.

## Related Work & Insights

- **CLEVR**: Early synthetic diagnostic dataset using simple 3D shapes.
- **MindCube**: Emphasizes cognitive mapping and cross-scene spatial tracking.
- **ViewSpatial-Bench**: Focuses on viewpoint-dependent localization.
- **OmniSpatial**: Logic tasks based on cognitive psychology, but entangled with physical common sense.
- **Key Insight**: Purely linguistic approaches have fundamental limits in spatial reasoning; **structured reasoning beyond language** is required.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)
- [\[ICLR 2026\] MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse](metaspatial_reinforcing_3d_spatial_reasoning_in_vlms_for_the_metaverse.md)
- [\[ICML 2025\] Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas](../../ICML2025/vlm_reasoning/why_is_spatial_reasoning_hard_for_vlms_an_attention_mechanism_perspective_on_foc.md)
- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](../../ICML2026/vlm_reasoning/3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICLR 2026\] SpatialLadder: Building Spatial Reasoning Capabilities for Vision-Language Models via Progressive Training](spatialladder_progressive_training_for_spatial_reasoning_in_vision-language_mode.md)

</div>

<!-- RELATED:END -->
