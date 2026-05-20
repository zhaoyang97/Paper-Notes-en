---
title: >-
  [Paper Note] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs
description: >-
  [ICLR 2026][Multimodal VLM][spatial reasoning] This paper introduces SpinBench, a cognitively grounded diagnostic benchmark that systematically evaluates spatial reasoning in 37 VLMs through 7 progressively structured ta…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "spatial reasoning"
  - "perspective taking"
  - "mental rotation"
  - "vision-language models"
  - "benchmark"
date: 2026-05-08
content_hash: 33dbc7f27bc0e355
---

# SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs

**Conference**: ICLR 2026
**arXiv**: [2509.25390](https://arxiv.org/abs/2509.25390)  
**Code**: [https://spinbench25.github.io/](https://spinbench25.github.io/)  
**Area**: Multimodal VLM
**Keywords**: spatial reasoning, perspective taking, mental rotation, vision-language models, benchmark

## TL;DR

This paper introduces SpinBench, a cognitively grounded diagnostic benchmark that systematically evaluates spatial reasoning in 37 VLMs through 7 progressively structured task categories—ranging from object identity recognition to perspective taking—revealing systemic deficiencies including egocentric bias and weak rotation understanding.

## Background & Motivation

Spatial reasoning is a fundamental component of human cognition and a critical capability for embodied agents operating in the physical world. Despite impressive progress in visual understanding, the spatial reasoning capabilities of vision-language models (VLMs) remain poorly understood and insufficiently diagnosed.

Several key issues with existing approaches:

**Entangled evaluation**: Existing applications (navigation, manipulation, autonomous driving, etc.) primarily reflect end-to-end performance, where spatial reasoning is entangled with high-level language and planning objectives, making it impossible to directly test whether models genuinely understand geometric primitives such as rotation, translation, relative object pose, and viewpoint change.

**Dataset bias reliance**: It remains unclear whether VLMs possess genuine spatial reasoning abilities or instead exploit dataset biases and shallow pattern matching.

**Inadequacy of existing benchmarks**: Prior spatial reasoning benchmarks (e.g., CLEVR, BLINK, SpaCE-10) fall short in the following respects: lack of controlled viewpoint variation, absence of reference frame change tests, no support for multi-frame reasoning, and entanglement of spatial reasoning with functional and physical commonsense knowledge.

The core research question is grounded in a foundational insight from cognitive science—the classic mental rotation experiments of Shepard & Metzler (1971) demonstrate that spatial cognition typically relies on simulated, imagery-based processes. This raises the central question: **Can VLMs perform such imagery-based spatial reasoning, or are they limited to symbolic and linguistic associations?**

## Method

### Overall Architecture

SpinBench is designed around the core challenge of **perspective taking**—a highly integrative ability requiring cross-viewpoint object recognition, relative localization, and mental simulation of transformations. SpinBench decomposes this high-level capability into a set of targeted diagnostic categories, each representing a fundamental spatial reasoning skill that underlies perspective taking.

To minimize confounding factors, all tasks are defined in a **horizontal 2D plane**, excluding vertical relations (e.g., above/below), with viewpoint changes restricted to horizontal orbits around the scene.

### Key Designs

1. **Progressive structure of seven task categories**:

    - **Identity Matching**: Evaluates whether models can consistently recognize the same object across different viewpoints—a prerequisite for cross-view reasoning. 405 samples.
    - **Object-Relation Grounding**: Tests understanding of relative object configurations in single static images, including directional relations (left/right, front/back) and distance relations (near/far). 636 samples, the largest category.
    - **Dynamic Translation**: Assesses reasoning about linear object displacement. Given two temporally ordered frames, the model must identify the direction of object motion relative to the observer. 156 samples.
    - **Dynamic Rotation**: Focuses on rotational transformations, requiring the model to judge rotation direction (e.g., clockwise vs. counterclockwise). 353 samples.
    - **Canonical View Selection**: Tests whether models can map between canonical viewpoints of an object (given a frontal view, select the left/right/back view). 358 samples.
    - **Mental Rotation**: Tests whether models can mentally simulate object transformations—given an object and a specified rotation angle/direction, select the correct resulting configuration. 78 samples.
    - **Perspective Taking**: The central task of SpinBench, requiring scene-level reasoning under viewpoint change. Includes two subtypes: (S) selecting the correct scene image from a new viewpoint; (T) predicting how object relations transform under a viewpoint change. 613 samples.

2. **Diversity across four data domains**:

    - **Infinigen Synthetic Scenes** (54.1%): Indoor tabletop multi-object scenes generated with Infinigen in Isaac Sim, using objects from the YCB dataset.
    - **ABO Objects** (23.7%): Everyday objects from the Amazon Berkeley Objects dataset, providing 360° views.
    - **Cars** (9.2%): Vehicle rotation sequences from the Multi-View Car Dataset.
    - **Faces** (13.0%): Eight-pose sequences from the Stereo Face Database.

3. **Fine-grained controlled variables**:

    - **Reference frame variation**: Tests the ability to switch between egocentric and allocentric reference frames.
    - **Symmetry augmentation**: Generates logically equivalent variants by flipping relations and answers (e.g., "left → right").
    - **Syntactic augmentation**: Rephrases questions while preserving semantics.
    - **Premise condition variants**: Distinguishes between failures in visual grounding and failures in linguistic reasoning.

### Evaluation Metrics

- **Raw accuracy**: Proportion of correctly answered questions.
- **Cohen's kappa ($\kappa$)**: Chance-corrected accuracy that accounts for differences in option cardinality, enabling fair cross-task comparison.
- **Pairwise consistency**: Measures whether models produce consistent responses across logically equivalent question pairs.

## Key Experimental Results

### Main Results

37 VLMs were evaluated, including 4 commercial and 33 open-source models.

| Model | Overall Accuracy | Consistency | Perfect Rate |
|-------|-----------------|-------------|--------------|
| InternVL3-38B | 73.8% | 95.7% | 71.1% |
| InternVL3.5-38B | 71.9% | 95.3% | 75.1% |
| InternVL3-14B | 70.3% | 91.4% | 63.7% |
| GPT-4.1 | 69.8% | 85.9% | 59.5% |
| GPT-4o | 67.8% | 79.6% | 51.2% |
| Claude Sonnet 4 | 64.8% | 71.7% | 42.8% |
| Human | 91.2% | — | — |

### Performance by Task Category

| Task Category | Best Model $\kappa$ | General Performance |
|--------------|--------------------|--------------------|
| Object-Relation Grounding | >0.6 | Best overall; most models perform reliably |
| Identity Matching | Bimodal distribution | Small models near chance; large models near perfect |
| Dynamic Rotation | Difficult | Most models perform poorly |
| Mental Rotation | Near chance | Most models at or below chance level |
| Perspective Taking | Near chance | Most challenging category |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|------------|-------|
| Egocentric vs. allocentric | $\kappa$ gap up to 1.6 | Molmo-7B: egocentric 0.94, allocentric −0.66 |
| CoT reasoning (Cosmos-Reason1) | Avg. +0.221 $\kappa$ | Largest gain on perspective taking: +0.650 |
| CoT reasoning (SpaceOm) | Avg. +0.118 $\kappa$ | Object-relation grounding benefits most |
| With vs. without premise | 41% of models below chance | Systemic failures persist even in pure language reasoning |

### Key Findings

1. **Systemic egocentric bias**: Models exhibit strong observer-perspective bias in dynamic rotation tasks. Models performing best on egocentric variants perform worst on allocentric variants, suggesting that first-person visual descriptions dominate inductive biases in training data.

2. **Severely insufficient rotation understanding**: In mental rotation and perspective taking tasks, the majority of models perform at or below chance, indicating the absence of robust internal representations for rotational transformations.

3. **Emergent capability**: Identity matching exhibits a clear emergent pattern—smaller models remain at chance level, while larger models (7B–8B+) achieve near-perfect accuracy, suggesting that cross-image 3D abstraction only becomes possible once models reach sufficient capacity.

4. **Strong accuracy–consistency correlation**: A strong positive correlation exists between overall accuracy and consistency (Pearson $r = 0.874$, $p < 0.05$), indicating that models unable to maintain equivalences such as "A is to the left of B" = "B is to the right of A" lack genuine spatial understanding.

5. **Human–VLM difficulty alignment**: A significant negative correlation is observed between human response times and VLM accuracy ($r = -0.54$, $p < 0.05$), validating that SpinBench captures spatial reasoning difficulties shared by both humans and VLMs.

## Highlights & Insights

1. **Cognitive science-driven benchmark design**: Inspired by the Shepard & Metzler mental rotation experiments, SpinBench decomposes spatial reasoning into progressive cognitive sub-skills. This hierarchical structure has been absent from prior benchmarks.

2. **Diagnostic value beyond leaderboards**: SpinBench provides not merely a ranking of scores but a diagnostic framework that precisely identifies where VLMs succeed and fail in spatial reasoning, and how these skills compose in perspective taking.

3. **Complementarity with existing benchmarks**: Weak and non-significant overall correlations with MindCube, ViewSpatial-Bench, OmniSpatial, and SpaCE-10 validate that SpinBench captures distinct foundational capabilities rather than general spatial intelligence.

4. **Practical implications**: Direct guidance for embodied AI—failures in reference frame reasoning or rotation understanding may cause catastrophic failures in navigation, manipulation, and other safety-critical tasks.

## Limitations & Future Work

1. **Incomplete spatial concept coverage**: Important spatial concepts such as containment (in), support (on), and vertical relations (above/below) are not yet covered.

2. **Tasks restricted to the horizontal plane**: All tasks are defined in a 2D horizontal plane, excluding height variation and vertical relations, limiting the comprehensiveness of the evaluation.

3. **Relatively small scale**: The benchmark comprises 2,599 samples in total, with some categories having limited sample sizes (e.g., mental rotation with only 78 samples).

4. **No training guidance**: The benchmark is primarily an evaluation tool; how to leverage its data to train models with improved spatial reasoning has not been explored.

5. **Synthetic data bias**: Infinigen synthetic scenes account for over 54% of samples, potentially introducing distributional discrepancies relative to real-world scenes.

## Related Work & Insights

- **CLEVR**: An early synthetic diagnostic dataset using simple 3D shapes only.
- **MindCube**: Emphasizes cognitive mapping, focusing on how models track spatial information across scenes.
- **ViewSpatial-Bench**: Focuses on viewpoint-dependent grounding.
- **OmniSpatial**: Spatial tasks grounded in cognitive psychology, but entangles spatial reasoning with functional and physical commonsense knowledge.
- **SpatialReasoner / SSR / APC**: Augment spatial reasoning via explicit 3D representations.
- **MetaSpatial / SpatialVLM**: Enhance spatial understanding through reinforcement learning or large-scale pretraining.
- Key implication: Purely language-based approaches face fundamental limitations in spatial reasoning, motivating the need for **structured reasoning beyond language**.

## Rating

- Novelty: ⭐⭐⭐⭐ (The cognitively grounded, progressive diagnostic design is creative, though the benchmark format itself is well-established)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (37 models, human baselines, consistency analysis, scaling laws, and correlation analyses—highly comprehensive)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, in-depth analysis, and professional presentation)
- Value: ⭐⭐⭐⭐ (Systematic diagnosis of VLM spatial reasoning offers important guidance, particularly for embodied AI)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)

</div>

<!-- RELATED:END -->
