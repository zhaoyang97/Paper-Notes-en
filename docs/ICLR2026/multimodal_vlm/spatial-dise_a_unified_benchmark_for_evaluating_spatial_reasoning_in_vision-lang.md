---
title: >-
  [Paper Note] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][spatial reasoning] This paper proposes Spatial-DISE, a unified spatial reasoning benchmark grounded in a cognitive-science-based 2×2 taxonomy (Intrinsic/Extrinsic × Static/Dynamic). The benchmark comprises 559 evaluation VQA pairs and 12K+ training instances. Evaluation across 32 state-of-the-art VLMs reveals a substantial gap between model performance and human-level capability, particularly on dynamic spatial reasoning tasks such as mental rotation and folding.
tags:
  - ICLR 2026
  - Multimodal VLM
  - spatial reasoning
  - VLM benchmark
  - cognitive taxonomy
  - DISE framework
  - mental transformation
date: 2026-05-08
content_hash: a34d763981520a7d
---

# Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.13394](https://arxiv.org/abs/2510.13394)
**Code**: [https://github.com/Spatial-DISE](https://github.com/Spatial-DISE)
**Area**: Multimodal VLM
**Keywords**: spatial reasoning, VLM benchmark, cognitive taxonomy, DISE framework, mental transformation

## TL;DR
This paper proposes Spatial-DISE, a unified spatial reasoning benchmark grounded in a cognitive-science-based 2×2 taxonomy (Intrinsic/Extrinsic × Static/Dynamic). The benchmark comprises 559 evaluation VQA pairs and 12K+ training instances. Evaluation across 32 state-of-the-art VLMs reveals a substantial gap between model performance and human-level capability, particularly on dynamic spatial reasoning tasks such as mental rotation and folding.

## Background & Motivation

**Background**: Spatial reasoning is critical for applications such as robotics, augmented reality, and autonomous driving. Numerous VLM spatial reasoning benchmarks have emerged in recent years, including SpatialRGPT, VSR, CV-Bench, and BLINK. These benchmarks primarily evaluate Extrinsic-Static (E-S) ability — i.e., understanding spatial relationships among objects in fixed scenes. Table 1 compares the DISE quadrant coverage of 18 existing benchmarks, showing that the vast majority cover only 1–2 quadrants.

**Limitations of Prior Work**: Existing benchmarks suffer from three key limitations: (1) they lack a systematic cognitive framework for categorizing and evaluating different types of spatial reasoning, resulting in fragmented and imbalanced assessments; (2) they over-focus on static spatial problems, neglecting tasks that require multi-step dynamic reasoning such as mental rotation and folding; and (3) the few benchmarks that do address dynamic tasks (e.g., SAT, SPACE) are too small to reliably evaluate model capability or support model training.

**Key Challenge**: Human spatial cognition encompasses rich dynamic mental simulation abilities — such as imagining the appearance of an object after rotation or folding — yet existing benchmarks have almost no systematic evaluation of this Intrinsic-Dynamic (I-D) capability. Models may perform adequately on static judgments while failing entirely on tasks requiring mental simulation, which are precisely the most critical abilities in real-world applications.

**Goal**: (1) How to establish a cognitively grounded unified taxonomy that covers all types of spatial reasoning? (2) How to generate large-scale, verifiable dynamic spatial reasoning data to address data scarcity? (3) What are the capability boundaries and failure modes of current VLMs across different spatial reasoning dimensions? (4) Can supplementary training data effectively improve spatial reasoning performance?

**Key Insight**: Drawing on Uttal et al.'s cognitive taxonomy of spatial abilities, this work organizes spatial reasoning along two dimensions — Intrinsic vs. Extrinsic and Static vs. Dynamic — into four DISE quadrants, and designs 10 cognitive tasks spanning all quadrants. A scalable synthetic data generation pipeline based on the Blender engine is constructed to address data scarcity in dynamic tasks.

**Core Idea**: Apply a cognitive-science-grounded 2×2 DISE taxonomy to unify spatial reasoning evaluation, with particular emphasis on filling the gap in the Intrinsic-Dynamic dimension left by existing benchmarks.

## Method

### Overall Architecture
Spatial-DISE consists of two datasets: **Spatial-DISE Bench** (559 VQA pairs covering 10 task types across 4 DISE quadrants, for evaluation) and **Spatial-DISE-12K** (12K+ VQA pairs covering 5 3D task types, for training). Data are sourced from two streams: real-world data (collected from academic psychometric tests and professional aptitude assessments, yielding an initial pool of 1,180 VQA pairs as conceptual templates) and Blender-based synthetic data (automatically generated at scale for dynamic tasks such as 3D rotation, 3D folding, and 3D shape finding).

The 10 tasks in the DISE taxonomy are inspired by classical psychometric tests: Intrinsic-Static (I-S) tasks include 2D/3D shape finding, testing static part-whole relationship analysis; Intrinsic-Dynamic (I-D) tasks include 2D/3D rotation, 2D/3D folding, and Fold&Punch, testing pure mental simulation of transformations; Extrinsic-Static (E-S) tasks use 3D projection to test spatial relationship understanding from fixed external viewpoints; Extrinsic-Dynamic (E-D) tasks use 2D/3D assembly to test dynamic multi-component assembly reasoning.

The overall pipeline consists of three stages: real-world data collection → scalable synthetic data generation → rigorous human quality control.

### Key Designs

1. **DISE Cognitive Taxonomy (2×2 Quadrants)**

   - **Function**: Provides a unified classification framework for spatial reasoning tasks.
   - **Mechanism**: The first dimension distinguishes Intrinsic (internal structure and part relationships of objects) from Extrinsic (spatial relationships between objects); the second dimension distinguishes Static (fixed, invariant information) from Dynamic (information requiring mental transformation). This yields four quadrants: I-S (analyzing internal static object properties, e.g., shape finding), I-D (mentally simulating object transformations, e.g., rotation and folding), E-S (inter-object relationships in fixed scenes, e.g., 3D projection), and E-D (reasoning about changing multi-object relationships, e.g., 2D/3D assembly).
   - **Design Motivation**: Prior benchmarks typically cover only 1–2 quadrants (predominantly E-S). The DISE framework ensures comprehensive coverage, with particular emphasis on filling the weakest quadrant, I-D.
   - **Known Limitation**: The framework does not explicitly distinguish difficulty differences between 2D and 3D spatial reasoning.

2. **Blender Automated Synthetic Data Pipeline**

   - **Function**: Generates large-scale, verifiable 3D spatial reasoning VQA data.
   - **Mechanism**: A five-step process — (1) generate reproducible random seeds via question\_id hashing; (2) construct core 3D objects (e.g., irregular shapes, textured cubes); (3) render question and correct-answer images from optimal viewpoints; (4) systematically generate multi-level distractors (geometric variants, texture/orientation errors, incorrect viewpoints, component substitutions); (5) render uniformly in a controlled virtual environment. Ground-truth answers for each instance are verifiable through Blender scene parameters.
   - **Design Motivation**: Dynamic spatial reasoning data are extremely scarce and difficult to collect at scale from the real world; the Blender synthetic pipeline is the key technical contribution addressing this bottleneck.
   - **Scalability**: The community can extend the framework with new task types by implementing corresponding object generation and distractor strategies.

3. **Hierarchical Distractor Generation Strategy**

   - **Function**: Ensures the diagnostic value and challenge level of each question.
   - **Mechanism**: Task-specific distractor strategies are designed — geometric variants (adding or removing components), pattern/orientation errors (texture misalignment), incorrect viewpoints (wrong orthographic projection direction), and component substitutions (replacing correct parts with geometrically similar but incorrect alternatives). Each strategy targets a specific type of error that models are prone to make.
   - **Design Motivation**: Simple distractors (e.g., completely different shapes) cannot effectively differentiate model capabilities; near-miss distractors force models to perform precise spatial reasoning rather than pattern matching.
   - **Implementation Details**: Distractor strategies are specifically designed per task type — e.g., distractors for 3D rotation are generated by fine-tuning rotation angles, while distractors for 3D folding are generated by swapping face textures.

### Quality Control
A three-stage quality control process is applied: (1) **answer uniqueness check** — each question must have exactly one correct answer; (2) **accuracy and clarity** — images must be free of rendering artifacts, questions must be clearly stated, and all options must conform to task standards; (3) **redundancy elimination** — logically or visually duplicate instances are removed. Instances failing any stage are excluded from the final dataset.

### Human Baseline Establishment
The human baseline is collected from 54 participants (ages 15–55), yielding 1,679 valid responses. A matrix sampling design ensures that each question is answered by an average of 3 independent participants. Mean accuracy across all responses is reported and cross-validated using Item Response Theory (IRT) to ensure psychometric reliability of the baseline.

## Key Experimental Results

### Main Results

| Model Type | Representative Model | Overall Acc. | I-D (Intrinsic-Dynamic) | E-D (Extrinsic-Dynamic) | I-S (Intrinsic-Static) |
|---|---|---|---|---|---|
| Closed-source Best | Doubao1.5VL-thinking | 42.0% | 40.9% | 61.9% | 35.6% |
| Closed-source Avg. | — | 31.9% | 35.2% | 26.0% | 27.7% |
| Open-source Best | Qwen2.5-VL-7B-sft | 47.0% | 43.1% | 66.7% | 51.7% |
| Open-source Avg. | — | 26.2% | 29.1% | 23.2% | 19.3% |
| **Human Baseline** | — | **76.8%** | **80.2%** | **61.1%** | **76.8%** |
| Random Chance | — | 24.8% | 24.3% | 25.4% | 24.7% |

### Fine-tuning Results (Spatial-DISE-12K)

| Model | Spatial-DISE | CVBench | SAT | SPACE | OmniSpatial |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B (Base) | 26.1% | — | — | — | — |
| Qwen2.5-VL-7B (SFT) | 47.0% (+20.9pp) | — | — | — | — |
| SpaceOm (Base) | 25.9% | 68.8% | 46.67% | 27.22% | 27.91% |
| SpaceOm (SFT) | 41.3% (+15.4pp) | 70.33% | 49.33% | — | — |

### Key Findings
- The average accuracy of all 32 models is only 28.4%, marginally above random chance (25%) and far below the human baseline (76.8%), indicating that spatial reasoning is a systemic weakness of current VLMs.
- On the Fold&Punch task — which requires three-step mental simulation (fold → punch → unfold) — the best-performing model achieves only 30.8%, with an average of 25.4% (equivalent to random chance), revealing a severe deficit in spatial working memory: models cannot maintain coherent mental states across multi-step transformations.
- Static capability is not a prerequisite for dynamic reasoning: several models perform better on dynamic tasks than on static ones (e.g., Gemini-2.0-Flash: 38.3% on dynamic vs. 23.6% on static), suggesting that models have learned fragmented strategies rather than systematic spatial cognition.
- Doubao-1.5-thinking surpasses humans on E-D tasks (61.9% vs. 61.1%) by converting cognitive simulation into a computational problem — algorithmically comparing geometric features rather than relying on mental simulation.
- Fine-tuning on Spatial-DISE-12K yields substantial gains (Qwen2.5-VL +20.9pp), with partial generalization to external benchmarks such as CVBench and SAT.
- Reasoning-enhanced training (e.g., RLHF, GRPO) provides limited and uneven improvements and does not fundamentally resolve the cognitive spatial reasoning deficiency.

## Highlights & Insights
- The DISE taxonomy unifies fragmented spatial reasoning research under a single framework, enabling precise diagnosis of which cognitive dimension is the weakest for a given model. This framework can be transferred to evaluation of other cognitive capabilities such as causal reasoning and temporal reasoning.
- The Blender synthetic pipeline is a reusable tool — seeded randomization ensures verifiability, and hierarchical distractors ensure diagnostic value. The community can build upon this foundation to extend new spatial reasoning task types.
- The finding that "static capability is not a prerequisite for dynamic reasoning" challenges intuition and suggests that current VLMs' spatial "understanding" may be pattern matching rather than genuine spatial cognition.
- The phenomenon of Doubao-1.5-thinking surpassing humans on E-D tasks suggests that models have a natural advantage on algorithmically tractable spatial tasks — pointing toward a research direction of "computational spatial reasoning."
- The fact that fine-tuning on only 12K instances yields 20pp+ gains indicates an extreme scarcity of training data for dynamic spatial reasoning; the dataset itself constitutes a significant contribution.

## Limitations & Future Work
- The VQA multiple-choice format may underestimate models' open-ended spatial reasoning capabilities (e.g., free-form description of spatial relationships).
- The visual style of synthetic data (plain backgrounds, simple geometric objects) diverges considerably from real-world scenes; further validation is needed to assess whether fine-tuning transfers to real-world scenarios.
- The benchmark focuses exclusively on 2D/3D geometric spatial reasoning, without addressing semantic spatial reasoning (e.g., "kitchens are typically adjacent to dining rooms") or navigational spatial reasoning.
- The human baseline of 54 participants is relatively small, and the age and educational distribution within the 15–55 age range is not detailed.
- The Blender pipeline currently covers only 5 3D task types and can be extended to occlusion reasoning, perspective transformation, mirror reflection, and other types.
- The Bench scale (559 pairs) may be insufficient for certain sub-categories (e.g., only 70 pairs for E-S), resulting in wide model confidence intervals.
- Video-based dynamic spatial reasoning is not explored; multi-frame input may improve models' mental simulation capabilities.

## Related Work & Insights
- **vs. SPARE3D**: SPARE3D covers only the I-S quadrant (testing 3D shape recognition with synthetic data), whereas Spatial-DISE covers all four quadrants with particular emphasis on filling the I-D gap.
- **vs. SPACE**: SPACE addresses dynamic reasoning but is small-scale (5K) and lacks a unified framework; Spatial-DISE provides a larger training set (12K+) and a systematic cognitive taxonomy.
- **vs. OmniSpatial**: OmniSpatial covers all four quadrants but is small-scale (1.5K) and uses real-world data that is difficult to scale; Spatial-DISE's Blender pipeline offers a scalable alternative.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The DISE taxonomy is grounded in cognitive science, and the systematic emphasis on Intrinsic-Dynamic reasoning fills an important evaluation gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation spans 32 models (closed-source, open-source, reasoning-enhanced, and spatially specialized), with quadrant-level analysis, fine-tuning experiments, and generalization tests across five external benchmarks — outstanding in both breadth and analytical depth.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, the DISE framework diagrams are intuitive, and the cognitive analysis is substantive; some experimental tables are overly dense, making it difficult to extract key information quickly.
- **Value**: ⭐⭐⭐⭐ The work reveals systemic weaknesses of current VLMs in cognitive spatial reasoning; the Blender synthetic pipeline and the 12K training set offer practical reuse value for the community.

<!-- END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)

</div>

<!-- RELATED:END -->
