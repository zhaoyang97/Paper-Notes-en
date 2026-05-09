---
title: >-
  [Paper Note] CodePercept: Code-Grounded Visual STEM Perception for MLLMs
description: >-
  [CVPR 2026][Multimodal VLM][MLLM] Through systematic scaling analysis, this paper reveals that **perception rather than reasoning** is the true bottleneck of MLLMs on STEM visual tasks. It proposes a paradigm that uses executable code as a medium to enhance perceptual capability, constructs ICC-1M — a 1M-scale Image-Caption-Code triplet dataset — and introduces two training tasks: code-grounded caption generation and STEM image-to-code translation.
tags:
  - CVPR 2026
  - Multimodal VLM
  - MLLM
  - STEM visual perception
  - code generation
  - perception bottleneck
  - ICC-1M
date: 2026-05-08
content_hash: 196da9ebb519fe67
---

# CodePercept: Code-Grounded Visual STEM Perception for MLLMs

**Conference**: CVPR 2026
**arXiv**: [2603.10757](https://arxiv.org/abs/2603.10757)
**Code**: [GitHub](https://github.com/TongkunGuan/Qwen-CodePercept)
**Area**: Code Intelligence / Multimodal Large Language Models
**Keywords**: MLLM, STEM visual perception, code generation, perception bottleneck, ICC-1M

## TL;DR

Through systematic scaling analysis, this paper reveals that **perception rather than reasoning** is the true bottleneck of MLLMs on STEM visual tasks. It proposes a paradigm that uses executable code as a medium to enhance perceptual capability, constructs ICC-1M — a 1M-scale Image-Caption-Code triplet dataset — and introduces two training tasks: code-grounded caption generation and STEM image-to-code translation.

## Background & Motivation

When MLLMs fail at STEM visual reasoning (mathematics, physics, chemistry, electrical engineering), a fundamental question arises: **does the failure stem from insufficient perception or insufficient reasoning?**

This paper answers the question through a novel **scaling analysis experiment**, decoupling STEM visual reasoning into two stages — perception (image → description) and reasoning (description → answer) — and independently scaling each stage while holding the other fixed:

- **Perception@4B + Reasoning scaled across 4/8/32B** (blue curve) vs. **Reasoning@4B + Perception scaled across 4/8/32B** (red curve)
- Results consistently show: **scaling perception yields greater gains than scaling reasoning**

This finding reveals that perception is the true lever for unlocking current STEM visual reasoning performance.

However, directly enhancing STEM perception via knowledge distillation (e.g., having GPT/Gemini generate descriptive captions) faces two major obstacles:
1. **Hallucination**: Teacher models produce erroneous descriptions of spatial positions, quantitative relationships, and element interactions.
2. **Descriptive inadequacy**: The complex spatial relationships and precise numerical values in many STEM images **cannot be adequately captured by natural language** (e.g., auxiliary line constructions in polyhedral geometry).

## Method

### Overall Architecture

The CodePercept pipeline consists of three core components:

1. **Image-Code Pair Construction** (data engine): Three complementary pipelines generate large-scale image-code pairs.
2. **Two code-grounded training tasks**: Code-Grounded Caption Generation + STEM Image-to-Code Translation.
3. **Two-stage post-training**: SFT (CodePercept-S1) + RL (CodePercept-R1).

### Key Designs

1. **Three data generation pipelines** producing the ICC-1M dataset (1M+ triplets):

    - **Image Reproduce (IR)**: $\mathbf{c} = G_{code}(\mathbf{I}, G_{caption}(\mathbf{I}))$. An MLLM first generates an image description, which is then used together with the image to generate reproduction code. Conceptually straightforward but constrained by the diversity of source datasets.
    - **Image Diversity (ID)**: $[\mathbf{c}_1, \dots, \mathbf{c}_K] = G_{code}(\mathbf{I}, G_{principle}(\mathbf{I}))$. Core insight: the underlying principles of STEM images can be abstracted and re-instantiated in different contexts. For example, starting from a domino logic puzzle seed image, variants such as circular domino wheels, triangular combinations, and ladybug-spot matrices are generated, preserving STEM rigor while introducing structural novelty.
    - **Solid Geometry Synthesis (SG)**: $\mathcal{C}_{geo} = \{\mathbf{c}_i \mid \mathbf{c}_i = \tilde{\mathbf{c}}_i(\boldsymbol{\theta})\}$. Parameterized code templates are used to generate solid geometry images, covering 8 canonical scenarios including cube unfolding, orthographic three-view drawings, cross-section analysis, and polyhedron construction. This addresses the fundamental deficiency of current MLLMs in generating solid geometry code.

2. **Code-Grounded Caption Generation**: Generates high-quality captions anchored to code as ground truth, eliminating hallucinations.
    - Step 1: The MLLM directly describes the image to obtain $\mathbf{t}_{draft}$ (fluent but factually erroneous).
    - Step 2: Code analysis combined with an execution tracer $\xi(\mathbf{c})$ extracts verified visual facts $\mathbf{t}_{code}$.
    - Step 3: The final caption $\mathbf{t}_{new} = G_{refine}(\mathbf{t}_{draft}, \mathbf{t}_{code})$ is synthesized, preserving linguistic fluency while correcting factual errors.

    The execution tracer $\xi(\mathbf{c})$ is the key innovation — it records all visual elements produced during code execution: geometric precision (coordinates, dimensions, spatial relationships), quantitative attributes (counts, RGB specifications), rendering semantics (z-order layering, transformation matrices), and STEM parameter-to-visual mappings.

3. **STEM Image-to-Code Translation**: Trains the model to directly generate executable reproduction code from images.
    - Explanatory code with pedagogical annotations is generated as $\mathbf{c}_{new} = G_{refine}(G_{code}(\mathbf{x}), \mathbf{c})$.
    - Draft code exhibits good pedagogical patterns (step-by-step decomposition, parameter explanation) but contains factual errors.
    - Ground-truth code is used to correct errors while preserving the explanatory structure.

### Loss & Training

**Stage 1: SFT (CodePercept-S1)**:
- Based on the Qwen3-VL series; jointly optimizes image captioning and image-to-code translation tasks.
- Trained for 1 epoch on ICC-1M using 32 A100 GPUs with the SWIFT framework.

**Stage 2: RL (CodePercept-R1)**:
- GRPO reinforcement learning applied to code generation only.
- Two categories of reward signals:
    - Format Reward $r_{fmt}$: validates code format (```python``` blocks).
    - Content Reward $r_{cnt}$: execution reward ($r_{exec}$, whether code is executable) + code-level reward ($r_{code}$, GPT-4o evaluates semantic equivalence) + image-level reward ($r_{image}$, GPT-4o evaluates visual similarity).
- Total reward: $r = r_{fmt} + r_{cnt}$.
- 10K samples selected from ICC-1M, trained for 1 epoch using the VeRL framework.

## Key Experimental Results

### Main Results: Perception Evaluation (Captioner-Solver Setting, LLM Solver: Qwen3-30A3-Thinking)

| Image Captioner | MathVision | MathVista | MathVerse | DynaMath | WeMath | LogicVista | Avg. |
|----------------|-----------|----------|----------|---------|-------|-----------|------|
| Gemini2.5-Pro | 66.80 | 74.80 | 73.47 | 81.42 | 60.29 | 66.44 | 70.53 |
| Claude-Opus 4.1 | 59.61 | 71.10 | 56.19 | 73.25 | 44.86 | 59.28 | 60.72 |
| Qwen3-VL-4B | 54.21 | 67.30 | 64.59 | 69.40 | 46.10 | 54.14 | 59.29 |
| **CodePercept-4B-S1** | **57.63** | **69.60** | **65.59** | **71.38** | **47.81** | **60.40** | **62.07 (+2.8)** |
| Qwen3-VL-8B | 54.37 | 69.60 | 63.75 | 72.19 | 45.43 | 56.82 | 60.36 |
| **CodePercept-8B-S1** | **59.31** | **70.20** | **66.52** | **73.20** | **49.14** | **61.52** | **63.32 (+3.0)** |
| Qwen3-VL-32B | 58.55 | 72.20 | 71.09 | 75.78 | 48.00 | 62.19 | 64.63 |
| **CodePercept-32B-S1** | **62.27** | **72.90** | **71.70** | **77.41** | **54.19** | **65.33** | **67.30 (+2.7)** |

### STEM2Code-Eval Benchmark (Image Reproduction Perception Evaluation)

| Model | Image Score | Code Score | Avg. | Exec Rate |
|------|-----------|-----------|------|----------|
| Gemini2.5-Pro-Thinking | 68.89 | 75.41 | 72.15 | 91.70% |
| GPT5-Thinking | 64.97 | 64.98 | 64.98 | 96.60% |
| Qwen3-VL-8B-Instruct | 28.59 | 28.23 | 28.41 | 85.30% |
| **CodePercept-8B-S1** | **44.53** | **46.78** | **45.66** | 87.60% |
| **CodePercept-8B-R1** | **50.25** | **47.04** | **48.65** | **93.40%** |
| Qwen3-VL-32B-Instruct | 36.85 | 39.98 | 38.42 | 81.80% |
| **CodePercept-32B-S1** | **61.14** | **56.99** | **59.07** | 93.00% |
| **CodePercept-32B-R1** | **68.97** | **62.53** | **65.75** | **95.90%** |

### Ablation Study

| Data Configuration | Avg. Perception Score | Gain |
|----------|:---:|:---:|
| Baseline (Qwen3-VL-8B) | 60.36 | - |
| + IR-CodeCap | 60.91 | +0.55 |
| + ID-CodeCap | 62.15 | +1.79 |
| + SG-CodeCap | 62.75 | +2.39 |
| NativeCap (direct captioning) | 60.78 | +0.42 |
| **CodeCap (code-grounded)** | **62.75** | **+2.39** |
| CodeCap + ImCode | **63.32** | **+2.96** |

### Key Findings

- **Perception is the STEM bottleneck**: Scaling perception consistently outperforms scaling reasoning across all datasets.
- **Code-grounded captions outperform direct captions**: CodeCap surpasses NativeCap by 2.0 percentage points, confirming that code effectively eliminates hallucinations.
- **Three pipelines are complementary**: ID (diversity) yields the largest gain, with SG (solid geometry) providing further improvement.
- **Code and captions are complementary**: Joint training on image-to-caption and image-to-code (63.32) outperforms caption-only training (62.75); code, functioning as "structured captions," supplies precise spatial and quantitative information.
- **RL substantially improves code quality**: CodePercept-8B-R1 outperforms S1 by +3.0 on STEM2Code-Eval (45.66 → 48.65).
- **Surpasses larger models**: CodePercept-8B-S1 (8B) exceeds Qwen2.5-VL-72B on perception tasks by 6.2 percentage points.

## Highlights & Insights

- **The core finding is highly significant**: Perception, not reasoning, is the STEM bottleneck — this challenges the prevailing narrative that MLLMs suffer primarily from insufficient reasoning ability and may redirect community research efforts.
- **"Code as perception" paradigm**: Executable code provides semantic precision that natural language cannot match — coordinates, quantities, and spatial relationships can all be precisely expressed in code and verified through execution.
- **Clever use of the execution tracer**: Code execution logs serve as an "external specification" for LLM analysis of code, addressing the difficulty LLMs face when analyzing complex recursive or nested logic.
- **STEM2Code-Eval benchmark**: The first benchmark to directly evaluate STEM visual perception through code generation — faithful reproduction of an image via code requires complete visual understanding.

## Limitations & Future Work

- The RL stage relies on GPT-4o as an evaluator, incurring high cost and potentially introducing bias.
- ICC-1M data is generated solely using matplotlib, limiting visual diversity (e.g., hand-drawn diagrams, photorealistic STEM images are not covered).
- STEM2Code-Eval contains only 1,000 images, which is a relatively small scale.
- Validation is currently limited to the Qwen3-VL series; applicability to other MLLM architectures remains unknown.
- Code generation is restricted to Python matplotlib, which cannot cover complex geometric scenes requiring 3D rendering.
- The decoupled analysis of perception and reasoning uses captions as an intermediate representation, which may introduce an information bottleneck.

## Related Work & Insights

- **Qwen3-VL [2026]**: The backbone model for CodePercept, developed by the Qwen team.
- **GRPO [DeepSeek 2024]**: Group Relative Policy Optimization, used in the RL stage.
- **MathVision, MathVista, MathVerse**: STEM visual reasoning benchmarks.
- **InternVL / MiniCPM-V**: Competitive MLLMs that trail CodePercept on STEM2Code-Eval.
- Insight: The paradigm of code as a perceptual medium is extensible to other domains requiring precise description (e.g., maps, architectural drawings, circuit diagrams, chemical structural formulas).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs](downscaling_intelligence_exploring_perception_and_reasoning_bottlenecks_in_small.md)
- [\[CVPR 2026\] Token Warping Helps MLLMs Look from Nearby Viewpoints](token_warping_helps_mllms_look_from_nearby_viewpoints.md)
- [\[CVPR 2026\] EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)
- [\[CVPR 2026\] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](linking_perception_confidence_and_accuracy_in_mllms.md)

<!-- RELATED:END -->
