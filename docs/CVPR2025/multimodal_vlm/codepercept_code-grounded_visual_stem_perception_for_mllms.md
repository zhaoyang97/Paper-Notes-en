---
title: >-
  [Paper Note] CodePercept: Code-Grounded Visual STEM Perception for MLLMs
description: >-
  [CVPR 2025][Multimodal VLM][STEM perception] Through scaling analysis, this work discovers that the true bottleneck of STEM visual reasoning is perception rather than reasoning, and proposes using executable Python code as a precise perceptual medium. By constructing the ICC-1M dataset (Image-Caption-Code triplets) for training, CodePercept-8B improves by $+3.0\%$ to $+12.3\%$ over Qwen3-VL-8B on STEM perception benchmarks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "STEM perception"
  - "code generation"
  - "visual reasoning"
  - "data synthesis"
  - "perception bottleneck"
date: 2026-05-08
content_hash: 1330c7e360fd337d
---

# CodePercept: Code-Grounded Visual STEM Perception for MLLMs

**Conference**: CVPR 2025  
**arXiv**: [2603.10757](https://arxiv.org/abs/2603.10757)  
**Code**: [https://github.com/TongkunGuan/Qwen-CodePercept](https://github.com/TongkunGuan/Qwen-CodePercept)  
**Area**: Multimodal VLM  
**Keywords**: STEM perception, code generation, visual reasoning, data synthesis, perception bottleneck

## TL;DR

Through scaling analysis, this work discovers that the true bottleneck of STEM visual reasoning is perception rather than reasoning, and proposes using executable Python code as a precise perceptual medium. By constructing the ICC-1M dataset (Image-Caption-Code triplets) for training, CodePercept-8B improves by $+3.0\%$ to $+12.3\%$ over Qwen3-VL-8B on STEM perception benchmarks.

## Background & Motivation

**Background**: Current improvements of MLLMs in the STEM field are concentrated on reasoning capabilities—such as cold-start data, RL training, and text-only thinking data transfer. A massive amount of work employs RL rewards to enhance mathematical and scientific reasoning.

**Limitations of Prior Work**: When a model fails on a STEM task, it remains unclear whether the cause is insufficient perception or inadequate reasoning. Traditional benchmarks only measure problem-solving accuracy and cannot separate these two abilities.

**Key Challenge**: The authors reveal through scaling experiments that by decoupling STEM reasoning into perception (image $\to$ caption) and reasoning (caption $\to$ answer), scaling perception consistently outperforms scaling reasoning when independently scaling both. This indicates that perception is the true lever.

**Goal**: How to systematically enhance the visual perception capabilities of MLLMs in the STEM domain?

**Key Insight**: Natural language descriptions of STEM images suffer from "descriptive aphasia"—complex spatial relationships and precise numerical values cannot be fully expressed in natural language. Conversely, executable code naturally possesses precise semantics and highly aligns with the structured characteristics of STEM images.

**Core Idea**: Use executable Python code as the ground truth and training medium for STEM visual perception; only by accurately reconstructing the image can the model prove it truly understands the image.

## Method

### Overall Architecture

Three main components: (1) ICC-1M dataset construction (1M Image-Caption-Code triplets), (2) two Code-Grounded training tasks (code-driven caption generation + image-to-code translation), and (3) the STEM2Code-Eval benchmark (evaluating perception capabilities through code reconstruction). Training consists of two phases: SFT + RL.

### Key Designs

1. **Scaling Analysis (Key Findings)**:

    - **Function**: Decouples STEM reasoning into two phases: perception ($MLLM_{captioner} \to$ description) and reasoning ($LLM_{solver} \to$ answer).
    - **Experimental Design**: Fixes one side with a 4B model and varies the other with 4B/8B/32B models to observe performance changes.
    - **Key Conclusion**: Scaling perception consistently yields larger gains than scaling reasoning. Verified on MathVision using Qwen3-VL-Thinking models. This answers a long-standing question—the root cause of STEM failures is perception.

2. **ICC-1M Dataset Construction (Three Pipelines)**:

    - **Image Reproduce (IR)**: Uses MLLMs to generate reconstruction code for existing STEM images—first generating a caption to understand the content, and then generating code based on both the caption and the original image.
    - **Image Diversity (ID)**: Extracts underlying STEM principles from seed images and then re-instantiates them in different visual contexts (e.g., from domino logic puzzles to circular domino wheels, triangular arrangements, etc.) to expand diversity.
    - **Solid Geometry (SG)**: Generates solid geometry images and code using parametric templates, addressing the issue where LLMs struggle to generate accurate 3D spatial code.
    - **Unified Quality Control**: A triple-filtering mechanism checking image quality, code quality, and image-code consistency.

3. **Code-Grounded Caption Generation**:

    - **Function**: Leverages executable code as ground truth to generate accurate captions.
    - **Mechanism**: First generates a native caption (which may contain hallucinations) $\to$ analyzes code and execution logs to extract verified visual facts $\to$ corrects errors in the caption using these visual facts.
    - **Highlight**: The execution tracer automatically records precise information such as geometric coordinates, quantities, and colors, resolving the difficulty of directly analyzing complex code logic.
    - **Design Motivation**: Directly asking MLLMs to describe STEM images leads to hallucinations regarding numerical values and spatial relationships.

4. **STEM Image-to-Code Translation**:

    - **Function**: Trains the model to directly generate executable reconstruction code from images.
    - **Mechanism**: First generates an explanatory code draft (with steps explained but potentially incorrect) $\to$ corrects errors using the ground truth code while preserving the explanatory structure.
    - **Code serves as a "structured caption,"** complementing the natural language caption.

### Loss & Training

- **SFT**: Jointly trains the tasks image $\to$ caption and image $\to$ code, allowing semantic understanding and structured understanding to mutually reinforce each other.
- **RL (GRPO)**: Conducts RL solely for code generation, where $\text{reward} = \text{format reward}$ (valid Python blocks) $+$ $\text{content reward}$ (executability $+$ GPT-4o rated code semantics $+$ GPT-4o rated image similarity).
- Based on the Qwen3-VL series, trained on $32 \times \text{A100}$ GPUs.

## Key Experimental Results

### Main Results (Captioner-Solver Perception Evaluation)

| Model | MathVision | MathVista | MathVerse | DynaMath | WeMath | LogicVista | Avg |
|------|-----------|-----------|-----------|---------|--------|------------|-----|
| Qwen3-VL-8B-Instruct | 54.37 | 69.60 | 63.75 | 72.19 | 45.43 | 56.82 | 60.36 |
| **CodePercept-8B-S1** | 59.31 (+5.0) | 70.20 (+0.6) | 66.52 (+2.8) | 73.20 (+1.0) | 49.14 (+3.7) | 61.52 (+4.7) | **63.32 (+3.0)** |
| CodePercept-32B-S1 | 62.27 (+3.7) | 72.90 | 71.70 | 77.41 | 54.19 (+6.2) | 65.33 | **67.30 (+2.7)** |

### STEM2Code-Eval (Code Reconstruction Evaluation)

| Model | Image Score | Code Score | Avg | Exec Rate |
|------|-----------|-----------|-----|-----------|
| Qwen3-VL-8B-Instruct | 28.59 | 28.23 | 28.41 | 85.3% |
| CodePercept-8B-S1 | 44.53 | 46.78 | 45.66 | 87.6% |
| CodePercept-8B-R1 | **50.25** | **47.04** | **48.65** | 93.4% |
| Gemini2.5-Pro-Thinking | 68.89 | 75.41 | 72.15 | 91.7% |

### Ablation Study

| Data Configuration | Avg Score |
|---------|----------|
| Qwen3-VL-8B baseline | 60.36 |
| + IR-CodeCap | 60.91 (+0.6) |
| + ID-CodeCap | 62.15 (+1.8) |
| + SG-CodeCap | 62.75 (+2.4) |
| NativeCap (direct caption, without code) | 60.78 |
| CodeCap (code-driven caption) | 62.75 (+2.0) |
| CodeCap + ImCode (joint training) | **63.32** (+2.5) |

### Key Findings

- **Empirical Evidence of Perception as Bottleneck**: The scaling analysis is highly convincing—independently scaling perception capabilities consistently yields higher gains than scaling reasoning.
- **Code-Driven Caption Outperforms Direct Caption by $+2.0\%$**: Verifies the effectiveness of code in eliminating hallucinations.
- **Complementarity of Caption and Code**: Joint training performs better than using caption alone ($+0.6\%$) or code alone.
- **RL Significantly Benefits Code Generation**: CodePercept-8B-R1 vs S1 improves by $+3.0$ on STEM2Code-Eval, with an increase of $+5.8\%$ in Exec Rate.
- **CodePercept-8B Outperforms Multiple 72B Models**: Surpasses Qwen2.5-VL-72B in captioner-solver evaluation.

## Highlights & Insights

- **"Perception as the Bottleneck" is a neglected yet highly crucial finding**. Current STEM AI research focused almost exclusively on reasoning (RL, CoT), but this paper demonstrates through controlled experiments that perception should be solved first. This could potentially shift research priorities in the field.
- **The insight of "code as a perceptual medium" is highly elegant**: Code naturally possesses precision (coordinates, color values), verifiability (executability), and structure (hierarchical descriptions), all of which are lacking in natural language.
- **Execution Tracer**: Executes code and logs all rendering details (coordinates, z-order, RGB colors) as a "manual" for code analysis, cleverly addressing the challenge where LLMs struggle to analyze complex code logic.
- **Design Philosophy of STEM2Code-Eval Benchmark**: An image is truly understood only if it can be reconstructed—evaluating perception more comprehensively than simple question-answering.

## Limitations & Future Work

- **Code generation is limited to Matplotlib**: Certain STEM images (e.g., real-world photos, hand-drawn diagrams) cannot be reconstructed using Matplotlib.
- **RL reward depends on GPT-4o**: GPT-4o scoring is unstable and expensive; future work could explore automatic rewards based on pixel similarity.
- **Evaluation restricted to the STEM domain**: Can the approach of "code as a perceptual medium" be generalized to other fields (e.g., medical imaging, remote sensing)?
- **Imperfect decoupling between perception and reasoning**: Caption quality simultaneously affects both perception and representation capabilities, leading to incomplete decoupling.

## Related Work & Insights

- **vs. RL reasoning works such as Vision-R1/Video-R1**: While these works put all effort into boosting reasoning, CodePercept points out they might be optimizing in the wrong direction—perception should be enhanced first.
- **vs. Chart/UI code generation**: Those target downstream code generation applications, whereas CodePercept utilizes code as a means of perception enhancement; the goals differ, but the techniques are transferable.
- **Insight**: Future work might apply the concept of "code as a perceptual medium" to other modalities (e.g., audio $\to$ MIDI, 3D $\to$ rendering code) to enhance understanding.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Highly original with the dual insight of "perception as the bottleneck" and code as the perception medium.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 6 STEM benchmarks, scaling analysis, comprehensive ablations, and 3 model scales (4B/8B/32B).
- **Writing Quality**: ⭐⭐⭐⭐ Structured and clear, though slightly formulas-heavy.
- **Value**: ⭐⭐⭐⭐⭐ High potential to shift the priority of STEM AI research, with comprehensive contributions in benchmarks, datasets, and methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] StarVector: Generating Scalable Vector Graphics Code from Images and Text](starvector_generating_scalable_vector_graphics_code_from_images_and_text.md)
- [\[CVPR 2025\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2025\] Evaluating Model Perception of Color Illusions in Photorealistic Scenes](evaluating_model_perception_of_color_illusions_in_photorealistic_scenes.md)
- [\[CVPR 2025\] VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?](vidcomposition_can_mllms_analyze_compositions_in_compiled_videos.md)
- [\[CVPR 2025\] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs](peace_empowering_geologic_map_holistic_understanding_with_mllms.md)

</div>

<!-- RELATED:END -->
