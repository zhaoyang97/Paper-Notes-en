---
title: >-
  [Paper Note] SketchAgent: Language-Driven Sequential Sketch Generation
description: >-
  [CVPR 2025][Multimodal VLM][Sketch Generation] Without any training or fine-tuning, SketchAgent achieves human-level sketch generation (reaching 85% of human Top-1 recognition rate) stroke-by-stroke using a grid-canvas coordinate system, in-context examples, and a Bézier curve fitting post-processing pipeline designed for pre-trained multimodal LLMs. It supports interactive collaborative drawing and conversational editing.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Sketch Generation"
  - "Language-driven"
  - "Zero-training"
  - "Bézier Curves"
  - "Stroke-by-stroke"
  - "Human-AI Collaboration"
date: 2026-05-08
content_hash: dfc69967c4bde542
---

# SketchAgent: Language-Driven Sequential Sketch Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.17673](https://arxiv.org/abs/2411.17673)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Agent  
**Keywords**: Sketch Generation, Language-driven, Zero-training, Bézier Curves, Stroke-by-stroke, Human-AI Collaboration

## TL;DR
Without any training or fine-tuning, SketchAgent achieves human-level sketch generation (reaching 85% of human Top-1 recognition rate) stroke-by-stroke using a grid-canvas coordinate system, in-context examples, and a Bézier curve fitting post-processing pipeline designed for pre-trained multimodal LLMs. It supports interactive collaborative drawing and conversational editing.

## Background & Motivation

**Background**: Sketch generation research mainly follows two directions: (1) RNN-based methods (such as SketchRNN) that learn to encode vector stroke sequences, which suffer from poor generalization and lack semantic control; (2) Diffusion-based optimization methods (such as SVGDreamer), which yield good results but are time-consuming (~1.6 hours/image) and produce outputs resembling design blueprints rather than hand-drawn sketches. Existing methods generally lack the progressive, semantic characteristics of human drawing.

**Limitations of Prior Work**: (1) Traditional methods require training/fine-tuning on sketch datasets, limiting generalization to new concepts; (2) Directly prompting LLMs to output SVG code results in mechanical, rigid lines lacking a hand-drawn feel; (3) LLMs possess weak spatial reasoning, easily making errors when using pixel coordinates; (4) Existing methods do not support human-AI collaborative drawing—unable to pause mid-way to allow user stroke additions before continuing.

**Key Challenge**: Multimodal LLMs have rich visual and semantic priors and theoretically "understand" the visual structure of objects, but they lack the ability to translate semantic understanding into precise spatial coordinates.

**Goal**: How to leverage the prior knowledge of pre-trained LLMs to generate human-style stroke-by-stroke sketches without training, while supporting interactive editing?

**Key Insight**: Instead of modifying the model, modify the interface: design a simple "drawing language" (grid coordinates + stroke formats) to teach LLMs sketch construction via in-context learning, and then use Bézier curve fitting to transform rough coordinates into smooth lines.

**Core Idea**: Using a grid canvas to mitigate the spatial reasoning weaknesses of LLMs + in-context learning to teach drawing formats to LLMs + Bézier curve post-processing to generate smooth, hand-drawn style sketches, all with zero training.

## Method

### Overall Architecture
Pipeline: User text prompt $\to$ LLM plans drawing strategy in the `<thinking>` tag $\to$ stroke-by-stroke output of grid coordinate sequences $\to$ Bézier curve fitting $\to$ rendering as vector sketches. Three modes are supported: independent generation, collaborative drawing (users can add strokes mid-way), and conversational editing (feedback drawn content to the model for linguistic instructions).

### Key Designs

1. **The Canvas**:

    - **Function**: Converts LLM coordinate outputs from pixel space to low-resolution grid space, significantly reducing spatial reasoning difficulty.
    - **Mechanism**: Defines a 50×50 numbered grid where each cell is identified by coordinates (e.g., `x2y8`). LLMs only need to output grid coordinate sequences in the format of `<points>x1y1, x15y20, ...</points>`. Each stroke includes a semantic ID label for analysis.
    - **Design Motivation**: Directly using pixel coordinates causes the spatial reasoning of LLMs to break down. A 50×50 grid is sufficient to express the rough structure of the sketch while keeping it easy for the LLM to understand and generate. Although resolution is sacrificed, sketches are inherently abstract, making this a perfect match.

2. **Bézier Curve Smoothing**:

    - **Function**: Transforms the rough grid coordinate points output by the LLM into smooth, natural hand-drawn lines.
    - **Mechanism**: Fits the coordinate sequence of each stroke to a cubic Bézier curve $B(t) = (1-t)^3P_0 + 3(1-t)^2tP_1 + 3(1-t)t^2P_2 + t^3P_3$. The LLM simultaneously outputs parameter $t$ values corresponding to each point, and control points are solved using least squares $P = \text{argmin}_P ||AP - B||$. Iterative splitting of curve segments is performed when fitting error is high.
    - **Design Motivation**: Simply connecting grid coordinates leads to jagged, mechanical lines; Bézier fitting gives the lines a hand-drawn smoothness. Having the LLM output $t$ values is a clever design—it essentially allows the model to express "where this point lies along the stroke's path", providing richer information than coordinates alone.

3. **In-Context Learning + Chain-of-Thought Drawing Planning**:

    - **Function**: Enables the LLM to learn drawing formats and strategies without training.
    - **Mechanism**: The system prompt introduces the grid canvas and drawing language rules; the user prompt contains a complete house sketch example (crucial) + several single-stroke primitive examples. Before drawing, the model describes its strategy in the `<thinking>` tag—decomposing object components, planning stroke order, and designing positions of each part, followed by stroke-by-stroke coordinate output.
    - **Design Motivation**: Ablative experiments show that complete examples (Modified ICL) have the greatest impact on performance (removing them drops Top-1 from 0.23 to 0.07), demonstrating that the LLM needs to "see a complete finished drawing" to understand the task. CoT planning helps the model decompose the component structure of complex objects.

### Loss & Training
**Zero-training/fine-tuning**—the model entirely utilizes pre-trained Claude 3.5-Sonnet, with only the least-squares optimization (closed-form solution) of Bézier curve fitting. Generation takes ~20 seconds/image.

## Key Experimental Results

### Main Results

| Method | Top-1 Recognition Rate | Top-5 Recognition Rate |
|--------|------|------|
| GPT-4o | 0.15 ± 0.04 | 0.30 ± 0.06 |
| Claude-3.5-Sonnet (Direct SVG) | 0.23 ± 0.05 | 0.44 ± 0.03 |
| **SketchAgent** | **0.23 ± 0.04** | **0.43 ± 0.06** |
| Human (QuickDraw) | 0.27 ± 0.07 | 0.49 ± 0.06 |

2AFC User Study: 74.9% of participants found SketchAgent's sketches to be more human-like than direct prompting. When compared to real human sketches, SketchAgent was chosen as "more human-like" with a probability of 45.3% (very close to a 50/50 split).

### Ablation Study

| Configuration | Top-1 | Top-5 |
|------|---------|---------|
| **SketchAgent (Full)** | **0.23** | **0.43** |
| w/o System Prompt | 0.20 | 0.42 |
| w/o CoT | 0.14 | 0.29 |
| Modified ICL (w/o complete example) | 0.07 | 0.16 |

### Key Findings
- **Complete examples are the most critical factor**: Removing them causes Top-1 to plummet by 70% (0.23 $\to$ 0.07), proving that the LLM needs demonstrations to understand the output format and strategy of drawing tasks.
- **CoT planning contributes significantly**: Removing it drops Top-1 by 39% (0.23 $\to$ 0.14), demonstrating that planning before drawing is vastly superior to direct drawing.
- **Direct SVG prompting has comparable CLIP scores but different visual styles**: Web/SVG methods yield mechanical, blueprint-like vector drawings, whereas SketchAgent yields hand-drawn, progressive sketches; both show similar recognition rates but very different human preferences.
- **Collaborative drawing is effective**: In a user study with 30 participants, the recognition rate of collaborative sketches was close to that of single players, and both parties actively contributed.
- **Conversational editing accuracy is 92%**: Spatial relationship instructions ("on the left") achieved 94% accuracy, and semantic relation inference ("put a hat on the animal's head") achieved 88% accuracy.

## Highlights & Insights
- **Elegance of the Zero-Training Paradigm**: Achieving near-human sketch generation behavior purely through prompt engineering + post-processing without training, fine-tuning, or optimization. This proves that LLM priors are far richer than we currently exploit, and the key lies in designing correct interfaces.
- **Grid Canvas as a General Spatial Reasoning Enhancement**: The 50×50 grid discretizes continuous space into a representation that is much easier for LLMs to manage. This concept can be generalized to any task requiring LLM spatial reasoning (e.g., layout design, map navigation).
- **Possibility of Human-AI Collaborative Sketching**: Fluid collaborative drawing between humans and AI is realized via stopping tokens and coordinate conversion mechanisms. This interactive paradigm holds immense potential for creative tools.

## Limitations & Future Work
- Constrained by LLM visual priors: The model can verbally describe complex concepts (e.g., a unicorn's horn) but cannot draw them well; a gap remains between semantic understanding and spatial execution.
- Poor performance on human figures and letters/symbols, due to LLM's insufficient spatial priors on human anatomy/character structures.
- The 50×50 grid is highly abstract, meaning detailed representation is limited (though this is somewhat a feature of sketches).
- Highly dependent on specific models (Claude 3.5-Sonnet); performance might degrade significantly with other LLMs.
- Generation time is ~20 seconds, which is slower than SketchRNN (~4 seconds) but much faster than optimization approaches (~1.6 hours).

## Related Work & Insights
- **vs SketchRNN**: SketchRNN requires training and generalizes poorly; SketchAgent requires zero training and can sketch any concept, albeit with lower precision.
- **vs SVGDreamer**: SVGDreamer generates high-quality SVGs via optimization but takes 1.6 hours; SketchAgent outputs in 20 seconds with a style closer to hand-drawn sketches.
- **vs Direct SVG prompting**: Similar recognition rates, but SketchAgent's strokes are more natural and human-preferred. The key distinction lies in Bézier fitting and stroke-by-stroke generation strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ The zero-training scheme is highly novel, and the combination of grid + Bézier + ICL is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Assessed across recognition rates, user studies, ablations, and collaboration/editing.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-illustrated, with clear descriptions of method motivations and design choices.
- Value: ⭐⭐⭐⭐ Demonstrates the potential of LLMs in visual creative tasks, with inspiring interaction design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[AAAI 2026\] PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography](../../AAAI2026/multimodal_vlm/pet2rep_towards_vision-language_model-drived_automated_radiology_report_generati.md)
- [\[ICLR 2026\] Reading Images Like Texts: Sequential Image Understanding in Vision-Language Models](../../ICLR2026/multimodal_vlm/reading_images_like_texts_sequential_image_understanding_in_vision-language_mode.md)
- [\[CVPR 2026\] Language-driven Fine-grained Retrieval](../../CVPR2026/multimodal_vlm/language-driven_fine-grained_retrieval.md)
- [\[AAAI 2026\] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model](../../AAAI2026/multimodal_vlm/o3slm_open_weight_open_data_and_open_vocabulary_sketch-language_model.md)

</div>

<!-- RELATED:END -->
