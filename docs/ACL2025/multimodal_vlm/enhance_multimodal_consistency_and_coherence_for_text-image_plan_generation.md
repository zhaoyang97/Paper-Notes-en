---
title: >-
  [Paper Note] Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation
description: >-
  [ACL 2025][Multimodal VLM][Multimodal planning] This paper proposes an autoregressive text-image plan generation framework (MPlanner) that effectively enhances the coherence of visual steps and text-image consistency in multimodal plans through a four-stage iteration: textual drafting, image editing, visual information extraction, and textual refinement.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal planning"
  - "text-image plan generation"
  - "visual coherence"
  - "cross-modal consistency"
  - "autoregressive framework"
date: 2026-05-08
content_hash: 5a1114a257492961
---

# Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation

**Conference**: ACL 2025  
**arXiv**: [2506.11380](https://arxiv.org/abs/2506.11380)  
**Code**: [Available](https://github.com/psunlpgroup/MPlanner)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal planning, text-image plan generation, visual coherence, cross-modal consistency, autoregressive framework

## TL;DR

This paper proposes an autoregressive text-image plan generation framework (MPlanner) that effectively enhances the coherence of visual steps and text-image consistency in multimodal plans through a four-stage iteration: textual drafting, image editing, visual information extraction, and textual refinement.

## Background & Motivation

Task planning in daily life (such as cooking steps or gardening guides) requires a combination of textual instructions and visual demonstrations. Although LLMs excel at generating textual plans, **multimodal plan generation** (simultaneously outputting textual instructions and paired image sequences) still faces two core challenges:

**Visual Coherence**: There is a lack of visual consistency across images of consecutive steps. For example, a clear glass cup in step 1 becomes a white opaque flowerpot in step 2.

**Text-Image Consistency**: The textual description does not match the content of the corresponding image. For example, the text mentions "indirect sunlight" while the image shows direct sunlight.

Existing methods (such as directly pipeline-coupling an LLM and a text-to-image model) cannot resolve these issues because the image generation module is **unaware of the visual state of the previous step** when generating the current step. While prior work TiP introduced cross-modal prompting, it relied on image descriptions as a bridge, which led to visual incoherence due to information loss.

## Method

### Overall Architecture

MPlanner adopts a four-stage autoregressive architecture, executed at each time step $k$:

**Stage 1 - Textual Plan Drafting**:
$$d_k = \begin{cases} \mathbf{G}_t(\mathcal{G}), & k=1 \\ \mathbf{G}_t(\mathcal{G}, \text{Concat}(t_1, ..., t_{k-1})), & k>1 \end{cases}$$
Generates the current step draft $d_k$ based on the task goal $\mathcal{G}$ and all previous textual steps.

**Stage 2 - Visual Plan Generation**:
$$i_k = \mathbf{G}_i(d_k, i_{k-1})$$
Uses the InstructPix2Pix image editing model to generate the current visual step $i_k$ based on the textual draft $d_k$ and the previous step's image $i_{k-1}$. **The key lies in conditioning on the previous step's image** rather than generating from scratch, ensuring visual coherence.

**Stage 3 - Visual Information Extraction**:
$$v_k = \mathbf{E}(i_k)$$
Drawing inspiration from PDDL in classical planning, a pseudo-PDDL (pPDDL) formatted representation is designed to extract four categories of structured information from the image: **involved objects (objects), tools (tools), action (action), and goal (goal)**.

**Stage 4 - Textual Plan Refinement**:
$$t_k = \mathbf{G}_t(d_k, v_k)$$
Refines the draft $d_k$ using the extracted visual information $v_k$, ensuring that the final textual version $t_k$ is consistent with the actual generated image.

### Key Designs

#### 1. Image Editing Instead of Image Generation

Choosing InstructPix2Pix instead of Stable Diffusion is a core design decision:
- **Image Editing**: Taking the previous step's image as input naturally preserves continuity in object appearance and scene layout.
- **Image Generation**: Generating from scratch at each step is highly prone to visual discontinuities.

To adapt to planning scenarios, InstructPix2Pix was fine-tuned:
- Collected {$i_{k-1}$, $t_k$, $i_k$} triplets for 20,000 tasks from wikiHow.
- Split the training data into 0.9/0.05/0.05.
- The final training loss was 0.100, and the validation loss was 0.105.

#### 2. pPDDL Structured Visual Information

Instead of allowing the model to freely describe the image, a fixed extraction format was designed:

```
Objects: [List]
Tools: [List]
Action: [Description]
Goal: [Description]
```

Ablation studies demonstrate the superiority of this structured representation—free-form text descriptions introduce noise, which then accumulates during the autoregressive process.

#### 3. Plug-and-Play Backbone Compatibility

The framework is compatible with different backbone LLMs:
- **Mistral-7B**: An open-source small model, using InstructBLIP-Vicuna-7B for visual information extraction.
- **Gemini-1.5-Flash**: A closed-source medium model, acting as its own visual information extractor.
- **GPT-4o**: The strongest closed-source model, acting as its own visual information extractor.

### Loss & Training

The main body of this framework requires no training, but the InstructPix2Pix fine-tuning uses standard diffusion model training:
- Fine-tuned on the wikiHow dataset.
- Followed the original hyperparameters of InstructPix2Pix.
- Trained for a maximum of 50 epochs.
- Goal: Learning to generate a reasonable next-step image given the previous step's image and textual instructions.

## Key Experimental Results

### Main Results

**Dataset**: 1,100 tasks collected from Instructables (100 cooking plans) and wikiHow (1,000 expert articles spanning 11 categories).

**Main Results under the GPT-4o Backbone (Summary of Table 1)**:

| Method | BertScore | R-1 | CLIP↑ | PPL↓ | T-I↑ | I-I↑ |
|------|----------|-----|-------|------|------|------|
| GPT (vanilla) | 0.827 | 27.8 | 12.32 | 5.75 | 1.53 | 2.47 |
| Sd | 0.805 | 19.4 | 9.65 | 5.09 | 1.10 | 1.24 |
| TiP | 0.840 | 29.8 | 13.19 | 6.27 | 1.68 | 2.30 |
| **Ours** | **0.849** | **33.7** | **27.14** | **5.21** | **2.47** | **2.76** |

**Consistency Across Backbones**: The three backbones (Mistral-7B, Gemini-1.5, GPT-4o) exhibit consistent trends across both datasets.

**Human Evaluation (Table 2)**: Ours vs. GPT-4o comparison

| Dimension | Win/Tie/Lose | κ |
|------|-------------|---|
| Text Quality | Slightly Better | 0.521 |
| Image Coherence | Significantly Better | 0.604 |
| Text-Image Alignment | Significantly Better | 0.699 |

### Ablation Study

**Ablation Study with GPT-4o Backbone on Instructables (Table 3)**:

| Variant | R-1 | CLIP↑ | PPL↓ | T-I↑ | I-I↑ |
|------|-----|-------|------|------|------|
| w/ Des (Free description replacing pPDDL) | 29.5 | 14.71 | 5.92 | 1.76 | 2.39 |
| w/ Img (Direct refinement with image) | 25.7 | 16.48 | 5.90 | 1.72 | 2.26 |
| pPDDL-to-NL (Format then natural language) | 26.1 | 12.04 | 6.25 | 1.58 | 2.09 |
| **Ours (Full Framework)** | **33.7** | **27.14** | **5.21** | **2.47** | **2.76** |

Key findings from the ablation study:
- **pPDDL > Free Description**: Structured extraction avoids verbose and noisy descriptions.
- **Explicit Extraction > Direct Multimodal Input**: Even though GPT-4o can understand images, it still benefits from an external extractor producing clean information.
- **NL-to-pPDDL > pPDDL-to-NL**: Generating the initial draft in natural language first is better suited to guide image generation.

### Key Findings

1. **Most Significant Improvement in CLIP Score**: The improvement of the Ours method on CLIP (from ~12 to ~27) far exceeds other metrics, indicating that text-image consistency is the most improved aspect.
2. **The SD Baseline Has the Lowest PPL but Is Not the Best**: Because SD-generated images revolve around the same theme with minimal variation, resulting in similar descriptions and low PPL, but they lack actual planning capabilities.
3. **Best Performance on Moderate-Complexity Tasks**: Tasks that are too simple or overly complex are less optimal for this framework.
4. **Visual/Alignment Metrics on wikiHow Are Generally Lower Than Instructables**: This is due to wikiHow tasks being longer and more complex.

## Highlights & Insights

1. **Autoregressive + Cross-Modal Loop**: The closed-loop design of "text $\rightarrow$ image $\rightarrow$ visual information $\rightarrow$ textual refinement" is elegant and effectively addresses the two core challenges.
2. **Introduction of pPDDL**: Bringing classical AI planning concepts (PDDL) into multimodal generation provides an effective template for structured visual information extraction.
3. **Clear Problem Definition**: This is the first work to systematically investigate the coherence and consistency issues in multimodal plan generation.
4. **High Practical Value**: The framework is plug-and-play, compatible with various LLMs and image generation models, and exhibits good scalability.

## Limitations & Future Work

1. **Limitations of InstructPix2Pix**: Visual incoherence still occurs in steps involving drastic scene transformations.
2. **Indirect Visual Evaluation**: Evaluating visual coherence via an "image $\rightarrow$ text $\rightarrow$ PPL" pipeline is indirect and may miss fine-grained nuances in the pixel space.
3. **Data Leakage Risk**: LLMs might have seen similar task plans during pre-training.
4. **Opportunities to Explore Better Image Editing Models**: Such as SDXL-based editing models or direct image generation using VLMs.
5. **Lack of User Studies**: Human evaluation was restricted to the paper's authors (3 annotators); the small pool of evaluators limits the generalizability of the findings.

## Related Work & Insights

- **Task Planning**: Zero-shot planning with LLMs (Huang et al., 2022), LLM + classical planner (Liu et al., 2023).
- **TiP (Lu et al., 2024)**: The most direct predecessor, which uses T2I-Bridge and I2T-Bridge for cross-modal prompting.
- **InstructPix2Pix (Brooks et al., 2023)**: A conditional image-editing model; this work fine-tunes it for planning tasks.
- **PDDL (Fox and Long, 2003)**: A classical planning domain definition language, which this work simplifies into pPDDL for visual information structuring.
- **Insights**: The concept of cross-modal iterative refinement can be extended to other tasks requiring multimodal alignment (such as multimodal story generation or educational content creation).

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Practicality | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Overall Rating | 4 |

The framework design is elegant, the problem definition is clear, and the ablation experiments are thorough. The introduction of pPDDL is a clever cross-disciplinary adaptation. The main limitations lie in the indirect nature of the visual evaluation metrics and the inherent constraints of the image editing model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](code_guided_text_rich_image.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](../../ICCV2025/multimodal_vlm/multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)
- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](../../CVPR2025/multimodal_vlm/comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)
- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](../../CVPR2025/multimodal_vlm/opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)

</div>

<!-- RELATED:END -->
