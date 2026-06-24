---
title: >-
  [Paper Note] A Unified Agentic Framework for Evaluating Conditional Image Generation
description: >-
  [ACL 2025][Image Generation][Conditional Image Generation] CIGEval is proposed as a unified evaluation framework based on Large Multimodal Model (LMM) Agents. By integrating various tools (Grounding, Highlight, Difference, Scene Graph) and adopting a divide-and-conquer evaluation strategy, it achieves correlation comparable to human annotators (0.4625 vs. human-to-human 0.47) across 7 conditional image generation tasks, and surpasses the SOTA GPT-4o baseline by fine-tuning a…
tags:
  - "ACL 2025"
  - "Image Generation"
  - "Conditional Image Generation"
  - "LMM Agent"
  - "Evaluation Framework"
  - "Tool Augmentation"
  - "Agent Tuning"
date: 2026-05-08
content_hash: a1aca62994d795f8
---

# A Unified Agentic Framework for Evaluating Conditional Image Generation

**Conference**: ACL 2025  
**arXiv**: [2504.07046](https://arxiv.org/abs/2504.07046)  
**Code**: [https://github.com/HITsz-TMG/Agentic-CIGEval](https://github.com/HITsz-TMG/Agentic-CIGEval)  
**Area**: Image Generation / Evaluation  
**Keywords**: Conditional Image Generation, LMM Agent, Evaluation Framework, Tool Augmentation, Agent Tuning

## TL;DR

CIGEval is proposed as a unified evaluation framework based on Large Multimodal Model (LMM) Agents. By integrating various tools (Grounding, Highlight, Difference, Scene Graph) and adopting a divide-and-conquer evaluation strategy, it achieves correlation comparable to human annotators (0.4625 vs. human-to-human 0.47) across 7 conditional image generation tasks, and surpasses the SOTA GPT-4o baseline by fine-tuning a 7B model on only 2.3K training samples.

## Background & Motivation

Conditional image generation is advancing rapidly, covering 7 mainstream tasks including text-guided generation/editing, subject-driven generation/editing, multi-concept composition, and control-signal-guided generation. However, existing evaluation metrics face three major limitations:

**Task-specific limitation**: LPIPS only measures perceptual similarity, while CLIP-Score only measures text alignment, making them non-generalizable across tasks.

**Lack of interpretability**: Existing methods only provide a single score, lacking reasoning processes and multi-dimensional fine-grained assessments.

**Misalignment with humans**: Traditional metrics (DINO, CLIP) exhibit a large gap compared to human ratings; even the GPT-4o-based VIEScore struggles to capture subtle differences between images.

The core observation of this paper is that **GPT-4o’s native perceptual capability is insufficient for capturing minor differences between highly similar images**, necessitating augmentation via external tools. For instance, in the subject-driven image editing case in Figure 1, GPT-4o directly assigns a high evaluation score, but once focused on the glasses region using the Grounding and Highlight tools, differences in shape and design become apparent.

## Method

### Overall Architecture

CIGEval models image evaluation as an agentic task:

$$f_{\text{eval}}(I, O, C^*) = (\text{rationale}, \text{score})$$

where $I$ represents the evaluation instructions, $O$ is the generated image, and $C^*$ is the set of conditions (text, subject images, control signals, etc.).

A **divide-and-conquer** strategy is employed: each evaluation task is decomposed into multiple fine-grained sub-problems. A suitable tool is selected for each sub-problem, scores are assigned based on tool outputs, and the minimum of the sub-scores is ultimately taken as the final score.

### Key Designs

**1. Multi-functional Toolbox**

| Tool | Input | Output | Function |
|------|------|------|------|
| **Grounding** | Image + Target Entity | Coordinates [x1,y1,x2,y2] | Localizes specific object regions in the image |
| **Highlight** | Image + Region Coordinates | Edited Image | Highlights specified regions (darkening the rest to 1/4 brightness) |
| **Difference** | Image 1 + Image 2 | Difference Region Coordinates | Pixel-level difference detection |
| **Scene Graph** | Image | Structured Description | Object, attribute, and relationship descriptions analyzed by LMM |

- Grounding is implemented based on GroundingDINO.
- Scene Graph is based on CCoT prompting (using either GPT-4o or open-source models).
- Highlight is typically utilized after Grounding or Difference to focus on the target areas.
- Difference detects the locations of differences between two images via pixel comparison.

**2. Fine-grained Evaluation Framework**

Each task is decomposed into a subset of the following sub-problems:
1. Does the generated image follow the text prompt?
2. Does the image editing follow the instructions?
3. Was minimal editing performed without altering the background?
4. Are the objects in the generated image consistent with the given subjects?
5. Does the image follow control signals (such as Canny edges, OpenPose)?

Each sub-problem adopts the **ReAct format** (Observation → Thought → Action), where CIGEval autonomously selects tools, analyzes outputs, and assigns a rating of 0-10.

**3. Score Aggregation**

$$O = \min(\alpha_1, ..., \alpha_i)$$

Using the min operation rather than the average emphasizes that every condition must be met, as a failure in any single dimension is unacceptable.

### Loss & Training

**Agent Tuning**: GPT-4o is employed to execute the evaluation process to generate trajectory data. Trajectories are filtered by excluding samples where the predicted score deviates from human scores by >0.3, ultimately yielding **2,274** high-quality trajectories.

Finetuning strategy (on Qwen2-VL-7B / Qwen2.5-VL-7B):
- Each trajectory is represented as $\langle o_0, t_1, a_1, ..., o_{n-1}, t_n, a_n, o_n \rangle$.
- The cross-entropy loss is computed only on the thoughts $t_i$ and actions $a_i$, while preceding trajectory contexts $c_i$ are masked.
- Learning rate is 1e-5, batch size is 128, and sequence length is 32768.
- Optimizing via AdamW with a cosine scheduler and 3% warmup.

## Key Experimental Results

### Main Results

**Spearman Correlation on the ImagenHub Benchmark** (across 7 tasks):

| Method | Avg Correlation |
|------|-----------|
| Human-to-Human | 0.4700 |
| VIEScore (GPT-4o) | 0.4459 |
| **CIGEval (GPT-4o)** | **0.4625** |
| CLIPScore / LPIPS / DINO | Only applicable to specific tasks |

CIGEval (GPT-4o) outperforms VIEScore on **all 7 tasks**, with particularly notable improvements in multi-conditional tasks:
- Multi-concept IC: 0.4516 → **0.4931**
- Control-guided IG: 0.4972 → **0.5402**

**Open-source models after Agent Tuning**:

| Model | Pre-tuning Avg | Post-tuning Avg | Gain |
|------|----------|----------|------|
| Qwen2-VL-7B | 0.2840 | **0.4997** | +76% |
| Qwen2.5-VL-7B | 0.3455 | **0.4631** | +34% |

Both fine-tuned 7B models surpass VIEScore (GPT-4o)'s correlation of 0.4459!

### Ablation Study

**Ablation on Tools** (CIGEval GPT-4o version):

| Configuration | Avg Correlation |
|------|-----------|
| Full CIGEval | **0.7262** |
| w/o Grounding | 0.6376 (-8.9%) |
| w/o Difference | 0.7020 (-2.4%) |
| w/o Scene Graph | 0.6471 (-7.9%) |
| Scene Graph using Qwen2.5-VL-7B | 0.7120 (-1.4%) |
| Scene Graph using Qwen2.5-VL-70B | 0.7311 (+0.5%) |

Every tool contributes, with Grounding and Scene Graph exerting the greatest impact. Replacing Scene Graph with open-source models leads to only minor degradation, indicating robust framework performance.

### Key Findings

1. Tool augmentation is crucial: LMM perceptual abilities alone cannot distinguish subtle differences between highly similar images.
2. A small amount of high-quality trajectory data (2.3K) can significantly boost the evaluation capacity of small open-source models.
3. Multi-conditional tasks (subject-driven editing, multi-concept composition, control-guided generation) present major evaluation challenges, where CIGEval demonstrates its most prominent advantages.
4. Images generated by GPT-4o still exhibit clear deficiencies on tasks requiring multiple input images and control signals.

## Highlights & Insights

1. **Agent Paradigm for Evaluation**: Formulating the evaluation task as an agentic tool-use process provides an interpretable and scalable evaluation pipeline.
2. **Autonomous Tool Selection**: The agent dynamically determines which tool to employ based on the task type and specific sub-problems, rather than relying on a static workflow.
3. **High Data Efficiency**: Improving 7B models to surpass the GPT-4o baseline with only 2.3K training trajectories highlights the value of high-quality trajectory data.
4. **Unified Framework**: A single framework covers 7 distinct conditional image generation tasks, eliminating the need to design specialized metrics for each task.

## Limitations & Future Work

1. Current work focuses exclusively on Semantic Consistency, leaving Perceptual Quality unaddressed.
2. The toolbox can be expanded further (e.g., OCR tools have already been integrated in case studies).
3. Aggregating scores via the min operation is conservative and may undervalue images that perform slightly worse in one dimension but excel overall.
4. The quality of fine-tuning data relies heavily on GPT-4o's evaluation outputs and the 0.3 threshold filtering, which may introduce GPT-4o's inherent biases.
5. The localization accuracy of GroundingDINO on certain fine-grained objects could pose a bottleneck.

## Related Work & Insights

- **VIEScore**: An evaluation metric using direct prompting with GPT-4o, serving as CIGEval's primary comparison baseline.
- **ImagenHub**: A standardized conditional image generation evaluation benchmark coupled with human-rated datasets.
- **GroundingDINO**: An open-world object detection model powering CIGEval's Grounding tool.
- **ReAct**: The Observation-Thought-Action agent reasoning framework adopted by CIGEval.
- **CCoT (Mitra et al.)**: Chain-of-Composition prompting, which serves as the foundation for the Scene Graph tool.

## Rating

- **Novelty**: ★★★★☆ — The agent-based tool-augmented evaluation paradigm is relatively novel in conditional image generation.
- **Value**: ★★★★★ — The unified framework, compatibility with open-source models, and coverage of 7 mainstream tasks exhibit high engineering maturity.
- **Experimental Thoroughness**: ★★★★★ — Rigorous evaluation across 7 tasks, comprehensive ablation studies, GPT-4o image generation case studies, and agent tuning.
- **Writing Quality**: ★★★★☆ — Clear structure, rich cases, and intuitive charts and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](../../ICCV2025/image_generation/a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](../../CVPR2026/image_generation/agentic_retoucher_for_texttoimage_generation.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[ICCV 2025\] UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](../../ICCV2025/image_generation/unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)
- [\[NeurIPS 2025\] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](../../NeurIPS2025/image_generation/toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)

</div>

<!-- RELATED:END -->
