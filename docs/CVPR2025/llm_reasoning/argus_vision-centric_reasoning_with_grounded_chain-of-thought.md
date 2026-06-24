---
title: >-
  [Paper Note] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought
description: >-
  [CVPR 2025][Reasoning][Visual Chain-of-Thought] Argus proposes a grounded visual CoT mechanism that enables explicit target-oriented visual attention by first making the MLLM predict a question-related bounding box (RoI), and then resampling/re-encoding the visual tokens of that region as reasoning context, achieving dual SOTA in visual reasoning and object grounding among 7B/8B-scale MLLMs.
tags:
  - "CVPR 2025"
  - "Reasoning"
  - "Visual Chain-of-Thought"
  - "Visual Attention Grounding"
  - "Multimodal Reasoning"
  - "RoI Re-engagement"
  - "Mixture of Vision Experts"
date: 2026-05-08
content_hash: ff03f368ca6f3141
---

# Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought

**Conference**: CVPR 2025  
**arXiv**: [2505.23766](https://arxiv.org/abs/2505.23766)  
**Code**: [https://yunzeman.github.io/argus/](https://yunzeman.github.io/argus/) (Project Page)  
**Area**: LLM Reasoning  
**Keywords**: Visual Chain-of-Thought, Visual Attention Grounding, Multimodal Reasoning, RoI Re-engagement, Mixture of Vision Experts

## TL;DR

Argus proposes a grounded visual CoT mechanism that enables explicit target-oriented visual attention by first making the MLLM predict a question-related bounding box (RoI), and then resampling/re-encoding the visual tokens of that region as reasoning context, achieving dual SOTA in visual reasoning and object grounding among 7B/8B-scale MLLMs.

## Background & Motivation

**Background**: Existing MLLMs perform well on general vision-language tasks, but fall short in vision-centric scenarios that require precise visual focusing, such as identifying spatial relations of small objects or reading specific data in charts.

**Limitations of Prior Work**: Existing MLLMs primarily rely on implicit self-attention mechanisms to handle interactions between visual tokens and language tokens, lacking explicit target-oriented visual search capabilities. Although Cambrian-1 and Eagle studied the complementarity of multiple vision encoders, they still did not introduce conscious visual attention control.

**Key Challenge**: Cognitive science distinguishes between two types of visual attention: stimulus-driven involuntary attention (bottom-up, triggered by salient objects in the image) and goal-directed voluntary attention (top-down, directed by task goals). The ViT encoders of existing MLLMs correspond to the former, while the cross-attention in LLMs implicitly implements the latter, but this implicit goal-directed attention is neither precise nor controllable.

**Goal**: (1) How to introduce explicit language-guided visual attention into MLLMs? (2) Whether this more explicit visual focus can improve reasoning task performance?

**Key Insight**: Drawing inspiration from the concept of voluntary attention in cognitive science, object-centric grounding (predicting a bounding box) is utilized as an intermediate signal for visual CoT—first letting the model find "where to look", and then letting the model "look closely" for reasoning.

**Core Idea**: Use the grounded bounding box prediction as a visual CoT signal to achieve explicit target-oriented visual attention through visual token resampling/re-encoding in the RoI area.

## Method

### Overall Architecture

The pipeline of Argus consists of a two-stage reasoning process: (1) Given an image and a question, the model first performs initial image encoding via the Mixture of Vision Experts (MoVE), and then **predicts the bounding box most relevant to the question** (with RoI coordinates output as text); (2) Based on the predicted bounding box, the model **crops/samples the corresponding region** from the original image, appends its visual tokens to the input sequence as additional CoT context, and then generates the final answer based on this focused visual information.

### Key Designs

1. **Mixture of Vision Experts (MoVE)**:

    - **Function**: Extract complementary visual features and minimize information loss during the image-to-token process.
    - **Mechanism**: Simultaneously utilize three vision experts: CLIP ViT-L/14 ($448 \times 448$, semantic alignment), ConvNeXt-XXL ($1024 \times 1024$, fine-grained spatial features), and EVA-02-L/16 ($1024 \times 1024$, detection-oriented features). Their features are interpolated to a unified $32 \times 32$ spatial resolution, concatenated along the channel dimension (5120-dim), and mapped to the LLM's 4096-dim space via an MLP, producing 1024 visual tokens.
    - **Design Motivation**: Different vision encoders have distinct strengths—CLIP excels in semantic alignment, ConvNeXt preserves detailed textures, and EVA-02 is proficient in object detection. Their complementarity enables a more comprehensive visual understanding.

2. **RoI Prediction and Visual Context Re-engagement**:

    - **Function**: Implement explicit goal-directed visual search, prompting the model to "find first, then look."
    - **Mechanism**: The model outputs a normalized bounding box as textual coordinates $[x_{min}, y_{min}, x_{max}, y_{max}]$, and performs RoI sampling based on this box. The paper systematically compares four visual attention re-engagement strategies: (a) Implicit self-attention (baseline, no additional processing); (b) Implicit box guidance (outputs box coordinates text as CoT only, without re-encoding visual tokens); (c) Explicit RoI re-encoding (cropping the region as a new image fed into the vision encoders); (d) Explicit RoI resampling (retrieving patch tokens overlapping with the box from the initial token cache).
    - **Design Motivation**: Implicit methods have limited control over the attention of visual tokens, while explicit methods force the model to focus on key areas by actually extracting the visual tokens of the RoI region. Resampling is more efficient (reusing cached tokens), while re-encoding is superior in small object perception (high-resolution processing).

3. **Grounded CoT Training Strategy**:

    - **Function**: Jointly train grounding and reasoning capabilities.
    - **Mechanism**: The SFT stage uses a mixture of three datasets for training—Eagle1.8M (general reasoning), VCoT dataset (CoT reasoning with bounding box annotations, including TextVQA/DocVQA/ScienceQA, etc.), and GRIT+Shikra (large-scale grounding data). The training format is a multi-turn conversation: the model first outputs `<roi-box>` coordinates, the user provides the `<visual-context>` tokens, and the model then generates the answer.
    - **Design Motivation**: Grounding data enhances the model's object perception capabilities, thereby improving the quality of bounding box predictions and maximizing the utility of the CoT mechanism.

### Loss & Training

Two-stage training: (1) Pre-training stage: utilizes LLaVA-595K, freezes the LLM, and trains the vision encoders and MLP (32×A100, 4 hours); (2) SFT stage: full-parameter fine-tuning for 1 epoch with batch size 256, learning rate 2e-5, AdamW, and cosine scheduler (64×A100, 28 hours).

## Key Experimental Results

### Main Results

| Model | Vision-Centric Avg | V-Star | CV-Bench-2D | MMVP | TextVQA | ChartQA | MMMU | MMBench |
|------|-------------------|--------|-------------|------|---------|---------|------|---------|
| GPT-4o | 73.7 | 70.7 | 79.8 | 58.5 | 79.7 | 86.9 | 68.9 | 87.1 |
| Qwen2.5-VL | 72.6 | 72.8 | 80.0 | 53.1 | 84.9 | 85.2 | 58.6 | 86.5 |
| Eagle-X3-8B | 59.6 | 60.7 | 66.4 | 45.1 | 70.9 | 70.4 | 39.8 | 70.9 |
| Visual-CoT-7B | 54.4 | 49.7 | 61.5 | 35.7 | 70.0 | 69.7 | 37.2 | 67.3 |
| **Argus-X3-8B** | **65.3** | **78.5** | **68.5** | **45.5** | **73.6** | **74.8** | 40.4 | **72.9** |

Referring Grounding (RefCOCO):

| Model | RefCOCO-val | RefCOCO-testA | RefCOCO+-val | RefCOCOg-val |
|------|-------------|---------------|--------------|--------------|
| G-DINO-L (Expert) | 90.6 | 93.2 | 82.8 | 86.1 |
| QwenVL-7B | 89.4 | 92.3 | 83.1 | 85.6 |
| **Argus-X3-8B** | **89.8** | **92.9** | **84.7** | **86.7** |

### Ablation Study

| Visual Attention Strategy | V-Star | CV-Bench-2D | TextVQA | ChartQA |
|---------------|--------|-------------|---------|---------|
| Implicit Self-Att | 58.6 | 64.5 | 69.2 | 67.3 |
| Implicit Box Guidance | 63.9 | 67.0 | 71.6 | 70.4 |
| Explicit RoI Re-encoding | 68.1 | 67.4 | 71.4 | 71.8 |
| Explicit RoI Resampling | 67.0 | 68.2 | 73.9 | 72.7 |

Joint Effects of CoT and Grounding:

| Configuration | V-Star | CV-Bench-2D | TextVQA | ChartQA |
|------|--------|-------------|---------|---------|
| Baseline (Eagle-X3) | 55.3 | 64.9 | 66.3 | 63.0 |
| + CoT signals | 62.7 | 65.5 | 71.1 | 69.4 |
| ++ Grounding (Argus) | 67.0 | 68.2 | 73.9 | 72.7 |

### Key Findings

- Explicit visual RoI re-engagement (both resampling and re-encoding) consistently outperforms implicit methods, proving the effectiveness of the "find first, then look" strategy.
- Resampling outperforms re-encoding on most tasks as it preserves the original positional information and avoids distribution shift caused by resolution transformation. However, on V-Star (small object perception), re-encoding is superior due to its ability to handle small areas with a larger patch size.
- Resampling is significantly more computationally efficient: GMACs are only 4355 vs 8711, extra tokens are only 26 vs 1024, and inference time is 492ms vs 827ms.
- Multi-RoI extension (expanding from single-target to multi-target reasoning) improves performance from 68.1 to 78.5 on V-Star, and from 64.2 to 69.6 on CV-Bench-2D.

## Highlights & Insights

- **Elegant analogy of Visual CoT**: Mapping the concepts of involuntary/voluntary attention in cognitive science to the MLLM's ViT encoding (stimulus-driven) and RoI re-engagement (goal-directed) provides clear theoretical motivation.
- **Note-worthy efficiency gains of the resampling strategy**: Adding only 26 tokens significantly improves reasoning performance, which is much more efficient than re-encoding 1024 tokens, making it highly suitable for practical deployment.
- **Positive feedback loop between grounding and reasoning**: Grounding data increases the accuracy of box predictions $\rightarrow$ better CoT signals $\rightarrow$ better reasoning results; this synergistic effect is key to the success of the methodology.

## Limitations & Future Work

- It only validates LLMs at the 8B scale, leaving untested whether larger scale models (e.g., 70B) could further amplify the benefits of visual CoT.
- Visual CoT training data is scarce; existing data primarily originates from text understanding and science QA scenarios, lacking more diverse visual reasoning CoT annotations.
- The multi-RoI extension currently requires multi-step sequential inference, leaving room for efficiency optimization.
- The improvement on MMMU and GQA is limited, which the authors attribute to these benchmarks relying more on language priors than on visual information.

## Related Work & Insights

- **vs Eagle**: Argus adds the visual CoT mechanism on top of Eagle-X3, boosting performance on V-Star from 60.7 to 78.5 ($+17.8$) while sharing the same MoVE architecture, demonstrating the immense value of explicit visual attention.
- **vs Visual-CoT**: Visual-CoT relies on an external object detector to provide the RoI, whereas Argus internalizes the grounding capability into the model itself to achieve end-to-end training, comprehensively outperforming it.
- **vs Cambrian-1**: Both emphasize vision-centric designs, but Cambrian-1 focuses on the combination of encoders while Argus focuses on the attention mechanism during reasoning, making them complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The formulation of visual CoT is clear and elegant, though the idea of grounding + re-engagement has roots in prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The ablation studies are comprehensive and systematic, covering four re-engagement strategies, encoder capacity, context expansion, multi-RoIs, etc.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-structured, the analogy to cognitive science is engaging, and the figures are intuitive.
- Value: ⭐⭐⭐⭐ It provides a clear direction for improving visual reasoning in MLLMs, and the resampling strategy is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization](../../CVPR2026/llm_reasoning/revisiting_the_necessity_of_lengthy_chain-of-thought_in_vision-centric_reasoning.md)
- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)
- [\[CVPR 2025\] VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)
- [\[ACL 2025\] Improve Vision Language Model Chain-of-thought Reasoning](../../ACL2025/llm_reasoning/improve_vlm_cot_reasoning.md)
- [\[ICLR 2026\] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](../../ICLR2026/llm_reasoning/scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)

</div>

<!-- RELATED:END -->
