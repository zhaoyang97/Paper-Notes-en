---
title: >-
  [Paper Note] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning
description: >-
  [ICCV 2025][Multimodal VLM][Continual Learning] This paper identifies a phenomenon termed "dual catastrophic forgetting" in continual visual instruction tuning (CVIT) of multimodal large models, wherein both visual understanding capability and instruction-following capability degrade simultaneously. To address this, SMoLoRA is proposed, employing a separable-routing mixture of LoRA experts to effectively mitigate both forms of forgetting.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Continual Learning
  - Visual Instruction Tuning
  - Catastrophic Forgetting
  - Mixture of LoRA Experts
  - Separable Routing
date: 2026-05-08
content_hash: f49b73b43e8f07df
---

# SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning

**Conference**: ICCV 2025  
**arXiv**: [2411.13949](https://arxiv.org/abs/2411.13949)  
**Code**: [https://github.com/Minato-Zackie/SMoLoRA](https://github.com/Minato-Zackie/SMoLoRA)  
**Area**: Multimodal VLM  
**Keywords**: Continual Learning, Visual Instruction Tuning, Catastrophic Forgetting, Mixture of LoRA Experts, Separable Routing

## TL;DR

This paper identifies a phenomenon termed "dual catastrophic forgetting" in continual visual instruction tuning (CVIT) of multimodal large models, wherein both visual understanding capability and instruction-following capability degrade simultaneously. To address this, SMoLoRA is proposed, employing a separable-routing mixture of LoRA experts to effectively mitigate both forms of forgetting.

## Background & Motivation

Multimodal large language models (MLLMs) acquire the ability to handle diverse visual tasks through visual instruction tuning (VIT). In practice, however, models must continuously learn new tasks to accommodate evolving requirements—this constitutes the continual visual instruction tuning (CVIT) setting.

Existing CVIT methods (e.g., EMT, Eproj, CoIN) largely adopt conventional continual learning paradigms to mitigate catastrophic forgetting, yet overlook challenges unique to CVIT. This paper is the first to identify **dual catastrophic forgetting** in CVIT:

**Visual Understanding Forgetting**: When learning new tasks, the model forgets previously acquired visual understanding capabilities. For example, after learning VQAv2 followed by ImageNet classification, performance on the earlier VQA tasks degrades significantly.

**Instruction-Following Forgetting**: The model's instruction-following capability progressively deteriorates as more tasks are introduced. For instance, a model originally capable of generating complete sentences as instructed may, after several rounds of training, produce only fragmented tokens or malformed outputs.

The two forms of forgetting have distinct origins—visual understanding forgetting stems from inter-task interference in visual features, while instruction-following forgetting arises from conflicts among different task instruction formats. **Conventional methods (e.g., EWC, Replay, vanilla LoRA) cannot address both simultaneously**, as they do not differentiate between these two dimensions.

The core idea of SMoLoRA is to design two independent routing modules that separately handle visual understanding and instruction following, enabling task-specific adaptation along each dimension independently, thereby preventing interference and alleviating dual forgetting.

## Method

### Overall Architecture

SMoLoRA builds upon the Mixture of LoRA Experts (MoLoRA) framework, introducing a separable routing mechanism. The model contains $N$ LoRA blocks divided into two groups: the first $M$ blocks serve the visual understanding module, and the remaining $N-M$ blocks serve the instruction-following module. Each group is activated by its own dedicated router, and the outputs are merged via an adaptive fusion module. SMoLoRA replaces the original LoRA in the FFN layers and adapter of the LLM backbone.

### Key Designs

#### 1. Visual Understanding Module — Instance-Based Routing

- **Function**: Selects the most appropriate LoRA blocks for visual understanding based on the holistic visual and textual features of the current input instance.
- **Mechanism**: The current layer input $x_{l-1} \in \mathbb{R}^{d \times s}$ is averaged along the sequence dimension to yield $\text{Avg}(x_{l-1})$, which is then passed through a routing matrix $R^{vu} \in \mathbb{R}^{M \times d}$ to compute activation weights for each LoRA block:
  $G^{vu}(z^{vu}) = \text{softmax}(\text{top}_k(R^{vu} \cdot \text{Avg}(x_{l-1})))$
- **Design Motivation**: Visual understanding requires integrating image content with text-guided contextual cues; therefore, the averaged representation of the entire instance is used as the routing signal. This enables similar visual understanding tasks (e.g., different VQA datasets) to share LoRA blocks, while substantially different tasks (e.g., classification vs. captioning) utilize distinct blocks.

#### 2. Instruction-Following Module — Instruction Embedding-Based Routing

- **Function**: Selects corresponding LoRA blocks for instruction comprehension and format control based on the semantic features of the current task instruction.
- **Mechanism**: Sentence-BERT encodes the instruction text $X^{ins}$ into an embedding $f_\sigma(X^{ins}) \in \mathbb{R}^{e \times 1}$, which is passed through a routing matrix $R^{if} \in \mathbb{R}^{(N-M) \times e}$ to compute activation weights:
  $G^{if}(z^{if}) = \text{softmax}(\text{top}_k(R^{if} \cdot f_\sigma(X^{ins})))$
- **Design Motivation**: Differences in instruction format across tasks (e.g., "Describe the image" vs. "Answer with a single word") are the proximate cause of instruction-following forgetting. Routing via instruction embeddings rather than instance features enables precise discrimination of different instructional requirements, allowing tasks with identical instruction formats to share LoRA blocks while tasks with divergent formats utilize separate ones.

#### 3. Adaptive Fusion Module

- **Function**: Dynamically weights and fuses the outputs of the visual understanding module and the instruction-following module.
- **Mechanism**: The outputs $x_l^{vu}$ and $x_l^{if}$ from the two modules are computed separately; trainable importance matrices $I^{vu}, I^{if} \in \mathbb{R}^{1 \times k}$ are used to compute fusion weights $[\alpha, \beta]^T = \text{softmax}(\text{concat}(I^{vu} x_l^{vu}, I^{if} x_l^{if}))$, yielding the final output $\mathcal{F} = \alpha \circ x_l^{vu} + \beta \circ x_l^{if}$
- **Design Motivation**: The contributions of the two modules are not equal—certain tasks rely more heavily on visual understanding, while others depend more on instruction following. Adaptive fusion allows the model to dynamically adjust the weighting of both modules based on the specific input, rather than applying a simple average.

### Loss & Training

- Standard language modeling loss (next-token prediction) is adopted.
- The model is initialized from LLaVA-v1.5-7B, with a learning rate of $1 \times 10^{-4}$, cosine decay, batch size of 64, and one epoch of training per task.
- Each module contains 4 LoRA blocks with rank=16 and top-1 routing.
- SMoLoRA is applied exclusively to FFN layers and the adapter; attention layers are not modified.

## Key Experimental Results

### Main Results — Upstream Continual Learning (Single Instruction Type)

| Method | ScienceQA | TextVQA | Flickr30k | ImageNet | GQA | VQAv2 | AP↑ | MAP↑ | BWT↑ | MIF↑ |
|--------|-----------|---------|-----------|----------|-----|-------|-----|------|------|------|
| Multitask (upper bound) | 83.49 | 61.93 | 169.21 | 96.53 | 60.07 | 65.80 | 89.51 | — | — | 98.38 |
| SeqLoRA | 55.31 | 50.22 | 33.89 | 22.73 | 50.52 | 64.61 | 46.21 | 57.41 | -48.10 | 78.35 |
| EWC | 57.04 | 50.02 | 32.96 | 22.85 | 50.16 | 64.54 | 46.26 | 56.19 | -49.71 | 78.90 |
| Eproj | 65.29 | 52.87 | 148.19 | 39.45 | 28.06 | 57.86 | 65.29 | 73.53 | -14.02 | 89.81 |
| **SMoLoRA** | **77.36** | **58.29** | **151.99** | **95.35** | **51.96** | **65.71** | **83.44** | **84.85** | **-3.23** | **97.79** |

SMoLoRA surpasses the second-best method Eproj by 18.15% in AP, improves BWT from -14.02 to -3.23, and nearly eliminates catastrophic forgetting.

### Ablation Study — Module Contributions

| VU | IF | AF | AP | MAP | BWT | MIF |
|----|----|----|-----|------|------|------|
| ✗ | ✗ | ✗ | 46.21 | 57.41 | -48.10 | 78.35 |
| ✓ | ✗ | ✗ | 53.49 | 67.84 | -33.06 | 80.12 |
| ✗ | ✓ | ✗ | 71.97 | 79.56 | -17.42 | **98.38** |
| ✓ | ✓ | ✗ | 75.16 | 78.72 | -10.99 | 97.43 |
| ✓ | ✓ | ✓ | **83.44** | **84.85** | **-3.23** | 97.79 |

The instruction-following module (IF) contributes most substantially (AP jumps from 46.21 to 71.97), indicating that instruction-following forgetting is the more severe problem in CVIT.

### Key Findings

- **Downstream Zero-Shot Transfer**: SMoLoRA achieves a zero-shot AP of 34.35% on VizWiz/TextCaps/OCRVQA, substantially outperforming the second-best DoRA (25.21%), with MIF reaching 90.94% (vs. DoRA 61.98%), demonstrating that separable routing significantly enhances generalization capability.
- **Routing Visualization**: Different tasks exhibit distinct LoRA block preferences in the VU module, whereas similar tasks (e.g., Flickr30k and TextCaps, both captioning tasks) converge to similar routing patterns. In the IF module, routing differences are more pronounced across tasks with divergent instruction formats.
- **Case Analysis**: With only the VU module, the model correctly understands images but produces fragmented outputs; with only the IF module, output format is correct but visual understanding is inaccurate. Both modules together are necessary to simultaneously ensure visual correctness and format compliance.

## Highlights & Insights

- **The identification of dual catastrophic forgetting** is the most significant contribution of this paper. Decomposing the forgetting problem in CVIT into two independent dimensions—visual understanding and instruction following—opens a new direction for subsequent research.
- **The separable routing paradigm** is broadly applicable—it extends beyond LoRA experts and can be generalized to other parameter-efficient methods.
- **The design of the MIF metric** fills a gap in existing CVIT evaluation by providing a measure of instruction-following capability.

## Limitations & Future Work

- Experiments are conducted solely on LLaVA-v1.5-7B; the severity of dual forgetting in larger models (e.g., 13B/70B) remains to be verified.
- The routing strategy employs a fixed top-1 selection; dynamic top-$k$ or soft routing may yield further improvements.
- The current benchmark comprises only 6 upstream and 4 downstream tasks, offering limited task diversity.
- The Sentence-BERT embeddings used in the IF module are frozen; whether joint fine-tuning would be beneficial remains unexplored.

## Related Work & Insights

- This work is related to methods that apply MoE to large multimodal models, such as MoE-LLaVA and MoCLE, but SMoLoRA specifically designs separable routing for the continual learning setting.
- The key distinction from CoIN (which also applies MoE to CVIT) is that CoIN employs a single unified router and does not differentiate between forgetting along the visual understanding and instruction-following dimensions.
- Insight: Continual alignment of MLLMs may similarly exhibit dual forgetting—concurrent degradation of safety alignment and task capability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models](instruction-grounded_visual_projectors_for_continual_learning_of_generative_visi.md)
- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](../../NeurIPS2025/multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)

</div>

<!-- RELATED:END -->
