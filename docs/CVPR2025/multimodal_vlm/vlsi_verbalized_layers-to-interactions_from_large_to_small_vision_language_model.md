---
title: >-
  [Paper Note] VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Knowledge Distillation] VLsI proposes a natural language-based layer-wise distillation method. By introducing "verbalizers" in the intermediate layers of large and small VLMs to map features into the language space, combined with an adaptive layer matching strategy to align inference processes, VLsI enables 2B/7B small models to outperform GPT-4V by an average of 11.0%/17.4% on 10 VL benchmarks without any architectural modifications or parameter i…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Knowledge Distillation"
  - "Layer-wise Alignment"
  - "Natural Language Distillation"
  - "Small VLM"
  - "Efficient Inference"
date: 2026-05-08
content_hash: 3cd2f150b18f3b5e
---

# VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.01822](https://arxiv.org/abs/2412.01822)  
**Code**: [https://github.com/byungkwanlee/VLsI](https://github.com/byungkwanlee/VLsI)  
**Area**: Multimodal VLM  
**Keywords**: Knowledge Distillation, Layer-wise Alignment, Natural Language Distillation, Small VLM, Efficient Inference

## TL;DR
VLsI proposes a natural language-based layer-wise distillation method. By introducing "verbalizers" in the intermediate layers of large and small VLMs to map features into the language space, combined with an adaptive layer matching strategy to align inference processes, VLsI enables 2B/7B small models to outperform GPT-4V by an average of 11.0%/17.4% on 10 VL benchmarks without any architectural modifications or parameter increases.

## Background & Motivation
Open-source VLMs (such as Qwen2-VL and LLaVA-OneVision) continuously improve performance by scaling up model size. However, the high computational overhead of large models makes them difficult to deploy on resource-constrained platforms, such as mobile devices and robotics. Existing solutions include adding extra modules (e.g., multi-vision encoder fusion) or modifying architectures (e.g., the dual forward propagation in TroL), but these approaches introduce engineering complexity and compatibility issues. Traditional knowledge distillation methods typically target the mimicry of final outputs, which often leads to training instability and neglects the discrepancies in the intermediate inference processes between large and small models. **Core Idea**: Since natural language is an efficient medium for knowledge transfer among humans, can it also serve as a bridge for knowledge transfer between large and small VLMs? By "verbalizing" intermediate layer features, the physical alignment of intermediate inference trajectories can be achieved.

## Method

### Overall Architecture
The training of VLsI consists of three stages: (1) Verbalization Step: Training a verbalizer for each designated intermediate target layer in both the large and small VLMs to map their hidden states to the natural language space; (2) Interaction Step: Utilizing adaptive layer matching to identify the optimal layer alignment between the large and small VLMs, and distilling the small model using KL divergence to mimic the step-by-step inference progress of the larger model; (3) SFT Step: Fine-tuning all parameters of the distilled small model with an autoregressive loss to further enhance its instruction-following capacity.

### Key Designs
1. **Verbalizer**:
    - Function: Projecting hidden states from intermediate Transformer layers into the natural language space, allowing them to be interpreted as textual responses.
    - Mechanism: Each target intermediate layer is equipped with a verbalizer, which consists of a lightweight FFN (verb-FFN, without dimension expansion/reduction) and the language head of the backbone VLM. Under frozen backbone parameters, the verb-FFN is trained using an autoregressive loss to ensure that the intermediate layer outputs, when mapped through the verbalizer, generate text aligned with the target response. Because the backbone weights are frozen, the gradient updates for individual verbalizers remain mutually independent.
    - Design Motivation: Drawing inspiration from speculative decoding—where a smaller LLM can mimic a larger one by reusing its word embeddings and language head. Verbalization enables the tracking of "key reasoning progress" within the highly interpretable natural language space, rather than relying on direct alignment in opaque latent feature spaces.

2. **Adaptive Layer Matching**:
    - Function: Resolving the cross-layer mapping challenge when the large and small VLMs contain different numbers of layers.
    - Mechanism: Employs a multinomial sampling strategy to match layers dynamically. For the $i_s$-th layer of the small model, all candidate layers within the search range of the large model are traversed to calculate the KL divergence against the current layer of the small model. A sampling distribution is then generated via $\text{softmax}(-\text{KLD}/T)$ for selection. There are two key constraints: **order-preservation**—if the $i$-th layer of the small model is matched with the $j$-th layer of the large model, the $(i+1)$-th layer must match a layer beyond the $(j+1)$-th; **search range limits**—preventing early layers of the large model from pairing with late layers of the small model.
    - Design Motivation: Uniform static mapping ignores the fact that different layers exhibit varied "reasoning progression." Sampling based on negative KL divergence biases matching towards better-aligned layer pairs, while the adaptive temperature dynamically regulates this process during training.

3. **Three-step Training Strategy**:
    - Function: Maximizing the effect of knowledge transfer step-by-step.
    - Mechanism: The Verbalization phase establishes "layer-to-language" mappings for both large and small models independently. The Interaction phase aligns the intermediate layer distributions using KL divergence loss, updating only the verbalizers and LoRA parameters of the small model. The SFT phase unfrees all parameters for full autoregressive fine-tuning.
    - Design Motivation: Simultaneously optimizing KL divergence and autoregressive losses during the Interaction phase causes performance degradation (as verified by ablation studies). The SFT stage acts similarly to recovery training after pruning, helping the small model fully digest the distilled knowledge.

### Loss & Training
- Verbalization: Autoregressive loss $\mathcal{L}_{AR}$ trains each layer's verbalizer independently.
- Interaction: Sum of KL divergence losses across all matched layers (including intermediate layers and the final layer).
- SFT: Standard autoregressive loss fine-tunes all parameters of the small model.
- Training Configuration: 8×A100 80GB, LoRA rank=64, AdamW + cosine schedule, 2.9M diverse visual instruction data.

## Key Experimental Results

### Main Results (7B Model Comparison)

| Benchmark | Qwen2-VL-7B | VLsI-7B | GPT-4V | Gain vs GPT-4V |
|-----------|------------|---------|--------|---------------|
| MM-Vet | 62.0 | **75.2** | 67.5 | +7.7 |
| MMMU | 54.1 | **69.3** | 61.7 | +7.6 |
| MathVista | 58.2 | **74.7** | 54.7 | +20.0 |
| MMB | 83.0 | **86.3** | - | - |
| AI2D | 77.5 | **87.3** | 78.6 | +8.7 |

### Small Model Comparison (2B)

| Benchmark | Qwen2-VL-2B | VLsI-2B | Gain |
|-----------|------------|---------|------|
| MM-Vet | 49.5 | **64.8** | +15.3 |
| MMMU | 41.1 | **51.4** | +10.3 |
| MathVista | 43.0 | **68.4** | +25.4 |
| AI2D | 60.2 | **89.0** | +28.8 |

### Ablation Study

| Configuration | MMB | MM-Vet | MMMU | Description |
|------|-----|--------|------|------|
| CE (Interaction) + w/o Last Layer | 79.2 | 64.5 | 56.5 | Intermediate layer distillation via cross-entropy |
| KLD + w/o Last Layer | 83.0 | 69.5 | 61.0 | KLD + w/o final layer distillation |
| KLD + KLD (Full) | **86.3** | **75.8** | **69.3** | KLD for both intermediate and final layers |
| w/o SFT | 82.1 | 60.5 | 52.9 | Skipping the SFT phase |
| w/ SFT | **86.3** | **75.8** | **69.3** | SFT yields significant improvement |

### Key Findings
- VLsI-7B **outperforms GPT-4V by up to 17.4%** on challenging benchmarks such as MM-Vet, MMMU, and MathVista.
- VLsI-2B even outperforms many 7B-13B models (such as LLaVA-NeXT-13B and Eagle-13B).
- KL divergence is better suited for layer-wise distillation than cross-entropy; simultaneous distillation of intermediate and final layers achieves optimal outcomes.
- Each component within adaptive layer matching (order-preservation, search range limits, adaptive temperature) contributes to the performance.
- Verbalizer architecture: verb-FFN (269M parameters) achieves the optimal balance between performance and efficiency.

## Highlights & Insights
- **Natural language as a distillation medium** is a highly elegant idea—transforming the characteristic-space alignment problem into distribution matching in the language space, which is more intuitive and effective.
- "Verbalization" allows visualizing the step-by-step reasoning progress at each layer (as shown in Fig. 3), offering strong interpretability.
- Modifies no model architecture and incurs no additional inference costs, allowing direct use of the small model once distilled.
- VLsI-0.5B under the guidance of LLaVA-OV-72B still yields competitive results (MMB=72.5), demonstrating the viability of distillation even under extreme scale disparities.

## Limitations & Future Work
- Large and small models must share the same tokenizer and vocabulary indices, which limits transfer across different model families (e.g., feasible within the Qwen family, but not from Qwen to LLaMA).
- Training requires loading both the large and small models simultaneously, leading to heavy GPU memory overhead (e.g., 72B teacher + 7B student).
- The verbalization phase trains each target layer independently, scaling up the training cost as the number of target layers increases.
- The volume of visual instruction data used in experiments is relatively large (2.9M), leaving data efficiency to be further explored.
- Currently evaluated only on Qwen2-VL and LLaVA-OV model families; wider architecture support warrants subsequent research.

## Related Work & Insights
- **vs LLaVA-KD / LLaVA-MoD**: These methods target only final-layer distillation, ignoring the alignment of intermediate reasoning processes.
- **vs TroL / Phantom**: These approaches enhance small models by modifying architectures (e.g., dual forward-propagation or expanding hidden dimensions) but introduce extra KV-cache and compatibility issues.
- **vs Align-KD / MoVE-KD**: These methods construct alignments within the feature space, whereas VLsI establishes them naturally and effectively in the language space.
- **vs DistiLLM / MiniLLM**: While these prioritize the direction of KL divergence in distillation, VLsI focuses on layer-wise alignment as an orthogonal dimension.

## Supplementary Notes
- VLsI-7B achieves a score of 92.0 on LLaVA-Wilder, outperforming GPT-4o (85.9) and Qwen2-VL-72B (84.1).
- Designated intermediate layers: The small model selects one layer every 4 layers (layers 2, 6, 10, ..., 26), with the large model selected at corresponding intervals.

## Rating
- Novelty (新颖性): ⭐⭐⭐⭐⭐ Using natural language for layer-wise distillation presents a highly novel paradigm, with elegant designs for verbalizer and adaptive layer matching.
- Experimental Thoroughness (实验充分度): ⭐⭐⭐⭐⭐ Evaluated across 10 benchmarks, with rich ablation studies, cross-backbone verification, and comparisons against closed-source models.
- Writing Quality (写作质量): ⭐⭐⭐⭐ Clear structure and well-executed verbalization visualizations.
- Value (价值): ⭐⭐⭐⭐⭐ Offers a new distillation paradigm for efficient VLMs, enabling a 2B model to outperform many 7B-13B alternatives.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)
- [\[CVPR 2025\] Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models](calico_part-focused_semantic_co-segmentation_with_large_vision-language_models.md)
- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)

</div>

<!-- RELATED:END -->
