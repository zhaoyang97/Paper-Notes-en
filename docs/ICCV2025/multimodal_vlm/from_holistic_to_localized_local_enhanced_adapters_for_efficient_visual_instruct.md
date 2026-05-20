---
title: >-
  [Paper Note] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning
description: >-
  [ICCV 2025][Multimodal VLM][Dual-LoRA] This paper proposes Dual-LoRA and Visual Cue Enhancement (VCE) modules that adopt a "from holistic to localized" paradigm to address data conflicts in efficient visual instruction f…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Dual-LoRA"
  - "LoRA-MoE"
  - "Visual Cue Enhancement"
  - "Data Conflict"
  - "Efficient Instruction Fine-Tuning"
date: 2026-05-08
content_hash: 92f93e9f5ca2bb2b
---

# From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning

**Conference**: ICCV 2025
**arXiv**: [2411.12787](https://arxiv.org/abs/2411.12787)  
**Code**: [https://github.com/pengkun-jiao/Dual-LoRA](https://github.com/pengkun-jiao/Dual-LoRA)  
**Area**: Multimodal VLM
**Keywords**: Dual-LoRA, LoRA-MoE, Visual Cue Enhancement, Data Conflict, Efficient Instruction Fine-Tuning

## TL;DR

This paper proposes Dual-LoRA and Visual Cue Enhancement (VCE) modules that adopt a "from holistic to localized" paradigm to address data conflicts in efficient visual instruction fine-tuning, surpassing LoRA-MoE methods with only a 1.16× inference time overhead.

## Background & Motivation

Efficient visual instruction tuning (EVIT) adapts multimodal large language models (MLLMs) to downstream tasks using adapters such as LoRA with modest computational overhead. However, as task diversity and complexity increase, LoRA fine-tuning suffers from severe **data conflicts**: for instance, in multi-task food-related training, knowledge inconsistencies may arise between ingredient recognition and recipe generation.

Existing solutions rely on LoRA-MoE (embedding LoRA within a mixture-of-experts framework) to mitigate data conflicts through localized expert activation. Nevertheless, LoRA-MoE exhibits two drawbacks: (1) it requires complex design choices to balance activation strategies, trainable parameters, and task complexity; and (2) multi-expert activation substantially increases inference latency (a 4-expert LoRA-MoE incurs 1.59× baseline latency).

The authors' core insight is drawn from human cognition: humans first acquire holistic knowledge (e.g., cooking in general) and then apply specific portions (e.g., ingredient recognition) to particular tasks — a "from holistic to localized" process.

## Method

### Overall Architecture

Two core components are proposed: (1) **Visual Cue Enhancement (VCE)**, which enriches local details in visual feature projection; and (2) **Dual Low-Rank Adaptation (Dual-LoRA)**, which transitions from holistic knowledge to localized task adaptation via a dual-subspace design. Training proceeds in two stages: Stage 1 pre-trains VCE and the visual projector; Stage 2 fine-tunes VCE, the projector, and Dual-LoRA jointly.

### Key Designs

1. **Visual Cue Enhancement (VCE)**: Typical MLLMs (e.g., LLaVA) rely solely on high-level feature maps from the penultimate ViT layer, which tend to overlook local visual details. VCE extracts local information from multiple intermediate feature maps via Deformable Attention, specifically selecting layers 2, 8, 14, and 20 of CLIP ViT-L, with layer 2 serving as the anchor feature. For each anchor patch $p_q$, deformable attention aggregates $K$ neighboring reference patches at each layer; multi-scale features are then concatenated and fused via linear projection, followed by residual fusion with the high-level features before being passed to the visual projector. VCE adds only 5.53 MB of parameters.

2. **Dual-LoRA (Skill Space + Task Space)**: Based on theoretical analysis (Proposition 1 & Corollaries 1–2), the paper proves that a single rank-$r$ LoRA is theoretically at least as expressive as $K$ LoRA experts. The empirical advantage of LoRA-MoE stems from its "local response" capability. To replicate this, a dual-space design is introduced:

    - **Skill Space** $S$: a low-rank matrix for stably learning holistic cross-task knowledge.
    - **Task Space** $T$: a rank-correction matrix that dynamically modulates the skill space via nonlinear activation $\sigma$ (ReLU).

   The output is formulated as: $z = Wx + \frac{r}{\alpha} B(\text{Norm}(Sx) \odot \sigma(Tx))$

   LayerNorm smooths the skill space distribution, while ReLU induces sparsity to enable local response.

3. **Theoretical Foundations**: The authors prove that (a) a single rank-$K$ LoRA subsumes the representation space of $K$ rank-1 LoRAs (Proposition 1); and (b) via element-wise multiplication with nonlinear activation, a single LoRA can be decomposed into an arbitrary combination of LoRAs (Corollary 2), providing theoretical justification for the Dual-LoRA design.

### Loss & Training

- Standard cross-entropy loss is used for instruction fine-tuning.
- Dual-LoRA is injected only into the query and value projection layers of the LLM for efficiency.
- $\alpha$ is set to 2× the rank; LoRA dropout is 0.05.
- Base model: LLaVA-1.5-7B with CLIP ViT-L/14 visual encoder.

## Key Experimental Results

### Main Results

| Method | UniFood IoU↑ | UniFood F1↑ | Recipe BLEU↑ | Recipe Rouge-L↑ | ScienceQA Acc↑ | Flickr30k BLEU↑ |
|------|------------|------------|-------------|----------------|---------------|----------------|
| Vanilla LoRA | 23.2 | 34.1 | 12.4 | 40.1 | 70.01 | 27.89 |
| LoRA-MoE (top-2) | 22.9 | 33.8 | 12.7 | 40.2 | 76.3 | 28.15 |
| LoRA-MoE (softmax) | 22.7 | 33.5 | 12.5 | 40.0 | 77.73 | 28.06 |
| RoDE | 23.6 | 34.6 | 13.8 | 41.4 | 78.39 | 28.17 |
| **Dual-LoRA** | **24.2** | **35.2** | **14.8** | **42.1** | **79.17** | **28.25** |
| **Dual-LoRA+VCE** | **24.5** | **35.5** | 14.7 | **42.2** | **80.01** | **28.71** |

General benchmarks (LLaVA-1.5-7B base): MMBench 65.6, POPE 87.2, MMVet 32.1 (all best or second-best).

### Ablation Study

| VCE | Dual-LoRA | IoU↑ | F1↑ | BLEU↑ | Rouge-L↑ |
|-----|-----------|------|-----|-------|----------|
| ✗ | ✗ (LoRA) | 23.2 | 34.1 | 12.4 | 40.1 |
| ✗ | ✓ | **24.2** | **35.2** | **14.8** | **42.1** |
| ✓ | ✗ | 23.3 | 34.2 | 12.6 | 40.5 |
| ✓ | ✓ | **24.5** | **35.5** | 14.7 | **42.2** |

Ablation on internal Dual-LoRA design:

| Skill @Norm | Task @Non-linear | IoU↑ | BLEU↑ |
|-------------|-----------------|------|-------|
| ✗ | ✗ | 21.4 | 13.1 |
| ✓ | ✗ | 22.6 | 14.1 |
| ✗ | ✓ | 21.9 | 13.4 |
| ✓ | ✓ | **24.2** | **14.8** |

### Key Findings

- Dual-LoRA **consistently outperforms** vanilla LoRA and LoRA-MoE methods across all downstream tasks.
- Inference time is only 1.16× that of vanilla LoRA (Dual-LoRA+VCE), compared to 1.59× for 4-expert LoRA-MoE.
- Attention heatmap visualizations of the VCE module confirm that the enhanced visual cues genuinely focus on salient image regions.
- The modulated skill space exhibits lower information entropy than the original, indicating greater task-specificity.
- Dual-LoRA's advantage over vanilla LoRA is more pronounced at lower parameter budgets, demonstrating superior mitigation of data conflicts.
- Both LayerNorm in the Skill Space and nonlinear activation in the Task Space are indispensable.

## Highlights & Insights

1. **Theory-driven design**: Rigorous mathematical proofs (Proposition 1, Corollaries 1–2) reveal the representational equivalence between a single LoRA and MoE, motivating a more efficient alternative.
2. **Structural simplicity**: Dual-LoRA is architecturally far simpler than LoRA-MoE, requiring neither a router nor expert selection strategies.
3. **Efficiency advantage**: Dual-LoRA outperforms all LoRA-MoE methods under equivalent parameter budgets, with negligible inference overhead.
4. **Lightweight VCE**: At only 5.53 MB, VCE effectively aggregates local information from multi-level feature maps.

## Limitations & Future Work

- Experiments are primarily conducted on LLaVA-1.5-7B; validation on larger models (13B/70B) or newer architectures (Qwen-VL-2, InternVL, etc.) is absent.
- The selection of intermediate layers in VCE (2, 8, 14, 20) is manually determined; adaptive layer selection warrants exploration.
- The effect of ReLU-induced sparsity within Dual-LoRA across different task types is not thoroughly analyzed.
- Effectiveness on sequential tasks such as video understanding remains unvalidated.

## Related Work & Insights

- Compared to concurrent works such as AdaMoLE and LLaVA-MoLE, Dual-LoRA offers a structurally simpler alternative.
- The VCE module shares conceptual similarities with the Spatial Vision Aggregator in Cambrian-1, but is considerably more lightweight.
- The dual-space design of Skill Space and Task Space may generalize to other parameter-efficient fine-tuning scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The theory-driven dual-space design is novel, and the cognitive inspiration of "holistic to localized" adds conceptual depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-baseline evaluation with thorough ablations, though limited to a single base model.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation figures are clear and theoretical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐ Provides practical guidance for the MLLM efficient fine-tuning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](../../NeurIPS2025/multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](../../NeurIPS2025/multimodal_vlm/coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)

</div>

<!-- RELATED:END -->
