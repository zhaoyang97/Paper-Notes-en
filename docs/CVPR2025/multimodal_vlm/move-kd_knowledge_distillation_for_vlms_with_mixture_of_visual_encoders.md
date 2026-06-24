---
title: >-
  [Paper Note] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders
description: >-
  [CVPR 2025][Multimodal VLM][Knowledge Distillation] This paper proposes MoVE-KD—the first framework to fuse the strengths of multiple visual encoders (CLIP/EVA/ConvNeXt/SAM) into a single encoder from the perspective of knowledge distillation. It alleviates multi-teacher knowledge conflicts through Mixture-of-LoRA-Experts (MoLE), adaptively weights distilled tokens and teachers using CLIP $[CLS]$ attention, and achieves consistent improvements on LLaVA/LLaVA-NeXT.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Knowledge Distillation"
  - "Visual Encoder Fusion"
  - "Mixture of Experts"
  - "LoRA"
  - "Attention Guidance"
date: 2026-05-08
content_hash: 65437e0ce599ef77
---

# MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders

**Conference**: CVPR 2025  
**arXiv**: [2501.01709](https://arxiv.org/abs/2501.01709)  
**Code**: [https://github.com/hey-cjj/MoVE-KD](https://github.com/hey-cjj/MoVE-KD)  
**Area**: Multimodal VLM  
**Keywords**: Knowledge Distillation, Visual Encoder Fusion, Mixture of Experts, LoRA, Attention Guidance

## TL;DR

This paper proposes MoVE-KD—the first framework to fuse the strengths of multiple visual encoders (CLIP/EVA/ConvNeXt/SAM) into a single encoder from the perspective of knowledge distillation. It alleviates multi-teacher knowledge conflicts through Mixture-of-LoRA-Experts (MoLE), adaptively weights distilled tokens and teachers using CLIP $[CLS]$ attention, and achieves consistent improvements on LLaVA/LLaVA-NeXT.

## Background & Motivation

Different pre-trained visual encoders (such as CLIP, EVA, ConvNeXt, DINOv2) have their own strengths and perform differently across various VL tasks. $\rightarrow$ Current multi-encoder methods (e.g., Eagle, Mini-Gemini, $S^2$) employ multiple encoders in parallel through feature concatenation or attention mechanisms, but the computational cost scales linearly with the number of encoders. $\rightarrow$ AM-RADIO attempts to replicate predictions from multiple foundation models using a single model with multiple heads, but sharing a backbone to learn diverse characteristics leads to conflicts. $\rightarrow$ **Key Challenge**: How to absorb the respective advantages of multiple teacher encoders without conflict while maintaining the efficiency of a single encoder? $\rightarrow$ **Key Insight**: A combination of knowledge distillation + MoE + LoRA—using MoLE to allow different tokens to select different LoRA experts based on inputs, and utilizing CLIP $[CLS]$ attention to guide distillation to focus on valuable features.

## Method

### Overall Architecture

Training pipeline: Multiple teacher encoders (CLIP/EVA/ConvNeXt, optionally SAM) individually process images to obtain visual tokens $\rightarrow$ each teacher's tokens are projected into a unified space via an independent 2-layer MLP encoder adapter $\rightarrow$ token weights $W^{(tok)}$ and teacher weights $W^{(tea)}$ are computed based on pre-trained CLIP $[CLS]$ attention $\rightarrow$ weighted MSE loss distills knowledge into the student encoder $\rightarrow$ a MoLE structure is integrated within the student encoder to prevent knowledge conflicts $\rightarrow$ Total Loss = Text Loss + KD Loss. Only the student encoder is used during inference.

### Key Designs

1. **Mixture-of-LoRA-Experts (MoLE)**:
    - **Function**: Alleviates multi-teacher knowledge conflicts within the student encoder.
    - **Mechanism**: Embeds the MoE architecture in each FFN layer of the student encoder, where each expert is a parameter-efficient LoRA module (two low-rank matrices); a Router linear layer dynamically selects the top-1 expert based on inputs: $F^\star(x) = F(x) + E_i(x)$, $i = \text{argmax}(\text{Softmax}(f(x)))$.
    - **Design Motivation**: Directly fine-tuning the student encoder $\rightarrow$ overfitting + catastrophic forgetting + training crash (abnormal loss); using complete FFNs as experts $\rightarrow$ parameter explosion; LoRA is both parameter-efficient (only 0.3% of total parameters) and has better generalization; ablation studies confirm that without MoLE, KD can even degrade performance below the baseline (VQAv2: 76.7 $\rightarrow$ 77.4 with MoLE vs 76.7 without MoLE/KD).

2. **Attention-Guided KD Regularization (Token Weights + Teacher Weights)**:
    - **Function**: Adaptively identifies which visual tokens and which teachers are more worthy of distillation.
    - **Mechanism**: Utilizes the cross-attention between the pre-trained CLIP $[CLS]$ token and other visual tokens as a measure of importance.
    - **Token Weights**: $W^{(tok)} = \text{Softmax}(\frac{V^{(cls)}W^{(Q)} \cdot (V^{(res)}W^{(V)})^T}{\sqrt{d}})$, focusing the student on semantically rich regions (e.g., foreground objects) and ignoring background noise.
    - **Teacher Weights**: $W^{(tea)} = \text{Softmax}(\text{mean}(\frac{V^{(cls)} \cdot V_i^{(t)T}}{\sqrt{d}}))$, reflecting the response intensity of each teacher to a specific image.
    - **Design Motivation**: $[CLS]$ attention naturally focuses on key areas (visualization shows CLIP's $[CLS]$ concentrates on meaningful objects), which is more efficient and generalizes better than learnable tokens; different teachers should contribute differently to different images, and uniform weighting would limit each teacher from exerting their unique strengths.

3. **Encoder Adapter**:
    - **Function**: Aligns outputs from different teacher encoders into a unified representation space.
    - **Mechanism**: Each teacher is equipped with an independent 2-layer MLP adapter to map outputs to the same dimension as the student.
    - **Design Motivation**: Encoder output spaces from different pre-training sources are inconsistent, and simple linear interpolation cannot bridge the gap; ablation shows using adapters outperforms interpolation by 0.4% (avg: 66.3 $\rightarrow$ 66.7).

### Loss & Training

Total Loss: 

$$\mathcal{L}_{total} = \mathcal{L}_{text} + \lambda_{kd} \cdot \mathcal{L}_{kd}$$

- $\mathcal{L}_{text}$: Standard cross-entropy loss for text generation in VLMs.
- $\mathcal{L}_{kd}$: Weighted MSE distillation loss with token weights and teacher weights.
- $\lambda_{kd} = 0.5$, CLIP teacher has a fixed weight of 0.8 (since the student is initialized from CLIP), number of MoLE experts = 3, LoRA rank = 32.
- Two-stage training: Pre-training stage freezes the LLM and only trains MoLE/adapters/projection layer; fine-tuning stage unfrezes all parameters except for the teachers.
- Trained on 16$\times$A800 GPUs.

## Key Experimental Results

### Main Results (LLaVA-1.5 7B + MoVE-KD)

| Method | VQAv2 | GQA | TextVQA | POPE | SQA | MME | MMB | Avg|
|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5 | 78.5 | 62.0 | 58.2 | 85.9 | 66.8 | 1510.7 | 64.3 | 66.5 |
| +RADIO | 76.3 | 63.0 | 56.3 | 86.2 | — | — | — | — |
| +MoVE-KD v1.0 | 79.5 | 63.2 | 58.3 | 86.9 | 69.3 | 1524.5 | 66.3 | 68.0 |
| **+MoVE-KD v1.1** | **79.9** | **63.9** | **59.6** | 86.3 | **69.8** | 1509.1 | **67.4** | — |

### Ablation Study

| Configuration (Added step-by-step) | VQAv2 | GQA | VizWiz | POPE | MMB | Avg |
|---|---|---|---|---|---|---|
| LLaVA-1.5 (baseline) | 78.5 | 62.0 | 50.0 | 85.9 | 64.3 | 66.5 |
| + KD (interpolation) | 79.0 | 62.4 | 50.9 | 84.7 | 62.9 | 66.3 |
| + Encoder adapter | 79.3 | 62.4 | 51.2 | 85.2 | 63.8 | 66.7 |
| + MoLE | 79.1 | 62.8 | 51.9 | **86.4** | 65.4 | 67.4 |
| + Token weight | 79.3 | 63.1 | 52.5 | 86.7 | 66.0 | 67.7 |
| **+ Teacher weight (full)** | **79.5** | **63.2** | **52.3** | **86.9** | **66.3** | **68.0** |

### Key Findings

- RADIO suffers from noticeable forgetting on VQAv2/TextVQA (-2.2/-1.9), whereas MoVE-KD achieves comprehensive improvements—indicating that MoLE effectively mitigates knowledge conflicts.
- Introducing MoLE alone without KD does not improve performance (Table 3), ruling out the hypothesis that "performance gains come merely from parameter increases".
- Directly unfreezing the encoder for fine-tuning leads to degraded performance, proving that the improvement of MoVE-KD is not due to encoder unfreezing.
- v1.0 $\rightarrow$ v1.1 (adding SAM teacher) brings further improvements, especially on TextVQA (+1.3), demonstrating the great scalability of the method regarding teachers.
- A CLIP teacher weight of 0.8 is optimal; a weight too low (0.6) leads to excessive forgetting of the original CLIP knowledge.

## Highlights & Insights

- **Pioneering Perspective**: First to address VLM multi-encoder fusion from the KD perspective, which is more elegant than feature concatenation and incurs no extra inference overhead.
- **Elegant MoLE Design**: LoRA as MoE experts = parameter efficiency + knowledge division of labor, with only 0.3% parameter overhead.
- **Reuse of [CLS] Attention**: Utilizing CLIP's $[CLS]$ attention as a distillation guidance signal is an ingenious "free lunch".
- **Plug-and-Play**: Can be directly applied to mainstream VLMs like LLaVA/LLaVA-NeXT without modifying the inference architecture.

## Limitations & Future Work

- Multi-teacher encoder forward passes are still required during training, leading to high training costs (16$\times$A800).
- The $[CLS]$ attention fully relies on CLIP, which may result in insufficient distillation in regions that CLIP fails to attend to.
- Validation is limited to the LLaVA series; generalizability has not been tested on architectures like Qwen-VL or InternVL.
- MoLE currently uses only top-1 routing; top-2 or soft routing strategies have not been explored.
- Occasional slight performance drops on TextVQA, possibly because enhancing visual tokens has an adverse effect on text understanding.

## Related Work & Insights

- **AM-RADIO**: Multi-head, single-backbone distillation trained on DataComp-1B; MoVE-KD requires no extra data and achieves better performance.
- **Eagle/Mini-Gemini**: Uses extra encoders for high-resolution refinement, which incurs higher inference costs.
- **OneS**: A pioneer in multi-teacher LLM distillation; MoVE-KD introduces multi-teacher KD to the visual side of VLMs.
- **Insight**: $[CLS]$ attention is an underestimated signal—it encodes information on "where is important in the image" and can be widely applied in scenarios like distillation, pruning, and token compression.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to apply multi-teacher KD + MoLE on VLM visual encoders, presenting a novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers LLaVA/LLaVA-NeXT, multiple scales (1.7B to 13B), 8 benchmarks, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and well-designed ablation studies.
- **Value**: ⭐⭐⭐⭐ Highly practical, plug-and-play improvement for VLMs; excellent teacher scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Instruction-based Image Manipulation by Watching How Things Move](instruction-based_image_manipulation_by_watching_how_things_move.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[CVPR 2025\] Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment](harnessing_frozen_unimodal_encoders_for_flexible_multimodal_alignment.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](multimodal_autoregressive_pre-training_of_large_vision_encoders.md)

</div>

<!-- RELATED:END -->
