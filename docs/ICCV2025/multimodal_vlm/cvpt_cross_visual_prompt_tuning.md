---
title: >-
  [Paper Note] CVPT: Cross Visual Prompt Tuning
description: >-
  [ICCV2025][Multimodal VLM][Visual Prompt Tuning] To address the computational redundancy and attention disruption caused by prompt tokens participating in self-attention in Visual Prompt Tuning (VPT), this paper proposes CVPT, which decouples the interaction between prompt and image tokens via cross-attention and leverages a weight-sharing mechanism to initialize the cross-attention module. CVPT significantly outperforms VPT across 25 datasets and achieves performance comparable to mainstream adapter-based methods.
tags:
  - ICCV2025
  - Multimodal VLM
  - Visual Prompt Tuning
  - Cross-Attention
  - PEFT
  - parameter-efficient fine-tuning
  - Weight Sharing
date: 2026-05-08
content_hash: e4ba812d00dc558f
---

# CVPT: Cross Visual Prompt Tuning

**Conference**: ICCV2025
**arXiv**: [2408.14961](https://arxiv.org/abs/2408.14961)
**Code**: [https://github.com/Lingyun0419/CVPT](https://github.com/Lingyun0419/CVPT)
**Area**: Multimodal VLM / Parameter-Efficient Fine-Tuning
**Keywords**: Visual Prompt Tuning, Cross-Attention, PEFT, parameter-efficient fine-tuning, Weight Sharing

## TL;DR
To address the computational redundancy and attention disruption caused by prompt tokens participating in self-attention in Visual Prompt Tuning (VPT), this paper proposes CVPT, which decouples the interaction between prompt and image tokens via cross-attention and leverages a weight-sharing mechanism to initialize the cross-attention module. CVPT significantly outperforms VPT across 25 datasets and achieves performance comparable to mainstream adapter-based methods.

## Background & Motivation

**State of the Field**: Full fine-tuning of large-scale pretrained models (e.g., ViT) incurs substantial costs, making parameter-efficient fine-tuning (PEFT) the dominant paradigm. Within PEFT, two major families exist: adapter-based methods and prompt-based methods. In the vision domain, adapter-based methods (e.g., AdaptFormer, LoRA) consistently outperform prompt-based methods (e.g., VPT), leading to the widespread belief that "prompt methods are inferior to adapters."

**Limitations of Prior Work**: VPT concatenates prompt tokens directly with image tokens and feeds the combined sequence into the self-attention of each Transformer block, introducing two critical issues:
   - **Computational complexity**: The self-attention complexity grows from $n^2$ to $(n+m)^2$, escalating sharply as the number of prompts increases.
   - **Attention disruption**: Prompt tokens compete for attention weights during softmax normalization. When the number of prompts reaches 196, they occupy more than 80% of the attention weights, severely degrading the original feature representations.

**Root Cause**: VPT requires a large number of prompts to adapt to downstream tasks, yet adding more prompts paradoxically degrades performance. The root cause lies in how prompts are deployed — they share the same self-attention with image tokens, coupling the two modalities.

**Paper Goals**: To preserve the flexibility of prompts while eliminating their interference with self-attention, enabling prompt-based methods to maintain high performance and efficiency even when a large number of prompts are used.

**Starting Point**: The authors observe that prompt tokens are not inherent components of the original input sequence — they carry no semantic information and serve only as indirect tuning factors. Therefore, decoupling prompts from self-attention is the fundamental solution.

**Core Idea**: Replace prompt concatenation in self-attention with cross-attention, where embedded tokens serve as queries and prompts serve as keys and values, decoupling their interaction.

## Method

### Overall Architecture
CVPT follows a pipeline similar to VPT: the pretrained ViT backbone is frozen, and only the prompt tokens and the final classification head are trained. The key distinction is that prompt tokens are no longer concatenated into the image token sequence; instead, they interact with embedded tokens through a dedicated cross-attention module. The forward pass proceeds as: Input → Patch Embedding → multiple Transformer Blocks (each executing frozen self-attention → cross-attention with prompts → frozen MLP in sequence) → classification head output.

### Key Designs

1. **Cross-Attention Decoupling Module**:

    - **Function**: A cross-attention layer is inserted after the self-attention module and before the MLP in each Transformer block.
    - **Mechanism**: Embedded tokens serve as queries ($Q = X_1 W^Q$), and prompt tokens serve as keys and values ($K = V = X_2 W^K$). The cross-attention is computed as $\text{CrossAttention}(X_1, X_2) = \text{Softmax}(\frac{Q \cdot K}{\sqrt{d_k}}) V$, and the result is added back to the embedded tokens via a residual connection.
    - **Design Motivation**: This design offers three benefits: (1) self-attention operates exclusively on image tokens, preserving full feature representation capacity; (2) the computational complexity of cross-attention is $O(n \cdot m)$, linear rather than quadratic; (3) the output dimensionality matches that of embedded tokens, enabling direct residual addition without imposing additional computation on the subsequent MLP.

2. **Weight Sharing Mechanism**:

    - **Function**: The cross-attention module is initialized with the pretrained weights of the self-attention module and frozen during training.
    - **Mechanism**: Since the cross-attention and self-attention share the same structural form (QKV projection + softmax), the pretrained self-attention weights can be directly reused as initialization.
    - **Design Motivation**: (1) This avoids introducing a large number of learnable parameters (frozen CA requires only 0.09M parameters vs. 28.4M for learnable CA); (2) the pretrained self-attention weights encode rich visual knowledge, providing a strong inductive bias; (3) experiments show that frozen CA with weight sharing achieves performance on par with learnable CA.

3. **Optimal Insertion Position**:

    - **Function**: Explores the optimal position within the Transformer block to insert the cross-attention module.
    - **Mechanism**: Five candidate positions are evaluated; inserting after the self-attention (position 3) yields the best performance at 74.0% accuracy.
    - **Design Motivation**: The self-attention module produces rich contextual features; performing cross-attention immediately thereafter allows more effective prompt adaptation based on these features.

### Loss & Training
The training strategy follows VPT, using the AdamW optimizer and optimizing only the prompt tokens and the classification head. The number of prompts is selected from {1, 5, 10, 20, 50, 100, 200}.

## Key Experimental Results

### Main Results

| Dataset | Metric | CVPT | VPT-Deep | LoRA | DMLoRA | Gain (vs. VPT) |
|--------|------|------|----------|------|--------|-------------|
| VTAB-1K (19ds avg) | Top-1 Acc | **77.2** | 72.0 | 74.5 | 77.0 | +5.2% |
| VTAB Natural (7ds) | Avg Acc | 83.3 | 71.6 | - | - | +11.7% |
| VTAB Structured (8ds) | Avg Acc | 61.7 | 55.0 | - | - | +6.7% |
| FGVC (5ds avg) | Top-1 Acc | **90.5** | 89.1 | 89.5 | 90.7 | +1.4% |
| ADE20K (P=200) | mIoU-SS | **45.66** | 42.11 | - | - | +3.55% |
| ADE20K (P=200) | mIoU-Ms | **47.92** | 44.06 | - | - | +3.86% |

### Ablation Study

| Configuration | VTAB Avg Acc | FGVC Avg Acc | Params (M) | Notes |
|------|-------------|-------------|------------|------|
| Weight Sharing + Frozen CA | **74.0** | **89.3** | 0.09 | Full model, most efficient |
| Weight Sharing + Learnable CA | 74.6 | 89.5 | 28.4 | 300× more parameters, marginal gain |
| Random Init + Learnable CA | 74.0 | 89.5 | 28.4 | Weight sharing offers no clear advantage when CA is learnable |
| Random Init + Frozen CA | 63.7 | 86.0 | 0.09 | Frozen CA collapses without weight sharing |
| Linear Probing | 57.6 | 79.3 | 0 | Baseline |

### Key Findings
- **Prompt count sensitivity**: VPT performance degrades sharply when the prompt count exceeds 50 (dropping from 73.0 to 64.0 at 200 prompts), while CVPT improves steadily with more prompts (reaching 74.8 at 200 prompts), validating the effectiveness of the decoupled design.
- **Pronounced advantage on OOD datasets**: The largest gains are observed on the Structured subset, which deviates significantly from the pretraining distribution, indicating that more prompts facilitate adaptation to out-of-distribution tasks.
- **Efficiency advantage**: At 200 prompts, CVPT incurs substantially lower FLOPs and memory overhead than VPT, with costs that do not grow explosively with prompt count.

## Highlights & Insights
- **Addressing the architectural flaw of VPT at its root**: Rather than patching VPT within its existing framework (e.g., adjusting prompt placement or initialization), this work reconceptualizes how prompts interact with image tokens. This "decoupling" perspective is broadly applicable to other scenarios involving auxiliary token injection.
- **Weight sharing as a zero-cost lunch**: Frozen CA with weight sharing matches the performance of learnable CA at only 0.09M parameters (vs. 28.4M), representing an exceptionally efficient design. This technique is transferable to any scenario requiring the insertion of new modules while minimizing parameter overhead.
- **Challenging the community consensus that "prompt methods are inferior to adapters"**: CVPT surpasses all adapter-based methods on VTAB-1K, demonstrating that the bottleneck of prompt-based methods lies in their deployment strategy rather than the paradigm itself.

## Limitations & Future Work
- **Prompt initialization strategies unexplored**: The authors acknowledge that no new prompt initialization method is proposed; the random initialization from VPT is retained. Improved initialization may further boost performance.
- **Validation limited to ViT-B/16 and ViT-L**: Larger-scale models (e.g., ViT-H, ViT-G) are not evaluated, leaving cross-model generalizability unconfirmed.
- **Fixed cross-attention placement**: The same cross-attention layout is applied across all layers; layer-adaptive deployment strategies are not explored.
- **Comparison with recent PEFT methods omitted**: Newer methods such as QLoRA and DoRA are not included in the evaluation.

## Related Work & Insights
- **vs. VPT**: VPT concatenates prompts into self-attention; CVPT decouples them via an independent cross-attention module. CVPT outperforms VPT by 5.2% on VTAB-1K and is not constrained by prompt count.
- **vs. Adapter (AdaptFormer)**: Adapters insert small learnable modules within Transformer blocks. CVPT achieves comparable or superior results via a fundamentally different approach (prompt + cross-attention), demonstrating that neither paradigm is inherently superior.
- **vs. E²VPT**: E²VPT restricts prompts to the key-value matrices but still operates within the self-attention framework. CVPT entirely removes prompts from self-attention.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The decoupling perspective is the key contribution; cross-attention itself is not novel, but its application to VPT improvement is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 25 datasets, 3 task categories, comprehensive ablation studies and efficiency analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear analysis with a coherent problem–method–experiment narrative.
- **Value**: ⭐⭐⭐⭐ — Challenges established beliefs about prompt vs. adapter methods and offers important insights for the PEFT community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[NeurIPS 2025\] VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion](../../NeurIPS2025/multimodal_vlm/vipamin_visual_prompt_initialization_via_embedding_selection_and_subspace_expans.md)

<!-- RELATED:END -->
