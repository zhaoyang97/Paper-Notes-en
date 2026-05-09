---
title: >-
  [Paper Note] Make Your Training Flexible: Towards Deployment-Efficient Video Models
description: >-
  [ICCV 2025][LLM Pretraining][Flexible Training] This paper proposes Flux — a data augmentation tool that enables flexible video model training through flexible sampling grids and group-dynamic token selection, allowing a single model to operate efficiently across varying computational budgets. The paper further introduces a Token Optimization test-time paradigm that matches previous SOTA performance using only 1/4 of the tokens, saving approximately 90% of computation.
tags:
  - ICCV 2025
  - LLM Pretraining
  - Flexible Training
  - Token Optimization
  - Video Pretraining
  - Deployment Efficiency
  - Data Augmentation
date: 2026-05-08
content_hash: bfa5f5f406e54ef5
---

# Make Your Training Flexible: Towards Deployment-Efficient Video Models

**Conference**: ICCV 2025
**arXiv**: [2503.14237](https://arxiv.org/abs/2503.14237)
**Code**: [https://github.com/OpenGVLab/FluxViT](https://github.com/OpenGVLab/FluxViT)
**Area**: LLM Pretraining
**Keywords**: Flexible Training, Token Optimization, Video Pretraining, Deployment Efficiency, Data Augmentation

## TL;DR
This paper proposes Flux — a data augmentation tool that enables flexible video model training through flexible sampling grids and group-dynamic token selection, allowing a single model to operate efficiently across varying computational budgets. The paper further introduces a Token Optimization test-time paradigm that matches previous SOTA performance using only 1/4 of the tokens, saving approximately 90% of computation.

## Background & Motivation

**State of the Field**: Video representation learning is a foundational task in computer vision, critical for multimodal LLMs and embodied AI. Current mainstream methods operate on fixed spatiotemporal sampling grids (e.g., 8 frames × 224²) with a fixed number of tokens, leading to substantial redundancy during both training and deployment.

**Limitations of Prior Work**:
   - **Redundancy from fixed sampling**: Videos contain abundant spatiotemporal redundancy; many tokens extracted via fixed sampling carry low information content.
   - **Inflexible deployment**: Models are trained at 8 frames × 224², yet real deployment may require adaptation to varying computational budgets. Directly reducing frames or resolution causes significant performance degradation.
   - **Limited token reduction**: Existing token pruning/merging methods perform poorly at high reduction ratios, and the strategies themselves introduce computational overhead.
   - **Incomplete flexible training methods**: ResFormer and FFN address spatial or temporal flexibility independently, without jointly handling both dimensions, and neither has been validated in large-scale pretraining.

**Root Cause**: How can a single model simultaneously satisfy deployment requirements under diverse computational budgets? Simply reducing frames or resolution is suboptimal — for a given token budget, the goal should be to select the most information-maximizing token set.

**Paper Goals**: Propose the Token Optimization paradigm — under a given token budget, select the optimal token set from better-sampled video to maximize information.

**Starting Point**: Combine flexible sampling and token selection as training-time data augmentation, enabling the model to naturally adapt to various resolutions and token counts. A test-time Token Optimization strategy is also introduced to identify the optimal sampling-selection combination.

**Core Idea**: Use flexible sampling and group-dynamic token selection as zero-cost training augmentation, enabling video models to identify the optimal token set via Token Optimization across all computational budgets.

## Method

### Overall Architecture
As illustrated in Figs. 2–3, Flux comprises three levels of design: (1) **Flexi-Sampling**: randomly selecting different frame counts and resolutions during training; (2) **Group-Dynamic Token Selector**: selecting a high-information token subset from the flexibly sampled token pool; (3) **FluxViT architectural enhancements**: GLPE (Global-Local Positional Embedding) and DPN (Dual Patch Normalization) to adapt ViT to variable token counts. At test time, Token Optimization searches for the optimal sampling-selection configuration.

### Key Designs

1. **Flexi-Sampling**:

    - Function: Each video during training adopts a randomly chosen spatiotemporal resolution.
    - Mechanism: For each video in the batch, a frame count $[F_{min}, F_{max}]$ (stride $t_s$) and spatial resolution $[R_{min}, R_{max}]$ (stride $r_s$) are randomly selected, with a token count threshold $T_{thres}$ to maintain a reasonable pool size. Default settings: 4–24 frames, 168–252 resolution.
    - Design Motivation: Models trained with fixed sampling have only observed a single resolution and generalize poorly to others. Flexible sampling exposes the model to diverse resolution combinations, naturally inducing cross-resolution robustness.

2. **Group-Dynamic Token Selector**:

    - Function: Selects the most informative token subset from the token pool for the teacher model.
    - Mechanism: The frame sequence is evenly divided into $N$ sparse groups $B_i$. Within each group, the dynamic value of each token is computed via inter-frame difference: $D(F_{t+1,i}) = \|F_{t+1,i} - F_{t,i}\|_p$. The top $K/N$ tokens with the highest dynamic values are selected. This ensures: (a) tokens with the greatest variation (highest information) are selected; (b) grouping guarantees full-video temporal coverage.
    - Design Motivation: A large proportion of video tokens represent static background (low information). Tokens with high inter-frame variation are more semantically meaningful. Grouping prevents tokens from localized rapid motion from dominating while neglecting other temporal segments.

3. **Double Mask Module**:

    - Function: Simultaneously augments both teacher and student within the UMT (Unmasked Teacher) framework.
    - Mechanism: On the teacher side, Flexi-Sampling combined with the Group-Dynamic Selector yields informative tokens; on the student side, masking is based on attention scores from the teacher's CLS token. The two masks are complementary — the teacher provides high-quality representations filtered from richly sampled video, while the student learns to understand video from a sparser perspective.
    - Design Motivation: Richer information is extracted from higher-resolution sampling without increasing the teacher's computational cost (the number of tokens after selection remains unchanged).

4. **Global-Local Positional Embedding (GLPE)**:

    - Function: Provides positional encoding for a flexible number and arrangement of tokens.
    - Mechanism: Globally, a learnable positional encoding (sine-cosine initialized) combined with a Depth-Wise Conv enhances local relationships. A linear projection encodes local position over Value vectors in the attention mechanism: $Z = (\text{Softmax}(\frac{QK^T}{\sqrt{D}}) + LPE) \cdot V$. The LPE is value-dependent and is not affected by the number of input tokens.
    - Design Motivation: Standard positional encodings assume a fixed token count and arrangement. When tokens are selected or masked, they originate from discrete spatiotemporal positions that require explicit positional encoding.

5. **Dual Patch Normalization (DPN)**:

    - Function: Stabilizes training under flexible sampling.
    - Mechanism: A LayerNorm is added after the standard Patch Embedding layer (to assist dynamic estimation) and another before the Patch Embedding layer (to stabilize gradients).
    - Design Motivation: Flexible sampling causes large distributional shifts in input tokens, potentially leading to excessively large gradients in Patch Embedding. Dual normalization stabilizes training.

### Loss & Training
- Flux-PT (Pretraining): Teacher-student alignment loss under the UMT framework, using InternVideo2-1B as the teacher.
- Flux-FT (Fine-tuning): Standard supervised training with self-distillation, where aggregated features from larger token counts guide training with smaller token counts.
- Multi-number co-training: Three different token counts are used within a single batch to train the student, maximizing the utilization of teacher computation.

## Key Experimental Results

### Main Results

| Model | K400 Top-1 | SSv2 Top-1 | MSRVTT R@1 | COIN | Scale |
|------|-----------|-----------|-----------|------|------|
| InternVideo2-S | 87.8 | - | - | - | Small |
| **FluxViT-S** | **90.0** | - | - | - | Small |
| InternVideo2-B | 89.0 | 73.5 | 48.2 | 92.5 | Base |
| **FluxViT-B** | **90.0** | **75.8** | **49.9** | **94.1** | Base |

### Token Optimization Results

| Configuration | Token Count | K400 | Relative to Full | Compute Saved |
|------|---------|------|---------|---------|
| FluxViT-B Full | 3072 | 90.0 | 100% | 0% |
| FluxViT-B TO (1/4) | 768 | ~89.0 | ~99% | ~90% |
| InternVideo2-B Full | 3072 | 89.0 | - | 0% |

### Ablation Study

| Configuration | K400 | SSv2 | Note |
|------|------|------|------|
| Baseline (InternVideo2 UMT) | 87.5 | 71.8 | Original pipeline |
| + Flexi-Sampling | 88.2 | 73.0 | Flexible sampling improves robustness |
| + Group-Dynamic Selector | 89.0 | 74.2 | Informative token selection is effective |
| + GLPE + DPN | 89.5 | 75.0 | Architectural enhancements are critical |
| + Multi-number training | 90.0 | 75.8 | Co-training with multiple token counts further improves performance |

### Key Findings
- **Remarkable Token Optimization efficiency**: 1/4 of the tokens suffice to match the performance of the previous SOTA (InternVideo2), saving approximately 90% of computation — demonstrating substantial redundancy in fixed sampling.
- **Generality of Flux as an augmentation tool**: Effective in both pretraining (UMT) and supervised fine-tuning without increasing training cost.
- **Joint spatiotemporal flexibility outperforms single-dimensional flexibility**: ResFormer (spatial) and FFN (temporal) each address only one dimension; Flux handles both simultaneously and achieves superior results.
- **FluxViT-B surpasses larger models across multiple tasks**: K400 90.0%, SSv2 75.8%, MSRVTT 49.9%, COIN 94.1% — new SOTA at the same scale.
- **Improvements on chat-centric tasks**: When used as a visual encoder integrated with an LLM, FluxViT outperforms SigLIP/CLIP on MVBench and Dream-1k.

## Highlights & Insights
- **Token Optimization as a new paradigm**: The shift from "fixed sampling + all tokens" to "flexible sampling + optimal token selection" represents a paradigm shift in video model deployment — not "fewer frames or lower resolution," but "finding the optimal token set under a given budget."
- **Zero-cost augmentation**: Token selection in Flux keeps the number of tokens processed by the teacher unchanged, incurring no additional training cost. Sampling from higher resolutions thus becomes a "free lunch."
- **Elegant design of the Group-Dynamic Selector**: Grouping ensures temporal coverage; only tokens with high inter-frame variation are selected. Simple yet highly effective — it guarantees informativeness while avoiding overfitting to rapid motion.
- **Validation with LLM integration**: Validation in chat-centric settings opens the door to applying Flux in multimodal LLMs.

## Limitations & Future Work
- The optimal configuration search in Token Optimization incurs non-trivial overhead (requiring evaluation of multiple configurations on a validation set).
- Flexible sampling increases data preprocessing complexity due to multi-resolution support.
- InternVideo2-1B is used as the teacher, and teacher quality constitutes an upper bound for the student model.
- GLPE and DPN introduce a small number of additional parameters and computations.
- A learned token selector (rather than the heuristic inter-frame-difference-based approach) could be explored.

## Related Work & Insights
- **vs. ResFormer/FFN**: ResFormer addresses only spatial flexibility and FFN only temporal flexibility; neither has been validated in large-scale pretraining. Flux handles both spatiotemporal dimensions and is validated at the InternVideo2 scale.
- **vs. MAR/MCM (token reduction)**: Traditional masking reduces tokens at fine-tuning or inference time but lacks training-time flexibility. Flux introduces flexibility during training, enabling the model to naturally adapt.
- **vs. InternVideo2**: FluxViT comprehensively outperforms the InternVideo2 series at the same scale, demonstrating the effectiveness of Flux augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐ The Token Optimization paradigm is innovative, though the core components (flexible sampling, dynamic selection) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive evaluation across multiple tasks (action recognition + retrieval + chat), scales, and settings.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and systematic ablation.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for efficient video model deployment; matching SOTA with 1/4 tokens is a strong result.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[CVPR 2026\] FlowMotion: Training-Free Flow Guidance for Video Motion Transfer](../../CVPR2026/llm_pretraining/flowmotion_training-free_flow_guidance_for_video_motion_transfer.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](syncity_training-free_generation_of_3d_worlds.md)
- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](../../NeurIPS2025/llm_pretraining/efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)

<!-- RELATED:END -->
