---
title: >-
  [Paper Note] Long-Context State-Space Video World Models
description: >-
  [ICCV 2025][Image Generation][World Models] This paper proposes integrating State Space Models (SSM/Mamba) into video world models. Through a block-wise SSM scan scheme that balances spatial consistency and temporal memory, combined with local frame attention, the method achieves persistent long-term spatial memory under linear training complexity and constant inference overhead, substantially outperforming finite-context Transformer baselines on Memory Maze and Minecraft datasets.
tags:
  - ICCV 2025
  - Image Generation
  - World Models
  - SSM
  - Mamba
  - Long-Term Memory
  - Video Diffusion Models
  - Autoregressive Generation
date: 2026-05-08
content_hash: bfd0bc02bd6fe1ac
---

# Long-Context State-Space Video World Models

**Conference**: ICCV 2025
**arXiv**: [2505.20171](https://arxiv.org/abs/2505.20171)
**Code**: To be confirmed
**Area**: Video World Models / State Space Models
**Keywords**: World Models, SSM, Mamba, Long-Term Memory, Video Diffusion Models, Autoregressive Generation

## TL;DR
This paper proposes integrating State Space Models (SSM/Mamba) into video world models. Through a block-wise SSM scan scheme that balances spatial consistency and temporal memory, combined with local frame attention, the method achieves persistent long-term spatial memory under linear training complexity and constant inference overhead, substantially outperforming finite-context Transformer baselines on Memory Maze and Minecraft datasets.

## Background & Motivation

**Background**: Video diffusion models have demonstrated promise as world models, enabling interactive environment simulation via autoregressive frame prediction conditioned on actions. Recent methods (OpenSora, CogVideoX, GameGen-X) employ Transformers with sliding-window inference for unbounded-length video generation.

**Limitations of Prior Work**:
- **Memory limitations of attention mechanisms**: The attention window in existing video world models is extremely limited (typically only a few seconds); a simple left-right camera pan in a game can cause the environment to change completely.
- **Tension between computational cost and memory**: Full causal attention incurs quadratic training complexity with respect to context length, and linear per-frame inference cost; sliding-window inference reduces complexity but entirely sacrifices long-term memory.

**Key Challenge**: Persistent and consistent world simulation requires the model to "remember" previously observed environments. However, Transformer architectures either incur prohibitive memory overhead or lose long-term information due to the sliding window. An architecture that is simultaneously efficient and capable of retaining long-term memory is needed.

**Goal**: Design a video world model architecture that maintains constant inference time and memory while possessing long-term spatial memory capability.

**Key Insight**: SSMs (Mamba) are inherently causal sequence models with fixed-size hidden states, requiring no linearly growing KV-cache during inference. Their compressed memory, while less precise than full attention, precisely suits the world model's need to "remember the big picture while refining locally."

**Core Idea**: A hybrid architecture combining a block-wise SSM scan scheme with local frame attention — SSM handles long-term temporal memory while attention handles short-range spatial refinement — achieving linear training complexity, constant inference overhead, and persistent memory.

## Method

### Overall Architecture
The model is trained with the diffusion forcing strategy (independent noise levels per frame) and supports autoregressive inference. Each network layer comprises two components: ① block-wise SSM scan → ② frame-local attention. The input is an action-frame sequence, and the output is a denoised prediction of the next frame. During inference, only a fixed-length KV-cache ($k$ frames) and SSM hidden states need to be maintained, keeping memory usage constant.

### Key Designs

1. **Block-wise SSM Scan**

    - **Function**: Balance spatial consistency and temporal memory within the SSM scan.
    - **Mechanism**: The spatial dimensions are partitioned into blocks of size $(b_h, b_w)$, each processed independently by a temporal SSM scan. Different layers use different block sizes — smaller blocks bring temporally adjacent tokens closer together (enhancing temporal memory), while larger blocks allow more spatial interaction (enhancing spatial consistency).
    - **Design Motivation**: In a standard spatial-major scan, temporally adjacent tokens are separated by $H \times W$ spatial tokens, making it difficult for the SSM's finite hidden state to retain temporal information effectively. Block partitioning reduces the distance between corresponding tokens in adjacent frames to $b_h \times b_w$, while also increasing the effective hidden state dimension per layer by allocating independent states to each block.

2. **Frame Local Attention**

    - **Function**: Compensate for SSM's weakness in precise local information retrieval.
    - **Mechanism**: Each SSM layer is followed by a block-causal attention layer, where each token can attend to all tokens in the current frame and the preceding $k$ frames. The attention mask is $M_{i,j} = 1$ iff $j \in [i-k, i]$ (in frame indices); during inference, only the KV-cache of $k$ frames is maintained.
    - **Design Motivation**: SSMs perform poorly on associative recall tasks (e.g., precisely retrieving a specific token). Local attention provides intra-frame bidirectional processing and short-range cross-frame precise alignment, complementing the SSM's long-range compressed memory.

3. **Long-Context Training**

    - **Function**: Encourage the model to learn to exploit distant context frames.
    - **Mechanism**: Training mixes standard diffusion forcing (all frames independently noised) with a modified scheme in which a randomly chosen prefix of frames is kept fully clean ($t_i = 0$), while only subsequent frames are noised and only their losses are computed.
    - **Design Motivation**: Under standard training, the model tends to rely on the most recent (information-rich) frames and does not actively attend to distant context. By adding heavier noise to nearby frames while keeping distant frames clean, the model is forced to learn to exploit long-range clean information.

### Inference Strategy
Autoregressive per-frame generation requires only: ① a fixed-length KV-cache of $k$ frames and ② the SSM hidden state for each block per layer. Both memory usage and per-frame inference time remain strictly constant regardless of generation length.

## Key Experimental Results

### Main Results — Memory Maze Spatial Retrieval Task (400-frame generation | 400-frame context)

| Model | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|------|--------|---------|--------|
| Causal (192-frame context) | 0.829 | 0.147 | 26.4 |
| Mamba2 | 0.747 | 0.313 | 20.4 |
| Mamba2 + Frame Local Attn | 0.735 | 0.336 | 19.3 |
| **Ours** | **0.898** | **0.069** | **30.8** |
| Causal (full context, reference) | 0.914 | 0.057 | 32.6 |

### Ablation Study — Memory Maze Spatial Reasoning Task (200-frame generation)

| Configuration | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|------|--------|---------|--------|
| w/o block-wise scan | 0.845 | 0.113 | 27.5 |
| w/ block size 1 | 0.766 | 0.198 | 23.1 |
| w/o long-context training | 0.809 | 0.143 | 25.3 |
| **Full model** | **0.855** | **0.099** | **28.2** |

### Minecraft Spatial Reasoning Task

| Model | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|------|--------|---------|--------|
| DFoT (SOTA) | 0.450 | 0.281 | 17.1 |
| Causal (25 frames) | 0.417 | 0.350 | 15.8 |
| **Ours** | **0.454** | **0.259** | **17.8** |

### Key Findings
- **The proposed method approaches the memory capacity of full-context Transformers** (PSNR 30.8 vs. 32.6), while reducing training complexity from quadratic to linear and inference complexity from linear to constant.
- Naively replacing attention with Mamba2 yields poor results (PSNR only 20.4), demonstrating that direct substitution is infeasible and that the block-wise design is critical.
- An excessively small block size (= 1) improves temporal memory but degrades spatial consistency, whereas an excessively large block size impairs temporal memory — necessitating the layerwise mixing of different block sizes.
- The long-context training strategy contributes substantially (PSNR improves from 25.3 to 28.2), confirming that standard diffusion forcing does not automatically enable the model to exploit distant context.
- As context distance increases, the retrieval PSNR of causal Transformers drops sharply (particularly beyond the training length), while the proposed method remains stable.

## Highlights & Insights
- **This is the first video generation work to leverage SSMs in their naturally advantageous setting (causal temporal modeling)**. Prior video SSM works apply bidirectional scans to replace attention in non-causal tasks, failing to exploit the core strengths of SSMs. By framing the problem as world modeling, this work fully utilizes SSM causality, fixed hidden state, and linear complexity.
- **The block-wise scan** is a flexible mechanism for trading off spatial versus temporal modeling. Small blocks strengthen temporal memory; large blocks strengthen spatial coherence; layerwise mixing achieves both. This idea is transferable to other tasks requiring joint spatial-temporal modeling.
- **The long-context training strategy** is simple yet effective — by adding noise to nearby frames while keeping distant frames clean, it breaks the model's tendency to rely on proximate information. This technique is broadly applicable.

## Limitations & Future Work
- Interactive frame rates (real-time inference) have not yet been achieved; future work may address this via timestep distillation.
- The model cannot effectively handle memories beyond its training context length; techniques from Mamba length extrapolation literature may be applicable.
- Experiments are limited to low-resolution synthetic environments (Memory Maze, Minecraft); validation on high-resolution real-world video remains to be done.
- The compressed memory inherent to SSMs inevitably discards information, which may be insufficient in scenarios requiring precise pixel-level recall.
- Incorporating explicit memory modules (e.g., a memory bank) alongside SSMs is a potential direction.

## Related Work & Insights
- **vs. DFoT (Diffusion Forcing Transformer)**: The current SOTA using bidirectional Transformer with diffusion forcing, but constrained by training context length (25 frames) and quadratic training complexity. The proposed method surpasses DFoT on Minecraft.
- **vs. Causal Transformer + Sliding Window**: Sliding-window inference achieves constant speed but completely sacrifices long-term memory. The proposed method maintains constant inference while retaining long-term memory.
- **vs. GameGen-X / Genie**: These open-world generation methods also rely on sliding windows and explicitly acknowledge the lack of long-term consistency — the core problem this paper addresses.
- **vs. Mamba-based Video Generation (DiS, ZigMa)**: These works use bidirectional Mamba to replace attention for non-causal generation, failing to exploit the causal advantages of SSMs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First application of SSMs to their naturally advantageous setting in video world modeling; the block-wise scan design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Dual-dataset evaluation on Memory Maze and Minecraft, with comprehensive ablations and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem definition is clear, motivation is well-articulated, and the complexity comparison in Tab. 1 is immediately informative.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the critical bottleneck of long-term memory in video world models, laying an architectural foundation for interactive persistent world simulation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](../../NeurIPS2025/image_generation/rlvr-world_training_world_models_with_reinforcement_learning.md)
- [\[AAAI 2026\] LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](../../AAAI2026/image_generation/longllada_unlocking_long_context_capabilities_in_diffusion_llms.md)
- [\[ICCV 2025\] AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction](animegamer_infinite_anime_life_simulation_with_next_game_state_prediction.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](aether_geometric-aware_unified_world_modeling.md)
- [\[ICCV 2025\] Less-to-More Generalization: Unlocking More Controllability by In-Context Generation](less-to-more_generalization_unlocking_more_controllability_by_in-context_generat.md)

<!-- RELATED:END -->
