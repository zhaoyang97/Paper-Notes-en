---
title: >-
  [Paper Note] Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation
description: >-
  [CVPR 2026][Multimodal VLM][multimodal fusion] This paper proposes Mixture of States (MoS), a novel fusion paradigm for multimodal diffusion models. A lightweight…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "multimodal fusion"
  - "state routing"
  - "T2I/image editing"
  - "asymmetric Transformer"
  - "token-level dynamics"
date: 2026-05-08
content_hash: 166115612a99c04a
---

# Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation

**Conference**: CVPR 2026
**arXiv**: [2511.12207](https://arxiv.org/abs/2511.12207)  
**Code**: None (but built on open-source components)  
**Area**: Image Generation / Multimodal Fusion / Diffusion Models
**Keywords**: multimodal fusion, state routing, T2I/image editing, asymmetric Transformer, token-level dynamics

## TL;DR
This paper proposes Mixture of States (MoS), a novel fusion paradigm for multimodal diffusion models. A lightweight, learnable token-level router dynamically routes hidden states from arbitrary layers of an understanding tower (frozen LLM/VLM) to arbitrary layers of a generation tower (DiT). With only 3–5B parameters, MoS matches or surpasses the 20B Qwen-Image on both image generation and editing benchmarks.

## Background & Motivation
The central challenge in multimodal diffusion models is effective alignment of text and visual signals. Existing fusion strategies each carry inherent limitations: (1) cross-attention uses only the final-layer features, providing limited information; (2) self-attention concatenates text and visual tokens, incurring $O(n^2)$ computational cost; (3) Mixture-of-Transformers (MoT) shares KV projections layer by layer, requiring both towers to be structurally symmetric and of equal depth, which is highly inflexible. Three key design principles are commonly overlooked: layer selection should be adaptive rather than fixed, conditioning signals should vary dynamically with the denoising timestep, and conditioning signals should be personalized at the token level.

## Core Problem
Can one design a flexible cross-modal fusion mechanism that permits fully asymmetric understanding and generation towers (differing in depth and width), while adapting the fusion strategy dynamically to input content and denoising progress?

## Method

### Overall Architecture
A dual-tower design is adopted: an understanding tower $\mathcal{U}$ (frozen PLM-8B or InternVL-14B) processes text and image conditions, while a generation tower $\mathcal{G}$ (a 3B/5B DiT trained from scratch) performs diffusion denoising. A lightweight router $\mathcal{R}$ (only 100M parameters, comprising 2 Transformer blocks) dynamically determines, given the prompt, the noisy image $z_t$, and the timestep $t$, which layers of the understanding tower should have their hidden states routed to which layers of the generation tower.

### Key Designs
1. **Token-level sparse routing**: Each context token independently predicts a logit matrix $\mathcal{W} \in \mathbb{R}^{m \times n}$ (where $m$ and $n$ denote the number of layers in the understanding and generation towers, respectively). Each entry $w_{ij}$ represents the routing weight from the $i$-th layer of the understanding tower to the $j$-th layer of the generation tower. After softmax normalization, top-$k$ ($k=2$) selection is applied, transmitting only the two most relevant hidden states. A key finding is that token-level routing outperforms sample-level routing (FID 20.17 vs. 21.66), as different tokens require features from different layers.

2. **Timestep-sensitive routing**: The router takes three inputs — the text prompt, the noisy latent $z_t$, and the denoising timestep $t$. Ablations confirm that all three are indispensable (FID: prompt only 21.12 → +latent 21.89 → **+timestep 20.15**). Visualizations reveal that routing patterns shift over the course of denoising: early steps exhibit sparse selection of specific layers, while later steps tend toward more uniform weight distributions — consistent with the diffusion model's "structure first, then details" denoising behavior.

3. **$\epsilon$-greedy exploration during training**: With probability $\epsilon=0.05$, a layer is selected at random rather than via top-$k$, preventing the router from collapsing into local optima. Ablations show that $\epsilon$-greedy training accelerates convergence and yields better final performance. $k=2$ is found optimal — $k=1$ is overly localized, while $k \geq 3$ dilutes information.

### Loss & Training
Standard rectified flow matching training: $\mathbb{E}[\|v_t - \mathcal{G}(z_t, t, \mathcal{R}(\cdot))\|^2]$. A four-stage progressive training schedule is adopted: $512^2$ resolution (1400 A100-days) → $1024^2$ (equivalent compute) → aesthetic fine-tuning (100 A100-days) → $2048^2$ super-resolution (80 A100-days). The total cost is approximately 3,000 A100-days — substantially lower than the 6,250 A100-days required for SD1.5.

## Key Experimental Results

| Method | Params | Fusion Type | GenEval↑ | DPG↑ | oneIG↑ | ImgEdit↑ |
|--------|--------|-------------|----------|-------|--------|----------|
| FLUX.1[Dev] | 12B | Self-Attn | 0.66 | 83.84 | 0.43 | — |
| SANA-1.5 | 4.8B | Cross-Attn | 0.81 | 84.70 | 0.33 | — |
| Bagel | 14B | MoT | 0.88 | — | 0.36 | 3.20 |
| Qwen-Image | **20B** | Self-Attn | 0.87 | 88.32 | 0.54 | 4.27 |
| **MoS-S** | **3B** | MoS | 0.89 | 86.33 | 0.50 | 4.17 |
| **MoS-L** | **5B** | MoS | **0.90** | **87.01** | **0.52** | **4.33** |

MoS-L (5B) even surpasses Qwen-Image (20B) on GenEval (0.90) and ImgEdit (4.33) while using only one-quarter of the parameters.

### Ablation Study
- **MoS > MoT > Cross-Attn**: FID 17.77 vs. 21.66 (manual), GenEval 0.79 vs. 0.74 (Cross-Attn)
- **Advantage of asymmetric towers**: The understanding tower can be independently scaled (8B → 14B yields consistent gains), which is not achievable under MoT
- **Negligible router overhead**: Only 0.008s per iteration
- **Lower total latency**: MoS < Qwen-Image ≈ Bagel (since the understanding tower is executed only once)
- **Equally effective for editing**: The dual towers capture features of the reference image at different granularities (semantic vs. pixel-level)

## Highlights & Insights
- **MoS breaks the symmetry constraint of MoT** — enabling fully heterogeneous understanding and generation towers to be freely combined, which is highly valuable for practical deployment
- The strategy of freezing the understanding tower and training only the generation tower substantially reduces training cost — achieving state-of-the-art performance at approximately 3,000 A100-days
- Token-level, timestep-sensitive routing represents a paradigm shift in diffusion model fusion — moving away from "a single embedding for all denoising steps"
- Router visualizations provide an interpretable window into cross-modal interaction — confirming that different tokens and different timesteps genuinely require features from different layers
- The efficiency story of 5B matching 20B is highly compelling and directly relevant to industrial deployment

## Limitations & Future Work
- Currently supports only unidirectional routing from understanding tower to generation tower; bidirectional MoS may yield further gains
- RLHF/GRPO-based human preference alignment has not been explored
- Small-object generation still exhibits visual artifacts
- Combinations with efficiency techniques such as quantization, distillation, and feature caching remain unexplored
- Validation is limited to image generation and editing; MoS for video generation has yet to be investigated

## Related Work & Insights
- **vs. MoT (Bagel/LMFusion)**: MoT requires symmetric towers with layer-to-layer correspondence, severely limiting flexibility. MoS achieves arbitrary layer-to-layer sparse connections via a router, and MoS at 3B outperforms Bagel at 14B.
- **vs. Cross-Attention (SANA/PixArt)**: Cross-attention uses only the final-layer embedding — static and informationally limited. MoS dynamically selects hidden states across all layers.
- **vs. Self-Attention (FLUX/SD3)**: Self-attention is computationally expensive and likewise static. MoS incurs lower computation (smaller generation tower) while adapting dynamically.
- **vs. Qwen-Image (20B)**: Qwen-Image is performant but 4× larger. MoS-L (5B) matches or exceeds its performance.

The "heterogeneous towers + router" architecture of MoS is directly extensible to **video generation**, where the understanding tower processes text and keyframes while the router adapts dynamically to timestep and frame position. MoS is orthogonally complementary to LinVideo — which replaces softmax with linear attention to reduce per-step cost — and the two approaches can be combined. Token-level routing may also inspire dynamic cross-modal interaction in **VLM reasoning**, where layer-fixed fusion is currently the norm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — MoS introduces a new fusion paradigm that breaks the symmetry constraint; token-level and timestep-level routing are both original contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablations (routing inputs/outputs/architecture/sparsity/scaling), multi-task evaluation (generation + editing), and multiple benchmarks (GenEval/DPG/WISE/oneIG/ImgEdit/GEdit)
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from three design principles → MoS design → systematic ablation → state-of-the-art results is presented with exceptional clarity
- Value: ⭐⭐⭐⭐⭐ — The 5B = 20B efficiency narrative, interpretable routing, and paradigm-level innovation together constitute a highly significant contribution to the image generation field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[CVPR 2026\] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)
- [\[CVPR 2026\] SALMUBench: A Benchmark for Sensitive Association-Level Multimodal Unlearning](salmubench_a_benchmark_for_sensitive_association-level_multimodal_unlearning.md)

</div>

<!-- RELATED:END -->
