---
title: >-
  [Paper Note] Mixture of States: Routing Token-Level Dynamics for Multimodal Generation
description: >-
  [CVPR 2026][Image Generation][Multimodal diffusion models] This paper proposes Mixture of States (MoS)—a multimodal fusion paradigm based on learnable token-level sparse routing—enabling visual tokens to adaptively selec…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Multimodal diffusion models"
  - "dynamic routing"
  - "Mixture of States"
  - "text-to-image generation"
  - "image editing"
  - "sparse interaction"
date: 2026-05-08
content_hash: 4ed0918bbb73b1ed
---

# Mixture of States: Routing Token-Level Dynamics for Multimodal Generation

**Conference**: CVPR 2026
**arXiv**: [2511.12207](https://arxiv.org/abs/2511.12207)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: Multimodal diffusion models, dynamic routing, Mixture of States, text-to-image generation, image editing, sparse interaction

## TL;DR

This paper proposes Mixture of States (MoS)—a multimodal fusion paradigm based on learnable token-level sparse routing—enabling visual tokens to adaptively select hidden states from arbitrary layers of a text encoder at each denoising step. With only 3–5B parameters, MoS matches or surpasses models at the 20B scale.

## Background & Motivation

**Modality representation gap**: Text models (contrastive learning / masked prediction / next-token prediction) and visual models (diffusion / flow matching) have fundamentally different training objectives, making alignment of their heterogeneous representations a core challenge.

**Inherent limitations of existing fusion strategies**: Cross-Attention uses only the final-layer features of the text encoder, limiting information richness; Self-Attention concatenates text and visual tokens, incurring quadratic complexity with respect to sequence length.

**Rigidity of MoT**: Mixture-of-Transformers requires text and visual branches to share the same depth and hidden dimensionality, enforcing strict layer-to-layer correspondence and precluding asymmetric architectures.

**Mismatch between static conditioning and dynamic denoising**: Existing methods encode text embeddings once and keep them fixed, whereas the noise level and visual features evolve dynamically across denoising timesteps, creating an information mismatch.

**Insufficiency of single-layer representations**: Experiments demonstrate that using a single fixed layer's global embedding to represent all tokens is suboptimal; different tokens should adaptively draw representations from different layers.

**Parameter efficiency demands**: Although existing SOTA models (e.g., Qwen-Image 20B) achieve strong performance, their parameter counts are prohibitively large, motivating the need for efficient alternatives that reach comparable performance at smaller scale.

## Method

### Overall Architecture

MoS adopts a **dual-tower architecture** comprising an Understanding Tower ($\mathcal{U}$) and a Generation Tower ($\mathcal{G}$), connected via a learnable router $\mathcal{R}$. The Understanding Tower processes multimodal context (text or text + image), while the Generation Tower handles visual synthesis. During training, the Understanding Tower is frozen; only the Generation Tower and the router are trained. The entire model is trained end-to-end with Rectified Flow Matching:

$$\mathbb{E}_{c,t,z_0,z_1}\Big[\big\|\mathcal{G}(z_t, t, \mathcal{R}(t, c, z_t, \mathcal{U}(c))) - v_t\big\|_2^2\Big]$$

### Key Designs: The MoS Router

**Router input space**: The router receives three types of signals simultaneously—(1) text prompt embeddings $c$ (dimension-aligned via a shared projection layer followed by a linear layer); (2) noised image latents $z_t$ (via a shared patchify layer and projection); and (3) denoising timestep $t$ (sinusoidal embedding followed by projection). All three signals are projected to the same hidden dimensionality before concatenation.

**Router output space**: For each context token, the router predicts a logit matrix $\mathcal{W} \in \mathbb{R}^{m \times n}$, where $m$ is the depth of the Understanding Tower and $n$ is the depth of the Generation Tower. Each entry $w_{ij}$ represents the affinity weight for routing the $i$-th layer state of the Understanding Tower to the $j$-th layer of the Generation Tower. **Each token independently predicts its own routing matrix**, rather than sharing a global strategy.

**Lightweight router architecture**: All input embeddings are tokenized, normalized, and concatenated into a sequence, then processed by a two-layer bidirectional self-attention Transformer block to capture contextual semantics, with a projection layer producing the logit matrix. The router contains only 100M parameters and introduces negligible latency overhead (0.008 s per iteration).

**Sparse Top-$k$ selection with $\varepsilon$-greedy exploration**: For each layer $j$ of the Generation Tower, softmax normalization is applied to the logit column $w_{:,j}$, and the top-$k$ Understanding Tower layers with the highest weights are selected for weighted aggregation of hidden states:

$$\mathbf{S}_j^c = \sum_{i \in I_j} \bar{w}_{ij} \cdot \mathcal{S}_i^c$$

During training, $k$ layers are selected randomly with probability $\varepsilon$ (exploration) and via top-$k$ with probability $1-\varepsilon$ (exploitation), preventing premature convergence to suboptimal routing. At inference time, $\varepsilon = 0$.

### Loss & Training

The standard Rectified Flow Matching loss is employed, with target velocity $v_t = z_1 - z_0$, where $z_t = (1-t)z_0 + tz_1$, $z_0$ denotes the VAE-encoded image latent, and $z_1 \sim \mathcal{N}(0, I)$.

### Task Extensions

- **MoS-Image** (text-to-image): The Understanding Tower processes text; features aggregated by the router are projected and concatenated with visual features as in-context tokens.
- **MoS-Edit** (image editing): The Understanding Tower processes both a reference image and a text instruction; the Generation Tower receives Gaussian noise and the clean reference image for iterative refinement.

### Training Strategy

A four-stage progressive training schedule is adopted: Stage 1—low resolution $512^2$ (1,400 A100-days) → Stage 2—high resolution $1024^2$ → Stage 3—aesthetic fine-tuning (10M high-quality samples, 100 A100-days) → Stage 4—ultra-high-resolution $2048^2$ fine-tuning (1M samples, 80 A100-days). MoS-Edit requires an additional 50 A100-days. The total cost is approximately 3,000 A100-days, substantially lower than the 6,250 A100-days of SD v1.5.

## Key Experimental Results

### Main Results

| Model | Interaction Type | Parameters | GenEval↑ | DPG↑ | GEdit↑ | ImgEdit↑ |
|-------|-----------------|------------|----------|------|--------|----------|
| Qwen-Image | Self-Attn | 20B | 0.87 | 88.32 | 7.56 | 4.27 |
| SANA-1.5 | Cross-Attn | 4.8B | 0.81 | 84.70 | — | — |
| FLUX.1[Dev] | Self-Attn | 12B | 0.66 | 83.84 | — | — |
| Bagel | MoT | 14B | 0.88 | — | 6.52 | 3.20 |
| **MoS-S** | **MoS** | **3B** | **0.89** | **86.33** | **7.41** | **4.17** |
| **MoS-L** | **MoS** | **5B** | **0.90** | **87.01** | **7.86** | **4.33** |

MoS-L (5B) outperforms Qwen-Image (20B) on GenEval, GEdit, and ImgEdit, using only one-quarter of its parameters.

### Ablation Study

| Ablation Dimension | Key Findings |
|-------------------|-------------|
| Router inputs | Full dynamic conditioning (Prompt + Latent + Timestep) is optimal (FID 20.15 vs. 21.12 with Prompt only) |
| Prediction granularity | Token-level prediction outperforms sample-level (FID 20.17 vs. 21.66) |
| Layer selection | Adaptive routing substantially outperforms manually fixed routing (FID 17.77 vs. 21.51) |
| MoS vs. MoT | Under identical parameters, data, and compute, MoS consistently outperforms MoT across all training stages |
| MoS vs. Cross-Attn | GenEval 0.79 vs. 0.74; DPG 85.61 vs. 83.40 |

### Key Findings

- Timestep-awareness in the router is critical—different denoising stages require different conditioning signals.
- Token-level routing patterns naturally exhibit diverse strategies without explicit regularization.
- The router introduces minimal latency overhead (0.008 s/iter), and overall inference speed is faster than Qwen-Image and Bagel.
- Combined with Self-CoT reasoning, MoS-L improves on the WISE benchmark from 0.54 to 0.65.

## Highlights & Insights

- **Clear core contribution**: The MoS router unifies three design principles—sparse, dynamic, and token-level routing—breaking the rigidity of MoT's symmetric architecture and enabling flexible fusion in an asymmetric dual-tower setting.
- **Exceptional parameter efficiency**: A 5B model matches or surpasses a 20B model, with a training cost of 3,000 A100-days far below prior methods.
- **Rigorous ablation design**: The three core hypotheses—dynamic conditioning, token-level prediction, and adaptive layer selection—are validated individually, yielding strong empirical support.
- **Task unification**: The same framework supports both text-to-image generation and image editing; freezing the Understanding Tower preserves the model's original comprehension capabilities.

## Limitations & Future Work

- MoS currently supports only unidirectional (understanding → generation) interaction; bidirectional fusion (e.g., joint training) remains unexplored.
- Only SFT is employed as the post-training strategy; methods such as GRPO and RLHF for human preference alignment have not been investigated.
- Visual artifacts persist when generating small objects.
- Although efficient, freezing the Understanding Tower may impose an upper bound on how effectively the Generation Tower exploits the understanding representations.

## Related Work & Insights

- **Cross-Attention family** (SD, PixArt-α, SANA-1.5): Uses only final-layer features, limiting information richness.
- **Self-Attention family** (FLUX, Qwen-Image): Full-sequence interaction achieves strong performance but at high computational cost.
- **MoT family** (LMFusion, Bagel, Mogao): Layer-wise shared KV, but requires symmetric architectures.
- **Dynamic networks** (MoE, MoD, MoR): Sparse adaptive computation concepts, primarily applied to intra-model routing; MoS extends this idea to inter-model collaboration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The MoS router constitutes a novel cross-modal fusion paradigm; the token-level dynamic sparse routing design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablations comprehensively cover inputs, outputs, layer selection, and efficiency; multi-benchmark, multi-task evaluation with fair comparisons against MoT, Cross-Attn, and Self-Attn baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The three design principles are introduced in a logically progressive manner, with clear figures and rigorous argumentation.
- **Value**: ⭐⭐⭐⭐⭐ — A 4× improvement in parameter efficiency carries significant practical impact, providing a general fusion solution for asymmetric multimodal architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BiGain: Unified Token Compression for Joint Generation and Classification](bigain_unified_token_compression_for_joint_generation_and_classification.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[CVPR 2026\] ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing](changebridge_spatiotemporal_image_generation_with_multimodal_controls_for_remote.md)
- [\[CVPR 2026\] Guiding a Diffusion Transformer with the Internal Dynamics of Itself](guiding_a_diffusion_transformer_with_the_internal_dynamics_of_itself.md)
- [\[CVPR 2026\] TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts](tag-moe_task-aware_gating_for_unified_generative_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
