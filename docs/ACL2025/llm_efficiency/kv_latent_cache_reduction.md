---
title: >-
  [Paper Note] KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding
description: >-
  [ACL 2025][LLM Efficiency][KV Cache compression] KV-Latent achieves 50-87% KV Cache compression with less than 1% of the pre-training tokens' computational cost while maintaining performance. It achieves this by directly shrinking Key/Value attention head dimensions (mapping KV vectors to a low-dimensional latent space) and adapting a two-stage fine-tuning strategy along with frequency-aware RoPE modifications.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "KV Cache compression"
  - "attention head dimension reduction"
  - "RoPE frequency awareness"
  - "knowledge distillation"
  - "inference acceleration"
date: 2026-05-08
content_hash: 1794664e72aa355b
---

# KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding

**Conference**: ACL 2025  
**arXiv**: [2507.11273](https://arxiv.org/abs/2507.11273)  
**Code**: [https://github.com/ShiLuohe/KV-Latent](https://github.com/ShiLuohe/KV-Latent)  
**Area**: LLM Efficiency  
**Keywords**: KV Cache compression, attention head dimension reduction, RoPE frequency awareness, knowledge distillation, inference acceleration

## TL;DR

KV-Latent achieves 50-87% KV Cache compression with less than 1% of the pre-training tokens' computational cost while maintaining performance. It achieves this by directly shrinking Key/Value attention head dimensions (mapping KV vectors to a low-dimensional latent space) and adapting a two-stage fine-tuning strategy along with frequency-aware RoPE modifications.

## Background & Motivation

**Background**: The Transformer decoder architecture requires preserving Key and Value states (KV Cache) for each token at inference time, which scales linearly with context length and poses a primary memory and bandwidth bottleneck. Existing KV Cache compression methods mainly operate on three levels: head-level (reusing KV heads like MQA/GQA), layer-level (reusing KV columns across layers), and token-level (evicting or merging low-importance tokens).

**Limitations of Prior Work**: Head-level methods (GQA) are widely adopted but have a limited compression ceiling; layer-level methods suffer from non-continuity across layers, rendering them incompatible with operational optimizations; token-level methods depend on dynamic attention scores, making them incompatible with kernel-level optimizations like FlashAttention and hard to control at a fine granularity. Crucially, the direction of directly reducing the dimension of Key/Value vectors within each attention head remains almost unexplored.

**Key Challenge**: In MHA, the constraint $d_h \times n_h = d$ is implicitly assumed to be unbreakable, yet works like GQA have proved the low-rank nature of KV Cache—storing full $d$-dimensional vectors is not necessary. The core contradiction is: can $d_{qk}$ and $d_{vo}$ be decoupled to compress the KV Cache directly at the dimension level?

**Goal**: (1) How to downsample attention head dimensions from pre-trained models with minimal extra training cost; (2) How to address severe RoPE instabilities at low dimensions (where high-frequency oscillation noise is of the same scale as decaying signals when the dimension is $<32$); (3) Whether Key and Value compression affects performance symmetrically.

**Key Insight**: The authors observe that $K$ and $V$ are essentially low-rank transformations downsampling the $d$-dimensional hidden state to $d_h$ dimensions, whereas $Q^\top$ and $O$ perform upsampling. The KV Cache stores the results of these low-rank transformations—since they are already latent representations, this latent space can be compressed further.

**Core Idea**: Uniformly downsampling and directly pruning the K/V projection matrix dimensions of pre-trained models, followed by a two-stage recovery (intra-layer distillation + end-to-end fine-tuning), while employing a frequency-aware RoPE sampling to eliminate high-frequency noise at low dimensions.

## Method

### Overall Architecture

Given a pre-trained LLM (e.g., LLaMA-3-8B), the column dimensions of $W_K, W_V$ and $W_Q$ as well as the row dimension of $W_O$ in each attention layer are compressed synchronously (reducing QK from $d_h$ to $d_{qk}$, and VO from $d_h$ to $d_{vo}$). A two-stage training recovers performance. During inference, the KV Cache size scales down proportionally to the dimensions.

### Key Designs

1. **Model Preparation via Uniform Downsampling**:

    - **Function**: Extracting compressed initial weights from the pre-trained model weights.
    - **Mechanism**: Due to Rotary Position Embedding (RoPE), channels within an attention head exhibit rotational symmetry, meaning a uniform sampling of channels (with a stride proportional to the reduction ratio) preserves key information. For example, to shrink $d_{qk}$ to 1/4, $\tilde{W_Q^{(i)}} = W_Q^{(i)}[:, ::4]$. FFN layers are adjusted using LoRA (rank=256) instead of full-parameter tuning.
    - **Design Motivation**: Although SVD decomposition is theoretically superior, the rotation introduced by RoPE does not commute with matrix multiplication, rendering SVD difficult to apply directly. Uniform sampling is simple, effective, and inherently compatible with the channel-pair rotation structure of RoPE.

2. **Two-Stage Training**:

    - **Function**: Recovering the performance of the pruned model using minimal data (only 1B tokens from FineWeb-edu) in two stages.
    - **Mechanism**: **Stage I (Intra-layer Distillation)**—The original model is frozen, and target layers are aligned layer-by-layer. Given each layer's input $H_i^{(l)}$, the original and modified layers output $H_t^{(l)}$ and $H_p^{(l)}$, respectively, and their discrepancy is minimized via MSE loss: $\frac{1}{L}\sum_{l=1}^{L}\frac{||H_t^{(l)} - H_p^{(l)}||_2}{x \cdot h}$. **Stage II (End-to-end Training)**—The entire model is fine-tuned end-to-end using NTP (cross-entropy) or distillation (KL divergence) to fix accumulated inter-layer errors.
    - **Design Motivation**: Stage I ensures the outputs of individual layers remain close under isolated conditions, but accumulated minor perturbations across deep LLM layers can still explode. Thus, Stage II end-to-end training is essential. The combination of both stages yields faster convergence and better results than direct end-to-end training alone.

3. **Frequency-aware RoPE**:

    - **Function**: Modifying the frequency sampling strategy of RoPE on low-dimensional Q/K to eliminate high-frequency noise.
    - **Mechanism**: In original RoPE, as the dimension gets smaller, the frequency component $\theta_j = \theta^{-(j-1)/\delta}$ of lower-indexed channels (high-frequency rotation) has an oscillation period shorter than the sampling interval, which causes numerical approximation failure. The modified sampling formula skips the highest frequency components and densifies the low-frequency sampling: $\theta_j = \theta^{-2(j-1+d/8)/d}$ (first half of the channels) and $\theta_j = \theta^{-(j-1+3d/4)/d}$ (second half of the channels). This maintains a decay behavior even when the dimension is reduced to 16.
    - **Design Motivation**: Starting from the stability analysis of $\text{RoPE}_{\theta,d}(x) = \mathbb{1}_d \cdot \mathcal{R}_{\theta,d/2}(x) \cdot \mathbb{1}_d^\top$, the authors discover that when $d < 32$, the autocorrelation function exhibits many negative values (meaning attention scores of identical vectors at far distances are even lower than random vectors). The root cause is that the high-frequency components $\cos(\theta^p)$ oscillate heavily when $p$ is large, and low-dimensional sampling points are insufficient to approximate the integral.

### Loss & Training

- Stage I: MSE loss for aligning intra-layer hidden states.
- Stage II: Two options—NTP with cross-entropy loss (resource-efficient) or distillation with KL divergence loss (richer information but requires an extra forward pass).
- Experiments show that NTP training yields better results than distillation in low-data regimes, as distillation typically requires more data to show its advantages.
- FFN layers use LoRA (Up/Down/Gate), where the rank has minimal footprint on PPL (only a 0.04 difference between rank 16 and 256).

## Key Experimental Results

### Main Results

| Model | $d_{qk}$ | $d_{vo}$ | Method | MMLU | OBQA | ARC | Avg | KV Cache↓ | TTFT↓ |
|------|-----------|-----------|------|------|------|-----|-----|-----------|-------|
| LLaMA3-8B | 128 | 128 | Base | 35.3 | 35.5 | 55.5 | 42.1 | - | - |
| LLaMA3-8B | 64 | 64 | Train | 35.0 | 35.1 | 53.8 | 41.3 | ↓50% | ↓8% |
| LLaMA3-8B | 64 | 64 | Distill | 31.0 | 29.1 | 39.1 | 33.1 | ↓50% | ↓8% |
| LLaMA3-8B | 16 | 16 | Train | 31.0 | 29.5 | 38.5 | 33.0 | ↓87% | ↓13% |
| LLaMA2-7B | 128 | 128 | Base | 28.9 | 29.4 | 30.7 | 29.7 | - | - |
| LLaMA2-7B | 64 | 64 | Train | 28.1 | 29.3 | 27.5 | 28.3 | ↓50% | ↓17% |

### Ablation Study

| $d_{qk}$ | $d_{vo}$ | LogPPL | KV Cache(MB) | Max Context (60GB) |
|-----------|----------|--------|-------------|--------------|
| 128 | 128 | baseline | 256 | 0.40M tokens |
| 64 | 128 | 2.47 | 172 | 0.61M tokens |
| 128 | 64 | 2.80 | 172 | 0.61M tokens |
| 64 | 64 | 2.74 | 128 | 0.81M tokens |
| 16 | 16 | 3.78 | 32 | 3.27M tokens |

### Key Findings

- **$d_{vo}$ is more crucial than $d_{qk}$**: Compressing to 172MB, retaining a larger $d_{vo}$ (PPL=2.47) is far better than a larger $d_{qk}$ (PPL=2.80), suggesting that Value holds more incompressible information than Key.
- **NTP training outperforms distillation**: With only 1B tokens of training data, the NTP approach (Avg=41.3) significantly outperforms distillation (Avg=33.1). Distillation needs more data to demonstrate its benefits.
- **GQA models are harder to compress**: Performance degradation in GQA-equipped LLaMA3 is more pronounced under the same compression ratio compared to MHA-equipped LLaMA2, because GQA already implements head-level compression.
- **Orthogonal to token-level methods**: KV-Latent + PyramidInfer (at 50% compression) can be stacked, yielding an extra 50% KV Cache reduction while PPL only shifts from 2.509 to 2.499.
- **Insensitive to LoRA rank**: Changing the rank from 16 to 256 results in a LogPPL change of only 0.04.

## Highlights & Insights

- **Dimension-level compression introduces a new paradigm**: It is fully orthogonal to head-level (GQA), layer-level (CLA), and token-level (eviction) compression, and can be stacked, opening a fresh dimension for compression.
- **Elegant theoretical analysis of frequency-aware RoPE**: The stability problem of RoPE is transformed into a numerical integration approximation challenge—where $\cos(\theta^p)$ oscillates wildy at large $p$ regions and low-dimensional sampling points are insufficient, prompting the design to skip high frequencies and densify low ones. This insight is valuable for all scenarios utilizing low-dimensional RoPE.
- **Asymmetry of Information between Key and Value**: This finding offers inspiration for model architecture designs—allocating more dimensions to Value and aggressively compressing Key in future architectures.

## Limitations & Future Work

- Validated only on 7B/8B scales; scalability to larger models (70B+) remains unknown.
- Trained on only 1B tokens; whether distillation surpasses NTP with more data needs further exploration.
- Missing comparisons with CLA (cross-layer attention), as CLA requires pre-training from scratch.
- Does not validate the impact during SFT/RLHF stages.
- Performance drops significantly at $d_{qk}=d_{vo}=16$ (NIH is only 6%), indicating a lower bound of compression.
- SVD initialization is unusable due to the non-commutativety of RoPE, though workarounds might exist.

## Related Work & Insights

- **vs GQA/MQA**: GQA shares KV across head numbers, whereas KV-Latent compresses the internal dimensions of each head. They are orthogonal and stackable.
- **vs DeepSeek-V2 (MLA)**: MLA maps KV to a joint latent space and then decodes, which is conceptually similar but requires pre-training from scratch. KV-Latent holds the advantage of starting from pre-trained models with minimal extra fine-tuning.
- **vs Token-level methods (H2O, PyramidInfer)**: These methods dynamically drop token KV states, while KV-Latent compresses the saved dimension of each token. They are orthogonal and experimentally proved to be stackable.

## Rating

- Novelty: ⭐⭐⭐⭐ Dimension-level KV Cache compression is a relatively new direction, though the overall idea (low-rank + distillation recovery) is expected.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation studies (independent analysis of QK/VO, LoRA rank, compatibility with other methods), but tested model scales are small.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations of the RoPE frequency analysis, structured and neat.
- Value: ⭐⭐⭐⭐ Offers a new perspective on KV Cache compression, compatible with existing methods, with strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](spindlekv_layered_kv_cache.md)
- [\[ACL 2025\] RefreshKV: Updating Small KV Cache During Long-form Generation](refreshkv_updating_small_kv_cache_during_long-form_generation.md)
- [\[ICML 2026\] RKSC: Reasoning-Aware KV Cache Sharing and Confident Early Exit for Multi-Step LLM Inference](../../ICML2026/llm_efficiency/rksc_reasoning-aware_kv_cache_sharing_and_confident_early_exit_for_multi-step_ll.md)
- [\[ICLR 2026\] ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing](../../ICLR2026/llm_efficiency/rest-kv_robust_kv_cache_eviction_with_layer-wise_output_reconstruction_and_spati.md)
- [\[ACL 2025\] LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training](adaptive_grouped_pe_context_window.md)

</div>

<!-- RELATED:END -->
