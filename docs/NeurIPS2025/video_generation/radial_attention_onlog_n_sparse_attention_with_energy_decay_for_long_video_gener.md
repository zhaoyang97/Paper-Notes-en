---
title: >-
  [Paper Note] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation
description: >-
  [NeurIPS 2025][Video Generation][Sparse Attention] Radial Attention identifies a "spatiotemporal energy decay" phenomenon in video diffusion models, wherein attention scores decay exponentially with spatiotemporal distance. Based on this finding, the authors design a static sparse attention mask with O(n log n) complexity, achieving up to 3.7× inference speedup on models such as HunyuanVideo and Wan2.1, and enabling 4× longer video generation via LoRA fine-tuning.
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Sparse Attention"
  - "Spatiotemporal Energy Decay"
  - "O(n log n)"
  - "Long Video Generation"
  - "LoRA Fine-tuning"
date: 2026-05-08
content_hash: 4bdd8de70d0532e8
---

# Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation

**Conference**: NeurIPS 2025
**arXiv**: [2506.19852](https://arxiv.org/abs/2506.19852)  
**Code**: [https://github.com/mit-han-lab/radial-attention](https://github.com/mit-han-lab/radial-attention)  
**Area**: Efficient Video Generation / Sparse Attention
**Keywords**: Sparse Attention, Spatiotemporal Energy Decay, O(n log n), Long Video Generation, LoRA Fine-tuning

## TL;DR
Radial Attention identifies a "spatiotemporal energy decay" phenomenon in video diffusion models, wherein attention scores decay exponentially with spatiotemporal distance. Based on this finding, the authors design a static sparse attention mask with O(n log n) complexity, achieving up to 3.7× inference speedup on models such as HunyuanVideo and Wan2.1, and enabling 4× longer video generation via LoRA fine-tuning.

## Background & Motivation
Attention computation in video diffusion models is a fundamental bottleneck:

**Quadratic complexity does not scale**: The O(n²) complexity of 3D dense attention makes long video training and inference extremely costly. Generating a 5-second 720p video with HunyuanVideo requires approximately 110K tokens.

**Limitations of existing sparse methods**:
   - SVG dynamically selects spatial/temporal attention heads, but profiling at inference time may misclassify heads and is incompatible with training.
   - STA employs a fixed 3D sliding window, but the fixed receptive field limits long-range dependencies.
   - Linear attention requires substantial architectural modifications, making quality recovery through lightweight fine-tuning difficult.

**Demand for long video generation**: Pretrained models are limited to short videos (5 seconds), while full-parameter training on long videos is prohibitively expensive.

Core insight: The decay of attention scores is analogous to the energy attenuation of signals or waves propagating over distance in physics — the greater the spatiotemporal distance, the lower the attention score. This motivates a sparse strategy that translates energy decay into computational density decay.

## Method

### Overall Architecture
Radial Attention is a static sparse attention mechanism that replaces dense attention with a predefined mask. The core design principle is that each token attends to spatially proximate tokens, with the attention window shrinking exponentially as temporal distance increases. The mask is directly compatible with the block-sparse implementation of FlashAttention.

### Key Designs

1. **Spatiotemporal Energy Decay Phenomenon**

    - Empirical observation: In HunyuanVideo, post-softmax attention scores decay as the spatial and temporal distance between tokens increases.
    - Decay model: $p_{js+l} \leq C_{rel} e^{-\alpha|j-i_0| - \beta|l-k_0|} p_{i_0 s+k_0}$
    - Parameter $\alpha$ governs temporal decay; $\beta$ governs spatial decay.
    - "Spatial attention" heads (as defined by SVG) exhibit high temporal decay and low spatial decay.
    - "Temporal attention" heads exhibit high spatial decay and low temporal decay.
    - Regression analysis confirms that the decay follows an approximately exponential distribution.

2. **Radial Attention Mask Design**

    - **Temporal density decay**: The computational density between frames $i$ and $j$ is $(1/2)^{\lfloor \log_2(\max(|i-j|, 1)) \rfloor}$.
    - This yields $2\lceil \log_2 f \rceil - 1$ diagonal bands, where the central band retains 100% density and each outer band halves the density.
    - **Spatial density decay**: The diagonal width from frame $i$ to frame $j$ is $\lfloor s / 2^{\lfloor \log_2 \max(|i-j|,1) \rfloor} \rfloor$.
    - When the diagonal width falls below 1, the diagonal frequency is reduced rather than further shrinking the width.
    - **Attention sink**: The first frame retains full attention, as its attention is critical to generation quality.
    - The final mask is a static 4D tensor $\tilde{M} \in \{-\infty, 0\}^{f \times f \times s \times s}$.

3. **Complexity Analysis**

    - Upper bound on the number of zero elements in the mask: $4s^2 f \log_2 f = 4sn(\log_2 n - \log_2 s)$.
    - For long videos (fixed spatial resolution $s$, large $f$), the complexity is O(n log n).
    - For 4× longer videos (509-frame 720p HunyuanVideo), attention computation is reduced by 9×.

4. **Error Analysis**

    - L1 attention error bound: $O(C_{rel} e^{-\min(\beta/2, \alpha)s})$.
    - The error decays exponentially with increasing decay rates $\alpha$, $\beta$ and spatial resolution $s$.
    - Empirically, Radial Attention incurs smaller approximation error than SVG.

5. **LoRA Adaptation for Long Video**

    - Key insight: Radial Attention only prunes unimportant token relationships while preserving the softmax attention mechanism, allowing pretrained weights to remain largely applicable.
    - Only lightweight LoRA fine-tuning on Q/K/V/O projections is required.
    - Empirically, LoRA + Radial Attention outperforms full-parameter fine-tuning, as LoRA concentrates updates on the most critical parameters.
    - The length-extension LoRA is compatible with existing style LoRAs.

### Loss & Training
- The original diffusion training objective is used directly.
- LoRA fine-tuning: rank 32, applied to Q/K/V/O projections of all attention layers.
- Hardware-friendly: implemented with 128×128 block sparsity, compatible with FlashAttention.

## Key Experimental Results

### Main Results (Default Video Length)

| Model | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Vision Reward↑ | Speedup |
|-------|--------|-------|-------|--------|----------------|---------|
| HunyuanVideo | Original | - | - | - | 0.141 | 1.0× |
| HunyuanVideo | STA | 26.7 | 0.866 | 0.167 | 0.132 | 2.29× |
| HunyuanVideo | SVG | 27.2 | 0.895 | 0.114 | 0.144 | 1.90× |
| HunyuanVideo | **Radial** | **27.3** | 0.886 | **0.114** | 0.139 | **1.88×** |
| Wan2.1-14B | Original | - | - | - | 0.136 | 1.0× |
| Wan2.1-14B | STA | 22.9 | 0.830 | 0.171 | 0.132 | 2.01× |
| Wan2.1-14B | **Radial** | **23.9** | **0.842** | **0.163** | 0.128 | 1.77× |

### Long Video Extension Results (HunyuanVideo)

| Scale | Method | Sparsity | Train Speedup | Infer Speedup | Vision Reward↑ |
|-------|--------|----------|---------------|---------------|----------------|
| 2× (253 frames) | Original | 0% | - | 1.0× | 0.122 |
| 2× | RIFLEx | 0% | - | 1.0× | 0.128 |
| 2× | Full Fine-tuning | 0% | 1.0× | 1.0× | 0.124 |
| 2× | **Radial+LoRA** | **80.8%** | **2.78×** | **2.35×** | **0.126** |
| 4× (509 frames) | Original | 0% | - | 1.0× | 0.054 |
| 4× | Full Fine-tuning | 0% | 1.0× | 1.0× | 0.133 |
| 4× | **Radial+LoRA** | **88.3%** | **4.37×** | **3.71×** | **0.134** |

### Key Findings
- Between 80–88% of attention computation can be safely skipped with negligible loss in video quality.
- Radial+LoRA achieves a Vision Reward of 0.134 on 4× longer videos, marginally surpassing full-parameter dense fine-tuning (0.133), while reducing training cost by 4.37×.
- RIFLEx (a training-free method) achieves only 0.037 Vision Reward at 4× length, indicating severe quality degradation.
- Unifying SVG's separate spatial/temporal attention patterns into a single Radial pattern eliminates the instability introduced by dynamic head classification.
- Performance across VBench dimensions — including subject consistency, aesthetic quality, and imaging quality — remains stable.

## Highlights & Insights
- "Spatiotemporal energy decay" is an elegant finding grounded in physical intuition, drawing an analogy between signal attenuation in the physical world and attention mechanisms.
- The static mask design is both simple and effective: it requires no runtime profiling, no architectural modifications, and can be used for both training and inference.
- The O(n log n) complexity offers a favorable trade-off between the O(n²) dense attention and O(n) linear attention.
- The observation that LoRA fine-tuning outperforms full-parameter fine-tuning is surprising, suggesting that lightweight adaptation can more precisely update the most critical parameters.
- Consistent performance across three models (Mochi 1, HunyuanVideo, and Wan2.1) demonstrates the generality of the approach.

## Limitations & Future Work
- The decay parameters $\alpha$ and $\beta$ may vary across models; the current formulation assumes a uniform decay pattern.
- The 128×128 block-sparse granularity may be insufficiently fine-grained for low-resolution videos.
- Narrative consistency and motion coherence in long videos remain open challenges.
- Integration with newer hardware acceleration schemes such as FlashAttention 4 has yet to be explored.

## Related Work & Insights
- This work unifies the separate spatial and temporal attention patterns in SVG, eliminating the need for manual decisions in pattern classification.
- Among O(n log n) complexity methods (Reformer, H-Transformer-1D, PowerAttention), Radial Attention offers the strongest physical motivation and is the most hardware-friendly.
- LoRA-based long video adaptation provides a low-cost pathway for extending the generation length of video diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[NeurIPS 2025\] VSA: Faster Video Diffusion with Trainable Sparse Attention](vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [\[ICLR 2026\] MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation](../../ICLR2026/video_generation/moga_mixture-of-groups_attention_for_end-to-end_long_video_generation.md)
- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](../../ICML2026/video_generation/veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](../../ICML2026/video_generation/dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)

</div>

<!-- RELATED:END -->
