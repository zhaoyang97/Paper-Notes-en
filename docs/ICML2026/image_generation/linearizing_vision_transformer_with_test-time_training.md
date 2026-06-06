---
title: >-
  [Paper Note] Linearizing Vision Transformer with Test-Time Training
description: >-
  [ICML 2026][Image Generation][Test-Time Training] The authors discovered that a two-layer TTT inner model is structurally equivalent to Softmax attention (where Softmax acts as a two-layer dynamic MLP). This observation…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Test-Time Training"
  - "Linear Attention"
  - "Weight Inheritance"
  - "Instance Normalization"
  - "DiT Acceleration"
date: 2026-05-08
content_hash: 78818bd973d252db
---

# Linearizing Vision Transformer with Test-Time Training

**Conference**: ICML 2026  
**arXiv**: [2605.02772](https://arxiv.org/abs/2605.02772)  
**Code**: Not yet released  
**Area**: Image Generation / Vision Transformer / Linear Attention / Stable Diffusion Acceleration  
**Keywords**: Test-Time Training, Linear Attention, Weight Inheritance, Instance Normalization, DiT Acceleration

## TL;DR
The authors discovered that a two-layer TTT inner model is structurally equivalent to Softmax attention (where Softmax acts as a two-layer dynamic MLP). This observation enables direct inheritance of all weights from Q/K/V/MLP. By incorporating key Instance Normalization to handle shift-invariance and depthwise convolutions on Q/K to restore locality, Stable Diffusion 3.5 was linearized and accelerated by 1.32×–1.47× with only one hour of fine-tuning.

## Background & Motivation

**Background**: Softmax attention in Vision Transformers is the de facto standard for vision foundation models (DiT, SD3.5, ViT), but its $\mathcal{O}(N^2)$ complexity makes inference on long sequences expensive. Numerous linear-complexity alternatives have been proposed—kernel approximations (Performer/Linear Attention), State Space Models (Mamba), and TTT—but the cost of training large models from scratch is prohibitive. The industry demands solutions that "replace attention in pre-trained Softmax models with zero-cost."

**Limitations of Prior Work**: (1) Hedgehog and LoLCATs only inherit partial weights (MLP), requiring Q/K to relearn activations; (2) CLEAR restricts Softmax to local windows, sacrificing global modeling; (3) LiT only inherits the MLP; (4) Diffusion Grafting requires multi-stage fine-tuning. None of these achieve "full weight inheritance + minimal fine-tuning."

**Key Challenge**: Softmax attention is mathematically equivalent to a two-layer MLP $\sigma(qK^\top)V$ constructed dynamically using $K$ and $V$. In contrast, standard linear attention only represents a single-layer dynamic linear transformation $\phi(q)(\phi(K)^\top V)$, which is an order of magnitude lower in representational capacity. Even if weights are transferred, the target space cannot accommodate the source space, leading to inheritance failure.

**Goal**: (i) Identify a linear-complexity structure capable of "containing" Softmax attention; (ii) Align the representational space with Softmax properties (shift-invariance, locality); (iii) Validate complete linearization on DiT and SD3.5.

**Key Insight**: The authors noted that if the inner model of Test-Time Training (TTT) is a two-layer MLP $f_W(x) = \sigma(xW_1)W_2$, supplemented with fast weights $W_1' = W_1 - \Delta_1, W_2' = W_2 - \Delta_2$ learned from the input sequence, the resulting output $\mathrm{TTT}(q) = \sigma(qW_1')W_2'$ matches the "dynamic two-layer MLP" form of Softmax. This represents structural isomorphism rather than mere symbolic similarity.

**Core Idea**: Use a two-layer TTT (specifically TTT-SwiGLU) as a linear-complexity proxy to directly reuse pre-trained Q/K/V/MLP weights. Apply key Instance Normalization to simulate Softmax's constant shift absorption and use depthwise convolutions to inject locality. Performance is restored after only one hour of post-training.

## Method

### Overall Architecture
For a pre-trained Softmax model, the authors only replace the attention modules. Specific modifications include: (1) Applying Instance Normalization to $K$: $\hat{k}_i = (k_i-\bar{k})/\sqrt{\frac{1}{N}\sum_j(k_j-\bar{k})^2+\varepsilon}$; (2) Adding depthwise convolution (DWC) branches to $Q$ and $K$: $\hat{q} = q + \mathrm{DWC}(q), \hat{k} = k + \mathrm{DWC}(k)$; (3) Replacing Softmax attention with a two-layer TTT-SwiGLU inner model, where $W_1$ and $W_2$ are inherited directly from Q/K/V projection matrices; (4) Optionally mixing with Neighborhood Attention (NAT) at a 50/50 ratio. Other components like MLP, LayerNorm, and embeddings are preserved and participate in fine-tuning.

### Key Designs

1.  **TTT as a "Structural Isomorphism" for Softmax**:
    - **Function**: Aligns Softmax attention $\mathrm{Attn}(q,K,V) = \sigma(qW_1^{dyn})W_2^{dyn}$ (where $W_1^{dyn} = K^\top, W_2^{dyn} = V$) with two-layer TTT $\mathrm{TTT}(q) = \sigma(qW_1')W_2'$ in terms of representational capacity to support full weight inheritance.
    - **Mechanism**: From the perspective of the query, Softmax attention is essentially a two-layer dynamic MLP—the first layer weights are $K^\top$ (constructed dynamically), the intermediate non-linearity is row-wise softmax, and the second layer is $V$. In TTT, the static weights $W_1, W_2$ plus fast weights $\Delta_1, \Delta_2$ derived from sequence gradients yield $W_1', W_2'$, which perfectly mirrors this structure. Table 1 demonstrates that vanilla linear attention with ProjQK achieves only 24.39% accuracy under the freeze protocol, whereas TTT-SwiGLU reaches 67.33%, proving that structural matching yields superior migration benefits.
    - **Design Motivation**: Previous linear attention methods condensed "dynamic weights" into a single layer $\phi(K)^\top V$, losing the non-linear intermediate layer of Softmax. TTT preserves both the non-linearity and the two-layer depth.

2.  **Key Instance Normalization for Shift-Invariance**:
    - **Function**: Eliminates systematic offsets in pre-trained keys to stabilize TTT's internal optimization.
    - **Mechanism**: Softmax is insensitive to constant shifts $\delta$ in $K$ (subtracting $q^\top\delta$ from exponents cancels out). However, the TTT inner loss $\mathcal{L}_t(k_t) = -v_t^\top f_W(k_t)$ is extremely sensitive to $\delta$. The first-order gradient expansion for $W_1$ includes terms like $-[W_2^\top v_t \odot \sigma'(W_1 k_t)]\delta^\top$, leading to gradient explosion during online accumulation. The authors measured a shift ratio $r = \|\bar{k}\|_2 / (\frac{1}{N}\sum_i \|k_i\|_2) \approx 0.5$ in pre-trained ViTs compared to 0.07 in random initialization, confirming systematic bias. Applying Instance Norm $\hat{k}_i = (k_i-\bar{k})/\sqrt{\mathrm{var}+\varepsilon}$ solves this. Table 2 shows that removing mean subtraction causes divergence (NaN), while removing std division has negligible impact, identifying "centering" as the critical factor.
    - **Design Motivation**: This aligns the representational space properties; Softmax implicitly absorbs constant offsets, while TTT requires explicit centering to prevent collapse.

3.  **Depthwise Conv on Q/K for Locality**:
    - **Function**: Complements the strong local bias of Softmax, which TTT inherently lacks, to improve fine-grained modeling.
    - **Mechanism**: The authors defined implicit attention as $A_{implicit}(i,j) = \partial o_i/\partial v_j$. Visualizations showed TTT to be more global and lacking the local spikes of Softmax. This is fixed by adding residual DWC branches: $\hat{q} = q + \mathrm{DWC}(q), \hat{k} = k + \mathrm{DWC}(k)$. This encourages the TTT inner target $L(f_W(k), v)$ to perceive local window information. Table 3 shows DWCQK outperforms CPE on inputs or DWC on values.
    - **Design Motivation**: Locality is an implicit inductive bias in vision Softmax. DWC is an inexpensive locality injector, adding only 0.5M parameters to recover significant accuracy.

### Loss & Training
Two fine-tuning protocols were used: (1) Freeze Protocol—training only the new TTT internal parameters and DWC weights with a high learning rate for structural validation; (2) Full Fine-Tuning (FT)—training all parameters. For SD3.5, only 3000 steps of fine-tuning (~1 hour on 4×H20) were performed using standard rectified flow loss and EMA teacher alignment. For DiT-XL/2, 8 epochs were used, representing only 0.57% of the original training steps.

## Key Experimental Results

### Main Results (ImageNet Classification, Fine-tuning after Weight Inheritance)

| Model | New Params | Freeze acc | FT acc | FLOPs |
|------|--------|-----------|--------|-------|
| Softmax (Original) | — | 72.05 | — | 1.25G |
| Linear Attn | 0 | 3.71 | 63.30 | 1.13G |
| Linear + ProjQK | 0.3M | 24.39 | 66.23 | 1.19G |
| TTT-1Layer-Gate | 0.3M | 61.95 | 67.59 | 1.25G |
| TTT-2Layer | 0.3M | 65.98 | 68.14 | 1.25G |
| TTT-3Layer | 0.5M | 67.09 | 68.93 | 1.37G |
| **TTT-SwiGLU** | 0.5M | **67.33** | **69.25** | 1.34G |

| Large Model Exp | Setting | Acceleration | Performance |
|-----------|------|------|------|
| DiT-XL/2 | 8 epochs (0.57% of total) | — | Comparable to Softmax |
| SD3.5-T5 (1K) | 3000 steps FT | 1.32× | Close to FT Softmax |
| SD3.5-T5 (2K) | 3000 steps FT | 1.47× | Close to FT Softmax |

### Ablation Study

| Normalization Strategy | Stable | Acc | Notes |
|---------------------|------|------|------|
| None | ✗ | 0.37 | Immediate divergence |
| RMSNorm | ✗ | 57.38 | Token-level; fails to remove key shift |
| LayerNorm | ✗ | 57.25 | Same as above |
| **InstanceNorm** (Ours) | ✓ | **71.19** | Cross-token centering; matches shift-invariance |
| InstanceNorm w/o ÷std | ✓ | 71.15 | Std scaling is negligible |
| InstanceNorm w/o mean sub. | ✗ | 51.43 | Mean subtraction is mandatory; otherwise NaN |

| Locality Strategy | Acc | Params | FLOPs |
|--------------------|------|------|-------|
| TTT (no locality) | 69.25 | 6.2M | 1.34G |
| + CPE on input | 69.64 | 6.2M | 1.34G |
| + DWC on Value | 70.47 | 6.2M | 1.34G |
| **+ DWCQK** (Ours) | **71.19** | 6.2M | 1.34G |
| + DWCQK + NAT3 | 71.67 | 6.2M | 1.36G |
| + DWCQK + NAT5 | 72.06 | 6.2M | 1.39G |

### Key Findings
- **Structural matching is an order of magnitude more important than activation replacement**: Linear + ProjQK yields only 24.39% under the Freeze protocol, while TTT-SwiGLU reaches 67.33%. Structural alignment is the key to migration gains.
- **Marginal benefits from TTT non-linearity depth**: Accuracy improves from 1 to 2 to 3 layers (61.95→65.98→67.09), but 2-layer SwiGLU is sufficient and more FLOP-efficient.
- **InstanceNorm requires mean subtraction but not std scaling**: This confirms the theoretical analysis that key shift is the root mathematical cause.
- **NAT is an optional bonus**: Unlike methods like Hedgehog or CLEAR that rely heavily on local windows, DWCQK achieves 71.19% independently.

## Highlights & Insights
- **"Softmax = Dynamic Two-Layer MLP" is the pivotal insight**: While similar observations exist in literature (e.g., Kristiadi et al.), operationalizing it into a TTT inner model capable of "containing" Softmax is an elegant engineering mapping.
- **Shift-invariance diagnosis is a valuable heuristic**: Using the ratio $r = \|\bar{k}\|/\mathrm{avg}\|k_i\|$ to quantify implicit invariances could be extended to other transfer learning scenarios.
- **Implicit attention via gradient is a universal diagnostic**: $A_{implicit} = \partial o/\partial v$ allows for the visualization of locality in any sub-quadratic architecture (SSM, TTT, RNN) that lacks an explicit attention map.
- **Training efficiency is striking**: Linearizing SD3.5 in one hour or DiT-XL in 0.57% of training steps offers significant industrial value for deployment.

## Limitations & Future Work
- The experiments focus on vision tasks (ViT, DiT, SD3.5); it remains to be seen if TTT can linearize language models (e.g., Llama) with similar efficiency.
- TTT fast weight updates introduce computational overhead; the 1.32×–1.47× speedup is observed at 1K-2K resolutions but may diminish at lower resolutions.
- Handling of the KV cache (TTT inner states) during auto-regressive inference was not discussed in detail.
- DWCQK is optimized for 16×16 patches; other patch sizes (e.g., 3D patches in video) may require kernel redesign.
- Scalability to larger models like Flux or SD3.5-Large is left for future research.

## Related Work & Insights
- **vs Hedgehog / LoLCATs**: These use learnable Q/K activations to approximate Softmax within a single-layer framework, failing to bridge the capacity gap. This work replaces the "kernel" with TTT.
- **vs CLEAR**: CLEAR retains Softmax but restricts it to local windows. This work uses global TTT complemented by local DWC, providing greater flexibility.
- **vs LiT**: LiT only inherits the MLP, whereas this work achieves full weight inheritance, improving migration efficiency.
- **vs ViT3 (Han 2025)**: While both use vision TTT, this work focuses on "converting Softmax to TTT," whereas ViT3 focuses on "designing TTT backbones from scratch."

## Rating
- Novelty: ⭐⭐⭐⭐ The "TTT-Softmax isomorphism" and shift-invariance fix are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of ImageNet, DiT-XL/2, and SD3.5; extensive ablations. Lacks NLP tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow from structural/representation alignment to experimental validation.
- Value: ⭐⭐⭐⭐ Provides a practical "one-hour" solution for linearizing SD3.5 and elucidates TTT's role in vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[NeurIPS 2025\] Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials](../../NeurIPS2025/image_generation/linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)
- [\[CVPR 2026\] Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling](../../CVPR2026/image_generation/test-time_instance-specific_parameter_composition_a_new_paradigm_for_adaptive_ge.md)
- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](../../ICLR2026/image_generation/vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)

</div>

<!-- RELATED:END -->
