---
title: >-
  [Paper Note] Linearizing Vision Transformer with Test-Time Training
description: >-
  [ICML 2026][Image Generation][Test-Time Training] The authors observe that the inner model of two-layer TTT is structurally equivalent to Softmax attention (Softmax can be viewed as a two-layer dynamic MLP). This enables…
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
The authors observe that the inner model of two-layer TTT is structurally equivalent to Softmax attention (Softmax can be viewed as a two-layer dynamic MLP). This enables direct inheritance of all Q/K/V/MLP weights. Key Instance Normalization is used to handle shift-invariance, and depthwise conv on Q/K is added to inject locality. With only 1 hour of fine-tuning, Stable Diffusion 3.5 is linearized and accelerated by 1.32×–1.47×.

## Background & Motivation

**Background**: Softmax attention in Vision Transformers is the de facto standard for vision foundation models (DiT, SD3.5, ViT), but its $\mathcal{O}(N^2)$ complexity makes long-sequence inference expensive. Many linear-complexity alternatives have been proposed—kernel approximations (Performer/Linear Attention), state space models (Mamba), TTT, etc.—but training a large model from scratch is costly. Industry prefers "zero-cost attention replacement on pretrained Softmax models."

**Limitations of Prior Work**: (1) Hedgehog / LoLCATs can only inherit part of the weights (MLP); Q/K activations must be relearned. (2) CLEAR restricts Softmax to local windows, losing global modeling. (3) LiT only inherits MLP. (4) Diffusion Grafting requires multi-stage fine-tuning. None achieve "full weight inheritance + short fine-tuning."

**Key Challenge**: Softmax attention is mathematically equivalent to a two-layer MLP $\sigma(qK^\top)V$ dynamically constructed from $K, V$, while standard linear attention only represents a single-layer dynamic linear transformation $\phi(q)(\phi(K)^\top V)$—a significant gap in expressiveness. Even with weight transfer, the target space cannot accommodate the source, so inheritance fails.

**Goal**: (i) Find a linear-complexity structure that can truly "accommodate" Softmax attention; (ii) Align Softmax's two key properties (shift-invariance, locality) at the representation level; (iii) Validate full linearization on DiT and SD3.5.

**Key Insight**: The authors note that if the inner model of TTT uses a two-layer MLP $f_W(x) = \sigma(xW_1)W_2$, with fast weights $W_1' = W_1 - \Delta_1, W_2' = W_2 - \Delta_2$ learned from the input sequence, the output $\mathrm{TTT}(q) = \sigma(qW_1')W_2'$ matches the "dynamic two-layer MLP" form of Softmax—this is a structural, not just symbolic, equivalence.

**Core Idea**: Use two-layer TTT (specifically TTT-SwiGLU) as a linear-complexity substitute, directly reusing all Softmax Q/K/V/MLP weights. Key Instance Normalization simulates Softmax's constant shift absorption, and depthwise conv injects locality. Only 1 hour of post-training is needed to recover original model performance.

## Method

### Overall Architecture
For a pretrained Softmax model, only the attention module is replaced. Specifically: (1) Apply instance norm to K: $\hat{k}_i = (k_i-\bar{k})/\sqrt{\frac{1}{N}\sum_j(k_j-\bar{k})^2+\varepsilon}$; (2) Add a depthwise conv branch to Q and K: $\hat{q} = q + \mathrm{DWC}(q), \hat{k} = k + \mathrm{DWC}(k)$; (3) Replace Softmax attention with a two-layer TTT-SwiGLU inner model, with $W_1, W_2$ directly inherited from Q/K/V projection matrices; (4) Optionally, mix with Neighborhood Attention (NAT) 50/50. All other MLP, LayerNorm, and embedding layers are retained and participate in fine-tuning.

### Key Designs

1. **TTT as a "Structurally Isomorphic" Substitute for Softmax**:

    - **Function**: Aligns Softmax attention $\mathrm{Attn}(q,K,V) = \sigma(qW_1^{dyn})W_2^{dyn}$ (where $W_1^{dyn} = K^\top, W_2^{dyn} = V$) with two-layer TTT $\mathrm{TTT}(q) = \sigma(qW_1')W_2'$ in representational capacity, enabling full weight inheritance.
    - **Mechanism**: From the query's perspective, Softmax attention is essentially a two-layer dynamic MLP—the first layer's weights are $K^\top$ (dynamically constructed from the sequence), the intermediate nonlinearity is row-wise softmax, and the second layer is $V$. In TTT, the two-layer MLP's "static weights" $W_1, W_2$ plus fast weights $\Delta_1, \Delta_2$ computed from sequence gradients yield $W_1', W_2'$, structurally identical to the above. Controlled experiments (Table 1) show that vanilla linear attention, even with ProjQK under freeze protocol, achieves only 24.39% acc, while TTT-SwiGLU reaches 67.33%—structural matching yields far greater transfer than activation replacement.
    - **Design Motivation**: Previous linear attention compresses "dynamic weights" into a single layer $\phi(K)^\top V$, losing the nonlinear intermediate layer of Softmax. TTT preserves the "nonlinearity + two layers" structure, making it a truly linear architecture capable of accommodating the Softmax representational space.

2. **Key Instance Normalization for Shift-Invariance Alignment**:

    - **Function**: Eliminates systematic bias in pretrained model keys, stabilizing TTT internal optimization.
    - **Mechanism**: Softmax is insensitive to constant shift $\delta$ in $K$—subtracting $q^\top\delta$ from both numerator and denominator leaves softmax unchanged. However, TTT's internal loss $\mathcal{L}_t(k_t) = -v_t^\top f_W(k_t)$ is highly sensitive to $\delta$; first-order expansion of the gradient w.r.t. $W_1$ yields extra terms like $-[W_2^\top v_t \odot \sigma'(W_1 k_t)]\delta^\top$, which accumulate online and cause gradient explosion. The authors define shift ratio $r = \|\bar{k}\|_2 / (\frac{1}{N}\sum_i \|k_i\|_2)$; in pretrained ViT, $r \approx 0.5$, while random initialization yields only 0.07, confirming systematic bias. The solution is to apply Instance Norm to K before TTT: $\hat{k}_i = (k_i-\bar{k})/\sqrt{\mathrm{var}+\varepsilon}$. Table 2 shows that removing mean subtraction causes immediate NaN during training, while removing std division has little effect—proving that "centering" is key, not "standardization."
    - **Design Motivation**: This is the most critical and easily overlooked aspect of representation alignment—Softmax implicitly "absorbs" constant bias in K, so pretrained K need not be centered. TTT, however, optimizes explicitly; uncentered K leads to failure, so invariance must be manually restored.

3. **Depthwise Conv on Q/K to Inject Locality**:

    - **Function**: Compensates for the strong local bias of Softmax, which TTT lacks, enhancing fine-grained modeling.
    - **Mechanism**: Since TTT lacks explicit $QK^\top$, the authors define implicit attention $A_{implicit}(i,j) = \partial o_i/\partial v_j$ for visualization, finding that TTT is more global and lacks local spikes compared to Softmax. The fix is to add depthwise conv residuals to Q/K: $\hat{q} = q + \mathrm{DWC}(q), \hat{k} = k + \mathrm{DWC}(k)$. This allows the TTT internal learning target $L(f_W(k), v)$ to "jointly predict v from keys within a local window," naturally expanding the receptive field. Table 3 shows DWCQK outperforms adding CPE to input or DWC to value; further mixing with NAT3/NAT5 yields additional gains.
    - **Design Motivation**: Locality is Softmax's implicit inductive bias in vision; linear/TTT models are inherently weaker on local textures. Depthwise conv is the cheapest locality injector—0.5M parameters can recover 2% acc.

### Loss & Training
Two fine-tuning protocols: (1) Freeze protocol—train only newly introduced TTT internal parameters and DWC weights, with a large learning rate (for structural validation); (2) Full Fine-Tuning (FT)—all parameters are trained. On SD3.5, only 3000 steps of fine-tuning (4×H20, about 1 hour) are performed, using standard rectified flow loss + EMA teacher alignment. On DiT-XL/2, 8 epochs are run, only 0.57% of original training steps.

## Key Experimental Results

### Main Results (ImageNet Classification, fine-tuning after weight inheritance, all TTT with InstanceNorm)

| Model | New Params | Freeze acc | FT acc | FLOPs |
|-------|------------|------------|--------|-------|
| Softmax (original) | — | 72.05 | — | 1.25G |
| Linear Attn | 0 | 3.71 | 63.30 | 1.13G |
| Linear + ProjQK | 0.3M | 24.39 | 66.23 | 1.19G |
| TTT-1Layer-Gate | 0.3M | 61.95 | 67.59 | 1.25G |
| TTT-2Layer | 0.3M | 65.98 | 68.14 | 1.25G |
| TTT-3Layer | 0.5M | 67.09 | 68.93 | 1.37G |
| **TTT-SwiGLU** | 0.5M | **67.33** | **69.25** | 1.34G |

| Large Model Exp. | Setting | Speedup | Performance |
|------------------|---------|---------|-------------|
| DiT-XL/2 | 8 epochs only (0.57% of original training) | — | On par with Softmax |
| SD3.5-T5 (1K) | 3000 steps fine-tuning | 1.32× | Close to fine-tuned Softmax |
| SD3.5-T5 (2K) | Same as above | 1.47× | Close to fine-tuned Softmax |

### Ablation Study

| Normalization Strategy | Stable | Acc | Note |
|-----------------------|--------|-----|------|
| None | ✗ | 0.37 | Diverges immediately |
| RMSNorm | ✗ | 57.38 | Token-level, cannot remove key shift |
| LayerNorm | ✗ | 57.25 | Same as above |
| **InstanceNorm** (Ours) | ✓ | **71.19** | Cross-token centering, matches shift-invariance |
| InstanceNorm w/o ÷std | ✓ | 71.15 | Std scaling almost irrelevant |
| InstanceNorm w/o mean sub. | ✗ | 51.43 | Mean subtraction essential, otherwise NaN |

| Locality Enhancement | Acc | Params | FLOPs |
|---------------------|-----|--------|-------|
| TTT (no locality) | 69.25 | 6.2M | 1.34G |
| + CPE on input | 69.64 | 6.2M | 1.34G |
| + DWC on Value | 70.47 | 6.2M | 1.34G |
| **+ DWCQK** (Ours) | **71.19** | 6.2M | 1.34G |
| + DWCQK + NAT3 | 71.67 | 6.2M | 1.36G |
| + DWCQK + NAT5 | 72.06 | 6.2M | 1.39G |

### Key Findings
- **Structural matching is an order of magnitude more important than activation replacement**: Linear + ProjQK under Freeze achieves only 24.39%, while TTT-SwiGLU reaches 67.33%—with the same 0.3–0.5M new parameters, structural alignment yields vastly superior transfer.
- **TTT nonlinearity depth has diminishing returns**: Freeze acc improves from 61.95→65.98→67.09 for 1→2→3 layers, but 3 layers only slightly outperform SwiGLU (2 layers), indicating two layers suffice to approximate Softmax, and more layers only increase FLOPs.
- **InstanceNorm must remove mean, std is optional**: Directly validates the theoretical analysis that "key shift is the mathematical root cause"—the most insightful diagnostic experiment in the paper.
- **NAT is a bonus, not a necessity**: Unlike methods like Hedgehog, CLEAR that heavily rely on local windows, DWCQK alone achieves 71.19%; NAT is only an optional enhancement—showing that structural and representational alignment are core, local windows are just a patch.

## Highlights & Insights
- **"Softmax = two-layer dynamic MLP" is the core insight**: While not entirely original (see kristiadi et al.), the authors' implementation—"TTT inner model with two layers can accommodate Softmax"—is an elegant engineering mapping.
- **Shift-invariance diagnostic is generalizable**: Defining $r = \|\bar{k}\|/\mathrm{avg}\|k_i\|$ as a simple ratio to quantify "model sensitivity to implicit invariances" can be extended to other transfer learning diagnostics (e.g., RMSNorm vs LayerNorm).
- **Implicit attention via gradient is a general tool for measuring locality**: $A_{implicit} = \partial o/\partial v$ applies to any model without explicit attention maps (SSM, TTT, RNN), useful for interpreting and visualizing sub-quadratic architectures.
- **Training cost data is impressive**: SD3.5 in one hour, DiT-XL at 0.57% steps—making "linearization" nearly free, with direct value for industrial deployment.

## Limitations & Future Work
- Experiments are mainly on vision tasks (ViT, DiT, SD3.5); it is unverified whether Llama-like large LMs can be linearized with 1-hour fine-tuning.
- TTT's fast weight updates incur extra computational overhead; the 1.32×/1.47× speedup is mainly at 1K–2K resolution, and at lower resolutions, TTT's overhead may outweigh Softmax.
- KV cache handling during inference is not discussed—efficiently caching TTT inner model states is crucial for autoregressive generation.
- DWCQK is friendly to 16×16 patches, but for other patch sizes (e.g., 3D patches in video), conv kernel design needs to be revisited.
- Only DiT-XL/2 + SD3.5 are reported; scalability to larger models like SD3.5-Large, Flux, etc., is left for future work.

## Related Work & Insights
- **vs Hedgehog / LoLCATs**: They use learnable Q/K activations to approximate Softmax but remain within the single-layer linear attention framework, unable to bridge the representational gap; this work replaces the "kernel" with TTT.
- **vs CLEAR**: CLEAR retains Softmax in local windows but restricts global modeling—this work uses global TTT + local DWC, offering more flexibility.
- **vs LiT**: LiT only inherits MLP; this work achieves "full weight inheritance," improving transfer efficiency by orders of magnitude.
- **vs Diffusion Grafting**: Grafting emphasizes multi-stage fine-tuning, while this work emphasizes "finding the right architecture + aligning representations"—the two are orthogonal and can be combined.
- **vs ViT3 (Han 2025)**: Also in vision TTT, this work focuses on "Softmax to TTT," while ViT3 focuses on "designing TTT vision backbones from scratch"; the choice of TTT inner model differs (SwiGLU is superior here).

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of "TTT structurally isomorphic to Softmax" and Instance Norm for shift-invariance are both novel; however, linearizing Transformers is a hot topic with dense related work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers ImageNet classification, DiT-XL/2, SD3.5; ablations thoroughly test normalization, locality, and structural choices; lacks NLP tasks and larger model extensions.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative—structural alignment → representational alignment → experimental validation; formula derivations (especially shift gradient expansion) are lucid.
- Value: ⭐⭐⭐⭐ Provides industry with a "1-hour SD3.5 linearization" solution and reveals the true use of TTT in vision.

## Related Papers

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[CVPR 2025\] LaVin-DiT: Large Vision Diffusion Transformer](../../CVPR2025/image_generation/lavin-dit_large_vision_diffusion_transformer.md)
- [\[CVPR 2026\] Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling](../../CVPR2026/image_generation/test-time_instance-specific_parameter_composition_a_new_paradigm_for_adaptive_ge.md)
- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](../../ICLR2026/image_generation/vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)

</div>

<!-- RELATED:END -->

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
