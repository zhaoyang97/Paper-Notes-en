---
title: >-
  [Paper Note] The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity
description: >-
  [ICML 2026][3D Vision][attention sink] This paper reveals the structural root of "attention sink" (attention pooling at the first token) in LLMs: the lack of value aggregation for the first token under causal masking lea…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "attention sink"
  - "variance discrepancy"
  - "super neurons"
  - "dimensional collapse"
  - "head-wise RMSNorm"
date: 2026-05-08
content_hash: c6a0b43e8b0bf186
---

# The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity

**Conference**: ICML 2026  
**arXiv**: [2605.06611](https://arxiv.org/abs/2605.06611)  
**Code**: None  
**Area**: Interpretability / Transformer Mechanism / LLM Optimization  
**Keywords**: attention sink, variance discrepancy, super neurons, dimensional collapse, head-wise RMSNorm

## TL;DR
This paper reveals the structural root of "attention sink" (attention pooling at the first token) in LLMs: the lack of value aggregation for the first token under causal masking leads to variance discrepancy. This is selectively amplified by "super neurons" in the FFN, forming extreme dimension disparity that eventually locks the QK projection and forces an attention sink. Ours proposes head-wise RMSNorm to suppress the sink at its root during pre-training.

## Background & Motivation

**Background**: Attention sink (where the first token in a decoder-only Transformer captures an disproportionately high attention score) is a universal phenomenon in GPT/LLaMA models. It has been utilized (StreamingLLM KV cache compression) but also criticized (causing activation outliers, representation collapse, and quantization difficulties). Past explanations include Softmax needing to "collect residual probability mass," position encoding artifacts, and spectral subspace issues (Xiao et al. 2023, Yan et al. 2024, Cancedda 2024).

**Limitations of Prior Work**: These explanations are either phenomenological (stating "Softmax needs a sink") or cover only partial cases (position encoding theories fail to explain why it appears suddenly at Layer 2 rather than Layer 0). None fully answer the chain of phenomena: **"Why specifically the first token, why at a specific layer, and why does the norm explode?"**

**Key Challenge**: The authors find that the onset of attention sink is a **structural invariant**. In Llama-2, regardless of input, the sink consistently appears at Layer 2, synchronized with a spike in the first token's $\ell_2$-norm. This indicates that the sink is not an emergent property but the inevitable result of a **deterministic causal chain** triggered at a fixed layer. However, the nature of this chain and whether it can be intervened upon remained unclear.

**Goal**: To fully map the three-stage causal chain from "token-level statistical differences → neuron-level amplification in FFN → locking of attention patterns" and verify causality through controlled experiments at each step.

**Key Insight**: Start with the **positional asymmetry** of value aggregation. Under a causal mask, the first token $i=0$ can only attend to itself ($a_{0,0}=1$), while subsequent tokens perform a convex combination of $i+1$ vectors. Variance naturally decays monotonically, making the first token a natural variance outlier. This simple observation is the starting point of the entire chain.

**Core Idea**: Attention sink = **Variance Discrepancy (from value aggregation) → Selective Activation of Super Neurons (FFN amplification) → Dimension Disparity (sparse channeling in down-projection) → QK Locking (RMSNorm projecting the first token to a fixed direction)**. Once this chain is understood, variance discrepancy can be suppressed at the source using head-wise RMSNorm.

## Method

### Overall Architecture
The authors perform "phenomenon diagnosis" (Sec 3) to prove Layer 2 onset and norm spikes; followed by "causal localization" (Sec 3.1-3.2) to prove value aggregation introduces positional variance discrepancy and replicate sinks at **arbitrary positions** via controlled interventions (masking, variance amplification). They then perform "propagation chain analysis" (Sec 4) to trace how variance discrepancy is preserved by $\mathbf{W}_O$, triggers super neurons, forms dimension disparity via sparse $\mathbf{W}_{\text{down}}$, and collapses into a single basis vector via RMSNorm to lock QK. Finally, "engineering intervention" (Sec 5) proposes head-wise RMSNorm to suppress variance discrepancy at the root.

### Key Designs

1.  **Diagnosis and Causal Verification of Variance Discrepancy**:
    *   **Function**: To locate the root cause of "why the first token" in the unique operation of value aggregation.
    *   **Mechanism**: Dimension-wise variance after Layer 1 value aggregation in Llama-2-7B (using random token sequences to exclude BOS bias) shows that Position 0 has far higher variance than other positions, decaying monotonically. Two **controlled interventions** verify causality: (a) **Mask intervention**—changing the mask of the $k$-th token to attend only to itself (simulating the first token's unaggregated state) immediately turns $k$ into a new sink. (b) **Variance amplification**—directly amplifying the variance of any token using $\mathbf{o}_k'^{(l)}=\boldsymbol{\mu}^{(l)}+\lambda\cdot(\mathbf{o}_k^{(l)}-\boldsymbol{\mu}^{(l)})$ for $\lambda>1$ also creates a new sink. A key control experiment: simply scaling the norm $\lambda\cdot \mathbf{o}_k$ **does not** replicate the sink, excluding the confounder that "large norm causes sinks."
    *   **Design Motivation**: This provides the causal anchor; without these experiments, the "variance discrepancy root cause" would only be a correlation.

2.  **Selective Activation of Super Neurons and Dimension Disparity**:
    *   **Function**: To explain how variance discrepancy is exponentially amplified into dimensional collapse by a **few specific neurons** in the FFN.
    *   **Mechanism**: In the SwiGLU FFN, $\mathbf{W}_{\text{gate}}$ and $\mathbf{W}_{\text{up}}$ contain "super neurons" with massive column norms (e.g., index 7890). These neurons show high cosine alignment with the first token's norm $\mathbf{x}_{\text{norm}}$ but nearly zero for others. Effectively, super neurons **"open the gate" only for the first token**. The corresponding row in $\mathbf{W}_{\text{down}}$ is **heavy-tailed and sparse**, channeling this massive activation into a few outlier dimensions (e.g., dim 2533). This is quantified by the Dominance Ratio $\text{DomRatio}(\mathbf{h}_0)=\max_j|\mathbf{h}_{0,j}|/(\frac{1}{d}\sum_k|\mathbf{h}_{0,k}|)$, which exceeds 200 in early Llama-2 layers.
    *   **Design Motivation**: Translates statistical variance discrepancy into parameter-level geometric collapse while explaining why the sink appears at a fixed layer (Layer 2) where learned super neurons are triggered.

3.  **RMSNorm Directional Collapse + QK Structural Locking**:
    *   **Function**: To explain why dimension disparity **necessarily** translates into the lock-in of QK attention scores for the first token.
    *   **Mechanism**: When $\mathbf{x}_0$ has an overwhelming value $\lambda$ in dimension $c$, the RMSNorm normalization constant is dominated by $\lambda$, and the output collapses to $\text{RMSNorm}(\mathbf{x}_0)\approx \text{sgn}(\lambda)\sqrt{d}\gamma_c\cdot \mathbf{e}_c$. After key projection, $\mathbf{k}_0^{(h)}\approx \pm\sqrt{d}\cdot (\mathbf{W}_K^{(h)})_{c,:}$ (the $c$-th row of $\mathbf{W}_K$). SVD shows high cosine alignment between the query matrix's principal direction and $\mathbf{k}_0^{(h)}$, leading to structurally positive QK dot products across all tokens for those heads.
    *   **Design Motivation**: Closes the causal chain with observable intermediate variables at every step.

### Loss & Training
**Head-wise RMSNorm Intervention** (Sec 5.1): Applies RMSNorm to every head output after value aggregation but before output projection $\mathbf{W}_O$: $\hat{\mathbf{o}}_t^{(h)}=\frac{\mathbf{o}_t^{(h)}}{\text{RMS}(\mathbf{o}_t^{(h)})}\odot \boldsymbol{\lambda}$, where $\boldsymbol{\lambda}\in\mathbb{R}^{d_k}$ is a learnable scaling vector shared across heads. This ensures (i) variance normalization across all positions and (ii) balanced contributions to $\mathbf{W}_O$ between low-entropy (high variance) and high-entropy (low variance) heads. Verified by pre-training a 152M parameter model on 20B tokens of OpenWebText.

## Key Experimental Results

### Main Results: Comparison of Three Architectures (Llama-2 config, 4 seeds avg)

| Metric | Baseline (Softmax) | Sigmoid Attention | **Ours (HeadNorm)** |
| :--- | :--- | :--- | :--- |
| Train Loss ↓ | 2.7483 ± 0.0118 | — | **2.7073 ± 0.0095** |
| Validation Loss ↓ | 2.7812 ± 0.0109 | (Slower/Higher) | **2.7421 ± 0.0066** |
| Effective Rank ↑ (Layer Avg) | 343.71 ± 15.63 | High | **445.96 ± 5.37** |
| Dimension Disparity ↓ (Layer Avg) | 82.67 ± 8.09 | Low | **33.74 ± 2.73** |
| Attention Sink Removed? | No (from Layer 5) | Yes | **Yes** |

### Ablation Study

| Experiment | Phenomenon | Conclusion |
| :--- | :--- | :--- |
| Mask block at $k=10$ | $k$ immediately becomes sink | Variance discrepancy is the causal start |
| Variance amplify $\lambda\uparrow$ at $k=10$ | Sink score rises monotonically | Magnitude control → Strong causality |
| Scale norm $\lambda\cdot \mathbf{o}_k$ (control) | No sink appears | Excludes "large norm" as the cause |
| $\mathbf{W}_O$ Kendall $\tau$ vs $\boldsymbol{\sigma}_{in}$ | Mean 0.32 (positive bias) | $\mathbf{W}_O$ structurally amplifies variance |
| RMSNorm after Layer 2 outlier dim 2533 | DomRatio 262.88× | Direction collapses almost entirely to $\mathbf{e}_{2533}$ |

### Key Findings
*   **Sigmoid attention removes sinks but performs worse**: While it validates the root cause, replacing the activation is not a "free lunch" because Sigmoid output scales with sequence length, introducing instability.
*   **HeadNorm both removes sinks and accelerates convergence**: A theoretical bonus where variance normalization improves the optimization landscape conditioning for AdamW.
*   **Effective rank increases from 343 to 446**: The sink is not just an attention phenomenon; it accompanies manifold collapse of hidden states. HeadNorm restores representation capacity.
*   **Super neurons are learned structures**: Their positions are fixed after pre-training, and their corresponding $\mathbf{W}_{\text{down}}$ row vectors are sparse, providing a root cause for outlier handling in low-bit quantization.

## Highlights & Insights
*   **Three-stage Causal Chain + Two Controlled Interventions**: Transforms attention sink from an empirical observation into an intervenable engineering problem. The mask and variance amplification experiments are definitive.
*   **Elegant and Low-cost Engineering Solution**: HeadNorm is just one line of RMSNorm plus a learnable $\boldsymbol{\lambda}$. It requires no change to attention math or Softmax and can be integrated into existing LLaMA-style pre-training pipelines.
*   **Super Neuron + Sparse Down-projection Perspective**: Unifies seemingly unrelated phenomena—attention sinks, activation outliers, quantization difficulties, and representation collapse—under a single structural explanation in the FFN.

## Limitations & Future Work
*   Intervention experiments were primary conducted on Llama-2-7B. While other open-source LLMs were verified in the appendix, they belong to the same architecture family (decoder-only + SwiGLU + RMSNorm). Generalization to GLU variants or different norm types (LayerNorm, DeepNorm) requires further testing.
*   HeadNorm pre-training was only verified at 152M parameters / 20B tokens. Whether industrial-scale 7B+ models maintain the same convergence acceleration is not guaranteed.
*   Did not analyze the impact of HeadNorm on downstream long-context performance (especially methods like StreamingLLM that rely on sinks for KV compression).
*   Future direction: Developing dynamic normalization (e.g., Mixture-of-Norm) to determine where and when to normalize based on head behavior.

## Related Work & Insights
*   **vs Xiao et al. 2023 (StreamingLLM)**: They discovered and utilized the sink; Ours unearths the underlying structural layers to show it is a fully avoidable artifact.
*   **vs Cancedda 2024 (Spectral Subspace)**: While Cancedda explains sinks via QK spectral subspaces, Ours explains **why** these subspaces form (directional collapse via RMSNorm).
*   **vs Liu et al. 2024 (Activation Outliers)**: Ours proves outliers and sinks share the same root (super neurons + sparse projection), suggesting that solving sinks simultaneously mitigates quantization issues.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ (The three-stage causal chain and super neuron perspective are genuine new explanations).
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Controlled interventions + multi-seed verification + cross-LLM replication).
*   Writing Quality: ⭐⭐⭐⭐⭐ (The logical flow from phenomenon to hypothesis to verification is very clean).
*   Value: ⭐⭐⭐⭐ (Immediately applicable engineering fix; insights for quantization and long-context).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion](splattn_bridging_2d_and_3d_with_gaussian_soft_splatting_and_attention_for_point_.md)
- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](../../AAAI2026/3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](../../AAAI2026/3d_vision/debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](../../ICCV2025/3d_vision/strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)

</div>

<!-- RELATED:END -->
