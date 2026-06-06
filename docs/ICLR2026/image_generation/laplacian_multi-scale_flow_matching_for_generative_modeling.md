---
title: >-
  [Paper Note] Laplacian Multi-scale Flow Matching for Generative Modeling
description: >-
  [ICLR 2026][Image Generation][Multi-scale Generation] This paper proposes LapFlow, which decomposes images into Laplacian pyramid residuals and models different scales in parallel via a Mixture-of-Transformers (MoT) arch…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Multi-scale Generation"
  - "Laplacian Pyramid"
  - "Flow Matching"
  - "Mixture-of-Transformers"
  - "Causal Attention"
date: 2026-05-08
content_hash: 833e147a145a8ec6
---

# Laplacian Multi-scale Flow Matching for Generative Modeling

**Conference**: ICLR 2026
**arXiv**: [2602.19461](https://arxiv.org/abs/2602.19461)  
**Code**: [GitHub](https://github.com/sjtuytc/gen)  
**Area**: Diffusion Models / Flow Matching
**Keywords**: Multi-scale Generation, Laplacian Pyramid, Flow Matching, Mixture-of-Transformers, Causal Attention

## TL;DR
This paper proposes LapFlow, which decomposes images into Laplacian pyramid residuals and models different scales in parallel via a Mixture-of-Transformers (MoT) architecture with causal attention, reducing computational cost while improving generation quality.

## Background & Motivation
- Diffusion models and Flow Matching have achieved state-of-the-art performance in image synthesis, yet scalability remains a critical challenge as resolution increases.
- Existing multi-scale approaches (Cascaded Diffusion, EdifyImage, Pyramidal Flow) each have limitations: they require training multiple independent networks, operate in pixel space leading to slow inference, or suffer from poor performance when trained from scratch.
- A multi-scale framework is needed that simultaneously improves generation quality, accelerates sampling, and scales to high resolutions.

## Method

### Overall Architecture
LapFlow decomposes images into Laplacian pyramid residuals at three scales and processes them in parallel through a unified MoT model. A progressive generation strategy is adopted: the coarsest scale is denoised first, with finer scales conditioned progressively thereafter.

### Key Designs

1. **Laplacian Decomposition**: The image is decomposed into residuals at three scales:
$$\mathbf{x}_1^{(2)} = \text{Down}(\text{Down}(\mathbf{x}_1)), \quad \mathbf{x}_1^{(1)} = \text{Down}(\mathbf{x}_1) - \text{Up}(\mathbf{x}_1^{(2)})$$
$$\mathbf{x}_1^{(0)} = \mathbf{x}_1 - \text{Up}(\text{Down}(\mathbf{x}_1))$$
   Reconstruction: $\mathbf{x}_1 = \mathbf{x}_1^{(0)} + \text{Up}(\mathbf{x}_1^{(1)}) + \text{Up}(\text{Up}(\mathbf{x}_1^{(2)}))$

2. **Multi-scale Noise Process**: Different scales are trained over different time intervals. Two key time points $T_1, T_2$ are defined: the smallest scale $k=2$ is trained over $[0,1]$, the intermediate scale $k=1$ over $[T_2,1]$, and the largest scale $k=0$ over $[T_1,1]$. The noise interpolation at each scale is:
$$\mathbf{x}_t^{(k)} = \alpha_t^{(k)} \mathbf{x}_1^{(k)} + \sigma_t^{(k)} \mathbf{x}_0^{(k)}$$

3. **MoT Architecture and Causal Attention**: Scale-specific QKV projections are combined with shared global attention. A causal mask enforces unidirectional information flow from low to high resolution:
$$\text{MaskedGlobalAttn}(Q,K,V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}} + M_c\right)V$$
   where $M_c$ is a block-causal mask ensuring scale $k$ only attends to scales $k' \geq k$.

### Loss & Training
The multi-scale conditional Flow Matching loss is:
$$\mathcal{L}_{mv} = \sum_{k=2}^{s} w_k \mathbb{E}_{t,q,p_t} \|\mathbf{v}_t^{(k)} - \mathbf{u}_t^{(k)}(\mathbf{x}_t^{(k)}|\mathbf{x}_1^{(k)})\|^2$$
A progressive training strategy is employed: at each sampled stage $s$, all scales $k \geq s$ are trained jointly.

## Key Experimental Results

### Main Results

| Method | Dataset | Resolution | FID↓ | GFLOPs | Inference Time (s) |
|--------|---------|------------|------|--------|--------------------|
| LFM | CelebA-HQ | 256 | 5.26 | 22.1 | 1.70 |
| Pyramidal Flow | CelebA-HQ | 256 | 11.20 | 14.2 | 1.85 |
| **LapFlow (Ours)** | CelebA-HQ | **256** | **3.53** | **16.5** | **1.51** |
| LFM | CelebA-HQ | 512 | 6.35 | 43.5 | 2.90 |
| **LapFlow (Ours)** | CelebA-HQ | **512** | **4.04** | **41.7** | **2.60** |
| LFM | CelebA-HQ | 1024 | 8.12 | 154.8 | 4.20 |
| **LapFlow (Ours)** | CelebA-HQ | **1024** | **5.51** | **148.2** | **3.30** |

### Ablation Study

| Configuration | FID (256×256) | GFLOPs | Note |
|---------------|--------------|--------|------|
| Separate Model | 3.60 | 38.9 | Independent model per scale |
| **MoT (Default)** | **3.53** | **16.5** | Shared parameters + experts |
| SDVAE | 4.37 | - | Standard VAE |
| **EQVAE (Default)** | **3.53** | - | Equivariant VAE |

### Key Findings
- LapFlow achieves FID=3.53 on CelebA-HQ 256, substantially outperforming LFM (FID=5.26).
- The MoT design reduces GFLOPs from 38.9 to 16.5 while marginally improving FID.
- Causal masking is critical: removing the mask or using self-attention only both degrade performance.
- The method scales effectively to 1024×1024 resolution while maintaining low computational overhead.

## Highlights & Insights
- The Laplacian pyramid's natural multi-scale structure is exploited to model different frequency components separately.
- The MoT architecture elegantly combines scale-specific processing with globally shared attention, achieving parameter-efficient computation.
- Causal attention enforces a natural information flow from structure to detail, realizing hierarchical generation.
- Theoretical analysis of time-weighted complexity demonstrates that the progressive multi-scale design incurs lower attention cost than DiT.

## Limitations & Future Work
- Evaluation is currently limited to CelebA-HQ and ImageNet; text-guided generation remains unassessed.
- The applicability of Laplacian decomposition in latent space may be less straightforward than in pixel space.
- The key time points $T_1, T_2$ require manual specification.
- No comparison with recent large-scale text-to-image models is provided.

## Related Work & Insights
- The multi-scale paradigm traces from LapGAN through Cascaded Diffusion and Pyramidal Flow; LapFlow achieves parallel processing by eliminating explicit bridging mechanisms.
- The MoT design draws on Mixture-of-Experts and represents its first application to multi-scale visual generation.
- The proposed framework offers a more efficient alternative for high-resolution visual generation.

## Technical Details
- The three-scale Laplacian decomposition operates in latent space (VAE downsampling factor 8), with a maximum latent size of 32×32.
- Training uses DiT-L/2 for CelebA-HQ and supports DiT-B/2 and DiT-XL/2 for ImageNet.
- Sampling employs the Dormand–Prince (dopri5) ODE solver.
- The GVP path generally outperforms the linear path (verified by ablation).
- Classifier-free guidance is supported and applied on ImageNet.
- Theoretical time-weighted complexity analysis confirms that effective attention cost is lower than that of DiT.
- EQVAE benefits LapFlow but not LFM (Table 2a), indicating that the multi-scale framework better exploits high-quality VAE representations.
- On ImageNet 256, LapFlow surpasses both single-scale and multi-scale baselines with lower GFLOPs.
- Both class-conditional generation on ImageNet and unconditional generation on CelebA-HQ are supported.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Laplacian pyramid, MoT, and causal attention is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation and ablation on two datasets, though text-to-image experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with thorough algorithmic descriptions.
- Value: ⭐⭐⭐⭐ Achieves a favorable balance between efficiency and quality, offering meaningful guidance for multi-scale generation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)
- [\[ICLR 2026\] GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/image_generation/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)

</div>

<!-- RELATED:END -->
