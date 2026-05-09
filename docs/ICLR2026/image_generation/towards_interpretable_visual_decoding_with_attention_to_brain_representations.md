---
title: >-
  [Paper Note] Towards Interpretable Visual Decoding with Attention to Brain Representations
description: >-
  [ICLR 2026][Image Generation][fMRI visual decoding] This paper proposes NeuroAdapter, which segments fMRI signals into independent tokens by brain region and conditions Stable Diffusion directly via cross-attention, bypassing conventional CLIP/DINO intermediate embedding spaces. On NSD and other datasets, NeuroAdapter matches or surpasses existing methods on high-level semantic metrics. It further introduces the IBBI bidirectional interpretability framework, which for the first time dynamically reveals how different cortical regions drive image generation along the denoising trajectory.
tags:
  - ICLR 2026
  - Image Generation
  - fMRI visual decoding
  - end-to-end brain-to-image reconstruction
  - cross-attention conditioning
  - bidirectional interpretability
  - brain region tokens
date: 2026-05-08
content_hash: b5a65078194d3eaa
---

# Towards Interpretable Visual Decoding with Attention to Brain Representations

**Conference**: ICLR 2026
**arXiv**: [2509.23566](https://arxiv.org/abs/2509.23566)
**Code**: [GitHub](https://github.com/kriegeskorte-lab/NeuroAdapter)
**Area**: Brain-Computer Interface / Visual Decoding / Medical Imaging
**Keywords**: fMRI visual decoding, end-to-end brain-to-image reconstruction, cross-attention conditioning, bidirectional interpretability, brain region tokens

## TL;DR

This paper proposes NeuroAdapter, which segments fMRI signals into independent tokens by brain region and conditions Stable Diffusion directly via cross-attention, bypassing conventional CLIP/DINO intermediate embedding spaces. On NSD and other datasets, NeuroAdapter matches or surpasses existing methods on high-level semantic metrics. It further introduces the IBBI bidirectional interpretability framework, which for the first time dynamically reveals how different cortical regions drive image generation along the denoising trajectory.

## Background & Motivation

**Background**: Reconstructing visual stimuli from human fMRI activity is a central challenge in computational neuroscience. Dominant approaches adopt a two-stage pipeline—first mapping fMRI signals into the embedding space of pretrained vision-language models (e.g., CLIP, DINO), then using these embeddings to guide generative models such as Stable Diffusion. Representative works include Brain Diffuser, MindEye1, and DREAM.

**Limitations of Prior Work**: Two-stage pipelines suffer from two fundamental issues. (1) **Information bottleneck**: The dimensionality and semantic coverage of intermediate embedding spaces are limited, potentially discarding rich low- and high-level neural information during mapping—reconstruction quality is thus bounded by fMRI-to-embedding alignment rather than the intrinsic information content of brain activity. (2) **Obscured interpretability**: The intermediate mapping severs the direct link between brain regions and generated outputs, making it impossible to trace "which brain regions drove which parts of the image," limiting the scientific utility of decoding methods in neuroscience research.

**Key Challenge**: Two-stage methods place "improving reconstruction quality" and "maintaining interpretability" in opposition—the introduction of embedding spaces improves generation quality at the cost of transparency in brain region attribution.

**Goal**: (1) Design an end-to-end framework that maps directly from fMRI to images without any intermediate embedding space; (2) Achieve brain-region-level interpretability without sacrificing reconstruction quality—dynamically tracking each cortical region's contribution to the generation process.

**Key Insight**: Each brain region is treated as an independent token, and a diffusion model directly "attends" to these brain region tokens via cross-attention. This design naturally establishes a one-to-one correspondence between attention weights and brain regions, making the attention matrix itself the object of interpretability analysis.

**Core Idea**: Condition the diffusion model directly via cross-attention using brain-region-granularity tokens, making attention weights a natural carrier of interpretability.

## Method

### Overall Architecture

Input: fMRI cortical surface data recorded while subjects view images → Schaefer parcellation yields 500 regions per hemisphere → the top $p=200$ regions by SNR are selected → each region is projected to a token embedding via an independent linear projection, yielding $E \in \mathbb{R}^{p \times f}$ → the cross-attention layers of the Stable Diffusion U-Net are replaced with IP-Adapter-style modules, allowing U-Net spatial queries to attend directly to fMRI tokens → reconstructed images are generated. During training, the main SD parameters are frozen; only the region projection matrices and new cross-attention modules are updated. At inference, a brain encoder selects the best reconstruction from multiple candidate images.

### Key Designs

1. **Brain-Region-Granularity Tokenization and Linear Projection**:

    - Function: Transforms high-dimensional fMRI cortical data into a structured token sequence as the conditioning input for the diffusion model.
    - Mechanism: For each brain region $P_i$, its vertex response vectors are zero-padded to the maximum vertex count $v_{max}$, then mapped to an $f$-dimensional token via an independent projection matrix $w \in \mathbb{R}^{v_{max} \times f}$ ($f=768$). The 200 regions yield 200 tokens, forming the conditioning sequence $E \in \mathbb{R}^{p \times f}$. Each region has its own projection matrix with no shared parameters.
    - Design Motivation: The key insight is the "one region = one token" correspondence. This differs from flattening the entire brain into a high-dimensional vector before projection—region-granularity tokenization ensures that each column of the cross-attention matrix naturally corresponds to an anatomically defined brain region, laying the groundwork for subsequent interpretability analysis. Linear rather than MLP mappings are used deliberately to avoid nonlinear transformations that would obscure the attribution of region-specific information.

2. **fMRI Token Dropout Regularization**:

    - Function: Prevents the model from over-relying on specific brain regions, improving decoding robustness.
    - Mechanism: During training, a dropout probability $r \sim \mathcal{U}(0,1)$ is independently sampled for each sample, generating a binary mask $M \in \{0,1\}^{p \times 1}$; each token is zeroed with probability $r$: $E' = E \odot M$. Since the dropout probability itself is uniformly random, the model is exposed during training to conditions ranging from nearly no dropout to almost complete dropout.
    - Design Motivation: Analogous to conditional dropout in classifier-free guidance, this trains the model to produce reasonable outputs under varying levels of available information. Ablation studies show this design is critical—removing it causes a significant drop in high-level semantic metrics.

3. **Brain Encoder–Assisted Image Selection**:

    - Function: Mitigates instability in generation quality caused by the stochasticity of diffusion models.
    - Mechanism: For each test fMRI sample, $n$ candidate images are generated using different random seeds. A pretrained whole-brain encoder (Transformer architecture) predicts the expected fMRI response $B'_i$ for each candidate image; the candidate whose predicted response has the highest Pearson correlation with the measured fMRI is selected as the final output.
    - Design Motivation: The stochasticity of diffusion models is a double-edged sword—images generated from the same condition can vary substantially in quality. The brain encoder "back-validates" candidates, selecting the reconstruction most consistent with the original brain activity pattern, essentially using encode-decode consistency as a quality criterion.

### IBBI Bidirectional Interpretability Framework

The IBBI (Image–Brain BI-directional) framework leverages the cross-attention matrix $A^{(\ell,h,t)} \in \mathbb{R}^{q \times p}$ to provide two complementary analytical perspectives:

**Brain-guided view**—"Which brain regions are driving generation?": The attention matrix is aggregated over the query dimension, layers, and heads, yielding a region contribution vector $B^{(t)} \in \mathbb{R}^p$ satisfying $\sum_j B_j^{(t)} = 1$. Projecting this onto the cortical surface visualizes the relative influence of each region at each denoising step.

**Image-guided view**—"Where in the image is a given brain region attending?": For a given ROI $\mathcal{R}$, attention is pooled over heads and tokens within the ROI to obtain a per-layer query-wise attention map $m_\mathcal{R}^{(\ell,t)} \in \mathbb{R}^{q^\ell}$, which is reshaped to 2D, upsampled to image resolution, and averaged across layers to produce an ROI attention map $I_\mathcal{R}^{(t)}$. This reveals the "functional footprint" of specific brain regions (e.g., FFA for faces, PPA for scenes) in image space.

### Loss & Training

The standard diffusion loss is combined with Min-SNR weighting, which down-weights high-SNR (low-noise, easier reconstruction) steps and preserves gradients at low-SNR steps, balancing the training signal. The text encoder receives empty input, ensuring fMRI tokens are the sole conditioning source. Training runs for 300 epochs with batch size 16 on 2 NVIDIA L40 GPUs, taking approximately 25 hours.

## Key Experimental Results

### Main Results

Comparison with mainstream methods on the NSD dataset (averaged over 4 subjects, expressed as percentage improvement relative to an ImageNet retrieval baseline):

| Method | Type | CLIP↑ | Incep↑ | Eff↑ | SwAV↑ | PixCorr↑ | SSIM↑ |
|--------|------|-------|--------|------|-------|----------|-------|
| Brain Diffuser (w/ VDVAE) | Two-stage | High | High | High | High | **Best** | **Best** |
| Brain Diffuser (w/o VDVAE) | Two-stage | Medium | Medium | Medium | Medium | Comparable to Ours | Comparable to Ours |
| MindEye1 | Two-stage | High | High | High | High | Medium | Medium |
| DREAM | Two-stage | High | High | High | High | Medium | Medium |
| MindFormer | Multi-subject | High | High | High | High | Medium | Medium |
| **NeuroAdapter (Ours)** | **End-to-end** | **Best/On-par** | **Best/On-par** | **Competitive** | **Competitive** | Medium | Medium |

Key takeaway: NeuroAdapter is competitive with or surpasses two-stage methods that rely on CLIP/DINO embeddings on high-level semantic metrics (CLIP, Incep, Eff, SwAV). On low-level metrics (PixCorr, SSIM), it is on par with Brain Diffuser without the VDVAE pathway—indicating that the gap in low-level metrics stems from the additional low-level feature pathway provided by VDVAE, not from an inherent limitation of the end-to-end approach.

### Ablation Study

Key ablations on NSD Subject 1 (direction of change relative to the full model):

| Configuration | High-level Metrics | Low-level Metrics | Notes |
|--------------|-------------------|-------------------|-------|
| Full NeuroAdapter | Best | Best | Complete model, $p=200$, $f=768$ |
| w/o fMRI Token Dropout | Significant drop | Drop | Dropout is critical for robustness |
| w/o Min-SNR weighting | Slight drop | Drop | Training signal balance helps |
| w/o Brain Encoder Selection | Drop | Drop | Stochasticity degrades quality |
| $p=50$ (fewer regions) | Significant drop | Drop | Insufficient information |
| $p=400$ (more regions) | Slight drop | Slight drop | Low-SNR regions introduce noise |
| Visual tokens only (no region structure) | Large drop | Large drop | Region-granularity tokenization is essential |

### Key Findings

- **fMRI Token Dropout is the most critical design**: Removing it causes a substantial performance drop, indicating that the model is prone to overfitting to specific region combinations. Uniformly sampling the dropout probability is more effective than using a fixed probability.
- **$p=200$ regions is the sweet spot**: Too few (50) provides insufficient information; too many (400) introduces low-SNR noise. The combination of SNR-based filtering and a moderate number of regions is most effective.
- **Higher-order visual areas dominate generation dynamics**: IBBI analysis shows that the attention contributions of high-order visual regions such as FFA (face-selective area) and PPA (scene-selective area) consistently far exceed those of low-level areas such as V1/V2.
- **Temporal dynamics of attention maps align with neuroscience**: Attention is distributed broadly across the image in early denoising steps and progressively concentrates on semantically relevant regions as denoising proceeds—face ROIs converge to faces, scene ROIs expand to backgrounds.
- **Causal perturbation validates functional selectivity**: Masking low-level visual ROIs (V1–V3) does not affect semantic content, whereas masking high-level ROIs (FFA, PPA, LOC) completely alters the generated image, confirming that higher-order areas carry core semantic information.
- **Generalization on NSD-Imagery and Deeprecon**: NeuroAdapter demonstrates reasonable generalization under mental imagery tasks and settings where training and test categories do not overlap, achieving high-level semantic metrics comparable to existing methods.

## Highlights & Insights

- **The "one region = one token" design is remarkably elegant**: This seemingly simple correspondence simultaneously addresses both conditioning and interpretability—each column of the cross-attention matrix directly corresponds to an anatomically defined brain region, enabling brain region contribution analysis without any additional probes or post-processing. This "design as interpretability" philosophy is worth emulating.
- **The IBBI framework provides a neuroscientific probe for generative models**: By linking the denoising trajectory of a diffusion model to brain region function, it not only validates the known visual hierarchy (low-level areas → low-level features; high-level areas → semantics) but also yields novel findings in the temporal dimension (dynamics of attention from diffusion to convergence).
- **The counter-intuitive finding that removing intermediate embeddings does not degrade performance**: This suggests that fMRI signals themselves carry sufficient information to drive high-quality generation directly, and that CLIP/DINO embedding spaces are a convenience rather than a necessity—raising the question of whether intermediate representations can be bypassed in other modalities as well.
- **The brain encoder selection strategy is a general paradigm for filtering generation quality**: Using "inverse encoding" to back-validate reconstructed outputs against input conditions is generalizable to any conditional generation task.

## Limitations & Future Work

- **Weak reconstruction of low-level visual features**: The end-to-end approach forgoes low-level feature pathways such as VDVAE, leading to a gap in PixCorr/SSIM. Adding a lightweight pixel-level auxiliary loss or a waveform VAE branch could address this.
- **Diffusion model stochasticity**: Even with brain encoder selection, generation quality remains unstable. Deterministic sampling or consistency models could be explored to reduce stochasticity.
- **Interpretability reflects correlation, not causation**: Cross-attention weights indicate "what the model chooses to attend to" rather than "what brain regions truly encode." Causal perturbation analysis is a step forward, but more rigorous causal inference methods remain necessary.
- **Limited subject scale**: NSD contains only 8 subjects; cross-subject generalization is insufficiently validated. Future work should incorporate functional alignment techniques to support cross-subject decoding.
- **Additional overhead from brain encoder**: The selection strategy requires a pretrained encoder and multiple forward passes, resulting in high inference cost.

## Related Work & Insights

- **vs Brain Diffuser**: Uses CLIP and VDVAE as separate pathways to capture high- and low-level features for SD guidance. NeuroAdapter replaces this with a single end-to-end pathway, matching or exceeding high-level metrics while being limited on low-level metrics due to the absence of VDVAE. The key distinction is that Brain Diffuser's embedding-space mapping severs the brain region attribution chain.
- **vs MindEye1/DREAM**: These methods use carefully designed CLIP alignment strategies and large-scale data augmentation to boost performance. NeuroAdapter demonstrates that direct conditioning without embedding alignment can achieve competitive results—suggesting that the gains from embedding alignment may be overstated.
- **vs DynaDiff**: Uses LoRA fine-tuning of SD to process dynamic fMRI, representing an alternative route toward single-stage decoding. However, LoRA-based conditioning lacks brain-region-granularity interpretability and cannot map attention analysis back to anatomical structures.
- **vs Brain encoding models (Adeli et al. 2025)**: Transformer attention maps in the encoding direction (image→fMRI) reveal "which visual features are routed to which brain regions." NeuroAdapter provides a complementary perspective from the decoding direction (fMRI→image); combining both yields a more complete understanding of visual representations in the brain.

## Rating

- Novelty: ⭐⭐⭐⭐ End-to-end conditioning combined with the "design as interpretability" paradigm represents an important methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, comprehensive ablations, and both qualitative and quantitative IBBI analysis; discussion of low-level metrics could be more in-depth.
- Writing Quality: ⭐⭐⭐⭐ Method and interpretability framework are described clearly with high-quality figures.
- Value: ⭐⭐⭐⭐ Provides a new interpretability tool for neural decoding research, with state-of-the-art performance on high-level semantic reconstruction.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)
- [\[ICLR 2026\] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)
- [\[CVPR 2026\] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](../../CVPR2026/image_generation/cognitioncapturerpro_towards_high-fidelity_visual_decoding_from_eegmeg_via_multi.md)
- [\[ICLR 2026\] Generating Directed Graphs with Dual Attention and Asymmetric Encoding](generating_directed_graphs_with_dual_attention_and_asymmetric_encoding.md)

<!-- RELATED:END -->
