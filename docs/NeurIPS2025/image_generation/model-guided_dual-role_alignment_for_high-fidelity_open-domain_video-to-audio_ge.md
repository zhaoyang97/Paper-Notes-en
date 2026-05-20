---
title: >-
  [Paper Note] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation
description: >-
  [NeurIPS 2025][Image Generation][Video-to-Audio] This paper proposes MGAudio, the first video-to-audio generation framework that replaces classifier-free guidance (CFG) with model-guided (MG) training…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Video-to-Audio"
  - "Model Guidance"
  - "Flow Matching"
  - "Dual-Role Alignment"
  - "CFG Alternative"
date: 2026-05-08
content_hash: 1f08f98b173a76f4
---

# MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.24103](https://arxiv.org/abs/2510.24103)  
**Code**: [GitHub](https://github.com/pantheon5100/mgaudio)  
**Area**: Image Generation
**Keywords**: Video-to-Audio, Model Guidance, Flow Matching, Dual-Role Alignment, CFG Alternative

## TL;DR

This paper proposes MGAudio, the first video-to-audio generation framework that replaces classifier-free guidance (CFG) with model-guided (MG) training, combined with a dual-role audio-video encoder (DRAVE) for simultaneous condition injection and feature alignment. With only 131M parameters, MGAudio achieves state-of-the-art performance on VGGSound (FAD=0.40) and surpasses most competing methods using only 10% of the training data.

## Background & Motivation

- **Background**: Video-to-audio (V2A) generation aims to synthesize semantically aligned and temporally synchronized audio for silent videos. Mainstream approaches—including Diff-Foley, FRIEREN, MDSGen, and MMAudio—rely on **classifier-free guidance (CFG)**, which randomly drops 10% of conditioning signals during training to jointly learn conditional and unconditional objectives.
- **Limitations of Prior Work**: CFG introduces two key problems:
    - **Multi-task dilution**: Simultaneously learning conditional and unconditional objectives may divide model capacity, leaving both sub-tasks underoptimized.
    - **Train-inference mismatch**: The condition-dropout behavior during training is inconsistent with the CFG-scaling behavior during inference, leading to misaligned sampling dynamics.
- **Key Challenge**: Vision Model-Guidance (VMG) has demonstrated the feasibility of replacing CFG with direct self-guidance in image generation, but its design targets discrete class labels and has not been explored for continuous video-conditioned audio generation. The authors further observe a **modality-specific behavior**: unlike in the visual domain, MG training in audio accelerates convergence but still requires CFG at inference time for peak quality, potentially due to the temporal sensitivity of audio signals. Additionally, prior methods typically use only the video encoder from CAVP for condition injection, discarding the audio encoder—which the authors argue can serve as an alignment target for intermediate representations, analogous to REPA in the visual domain.
- **Goal**: To develop a V2A generation framework that (1) replaces CFG training with model-guided objectives, (2) fully exploits both branches of a pretrained audio-video encoder for condition injection and representation alignment, and (3) achieves strong performance with reduced model size and data requirements.

## Method

### Overall Architecture

MGAudio consists of three core components: (1) a scalable flow-matching-based Transformer denoiser (FBDT); (2) a dual-role audio-video encoder (DRAVE) for simultaneous condition injection and feature alignment; and (3) an audio model-guided (AMG) training objective as a replacement for CFG. Input video frames are encoded by the CAVP video encoder and aggregated into a global conditioning vector, which guides a flow-based Transformer to generate audio latent representations from noise.

### Key Designs

1. **Flow-Matching-Based Denoising Transformer (FBDT)**: The architecture follows SiT (Scalable Interpolant Transformer). Input audio is converted to a mel spectrogram $\mathbf{X} \in \mathbb{R}^{64 \times 816}$, encoded by the AudioLDM VAE into a latent representation $\mathbf{x} \in \mathbb{R}^{8 \times 16 \times 204}$, and patchified into a token sequence $\mathbf{x}' \in \mathbb{R}^{816 \times D}$ (where $D=768$ for the Base model). Video frames are encoded by CAVP and aggregated via a 1×1 convolution into a global vector $\vec{v} \in \mathbb{R}^{1 \times 768}$, injected through Adaptive LayerNorm. The flow matching objective is:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \vec{v}, \epsilon} \| u_\theta(\mathbf{x}_t, \vec{v}, t) - u_t(\mathbf{x}_t | \mathbf{x}_0) \|^2$$

where $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\epsilon$ and the ground-truth flow direction is $u_t = \mathbf{x}_0 - \epsilon$.

2. **Dual-Role Audio-Video Encoder (DRAVE)**: Both encoders of CAVP are utilized: the video encoder provides conditioning signals to guide denoising (via AdaLN), while the audio encoder provides reference representations of clean audio for intermediate-layer alignment. The alignment loss is:

$$\mathcal{L}_{\text{align-audio}} = -\mathbb{E}_{\mathbf{x}, \epsilon, t} \left[ \frac{1}{\mathcal{B}} \sum_{i=1}^{\mathcal{B}} \text{similarity}(\mathbf{G}_0^i, h_\phi(\mathbf{H}_t^i)) \right]$$

where $\mathbf{G}_0$ denotes the clean audio features encoded by CAVP, $\mathbf{H}_t$ denotes the intermediate latent representations of the denoising Transformer at timestep $t$, and $h_\phi$ is an MLP projection layer. Cosine similarity is used as the similarity metric. This dual-role design allows the CAVP encoder to contribute both as a conditioning provider (video branch) and a learning signal (audio branch), fully exploiting the potential of the pretrained encoder.

3. **Audio Model Guidance (AMG)**: As a replacement for CFG, the training objective leverages self-guidance. The model-guided target is defined as:

$$u' = u + w \cdot \text{sg}(u_\theta(\mathbf{x}_t, \vec{v}, t)) - u_\theta(\mathbf{x}_t, \varnothing, t)$$

$$\mathcal{L}_{\text{AMG}} = \mathbb{E}_{t, \mathbf{x}_0, \vec{v}, \epsilon} \| u_\theta(\mathbf{x}_t, \vec{v}, t) - u' \|^2$$

where $w$ is the guidance scale, $\text{sg}(\cdot)$ denotes the stop-gradient operator to prevent degenerate gradients, and $\varnothing$ is a zero-vector condition. An EMA model is used to compute $u'$ for improved training stability. During training, the condition is replaced with a zero vector with probability $\psi$.

### Loss & Training

$$\mathcal{L}_{\text{FM-align}} = \mathcal{L}_{\text{AMG}} + \lambda \mathcal{L}_{\text{align-audio}}$$

$\lambda = 0.5$. The model is trained for 1.1M steps with batch size 64, learning rate 1e-4, on a single A100 80GB GPU. Inference uses 50 sampling steps with CFG=1.45.

## Key Experimental Results

### Main Results

**VGGSound Test Set**

| Method | FAD↓ | FD↓ | IS↑ | KL↓ | Align Acc↑ | Params |
|--------|------|-----|-----|-----|-----------|--------|
| Diff-Foley | 6.25 | 23.07 | 10.85 | 3.18 | 93.94 | 860M |
| FRIEREN | 1.38 | 12.36 | 12.12 | 2.73 | 97.25 | 157M |
| MMAudio | 0.71 | 6.97 | 11.09 | 2.07 | 92.28 | 157M |
| **MGAudio** | **0.40** | **6.16** | **12.82** | 2.76 | 95.65 | **131M** |

MGAudio achieves an FAD of 0.40, representing a 43% improvement over the second-best method MMAudio (0.71).

**UnAV-100 Generalization Test (Zero-Shot)**

| Method | FAD↓ | FD↓ | IS↑ | Align Acc↑ |
|--------|------|-----|-----|-----------|
| MMAudio | 0.93 | 8.63 | 11.37 | 85.68 |
| **MGAudio** | **0.54** | **5.40** | **13.90** | 97.54 |

MGAudio demonstrates strong cross-dataset generalization, with Align Acc substantially outperforming MMAudio.

### Ablation Study

**Data Efficiency (300k steps)**

| Training Data Ratio | FAD↓ | FD↓ | IS↑ | KL↓ | Align Acc↑ |
|--------------------|------|-----|-----|-----|-----------|
| 5% | 1.16 | 9.72 | 9.80 | 2.67 | 93.37 |
| 10% | **0.73** | **8.79** | 10.28 | 2.64 | 95.73 |
| 100% | 0.81 | 10.25 | 9.85 | 2.71 | 95.14 |

Notably, training on 10% of the data (FAD=0.73) outperforms training on 100% (FAD=0.81), suggesting that approximately 15% of VGGSound samples are noisy.

**Effect of Model Guidance**

| Setting | CFG | MG | Dual Align | FAD↓ | FD↓ | Align Acc |
|---------|-----|-----|-----------|------|-----|-----------|
| No guidance | ✗ | ✗ | ✗ | 2.67 | 17.23 | 78.20 |
| SiT (CFG only) | ✓ | ✗ | ✗ | 1.52 | 16.25 | 87.07 |
| MG only | ✗ | ✓ | ✗ | 2.13 | 16.20 | 83.87 |
| MG + CFG | ✓ | ✓ | ✗ | 1.19 | 14.26 | 92.37 |
| MGAudio (full) | ✓ | ✓ | ✓ | **1.14** | **13.09** | **93.67** |

**Alignment Encoder Selection**

| Encoder | FAD↓ | Align Acc↑ | Notes |
|---------|------|-----------|-------|
| DINOv2 | 5.52 | 21.66 | Visual encoder unsuitable for audio alignment |
| CLAP | 1.35 | 93.19 | Audio-language pretrained encoder |
| CAVP | **1.14** | **93.67** | Audio-visual contrastive pretraining is superior |

### Key Findings

- **MG in audio ≠ VMG in vision**: In the visual domain, MG alone suffices to reach SOTA; in the audio domain, MG must be combined with CFG at inference time for optimal results, likely because the temporal structure of audio requires additional inference-time guidance.
- Data quality outweighs data quantity: Approximately 6% of VGGSound samples exhibit weak audio-visual alignment, 9% contain static frames or silence, leaving only ~85% effective samples.
- The model scales well: FAD decreases consistently from 2.54 (S/2, 34M) to 0.90 (XL/2, 680M).
- UMAP visualization shows that MGAudio's generated audio distribution clusters more tightly around the real data distribution compared to CFG-based methods.

## Highlights & Insights

- **Elegant core idea**: The model uses the difference between its own EMA conditional and unconditional predictions to guide itself, replacing condition dropout in CFG training.
- **Remarkable data efficiency**: Training on 10% of the data surpasses most methods trained on 100%, demonstrating that AMG effectively exploits the conditioning structure.
- **Practical dual-role design**: No new external models are introduced; the two branches of the existing CAVP encoder are fully utilized.
- At 131M parameters, MGAudio substantially outperforms much larger models such as Diff-Foley (860M) and See&Hear (1099M) across all metrics.

## Limitations & Future Work

- The model performs poorly on linguistically complex audio such as human speech and singing, due to a lack of phoneme and speech structure awareness.
- Ambiguous visual semantics may introduce confusion, as a single scene can correspond to multiple plausible sounds.
- Inference speed is constrained by the VAE and iterative sampling; acceleration via consistency models or distillation is a natural future direction.
- More fine-grained temporal alignment strategies remain unexplored, as the current design relies on global aggregation of video features.

## Related Work & Insights

- The AMG training strategy is broadly applicable to other conditional generation tasks, including text-to-audio and text-to-video generation.
- The dual-role encoder concept can be extended to other pretrained models: leveraging encoders not only for condition injection but also for intermediate representation supervision.
- The data quality findings suggest that cleaning VGGSound—constructing a high-quality subset—may be a cost-effective strategy for improving performance.
- The synergy between AMG and CFG warrants further investigation across additional modalities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First work to extend model guidance from image generation to audio generation; the dual-role alignment design is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five ablation dimensions with comprehensive coverage; in-depth analysis of data efficiency and scalability; intuitive UMAP visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear, though the related work section is somewhat lengthy.
- **Value**: ⭐⭐⭐⭐⭐ Achieves substantial SOTA improvements with a compact model; the data efficiency findings offer meaningful guidance for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](../../ICLR2026/image_generation/flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [\[NeurIPS 2025\] Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation](vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)
- [\[ICCV 2025\] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](../../ICCV2025/image_generation/structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)
- [\[NeurIPS 2025\] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)
- [\[ICCV 2025\] PersonalVideo: High ID-Fidelity Video Customization without Dynamic and Semantic Degradation](../../ICCV2025/image_generation/personalvideo_high_id-fidelity_video_customization_without_dynamic_and_semantic_.md)

</div>

<!-- RELATED:END -->
