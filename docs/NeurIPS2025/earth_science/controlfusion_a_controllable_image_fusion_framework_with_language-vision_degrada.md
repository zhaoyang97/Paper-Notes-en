---
title: >-
  [Paper Note] ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts
description: >-
  [NeurIPS 2025][Infrared-visible fusion] This paper proposes ControlFusion, a controllable infrared-visible image fusion framework based on language-vision degradation prompts. It employs a physics-driven degradation imaging model to simulate compound degradations, and uses a prompt-modulated network to perform dynamic restoration and fusion, achieving comprehensive state-of-the-art performance under both real-world and compound degradation scenarios.
tags:
  - NeurIPS 2025
  - Infrared-visible fusion
  - degradation restoration
  - language-vision prompts
  - CLIP
  - controllable fusion
date: 2026-05-08
content_hash: 33cb39d472aa291e
---

# ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts

**Conference**: NeurIPS 2025
**arXiv**: [2503.23356](https://arxiv.org/abs/2503.23356)
**Code**: [https://github.com/Linfeng-Tang/ControlFusion](https://github.com/Linfeng-Tang/ControlFusion)
**Area**: Image Fusion / Multimodal
**Keywords**: Infrared-visible fusion, degradation restoration, language-vision prompts, CLIP, controllable fusion

## TL;DR
This paper proposes ControlFusion, a controllable infrared-visible image fusion framework based on language-vision degradation prompts. It employs a physics-driven degradation imaging model to simulate compound degradations, and uses a prompt-modulated network to perform dynamic restoration and fusion, achieving comprehensive state-of-the-art performance under both real-world and compound degradation scenarios.

## Background & Motivation

**State of the Field**: Infrared-visible image fusion (IVIF) integrates thermal information and texture details, with broad applications in security, military detection, and autonomous driving. Existing methods encompass CNN/AE/GAN/Transformer/diffusion model architectures.

**Limitations of Prior Work**:
   - Degradation-robust methods rely on **simple data construction strategies**, resulting in a domain gap between synthetic and real images.
   - Existing methods handle only **single-type degradations** and cannot address compound degradations in real-world scenarios (e.g., simultaneous low-light, noise, and blur).
   - There is a lack of **degradation severity modeling**, causing sharp performance drops as degradation intensifies; personalized user requirements also cannot be accommodated.

**Root Cause**: Real-world degradations vary widely in type and severity combinations, making fixed fusion networks inflexible.

**Starting Point**: Construct a physics-based degradation imaging model to reduce the synthetic-real domain gap; use language prompts for explicit modeling of degradation type and severity; and employ a visual adapter for automatic degradation awareness.

**Core Idea**: Language prompts specify degradation type/severity + visual adapter automatically perceives degradation → dynamic modulation of feature restoration and fusion.

## Method

### Overall Architecture
Two-stage training: Stage I aligns text embeddings with visual embeddings (training the visual adapter); Stage II trains the end-to-end restoration-fusion network. At inference, two modes are supported: user-provided text prompts to specify degradation, or automatic degradation perception via the visual adapter.

### Key Designs

1. **Physics-Driven Degradation Imaging Model**:

    - Function: Simulates degradation for infrared/visible images separately, constructing the DDL-12 training dataset (12 degradation types × 4 severity levels, approximately 48,000 training pairs).
    - Mechanism: $D_m = \mathcal{P}_s(\mathcal{P}_w(\mathcal{P}_i(I_m)))$, employing three nested degradation layers — illuminance degradation (Retinex theory, $\gamma \in [0.5, 3]$), weather degradation (atmospheric scattering model, including rain and fog), and sensor degradation (noise + motion blur + contrast reduction).
    - Design Motivation: Physics-based degradation simulation is closer to real-world scenarios than random degradation. Infrared and visible modalities exhibit different degradation types (infrared primarily suffers from stripe noise and low contrast; visible images mainly suffer from low-light/overexposure/rain/fog), necessitating separate modeling.

2. **Spatial-Frequency Collaborative Visual Adapter (SFVA)**:

    - Function: Automatically extracts degradation descriptor embeddings from degraded images, replacing manual text input.
    - Mechanism: A frequency branch applies FFT to extract frequency-domain degradation priors ($F_{fre}^m = \sum_{x,y} D_m(x,y) e^{-j2\pi(\frac{ux}{W} + \frac{vy}{H})}$); a spatial branch employs CNN to extract spatial features; the two branches are concatenated and linearly projected to obtain the visual embedding $p_{vis}$.
    - Design Motivation: Different degradations exhibit distinct frequency-domain characteristics (e.g., noise concentrates in high frequencies; blur causes high-frequency attenuation), which the frequency branch effectively captures. MSE and cosine similarity losses ensure semantic alignment between $p_{vis}$ and $p_{text}$.

3. **Prompt-Modulated Module (PMM)**:

    - Function: Dynamically modulates fusion features according to degradation prompts.
    - Mechanism: An MLP generates scaling parameter $\gamma_p$ and shift parameter $\beta_p$ from prompt $p$: $\hat{F}_f = (1 + \gamma_p) \odot F_f + \beta_p$, implementing FiLM-style feature modulation.
    - Design Motivation: Different degradations require distinct feature enhancement strategies; learnable affine transformations enable conditional restoration.

4. **Cross-Modal Cross-Attention Fusion Layer**:

    - Function: Exchanges Query vectors between modalities to facilitate cross-modal feature interaction.
    - Mechanism: $F_f^{ir} = \text{softmax}(\frac{Q_{vi}K_{ir}}{\sqrt{d_k}})V_{ir}$, using the visible-domain Query to retrieve infrared Key-Value pairs, and vice versa.
    - Design Motivation: Cross-modal Query exchange promotes spatially aligned complementary information fusion.

### Loss & Training
- Stage I: $\mathcal{L}_I = \lambda_1 \|p_{vis} - p_{text}\|^2 + \lambda_2 (1 - \cos(p_{vis}, p_{text}))$
- Stage II: Weighted combination of intensity loss + SSIM loss + maximum gradient loss + color consistency loss.

## Key Experimental Results

### Main Results (Standard Fusion Benchmarks)

| Method | MSRS-VIF | LLVIP-VIF | RoadScene-VIF | FMB-VIF |
|--------|----------|-----------|---------------|---------|
| Text-DiFuse | 0.850 | 0.883 | 0.683 | 0.793 |
| ControlFusion | **0.927** | **0.968** | **0.817** | **0.872** |

### Performance under Degradation (CLIP-IQA / MUSIQ Metrics)
ControlFusion achieves the best or second-best results across all degradation types (blur, rain, low-light, overexposure, noise, stripe noise, low contrast) and compound degradations. The advantage is particularly pronounced under compound degradations (e.g., simultaneous low-light, noise, and rain).

### Ablation Study

| Configuration | EN | SD | VIF | Qabf |
|---------------|----|----|-----|------|
| Full model | Best | Best | Best | Best |
| w/o SFVA (text only) | Significant drop | - | - | - |
| w/o PMM | Notable drop | - | - | - |
| w/o physics degradation model | Poor real-world generalization | - | - | - |

### Key Findings
- Visual embeddings generated by SFVA are highly aligned with manual text embeddings, enabling fully automated deployment.
- The physics-driven degradation model significantly reduces the synthetic-real domain gap.
- Performance remains stable across all 4 severity levels, without sharp degradation under severe conditions.

## Highlights & Insights
- **Dual-channel language-vision degradation description paradigm**: Text prompts enable user controllability while the visual adapter enables automation, with both channels semantically aligned — this paradigm is transferable to any conditional image restoration task.
- **FiLM-style modulation for degradation adaptation**: Simple affine transformations achieve powerful conditional effects, avoiding the need to train dedicated models for each degradation type.
- **Physics-driven degradation simulation**: The combination of Retinex, atmospheric scattering, and sensor noise modeling is more reliable than purely data-driven approaches.

## Limitations & Future Work
- Text prompt templates are relatively fixed, limiting flexibility.
- The discrete 4-level severity quantization may lack sufficient granularity.
- SFVA's degradation awareness depends on the alignment quality achieved in Stage I.
- Generalizability to other multimodal fusion tasks (e.g., MRI-CT, multispectral) has not been validated.

## Related Work & Insights
- **vs. Text-IF**: Text-IF requires manually crafted text prompts for each scene; ControlFusion achieves automation via SFVA.
- **vs. Text-DiFuse**: Text-DiFuse is diffusion-based but does not address compound degradations; ControlFusion explicitly models combinations of multiple degradation types.
- **vs. DA-CLIP**: DA-CLIP achieves degradation awareness by fine-tuning CLIP, but targets natural images only; ControlFusion is designed specifically for multimodal imagery.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual-channel language-vision degradation control and a physics-based degradation model is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 datasets, 7 degradation types, compound degradations, and ablation studies — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-organized with complete formulations.
- Value: ⭐⭐⭐⭐ Practically significant for industrial deployment; handling real-world degradations is a critical pain point.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics](../../AAAI2026/earth_science/mdaif_robust_one-stop_multi-degradation-aware_image_fusion_with_language-driven_.md)
- [\[NeurIPS 2025\] Predicting Public Health Impacts of Electricity Usage](predicting_public_health_impacts_of_electricity_usage.md)
- [\[NeurIPS 2025\] Adaptive Online Emulation for Accelerating Complex Physical Simulations](adaptive_online_emulation_for_accelerating_complex_physical_simulations.md)
- [\[NeurIPS 2025\] Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning](reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi.md)
- [\[NeurIPS 2025\] A Probabilistic U-Net Approach to Downscaling Climate Simulations](a_probabilistic_unet_approach_to_downscaling_climate_simulat.md)

<!-- RELATED:END -->
