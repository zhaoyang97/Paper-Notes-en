---
title: >-
  [Paper Note] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss
description: >-
  [AAAI 2026][Medical Imaging][3D Medical Image Synthesis] This paper presents MAISI-v2, the first framework to introduce Rectified Flow into 3D medical image synthesis. By replacing DDPM with Rectified Flow…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "3D Medical Image Synthesis"
  - "Rectified Flow"
  - "Region-specific Contrastive Loss"
  - "Latent Diffusion Model"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 648b1066831b61a8
---

# MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss

**Conference**: AAAI 2026
**arXiv**: [2508.05772](https://arxiv.org/abs/2508.05772)  
**Code**: [GitHub](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/tree/main)  
**Area**: Medical Image Synthesis / Diffusion Models
**Keywords**: 3D Medical Image Synthesis, Rectified Flow, Region-specific Contrastive Loss, Latent Diffusion Model, Data Augmentation

## TL;DR

This paper presents MAISI-v2, the first framework to introduce Rectified Flow into 3D medical image synthesis. By replacing DDPM with Rectified Flow, it achieves a 33× speedup, and a novel region-specific contrastive loss is designed to improve conditioning fidelity for small regions such as tumors. The utility of synthesized data is validated on downstream tumor segmentation tasks.

## Background & Motivation

Medical image synthesis has significant application value in data augmentation, modality translation, anomaly simulation, and privacy-preserving data sharing. Diffusion models have become the dominant approach in image generation, yet their clinical deployment in 3D medical imaging is constrained by three bottlenecks:

**(1) Poor generalization**: Existing methods are typically trained on specific organs, modalities, or voxel spacings, and fail to accommodate the substantial variability in resolution, anatomical structures, and acquisition protocols present in clinical data. For instance, MedSyn can only generate images of fixed size $256^3$ at fixed spacing, and GenerateCT also supports only fixed dimensions.

**(2) Slow inference**: DDPM-based models typically require hundreds of iterative denoising steps. For 3D high-resolution volumes (e.g., $512^3$), this incurs prohibitive computational cost — MAISI with 1000-step DDPM requires 198 seconds (plus 15 seconds for VAE decoding), and a generation time exceeding 10 minutes severely limits practical utility.

**(3) Poor conditioning fidelity**: Conditional guidance mechanisms such as ControlNet perform well for 2D natural images, but frequently produce misalignment between generated outputs and input conditions (e.g., segmentation masks) in 3D medical imaging — a critical failure mode in medical applications, as data augmentation loses its value if generated tumor locations do not match the mask.

The MAISI framework has addressed the generalization issue (unified handling of diverse voxel spacings and anatomical structures), but inherits both the slow inference of DDPM and the weak conditional control of ControlNet. In 2D, ControlNet++ uses cycle-consistency loss to improve conditioning fidelity, but requires an additional inverse network, introducing pipeline complexity and error propagation.

The core idea is to replace DDPM with Rectified Flow for efficient deterministic sampling, and to design a region-specific contrastive loss that directly enhances ROI sensitivity without requiring additional networks.

## Method

### Overall Architecture

MAISI-v2 builds upon the MAISI architecture and consists of three components:
- **VAE**: Reuses the pretrained variational autoencoder from MAISI, compressing single-channel 3D volumes into 4-channel latent features with $4 \times 4 \times 4$ spatial compression (total compression ratio: 16); no fine-tuning is performed.
- **Rectified Flow LDM**: Replaces the original DDPM-based latent diffusion model, conditioned on voxel spacing.
- **ControlNet + Region-specific Contrastive Loss**: A control branch encodes the segmentation mask and injects it into the LDM; a contrastive loss is incorporated during training, conditioned on voxel spacing and segmentation mask.

### Key Designs

1. **Replacing DDPM with Rectified Flow**:

    - **Function**: Replaces the stochastic denoising process with a deterministic ODE transport.
    - **Mechanism**: Conventional diffusion models model curved or noisy trajectories through stochastic processes, requiring a large number of steps to travel from noise to data. Rectified Flow learns a time-dependent velocity field $v_t(x)$ that encourages straight-line transport between source distribution $\pi_0$ and target distribution $\pi_1$. The training objective is $\mathcal{L}_{\text{flow}} = \int_0^1 \mathbb{E}_{x_0, x_1, t} [\|v_t(x_t, c) - (x_1 - x_0)\|^2] dt$, where $x_t = (1-t)x_0 + tx_1$ is a linear interpolation. Straight-line transport enables high-quality sampling with far fewer steps.
    - **Design Motivation**: Rectified Flow has been validated for efficiency in Stable Diffusion 3 and Open Sora, but had not yet been introduced to 3D medical imaging.

2. **Three-stage Training Strategy**:

    - **Function**: Addresses batch size limitations and numerical stability issues arising from mixed training on 3D images of varying sizes.
    - **Mechanism**:
        - **Pre-training stage**: Training on $128^3$ low-resolution images with batch size 96 and learning rate 1e-3, completed in 1 day. Uniform size permits large batches and avoids NaN issues.
        - **Main training stage**: Full-resolution mixed training using **bucketed data parallelism** — images are grouped by size onto different GPUs ($128^3$: batch size 96; $256^2 \times 128$: batch size 24; $512^2 \times 768$: batch size 1), for 16,000 epochs over approximately 10 days.
        - **Fine-tuning stage**: Corrects data imbalance from the second stage by mixing all images with batch size 1 and sampling weights to balance dataset contributions, for 2,000 epochs over approximately 10 days.
    - **Design Motivation**: Naïve mixed training forces batch size to 1, and mixed-precision optimization is prone to NaN; bucketed parallelism accelerates training but introduces data imbalance; the three-stage progression addresses all these issues. Total training utilizes 64 A100 80GB GPUs over approximately three weeks.

3. **Region-specific Contrastive Loss**:

    - **Function**: Enhances the sensitivity of generated outputs to small-region conditions such as tumor masks.
    - **Mechanism**: Two versions are generated from the same noisy input — one conditioned on the original mask $c_{\text{orig}}$ and one on a perturbed mask $c_{\text{perturb}}$ (replacing the ROI label with the corresponding background label, e.g., pancreatic tumor label → pancreas label). The two outputs should differ within the ROI (reflecting the changed condition) and agree in the background (where the condition is unchanged).
        - **ROI sensitivity loss**: $\mathcal{L}_{\text{roi}} = -\min(\mathcal{D}_{\text{roi}}, \delta)$, encouraging large output discrepancy within the ROI, with upper bound $\delta=2$ to prevent gradient explosion.
        - **Background consistency loss**: $\mathcal{L}_{\text{bg}} = \|(G_\theta(x_t, c_{\text{orig}}) - G_\theta(x_t, c_{\text{perturb}})) \odot m^-\|_{1,m^-}$, using the complement of the dilated mask $m^- = 1 - \text{dilate}(m)$ to enforce background invariance.
        - Total objective: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{contrast}}(\mathcal{L}_{\text{roi}} + \mathcal{L}_{\text{bg}})$
    - **Design Motivation**: Weighted average loss alone (tumor weight 100) is insufficient to ensure small tumors appear clearly in generated images; the cycle-consistency approach of ControlNet++ requires an additional inverse network with associated error propagation; the contrastive loss requires no extra network and directly exploits condition perturbation to distinguish ROI from background.

4. **Memory-aware Strategy**:

    - **Function**: Adaptively selects where to compute the contrastive loss to accommodate varying GPU memory capacities.
    - **Mechanism**: For small-to-medium inputs, the loss is computed on the final output of ControlNet combined with the frozen diffusion model (higher spatial fidelity); for large inputs, it is computed on intermediate features of the ControlNet encoder (coarser but memory-efficient).
    - **Design Motivation**: 3D volumes at the $512^2 \times 768$ scale may exceed memory capacity even on 80GB GPUs.

### Loss & Training

- ControlNet training uses 8 A100 GPUs, AdamW optimizer, learning rate 5e-5 with polynomial decay, 60 epochs over approximately 2 days.
- $\lambda_{\text{contrast}}$ scheduling: set to 0.01 for the first 30 epochs to ensure tumor presence, then reduced to 0.001 for the remaining 30 epochs to correct body structure. The reverse order (small then large) results in tumors failing to appear.
- Quality check: the median HU values of major organs in generated CTs are verified to fall within physiological ranges (based on 5th/95th percentiles or 6-sigma bounds from training data); the final model achieves a 100% pass rate.

## Key Experimental Results

### Main Results — FID Comparison (AutoPET2023 OOD Dataset, $512^3$ volumes)

| Method | Steps | Time (s) | FID_avg↓ |
|--------|-------|----------|----------|
| HA-GAN | 1 | 1 | 13.595 |
| MedSyn (2-stage DDIM) | 50+20 | 100 | 24.709 |
| GenerateCT (2D EDM) | 25×201 | 89 | 10.757 |
| MAISI (DDPM) | 1000 | 198+15 | 2.441 |
| MAISI (DDIM) | 30 | 6+15 | 4.776 |
| **MAISI-v2 (Rectified Flow)** | **30** | **6+15** | **2.322** |

MAISI-v2 achieves FID on par with or better than MAISI (1000-step DDPM) using only 30 steps, realizing a **33× speedup** (198s → 6s for the LDM component).

### Ablation Study — Effect of Inference Steps

| Steps | 5 | 10 | 20 | 30 | 50 | 100 |
|-------|-----|------|------|------|------|------|
| FID_avg | 20.334 | 4.421 | 2.645 | 2.322 | 2.064 | 1.967 |

Diminishing returns are evident beyond 30 steps; 30 steps is adopted as the default setting.

### Downstream Segmentation Data Augmentation

| Method | Liver Tumor | Lung Tumor | Pancreas Tumor | Colon Tumor | Bone Lesion |
|--------|-------------|------------|----------------|-------------|-------------|
| Real Only | 0.662 | 0.581 | 0.433 | 0.449 | 0.504 |
| DiffTumor | 0.684(+2.2%) | — | 0.511(+7.9%) | — | — |
| MAISI | 0.688(+2.6%) | 0.635(+5.5%) | 0.482(+4.9%) | 0.485(+3.6%) | 0.539(+3.6%) |
| ControlNet only | 0.693(+3.0%) | 0.627(+4.7%) | 0.484(+5.1%) | 0.402(-4.7%) | 0.520(+1.6%) |
| **MAISI-v2** | **0.695(+3.3%)** | **0.655(+7.5%)** | **0.497(+6.4%)** | **0.491(+4.2%)** | **0.537(+3.3%)** |

ControlNet without contrastive loss yields a negative gain on colon tumor (−4.7%), which is reversed to +4.2% upon adding the contrastive loss; improvements are statistically significant for 4 of 5 tumor types.

### Key Findings

- Rectified Flow's deterministic transport is more inference-efficient than DDPM, but exhibits weaker conditioning fidelity for small or low-contrast regions — as deterministic trajectories reduce diversity and prediction errors accumulate during integration. The contrastive loss effectively compensates for this limitation.
- The scheduling order of $\lambda_{\text{contrast}}$ (large then small) is critical: a large initial value forces tumors to appear, while the subsequent smaller value corrects global body structure. The reverse order fails.
- Compared against video generation models (SVD, Open Sora 2.0) at comparable voxel counts, MAISI-v2 demonstrates superior inference efficiency ($1.3 \times 10^8$ voxels in 26s unconditional / 34s conditional vs. Open Sora 2.0 at $2.3 \times 10^8$ voxels in 162s).

## Highlights & Insights

- This work is the first to systematically introduce Rectified Flow into 3D medical image synthesis; a 33× speedup makes large-scale synthetic data generation practically feasible.
- The region-specific contrastive loss design is particularly elegant — it requires no additional network or generation pass, achieving ROI sensitivity enhancement solely through condition perturbation that constructs implicit treatment/control pairs.
- The three-stage training strategy (pre-training → bucketed parallelism → balanced fine-tuning) represents practical engineering experience for handling variable-size 3D data, offering high reference value to the community.
- The full release of code, model weights, and a GUI demo demonstrates NVIDIA's commitment to advancing community development.

## Limitations & Future Work

- Training and validation are limited to the CT modality; other modalities such as MRI and PET have not been covered.
- Downstream validation is restricted to segmentation tasks; applications such as detection, registration, and image translation remain unexplored.
- Large-scale volumes (e.g., $512^2 \times 768$) still require 40GB (inference) / 80GB (training) GPUs, placing them beyond broad accessibility.
- Training resource requirements are extremely high (64 A100 GPUs for three weeks), imposing a substantial barrier to reproducibility.
- Although the contrastive loss mitigates Rectified Flow's inherently weaker conditioning fidelity for small tumors compared to DDPM, the underlying mechanism warrants further investigation.

## Related Work & Insights

- The MAISI framework provides the complete infrastructure (VAE, ControlNet architecture, data pipeline) upon which this work builds; MAISI-v2 represents a successful incremental improvement.
- Rectified Flow is transferred cross-domain from Stable Diffusion 3 and Open Sora to 3D medical imaging, serving as an exemplary instance of diffusion model acceleration being adopted in a specialized domain.
- The cycle-consistency idea of ControlNet++ motivates the direction of conditioning fidelity improvement, but the present work replaces the complex inverse-network scheme with a simpler contrastive loss.
- DiffTumor's tumor inpainting approach, while differently formulated in task definition, provides a useful quantitative comparison baseline.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)
- [\[AAAI 2026\] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)
- [\[AAAI 2026\] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](../../NeurIPS2025/medical_imaging/surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [\[AAAI 2026\] EgoEMS: A High-Fidelity Multimodal Egocentric Dataset for Cognitive Assistance in Emergency Medical Services](egoems_a_high-fidelity_multimodal_egocentric_dataset_for_cognitive_assistance_in.md)

</div>

<!-- RELATED:END -->
