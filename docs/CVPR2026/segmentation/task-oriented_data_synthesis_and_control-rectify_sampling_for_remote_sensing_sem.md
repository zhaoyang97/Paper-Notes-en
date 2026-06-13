---
title: >-
  [Paper Note] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][Remote sensing semantic segmentation] This paper proposes the TODSynth framework, which achieves joint text-image-mask controlled remote sensing image synthesis via unified tri-modal attention i…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Remote sensing semantic segmentation"
  - "data synthesis"
  - "controllable generation"
  - "diffusion models"
  - "flow matching"
date: 2026-05-08
content_hash: 1be0ed387f672e10
---

# Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2512.16740](https://arxiv.org/abs/2512.16740)  
**Code**: [GitHub](https://github.com/Yunkai-Yang/crfm)  
**Area**: Segmentation / Remote Sensing
**Keywords**: Remote sensing semantic segmentation, data synthesis, controllable generation, diffusion models, flow matching

## TL;DR

This paper proposes the TODSynth framework, which achieves joint text-image-mask controlled remote sensing image synthesis via unified tri-modal attention in MM-DiT, and introduces Control-Rectify Flow Matching (CRFM), a novel sampling-stage method that dynamically adjusts the generation trajectory using semantic loss from a downstream segmentation model. The synthesized data improves mIoU by 4.14% on FUSU-4k and 2.08% on LoveDA.

## Background & Motivation

**Background**: Remote sensing semantic segmentation is a fundamental task for land-use classification and environmental monitoring, yet constructing large-scale pixel-level annotated datasets is prohibitively costly. Diffusion model-based data synthesis has emerged as a promising approach for augmenting training sets, with methods such as ControlNet enabling semantic mask-conditioned image generation.

**Limitations of Prior Work**: (1) **Immature control schemes**: DiT-based generative models (e.g., SD v3.5) substantially outperform UNet-based architectures, yet effectively injecting semantic mask control into DiT remains an open problem. Adapter-based cross-attention control suffers from low efficiency and modality conflicts. (2) **Unstable sampling quality**: Even with reasonable control schemes, the stochasticity inherent in diffusion/flow matching sampling can cause generated images to locally deviate from mask constraints (semantic drift), reducing the utility of synthesized data for downstream tasks. (3) **Limited post-hoc filtering**: Existing approaches (e.g., CLIP-based scoring, FreeMask adaptive filtering) are remedies applied after generation; in complex scenes or few-shot categories, aggressive filtering discards valuable annotations.

**Key Challenge**: A fundamental tension exists between the stochasticity of generative models and the deterministic semantic control required by downstream tasks. The large domain gap in remote sensing imagery, the absence of DiT models pretrained on remote sensing data, and the scarcity of fine-grained textual descriptions further exacerbate this conflict.

**Goal**: (1) Identify an effective DiT control scheme for remote sensing mask-to-image (M2I) synthesis; (2) Correct semantic drift during the sampling process (rather than post hoc), improving the task relevance of synthesized data.

**Key Insight**: The authors observe that directly optimizing the latent variable $z_t$ leads to mode collapse, whereas optimizing the velocity field $v_\Theta$ enables stable and continuous correction. Based on this insight, gradient signals from a downstream segmentation model are injected during the early high-plasticity phase of flow matching to rectify the generation trajectory.

**Core Idea**: Employ tri-modal joint attention for architecture-level control, and apply velocity field correction (CRFM) guided by downstream segmentation loss gradients in the early sampling phase, achieving task-oriented remote sensing data synthesis.

## Method

### Overall Architecture

TODSynth consists of three stages: (1) **Training**: An MM-DiT model with unified tri-modal attention (Tri-Attention) is trained on top of SD v3.5, conditioned on text and semantic masks, to generate remote sensing images. (2) **Sampling**: CRFM corrects the velocity field in the early sampling steps using cross-entropy loss gradients from a pretrained segmentation model. (3) **Downstream Training**: Synthesized data is mixed with real data to train the segmentation model.

### Key Designs

1. **Unified Tri-Modal Attention (Tri-Attention)**:

    - **Function**: Enables deep fusion of text, image, and semantic mask modalities within the DiT architecture.
    - **Mechanism**: A third modality stream processing the mask sequence $h^m$ is introduced alongside the original text-image dual-modal joint attention in MM-DiT. Each modality has independent $W_q, W_k, W_v$ projection matrices, and all tokens are concatenated for attention computation within a single attention operation: $h_o^t, h_o^z, h_o^m = \text{Attn}([h^t W_q^t, h^z W_q^z, h^m W_q^m], ...)$. This allows mask information to directly interact with text embeddings, enhancing global semantic understanding.
    - **Design Motivation**: In the mask-adapter approach, masks do not fuse with text, leading to underutilization of semantic information and static mask representations throughout denoising. In the Siamese approach, purely mask-conditioned M2I lacks local textual descriptions, weakening the advantages of decoupled processing. Tri-Attention achieves tri-modal cross-attention in the most concise manner.

2. **Control-Rectify Flow Matching (CRFM)**:

    - **Function**: Dynamically corrects the generation trajectory during sampling to ensure synthesized images more faithfully adhere to semantic mask constraints.
    - **Mechanism**: In the early (high-plasticity) steps of flow matching sampling, the current state $z_t$ and predicted velocity field $v^P$ are used to estimate the final output $z_0^t = z_t - \sigma_t v^P$. The VAE decoder yields a pre-synthesized image $x_0^t$, which is fed into a pretrained segmentation network to compute the cross-entropy loss $\mathcal{L}_{CE}(\mathcal{S}(x_0^t), C^m)$. The gradient with respect to the velocity field yields a correction vector $v_{rec}' = -\nabla_{v_t} \mathcal{L}_{CE}$. The final corrected velocity field is $v' = v^P + \alpha \cdot v_{rec}'$.
    - **Design Motivation**: (1) Directly optimizing $z_t$ causes mode collapse (loss of sample diversity), whereas optimizing the velocity field indirectly updates $z_t$ via ODE integration, providing more stable correction. (2) Correction is applied only in the early steps, where stochasticity is high and plasticity is strong, and the impact of segmentation model prediction errors remains controllable at coarse-grained adjustments. Late-stage correction risks amplifying segmentation model errors and inducing adversarial perturbations.

3. **Full-Parameter Fine-Tuning of Image and Mask Branches**:

    - **Function**: Bridges the domain gap caused by the absence of DiT models pretrained on remote sensing data.
    - **Mechanism**: Since no DiT-based generative model pretrained on remote sensing data exists, the frozen backbone + adapter strategy yields limited improvements. This work opts for full-parameter fine-tuning of the image and mask branches to fully adapt the model to the distinctive distribution of remote sensing imagery.
    - **Design Motivation**: Remote sensing images differ substantially from natural images in terms of nadir viewpoint, spectral characteristics, and scale variation, necessitating more thorough domain adaptation.

### Loss & Training

The training stage employs the standard Rectified Flow loss (MSE loss on velocity field prediction). The correction strength in CRFM is governed by the hyperparameter $\alpha$. Post-hoc filtering follows FreeMask's pixel-level filtering strategy. The synthesized-to-real data ratio is 3:1. The model is trained for 200K steps on 8×RTX 4090 GPUs at 512×512 resolution using the AdamW optimizer with a learning rate of $10^{-5}$.

## Key Experimental Results

### Main Results

| Method | Post-processing | Synth/Real | FUSU-4k OA | FUSU-4k mIoU | FUSU-4k mAcc |
|--------|----------------|------------|------------|--------------|--------------|
| Baseline (real only) | - | - | 74.27 | 45.27 | 56.44 |
| ControlNet (SD v1.5) | × | ×10 | 73.85 | 45.13 | 56.77 |
| FreeMask | FM | ×5 | 74.23 | 45.83 | 56.29 |
| SynthEarth | CLIP | ×5 | 75.35 | 47.53 | 58.91 |
| SD v3.5 (Tri-Attn) | FM | ×3 | 75.41 | 48.57 | 61.67 |
| **TODSynth (Ours)** | FM | **×3** | **75.66** | **49.41** | **63.27** |

On the LoveDA dataset, TODSynth achieves OA +1.60% / mIoU +2.08% / mAcc +2.22% over the baseline.

### Ablation Study

Control strategy comparison (FUSU-4k):

| Method | OA | mIoU | mAcc |
|--------|----|------|------|
| ControlNet (SD v1.5) | 73.85 | 45.13 | 56.77 |
| Mask-adapter | 74.94 | 47.41 | 59.62 |
| Siamese MM-attention | 74.94 | 48.46 | 61.44 |
| **Tri-Attention** | **75.41** | **48.57** | **61.67** |

CRFM step ablation (total steps = 23):

| CRFM Steps | mIoU | mAcc | FID |
|------------|------|------|-----|
| 0 (no correction) | 48.57 | 61.67 | 35.85 |
| 2 | 48.80 | 61.05 | 34.86 |
| **4** | **49.41** | **63.27** | 38.65 |
| 6 | 48.74 | 61.30 | 66.95 |

### Key Findings
- **DiT >> UNet**: MM-DiT-based controllable generation substantially outperforms UNet-based ControlNet and FreeMask, even compared to SynthEarth, a dedicated remote sensing generative foundation model.
- **CRFM is effective but requires step control**: Four correction steps achieves the best mIoU/mAcc; excessive correction steps (6 steps) causes FID to spike dramatically (66.95), indicating that over-correction degrades image quality.
- **Pixel-level filtering > image-level filtering**: FreeMask's pixel-level filtering significantly outperforms CLIP's image-level filtering, with finer-grained selection being more appropriate for segmentation tasks.
- TODSynth achieves superior performance with **fewer synthesized samples** (×3 vs. ×5/×10), demonstrating the efficiency of task-oriented synthesis.
- Direct latent variable optimization leads to mode collapse, validating the necessity of correcting the velocity field rather than the latent variable.

## Highlights & Insights
- **Velocity field correction vs. latent variable optimization**: This is the paper's most central insight. Within the flow matching framework, optimizing the velocity field rather than directly modifying the latent variable avoids mode collapse and provides stable trajectory correction—a principle generalizable to other conditional generation tasks.
- **Task-feedback-driven sampling**: Rather than filtering after generation, downstream task signals are used to guide sampling during generation—a paradigm shift from "select the best after generation" to "correct during generation."
- **Early-stage plasticity window**: The early steps of flow matching are identified as the optimal window for correction, while late-stage correction is found to be harmful. This is consistent with the observation in diffusion models that early steps determine semantics and later steps determine fine details.
- The concise implementation of Tri-Attention validates the applicability of "unified fusion over decoupled processing" in remote sensing M2I scenarios.

## Limitations & Future Work
- CRFM relies on the quality of the pretrained segmentation model; if the segmentation model itself performs poorly on the target domain, the correction signal may be inaccurate.
- The hyperparameters $\alpha$ and CRFM step count currently require manual tuning; adaptive scheduling strategies may be more robust.
- Validation is limited to two remote sensing datasets; whether the approach transfers to other annotation-scarce domains such as medical imaging remains to be verified.
- The 512×512 resolution may be insufficient for high-resolution remote sensing imagery, and higher-resolution generation warrants further exploration.
- Full-parameter fine-tuning incurs high computational cost (8×4090); the effectiveness of lightweight alternatives such as LoRA has not been compared.

## Related Work & Insights
- **vs. ControlNet**: A UNet-based control scheme with limited effectiveness in the remote sensing domain. The proposed Tri-Attention in DiT substantially outperforms it.
- **vs. FreeMask**: A post-hoc filtering approach that is complementary to CRFM. This work demonstrates that stacking CRFM on top of FreeMask yields further improvements.
- **vs. SynthEarth**: A remote sensing generative foundation model using CLIP-score filtering. TODSynth achieves better results with fewer synthesized samples (×3 vs. ×5).
- **vs. training-free L2I editing**: Methods that directly optimize latent variables lead to mode collapse. Velocity field correction provides a superior alternative.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The CRFM velocity field correction is a novel idea; the "correct during generation" paradigm deserves attention.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies on control strategies and CRFM hyperparameters are thorough, though evaluation on only two datasets is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear and the experimental framework is complete.
- **Value**: ⭐⭐⭐⭐ Practically valuable for remote sensing data synthesis; the CRFM idea is transferable to other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[NeurIPS 2025\] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis](../../NeurIPS2025/segmentation/fast_foreground-aware_diffusion_with_accelerated_sampling_trajectory_for_segment.md)
- [\[CVPR 2026\] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)
- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](../../AAAI2026/segmentation/s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[CVPR 2026\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)

</div>

<!-- RELATED:END -->
