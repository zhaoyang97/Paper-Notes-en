---
title: >-
  [Paper Note] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis
description: >-
  [CVPR 2026][3D Vision][grasp generation] AffordGrasp proposes a diffusion-based cross-modal framework that generates physically plausible and semantically consistent hand grasp poses from text instructions and object point clouds, via affordance-guided latent diffusion and a Distribution Adjustment Module (DAM), significantly outperforming existing methods on four benchmarks.
tags:
  - CVPR 2026
  - 3D Vision
  - grasp generation
  - affordance
  - cross-modal diffusion
  - hand-object interaction
  - semantic instruction
date: 2026-05-08
content_hash: 03e66842bacb8bdf
---

# AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis

**Conference**: CVPR 2026
**arXiv**: [2603.08021](https://arxiv.org/abs/2603.08021)
**Code**: [Project Page](https://affordgrasp.github.io/)
**Area**: 3D Vision / Hand-Object Interaction
**Keywords**: grasp generation, affordance, cross-modal diffusion, hand-object interaction, semantic instruction

## TL;DR
AffordGrasp proposes a diffusion-based cross-modal framework that generates physically plausible and semantically consistent hand grasp poses from text instructions and object point clouds, via affordance-guided latent diffusion and a Distribution Adjustment Module (DAM), significantly outperforming existing methods on four benchmarks.

## Background & Motivation
**Background**: Semantic grasp generation aims to synthesize hand poses for object interaction conditioned on user instructions, which is a critical capability for AR/VR and embodied intelligence.

**Limitations of Prior Work**:
   - A large modality gap exists between 3D geometry and natural language, making fine-grained geometry–semantic alignment (e.g., distinguishing "grasp the handle" from "grasp the rim") difficult to achieve through direct fusion;
   - Existing diffusion pipelines lack explicit spatial and semantic constraints, frequently producing physically implausible or semantically inconsistent grasps.

**Key Challenge**: How can generated grasp poses precisely correspond to the interaction intent described by a language instruction while remaining physically feasible?

**Key Insight**: Object affordance is introduced as a cross-modal bridge, linking linguistic semantics with 3D geometry through affordance regions, supplemented by a Distribution Adjustment Module that enforces physical and semantic consistency after sampling.

**Core Idea**: Affordance-driven latent diffusion + Distribution Adjustment Module = physical plausibility + semantic precision.

## Method

### Overall Architecture
Input: object point cloud $P_g$ + text instruction $I$ → Affordance Generator predicts affordance map $P_a$ → multimodal encoding fusion $f = \{f_I, f_{pg}, f_{pa}\}$ → latent diffusion model generates hand latent code → DAM refinement → MANO layer outputs hand mesh $h_m$.

### Key Designs
1. **Affordance Generator**:

    - **Function**: Predicts a per-point affordance probability map on the object point cloud conditioned on the instruction.
    - **Mechanism**: Built on the LASO architecture, initially trained on AffordPose, then extended to OakInk and GRAB via a self-training loop. Focal Loss + Dice Loss is used to handle positive–negative sample imbalance: $\mathcal{L} = \mathcal{L}_{\text{focal}} + \lambda_1 \mathcal{L}_{\text{dice}}$
    - **Design Motivation**: Affordance regions provide an explicit intermediate representation that decouples the spatial information of "where to grasp" from the language instruction, reducing the difficulty of cross-modal fusion.

2. **Cross-Modal Latent Diffusion Model**:

    - **Function**: Learns the conditional distribution in the latent space of hand poses.
    - **Mechanism**:
        - A pretrained VAE encodes hand mesh vertices $h_v^{gt} \in \mathbb{R}^{778 \times 3}$ into a compact latent code $h_z$
        - Forward diffusion: $z^t = \sqrt{\alpha_t} z^0 + \sqrt{1 - \alpha_t} \epsilon$
        - A conditional U-Net is trained to predict noise: $L_{LDM} = \mathbb{E}\|\epsilon - \epsilon_\theta(z^t, f, t)\|_2^2$
        - Conditioning features $f = \{f_I, f_{pg}, f_{pa}\}$ are encoded by RoBERTa and two independent PointNets, respectively
    - **Design Motivation**: Operating in latent space is more efficient and better preserves the spatial structure of hand poses.

3. **Distribution Adjustment Module (DAM)**:

    - **Function**: Refines the latent code after diffusion sampling to enforce physical constraints and semantic alignment.
    - **Mechanism**:
        - Recovers the latent hand representation from noise prediction: $\hat{h}_z = \frac{1}{\sqrt{\alpha_t}}(z^t - \sqrt{1-\alpha_t}\epsilon_\theta(z^t, f, t))$
        - Fuses spatial features: $f_{\text{spatial}} = \text{Norm}(f_{pg} + f_{pa}) + \hat{h}_z$
        - Multi-head attention interacts with the instruction: $f_{\text{align}} = \text{Attention}(f_I, f_{\text{spatial}}, f_{\text{spatial}}) + f_I$
        - Dual residual refinement: $\tilde{h}_z = \text{Norm}(\text{MLP}(f_{\text{align}}) + \hat{h}_z)$
    - **Design Motivation**: Denoised outputs from diffusion models may be insufficiently precise regarding contact constraints and semantic details. DAM corrects these in a single forward pass as a lightweight post-processing module, avoiding the high computational cost of gradient-based sampling correction.

### Loss & Training
- Two-stage training: the diffusion model is trained first (DAM frozen), then the diffusion model is frozen and DAM is trained.
- DAM loss: $\mathcal{L} = \mathcal{L}_{\text{recon}}(h_v, h_p, h_v^{gt}, h_p^{gt}) + \lambda_2 \mathcal{L}_{\text{physical}}(h_m, h_m^{gt}, P_g)$
- Reconstruction loss covers MANO parameter and vertex alignment; physical loss penalizes interpenetration and contact inconsistency.

## Key Experimental Results

### Main Results

| Dataset | Method | Penetration Vol.↓ | Displacement↓ | Contact Rate↑ | Semantic Acc.↑ |
|--------|------|----------|-------|---------|-----------|
| OakInk | FastGrasp | 7.88 | 2.27 | 88% | 78.05% |
| OakInk | **AffordGrasp** | **7.31** | **1.43** | **98%** | **80.08%** |
| GRAB | FastGrasp | 4.61 | 1.20 | 94% | 61.50% |
| GRAB | **AffordGrasp** | **3.06** | 1.66 | **94%** | **62.50%** |
| HO-3D (OOD) | D-VQVAE | 13.12 | 2.33 | 95% | 64.00% |
| HO-3D (OOD) | **AffordGrasp** | **7.38** | **2.33** | **97%** | **72.00%** |

### Ablation Study

| Configuration | Penetration Vol.↓ | Contact Rate↑ | Semantic Acc.↑ | Note |
|------|----------|---------|-----------|------|
| w/o affordance | 8.27 | 97% | 76.56% | Removing affordance increases penetration |
| w/o DAM | 8.12 | 97% | 79.11% | Removing DAM slightly reduces semantics |
| Full pipeline | **7.31** | **98%** | **80.08%** | Both modules synergize optimally |

### Key Findings
- Outstanding cross-domain generalization: the model trained on GRAB substantially outperforms baselines in zero-shot settings on HO-3D and AffordPose.
- Affordance regions effectively reduce penetration volume (object–hand collision), demonstrating the importance of spatial guidance.
- DAM's dual residual mechanism preserves the core structure of the diffusion output while effectively correcting local details.

## Highlights & Insights
- The concept of **affordance as a cross-modal bridge** is concise and effective, avoiding the instability of multi-round VLM inference.
- The self-training annotation pipeline addresses the scarcity of affordance-labeled data.
- DAM, as a lightweight post-processing module, is transferable to other conditional generation tasks.

## Limitations & Future Work
- Physical priors (gravity, friction) are not explicitly modeled, which may limit performance in certain real-world scenarios.
- AffordPose is excluded from training due to the absence of MANO parameters, limiting object diversity.
- Multi-step DDIM sampling is still required at inference, leaving room for improvement in real-time applicability.

## Related Work & Insights
- Compared to SemGrasp, the method avoids 2D projection occlusion issues by operating directly in 3D space.
- The DAM design is conceptually similar to ControlNet's post-processing refinement, but requires no retraining of the base model.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of affordance guidance and DAM is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, in-domain and out-of-domain evaluation, extensive ablations, and physics simulation validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with high-quality figures and tables.
- Value: ⭐⭐⭐⭐ Practically advances semantic grasping for embodied intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](cmhanet_a_crossmodal_hybrid_attention_network_for.md)
- [\[CVPR 2026\] Affostruction: 3D Affordance Grounding with Generative Reconstruction](affostruction_3d_affordance_grounding_with_generative_reconstruction.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)
- [\[CVPR 2026\] AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers](affordmatcher_affordance_learning_in_3d_scenes_from_visual_signifiers.md)
- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](../../AAAI2026/3d_vision/stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)

</div>

<!-- RELATED:END -->
