---
title: >-
  [Paper Note] DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness
description: >-
  [3D Vision] This paper proposes the Direct Simulation Optimization (DSO) framework, which uses stability feedback from a (non-differentiable) physics simulator as a reward signal to fine-tune 3D generators via DPO or the newly proposed DRO objective, enabling feed-forward generation of physically self-supporting 3D objects without test-time optimization.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: fcd4188acde4f2bb
---

# DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.22677](https://arxiv.org/abs/2503.22677)
- **Code**: [ruiningli.com/dso](https://ruiningli.com/dso)
- **Area**: 3D Vision / Physical Stability
- **Keywords**: 3D generation, physical simulation feedback, DPO/DRO alignment, diffusion model fine-tuning, self-supporting
- **Authors**: Ruining Li, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi (Oxford VGG)

## TL;DR

This paper proposes the Direct Simulation Optimization (DSO) framework, which uses stability feedback from a (non-differentiable) physics simulator as a reward signal to fine-tune 3D generators via DPO or the newly proposed DRO objective, enabling feed-forward generation of physically self-supporting 3D objects without test-time optimization.

## Background & Motivation

State-of-the-art 3D generators (e.g., TRELLIS, Hunyuan3D 2.0) prioritize geometric and appearance quality while neglecting physical constraints—particularly self-supporting stability under gravity. Experiments show that even when the input image depicts a stable object, TRELLIS still generates unstable 3D models approximately 30% of the time.

Existing methods (e.g., Atlas3D, PhysComp) rely on differentiable physics simulators to optimize geometry at test time, but differentiable simulators are slow, numerically unstable, and prone to local optima. The core motivation of this paper is: can a generator be trained to produce physically stable objects directly, eliminating the need for additional optimization at inference time?

The nature of stability is discrete (either stable or not), making it unsuitable for direct gradient-based optimization. However, physics simulators can readily determine stability—a property that naturally suits reward-based learning paradigms.

## Method

### Overall Architecture

The DSO framework proceeds in three steps:
1. **Data Generation**: Sample a large number of 3D objects from images using the reference model $p_{\text{ref}}$.
2. **Simulation Annotation**: Apply a non-differentiable physics simulator (MuJoCo) to assign binary stability labels $o(\mathbf{x}_0) \in \{0, 1\}$ to each generated object.
3. **Alignment Training**: Fine-tune the generator using DPO or DRO objectives to reinforce stable samples and suppress unstable ones.

### Core Optimization Objective

The original objective maximizes the expected stability of generated objects while constraining the updated model from deviating from the reference via KL divergence:

$$\max_\theta \mathbb{E}[o(\mathbf{x}_0)] - \beta \mathbb{D}_{\text{KL}}[p_\theta \| p_{\text{ref}}]$$

Since $o$ is non-differentiable and direct RL optimization is prohibitively expensive due to the cost of decoding 3D representations, alternative approaches are required.

### Direct Reward Optimization (DRO) — Proposed in This Work

Using a reparameterization trick, the reward signal $o(\mathbf{x}_0)$ is expressed in terms of the optimal reverse diffusion process, yielding a training objective that requires no paired preference data:

$$\mathcal{L}_{\text{DRO}} = -T \mathbb{E}\left[ w(t)(1 - 2o(\mathbf{x}_0)) \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2_2 \right]$$

Intuitively: for stable samples ($o=1$), the coefficient is $-1$, encouraging better denoising; for unstable samples ($o=0$), the coefficient is $+1$, causing the model to "unlearn" denoising those samples.

**Advantages of DRO over DPO**:
- Does not require paired preference data
- Does not require querying the reference model $\epsilon_{\text{ref}}$ during training
- Converges faster and achieves better alignment

### Direct Preference Optimization (DPO)

As a baseline comparison, the Diffusion-DPO objective can also be applied, requiring paired stable/unstable 3D models generated from the same image:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[ \log \sigma\left( -\beta T w(t) \left( \Delta_\theta^w - \Delta_\theta^l \right) \right) \right]$$

### Self-Improving Data Construction

A key innovation is that training data is derived entirely from the generator's own outputs, with no real 3D objects required.
- The reference model generates 3D models from rendered images in Objaverse (13k objects × 6 views × 4 generations = 312k models).
- MuJoCo simulation is used, with a tilt angle < 20° classifying an object as stable.
- Synthetic 2D images (descriptions generated by GPT-4 → images generated by FLUX) can fully replace rendered images.

### Training Details

- Base model: TRELLIS (rectified flow transformer)
- Only the linear layers of the first coarse geometry transformer are fine-tuned (LoRA rank 64)
- AdamW optimizer, batch size 48, 4× A100
- DRO training: 4,000 steps; DPO training: 8,000 steps

## Key Experimental Results

### Main Results (Table 1: PhysComp Evaluation Set)

| Method | % Stable ↑ | Rot. ↓ | CD ↓ | F-Score ↑ |
|--------|-----------|--------|------|-----------|
| TRELLIS (baseline) | 85.1 | 14.14° | 0.0485 | 73.12 |
| Atlas3D | 69.4 | 32.86° | — | — |
| DSO + DPO | 95.1 | 5.42° | 0.0480 | 73.62 |
| **DSO + DRO** | **99.0** | **1.88°** | **0.0440** | **76.17** |

On the hard subset (11 unstable objects):

| Method | % Stable ↑ | % Output ↑ | Rot. ↓ |
|--------|-----------|-----------|--------|
| TRELLIS | 54.5 | 100 | 39.18° |
| PhysComp | 80.3 | 46.2 | 18.14° |
| DSO + DPO | 82.6 | 100 | 16.83° |
| **DSO + DRO** | **95.5** | **100** | **5.58°** |

### Ablation Study (Tables 2 & 3)

| Method | % Stable ↑ | Rot. ↓ | CD ↓ | F-Score ↑ |
|--------|-----------|--------|------|-----------|
| TRELLIS | 85.1 | 14.14° | 0.0485 | 73.12 |
| TRELLIS + SFT | 89.5 | 10.22° | 0.0440 | 76.17 |
| DSO + DPO | 95.1 | 5.42° | 0.0480 | 73.62 |
| DSO + DRO | 99.0 | 1.88° | 0.0440 | 76.17 |

Synthetic data ablation (Table 3):

| Method | Synthetic Only? | Loss | % Stable ↑ | Rot. ↓ |
|--------|----------------|------|-----------|--------|
| DSO | ✓ | DPO | 93.5 | 6.92° |
| DSO | ✗ | DPO | 95.1 | 5.42° |
| DSO | ✓ | DRO | 97.6 | 3.17° |
| DSO | ✗ | DRO | 99.0 | 1.88° |

### Key Findings

1. **DRO consistently outperforms DPO**: stability rate 99.0% vs. 95.1%, with faster convergence.
2. **No trade-off between physical stability and geometric quality**: fine-tuning with DSO slightly improves geometric quality (CD reduced from 0.0485 to 0.0440).
3. **SFT is inferior to DSO**: supervised fine-tuning on stable data alone is insufficient; simultaneous exposure to both stable and unstable samples is necessary.
4. **High data efficiency**: using only 1/16 of the data (~19.2k models) achieves performance close to the full dataset.
5. **Purely synthetic data is viable**: effective models can be trained using only synthetic images without any real 3D data.

## Highlights & Insights

- **DRO as a general contribution**: requiring neither paired preference data nor reference model queries during training, DRO is generalizable to other diffusion model alignment scenarios.
- **Self-improving closed loop**: generator produces outputs → simulator evaluates stability → feedback drives fine-tuning, forming an automatic improvement pipeline requiring no human annotation.
- **Effective use of non-differentiable simulators**: circumvents the limitations of differentiable simulation by reformulating physical property optimization as a reward alignment problem.
- 3D printing experiments validate effectiveness in the real physical world.

## Limitations & Future Work

- Only gravitational self-supporting stability is addressed; other physical properties (deformability, articulation, etc.) are not considered.
- Prolonged training leads to "cheating"—the model generates flat base plates to prevent toppling.
- Validation is limited to TRELLIS; applicability to other 3D generators remains to be examined.
- The choice of stability threshold (20°) lacks sufficient discussion.

## Related Work & Insights

- **RLHF → simulation feedback**: extends the DPO alignment paradigm from LLMs to physical property alignment in 3D generation.
- **Generalizable to broader physical properties**: including scene decomposition, part interaction, graspability, and more.
- **Implications for 3D generation quality evaluation**: geometric quality and functional utility (physical stability) should be jointly considered.

## Rating ⭐⭐⭐⭐

A well-motivated and elegantly designed work. DRO, as a new alignment objective for diffusion models that requires no paired preference data, demonstrates strong generalizability. The self-improving pipeline and purely synthetic data approach showcase substantial practical value. The experiments are comprehensive, and physical validation via 3D printing is convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)

</div>

<!-- RELATED:END -->
