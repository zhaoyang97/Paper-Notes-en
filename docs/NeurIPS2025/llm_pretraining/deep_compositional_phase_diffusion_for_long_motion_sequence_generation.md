---
title: >-
  [Paper Note] Deep Compositional Phase Diffusion for Long Motion Sequence Generation
description: >-
  [NeurIPS 2025][LLM Pretraining][motion generation] This paper proposes the Compositional Phase Diffusion framework, which employs SPDM and TPDM to handle semantic alignment and transition continuity, respectively, within the frequency-domain phase space established by ACT-PAE. The framework enables long-range compositional motion sequence generation and achieves state-of-the-art performance on BABEL-TEACH.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - motion generation
  - diffusion model
  - phase representation
  - compositional generation
  - motion inbetweening
date: 2026-05-08
content_hash: d86830d8b3f947a9
---

# Deep Compositional Phase Diffusion for Long Motion Sequence Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.14427](https://arxiv.org/abs/2510.14427)
**Code**: [GitHub](https://github.com/asdryau/TransPhase)
**Area**: LLM Pretraining
**Keywords**: motion generation, diffusion model, phase representation, compositional generation, motion inbetweening

## TL;DR
This paper proposes the Compositional Phase Diffusion framework, which employs SPDM and TPDM to handle semantic alignment and transition continuity, respectively, within the frequency-domain phase space established by ACT-PAE. The framework enables long-range compositional motion sequence generation and achieves state-of-the-art performance on BABEL-TEACH.

## Background & Motivation
**State of the Field**: Current human motion generation models (e.g., MDM, MLD) excel at generating variable-length motion clips with a single semantic label, but exhibit significant issues in compositional long-sequence generation tasks—specifically, at the boundaries between consecutively executed semantic action segments.

**Limitations of Prior Work**: Directly concatenating independently generated motion clips produces motion discontinuities at transition boundaries, manifesting as abrupt stops, over-smoothing, and foot-sliding artifacts. Methods such as priorMDM introduce additional transition segments to smooth pose differences, but neglect the intrinsic kinematic properties of each clip.

**Root Cause**: How can semantic alignment within each action segment be maintained while simultaneously ensuring kinematic continuity across adjacent segments?

**Paper Goals**: Jointly address semantic alignment and transition smoothness, supporting flexible compositional generation with arbitrary numbers of variable-length segments.

**Starting Point**: Perform diffusion in the motion frequency domain (phase parameter space) rather than in the raw pose space, leveraging the fact that phase representations naturally capture the periodicity and dynamics of motion.

**Core Idea**: A Transformer-based periodic autoencoder encodes motions into unified phase parameters, after which two specialized diffusion modules—one for semantics and one for transitions—collaboratively denoise in the phase space.

## Method

### Overall Architecture
The framework consists of three components: (1) ACT-PAE encodes variable-length motions into unified phase parameters $\mathbf{P} = [\mathbf{F}, \mathbf{A}, \mathbf{B}, \mathbf{S}]$; (2) SPDM denoises phase parameters conditioned on text to ensure semantic alignment; (3) TPDM denoises phase parameters in transition regions using phase information from adjacent segments to ensure continuity. Multiple modules can be parallelized to handle an arbitrary number of segments.

### Key Designs

1. **ACT-PAE (Action-Centric Periodic Autoencoder)**:

    - **Function**: Encodes a variable-length motion sequence $\mathbf{X} \in \mathbb{R}^{N \times E}$ into fixed-dimensional phase parameters $\mathbf{P} = [\mathbf{F}, \mathbf{A}, \mathbf{B}, \mathbf{S}] \in \mathbb{R}^Q$.
    - **Mechanism**: A Transformer encoder based on the ACTOR architecture directly processes variable-length inputs and predicts four parameters—frequency $\mathbf{F}$, amplitude $\mathbf{A}$, offset $\mathbf{B}$, and phase shift $\mathbf{S}$—which are parameterized as a periodic signal via $\mathbf{Q} = \mathbf{A}\sin(\mathbf{F} \cdot (T - \mathbf{S})) + \mathbf{B}$; a decoder then reconstructs the motion.
    - **Design Motivation**: The original DeepPhase PAE relies on fixed-length convolutions, causing variable-length motions to be encoded into differing numbers of phase codes with inconsistent training objectives. ACT-PAE addresses this by using a Transformer to process variable-length inputs and produce fixed-dimensional parameters.

2. **SPDM (Semantic Phase Diffusion Module)**:

    - **Function**: Guides phase parameter denoising using CLIP text embeddings to ensure semantic consistency between generated motion and input text.
    - **Mechanism**: Phase parameters are represented simultaneously as param-level tokens $[\mathbf{F}, \mathbf{A}, \mathbf{B}, \mathbf{S}]$ and frame-level tokens (the periodic signal $\mathbf{Q}$); a self-attention Transformer fuses text conditioning with both levels of phase information for denoising.
    - **Design Motivation**: Frame-level tokens explicitly provide spatiotemporal action context, enabling SPDM to monitor the current semantic state throughout the denoising process.

3. **TPDM (Transitional Phase Diffusion Module)**:

    - **Function**: Guides denoising of the current segment using clean phase parameters from adjacent segments to ensure transition continuity.
    - **Mechanism**: Two sub-modules are designed—TPDMf (conditioned on the preceding motion) and TPDMb (conditioned on the succeeding motion)—each using cross-attention over the current noisy phase and the adjacent clean phase. A Phase Mixing formula $\mathbf{P}^0 = r\frac{\mathbf{P}_f^0 + \mathbf{P}_b^0}{2} + (1-r)\mathbf{P}_c^0$ blends transition and semantic information, where $r$ decreases over denoising steps (establishing transitions first, then refining semantics).
    - **Design Motivation**: Bidirectional TPDM ensures phase dynamics are aligned in both forward and backward directions, and bidirectional information propagation prevents progressive error accumulation in long sequences.

### Loss & Training
- ACT-PAE: L2 reconstruction loss.
- SPDM and TPDM: Standard $\epsilon$-prediction diffusion loss with DDIM sampler.

## Key Experimental Results

### Compositional Action Pair Generation (BABEL-TEACH Test)

| Method | FID↓ Overall | MMD↓ Overall |
|--------|-------------|-------------|
| MDM-30 | 1.146 | 4.923 |
| TEACH | 1.041 | 4.821 |
| PCMDM | 0.837 | 5.423 |
| priorMDM | 0.839 | 5.025 |
| **Ours** | **0.782** | **4.711** |

### Long-Range Motion Generation (3,164 text segments, 168 minutes)

| Method | FID↓ Overall | MMD↓ Overall |
|--------|-------------|-------------|
| TEACH | 1.780 | 4.984 |
| PCMDM | 0.876 | 5.156 |
| priorMDM | 1.536 | 5.060 |
| **Ours** | **0.766** | **4.680** |

### Ablation Study

| Configuration | FID↓ | Note |
|---------------|------|------|
| w/o TPDM | Significant increase | Degraded transition quality |
| w/o SPDM | Moderate increase | Degraded semantic alignment |
| w/o Phase Mixing | Increase | Lack of progressive blending |
| Full model | Best | All components are complementary |

### Key Findings
- Operating in the frequency-domain phase space yields smoother transitions than operating in the raw pose space, as phase parameters naturally capture motion periodicity.
- The progressive information propagation mechanism of bidirectional TPDM sustains continuity across very long sequences (168 minutes).
- The framework scales well—by parallelizing module processing, generation time is decoupled from the number of segments.

## Highlights & Insights
- **Advantage of frequency-domain operation**: Performing diffusion in phase parameter space rather than raw joint space reformulates the motion continuity problem as a phase continuity problem, which is more tractable to model.
- **Progressive blending strategy**: The decreasing schedule of $r$ in Phase Mixing (transitions first, semantics second) is intuitively grounded—it first establishes kinematic coherence before refining semantic details.
- **Unified framework**: The same architecture handles compositional generation, motion inbetweening, and long-range generation through module addition or removal.

## Limitations & Future Work
- **Dataset limitations**: Evaluation is conducted solely on BABEL-TEACH, which covers a limited range of action categories with a maximum clip length of 250 frames.
- **Information loss in phase representation**: Periodic sinusoidal parameterization may not fully capture complex aperiodic motions.
- **Limitations of linear blending**: Final segment concatenation still relies on linear blending, which may introduce minor discontinuities at high-frequency motion transitions.

## Related Work & Insights
- **vs. priorMDM**: priorMDM independently generates semantic segments and then synthesizes transition segments; the proposed method exchanges information during the denoising process itself, resolving transition issues at the source.
- **vs. TEACH**: TEACH connects boundary poses via SLERP interpolation, ignoring intrinsic kinematics; the proposed method models dynamics in the phase space.

## Rating
- Novelty: ⭐⭐⭐⭐ — Phase-space diffusion with dual semantic/transition module design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluation across three tasks, ablation study, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ — Framework description is clear and diagrams are intuitive.
- Value: ⭐⭐⭐⭐ — Constitutes a meaningful advancement for long-range motion generation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Practical Guide for Incorporating Symmetry in Diffusion Policy](a_practical_guide_for_incorporating_symmetry_in_diffusion_policy.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](../../ICCV2025/llm_pretraining/syncity_training-free_generation_of_3d_worlds.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)

<!-- RELATED:END -->
