---
title: >-
  [Paper Note] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics
description: >-
  [CVPR 2026][Video Generation][Physically consistent video generation] This paper proposes Phantom, a framework that augments a pretrained video diffusion model (Wan2.2-TI2V) with a dedicated physical dynamics branch. Physics-aware embeddings extracted by V-JEPA2 serve as latent physical states, and bidirectional cross-attention is employed to jointly model visual content and physical dynamics evolution. Phantom achieves substantial improvements over baselines on physics consistency benchmarks (VideoPhy PC +50.4%) while preserving visual quality.
tags:
  - CVPR 2026
  - Video Generation
  - Physically consistent video generation
  - flow matching
  - dual-branch architecture
  - V-JEPA2
  - latent physical dynamics
date: 2026-05-08
content_hash: 14c295dc602e0662
---

# Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics

**Conference**: CVPR 2026
**arXiv**: [2604.08503](https://arxiv.org/abs/2604.08503)
**Code**: [https://plan-lab.github.io/phantom](https://plan-lab.github.io/phantom)
**Area**: Video Generation / Physical Consistency
**Keywords**: Physically consistent video generation, flow matching, dual-branch architecture, V-JEPA2, latent physical dynamics

## TL;DR
This paper proposes Phantom, a framework that augments a pretrained video diffusion model (Wan2.2-TI2V) with a dedicated physical dynamics branch. Physics-aware embeddings extracted by V-JEPA2 serve as latent physical states, and bidirectional cross-attention is employed to jointly model visual content and physical dynamics evolution. Phantom achieves substantial improvements over baselines on physics consistency benchmarks (VideoPhy PC +50.4%) while preserving visual quality.

## Background & Motivation

**Background**: Video generation models represented by Sora, HunyuanVideo, and Wan2.2 can produce visually realistic videos, yet they exhibit clear deficiencies in physical consistency—generated objects frequently violate fundamental physical laws such as gravity, inertia, and collision dynamics.

**Limitations of Prior Work**: (1) Simply scaling model size and training data is insufficient to learn generalizable physical laws; models tend to memorize case-specific patterns rather than abstract physical principles. (2) Existing physics-aware methods either rely on external physics simulators (limited by simulator coverage), employ LLM prompt engineering at inference time to guide generation (which does not enhance the model's intrinsic physical understanding and introduces inference overhead), or inject physical priors via representation alignment (which cannot explicitly model physical state evolution).

**Key Challenge**: Current video generation models primarily rely on a next-frame prediction objective that optimizes visual fidelity without explicitly enforcing physical reasoning, making it difficult for models to internalize and comply with real-world physical laws.

**Goal**: How to directly integrate reasoning over latent physical properties into the video generation process, enabling models to produce videos that are not only visually realistic but also physically consistent?

**Key Insight**: The authors hypothesize that the inability to learn physical dynamics stems from reliance solely on next-frame prediction objectives. The proposed solution is to have the model simultaneously predict video content and latent physical parameters.

**Core Idea**: A dedicated physical branch is introduced into the video generation pipeline. Self-supervised representations from V-JEPA2 serve as "latent physical states" and are jointly trained with the visual branch, enabling the model to reason about physical dynamics while generating video.

## Method

### Overall Architecture
Phantom is built upon Wan2.2-TI2V-5B and adopts a dual-branch parallel latent flow-matching architecture: (1) the **video branch** reuses pretrained Wan2.2 modules to process visual latent sequences; (2) the **physics branch** mirrors the architecture of the video branch but is initialized from scratch, predicting physical dynamics in the V-JEPA2 latent space. The two branches exchange information via bidirectional cross-attention layers—Vis-Attention allows the video branch to attend to hidden states of the physics branch, while Phy-Attention allows the physics branch to attend to hidden states of the video branch.

The observed input video $\mathbf{x}^o$ is encoded into two complementary latent spaces: (1) a visual latent sequence $\mathbf{v}^o$ obtained via the video VAE encoder; and (2) a physical latent sequence $\mathbf{z}^o$ obtained via V-JEPA2. The model is conditioned on observed frames and physical states, and jointly predicts future video frames along with their corresponding physical dynamics.

### Key Designs

1. **Physics-Aware Latent Representation (V-JEPA2 Embeddings)**:

    - **Function**: Provides abstract representations of physical states for the video generation model.
    - **Mechanism**: Representations extracted by V-JEPA2 (a self-supervised video encoder) are used as latent physical states. V-JEPA2 representations have been shown to encode intuitive physical concepts such as object permanence, collision, and gravity, enabling the model to reason in a learned abstract physical space without requiring explicit specification of physical attributes, simulators, or external reasoning modules.
    - **Design Motivation**: Compared to explicit physics simulators, latent physical representations are not constrained by simulator assumptions and can cover a broader range of physical phenomena. Compared to representation alignment approaches, Phantom explicitly models the temporal evolution of physical states rather than performing only static alignment.

2. **Bidirectional Cross-Attention Coupling**:

    - **Function**: Dynamically exchanges information between the video and physics branches.
    - **Mechanism**: Cross-attention layers are inserted at corresponding depths in both branches. Vis-Attention uses video hidden states as queries and physics hidden states as keys/values: $\mathbf{h}'_v = \text{Softmax}(\frac{\mathbf{W}^Q_v\mathbf{h}_v \cdot (\mathbf{W}^K_v\mathbf{h}_z)^T}{\sqrt{d}}) \mathbf{W}^V_v\mathbf{h}_z$; Phy-Attention is handled symmetrically. This allows physical cues to guide visual generation while visual evidence refines physical reasoning.
    - **Design Motivation**: Compared to joint attention that intermixes the two modalities, bidirectional cross-attention provides finer-grained control and avoids training instability caused by excessive entanglement of visual and physical features.

3. **Selective Freezing Training Strategy**:

    - **Function**: Injects physical reasoning while preserving pretrained visual generation capabilities.
    - **Mechanism**: All pretrained parameters of the video branch are frozen during training; only the physics branch and the bidirectional cross-attention layers are updated. In 50% of training instances no conditioning frames are provided (corresponding to text-to-video), while in the remaining 50%, 1–45 conditioning frames are randomly sampled (corresponding to video-to-video).
    - **Design Motivation**: Protects the strong generative prior of Wan2.2 from being corrupted by gradients from the physics branch.

4. **Recursive Loss Weight Scheduling**:

    - **Function**: Stabilizes joint training of the visual and physics branches.
    - **Mechanism**: The joint loss is $\mathcal{L} = \mathcal{L}_v + \alpha_z \mathcal{L}_z$, where the gradient norm of the physics loss is substantially larger than that of the visual loss. The schedule initializes $\alpha_z=0$ and gradually increases it; when the gradient norm of the physics branch exceeds a threshold $\eta_z$, $\alpha_z$ is reset to $0$ and the schedule restarts. This cyclic weighting prevents the physics branch from overwhelming the shared architecture.
    - **Design Motivation**: Direct joint training leads to instability due to the large magnitude of the physics loss; cyclic scheduling allows the physics branch to contribute meaningful gradients progressively.

### Loss & Training
The standard flow-matching objective is extended to jointly predict visual and physical velocity fields. Training is conducted on OpenVidHD-0.4M (approximately 400K high-quality video-text pairs), supporting up to 121 frames at a resolution of 480×832. A recursive weight scheduling strategy is adopted to balance dual-branch training.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Phantom | Wan2.2-TI2V | Gain |
|--------|------|------|----------|------|
| VideoPhy | SA | 47.5 | 41.5 | +14.5% |
| VideoPhy | PC | **37.9** | 25.2 | **+50.4%** |
| VideoPhy-2 | SA | 27.75 | 24.53 | +13.1% |
| VideoPhy-2 | PC | 71.74 | 69.20 | +2.6% |
| Physics-IQ (single-frame) | Score | **29.59** | 22.10 | **+33.9%** |
| Physics-IQ (multi-frame) | Score | 27.53 | - | - |

Note: Phantom achieves the highest VideoPhy PC score (37.9) among all methods, surpassing dedicated physics methods such as PhyT2V (37) and WISA (33).

### VBench-2 Comprehensive Evaluation

| Dimension | Phantom | Wan2.2-TI2V | Change |
|------|---------|-------------|------|
| Total | 51.84 | 51.57 | +0.5% |
| Physics | 43.61 | 40.19 | +6.0% |
| Human Fidelity | 88.39 | 86.10 | +2.7% |
| Controllability | 20.23 | 18.50 | +9.4% |
| Commonsense | 61.43 | 60.57 | +1.4% |

### Physics-IQ Breakdown (Single-Frame)

| Metric | Phantom | Wan2.2-TI2V | Gain |
|------|---------|-------------|------|
| Spatial IoU | 0.245 | 0.164 | +49.4% |
| Spatiotemporal IoU | 0.146 | 0.132 | +10.6% |
| Weighted Spatial IoU | 0.140 | 0.102 | +37.3% |
| MSE↓ | 0.009 | 0.010 | +11.1% |

### Key Findings
- **Physical consistency is substantially improved without sacrificing visual quality**—the VBench-2 total score is on par with or slightly higher than the baseline, demonstrating that physical reasoning and visual generation are mutually compatible.
- Diversity under Creativity decreases (64.67→45.95), while Composition improves from 40.35 to 45.07; the authors suggest that physically implausible videos may artificially inflate diversity metrics.
- Phantom achieves a Physics-IQ single-frame score of 29.59, surpassing all methods including CogVideoX-I2V (27.90) and RDPO (25.21).
- Phantom is trained on only 400K videos without physics-specific data, yet achieves significant gains in physical consistency, validating the effectiveness of V-JEPA2 physical representations combined with joint modeling.

## Highlights & Insights
- **V-JEPA2 as a physics prior is an elegant choice**: No physics simulator or physical parameter annotation is required; the intuitive physical knowledge already encoded in self-supervised visual representations is directly leveraged. This constitutes a form of "free lunch"—exploiting the physics-awareness of an existing large model to enhance another.
- **Dual-branch flow-matching design**: Two parallel ODE processes for vision and physics are coupled via cross-attention, enabling information exchange while preserving the characteristics of each modality. This design is more principled than naively concatenating physical information to the input and offers better scalability.
- **Recursive loss weight scheduling** is a practical training trick—when the gradient scales of two learning objectives differ substantially, periodic weight resets are more stable than fixed ratio weighting. This approach is transferable to other multi-task learning scenarios.
- **Zero additional physical input at inference**: In text-to-video mode, the model performs fully joint denoising from pure noise, indicating that physical understanding has been internalized by the model.

## Limitations & Future Work
- The physics branch is initialized from scratch, which may be less training-efficient compared to initialization from an existing physical model.
- V-JEPA2's physics-awareness remains limited and may insufficiently encode complex phenomena such as fluid dynamics and deformable objects.
- Training is conducted on only 400K videos, whereas the baseline Wan2.2 is pretrained on substantially larger data—larger-scale training may yield further improvements.
- Recursive weight scheduling requires manual specification of the threshold $\eta_z$ and may be sensitive to this hyperparameter.
- The observed decrease in Diversity on VBench-2 warrants attention and may limit applicability in creative generation scenarios.

## Related Work & Insights
- **vs. PhyT2V/DiffPhy**: These methods refine prompts at inference time using LLM reasoning to guide diffusion, which is an external approach that does not enhance the model's intrinsic physical understanding and incurs inference overhead. Phantom internalizes physical reasoning into the generation process.
- **vs. VideoREPA**: VideoREPA injects physical priors indirectly via representation alignment, performing static alignment without modeling temporal physical state evolution. Phantom explicitly predicts the temporal evolution of physical dynamics.
- **vs. PhysAnimator/PhysGen**: These methods rely on external physics simulators and are constrained by simulator coverage and fidelity. Phantom requires no simulator.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Dual-branch joint modeling of visual and physical dynamics constitutes a novel paradigm; the use of V-JEPA2 as latent physical representations is an elegant choice.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation covers four benchmarks (VideoPhy, VideoPhy-2, Physics-IQ, VBench-2), though ablation studies analyzing individual component contributions are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated and the methodology is presented systematically.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new direction for physically consistent video generation; the paradigm of dual-branch joint modeling combined with self-supervised physical representations is broadly influential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)
- [\[CVPR 2026\] Physical Simulator In-the-Loop Video Generation](physical_simulator_in-the-loop_video_generation.md)
- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](../../ICLR2026/video_generation/javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)

</div>

<!-- RELATED:END -->
