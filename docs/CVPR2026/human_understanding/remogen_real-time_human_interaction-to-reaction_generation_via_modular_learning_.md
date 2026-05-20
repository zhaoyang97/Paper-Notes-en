---
title: >-
  [Paper Note] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data
description: >-
  [CVPR 2026][Human Understanding][Interaction-to-reaction generation] This paper proposes ReMoGen, a modular framework for real-time human interaction-to-reaction motion generation. It learns a general motion prior from l…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Interaction-to-reaction generation"
  - "modular learning"
  - "motion prior"
  - "real-time generation"
  - "human-human/human-scene interaction"
date: 2026-05-08
content_hash: 786254e43ae47bff
---

# ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data

**Conference**: CVPR 2026
**arXiv**: [2604.01082](https://arxiv.org/abs/2604.01082)  
**Code**: None  
**Area**: Human Motion Generation / Human Understanding / Interactive Motion Generation
**Keywords**: Interaction-to-reaction generation, modular learning, motion prior, real-time generation, human-human/human-scene interaction

## TL;DR

This paper proposes ReMoGen, a modular framework for real-time human interaction-to-reaction motion generation. It learns a general motion prior from large-scale single-person motion data (frozen during downstream training), adapts to different interaction domains (human-human/human-scene) via independently trained Meta-Interaction modules, and achieves per-frame low-latency online updates (0.047 s/frame) through Frame-wise Segment Refinement. ReMoGen comprehensively surpasses state-of-the-art methods on the Inter-X and LINGO benchmarks.

## Background & Motivation

1. **Background**: Human motion generation has evolved from text-driven single-person synthesis to multi-agent interaction scenarios. Existing approaches include: text-to-motion methods (T2M, MotionDiffuse) that generate isolated motions only; human-scene interaction methods (TRUMANS, LINGO) that introduce spatial awareness but are limited to single agents; and human-human interaction methods (ReGenNet, FreeMotion) that attempt joint generation but operate predominantly in offline mode.
2. **Limitations of Prior Work**:
    - **Data scarcity and heterogeneity**: Single-person motion data is abundant (HumanML3D), whereas human-human interaction (Inter-X) and human-scene interaction (LINGO) datasets are scarce and exhibit large distributional discrepancies, causing end-to-end models trained on a single domain to overfit.
    - **Real-time responsiveness**: Diffusion models yield high quality but incur large latency incompatible with real-time use; autoregressive models are fast but suffer from error accumulation and drift.
    - Most existing methods assume full observation of the counterpart's complete trajectory, which is infeasible in practical online interaction settings.
3. **Key Challenge**: How to simultaneously achieve high-fidelity and low-latency interaction-to-reaction generation under data-scarce conditions.
4. **Goal**: (1) Efficient knowledge transfer across heterogeneous interaction domains; (2) Real-time responsiveness without sacrificing motion quality.
5. **Key Insight**: Decouple general motion prior learning from interaction-specific adaptation — freeze a backbone pretrained on large-scale single-person data and inject interaction awareness via lightweight modules.
6. **Core Idea**: Prior-guided modular learning + frame-level intra-segment refinement; the former addresses data heterogeneity and the latter addresses real-time requirements.

## Method

### Overall Architecture

The inputs are textual intent, observed motions of other agents, and scene context. ReMoGen comprises three components: (1) a frozen text-conditioned single-person motion prior (a VAE and latent diffusion model pretrained on HumanML3D); (2) Meta-Interaction modules (independently trained adapters for HHI and HSI, respectively); and (3) Frame-wise Segment Refinement (FWSR), a lightweight per-frame correction module. Motion is generated via segment-level autoregression: conditioned on a history window $M_h^i \in \mathbb{R}^{H \times D}$ and text $W$, the model predicts a future segment $\hat{M}_f^i \in \mathbb{R}^{F \times D}$ ($H=2$ history frames, $F=8$ future frames).

### Key Designs

1. **Universal Motion Prior**:

    - **Function**: Provides a strong generative foundation encoding basic kinematic structure, temporal dynamics, and language-motion correspondences.
    - **Mechanism**: Built on the DART architecture, a Transformer VAE encoder-decoder compresses motion segments into a latent space, and a conditional diffusion model generates within that latent space. The encoder maps a motion segment to latent representation $z$; the denoiser $G_\psi$ iteratively denoises under text embedding $w$: $\hat{z}_0 = G_\psi(z_t, t, M_h^i, w)$; the decoder reconstructs the motion. Generation uses 10 diffusion steps at 10 FPS.
    - **Design Motivation**: The motion prior learned from large-scale single-person data is already highly expressive. Joint fine-tuning destroys this knowledge — experiments show that joint fine-tuning degrades motion quality — making freezing the prior a critical design choice.

2. **Meta-Interaction Module**:

    - **Function**: Injects interaction awareness into the frozen motion prior.
    - **Mechanism**: Two independent encoders process interaction cues separately — an Others Encoder (TCN-based) extracts relative velocity, approach direction, and spatial relationships; a Scene Encoder (ViT-based) summarizes surrounding geometry and functional space. Interaction cues are injected via a Meta-Interaction Block: ego features first undergo self-attention to obtain $h'$; cross-attention is then applied over interaction cues to extract interaction signals, which are transformed into FiLM-style affine parameters $(\gamma, \beta)$ and applied as $h_{mod} = (1 + \tanh\gamma) \odot h' + \tanh\beta$.
    - **Design Motivation**: Each module is trained independently on its respective domain (HHI on Inter-X, HSI on LINGO) for 65k iterations, avoiding the difficulties of joint training on heterogeneous data. At inference, effects from multiple modules are composited as $\Delta_{total} = \sum_i \alpha_i \Delta_i$ (with L2-norm clamping), enabling flexible mixing.

3. **Frame-wise Segment Refinement (FWSR)**:

    - **Function**: Provides per-frame low-latency reactive updates on top of segment-level generation.
    - **Mechanism**: Standard segment-level autoregression faces a latency-quality trade-off — longer segments improve quality but slow updates, while shorter segments improve responsiveness but introduce jitter. At each frame within a segment, FWSR fine-tunes the initial segment latent $z_0$ using a lightweight Meta-Interaction Block: $\hat{z}^f = \text{Modulate}(z_0, \text{concat}(M_h^{(f-1)}, X_{dyn}^{(f)}))$, incorporating the latest observed interaction cues. Only the prediction at the corresponding frame position is retained; the history buffer is updated before processing the next frame.
    - **Design Motivation**: The large backbone provides stable long-term dynamics, while the lightweight adapter enables fast fine-grained reactivity. FWSR is trained independently (with the prior and Meta-Interaction modules frozen), ensuring it acts as a stable local adapter without altering the global motion structure.

### Loss & Training

- **Stage-wise training**: Pretraining the prior on HumanML3D → independent training of each Meta-Interaction module for 65k iterations (prior frozen) → training FWSR for 65k steps (prior and Meta-Interaction frozen).
- **Training objectives**: Reconstruction loss $L_{rec}$, latent space loss $L_{latent}$, and auxiliary temporal increment loss $L_{aux}$.
- **Optimizer**: AdamW (lr=1e-4), batch size 1024, gradient clipping 1.0, EMA 0.999.
- Training is feasible on a single NVIDIA RTX 3090.

## Key Experimental Results

### Main Results

**Human-Human Interaction (Inter-X)**

| Method | FID ↓ | R-Prec. (Top3) ↑ | MM Dist. ↓ | Latency (s/frame) ↓ |
|---|---|---|---|---|
| ReGenNet | 11.622 | 0.269 | 6.092 | 0.210 |
| FreeMotion | 3.383 | 0.284 | 5.438 | 0.221 |
| SymBridge | 2.569 | 0.355 | 4.955 | 0.040 |
| **Ours** | **0.181** | **0.464** | **4.076** | 0.042 |
| **Ours+FWSR** | **0.166** | 0.462 | **4.076** | 0.047 |

**Human-Scene Interaction (LINGO)**

| Method | FID ↓ | R-Prec. (Top3) ↑ | MM Dist. ↓ | Latency ↓ |
|---|---|---|---|---|
| TRUMANS | 4.731 | 0.178 | 10.822 | 0.074 |
| LINGO | 3.633 | 0.218 | 9.597 | 0.189 |
| **Ours** | **1.201** | **0.530** | **3.408** | **0.042** |

### Ablation Study

**Ablation on Motion Prior Usage (Inter-X)**

| Configuration | FID ↓ | R-Prec. ↑ | MM Dist. ↓ |
|---|---|---|---|
| Prior Only | 3.735 | 0.231 | 5.736 |
| No Prior (from scratch) | 0.270 | 0.412 | 4.385 |
| Joint-Finetune | 0.298 | 0.439 | 4.188 |
| **Ours (Frozen Prior + Module)** | **0.181** | **0.464** | **4.076** |

**FWSR Ablation**

| Configuration | FID ↓ | Latency ↓ |
|---|---|---|
| Segment-level autoregression (Seg.) | 0.181 | 0.042 |
| Frame-level sliding window (Slide) | 4.136 | 0.305 |
| **Segment + FWSR** | **0.166** | 0.047 |

### Key Findings

- **Frozen prior + modular adaptation substantially outperforms joint fine-tuning**: FID 0.181 vs. 0.298; joint fine-tuning erodes the pretrained kinematic knowledge.
- **FWSR achieves significant quality improvement** (FID 0.181→0.166) **at negligible additional latency** (0.042→0.047 s/frame).
- **Zero-shot compositional generalization**: Directly combining HHI and HSI modules on EgoBody ($\alpha_{HHI}=\alpha_{HSI}=0.5$) without retraining already outperforms zero-shot single-module inference, though it falls short of fine-tuning.
- **Prior initialization followed by only 2k–10k fine-tuning steps surpasses training from scratch for 500k steps** (EgoBody), demonstrating the strong transfer efficiency of the pretrained prior.
- The real-time threshold of 0.1 s/frame is comfortably satisfied (0.042–0.047 s/frame), whereas ReGenNet and FreeMotion both fail to meet it.

## Highlights & Insights

- **The modular decoupling philosophy is particularly elegant**: the prior supplies fundamental motion capability, the Meta-Interaction module provides interaction awareness, and FWSR provides real-time responsiveness — three orthogonal components that can be optimized independently. This design paradigm is transferable to any generative task requiring adaptation under data-scarce conditions.
- **The FiLM-modulated Meta-Interaction Block** establishes a sound adapter design pattern for motion generation — injecting conditional signals via feature-level affine transformations without modifying original model parameters.
- **Compositional inference** (weighted combination of multiple modules) naturally extends the framework to mixed interaction scenarios without retraining.

## Limitations & Future Work

- Currently limited to dyadic interaction and simple scene interaction; multi-person group interaction is not addressed.
- The combination weights $\alpha_i$ for Meta-Interaction modules must be set manually; adaptive learning of these weights warrants exploration.
- Scene encoding uses voxelized 3D occupancy representations, which may be insufficient for fine-grained object interactions (e.g., manipulating objects on a table).
- Fine-grained hand interactions (e.g., handshakes, object handovers) are not handled.
- FWSR's per-frame correction may exhibit insufficient responsiveness under abrupt motion changes.

## Related Work & Insights

- **vs. SymBridge**: Both target real-time interaction; SymBridge focuses on human-robot interaction but achieves higher FID (2.569 vs. 0.181). ReMoGen substantially improves quality through a stronger motion prior.
- **vs. FreeMotion**: The offline version of FreeMotion achieves FID of only 0.492, but its latency is unacceptable. ReMoGen attains superior FID (0.166) under real-time constraints.
- **vs. ControlNet/LoRA paradigm**: ReMoGen successfully transfers the adapter design pattern from image generation to motion generation; FiLM modulation replaces the additive injection used in ControlNet.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-tier design of prior freezing + modular adaptation + frame-level refinement is clear and innovative, though individual components (FiLM, segment-level autoregression) are established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers HHI, HSI, and mixed scenarios; provides detailed ablations on prior usage strategy and FWSR effectiveness; includes EgoBody transfer experiments — comprehensive overall.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; the four research questions effectively organize the experimental section.
- **Value**: ⭐⭐⭐⭐ Real-time interaction-to-reaction generation is an important yet underexplored problem, and the proposed framework offers a scalable solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HandX: Scaling Bimanual Motion and Interaction Generation](handx_scaling_bimanual_motion_and_interaction_generation.md)
- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](../../ICCV2025/human_understanding/posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[CVPR 2026\] HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation](humanorbit_3d_human_reconstruction_as_360_orbit_generation.md)
- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](molingo_motion-language_alignment_for_text-to-motion_generation.md)
- [\[NeurIPS 2025\] HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion](../../NeurIPS2025/human_understanding/hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)

</div>

<!-- RELATED:END -->
