---
title: >-
  [Paper Note] From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation
description: >-
  [ICML 2026][Robotics][Latent Actions] MoLA employs a set of "modal-aware inverse dynamics models (IDM)" pre-trained on large-scale robotics data to translate future frames predicted by a video generation model into three…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Latent Actions"
  - "Inverse Dynamics"
  - "Video Generation"
  - "Robot Manipulation"
  - "Modal Perception"
date: 2026-05-08
content_hash: 52a1124d1cb811fa
---

# From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation

**Conference**: ICML 2026  
**arXiv**: [2605.12167](https://arxiv.org/abs/2605.12167)  
**Code**: https://logosroboticsgroup.github.io/MoLA (available)  
**Area**: Robotics / Embodied Intelligence / Video Generation / VLA  
**Keywords**: Latent Actions, Inverse Dynamics, Video Generation, Robot Manipulation, Modal Perception

## TL;DR
MoLA employs a set of "modal-aware inverse dynamics models (IDM)" pre-trained on large-scale robotics data to translate future frames predicted by a video generation model into three discrete latent actions—semantic, depth, and optical flow. The policy head then controls based on these action-centric representations, achieving robust and accurate "imagination-to-execution" interfaces on CALVIN, LIBERO, LIBERO-Plus, and real UR5e robots.

## Background & Motivation

**Background**: Current robot manipulation research follows two main paths. One is VLA (e.g., RT-1/π0/OpenVLA), which learns an end-to-end action head from vision and language. The other is imagination-based policy (e.g., UniPi/VPP/DreamGen), which first predicts future frames using a video generation model and then feeds these "imagined futures" to the policy. The imagination-based approach is attractive for its ability to foresee long-horizon outcomes.

**Limitations of Prior Work**: There are two naive ways to drive policy with imagined frames, both suboptimal. The first treats predicted frames as extra visual conditions for the action head, forcing it to learn all control signals from visual changes. The second directly decodes actions from video, making control entirely dependent on video prediction accuracy—any drift in long-horizon video prediction leads to action drift.

**Key Challenge**: Video generation models are optimized for "perceptual realism"—pixel MSE, perceptual loss, etc.—not for "control relevance." Thus, even if predicted frames look realistic, they do not explicitly reveal the physical actions driving state transitions; the mapping from image space to action space is inherently indirect and unstable.

**Goal**: Insert an "action-centric interface" between video imagination and policy execution, enabling the policy to reason in action space rather than pixel space. This interface must reliably infer actions from generated (noisy) future frames and support multi-modal information.

**Key Insight**: The authors observe that inverse dynamics models (IDM) are naturally suited for this—by definition, they infer the action between $o_t$ and $o_{t+k}$ given visual changes. Applying IDM to generated future frames effectively "reverse decodes" video outputs into actions. However, a single IDM tends to overload one codebook; the authors further note that manipulation depends on three complementary cues: semantics (task intent), depth (3D structure), and flow (interaction dynamics).

**Core Idea**: Use "multi-path inverse dynamics models, pre-trained on large-scale robotics data and specialized by modality," to translate generated future videos into a set of discrete latent actions (Mixture of Latent Actions), serving as the interface between video imagination and the policy head.

## Method

### Overall Architecture
Given current RGB observation $o_t$ and language instruction $l$, MoLA proceeds in four steps: (1) The video generation model (Stable Video Diffusion backbone, single-step denoising at inference) synthesizes future frames $\hat{o}_{t:t+H}$ as the "imagination space"; (2) Three IDMs (semantic/depth/flow) each infer discrete latent actions $z_{t \to t+k}^{(m)}$ from $(o_t, \hat{o}_{t+k})$; (3) The three latent actions are concatenated into a mixture $\mathcal{Z}$; (4) A Diffusion Transformer + flow matching policy head consumes the latent actions and generated visual features to output a continuous, executable control sequence. Training is staged: fine-tune the video generator, pre-train MoIDM, then end-to-end fine-tune MoIDM + policy head (video generator frozen).

### Key Designs

1. **Modal-Aware Inverse Dynamics Model (MoIDM)**:

    - **Function**: Translates "visual change between two RGB frames" into discrete latent action tokens, serving as the core interface between imagination and control.
    - **Mechanism**: Each IDM has an independent spatiotemporal Transformer $T^{(m)}$ and VQ codebook. A ViT encoder extracts features from $o_t$ and $o_{t+k}$; learnable latent action queries interact with these features via $T^{(m)}$, outputting $\tilde{h} = T^{(m)}(q^{(m)}, h_t, h_{t+k})$, then VQ yields $z^{(m)} = \text{VQ}^{(m)}(\tilde{h})$. All three IDMs share the RGB inference pipeline but are separated by different reconstruction targets (SAM2 for semantics, Depth Anything v2 for depth, CoTracker3 for flow), enforcing modal bias.
    - **Design Motivation**: A single IDM tends to entangle all factors in one codebook, causing interference. Modal separation constrains each codebook to capture only one aspect—task semantics, geometry, or interaction dynamics—maximizing complementarity. Ablations show merging or sharing codebooks degrades performance.

2. **Large-Scale Pretraining + Joint Fine-Tuning Strategy**:

    - **Function**: Ensures IDM works robustly not only on real frames but also on generated (noisy, slightly drifting) future frames.
    - **Mechanism**: Independently pre-train three IDMs on large-scale robotics data (Open X-Embodiment + AgiBot) using real future frames, then freeze the video generator and jointly fine-tune MoIDM and policy head on downstream tasks. This preserves general action semantics learned from large-scale visual changes and allows co-adaptation to the "generated frame" distribution.
    - **Design Motivation**: Ablations (Q2/Q3) show that removing pretraining (training from scratch) severely hurts performance, while freezing MoIDM prevents it from adapting to evolving policies—MoIDM must be both "broadly pre-trained" and "adaptable downstream," both are essential.

3. **Flow Matching-Based Diffusion Policy Head**:

    - **Function**: Decodes discrete latent actions + generated visual features into continuous, temporally correlated, multi-modal real robot control sequences.
    - **Mechanism**: The policy head uses a Diffusion Transformer (DiT) trained from scratch; the training objective is flow matching (not standard diffusion loss), learning a continuous transformation from noisy action samples to target actions. Inputs are the three latent actions and predicted visual features.
    - **Design Motivation**: Latent actions are "intermediate cues distilled from imagination," but final control remains a continuous, multi-modal distribution. Ablation (Q4) shows that a lighter autoregressive token head also works (indicating sufficient information in latent actions), but DiT + flow matching is more robust for long-horizon dependencies, multi-modality, and imperfect latent actions.

### Loss & Training
During IDM pretraining, each path uses two reconstruction targets: a shared RGB decoder reconstructs future RGB frames, and a modality-specific decoder reconstructs features/labels from the corresponding foundation model. In downstream tasks, the video generator is frozen, and MoIDM + policy head are trained end-to-end with flow matching loss. The overall process is "video fine-tuning → MoIDM pretraining → end-to-end fine-tuning," each stage serving a distinct purpose.

## Key Experimental Results

### Main Results
Evaluation covers CALVIN ABC-D long-horizon tasks, four LIBERO lifelong subsets, LIBERO-Plus (10,030 robust tasks with perturbations), and real UR5e robots.

| Benchmark | Metric | Ours | Prev. SOTA | Gain |
|-----------|--------|------|------------|------|
| CALVIN ABC-D | Avg. Len. | 4.55 | DreamVLA 4.44 | +0.11 |
| LIBERO (mean of 4 sets) | Success % | 97.0 | VPP 90.9 | +6.1 |
| LIBERO-Plus | Avg. % | 92.7 | OpenVLA-OFT+ 79.5 | +13.2 |
| Real Robot (in+OOD avg.) | Success % | 73.0 | VPP 62.0 | +11.0 |

On CALVIN, MoLA raises the "success rate for completing 5 consecutive subtasks" from the previous SOTA 76.9% (VPP) to 82.6%. The 13.2-point gain on LIBERO-Plus especially demonstrates robustness to perturbations, lighting changes, and initial state variations. On real robots, while not surpassing commercial-grade π0.5, MoLA outperforms similar "video imagination + policy" approaches (VPP) by 11 points in stability.

### Ablation Study

| Modal Combination | CALVIN Avg. Len. ↑ | Notes |
|-------------------|--------------------|-------|
| Baseline (no MoIDM, direct future frames) | 4.24 | Weakest |
| Sem only | 4.31 | Task semantics alone is useful |
| Depth only | 4.35 | 3D structure information is stronger |
| Flow only | 4.39 | Strongest single modality (interaction dynamics most important) |
| Flow + Depth | 4.46 | Dual modalities are complementary |
| All (Sem + Depth + Flow) | **4.55** | Full tri-modal is optimal |

### Key Findings
- Adding the three IDMs increases Avg. Len. monotonically (4.24→4.31→4.35→4.39→4.46→4.55), confirming their true complementarity rather than redundancy. Among single modalities, flow contributes most—supporting the view that "manipulation is fundamentally about interaction dynamics."
- Ablation Q1 also shows that even the "baseline (no IDM)" is weaker than using the weakest single-path IDM, indicating that the VQ discrete latent action bottleneck itself is an effective inductive bias, making the "image→action" mapping easier to learn.
- In terms of data efficiency, MoLA significantly outperforms VPP with only 10% of CALVIN data, demonstrating that "pre-trained IDM + latent action interface" provides a strong prior, especially advantageous in low-data regimes.
- At inference, replacing "imagined future" with "two copies of the same frame / noisy current frame" significantly degrades performance, indicating that both the latent action mechanism and genuine future cues are indispensable; MoIDM's inductive bias alone is insufficient.

## Highlights & Insights
- **The observation that a single IDM is insufficient directly motivated the modal separation approach**, avoiding the common pitfall of overloading one codebook with conflicting signals. This "train a specialist for each modality + use foundation models as supervision" paradigm is transferable to other tasks bridging vision and action/language.
- **The discrete VQ bottleneck is more critical than expected**: the baseline without IDM is even weaker than the weakest single-path IDM, indicating that compressing high-dimensional visual changes into discrete tokens helps the policy filter out control-irrelevant details. This applies to other "high-dimensional condition → low-dimensional decision" problems (e.g., GUI agents).
- **The three-stage division—video generator as 'producer', IDM as 'interpreter', policy head as 'executor'—enables MoLA to seamlessly swap underlying video models** (ablation Q5 shows replacing SVD with Wan2.2 improves performance), providing a clean way to integrate video generation advances into robot control.

## Limitations & Future Work
- The video generation model is kept frozen, limiting the upper bound of three-way co-optimization; using a lighter video model that can also be fine-tuned may further reduce the distribution gap between generated and real frames.
- In real robot experiments, commercial-scale VLA models like π0.5 still outperform MoLA on average, indicating that the "video imagination + interface" approach does not yet fully replace end-to-end VLA trained on massive datasets, and is currently better suited for medium-scale data scenarios.
- Modal selection is fixed to semantic/depth/flow; the inclusion of tactile, kinematic, or force feedback modalities for real robots is not discussed. Multi-modal extension and learning inter-modal weights are natural next steps.
- Although only single-step denoising is used at inference, the process still requires running video generation + three IDMs + DiT policy head, so overall latency may be tight for high-frequency control scenarios.

## Related Work & Insights
- **vs VLA (OpenVLA / π0 / GR00T)**: These map perception directly to action without an explicit "imagination" step; MoLA adds an imagination layer, yielding greater stability on long-horizon and out-of-distribution tasks, at the cost of higher inference overhead.
- **vs VPP (Video Prediction Policy)**: Also "predict future then drive policy," but VPP directly conditions on generated frames, requiring the action head to learn "visual change→control" itself; MoLA inserts MoIDM to pre-digest this, raising CALVIN Avg. Len. from 4.33 to 4.55.
- **vs DreamGen / UniPi**: These use video generation as the policy backbone, directly decoding actions; MoLA emphasizes the latent action interface as key, offering greater robustness than pure video-to-action.
- **vs LAPA / UniVLA**: Also use IDM/latent action ideas, but with single encoding and no explicit modal decomposition; MoLA's "modal separation + VQ codebook" further refines this line.

## Rating
- Novelty: ⭐⭐⭐⭐ "Tri-modal IDM + discrete latent action interface" is a clean design, though IDM itself and single-path LAPA have prior prototypes in the literature; the main contribution is solidifying this approach.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three simulation suites + real robot + data efficiency + 7 ablation questions, covering a wide range.
- Writing Quality: ⭐⭐⭐⭐ The narrative is clear, logically connecting the "visual realism vs control relevance" conflict to MoIDM; formulas are dense but necessary.
- Value: ⭐⭐⭐⭐ Provides the most convincing interface design for "video imagination-driven robots" to date, likely to become a strong baseline for future video-based VLA work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)
- [\[ICML 2026\] Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](decompose_and_recompose_reasoning_new_skills_from_existing_abilities_for_cross-t.md)
- [\[ICML 2026\] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)
- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](../../CVPR2026/robotics/dawn_pixel_motion_diffusion_robot_control.md)
- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](../../AAAI2026/robotics/semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)

</div>

<!-- RELATED:END -->
