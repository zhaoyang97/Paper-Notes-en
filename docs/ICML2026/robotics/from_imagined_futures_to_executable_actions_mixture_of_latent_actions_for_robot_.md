---
title: >-
  [Paper Note] From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation
description: >-
  [ICML 2026][Robotics][Latent Actions] MoLA employs a set of "Modality-aware Inverse Dynamics Models (IDM)" pre-trained on large-scale robotics data to translate future frames predicted by a video generation model into th…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Latent Actions"
  - "Inverse Dynamics"
  - "Video Generation"
  - "Robot Manipulation"
  - "Modality Awareness"
date: 2026-05-08
content_hash: e9db84d9ca87c264
---

# From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation

**Conference**: ICML 2026  
**arXiv**: [2605.12167](https://arxiv.org/abs/2605.12167)  
**Code**: https://logosroboticsgroup.github.io/MoLA (Available)  
**Area**: Robotics  
**Keywords**: Latent Actions, Inverse Dynamics, Video Generation, Robot Manipulation, Modality Awareness

## TL;DR
MoLA employs a set of "Modality-aware Inverse Dynamics Models (IDM)" pre-trained on large-scale robotics data to translate future frames predicted by a video generation model into three-way discrete latent actions (semantic/depth/optical flow). A policy head then performs control based on these action-centric representations, rendering the "imagine-execute" interface both stable and precise across CALVIN, LIBERO, LIBERO-Plus, and real-world UR5e platforms.

## Background & Motivation

**Background**: Current robot manipulation follows two main paradigms. One is VLA (e.g., RT-1, $\pi_0$, OpenVLA), which learns an end-to-end action head directly from vision and language. The other is imagination-based policy (e.g., UniPi, VPP, DreamGen), which predicts several future frames using video generation models and then feeds these "imagined futures" to the policy. The video imagination route naturally anticipates long-term outcomes, making it a compelling direction.

**Limitations of Prior Work**: There are two naive ways to drive policies using imagined frames, both of which are suboptimal. The first treats predicted frames as additional visual conditions for the action head, effectively offloading the entire task of "extracting control signals from visual changes" to the action head. The second directly decodes video into actions, making control entirely dependent on video prediction accuracy; if long-range video prediction drifts, the actions drift accordingly.

**Key Challenge**: The optimization objective of video generation models is "perceptual realism"—such as pixel MSE or perceptual loss—rather than "control relevance." Consequently, even if the predicted frames appear realistic, they do not explicitly reveal the physical actions driving the transitions between states. The path from image space to action space is inherently indirect and unstable.

**Goal**: To insert an "action-centric interface" between video imagination and policy execution, allowing the policy to reason in action space rather than pixel space. This interface must reliably infer actions from (potentially noisy) generated future frames and remain compatible with multi-modal information.

**Key Insight**: The authors observe that Inverse Dynamics Models (IDM) are naturally suited for this task—defined as "inferring the intermediate action given the visual transition from $o_t$ to $o_{t+k}$." Applying an IDM to generated future frames is equivalent to "back-decoding" the video generation output into actions. Since a single IDM might compress all signals into one codebook, the authors further observe that manipulation relies on three complementary cues: semantics (task intent), depth (3D structure), and optical flow (interaction dynamics).

**Core Idea**: Use "modality-specialized multi-path inverse dynamics models pre-trained on large-scale robotics data" to translate generated future videos into a set of discrete latent actions (Mixture of Latent Actions), serving as the interface between video imagination and the policy head.

## Method

### Overall Architecture
Given the current RGB observation $o_t$ and language instruction $l$, MoLA follows four steps: (1) A video generation model (using Stable Video Diffusion as the backbone with single-step denoising during inference) synthesizes future frames $\hat{o}_{t:t+H}$ as the "imagination space"; (2) Three-way IDMs (semantic/depth/flow) infer transitions from the frame pair $(o_t, \hat{o}_{t+k})$, each outputting discrete latent actions $z_{t \to t+k}^{(m)}$; (3) The three-way latent actions are concatenated into a mixture $\mathcal{Z}$; (4) An action head based on Diffusion Transformer + flow matching consumes the latent actions and generated visual features to output continuous executable control sequences. Training occurs in three stages: fine-tuning the video generation model, pre-training the MoIDM, and finally end-to-end fine-tuning of the MoIDM and action head (while the video generation model remains frozen).

### Key Designs

1.  **Modality-aware Inverse Dynamics Model (MoIDM)**:
    *   **Function**: Translates "visual changes between two RGB frames" into discrete latent action tokens, serving as the core interface connecting imagination and control.
    *   **Mechanism**: Each IDM path has an independent spatio-temporal Transformer $T^{(m)}$ and an independent VQ codebook. A ViT encoder processes $o_t$ and $o_{t+k}$ into features, and a set of learnable latent action queries interacts with these features through $T^{(m)}$ to output $\tilde{h} = T^{(m)}(q^{(m)}, h_t, h_{t+k})$. Finally, VQ produces $z^{(m)} = \text{VQ}^{(m)}(\tilde{h})$. The three IDMs share the RGB inference pipeline but are biased toward different modalities via reconstruction targets (SAM2 for semantics, Depth Anything v2 for depth, and CoTracker3 for optical flow).
    *   **Design Motivation**: A single IDM tends to conflate various factors in one codebook, leading to interference. By specializing via modalities, each codebook is explicitly constrained to capture one of "task semantics, geometric structure, or interaction motion," maximizing complementarity. Ablations show significant performance drops when merging the three paths into one or sharing a codebook.

2.  **Large-scale Pre-training + Joint Fine-tuning Strategy**:
    *   **Function**: Enables the IDM to work not only on real frames but also to infer actions stably from generated future frames that may contain noise or slight drift.
    *   **Mechanism**: The three IDM paths are first pre-trained independently on large-scale robotics data (e.g., Open X-Embodiment + AgiBot) using real future frames. Subsequently, on downstream tasks, the video generation model is frozen while the MoIDM and action head are fine-tuned end-to-end. This allows the MoIDM to retain general action semantics from large-scale visual changes while co-adapting to the "generated frame" distribution.
    *   **Design Motivation**: Ablations (Q2/Q3) indicate that removing pre-training (training from scratch) leads to severe performance degradation, while freezing the MoIDM prevents it from evolving with the policy. The MoIDM must be both "well-informed via pre-training" and "tunable downstream."

3.  **Flow Matching-based Diffusion Action Head**:
    *   **Function**: Decodes discrete latent actions and generated visual features into continuous, time-correlated, multi-modal robot control sequences.
    *   **Mechanism**: The action head utilizes a Diffusion Transformer (DiT) architecture trained from scratch. The training objective is flow matching rather than standard diffusion loss, learning the continuous transformation from noise action samples to target actions. Conditional inputs include the three-way latent actions and visual features from predicted frames.
    *   **Design Motivation**: Latent actions are "intermediate cues distilled from imagination," but final control remain a continuous multi-modal distribution. Ablation Q4 shows that while a light autoregressive token head can work (proving the informativeness of latent actions), DiT + flow matching is superior in handling long-term dependencies, multi-modality, and robustness to imperfect latent actions.

### Loss & Training
During the IDM pre-training stage, each path uses two reconstruction objectives: a shared RGB decoder to reconstruct future RGB frames, and a modality-specific decoder to reconstruct features/labels from foundation models. In the downstream stage, the video generation model is frozen, and the MoIDM and action head are trained end-to-end using flow matching loss. The overall pipeline follows a three-stage process: "video fine-tuning $\to$ MoIDM pre-training $\to$ end-to-end fine-tuning."

## Key Experimental Results

### Main Results
Evaluation covers CALVIN ABC-D long-horizon tasks, four LIBERO lifelong suites, LIBERO-Plus (10,030 tasks with perturbations), and a real UR5e robot.

| Benchmark | Metric | MoLA | Prev. SOTA | Gain |
|-----------|------|------|----------|------|
| CALVIN ABC-D | Avg. Len. | 4.55 | DreamVLA 4.44 | +0.11 |
| LIBERO (Avg of 4) | Success % | 97.0 | VPP 90.9 | +6.1 |
| LIBERO-Plus | Avg. % | 92.7 | OpenVLA-OFT+ 79.5 | +13.2 |
| Real Robot (Avg in+OOD) | Success % | 73.0 | VPP 62.0 | +11.0 |

On CALVIN, MoLA pushes the success rate for "completing 5 consecutive sub-tasks" from the previous SOTA of 76.9% (VPP) to 82.6%. The 13.2 point gain on LIBERO-Plus specifically highlights its stability across robustness dimensions such as interference, lighting changes, and initial state variations. On the real robot, while it does not surpass the commercial-grade $\pi_{0.5}$, it maintains a steady 11-point advantage over similar "video imagination + policy" routes (VPP).

### Ablation Study

| Modality Combination | CALVIN Avg. Len. ↑ | Description |
|---------|--------------------|------|
| Baseline (No MoIDM, direct future frames) | 4.24 | Weakest |
| Sem only | 4.31 | Task semantics are independently useful |
| Depth only | 4.35 | 3D structure provides stronger cues |
| Flow only | 4.39 | Strongest single modality (interaction dynamics is most critical) |
| Flow + Depth | 4.46 | Bi-modal complementarity |
| All (Sem + Depth + Flow) | **4.55** | Tri-modal full configuration is optimal |

### Key Findings
- The addition of the three-way IDM leads to a monotonic increase in Avg. Len. (4.24 $\to$ 4.31 $\to$ 4.35 $\to$ 4.39 $\to$ 4.46 $\to$ 4.55), proving they are truly complementary rather than redundant. Among them, optical flow contributes the most, reinforcing that "the essence of manipulation is interaction dynamics."
- Ablation Q1 shows that even the "baseline (no IDM)" is weaker than schemes that simply feed visual features after removing the IDM, suggesting that the VQ discrete latent action bottleneck itself is an effective inductive bias that makes "image $\to$ action" mapping easier to learn.
- In terms of data efficiency, MoLA significantly outperforms VPP with only 10% of CALVIN data, indicating that the "pre-trained IDM + latent action interface" provides a strong prior, particularly advantageous in low-data scenarios.
- Replacing "imagined futures" with "two copies of the same frame" or "noisy current frames" during inference leads to significant performance drops, showing that the latent action mechanism and genuine future cues are both indispensable.

## Highlights & Insights
- **The observation that a single IDM is insufficient directly led to the modality-specialized scheme**, avoiding the common pitfall of conflating all signals into one codebook. This pattern of "training one expert per modality + using foundation models as supervision" is highly transferable to other tasks bridging vision and action/language.
- **The discrete VQ bottleneck is more critical than expected**: The baseline without IDM was weaker than the weakest single-path IDM, suggesting that compressing high-dimensional visual changes into discrete tokens helps the policy filter out control-irrelevant details—a principle applicable to other "high-dimensional condition $\to$ low-dimensional decision" problems (e.g., GUI agents).
- **The three-part division of video model as "generator," IDM as "interpreter," and action head as "executor"** allows MoLA to seamlessly swap underlying video models (Ablation Q5 showed gains when replacing SVD with Wan2.2). This is a clean way to automatically incorporate progress in video generation into robot control.

## Limitations & Future Work
- The video generation model remains frozen, limiting the upper bound of joint optimization; future work using lighter video models that can be fine-tuned might further reduce the distribution gap between generated and real frames.
- In real-world experiments, "ultra-large-scale corporate VLAs" like $\pi_{0.5}$ still outperform MoLA in average scores, suggesting that the "video imagination + interface" route is not yet a total replacement for end-to-end VLAs trained on massive datasets, but rather a superior compromise for medium-scale data.
- Modality selection is fixed to semantics, depth, and flow; adding tactile, kinematic, or force feedback for real robots was not discussed. Multi-modal expansion and learned modality weighting are natural next steps.
- Although only single-step denoising is used during inference, the pipeline still requires running video generation, three IDMs, and a DiT action head; the total latency might be tight for high-frequency control scenarios.

## Related Work & Insights
- **vs VLA (OpenVLA / $\pi_0$ / GR00T)**: These map perception directly to action without explicit "imagination" steps; MoLA adds an imagination layer, proving more stable in long-horizon tasks and OOD scenarios at the cost of higher inference overhead.
- **vs VPP (Video Prediction Policy)**: Both predict the future to drive policy, but VPP treats generated frames as conditions, forcing the action head to learn "visual change $\to$ control" from scratch; MoLA inserts MoIDM to "pre-digest" this mapping, improving CALVIN Avg. Len. from 4.33 to 4.55.
- **vs DreamGen / UniPi**: These use video generation as the policy backbone to decode actions directly; MoLA emphasizes that the latent action interface is key, offering more robustness than pure video $\to$ action.
- **vs LAPA / UniVLA**: These also use IDM/latent action ideas but rely on single encoding without explicit modality decomposition; MoLA’s "modality specialization + VQ codebook" represents a further refinement of this lineage.

## Rating
- Novelty: ⭐⭐⭐⭐ The "three-way modality-aware IDM + discrete latent action interface" is a clean design, though IDM and LAPA-like concepts exist in the literature; the strength lies in the solid execution and refinement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three simulation suites, real-world deployment, data efficiency, and 7 ablation questions; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The narrative is clear, moving logically from the "visual realism vs. control relevance" contradiction to the MoIDM solution.
- Value: ⭐⭐⭐⭐ Provides the most convincing interface design to date for "video imagination-driven robotics" and will likely serve as a strong baseline for future video-based VLA research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](../../ICLR2026/robotics/from_spatial_to_actions_grounding_vision-language-action_model_in_spatial_founda.md)
- [\[ICML 2026\] Mixture of Horizons in Action Chunking](mixture_of_horizons_in_action_chunking.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](../../CVPR2026/robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)

</div>

<!-- RELATED:END -->
