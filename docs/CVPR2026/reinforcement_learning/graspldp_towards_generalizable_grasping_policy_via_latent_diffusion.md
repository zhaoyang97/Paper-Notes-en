---
title: >-
  [Paper Note] GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion
description: >-
  [CVPR 2026][Reinforcement Learning][Robot Grasping] This paper proposes GraspLDP, which injects grasp pose priors from a pretrained grasp detector and graspness map visual cues into a latent diffusion policy framework. B…
tags:
  - "CVPR 2026"
  - "Reinforcement Learning"
  - "Robot Grasping"
  - "Latent Diffusion Policy"
  - "Grasp Prior"
  - "Imitation Learning"
  - "Generalization"
date: 2026-05-08
content_hash: f243faddc17bc052
---

# GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion

**Conference**: CVPR 2026
**arXiv**: [2602.22862](https://arxiv.org/abs/2602.22862)
**Code**: Available (Project Page)
**Area**: Reinforcement Learning
**Keywords**: Robot Grasping, Latent Diffusion Policy, Grasp Prior, Imitation Learning, Generalization

## TL;DR
This paper proposes GraspLDP, which injects grasp pose priors from a pretrained grasp detector and graspness map visual cues into a latent diffusion policy framework. By leveraging VAE-encoded action latent spaces for guidance and a self-supervised reconstruction objective, GraspLDP substantially improves grasping accuracy and generalization.

## Background & Motivation
Grasping is the critical initial step for physical interaction in robotic manipulation pipelines. Visuomotor policies based on imitation learning (e.g., Diffusion Policy, ACT) have demonstrated promise on general manipulation tasks, yet they often underperform specialized grasp detection methods on the grasping subtask, as modeling the full action sequence for grasping is inherently more complex.

Existing methods that integrate grasp priors into imitation learning (e.g., Robograsp, GPA-RAM) suffer from two problems: (1) grasp poses are simply concatenated as conditional inputs to the policy model, resulting in weak association between the grasp pose and the output action sequence and thus providing ineffective guidance; (2) the modality mismatch between low-semantic grasp poses and high-dimensional visual inputs makes it difficult for the policy model to fully extract the spatial distribution of grasping. On the other hand, the data-driven approach GraspVLA, while strong, requires 160 RTX 4090 GPUs training for 10 days to generate 1B frames of simulation data, imposing extremely high costs.

**Core Idea**: Drawing inspiration from the success of latent diffusion models in image generation, GraspLDP injects precise target grasp poses into the action latent space to guide action generation, while using graspness maps as visual cues to condition the diffusion process, bridging static target grasp poses and dynamic action sequences through a shared latent space.

## Method

### Overall Architecture
GraspLDP adopts a two-stage training procedure: (1) **Action Latent Learning**: a VAE encodes action sequences into a compact latent space, with grasp pose priors injected into the decoder to guide action reconstruction; (2) **Diffusion on Latent Action Space**: a diffusion policy is trained on the latent space, conditioned on graspness map visual cues during denoising, with an additional self-supervised reconstruction objective to enforce cue utilization.

### Key Designs

1. **Grasp Guidance in Latent Space**:

    - **Function**: A lightweight VAE compresses action sequences into compact latent features; the target grasp pose is concatenated at the decoder for reconstruction.
    - **Mechanism**: Encoding $\mathbf{Z} = \mathcal{E}(A)$, decoding $\hat{A} = \mathcal{D}(\mathbf{Z} \oplus \mathcal{G})$, with loss $\mathcal{L}_{VAE} = \text{MSE}(A, \hat{A}) + \lambda \mathcal{L}_{KL}$. The diffusion model denoises over the compact latent features rather than directly in the high-dimensional action space.
    - **Design Motivation**: Directly conditioning on grasp poses dilutes guidance strength and increases training difficulty. Injection into the latent space allows the grasp pose to exert a direct, strongly constrained influence on action reconstruction at the decoder stage. The lower-dimensional latent space also accelerates inference.

2. **Visual Graspness Cue**:

    - **Function**: Per-point graspness scores are obtained from a pretrained graspness network on the point cloud, back-projected into pixel space to form a graspness map, and overlaid onto the wrist camera image as a visual cue.
    - **Mechanism**: $O_{cue}(j,k) = \text{masked\_color}$ if $M(j,k) > \tau$, otherwise the original pixel is retained. A self-supervised reconstruction target is additionally imposed at each reverse diffusion step: $\mathcal{L}_{Recon.} = \text{MSE}(O_{cue}, \hat{O}_{cue})$, with total loss $\mathcal{L}_{LDP} = \mathcal{L}_{Diff.} + \lambda_{Recon.} \mathcal{L}_{Recon.}$.
    - **Design Motivation**: The graspness map is a geometry-driven, illumination-invariant indicator of grasp feasibility, guiding the end-effector toward graspable regions. The self-supervised reconstruction ensures the model genuinely attends to these visual cues rather than ignoring them.

3. **Heuristic Pose Selector (HPS)**:

    - **Function**: At inference time, selects the most appropriate grasp pose from the candidates predicted by the grasp detector to serve as guidance.
    - **Mechanism**: Candidates are first filtered via collision detection and NMS, retaining the top-k quality candidates. The SE(3) geodesic distance $d_{\mathcal{G}_j, W} = \sqrt{\xi^\top W \xi}$ between the current end-effector pose $P$ and each candidate $\mathcal{G}_j$ is computed, and the closest candidate is selected: $\mathcal{G}^* = \arg\min d(\mathcal{G}_j)$.
    - **Design Motivation**: Jointly considering grasp quality and kinematic proximity balances grasp feasibility with trajectory smoothness, avoiding degraded success rates caused by unreasonable pose guidance.

### Loss & Training
Two stages: Stage 1 trains the VAE (MSE + KL); Stage 2 trains the latent diffusion policy (diffusion loss + self-supervised reconstruction loss). Training data consists of approximately 12K high-quality grasping demonstrations on the LIBERO benchmark.

## Key Experimental Results

### Main Results

| Method | In Domain | Spatial Gen. | Object Gen. | Visual Gen. | Avg. |
|--------|-----------|-------------|-------------|-------------|------|
| Diffusion Policy | 62.8 | 48.9 | 11.4 | 16.3 | 34.9 |
| GraspVLA | 50.8 | 49.5 | 46.8 | 51.7 | 49.7 |
| Ours Baseline (CG) | 72.3 | 59.1 | 48.3 | 47.7 | 56.9 |
| **GraspLDP** | **80.3** | **71.1** | **58.2** | **64.6** | **68.6** |

### Ablation Study

| Configuration | ID SR | SG SR | OG SR | VG SR |
|---------------|-------|-------|-------|-------|
| GraspLDP (full) | 80.3 | 71.1 | 58.2 | 64.6 |
| w/o Graspness Cue | 77.4 (-2.9) | 67.3 (-3.8) | 54.2 (-4.0) | 57.5 (-7.1) |
| w/o Latent Guidance w/ CG | 73.5 (-6.8) | 62.2 (-8.9) | 52.3 (-5.9) | 54.5 (-10.1) |
| w/o Latent Guidance | 60.6 (-19.7) | 49.8 (-21.3) | 21.2 (-37.0) | 19.4 (-45.2) |
| w/o GC & LG | 55.1 (-25.2) | 46.2 (-24.9) | 16.0 (-42.2) | 15.7 (-48.9) |

### Key Findings
- GraspLDP improves in-domain grasping success rate by 17.5% over Diffusion Policy, with spatial/object/visual generalization gains of 22.2%, 46.8%, and 48.3%, respectively.
- Latent Guidance is the most critical component; its removal causes OG/VG to drop to ~20%, demonstrating that latent-space grasp pose guidance is essential for generalization.
- Graspness Cue yields the largest improvement on Visual Generalization (-7.1%), as geometry-driven graspness cues are robust to illumination variations.
- HPS outperforms random/highest/nearest selection strategies, confirming that jointly considering quality and kinematic proximity is necessary.
- Inference latency is only ~15% higher than Diffusion Policy, far faster than GraspVLA.

## Highlights & Insights
- The paper transfers the latent diffusion paradigm from image generation to robotic action generation; injecting priors in the latent space proves more effective than conditional concatenation.
- The graspness map as a visual prompt is an elegant and effective design; self-supervised reconstruction enforces information utilization.
- In real-world experiments, GraspLDP achieves a Scene Completion Rate of 96.2% in cluttered scenes, approaching the open-loop method AnyGrasp at 92.3%.

## Limitations & Future Work
- The approach relies on a pretrained grasp detection network (e.g., AnyGrasp); if the detector fails on novel objects, the quality of pose priors degrades.
- The current method targets only the grasping subtask and has not been extended to full long-horizon manipulation.
- VAE training and diffusion training are conducted in two separate stages; end-to-end joint training may yield further improvements.

## Related Work & Insights
- Diffusion Policy established the paradigm of diffusion models for action generation; GraspLDP extends this to the latent space and incorporates task-specific priors.
- GraspVLA pursues a data-driven route (1B frames), whereas GraspLDP adopts a prior-injection route, achieving higher efficiency with substantially less data.
- The latent-space guidance idea is generalizable to other manipulation tasks that require prior knowledge (e.g., assembly, tool use).
- PPI guides continuous action generation with discrete keyposes; GraspLDP achieves finer-grained guidance by operating within the latent space.
- The graspness concept from GSNet is repurposed as a visual prompt, demonstrating the synergistic potential between grasp detection and policy learning.

## Supplementary Details
- Real-world cluttered scene grasping: GraspLDP achieves SCR of 96.2%, with an average SR of 80% across four scenes.
- Graspness computation requires only 36 ms at inference; latent decoding takes less than 1 ms, keeping overall latency within ~100 ms.
- The VAE employs an asymmetric decoder: the encoder does not receive the grasp pose, while the decoder does — this information flow ensures the latent encodes the action itself, with the decoder injecting pose modulation.
- The GFE (Grasp Frame Error) metric is introduced as a novel evaluation measure based on SE(3) geodesic distance to assess how accurately the policy follows grasp pose guidance.
- Training requires only 12K demonstrations, far fewer than GraspVLA's 1B frames, yet achieves comprehensively superior generalization performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Latent-space grasp pose injection and self-supervised reconstruction of graspness visual cues are noteworthy contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Simulation and real-world evaluation, multi-dimensional generalization assessment, detailed ablations, and HPS ablations are all comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, detailed method description, and well-designed experiments.
- **Value**: ⭐⭐⭐⭐⭐ Addresses practical challenges in policy-based grasping generalization; the +46.8% OG improvement is of significant importance for real-world deployment.

## Key Terminology
- **Graspness**: Per-point graspability score on a point cloud; a geometry-driven measure of grasp feasibility.
- **Latent Diffusion Policy**: Diffusion-based denoising performed in the VAE-encoded action latent space.
- **SE(3) Geodesic Distance**: Geodesic distance on the special Euclidean group, providing a unified measure of rotational and translational discrepancy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](../../ICLR2026/reinforcement_learning/thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)
- [\[AAAI 2026\] Formal Verification of Diffusion Auctions](../../AAAI2026/reinforcement_learning/formal_verification_of_diffusion_auctions.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](../../AAAI2026/reinforcement_learning/object-centric_latent_action_learning.md)
- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](../../ACL2026/reinforcement_learning/spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)

</div>

<!-- RELATED:END -->
