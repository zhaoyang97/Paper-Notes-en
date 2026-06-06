---
title: >-
  [Paper Note] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges
description: >-
  [NeurIPS 2025][Robotics][Dexterous Grasp Transfer] This paper proposes modeling cross-morphology visual dexterous grasp transfer as a Schrödinger Bridge problem. By learning Score and Flow Matching ([SF]²M) in a latent s…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Dexterous Grasp Transfer"
  - "Schrödinger Bridge"
  - "Optimal Transport"
  - "Score and Flow Matching"
  - "Physics-Guided Cost Function"
date: 2026-05-08
content_hash: 2fdc218d8f58e35f
---

# Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges

**Conference**: NeurIPS 2025
**arXiv**: [2506.02489](https://arxiv.org/abs/2506.02489)  
**Code**: [grasp2grasp.github.io](https://grasp2grasp.github.io)  
**Area**: Robotics
**Keywords**: Dexterous Grasp Transfer, Schrödinger Bridge, Optimal Transport, Score and Flow Matching, Physics-Guided Cost Function

## TL;DR

This paper proposes modeling cross-morphology visual dexterous grasp transfer as a Schrödinger Bridge problem. By learning Score and Flow Matching ([SF]²M) in a latent space and designing physics-aware optimal transport cost functions (over pose, contact maps, grasp wrench space, and Jacobian manipulability), the method achieves distribution-level transfer of grasp intent across different robot hands without requiring paired data.

## Background & Motivation

Dexterous multi-finger robotic hands offer high degrees of freedom and flexibility in grasping tasks; however, their high-dimensional configuration spaces and complex contact dynamics create a "hand-specific" bottleneck for data-driven approaches—models trained for one hand cannot be directly transferred to another. Existing grasp transfer methods suffer from the following limitations:

**Teleoperation / Behavioral Cloning**: Directly copying joint angles or end-effector poses yields physically infeasible grasps when the source and target hand morphologies differ significantly.

**Retargeting Methods** (e.g., Dex-Retargeting, CrossDex): Rely on manually annotated link correspondences, limiting generality, and perform point-wise mapping rather than distribution-level transfer.

**RobotFingerPrint**: Maps grasps through a unified coordinate space but requires simulation preprocessing and is object-agnostic, ignoring functional grasp intent.

**Lack of Paired Data**: Obtaining one-to-one paired grasp data across source and target hands is extremely difficult.

The core motivation of this paper is: **Is it possible, without paired data, to learn a distribution-level mapping that preserves functional grasp intent, starting only from unpaired grasp datasets of each hand?**

## Method

### Overall Architecture

The method adopts a two-stage pipeline (illustrated in Figure 3 of the paper):

**Stage 1: VAE Encoding of Grasp Observations**

- A VAE with a PVCNN backbone encodes the segmented source-hand grasp visual observations into a latent space $z \in \mathcal{Z}$.
- The decoder outputs the grasp configuration $\hat{g} = \text{Dec}(z)$, which is passed through a differentiable forward kinematics layer FK to obtain 3D hand mesh vertices.
- VAE loss: $\mathcal{L}_{VAE} = \mathbb{E}[\|\hat{g} - g\|^2 + \alpha\|\hat{v} - \text{FK}(g)\|^2] + \beta \text{KL}(q_{\text{Enc}}(z|o) \| \mathcal{N}(0,I))$

**Stage 2: Latent-Space Schrödinger Bridge**

- A Schrödinger Bridge is formulated in the VAE latent space to model the transport from the source hand distribution $q_0(z_{\text{source}}|o_{\text{obj}})$ to the target hand distribution $q_1(z_{\text{target}}|o_{\text{obj}})$.
- A U-ViT backbone is used to learn the score function $s_\theta(t,z)$ and the velocity field $v_\theta(t,z)$.
- Conditioning inputs include: the object point cloud (encoded by LION VAE as 1 global token + 512 local tokens), the hand observation latent (1 token), and 5 contact anchor tokens.

**Inference Pipeline**: Source hand observation → VAE encoding $z_0$ → Forward evolution along the learned SDE dynamics → Translated latent code $z_1$ → VAE decoding → Target hand grasp configuration.

### Key Designs: Physics-Aware OT Cost Functions

Conventional Schrödinger Bridges use Euclidean distance as the transport cost, which fails to capture functional grasp equivalence. This paper designs four physics-aware cost terms:

**1. Base Pose Cost $d_{\text{pose}}$**

$$d_{\text{pose}} = \|h_{\text{source}} - h_{\text{target}}\|_2^2 + \|R_{\text{source}} - R_{\text{target}}\|_F^2$$

The translation component uses the L2 norm; the rotation component uses the Frobenius norm (rotation matrices converted via the 6D continuous representation), encouraging coarse workspace alignment.

**2. Contact Map Similarity $d_{\text{contact}}$**

$$d_{\text{contact}} = \text{Chamfer}(C_{\text{source}}, C_{\text{target}})$$

Contact maps are represented as 3D point clouds, and the bidirectional Chamfer distance is computed to preserve local interaction geometry on the object surface.

**3. Grasp Wrench Space Overlap $d_{\text{wrench}}$**

$$d_{\text{wrench}} = 1 - \frac{\text{Vol}(\text{Hull}(GWS_{\text{source}}) \cap \text{Hull}(GWS_{\text{target}}))}{\text{Vol}(\text{Hull}(GWS_{\text{source}}) \cup \text{Hull}(GWS_{\text{target}}))}$$

Convex hulls of the wrench sets are computed from contact positions and normals; the IoU measures the mechanical capability similarity between the two grasps. In practice, the computation is reduced to the first 3 dimensions (centroidal forces) and IoU is estimated via Monte Carlo sampling.

**4. Jacobian Manipulability $d_{\text{jac}}$**

$$d_{\text{jac}} = \|m_{\text{source}} - m_{\text{target}}\|^2$$

Grasps are executed in the differentiable simulator Warp; the Jacobian of the object pose with respect to joint angles is computed, and the maximum-effect 6D vector across joint dimensions is extracted. This encourages the translated grasp to preserve similar object controllability.

### Loss & Training

The core training objective is the [SF]²M (Score and Flow Matching) loss:

$$\mathcal{L}_{[\text{SF}]^2\text{M}}(\theta) = \mathbb{E}\left[\|v_\theta(t,z) - u_t^\circ(z|z_0,z_1)\|^2 + \lambda(t)^2\|s_\theta(t,z) - \nabla\log p_t(z|z_0,z_1)\|^2\right]$$

where $(z_0, z_1) \sim \pi_\varepsilon^*$ (the entropy-regularized OT plan computed via the Sinkhorn algorithm), and $z \sim p_t(z|z_0,z_1)$ (conditional Brownian bridge sampling). Minibatch OT approximation is employed for computational efficiency.

## Training & Inference

### Training Pipeline

Training proceeds in two sequential and independent stages:

**Stage 1 (VAE Training)**: A PVCNN-based VAE is trained separately for each hand morphology. The input is the segmented hand point cloud observation $o$, encoded into latent variable $z \sim \text{Enc}(o)$, decoded to grasp configuration $\hat{g} = \text{Dec}(z)$, and passed through the differentiable FK layer to obtain 3D mesh vertices $\hat{v} = \text{FK}(\hat{g})$. The loss comprises three terms: configuration reconstruction error, mesh vertex reconstruction error, and KL regularization. Encoder and decoder parameters are frozen after training.

**Stage 2 (Schrödinger Bridge Training)**: With the VAE frozen, a U-ViT network is trained in the latent space to learn the SB dynamics. Each minibatch training step proceeds as follows:
1. Sample a batch of observations $o_s, o_t$ from the source and target hand datasets, respectively.
2. Obtain latent codes $z_s = \text{Enc}(o_s)$ and $z_t = \text{Enc}(o_t)$ via the VAE encoder.
3. Compute the entropy-regularized OT plan $\pi_\varepsilon^*$ on the current minibatch via the Sinkhorn algorithm, with the cost matrix determined by the selected physics-aware cost function.
4. Sample pairs $(z_0, z_1)$ from $\pi_\varepsilon^*$ and sample intermediate points $z_t$ from the conditional Brownian bridge.
5. Update $v_\theta$ and $s_\theta$ using the [SF]²M loss.

Key implementation details: Minibatch OT avoids the $O(n^2)$ cost of exact OT; the GWS cost reduces the 6D convex hull to the first 3 dimensions (centroidal force directions) with IoU estimated by Monte Carlo; the Jacobian cost is computed online using the Warp differentiable simulator.

### Inference Pipeline

Given a visual observation of a source hand grasp $(o_{\text{obj}}, o_{\text{hand}})$:
1. Encode the initial latent code $z_0 = \text{Enc}(o_{\text{hand}})$ via the source hand VAE encoder.
2. Integrate forward along the learned SDE dynamics using Euler–Maruyama integration with an adjustable number of steps (default: 100).
3. Obtain the translated latent code $z_1$.
4. Decode the target hand grasp configuration $g_{\text{target}} = \text{Dec}(z_1)$ via the target hand VAE decoder.

Inference requires **no test-time fine-tuning**; the entire process is purely feed-forward amortized inference. Conditioning information (object point cloud encoding, contact anchors) is injected via cross-attention in the U-ViT.

## Key Experimental Results

### Main Results: Grasp Translation Quality (Table 1)

Dataset: MultiGripperGrasp (30.4M grasps, 11 robot hands, 345 objects). Training on 138 objects; testing on 34 unseen objects.

| Method | Mean Success Rate↑ | Diversity (rad)↑ | 6D GWH IoU↑ |
|---|---|---|---|
| CrossDex | 26.87% | 0.206 | 5.26% |
| Dex-Retargeting | 43.19% | 0.201 | 5.68% |
| RobotFingerPrint | 57.00% | 0.203 | 7.77% |
| Diffusion baseline | 60.83% | 0.271 | 9.58% |
| **Ours (pose)** | 64.32% | 0.250 | 10.68% |
| **Ours (contact)** | 62.75% | 0.272 | **14.54%** |
| **Ours (GWH)** | 66.34% | 0.266 | **14.63%** |
| **Ours (jacobian)** | **67.45%** | 0.258 | 10.88% |

Key findings:
- The Jacobian cost achieves the highest success rate (67.45%), outperforming the strongest baseline by 6.6 percentage points.
- The GWH and Contact costs substantially outperform baselines on functional alignment (GWH IoU), achieving approximately 1.5–2.5× the baseline values.
- Optimization-based methods (Dex-Retargeting/CrossDex) produce many physically invalid grasps due to neglect of palm collisions.

### Ablation Study

**Effect of Diffusion Rate σ (Table 2, H→A task):**

| σ | Success Rate↑ | IoU↑ |
|---|---|---|
| 0.01 | 77.16% | 16.72% |
| 0.1 | 74.78% | 15.89% |
| 1.0 | 64.39% | 13.83% |

Smaller diffusion rates yield more precise transport trajectories at the cost of slightly reduced diversity.

**Effect of Discretization Steps (Table 3, H→A task):**

| Steps | Ours Success Rate | Diffusion Success Rate | Ours IoU | Diffusion IoU |
|---|---|---|---|---|
| 10 | 71.45% | 60.00% | 13.80% | 6.19% |
| 100 | 75.12% | 72.57% | 14.00% | 9.54% |
| 200 | 74.78% | 73.17% | 15.89% | 9.66% |

The SB method significantly outperforms the Diffusion baseline even at low step counts, demonstrating more stable and sample-efficient generation dynamics.

### Physical Functional Alignment Analysis

The 6D GWH IoU results reveal a core trade-off:
- The **Jacobian cost** achieves the highest success rate (67.45%), indicating that manipulability best captures the kinematic relationships necessary for stability.
- The **GWH and Contact costs** substantially lead on functional alignment metrics (14.63% and 14.54% vs. 9.58% for the baseline), indicating that directly optimizing for mechanical capability or contact patterns more effectively transfers the underlying mechanical intent of grasps.
- This trade-off implies a tension between stability (Jacobian) and functional equivalence (GWH/Contact); no single cost function simultaneously optimizes both.

### Efficiency Comparison

- RobotFingerPrint: >5 seconds per grasp (per-sample iterative optimization)
- Proposed method (100 steps): ~0.8 seconds per grasp (batch generation, amortized inference cost)

## Highlights & Insights

1. **Problem Formulation Innovation**: This is the first work to model cross-morphology grasp transfer as a Schrödinger Bridge problem, naturally handling distribution-level mapping without paired data.
2. **Physics-Aware Cost Design**: Each of the four cost functions serves a distinct purpose—Jacobian best supports stability, GWH/Contact best supports functional alignment—providing choices for different application scenarios.
3. **No Hand-Specific Simulation Preprocessing**: Unlike RobotFingerPrint, the proposed method requires no hand-specific coordinate mapping for each new hand morphology.
4. **Simulation-Free Training**: Full stochastic process simulation is avoided, making training scalable.
5. **Low-Step Robustness**: Even at 10 steps, the method substantially outperforms the Diffusion baseline, making it deployment-friendly.

## Limitations & Future Work

1. **Degraded Performance on Thin-Shell Objects**: Performance degrades on thin-shell objects with ambiguous or minimal contact regions (as shown in the right portion of Figure 4), where the space of valid grasp configurations is severely constrained.
2. **Lower Consistency on the Shadow Hand**: Due to the Shadow hand's more complex kinematics and larger workspace, all methods perform lower on the H→S task compared to H→A.
3. **No Support for Unseen Hand Morphologies at Inference**: The VAE decoder implicitly encodes the kinematic structure of the target hand; retraining is required for unseen morphologies.
4. **Trade-off Among Cost Functions**: The optimum for stability (Jacobian) and the optimum for functional alignment (GWH) cannot be achieved simultaneously; a unified cost function is lacking.
5. **Evaluation Limited to Simulation**: Real-robot transfer results are not demonstrated.

## Related Work & Insights

- **Relation to GenDexGrasp/UGG**: These methods also address multi-hand grasping but remain hand-specific generative models and do not perform cross-morphology transfer.
- **Relation to Schrödinger Bridge Literature**: The method inherits the simulation-free training paradigm of the [SF]²M framework, with the key innovation being the replacement of general geometric distance in OT costs with physically semantic cost functions.
- **Broader Inspiration**: The physics-aware transport cost design can be generalized to other distribution transfer tasks requiring semantic invariance (e.g., cross-domain action transfer, cross-morphology motion retargeting).
- **Potential Extensions**: Combining multiple cost functions (multi-objective OT) or using conditioned cost weights for task-adaptive transfer are natural next steps.

## My Notes

### Assessment of Methodological Contribution
The most valuable contribution of this paper is not the Schrödinger Bridge framework itself (which directly adopts [SF]²M) but rather the **design of physics-aware cost functions**. Within the OT/SB framework, the choice of ground cost determines the semantic meaning of transport—generic Euclidean distance only achieves geometric alignment and cannot preserve functional intent. The four costs encode grasp equivalence from distinct physical dimensions: pose (coarse alignment), contact map (local interaction), grasp wrench space (mechanical capability), and Jacobian (controllability). This design philosophy has strong transferability.

### Fundamental Distinction from Existing Methods
Traditional retargeting methods (Dex-Retargeting, CrossDex) perform **point-wise mapping**—finding one target grasp for each source grasp—and require manually annotated link correspondences. The proposed method is a **distribution-level mapping**—learning the transport from the source hand grasp distribution to the target hand grasp distribution—naturally handling one-to-many and many-to-one scenarios without requiring explicit hand structure correspondences.

### Deeper Reflection on Limitations
The most significant limitation lies in the VAE decoder's implicit binding to the target hand kinematics, which prevents direct transfer to unseen hand morphologies at inference time, leaving a gap from the ultimate goal of "universal cross-morphology" transfer. A possible remedy would be to condition the decoder on hand morphology, or to use a hypernetwork that dynamically generates decoder weights from hand morphology parameters. Furthermore, the trade-off among the four cost functions suggests the existence of Pareto-optimal solutions—multi-objective OT or weighted combinations are natural next steps.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing Schrödinger Bridges to grasp transfer is a novel problem formulation; the physics-aware cost design also demonstrates originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple hand morphology combinations, multi-metric evaluation, and complete ablation studies; however, real-robot validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — The mathematical framework is clearly presented and the derivation from SB to the specific application is logically coherent.
- **Value**: ⭐⭐⭐⭐ — Addresses a practical pain point with a generalizable method; provides meaningful reference value for the robot learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](../../ICML2026/robotics/optimal_and_scalable_mapf_via_multi-marginal_optimal_transport_and_schrödinger_b.md)
- [\[ICCV 2025\] DexVLG: Dexterous Vision-Language-Grasp Model at Scale](../../ICCV2025/robotics/dexvlg_dexterous_vision-language-grasp_model_at_scale.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[NeurIPS 2025\] SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)

</div>

<!-- RELATED:END -->
