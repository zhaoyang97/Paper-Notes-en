---
title: >-
  [Paper Note] InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions
description: >-
  [CVPR 2026][Human Understanding][Human-Object Interaction] InterPrior utilizes a three-stage recipe of "large-scale imitation distillation + RL fine-tuning" to distill a full-reference imitation expert into a goal-conditioned variational policy. This policy is then refined using RL into a generalizable generative controller capable of generating full-body human-object interactions from sparse goals (snapshots/trajectories/contacts) and self-correcting after failures.
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Human-Object Interaction"
  - "Physics Simulation"
  - "Generative Controller"
  - "Variational Distillation"
  - "RL Fine-Tuning"
date: 2026-05-08
content_hash: cfa3a79704406b5f
---

# InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_InterPrior_Scaling_Generative_Control_for_Physics-Based_Human-Object_Interactions_CVPR_2026_paper.html)  
**Code**: Project Page https://sirui-xu.github.io/InterPrior  
**Area**: Human Understanding / Physics-Based Character Interaction / Reinforcement Learning  
**Keywords**: Human-Object Interaction, Physics Simulation, Generative Controller, Variational Distillation, RL Fine-Tuning

## TL;DR
InterPrior utilizes a three-stage recipe of "large-scale imitation distillation + RL fine-tuning" to distill a full-reference imitation expert into a goal-conditioned variational policy. This policy is then refined using RL into a generalizable generative controller capable of generating full-body human-object interactions from sparse goals (snapshots/trajectories/contacts) and self-correcting after failures.

## Background & Motivation

**Background**: Human-Object Interaction (HOI) is naturally hierarchical—humans plan sparse intentions at the high level (e.g., "reach for the bottle"), while low-level details such as balance, contact, and limb coordination emerge spontaneously from the low-level motion prior. In physics simulation, enabling a simulated humanoid or robot to perform such full-body loco-manipulation relies on two main approaches: 1) **adversarial generative controllers** (utilizing discriminators for distribution matching combined with RL), which can extend motion coverage beyond demonstrations but suffer from unstable optimization, discriminator mode collapse, and require hand-programmed task rewards, making them difficult to scale; 2) **distilling reference imitation policies** (distilling an imitation expert into a goal-conditioned policy), which can leverage large-scale datasets without requiring task-specific reward engineering.

**Limitations of Prior Work**: Distillation-based approaches are highly fragile in loco-manipulation tasks. The root cause is that **reference coverage falls far short of the configuration space**—as the degrees of freedom of objects increase, contact modes and relative poses explode combinatorially with geometry. Since distillation is essentially "replaying dataset trajectories", the policy easily destabilizes, fails to grasp thin/small objects, and compounding errors lead to catastrophic failures once the goal or human-object state drifts out of the training distribution (such as during transitions between skills or from random initializations). Conversely, training purely with RL leads to unnatural motions due to reward hacking.

**Key Challenge**: Distillation provides a natural and diverse prior but is **not robust** (fails to cover the entire configuration space), whereas pure RL is robust but **unnatural** (prone to reward hacking). Neither approach is sufficient on its own.

**Goal**: To learn a **single policy** that scales across four dimensions: task coverage (supporting multiple sparse goals and their combinations within one policy), skill coverage (handling large-scale HOI data with the same recipe), motion coverage (generating expressive trajectories instead of merely reproducing demonstrations), and dynamics coverage (operating successfully under varying physical properties).

**Key Insight / Core Idea**: The key insight of the authors is that **RL fine-tuning is the key to transforming distillation from "data reconstruction" into "generalizable policies."** Therefore, distillation is used to provide a strong and natural initialization, after which RL serves as a **local optimizer anchored near the pre-trained model**. This simultaneously expands the capability boundaries (e.g., recovering from failures, exploring unseen configurations) and preserves the naturalness learned during pre-training via regularization.

## Method

### Overall Architecture

InterPrior aims to solve the following: given the current human-object state + sparse future goals, sample low-level control commands to enable the humanoid character in a physics simulator to complete natural, physically feasible, and goal-respecting interactions. The proposed method is a **three-stage paradigm**: first, train a full-reference imitation expert that strictly mimics reference motions (Stage 1); second, **distill** it into a variational policy capable of sampling actions from sparse, multi-modal goals (Stage 2); finally, use **RL fine-tuning** to push this variational policy from "simply replaying data" to "generalizing and recovering from failures" (Stage 3). All three stages are modeled as MDPs sharing the same input/output format: the input consists of the observation $x_t$ (human kinematics, object kinematics, signed distance $D_t$ between the human and object, and binary contact indicators $C_t$) along with goal conditions, and the output is the joint position target $a_t$, which is converted to joint torques via PD control to drive the simulation.

The goal condition serves as the unified task interface: the reference frame $y_t$ shares the state space with the observation $x_t$, accompanied by a binary mask $m_t$ indicating which components are provided to the policy. The authors employ two types of goals—short-horizon preview sequences and long-horizon snapshots—and perform **masked residual encoding** for each goal: $\tilde{y}_{t+k} = m_{t+k} \odot \ominus(y_{t+k}, x_t)$ (where $\ominus$ denotes the log-map for rotations and difference for Euclidean quantities). During inference, users can express snapshot goals, trajectory goals, contact goals, or their combinations under a single unified interface by specifying the constrained components, setting their masks to 1, and zeroing the rest.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Large-Scale HOI Data<br/>+ Sparse Goals (Snapshot/Trajectory/Contact)"] --> B["Stage 1: InterMimic+ Full-Reference Imitation Expert<br/>Expanded Reference Range + Reference-Free Grasping Reward"]
    B -->| "Distillation as Teacher" | C["Stage 2: Variational Distillation<br/>Mask-Conditioned Variational Policy + Hyperspherical Latent Space"]
    C -->| "RL as Local Optimizer" | D["Stage 3: RL Post-Training In-Betweening<br/>Single-Frame Goal Completion + Prior Preservation"]
    D --> E["Generalizable Generative Controller<br/>(Goal-Conditioned Policy π)"]
```

### Key Designs

**1. InterMimic+ Full-Reference Imitation Expert: Making the "Strict Copying" Teacher More Robust to Disturbances**

The Stage 1 expert $\pi_E$ follows the large-scale human-object co-tracking paradigm of InterMimic: at each step, it receives the complete, unmasked future reference and uses PPO to maximize $r = r_{\text{track}} - r_{\text{energy}}$ to force the policy to strictly track the reference. However, the authors identified two major issues with the original expert: degradation in performance when grasping **thin/small objects** (as it stiffly replays reference trajectories without utilizing fine-grained hand-object relationships) and catastrophic failures once the rollout drifts from the reference. To address this, InterMimic+ implements two designs. First, it **expands reference coverage**: by initializing rollouts from reference frames with random perturbations, applying sparse impulses (random velocity perturbations) to the pelvis and the object during rollouts to force the policy to drift, and augmenting object shapes and randomizing mass density, center of mass offset, inertia, and friction, the expert is exposed to diverse dynamics **without altering the reference itself**. Furthermore, an early termination penalty $r_{\text{ter}} = -w_{\text{ter}} \cdot c_{\text{ter}}$ is added (triggered when the humanoid falls or the state severely drifts from the reference) to prevent the agent from immediately entering failure states under perturbation. Second, a **reference-free grasping reward** $r_h$ is introduced: since strict reference tracking becomes unreliable under randomization and perturbations, $r_h$ encourages the hands to align with, touch, and wrap around the object based on the **current simulated state**, rather than blindly following the reference trajectory. This acts as a correction term that guides hands back to the real object and offsets the drift caused by perturbations. The complete reward is formulated as $r_t = (r_{\text{track}} - r_{\text{energy}} - r_h) + r_{\text{ter}}$.

**2. Variational Distillation: Distilling the Expert into a Multimodal Policy for "Sampling Actions from Sparse Goals"**

The expert tracks dense references, but the final policy must maintain naturalness and diversity under **sparse cues**, which requires sampling from a latent skill distribution. The authors model the policy $\pi$ using a latent variable $z_t \in \mathbb{R}^{d_z}$, which consists of three components: a prior $p_\psi(z_t \mid x_{t-\delta:t}, \mathcal{G}_t)$ (a 4-layer Transformer that encodes recent history + sparse goals to output Gaussian $\mathcal{N}(\mu_p, \Sigma_p)$), an encoder $q_\phi(z_t \mid x_t, \mathcal{G}_t, y_{t:t+H}, y_{t+L})$ (an MLP used only during training that observes the complete future reference to output $\mathcal{N}(\mu_q, \Sigma_q)$), and a decoder $f_\theta(a_t\mid x_{t-\delta:t}, z_t)$ (which maps the latent variable and observation to actions). During training, $z_t$ is sampled via reparameterization from the residual posterior $\mathcal{N}(\mu_p + \mu_q, \Sigma_q)$ as $z_t = (\mu_p + \mu_q) + \Sigma_q^{1/2}\epsilon$, where $\epsilon$ is kept constant within an episode to ensure temporal consistency; during inference, only the prior is used to sample from $\mathcal{N}(\mu_p, \Sigma_p)$. Two new designs are key: first, **multimodal conditioning** (incorporating contact into the goals to support flexible human-object constraints); second, **latent space regularization and bounding**—the sampled latent variable is projected onto a hypersphere via $z_t \leftarrow z_t / \lVert z_t \rVert$. This suppresses unnatural movements induced by rare outlier latent variables while preserving multimodal variations in direction (the KL regularization is still calculated on the Gaussian prior before projection). Distillation is performed using an online DAgger framework: the student learns from a mixture of expert rollouts and its own rollouts, with the expert's control ratio decaying over training. The total loss is formulated as $L_{\text{total}} = L_{\text{ELBO}} + \lambda_{\text{scale}} L_{\text{scale}} + \lambda_{\text{tc}} L_{\text{tc}}$, where the ELBO term contains the imitation loss (matching expert actions) + goal reconstruction loss (completing masked goal elements, forcing the latent space to learn "intent inference from context") + KL regularization; $L_{\text{scale}}$ constrains the prior mean $\mu_p$ to maintain a unit norm (preventing degradation in conjunction with hyperspherical normalization); and $L_{\text{tc}}$ encourages prior distributions of adjacent timesteps to be similar.

**3. RL Post-Training In-Betweening Fine-Tuning: Pushing the Distilled Policy Beyond Reference Coverage**

The distilled policy $\pi$ can follow goals, but is highly fragile once the state drifts out of the training distribution (most typically during transition phases between skills). The authors' core approach is to model RL fine-tuning as an **in-betweening task**: starting from a randomly sampled initial configuration, the agent is tasked with reaching a **single-frame goal randomly sampled from the dataset**. This sidesteps the issue of "requiring a powerful multi-frame trajectory sampler" (which is extremely difficult to train at the scale of loco-manipulation) while systematically expanding the state distribution encountered by RL via the combination of "single-frame goals + random initializations/offsets". The reward is formulated as $r_t^{\text{PT}} = (r_{\text{energy}} - r_h) + r_{\text{goal}} + r_{\text{ter}}$. Since the goals are arbitrarily specified by random masks, **dense distance rewards are avoided** and a sparse success signal $r_{\text{goal}}$ is used instead: a constant success reward $r_{\text{succ}}$ is given when the masked feature distance between the current state and the goal $\lVert m_{t+L} \odot \ominus(\tilde{y}_{t+L}, x_t)\rVert_1 < \xi$, and 0 otherwise. On top of this, two types of new skills are learned: **In-distribution expansion** (e.g., regrasping, where "self-correction before failure" emerges naturally as the agent attempts to reach goals from diverse starting and perturbed states, requiring no extra supervision); **Out-of-distribution skills** (e.g., getting up from falls, achieved by appending a learnable token to identify the sub-task and adding auxiliary rewards to encourage an upright posture and raising the center of mass). Crucially, for **prior preservation**, instead of freezing the network to prevent forgetting, the authors employ a simple **multi-objective scheduling**: a subset of simulation environments continues optimizing the original distillation loss while the remaining environments perform RL fine-tuning. This anchors the policy to the pre-trained prior without restricting network capacity; gradients are aggregated across multiple GPUs via map-reduce.

### Loss & Training
- Stage 1: PPO, $r_t = (r_{\text{track}} - r_{\text{energy}} - r_h) + r_{\text{ter}}$, 30 Hz simulation in IsaacGym.
- Stage 2: DAgger online distillation, $L_{\text{total}} = L_{\text{ELBO}} + \lambda_{\text{scale}} L_{\text{scale}} + \lambda_{\text{tc}} L_{\text{tc}}$; expert/decoder/encoder are MLPs (1024, 1024, 512), and the prior is a 4-layer Transformer.
- Stage 3: RL (PPO-variant) fine-tuning + multi-objective environment mixture scheduling with parallel distillation; trajectory-conditional inputs are explicitly "protected" from being distorted during fine-tuning by the parallel distillation loss.

## Key Experimental Results

The dataset used is InterAct (including the OMOMO subset, recovered via teacher rollouts), and generalization evaluations are transferred to subsets of BEHAVE / HODome (excluding soft-body-dominated interactions). Tasks are categorized into two types: full-reference tracking and sparse goal following; the latter covers snapshots, trajectories, contact, and their combinations, along with two stress tests: long-horizon multi-goal chains (Chain) and random initialization (Rand Init).

### Main Results (Goal-Conditioned Tasks, Excerpt from Table 1, Succ↑ / Eh↓ / Eo↓ / Fail↓)

| Configuration | Snapshot Succ | Contact Succ | Chain Succ | Rand Init Succ |
|---|---|---|---|---|
| MaskedMimic (InterMimic Expert) | 64.2 | 52.2 | 29.1 | 31.7 |
| InterPrior (Ours) | **90.0** | **90.7** | **68.8** | **88.6** |

The full InterPrior improves snapshot success rate from 64.2 to 90.0 and contact success rate from 52.2 to 90.7. The most significant improvements are observed in the two stress tests: multi-goal chain (29.1→68.8) and random initialization (31.7→88.6). This supports the core argument—distilled policies fit the "demonstration-guided state distribution," resulting in drift failures once long rollouts enter poorly-covered intermediate states. Directly training with RL to "reach sparse targets from diverse initializations" significantly improves interpolation between target sequences and out-of-distribution recovery.

### Full-Reference Tracking and Reusable Priors (Table 2, SR↑)

| Method | OMOMO SR | BEHAVE SR | HODome SR |
|---|---|---|---|
| InterMimic | 63.9 | 10.7 | 27.8 |
| InterMimic + Fine-Tuning | / | 38.9 | 55.5 |
| InterPrior | **83.2** | 27.4 | 40.1 |
| InterPrior + Fine-Tuning | / | **52.0** | **72.4** |

On OMOMO with thin objects and initialization perturbations, InterPrior improves the success rate from 63.9 to 83.2 (at the cost of a slight increase in pelvic position error $E_h$ from 7.1 to 8.9, as it **actively and slightly drifts from reference** to realign contact, trading "strict tracking" for "successful interaction"). When serving as a reusable prior adapted to new objects or interactions, InterPrior proves significantly more stable than the full-reference InterMimic, whether fine-tuned or not (reaching 52.0 after BEHAVE fine-tuning from 38.9).

### Ablation Study (Cumulative Ablation of Table 1, Succ of Snapshot/Contact/Chain/RandInit)

| Cumulative Configuration | Snapshot | Contact | Chain | Rand Init |
|---|---|---|---|---|
| InterMimic+ Expert | 71.4 | 69.3 | 33.9 | 30.1 |
| + Latent Space Shaping Loss | 74.9 | 71.9 | 40.0 | 30.9 |
| + Bounded Latent Space & Observations | 89.1 | 88.5 | 45.1 | 41.1 |
| + RL Fine-Tuning (=Ours) | 90.0 | 90.7 | 68.8 | 88.6 |

### Key Findings
- **The bounded latent space is the biggest contributor to single-step accuracy**: Adding the hyperspherical bounding improves snapshot success from 74.9 to 89.1 and contact success from 71.9 to 88.5, indicating that restricting latent variables to a proper manifold is crucial for suppressing drift in contact-heavy tasks.
- **RL fine-tuning is the main driver behind the stress test performance**: It has minimal impact on standard task accuracy (snapshot only moves from 89.1 to 90.0) but dramatically improves Chain (45.1→68.8) and Rand Init (41.1→88.6), proving its primary contribution lies in "robustness and out-of-distribution recovery" rather than "more precise fitting".
- **Trajectory following is not sacrificed**: Even though in-betweening is only fine-tuned on single-frame snapshot goals, the trajectory success rate actually improves from 93.6 to 94.6. This is because the trajectory-conditional inputs are explicitly protected by the parallel distillation loss, and when they drift, they are redefined as snapshot goals, thereby indirectly benefiting from the snapshot fine-tuning.
- **Failure modes**: Extremely thin or small unseen objects remain difficult to grasp; large alignment errors introduced by canonicalization in multi-goal chains cause the policy to "prefer maintaining balance rather than forcing precise goal achievement."

## Highlights & Insights
- **The recipe of "distillation as initialization, RL as a local optimizer" is highly transferable**: It decouples the contradiction between "naturalness but lack of robustness" and "robustness but unnaturalness." By anchoring RL with pre-training priors to prevent reward hacking, this paradigm can be directly transferred to other "imitation-first, reinforcement-later" embodied control tasks.
- **Unified sparse goals via masked residual encoding**: Expressing snapshots, trajectories, contacts, and their combinations solely through "which components to fill + which masks to enable" allows a single interface to handle a wide range of task formulations, serving as the core mechanism for scalable task coverage.
- **In-betweening bypasses multi-frame trajectory samplers**: Modeling fine-tuning as "reaching a random single-frame goal from random initializations" systematically broadens the state distribution while avoiding the hard problem of training a robust trajectory sampler at the scale of loco-manipulation. Behaviors like regrasping and self-correction emerge spontaneously without explicit supervision.
- **Hyperspherical latent space with unit norm constraint**: Constraining the latent variables to limit unnatural motions induced by rare outliers while preserving directional multimodality is a subtle yet crucial design for achieving both robustness and diversity.

## Limitations & Future Work
- The authors acknowledge remaining failure modes: extremely thin or narrow unseen objects, and alignment errors introduced by canonicalization in multi-goal chains.
- Evaluations excluded interactions dominated by soft bodies (e.g., backpack shoulder straps); the method assumes all objects are rigid, and its applicability to deformable objects remains unverified.
- On full-reference tracking, InterPrior's position error is slightly higher than that of InterMimic (as it actively departs from references to realign contact), which acts as a disadvantage in scenarios requiring strict adherence to references.
- The G1 robot lacks dexterous hands, which ruled out single-handed grasping of thin geometric objects; real-robot capabilities rely on deployment from another work [17], with the main body of our work remaining in simulation. ⚠️ Real robot details are subject to the original text.
- Future Work: Integrating perception, language-conditioned goals, and richer affordance, with steps toward sim-to-real assistive manipulation and teleoperation.

## Related Work & Insights
- **vs Adversarial Generative Controllers (e.g., AMP / ASE series)**: They employ discriminators for distribution matching combined with RL. While they can extend motion coverage, they suffer from unstable optimization, are prone to discriminator mode collapse, require hand-programmed task rewards, and are hard to scale. InterPrior takes a distillation + RL approach, eliminating the need for task-specific discriminators and digesting large-scale data.
- **vs MaskedMimic**: MaskedMimic also utilizes masked goal-conditioning for physics control but directly distills the InterMimic expert, lacking the bounded latent space and RL fine-tuning, which makes it significantly weaker in stress tests (snapshot success 64.2 vs 90.0, Rand Init 31.7 vs 88.6). InterPrior builds on this by adding the InterMimic+ expert, latent space shaping/bounding, and RL in-betweening.
- **vs InterMimic (The predecessor of our expert)**: InterMimic operates on full-reference imitation, strictly replaying dense references, which yields high accuracy but fragility (failing to grasp thin objects and crashing immediately upon departing from reference). InterPrior distills it into a goal-conditioned generative policy and fine-tunes with RL, obtaining generalization and failure recovery at the cost of slightly lower strict tracking accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ Solid combination of the "distillation initialization + RL local optimizer" recipe, in-betweening fine-tuning, and unified masked goals.
- Experimental Thoroughness: ⭐⭐⭐⭐ Both quantitative and qualitative, covering multi-task, stress-tests, cumulative ablations, cross-dataset generalization, and sim-to-sim.
- Writing Quality: ⭐⭐⭐⭐ The three-stage logic is clear, and the core claim (RL is key to generalization) is consistently threaded throughout; individual math notations in the CVF paper are slightly cluttered.
- Value: ⭐⭐⭐⭐ Provides a scalable and reusable prior recipe for humanoid loco-manipulation, offering practical reference value for embodied interaction generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Decoupled Generative Modeling for Human-Object Interaction Synthesis](decoupled_generative_modeling_for_human-object_interaction_synthesis.md)
- [\[CVPR 2026\] InterPhys: Physics-aware Human Motion Synthesis in a Dynamic Scene](interphys_physics-aware_human_motion_synthesis_in_a_dynamic_scene.md)
- [\[CVPR 2026\] Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors](ground_reaction_inertial_poser_physics-based_human_motion_capture_from_sparse_im.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](../../ICCV2025/human_understanding/humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[CVPR 2026\] PAMotion: Physics-Aware Motion Generation for Full-Body Interaction with Multiple Objects](pamotion_physics-aware_motion_generation_for_full-body_interaction_with_multiple.md)

</div>

<!-- RELATED:END -->
