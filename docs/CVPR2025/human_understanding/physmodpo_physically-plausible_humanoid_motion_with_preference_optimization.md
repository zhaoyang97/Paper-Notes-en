---
title: >-
  [Paper Note] PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization
description: >-
  [CVPR 2025][Human Understanding][Motion Generation] This paper proposes PhysMoDPO, which applies Direct Preference Optimization (DPO) to text-driven human motion generation. By integrating a Whole-Body Controller (WBC) into the training pipeline to calculate physics-based rewards and construct preference data, the generated motions satisfy both physical constraints and text instructions. Zero-shot deployment is achieved on the Unitree G1 robot.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Motion Generation"
  - "DPO"
  - "Physical Plausibility"
  - "Humanoid Robot"
  - "Whole-Body Controller"
date: 2026-05-08
content_hash: dd55adf4a47b3468
---

# PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization

**Conference**: CVPR 2025  
**arXiv**: [2603.13228](https://arxiv.org/abs/2603.13228)  
**Code**: [https://mael-zys.github.io/PhysMoDPO/](https://mael-zys.github.io/PhysMoDPO/)  
**Area**: LLM Alignment / Human Motion Generation  
**Keywords**: Motion Generation, DPO, Physical Plausibility, Humanoid Robot, Whole-Body Controller

## TL;DR
This paper proposes PhysMoDPO, which applies Direct Preference Optimization (DPO) to text-driven human motion generation. By integrating a Whole-Body Controller (WBC) into the training pipeline to calculate physics-based rewards and construct preference data, the generated motions satisfy both physical constraints and text instructions. Zero-shot deployment is achieved on the Unitree G1 robot.

## Background & Motivation
**Background**: Text-driven human motion generation is mainly driven by diffusion models, but the generated motions often exhibit physical implausibility (e.g., foot sliding, center-of-mass imbalance) when deployed in physical simulations.

**Limitations of Prior Work**: (1) Models trained in kinematic space cannot guarantee dynamic feasibility. (2) Manually designed physical constraints (such as foot-sliding penalties) struggle to cover complex physical dynamics. (3) When executed via a WBC, the controller may severely modify the motion to satisfy constraints, leading to a deviation from the original intent.

**Key Challenge**: The mismatch between generation quality (kinematic metrics) and physical feasibility (dynamic constraints).

**Goal**: To enable the diffusion motion generator to produce motions that are both physically compliant and faithful to text instructions.

**Key Insight**: Integrate a WBC into the training pipeline and directly use the physical simulation results as preference signals to fine-tune the generator.

**Core Idea**: Utilizing WBC simulation results as the preference data source, combining physical and task rewards, and fine-tuning the motion generator via DPO.

## Method

### Overall Architecture
Given a conditioning signal $C$ (text $C_t$ or text + spatial control $(C_t, C_s)$), the diffusion generator $G_\theta$ samples $K$ candidate motions from noise, $X_k = G_\theta(\epsilon_k, C)$. Each candidate is projected onto the physically feasible space via a fixed WBC (DeepMimic tracking policy) $\mathcal{T}$: $X'_k = \mathcal{T}(X_k)$. Physics rewards and task rewards are calculated on $X'_k$ to choose the best and worst candidates to construct preference pairs $(X_{win}, X_{lose})$. Finally, $G_\theta$ is fine-tuned using the DPO loss. This entire process can be iterative: sampling is repeated to construct new preference pairs after updating the generator.

### Key Designs

1. **Formalization of Kinematic Space vs. Physically Feasible Space (Section 3.1)**:

    - **Function**: Clearly separate the kinematic space $\mathcal{X}_{kin}$ and the physically feasible space $\mathcal{X}_{phys} \subset \mathcal{X}_{kin}$, defining the tracking distortion as $\Delta(X) = \|X' - X\|_2^2$.
    - **Mechanism**: A kinematically reasonable motion ($X \in \mathcal{X}_{kin}$) is not necessarily physically feasible ($X \in \mathcal{X}_{phys}$). A small $\Delta(X)$ indicates that the motion is close to physically feasible, requiring only minor adjustments from the WBC; a large $\Delta(X)$ indicates the motion is physically implausible, requiring major modifications from the WBC.
    - **Design Motivation**: **Shifting the evaluation target from the kinematic space to the deployment space**—evaluating the WBC-executed $X' = \mathcal{T}(X)$ rather than the direct output of the generator $X$, which directly resolves the core challenge of "good kinematic metrics but poor deployment performance".

2. **Physics-Based Post-Training Pipeline (Section 3.2)**:

    - **Function**: For each conditioning signal $C$, sample $K$ candidates, score them after WBC simulation, and construct preference pairs for DPO.
    - **Mechanism**: The training loss is $\mathcal{L} = \mathcal{L}_{DPO}(X_{win}, X_{lose}) + \lambda_{SFT}\mathcal{L}_{SFT}(X_{win})$, where $\lambda_{SFT}$ balances preference learning and the retention of generation quality. Note: **Rewards are computed on $X'$, but preference pairs are defined on $X$** (respecting the DPO requirement that data must come from the model's own sampling).
    - **Design Motivation**: Since $\mathcal{T}$ (the physical simulator) is non-differentiable, gradients cannot be propagated directly. Using DPO's preference learning bypasses this non-differentiability issue. The SFT term is added to prevent mode drift or degradation in generation quality caused by pure preference optimization.

3. **Four Types of Reward Signals (Section 3.3)**:

    - **Tracking Reward** $\mathcal{R}_{track}(X', X) = -\|X' - X\|_2^2$: Directly minimizes the magnitude of WBC corrections.
    - **Foot-sliding Reward** $\mathcal{R}_{slide}(X') = -\frac{1}{N}\sum_i \mathbf{1}[h_{feet} < h_0]\mathbf{1}[v_{feet}^{xy} > v_0]$: Penalizes cases where feet touch the ground but slide horizontally.
    - **Text Alignment Reward** $\mathcal{R}_{M2T}(X', C_t) = \cos(TMR_{text}(C_t), TMR_{mot}(X'))$: Computes text-motion cosine similarity using the TMR encoder (importantly, calculated on $X'$!).
    - **Spatial Control Reward** $\mathcal{R}_{control}(X', C_s) = -\frac{\|W \odot (X' - C_s)\|_2^2}{\|W\|_1}$: Measures spatial constraint satisfaction (enabled only when $C_s$ is present).
    - Preference pair construction uses **dominance rules**: $X_k$ is preferred over $X_l$ if and only if **all** reward terms are better, avoiding the introduction of sensitive reward weights.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{DPO}(X_{win}, X_{lose}) + \lambda_{SFT}\mathcal{L}_{SFT}(X_{win})$
- $\mathcal{L}_{SFT}$ uses a Two-Forward objective (only on win samples)
- Iterative DPO: Resample using the new model after each round of updates to refresh the preference data
- Only the diffusion head is updated, optimized using AdamW
- 12 candidates are generated per training prompt ($K=12$)

## Key Experimental Results

### Main Results: Text-to-Motion (HumanML3D, SMPL Post-Simulation Evaluation)

| Method | MM-Dist↓ | R@1↑ | R@3↑ | FID↓ | Jerk↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| Real after sim | 16.02 | 0.668 | 0.895 | 34.07 | 35.87 |
| MaskedMimic | 19.73 | 0.413 | 0.631 | 73.79 | 66.08 |
| MotionStreamer | 17.17 | 0.583 | 0.831 | 49.14 | 46.75 |
| SFT (only win) | 17.23 | 0.578 | 0.836 | 49.22 | 48.30 |
| **PhysMoDPO** | **16.95** | **0.585** | **0.852** | **48.29** | **43.60** |

### Ablation Study: Zero-Shot Deployment on G1 Robot

| Method | M2T↑ | R@3↑ | FID↓ | Jerk↓ |
|------|:---:|:---:|:---:|:---:|
| Real after sim | 0.828 | 0.848 | 0.120 | 87.76 |
| MaskedMimic | 0.716 | 0.576 | 0.367 | **83.58** |
| MotionStreamer | 0.790 | 0.756 | 0.303 | 95.08 |
| **PhysMoDPO** | **0.792** | **0.764** | **0.303** | 90.14 |

### Key Findings
- **SFT baseline is largely ineffective**: Performing SFT solely on win samples yields negligible improvements or even degradation in some metrics, demonstrating that DPO's contrastive learning signals are far more effective than pure positive-sample SFT.
- **MaskedMimic is physically safe but has poor text consistency** (R@1 of only 0.413 vs. PhysMoDPO's 0.585), illustrating that end-to-end physical control policies cannot replace the paradigm of diffusion generators combined with post-training.
- **Successful zero-shot transfer to G1 robot**: The model fine-tuned by PhysMoDPO can be directly deployed to the Unitree G1 (without additional refinement), reducing Jerk from 95.08 to 90.14, and outperforming all baselines in text consistency.
- **OOD Generalization (OMOMO)**: On unseen human-object interaction data, PhysMoDPO still significantly reduces Err, FID, and Jerk, indicating that physical preference learning possesses cross-distribution transferability.
- **Dominance rule for preference construction outclasses weighted sum**: It successfully avoids the need for tedious reward weight tuning.

## Highlights & Insights
- **Transferring DPO from LLM alignment to motion generation** is an ingenious cross-domain application. The key insight is that the physical simulator acts like a "human annotator"—automatically assessing the physical plausibility of motions and bypassing the limitations of manually designed reward functions.
- **Using WBC as an automatic preference annotator** is highly elegant—eliminating the need for human preference labeling, as the simulator itself serves as the perfect judge for physical plausibility. This makes preference data collection practically zero-cost (excluding computing overhead).
- **Direct resolution of the evaluation mismatch**—shifting the evaluation target from $\mathcal{X}_{kin}$ to $\mathcal{X}_{phys}$, and computing all metrics (including text alignment) *after* WBC execution, ensuring that "what is optimized is exactly what is deployed".
- **Replacing weighted sum with dominance rules (Pareto dominance)** to aggregate multidimensional rewards—avoiding sensitive hyperparameter tuning, maintaining simplicity and robustness. The limitation is the potential scarcity of valid preference pairs, though $K=12$ candidates proved sufficient in experiments.
- **Iterative DPO for refreshing preference data**—resampling after each round of updates avoids the issue in offline DPO where preference data mismatches the current model distribution.

## Limitations & Future Work
- High computational cost of WBC simulation—sampling 12 candidates per training prompt and simulating each of them separately multiplies the training time.
- Zero-shot deployment has only been validated on the Unitree G1; whether it generalizes to other robot morphologies (bipedal, quadrupedal, or humanoids of different scales) remains to be verified.
- The generative diversity of the diffusion model might shrink due to DPO fine-tuning (risk of mode collapse in DPO).
- Although the dominance rule for preference construction avoids reward weight tuning, it can lead to an insufficient number of available preference pairs when no candidate among the $K$ samples dominates another across all rewards.

## Related Work & Insights
- **vs ReinDiffuse/HY-Motion**: These methods employ RL with hand-crafted floating/sliding rewards + PPO/GRPO, whereas PhysMoDPO uses the WBC simulator as the reward source + DPO. The core difference lies in the source of reward signals—manual heuristics struggle to cover complex physical dynamics like center-of-mass anomalies, while WBC simulation provides comprehensive coverage.
- **vs PhysPT/Zhang et al.**: These are test-time optimization/projection methods, whereas PhysMoDPO solves the issue during training, incurring zero overhead during inference. Furthermore, projection methods might shift the output distribution and impair task performance.
- **vs Morph**: Morph refines datasets using physical models and then trains the generator, but noisy motions can adversely impact the physical model. In contrast, the WBC in PhysMoDPO is read-only (frozen parameters) and remains unaffected by the quality of the generator.
- Shared philosophy with SceneAssistant: both utilize **non-differentiable external systems** (rendering engines/physical simulators) as feedback signal sources to improve the generator.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of WBC and DPO is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly thorough, including simulation, real hardware, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation and detailed methodology.
- **Value**: ⭐⭐⭐⭐⭐ Possesses significant value for both motion generation and robot control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] KinemaDiff: Towards Diffusion for Coherent and Physically Plausible Human Motion Prediction](../../ICLR2026/human_understanding/kinemadiff_towards_diffusion_for_coherent_and_physically_plausible_human_motion_.md)
- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](../../NeurIPS2025/human_understanding/vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](motionrefit_motion_editing.md)
- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2025\] StickMotion: Generating 3D Human Motions by Drawing a Stickman](stickmotion_generating_3d_human_motions_by_drawing_a_stickman.md)

</div>

<!-- RELATED:END -->
