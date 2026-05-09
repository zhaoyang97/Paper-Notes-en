---
title: >-
  [Paper Note] PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization
description: >-
  [CVPR 2026][LLM Alignment][DPO] This work introduces DPO preference optimization into the post-training stage of diffusion-based motion generation models. A physics simulation controller automatically constructs preference data pairs, enabling generated human motions to satisfy both text/spatial control instructions and physical constraints. The approach successfully transfers zero-shot to a real Unitree G1 robot.
tags:
  - CVPR 2026
  - LLM Alignment
  - DPO
  - human motion generation
  - physics simulation
  - diffusion models
  - humanoid robots
date: 2026-05-08
content_hash: 167146d8c7326d1d
---

# PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization

**Conference**: CVPR 2026
**arXiv**: [2603.13228](https://arxiv.org/abs/2603.13228)
**Code**: Available ([Project Page](https://mael-zys.github.io/PhysMoDPO/))
**Area**: Alignment RLHF / Human Motion
**Keywords**: DPO, human motion generation, physics simulation, diffusion models, humanoid robots

## TL;DR

This work introduces DPO preference optimization into the post-training stage of diffusion-based motion generation models. A physics simulation controller automatically constructs preference data pairs, enabling generated human motions to satisfy both text/spatial control instructions and physical constraints. The approach successfully transfers zero-shot to a real Unitree G1 robot.

## Background & Motivation

**Background**: Diffusion-based text-driven human motion generation has achieved significant progress (MDM, MoMask, MotionStreamer, etc.), producing motion sequences conditioned on text descriptions or spatial constraints.

**Limitations of Prior Work**: These models are trained and evaluated in kinematic space, but serious deficiencies are exposed when deployed to physical simulation or real robots—foot sliding, unstable center of mass, dynamically implausible motion—making generated motions unexecutable.

**Key Challenge**: When motions are mapped to executable trajectories via a Whole-Body Controller (WBC), the controller substantially modifies implausible motions to satisfy physical constraints, causing large deviations between the executed trajectory and the generated motion. In other words, "good kinematic metrics ≠ good deployment performance."

**Goal**: Enable the diffusion motion generator to directly output physically feasible motions that are faithful to text/spatial conditions.

**Limitations of Prior Work**: (a) Post-processing/projection methods (PhysPT, PhysDiff) alter the output distribution and degrade task performance; (b) RL fine-tuning with hand-crafted rewards (foot sliding, floating penalties) (ReinDiffuse, HY-Motion) struggles to cover complex dynamics.

**Core Idea**: Integrate a pretrained physics tracking controller (DeepMimic) directly into the training pipeline—execute candidate motions through it, compute physical and task rewards on the resulting trajectories, automatically construct DPO preference pairs to fine-tune the generator, forming an iterative generate–finetune optimization loop.

## Method

### Overall Architecture

PhysMoDPO operates as a three-step iterative loop:

1. **Generation Stage**: For each condition $C$ (text or text + spatial control), sample $K$ candidate motions $X_k$ from the diffusion generator.
2. **Simulation Stage**: Feed each candidate into the fixed physics tracking policy $\mathcal{T}$ (DeepMimic), obtaining simulated trajectories $X_k' = \mathcal{T}(X_k)$.
3. **Preference Optimization Stage**: Compute rewards on simulated trajectories, construct preference pairs $(X_{\text{win}}, X_{\text{lose}})$, and fine-tune the generator with a joint DPO + SFT loss.

The entire process can be iterated over multiple rounds: after each round of generator updates, new candidates are sampled and new preference data is constructed.

### Key Designs

**Introduction of the Physics Operator $\mathcal{T}$**:

- **Function**: Defines a mapping from kinematic space to physically feasible space, $\mathcal{T}: \mathcal{X}_{\text{kin}} \to \mathcal{X}_{\text{phys}}$, implemented via the DeepMimic tracking policy.
- **Mechanism**: Evaluation is no longer performed in kinematic space but in the deployment space (after WBC execution).
- **Design Motivation**: Directly measures the physical deployability of motions; physical consistency is quantified by tracking deviation $\Delta(X) = \|X' - X\|_2^2$.

**Pareto Dominance Rule for Preference Pair Construction**:

- **Function**: Requires $X_{\text{win}}$ to outperform $X_{\text{lose}}$ on all reward dimensions.
- **Mechanism**: Among the $K$ simulated candidates, a valid preference pair is formed only when one sample dominates another across all rewards—tracking, sliding, text alignment (and spatial control).
- **Design Motivation**: Avoids sensitive reward weighting hyperparameters and maintains consistency across different objectives.

**Iterative Training**:

- **Function**: After each DPO update round, regenerate candidates with the improved model and rebuild preference pairs.
- **Mechanism**: As the model improves, old preference data no longer reflects current failure modes and must be refreshed.
- **Design Motivation**: Analogous to online DPO, continuously targeting the model's current weaknesses.

### Loss & Training

Total training objective:

$$\mathcal{L} = \mathcal{L}_{\text{DPO}}(X_{\text{win}}, X_{\text{lose}}) + \lambda_{\text{SFT}} \mathcal{L}_{\text{SFT}}(X_{\text{win}})$$

Four reward functions:

| Reward | Formula / Meaning | Role |
|--------|------------------|------|
| $\mathcal{R}_{\text{track}}$ | $-\|X' - X\|_2^2$, negative tracking deviation | Encourages minimal change after physical execution |
| $\mathcal{R}_{\text{slide}}$ | Penalty on horizontal foot velocity during ground contact | Reduces foot sliding artifacts |
| $\mathcal{R}_{\text{M2T}}$ | Cosine similarity between text and motion in TMR encoder | Maintains text semantic consistency |
| $\mathcal{R}_{\text{control}}$ | Weighted MSE at control joints | Satisfies spatial control constraints (spatial control tasks only) |

Training details: 12 samples per training prompt are generated for MotionStreamer; only the diffusion head is updated; SMPL joint rotation representation replaces the HumanML3D representation.

## Key Experimental Results

### Main Results

**Table 1: Text-Driven Motion Generation (SMPL Simulation, HumanML3D Dataset)**

| Method | MM-Dist ↓ | R@3 ↑ | FID ↓ | Jerk ↓ |
|--------|-----------|-------|-------|--------|
| MaskedMimic | 19.73 | 0.6305 | 73.79 | 66.08 |
| MotionStreamer | 17.17 | 0.8310 | 49.14 | 46.75 |
| SFT | 17.23 | 0.8355 | 49.22 | 48.30 |
| **PhysMoDPO** | **16.95** | **0.8517** | **48.29** | **43.60** |

**Table 2: Spatial-Text Control (Cross Setting, SMPL Simulation, HumanML3D)**

| Method | Err. ↓ | MM-Dist ↓ | FID ↓ | Jerk ↓ |
|--------|--------|-----------|-------|--------|
| OmniControl (Cross) | 0.0938 | 3.086 | 0.75 | 64.07 |
| SFT | 0.0972 | 3.075 | 0.68 | 61.22 |
| **PhysMoDPO** | **0.0923** | 3.099 | **0.66** | **58.02** |

**Zero-Shot Transfer to G1 Robot** (text-driven, Table 4): PhysMoDPO achieves the best results on M2T (0.7919), R@1 (0.4707), R@3 (0.7640), and FID (0.3029), with Jerk substantially reduced from MotionStreamer's 95.08 to 90.14.

### Ablation Study

**Ablation on Number of Iterations** (OmniControl, HumanML3D):

| Rounds | Err. ↓ | FID ↓ | Jerk ↓ |
|--------|--------|-------|--------|
| 1 | 0.1421 | 1.17 | 72.13 |
| 2 | 0.1324 | 0.97 | 63.55 |
| 3 | **0.1298** | **0.93** | **62.31** |

**Ablation on Reward Components**:

| Reward Combination | Err. ↓ | FID ↓ | Jerk ↓ |
|-------------------|--------|-------|--------|
| Tracking only | 0.1467 | 1.61 | 74.76 |
| + Control | 0.1447 | 1.45 | 74.47 |
| + Sliding | 0.1422 | 1.21 | **68.43** |
| + M2T | **0.1421** | **1.17** | 72.13 |

### Key Findings

- MaskedMimic, despite being an end-to-end physics policy, exhibits poor text-following capability (R@3 of only 0.6305 vs. PhysMoDPO's 0.8517).
- Three rounds of iterative training improve FID by 20% over one round (1.17→0.93), demonstrating the critical importance of refreshing preference data.
- The sliding reward yields the greatest improvement in Jerk (74.76→68.43), while the M2T reward yields the greatest improvement in FID (1.21→1.17).
- PhysMoDPO also substantially outperforms baselines on the OOD dataset OMOMO, demonstrating generalization capability.
- In a user study, PhysMoDPO consistently outperforms OmniControl and MaskedMimic on text adherence, motion fluency, and stability.

## Highlights & Insights

1. **Paradigm Shift in Evaluation**: Moving evaluation from kinematic space to the physical deployment space (post-WBC execution) is a critical perspective shift that exposes a blind spot in prior work that relied solely on kinematic metrics.
2. **Clever Adaptation of DPO for Continuous Motion**: Computing rewards on simulated trajectories while performing DPO training on kinematic samples elegantly circumvents the non-differentiability of the physics simulator.
3. **Pareto Dominance for Preference Pair Construction**: Rather than a weighted sum, requiring dominance across all dimensions to form a preference pair avoids the burden of tuning reward weights while ensuring signal consistency.
4. **Real-Robot Deployment Validation**: Beyond simulation experiments, zero-shot deployment on the Unitree G1 with recorded video is highly convincing.

## Limitations & Future Work

1. **Limited Scenarios**: Validation is currently restricted to flat ground; complex scenarios such as stairs, slopes, and uneven terrain remain unaddressed.
2. **Dependence on a Fixed WBC**: The quality of preference pairs is bounded by the capability of the tracking policy; systematic biases in the WBC propagate into the training signal.
3. **Iterative Training Cost**: Each round requires $K$ sampling runs plus physics simulation for a large number of prompts, incurring substantial computational overhead.
4. **SMPL Representation Constraint**: Converting data from HumanML3D format to SMPL joint rotations and retraining raises the barrier for switching models.
5. **Lack of Direct Comparison with Morph**: As a closely related method that refines motions with a physics model before fine-tuning, Morph is not directly compared in the experiments.

## Related Work & Insights

- **DPO for Diffusion**: The framework draws on Diffusion-DPO (images) and VideoDPO (video), extending it for the first time to human motion as a structured sequence generation task.
- **ReinDiffuse / HY-Motion**: These methods fine-tune motion generators with PPO/GRPO and hand-crafted rewards; PhysMoDPO replaces these with simulator-based automatic rewards for broader coverage.
- **BeyondMimic / HOVER**: These works improve WBC robustness; PhysMoDPO takes the complementary approach of keeping the WBC fixed and improving the generator instead.
- Core Insight: **Using a downstream executor as an evaluation signal to provide feedback to an upstream generator** is a broadly applicable pattern in other generation-deployment pipelines (e.g., code generation → compilation/execution, text → TTS).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Transferring the DPO alignment framework from LLMs/images to physics-grounded motion generation is a compelling cross-domain idea; the Pareto dominance construction for preference pairs is also a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two tasks × three robots × multiple datasets × OOD evaluation × ablations × user study × real-robot deployment; exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematically rigorous, logically clear, with an intuitive related work comparison table; slightly verbose but does not impede readability.
- **Value**: ⭐⭐⭐⭐ — Provides a scalable technical pathway for bridging the gap between motion generation and robot deployment; real-robot validation adds practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering](glyphprinter_region-grouped_direct_preference_optimization_for_glyph-accurate_vi.md)
- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/llm_alignment/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [\[CVPR 2026\] $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)

</div>

<!-- RELATED:END -->
