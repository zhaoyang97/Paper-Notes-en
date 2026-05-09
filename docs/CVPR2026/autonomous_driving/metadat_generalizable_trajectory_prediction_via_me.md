---
title: >-
  [Paper Note] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating
description: >-
  [CVPR 2026][Autonomous Driving][Test-Time Training] This paper proposes MetaDAT, a framework that obtains an initialization amenable to online adaptation via meta pre-training, and further employs dynamic learning rate optimization (DLO) and hard-sample-driven updates (HSD) at test time to achieve trajectory prediction adaptation under cross-dataset distribution shifts. MetaDAT consistently outperforms existing test-time training (TTT) methods across diverse cross-domain configurations on nuScenes, Lyft, and Waymo.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Test-Time Training
  - Meta-Learning
  - Trajectory Prediction
  - Distribution Shift
  - Online Adaptation
date: 2026-05-08
content_hash: 339d374e6184ed21
---

# MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating

**Conference**: CVPR 2026
**arXiv**: [2603.09419](https://arxiv.org/abs/2603.09419)
**Code**: None (not released)
**Area**: Autonomous Driving / Trajectory Prediction
**Keywords**: [Test-Time Training, Meta-Learning, Trajectory Prediction, Distribution Shift, Online Adaptation]

## TL;DR
This paper proposes MetaDAT, a framework that obtains an initialization amenable to online adaptation via meta pre-training, and further employs dynamic learning rate optimization (DLO) and hard-sample-driven updates (HSD) at test time to achieve trajectory prediction adaptation under cross-dataset distribution shifts. MetaDAT consistently outperforms existing test-time training (TTT) methods across diverse cross-domain configurations on nuScenes, Lyft, and Waymo.

## Background & Motivation
Trajectory predictors in autonomous driving are typically trained offline on pre-collected datasets, yet face severe distribution shifts when deployed in new environments—differences in road structure, interaction patterns, and driving styles can lead to significant performance degradation. Test-time training (TTT) is a promising solution: by exploiting the "self-labeling" property of trajectory prediction (past observations serve as ground truth for prior predictions), the model can be updated online at inference time.

However, existing methods suffer from two fundamental limitations: (1) **offline–online objective misalignment**—conventional offline pre-training optimizes only in-distribution prediction accuracy without considering online adaptability, causing learned representations to degrade rapidly during online updates; (2) **fixed online update strategies**—learning rates and update frequencies are predetermined and cannot adapt to the characteristics of unseen test data (e.g., degree of distribution shift, difficulty distribution of samples).

## Core Problem
How can a trajectory prediction model be both "ready to learn online" from the pre-training stage and capable of dynamically adjusting its learning strategy at test time based on actual data characteristics? In short, the paper addresses two simultaneous challenges: *what initialization to learn* and *how to update adaptively*.

## Method

### Overall Architecture
MetaDAT consists of two stages: (1) **Meta Pre-training**: TTT tasks are simulated on the source dataset, and a MAML-style bi-level optimization is used to obtain an initialization $\theta^*$ suited for online adaptation; (2) **Data-Adaptive Test-Time Updating**: online learning is performed on the target domain, with DLO and HSD adaptively adjusting the update strategy.

The prediction network adopts ForecastMAE as the backbone (consistent with prior work T4P for fair comparison), comprising an embedding layer, encoder, decoder, and MAE reconstruction branch. The training objective is a joint MAE loss combining regression and reconstruction losses.

### Key Designs
1. **Meta Pre-training (MP)**: Individual driving scenarios in the source dataset are treated as sub-domains and organized in temporal order to simulate TTT tasks. MAML-style bi-level optimization is applied: the inner loop performs $K$ online update steps on each simulated TTT task to obtain $\theta'$; the outer loop evaluates post-adaptation prediction performance and optimizes the initial parameters $\theta$. A first-order approximation is used to reduce computational cost, and the offline pre-trained parameters $\theta_\text{off}$ serve as the meta-learning starting point to accelerate convergence and improve generalization.
2. **Dynamic Learning Rate Optimization (DLO)**: Per-layer learning rates are dynamically adjusted using online partial derivatives $\partial \mathcal{L}/\partial \alpha$. The core idea is: when the gradient directions of two consecutive steps are consistent, the learning rate should increase (to accelerate convergence); when they are opposite, it should decrease (to avoid oscillation). To stabilize training, a window of length $\tau_\alpha$ is used to average gradients in practice. Each network layer has an independent learning rate that adapts from a shared initial value.
3. **Hard-Sample-Driven Updates (HSD)**: Autonomous driving data exhibits long-tail distributions, where a minority of difficult scenarios (e.g., dense interactions, complex intersections) are most susceptible to distribution shift and most informative. Hard samples are identified by comparing the prediction error $e$ against the running mean $m$ and standard deviation $\sigma$ (i.e., $e > m + k\sigma$), and additional update steps are applied to these samples. Since only a small number of samples are selected, overall efficiency is not compromised.

### Loss & Training
- Both pre-training and TTT use a joint MAE loss: $\mathcal{L}_\text{mae} = \mathcal{L}_\text{reg}(X, Y) + \mathcal{L}_\text{recon}(X, Y)$, where $\mathcal{L}_\text{reg}$ is the regression loss and $\mathcal{L}_\text{recon}$ is the masked autoencoder reconstruction loss.
- Meta pre-training uses AdamW with meta batch size $B=4$, inner loop steps $K=4$, meta learning rate $\beta=5\text{e-}4$ (cosine decay to $1\text{e-}6$), trained for 8 epochs.
- Test-time training uses AdamW with time interval $\tau = t_f$ (so past ground truth covers the full prediction horizon), DLO parameters $\gamma=1\text{e-}4$ and $\tau_\alpha=8$, and HSD threshold $k=3$.
- Actor-specific tokens are adopted to learn individual behavioral patterns.

## Key Experimental Results

| Configuration | Metric | MetaDAT | T4P (Prev. SOTA) | Gain |
|---|---|---|---|---|
| Lyft→nuS Short | mADE6/mFDE6 | 0.332/0.683 | 0.408/0.847 | 18.6%/19.4% |
| nuS→Way Short | mADE6/mFDE6 | 0.305/0.712 | 0.343/0.792 | 11.1%/10.1% |
| Way→nuS Short | mADE6/mFDE6 | 0.266/0.548 | 0.284/0.585 | 6.3%/6.3% |
| Short-term Avg. | mADE6/mFDE6 | 0.301/0.648 | 0.345/0.741 | 12.7%/12.5% |
| nuS→Lyft Long | mADE6/mFDE6 | 0.648/1.472 | 0.711/1.578 | 8.9%/6.7% |
| Lyft→nuS Long | mADE6/mFDE6 | 1.177/2.551 | 1.260/2.742 | 6.6%/7.0% |

### Ablation Study
- All three modules are individually effective and complementary: on Lyft→nuS (short-term), MP alone reduces mADE6 from 0.408 to 0.355, DLO to 0.376, HSD to 0.400, and all three combined achieve 0.332.
- MP contributes the most (long-term average mADE6 from 0.560 to 0.514), followed by DLO (0.530), and HSD the least—but HSD further improves the combined system.
- Learning rate robustness: at suboptimal $\alpha=0.01$, T4P yields mADE6 of 0.518 while MetaDAT remains at 0.407; at $\alpha=0.0001$, T4P yields 0.393 and MetaDAT 0.341.
- Few-shot scenario: with only 2,000 samples, MetaDAT (0.327/0.743) nearly matches T4P trained on 10,000 samples (0.343/0.792).
- Efficiency: MetaDAT achieves better prediction accuracy than T4P at the same FPS.

## Highlights & Insights
- The meta pre-training design is elegant—directly encoding "fast online adaptability" as the pre-training objective fundamentally resolves the offline–online misalignment problem.
- DLO's approach of adjusting learning rates based on the consistency of consecutive gradient directions is concise and effective, requiring no additional hyperparameter search to accommodate different shift magnitudes.
- The framework simultaneously optimizes two dimensions of TTT: *what to update on* (all samples → hard samples) and *how to update* (fixed → adaptive learning rate), with these two axes being complementary.
- Robustness to suboptimal hyperparameters is an important advantage for practical deployment.

## Limitations & Future Work
- The method relies on accurate online detection and tracking to obtain observed trajectories for training; perceptual noise in practice will degrade performance (acknowledged by the authors).
- The inner loop of meta pre-training increases training time, although mitigated by the first-order approximation and offline pre-training initialization.
- Validation is limited to the ForecastMAE backbone; generalizability to other predictors (e.g., HiVT, QCNet) has not been explored.
- The HSD threshold parameter $k=3$ is manually set and could be further automated.

## Related Work & Insights
- **vs T4P** (CVPR 2024): T4P introduces MAE loss and actor-specific tokens for TTT but retains standard offline pre-training and a fixed online update strategy. MetaDAT addresses both root problems (pre-training objective alignment + adaptive updates) on top of T4P, achieving a 12.7% improvement in short-term average mADE6.
- **vs AML** (ICRA 2023): AML also applies meta-learning but only adapts the final Bayesian linear regression layer of the decoder, limiting the adaptability of deep representations. MetaDAT applies meta pre-training to the full model parameters, offering greater flexibility; the performance gap is substantial (AML short-term average mADE6: 0.567 vs. MetaDAT: 0.301).
- **vs MEK** (2021): MEK uses an extended Kalman filter as the online optimizer without optimizing the pre-training stage, and is unstable in certain configurations (mFDE6=1.806).

The meta pre-training + test-time adaptive updating framework is transferable to other online learning scenarios (e.g., domain adaptation in medical imaging, cross-sensor adaptation in point cloud understanding). The DLO strategy of using consecutive gradient direction consistency to adjust learning rates is applicable to any online/continual learning setting. Adaptive thresholding via running statistics in HSD is more flexible than fixed-ratio selection. This work is related to ideas in `20260316_foundation_model_tta.md` on TTA, but MetaDAT emphasizes pre-training alignment—a dimension commonly overlooked by TTA methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Meta pre-training for offline–online alignment, combined with DLO and HSD, constitutes a complete and theoretically grounded design with clear contributions to TTT for trajectory prediction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six cross-domain configurations across three datasets, both short- and long-term settings, multiple baselines, full ablations, and robustness/efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, method description is rigorous, experiments are well presented, and algorithm pseudocode is readable.
- Value: ⭐⭐⭐⭐ Practically meaningful for online deployment of trajectory predictors in autonomous driving; the framework has reasonable generality.
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value to Me: ⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction](den_tp_a_density_balanced_data_curation_and_evaluation_framework_for_trajectory.md)
- [\[CVPR 2026\] FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](foss_modeling_long_range_dependencies_and_multimodal_uncertainty_in_trajectory_p.md)
- [\[CVPR 2026\] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving](dlwm_dual_latent_world_models_enable_holistic_gaussian-centric_pre-training_in_a.md)
- [\[CVPR 2026\] Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction](recover_to_predict_progressive_retrospective_learning_for_variable-length_trajec.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](test-time_3d_occupancy_prediction.md)

<!-- RELATED:END -->
