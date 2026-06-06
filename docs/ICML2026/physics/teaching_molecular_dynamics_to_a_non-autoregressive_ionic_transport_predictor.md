---
title: >-
  [Paper Note] Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor
description: >-
  [ICML 2026][Physics & Scientific Computing][Ionic Transport] This paper treats expensive atomic trajectories as "privileged auxiliary modalities" during training. A dual-modality trainer first learns dynamics from trajec…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Ionic Transport"
  - "Molecular Dynamics"
  - "Auxiliary Modality Learning"
  - "Closed-form Ridge Regression Initialization"
  - "Privileged Information"
date: 2026-05-08
content_hash: 55da3f1e2f6cedac
---

# Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor

**Conference**: ICML 2026  
**arXiv**: [2605.09311](https://arxiv.org/abs/2605.09311)  
**Code**: https://github.com/jykim-git/MD.git (Available)  
**Area**: AI for Science / Material Prediction / Auxiliary Modality Learning  
**Keywords**: Ionic Transport, Molecular Dynamics, Auxiliary Modality Learning, Closed-form Ridge Regression Initialization, Privileged Information

## TL;DR
This paper treats expensive atomic trajectories as "privileged auxiliary modalities" during training. A dual-modality trainer first learns dynamics from trajectories, which are then distilled via closed-form ridge regression into the hidden representations of a non-autoregressive predictor that only views equilibrium structures. This approach is 200× faster and more accurate than autoregressive SOTA in lithium-ion Mean Squared Displacement (MSD) prediction.

## Background & Motivation
**Background**: Predicting ionic transport properties (MSD, diffusivity, conductivity) of battery materials primarily relies on Molecular Dynamics (MD) simulations: numerically integrating Newton's equations starting from equilibrium structures to obtain atomic trajectories for property calculation. Even with MLIP acceleration, a single material still requires hours. The machine learning community offers two alternatives: autoregressive MD acceleration (e.g., LiFlow, generating trajectories step-by-step) and non-autoregressive material property prediction (e.g., MatFormer, ComFormer, DenseGNN, mapping structure to property in one forward pass).

**Limitations of Prior Work**:
- Autoregressive solutions are slow to infer and accumulate errors, leading to trajectory divergence.
- Non-autoregressive solutions offer fast inference but sacrifice accuracy as they cannot observe dynamic information.
- Existing methods are restricted to either "trajectory-available" or "structure-only" datasets, unable to leverage both when data is scarce in real-world scenarios.

**Key Challenge**: Ionic transport inherently involves long-term dynamics (rare jump events + vibrational background), but fast inference necessitates starting from static structures. Consuming dynamics while maintaining fast inference presents a fundamental contradiction between "input modality vs. inference cost." Furthermore, iterative optimization in traditional KD suffers from high variance in few-shot scenarios, making reliable knowledge transfer difficult.

**Goal**: (i) Enable a non-autoregressive predictor to learn dynamic priors without requiring trajectories at inference time; (ii) Utilize both "trajectory-labeled" and "structure-only" datasets; (iii) Achieve stable knowledge transfer even in data-scarce scenarios with minimal trajectory data.

**Key Insight**: Atomic trajectories are positioned as "privileged information" / "auxiliary modalities" (auxiliary modality learning, AML), present only during training. Strong priors are provided by pre-trained scientific foundation models (SevenNet for structural embeddings, MOMENT for temporal embeddings) to avoid learning from scratch on scarce data. Closed-form ridge regression replaces iterative optimization for modality alignment to circumvent variance explosion in SGD under small sample sizes.

**Core Idea**: A trio of "Privileged Dynamic Modalities + Closed-form Distillation + Cross-dataset Representation Initialization" allows a structure-only predictor to implicitly inherit dynamic representations learned from trajectories.

## Method

### Overall Architecture
Two levels of auxiliary modality learning: (1) **Model-level** —— A dual-modality trainer $g$ is first trained on the trajectory dataset $\mathcal{D}^{trj}$ (consuming trajectory embeddings $\mathbf{E}_\mathbf{p}$, structural embeddings $\mathbf{E}_\mathbf{x}$, and temperature embeddings $\mathbf{E}_T$). Then, the merged hidden representation $\mathbf{H}=\mathbf{H}_\mathbf{p}+\mathbf{H}_{\mathbf{x},T}$ of $g$ is distilled via closed-form ridge regression into the encoder of predictor $f_1$, followed by fine-tuning; (2) **Data-level** (Optional) —— When training predictor $f_2$ for structure-only datasets $\mathcal{D}^{str}$, the encoder is initialized from $g$'s structural encoder, and the decoder is initialized from $f_1$'s decoder, transferring dynamic knowledge across datasets.

### Key Designs

1. **Dual-modality Trainer $g$ + Structure-only Regularization**:
    - **Function**: Forces the structural encoder to learn useful dynamics-related representations during trajectory-based training, preventing it from being overshadowed by the trajectory encoder.
    - **Mechanism**: $g$ consists of two linear layers $\mathbf{W}_\mathbf{p}$ (trajectory) and $\mathbf{W}_{\mathbf{x},T}$ (structure + temperature). Their sum $\mathbf{H}$ is passed through an MLP decoder. An auxiliary loss term requires accurate prediction using only structural embeddings: $\mathcal{L}(\hat y^i,y_s^i)+\lambda_b\mathcal{L}(\hat y^i_{\mathbf{x},T},y_s^i)$. Structural embeddings utilize third-order polynomial expansion $\mathbf{E}_\mathbf{x}=[\mathbf{E}_{a,s};\mathbf{E}_{a,s}\odot\mathbf{E}_{a,s};\mathbf{E}_{a,s}^{\odot 3}]$ after SevenNet node/edge feature aggregation to supplement linear layer expressiveness.
    - **Design Motivation**: Trajectory signals are too strong; without regularization, the structural encoder becomes an empty "placeholder," leaving nothing for the structure-only model to inherit during closed-form distillation.

2. **Closed-form Ridge Regression Distillation Initialization**:
    - **Function**: Directly transfers the hidden representation $\mathbf{H}^i$ of $g$ to the encoder $\mathbf{W}^{trj}$ of the structure-only predictor $f_1$.
    - **Mechanism**: Solving $\min_{\mathbf{W}^{trj}}\sum_i\|\mathbf{X}^i\mathbf{W}^{trj}-\mathbf{H}^i\|_F^2+\lambda_r\|\mathbf{W}^{trj}\|_F^2$ yields the closed-form solution $\mathbf{W}^{trj}=(\sum_i(\mathbf{X}^i)^\top\mathbf{X}^i+\lambda_r\mathbf{I})^{-1}(\sum_i(\mathbf{X}^i)^\top\mathbf{H}^i)$, computed via Cholesky decomposition to floating-point precision. The decoder $g_{\text{dec}}$ is reused directly. Subsequent fine-tuning on $\mathcal{D}^{trj}$ uses only structure-temperature embeddings without trajectories.
    - **Design Motivation**: Traditional KD using iterative gradient optimization exhibits extreme variance in small-data scenarios (dozens to hundreds of samples) typical of ionic transport. The closed-form solution is determinant and requires no learning rate tuning or early stopping, making it a natural choice for data-scarce settings.

3. **Cross-dataset Initialization (data-level AML)**:
    - **Function**: Migrates dynamic priors learned from trajectory datasets to structure-only datasets $\mathcal{D}^{str}$ that lack trajectories entirely.
    - **Mechanism**: The encoder $\mathbf{W}^{str}$ of $f_2$ is initialized from the structural encoder $\mathbf{W}_{\mathbf{x},T}$ of $g$ (as its learned representations are more general under structure-only regularization), while the decoder is initialized from $f_1$'s $f_{\text{dec}}^{trj}$ (which captures robust mappings from hidden representations to transport properties). This "structural-path encoder + trajectory-path decoder" cross-initialization avoids the issue where $\mathbf{W}^{trj}$ is overfitted to the trajectory distribution.
    - **Design Motivation**: Since $\mathbf{W}^{trj}$ was fitted to $\mathbf{H}$ via the closed-form solution, it is biased toward the trajectory distribution and generalizes poorly to structure-only data. Conversely, $\mathbf{W}_{\mathbf{x},T}$ has been regularized to learn structural representations, making it a better starting point for new datasets.

### Loss & Training
The entire process uses $L_1$ loss to predict transport properties on a $\log_{10}$ scale. The dual-modality trainer includes an auxiliary structure-only term weighted by $\lambda_b$. Closed-form initialization uses $\lambda_r$ to balance fitting vs. overfitting. The three datasets are: Dataset 1 (MD-calculated Li-MSD, trajectory-based), Dataset 2 (MD-calculated multi-element diffusivity, structure-only, Na reserved for unseen species testing), and Dataset 3 (Experimental Li-conductivity, structure-only).

## Key Experimental Results

### Main Results

| Method | Type | Dataset 1 Inference Time (s) | MAE@600K | MAE@800K | MAE@1000K | MAE@1200K |
|------|------|----------------------|----------|----------|-----------|-----------|
| LiFlow (Nam 2025) | Autoregressive | 2910 | 0.378 | 0.392 | 0.457 | 0.407 |
| MatFormer | Non-autoregressive | 22 | 0.604 | 0.685 | 0.894 | 1.207 |
| ComFormer | Non-autoregressive | 14 | 0.451 | 0.531 | 0.642 | 0.760 |
| DenseGNN | Non-autoregressive | 29 | 0.412 | 0.472 | 0.531 | 0.523 |
| **Ours** | Non-autoregressive | **14** | **0.344** | **0.367** | **0.402** | **0.390** |

Ours is approximately 200× faster than LiFlow, with lower MAE across all temperatures (preserving dynamic knowledge).

Cross-dataset results:

| Method | Dataset 2 MAE($\log_{10}D_{Na}$)@2500K | Dataset 3 MAE($\log_{10}\sigma_{Li}$)@300K |
|------|----------------------------------------|--------------------------------------------|
| MatFormer | 0.651 | 2.090 |
| ComFormer | 0.517 | 2.150 |
| DenseGNN | 0.312 | 2.048 |
| **Ours** | **0.064** | **1.388** |

Gain of 5× on Dataset 2; Dataset 3 (real experimental data) saw an MAE reduction of 0.66.

### Ablation Study

| Configuration | Dataset 1 MAE@600K |
|------|-------------------|
| Full | 0.344 |
| w/o model-level AML | 0.395 |

The appendix further verifies: Removing structure-only regularization makes the structural encoder useless after closed-form distillation; removing data-level AML eliminates most gains on Datasets 2/3; substituting the closed-form solution with iterative SGD decreases accuracy in data-scarce scenarios.

### Key Findings
- **Dynamic priors are distillable**: Even without trajectories at inference, models can inherit vibration + jump patterns learned from trajectories. The key is Fourier transformation to the frequency domain + MOMENT temporal foundation model for compact representation.
- **Cross-dataset transfer holds across ionic species**: Na-ions benefited from representations learned on Li-data despite being excluded during training.
- **Small data + Strong priors**: On scales of hundreds of samples, closed-form solutions + pre-trained foundation model embeddings far outperform deep networks trained from scratch.

## Highlights & Insights
- **Practical combination of "Privileged Information + Closed-form Distillation"**: Implements the LUPI framework in a material science context, demonstrating that closed-form solutions are more stable than SGD distillation for small data—a recipe for "few-shot knowledge transfer" worth promoting.
- **Polynomial embeddings for linear layers**: Using $[\mathbf{E}; \mathbf{E}^{\odot 2}; \mathbf{E}^{\odot 3}]$ enables finite-order nonlinearity for linear mappings. Combined with SevenNet's strong priors, this enhances expressiveness without adding significant parameters.
- **Cross-dataset encoder/decoder cross-initialization**: Avoids the pitfall where the encoder becomes locked to the source domain distribution after closed-form distillation. This logic is transferable to any "pre-train on rich domain, transfer to poor domain" scenario.

## Limitations & Future Work
- Closed-form solutions require $D \times D$ matrix inversion; caution is needed for high embedding dimensions (currently mitigated by linear layers + polynomial expansion).
- Experiments were validated primarily on Li/Na; more complex multi-element co-diffusion scenarios require further testing.
- MAE on real experimental Dataset 3 remains high (1.388, an order of magnitude error!), indicating that sim-to-real remains an open problem requiring more experimental data and domain adaptation.
- Assumes consistent trajectory length $L$; more complex MD protocols (variable length/temperature ramps) require redesigned Fourier representations.

## Related Work & Insights
- **vs. LiFlow (autoregressive)**: LiFlow samples atomic trajectories via generative models to compute properties, which is slow and accumulates error; Ours uses direct NAR prediction and is more accurate.
- **vs. MatFormer / ComFormer / DenseGNN**: These share the structure-to-property NAR approach but lack dynamic knowledge; Ours injects this via AML.
- **vs. Traditional KD (Hinton 2015)**: Traditional KD uses iterative gradients for logit distillation; Ours uses closed-form representation distillation tailored for data-scarce settings.
- **vs. LUPI / Generalized Distillation**: This marks the first time atomic trajectories have been positioned as privileged modalities for material prediction.
- **Insight**: In scenarios like biomedicine or chemistry where data is scarce but "expensive oracles" (e.g., wet labs, quantum simulations) exist, this paradigm of "pre-train on rich modality, distill via closed form, transfer across datasets" is highly reusable.

## Rating
- Novelty: ⭐⭐⭐⭐ First use of atomic trajectories as privileged modality + closed-form distillation; highly novel in AI4Science.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets covering sim+real, extensive comparison with autoregressive and NAR baselines, complete ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear three-part motivation (cost/accuracy/data scarcity), intuitive methodology diagrams.
- Value: ⭐⭐⭐⭐ 200× acceleration + cross-dataset transfer, direct engineering value for battery material screening.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Speculative Sampling for Faster Molecular Dynamics](speculative_sampling_for_faster_molecular_dynamics.md)
- [\[ICML 2026\] Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)
- [\[ICLR 2026\] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](../../ICLR2026/physics/hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)
- [\[NeurIPS 2025\] FEAT: Free Energy Estimators with Adaptive Transport](../../NeurIPS2025/physics/feat_free_energy_estimators_with_adaptive_transport.md)
- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](../../AAAI2026/physics/pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)

</div>

<!-- RELATED:END -->
