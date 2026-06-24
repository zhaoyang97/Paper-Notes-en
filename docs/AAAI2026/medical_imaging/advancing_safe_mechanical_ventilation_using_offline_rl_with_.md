---
title: >-
  [Paper Note] Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards
description: >-
  [AAAI 2026][Medical Imaging][Offline reinforcement learning] This paper addresses the problem of optimizing mechanical ventilation (MV) settings in the ICU via offline RL. A hybrid action space approach (HybridIQL/HybridEDAC) is proposed to avoid distributional shift caused by conventional discretization. Clinically aligned reward functions are introduced based on ventilator-free days (VFD) and physiological safety ranges, with multi-objective optimization used to select the…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Offline reinforcement learning"
  - "mechanical ventilation"
  - "hybrid action space"
  - "clinical reward design"
  - "ICU decision support"
date: 2026-05-08
content_hash: 4abe8910d478d7e5
---

# Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards

**Conference**: AAAI 2026
**arXiv**: [2506.14375v2](https://arxiv.org/abs/2506.14375v2)  
**Code**: [Available](https://github.com/NIMI-research/intellilung-advancing-mechanical-ventilation)  
**Area**: Medical AI / Offline Reinforcement Learning
**Keywords**: Offline reinforcement learning, mechanical ventilation, hybrid action space, clinical reward design, ICU decision support

## TL;DR
This paper addresses the problem of optimizing mechanical ventilation (MV) settings in the ICU via offline RL. A hybrid action space approach (HybridIQL/HybridEDAC) is proposed to avoid distributional shift caused by conventional discretization. Clinically aligned reward functions are introduced based on ventilator-free days (VFD) and physiological safety ranges, with multi-objective optimization used to select the optimal reward. The number of optimizable ventilation parameters is scaled from 2–3 to 6, and HybridIQL achieves the best balance between performance and policy coverage.

## Background & Motivation
Invasive mechanical ventilation (MV) is one of the most commonly used life-sustaining interventions in the ICU, particularly prominent during the COVID-19 pandemic. However, MV itself can induce ventilator-induced lung injury (VILI). Clinical guidelines provide only general recommendations, and actual parameter settings are heavily dependent on clinician experience; adherence to lung-protective ventilation strategies remains globally low. MV also demands high nurse-to-patient ratios, and elevated workloads can lead to suboptimal recovery and prolonged ICU stays.

Existing offline RL methods for MV optimization face three critical limitations:
1. **Action space explosion**: Discretizing continuous settings leads to exponential growth in the action space (6 parameters = 18,144 combinations), forcing prior work to optimize only 2–3 parameters.
2. **Distributional shift from discretization**: Training uses discrete bins, but inference requires mapping back to continuous values; different mapping strategies introduce varying degrees of distributional shift, compromising safety.
3. **Poorly designed rewards**: Most prior work uses mortality as a terminal reward, yet medical research shows mortality is a highly confounded outcome that does not reliably reflect MV treatment quality.

## Core Problem
How can a safe and reliable MV parameter recommendation policy be learned from offline ICU data? Specific challenges include: (1) how to natively handle a mixed continuous-discrete action space; (2) how to design rewards genuinely aligned with clinical objectives; and (3) how to ensure the learned policy does not deviate from clinically safe ranges.

## Method

### Overall Architecture
The system is an offline RL-based AI decision support system (AI-DSS). It takes a 26-dimensional clinical state as input (vital signs + respiratory parameters + laboratory values + fluid balance + demographics) and outputs settings for 6 MV control parameters (ventilation mode, respiratory rate, tidal volume, driving pressure, PEEP, FiO2). The system is trained on three public ICU databases—MIMIC-IV, eICU, and HiRID (12,572 patients in total, 1.25 million ventilation-hours)—and was developed in close collaboration with clinical experts from multiple hospitals.

### Key Designs

1. **Clinically Aligned Reward Function (C1)**:

    - **Primary objective**: Ventilator-free days (VFD) replace mortality. $\text{VFD} = \text{alive} \times \max(0, \min(\text{reintubation time}, 30\text{ days}) - \text{ventilation days})$, jointly capturing survival and ventilation duration.
    - **Secondary objective**: Range reward $r_{range}$, checking whether 7 physiological parameters (blood pH, MAP, PaO2, SaO2, PaCO2, heart rate, SpO2) fall within safety ranges (parameters and weights determined by multi-hospital expert Delphi consensus).
    - **Time penalty** $r_{tp} = -1$, preventing the policy from prolonging ventilation to accumulate positive range rewards.
    - Total reward: $r = r_{range} + r_{tp} + r_{vfd}$.
    - The application mode of VFD reward (stepwise vs. terminal) and weight $w_{vfd}$ are selected via **Tchebycheff multi-objective optimization**, finding the Pareto-optimal trade-off between Corr@VFD and Corr@RangeReward.

2. **Constrained Discrete Action Space (C2)**:

    - **Action space constraint**: Only action combinations actually used by clinicians in the dataset are retained, reducing the space to 53.6% of the original; further leveraging mutual exclusivity of ventilation modes (masking driving pressure under VCV, masking tidal volume under PCV) reduces it to 6.9%.
    - **Factorized Q decomposition**: $Q(s,a) \approx \sum_{k=1}^{K} q_k(s, a_k)$, reducing output dimensionality from $O(\prod_k |A_k|)$ to $O(\sum_k |A_k|)$. This reduces variance at the cost of a bias (ignoring cross terms), which is a favorable variance-bias trade-off in low-coverage settings.

3. **Offline RL with Hybrid Action Space (C3)**:

    - **HybridIQL**: The critic receives continuous actions concatenated with one-hot discrete actions. The policy is optimized with AWR: $\log\pi_\phi(a|s) = \log\pi^d_\phi(a_d|s) + \log\pi^c_\phi(a_c|s)$.
    - **HybridEDAC**: An ensemble method based on SAC, with the critic adapted to accept hybrid inputs rather than outputting Q-values for each discrete combination (empirically more stable). The discrete component is updated by computing the exact expectation over the discrete distribution directly (rather than using Gumbel-Softmax), substantially reducing policy update variance.

4. **Distributional Shift Analysis of Discretization (C4)**:

    - Four bin-to-value mapping strategies are compared: bin mode / Gaussian sampling / bin mean / uniform sampling.
    - Uniform sampling yields the lowest coverage (−1.26); bin mode yields the highest (−0.62), demonstrating that discretization introduces non-trivial distributional shift.

### Loss & Training
- Discrete setting: CQL with $\alpha=0.1$; IQL with $\tau=0.8, \beta=5$; learning rates $10^{-6}$ (CQL) / $5 \times 10^{-5}$ (IQL); 100K training steps.
- Hybrid setting: HybridIQL learning rate $10^{-4}$; HybridEDAC learning rate $3 \times 10^{-5}$; 25 critic ensemble members; continuous entropy target $\mathcal{H}_c=-0.3$; discrete entropy target $\mathcal{H}_d=0.3$.
- Evaluation uses Fitted Q-Evaluation (FQE) and distributional FQE (QR-DQN); policy coverage $d_\pi$ is measured using a fitted behavioral policy model.

## Key Experimental Results

| Method | $V^\pi$ (vs. clinician) | $d_\pi$ (policy coverage) | Characteristics |
|--------|------------------------|--------------------------|-----------------|
| CQL (baseline) | Highest | Lowest | Severe OOD overestimation |
| CF-CQL (proposed) | Slightly below CQL | Substantially improved | Safe and reliable |
| HybridIQL (proposed) | Higher than CF-CQL | **Highest** | Best performance–safety balance |
| HybridEDAC | Highest (hybrid) | Low | Overestimation issue similar to CQL |
| DiscreteIQL | Lower than HybridIQL | Lower than HybridIQL | Confirms hybrid advantage |

- VFD@EachStep + $w_{vfd}=0.5$ is the Tchebycheff-optimal reward: Corr@VFD improves from 0.48 to 0.56 while maintaining high Corr@RangeReward.
- Mortality-based reward only achieves high Corr@VFD at $w_{morta}=100$, at which point Corr@RangeReward drops to 0.13.
- Cross-dataset generalization (training on eICU+HiRID, testing on MIMIC-IV): HybridIQL performs best with coverage exceeding the clinician policy.

### Ablation Study
- **Factorized critic contributes most**: F-CQL performance approximates CF-CQL, indicating that coverage gains primarily stem from the factorized critic rather than action constraints.
- Action constraints mainly serve to eliminate unsafe actions with limited independent contribution to coverage improvement.
- Increasing CQL regularization $\alpha$ improves $d_\pi$ but significantly degrades $V^\pi$, making it inferior to the factorized critic approach.
- HybridIQL is the most robust to hyperparameters (162 configuration sweeps); CF-CQL ranks second; CQL and HybridEDAC exhibit high variance and poor coverage.

## Highlights & Insights
- **Elegant adaptation of offline RL to hybrid action spaces**: Computing the exact expectation over the discrete distribution instead of using Gumbel-Softmax substantially reduces variance; AWR log-probability in IQL is directly decomposed into discrete and continuous components.
- **Tchebycheff multi-objective reward selection** is practically useful: it eliminates manual trial-and-error weight tuning and automatically balances multiple clinical objectives.
- **Emphasis on policy coverage $d_\pi$ as a safety metric**: evaluating $V^\pi$ alone is insufficient—high $V^\pi$ with low $d_\pi$ implies OOD action overestimation, which is problematic for real deployment.
- **Scaling from 2–3 to 6 parameters** carries substantial practical significance: prior work was constrained to a small number of parameters due to action space explosion.
- The entire system was developed in close collaboration with clinical experts from multiple hospitals in Europe and North America, with Delphi consensus used for parameter selection and a clear deployment orientation.

## Limitations & Future Work
- When offline data itself contains unsafe actions, constraining to the data distribution may still result in recommending unsafe operations → future work should incorporate hard constraints from ventilation guidelines.
- Data from public databases contain substantial noise and limited resolution; high-quality prospective data are needed.
- The reward function is defined at the cohort level and may not capture the needs of specific subgroups or individual patients.
- All evaluations are offline and retrospective: FQE and coverage are proxy metrics; true efficacy requires prospective RCTs (planned).
- Factorized Q decomposition ignores interactions between action dimensions, which may introduce excessive bias in settings with strong cross-dimensional dependencies.

## Related Work & Insights
- **Kondrup et al. (AAAI 2023)**: Uses CQL with mortality reward and Apache-II intermediate reward, optimizing only 2–3 parameters; severe OOD issues after discretization. The present work comprehensively advances over this in reward design (VFD), action space (hybrid), and scale (6 parameters).
- **Chen et al. (2022)**: Also employs a hybrid action space but based on off-policy SAC, lacking offline safety regularization and potentially overestimating. The conservative estimation via IQL/EDAC in this work is safer.
- **Eghbali et al. (2024)**: Uses conformal prediction for uncertainty quantification but still relies on discretization and mortality reward. The VFD reward and hybrid actions proposed here offer a distinct approach.

The hybrid action space RL adaptation methodology is transferable to other medical settings involving mixed continuous-discrete control (e.g., discrete drug selection combined with continuous infusion rates). The Tchebycheff multi-objective reward selection framework is applicable for balancing multiple clinical indicators in other RL-based medical systems. Factorized Q decomposition has broad applicability in medical decision-making with large action spaces.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hybrid action space + clinically aligned reward + multi-objective selection represents a valuable engineering contribution, though individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three databases, multiple baselines, cross-dataset generalization, hyperparameter robustness, ablation studies, and distributional shift analysis — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, thorough explanation of clinical motivation, and well-organized C1–C4 contribution labeling.
- Value: ⭐⭐⭐⭐ Complete system with a clear deployment orientation (multi-hospital collaboration + planned RCT), though offline evaluation remains a limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Clinically-Grounded Counterfactual Reasoning for Medical Video Diagnosis](../../CVPR2026/medical_imaging/clinically-grounded_counterfactual_reasoning_for_medical_video_diagnosis.md)
- [\[CVPR 2026\] fMRI-LM: Towards a Universal Foundation Model for Language-Aligned fMRI Understanding](../../CVPR2026/medical_imaging/fmri-lm_towards_a_universal_foundation_model_for_language-aligned_fmri_understan.md)
- [\[CVPR 2026\] Virtual Immunohistochemistry Staining with Dual-Aligned Multi-Task Feature Guidance](../../CVPR2026/medical_imaging/virtual_immunohistochemistry_staining_with_dual-aligned_multi-task_feature_guida.md)
- [\[CVPR 2026\] SurgCoT: Advancing Spatiotemporal Reasoning in Surgical Videos through a Chain-of-Thought Benchmark](../../CVPR2026/medical_imaging/surgcot_advancing_spatiotemporal_reasoning_in_surgical_videos_through_a_chain-of.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](../../CVPR2026/medical_imaging/chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)

</div>

<!-- RELATED:END -->
