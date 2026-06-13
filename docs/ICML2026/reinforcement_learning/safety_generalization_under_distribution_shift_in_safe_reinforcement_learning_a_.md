---
title: >-
  [Paper Note] Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed
description: >-
  [ICML 2026][Reinforcement Learning][Safe Reinforcement Learning] The authors constructed a unified T1D/T2D diabetes simulator based on the UVA-Padova physical model. They found that while 8 mainstream Safe RL algorithms…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Safe Reinforcement Learning"
  - "Distribution Shift"
  - "Test-time Shielding"
  - "Neural ODE"
  - "Diabetes Decision Making"
date: 2026-05-08
content_hash: fe37a6b7f2f5076a
---

# Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed

**Conference**: ICML 2026  
**arXiv**: [2601.21094](https://arxiv.org/abs/2601.21094)  
**Code**: GlucoSim + GlucoAlg (GitHub, safe-autonomy-lab)  
**Area**: Reinforcement Learning / Safe RL  
**Keywords**: Safe Reinforcement Learning, Distribution Shift, Test-time Shielding, Neural ODE, Diabetes Decision Making  

## TL;DR
The authors constructed a unified T1D/T2D diabetes simulator based on the UVA-Padova physical model. They found that while 8 mainstream Safe RL algorithms can satisfy safety constraints on training patients, their Time-in-Range generally drops by 8–13% when deployed to unseen patients. Consequently, they proposed using Basis-Adaptive Neural ODE to predict blood glucose trajectories and applying predictive shielding at test-time to filter dangerous actions, allowing baselines like PPO-Lag / CPO to recover 13–14% TIR on OOD patients.

## Background & Motivation

**Background**: Safe RL models control problems as CMDPs, using techniques like Lagrangians, trust regions (CPO), or projections (PCPO) to enforce cumulative cost constraints during training. These algorithms provide safety guarantees in an "expected sense" under fixed dynamics. Diabetes management is a frequently studied safety-critical application.

**Limitations of Prior Work**: All Safe RL guarantees are "distribution-dependent"—the safety boundaries learned during training are tied to the physical parameters encountered at that time. However, in real-world deployment, insulin sensitivity, absorption rates, and metabolic rates vary by individual and cannot be exhaustively sampled in a training set. Existing RL work for diabetes (fox2020deep / zhu2023offline) defaults to matching training and testing dynamics and mostly studies T1D, lacking systematic evaluation of safety robustness under population diversity.

**Key Challenge**: "Constraint satisfaction" during training and "persistent safety" during deployment are two different things under distribution shift. Especially in healthcare, shifts are latent and structural (unobservable metabolic parameters) rather than observable geometric parameters as in robotics; ethics also prohibit online trial-and-error retraining.

**Goal**: (i) Quantify the "safety generalization gap" of mainstream Safe RL; (ii) Provide an algorithm-agnostic test-time mechanism that requires no retraining to bridge this gap; (iii) Provide a unified clinical simulator supporting T1D/T2D + pump/non-pump as a reusable testbed.

**Key Insight**: Transform the traditional test-time shielding approach—which typically relies on formal logic or known dynamics—to be driven by a "learned dynamics predictor + probabilistic bounds," explicitly decoupling "training-time constraints" from "deployment-time verification."

**Core Idea**: Use a continuous-time dynamics predictor (BA-NODE) capable of function-space adaptation for individual patients to predict blood glucose trajectories $H$ steps after a candidate action. Actions predicted to exceed boundaries are shielded. The shielding threshold is tighter than the clinical failure threshold, with the buffer margin exactly matching the predictor's error bound.

## Method

### Overall Architecture
The system consists of three layers: (i) GlucoSim, a unified diabetes simulator extended from the UVA-Padova model, supporting T1D pump, T2D pump, and T2D non-pump scenarios. It provides a 14-dimensional CGM/IOB/meal history observation to the agent, which outputs discrete bolus + meal recommendations filtered by a "patient acceptance model"; (ii) BA-NODE individualized dynamics predictor, which predicts the future $H$-step blood glucose trajectory given historical observations; (iii) Test-time predictive shielding, which filters the action distribution of any pre-trained Safe RL policy online. During training, one representative patient (Child#01 / Adolescent#01 / Adult#01) is used for 11 days; during deployment, zero-shot evaluation is conducted on 9 unseen patients for 77 days.

### Key Designs

1. **Unified Diabetes Simulator + OOD Safety Benchmark**:
    - **Function**: Consolidates T1D/T2D + pump/non-pump into a single MDP interface, capturing realistic clinical scenarios of "treatment decision support" rather than "direct continuous control," and explicitly creating parameter-level and duration-level distribution shifts.
    - **Mechanism**: The physical layer is based on UVA-Padova. T2D is modeled using a hybrid of Hovorka secretion dynamics and the Dalla Man transport model for insulin resistance. The MDP reward uses a two-hour prediction window to calculate risk delta, incorporating delayed effects like insulin stacking and rebound hyperglycemia. Costs include "frequent intervention" penalties for clinical realism. The Risk Index uses the Kovatchev asymmetric transform $r_t = 10 f(G_t)^2$, where $f(G) = 1.509 (\ln(G)^{1.084} - 5.381)$, penalizing hypoglycemia more heavily than hyperglycemia. Evaluation splits shifts into parameter generalization (testing on Patient #02–#10) and duration generalization (11 days training vs. 77 days testing).
    - **Design Motivation**: Controlling continuous basal rates directly is an unrealistic clinical assumption. The simulator must be realistic enough for "training-time compliance vs. deployment-time failure" to emerge naturally to serve as a Safe RL testbed.

2. **Basis-Adaptive Neural ODE (BA-NODE) Individualized Dynamics Predictor**:
    - **Function**: Zero-shot adapts to specific patients and stably predicts multi-step blood glucose with a small context window, even when latent variables (sensitivity, absorption) are unobservable.
    - **Mechanism**: Three modules in series—(a) ITransformer treats each physiological variable as a token for cross-variable self-attention, obtaining a per-variate summary projected into the initial latent state $h_0$; (b) $K$ parallel neural ODE vector fields $\{f_{\theta_k}\}$ advance one step using RK4 to obtain $K$ candidate latent trajectories, combined into a single latent trajectory via a shared linear projection $W_{\mathrm{proj}} \in \mathbb{R}^{K \times 1}$; (c) Treating these $K$ rollouts as "basis trajectories" $\{G_k(\cdot)\}$, collect $N$ context windows for each patient and solve for patient-specific weights $w^\star$ via regularized least squares: $w^\star = \arg\min_w \|\tilde G w - \tilde y_{\text{ctx}}\|_2^2 + \lambda \|w\|_2^2$. The final prediction is $\hat y_{T+P} = y_T + \sum_{i=1}^{P} (G(x_{\text{pred}}) w^\star)_i$.
    - **Design Motivation**: Original Function Encoders only perform static regression and cannot handle historical dependencies in blood glucose. BA-NODE replaces "static function bases" with "dynamic basis trajectories," allowing function-space adaptation to support time series naturally. The ODE ensemble satisfies both high expressivity and single-trajectory coherence.

3. **Predictive Shielding at Test-Time**:
    - **Function**: An algorithm-agnostic runtime safety wrapper that masks the policy's action distribution to lower the probability of actions predicted to violate safety, without zeroing them out (maintaining exploration tolerance).
    - **Mechanism**: The shielded policy is $\pi_{\text{shielded}}(a|s) = \mathrm{Softmax}(\log \pi_\theta(a|s) + M(s,a))$, where $M$ is formed by three rules: (i) **Critical Rescue**: If $BG_t < G_{\text{rescue}} = 60\,\text{mg/dL}$, mandatory 15g rescue carbs are administered and all insulin actions are shielded; (ii) **Predictive Safety**: If $BG_t \geq G_{\text{shield}}^\downarrow = 80\,\text{mg/dL}$, BA-NODE predicts $H$-step trajectories for top-$k$ bolus candidates × all discrete meal combinations, penalizing actions where $m(a) < G_{\text{shield}}^\downarrow$ or $M(a) > G_{\text{shield}}^\uparrow$; (iii) **Gating**: In the transition zone $[G_{\text{rescue}}, G_{\text{shield}}^\downarrow)$, prediction verification is paused to avoid repeated rescue from small noise. Accompanied by a **probabilistic safety bound**: Under a one-sided $(\varepsilon, \alpha)$-reliable hypothesis, setting the shielding threshold to $G_{\text{shield}}^\downarrow = G_{\text{fail}}^\downarrow + \varepsilon$ guarantees $\Pr(\min_\tau BG_\tau(a) \geq G_{\text{fail}}^\downarrow) \geq 1 - \alpha$.
    - **Design Motivation**: Training constraints inevitably fail under OOD; test-time remediation is required. Using a predictor for trajectory-level checks prevents delayed risks like "insulin stacking" better than single-step rules. Using margins in the threshold converts model prediction error into a safety margin above clinical failure lines.

### Loss & Training
8 Safe RL baselines (PPO-Lag, TRPO-Lag, CPO, RCPO, FOCOPS, PCPO, CRPO, CUP) were trained on each cohort × representative patient using original hyperparameters, with a Rule-Based Shield (RBS) as a control. BA-NODE was trained on 15 days of pre-trained policy trajectories per cohort, with 5 days for evaluation. Prediction windows extended up to 120 minutes (24 steps). Evaluation utilized clinical metrics (TIR, CV, Risk Index), avoiding circular validation with training reward/cost.

## Key Experimental Results

### Main Results

**Safety Generalization Gap (No Shielding)**: Comparing training patients (ID) vs. unseen patients (OOD) across 8 algorithms (selected):

| Algorithm | TIR ID (%) ↑ | TIR OOD (%) ↑ | ΔTIR | ΔRisk |
|---|---|---|---|---|
| CPO | 87.28 | 76.73 | -10.55 | +1.62 |
| CUP | 89.36 | 77.70 | -11.66 | +2.61 |
| FOCOPS | 88.08 | 75.42 | -12.66 | +2.77 |
| PPO-Lag | 85.29 | 75.20 | -10.09 | +2.16 |
| TRPO-Lag | 82.79 | 74.43 | -8.37 | +1.79 |

7 out of 8 algorithms saw a TIR drop $\geq 8\%$ and a Risk Index increase $\geq 1.6$ on OOD, confirming that the safety generalization gap is structural rather than a flaw of specific algorithms.

**BA-NODE Prediction Accuracy (24 steps = 120 mins)**:

| Model | MAE ↓ | FDE ↓ | RMSE ↓ |
|---|---|---|---|
| ITransformer | 4.11 | 5.39 | 7.25 |
| NODE | 3.18 | 4.11 | 5.10 |
| **BA-NODE** | **2.82** | **3.63** | **4.42** |

BA-NODE reduced both mean and variance across all metrics, proving that function-space adaptation stabilizes multi-step prediction.

### Ablation Study

**T1D Before and After Shielding (Selected)**:

| Algorithm | TIR (%) | ΔTIR | Risk Index | ΔRisk | CV (%) | ΔCV |
|---|---|---|---|---|---|---|
| CPO | 85.50 | +4.70 | 3.44 | -1.51 | 25.40 | -2.56 |
| FOCOPS | 82.63 | +3.07 | 4.53 | -0.97 | 29.63 | -4.26 |
| PPO-Lag | 85.59 | +8.05 | 3.96 | -1.79 | 29.18 | -3.21 |
| RCPO | 86.45 | +6.90 | 3.34 | -2.26 | 26.22 | -3.92 |

**Greater Gains in T2D**: CPO ΔTIR +13.54%, ΔCV −6.68%; PPO-Lag ΔTIR +14.15%, ΔCV −5.5%. Predictive shielding consistently outperformed RBS across 72 settings (8 algorithms × 3 diabetes types × 3 age groups), except where the base policy itself was poor (e.g., PCPO TIR < 40%)—shielding can only adjust action distributions and cannot manifest good actions from zero probability.

### Key Findings
- **Training Compliance $\neq$ Deployment Safety**: Nearly all policies with Risk Index < 5 and TIR > 80% during training failed OOD, showing that expected-sense safety guarantees in CMDPs are insufficient under distribution shift.
- **Predictive Shielding > Rule-Based Shielding**: RBS often exacerbated CV through positive feedback loops (high glucose $\rightarrow$ insulin $\rightarrow$ hypoglycemia $\rightarrow$ carbs $\rightarrow$ high glucose), whereas predictive shielding avoided short-sightedness by looking $H$ steps ahead.
- **Base Policy Lower Bound is Critical**: The effectiveness of shielding is limited by the base policy's probability mass; if the base policy places all mass on dangerous actions, the shield can only retain the "second worst" option.
- **$(\varepsilon, \alpha)$ Bounds Unify Theory and Engineering**: Clinicians only need to select an acceptable $\alpha$, and the system uses the model's $\varepsilon$ error bound to calculate the required shielding tightness.

## Highlights & Insights
- **Converting "Dynamics Prediction Accuracy" into a "Safety Margin"** is the most clever design: The shielding threshold isn't arbitrary; it is $G_{\text{shield}}^\downarrow = G_{\text{fail}}^\downarrow + \varepsilon$, allowing prediction error to be absorbed as a safety buffer. This provides a high-probability bound of $1 - \alpha$ that is interpretable and engineerable.
- **Function-Space Adaptation + ODE Ensembles** are more practical than training a separate NODE for every patient—solving one least-squares problem gives patient-specific weights zero-shot. No online gradient updates are needed, fitting the clinical constraint of "no trial-and-error on patients."
- **Algorithm-Agnostic Wrapper** is transferable: Any high-risk control scenario (drug dosing, power dispatch, low-speed autonomous driving) can adopt this "predict candidate trajectory $\rightarrow$ shield sub-standard actions" paradigm, provided a trajectory predictor can be trained offline.
- **Introduction of CV Metric** shows clinical intuition: While TIR measures time in range, it tolerates oscillations. CV measures trajectory smoothness. Predictive shielding truly maintains safety "gently."

## Limitations & Future Work
- The patient-specific weight $w^\star$ assumes the context window is "representative enough"; cold-start robustness for the first few hours with a new patient is not systematically verified.
- Test-time shielding filters actions rather than retraining the policy; over time, it may mask base policy flaws, leading to behaviors drifting toward shielding boundaries.
- Currently, $\varepsilon$ is estimated on an IID calibration set. Since distribution shifts increase $\varepsilon$, the "second-order effect" of prediction error drifting OOD is not built into the probabilistic bound.

## Related Work & Insights
- **vs. Traditional Shielding (alshiekh2018safe)**: They rely on formal logic or known dynamics. This paper embeds a differentiable dynamics predictor into the shield, using Monte Carlo-style trajectory prediction instead of formal verification to adapt to unknown dynamics.
- **vs. Adaptive Conformal Prediction Shielding (sheng2024)**: Those works use adjustable confidence bands but do not extend formal guarantees to distribution shifts. This paper explicitly ties the margin between shielding and failure thresholds to the prediction error bound for OOD scenarios.
- **vs. Safe Meta-RL (khattar2023a / xu2025efficient)**: Meta-RL adapts via parameter fine-tuning on new tasks, but fine-tuning requires interaction that may violate safety. This paper follows "test-time verification + zero parameter updates," avoiding ethical hurdles in clinical settings.

## Rating
- Novelty: ⭐⭐⭐⭐ Non-trivial combination of function-space adaptation and probabilistic shielding for Safe RL under distribution shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 72 settings (8 algorithms × 3 types × 3 age groups) fully evaluated.
- Writing Quality: ⭐⭐⭐⭐ Probabilistic bounds are rigorous, though the BA-NODE module concatenation is slightly brief in the main text.
- Value: ⭐⭐⭐⭐ Establishes an engineering paradigm of "training compliance + test-time verification" for safety-critical RL; the OOD safety benchmark is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Partial Action Replacement: Tackling Distribution Shift in Offline MARL](../../AAAI2026/reinforcement_learning/partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)
- [\[ICML 2026\] Safe In-Context Reinforcement Learning](safe_in-context_reinforcement_learning.md)
- [\[ICML 2026\] Safe Reinforcement Learning with Preference-Based Constraint Inference](safe_reinforcement_learning_with_preference-based_constraint_inference.md)
- [\[ICML 2026\] Learning to Search and Searching to Learn for Generalization in Planning](learning_to_search_and_searching_to_learn_for_generalization_in_planning.md)
- [\[ICML 2026\] DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning](darts_distribution-aware_active_rollout_trajectory_shaping_for_accelerating_llm_.md)

</div>

<!-- RELATED:END -->
