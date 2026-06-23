---
title: >-
  [Paper Note] Learning Human Habits with Rule-Guided Active Inference
description: >-
  [ICLR 2026][Reinforcement Learning][Active Inference] This work extends active inference (AIF) into a "habit-forming" framework: it employs a bio-inspired wake–sleep algorithm to jointly learn world models and symbolic rules under a unified free energy objective. This allows agents to react instantaneously using high-confidence rules in familiar contexts while falling bac
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Active Inference
  - Free Energy
  - Habit Learning
  - Symbolic Rules
  - Wake-Sleep
  - Neuro-Symbolic
date: 2026-05-08
content_hash: c2bdd034e2db90b7
---
# Learning Human Habits with Rule-Guided Active Inference

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FZXwkBH6s7](https://openreview.net/forum?id=FZXwkBH6s7)  
**Code**: [https://github.com/GongZhiren/human-action-active-inference](https://github.com/GongZhiren/human-action-active-inference)  
**Area**: Active Inference / Sequential Decision Making / Neuro-Symbolic / Human Behavior Modeling  
**Keywords**: Active Inference, Free Energy, Habit Learning, Symbolic Rules, Wake-Sleep, Neuro-Symbolic  

## TL;DR
This work extends active inference (AIF) into a "habit-forming" framework: it employs a bio-inspired wake–sleep algorithm to jointly learn world models and symbolic rules under a unified free energy objective. This allows agents to react instantaneously using high-confidence rules in familiar contexts while falling back to EFE planning in novel scenarios, resulting in more accurate and faster human behavior prediction and interpretable "habits."

## Background & Motivation
**Background**: Human decision-making comprises two complementary systems: goal-oriented deliberative planning for novel situations (based on simulating consequences via a world model) and stimulus-response habit shortcuts for familiar situations (bypassing deliberation for rapid action). Active Inference (AIF) conceptualizes the brain as a predictive machine minimizing free energy, performing perceptual inference via Variational Free Energy (VFE) and prospective planning via Expected Free Energy (EFE), providing an elegant, biologically plausible framework unifying perception, learning, and action.

**Limitations of Prior Work**: Classical AIF operationalizes behavior almost entirely as "step-by-step prospective planning," lacking three key elements of human behavior: (i) a mechanism to compress repeated successes into compact, reusable rules with confidence levels; (ii) a principled mode-switching mechanism (using immediate rules in familiar contexts and only investing in expensive look-ahead under high uncertainty); and (iii) an offline process to consolidate, prune, and semantically anchor these rules.

**Key Challenge**: Multi-step EFE rollout in AIF becomes exponentially expensive as the planning horizon $H$ and action space increase. Repeated expensive planning in familiar contexts is neither efficient nor human-like—while pure deep/logic/LLM approaches are either black boxes, rely on static post-hoc rules, or suffer from extreme latency.

**Goal**: To fit and explain human (and human-like) action sequences through the perspective of control-as-inference, enabling the framework to maintain flexible planning while providing instantaneous habitual responses and generating interpretable rules.

**Core Idea (rule-guided AIF + wake–sleep)**: Symbolic rules are directly embedded into the AIF generative process using a bio-inspired wake–sleep cycle. In the **wake phase**, the agent collects state–intention–action triplets that consistently reduce free energy from real experience as candidate rules. In the **sleep phase**, generative replay is used to consolidate, prune, and semantically anchor these rules. Each rule is anchored to latent state prototypes and interpretable discrete intentions, forming neuro-symbolic units that bridge continuous world models and symbolic decision-making.

## Method

### Overall Architecture
The method decomposes the general latent state into $Z_t=(S_t, m_t)$ (continuous external world state $S_t$ + discrete mental state $m_t$). Symbolic "condition → action" rules are defined atop this representation, and a wake–sleep algorithm jointly trains the encoder, decoder (world model), and rules under a unified total free energy objective. During decision-making, a habit shortcut is taken if a rule hits; otherwise, the agent falls back to EFE planning.

```mermaid
flowchart TD
    O[Observation Sequence O_t] --> ENC[Encoder q_ϑ S_t, m_t | H_t]
    ENC --> S[Continuous World State S_t]
    ENC --> M[Discrete Mental State m_t]
    S --> MATCH{Rule Hit?<br/>κ S_t,S*_r ≥ τ_r AND m_t=m*_r}
    M --> MATCH
    MATCH -- Yes --> RULE[Habit Policy: Rule Action a_f<br/>Instantaneous · Interpretable]
    MATCH -- No --> EFE[EFE Planning beam/MCTS<br/>Multi-step rollout]
    RULE --> ACT[Mixed Policy p_ϕπ a_t]
    EFE --> ACT
    ACT --> WAKE[Wake: Real Trajectories<br/>Update Model + Collect New Rules]
    WAKE --> SLEEP[Sleep: Generative Replay<br/>Consolidate/Prune/Tune Confidence]
    SLEEP -.Shared Free Energy Objective.-> ENC
```

### Key Designs

**1. Latent State Bifurcation: Continuous World State + Discrete Mental State—Providing dual anchoring for rules.** The method decomposes the general latent variable $Z_t$ into $Z_t=(S_t, m_t)$: $S_t\in\mathcal{S}$ is a continuous low-dimensional embedding of the external world responsible for precise reconstruction of observations; $m_t\in\{1,\dots,K\}$ is a discrete mental state encoding intentions, modes, or sub-goals (e.g., "cautious/aggressive/energy-saving"). The generative model is rewritten as $p_\phi(O_{1:T},S_{1:T},m_{1:T},a_{1:T})=p_\phi(S_1)p_\phi(m_1)\prod_t p_\phi(O_t|S_t)\,p_\phi(S_t|S_{t-1},a_{t-1})\,p_\phi(m_t|m_{t-1},S_t)\,p_{\phi_\pi}(a_t|S_t,m_t)$. The slower evolution of $m_t$ acts as an "intention bottleneck"—a prerequisite for rules to attach simultaneously to "environmental context" and "internal goals," making rules context-sensitive and intention-driven, echoing cognitive science views on habits.

**2. Anchored Symbolic Rules and Mixed Strategy—Formulating habits as interpretable condition-action units for shortcuts.** Each rule is defined as an anchored condition-action pair $f:(S^\star_f, m^\star_f)\Rightarrow a_f$, where the continuous anchor $S^\star_f$ is an external environment prototype, $m^\star_f$ specifies the intention mode, $a_f$ is the prescribed action, and a confidence level $\rho_f\in[0,1]$ is included. The rule library acts as an amortized mixture model over context–action pairs. Recognition uses MAP estimation: rule $r$ is activated when $\kappa(S^{MAP}_t, S^\star_r)\ge\tau_r$ and $m^{MAP}_t=m^\star_r$, where $\kappa$ is a Gaussian similarity kernel. The final action distribution merges rule priors with EFE planning into a mixed strategy:

$$p_{\phi_\pi}(a_t|S_t,m_t)\propto \pi(a_t|S^{MAP}_t,m^{MAP}_t)+\bigl(1-\mathbb{1}_{\text{rule hit}}\bigr)\exp\bigl(-\tau\,\text{EFE}_t(a_t)\bigr)$$

When a reliable rule hits, its prior dominates, bypassing expensive rollouts; otherwise, it falls back to multi-step EFE minimization for deliberative planning.

**3. Unified Total Free Energy Objective + Wake–Sleep Joint Learning—Collecting rules while awake, consolidating while dreaming.** The generative model $p_\phi$, inference network $q_\vartheta$, and policy parameters $\phi_\pi$ (including rule prototypes) are jointly optimized under a unified total free energy objective:

$$\mathcal{F}_t(\phi,\vartheta,\phi_\pi)=\underbrace{\text{VFE}_t(O_t;\phi,\vartheta)}_{\text{Fitting real data}}+\eta\,\underbrace{\text{EFE}_t(\phi,\phi_\pi)}_{\text{Acting on rollout}}+\gamma\,\underbrace{D_{KL}\!\bigl(q_\vartheta(m_{t-1}|H_{t-1})\,\|\,q_\vartheta(m_t|H_t)\bigr)}_{\text{Consistency of mental states}}$$

The KL term acts as a "sticky prior" for discrete mental states, encouraging slow, interpretable transitions. During the **Wake phase**, $(\phi, \vartheta)$ are updated on real trajectories $\mathcal{D}_{real}$ to minimize free energy while "growing" rules: when triplets $(S^{MAP}_t, m^{MAP}_t, a_t)$ recur with low free energy, new rules are created or nearby rules' confidence is increased. During the **Sleep phase**, $p_\phi$ generates replay trajectories to jointly update $(\phi, \phi_\pi)$, consolidating or pruning rules on imagined data.

## Key Experimental Results

### Main Results
Evaluated across four domains (NBA player trajectories / Car-following / DDXPlus medical diagnosis / Atari-Berzerk), Acc denotes Acc@1/3/5 (%), and Lat/CT denotes Latency (ms)/Convergence Time (h):

| Category | Method | NBA Acc | NBA Lat/CT | Car-Follow Acc | DDXPlus Acc | Berzerk Acc |
|---|---|---|---|---|---|---|
| Logic | RNNLogic | 67.2/60.6/51.8 | 26.9/1.20 | 72.3/68.1/57.6 | 18.8/16.3/13.3 | 33.9/27.5/24.4 |
| Logic | STLR | 75.3/74.7/70.2 | 174/3.35 | 78.9/76.6/75.0 | 22.5/18.3/15.6 | 45.5/38.7/37.2 |
| DeepNN | Re-Net | 72.2/68.5/62.0 | 218/2.34 | 76.3/70.7/67.3 | 27.3/20.2/16.2 | 40.7/32.5/29.3 |
| AIF | DAI | 75.4/70.6/62.3 | 262/1.24 | 78.9/73.4/68.5 | 46.8/39.3/34.2 | 60.0/52.3/41.5 |
| AIF | DAI-MC | 82.3/80.6/76.5 | 387/1.52 | 84.5/82.9/80.3 | 57.2/52.2/43.7 | 66.8/58.2/48.2 |
| LLM | LaTee | 78.5/73.3/64.5 | 1244/4.65 | 82.4/74.8/71.8 | 28.2/22.1/20.4 | 62.2/54.2/49.3 |
| MBRL | DreamerV2 | 86.4/83.6/81.7 | 52.7/1.75 | 88.4/85.4/82.3 | 64.1/61.5/58.2 | 76.3/72.2/69.5 |
| **Ours** | **Ours** | **97.0/91.3/85.7** | 35.9/2.59 | **96.8/95.9/94.2** | **79.6/73.6/68.1** | **85.6/77.2/72.4** |

Ours leads across domains: NBA Acc@1 97.0% (vs. DreamerV2 at 86.4%). Latency in NBA is only 35.9ms due to rule hits (compared to DAI-MC at 386ms).

### Ablation Study
Pareto tradeoff between rule count (RC) and accuracy/latency on NBA dataset (RHR = Rule Hit Rate):

| Rule Count (RC) | RHR | Trend |
|---|---|---|
| 0 | 0% | Pure planning, highest latency |
| 3 | 31.6% | Accuracy increases |
| **6** | **39.9%** | **Optimal Acc@3/Acc@5 point** |
| 64 | 82.9% | High hit rate but accuracy drops |
| 256 | 98.7% | Overfitting trivial rules, accuracy decreases |

### Key Findings
- **Rules accelerate inference; accuracy follows an inverted U-curve with rule count**: Increasing RC monotonically reduces latency, but accuracy peaks at a compact rule set (RC≈6).
- **Significant gains for rare critical actions (HHAR)**: In DDXPlus (225 actions), rule envelopes reliably capture low-frequency but vital diagnostic operations.
- **Healthy training dynamics**: Persistent decreases in $\Delta F$, VFE, EFE, and KL terms indicate synchronized improvements in world model reconstruction and decision quality.

## Highlights & Insights
- **Habits as first-class citizens**: Unlike prior AIF works where habits were ad hoc, this work uses a wake–sleep cycle to provide a principled mechanism for habit acquisition, consolidation, and meta-control.
- **Neuro-symbolic bridging grounded in Free Energy**: Rules are not post-extracted; they are embedded in the generative process and updated dynamically under a unified free energy objective.
- **Engineering dual-process systems**: The implementation of a "fast rules + slow planning" strategy achieves both higher accuracy and lower latency.

## Limitations & Future Work
- **Continuous anchors are not directly readable**: $S^\star_f$ must be decoded back to the observation space via the world model for visualization, making its interpretability indirect.
- **Latency in large action spaces**: In DDXPlus (225 actions), 159ms latency remains higher than low-dimensional domains, suggesting rule triggers mitigate but do not eliminate planning costs.
- **Engineering approximations**: Current M-step updates for the mixture model are engineering approximations of full variational learning.
- **Offline/Demonstration setting**: The method fits human trajectories in an offline replay buffer setting without explicitly recovering reward distributions, leaving online interactive transfer for future work.

## Related Work & Insights
- **Active Inference**: Extends the Friston-style VFE/EFE framework and habit networks (Fountas et al., 2020) by introducing a generative-consolidation closed loop using wake–sleep.
- **Wake–Sleep / Program Synthesis**: Adopts the sleep-phase consolidation ideas from Hinton et al. (1995) and DreamCoder (Ellis et al., 2023) for rule library growth and pruning.
- **Neuro-symbolic Logic**: Bridges the gap with static/post-hoc rule methods (RNNLogic, STLR) by coupling rules with latent states for joint optimization under free energy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ principled embedding of rules + wake-sleep in AIF.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ extensive cross-domain coverage and strong baselines.
- **Writing Quality**: ⭐⭐⭐⭐ clear motivation and well-structured method.
- **Value**: ⭐⭐⭐⭐ strong potential for human behavior modeling and embodied decision-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] Toward Conservative Planning from Human-AI Preferences in Reinforcement Learning](toward_conservative_planning_from_human-ai_preferences_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](../../NeurIPS2025/reinforcement_learning/aline_joint_amortization_for_bayesian_inference_and_active_data_acquisition.md)
- [\[ICLR 2026\] Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions](using_reinforcement_learning_to_train_large_language_models_to_explain_human_dec.md)
- [\[ICLR 2026\] Learning What Matters Now: Dynamic Preference Inference under Contextual Shifts](learning_what_matters_now_dynamic_preference_inference_under_contextual_shifts.md)

</div>

<!-- RELATED:END -->
