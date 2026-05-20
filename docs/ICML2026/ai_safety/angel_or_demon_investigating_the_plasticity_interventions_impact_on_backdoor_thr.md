---
title: >-
  [Paper Note] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning
description: >-
  [ICML 2026][AI Safety][DRL backdoor] The authors systematically evaluate, for the first time, the impact of 7 mainstream plasticity interventions (SAM/Shrink&Perturb/Weight Clip/SN/WD/LN/ReDo) on deep reinforcement learn…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "DRL backdoor"
  - "plasticity intervention"
  - "SAM"
  - "loss landscape sharpness"
  - "robust backdoor injection"
date: 2026-05-08
content_hash: 3805bd3597bf7e08
---

# Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.14587](https://arxiv.org/abs/2605.14587)  
**Code**: <https://github.com/maoubo/Plasticity>  
**Area**: AI Security / Deep RL Backdoor Attacks / Plasticity Intervention  
**Keywords**: DRL backdoor, plasticity intervention, SAM, loss landscape sharpness, robust backdoor injection

## TL;DR
The authors systematically evaluate, for the first time, the impact of 7 mainstream plasticity interventions (SAM/Shrink&Perturb/Weight Clip/SN/WD/LN/ReDo) on deep reinforcement learning (DRL) backdoor attacks (14,664 experiments), finding that only SAM is a "demon"—significantly exacerbating backdoor threats. Based on this, they propose the "Sweeper-Converter-Connector" robust backdoor injection framework and provide a detection signal based on loss landscape sharpness.

## Background & Motivation

**Background**: DRL is widely used in robotics control, UAV navigation, and autonomous driving, but is vulnerable to backdoor attacks (TrojDRL/BadRL/SleeperNets/UNIDOOR, etc.). Meanwhile, DRL training suffers from "plasticity loss" (non-stationary input + drifting objectives cause agents to lose learning ability), so modern DRL pipelines routinely include plasticity interventions: Shrink & Perturb, Weight Clipping, Spectral Normalization, Weight Decay, Layer Normalization, ReDo, SAM, etc.

**Limitations of Prior Work**: (1) Backdoor and plasticity research have long been separate, and no one has systematically asked "do plasticity interventions make backdoors easier or harder?" (2) In real DRL deployment, both techniques are almost always present, but lack of guidance can lead to "thinking LN/SAM improves performance, but actually introducing a security vulnerability."

**Key Challenge**: Plasticity interventions are designed to stabilize training, but do they have side effects on learning the "malicious trigger → target action" mapping? If some interventions actually help backdoors become more stable and potent, they inadvertently become "attack amplifiers."

**Goal**: (1) Quantify the effect of each intervention on ASR (attack success rate) and BTP (benign task performance) under two threat models (TM-Scratch: injected during training / TM-Post: injected post-training); (2) Identify the underlying mechanisms; (3) Design a more robust backdoor injection framework and propose a detection signal based on these mechanisms.

**Key Insight**: Directly repurpose three mature pathological metrics from the plasticity field—**weight magnitude / effective rank / loss landscape sharpness**—as a diagnostic dashboard for backdoor properties, ranking backdoored agents under each intervention.

**Core Idea**: Use large-scale (14,664 cases) controlled experiments + three pathological metrics to decompose "intervention effects" into three mechanisms (M1 activation pathway disturbance / M2 representation space compression / M3 backdoor gradient amplification), then use these mechanisms to reverse-engineer robust attack and detection strategies.

## Method

### Overall Architecture
This is a "systematic empirical + mechanistic analysis + derived design" three-stage work: (1) **Empirical RQ1**—construct 2 (TM-Scratch/Post) × 8 (interventions) × 47 (backdoor tasks) × 4 (attack algorithms) × 3 (seed) = 9,024 cases, plus 5,640 cases for intervention combinations, totaling 14,664 cases, to map ASR/BTP spectra; (2) **Mechanistic RQ2**—for each intervention, measure three pathological metrics (weight magnitude / effective rank / loss landscape sharpness) on backdoored agents, forming an 8×3 pathological vector $\mathbf{v}(p_i)$ and ranking them; (3) **Design RQ3**—design the SCC injection framework and sharpness-based detection based on mechanisms; quantify synergy of intervention combinations using Pathological Distance.

### Key Designs

1. **Empirical Decomposition of Three Pathological Mechanisms (M1 / M2 / M3)**:

    - Function: Reduce the complex "intervention ASR ±x%" phenomena to three interpretable internal mechanisms.
    - Mechanism: (M1) **Activation Pathway Disturbance**—Shrink&Perturb / Weight Clipping / ReDo prune or reset weights, making "backdoor pathway" and "benign pathway" compete for resources; Fig.6 shows backdoor attacks cause a few weights in the actor net's second layer to surge (sparse backdoor pathway), Weight Clip suppresses them, leading to renewed competition. (M2) **Representation Space Compression**—Spectral Norm / Weight Decay / Layer Norm restrict the Lipschitz constant or smooth activations, aligning backdoor gradients (originally nearly orthogonal to benign gradients, dot product ≈ 0) to almost fully aligned (≈1.0), turning the backdoor from a sparse single pathway to a multi-pathway shared with benign, making it less stable under non-stationary training. (M3) **Backdoor Gradient Amplification**—SAM captures sharp loss directions via adversarial perturbation, which coincides with the backdoor direction (backdoor samples expand loss landscape sharpness by over 6×); SAM amplifies these gradients and pushes the backdoor pathway to flat minima, making it robust to parameter perturbations.
    - Design Motivation: ASR/BTP numbers alone cannot explain counterintuitive results like "why does SAM have the opposite effect"; pathological diagnosis links statistics to concrete network changes, making conclusions generalizable (not just hyperparameter-specific).

2. **SCC Robust Backdoor Injection Framework (Sweeper-Converter-Connector)**:

    - Function: Reverse-engineer "which interventions benefit attacks" into an attack design cookbook, constructing robust backdoors under TM-Post.
    - Mechanism: Observing that combined interventions (Plastic/SLac/SSW) outperform SAM alone (ASR 0.178→0.418, BTP 0.745→0.915), the injection process is distilled into three steps—(a) **Sweeper**: Use Shrink&Perturb / Weight Clip / ReDo to clear part of the benign pathway, making room for the backdoor (leveraging M1); (b) **Converter**: Use Spectral Norm / Weight Decay / LN to align backdoor gradients with benign gradients, turning the backdoor into a multi-pathway structure (leveraging M2); (c) **Connector**: Use SAM to jointly optimize multi-pathways to flat minima, stabilizing representations (leveraging M3). Pathological Distance $PD(A)=\sum_{i<j}\|\mathbf{v}(p_i)-\mathbf{v}(p_j)\|_2$ measures pathological differences among interventions; experiments confirm that higher $PD$ (e.g. SSW=18.64) correlates with stronger backdoor threats.
    - Design Motivation: Real deployments commonly use multiple interventions (Plastic/Swiss Cheese, etc.), so attackers can simply follow the SCC template to select complementary interventions for free attack amplification; this provides a concrete threat model for "plasticity-aware security assessment."

3. **Loss Landscape Sharpness-Based Backdoor Detection Signal**:

    - Function: Transform the most prominent external manifestation of backdoors (abnormal loss sharpness) into a monitorable defense-side metric.
    - Mechanism: Backdoor attacks expand the fluctuation range of loss landscape sharpness by 635.22%, the most significant among the three pathologies; all interventions except SAM further exacerbate this anomaly ($v_{i3}>v_{13}$). Defenders can monitor sharpness time series throughout agent training; significant spikes or drops are suspicious. With task-adaptive thresholds and multi-source noise decoupling, this can serve as a general DRL backdoor warning.
    - Design Motivation: Existing DRL backdoor detection mostly relies on triggers or special probes; the proposed sharpness signal requires no trigger knowledge and is compatible with any DRL training process. Drawbacks are large baseline differences in sharpness across tasks and potential false positives from other anomalies—both are acknowledged as open problems.

### Loss & Training
No new loss is proposed; the focus is on evaluation protocol design—attacks use transition tampering to inject triggers into (state, action, reward) tuples, with backdoor reward for reinforcement; no defense-side interventions are applied (only side effects of interventions are studied). Tasks cover OpenAI Gym's 4 classic control + 2 physical control + PyBullet's 3 robotics, including discrete/continuous actions, sparse/dense rewards, cold/non-cold starts; 4 attack types (TrojDRL/BadRL/SleeperNets/UNIDOOR), 47 backdoor tasks, single/multi-backdoor. Hyperparameters for each intervention follow their respective original papers.

## Key Experimental Results

### Main Results
In the TM-Post scenario (interventions have greater impact on pre-trained agents), representative ASR / BTP changes for robotics control tasks:

| Intervention | ASR (Robotics) | BTP (Robotics) | Main Pathological Effect |
|--------------|---------------|---------------|-------------------------|
| None (baseline) | 0.178 ± 0.157 | 0.745 ± 0.230 | — |
| Weight Clipping | ↓ 17.46% | ↓ 20.19% | M1 pathway disturbance |
| Spectral Norm | ↓ 11.78% | ↓ moderate | M2 representation compression |
| Layer Norm | ↓ moderate | ↓ 11.93% | M2 representation compression |
| Weight Decay | ↓ mild | ↓ mild | M2 representation compression |
| Shrink & Perturb | ↓ mild | ↓ mild | M1 soft disturbance |
| ReDo | ↓ mild | ↓ mild | M1 neuron reset |
| **SAM** | **↑ 0.326 (+83%)** | **↑ 0.814 (+9%)** | **M3 gradient amplification** |

Comparison of intervention combinations (robotics control + SAM series):

| Combination | Includes SAM? | ASR | BTP | Pathological Distance |
|-------------|--------------|------|------|----------------------|
| None | — | 0.178 ± 0.157 | 0.745 ± 0.230 | N/A |
| Plastic | ✓ | 0.368 ± 0.144 | 0.724 ± 0.362 | 9.43 |
| SLac | ✓ | 0.417 ± 0.146 | 0.816 ± 0.276 | 17.42 |
| **SSW** | ✓ | **0.418 ± 0.092** | **0.915 ± 0.131** | **18.64** |
| Swiss Cheese (WD+LN) | ✗ | ≈ LN alone | ≈ LN alone | 0.52 |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|---------------|------------|----------------|
| TM-Scratch (injection during training) | ASR only slightly changes (LN max -8.84%) | Representations not yet stable, intervention effects diluted by training dynamics |
| TM-Post (post-training injection) | Significant ASR/BTP changes | Only stabilized models reveal intervention effects |
| Backdoor vs. normal training (Fig.4) | weight magnitude range +98.63%, effective rank +19.16%, sharpness +635.22% | sharpness is the strongest external backdoor marker |
| Single vs. multi-intervention combination | Higher $PD$ yields stronger attacks; same-mechanism combo (Swiss Cheese) has almost no gain | Only complementary mechanisms jointly amplify threats |
| Spectral Norm gradient alignment analysis (Fig.7) | Backdoor-benign gradient rises from ≈0 to ≈1.00 | Confirms M2 representation compression → pathway sharing |
| Weight Clipping 3D weight visualization (App. Fig.13) | TM-Scratch minor effect, TM-Post strong | Parameter flexibility is key variable |

### Key Findings
- **Counterintuitive**: SAM (intended to stabilize training) is the only intervention that exacerbates backdoors, as it is sensitive to and amplifies sharp loss directions caused by backdoors.
- **TM-Post is more sensitive than TM-Scratch**: Injecting into a converged benign representation requires "squeezing out space" for the backdoor, making intervention-imposed parameter flexibility constraints more impactful.
- **BTP is more sensitive than ASR**: Benign representations are complex and require many cooperating parameters; once disrupted by interventions, they are hard to recover. Backdoor representations are sparse and local, easily rebuilt.
- **Intervention combinations are non-additive**: Same-mechanism combos (Swiss Cheese = WD+LN, $PD$=0.52) have almost no cumulative effect; cross-mechanism combos (SSW, $PD$=18.64) significantly amplify attacks—SCC's Pathological Distance is an effective design metric.
- **Among the three pathologies, sharpness is most valuable for detection**: Backdoor attacks expand sharpness fluctuation range by 6×; all interventions (except SAM) further exacerbate this anomaly, which can be exploited by defenders.

## Highlights & Insights
- **Large-scale controlled experimental design**: 14,664 cases covering the Cartesian product of 2 threat models × 8 interventions × 5 combinations × 4 attacks × 9 tasks × multiple seeds; such scale is rare in DRL security literature, lending high credibility to conclusions.
- **Cross-domain bridging**: Connects "plasticity" and "backdoor security" subcommunities via three pathological metrics—a rare "cross-subfield diagnosis" effort; offers methodological inspiration for other security areas (fairness, privacy).
- **"Role reuse" thinking**: SAM is lauded in defense literature for improving generalization, but this work reveals it is also an attack amplifier, reminding that any generalization tool can be a double-edged sword.
- **From mechanism to design**: The SCC triangle (Sweeper-Converter-Connector) translates diagnostic results directly into an attack design cookbook, providing PD as a quantifiable synergy metric—this "mechanism→process→metric" trio is a reusable pattern.
- **Feasibility of sharpness detection**: Since sharpness is already a routine optimizer monitoring metric, deployment cost is extremely low; this is an underrated free defense signal.

## Limitations & Future Work
- Experiments focus on low-dimensional control tasks (Gym/PyBullet); applicability to high-dimensional pixel-based tasks (Atari/StarCraft) is unknown.
- The SCC framework is only conceptually designed, not formally implemented or compared with real "unified injection algorithms"—readers must assemble it themselves.
- Sharpness-based detection faces two major challenges (as acknowledged): (1) Large baseline variance in sharpness across tasks, making unified thresholds difficult; (2) Other training anomalies (reward hacking, unstable critic) may also cause abnormal sharpness.
- Although intervention hyperparameter sensitivity is ablated in App.E, only "trend consistency" is verified, not the worst-case combinations.
- The "combination intervention amplifies attack" conclusion relies on five existing combos (Plastic/Swiss Cheese/Lac/SLac/SSW), without systematic combination search; theoretically, even stronger combos may exist.
- No defense strategies are proposed (except sharpness detection); a closed-loop security solution is still lacking.

## Related Work & Insights
- **vs TrojDRL/BadRL/SleeperNets/UNIDOOR**: These works only study attacks on vanilla DRL; this work evaluates with "modern DRL pipeline defaults," revealing compound effects of attacks/interventions.
- **vs Klein et al. 2024 (plasticity survey)**: Provides four categories of intervention motivations; this work adopts their framework but shifts perspective from "plasticity preservation" to "security side effects."
- **vs Lee et al. 2023 (SAM for DRL)**: Treats SAM as a plasticity-preserving panacea; this work effectively adds a "security warning label" to SAM.
- **vs deep learning backdoor defenses (Li et al. 2024b)**: DL backdoors can be mitigated by finetune-pruning; this work shows that "pruning" interventions (Weight Clip) in DRL have similar effects but at the cost of BTP, so the trade-off remains open.
- **vs Lyle et al. 2024 (Swiss Cheese / multi-intervention plasticity)**: They advocate combined interventions for better generalization; this work shows the same combos may become "vulnerability amplifiers" from a backdoor perspective, leaving "plasticity-aware security assessment" as an open problem.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic characterization of the interaction between plasticity interventions and DRL backdoors, proposing SCC + sharpness detection as two new approaches.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Cartesian evaluation of 14,664 cases, with three pathological diagnostic analyses, providing robust evidence.
- Writing Quality: ⭐⭐⭐⭐ RQ-driven structure, clear concept naming (M1/M2/M3 → SCC), well-integrated formulas and figures.
- Value: ⭐⭐⭐⭐⭐ Directly impacts security practices for all DRL systems with plasticity interventions, providing actionable signals for both attackers and defenders.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](../../ICML2025/medical_imaging/network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)
- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)
- [\[AAAI 2026\] Personalization of Large Foundation Models for Health Interventions](../../AAAI2026/medical_imaging/personalization_of_large_foundation_models_for_health_interventions.md)
- [\[ACL 2025\] ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking](../../ACL2025/medical_imaging/learning_from_negative_samples_in_biomedical_generative_entity_linking.md)
- [\[AAAI 2026\] Investigating Data Pruning for Pretraining Biological Foundation Models at Scale](../../AAAI2026/medical_imaging/investigating_data_pruning_for_pretraining_biological_foundation_models_at_scale.md)

</div>

<!-- RELATED:END -->
