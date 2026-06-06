---
title: >-
  [Paper Note] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning
description: >-
  [ICML 2026][AI Safety][DRL backdoor] This work presents the first systematic evaluation of the impact of seven mainstream plasticity interventions (SAM, Shrink&Perturb, Weight Clip, SN, WD, LN…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "DRL backdoor"
  - "plasticity intervention"
  - "SAM"
  - "loss landscape sharpness"
  - "robust backdoor injection"
date: 2026-05-08
content_hash: b064cabb5798e699
---

# Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.14587](https://arxiv.org/abs/2605.14587)  
**Code**: <https://github.com/maoubo/Plasticity>  
**Area**: AI Security / DRL Backdoor Attack / Plasticity Intervention  
**Keywords**: DRL backdoor, plasticity intervention, SAM, loss landscape sharpness, robust backdoor injection

## TL;DR
This work presents the first systematic evaluation of the impact of seven mainstream plasticity interventions (SAM, Shrink&Perturb, Weight Clip, SN, WD, LN, ReDo) on deep reinforcement learning (DRL) backdoor attacks through 14,664 experiments. It identifies SAM as a "demon" that significantly exacerbates backdoor threats. Consequently, the "Sweeper-Converter-Connector" robust backdoor injection framework is proposed, along with a detection signal based on loss landscape sharpness.

## Background & Motivation

**Background**: DRL is widely applied in robotic control, UAV navigation, and autonomous driving, yet it is susceptible to backdoor attacks (e.g., TrojDRL, BadRL, SleeperNets, UNIDOOR). Conversely, DRL suffers from "loss of plasticity" due to non-stationary inputs and drifting optimization targets. Modern DRL pipelines thus incorporate plasticity interventions: Shrink & Perturb, Weight Clipping, Spectral Normalization, Weight Decay, Layer Normalization, ReDo, and SAM.

**Limitations of Prior Work**: (1) Research on backdoors and plasticity has historically developed independently; no systematic study has investigated whether plasticity interventions facilitate or hinder backdoors. (2) Both technologies are often co-present in practical DRL deployments, but a lack of guidance may lead users to adopt interventions like LN or SAM for performance, unaware of potential security vulnerabilities.

**Key Challenge**: Since plasticity interventions are designed to stabilize training, do they have the side effect of facilitating the mapping from "malicious triggers" to "target actions"? If certain interventions inadvertently strengthen backdoors, they act as unintentional "attack amplifiers."

**Goal**: (1) Quantify the impact of each intervention on Attack Success Rate (ASR) and Benign Task Performance (BTP) under two threat models: TM-Scratch (injection during training) and TM-Post (injection into a pre-trained model). (2) Uncover the underlying mechanisms of these effects. (3) Design a robust backdoor injection framework based on these mechanisms and propose detection signals.

**Key Insight**: Three established pathological metrics from the plasticity field—**weight magnitude**, **effective rank**, and **loss landscape sharpness**—are utilized as diagnostic tools to analyze the internal attributes of backdoored agents under different interventions.

**Core Idea**: Through large-scale (14,664 cases) controlled experiments and pathological diagnosis, the intervention effects are decomposed into three mechanisms: M1 (Activation Pathway Perturbation), M2 (Representation Space Compression), and M3 (Backdoor Gradient Amplification). These mechanisms then inform robust attack and detection strategies.

## Method

### Overall Architecture
The work follows a three-stage structure: (1) **Empirical RQ1**: Construction of 14,664 cases (encompassing 2 threat models, 8 interventions, 47 tasks, 4 attack algorithms, and 3 seeds) to map the ASR/BTP spectrum. (2) **Mechanism RQ2**: Analysis of backdoored agents via three pathological metrics to establish and rank intervention pathology vectors $\mathbf{v}(p_i)$. (3) **Design RQ3**: Development of the SCC injection framework and sharpness-based detection based on the identified mechanisms.

### Key Designs

1.  **Empirical Decomposition of Three Pathological Mechanisms (M1 / M2 / M3)**:
    - **Function**: To reduce complex ASR observations into three interpretable internal mechanisms.
    - **Mechanism**: (M1) **Activation Pathway Perturbation**—Interventions like Shrink&Perturb, Weight Clipping, and ReDo reset or clip weights, causing "backdoor pathways" and "benign pathways" to compete for resources. (M2) **Representation Space Compression**—Spectral Norm, Weight Decay, and Layer Norm limit Lipschitz constants or smooth activations, aligning backdoor gradients (originally near-orthogonal, dot product $\approx 0$) with benign gradients ($\approx 1.0$), forcing backdoors to share multi-pathway structures with benign tasks. (M3) **Backdoor Gradient Amplification**—SAM captures sharp loss directions via adversarial perturbations, which correspond to backdoor directions. SAM amplifies these gradients and guides the backdoor pathway toward a flat minimum, enhancing robustness to parameter perturbations.
    - **Design Motivation**: ASR/BTP metrics alone cannot explain counter-intuitive results like SAM's behavior. Pathological diagnosis links statistical results to specific network changes, ensuring the conclusions are generalizable.

2.  **SCC Robust Backdoor Injection Framework (Sweeper-Converter-Connector)**:
    - **Function**: To translate findings about beneficial interventions into a "cookbook" for designing highly robust backdoors under TM-Post.
    - **Mechanism**: Observing that combined interventions (e.g., SSW) outperform single ones, a three-step process is defined: (a) **Sweeper**: Use M1-type interventions (Shrink&Perturb) to clear benign pathways. (b) **Converter**: Use M2-type interventions (SN/WD/LN) to transition the backdoor from orthogonal to aligned, creating multi-pathway structures. (c) **Connector**: Use SAM (M3) to optimize these pathways toward flat minima for stable co-existence. The Pathological Distance $PD(A)=\sum_{i<j}\|\mathbf{v}(p_i)-\mathbf{v}(p_j)\|_2$ measures the synergy between interventions.
    - **Design Motivation**: Since real-world deployments often combine interventions, attackers can select complementary ones using the SCC template to amplify attack effectiveness.

3.  **Loss Landscape Sharpness-based Backdoor Detection**:
    - **Function**: To convert the most significant pathological manifestation (sharpness anomaly) into a monitorable defense metric.
    - **Mechanism**: Backdoor attacks increase the fluctuation range of loss landscape sharpness by 635.22%. Almost all interventions (except SAM) further exacerbate this anomaly ($v_{i3} > v_{13}$). Defenders can monitor the sharpness time series during training; significant spikes or drops indicate potential backdoor activity.
    - **Design Motivation**: Unlike existing DRL detectors that require triggers or probes, sharpness signals are trigger-agnostic and compatible with standard DRL training workflows.

### Loss & Training
The study utilizes transition tampering to inject triggers into (state, action, reward) triplets and reinforces the backdoor via a specific reward. The evaluation covers 4 classic control, 2 physical control (OpenAI Gym), and 3 robotic (PyBullet) tasks, including discrete/continuous actions and sparse/dense rewards. Four attack algorithms (TrojDRL, BadRL, SleeperNets, UNIDOOR) are tested. Hyperparameters for each intervention follow the original literature.

## Key Experimental Results

### Main Results
Representing ASR / BTP changes in robotic control tasks under TM-Post (where interventions significantly impact pre-trained agents):

| Intervention | ASR (Robot) | BTP (Robot) | Primary Pathological Impact |
| :--- | :--- | :--- | :--- |
| None (baseline) | 0.178 ± 0.157 | 0.745 ± 0.230 | — |
| Weight Clipping | ↓ 17.46% | ↓ 20.19% | M1 Pathway Perturbation |
| Spectral Norm | ↓ 11.78% | ↓ Moderate | M2 Rep. Compression |
| Layer Norm | ↓ Moderate | ↓ 11.93% | M2 Rep. Compression |
| Weight Decay | ↓ Slight | ↓ Slight | M2 Rep. Compression |
| Shrink & Perturb | ↓ Slight | ↓ Slight | M1 Soft Perturbation |
| ReDo | ↓ Slight | ↓ Slight | M1 Neuron Reset |
| **SAM** | **↑ 0.326 (+83%)** | **↑ 0.814 (+9%)** | **M3 Gradient Amplification** |

Comparison of intervention combinations (Robot control + SAM variants):

| Combination | Includes SAM? | ASR | BTP | Pathological Distance |
| :--- | :--- | :--- | :--- | :--- |
| None | — | 0.178 ± 0.157 | 0.745 ± 0.230 | N/A |
| Plastic | ✓ | 0.368 ± 0.144 | 0.724 ± 0.362 | 9.43 |
| SLac | ✓ | 0.417 ± 0.146 | 0.816 ± 0.276 | 17.42 |
| **SSW** | ✓ | **0.418 ± 0.092** | **0.915 ± 0.131** | **18.64** |
| Swiss Cheese | ✗ | $\approx$ LN Alone | $\approx$ LN Alone | 0.52 |

### Ablation Study

| Configuration | Observation | Interpretation |
| :--- | :--- | :--- |
| TM-Scratch | Minor ASR changes (LN max -8.84%) | Training dynamics dilute intervention effects. |
| TM-Post | Significant ASR/BTP changes | Effects manifest only after representations stabilize. |
| Backdoor vs. Normal | Sharpness range +635.22% | Sharpness is the strongest external sign of a backdoor. |
| Single vs. Combined | Higher $PD$ leads to stronger attacks | Complementary mechanisms are required for synergy. |
| SN Gradient Alignment | Backdoor-Benign alignment $0 \to 1.0$ | Validates M2 mechanism (pathway sharing). |

### Key Findings
- **Counter-intuitive**: SAM, intended to stabilize training, is the only intervention that exacerbates backdoors because it amplifies and flattens the sharp loss directions introduced by the backdoor.
- **TM-Post is more sensitive than TM-Scratch**: Injecting backdoors into converged representations requires "squeezing out" space, magnifying the constraints interventions place on parameter flexibility.
- **BTP is more sensitive than ASR**: Benign representations are complex/coordinated and hard to rebuild after disruption; backdoor pathways are sparse and easier to recover.
- **Intervention combinations are non-additive**: Combinations using the same mechanism (e.g., Swiss Cheese) provide no gain, while heterogeneous combinations (SSW) significantly amplify threats.

## Highlights & Insights
- **Scale of Evaluation**: 14,664 cases provide a robust statistical foundation rarely seen in DRL security literature.
- **Cross-Domain Bridge**: Linking "plasticity" and "backdoor security" via pathological metrics provides a methodological template for other security domains (e.g., fairness, privacy).
- **"Dual-Role" Awareness**: Reveals that generalization tools like SAM can act as unintended security vulnerabilities.
- **Mechanistic to Practical**: The SCC framework translates diagnostic insights into an actionable attack design cookbook.

## Limitations & Future Work
- Tasks are primarily low-dimensional state space controls; generalizability to pixel-based observations (Atari) is unverified.
- The SCC framework is a conceptual design; a unified implementation for automated optimization is lacking.
- Sharpness-based detection faces challenges with high baseline variance across tasks and potential false positives from other training instabilities.
- Specific worst-case hyperparameter combinations for interventions were not exhaustively searched.

## Related Work & Insights
- **vs. TrojDRL/BadRL**: These focus on vanilla DRL; this work introduces standard modern pipeline interventions to reveal composite effects.
- **vs. Lee et al. 2023 (SAM for DRL)**: SAM was promoted as a cure for plasticity loss; this work attaches a "security warning label" to its use.
- **vs. Lyle et al. 2024 (Swiss Cheese)**: Advocates for combined interventions to improve generalization; this work shows these same combinations can act as "vulnerability amplifiers."

## Rating
- Novelty: ⭐⭐⭐⭐ (First to characterize plasticity-backdoor interaction).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (massive 14k+ case study).
- Writing Quality: ⭐⭐⭐⭐ (Clear RQ-driven structure).
- Value: ⭐⭐⭐⭐⭐ (Directly impacts safety practices in modern DRL deployment).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](../../ICLR2026/ai_safety/beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](../../NeurIPS2025/ai_safety/impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[ICML 2026\] Regret-Based Federated Causal Discovery with Unknown Interventions](regret-based_federated_causal_discovery_with_unknown_interventions.md)
- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](../../CVPR2026/ai_safety/tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)

</div>

<!-- RELATED:END -->
