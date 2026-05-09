---
title: >-
  [Paper Note] All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference
description: >-
  [CVPR 2026][AI Safety][collaborative perception] This paper proposes the Pseudo-Random Bayesian Inference (PRBI) framework for collaborative perception scenarios where **all vehicles are untrusted**. By leveraging inter-frame temporal consistency as a self-referential signal, PRBI employs pseudo-random grouping combined with Bayesian inference to efficiently identify and exclude malicious vehicles at an average cost of only 2.5 validations per frame, recovering detection accuracy to 79.4%–86.9% of the pre-attack baseline.
tags:
  - CVPR 2026
  - AI Safety
  - collaborative perception
  - adversarial defense
  - Bayesian inference
  - autonomous driving
  - V2V communication
date: 2026-05-08
content_hash: 656279992791642a
---

# All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference

**Conference**: CVPR 2026
**arXiv**: [2603.08498](https://arxiv.org/abs/2603.08498)
**Code**: To be confirmed
**Area**: AI Security
**Keywords**: collaborative perception, adversarial defense, Bayesian inference, autonomous driving, V2V communication

## TL;DR

This paper proposes the Pseudo-Random Bayesian Inference (PRBI) framework for collaborative perception scenarios where **all vehicles are untrusted**. By leveraging inter-frame temporal consistency as a self-referential signal, PRBI employs pseudo-random grouping combined with Bayesian inference to efficiently identify and exclude malicious vehicles at an average cost of only 2.5 validations per frame, recovering detection accuracy to 79.4%–86.9% of the pre-attack baseline.

---

## Background & Motivation

**Security vulnerabilities in Collaborative Perception (CP)**: Multiple vehicles share feature maps via V2V communication to extend perception range, but the feature fusion mechanism is inherently exposed to adversarial attacks — malicious vehicles can inject perturbations into shared features, causing severe misperception in the ego vehicle.

**Existing methods rely on the "trusted ego" assumption**: Sampling-based defenses (ROBOSAC, PASAC) use the ego vehicle's perception as a reliable reference for consistency verification; classifier-based defenses train binary networks to distinguish benign from malicious features — both assume the ego vehicle itself is not under attack.

**The ego vehicle can also be attacked in practice**: Through LiDAR injection attacks or data interception attacks, adversaries can interfere with the ego vehicle's feature maps without directly compromising its internal systems. Thus, "All Vehicles Can Lie" reflects the realistic threat model.

**Linear growth in verification overhead**: The per-frame verification cost of existing sampling and classifier methods scales linearly with the total number of vehicles, making them impractical for large-scale real-time collaborative perception.

**Detection latency from random sampling**: Purely randomized sampling may require many frames to converge to the complete set of attackers, resulting in prolonged risk exposure.

**Need for zero-trust, low-overhead defense**: An ideal solution should require no trust assumption about any vehicle, no prior knowledge of the number or proportion of attackers, and should maintain a constant per-frame verification cost.

---

## Method

### Overall Architecture

The core pipeline of PRBI operates in a four-step cycle:

1. **Initialization**: At $t=0$, the initial frame perception is assumed correct; normal detection counts $\mathbf{c}_{\text{normal}}$ and abnormal detection counts $\mathbf{c}_{\text{abnormal}}$ are initialized for each vehicle.
2. **Pseudo-random grouping + consistency verification (Soft Sampling)**: Each frame, all vehicles are divided into two groups; each group is compared against the previous frame's benign perception result using Jaccard similarity to update the counts.
3. **Attacker evaluation**: The empirical normal ratio $\eta$ is used to estimate the number of attackers $m$, and Bayesian inference is applied to compute the benign probability $P_{\text{benign}}[j]$ for each vehicle, identifying the $m$ most suspicious vehicles.
4. **Hypothesis testing + defensive perception**: A T-test determines whether $m$ has converged; if convergence is confirmed, the attacker set is output; otherwise, vehicles with nonzero benign probability are used for the current frame's collaborative perception and the reference is updated.

### Key Design 1: Inter-Frame Temporal Consistency as a Self-Referential Signal

- **Function**: The verified benign perception output $D_{\text{ref}}$ from the previous frame serves as the detection reference for the current frame, entirely replacing the "trusted ego" assumption.
- **Mechanism**: Under benign conditions, the Jaccard similarity between adjacent frames remains stable at approximately 0.8, whereas under adversarial conditions it drops sharply below 0.3. This pronounced distributional difference allows inter-frame temporal consistency to be converted into a self-supervised defense signal.
- **Design Motivation**: LiDAR perception outputs exhibit natural spatial continuity, with smooth inter-frame transitions under normal conditions — providing the only vehicle-agnostic reference anchor available in a zero-trust environment.

### Key Design 2: Pseudo-Random Binary Grouping Strategy

- **Function**: Only 2 additional validations are performed per frame — vehicles are sorted by suspicion level and the $\lfloor m \rfloor$ most suspicious vehicles are placed in one group, with the remainder in the other.
- **Mechanism**: Binary grouping statistically approximates a sampling-without-replacement process. The probability of sampling an all-benign group is $P_{ideal}' = 2^{n-k}/2^n = 2^{-k}$, depending solely on the number of attackers $k$. From the empirical normal ratio $\eta \approx 2^{-k}$, the attacker count can be back-calculated as $m = \log_2(1/\eta)$.
- **Design Motivation**: This reduces the verification overhead from $O(n)$ to a constant (2 per frame), fully decoupling the number of validations from the total number of vehicles. The pseudo-random grouping based on $m$ and $P_{\text{benign}}$ guarantees monotonic convergence of $m$ to the true value $k$ (Theorem 1).

### Key Design 3: Bayesian Probabilistic Inference for Attacker Identification

- **Function**: For each vehicle $j$, the posterior benign probability $P_{\text{benign}}[j] = P(\mathcal{B}_j | \mathcal{A})$ is computed, and the $\lfloor m \rceil$ vehicles with the lowest probability are selected as suspected attackers.
- **Mechanism**: The prior $P(\mathcal{B}_j)$ combines short-term (previous frame Bayesian result) and long-term (historical normal detection ratio) components with weighted fusion, incorporating temporal memory to mitigate instability in early grouping stages. The likelihood $P(\mathcal{A}|\mathcal{B}_j)$ is estimated via the system's anomaly ratio after excluding vehicle $j$.
- **Design Motivation**: Malicious vehicles necessarily appear in every anomalous detection event ($\beta_j = 0$), so their benign probability remains persistently at 0, guaranteeing eventual exclusion.

### Key Design 4: T-Test Convergence Criterion

- **Function**: A window $W$ stores the estimated $m$ values from the most recent $w_p$ frames; the null hypothesis $H_0: k = m$ is tested, and convergence is declared when the T-test passes and the number of zero-probability vehicles equals exactly $m$.
- **Mechanism**: $m$ fluctuates slightly around $k$; when the fluctuation remains within the confidence interval for an extended period, convergence is confirmed. The dual condition prevents premature acceptance of $H_0$ during slow convergence of $m$.
- **Design Motivation**: In continuous scenarios, earlier convergence confirmation enables earlier termination of redundant validations and output of defensive results; empirically, convergence is achieved in approximately 4 frames on average.

---

## Loss & Training

PRBI is an **inference-stage defense framework** that involves no additional training or loss functions. It operates on top of pre-trained collaborative perception models, directly utilizing Jaccard similarity between detection outputs for consistency verification. The adversarial perturbation optimization objective follows the standard multi-agent detection loss:

$$\max_{\|\delta\| \leq \Delta} \sum_{j=1}^{L} \mathcal{L}_{\text{det}}(d_j, d_j')$$

where $\Delta$ constrains the perturbation magnitude and $\mathcal{L}_{\text{det}}$ denotes the detection loss. The defense distinguishes normal from anomalous frames via the Jaccard threshold $\epsilon$, requiring no fine-tuning of the perception model.

---

## Key Experimental Results

### Table 1: Per-Frame Verification Count Comparison ($n=5$)

| Method | Metric | Attack ratio 20% | 40% | 60% | 80% | Average ↓ |
|:-----|:-----|:---:|:---:|:---:|:---:|:---:|
| ROBOSAC | Avg | 4.89 | 10.36 | 8.29 | 4.73 | 7.1 |
| PASAC | Avg | 4.79 | 6.60 | 7.59 | 8.00 | 6.7 |
| **PRBI (Ours)** | **Avg** | **2.00** | **2.35** | **2.61** | **2.86** | **2.5** |

- PRBI averages only 2.5 validations per frame, significantly lower than ROBOSAC (7.1) and PASAC (6.7).
- ROBOSAC reaches a maximum of 30.3 validations per frame; PRBI peaks at only 5.0.

### Table 2: Detection Performance Comparison ($n=5, k=2$, V2VNet backbone + three attack types)

| Setting | AP@0.5 | AP@0.7 |
|:-----|:---:|:---:|
| Upper-Bound (attack-free collaborative) | 80.73 | 78.35 |
| Attack w/ PGD | 17.02 | 14.53 |
| PRBI against PGD | **68.93** (+51.91) | **63.82** (+49.29) |
| Attack w/ BIM | 13.51 | 11.69 |
| PRBI against BIM | **68.76** (+55.25) | **64.88** (+53.19) |
| Attack w/ C&W | 10.68 | 6.04 |
| PRBI against C&W | **71.87** (+61.19) | **68.54** (+62.50) |
| Lower-Bound (single-vehicle perception) | 56.35 | 52.89 |
| ROBOSAC | 64.13 (+7.78) | 61.01 (+8.12) |
| PASAC | 68.39 (+12.04) | 64.73 (+11.83) |

- On V2VNet, PRBI recovers 86.9% of the AP loss incurred under C&W attack, substantially outperforming ROBOSAC and PASAC.
- Consistent robustness is maintained across multiple fusion strategies (Mean/Max/Sum/V2VNet/DiscoNet).

### Convergence and Identification Rate

| Attack ratio | Avg. convergence frames | Malicious vehicle identification rate | Benign vehicle false positive rate |
|:---:|:---:|:---:|:---:|
| 20% | 2.25 | 100% | 0% |
| 40% | 2.77 | 100% | 6% |
| 60% | 3.36 | 100% | 0% |
| 80% | 4.27 | 100% | 0% |

- The malicious identification rate is 100% across all attack ratios, with convergence achieved in approximately 4 frames on average.

---

## Highlights & Insights

1. **First efficient defense for the fully untrusted setting**: The framework eliminates reliance on a trusted ego vehicle by introducing inter-frame temporal consistency as a self-referential signal — a conceptually simple yet highly insightful approach.
2. **Decoupling verification overhead from vehicle count**: The binary grouping strategy reduces per-frame validations from $O(n)$ to a constant of 2, achieving theoretical optimality.
3. **Rigorous theoretical guarantees**: Theorem 1 proves the monotonic convergence of $m$ to $k$; Theorem 2 proves that floor rounding guarantees exact convergence — the theoretical analysis is thorough.
4. **Invariant guaranteeing attacker exclusion**: Malicious vehicles satisfy $\beta_j = 0$, keeping their benign probability permanently at 0 and ensuring no missed detections.
5. **Strong practical utility**: No additional training, no prior knowledge, and no modification of the perception model are required; PRBI operates as a plug-and-play inference-stage defense module.

---

## Limitations & Future Work

1. **Fragile scenarios for inter-frame consistency assumption**: Under extreme vehicle motion such as sharp turns or sudden braking, inter-frame perception changes can be large, causing natural drops in Jaccard similarity that may trigger false positives.
2. **Initial frame correctness assumption**: At $t=0$, the perception is assumed fully correct; if the system is under attack from the very first frame, a reliable reference cannot be established.
3. **Evaluation limited to small-scale $n=5$ scenarios**: Real-world urban autonomous driving may involve tens of collaborating vehicles; convergence speed and stability at larger scales remain to be validated.
4. **Static attacker set assumption**: The framework assumes the set of attackers remains fixed throughout the sequence; its adaptability to dynamically joining or leaving attackers is uncertain.
5. **6% false positive rate at 40% attack ratio**: Premature stabilization of $m$ may cause benign vehicles to be incorrectly excluded; while this does not affect malicious identification rate, it reduces collaborative gain.
6. **Adaptive attacks not considered**: An adversary aware of PRBI's detection logic could design slowly varying perturbations to maintain inter-frame similarity above the threshold, potentially evading detection.

---

## Related Work & Insights

- **ROBOSAC / PASAC**: Classical sampling-based defenses that use the ego vehicle as a reference for iterative consistency verification. PRBI's key improvement is replacing ego trust with inter-frame temporal consistency and replacing linear sampling with binary grouping.
- **MATE**: A geometry-based multi-agent trust estimator requiring object tracking and visibility reasoning. PRBI requires no scene geometry modeling and is substantially more lightweight.
- **Classifier-based defenses**: Binary networks trained to detect malicious features suffer from poor generalization and expand the attack surface. PRBI requires no training and introduces zero attack surface expansion.
- **Broader inspiration**: The inter-frame consistency signal is not limited to collaborative perception defense; it can be generalized to anomaly detection in any multi-source fusion system, such as malicious client detection in federated learning. The paradigm of pseudo-random grouping combined with Bayesian inference is broadly applicable.

---

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First constant-overhead defense under the fully untrusted setting; the inter-frame self-referential signal is a novel contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage of multiple attack types, fusion strategies, and parameter sensitivity analyses, though evaluation at only $n=5$ is a mild limitation
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear problem formulation, rigorous theoretical analysis, and complete mathematical derivations
- **Value**: ⭐⭐⭐⭐ — Practically significant for collaborative autonomous driving security; the plug-and-play design facilitates real-world deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](../../AAAI2026/ai_safety/detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[AAAI 2026\] SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute](../../AAAI2026/ai_safety/secmoe_communication-efficient_secure_moe_inference_via_select-then-compute.md)
- [\[CVPR 2026\] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)

</div>

<!-- RELATED:END -->
