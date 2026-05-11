---
title: >-
  [Paper Note] Predict and Resist: Long-Term Accident Anticipation under Sensor Noise
description: >-
  [AAAI 2026][accident anticipation] A unified framework is proposed that integrates a diffusion-based dual-level denoising module with a temporally-aware Actor-Critic reinforcement learning model to enable robust long-ter…
tags:
  - "AAAI 2026"
  - "accident anticipation"
  - "diffusion denoising"
  - "Actor-Critic reinforcement learning"
  - "sensor noise"
  - "long-term temporal reasoning"
date: 2026-05-08
content_hash: 9f7be8debddf618f
---

# Predict and Resist: Long-Term Accident Anticipation under Sensor Noise

**Conference**: AAAI 2026
**arXiv**: [2511.08640](https://arxiv.org/abs/2511.08640)
**Code**: None
**Area**: Other
**Keywords**: accident anticipation, diffusion denoising, Actor-Critic reinforcement learning, sensor noise, long-term temporal reasoning

## TL;DR

A unified framework is proposed that integrates a diffusion-based dual-level denoising module with a temporally-aware Actor-Critic reinforcement learning model to enable robust long-term traffic accident anticipation under sensor noise, achieving state-of-the-art performance on three benchmark datasets in terms of both average precision (AP) and mean time-to-accident (mTTA).

## Background & Motivation

Traffic accident anticipation—predicting collision likelihood before impact occurs—is a critical capability for autonomous driving. Unlike conventional perception systems that only detect accidents after they occur, anticipation enables proactive safety interventions such as timely braking or evasive maneuvers, shifting the safety paradigm from reactive to preventive.

However, achieving reliable accident anticipation faces two **mutually coupled** core challenges:

**Robustness under imperfect perception**: Sensors on autonomous vehicles are inevitably affected by rain, glare, dust, lens damage, and motion blur. Under such noise conditions, short-term single-frame predictions become unreliable. Ironically, these are precisely the scenarios that demand longer temporal reasoning—by accumulating weak signals across time, a model can still extract meaningful patterns even when individual frames are corrupted.

**The "when to warn" problem**: Most existing methods focus on frame-level classification or short-term prediction, indicating whether an accident may occur but rarely optimizing *when* to issue a warning. In safety-critical scenarios, timing is as important as accuracy: warnings issued too late are useless, while premature or overly frequent alerts erode trust and may even induce unsafe responses. This is fundamentally a **long-term credit assignment problem**: the model must identify subtle early cues, maintain temporal reasoning, and determine the optimal moment to act.

**Key insight**: These two challenges amplify each other. Sensor degradation increases uncertainty in immediate observations, which in turn magnifies the need for long-term temporal reasoning to stabilize predictions; conversely, models lacking effective temporal credit assignment cannot exploit cross-frame redundancy to overcome noisy perception. A truly deployable anticipation system must therefore address timing and robustness as a **single coupled problem**.

## Method

### Overall Architecture

This paper reformulates accident anticipation as a **sequential decision-making problem under uncertainty**. The framework consists of five core components: an object detector, a feature extractor, a self-adaptive object-aware module, a dual-level (image/object) diffusion denoising module, and an Actor-Critic decision module.

Given a video $V = \{V_t\}_{t=1}^T$, a learnable function $f_\theta$ predicts per-frame accident probabilities:

$$p_t = f_\theta(V_{1:t}), \quad t = 1, \ldots, T$$

The time-to-accident (TTA) is defined as the interval from the first confident prediction to the actual accident frame:

$$\Delta t = \tau - t_o \quad \text{where} \quad t_o = \min\{t \in \{1,\ldots,T\} \mid p_t \geq p_{th}\}$$

### Key Designs

1. **Object Detection and Feature Extraction**: Each frame is processed by Cascade R-CNN; the top-K dynamic agents are selected and encoded into feature vectors $\mathbf{F}_{obj}$ via VGG-16. Global features $\mathbf{F}_{img}$ are extracted using VGG-16 and MLP to encode scene context.

2. **Self-Adaptive Object-Aware Module**: Dynamically attends to the most informative traffic participants based on temporal context and inter-object interactions. The attention mechanism is computed as:

    $\mathbf{e}_t = \tanh(\mathbf{W}_{wa}\mathbf{h}_{t-1} + \mathbf{W}_{ua}\mathbf{F}_{obj} + \mathbf{b}_a)$
    $\alpha_t = \text{softmax}(\mathbf{W}_w \mathbf{e}_t)$
    $\bar{\mathbf{F}}_{obj} = \alpha_t \odot \mathbf{F}_{obj}$

   This mechanism adaptively prioritizes high-risk objects while encoding critical temporal interaction information.

3. **Diffusion-Based Hierarchical Feature Enhancement**: One of the paper's core innovations, applying diffusion denoising separately at the image level and the object level.

   The **forward diffusion process** employs a variance-preserving Markov chain:
    $\mathbf{F}_{img}^{noisy} = \sqrt{\bar{\alpha}_t}\mathbf{F}_{img} + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$

   A linear noise schedule ranges from $\beta_{start} = 0.001$ to $\beta_{end} = 0.02$.

   The **denoising network** is a lightweight feed-forward network:
    $p_\theta(\mathbf{F}_{img}^{noisy}, t) = W_2(\text{ReLU}(W_1\mathbf{F}_{img}^{noisy} + b_1)) + b_2$

   A **residual fusion strategy** preserves semantic fidelity:
    $\mathbf{F}_{img}^{enhanced} = \mathbf{F}_{img} + \lambda \cdot p_\theta(\mathbf{F}_{img}^{noisy}, t), \quad \lambda = 0.15$

   The same residual enhancement is applied to object-level features $\bar{\mathbf{F}}_{obj}$. The essential function of this design is a probabilistic feature stabilizer, analogous to Bayesian evidence accumulation—iteratively refining noisy inputs into structurally faithful and temporally consistent representations to reduce jitter and spurious activations.

4. **Temporal Fusion and GRU Reasoning**: Enhanced image and object features are concatenated and fed into a GRU to capture sequential dependencies:
    $\mathbf{X}_t, \mathbf{h}_t = \text{GRU}(\text{concat}(\mathbf{F}_{img}^{enhanced}, \mathbf{F}_{obj}^{enhanced}))$
   An MLP predicts per-frame accident probability $p_t = \text{MLP}(\mathbf{X}_t)$, and a temporal weighting layer computes temporal-weight loss $w_t = \text{fc}(\mathbf{h}_t)$.

5. **Actor-Critic Long-Term Decision Module**: A rolling buffer stores the most recent $W$ hidden states, and a summary vector $\bar{\mathbf{h}}_t$ is obtained via mean pooling.

   The **Actor** maps historical states to a discrete action distribution:
    $\pi_t = \text{softmax}(\mathbf{W}_p \bar{\mathbf{h}}_t + \mathbf{b}_p)$

   The **Critic** estimates expected cumulative reward:
    $V_t = \mathbf{w}_v^\top \bar{\mathbf{h}}_t + b_v$

   The **reward function** balances prediction correctness and temporal urgency:
    $r_t = \mathbb{I}(a_t = y_t) \cdot e^{-t/\tau} + \mathbb{I}(a_t \neq y_t) \cdot \gamma$
   where the reward for correct predictions decays exponentially over time (encouraging early prediction) and incorrect predictions receive a fixed penalty ($\tau=5, \gamma=-0.5$).

### Loss & Training

The total loss consists of three components:

$$\mathcal{L}_{total} = \mathcal{L}_{an} + \alpha(\mathcal{L}_{actor} + \beta \mathcal{L}_{critic})$$

where $\alpha = \beta = 0.5$.

- **Anticipation loss $\mathcal{L}_{an}$**: Applies a temporal penalty $p = -\max(0, (t_{accident} - t_{current} - 1)/\text{fps})$ and adaptive temporal weight $\omega_t = 1 + \sigma(h_t)$ to positive samples, encouraging earlier predictions.
- **Policy gradient loss**: $\mathcal{L}_{actor} = -\mathbb{E}[\log \pi_t(a_t) \cdot A_t] - \lambda_e \mathcal{H}(\pi_t)$, where $A_t = \tilde{r}_t - V_t$ is the advantage function and $\lambda_e = 0.1$ controls entropy regularization.
- **Value loss**: $\mathcal{L}_{critic} = \frac{1}{2}(\tilde{r}_t - V_t)^2$

Training uses PyTorch 2.0 on an NVIDIA RTX 3050 for 30 epochs, with batch size 10, Adam optimizer (initial learning rate $3 \times 10^{-4}$), and a ReduceLROnPlateau scheduler. Each frame contains up to 19 objects; 4096-dimensional VGG-16 features and a 256-unit GRU are used to model temporal dynamics.

## Key Experimental Results

### Main Results

Comparison with state-of-the-art methods on three benchmark datasets:

| Method | Conference | DAD AP(%) | DAD mTTA(s) | CCD AP(%) | CCD mTTA(s) | A3D AP(%) | A3D mTTA(s) |
|--------|------------|-----------|-------------|-----------|-------------|-----------|-------------|
| DSA | ACCV | 48.1 | 1.34 | 98.7 | 3.08 | 92.3 | 2.95 |
| ACRA | CVPR | 51.4 | 3.01 | 98.9 | 3.32 | - | - |
| AdaLEA | CVPR | 52.3 | 3.44 | 99.2 | 3.45 | 92.9 | 3.16 |
| UString | TIV | 53.7 | 3.53 | 99.5 | 3.74 | 93.2 | 3.24 |
| AccNet | AAP | 60.8 | 3.58 | 99.5 | 3.78 | 95.1 | 3.26 |
| LATTE | IF | 89.7 | 4.49 | 98.8 | 4.53 | 92.5 | 4.52 |
| **Ours** | - | **91.2** | **4.59** | **99.8** | **4.29** | **95.7** | **4.60** |

The proposed method achieves 91.2% AP and 4.59s mTTA on DAD, with consistent improvements on CCD and A3D as well.

### Ablation Study

Core module ablation (CCD dataset):

| Configuration | AP(%) | mTTA(s) | Note |
|---------------|-------|---------|------|
| Full model | 99.8 | 4.29 | Baseline |
| w/o Object-Aware Module | 99.3 | 4.61 | mTTA slightly increases but AP drops |
| w/o Temporal Weight Layer | 99.5 | 4.47 | Both metrics slightly decrease |
| w/o Anticipation Loss | 33.3 | 5.00 | AP collapses, confirming criticality |
| w/o Policy Gradient Loss | 99.6 | 4.47 | Minor impact |
| w/o Value Loss | 92.8 | 3.03 | Both AP and mTTA drop significantly |

Noise robustness (CCD dataset, Gaussian noise):

| Noise $\sigma$ | Full Model AP(%) | Full Model mTTA(s) | w/o Image Diffusion AP | w/o Object Diffusion AP | w/o All Diffusion AP |
|----------------|-----------------|-------------------|----------------------|------------------------|---------------------|
| Original | 99.8 | 4.29 | 99.6 | 99.6 | 99.6 |
| 0.5 | 99.6 | 4.00 | 99.4 | 99.5 | 99.4 |
| 1.0 | 99.6 | 4.04 | 99.5 | 99.5 | 99.4 |
| 5.0 | 99.6 | 4.35 | 99.5 | 99.3 | 99.0 |
| 10.0 | 98.0 | 3.43 | 98.6 | 98.8 | 98.2 |
| 20.0 | 91.6 | 3.05 | 92.8 | 91.4 | 91.0 |

Reward–penalty trade-off (A3D dataset):

| Reward Scale | Penalty Scale | AP(%) | mTTA(s) |
|--------------|---------------|-------|---------|
| ×1 (5.0) | ×1 (-0.5) | 95.7 | 4.60 |
| ×10 | ×1 | 93.6 | 4.77 |
| ×50 | ×1 | 92.7 | 4.70 |
| ×0.1 | ×1 | 96.2 | 4.47 |
| ×1 | ×10 | 91.2 | 4.92 |
| ×1 | ×0.1 | 92.1 | 4.71 |

### Key Findings

- **Anticipation loss is indispensable**: Removing it causes AP to collapse from 99.8% to 33.3%, demonstrating that simply maximizing mTTA does not necessarily yield better performance.
- **Value loss is critical for stable long-term decision-making**: Its removal causes AP to drop to 92.8% and mTTA to decrease from 4.29s to 3.03s, highlighting the essential role of value estimation in stabilizing long-horizon decisions.
- **Dual-level diffusion is most effective under moderate noise**: The full model maintains 99.6% AP at $\sigma \leq 5$; under extreme noise ($\sigma = 20$), all variants degrade, and removing image diffusion sometimes yields higher AP, suggesting that over-denoising may harm severely corrupted inputs.
- **Delicate balance between reward and penalty**: Increasing the reward weight degrades AP but improves mTTA (encouraging but overly aggressive predictions), while increasing the penalty weight makes the model overly conservative (highest mTTA = 4.92s but lowest AP = 91.2%).
- **Long-horizon training (history window = 10) vs. frame-level baseline (window = 0)**: The long-horizon model produces shorter and fewer false alarms in complex multi-agent rainy scenes, and anticipates collisions nearly one second earlier in typical crash scenarios.

## Highlights & Insights

1. **Unified treatment of a coupled problem**: The core design philosophy is to address noise robustness and warning timing as a single coupled problem rather than solving them separately.
2. **Diffusion in feature space**: Unlike conventional image-level denoising, diffusion denoising is applied in feature space, making it more lightweight and directly beneficial to downstream tasks.
3. **Conservative design of residual fusion**: The small coefficient $\lambda = 0.15$ ensures that the denoised output serves as an incremental refinement of the original features, avoiding disruption of existing semantic information.
4. **Elegant application of the RL framework**: Modeling "when to warn" as a sequential decision problem, the time-weighted reward naturally encodes the safety requirement that "earlier is better."
5. **Practical deployment orientation**: The model is trainable on an RTX 3050 and is lightweight (256-unit GRU), offering strong potential for real-world deployment.

## Limitations & Future Work

- **Over-denoising under extreme noise**: At $\sigma \geq 10$, removing image diffusion sometimes performs better, suggesting that the current fixed residual coefficient $\lambda$ and number of diffusion steps may require adaptive adjustment.
- **Notable performance drop under 50% impulse noise**: AP drops to 91.6% (A3D), which may be insufficient for dense urban environments.
- **Dependence on the object detector**: The use of Cascade R-CNN as the object detector introduces the risk of missing critical objects under noisy conditions, potentially causing cascading errors.
- **Single-viewpoint limitation**: Only dashcam footage is used; multi-sensor fusion (LiDAR, radar, etc.) is not exploited.
- **Dataset limitations**: All three datasets (DAD, CCD, A3D) cover urban scenes from specific regions; generalization to highway, rural road, and other settings remains unvalidated.
- **Fixed temporal window $W$**: The rolling buffer size is fixed and may need to be dynamically adjusted based on scene complexity.
- **Unimodal input**: Only visual features are used; driving behavior, vehicle dynamics, and other modalities are not integrated.

## Related Work & Insights

This paper fits within the evolutionary trajectory of visual accident anticipation research:

- **From frame-level perception to temporal reasoning**: A development path from CNN to RNN/LSTM/GRU.
- **From isolated objects to interaction modeling**: GNNs explicitly encode multi-agent relationships; Transformers capture long-range dependencies.
- **Scarcity and interpretability**: GANs/VAEs synthesize scarce accident scenarios; attention mechanisms enhance interpretability.

Consistent with the trend of applying diffusion models to perception tasks (e.g., denoising and data augmentation), this paper innovatively applies the diffusion process in feature space rather than image space. The introduction of reinforcement learning draws on classical methods from sequential decision-making, reformulating the timing optimization of safety warnings as a credit assignment problem.

**Implications for future research**: The dual-level denoising + RL decision paradigm proposed in this framework is generalizable to other safety-critical scenarios requiring sequential decision-making under noise, such as industrial equipment fault prediction and medical anomaly detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of diffusion denoising and Actor-Critic is novel, with a distinctive problem formulation perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, multiple noise conditions, thorough ablation and hyperparameter analysis.
- **Value**: ⭐⭐⭐⭐ — Lightweight design, noise robustness, and strong real-world deployment potential.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated, with deep analysis of the coupling between challenges.
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](../../NeurIPS2025/others/coresets_for_clustering_under_stochastic_noise.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[AAAI 2026\] ShortageSim: Simulating Drug Shortages under Information Asymmetry](shortagesim_simulating_drug_shortages_under_information_asymmetry.md)
- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](optimal_welfare_in_noncooperative_network_formation_under_attack.md)
- [\[AAAI 2026\] Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast](radar-aplanc_unsupervised_radar-based_heartbeat_sensing_via_augmented_pseudo-lab.md)

</div>

<!-- RELATED:END -->
