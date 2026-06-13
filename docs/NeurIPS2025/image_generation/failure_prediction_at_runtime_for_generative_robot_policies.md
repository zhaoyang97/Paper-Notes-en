---
title: >-
  [Paper Note] Failure Prediction at Runtime for Generative Robot Policies
description: >-
  [NeurIPS 2025][Image Generation][Failure Prediction] This paper proposes FIPER, a framework for runtime failure prediction in generative robot policies (diffusion/flow matching). It jointly evaluates an observation-side…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Failure Prediction"
  - "Generative Policies"
  - "RND"
  - "Action Chunk Entropy"
  - "Conformal Prediction"
date: 2026-05-08
content_hash: 5cb97c69a5123d2a
---

# Failure Prediction at Runtime for Generative Robot Policies

**Conference**: NeurIPS 2025
**arXiv**: [2510.09459](https://arxiv.org/abs/2510.09459)  
**Code**: [GitHub](https://github.com/utiasDSL/fiper)  
**Project Page**: [FIPER Website](https://tum-lsy.github.io/fiper_website)
**Authors**: Ralf Römer*, Adrian Kobras*, Luca Worbis, Angela P. Schoellig (TUM Learning Systems and Robotics Lab)
**Area**: Image Generation
**Keywords**: Failure Prediction, Generative Policies, RND, Action Chunk Entropy, Conformal Prediction

## TL;DR

This paper proposes FIPER, a framework for runtime failure prediction in generative robot policies (diffusion/flow matching). It jointly evaluates an observation-side metric RND-OE (OOD detection) and an action-side metric ACE (Action Chunk Entropy) to enable early and accurate failure prediction without any failure data, with statistical guarantees provided via conformal prediction.

## Background & Motivation

**Background**: Generative imitation learning methods—including diffusion policy and flow matching—have achieved remarkable progress in recent years, enabling robots to perform complex long-horizon manipulation tasks. By learning multimodal conditional action distributions, these approaches demonstrate strong task generalization.

**Limitations of Prior Work**:
- Distribution shifts at deployment (unseen environments, lighting changes, object position variations) or accumulated action errors can lead to unpredictable and dangerous behavior.
- Existing OOD detection methods operate solely on the observation side, producing numerous false positives for **benign OOD states** that the policy can actually generalize to.
- VLM-based methods can only **retrospectively** detect failures that have already occurred, offering no early warning capability.
- Many methods rely on failure data collection, which is both unsafe and impractical in real-world settings.
- Existing uncertainty measures fail to properly handle the **multimodal action distributions** of generative policies.

**Key Challenge**: Safe deployment requires precise runtime failure prediction, yet no single signal source—observation or action alone—can reliably distinguish genuine failure precursors from situations the policy can handle.

**Goal**: To provide an early failure prediction mechanism for generative robot policies at runtime, without requiring failure data, while minimizing false positives on benign OOD situations.

**Key Insight**: Based on the observation that *failures are typically accompanied by both unfamiliar observations and incoherent actions*, the paper designs a dual-metric detection framework that only triggers an alarm when both signals are simultaneously anomalous.

**Core Idea**: Combining observation-space OOD detection with action-space uncertainty quantification in a complementary, noise-reducing fashion, with conformal prediction for threshold calibration, to achieve runtime failure prediction requiring no failure data.

## Method

### Overall Architecture

FIPER (**Fi**ailure **P**rediction at **R**untime) is a modular runtime failure prediction framework built on a key insight: **failures tend to coincide with both unfamiliar observations and ambiguous/incoherent actions**. The framework consists of three core components:

1. **Observation-side metric RND-OE**: Applies Random Network Distillation within the policy's own observation embedding space to detect whether the current observation deviates from the training distribution.
2. **Action-side metric ACE**: Proposes Action Chunk Entropy, which samples multiple batches of action chunks from the conditional action distribution, computes entropy scores in end-effector space, and quantifies action uncertainty.
3. **Temporal window aggregation + dual-threshold triggering**: Both scores are smoothed over a short temporal window, and calibrated thresholds from conformal prediction are applied; a failure alarm is raised only when **both** scores simultaneously exceed their respective thresholds.

Overall pipeline: observations are encoded by the policy encoder to obtain embeddings → RND-OE computes an OOD score → the policy samples multiple action chunk batches → ACE computes an entropy score → both scores are aggregated over a temporal window → dual-threshold detection → alarm triggered or not.

### Key Designs

1. **Random Network Distillation in Observation Embeddings (RND-OE)**

    - **Function**: Detects whether the current observation deviates from the policy's training data distribution.
    - **Mechanism**: RND is deployed in the policy's own observation embedding space rather than the raw pixel space. A randomly initialized teacher network is fixed, and a student network is trained to replicate the teacher's outputs on training embeddings. For in-distribution embeddings, the student accurately matches the teacher (low prediction error); for OOD embeddings, large prediction errors emerge and serve as the OOD signal.
    - **Design Motivation**: Performing OOD detection in raw observation space is easily perturbed by task-irrelevant visual changes (e.g., lighting, background texture), causing false positives. Using the policy's learned embedding space naturally filters out task-irrelevant visual variation and focuses on semantic representations that truly influence policy decisions, improving detection robustness.

2. **Action Chunk Entropy (ACE)**

    - **Function**: Quantifies the uncertainty of actions produced by a generative policy at the current state.
    - **Mechanism**: A batch of action chunks is sampled from the policy's conditional action distribution. Each chunk is transformed into end-effector space, and an entropy score is computed over the batch. The specially designed entropy measure distinguishes between "multimodal yet each mode is confident" (benign, low uncertainty) and "disordered across modes" (high uncertainty / failure precursor).
    - **Design Motivation**: A core strength of generative policies (diffusion/flow matching) is the ability to learn multimodal action distributions. Conventional variance/entropy measures incorrectly classify reasonable multimodal behavior as high uncertainty. ACE is computed in end-effector space and properly accounts for temporal modal consistency, effectively distinguishing benign multimodality from genuine action confusion.

3. **Conformal Prediction Calibration and Dual-Metric Joint Decision**

    - **Function**: Sets statistically guaranteed thresholds for both metrics and reduces false positives through joint decision-making.
    - **Mechanism**: A small set of successful demonstration rollouts (50 in simulation, only 10 in the real world) serves as the calibration set. Conformal prediction is used to independently derive thresholds for RND-OE and ACE. At inference time, both scores are averaged over a short moving window to smooth noise; a failure alarm is triggered only when **both scores simultaneously** exceed their respective thresholds within the window.
    - **Design Motivation**: A single metric tends to produce characteristic false positives—RND-OE is sensitive to benign OOD, while ACE may miss failures in which the observation is anomalous but the action appears confident. The intersection logic of dual metrics naturally filters out each metric's specific false positive sources. Conformal prediction provides a statistical upper bound on the false positive rate, enhancing the method's credibility in safety-critical scenarios.

### Loss & Training

- **RND student network training**: Trained using MSE loss $\mathcal{L}_{\text{RND}} = \| f_\theta(\mathbf{z}) - f_{\text{teacher}}(\mathbf{z}) \|^2$ on observation embeddings from successful rollouts, where $\mathbf{z}$ is the output of the policy encoder.
- **No policy training data required**: Neither RND training nor ACE computation requires access to the policy's original training dataset.
- **No failure data required**: The entire calibration process relies solely on successful rollouts, without any failure demonstrations.
- **Conformal prediction calibration**: The $(1-\alpha)$ quantile of scores on the calibration set is used as the threshold, where $\alpha$ controls the upper bound on the allowed false positive rate.

## Key Experimental Results

### Main Results

FIPER is evaluated across 5 diverse task environments (3 simulation + 2 real-world), covering both diffusion policy and flow matching, and spanning multiple failure modes including grasping, rope manipulation, and bimanual coordination.

| Method | Accuracy | Early Prediction Time | False Positive Rate | Benign OOD Discrimination |
|---|---|---|---|---|
| OOD-only (RND) | Moderate | Earlier | **High** | Poor (benign OOD also triggers) |
| Action-only (Variance) | Moderate | Moderate | Moderate | Moderate |
| VLM-based | Relatively high | **Late** (retrospective) | Low | — |
| FIPER (RND-OE + ACE) | **Highest** | **Earliest** | **Lowest** | **Best** |

- FIPER outperforms all baselines across all 5 environments, with especially significant advantages in distinguishing genuine failures from benign OOD situations.
- In real-world rope manipulation tasks, FIPER can provide warnings several seconds before failure occurs, allowing sufficient time for human intervention.

### Ablation Study

| Ablation Configuration | Result |
|---|---|
| RND-OE only | Detects OOD but high false positive rate; benign OOD frequently triggers alarms |
| ACE only | Captures action uncertainty but misses failures caused by observation anomalies |
| RND-OE + ACE (no temporal window) | Unstable detection; single-frame noise causes spurious triggers |
| RND-OE + ACE (with temporal window) | Prediction stability and accuracy significantly improved |
| Raw pixel-space RND vs. embedding-space RND-OE | Embedding-space version is more robust with fewer false positives |
| Different policy types (diffusion vs. flow matching) | FIPER is effective on both, validating framework generality |
| Calibration data volume sensitivity | 50 successful rollouts in simulation and 10 in the real world are sufficient |

### Key Findings

1. **Dual-metric complementarity is central**: Neither RND-OE nor ACE alone can reliably predict failures; their combination substantially reduces false positives while maintaining high detection rates.
2. **Embedding space >> raw space**: Applying RND in the policy's embedding space is more effective than in the raw pixel space, as the embedding space filters out task-irrelevant visual variation.
3. **ACE correctly handles multimodality**: Conventional variance measures incorrectly flag multimodal action distributions as high uncertainty; ACE properly distinguishes "multimodal but confident" from "genuinely disordered."
4. **Minimal calibration data is sufficient**: Only 10 successful rollouts in the real world are needed for effective calibration, demonstrating strong practicality.
5. **Cross-policy generalization**: The same framework performs well on both diffusion policy and flow matching without task-specific modifications.

## Highlights & Insights

1. **Zero failure data requirement**: From a safe deployment perspective, independence from failure data is a substantial practical advantage—collecting failure data is itself dangerous and costly.
2. **Input–output dual-side detection**: The observation side checks "whether what is seen is anomalous," and the action side checks "whether what is to be done is incoherent." This I/O dual-side detection philosophy is both elegant and effective.
3. **Deep understanding of generative policies**: The design of ACE reflects a thorough understanding of the core characteristic of diffusion/flow matching models—namely, producing multimodal action distributions—rather than naively applying conventional uncertainty methods.
4. **Statistical guarantees enhance credibility**: Conformal prediction provides a mathematical upper bound on false positive rates, which is critical for safety-critical robot deployment.
5. **Interpretability**: The framework can distinguish whether a failure is caused by an observation anomaly or action confusion, providing valuable diagnostic information for debugging and human–robot interaction.
6. **Modular design**: RND-OE and ACE function as independent plug-and-play modules that can be attached to any generative policy without modifying the policy itself.

## Limitations & Future Work

1. **Theoretical assumptions of conformal prediction**: Coverage guarantees rely on the exchangeability assumption of data, which may not hold in highly non-stationary or adversarial environments.
2. **Passive prediction rather than active recovery**: The current framework only predicts failures and does not integrate active recovery mechanisms (e.g., automatically requesting human takeover, switching to a safe policy, or executing fallback actions).
3. **Temporal window hyperparameter**: The window size must be manually selected; different tasks may require different window lengths, and no adaptive tuning mechanism is provided.
4. **Detection latency in extreme unseen scenarios**: In novel scenarios that differ substantially from the training distribution, some detection delay may still occur.
5. **End-effector space assumption**: ACE is computed in end-effector space, which may require redefinition of an appropriate action space for non-manipulation tasks such as navigation.

## Related Work & Insights

- **Random Network Distillation (RND)** was originally proposed for curiosity-driven exploration in deep reinforcement learning (Burda et al., 2019); this paper innovatively transfers it to the observation embedding space for OOD detection.
- **Conformal prediction** is increasingly popular in uncertainty quantification (Angelopoulos & Bates, 2023); this paper demonstrates its practical value for providing guaranteed threshold calibration in robot safety monitoring.
- Compared to ensemble-based uncertainty methods, FIPER does not require training multiple policy copies, making it more computationally tractable.
- The dual-metric joint-triggering design philosophy is generalizable to other safety-critical systems requiring low false positive detection rates.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The joint observation-plus-action dual-side detection framework is novel; ACE as a tailored entropy measure for multimodal action distributions represents a valuable technical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five environments spanning simulation and the real world, two policy types, and thorough ablations; some quantitative comparison details could be more comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ — Requires no failure data, no policy training data, minimal calibration data, lightweight computation, and plug-and-play deployment; extremely practical.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, the method is intuitive, and the project page videos provide effective illustration.

---
title: >-
  [Paper Notes] Failure Prediction at Runtime for Generative Robot Policies
description: >-
  [NeurIPS 2025][Image Generation][Failure Prediction] Proposes the FIPER framework, which combines observation-space OOD detection (RND) and action-space uncertainty quantification (ACE) to achieve early failure prediction for generative robot policies at runtime without requiring any failure data.
tags:
  - NeurIPS 2025
  - Image Generation
  - Failure Prediction
  - Imitation Learning
  - Diffusion Models
  - Out-of-Distribution Detection
  - Conformal Prediction
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[NeurIPS 2025\] Real-Time Execution of Action Chunking Flow Policies](real-time_execution_of_action_chunking_flow_policies.md)
- [\[NeurIPS 2025\] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](image_super-resolution_with_guarantees_via_conformalized_generative_models.md)

</div>

<!-- RELATED:END -->
