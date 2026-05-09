---
title: >-
  [Paper Note] Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling
description: >-
  [NeurIPS 2025][Autonomous Driving][Autonomous Driving Planning] This paper proposes Flow Planner—a system combining three synergistic innovations: fine-grained trajectory tokenization, an interaction-enhanced spatiotemporal fusion architecture, and flow matching with classifier-free guidance. It is the first purely learning-based method to surpass 90 points on nuPlan Val14 (90.43), and outperforms Diffusion Planner by 8.92 points on the interaction-intensive interPlan benchmark.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - Autonomous Driving Planning
  - Flow Matching
  - Interactive Behavior Modeling
  - Classifier-Free Guidance
  - Trajectory Generation
date: 2026-05-08
content_hash: 4d23bc7403aa1e9d
---

# Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling

**Conference**: NeurIPS 2025  
**arXiv**: [2510.11083](https://arxiv.org/abs/2510.11083)  
**Code**: [https://github.com/DiffusionAD/Flow-Planner](https://github.com/DiffusionAD/Flow-Planner)  
**Area**: Autonomous Driving  
**Keywords**: Autonomous Driving Planning, Flow Matching, Interactive Behavior Modeling, Classifier-Free Guidance, Trajectory Generation  

## TL;DR
This paper proposes Flow Planner—a system combining three synergistic innovations: fine-grained trajectory tokenization, an interaction-enhanced spatiotemporal fusion architecture, and flow matching with classifier-free guidance. It is the first purely learning-based method to surpass 90 points on nuPlan Val14 (90.43), and outperforms Diffusion Planner by 8.92 points on the interaction-intensive interPlan benchmark.

## Background & Motivation

**Background**: Autonomous driving planning methods are broadly divided into rule-based approaches (e.g., PDM-Closed) and learning-based approaches (imitation learning + generative models). Learning-based methods have advanced rapidly with transformers and diffusion models, yet remain insufficient in interactive scenarios.

**Limitations of Prior Work**: (a) Naively stacking transformer blocks lacks effective fusion mechanisms for heterogeneous information (static lanes + dynamic neighboring vehicles); (b) high-quality interactive scenarios are scarce in training data, causing naive behavior cloning to converge to distributions that deviate from real interaction behaviors; (c) auxiliary losses (e.g., collision penalties) require case-by-case design and compromise training stability; (d) Diffusion Planner's interaction modeling is limited to a few nearest vehicles and lacks a dedicated fusion architecture.

**Key Challenge**: Effective interactive behavior modeling requires three conditions to be satisfied simultaneously—(i) expressive trajectory representations, (ii) efficient heterogeneous information fusion, and (iii) dynamic enhancement of conditional signals to compensate for the scarcity of interaction data. Prior methods satisfy at most one of these.

**Key Insight**: Co-design across three dimensions: data modeling (trajectory tokenization), model architecture (spatiotemporal fusion), and learning paradigm (flow matching + CFG).

**Core Idea**: Flow Planner = fine-grained trajectory tokens + scale-adaptive attention spatiotemporal fusion + flow matching CFG for dynamic augmentation of neighbor conditions.

## Method

### Overall Architecture
The input is vectorized scene information (neighbor history, lanes, static objects, navigation), encoded into tokens via MLP-Mixer. The ego trajectory is decomposed into overlapping segment tokens through fine-grained tokenization. All tokens are fused spatiotemporally in a unified latent space via scale-adaptive attention. Training employs a flow matching loss with Bernoulli condition masking; inference uses classifier-free guidance to enhance neighbor interaction conditioning.

### Key Designs

1. **Fine-grained Trajectory Tokenization**:

    - Function: Decomposes the full trajectory into overlapping segment tokens, balancing expressiveness and consistency.
    - Mechanism: A trajectory $\tau_t$ of $L$ waypoints is divided into $K$ segments, each containing $L_{seg}$ points with overlap $L_{overlap}$ between adjacent segments: $F_{ego}^k = \text{MLP}((x_{l^k}, \ldots, x_{r^k}))$, where $l^k = (k-1)(L_{seg} - L_{overlap})$. After adding sinusoidal positional encodings, the segments are concatenated as $F_{ego} = \text{Concat}(F_{ego}^1, \ldots, F_{ego}^K)$.
    - A consistency loss is applied over overlapping regions: $\mathcal{L}_{consist} = \frac{1}{K-1} \sum_{k=1}^{K-1} \|\hat{\tau}^{k:k+1} - \hat{\tau}^{k+1:k}\|^2$
    - Design Motivation: Representing the entire trajectory as a single token leads to excessive compression and insufficient scene information fusion, while per-timestep tokens suffer from severe error accumulation. Overlapping segments strike a balance between the two. Ablation studies verify that $K=20$ segments is optimal.

2. **Interaction-enhanced Spatiotemporal Fusion**:

    - Function: Efficiently fuses heterogeneous scene tokens to enhance interaction modeling.
    - Mechanism:
        - Heterogeneous features (lane, neighbor, ego) are first projected into a shared latent space via separate adaptive LayerNorm (adaLN) layers, with timestep and navigation conditions injected.
        - After concatenation, **scale-adaptive self-attention** performs global fusion: $F_{global} = \text{Softmax}\left(\frac{F_{global}W^Q (F_{global}W^K)^T}{\sqrt{d}} - \lambda \cdot D\right) F_{global}W^V$
        - Here $D$ is the Euclidean distance matrix between tokens, and $\lambda$ is a learnable receptive field scaling factor generated by a linear projection of the tokens themselves. Tokens that are spatially farther apart receive smaller attention scores.
        - After fusion, tokens are decomposed back into modality-specific representations, each processed through independent adaLN + FFN layers to further reduce the modality gap.
    - Design Motivation: Vanilla attention cannot effectively handle heterogeneous information fusion; scale-adaptive attention enables the model to adaptively attend to relevant neighboring vehicles based on spatial distance.

3. **Flow Matching + Classifier-Free Guidance**:

    - Function: Enables multimodal interactive behavior generation through conditioning-enhanced flow matching.
    - The conditional generative distribution is: $\tilde{q}(\tau_1|C) \propto q(\tau_1)^{1-\omega} q(\tau_1|C)^{\omega}$, with guided velocity field: $\tilde{v}_t(\tau_t, t|C) = (1-\omega) v_t(\tau_t, t) + \omega \cdot v_t(\tau_t, t|C)$
    - Training uses Bernoulli condition masking: $\mathcal{L}_{flow} = \mathbb{E}_{t, b \sim \mathcal{B}} \|\tau_\theta(\tau_t, t|(1-b) \cdot C + b \cdot \emptyset) - \tau_1\|^2$
    - In practice, only neighbor information is masked (identified experimentally as the most critical element for interaction modeling).
    - Inference employs optimal transport paths and a second-order midpoint ODE solver.
    - Design Motivation: CFG enables the model to jointly learn unconditional planning and conditional planning; the difference between them captures behavior changes induced by neighboring vehicles. At inference time, this difference can be amplified via $\omega$ to enhance interaction awareness.

### Loss & Training
- Final loss: $\mathcal{L} = \mathcal{L}_{flow} + \alpha \cdot \mathcal{L}_{consist}$
- Data augmentation: Random perturbations are applied to the ego vehicle's current frame state, with new ground-truth trajectories generated via quintic polynomial interpolation.

## Key Experimental Results

### Main Results — nuPlan Closed-Loop Evaluation

| Type | Method | Val14 NR | Val14 R | Test14-hard R |
|------|------|---------|--------|--------------|
| Rule-based | PDM-Closed | 92.84 | 92.12 | 75.19 |
| Hybrid | PDM-Hybrid | 92.77 | 92.11 | 76.07 |
| Learning | PLUTO (w/o refine.) | 88.89 | 78.11 | 59.74 |
| Learning | Diffusion Planner | 89.87 | 82.80 | 69.22 |
| **Learning** | **Flow Planner** | **90.43** | **83.31** | **70.42** |
| Hybrid | Flow Planner w/ refine. | 94.31 | 92.38 | 80.25 |

A score of **90.43** on Val14 marks the first time a purely learning-based method surpasses 90 points.

### interPlan Interactive Benchmark

| Method | Overall | Nudge Around | High Traffic | Jaywalk |
|------|---------|-------------|-------------|---------|
| PlanTF | 47.70 | 49.40 | 58.85 | 33.94 |
| PLUTO | 58.47 | 71.56 | 67.25 | 25.48 |
| Diffusion Planner | 52.90 | 60.48 | 49.71 | 26.20 |
| **Flow Planner** | **61.82** | **72.96** | **67.21** | **43.57** |

Flow Planner surpasses Diffusion Planner by 8.92 overall and by 17.37 on the Jaywalk scenario—the most challenging interaction case involving unpredictable pedestrian crossing.

### Ablation Study

| Configuration | nuPlan Val14 | interPlan |
|------|-------------|-----------|
| Base (vanilla self-attention) | 88.10 | 41.27 |
| + Trajectory Tokenization | 88.33 | 44.14 |
| + Scale-Adaptive Attention | 88.77 | 46.25 |
| + Separate adaLN & FFN | 89.54 | 58.22 |
| + Classifier-Free Guidance | **90.43** | **61.82** |

### CFG Scale Ablation

| CFG $\omega$ | Val14 Score |
|-----|-----------|
| 1.65 | 89.64 |
| 1.75 | 90.14 |
| **1.80** | **90.43** |
| 1.85 | 90.00 |
| 1.90 | 89.63 |

### Key Findings
- **Separate adaLN + FFN** contributes the most: interPlan improves from 46.25 to 58.22 (+11.97), indicating that heterogeneous feature fusion is the key bottleneck in interaction modeling.
- **CFG** yields a further +3.6 gain on interPlan, validating the importance of dynamically amplifying conditional signals at inference time for interactive scenarios.
- **$K=20$ trajectory segments** achieves the best performance (Table 5); too few segments (1) fail to model multimodal behavior, while too many (80) impose excessive token overhead.
- Flow Planner substantially outperforms Diffusion Planner in interaction-intensive scenarios such as unprotected left turns and pedestrian crossings; case studies show it can recognize approaching vehicles from behind and abort lane changes accordingly.

## Highlights & Insights
- **Applying CFG to autonomous driving planning** is the genuinely insightful contribution—by masking neighbor information to train an unconditional branch, setting $\omega > 1$ at inference amplifies the conditional influence, implicitly learning "behavior changes caused by neighboring vehicles." This is more elegant and more effective than explicit collision penalties.
- **Fine-grained trajectory tokenization with overlapping consistency loss** resolves the dilemma between single-token and per-timestep tokenization—avoiding both over-compression and error accumulation.
- **Scale-adaptive attention** incorporates spatial distance priors into attention scores—intuitively correct (distant vehicles have less influence) and architecturally simple (a single learnable scalar plus a distance matrix bias).

## Limitations & Future Work
- The method relies on preprocessed perception inputs from nuPlan (vectorized representations) and does not process raw sensor data end-to-end.
- The CFG scale $\omega$ requires manual tuning and lacks an adaptive mechanism.
- Uncertainty in pedestrian and cyclist intent is not modeled—the Jaywalk scenario shows substantial improvement but absolute scores remain modest (43.57).
- Flow Planner w/ refine. and Diffusion Planner w/ refine. achieve similar performance, suggesting that post-processing may obscure differences between the underlying models.

## Related Work & Insights
- **vs. Diffusion Planner**: Diffusion Planner employs DDPM with joint ego-neighbor generation, but interaction is limited to a fixed set of nearest vehicles and the architecture lacks a dedicated fusion design. Flow Planner uses flow matching (faster convergence) + CFG (dynamic interaction enhancement) + scale-adaptive fusion.
- **vs. PLUTO**: PLUTO relies on reference line priors, contrastive loss, and post-processing; Flow Planner achieves superior performance without any such priors.
- **vs. PDM-Closed**: Rule-based methods have a clear advantage in common scenarios (92.12 R) but underperform on hard scenarios (75.19) compared to Flow Planner, which captures out-of-distribution interactions through learned flow.

## Rating
- Novelty: ⭐⭐⭐⭐ Three co-designed innovations; the application of CFG to planning is a genuinely novel insight
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three nuPlan benchmarks + interPlan + comprehensive ablations + case studies
- Writing Quality: ⭐⭐⭐⭐ Clear structure; case studies are intuitive and illustrative
- Value: ⭐⭐⭐⭐⭐ A milestone as the first purely learning-based method to exceed 90 points, introducing a new paradigm for interaction modeling

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[NeurIPS 2025\] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)
- [\[NeurIPS 2025\] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning](unimotion_a_unified_motion_framework_for_simulation_prediction_and_planning.md)
- [\[AAAI 2026\] ReflexDiffusion: Reflexion-Enhanced Trajectory Planning for High Lateral Acceleration in Autonomous Driving](../../AAAI2026/autonomous_driving/reflexdiffusion_reflection-enhanced_trajectory_planning_for_.md)

</div>

<!-- RELATED:END -->
