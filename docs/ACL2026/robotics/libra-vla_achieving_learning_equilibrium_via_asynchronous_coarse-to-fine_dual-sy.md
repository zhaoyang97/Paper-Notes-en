---
title: >-
  [Paper Note] Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System
description: >-
  [ACL 2026][Robotics][Vision-Language-Action Models] Libra-VLA decomposes robotic actions into a hybrid action space of "discrete macro-intents + continuous micro-poses." It employs System 2 (VLM + parallel coarse-action…
tags:
  - "ACL 2026"
  - "Robotics"
  - "Vision-Language-Action Models"
  - "Hybrid Action Space"
  - "Dual-System"
  - "Asynchronous Execution"
  - "Coarse-to-Fine"
date: 2026-05-08
content_hash: 2a5cce4df40be644
---

# Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System

**Conference**: ACL 2026  
**arXiv**: [2604.24921](https://arxiv.org/abs/2604.24921)  
**Code**: <https://libra-vla.github.io/>  
**Area**: VLA / Embodied AI / Dual-System Architecture  
**Keywords**: Vision-Language-Action Models, Hybrid Action Space, Dual-System, Asynchronous Execution, Coarse-to-Fine

## TL;DR
Libra-VLA decomposes robotic actions into a hybrid action space of "discrete macro-intents + continuous micro-poses." It employs System 2 (VLM + parallel coarse-action head) for low-frequency planning and System 1 (diffusion transformer + independent SigLIP encoder) for high-frequency refinement. Facilitated by an intent buffer for asynchronous execution, it achieves a State-of-the-Art (SoTA) success rate of 97.2% on LIBERO and 79.5% zero-shot on LIBERO-Plus (10% higher than the previous OpenVLA-OFT+).

## Background & Motivation
**Background**: VLA models (OpenVLA, π0, π0.5, GR00T-N1, etc.) have become the mainstream paradigm for open-world general-purpose robots, directly grounding language instructions into motor commands. Predominant approaches follow two paths: (a) discretizing continuous actions into 256 bins for Autoregressive (AR) prediction (OpenVLA, π0-FAST); (b) attaching a diffusion head to a VLM backbone to output continuous actions directly (π0, GR00T-N1, Diffusion Policy).

**Limitations of Prior Work**: Both approaches are **monolithic** "flat mappings"—a single network simultaneously processes high-level abstract semantic reasoning and low-level high-frequency motor control. This unified architecture ignores the natural hierarchical structure of robotic manipulation (coarse positioning followed by fine alignment), placing the massive "semantic-execution" gap on a single model, which leads to heavy representation burdens.

**Existing hierarchical attempts** are insufficient: HAMSTER/MOKA use keypoints, while ViLA/Hi Robot use sub-instructions, focusing on **temporal** decomposition (shortening planning horizons). However, each step still must bridge high-level modalities to continuous motor commands, leaving single-step representation complexity unsimplified. HybridVLA, despite its name, uses two independent branches for fine-grained prediction followed by arithmetic averaging, which is essentially a parallel structure without a hierarchy.

**Key Challenge**: There is a lack of hierarchy in the **action representation space**. Finer discrete bins reduce quantization error but deviate from VLM semantic abstraction; meanwhile, continuous outputs demand excessive geometric precision from the VLM. In dual-system architectures, GR00T-N1 uses static latents as bridges that can become "outdated," FiS-VLA suffers from "feature squeezing" due to multi-tasking on a single backbone, and OpenHelix relies on uninterpretable high-dimensional black-box latents.

**Goal**: (1) Decompose actions hierarchically within the **action representation space** rather than just the timeline; (2) Balance the learning difficulty between two subsystems through task division; (3) Achieve truly asynchronous, interpretable, and low-latency execution.

**Key Insight**: Explicitly split actions into a hybrid space—discrete coarse directions (macro-intent, answering "where to go") + continuous micro-poses (micro-alignment, answering "how to interact"). The former naturally aligns with the discrete token output space of VLMs, while the latter only needs to generate residuals around anchors, significantly compressing the search space.

**Core Idea**: Replace "flat modality translation" with "two-stage simple mapping" + asynchronous dual-system execution + an intent buffer for multi-step coarse direction look-ahead.

## Method

### Overall Architecture
- **Input**: Instruction $L$ + Observation $\mathbf{o}_t$
- **Probability Decomposition**: $P(\mathbf{a}_t \mid \mathbf{o}_t, L) \approx \underbrace{P(\mathbf{a}_t^f \mid \mathbf{a}_t^c, \mathbf{o}_t)}_{\text{Action Refiner}} \cdot \underbrace{P(\mathbf{a}_t^c \mid \mathbf{o}_t, L)}_{\text{Semantic Planner}}$
- **System 2 (Semantic Planner)**: InternVL2.5-2B backbone + Parallel Coarse-Action Head (12-layer transformer, hidden=1024), outputting $L_{\text{macro}} = M \times H_{\text{chunk}}$ coarse tokens at low frequency (per step $D$-dimensional action, $N$ bins per dimension).
- **System 1 (Action Refiner)**: Diffusion transformer + independent SigLIP vision encoder $\mathcal{E}_{\text{vis}}$, denoising continuous actions at high frequency by taking slices from the intent buffer as conditions.
- **Bridge**: FIFO Intent Buffer $\mathcal{Q}$ + a learnable codebook $\mathbf{E} \in \mathbb{R}^{N \times D}$ that maps discrete bins to embeddings.
- **Output**: Continuous actions $\mathbf{a}_t \in [-1, 1]^D$.

### Key Designs

1.  **Hybrid Action Space + Coarse-Grained Directional Discretization**:
    - **Function**: Quantizes each dimension of the action into $N$ uniform bins, $y_{t,i}^{gt} = \mathrm{clip}(\lfloor (a_{t,i}+1)/2 \times N \rfloor, 0, N-1)$, but **intentionally uses a very small $N \ll 256$**.
    - **Mechanism**: Previous discrete VLAs (OpenVLA) pursued $N=256$ to approximate continuous control, which bloated the token space beyond the VLM's capacity. Ours reduces $N$ significantly (ablations show an inverted U-curve where moderate $N$ performs best). Tokens represent "coarse directions (macro-intent)," allowing the VLM's semantic abstraction to hit a sweet spot, while quantization loss is compensated by the subsequent continuous refiner.
    - **Design Motivation**: The authors identify the "**learning complexity equipartition**" principle—performance peaks when the learning difficulty is balanced between the two subsystems.

2.  **Parallel Coarse-Action Head + Adaptive Intent Injection (Curriculum to solve teacher forcing bias)**:
    - **Function**: Uses $K$ learnable queries to perform self-attention with VLM outputs $\mathbf{H}_t$, predicting coarse tokens for the entire chunk in parallel: $\mathbf{Z}_{\text{act}} = \mathrm{SelfAttn}([\mathbf{Q}_{\text{act}}; \mathbf{H}_t])_{0:K}$, followed by Linear+Softmax to output $P(\mathbf{a}_t^c)$.
    - **Mechanism**: During training, System 1 requires $\mathbf{e}_{\text{intent}}$ as a condition. Since System 2 might be unreliable early on, using its predictions directly introduces noise that harms System 1 training; conversely, standard teacher forcing causes System 1 to fail at inference when it encounters noisy predictions. Ours uses a **dynamic curriculum**: when System 2's accuracy is below a threshold $\tau$, it uses GT codebook embeddings; once above $\tau$, it switches to sampling from $P(\mathbf{a}_t^c)$.
    - **Design Motivation**: The parallel head ensures inference speed (no AR); the curriculum bridges the train-test gap and enables the refiner to internalize tolerance for planner deviations.

3.  **Asynchronous Execution with Intent Buffer + Horizon Expansion**:
    - **Function**: System 2 predicts macro tokens for $L_{\text{macro}} = M \cdot H_{\text{chunk}}$ future steps into a FIFO buffer $\mathcal{Q}$. System 1 pops $H_{\text{chunk}}$ tokens as conditions at each control step. System 2 remains dormant for the next $M-1$ chunks.
    - **Mechanism**: Unlike traditional dual-systems (GR00T-N1) that use static latents prone to environment decoupling, our **predictive intent buffer** allows System 2 to plan the discrete direction sequence for the entire horizon. System 1's slices are **time-synchronized**, preventing lagging. Discrete tokens are also physically interpretable (e.g., "+x large, +y small"), offering more transparency than black-box latents.
    - **Design Motivation**: Amortizes the expensive VLM computation across $M$ chunks, decoupling control frequency from planning frequency—a critical engineering breakthrough for real-time robotics.

### Loss & Training
Jointly optimizes two losses:
- **Planner loss**: $\mathcal{L}_{\text{plan}} = \mathcal{L}_{\text{CE}}(P(\mathbf{a}_t^c), \mathbf{y}_t^{gt})$ (Standard cross-entropy, $N$-way classification per dimension).
- **Refiner loss**: $\mathcal{L}_{\text{diff}} = \mathbb{E}_{k, \mathbf{x}_0, \epsilon}[\|\epsilon - \epsilon_\theta(\mathbf{x}_k, \mathbf{F}_t^{\text{geo}}, \mathbf{e}_{\text{intent}})\|^2]$ (Standard DDPM noise prediction).
- **Total loss**: $\mathcal{L}_{\text{total}} = \lambda_{\text{diff}} \mathcal{L}_{\text{diff}} + \lambda_{\text{plan}} \mathcal{L}_{\text{plan}}$, with weights calibrated for gradient magnitude balance.
- **Key Hyperparameters**: $M=2$ (horizon expansion factor), $H_{\text{chunk}}=5$, $L_{\text{macro}}=10$, moderate $N$.
- **Training**: All experiments are conducted **without massive robot data pre-training**, fine-tuning directly from InternVL2.5-2B + SigLIP.

## Key Experimental Results

### Main Results: LIBERO Benchmark (4 task suites, 50 rollouts per task, 500 total)

| Method | Action Space | Spatial | Object | Goal | Long | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OpenVLA | Discrete | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0-FAST | Discrete | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| DD-VLA | Discrete | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 |
| Diffusion Policy | Continuous | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | Continuous | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| GR00T-N1 | Continuous | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| GO-1 | Continuous | 96.2 | 97.8 | 96.0 | 89.2 | 94.8 |
| F1 | Continuous | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| GE-Act | Continuous | 98.2 | 97.6 | 95.8 | 94.4 | 96.5 |
| π0 | Continuous | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | Continuous | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **Libra-VLA (Ours)** | **Hybrid** | **98.6** | **99.4** | **98.0** | **92.8** | **97.2** |

Highlights: Object 99.4 (validating refiner geometric precision), Long 92.8 (validating planner long-range guidance), Average 97.2 (Rank 1).

### LIBERO-Plus Robustness (7 perturbations: Camera/Robot/Lang/Light/BG/Noise/Layout)

| Method | Camera | Robot | Lang | Light | BG | Noise | Layout | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot Transfer** | | | | | | | | |
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| π0-FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| **Ours (Hybrid)** | **68.9** | **48.8** | **92.7** | **97.9** | **93.4** | **86.3** | **77.5** | **79.5** |
| **Supervised Fine-Tuning** | | | | | | | | |
| π0.5* | 70.3 | 41.7 | 81.1 | 97.3 | 94.6 | 71.8 | 84.9 | 75.7 |
| OpenVLA-OFT+ | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | 79.6 |
| **Ours (Hybrid)** | **94.5** | 41.8 | — | — | — | — | — | — |

**Key Finding**: In zero-shot transfer, Libra-VLA outperforms the runner-up OpenVLA-OFT by **+9.9 points** (79.5 vs 69.6), with notable gains in Language (92.7 vs 79.5) and Light (97.9 vs 88.7), demonstrating that the hybrid space significantly reduces reliance on the training distribution.

### Ablation Study

| Configuration | Trend | Description |
| :--- | :--- | :--- |
| Varying bin count $N$ | **Inverted U-curve** | Low $N$ lacks detail; high $N$ makes VLM learning difficult. Verification of "learning complexity equipartition." |
| Removing Adaptive Intent Injection | Performance drops | Train-test gap; refiner is unprepared for planner noise. |
| Removing independent SigLIP | Performance drops | Verification of "feature squeezing" bottleneck seen in FiS-VLA. |
| $M=1$ (Synchronous) | High latency | Asynchronous design primarily provides latency benefits. |
| Static latent bridge | Long-horizon drops | Predictive intent buffer mitigates lagging. |

## Key Findings
- The **inverted U-curve** is the most valuable methodological discovery: elevating "hyperparameter $N$" to the "learning complexity equipartition principle" provides clear guidance for VLA design.
- The **+10 point zero-shot gain** on LIBERO-Plus indicates that hybrid action space is more than just a performance trick; it significantly improves OOD generalization because macro-intents are naturally robust to environment perturbations.
- **SoTA without large-scale pre-training**: Unlike π0/GR00T which require massive robot data, Ours wins by fine-tuning InternVL2.5.
- **Latency Advantage**: Asynchronous execution amortizes VLM costs, significantly reducing measured latency.

## Highlights & Insights
- **Hierarchy in "Action Representation Space"** is a paradigm shift: previous hierarchical VLAs decomposed the temporal axis (waypoints/sub-instructions). Ours explores the orthogonal discrete/continuous dichotomy.
- **"Learning Complexity Equipartition" Principle**: This design philosophy, backed by the **empirical** inverted U-curve, serves as a "Rosetta Stone" for dual-system VLA architectures.
- **Predictive Intent Buffer as a Key to Asynchrony**: Using future discrete direction sequences instead of static latents ensures time-synchronized guidance for System 1.
- **Inter-system Communication via Discrete Tokens**: Unlike black-box latents, discrete tokens are physically interpretable, aiding debugging and safety audits.
- **Independent SigLIP for Refiner**: Assigning a high-resolution vision encoder to the fast system resolves the "feature squeezing" bottleneck in a cost-effective way.

## Limitations & Future Work
- **Task/Dataset-dependent optimal $N$**: No automated method for selecting $N$ is provided.
- **Static Hyperparameter $M$**: If $M$ is too large, macro-intents become outdated; if $M$ is too small, VLM calls become too frequent.
- **Uniform Discretization Assumption**: Assumes dimensions are decomposable; might fail for highly coupled multi-joint coordination (e.g., dexterous hands).
- **Scale and Domain**: Validated primarily on LIBERO/LIBERO-Plus; testing on industrial-grade long-horizon tasks or massive data scaling remains for the future.

## Related Work & Insights
- **vs. OpenVLA / π0-FAST (Discrete AR VLA)**: They use $N=256$ to approximate continuous control; Ours uses $N \ll 256$ to treat discretization as a semantic abstraction tool.
- **vs. π0 / GR00T-N1 (Continuous Diffusion VLA)**: They force the VLM to handle continuous output directly; Ours uses macro-intents to anchor the search space, letting diffusion learn only residuals.
- **vs. HAMSTER / Hi Robot (Temporal Hierarchical VLA)**: They decompose time; Ours decomposes the action **representation**, making them orthogonal and potentially combinable.
- **vs. HybridVLA**: HybridVLA is a parallel structure; Ours is a strictly hierarchical coarse-to-fine dependency.
- **vs. FiS-VLA**: Ours decouples features using an independent SigLIP encoder.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Decoupling hierarchy in the action representation space is a significant insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong results on LIBERO and robustness benchmarks; lacks a scaling curve.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear justifications for the architecture and findings.
- Value: ⭐⭐⭐⭐⭐ Provides actionable principles for dual-system VLA design with impressive zero-shot gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](../../AAAI2026/robotics/affordance-guided_coarse-to-fine_exploration_for_base_placem.md)
- [\[ICML 2026\] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](../../ICML2026/robotics/any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)
- [\[ICML 2026\] Dual Advantage Fields](../../ICML2026/robotics/dual_advantage_fields.md)
- [\[ICLR 2026\] Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](../../ICLR2026/robotics/sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)
- [\[ICML 2026\] Dual Quaternion SE(3) Synchronization with Recovery Guarantees](../../ICML2026/robotics/dual_quaternion_se3_synchronization_with_recovery_guarantees.md)

</div>

<!-- RELATED:END -->
