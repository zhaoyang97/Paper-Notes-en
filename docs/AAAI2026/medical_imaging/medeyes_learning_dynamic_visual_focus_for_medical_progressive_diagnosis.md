---
title: >-
  [Paper Note] MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis
description: >-
  [AAAI 2026][Medical Imaging][Medical VQA] MedEyes is a hybrid-policy reinforcement learning framework that introduces a Gaze-guided Reasoning Navigator (GRN) to simulate the "scan-and-drill" visual search pattern of clin…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Medical VQA"
  - "Reinforcement Learning"
  - "Visual Chain-of-Thought"
  - "Dynamic Attention"
  - "GRPO"
date: 2026-05-08
content_hash: 71c30cf14c424b06
---

# MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis

**Conference**: AAAI 2026
**arXiv**: [2511.22018](https://arxiv.org/abs/2511.22018)
**Code**: [GitHub](https://github.com/zhcz328/MedEyes)
**Area**: Medical Imaging
**Keywords**: Medical VQA, Reinforcement Learning, Visual Chain-of-Thought, Dynamic Attention, GRPO

## TL;DR
MedEyes is a hybrid-policy reinforcement learning framework that introduces a Gaze-guided Reasoning Navigator (GRN) to simulate the "scan-and-drill" visual search pattern of clinical physicians. Combined with a Confidence Value Sampler (CVS) and dual-stream GRPO optimization, the framework enables dynamic visual focus for progressive medical diagnostic reasoning, achieving an average improvement of 8.5 pp across five medical VQA benchmarks.

## Background & Motivation

Recent advances in medical vision-language models (VLMs) have demonstrated progress in question answering and report generation, yet significant gaps remain in diagnostic scenarios requiring progressive visual reasoning. The limitations of existing approaches are as follows:

**SFT Overfitting**: Supervised fine-tuning captures task knowledge but tends to overfit memorized reasoning trajectories, resulting in poor generalization and ambiguous responses in unseen clinical scenarios.

**Text-only CoT Lacks Visual Grounding**: Textual reasoning steps lack explicit alignment with visual evidence, leading to information loss and visual hallucination.

**Advantage Collapse in Pure On-policy RL**: When a model's initial capability is limited, autonomous exploration easily converges to local optima, producing reasoning paths that appear superficially coherent but are clinically incorrect — a phenomenon termed "cognitive traps."

**Limitations of Naive Behavior Cloning**: Simple imitation of expert trajectories replicates action sequences without capturing the underlying reasoning logic.

**Core Problem**: How can a model acquire the ability to perform progressive visual focusing and iterative diagnostic refinement analogous to expert physicians?

The paper's key insight is to introduce structured off-policy expert trajectories as "cognitive anchors," combined with on-policy autonomous exploration, to balance imitation of expert behavior with independent discovery. Critically, each reasoning step is explicitly grounded to a visual region, establishing a consistent mapping between "precise observation" and "structured reasoning."

## Method

### Overall Architecture

MedEyes is a hybrid-policy RL framework consisting of two collaborative streams:
- **On-policy Exploration Stream**: The policy model $\pi_\theta$ autonomously samples diagnostic trajectories.
- **Off-policy Guidance Stream**: Expert trajectories are constructed as cognitive anchors via GRN and CVS.

Medical visual reasoning is formalized as a Markov Decision Process: given a medical image $I$ and query $q$, the policy generates a diagnostic trajectory $\tau = [n_1, n_2, \ldots, n_T, a]$, where each step $n_t = \langle s_t, \mathcal{G}_t \rangle$ comprises a textual cognition $s_t$ and a visual anchor $\mathcal{G}_t$ (bounding box coordinates), ultimately producing a diagnostic answer $a$.

### Key Designs

1. **Gaze-guided Reasoning Navigator (GRN)**:

    - **Function**: Simulates the visual search pattern of clinical physicians by generating structured expert trajectories through dual-mode exploration.
    - **Mechanism**: Maintains a ternary attention state $\psi_t = (\mathcal{R}_t, \mathcal{C}_t, \mathcal{F}_t)$, representing the candidate region set, confidence distribution, and exploration mode, respectively. Candidate regions are generated via region-level VQA queries to a large-scale multimodal expert model (MedPLIB). State transitions $\psi_{t+1} = \mathcal{T}(\psi_t, a_t, o_t)$ are driven by two complementary modes:
        - **Scan Mode** ($\mathcal{F}_t = \text{global}$): Prompts the expert model to localize all anomalous regions in the image, producing global candidates.
        - **Drill Mode** ($\mathcal{F}_t = \text{local}$): Performs targeted analysis on specific candidate regions to generate refined confidence scores.
    - **Mode-switching Rule**: The confidence change rate is computed as $\Delta c = \frac{c_{t+1}(r_i) - c_t(r_i)}{c_t(r_i) + \epsilon}$. If $\Delta c \geq \delta$, drilling continues (sufficient information gain); otherwise, the mode switches back to scanning (information saturation in the current region).
    - **Design Motivation**: Directly inspired by eye-tracking studies of radiologists — expert diagnosis begins with a systematic scan to localize suspicious regions, followed by in-depth analysis of specific areas.

2. **Confidence Value Sampler (CVS)**:

    - **Function**: Generates diverse yet reliable exploration paths from multi-round GRN trajectories.
    - **Mechanism**: Applies nucleus sampling at each decision step, sampling from the top-$p_0$ confidence regions: $\mathcal{P}_{\text{nucleus}} = \{a_i : \sum_{j=1}^i P(a_j|\psi_t) \leq p_0\}$, producing $N_\text{expert}=6$ variable-length trajectories. Termination conditions: local confidence exceeds threshold $\xi=0.85$ or maximum length $T_\max=4$ is reached.
    - **Design Motivation**: A single expert trajectory is insufficient to cover the diversity of diagnostic strategies. Nucleus sampling introduces diversity while maintaining reliability. Adaptive trajectory length reflects differences in diagnostic complexity across cases.

3. **Dual-stream GRPO Optimization**:

    - **Function**: Decouples on-policy and off-policy learning signals to prevent reward assimilation and entropy collapse.
    - **Mechanism**:
        - **Source-adaptive Importance Ratio**: On-policy trajectories use $\rho_i^\theta = \pi_\theta(\tau_i|I,q) / \pi_{\theta_\text{old}}(\tau_i|I,q)$; off-policy trajectories use $\rho_i^\theta = \pi_\theta(\tau_i|I,q) / \pi_\text{expert}(\tau_i|I,q)$ (where $\pi_\text{expert}=1$).
        - **Advantage Decoupling**: Normalization statistics are computed independently over on-policy and off-policy data: $A_i = \frac{R(\tau_i) - \mu^{s(i)}}{\sigma^{s(i)} + \varepsilon}$.
    - **Design Motivation**: Unified normalization would allow high-reward expert trajectories to dominate and suppress the on-policy learning signal, causing gradient dominance. Decoupling preserves independent learning rates for both streams, enabling learning from experts without sacrificing adaptability to novel scenarios.

4. **Multi-component Verifiable Reward Function**:

    - **Accuracy Reward** $r_\text{acc}$: Correctness of the answer (indicator function), $\lambda_\text{acc}=0.7$.
    - **Grammar Reward** $r_\text{grammar}$: Correctness of the reasoning–action–perception tag structure (binary), $\lambda_\text{grammar}=0.2$.
    - **Diversity Reward** $r_\text{div}$: Number of unique regions explored plus the proportion of region pairs with IoU below a threshold, $\lambda_\text{div}=0.1$.

### Loss & Training

A PPO clip-style objective is employed:

$$\mathcal{J}(\theta) = \frac{1}{N}\sum_{i=1}^N \frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|} \min(\rho_{i,t}^\theta A_i, \text{clip}(\rho_{i,t}^\theta, 1-\epsilon, 1+\epsilon)A_i)$$

The model is built on Qwen2.5-VL-3B, trained with the AdamW optimizer at a learning rate of $1 \times 10^{-6}$ for 3 epochs, with a rollout batch size of 98 and 8 rollouts per prompt. Training is conducted on 6 RTX 3090 GPUs.

## Key Experimental Results

### Main Results

Evaluation is conducted on five medical VQA benchmarks: VQA-RAD, SLAKE, PathVQA, PMC-VQA, and MMMU (Health).

| Dataset | MedEyes | GMAI-VL (Medical SOTA) | MedVLM-R1 (RL SOTA) | GPT-4o |
|--------|---------|---------------------|----------------------|--------|
| VQA-RAD | **70.7** | 64.6 | 61.4 | 54.2 |
| SLAKE | **79.1** | 71.9 | 65.9 | 50.1 |
| PathVQA | **64.8** | 47.2 | 55.2 | 59.2 |
| PMC-VQA | **55.3** | 52.3 | 44.8 | 40.8 |
| MMMU* | **59.7** | 51.2 | 35.5 | - |
| Average | **65.9** | 57.4 (-8.5) | 52.5 (-13.4) | 51.1 (-14.8) |

### Ablation Study

| Configuration | VQA-RAD | SLAKE | PathVQA | Avg. | Notes |
|------|---------|-------|---------|------|------|
| MedEyes (Full) | 70.7 | 79.1 | 64.8 | 71.5 | Baseline |
| w/o GRN | 62.4 | 69.8 | 56.2 | 62.8 | −8.7 pp |
| w/o CVS | 65.3 | 73.5 | 59.1 | 66.0 | −5.5 pp |
| w/o Off-policy | 61.2 | 67.4 | 54.3 | 61.0 | −10.5 pp, largest drop |
| Scan Mode Only | 66.8 | 74.2 | 58.7 | 66.6 | Degraded on fine-grained tasks |
| Drill Mode Only | 64.5 | 71.9 | 60.3 | 65.6 | Lacks systematic exploration |

### Key Findings

- **Off-policy expert trajectories are the most critical component**: Removing them causes an average drop of 10.5 pp, demonstrating that pure on-policy learning is insufficient for medical reasoning.
- **Both scan and drill modes are indispensable**: Each mode alone underperforms on different task types; their complementary nature is essential for covering the full diagnostic workflow.
- **Training dynamics exhibit an "exploration–efficiency" transition**: Trajectory length first increases from 2.1 to 3.0 steps (learning when visual grounding is needed), then decreases to 2.6 steps (learning to reason more efficiently).
- **Six expert trajectories and a reasoning length of 3 steps represent the optimal trade-off**: More trajectories or longer sequences yield diminishing or negative returns.
- **Trajectory quality is critical**: GRN+CVS trajectories substantially outperform random sampling (+12.8 pp) and DeepSeek-R1-generated trajectories (+7.6 pp).

## Highlights & Insights

- **Clinically Aligned Design**: The scan-and-drill dual-mode strategy directly mirrors the visual search patterns identified in radiologist eye-tracking studies, serving as an exemplary instance of human cognition-inspired AI design.
- **Novel Hybrid-policy Training Paradigm**: Rather than a simple off-policy pre-training followed by on-policy fine-tuning, the framework employs dual-stream synchronous training with decoupled advantage normalization, demonstrating notable technical depth.
- **Strong Interpretability**: Each reasoning step is explicitly bound to an image region, and attention heatmaps intuitively visualize the progressive focusing process.
- **Inspiring "Cognitive Anchor" Concept**: Off-policy trajectories as cognitive anchors help models with limited initial capability escape local optima — a principle generalizable to other RL settings.

## Limitations & Future Work

- **Quantitative Measurement**: Tasks requiring pixel-to-centimeter calibration, such as tumor size measurement, are prone to large errors when using ratio-based estimation.
- **Fine-grained Concept Discrimination**: The model occasionally confuses morphologically similar pathological subtypes (e.g., aneurysm vs. dissecting aneurysm).
- **Dependence on External Expert Models**: GRN relies on MedPLIB as a visual expert, and trajectory quality is bounded by the capability of this external model.
- **Small Base Model Scale**: Validation is limited to a 3B parameter model; performance when scaled to larger models remains unexplored.
- **Computational Cost**: Each prompt requires 8 rollouts plus 6 expert trajectories, resulting in non-trivial training resource demands.

## Related Work & Insights

- Compared to visual CoT methods such as GRIT and DeepEyes, MedEyes addresses the training difficulty arising from insufficient initial model capability through off-policy guidance.
- The advantage decoupling mechanism in dual-stream GRPO is generalizable to any mixed on/off-policy RL scenario.
- The paradigm of using eye-tracking data to drive AI design holds broader potential in medical image analysis.
- Future directions include richer tool use (measurement, contrast enhancement), multi-turn conversational diagnosis, and integration with real-world clinical workflows.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_fusion_for_medical_visual_question_a.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[AAAI 2026\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)
- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](../../ICLR2026/medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)
- [\[ICLR 2026\] Boosting Medical Visual Understanding From Multi-Granular Language Learning](../../ICLR2026/medical_imaging/boosting_medical_visual_understanding_from_multi-granular_language_learning.md)

</div>

<!-- RELATED:END -->
