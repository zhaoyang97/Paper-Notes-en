---
title: >-
  [Paper Note] AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift
description: >-
  [NeurIPS 2025][AI Safety][Adaptive Sensing] Inspired by biological sensory systems, this position paper argues that AI research must shift from simply scaling models to optimizing inputs—by dynamically adjusting sensor-l…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Adaptive Sensing"
  - "Paradigm Shift"
  - "Sensor Optimization"
  - "Closed-Loop Perception"
  - "Embodied AI"
date: 2026-05-08
content_hash: 65163a8dfa17c360
---

# AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift

**Conference**: NeurIPS 2025
**arXiv**: [2507.07820](https://arxiv.org/abs/2507.07820)  
**Code**: None (Position Paper)  
**Area**: Perception Systems / Embodied AI
**Keywords**: Adaptive Sensing, Paradigm Shift, Sensor Optimization, Closed-Loop Perception, Embodied AI

## TL;DR

Inspired by biological sensory systems, this position paper argues that AI research must shift from simply scaling models to optimizing inputs—by dynamically adjusting sensor-level parameters (exposure, gain, multimodal configuration, etc.) to produce inputs most favorable to the model. Under ideal sensor adaptation, a small model (EfficientNet-B0, 5M parameters) can outperform a large model (OpenCLIP-H, 632M parameters), and the paper proposes a progressive formalization framework ranging from single-shot perception to closed-loop perception–action coupling.

## Background & Motivation

Current AI progress relies primarily on scaling model size and training datasets, but this trajectory faces fundamental sustainability challenges:

**Environmental Cost**: Training GPT-3 alone consumed approximately 1.287 GWh of electricity and emitted roughly 552 tonnes of CO₂—equivalent to driving a car over one million kilometers. As models continue to grow, environmental costs scale exponentially.

**Equity Concerns**: Only well-funded institutions can train and deploy frontier models, concentrating innovation opportunities among a small number of elite organizations and widening the global digital divide.

**Generalization Failures**: Models trained on large static datasets frequently fail under real-world covariate shifts such as sensor variation, lighting changes, and weather conditions. Existing robustness benchmarks inadequately capture this complexity.

**Biological Systems Offer an Alternative**: The human sensory system performs extensive adaptive adjustments before and during neural processing. The pupil adjusts from 2–8 mm in diameter within 200 ms (a 16× gain change in light); saccades redirect gaze within 3–5 ms; dark adaptation restores sensitivity; and ciliary muscles accommodate focus from 10 cm to infinity. These represent solutions to perceptual problems at the *sensor level*, rather than enlarging the brain. By contrast, artificial sensors remain almost entirely static—cameras use fixed or coarsely stepped apertures, fixed quantum efficiency, and fixed color filter arrays, while microphones have only half the dynamic range of the human ear.

**Core Position**: AI needs not only a "larger brain" but also "more capable senses." Adaptive sensing—dynamically optimizing parameters at the sensor level to produce inputs most amenable to the model—cannot be replaced by post-hoc methods such as domain adaptation or test-time adaptation, because once an analog signal is digitized, information lost due to sensor configuration is irreversibly gone.

## Method

### Overall Architecture

As a position paper, this work does not present a single method but constructs a complete argument from existing evidence to a future roadmap: (1) summarizing preliminary empirical evidence for adaptive sensing via the Lens framework; (2) proposing a four-stage progressive formalization from simple to complex settings (baseline MDP → single-shot perception → continuous perception → perception–action coupling); (3) mapping cross-domain application scenarios (humanoid robotics, healthcare, autonomous driving, agriculture, environmental monitoring); and (4) analyzing technical and ethical challenges while proposing research directions.

### Key Designs

1. **Lens Framework — Existing Empirical Foundation**:
    - **Function**: The first "model-friendly" test-time input adaptation framework, validating the feasibility of adaptive sensing on image classification tasks.
    - **Mechanism**: Dynamically responds to scene characteristics based on VisiT scores, selecting the optimal sensor parameter configuration for neural networks. Evaluated on the ImageNet-ES and ImageNet-ES-Diverse benchmarks.
    - **Key Findings**: Adaptive sensing can improve accuracy by up to 47.58 percentage points without any modification to the model architecture. Under ideal sensor adaptation, EfficientNet-B0 (5M parameters) matches or exceeds OpenCLIP-H (632M parameters, trained on 160× more data). Notably, **the image optimal for the model differs from the image optimal for humans**—meaning conventional "human-vision-friendly" auto-exposure strategies are suboptimal for AI models.
    - **Design Motivation**: Lens validates the core hypothesis but is limited to single-shot static classification scenarios.

2. **Progressive Closed-Loop Perception Framework**:
    - **Function**: Extends adaptive sensing from simple single-shot classification to continuous closed-loop embodied AI scenarios.
    - **Mechanism** — Four-Stage Progressive Design:
        - **Stage 1 (Baseline MDP)**: Standard $\mathcal{M}=(S,A,P_E,R)$ with fixed sensor configuration $o_{fixed}$; no adaptation.
        - **Stage 2 (Single-Shot Adaptive Sensing)**: Augmented stochastic process $\mathcal{P}=(S,O,P'_E,Q_M)$ with no action policy; selects the optimal configuration from $k$ candidates via perceptual quality metric $Q_M$: $o^*_{t+1} = \arg\max_i Q_M(s^{(i)}_{t+1}, o^{(i)}_{t+1})$.
        - **Stage 3 (Continuous Sensing MDP)**: Extends single-shot to sequential decision-making; sensing policy $\pi_{sense}(o_{t+1}|s_t,o_t,Q_M)$ learns the optimal sensor trajectory over continuous time steps.
        - **Stage 4 (Perception–Action Coupling)**: Joint MDP $\mathcal{M}=(S,A,O,P_E,Q_M)$ simultaneously optimizes action and sensing policies, with reward $r_{t+1} = R_{task}(s_t,a_t) + \lambda Q_M(s_t,o_t)$.
    - **Design Motivation**: The intuition draws from infant development—infants progressively improve visual-motor control through continuous perception–action feedback; similarly, AI agents must couple sensor adjustment with action decisions in dynamic environments.

3. **Multimodal Adaptive Extension**:
    - **Function**: Moves beyond single sensors to achieve adaptive resource allocation across modalities.
    - **Mechanism**: Introduces a modality weight vector $w_t \in \mathbb{R}^N$ (normalized); the sensing policy jointly outputs sensor parameters and modality weights: $(o_{t+1}, w_{t+1}) = \pi_{multi-sense}(o_{t+1}, w_{t+1} | s_t, o_t, w_t, Q_M)$. For example, a humanoid robot increases the weight of toe pressure sensors when its center of mass shifts forward during standing, and increases ankle proprioceptor weight when experiencing lateral perturbations.
    - **Sparse Reward Scenarios**: Intermediate perceptual quality metrics (e.g., grasp stability $Q_{grip}$, visual alignment $Q_{vis}$) are introduced as dense feedback signals, with composite reward $R_t = R_{sparse} + \lambda_{tact} Q_{grip} + \lambda_{vis} Q_{vis}$, mitigating exploration difficulty caused by sparse rewards.

### Loss & Training

As a framework paper, no models are actually trained. The proposed reward design consists of a weighted combination of task reward and perceptual quality, where $\lambda$ controls the balance between task orientation and sensing quality. In multimodal scenarios, the per-modality weights $\lambda_{tact}$ and $\lambda_{vis}$ are treated as hyperparameters.

## Key Experimental Results

### Main Results

The core empirical data cited in this paper are drawn from prior works Lens and SenseShift6D:

| Scenario | Metric | Adaptive Sensing | Baseline | Notes |
|----------|--------|-----------------|----------|-------|
| ImageNet-ES Classification | Accuracy Gain | +47.58 pp | Standard auto-exposure | No model architecture modification |
| Model Size Comparison | Classification Accuracy | EfficientNet-B0 (5M) ≈/> OpenCLIP-H (632M) | — | 50× smaller but comparable or superior accuracy |
| 6D Pose Estimation | Precision & Stability | Multimodal Adaptive > Single-modal > Factory Default | — | Validated on SenseShift6D |
| Object Detection/Segmentation | Robustness | Adaptive Exposure > Fixed Auto-Exposure | — | Consistent gains on complex vision tasks |

### Ablation Study

| Configuration | Key Finding | Notes |
|---------------|-------------|-------|
| Model Specificity | Optimal sensor configuration differs per model | Not a one-size-fits-all setting |
| Scene Specificity | Different scenes require different sensor parameters | Dynamic adaptation is necessary |
| Relation to Model Improvement | Adaptive sensing and model improvement are synergistic | Complementary, not substitutive |
| Human vs. Model Preference | Image optimal for the model ≠ image optimal for humans | Traditional auto-exposure is suboptimal for AI |

### Key Findings

- **Adaptive sensing and model scaling are complementary orthogonal strategies**: Adaptive sensing is more effective when covariate shift dominates; model-level adaptation is more effective under semantic shift (e.g., unseen categories); combining both yields the best results.
- **Closed-loop design is necessary in dynamic environments**: Independent controllers suffice in low-dynamic settings (single-shot classification), but in embodied AI scenarios, sensing policy must be closed-loop coupled with action policy.
- **50× model compression equivalence**: Correct sensor parameter selection can compensate for a 50× parameter gap and a 160× training data gap—fundamentally challenging the "scaling is all you need" paradigm.

## Highlights & Insights

- **The biological analogy is highly persuasive**: The comparison table contrasting the human pupil's 16× light gain adjustment in under 200 ms with a camera's fixed aperture and coarse-step ISO immediately illustrates how primitive and rigid artificial sensors are.
- **The 5M vs. 632M empirical result is the most striking**: With correctly tuned sensor parameters, a model 50× smaller can outperform one trained on 160× more data. This is not merely a technical finding but a fundamental challenge to the prevailing AI development paradigm.
- **The four-stage progressive formalization is well-structured**: The progression from no adaptation → single-shot → continuous → perception–action coupling is accompanied by clear mathematical definitions and intuitive explanations, providing the research community with a complete roadmap.
- **Cross-domain application perspective**: From humanoid robotics to medical imaging to autonomous driving to agricultural monitoring, the argument for the universal applicability of adaptive sensing is well-developed.
- **The finding that "model-optimal ≠ human-optimal"** implies that decades of auto-exposure and auto-gain algorithms designed around human visual preferences may all be suboptimal for AI applications.

## Limitations & Future Work

- **Core empirical evidence is primarily from image classification**: Evidence on more complex tasks (detection, segmentation, embodied interaction, language models) is very limited; the majority of the paper consists of vision and framework rather than experimental validation.
- **The closed-loop framework is entirely conceptual**: From Stage 2 onward (continuous sensing, perception–action coupling, multimodal adaptation), there is no experimental implementation—only mathematical formalization.
- **Scalability of the sensor parameter space**: As the number of modalities and parameter dimensions increases, the search space grows exponentially; no solution for efficient exploration is discussed.
- **Dependence on hardware ecosystem cooperation**: Open API control interfaces from sensor manufacturers are required, raising commercial and intellectual property concerns.
- **Blurry boundary with domain adaptation**: The paper claims that "adaptive sensing addresses covariate shift while domain adaptation addresses semantic shift," but in practice the two are intertwined, and practical guidance on when to apply each strategy is absent.
- **Latency cost of adaptation is not discussed**: In real-time systems, the computational overhead of sensor parameter search may offset part of the performance gain.

## Related Work & Insights

- The essential distinction from domain adaptation/test-time adaptation lies in the point of intervention: adaptive sensing operates *before* signal digitization, preserving information that post-hoc methods cannot recover.
- Distinction from active perception (changing the robot's viewpoint): adaptive sensing adjusts internal sensor parameters rather than external position.
- Distinction from physics-informed simulation (PINNs/DSE): the latter simulates after the fact and cannot recover information already lost at the sensor level.
- Particularly relevant for resource-constrained settings (edge computing, wearables, micro-UAVs): rather than deploying larger models, equipping smarter sensors is a viable alternative.
- The potential synergy between adaptive sensing and model compression/efficient inference warrants deeper exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A paradigm-level perspective shift from "scaling models" to "optimizing inputs"; highly inspiring at the conceptual level.
- **Experimental Thoroughness**: ⭐⭐⭐ — As a position paper, the work focuses on argumentation and roadmap; original experiments are very limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Arguments build progressively; biological analogies are precise and compelling; formalization is clear.
- **Value**: ⭐⭐⭐⭐ — The direction is correct and important, but substantial follow-up empirical work is needed to fulfill its promises.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[NeurIPS 2025\] CTRL-ALT-DECEIT: Sabotage Evaluations for Automated AI R&D](ctrl-alt-deceit_sabotage_evaluations_for_automated_ai_rd.md)
- [\[NeurIPS 2025\] Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping](enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)

</div>

<!-- RELATED:END -->
