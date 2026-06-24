---
title: >-
  [Paper Note] Reactivation: Empirical NTK Dynamics Under Task Shifts
description: >-
  [ICML 2025][Interpretability][Neural Tangent Kernel] This work presents the first systematic empirical study on NTK dynamics in continual learning, finding that task shifts consistently trigger abrupt deviations in the NTK. Even in the lazy learning regime, NTK norm, velocity, and alignment metrics deviate sharply at task boundaries, revealing a feature learning phenomenon termed "reactivation." The driving factors are precisely pinpointed by distinguishing between conceptual…
tags:
  - "ICML 2025"
  - "Interpretability"
  - "Neural Tangent Kernel"
  - "Continual Learning"
  - "Feature Learning"
  - "Distribution Shift"
  - "Reactivation Phenomenon"
date: 2026-05-08
content_hash: ae67acd8f38798a4
---

# Reactivation: Empirical NTK Dynamics Under Task Shifts

**Conference**: ICML 2025  
**arXiv**: [2507.16039](https://arxiv.org/abs/2507.16039)  
**Code**: None  
**Area**: Learning Theory / Continual Learning  
**Keywords**: Neural Tangent Kernel, Continual Learning, Feature Learning, Distribution Shift, Reactivation Phenomenon

## TL;DR

This work presents the first systematic empirical study on NTK dynamics in continual learning, finding that task shifts consistently trigger abrupt deviations in the NTK. Even in the lazy learning regime, NTK norm, velocity, and alignment metrics deviate sharply at task boundaries, revealing a feature learning phenomenon termed "reactivation." The driving factors are precisely pinpointed by distinguishing between conceptual and frequency distribution shifts.

## Background & Motivation

**Background**: Neural Tangent Kernel (NTK) theory is a core tool for understanding neural network learning dynamics. In the lazy/kernel regime, the NTK remains static during training, and the network behaves as a linear kernel machine; in the rich/feature learning regime, the evolution of the NTK is a necessary condition for feature learning. Recent studies on NTK dynamics have revealed several key phenomena: kernel alignment (aligning with the task direction), progressive sharpening (increase in curvature), etc.

**Limitations of Prior Work**: All existing analyses of NTK dynamics are limited to the **single-task setting**, assuming that the data distribution remains constant during training. Several theoretical works analyzing continual learning (Karakida & Akaho 2022; Doan et al. 2021; Bennani et al. 2020) **assume that the NTK remains static under distribution shifts** during derivation, but this critical assumption has never been validated.

**Key Challenge**: The core feature of continual learning is that the data distribution changes over time, while the fundamental assumption of NTK theory is that the data distribution is stationary. If the NTK changes under distribution shifts, many theoretical analyses of continual learning based on the static NTK assumption will be called into question.

**Goal**: To systematically and empirically examine the dynamic behavior of the NTK in continual learning, particularly whether and how the NTK changes during task shifts, and which factors govern these changes.

**Key Insight**: To design carefully controlled experiments to examine the effects of network width, learning rate, task similarity, and types of distribution shifts (conceptual vs. frequency) on NTK dynamics respectively, using four complementary NTK metrics for a comprehensive diagnostic.

**Core Idea**: Task shifts consistently trigger a "reactivation" of the NTK. Even in the lazy regime, the NTK undergoes a "checkmark-shaped" trajectory of a sharp drop followed by recovery at task boundaries, and the intensity of this phenomenon is controlled by the semantic novelty introduced by the new task (rather than frequency changes).

## Method

### Overall Architecture

This paper is a purely empirical study and does not propose new models or algorithms. Experiments are conducted on CIFAR-10/100 and ImageNet-100 for continual image classification, splitting classes into multiple tasks for sequential training. The core innovation lies in the experimental design: systematically measuring NTK dynamics across carefully controlled variables using four NTK metrics, all evaluated on the **data of the first task** (to observe the impact of new tasks on existing representations).

### Key Designs

1. **Four Complementary NTK Metrics System**
    - Function: Fully describe the dynamic behavior of the NTK from different perspectives.
    - Mechanism: (a) Spectral norm of the kernel (maximum eigenvalue), which controls the convergence rate of feature modes; (b) Kernel distance $S(\Theta, \Theta') = 1 - \text{CKA}(\Theta, \Theta')$, which measures the degree of deviation of the NTK from its initial state; (c) Kernel velocity $v(t) = S(\Theta_t, \Theta_{t+dt})/dt$, which quantifies the instantaneous rate of change of the NTK at time $t$; (d) Kernel alignment $A(t) = \text{CKA}(\Theta_t, \mathbf{y}\mathbf{y}^\top)$, which measures the similarity between the NTK and the target label kernel.
    - Design Motivation: A single metric might miss crucial dynamics. Kernel velocity reveals when changes occur, norm reveals the magnitude, and alignment reveals the direction.

2. **Distinction Experiments between Conceptual and Frequency Distribution Shifts**
    - Function: Pinpoint the types of factors that drive NTK changes.
    - Mechanism: Experiment 1—gradually introducing new classes, task similarity $= |\mathcal{D}_0 \cap \mathcal{D}_i| / |\mathcal{D}_0 \cup \mathcal{D}_i|$; Experiment 2—fixing the set of classes and changing the class frequency ratio $\mathcal{D}_\alpha = (1-\alpha)\tilde{\mathcal{D}}_0 + \alpha\tilde{\mathcal{D}}_1$.
    - Design Motivation: Disentangle the two types of distribution shifts—"presence of new concepts" and "only frequency changes"—to reveal that semantic novelty is the true driver of NTK changes.

3. **Dual-line Comparison between Lazy and Feature Learning Regimes**
    - Function: Verify the critical assumption that "wide networks behave as fixed kernel learners under non-stationary settings."
    - Mechanism: The lazy regime is achieved by scaling the learning rate inversely with width ($\eta \propto 1/N$); the feature learning regime uses Kaiming uniform initialization to reflect common practices in continual learning. Systematic testing is conducted across widths from 64 to 2048.
    - Design Motivation: If reactivation only occurred in the feature learning regime, it might be expected; however, if it also appears in the lazy regime, it directly challenges the core assumptions of NTK theory.

### Loss & Training

Standard cross-entropy loss is used to train image classification. The focus of the experiments is not on training strategies but on observing NTK dynamics. Two-stage sequential training: first training to convergence on Task 1 (5 classes), then shifting to Task 2 (another 5 classes) to continue training. Results for multiple task shifts are also reported in the appendix.

## Key Experimental Results

### Key Findings: NTK Dynamics at Task Shifts (CIFAR-10)

| Phenomenon | Lazy Regime | Feature Learning Regime |
|------|---------|-------------------|
| Kernel velocity at task shift | Shows a noticeable spike | Shows a larger spike |
| NTK norm trajectory | "Checkmark-shaped": sharp drop followed by recovery | Also "checkmark-shaped" |
| Kernel alignment change | Sharp deflection | Sharp deflection |
| Consistency across widths (64-2048) | Consistently appears across all widths | Consistently appears across all widths |

### Influence of Task Similarity on NTK Dynamics

| Type of Distribution Shift | Magnitude of NTK Norm Change | Spike Size of Kernel Velocity | Key Feature |
|------------|-------------|-------------|---------|
| Conceptual (new classes, 0% overlap) | Maximum | Maximum | Most pronounced checkmark shape |
| Conceptual (50% class overlap) | Medium | Medium | Monotonically decreasing relationship |
| Conceptual (100% overlap = no change) | None | None | Stable |
| Frequency (changing class ratios) | Minimal | Extremely low | Smooth evolution, no discontinuity |

### Impact of Learning Rate

| Learning Rate | Checkmark Shape | Recovery Speed | Reason |
|--------|---------|---------|------|
| High | More pronounced | Slower | Overfitting |
| Medium | Moderate | Fastest | Best balance |
| Low | Concentrated in the first few steps | Slower | Undertraining |

### Key Findings

- Even in the lazy regime, task shifts consistently trigger NTK reactivation, challenging the assumption that "wide networks are fixed kernel learners under non-stationary settings."
- Conceptual distribution shifts (introducing new classes) show a clear monotonic relationship with NTK changes, with a diminishing returns effect: the first few new classes cause disproportionately large changes.
- Frequency distribution shifts (changing class proportions) do not trigger reactivation—the kernel velocity remains low, and the NTK evolves smoothly.
- The "checkmark-shaped" (V-shape) trajectory consistently appears across all widths, learning rates, and training durations, hinting at a shared underlying mechanism.
- The same conclusions were confirmed on ImageNet-100, ruling out dataset specificity.
- During sequential transitions across multiple tasks, reactivation is triggered at every task boundary (appendix experiments).

## Highlights & Insights

- **First to systematically reveal the structured evolutionary patterns of NTK in non-stationary settings**: The reactivation phenomenon was completely undocumented before, serving as an important empirical complement to NTK theory.
- **Precisely distinguished the different effects of conceptual and frequency shifts**: Not all distribution shifts are equally important; semantic novelty is the key driving force. This has direct guiding implications for the algorithm design in continual learning—special treatment may be required when a new task with high semantic novelty is detected.
- **Directly challenged the validity of theoretical assumptions**: The static NTK assumption relied upon by multiple theoretical works in continual learning is empirically refuted; in particular, the discovery of reactivation in the lazy regime is surprising.
- **Rigor of experimental design**: By isolating variables (width, learning rate, shift type, shift intensity), every conclusion is supported by precise control experiments.

## Limitations & Future Work

- It is a purely empirical study and does not propose a theoretical framework to explain the reactivation phenomenon—why does a task shift trigger NTK changes? What are the underlying mechanisms?
- It only uses simple fully connected and convolutional networks, without involving modern architectures such as Transformers.
- It does not explore the impact of practical continual learning algorithms (e.g., EWC, PackNet) on NTK dynamics.
- It only considers sequences of two or a few tasks; whether reactivation accumulates or decays in longer task sequences remains unknown.
- It is limited to image classification; NTK dynamics in other domains like NLP may exhibit different patterns.
- It does not discuss whether the reactivation phenomenon can be utilized—for example, whether continual learning can be improved by actively managing NTK dynamics.

## Related Work & Insights

- **Fort et al. (2020)**: Empirically studied NTK dynamics under a single task, finding that the NTK changes significantly in early training stages—this paper extends this to multi-task settings.
- **Baratin et al. (2021)**: Kernel alignment study, finding that NTK alignment with the task direction improves learning efficiency.
- **Cohen et al. (2021)**: Progressive sharpening phenomenon (edge of stability), revealing the dynamic changes of curvature during training.
- **Karakida & Akaho (2022)**: Analyzed continual learning based on the static NTK assumption—experiments in this paper directly query the validity of this assumption.
- Insights: Future continual learning theories need to explicitly incorporate distribution shifts into NTK dynamic modeling; in practice, semantic overlap might serve as a signal to predict representation changes.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First to systematically study NTK dynamics in continual learning; the reactivation phenomenon is a brand-new discovery)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Very rigorous controlled variable experimental design, validated across multiple datasets, widths, and learning rates)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, detailed descriptions of observations, but slightly lacking in depth due to the lack of theoretical explanations)
- **Value**: ⭐⭐⭐⭐ (Raises important questions regarding the theoretical foundations of continual learning, pointing the direction for future theoretical developments)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond](../../ICCV2025/interpretability/learnable_fractional_reaction-diffusion_dynamics_for_under-display_tof_imaging_a.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](../../ACL2025/interpretability/an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ICML 2025\] On the Effect of Uncertainty on Layer-wise Inference Dynamics](on_the_effect_of_uncertainty_on_layer-wise_inference_dynamics.md)
- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](../../AAAI2026/interpretability/pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](../../NeurIPS2025/interpretability/are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)

</div>

<!-- RELATED:END -->
