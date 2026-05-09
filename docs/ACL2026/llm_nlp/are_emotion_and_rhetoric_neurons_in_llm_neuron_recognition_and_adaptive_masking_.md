---
title: >-
  [Paper Note] Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering
description: >-
  [ACL 2026][LLM/NLP][emotion neurons] This paper systematically investigates the representational mechanisms of emotion and rhetoric neurons in LLMs and their intrinsic relationships. It proposes a multi-dimensional neuron recognition framework and an adaptive masking validation method, enabling targeted steering of emotion/rhetoric predictions and rhetoric-neuron-assisted emotion recognition.
tags:
  - ACL 2026
  - LLM/NLP
  - emotion neurons
  - rhetoric neurons
  - adaptive masking
  - neuron intervention
  - LLM interpretability
date: 2026-05-08
content_hash: be2254d8cba16d87
---

# Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering

**Conference**: ACL 2026
**arXiv**: [2604.17255](https://arxiv.org/abs/2604.17255)
**Code**: None
**Area**: Text Generation
**Keywords**: emotion neurons, rhetoric neurons, adaptive masking, neuron intervention, LLM interpretability

## TL;DR

This paper systematically investigates the representational mechanisms of emotion and rhetoric neurons in LLMs and their intrinsic relationships. It proposes a multi-dimensional neuron recognition framework and an adaptive masking validation method, enabling targeted steering of emotion/rhetoric predictions and rhetoric-neuron-assisted emotion recognition.

## Background & Motivation

**Background**: LLMs are increasingly capable of emotion understanding and rhetorical generation. Existing improvements rely primarily on external optimization (prompt engineering, fine-tuning), with insufficient exploration of internal representational mechanisms. The limited neuron-level studies focus exclusively on emotion neurons, neglecting rhetoric neurons and their interplay with emotion.

**Limitations of Prior Work**: Conventional neuron validation methods (zero-forcing, mean substitution) exhibit counter-intuitive behavior on emotion and rhetoric tasks — masking highly relevant target neurons paradoxically increases task accuracy rather than decreasing it, rendering reliable causal validation infeasible.

**Key Challenge**: While neuron intervention is desirable for verifying and steering emotion/rhetoric expression, existing masking methods are unreliable. Zero-forcing may trigger functional compensation (complementary neuron clusters taking over), while mean substitution fails to genuinely disrupt the specific representations encoded by core neurons.

**Goal**: (1) Systematically identify neurons associated with 6 emotion and 4 rhetoric categories; (2) propose a reliable causal validation method; (3) achieve targeted steering and cross-signal enhancement.

**Key Insight**: Building on the differential activation of neurons, the paper designs decay-based masking rather than zero-forcing, coupled with feedback optimization to ensure reliable masking outcomes.

**Core Idea**: Replace conventional hard masking with adaptive masking (dynamic selection + decay masking + feedback optimization) to achieve reliable neuron function validation and intervention.

## Method

### Overall Architecture

Built upon Llama-3.1-8B-Instruct, focusing on FFN-layer neurons. The pipeline consists of three stages: neuron recognition (activation frequency + probability normalization + entropy filtering) → adaptive masking validation (dynamic selection + decay masking + feedback optimization) → neuron intervention (targeted steering + cross-signal enhancement).

### Key Designs

1. **Neuron Recognition**:

    - Function: Locate neurons that respond selectively to specific emotion/rhetoric categories.
    - Mechanism: A three-step filtering procedure — first, compute the activation frequency $f_{u,e}$ of each FFN neuron under inputs of each emotion/rhetoric label; then normalize to activation probability $P_{u,e} = f_{u,e}/T_e$; finally, measure distributional concentration via information entropy $H = -\sum P_{u,e}\log(P_{u,e})$, selecting the top 1% neurons with lowest entropy as target neurons.
    - Design Motivation: Low entropy indicates high selectivity toward a specific category, enabling reliable discrimination across different emotions/rhetorical devices.

2. **Adaptive Masking**:

    - Function: Provide reliable causal validation, resolving the counter-intuitive phenomenon of conventional masking.
    - Mechanism: (a) Compute the activation differential $D_i = |A_{i,target} - A_{i,non\text{-}target}| / A_{all}$ for each neuron and select the top 10% with the largest differentials as core neurons; (b) apply decay masking $n'_u = n_u \times (1 - \alpha \times A_i/A_i^{max})$ to core neurons rather than zero-forcing; (c) introduce feedback optimization: if accuracy does not decrease after masking, automatically increase the decay coefficient $\alpha$ and selection threshold $\tau$.
    - Design Motivation: Decay masking avoids triggering functional compensation, while feedback optimization ensures stable and reliable masking outcomes.

3. **Neuron Intervention and Cross-Signal Enhancement**:

    - Function: Enable targeted steering of emotion/rhetoric predictions, and rhetoric-assisted emotion recognition.
    - Mechanism: For non-target-category inputs, steer predictions by injecting the functional vector of target neurons: $n'_{s_i} = n_{s_i} + \beta \times \bar{n}_{s_i}$. For cross-signal enhancement, inject the mean activation of rhetoric neurons into emotion neurons: $a_{i,joint} = a_i + \omega \cdot \bar{a}_{i,meta}$.
    - Design Motivation: Validates the causal functionality of neurons while exploring the intrinsic synergy between rhetoric and emotion.

### Loss & Training

LLM parameters remain frozen throughout; interventions are applied solely at the FFN activation level. Feedback optimization (adjusting $\alpha$ and $\tau$) is performed on the development set only.

## Key Experimental Results

### Main Results

Accuracy change ($\Delta$ACC %) under different masking methods:

| Masking Method | Happiness | Sadness | Anger | Fear | Metaphor | Sarcasm |
|----------------|-----------|---------|-------|------|----------|---------|
| Zero (zero-forcing) | +5.46 | +4.32 | +7.92 | +1.22 | +3.37 | -4.17 |
| Mean (mean substitution) | +6.79 | -1.13 | +5.84 | +5.37 | +5.62 | -5.78 |
| **Adaptive** | **-9.25** | **-8.63** | **-10.14** | **-7.85** | **-7.29** | **-10.61** |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| All-layer masking | Largest performance drop | Emotion/rhetoric relies on cross-layer collaboration |
| Top-layer masking only | Second-largest drop | Consistent with dense neuron distribution in upper layers |
| Bottom-layer masking only | Smallest drop | Fewer neurons in lower layers |
| Rhetoric injected into emotion | 1–5% improvement | Validates effectiveness of cross-signal enhancement |

### Key Findings

- Conventional masking methods (Zero/Mean) consistently produce counter-intuitive accuracy gains on emotion tasks, demonstrating the unreliability of hard masking.
- Adaptive masking yields stable accuracy decreases (−4% to −14%) across all 10 emotion/rhetoric categories, confirming the method's reliability.
- Emotion and rhetoric neurons are predominantly concentrated in the upper layers of the model (layers 25–32), particularly in the 8B model.
- Hyperbole rhetoric neurons positively enhance all emotion categories, consistent with hyperbole's natural tendency to amplify emotional intensity.
- Sarcasm rhetoric most significantly promotes sadness, stemming from the expressive compatibility between sarcasm's implicit criticism and the introverted nature of sadness.

## Highlights & Insights

- Adaptive masking resolves a longstanding challenge in interpretability research: the counter-intuitive behavior of conventional masking. The combination of decay masking and feedback optimization is both elegant and effective, and generalizes to other neuron function validation settings.
- The finding that rhetoric neurons assist emotion recognition is particularly compelling: rhetoric is not merely a linguistic "ornament" — internally, it genuinely influences the model's emotion representations. This opens a new direction for multi-signal synergistic LLM control.
- The observed layer-wise distribution of neurons (upper-layer concentration) is consistent with existing knowledge storage studies, further confirming that upper FFN layers are critical regions for semantic integration.

## Limitations & Future Work

- The study is limited to 6 basic emotion categories and 4 rhetorical devices, leaving more complex affective states and additional rhetorical forms unexplored.
- Neuron intervention relies on static functional vectors without accounting for context-dependent dynamic adjustment.
- Validation is conducted solely on Llama-3.1; whether similar distributions hold for other architectures (e.g., Qwen, Mistral) remains to be confirmed.
- The strategy for adjusting the decay coefficient $\alpha$ is relatively simple; more refined adaptive strategies may further improve effectiveness.

## Related Work & Insights

- **vs Lee et al. (2025)**: Confirms the existence of emotion neurons in the Llama series but does not study rhetoric; this paper is the first to systematically investigate rhetoric neurons and reveal emotion–rhetoric synergy.
- **vs Di Palma et al. (2025)**: Achieves emotion recognition via probing techniques, but constitutes passive analysis; this paper realizes active intervention and targeted steering.
- **vs Radford et al. (2018)**: Introduced the earliest concept of sentiment neurons; this paper conducts a more systematic and reliable investigation on modern LLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of rhetoric neurons and emotion–rhetoric synergy
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets + cross-dataset validation + multi-layer masking analysis
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous experimental design
- Value: ⭐⭐⭐⭐ Provides reliable tools for neuron-level LLM interpretability and intervention

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[CVPR 2026\] Sign Language Recognition in the Age of LLMs](../../CVPR2026/llm_nlp/sign_language_recognition_llms.md)
- [\[AAAI 2026\] Smart: A GNN-LLM Hybrid Surrogate Model for Dragonfly System Application Runtime Prediction](../../AAAI2026/llm_nlp/smart_a_surrogate_model_for_predicting_application_runtime_in_dragonfly_systems.md)
- [\[ACL 2026\] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)

</div>

<!-- RELATED:END -->
