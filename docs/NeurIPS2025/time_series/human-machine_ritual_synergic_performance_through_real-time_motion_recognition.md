---
title: >-
  [Paper Note] Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition
description: >-
  [NeurIPS 2025][Time Series][IMU sensors] This paper proposes a lightweight real-time motion recognition system that leverages wearable IMU sensors combined with the MiniRocket time-series classifier to achieve dancer-spe…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "IMU sensors"
  - "motion recognition"
  - "MiniRocket"
  - "real-time interaction"
  - "dance-music synergy"
date: 2026-05-08
content_hash: 57143152d71b5236
---

# Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition

**Conference**: NeurIPS 2025
**arXiv**: [2511.02351](https://arxiv.org/abs/2511.02351)  
**Code**: None  
**Area**: Human-Computer Interaction / Creative AI
**Keywords**: IMU sensors, motion recognition, MiniRocket, real-time interaction, dance-music synergy

## TL;DR
This paper proposes a lightweight real-time motion recognition system that leverages wearable IMU sensors combined with the MiniRocket time-series classifier to achieve dancer-specific motion recognition with <50ms latency and 96.05% accuracy. Through "embodied memory mapping," the system encodes each dancer's personal movement-sound associations, establishing a human-machine collaborative performance paradigm that respects the expressive depth of the human body.

## Background & Motivation

**Background**: Human-machine collaborative performance is a prominent direction in new media art. Existing systems (e.g., EDGE dance-music generation, LuminAI improvisational dance partner) tend to have AI generate dance movements or music, positioning the machine as a "creator."

**Limitations of Prior Work**:
- Generative AI systems rely on predefined music genre labels and generic datasets (e.g., AIST++), neglecting artists' personal embodied experiences and memory associations.
- Discrete gesture input interfaces (e.g., Wekinator) are ill-suited for continuous dance movement.
- Most systems emphasize the machine's creativity rather than "deep listening" to human expression.

**Key Challenge**: AI-driven performance systems pursue autonomous machine creativity, which may obscure rather than amplify the expressive depth of the human body—bodily knowledge in dance (tactile sensation, memory, intuition) cannot be replaced by algorithms.

**Goal**: Design a collaborative paradigm in which "the machine does not create, only remembers"—the machine learns to recognize the dancer's movements and triggers sounds that the dancer personally associates with those movements, rather than generating new content.

**Key Insight**: Drawing from somatic philosophy, the paper treats the dancer's body as an "archive and oracle"—each movement carries personal memory and imagery, and the machine's role is that of an "attentive stage manager" rather than a "co-creator."

**Core Idea**: IMU + MiniRocket real-time recognition of personalized dancer movements → triggers the dancer's own associated sonic memories = a human-machine collaborative performance based on recollection (not generation).

## Method

### Overall Architecture
A two-stage pipeline: (1) **Training phase**: the dancer listens to personally meaningful sounds → improvises movement → 4 IMU sensors (wrists + ankles) collect 6-axis data (accelerometer + gyroscope, 24 channels total, 48 Hz) → segmentation + augmentation (jittering / time warping) → MiniRocket feature extraction + Ridge classifier training. (2) **Performance phase**: real-time IMU data stream → transmitted via BLE to a GPU server → MiniRocket inference → motion class and probability returned → corresponding sound/projection triggered.

### Key Designs

1. **Embodied Memory Mapping**:

    - **Function**: Establishes the dancer's personal movement-sound associations.
    - **Mechanism**: Rather than using predefined labels or AI-generated mappings, the dancer improvises movement in response to specific sounds, encoding the "memory association" between movement and sound into the training data. The dancer verbally describes the memory or imagery evoked by each sound (e.g., "the repetitive feeling of a subway commute"), and these descriptions guide the narrative organization of sounds during performance.
    - **Design Motivation**: To ensure that sounds are meaningful to the dancer and that movements are naturally elicited by sounds, forming a tight semantic feedback loop—the machine serves as a "bridge to memory" rather than a "source of creativity."

2. **IMU + MiniRocket Real-Time Classification**:

    - **Function**: High-accuracy, low-latency continuous motion recognition.
    - **Mechanism**: 4 IMUs (~25g each, BLE wireless) → 24-channel time series → 2-second sliding window segmentation → MiniRocket generates 10,000 minimum random convolutional kernel features → Ridge regression classifier. Full inference (data stream → server → result returned) < 50ms; inference alone ~15ms.
    - **Design Motivation**: MiniRocket achieves high accuracy on time-series classification without requiring GPU training; the Ridge classifier is simple and fast. The overall system is lightweight enough for real-time use in live performance.

3. **Data Augmentation Strategy**:

    - **Function**: Obtain sufficient training data from a small dataset (648 samples, 7 classes).
    - **Mechanism**: Applies jittering (additive Gaussian noise) and time warping (random stretching/compression of the time axis) to IMU time-series data.
    - **Design Motivation**: Dancer-specific training data is inherently limited (collected on-site); augmentation is therefore necessary.

### Loss & Training
- Ridge regression classifier (L2-regularized linear model)
- 10-fold stratified cross-validation
- Training data: 648 samples across 7 motion classes

## Key Experimental Results

### Main Results
10-fold cross-validation:

| Metric | Value |
|--------|-------|
| Mean Accuracy | 96.05% $\pm$ 2.89% |
| Macro-average F1 | 96.62% |
| AUC (all classes) | > 0.99 |
| End-to-end latency | < 50ms |
| Inference latency | ~15ms |

### Ablation Study: Per-Class Confusion Matrix

| Class | Accuracy | Notes |
|-------|----------|-------|
| Class 0 (stillness) | Highest | Clearly distinguished from other movements |
| Classes 1–6 (dance) | High | Probability decreases only during transitions |
| Transition segments | Reduced probability | Temporal ambiguity arises when a 2-second window spans two distinct movements |

### Key Findings
- **96% accuracy with only 648 samples**: MiniRocket exhibits very high sample efficiency, making it well-suited for personalized small-data scenarios.
- **<50ms latency satisfies real-time performance requirements**: The human perceptual latency threshold is approximately 100ms; the system operates well below this bound.
- **Transition segments are the primary source of error**: Classification uncertainty increases when a 2-second window spans two movements, though this can be exploited in performance—gradually changing probabilities can trigger gradual sound transitions.
- **All AUC > 0.99**: All motion classes are highly discriminable.

## Highlights & Insights
- The **"recollection rather than generation" philosophy of human-machine collaboration** is highly distinctive—repositioning AI from "co-creator" to "carrier and trigger of memory," thereby respecting the irreplaceability of the human body.
- **Engineering practicality of IMU + MiniRocket**: The entire system has very low cost (4 IMUs + laptop/phone), is highly reproducible, and is suitable for resource-constrained performing arts contexts.
- MiniRocket's first application in creative AI / interactive machine learning is noteworthy.

## Limitations & Future Work
- The 648-sample dataset is small; retraining is required when generalizing to other dancers.
- No handling of transitional movements—a 2-second window may truncate rapidly successive gestures.
- The system has only been tested in rehearsal; stability and audience experience in formal public performances have not been evaluated.
- Currently supports only discrete class recognition; continuous assessment of movement quality (e.g., force or fluidity of a spin) is not supported.
- Sound mapping is entirely manual at present; semi-automatic association discovery has not been explored.

## Related Work & Insights
- **vs. EDGE / MusicGen systems**: These systems have AI generate dance or music content. The proposed system restricts AI to recognition and triggering only, keeping the human as the sole content source.
- **vs. Wekinator (Fiebrink)**: Wekinator uses simple classifiers for discrete gesture recognition. This paper employs MiniRocket for continuous time-series processing, which better accommodates the fluid nature of dance.
- **vs. LuminAI / MIT Co-Dancing**: These systems position AI as a virtual dance partner that generates movement. This paper positions AI as a "stage manager" that controls media output.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "recollection rather than generation" human-machine collaboration philosophy is distinctive; first application of MiniRocket in creative AI.
- **Experimental Thoroughness**: ⭐⭐⭐ Technical validation is complete, but evidence is limited to rehearsal settings and the dataset is small.
- **Writing Quality**: ⭐⭐⭐⭐ The integration of philosophical motivation and technical implementation is fluent.
- **Value**: ⭐⭐⭐ Relevant to new media art and interactive AI, though technical contributions are modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Human Brain as a Combinatorial Complex](the_human_brain_as_a_combinatorial_complex.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] PlanU: Large Language Model Reasoning through Planning under Uncertainty](planu_large_language_model_reasoning_through_planning_under_uncertainty.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICLR 2026\] Tuning the Burn-in Phase in RNN Training Improves Performance](../../ICLR2026/time_series/tuning_the_burn-in_phase_in_training_recurrent_neural_networks_improves_their_pe.md)

</div>

<!-- RELATED:END -->
