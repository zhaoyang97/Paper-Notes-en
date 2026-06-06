---
title: >-
  [Paper Note] TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models
description: >-
  [ICLR 2026][Reinforcement Learning][Temporal Understanding] TPRU constructs a large-scale multi-image temporal understanding dataset (24,750 QA pairs, 126…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Temporal Understanding"
  - "Procedural Reasoning"
  - "Multi-Image Understanding"
  - "RL Fine-Tuning"
  - "MLLM"
  - "Embodied AI"
date: 2026-05-08
content_hash: 053d40b9d41b8402
---

# TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models

**Conference**: ICLR 2026
**arXiv**: [2602.18884](https://arxiv.org/abs/2602.18884)  
**Code**: [https://github.com/Stephen-gzk/TPRU/](https://github.com/Stephen-gzk/TPRU/)  
**Area**: Reinforcement Learning / Multimodal Large Language Models
**Keywords**: Temporal Understanding, Procedural Reasoning, Multi-Image Understanding, RL Fine-Tuning, MLLM, Embodied AI

## TL;DR
TPRU constructs a large-scale multi-image temporal understanding dataset (24,750 QA pairs, 126,000 images) spanning 3 complementary task types (temporal ordering, next-frame prediction, previous-frame review) across 4 embodied scenarios including robotic manipulation and GUI navigation, and demonstrates that RL fine-tuning enables a 7B model to surpass GPT-4o on temporal understanding benchmarks.

## Background & Motivation

**Background**: MLLMs excel at single-image tasks, but smaller deployable models exhibit severe deficiencies in understanding temporal and procedural image sequences—a critical bottleneck for embodied AI deployment in robotics, navigation, and instruction following.

**Limitations of Prior Work**: (1) Training paradigms systematically fail—existing datasets treat multi-image inputs as unordered collections, ignoring the critical distinction between "understanding an image set" and "understanding an image sequence"; (2) the community's response has been to create evaluation-only benchmarks that repeatedly confirm failure, rather than addressing the root cause: the absence of large-scale real-world sequential training data.

**Key Challenge**: Edge devices with limited resources cannot deploy models with tens of billions of parameters, yet smaller models' deficiency in procedural understanding renders them unsuitable for embodied scenarios. The question is whether this is a matter of model scale or training data.

**Goal**: (1) Provide a large-scale, structured temporal understanding training set to close the train-test gap; (2) validate whether small models, given appropriate data and training, can achieve temporal understanding on par with large models.

**Key Insight**: Real sequential data is collected from 4 embodied scenarios; 3 complementary tasks are designed to cover different aspects of temporal reasoning; hard negative samples are introduced to compel active cross-modal verification rather than passive observation.

**Core Idea**: Structured data combined with RL training enables small models to surpass large models in temporal and procedural understanding, demonstrating that this gap is a training challenge rather than an intrinsic limitation of scale.

## Method

### Overall Architecture
TPRU comprises two components: TPRU-25k (fine-tuning set, 24,750 samples) and TPRU-Test (461 manually annotated evaluation samples). Data is drawn from 4 embodied scenarios and processed through a 3-stage pipeline: sequence filtering → caption generation with robustness augmentation → task formulation.

### Key Designs

1. **Data Source Design**:

    - Robotic Manipulation: video frame sampling from ShareRobot planning tasks
    - LEGO Assembly: 36 high-quality stop-motion animation videos with clear state transitions and no motion blur
    - GUI Operation: 4-step screenshot sequences from GUI Odyssey
    - Embodied Navigation: ordered visual observations from the Habitat simulation environment
    - **Design Motivation**: Multi-source diversity ensures the model learns generalizable temporal reasoning rather than domain-specific patterns.

2. **Three Complementary Tasks**:

    - **Temporal Ordering**: frames are shuffled and the model must restore the correct order based on a textual description—evaluating comprehension of the complete temporal trajectory.
    - **Next-Frame Prediction**: given frames 1, 2, and 4, the model selects the correct frame 3 from candidates drawn from similar scenes—simulating an agent's anticipation of action consequences.
    - **Previous-Frame Review**: given the last 3 frames, the model selects the correct initial frame from candidates—evaluating understanding of procedural preconditions and event provenance.
    - **Design Motivation**: Forward prediction, backward review, and global ordering jointly cultivate structured, procedural dynamic understanding.

3. **Negative Sampling Strategy**:

    - **Function**: Creates instances with deliberate text-image mismatches (e.g., pairing "pick up the fork" with an image of "putting down a knife").
    - **Target Output**: "None of the choices provided."
    - **Design Motivation**: Forces explicit cross-modal verification, preventing the model from relying solely on textual priors.

### Loss & Training
Reinforcement learning (RL) is used to fine-tune Qwen2.5-VL models (3B/7B/32B), focusing on improving temporal reasoning in resource-constrained models. Details of RL training and reward design are provided in the paper's appendix.

## Key Experimental Results

### Main Results on TPRU-Test

| Model | Parameters | Accuracy |
|-------|-----------|----------|
| Qwen2.5-VL-7B (base) | 7B | 50.33% |
| **TPRU-7B** | **7B** | **75.70%** |
| GPT-4o | Closed-source | <75.70% |
| Qwen2.5-VL-72B | 72B | — |
| Gemini-2.5-Flash | Closed-source | — |

TPRU-7B achieves a 25.37 percentage point improvement over the base model, substantially outperforming GPT-4o.

### MuirBench Evaluation

| Model | Overall |
|-------|---------|
| Qwen2.5-VL-7B (base) | 58.35% |
| **TPRU-7B** | Significant improvement |
| **TPRU-32B** | **68.42%** |
| GPT-4o | 68.00% |
| Qwen2.5-VL-72B | 69.35% |

TPRU-32B surpasses GPT-4o and approaches the performance of the 72B model. Gains in the Ordering category are particularly pronounced.

### LEGO-Puzzles Evaluation
TPRU-7B demonstrates substantial improvements on LEGO-Puzzles, reflecting enhanced capacity for multi-step planning and procedural reasoning.

### Key Findings
- The temporal reasoning gap is not an intrinsic limitation of model scale but a challenge addressable through targeted data and RL training.
- The complementarity of the three tasks is essential—ablation studies show that removing any single task degrades performance.
- Negative samples are critical for preventing hallucination and over-reliance on textual priors.
- RL fine-tuning is more effective than SFT for this class of tasks.

## Highlights & Insights
- **Unified Training and Evaluation**: Breaks the cycle of "create benchmark → confirm failure → create new benchmark" by simultaneously providing training data and evaluation tools.
- **Small Models Outperforming Large Ones**: A 7B model surpassing GPT-4o challenges the assumption that temporal understanding requires large-scale models.
- **Practically-Oriented Design**: Data originates from real embodied scenarios rather than synthetic sources, with task designs directly corresponding to agent capability requirements.
- **RL Training Paradigm**: Demonstrates the effectiveness of RL in enhancing MLLM reasoning, outperforming conventional SFT.

## Limitations & Future Work
- TPRU-25k, while diverse, remains expandable in scale.
- The current focus on short sequences of 3–4 frames leaves procedural understanding over longer sequences unexplored.
- The relationship and distinctions between video understanding and multi-frame image understanding warrant further investigation.
- The reward design for RL training may admit better formulations.

## Related Work & Insights
- **vs. Mantis-Instruct / LLaVA-NeXT-Interleave**: These datasets treat multiple images as unordered collections and lack systematic temporal structure.
- **vs. MuirBench / LEGO-Puzzles**: These works provide evaluation benchmarks only; TPRU additionally supplies training data to close the gap.
- **vs. Jigsaw-R1 / MiCo**: These use procedurally generated puzzles to train spatial reasoning; TPRU employs real-world scenarios to train temporal reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The dataset design and three-task framework are innovative; the combination of RL with temporal reasoning is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple benchmarks (TPRU / MuirBench / LEGO), multiple model scales, and comprehensive method comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear and the pipeline figure is well-designed, though the dataset description is somewhat verbose.
- **Value**: ⭐⭐⭐⭐⭐ — Directly valuable for embodied AI and deployable MLLMs; dataset and code are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)
- [\[ICLR 2026\] When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift](when_sensors_fail_temporal_sequence_models_for_robust_ppo_under_sensor_drift.md)

</div>

<!-- RELATED:END -->
