---
title: >-
  [Paper Note] RLZero: Direct Policy Inference from Language Without In-Domain Supervision
description: >-
  [NeurIPS 2025][Image Generation][Zero-shot policy inference] This paper proposes RLZero, a framework that converts natural language instructions into behavioral policies in target environments via an "Imagine → Project → Imitate" pipeline. A video generation model is used to "imagine" observation sequences from language; these are then projected into the target domain; finally, an unsupervised pretrained RL agent imitates the projected sequences via a closed-form solution — all without any in-domain supervision or annotated trajectories.
tags:
  - NeurIPS 2025
  - Image Generation
  - Zero-shot policy inference
  - language-conditioned RL
  - video generation models
  - unsupervised RL
  - cross-embodiment transfer
date: 2026-05-08
content_hash: e04c8eea16a527fc
---

# RLZero: Direct Policy Inference from Language Without In-Domain Supervision

**Conference**: NeurIPS 2025
**arXiv**: [2412.05718](https://arxiv.org/abs/2412.05718)
**Code**: None
**Area**: Image Generation
**Keywords**: Zero-shot policy inference, language-conditioned RL, video generation models, unsupervised RL, cross-embodiment transfer

## TL;DR

This paper proposes RLZero, a framework that converts natural language instructions into behavioral policies in target environments via an "Imagine → Project → Imitate" pipeline. A video generation model is used to "imagine" observation sequences from language; these are then projected into the target domain; finally, an unsupervised pretrained RL agent imitates the projected sequences via a closed-form solution — all without any in-domain supervision or annotated trajectories.

## Background & Motivation

**Background**: The reward hypothesis posits that all goals can be expressed as the maximization of a scalar reward signal, yet defining appropriate reward functions in practice is notoriously difficult. Natural language offers an intuitive alternative for guiding RL agents, but existing language-conditioned RL approaches either require costly in-domain supervision (annotated trajectories, language–action pairs) or necessitate test-time training upon receiving new language instructions.

**Limitations of Prior Work**: (1) Traditional language-conditioned RL demands large quantities of manually annotated language–trajectory pairs, which are expensive to collect; (2) reward-function learning methods still require in-domain training data to bridge language and environment; (3) test-time training approaches reduce the need for pre-annotation but require retraining for every new instruction, precluding immediate execution.

**Key Challenge**: The language instruction space is open-ended and environment-agnostic, whereas RL policies must be grounded in the dynamics of a specific environment — bridging these two spaces without any task-specific supervision is the central challenge.

**Goal**: (1) Achieve fully zero-shot language-to-behavior conversion — no in-domain supervision, no annotated trajectories, no test-time training; (2) enable pretrained RL agents to respond to arbitrary natural language instructions; (3) support cross-embodiment transfer, including transfer from YouTube videos to robotic systems.

**Key Insight**: The problem is decomposed into three independently solvable sub-problems — language → vision (leveraging the text-to-video capabilities of video generation models), vision → cross-domain (domain transfer), and observation → action (closed-form imitation via unsupervised RL pretraining).

**Core Idea**: Video generation models serve as translators from language to vision; a unsupervised pretrained RL agent then directly imitates the translated observation sequences, thereby circumventing the need for any in-domain supervision.

## Method

### Overall Architecture

The RLZero pipeline consists of three sequentially composed, independent modules: (1) **Imagine** — given a natural language instruction, a pretrained text-to-video model generates an observation sequence depicting the described task execution; (2) **Project** — the generated video frames are projected into the visual space of the target environment via domain transfer, eliminating the domain gap between source and target; (3) **Imitate** — in the target environment, an unsupervised pretrained RL agent computes, via a closed-form solution, a policy that imitates the projected observation sequence instantaneously, without any additional training.

### Key Designs

1. **Imagine: Video Generation Models as Language→Vision Translators**

    - **Function**: Converts arbitrary natural language instructions into visually grounded task execution sequences.
    - **Mechanism**: A large-scale pretrained text-to-video generative model (e.g., a diffusion-based model) takes a language description as input and outputs a sequence of video frames depicting an agent executing the corresponding task in the environment. The general knowledge encoded in the video generation model serves as a bridge between language and behavior.
    - **Design Motivation**: Video generation models pretrained on internet-scale data already encode rich world knowledge and language–vision associations, requiring no fine-tuning on any specific RL environment.

2. **Project: Cross-Domain Visual Projection**

    - **Function**: Eliminates the domain gap between generated video frames and observations in the target environment.
    - **Mechanism**: Video frames produced in the Imagine stage (which may exhibit cartoon-style rendering, simulator aesthetics, or real-world viewpoints) are mapped into the observation space of the target environment, so that the subsequent imitation stage can treat them as valid environmental observations. Applicable techniques include style transfer and feature alignment via visual encoders.
    - **Design Motivation**: Images produced by video generation models typically differ in style and viewpoint from the target environment, and directly using them would lead to imitation failure.

3. **Imitate: Closed-Form Unsupervised RL Imitation**

    - **Function**: Instantaneously infers an executable policy from the projected observation sequence.
    - **Mechanism**: During pretraining, the agent undergoes unsupervised RL (e.g., goal-conditioned RL or successor features) in the target environment without task labels, acquiring general state–action mapping capabilities through exploratory interaction. At inference time, given a projected target observation sequence, the agent computes the actions required to imitate that sequence via a closed-form solution (e.g., least-squares or linear programming), with no gradient descent or additional training required.
    - **Design Motivation**: The closed-form solution guarantees zero-latency policy generation, enabling the system to achieve genuine "language input → immediate behavioral output."

### Loss & Training

During pretraining, the unsupervised RL objective is employed (the specific form depends on the chosen unsupervised RL method, e.g., mutual information maximization or goal-conditioned reward shaping). No training is required at inference time.

## Key Experimental Results

### Main Results

| Environment | Modality | Task Type | RLZero Success Rate | Baseline Comparison | Notes |
|---|---|---|---|---|---|
| Multiple continuous control | Language → behavior | Various manipulation tasks | Effective | First zero-shot method | No in-domain supervision |
| Cross-embodiment | YouTube video → behavior | Humanoid robot | Effective | — | YouTube to humanoid transfer |

### Ablation Study

| Configuration | Performance | Notes |
|---|---|---|
| Full RLZero | Best | All three stages combined |
| Without Project stage | Degraded | Domain gap causes imitation deviation |
| Different video generation models | Variable | Generation quality affects downstream performance |

### Key Findings

- **First zero-shot language→behavior method**: RLZero is the first approach to demonstrate direct language-to-behavior generation across multiple tasks and environments without any in-domain supervision.
- **Cross-embodiment transfer is feasible**: Beyond processing language instructions, the framework can infer humanoid robot policies from cross-embodiment YouTube videos (e.g., human demonstrations).
- **Advantages of modular design**: Each of the three modules can be upgraded independently — improvements in video generation models, domain adaptation methods, or unsupervised RL algorithms directly translate into overall system gains.

## Highlights & Insights

- The **three-stage decoupling strategy** decomposes an intractable end-to-end problem into three relatively independent sub-problems, each addressable with established tools. This divide-and-conquer system design is particularly elegant and transferable to other cross-modal robot planning problems.
- **Closed-form imitation** achieves truly zero-latency inference, offering a qualitative advantage over methods that require test-time training — in practical deployment, retraining for every new instruction is infeasible.
- The use of **video generation models as "world models"** represents a noteworthy paradigm shift: rather than predicting environment dynamics, they are employed to translate language into imitable behavioral references.

## Limitations & Future Work

- **Bottleneck of video generation quality**: System performance is upper-bounded by the quality of the video generation model; when language instructions describe complex or rare actions, the generated videos may be inaccurate.
- **Robustness of the projection step**: Domain transfer may discard critical spatial or temporal information, particularly when the visual disparity between source and target domains is large.
- **Coverage of unsupervised RL pretraining**: If the pretraining exploration fails to cover the state–action combinations required by the task, the closed-form imitation may produce infeasible actions.
- **Limited quantitative comparison**: As the first zero-shot method, fair comparison against supervised approaches is inherently difficult; the scale and depth of quantitative experiments warrant strengthening.
- **Long-horizon tasks**: The sequence length produced by video generation models is limited, making the approach potentially unsuitable for tasks requiring extended temporal planning.

## Related Work & Insights

- **vs. SayCan / Language-to-Reward**: SayCan requires a pretrained skill library; Language-to-Reward requires reward function specification; both demand some degree of in-domain supervision. RLZero bypasses these requirements entirely.
- **vs. UniPi**: UniPi also employs video generation models for planning, but requires training an inverse dynamics model within the target domain to convert videos into actions. RLZero replaces this step with unsupervised RL and a closed-form solution.
- **vs. DALL-E-Bot / SuSIE**: These methods use image generation models to produce subgoal images, but still require in-domain training of goal-conditioned policies. RLZero's unsupervised pretraining is considerably more general.
- The paper demonstrates the substantial potential of combining foundation models (video generation, unsupervised RL) in a compositional manner.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "Imagine–Project–Imitate" system architecture addresses a long-standing open problem and achieves genuinely zero-shot language→behavior generation.
- Experimental Thoroughness: ⭐⭐⭐ Demonstrates capabilities across multiple environments and cross-embodiment settings, though quantitative comparisons and ablations could be more extensive.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly, and the motivation is presented persuasively.
- Value: ⭐⭐⭐⭐⭐ As the first zero-shot language-conditioned RL method, this work opens an entirely new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[NeurIPS 2025\] Flattening Hierarchies with Policy Bootstrapping](flattening_hierarchies_with_policy_bootstrapping.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency](freqpolicy_efficient_flow-based_visuomotor_policy_via_frequency_consistency.md)
- [\[ICLR 2026\] Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations](../../ICLR2026/image_generation/translate_policy_to_language_flow_matching_generated_rewards_for_llm_explanation.md)

</div>

<!-- RELATED:END -->
