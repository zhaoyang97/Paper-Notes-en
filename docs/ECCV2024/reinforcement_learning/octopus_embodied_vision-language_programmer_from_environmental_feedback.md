---
title: >-
  [Paper Note] Octopus: Embodied Vision-Language Programmer from Environmental Feedback
description: >-
  [ECCV2024][Reinforcement Learning][Embodied AI] This paper proposes Octopus, an embodied vision-language programming model that bridges high-level planning and low-level manipulation by generating executable code. It introduces a Reinforcement Learning with Environmental Feedback (RLEF) training scheme to enhance decision-making quality.
tags:
  - "ECCV2024"
  - "Reinforcement Learning"
  - "Embodied AI"
  - "vision-language model"
  - "code generation"
  - "environmental feedback"
date: 2026-05-08
content_hash: c17bda2138a98533
---

# Octopus: Embodied Vision-Language Programmer from Environmental Feedback

**Conference**: ECCV2024  
**arXiv**: [2310.08588](https://arxiv.org/abs/2310.08588)  
**Code**: [GitHub](https://github.com/dongyh20/Octopus)  
**Area**: Robotics  
**Keywords**: Embodied AI, vision-language model, code generation, reinforcement learning, environmental feedback

## TL;DR

This paper proposes Octopus, an embodied vision-language programming model that bridges high-level planning and low-level manipulation by generating executable code. It introduces a Reinforcement Learning with Environmental Feedback (RLEF) training scheme to enhance decision-making quality.

## Background & Motivation

Large vision-language models (VLMs) have made remarkable progress in multimodal perception and reasoning. However, a significant gap remains when applying them to embodied agents: existing approaches either output only low-level manipulation action sequences or provide abstract, high-level plans, **leaving a lack of effective bridging between high-level planning and real-world manipulation**.

Prior programming paradigms (such as Voyager, VisProg) utilize LLMs to generate programs that call APIs, but they **lack visual perception** and parse textual data directly from the environment. A few works attempting to integrate vision (such as TAPA, SayPlan) over-rely on pre-trained vision models to convert images into language descriptions, potentially losing critical information during conversion, and are unable to generate executable code. Although EmbodiedGPT combines vision-language modeling with policy mapping, the capability of embodied VLMs to directly generate executable programs remains under-explored.

## Core Problem

1. How can embodied vision-language models be enabled to **directly generate executable code from visual inputs**, rather than merely outputting abstract plans?
2. How to construct **vision-dependent** simulation environments and function call systems to ensure that agents cannot bypass visual information to complete tasks?
3. How to leverage **environmental feedback** to further optimize the planning and code generation capabilities of the model?

## Method

### OctoVerse Environment Suite

The authors carefully designed three simulation environments, collectively termed OctoVerse. The core design principle is that all function calls must be **vision-dependent**:

- **OctoGibson**: Built on OmniGibson, supporting 50 scenes, 5000+ annotated objects, and 476 tasks (367 regular tasks + 109 reasoning tasks). It integrates 16 executable functions (e.g., `moveBot()`, `easyGrasp()`). Key design: restricting the arguments of `moveBot(object)` only to large objects (tables, cabinets), forcing the agent to infer the positions of small objects through vision.
- **OctoMC**: Built on Minecraft, containing 6 functional actions, 40 tasks, and 10 different biomes. The vision-independent `exploreUntil()` function from Voyager is removed and replaced with `teleport(yaw, distance)`, requiring the agent to actively navigate using visual cues.
- **OctoGTA**: Built on GTA-V, containing 19 functions, 25 tasks, and 5 task groups. It replaces `walkTo(location)` with `goForward(distance)` and `turnPlayer(degree)` to ensure operations rely on visual perception.

### Octopus Model Architecture

The model is based on the Otter architecture, with core components including:

- **Vision Encoder**: CLIP ViT-L/14
- **Language Decoder**: MPT-7B
- **Cross-Modal Module**: Perceiver Resampler + Cross-Gated Attention (from Flamingo's design)
- **Visual Input**: 8 first-person perspective images (one every 45°, covering 360°) + 2 bird's-eye view (BEV) images (OctoMC uses only 4 first-person images)

### Training Data Collection

1. **Environmental Information Collection**: Each state is formatted as an environment message containing Observed Objects, Observed Relations, Inventory, and Task Goal.
2. **GPT-4 Automatic Collection**: System prompts are carefully designed, and the GPT-4 32K model is used to control an exploratory agent to execute tasks and generate action blueprints and executable code within the simulation environments.
3. **Error Management**: Errors during GPT-4 execution (syntax errors, physical constraint failures, etc.) are also recorded; the task is terminated if the main task is not completed within 10 steps; all dataset pairs without syntax errors are retained for training.
4. **Environmental Feedback Collection**: Binary feedback is automatically labeled at the step-level (whether the state matches the target after a single step execution) and task-level (whether the overall task is completed).

### Training Pipeline

**Phase 1: SFT (Supervised Fine-Tuning)**

Token-level supervised fine-tuning is conducted on the collected dataset $\mathcal{D}_E = \{(\mathbf{X}_v, \mathbf{T}_i, \mathbf{T}_r)\}$, with the training objective of next-token prediction:

$$p(\mathbf{T}_r | \mathbf{T}_i, \mathbf{X}_v) = \prod_{l=1}^{L} p(t_l | \mathbf{X}_v, \mathbf{T}_i, \mathbf{T}_{r,<l})$$

**Phase 2: RLEF (Reinforcement Learning with Environmental Feedback)**

- **Tree-Structured Task Representation**: Modeling the task execution process as a tree structure, where each node represents a subtask with associated binary success/failure labels.
- **Reward Model**: Utilizing CodeLLaMA-7B with an additional value head as the reward model $r_\phi$, fine-tuned on preference data $\mathcal{D}_R$ to evaluate the quality of state transitions.
- **Policy Optimization**: Taking the SFT model as the initial policy $\pi^{\text{INIT}}$ and cloning it to initialize the RL policy $\pi_\theta^{\text{RL}}$, which is optimized using the PPO algorithm. The loss function includes a reward term and a KL divergence penalty term.

## Key Experimental Results

### OctoGibson Main Results (Success Rate / Planning Score)

| Model | Overall | Seen Env | Unseen Env | Regular Tasks | Reasoning Tasks |
|------|------|---------|---------|---------|---------|
| GPT-4 (blind) | 0.43/0.68 | 0.42/0.69 | 0.46/0.67 | 0.49/0.78 | 0.27/0.40 |
| GPT-4V | 0.45/0.63 | 0.40/0.62 | 0.60/0.67 | 0.42/0.67 | 0.53/0.53 |
| CodeLLaMA | 0.12/0.25 | 0.09/0.20 | 0.20/0.40 | 0.16/0.31 | 0.00/0.07 |
| TAPA (step) | 0.15/0.38 | 0.16/0.42 | 0.13/0.27 | 0.18/0.38 | 0.07/0.40 |
| EmbodiedGPT | 0.10/0.40 | 0.04/0.36 | 0.27/0.53 | 0.13/0.38 | 0.00/0.40 |
| **Octopus (SFT)** | 0.15/0.37 | 0.11/0.33 | 0.27/0.47 | 0.16/0.38 | 0.13/0.33 |
| **Octopus (SFT+RLEF)** | **0.18/0.42** | 0.13/0.38 | **0.33/0.53** | 0.18/0.40 | **0.20/0.53** |

### OctoMC & OctoGTA Results

- OctoMC: Octopus (SFT+RLEF) achieves an overall task success rate of 0.30 and a planning score of 0.65, outperforming EmbodiedGPT (0.25/0.58).
- OctoGTA: Octopus (SFT+RLEF) achieves an overall success rate of 0.20 and planning score of 0.56, outperforming EmbodiedGPT (0.16/0.42).
- GPT-4V remains the strongest baseline across all environments (OctoMC: 0.73/0.85, OctoGTA: 0.56/0.82).

### Ablation Study

- **Model Components**: Fine-tuning only the connector completes only 4/60 tasks; tuning both the connector and language decoder completes 5/60 tasks; full-parameter fine-tuning yields the best performance.
- **Model Scale**: The 3B model is significantly outperformed by the 7B model under both SFT and RLEF settings.
- **Visual Input**: Randomly shuffling the sequence of visual inputs leads to a substantial performance drop, demonstrating that the model indeed relies on structured visual signals.
- **Code Execution Rate**: CodeLLaMA achieves a code execution success rate of 92%, LLaMA only achieves 24%, while Octopus (based on MPT-7B) achieves 72%.

## Highlights & Insights

1. **Innovative Positioning of the Programming Paradigm**: Utilizing executable code as a bridge between high-level planning and low-level manipulation. This retains the flexibility of planning while enabling direct execution.
2. **Vision-Dependent Environment Design**: Function calls in the three environments are carefully constrained to ensure that task completion must rely on visual information, avoiding simple textual shortcuts.
3. **RLEF Training Scheme**: Preference data is constructed using environmental feedback from the simulator, and the policy model is optimized via PPO. The performance gain is particularly notable in reasoning tasks and unseen environments (e.g., reasoning tasks improved from 0.13 to 0.20).
4. **Complete Data Collection Pipeline**: GPT-4-driven automated data collection with an integrated error management mechanism offers high scalability.

## Limitations & Future Work

1. **Absolute Performance Gap**: The best Octopus model achieves only an 0.18 success rate on OctoGibson, which is far behind GPT-4V's 0.45 (and the latter is also far from saturated).
2. **Insufficient Spatial Reasoning**: In OctoMC and OctoGTA, the model frequently fails at precise spatial reasoning (angles, distances). Even if the planning is correct, tasks still fail due to imprecise physical execution.
3. **Limited Complex Decision-Making**: In OctoGTA, when facing non-trivial obstacles (e.g., navigating around walls or climbing), the model struggles to generalize even if similar cases exist in the training data.
4. **GTA Data Reliance on Manual Annotation**: Since textual environment states cannot be extracted in OctoGTA, the training dataset is entirely hand-crafted with only 160 samples, restricting model performance.
5. **Post-Processing Dependency**: Generalized object names generated in the code must be matched via string similarity to simulator-specific API variables, a step that may not be feasible in real-world scenarios.
6. **Failure of CodeLLaMA Integration**: Replacing MPT with CodeLLaMA generates nonsensical outputs, indicating that the vision-code alignment data may be insufficient.

## Related Work & Insights

- **Code as an Intermediate Representation** is a promising direction: compared to directly outputting actions or natural language plans, code naturally possesses compositionality, debuggability, and executability, making it a promising universal output format for embodied agents.
- **The RLEF Paradigm** can be extended to other scenarios where environmental feedback is obtainable (e.g., code execution, web interaction, tool usage), with the core concept of converting environmental execution results into preference signals.
- The philosophy of vision-dependent environment design can be utilized to evaluate whether other embodied models truly utilize visual information.
- The ceiling of 7B-scale models on complex embodied tasks remains limited; larger model scales or better vision-to-code alignment schemes may be paths forward.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of the programming paradigm, RLEF, and vision-dependent environment design is highly creative.
- Experimental Thoroughness: ⭐⭐⭐ — Broad coverage across three environments, but absolute performance remains low; the ablation studies are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, visually rich, with clear motivation.
- Value: ⭐⭐⭐⭐ — Open-sourcing environments, data, and models provides fundamental contributions to research on embodied visual programming.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](../../AAAI2026/reinforcement_learning/vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICCV 2025\] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](../../ICCV2025/reinforcement_learning/navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)
- [\[AAAI 2026\] STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](../../AAAI2026/reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)
- [\[ACL 2025\] Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback](../../ACL2025/reinforcement_learning/align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe.md)
- [\[ICLR 2026\] DVLA-RL: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning](../../ICLR2026/reinforcement_learning/dvla-rl_dual-level_vision-language_alignment_with_reinforcement_learning_gating_.md)

</div>

<!-- RELATED:END -->
