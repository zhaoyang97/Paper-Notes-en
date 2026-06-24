---
title: >-
  [Paper Note] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models
description: >-
  [ICML 2025][Robotics][Hierarchical Robot Control] Proposes Hi Robot, a hierarchical VLM system: a high-level VLM reasons about complex user instructions/feedback to generate atomic commands, while a low-level VLA ($\pi_0$) executes actions. Combined with a synthetic data generation scheme, it achieves open-ended instruction following capabilities far surpassing GPT-4o and flat VLAs across three types of robotic platforms.
tags:
  - "ICML 2025"
  - "Robotics"
  - "Hierarchical Robot Control"
  - "Vision-Language-Action Model (VLA)"
  - "Synthetic Data"
  - "Open-Ended Instruction Following"
  - "Human-Robot Interaction"
date: 2026-05-08
content_hash: 3e845ae7f31b0ffc
---

# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models

**Conference**: ICML 2025  
**arXiv**: [2502.19417](https://arxiv.org/abs/2502.19417)  
**Code**: [Project Page](https://www.pi.website/research/hirobot)  
**Area**: Robotics  
**Keywords**: Hierarchical Robot Control, Vision-Language-Action Model (VLA), Synthetic Data, Open-Ended Instruction Following, Human-Robot Interaction

## TL;DR

Proposes Hi Robot, a hierarchical VLM system: a high-level VLM reasons about complex user instructions/feedback to generate atomic commands, while a low-level VLA ($\pi_0$) executes actions. Combined with a synthetic data generation scheme, it achieves open-ended instruction following capabilities far surpassing GPT-4o and flat VLAs across three types of robotic platforms.

## Background & Motivation

Current robot instruction-following systems face a fundamental challenge: **simple atomic instructions vs. complex open-ended interactions**. Although existing VLA models (such as RT-2, $\pi_0$) can execute simple commands like "pick up the cup", they fail to handle complex requirements in real-world scenarios, such as:

- **Composite intent instructions**: "Help me make a veggie sandwich without tomatoes; also, if there is beef, make one for my friend."
- **Contextual feedback**: "That's not trash", "Ignore the rest".
- **Dynamic correction**: "You need to go slightly lower, otherwise you won't be able to grasp it."

The authors draw an analogy to Kahneman's Dual-System Theory:
- **System 1 (Fast Thinking)** = Low-level policy executing atomic operations (grasping, placing)
- **System 2 (Slow Thinking)** = High-level reasoning parsing complex instructions, integrating feedback, and planning the next step.

Prior work has mainly focused on System 1-level behaviors (simple instruction execution) or used LLMs/VLMs combined with pre-defined skills (limiting physical dexterity). The core motivation of Hi Robot is to **simultaneously achieve the flexibility of high-level reasoning and the physical dexterity of low-level control** while handling open-ended user interactions.

## Method

### Overall Architecture

Hi Robot decomposes the policy into a two-layer VLM inference process:

```text
Complex user instruction ℓ_t + Image observation I_t → [High-level VLM] → Atomic command ℓ̂_t (+ Voice response u_t)
                                                                              ↓
Image observation I_t + Atomic command ℓ̂_t + Robot state q_t → [Low-level VLA (π0)] → Action chunk A_t
```

**Frequency Separation**:
- Low-level policy: Outputs action chunks at high frequency (~10 Hz, up to 50 Hz with action chunking)
- High-level policy: Performs low-frequency reasoning (re-evaluates every 1 second, or triggers immediately upon receiving new user input)

**Key Interface**: The high and low levels are connected through **natural language**—the atomic commands output by the high-level policy are essentially short language instructions (e.g., "pick up one piece of lettuce") that the low-level VLA has encountered during training. This constitutes a flexible and interpretable intermediate representation.

### Key Designs

#### 1. Hierarchical Inference

The high-level policy $p_{\text{hi}}(\hat{\ell}_t | \mathbf{I}_t^1, \dots, \mathbf{I}_t^n, \ell_t)$ receives multi-camera images and open-ended instructions to output atomic commands. The low-level policy $p_{\text{lo}}(\mathbf{A}_t | \mathbf{I}_t^1, \dots, \mathbf{I}_t^n, \hat{\ell}_t, \mathbf{q}_t)$ uses this command to generate actions.

For simple and familiar tasks, one can directly set $\hat{\ell}_t = \ell_t$; the advantages of the hierarchical structure lie in:
- Instructions being too complex for the low-level policy to parse directly
- Instructions being out-of-distribution (uncommon) in the context of the robot's training data
- Involving dynamic interaction with the user

#### 2. User Interaction

Users can intervene at any time during task execution (via text or speech-to-text), which immediately triggers high-level re-reasoning. The high-level policy can also output voice replies $u_t$ (e.g., confirmations, clarifications), which are played to the user via TTS and then stripped from the command before being passed to the low-level policy.

The key is that the response of the high-level policy is **contextualized**: it not only processes the instruction $\ell_t$ but also observes the current images, allowing it to correctly understand feedback that requires visual grounding—such as "that is not trash"—which language-only systems cannot achieve.

#### 3. Synthetic Data Generation

This is one of the most unique contributions of this paper. Key Challenge: robot demonstration data only contains simple atomic annotations (e.g., "pick up lettuce"), but the high-level policy needs to learn to handle complex, open-ended instructions.

**Reverse Generation Strategy**: Given (observation image $\mathbf{I}_t$, skill label $\hat{\ell}_t$), a large VLM $p_{\text{gen}}$ is used to reverse-generate "complex user instructions $\ell_t$ that could lead to this skill":

$$p_{\text{gen}}(\ell_t, u_t | \mathbf{I}_t^1, \dots, \mathbf{I}_t^n, \hat{\ell}_0, \dots, \hat{\ell}_t, \mathcal{P})$$

Where $\mathcal{P}$ is a carefully designed prompt template. For example:
- Skill label "pick up the lettuce" $\rightarrow$ generates instruction "Can you help me add some veggies?"
- Skill label "put cup in bin" $\rightarrow$ generates instruction "Only clean up the paper cups, keep the plastic ones."

**Guaranteeing Data Diversity**:
- **Scenarios**: Negative tasks ("don't do X"), contextual corrections ("that is not Y"), specific constraints ("I'm allergic to peanuts")
- **Responses**: Simple confirmation, clarification queries, error handling
- **Context conditioning**: When generating, the skill sequence preceding the current timestep $\hat{\ell}_0, \dots, \hat{\ell}_{t-1}$ is considered to ensure the coherence of instructions across multi-step tasks.

#### 4. Model Architecture

| Component | Base Model | Parameters | Special Design |
|------|---------|--------|---------|
| High-level policy | PaliGemma-3B | 3B | Standard VLM, outputs language |
| Low-level policy ($\pi_0$) | PaliGemma-3B | 3B | Additional flow matching action expert, outputs continuous actions |

Both policy layers share the same VLM base model. The only difference is that the low-level policy features an additional flow matching module to output continuous actions. The framework is highly modular: the low-level component can be replaced with any other language-conditioned policy.

### Loss & Training

**High-level Policy Training**:
- Data: $\mathcal{D}_{\text{syn}} \cup \mathcal{D}_{\text{labeled}}$ (synthetic data + human-labeled data)
- Loss: Standard cross-entropy loss (next-token prediction)
- Full-parameter fine-tuning of PaliGemma-3B

**Low-level Policy Training**:
- Data: $\mathcal{D}_{\text{labeled}} \cup \mathcal{D}_{\text{demo}}$ (human-labeled skills + teleoperated demonstrations)
- Loss: Flow-matching objective (continuous action prediction)

**Training Hyperparameters**:
- Optimizer: AdamW ($\beta_1=0.9$, $\beta_2=0.95$, no weight decay)
- Gradient clipping: Max norm 1.0
- EMA weights: Decay factor 0.999
- Learning rate: Constant $1 \times 10^{-5}$ after 1000 warmup steps
- Batch size: 512
- High-level policy training takes only **~2 hours (8 $\times$ H100)**, which is highly efficient.

## Key Experimental Results

### Main Results

Evaluation is conducted across three task domains and three robotic platforms (20 trials per task for each method):

| Task | Robot | Metrics | Hi Robot | GPT-4o High-level | Flat VLA | Human Expert High-level |
|------|--------|------|----------|------------|----------|------------|
| Table Bussing | Single-arm UR5e | IA / TP | **Best** | Low (object misidentification) | Low (ignoring constraints) | Best (Oracle) |
| Sandwich Making | Dual-arm ARX | IA / TP | **Best** | Low (context loss) | Low (default behavior) | Best (Oracle) |
| Grocery Shopping | Mobile dual-arm ARX | IA / TP | **Best** | Low (instruction inconsistency) | Low (no feedback capability) | Best (Oracle) |

**Key Finding**: Hi Robot's Instruction Accuracy (IA) across all tasks is on average **over 40% higher** than the GPT-4o high-level policy, approaching the performance of human expert guidance.

### Ablation Study

| Configuration | Key Metric (IA / TP) | Description |
|------|-------------------|-------------|
| Hi Robot (Full) | **Best** | Synthetic data + Hierarchical |
| Hi Robot w/o Synthetic Data | Significant drop | Ignores clarifications ("this is not trash"), includes forbidden ingredients |
| Flat VLA + Synthetic Data | Lower than Hierarchical | Has synthetic data but lacks high-level reasoning, reverts to default behavior of sweeping all objects |
| Flat VLA (Original $\pi_0$) | Lowest | Cannot handle complex instructions and real-time feedback |

### Key Findings

1. **Synthetic data is crucial**: Without synthetic data, although the high-level policy aligns with image observations, it completely ignores user constraints (e.g., dietary restrictions, selective cleaning). The compositional language coverage provided by synthetic data is the key to generalization.
2. **Hierarchical structure outperforms flat architecture**: Even under the same data conditions, the hierarchical design outperforms the flat policy—re-evaluating instructions at each high-level step ensures coherence across multi-step tasks.
3. **GPT-4o lacks physical grounding**: GPT-4o frequently issues nonsensical commands (such as "pick up bermuda triangle") and labels all objects as "plate", demonstrating that while large LLMs are powerful, they lack an understanding of robotic capabilities.
4. **Human-expert experiments show the bottleneck is in reasoning, not execution**: Given the correct atomic commands, the low-level policy executes them almost flawlessly.

**Inference Latency**:

| Component | RTX 4090 | H100 |
|------|----------|------|
| Low-level (Single-step) | 73 ms (onboard) / 86 ms (WiFi) | — |
| High-level (prefill) | 47 ms | 17.3 ms |
| High-level (decode/step) | 13.2 ms | 5.7 ms |

The system achieves ~10 Hz control on consumer-grade hardware, reaching up to 50 Hz with action chunking.

## Highlights & Insights

1. **Reverse synthetic data generation** is an elegant and scalable solution—it avoids the need to collect complex interaction data by reverse-generating complex instructions from existing atomic annotations at an extremely low cost.
2. **Language as an intermediate representation** makes the system highly modular and interpretable—system debugging can be performed directly by observing the commands output by the high-level policy.
3. **The System 1 / System 2 analogy** provides a clear design philosophy—both levels utilize VLMs but have a distinct division of labor.
4. Training the high-level policy takes only 2 hours ($8 \times \text{H100}$), demonstrating the efficiency advantages of synthetic data generation combined with VLM fine-tuning.
5. The framework naturally supports **multimodal human-robot interaction**: voice input $\rightarrow$ Whisper ASR $\rightarrow$ high-level reasoning $\rightarrow$ voice reply + action.

## Limitations & Future Work

1. **Lack of memory mechanism**: The high-level policy cannot handle instructions that require long-context reasoning, and there is no memory across timesteps.
2. **Decoupled training of high-level and low-level policies**: The two layers are unaware of each other's capabilities, meaning the high-level policy might generate commands that the low-level policy cannot execute.
3. **Synthetic data dependence on prompt engineering**: Each task domain requires carefully designed generation prompts.
4. **Low-level bias**: Training biases toward grasping nearby objects, which can temporarily override instructions (e.g., grasping when close to cheese, despite the user explicitly asking for no cheese).
5. **Error accumulation and OOD recovery**: Recovery capability after dropping objects is limited.
6. **Training high-level policies separately per task**: A unified multi-task high-level model has not yet been realized.

**Future Directions**: Merging both layers into a single model and distinguishing System 1 / System 2 only during inference; asynchronous multi-level reasoning; introducing a closed-loop mechanism that allows the high-level policy to perceive the execution success rate of the low-level policy.

## Related Work & Insights

- **$\pi_0$ (Black et al., 2024)**: The foundation of the low-level policy in this paper, utilizing a PaliGemma + flow matching VLA.
- **YAY Robot (Shi et al., 2024)**: Previous hierarchical approach, but limited to single instructions and correction types observed in the training data.
- **RACER (Dai et al., 2024)**: Requires a physical simulator to construct recovery behaviors, whereas Hi Robot only uses real demonstrations.
- **SayCan (Brohan et al., 2023)**: LLM + pre-defined skills, lacking visual understanding and physical dexterity.
- **RT-2 (Brohan et al., 2023)**: VLA model but only handles simple commands.

The **reverse synthetic data generation** concept of this paper is highly transferrable: it can be applied to any scenario that has low-level annotations but lacks high-level complex instructions.

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 8 | Elegant hierarchical VLA design, with a novel reverse synthetic data generation scheme |
| Technical Quality | 8 | Sound system design, extensively validated across three platforms |
| Experimental Design | 8 | Comprehensive multi-task, multi-platform, comparative, and ablation studies, though lacking quantitative tables |
| Writing Quality | 9 | Clear analogy of System 1/2, with logically rigorous arguments |
| Practical Value | 9 | Directly addresses real-world scenarios, runnable on consumer-grade hardware |
| **Overall Score** | **8.4** | High-quality work from Physical Intelligence + Stanford |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[NeurIPS 2025\] COOPERA: Continual Open-Ended Human-Robot Assistance](../../NeurIPS2025/robotics/coopera_continual_open_ended_human_robot_assistance.md)
- [\[ECCV 2024\] ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments](../../ECCV2024/robotics/realfred_an_embodied_instruction_following_benchmark_in_photo-realistic_environm.md)
- [\[AAAI 2026\] 10 Open Challenges Steering the Future of Vision-Language-Action Models](../../AAAI2026/robotics/10_open_challenges_steering_the_future_of_vision-language-ac.md)
- [\[ICLR 2026\] Vision-Language-Action Instruction Tuning: From Understanding to Manipulation](../../ICLR2026/robotics/vision-language-action_instruction_tuning_from_understanding_to_manipulation.md)

</div>

<!-- RELATED:END -->
