---
title: >-
  [Paper Note] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning
description: >-
  [NeurIPS 2025][Robotics][VLA reasoning] ThinkAct proposes a dual-system framework that applies action-aligned visual rewards to fine-tune MLLMs via reinforcement learning…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "VLA reasoning"
  - "reinforcement learning"
  - "visual latent planning"
  - "embodied reasoning"
  - "dual-system architecture"
date: 2026-05-08
content_hash: 484920abfcb6fb98
---

# ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning

**Conference**: NeurIPS 2025
**arXiv**: [2507.16815](https://arxiv.org/abs/2507.16815)  
**Code**: [Project Page](https://jasper0314-huang.github.io/thinkact-vla/)  
**Area**: Robotics
**Keywords**: VLA reasoning, reinforcement learning, visual latent planning, embodied reasoning, dual-system architecture

## TL;DR

ThinkAct proposes a dual-system framework that applies action-aligned visual rewards to fine-tune MLLMs via reinforcement learning, eliciting embodied reasoning capabilities and compressing reasoning plans into visual latent representations to guide a downstream action model—realizing a "think before act" VLA reasoning paradigm.

## Background & Motivation

The central dilemma of current VLA models lies in their end-to-end mapping from visual and textual inputs directly to low-level actions, lacking explicit reasoning and planning capabilities. Three specific limitations are identified:

**Lack of long-horizon planning**: Existing VLA models such as OpenVLA and TraceVLA perform well on short-horizon skills but struggle with long-horizon manipulation tasks requiring multi-step reasoning, as they lack intermediate reasoning steps to decompose complex goals.

**Costly CoT supervision data and overfitting risk**: Methods such as ECoT and RAD attempt to incorporate chain-of-thought (CoT) reasoning but rely on supervised fine-tuning with CoT annotations generated offline by MLLMs. Generating high-quality reasoning trajectories is expensive, and models tend to overfit to specific visual scenes or reasoning patterns.

**Existing RL reasoning lacks action alignment**: Works such as Video-R1 apply RL to VLM reasoning but use QA-style accuracy rewards, which cannot support long-horizon planning and fail to establish a meaningful connection between reasoning and real action execution.

The core insight of ThinkAct is that **action-aligned visual feedback** (rather than QA-style rewards) should be used to guide MLLMs in learning embodied reasoning, and that compressing reasoning outputs into compact visual latent representations bridges high-level planning and low-level control.

## Method

### Overall Architecture

ThinkAct adopts a dual-system architecture: a reasoning MLLM $\mathcal{F}_\theta$ handles high-level planning (System 2 slow thinking), while a DiT action model $\pi_\phi$ handles low-level control (System 1 fast control). The two components are connected via a **visual plan latent representation** $c_t$. At each reasoning step, the MLLM receives the current observation $o_t$ and instruction $l$, and generates reasoning text along with a visual plan $c_t$; the action model then predicts the next $N$ executable actions conditioned on $c_t$. The two modules operate asynchronously—the MLLM reasons at low frequency while the action model controls at high frequency.

### Key Designs

1. **Action-aligned visual reward design**: This is ThinkAct's most central contribution. High-level planning is formulated as predicting a 2D gripper trajectory $\tau = [p_k]_{k=1}^K$ (with $K=8$ keypoints), and two reward signals are designed:

    - **Goal reward** $r_{\text{goal}}$: Measures proximity between the start and end points of the predicted and ground-truth trajectories, $r_{\text{goal}} = \frac{1}{2}(f(p_1, \hat{p}_1) + f(p_K, \hat{p}_K))$, where $f(p, p') = \max(0, 1 - \|p - p'\|_2^2)$. This encourages the model to anticipate task goal achievement.
    - **Trajectory reward** $r_{\text{traj}}$: Uses Dynamic Time Warping (DTW) distance to measure distributional alignment between the predicted and ground-truth trajectories, $r_{\text{traj}} = \max(0, 1 - d(\tau, \hat{\tau}))$. This ensures the predicted trajectory corresponds to physically plausible gripper motion.
    - The final reward is $r = 0.9 r_{\text{visual}} + 0.1 r_{\text{format}}$, with the visual reward being dominant.

2. **GRPO reinforcement fine-tuning**: Group Relative Policy Optimization is used to fine-tune the MLLM. Given input $(o_t, l)$, $M$ diverse responses are first sampled from the old policy; their respective rewards are computed and used to derive group-relative advantages $A_i$ to guide optimization. Compared to standard SFT, RL allows the model to freely explore reasoning paths rather than memorizing annotated data, while the visual feedback in the reward provides an embodied grounding anchor. QA data (RoboVQA, failure detection, etc.) is also incorporated to enhance general reasoning capabilities.

3. **Visual latent planning bridging reasoning and execution**: The reasoning embedding $v_t$ and visual plan embedding $c_t$ generated by the MLLM are produced internally. $c_t$ is mapped to the action model's input space via a Q-Former latent projector (32 queries), serving as conditioning information to guide the DiT diffusion policy in predicting actions. Crucially, $c_t$ distills long-horizon spatiotemporal planning intent, enabling the action model to leverage high-level reasoning to improve the robustness of low-level control.

### Loss & Training

A multi-stage training procedure is adopted:
- **SFT cold start**: The MLLM is fine-tuned for 20K steps on OXE trajectory data, RoboVQA, EgoPlan-IT, and Video-R1-CoT data to learn the correct output format and basic reasoning capabilities.
- **GRPO reinforcement fine-tuning**: RL training is conducted for 6K steps using visual trajectory data from OXE and Something-Something V2, along with QA data.
- **Reasoning-augmented action adaptation**: The MLLM is frozen, and the action model is trained via imitation learning on the target environment (e.g., LIBERO) with loss $\mathcal{L}_{\text{IL}}(\phi) = \mathbb{E}[\ell(\pi_\phi(c_t, o_i, l), a_i)]$.

## Key Experimental Results

### Main Results

**Robot Manipulation — SimplerEnv & LIBERO**

| Benchmark | Metric | ThinkAct | DiT-Policy | CoT-VLA | Magma | Gain (vs DiT) |
|-----------|--------|----------|-----------|---------|-------|---------------|
| SimplerEnv-Google-VM | Success Rate | **71.5%** | 56.0% | – | 68.4% | +15.5% |
| SimplerEnv-Google-VA | Success Rate | **65.1%** | 48.2% | – | 62.6% | +16.9% |
| SimplerEnv-Bridge-VM | Success Rate | **43.8%** | 32.4% | – | 35.4% | +11.4% |
| LIBERO Overall | Success Rate | **84.4%** | 76.8% | 83.9% | – | +7.6% |
| LIBERO-Long | Success Rate | **70.9%** | 57.6% | 69.0% | – | +13.3% |

**Embodied Reasoning Tasks**

| Benchmark | Metric | ThinkAct | Qwen2.5-VL* | InternVL3 | Gain |
|-----------|--------|----------|-------------|-----------|------|
| EgoPlan-Bench2 | Accuracy | **48.2%** | 45.7% | 36.2% | +2.5% |
| RoboVQA | BLEU Mean | **59.8** | 55.7 | 35.3 | +4.1 |
| OpenEQA | LLM Score | **56.2%** | 52.0% | 55.5% | +0.7% |

### Ablation Study

| Configuration | SimplerEnv | EgoPlan | RoboVQA | Notes |
|---------------|-----------|---------|---------|-------|
| ThinkAct (Full) | **60.1** | **48.2** | **59.8** | Both rewards used |
| w/o $r_{\text{traj}}$ | 59.2 | 47.9 | 58.5 | Planning coherence degrades |
| w/o $r_{\text{goal}}$ | 59.1 | 47.6 | 58.9 | Long-horizon reasoning weakened |
| w/o both visual rewards | 56.9 | 47.2 | 58.3 | QA reward only, nearly at SFT level |
| SFT cold start | 56.4 | 46.4 | 57.9 | No RL, lowest performance |

### Key Findings

- **RL-enhanced reasoning significantly outperforms SFT**: After RL fine-tuning, the model performs more nuanced environmental analysis and multi-step reasoning rather than merely attending to the current state.
- **Strong few-shot adaptation**: In the LIBERO 10-shot setting, ThinkAct surpasses state-of-the-art methods on all tasks, outperforming Magma by 7.3% on LIBERO-Goal and 9.5% on LIBERO-Spatial.
- **Emergent self-correction behavior**: Upon perceiving execution failure (e.g., object dropping), the reasoning MLLM can generate "Let's reconsider" and revise its plan, guiding the gripper back to the drop location for re-grasping.

## Highlights & Insights

- **Elegant visual reward design**: Translating the abstract notion of "reasoning quality" into a quantifiable 2D trajectory matching problem resolves the long-standing challenge of defining reward signals for embodied reasoning.
- **Asynchronous dual-system architecture**: The combination of slow MLLM reasoning and fast action model control is conceptually elegant; each reasoning step corresponds to $N$ action execution steps, balancing reasoning depth with control frequency.
- **Paradigm shift from SFT to RL training**: The work demonstrates that, in embodied AI, RL can elicit reasoning capabilities beyond what supervised data provides—analogous to its impact in language model reasoning.

## Limitations & Future Work

- The system inherits hallucination issues from the pre-trained MLLM, potentially generating plans that reference incorrect object attributes or spatial relations.
- The reasoning overhead increases inference latency (approximately 17% slower than OpenVLA), which constrains real-time applicability despite significant performance gains.
- 2D trajectories as a planning representation have limited expressiveness, failing to encode depth information and complex 3D interactions.
- The reward signal relies on an off-the-shelf gripper detector, whose precision directly affects training quality.

## Related Work & Insights

- Compared to CoT-VLA (which replaces language CoT with visual sub-goal frames), ThinkAct uses RL instead of SFT to generate reasoning, offering greater scalability.
- ThinkAct is complementary to RAD (which learns reasoning from action-free human videos) by further aligning reasoning with action execution.
- Future directions include extending visual reward signals to 3D space, or introducing online RL to enable the MLLM to learn through direct interaction in simulators.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Action-aligned visual rewards combined with visual latent planning to bridge reasoning and execution—original and complete contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 5 datasets across manipulation and reasoning benchmarks, with detailed ablations, few-shot, and self-correction analyses
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and polished figures, though notation is occasionally dense
- **Value**: ⭐⭐⭐⭐⭐ Introduces scalable reasoning capabilities to VLA models; the dual-system architecture and RL training paradigm carry broad influence

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](../../CVPR2026/robotics/fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[NeurIPS 2025\] SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)
- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)

</div>

<!-- RELATED:END -->
