---
title: >-
  [Paper Note] Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation
description: >-
  [CVPR2026][Multimodal VLM][GUI automation] This paper proposes CES (Coordinator-Executor-State Tracker), a multi-agent framework coupled with a staged execution-feedback reinforcement learning algorithm. By decoupling hi…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "GUI automation"
  - "long-horizon tasks"
  - "multi-agent framework"
  - "reinforcement learning"
  - "state tracking"
  - "task scheduling"
date: 2026-05-08
content_hash: a0d27faa961cd8cf
---

# Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation

**Conference**: CVPR2026
**arXiv**: [2511.22235](https://arxiv.org/abs/2511.22235)
**Code**: [hehehahi4/CES](https://github.com/hehehahi4/CES)
**Area**: Multimodal VLM
**Keywords**: GUI automation, long-horizon tasks, multi-agent framework, reinforcement learning, state tracking, task scheduling

## TL;DR

This paper proposes CES (Coordinator-Executor-State Tracker), a multi-agent framework coupled with a staged execution-feedback reinforcement learning algorithm. By decoupling high-level task planning from low-level execution, and through dedicated training of the Coordinator and State Tracker, CES significantly improves GUI agent planning and state management capabilities on long-horizon tasks.

## Background & Motivation

1. **Conflicting capabilities in single-agent systems**: Existing end-to-end GUI agents attempt to couple heterogeneous capabilities—task planning, multi-step reasoning, GUI element grounding, and precise action execution—within a single model. With limited parameters, simultaneously mastering both high-level and low-level abilities is difficult, and catastrophic capability collapse tends to occur as task complexity increases.
2. **Lack of task state awareness**: In long-horizon tasks, agents primarily rely on screenshots to infer progress. However, screenshots are insufficient and unreliable state representations—recurring home screens, out-of-distribution interfaces, and similar situations make progress estimation difficult.
3. **Limitations of the SFT paradigm**: Supervised fine-tuning relies heavily on large-scale, high-quality trajectory annotations, which are costly to acquire and generalize poorly.
4. **Insufficiency of single-step RL**: Although existing RL methods achieve reasonable performance on simple tasks, they still train a single policy network and do not address the capability coupling problem.
5. **Lack of optimization in multi-agent frameworks**: Existing multi-agent approaches mostly assign roles via general-purpose VLMs and prompt engineering, without deep specialization for each role.
6. **Temporal verification experiments**: The paper designs screenshot temporal ordering experiments and finds that accuracy drops sharply as the step gap increases, empirically demonstrating that screenshots cannot adequately represent task state.

## Method

### Overall Architecture: CES Collaborative Loop

Drawing an analogy to operating system design, three specialized agents form a cyclic collaboration:

- **Coordinator (CPU / planning core)**: Integrates the user's high-level instruction $q$, the state summary $m^{t-1}$ provided by the State Tracker, and the current screenshot $s^t$ to decompose and generate a clear atomic instruction $l^t = \pi_c(q, m^{t-1}, s^t)$.
- **Executor (I/O device / execution terminal)**: A frozen pretrained GUI model that executes action $u^t = (th^t, a^t) = \pi_e(l^t, s^t)$ based solely on the atomic instruction $l^t$ and current screenshot $s^t$, without needing to understand long-term intent.
- **State Tracker (dynamic memory)**: A language model that does not directly perceive the GUI environment. It generates a new high-semantic natural-language state summary $m^t = \pi_s(q, m^{t-1}, u^t)$ by interpreting the Executor's output $u^t$, the user intent $q$, and the previous state $m^{t-1}$.

### Staged Execution-Feedback Reinforcement Learning

**Warm-up SFT**: The Coordinator and State Tracker are first fine-tuned with supervised learning on existing trajectory data, enabling them to learn basic role responsibilities and output formats.

**Execution-feedback reward function**: Rather than directly evaluating the quality of intermediate outputs, outputs are passed to the Executor for execution, and a rule-based reward function provides objective scoring:

$$R = \alpha_1 R_{format} + \alpha_2 R_{executor}, \quad R_{executor} = \gamma_1 R_{type} + \gamma_2 R_{param}$$

where $R_{format}$ rewards format correctness, $R_{type}$ rewards correct action type, and $R_{param}$ rewards correct action parameters.

**Stage 1 — Training the Coordinator**: The Executor is frozen, and ground-truth states are used as $m^{t-1}$ input. The Coordinator's planning policy is optimized using the GRPO algorithm with execution-feedback rewards.

**Stage 2 — Training the State Tracker**: The trained Coordinator and Executor are frozen. State summaries generated by the State Tracker pass through the full CES loop, and the resulting execution-feedback reward is backpropagated to optimize the State Tracker, enabling it to learn to produce state information most valuable to the Coordinator.

### Training Details

- Coordinator backbone: Qwen2.5-VL-7B; State Tracker backbone: Qwen3-4B
- SFT stage: LLaMA Factory, 1 epoch, lr=5e-5
- RL stage: Verl framework; Coordinator 10 epochs lr=1e-6; State Tracker 5 epochs
- Reward coefficients: $\alpha_1=0.1, \alpha_2=0.9$; $\gamma_1=0.2, \gamma_2=0.8$
- Training Executor: GUI-R1-7B (frozen); Hardware: 8×80G GPU

## Key Experimental Results

### Main Results: Long-Horizon Task Performance (Table 1)

Evaluated on three benchmarks—AITZ (avg. 7.5 steps), AMEX (avg. 12.8 steps), and GUI-Odyssey (avg. 15.3 steps):

| Model | Method | AITZ SR | AMEX SR | GUI-Odyssey SR |
|-------|--------|---------|---------|----------------|
| Qwen2.5-VL-7B | Zero Shot | 18.11 | 35.10 | 34.37 |
| GUI-R1-7B | RL | 30.59 | 43.69 | 38.79 |
| GUI-Owl-7B | RL | 32.70 | 40.48 | 35.82 |
| + GPT-5 | Multi-Agent | 40.55 | 35.80 | 42.47 |
| **+ CES (Ours)** | **Multi-Agent** | **43.05** | **48.48** | **53.69** |

CES improves average Type accuracy by 10.38% over the GUI-R1-7B baseline, and raises GUI-Odyssey SR from 38.79% to 53.69% (+14.9%).

### Generalization (Table 2)

CES functions as a plug-and-play module and yields significant improvements across Executors of different scales:

| Executor | Setting | AMEX SR | GUI-Odyssey SR |
|----------|---------|---------|----------------|
| UI-R1-3B | Baseline → CES | 35.81 → 43.38 (+7.57) | 32.49 → 38.04 (+5.55) |
| GUI-Owl-7B | Baseline → CES | 40.48 → 47.24 (+6.76) | 35.82 → 46.65 (+10.83) |
| GUI-Owl-32B | Baseline → CES | 43.16 → 52.05 (+8.89) | 39.60 → 56.75 (+17.15) |

### Ablation Study (Table 3)

| Configuration | AMEX SR | GUI-Odyssey SR |
|---------------|---------|----------------|
| Full CES | 48.48 | 53.69 |
| w/o Coordinator | 33.27 (−15.21) | 39.15 (−14.54) |
| w/o State Tracker | 42.08 (−6.40) | 42.52 (−11.17) |
| w/o RL (SFT only) | 36.54 (−11.94) | 42.89 (−10.80) |

Removing any component or the RL stage leads to significant performance degradation, validating the necessity of each component and training strategy.

## Highlights & Insights

1. **OS-inspired design philosophy**: GUI automation is analogized to the CPU-I/O-Memory architecture of an operating system, elegantly decoupling planning, execution, and state management.
2. **State Tracker innovation**: A pure language model is used for dynamic context compression and state summarization, shifting state understanding from a high-dimensional visual space to a low-dimensional semantic space, nearly eliminating State Loss errors (14% → 2%).
3. **Execution-feedback reward**: This design cleverly addresses the difficulty of directly evaluating abstract tasks (planning/state tracking) by using downstream execution outcomes to guide upstream optimization.
4. **Plug-and-play generalizability**: The lightweight combination of a 7B Coordinator and 4B State Tracker substantially benefits diverse Executors; notably, the 7B+4B combination achieves performance comparable to a 32B single model.
5. **Thorough empirical validation**: Preliminary temporal ordering experiments, three long-horizon benchmarks, multi-scale Executor generalization, detailed ablations, and failure case analysis are all provided.

## Limitations & Future Work

1. **Executor remains a bottleneck**: Failure case analysis shows that the performance bottleneck has shifted to the Executor's perceptual limitations (Perception Error and Generalization Failure), which CES cannot address.
2. **Staged rather than joint training**: The Coordinator and State Tracker are trained separately; the possibility of joint training or co-evolution has not been explored (noted in the paper's Future Work).
3. **Domain applicability**: Validation is limited to mobile GUI scenarios and has not been extended to other GUI environments such as web or desktop.
4. **Dependency on state annotation quality**: Stage 1 relies on ground-truth state annotations; the feasibility of obtaining such annotations in real-world settings remains to be verified.
5. **Computational overhead**: Three models are executed serially (7B + frozen Executor + 4B), and inference latency may be an obstacle to practical deployment.

## Related Work & Insights

- **vs GUI-R1**: Both apply RL to train GUI agents, but GUI-R1 trains a single end-to-end model, whereas CES decouples high-level and low-level components and specifically optimizes high-level scheduling, raising GUI-Odyssey SR from 38.79% to 53.69%.
- **vs SWIRL**: Both are multi-stage workflow methods. SWIRL achieves SR=51.65% on GUI-Odyssey; CES reaches 53.69% and leads on other benchmarks as well.
- **vs Mobile-Agent-v3 / MobiAgent**: Both are multi-agent frameworks, but these methods assign roles via prompt engineering without dedicated optimization. CES trains each role deeply via execution-feedback RL.
- **vs GPT-5 Multi-Agent**: Using GPT-5 as Coordinator and State Tracker produces unstable results (with some metrics declining), whereas CES's dedicated trained models consistently and significantly outperform the prompt-based approach.

## Rating

- Novelty: ⭐⭐⭐⭐ — The OS-inspired three-role decoupled design combined with the execution-feedback RL training paradigm demonstrates good originality.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks, multi-scale generalization, detailed ablations, failure case analysis, and preliminary experiments are all included.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear, motivation is well articulated, and the preliminary experiment design is elegant.
- Value: ⭐⭐⭐⭐ — The plug-and-play high-level scheduling module has practical value for the GUI agent community, and the staged training paradigm is broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)
- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](../../ACL2026/multimodal_vlm/don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)
- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)

</div>

<!-- RELATED:END -->
