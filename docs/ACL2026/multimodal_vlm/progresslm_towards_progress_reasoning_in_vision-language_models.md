---
title: >-
  [Paper Note] PROGRESSLM: Towards Progress Reasoning in Vision-Language Models
description: >-
  [ACL2026][Multimodal VLM][Progress Reasoning] This paper defines the "ability to judge task completion stages from a single-frame observation" as the progress reasoning ability of VLMs. It constructs Progress-Bench and P…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Progress Reasoning"
  - "VLM Evaluation"
  - "Embodied AI"
  - "Two-stage Reasoning"
  - "RL Fine-tuning"
date: 2026-05-08
content_hash: 286a1bb2ac47b7cb
---

# PROGRESSLM: Towards Progress Reasoning in Vision-Language Models

**Conference**: ACL2026  
**arXiv**: [2601.15224](https://arxiv.org/abs/2601.15224)  
**Code**: Website / Code / Model / Dataset are mentioned in cache, but specific URLs are not expanded.  
**Area**: Multimodal VLM / Embodied Task Progress Reasoning  
**Keywords**: Progress Reasoning, VLM Evaluation, Embodied AI, Two-stage Reasoning, RL Fine-tuning

## TL;DR
This paper defines the "ability to judge task completion stages from a single-frame observation" as the progress reasoning ability of VLMs. It constructs Progress-Bench and ProgressLM-45K, and demonstrates that explicit learning of "scenario retrieval + mental simulation" is more stable than simple zero-shot prompting.

## Background & Motivation
**Background**: Existing VLMs are proficient at describing "what is present" in a single image and answering local state questions in robotic or embodied tasks. However, many practical systems are more concerned with "where the task has progressed to," such as whether a robot execution has stalled, whether a web agent is close to finishing a goal, or if online RL requires dense rewards.

**Limitations of Prior Work**: Traditional progress estimation often relies on task-specific regressors or converts the problem into indirect objectives like trajectory re-ranking or pairwise comparison. These methods either depend heavily on specific task distributions or require the entire trajectory as context, failing to answer a more general question: given a complete task demonstration and a current single-frame observation, can a VLM directly infer normalized progress.

**Key Challenge**: A single-frame observation contains static visual evidence, whereas progress is essentially a state variable over the temporal dimension. The model cannot simply perform image matching; it must place the current observation back into the task trajectory, determine which stage it belongs to, and how much it has advanced within that stage.

**Goal**: The authors first construct a controllable benchmark to systematically distinguish between visual and text demonstrations, same-view and cross-view scenarios, and answerable versus unanswerable samples. They then evaluate 14 VLMs and finally train a small-scale ProgressLM-3B to verify if progress reasoning can be acquired through explicit supervision and reinforcement learning.

**Key Insight**: The paper draws inspiration from how humans understand task progress: first finding a coarse-grained reference point in memory, then imagining how the state continues to evolve around that reference point. This approach is more interpretable than direct percentage regression and is better at handling uncertainty from cross-view or text demonstrations.

**Core Idea**: Decompose progress estimation into two stages: "scenario retrieval to locate anchors" and "mental simulation to refine percentages," and explicitly teach this reasoning format to VLMs using ProgressLM-45K.

## Method
The proposed method consists of a benchmark, a reasoning paradigm, and a training pipeline. Progress-Bench standardizes the progress reasoning problem, training-free prompting tests whether existing VLMs can activate such capabilities through prompts, and ProgressLM-3B internalizes the two-stage reasoning into model parameters.

### Overall Architecture
The input is a task demonstration $D$ and a current observation $o$. The demonstration can be a sequence of keyframes labeled with progress or text action steps; the observation is a single frame at an intermediate moment of task execution. The model outputs a progress score within $[0, 100\%]$; if the demonstration and observation are inconsistent or if progress cannot be inferred, it outputs N/A.

Progress-Bench is built on RoboMind. Each task trajectory is first labeled with several key steps, then intermediate frames are sampled between adjacent key steps, and fine-grained progress is obtained via linear interpolation $p^*=p_j+\delta(p_{j+1}-p_j)$. Visual demonstrations further distinguish between same-view and cross-view to test whether the model truly understands task states rather than performing pixel-level similarity matching. Unanswerable samples are constructed by modifying demonstrations or editing observations to check if the model actively refuses when semantics do not match.

The inference output of ProgressLM adopts a fixed structure: it first generates `<ref_think>` and `<ref>` to find the demonstration step closest to the current observation, then generates `<score_think>` and `<score>` to estimate whether the current state is ahead of, near, or after this anchor. The training phase uses supervised fine-tuning (SFT) to learn this format, followed by GRPO reinforcement learning to optimize the format, reference points, and progress scores.

### Key Designs
1. **Three-axis Control Evaluation of Progress-Bench**:
	- **Function**: Uses the same task definition to simultaneously test progress error, sequential consistency, and refusal capability.
	- **Mechanism**: Demonstration modalities are divided into visual keyframes and text steps, with visual demonstrations further split into same-view/cross-view; samples are also divided into answerable and unanswerable. This allows decomposing model failures into modes like "not understanding progress," "lack of robustness to cross-view," or "failure to refuse."
	- **Design Motivation**: If only final percentage error is considered, models might guess correctly using fixed templates or few discrete scores; the three-axis design exposes behaviors like score collapse, cumulative failure in text states, and over-refusal.

2. **Two-stage Reasoning: Scenario Retrieval + Mental Simulation**:
	- **Function**: Converts continuous progress estimation into a reasoning process of first selecting a reference step and then performing local comparison.
	- **Mechanism**: The first stage retrieves the step $j^\star$ in the demonstration closest to the current observation; the second stage judges whether the current frame is before, near, or after $j^\star$ based on state differences, then provides a score.
	- **Design Motivation**: Directly outputting a progress percentage often degrades into heuristics like 0/50/100; explicit anchors constrain the model's scores, aligning progress estimation with interpretable task states.

3. **SFT Cold Start + GRPO Calibration**:
	- **Function**: Enables small models to not only write in the two-stage format but also bind the format to correct anchors and scores.
	- **Mechanism**: ProgressLM-25K-CoT uses reasoning sequences with `<ref>` and `<score>` for autoregressive supervision, with the loss $\mathcal{L}_{SFT}=-\frac{1}{N}\sum_i\log P_\theta(r_i^*|D_i,o_i)$; subsequently, 20K samples are used for GRPO with the reward $R=\alpha R_{format}+\beta R_{ref}+\gamma R_{score}$, with weight ratios $1:6:3$.
	- **Design Motivation**: SFT establishes the reasoning skeleton, while RL further penalizes anchor errors and score inaccuracies, particularly improving calibration in cross-view and unanswerable scenarios.

### Loss & Training
The training data is ProgressLM-45K, where 25K is for CoT supervised cold start and 20K is for RL refinement; training tasks do not overlap with Progress-Bench. SFT uses LLaMA-Factory and LoRA rank 8, learning rate $1\times10^{-4}$, effective batch size of 64 on 4 H100 GPUs, for 2 epochs; RL uses EasyR1/GRPO, actor learning rate $1\times10^{-6}$, KL coefficient 0.01, with $n=16$ rollouts sampled per prompt, trained on 16 H100 GPUs for 2 epochs (approx. 23 hours).

Inference evaluation uses a large maximum output length to ensure the model can generate the full two-stage reasoning. The paper compares direct prediction, training-free prompting, and training-based ProgressLM to avoid misattributing prompt format gains to training gains.

## Key Experimental Results

### Main Results
Progress-Bench contains 240 task trajectories and 3,325 sampled observations, evaluating 14 VLMs. Metrics for answerable samples include NSE↓, PRC↑, and AFRR↓; lower NSE indicates smaller percentage error, higher PRC indicates more consistent progress ranking along the trajectory, and lower AFRR indicates a lower ratio of answerable samples being incorrectly refused.

| Model | Macro NSE↓ | Macro PRC↑ | Macro AFRR↓ | Description |
|------|-------------|-------------|--------------|------|
| GPT-5 | 21.3 | 72.6 | 4.2 | Strong closed-source baseline, still affected by modality |
| GPT-5-mini | 20.9 | 71.4 | 5.1 | Small closed-source model, PRC close to GPT-5 |
| Qwen2.5-VL-3B | 39.0 | 20.2 | 0.01 | Base for ProgressLM, weak at direct prediction |
| ProgressLM-3B-SFT | 24.0 | 59.3 | 7.8 | NSE dropped significantly and PRC improved after SFT |
| ProgressLM-3B-RL | 17.5 | 77.0 | 7.0 | Better Macro NSE/PRC than GPT-5 |

### Ablation Study
| Configuration | Key Metric | Description |
|------|----------|------|
| Qwen2.5-VL-3B, same-view | NSE 29.2 / PRC 43.0 / AFRR 9.9 | Small model has basic visual matching capability in same-view |
| Qwen2.5-VL-3B, cross-view | NSE 33.4 / PRC 28.9 / AFRR 6.5 | Ranking capability drops significantly in cross-view |
| ProgressLM-3B-RL, same-view | NSE 10.3 / PRC 93.5 / AFRR 0.1 | Two-stage training is very stable in same-view |
| ProgressLM-3B-RL, cross-view | NSE 15.2 / PRC 88.8 / AFRR 11.7 | Cross-view error increases, but PRC remains high |
| Qwen2.5-VL-7B, vision no-think | PRC 33.7 / NSE 34.0 / AFRR 28.3 | Direct prediction by 7B remains unreliable |
| Qwen2.5-VL-7B, vision training-based | PRC 85.7 / NSE 13.4 / AFRR 32.4 | Training-based reasoning significantly improves ranking and error |
| Qwen2.5-VL-7B, text training-based | PRC 50.5 / NSE 26.6 / AFRR 1.4 | Text demo is harder, but training still yields gains |

### Key Findings
- Existing VLMs often collapse progress scores into a few heuristic values like 0%, 50%, or 100%, leading to negative or undefined PRC; ProgressLM-SFT/RL shows more continuous distributions.
- Training-free reasoning only provides conditional gains on strong models; small models often just comply with the output format without truly improving progress understanding.
- Text demonstrations are more difficult than visual demonstrations because the model must accumulate implicit states; e.g., the same text step "manipulating the pot lid" may correspond to entirely different object states.
- Identifying unanswerable samples cannot rely solely on UDA; while Intern3.5-VL-38B can refuse many abnormal samples, its high AFRR on answerable samples suggests that over-conservatism also harms system usability.

## Highlights & Insights
- Transforming VLM "task progress" from a vague capability into a measurable metric is the most valuable contribution of this paper. It asks not just if the model understands the image, but if it can place a single-frame state into a complete task timeline.
- The two-stage reasoning design is simple but captures the essence: finding an anchor then estimating details is more aligned with human progress judgment and makes errors easier to diagnose.
- ProgressLM-3B exceeding GPT-5 in Macro NSE/PRC shows that such capabilities are not exclusive to model scale; targeted supervision and reward design can push small models to a more stable state.
- Unanswerable samples in the benchmark are crucial. Once progress estimation is used for robot monitoring or agent self-improvement, incorrectly giving a "seemingly precise" percentage can be more dangerous than refusing to answer.

## Limitations & Future Work
- The authors acknowledge that Progress-Bench primarily focuses on robotic manipulation tasks where progress is relatively clear and monotonic; in open-ended tasks, cyclic tasks, or scenarios with shifting goals, progress might not be describable by a single percentage.
- The training data for ProgressLM also consists of structurally similar manipulation tasks; migrating to web agents, code agents, or human activities may require new demonstration formats and progress notations.
- Experiments show that cross-view AFRR still increases, indicating the model is not yet perfectly calibrated between "conservative refusal" and "robust estimation."
- Future work could integrate progress scores into online control: e.g., using low progress growth to detect stalls, using high uncertainty to trigger re-planning, or using ProgressLM as a dense reward generator to assist RL.

## Related Work & Insights
- **vs. Task-specific Progress Regressors**: Traditional methods usually train regression models in fixed tasks or environments, with generalization depending on the training distribution; this paper treats progress estimation as a general reasoning problem for VLMs, requiring only a single-frame observation.
- **vs. Trajectory Re-ranking / Pairwise Comparison**: These methods obtain estimates indirectly by re-ranking frames or comparing relative progress, depending on full trajectory context; ProgressLM infers absolute progress from demonstrations and single frames, making it more suitable for online monitoring.
- **vs. Inference-time CoT Prompting**: Training-free prompts can make models explicitly state reference steps, but gains are unstable; the results here remind us that complex reasoning formats require training and reward binding, otherwise they are just "formatted explanations."
- **Insight**: For agent evaluation, a "task progress benchmark" could be designed similarly: given a goal, history trajectory, and current state, have the model judge completion, next bottleneck, and whether it is unanswerable.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Defines progress estimation as a structured reasoning capability for VLMs and builds a three-axis benchmark with a clear problem setting.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 14 VLMs, same/cross-view, text/visual demonstrations, unanswerable samples, and 3B/7B training scaling, providing solid evidence.
- Writing Quality: ⭐⭐⭐⭐☆ Clear storyline with information-dense tables; some large tables require readers to synthesize macro conclusions.
- Value: ⭐⭐⭐⭐⭐ Provides direct inspiration for embodied AI, long-horizon agent monitoring, and RL reward shaping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[ACL 2026\] VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning](vl-calibration_decoupled_confidence_calibration_for_large_vision-language_models.md)
- [\[ICLR 2026\] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](../../ICLR2026/multimodal_vlm/seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)
- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)

</div>

<!-- RELATED:END -->
