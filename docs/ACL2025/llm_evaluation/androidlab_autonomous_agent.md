---
title: >-
  [Paper Note] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents
description: >-
  [LLM Evaluation] AndroidLab is proposed as a systematic Android agent evaluation and training framework, consisting of a unified operating environment, a reproducible benchmark with 138 tasks, and an instruction dataset of 94.3K steps. Through fine-tuning, the success rate of open-source LLMs is improved from 4.59% to 21.50%.
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 96ea22d48366e861
---

# AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents

## Paper Information

- **Conference**: ACL 2025
- **arXiv**: [2410.24024](https://arxiv.org/abs/2410.24024)
- **Code**: [https://github.com/THUDM/Android-Lab](https://github.com/THUDM/Android-Lab)
- **Area**: LLM Evaluation
- **Keywords**: Android Agent, Mobile Agent, Benchmark, Instruction Tuning, Multimodal Models

## TL;DR

AndroidLab is proposed as a systematic Android agent evaluation and training framework, consisting of a unified operating environment, a reproducible benchmark with 138 tasks, and an instruction dataset of 94.3K steps. Through fine-tuning, the success rate of open-source LLMs is improved from 4.59% to 21.50%.

## Background & Motivation

- **Background**: While the use of LLMs/LMMs as mobile autonomous agents is increasingly studied, existing training and evaluation efforts lack systematicity—almost all benchmarks only test the performance of prompt engineering in closed-source models.
- **Limitations of Prior Work**: Static benchmarks (PixelHelp, AITW) lack interactivity, while dynamic benchmarks (AndroidEnv, AndroidWorld) suffer from poor reproducibility or lack support for constructing fine-tuning training datasets. Furthermore, no unified framework exists to compare open-source and closed-source, as well as text and multimodal models simultaneously.
- **Key Challenge**: A unified framework is needed to simultaneously support (1) fair comparison across different modalities, (2) reproducible evaluation, and (3) efficient construction of training data to improve open-source models.
- **Design Motivation**: Improvements achieved through prompt engineering are limited (complex reasoning frameworks only bring marginal gains while drastically increasing inference time); fine-tuning small-scale open-source models is essential to narrow the gap with closed-source models.

## Method

### Overall Architecture

AndroidLab comprises three components: (1) a standardized environment that unifies the action spaces for LLMs and LMMs; (2) a reproducible benchmark with 138 tasks across 9 pre-installed apps based on the Android Virtual Device (AVD); and (3) the Android Instruct dataset, containing 10.5K interaction trajectories and 94.3K steps.

### Key Designs

1. **Unified Dual-Modal Operating Mode**: XML mode (text input for LLMs) and SoM mode (screenshots with layout tags for LMMs) share the exact same action space and operating targets, ensuring fair comparisons between current LLMs and LMMs.
2. **Subgoal Decomposition Evaluation**: Each task is decomposed into multiple subgoals, with each subgoal's status independently validated via UI tree structure matching, bypassing the limitations of traditional action path matching.
3. **Reproducible Offline Environment**: All target applications are pre-installed and run offline in the AVD image with frozen system time and location, eliminating external network and temporal dependencies.

### Action Space

Six basic operations (Tap, Swipe, Type, Long Press, Home, Back) plus one termination operation (Finish), supporting the return of execution feedback.

### Data Construction Pipeline

Task Derivation & Expansion → LLM/LMM Self-Exploration (automatic trajectory generation) → Human Annotation (a 4-step workflow: feasibility check → UI familiarity → execution logging → cross-verification).

## Key Experimental Results

### Main Results: Comparison of Success Rates in XML and SoM Modes

| Mode | Model | Success Rate (SR) | Subgoal SR | Reverse Redundancy Ratio | Ratio of Reasonable Operations |
|------|------|-----------|---------|-----------|----------|
| XML | GPT-4-1106-Preview | **31.16%** | 38.21% | 66.34 | 86.24 |
| XML | GPT-4o | 25.36% | 30.56% | **107.45** | 86.56 |
| XML | Qwen2-7B (Base) | 4.35% | 4.95% | - | 67.26 |
| XML+SFT | LLaMA3.1-8B-ft | **23.91%** | 30.31% | 75.58 | **92.46** |
| SoM | GPT-4o | **31.16%** | 35.02% | 87.32 | 85.36 |
| SoM | Claude-3.5-Sonnet | 28.99% | 32.66% | **113.41** | 81.16 |
| SoM | CogVLM2 (Base) | 0.72% | 0.72% | - | 17.97 |
| SoM+SFT | Qwen2-VL-7B-ft | **18.12%** | 22.64% | 65.23 | 88.29 |

### Ablation Study: Impact of Different Reasoning Frameworks on Success Rate

| Mode | Model | Base | +ReAct | +SeeAct |
|------|------|------|--------|---------|
| XML | GPT-4o | 25.36% | **33.33%** | 24.64% |
| XML | Gemini-1.5-Pro | 18.84% | **31.16%** | 21.01% |
| SoM | GPT-4o | 31.16% | **31.88%** | 30.43% |

### Key Findings

1. **Significant Fine-Tuning Performance**: LLM success rate increased from 4.59% to 21.50% (+368%), and LMM from 1.93% to 13.28% (+588%). Post-fine-tuning, open-source models approach or even surpass some closed-source models.
2. **ReAct Framework is Only Significantly Effective in XML Mode**: XML + ReAct improves GPT-4o's SR from 25.36% to 33.33%, but the improvement is marginal under the SoM mode.
3. **Trade-off Between Efficiency and Quality**: Fine-tuned models generate an average of only 4.96 tokens/step, whereas ReAct requires 23.56 tokens/step and SeeAct requires 129.12 tokens/step.
4. **Substantial Improvement in Action Efficiency**: Post-fine-tuning, the Ratio of Reasonable Operations (ROR) typically exceeds 88%, which is much higher than the pre-fine-tuning range of 17% to 67%.
5. **Screen Size Affects Performance**: Standard smartphone sizes (Pixel 7/8 Pro) perform the best, while screens that are too small or too large degrade performance.
6. **Best Closed-Source Performance is only 31.16%**: The AndroidLab benchmark is heavily challenging, with even the most powerful models achieving less than 50% success rate.

## Highlights & Insights

- The first Android agent framework offering unified LLM and LMM evaluation, with fully aligned action spaces.
- The subgoal-decomposed evaluation mechanism is more accurate and flexible than trajectory path matching.
- The open-source training dataset effectively narrows the gap between open-source and closed-source models, validating the feasibility of the fine-tuning approach.
- A complete data construction toolchain is provided (online annotation tool + ADB + Accessibility Service).

## Limitations & Future Work

- It only covers 138 tasks across 9 applications, which is limited in scale compared to real-world application diversity.
- The offline environment cannot support task scenarios requiring online network interaction.
- The preset maximum of 25 steps per task in the evaluation might be insufficient for some complex tasks.
- The training data is derived from the benchmark applications themselves; generalization to unseen applications remains to be validated.
- Discrepancies still exist between the AVD environment and physical mobile devices.

## Related Work & Insights

- **Mobile Benchmarks**: PixelHelp (Li et al., 2020), AITW (Rawles et al., 2023), AndroidWorld (Rawles et al., 2024), B-MOCA (Lee et al., 2024)
- **Mobile Agents**: AppAgent (Yang et al., 2023b), Auto-GUI (Zhan & Zhang, 2023), CogAgent (Hong et al., 2023)
- **Web Agents**: WebGPT (Nakano et al., 2021), AutoWebGLM (Lai et al., 2024), MindAct (Deng et al., 2023)
- **General Code/API Agents**: HumanEval (Chen et al., 2021), ToolBench (Guo et al., 2024)

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Rating| 8.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)

</div>

<!-- RELATED:END -->
