---
title: >-
  [Paper Note] RISK: A Framework for GUI Agents in E-commerce Risk Management
description: >-
  [ACL 2026][LLM Safety][GUI agents] The RISK framework is proposed, comprising a domain dataset (RISK-Data: 8,492 single-step + 2,386 multi-step trajectories), a benchmark (RISK-Bench)…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "GUI agents"
  - "E-commerce risk management"
  - "Reinforcement fine-tuning"
  - "Web interaction"
  - "Multi-step reasoning"
date: 2026-05-08
content_hash: d6e6b904e9591d7b
---

# RISK: A Framework for GUI Agents in E-commerce Risk Management

**Conference**: ACL 2026  
**arXiv**: [2509.21982](https://arxiv.org/abs/2509.21982)  
**Code**: [GitHub](https://github.com/RenqiChen/RISK-GUI)  
**Area**: GUI Agents  
**Keywords**: GUI agents, E-commerce risk management, Reinforcement fine-tuning, Web interaction, Multi-step reasoning

## TL;DR

The RISK framework is proposed, comprising a domain dataset (RISK-Data: 8,492 single-step + 2,386 multi-step trajectories), a benchmark (RISK-Bench), and a GRPO-based reinforcement fine-tuning method (RISK-R1). For GUI agents in e-commerce risk management, the 7B model achieves a 70.5% online success rate, outperforming SOTA baselines with only 7.2% of the parameters.

## Background & Motivation

**Background**: E-commerce risk management requires aggregating heterogeneous information (transaction details, user profiles, site verification, etc.) from various external websites. This information is often embedded in dynamically loaded sub-pages, interactive elements, or complex DOM structures, requiring multi-step stateful web interactions.

**Limitations of Prior Work**: Traditional crawlers cannot handle stateful event-driven interactions. Existing GUI agents are mostly limited to single-step operations and lack multi-step reasoning and dynamic content processing capabilities. There is a lack of specialized datasets and benchmarks for e-commerce risk management. Furthermore, GUI models trained on coordinate positioning face a training-deployment gap when deployed using DOM indices and tool calling.

**Key Challenge**: General GUI agents perform poorly in e-commerce risk management because they lack domain knowledge, multi-step reasoning abilities, and experience in handling complex webpages.

**Goal**: Build a complete GUI agent framework for e-commerce risk management—covering data collection, model training, and practical deployment.

**Key Insight**: Utilize the Browser Use framework to collect high-quality domain data and implement GRPO reinforcement fine-tuning to achieve a seamless transition from training to deployment.

**Core Idea**: Bridge the gap between GUI agent training and real deployment through a four-dimensional reward design (format reward + stepwise accuracy reward + process reweighting + difficulty reweighting).

## Method

### Overall Architecture

RISK consists of three components: (1) RISK-Data—data collected via the Qwen-VL-Max-driven Browser Use framework and refined through trajectory filtering, step cleaning, information refinement, data augmentation, multi-step generation, and difficulty grading to obtain 8,492 single-step and 2,386 multi-step trajectories; (2) RISK-Bench—802 single-step and 320 multi-step trajectories categorized into easy, moderate, and difficult levels; (3) RISK-R1—a reinforcement fine-tuning framework based on GRPO, establishing base capabilities through SFT followed by RFT refinement.

### Key Designs

1.  **Framework-Driven Reward Functions**:
    - **Function**: Bridges the gap between GUI model training and framework deployment.
    - **Mechanism**: Four dimensions—(a) Format reward: checks if output contains `think`/`action`/`evaluation_previous_goal`/`memory`/`next_goal` structures and if action format uses DOM indices and tools instead of coordinates; (b) Stepwise accuracy reward: early training assigns rewards based on whether each action in the tool list reaches F1 > 0.5, while late training switches to a global binary reward; (c) Process reweighting: uses a sigmoid function to assign higher weights to later steps in a trajectory; (d) Difficulty reweighting: assigns higher weights to difficult samples in the optimization objective.
    - **Design Motivation**: Coordinate-based rewards in existing methods like GUI-R1 are unsuitable for actual deployment frameworks based on DOM indices. A single binary reward provides insufficient guidance for exploration during early training.

2.  **Domain Data Collection and Refinement Pipeline**:
    - **Function**: Constructs high-quality data specialized for e-commerce risk management.
    - **Mechanism**: Leverages Browser Use + Qwen-VL-Max for multi-turn interaction on real websites to collect raw data, followed by a six-step refinement process (trajectory filtering → step cleaning → information refinement → data augmentation → multi-step generation → difficulty grading) to ensure data quality.
    - **Design Motivation**: General GUI datasets lack information searching and website verification tasks specific to e-commerce risk management.

3.  **SFT → RFT Two-Stage Training**:
    - **Function**: Progressives from base capability establishment to fine-grained enhancement.
    - **Mechanism**: Phase one uses the entire RISK-Data for SFT to establish basic interaction capabilities; phase two uses only single-step trajectories for RFT (as multi-step trajectories are too long for GPU memory), transitioning stepwise accuracy rewards from fine-grained to coarse-grained.
    - **Design Motivation**: Direct RFT is unstable; SFT provides the necessary foundation for format and basic capabilities.

## Key Experimental Results

### Main Results

| Model | Single-step Overall | Multi-step Success Rate | OS-Genesis Web |
| :--- | :--- | :--- | :--- |
| GPT-4o | 81.5 | 74.0 | 55.3 |
| Qwen2.5-VL-72B | 80.6 | 67.8 | 50.0 |
| RISK-R1-7B (Ours) | **88.3** | **82.8** | **57.1** |
| GUI-R1-7B | 74.3 | 0.0 | 49.1 |
| UI-TARS-72B | 13.0 | 0.0 | 5.8 |

### Ablation Study

| Configuration | Single-step | Multi-step | Description |
| :--- | :--- | :--- | :--- |
| RISK-R1 Complete | 88.3 | 82.8 | All components |
| - Process Reweighting | 86.5 | 79.1 | Equal weights for late steps |
| - Stepwise Reward | 85.8 | 78.3 | Binary reward throughout |
| - Difficulty Reweighting | 87.1 | 80.5 | Equal sample weights |
| SFT Only | 83.2 | 74.7 | No RFT |

### Key Findings
- RISK-R1 (7B) outperforms 72B general models and GPT-4o, achieving SOTA with only 7.2% of the parameters.
- General GUI SFT models (UI-TARS) fail almost completely on domain tasks, highlighting the necessity of domain data.
- GUI-R1 (coordinate-based RFT) has a 0% success rate on multi-step tasks, confirming the severity of the training-deployment gap.
- An online evaluation success rate of 70.5% validates the practical deployment value of the method.
- Process reweighting and stepwise rewards yield the most significant improvements for multi-step tasks.

## Highlights & Insights
- **Complete Data-to-Deployment Framework**: RISK covers the entire pipeline from data collection and benchmark construction to model training and actual deployment.
- **Small Model Outperforms Large Model**: 7B exceeding 72B and GPT-4o proves the value of domain focus combined with correct training strategies.
- **Training-Deployment Consistency**: Reward functions based on DOM indices and tool calling instead of coordinates ensure seamless integration between training and deployment.
- **Online Evaluation**: Beyond offline benchmarks, real-world online evaluation strengthens the findings.

## Limitations & Future Work
- **RFT limited to single-step data**: Multi-step trajectories are too long for current GPUs; multi-step capability relies primarily on SFT and transfer from single-step RFT.
- **Domain limitation**: Only covers e-commerce risk management; applicability to other domains needs verification.
- **Framework dependency**: Tight coupling with the Browser Use framework.
- **Future directions**: Support for multi-step RFT training, expansion to more domains, and integration of more complex risk decision-making.

## Related Work & Insights
- **vs GUI-R1**: General GUI RFT using coordinate positioning fails completely in DOM index scenarios; RISK-R1's framework-driven reward addresses this.
- **vs UI-TARS**: General GUI SFT models perform poorly on domain tasks, emphasizing the importance of domain-specific data.
- **vs Browser Use**: RISK utilizes it as both a data collection tool and a deployment framework, achieving a closed loop between training and deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative framework-driven reward design and training-deployment consistency.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes offline and online evaluation, multiple baseline comparisons, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description, though some details require the appendix.
- Value: ⭐⭐⭐⭐ Provides a reproducible, end-to-end solution for domain-specific GUI agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)
- [\[ICML 2026\] Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model](../../ICML2026/llm_safety/anchored_decoding_provably_reducing_copyright_risk_for_any_language_model.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](../../ICML2026/llm_safety/from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[AAAI 2026\] An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling](../../AAAI2026/llm_safety/an_llm-based_simulation_framework_for_embodied_conversationa.md)
- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](../../ICML2026/llm_safety/less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)

</div>

<!-- RELATED:END -->
