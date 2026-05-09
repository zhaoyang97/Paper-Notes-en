---
title: >-
  [Paper Note] RISK: A Framework for GUI Agents in E-commerce Risk Management
description: >-
  [ACL 2026][LLM Agent][GUI agents] This paper proposes the RISK framework, comprising a domain dataset (RISK-Data: 8,492 single-step + 2,386 multi-step trajectories), a benchmark (RISK-Bench), and a GRPO-based reinforcement fine-tuning method (RISK-R1) for GUI agents in e-commerce risk management. The 7B model surpasses state-of-the-art baselines with only 7.2% of their parameter count, achieving an online task success rate of 70.5%.
tags:
  - ACL 2026
  - LLM Agent
  - GUI agents
  - e-commerce risk management
  - reinforcement fine-tuning
  - web interaction
  - multi-step reasoning
date: 2026-05-08
content_hash: 7af50fcfc70412f1
---

# RISK: A Framework for GUI Agents in E-commerce Risk Management

**Conference**: ACL 2026
**arXiv**: [2509.21982](https://arxiv.org/abs/2509.21982)
**Code**: [GitHub](https://github.com/RenqiChen/RISK-GUI)
**Area**: GUI Agents
**Keywords**: GUI agents, e-commerce risk management, reinforcement fine-tuning, web interaction, multi-step reasoning

## TL;DR

This paper proposes the RISK framework, comprising a domain dataset (RISK-Data: 8,492 single-step + 2,386 multi-step trajectories), a benchmark (RISK-Bench), and a GRPO-based reinforcement fine-tuning method (RISK-R1) for GUI agents in e-commerce risk management. The 7B model surpasses state-of-the-art baselines with only 7.2% of their parameter count, achieving an online task success rate of 70.5%.

## Background & Motivation

**State of the Field**: E-commerce risk management requires aggregating heterogeneous information from multiple external websites (transaction details, user profiles, site verification, etc.), which is often embedded in dynamically loaded sub-pages, interactive elements, or complex DOM structures, necessitating multi-step stateful web interaction.

**Limitations of Prior Work**: Traditional web crawlers cannot handle stateful, event-driven interactions; existing GUI agents are largely limited to single-step operations and lack multi-step reasoning and dynamic content processing capabilities; dedicated datasets and benchmarks for e-commerce risk management are absent; furthermore, GUI models trained with coordinate-based localization face a train-deploy gap when deployed in frameworks using DOM indices and tool calls.

**Root Cause**: General-purpose GUI agents perform poorly in e-commerce risk management scenarios due to lack of domain knowledge, multi-step reasoning ability, and experience handling complex web pages.

**Paper Goals**: To construct a complete GUI agent framework for e-commerce risk management, spanning data collection, model training, and real-world deployment.

**Starting Point**: High-quality domain data is collected using the Browser Use framework, and GRPO-based reinforcement fine-tuning is applied to achieve a seamless transition from training to deployment.

**Core Idea**: A four-dimensional reward design (format reward + step-wise accuracy reward + process re-weighting + difficulty re-weighting) bridges the gap between GUI agent training and real-world deployment.

## Method

### Overall Architecture

RISK consists of three components: (1) **RISK-Data** — data is collected by driving the Browser Use framework with Qwen-VL-Max, then refined through a six-step pipeline (trajectory filtering → step cleaning → information refinement → data augmentation → multi-step generation → difficulty grading), yielding 8,492 single-step and 2,386 multi-step trajectories; (2) **RISK-Bench** — 802 single-step and 320 multi-step trajectories across three difficulty levels: easy/moderate/difficult; (3) **RISK-R1** — a GRPO-based reinforcement fine-tuning framework that first applies SFT to establish foundational capabilities, followed by RFT for further refinement.

### Key Designs

1. **Framework-Driven Reward Function**:

    - **Function**: Bridges the gap between GUI model training and framework-based deployment.
    - **Mechanism**: Four dimensions — (a) *Format reward*: verifies whether the output contains structural elements such as think/action/evaluation_previous_goal/memory/next_goal, and whether the action format uses DOM indices and tool calls rather than coordinates; (b) *Step-wise accuracy reward*: in early training, rewards are assigned per action based on F1 > 0.5 against the tool list; in later training, this transitions to a holistic binary reward; (c) *Process re-weighting*: a sigmoid function assigns higher weights to later steps within a trajectory; (d) *Difficulty re-weighting*: samples at the difficult level receive higher weights in the optimization objective.
    - **Design Motivation**: Coordinate-based rewards used in existing methods such as GUI-R1 are incompatible with DOM-index-based deployment frameworks; a single binary reward provides insufficient guidance for model exploration during early training.

2. **Domain Data Collection and Refinement Pipeline**:

    - **Function**: Constructs high-quality domain-specific data for e-commerce risk management.
    - **Mechanism**: Raw data is collected through multi-turn real webpage interactions using the Browser Use framework and Qwen-VL-Max, then refined via a six-step pipeline (trajectory filtering → step cleaning → information refinement → data augmentation → multi-step generation → difficulty grading) to ensure data quality.
    - **Design Motivation**: General-purpose GUI datasets lack the information retrieval and website verification tasks characteristic of e-commerce risk management.

3. **Two-Stage SFT→RFT Training**:

    - **Function**: Progresses from establishing foundational capabilities to fine-grained improvement.
    - **Mechanism**: Stage one applies SFT on the full RISK-Data to establish basic interaction capabilities; stage two applies RFT using only single-step trajectories (multi-step trajectories are too long to fit on GPU), with the step-wise accuracy reward transitioning from fine-grained to coarse-grained.
    - **Design Motivation**: Direct RFT is unstable; SFT first establishes the foundation for output format and basic capabilities.

## Key Experimental Results

### Main Results

| Model | Single-Step Overall | Multi-Step Success Rate | OS-Genesis Web |
|-------|-------------------|------------------------|----------------|
| GPT-4o | 81.5 | 74.0 | 55.3 |
| Qwen2.5-VL-72B | 80.6 | 67.8 | 50.0 |
| RISK-R1-7B (Ours) | **88.3** | **82.8** | **57.1** |
| GUI-R1-7B | 74.3 | 0.0 | 49.1 |
| UI-TARS-72B | 13.0 | 0.0 | 5.8 |

### Ablation Study

| Configuration | Single-Step | Multi-Step | Notes |
|---------------|------------|-----------|-------|
| RISK-R1 (Full) | 88.3 | 82.8 | All components |
| − Process Re-weighting | 86.5 | 79.1 | Uniform weights for later steps |
| − Step-wise Reward | 85.8 | 78.3 | Binary reward throughout |
| − Difficulty Re-weighting | 87.1 | 80.5 | Uniform sample weights |
| SFT Only | 83.2 | 74.7 | No RFT |

### Key Findings
- RISK-R1 at 7B surpasses general-purpose 72B models and GPT-4o, achieving state-of-the-art performance with only 7.2% of the parameter count.
- General-purpose GUI SFT models (UI-TARS) nearly completely fail on domain-specific tasks, demonstrating the necessity of domain data.
- GUI-R1 (coordinate-based RFT) achieves 0% success rate on multi-step tasks, confirming the severity of the train-deploy gap.
- An online evaluation task success rate of 70.5% validates the practical deployment value of the proposed approach.
- Process re-weighting and step-wise rewards yield the most significant gains on multi-step tasks.

## Highlights & Insights
- **End-to-end framework from data to deployment**: RISK covers the complete pipeline of data collection, benchmark construction, model training, and real-world deployment.
- **Small model outperforms large models**: The 7B model surpasses 72B models and GPT-4o, demonstrating the value of domain focus combined with the correct training strategy.
- **Training-deployment consistency**: The reward function is grounded in DOM indices and tool calls rather than coordinates, ensuring seamless alignment between model training and framework deployment.
- **Online evaluation included**: In addition to offline benchmarks, real-environment online evaluation is conducted, enhancing the credibility of the results.

## Limitations & Future Work
- **RFT uses only single-step data**: Multi-step trajectories are too long to fit on GPU; multi-step capabilities are primarily transferred via SFT and single-step RFT.
- **Domain scope**: Coverage is limited to e-commerce risk management; the generalizability of the framework to other domains remains to be validated.
- **Framework dependency**: The approach is tightly coupled to the Browser Use framework.
- Future directions include supporting multi-step RFT training, extending to broader domains, and integrating more complex risk management decision-making.

## Related Work & Insights
- **vs. GUI-R1**: A general-purpose GUI RFT method using coordinate-based localization that completely fails in DOM-index-based scenarios; RISK-R1's framework-driven rewards resolve this issue.
- **vs. UI-TARS**: A general-purpose GUI SFT model that performs extremely poorly on domain-specific tasks, highlighting the importance of domain data.
- **vs. Browser Use**: RISK leverages it as both a data collection tool and a deployment framework, closing the loop between training and deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework-driven reward design and training-deployment consistency are innovative contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Combines offline and online evaluation, multiple baseline comparisons, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, though some details require reference to the appendix.
- Value: ⭐⭐⭐⭐ Provides a reproducible end-to-end solution for domain-specific GUI agents.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems](../../AAAI2026/llm_agent/with_great_capabilities_come_great_responsibilities_introducing_the_agentic_risk.md)
- [\[NeurIPS 2025\] Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction](../../NeurIPS2025/llm_agent/traj-coa_patient_trajectory_modeling_via_chain-of-agents_for_lung_cancer_risk_pr.md)
- [\[ACL 2026\] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)
- [\[ACL 2026\] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)

<!-- RELATED:END -->
