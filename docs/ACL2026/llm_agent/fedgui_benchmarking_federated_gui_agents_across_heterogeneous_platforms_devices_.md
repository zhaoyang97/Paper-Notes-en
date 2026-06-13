---
title: >-
  [Paper Note] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems
description: >-
  [ACL 2026][LLM Agent][Federated Learning] FedGUI is the first comprehensive federated learning benchmark for cross-platform GUI agents, comprising six datasets covering mobile, web…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Federated Learning"
  - "GUI Agent"
  - "Cross-platform Heterogeneity"
  - "Privacy Protection"
  - "Distributed Training"
date: 2026-05-08
content_hash: 850ffe2afb8c1a9e
---

# FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.14956](https://arxiv.org/abs/2604.14956)  
**Code**: [GitHub](https://github.com/wwh0411/FedGUI)  
**Area**: Agent / GUI Interaction  
**Keywords**: Federated Learning, GUI Agent, Cross-platform Heterogeneity, Privacy Protection, Distributed Training

## TL;DR
FedGUI is the first comprehensive federated learning benchmark for cross-platform GUI agents, comprising six datasets covering mobile, web, and desktop platforms. It systematically investigates the impact of four dimensions of heterogeneity—platform, device, operating system, and data source—on the training of federated GUI agents.

## Background & Motivation

**Background**: GUI agents utilize Vision-Language Models (VLMs) to perceive graphical interfaces and execute user instructions. Traditional methods rely on centralized data collection and annotation, which is costly and difficult to scale. Federated learning (FL) offers a privacy-preserving distributed training paradigm.

**Limitations of Prior Work**: (1) Existing federated GUI benchmarks (e.g., FedMABench) are restricted to collaboration among Android users, neglecting the potential contributions of web and desktop users; (2) In real-world scenarios, GUI data is distributed across various platforms (mobile/web/desktop), devices (different smartphone models), and operating systems (Android/macOS/Windows/Ubuntu). The impact of these heterogeneities on federated training has not been explored.

**Key Challenge**: GUI devices naturally generate rich supervisory signals but cannot be shared due to privacy concerns. While FL can address this, there is no benchmark that captures real-world cross-platform heterogeneity to guide algorithm selection.

**Goal**: Construct a federated GUI agent benchmark covering multiple platforms, devices, and OSs to answer two critical questions: Does cross-platform collaboration improve performance? How can different dimensions of heterogeneity be quantified and mitigated?

**Key Insight**: Build six datasets from nine data sources to correspond to the four dimensions of heterogeneity, followed by systematic evaluation using seven federated learning algorithms and over 20 base models.

**Core Idea**: Four-dimensional heterogeneity modeling (Platform $\times$ Device $\times$ OS $\times$ Source) + Unified Action Space + Systematic Federated Learning Evaluation.

## Method

### Overall Architecture
FedGUI follows the standard federated learning protocol: a central server coordinates heterogeneous clients. Each client trains locally on its own GUI interaction data, and the global model is updated by aggregating these local updates. A unified action space (consisting of six basic actions like CLICK and TYPE) is provided to achieve consistent policy learning across platforms.

### Key Designs

1.  **Four-dimensional Heterogeneity Dataset Construction**:
    - **Function**: Systematically isolate and study the impact of different sources of heterogeneity.
    - **Mechanism**: Construction of six datasets—FedGUI-Platform (Mobile/Web/Desktop, 15 clients), FedGUI-Device (5 Android devices), FedGUI-OS (Ubuntu/macOS/Windows), FedGUI-Web (various web sources), FedGUI-Mobile (various mobile sources), and FedGUI-Full (combined cross-platform and cross-source).
    - **Design Motivation**: Different sources of heterogeneity affect federated training differently; platform-level heterogeneity may be more challenging than device-level heterogeneity, requiring independent study.

2.  **Unified Action Space Design**:
    - **Function**: Enable GUI interaction data from different platforms to be trained and aggregated within a single model.
    - **Mechanism**: Identification of six basic actions shared across platforms (CLICK, TYPE, etc.) and mapping platform-specific actions into this unified domain. This ensures consistency in federated aggregation at the action level even when GUI appearances differ drastically.
    - **Design Motivation**: Without a unified action space, model parameters from different platforms cannot be meaningfully aggregated.

3.  **Systematic Federated Algorithm Evaluation**:
    - **Function**: Provide an empirical guide for selecting federated GUI agent algorithms.
    - **Mechanism**: Integration of seven representative FL algorithms (FedAvg, FedProx, FedYogi, etc.) for comprehensive comparison across all datasets and heterogeneity settings. Evaluation metrics include action type accuracy, grounding precision, and success rate.
    - **Design Motivation**: The optimal algorithm varies under different heterogeneities; benchmark data is essential to guide algorithm selection for practical deployment.

### Loss & Training
Standard federated learning setup: local training employs cross-entropy loss, and global aggregation uses the specific strategies of the chosen FL algorithms. Support for LoRA fine-tuning is included to reduce communication and computational costs.

## Key Experimental Results

### Main Results

| Finding | Description |
|------|------|
| Cross-platform collaboration is beneficial | Increasing the number of participating users (even from different platforms) enhances model performance. |
| Platform-level heterogeneity has the greatest impact | Cross-platform heterogeneity is more challenging than intra-platform heterogeneity (device/OS/source). |
| Adaptive algorithms are optimal | Adaptive algorithms such as FedYogi demonstrate the most robust performance in cross-platform settings. |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Mobile-only vs. Full-platform FL | Full-platform is superior | Cross-platform data diversity contributes positively. |
| IID vs. Non-IID Device Distribution | Performance drops under Non-IID | Device heterogeneity leads to data skew. |
| Different Base Models | Larger models yield higher gains | The scale of the VLM significantly impacts FL effectiveness. |

### Key Findings
- Even with highly heterogeneous platforms and devices, increasing the number of federated participants still improves global model performance. This provides confidence for large-scale distributed training of GUI agents.
- Platform-level heterogeneity represents the most significant performance challenge, followed by the operating system, while the impacts of device and data source are relatively minor.
- Adaptive learning rate algorithms like FedYogi are particularly effective in cross-platform scenarios, likely because adaptive aggregation handles the variance in gradient distributions across different platforms more effectively.

## Highlights & Insights
- **Four-dimensional heterogeneity decomposition** is a systematic experimental design that allows the impact of each type of heterogeneity to be analyzed independently.
- The finding that **cross-platform collaboration is beneficial** has significant value for practical deployment, suggesting that user data from various device types can be leveraged to train a superior unified GUI agent.
- The **Unified Action Space** is a critical engineering contribution that enables cross-platform federated learning.

## Limitations & Future Work
- Only LoRA fine-tuning was evaluated; full-parameter federated learning might exhibit different heterogeneity dynamics.
- Data privacy is provided only through the basic FL framework; additional measures like differential privacy were not introduced.
- Evaluation is primarily based on offline data and lacks online evaluation involving real-time user interactions.
- A unified action space may lose fine-grained, platform-specific interactions.

## Related Work & Insights
- **vs FedMABench**: FedMABench is restricted to mobile Android, whereas FedGUI extends to mobile, web, and desktop platforms.
- **vs Centralized Cross-platform Agents (ShowUI, UI-TARS)**: These rely on centralized data collection; FedGUI demonstrates a distributed alternative.
- **vs Single-platform GUI Benchmarks**: Single-platform methods lack generalization; federated cross-platform training provides a more scalable path.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First cross-platform federated GUI benchmark with a systematic four-dimensional heterogeneity analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Utilizes six datasets, seven algorithms, and over 20 base models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear description of dataset construction and systematic experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](../../ICML2026/llm_agent/recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)
- [\[ICML 2026\] Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](../../ICML2026/llm_agent/scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](../../AAAI2026/llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)

</div>

<!-- RELATED:END -->
