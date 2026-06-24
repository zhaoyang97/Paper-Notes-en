---
title: >-
  [Paper Note] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems
description: >-
  [ACL 2026 Findings][LLM Agent][Federated Learning] FedGUI is the first comprehensive federated learning benchmark for cross-platform GUI agents, containing six datasets covering mobile, web, and desktop platforms. It systematically investigates the impact of four dimensions of heterogeneity—platform, device, operating system, and data source—on the training of federated GUI agents.
tags:
  - "ACL 2026 Findings"
  - "LLM Agent"
  - "Federated Learning"
  - "GUI Agents"
  - "Cross-platform Heterogeneity"
  - "Privacy Protection"
  - "Distributed Training"
date: 2026-05-08
content_hash: e22dc3f0f74748d5
---

# FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.14956](https://arxiv.org/abs/2604.14956)  
**Code**: [GitHub](https://github.com/wwh0411/FedGUI)  
**Area**: Agent / GUI Interaction  
**Keywords**: Federated Learning, GUI Agents, Cross-platform Heterogeneity, Privacy Protection, Distributed Training

## TL;DR
FedGUI is the first comprehensive federated learning benchmark for cross-platform GUI agents, containing six datasets covering mobile, web, and desktop platforms. It systematically investigates the impact of four dimensions of heterogeneity—platform, device, operating system, and data source—on the training of federated GUI agents.

## Background & Motivation

**Background**: GUI agents perceive graphical interfaces and execute user instructions through Vision-Language Models (VLMs). Traditional methods rely on centralized data collection and annotation, which are high-cost and non-scalable. Federated learning (FL) provides a privacy-preserving distributed training paradigm.

**Limitations of Prior Work**: (1) Existing federated GUI benchmarks (e.g., FedMABench) are limited to collaboration among Android users, ignoring the contribution potential of web and desktop users; (2) In the real world, GUI data is distributed across different platforms (mobile/web/desktop), devices (different phone models), and operating systems (Android/macOS/Windows/Ubuntu), yet the impact of these heterogeneities on federated training remains unstudied.

**Key Challenge**: GUI devices naturally generate rich supervisory signals, but these cannot be shared due to privacy concerns. Federated learning can address this, but there is a lack of benchmarks that capture real-world cross-platform heterogeneity to guide algorithm selection.

**Goal**: To build a federated GUI agent benchmark covering multiple platforms, devices, and OSs to answer two key questions: Can cross-platform collaboration improve performance? How can different dimensions of heterogeneity be quantified and addressed?

**Key Insight**: Construct six datasets from nine data sources corresponding to four dimensions of heterogeneity, and conduct a systematic evaluation combining seven federated learning algorithms and 20+ foundation models.

**Core Idea**: Four-dimensional heterogeneity modeling (Platform × Device × OS × Source) + Unified action space + Systematic federated learning evaluation.

## Method

### Overall Architecture
Ours establishes a quantifiable benchmark for whether cross-platform GUI agents can be trained collaboratively via federated learning. It follows standard federated learning protocols where a central server coordinates a set of heterogeneous clients. Each client trains only on local GUI interaction data and then aggregates updates into a global VLM. To aggregate mobile, web, and desktop interfaces with distinct appearances into a single model, FedGUI provides a unified action space and constructs six datasets from eight data sources, isolating platform, device, OS, and source heterogeneities for systematic evaluation.

### Key Designs

**1. Four-Dimensional Heterogeneity Dataset Construction: Isolating "Which Heterogeneity is Most Difficult"**

Real-world GUI data is simultaneously confounded by differences in platform, device, OS, and source. To disentangle these effects, FedGUI derives six isolated datasets from eight data sources: FedGUI-Platform (15 clients), FedGUI-Device (5 Android devices), FedGUI-OS, FedGUI-Web, FedGUI-Mobile, and FedGUI-Full (combining cross-platform and cross-source). Each dataset amplifies a single dimension of heterogeneity, allowing the impacts of platform-level vs. device-level heterogeneity to be measured independently, leading to the conclusion that "Platform > OS > Device/Source" in terms of difficulty.

**2. Unified Action Space: Aligning Diverse Interfaces at the Action Level**

GUI layouts across platforms are often unrelated, making direct parameter aggregation difficult. Ours resolves this by finding common denominators at the action level: identifying six basic cross-platform shared actions (e.g., CLICK, TYPE) and mapping platform-specific interactions to two independent action domains defined in system prompts. This ensures that even if mobile and desktop interfaces share no commonality at the pixel level, federated aggregation maintains consistency in "action types," preventing the global model from becoming mere noise.

**3. Systematic FL Algorithm Evaluation: An Empirical Selection Guide for Cross-Platform Deployment**

To assist practitioners, FedGUI conducts a horizontal comparison of seven representative algorithms—including FedAvg, FedProx, and FedYogi—across all six datasets and heterogeneous settings. Evaluation metrics cover action type accuracy, grounding precision, and success rate. The value of this comparison lies in revealing how the "optimal algorithm varies with heterogeneity dimensions," showing that adaptive learning rate algorithms like FedYogi are the most robust in cross-platform scenarios, likely because adaptive aggregation better handles gradient distribution discrepancies between platforms.

### Loss & Training
Standard federated learning setup: Local training utilizes cross-entropy loss, while the global side merges updates according to the specific aggregation strategy of each FL algorithm. LoRA fine-tuning is supported to reduce communication and computation costs.

## Key Experimental Results

### Main Results

| Finding | Description |
|------|------|
| Cross-platform collaboration is beneficial | Increasing participating users (even from different platforms) improves model performance. |
| Platform-level heterogeneity has the greatest impact | Cross-platform heterogeneity is more challenging than intra-platform heterogeneity (Device/OS/Source). |
| Adaptive algorithms are optimal | Adaptive algorithms like FedYogi exhibit the most robust performance in cross-platform settings. |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Mobile-only vs. Full-platform FL | Full-platform is superior | Cross-platform data diversity makes a positive contribution. |
| IID vs. Non-IID Device Distribution | Non-IID performance drops | Device heterogeneity leads to data skew. |
| Different Foundation Models | Larger models gain more | VLM scale influences the effectiveness of federated learning. |

### Key Findings
- Even with data from highly heterogeneous platforms and devices, increasing federated participants still enhances global model performance, providing confidence for large-scale distributed GUI agent training.
- Platform-level heterogeneity is the primary performance challenge, followed by the operating system; the impact of devices and data sources is relatively small.
- Adaptive learning rate algorithms such as FedYogi are particularly effective in cross-platform scenarios, likely due to their ability to better handle gradient distribution variances across platforms.

## Highlights & Insights
- **Four-dimensional heterogeneity decomposition** is a systematic experimental design that allows the impact of each heterogeneity type to be analyzed independently.
- The finding that **cross-platform collaboration is beneficial** has practical deployment value, suggesting that user data from various device types can be leveraged to train superior unified GUI agents.
- The **unified action space** is a key engineering contribution that enables cross-platform federated learning.

## Limitations & Future Work
- Only LoRA fine-tuning was evaluated; full-parameter federated learning might exhibit different heterogeneity dynamics.
- Data privacy is provided only through the basic FL framework; additional protections like Differential Privacy (DP) were not introduced.
- Evaluation is primarily based on offline data, lacking online evaluation in real user interactions.
- The unified action space might lose fine-grained, platform-specific interactions.

## Related Work & Insights
- **vs. FedMABench**: Restricted to mobile Android; FedGUI expands to mobile, web, and desktop platforms.
- **vs. Centralized Cross-platform Agents (ShowUI, UI-TARS)**: These rely on centralized data collection; FedGUI demonstrates a distributed alternative.
- **vs. Single-platform GUI Benchmarks**: Single-platform methods offer poor generalization; federated cross-platform training provides a more scalable path.

## Rating
- Novelty: ⭐⭐⭐⭐ First cross-platform federated GUI benchmark with systematic 4D heterogeneity analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets, seven algorithms, and 20+ foundation models.
- Writing Quality: ⭐⭐⭐⭐ Clear description of dataset construction and systematic experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Helmsman: Autonomous Synthesis of Federated Learning Systems via Collaborative LLM Agents](../../ICLR2026/llm_agent/helmsman_autonomous_synthesis_of_federated_learning_systems_via_collaborative_ll.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](../../ICML2026/llm_agent/recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)
- [\[ICML 2026\] Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](../../ICML2026/llm_agent/scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)

</div>

<!-- RELATED:END -->
