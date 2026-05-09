---
title: >-
  [Paper Note] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems
description: >-
  [ACL 2026][LLM Agent][Federated Learning] FedGUI is the first comprehensive federated learning benchmark for cross-platform GUI agents, comprising six datasets covering mobile, web, and desktop environments. It systematically investigates the effects of four types of heterogeneity—cross-platform, cross-device, cross-OS, and cross-source—on federated GUI agent training.
tags:
  - ACL 2026
  - LLM Agent
  - Federated Learning
  - GUI Agents
  - Cross-Platform Heterogeneity
  - Privacy Preservation
  - Distributed Training
date: 2026-05-08
content_hash: 5f40a087e1bfad34
---

# FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems

**Conference**: ACL 2026
**arXiv**: [2604.14956](https://arxiv.org/abs/2604.14956)
**Code**: [GitHub](https://github.com/wwh0411/FedGUI)
**Area**: Agent / GUI Interaction
**Keywords**: Federated Learning, GUI Agents, Cross-Platform Heterogeneity, Privacy Preservation, Distributed Training

## TL;DR
FedGUI is the first comprehensive federated learning benchmark for cross-platform GUI agents, comprising six datasets covering mobile, web, and desktop environments. It systematically investigates the effects of four types of heterogeneity—cross-platform, cross-device, cross-OS, and cross-source—on federated GUI agent training.

## Background & Motivation

**Background**: GUI agents leverage vision-language models (VLMs) to perceive graphical interfaces and execute user instructions. Conventional approaches rely on centralized data collection and annotation, which are costly and difficult to scale. Federated learning offers a privacy-preserving paradigm for distributed training.

**Limitations of Prior Work**: (1) Existing federated GUI benchmarks (e.g., FedMABench) are limited to collaboration among Android users, overlooking the potential contributions of web and desktop users. (2) In practice, GUI data are distributed across diverse platforms (mobile/web/desktop), devices (different phone models), and operating systems (Android/macOS/Windows/Ubuntu), yet the impact of such heterogeneity on federated training remains unexplored.

**Key Challenge**: GUI devices naturally generate rich supervision signals, but privacy concerns prevent data sharing. While federated learning addresses this issue, the field lacks benchmarks that capture real-world cross-platform heterogeneity to guide algorithm selection.

**Goal**: To construct a federated GUI agent benchmark covering multiple platforms, devices, and operating systems, and to answer two key questions: Does cross-platform collaboration improve performance? How can heterogeneity along different dimensions be quantified and addressed?

**Key Insight**: Six datasets are constructed from nine data sources, each corresponding to one of four heterogeneity dimensions. A systematic evaluation is conducted across seven federated learning algorithms and over 20 base models.

**Core Idea**: Four-dimensional heterogeneity modeling (Platform × Device × OS × Source) combined with a unified action space and systematic federated learning evaluation.

## Method

### Overall Architecture
FedGUI follows the standard federated learning protocol: a central server coordinates heterogeneous clients, each of which trains locally on GUI interaction data, and updates are aggregated to form a global model. A unified action space (six fundamental actions including CLICK and TYPE) is provided to enable consistent policy learning across platforms.

### Key Designs

1. **Four-Dimensional Heterogeneous Dataset Construction**:

    - Function: Systematically isolate and study the effects of different sources of heterogeneity.
    - Mechanism: Six datasets are constructed—FedGUI-Platform (mobile/web/desktop, 15 clients), FedGUI-Device (5 Android device types), FedGUI-OS (Ubuntu/macOS/Windows), FedGUI-Web (different web data sources), FedGUI-Mobile (different mobile data sources), and FedGUI-Full (combined cross-platform and cross-source).
    - Design Motivation: Different sources of heterogeneity have distinct effects on federated training—platform-level heterogeneity may be more challenging than device-level heterogeneity, necessitating separate investigation.

2. **Unified Action Space Design**:

    - Function: Enable GUI interaction data from different platforms to be trained and aggregated within a single model.
    - Mechanism: Six cross-platform fundamental actions (CLICK, TYPE, etc.) are identified, and platform-specific actions are mapped into a unified domain. This ensures consistent federated aggregation at the action level, even when GUI appearances differ substantially across platforms.
    - Design Motivation: Without a unified action space, model parameters from different platforms cannot be meaningfully aggregated.

3. **Systematic Federated Algorithm Evaluation**:

    - Function: Provide empirical guidance for federated GUI agent algorithm selection.
    - Mechanism: Seven representative federated learning algorithms (FedAvg, FedProx, FedYogi, etc.) are integrated and comprehensively compared across all datasets and heterogeneity settings. Evaluation metrics include action type accuracy, grounding precision, and success rate.
    - Design Motivation: The optimal algorithm varies with the type of heterogeneity; benchmark data are necessary to guide algorithm selection in practical deployments.

### Loss & Training
Standard federated learning setup: local training uses cross-entropy loss, and global aggregation follows the respective aggregation strategy of each federated algorithm. LoRA fine-tuning is supported to reduce communication and computational costs.

## Key Experimental Results

### Main Results

| Finding | Description |
|---------|-------------|
| Cross-platform collaboration is beneficial | Adding more participants—even from different platforms—improves model performance |
| Platform-level heterogeneity has the greatest impact | Cross-platform heterogeneity is more challenging than intra-platform heterogeneity (device/OS/source) |
| Adaptive algorithms perform best | Adaptive algorithms such as FedYogi are most robust in cross-platform settings |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Mobile-only vs. full-platform federation | Full-platform is superior | Cross-platform data diversity contributes positively |
| IID vs. Non-IID device distribution | Performance drops under Non-IID | Device heterogeneity induces data skew |
| Different base models | Larger models benefit more | VLM scale affects federated learning efficacy |

### Key Findings
- Adding federated participants—even from highly heterogeneous platforms and devices—consistently improves global model performance, providing confidence for large-scale distributed GUI agent training.
- Platform-level heterogeneity poses the greatest performance challenge, followed by operating system heterogeneity; the effects of device and data source heterogeneity are comparatively smaller.
- Adaptive learning rate algorithms such as FedYogi are particularly effective in cross-platform scenarios, likely because adaptive aggregation better accommodates differences in gradient distributions across platforms.

## Highlights & Insights
- The **four-dimensional heterogeneity decomposition** constitutes a systematic experimental design, enabling independent analysis of each heterogeneity source.
- The finding that cross-platform collaboration is beneficial has practical deployment value, suggesting that data from diverse device types can be leveraged to train superior unified GUI agents.
- The unified action space is a key engineering contribution that makes cross-platform federated learning feasible.

## Limitations & Future Work
- Only LoRA fine-tuning is evaluated; full-parameter federated learning may exhibit different heterogeneity dynamics.
- Privacy protection relies solely on the basic federated learning framework, without additional mechanisms such as differential privacy.
- Evaluation is primarily conducted on offline data, lacking online assessment under real user interactions.
- The unified action space may sacrifice platform-specific fine-grained interactions.

## Related Work & Insights
- **vs. FedMABench**: Restricted to mobile Android; FedGUI extends coverage to mobile, web, and desktop platforms.
- **vs. Centralized Cross-Platform Agents (ShowUI, UI-TARS)**: These rely on centralized data collection; FedGUI demonstrates a viable distributed alternative.
- **vs. Single-Platform GUI Benchmarks**: Single-platform approaches generalize poorly; federated cross-platform training represents a more scalable path forward.

## Rating
- Novelty: ⭐⭐⭐⭐ First cross-platform federated GUI benchmark with systematic four-dimensional heterogeneity analysis
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets, seven algorithms, 20+ base models
- Writing Quality: ⭐⭐⭐⭐ Dataset construction is clearly described; experimental design is systematic

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](../../NeurIPS2025/llm_agent/benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)

<!-- RELATED:END -->
