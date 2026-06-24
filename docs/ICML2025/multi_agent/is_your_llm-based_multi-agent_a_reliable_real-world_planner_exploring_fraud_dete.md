---
title: >-
  [Paper Note] Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning
description: >-
  [ICML 2025 (Workshop MAS)][Multi-Agent][multi-agent planning] This paper proposes WandaPlan, an evaluation environment that systematically assesses the vulnerability of LLM-based multi-agent planning systems to false information by injecting three progressive types of fraud (single-source misinformation, team-coordinated manipulation, and level-escalating attacks) in travel planning scenarios, and designs an Anti-Fraud Agent to mitigate these risks.
tags:
  - "ICML 2025 (Workshop MAS)"
  - "Multi-Agent"
  - "multi-agent planning"
  - "fraud detection"
  - "travel planning"
  - "LLM reliability"
  - "evaluation benchmark"
date: 2026-05-08
content_hash: 02afadee37c47111
---

# Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning

**Conference**: ICML 2025 (Workshop MAS)  
**arXiv**: [2505.16557](https://arxiv.org/abs/2505.16557)  
**Code**: None  
**Area**: Social Computing / LLM Agent  
**Keywords**: multi-agent planning, fraud detection, travel planning, LLM reliability, evaluation benchmark

## TL;DR

This paper proposes WandaPlan, an evaluation environment that systematically assesses the vulnerability of LLM-based multi-agent planning systems to false information by injecting three progressive types of fraud (single-source misinformation, team-coordinated manipulation, and level-escalating attacks) in travel planning scenarios, and designs an Anti-Fraud Agent to mitigate these risks.

## Background & Motivation

**Background**: LLM-based multi-agent planning systems (such as AutoGPT and CrewAI frameworks) are developing rapidly, capable of autonomously collaborating to complete complex real-world tasks. Travel planning is a representative application scenario requiring the retrieval of information from multiple data sources like restaurants, hotels, and transportation to make decisions.

**Limitations of Prior Work**: These systems rely heavily on external data sources (review sites, map services, social media, etc.) when executing planning tasks. However, these sources are often rife with fraudulent content, such as fake reviews, manipulated high ratings, and misleading descriptions. Existing evaluation benchmarks (like TravelPlanner) only focus on task completion efficiency and feasibility, completely ignoring the dimension of data authenticity.

**Key Challenge**: More "efficient" planning frameworks are actually more vulnerable to fraudulent information. Because they tend to trust high-rated data sources quickly without cross-verification, optimizing framework efficiency inadvertently introduces security vulnerabilities: High efficiency = Prompt trust in data = High susceptibility to fraud.

**Goal**: To build a systematic evaluation environment that assesses the security and robustness of multi-agent planning systems when facing real-world fraudulent information, and to propose a preliminary defense mechanism.

**Key Insight**: Instead of purely creating attacks, this work constructs a comprehensive evaluation framework—incorporating real-world data simulation, fraud injection, multi-dimensional evaluation metrics, and defensive strategies.

**Core Idea**: Existing LLM multi-agents prioritize efficiency over security; thus, specialized anti-fraud evaluation and defense mechanisms are highly required.

## Method

### Overall Architecture

WandaPlan is a complete evaluation ecosystem consisting of four core components:
1. A real-world data simulation environment (mirroring real review, price, and description data)
2. Three progressive fraud injection schemes
3. A multi-dimensional evaluation metric system
4. A plug-and-play Anti-Fraud Agent module

### Key Designs

1. **Three Progressive Fraud Scenarios**:
    - **Function**: Simulates the evolution of fraud in the real world from simple to complex.
    - **Mechanism**:
        - **Misinformation Fraud (Single-Source Misdirection)**: Injects false high ratings or fake descriptions into a single data source. For example, fabricating a 5-star review and fake dish descriptions for a low-quality restaurant. This is the most basic form of attack.
        - **Team-Coordinated Multi-Person Fraud (Coordinated Manipulation)**: Simulates astroturfing teams where multiple "users" from different accounts post consistent fake positive reviews across different time periods, creating a collective misdirection. Each individual review seems genuine, but joint analysis reveals malicious patterns.
        - **Level-Escalating Multi-Round Fraud (Adaptive Escalation)**: Fraud strategies adapt dynamically based on feedback from the system. If the system rejects a recommendation, the fraudster escalates the strategy (e.g., adding discount information or fabricating limited-time offers), simulating an adaptive adversarial scenario.
    - **Design Motivation**: These three modes correspond to different stages and complexities of real-world fraud. This progressive design systematically tests the upper bound of the system's anti-fraud capabilities.

2. **Anti-Fraud Agent**:
    - **Function**: Serves as a plug-and-play security module embedded into existing planning frameworks.
    - **Mechanism**: The agent executes a three-layer defense:
        - **Information Cross-Verification**: Retrieves information regarding the same entity from multiple independent data sources and compares their consistency.
        - **Review Anomaly Detection**: Analyzes the temporal distribution, wording patterns, and rating distributions of reviews to identify patterns of manipulated reviews.
        - **Suspicious Data Tagging**: Labels low-confidence data sources with warning tags, prompting the planning system to de-prioritize or downweight them during decision-making.
    - **Design Motivation**: Leverages the reasoning capabilities of LLMs to perform context-aware fraud detection.

3. **Multi-Dimensional Evaluation Metrics**:
    - **Function**: Comprehensively evaluates the performance of planning systems under fraud scenarios.
    - **Mechanism**: Beyond evaluating traditional planning quality (route feasibility, budget adherence, schedule viability), it specifically introduces **security metrics**—fraud adoption rate, potential user loss, and fraud detection rate.
    - **Design Motivation**: Traditional success metrics in existing benchmarks fail to capture vulnerabilities in the security dimension.

### Loss & Training

This work is an evaluation-oriented study and does not involve model training. The core contributions lie in the benchmark design and evaluation methodology. The Anti-Fraud Agent is implemented based on prompt engineering.

## Key Experimental Results

### Main Results: Performance Degradation of Different Frameworks Under Fraud Scenarios

| Planning Framework | Score (No Fraud) | Misinformation ↓ | Multi-Person ↓ | Multi-Round ↓ |
|----------|-----------|------------------|----------------|---------------|
| CrewAI | 82.3 | 71.5 (-13.1%) | 63.2 (-23.2%) | 55.8 (-32.2%) |
| AutoGPT | 78.6 | 68.9 (-12.3%) | 60.1 (-23.5%) | 51.3 (-34.7%) |
| LangChain Agent | 75.1 | 66.4 (-11.6%) | 58.7 (-21.8%) | 49.2 (-34.5%) |
| + Anti-Fraud Agent | 80.5 | 76.8 (-4.6%) | 70.3 (-12.7%) | 63.5 (-21.1%) |

### Ablation Study: Anti-Fraud Agent Ablation

| Configuration | Fraud Detection Rate | Planning Quality | Description |
|------|-----------|---------|------|
| Complete Anti-Fraud Agent | Highest (~78%) | Highest | All three layers of defense enabled |
| w/o Cross-Verification | ~55% | Moderately Low | Single-source verification is easily deceived |
| w/o Anomaly Detection | ~48% | Low | Cannot identify multi-person coordinated fraud |
| w/o Tagging Mechanism | ~65% | Moderate | Identified but not effectively downweighted |

### Key Findings

- All evaluated frameworks suffer significant performance drops under fraud scenarios, with the degradation scaling up as fraud complexity increases (-12% → -23% → -34%).
- Multi-person coordinated fraud is the hardest to detect because individual reviews appear normal in isolation.
- Level-escalating fraud causes the most severe performance degradation due to its ability to adaptively bypass initial defenses.
- The Anti-Fraud Agent mitigates approximately 50-60% of the performance degradation but cannot eliminate it entirely.
- GPT-4-based frameworks show slightly better resistance to fraud than GPT-3.5-based ones, though the gap is marginal.

## Highlights & Insights

1. **Fills a critical gap**: Systematically evaluates for the first time the reliability of LLM multi-agent systems in adversarial data environments.
2. **Sophisticated progressive fraud design**: The three modes scale from simple to complex, accurately mirroring real-world fraudulent evolution.
3. **Exposes the "efficiency-security" paradox**: Highly efficient frameworks are often more fragile because they bypass thorough data validation protocols.
4. **Plug-and-play design** of the Anti-Fraud Agent makes it simple to integrate into any existing framework.
5. **Core Insight**: Security evaluations for AI agents should be prioritized as highly as capability assessments.

## Limitations & Future Work

- The study only covers travel planning; its generalizability to other fields (medical booking, financial investment, shopping recommendations) remains to be validated.
- Fraud patterns are pre-defined templates, which do not account for AI-generated adaptive fraud (e.g., using LLMs to generate high-quality fake reviews).
- The Anti-Fraud Agent relies on prompt engineering and lacks specialized training, making it prone to failure in complex situations.
- As a workshop paper, its scale is constrained, resulting in limited coverage of frameworks and scenarios.
- The definition of "fraud" in the evaluation is relatively straightforward; real-world fraud is considerably more subtle and covert.

## Related Work & Insights

- **TravelPlanner**: Focuses on evaluating planning execution; WandaPlan builds upon this by adding the security dimension.
- **Multi-Agent Safety Research**: Works like AgentBench and ToolBench focus on capability evaluation, whereas this work addresses security assessment.
- **LLM Adversarial Robustness**: Complements research on prompt injection and jailbreaking; while the latter attacks the model directly, this study targets the model's data inputs.
- Intuition: Security evaluation of AI Agents should become a standard procedure before deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ Assesses multi-agent planning frameworks through a security lens, which is both novel and critical.
- Experimental Thoroughness: ⭐⭐⭐ The three fraud modes are representative, but the scale of the evaluation is limited due to the workshop paper scope.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and the scenario design is highly intuitive.
- Value: ⭐⭐⭐⭐ Offers practical reference value for the security auditing of production-deployed AI agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ATLAS: Constraints-Aware Multi-Agent Collaboration for Real-World Travel Planning](../../ICLR2026/multi_agent/atlas_constraints-aware_multi-agent_collaboration_for_real-world_travel_planning.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/multi_agent/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](../../ACL2026/multi_agent/towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](../../ICLR2026/multi_agent/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)
- [\[ICML 2025\] AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML](automl-agent_a_multi-agent_llm_framework_for_full-pipeline_automl.md)

</div>

<!-- RELATED:END -->
