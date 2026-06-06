---
title: >-
  [Paper Note] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets
description: >-
  [AAAI 2026][Robotics][AI Agent Evaluation] This paper introduces the CAIA benchmark, which leverages cryptocurrency markets as a natural adversarial laboratory to evaluate 17 state-of-the-art LLMs on agent capabilities i…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "AI Agent Evaluation"
  - "Adversarial Benchmark"
  - "Cryptocurrency"
  - "Tool Selection"
  - "Hallucination"
date: 2026-05-08
content_hash: e9d4b5d033f993dd
---

# When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets

**Conference**: AAAI 2026
**arXiv**: [2510.00332](https://arxiv.org/abs/2510.00332)  
**Code**: [GitHub](https://github.com/SurfAI-CybertinoLab/CAIA)  
**Area**: Robotics
**Keywords**: AI Agent Evaluation, Adversarial Benchmark, Cryptocurrency, Tool Selection, Hallucination

## TL;DR

This paper introduces the CAIA benchmark, which leverages cryptocurrency markets as a natural adversarial laboratory to evaluate 17 state-of-the-art LLMs on agent capabilities in high-stakes adversarial environments. Results reveal that frontier models achieve only 67.4% accuracy (GPT-5) compared to a human baseline of 80%, and expose systematic tool selection failures.

## Background & Motivation

**Blind Spots in AI Benchmarking**: Existing AI evaluations (GLUE, ImageNet, ICPC, IMO) measure capabilities in closed-world settings, assuming tools behave as expected, information is trustworthy, and other agents are cooperative. Real-world autonomous deployment, however, demands **adversarial robustness**—the ability to operate in open systems rife with uncertainty, misinformation, and adversarial incentives.

**Benchmark Scores ≠ Deployment Readiness**: Agents that score highly on reasoning benchmarks may still fall for fabricated news, purchase manipulated assets, or be deceived by social engineering—because their evaluations never included deceptive scenarios. As AI agents increasingly interact with untrusted users, real funds, and critical infrastructure, this gap poses a significant safety risk.

**Why Cryptocurrency Markets**: 

**Genuinely Adversarial Environment**: Anonymous blockchains free malicious actors from reputational consequences; profit motives drive sophisticated attack strategies; regulatory vacuums permit deceptive practices illegal in traditional markets. Everyday attacks include honeypot contracts, flash loan exploits, and coordinated social engineering.

**High-Stakes and Irreversible**: No traditional financial safety nets exist; transactions are irreversible; smart contract execution is final. Losses exceeded $30 billion in 2024.

**Verifiable Ground Truth**: Despite adversarial chaos, blockchains provide complete transparency and immutability—every transaction is permanently recorded and cryptographically verifiable.

## Method

### Overall Architecture

The CAIA benchmark follows an "adversarial-first" design principle, comprising 178 temporally anchored tasks that evaluate agent knowledge, planning, and execution capabilities in cryptocurrency analysis.

### Key Designs

#### 1. **Three Pillars of Quality Assurance**

- **Knowledge**: Assesses foundational understanding of crypto-native concepts (AMM mechanisms, governance structures, etc.), testing conceptual mastery rather than definitional recall.
- **Planning**: Evaluates the ability to decompose complex problems into executable analytical workflows, requiring agents to specify tool selection and sequencing prior to execution.
- **Execution**: Tests real-world execution using production-grade APIs (Etherscan, CoinGecko, DefiLlama), assessing both technical capability and judgment.

#### 2. **Five-Stage Data Curation Pipeline**

1. **Automated Filtering**: LLM reviewers filter irrelevant, ambiguous, or unanswerable queries, retaining the top 15% (~1,000 tasks).
2. **Expert Review**: 92 domain experts, with at least 4 reviewers per task; extreme scores are dropped and the top 200 are selected by average.
3. **Format Standardization**: Uniform formatting with block number/timestamp anchoring to ensure objective evaluation.
4. **Ground Truth Verification**: Reproducible tool-chain solutions are validated; tasks that cannot be reproduced are removed → final set of 178 tasks.
5. **Categorization**: Six analytical categories for diagnostic purposes.

Source: Over 10,000 real-world queries from 3,000+ active users.

#### 3. **Task Category Distribution**

| Category | Count | Proportion | Focus |
|----------|-------|------------|-------|
| On-Chain Analysis | 77 | 43.3% | Transaction patterns, MEV, fund flows |
| Project Discovery | 49 | 27.5% | Protocol evaluation, security analysis |
| Token Economics | 23 | 12.9% | Incentive design, value accrual |
| Cross-Domain | 14 | 7.9% | Multi-domain synthesis |
| Trend Analysis | 8 | 4.5% | Temporal patterns, adoption metrics |
| General Knowledge | 7 | 3.9% | Foundational concepts |

### Experimental Setup

- **Evaluated Models**: 17 SOTA models (GPT-5, GPT-4.1, Claude Opus 4, Gemini 2.5 Pro, Grok 4, DeepSeek R1/V3.1, Llama 4, Qwen 3, etc.)
- **Two Conditions**: Without tools (closed-book) vs. with tools (open-book, 23 specialized tools)
- **Agent Framework**: Standard ReAct framework, ensuring evaluation is not confounded by implementation differences
- **Human Baseline**: 16 participants from university blockchain clubs and early-stage companies (junior analysts) completing a 10% stratified sample; average accuracy **80%**
- **Metrics**: Majority-vote accuracy over 5 independent runs + Pass@1/Pass@5 + cost efficiency

## Key Experimental Results

### Main Results

**Without Tools** (catastrophic failure across all models):

| Model | Majority-Vote Accuracy | Pass@1 | Pass@5 |
|-------|----------------------|--------|--------|
| GPT-5 | 27.5% | 28.1% | 42.7% |
| Gemini 2.5 Pro | 22.5% | 20.2% | 29.8% |
| GPT-o3 | 20.8% | 22.5% | 29.2% |
| Claude Opus 4 | 13.5% | 13.5% | 16.9% |
| DeepSeek R1 | 20.8% | 21.9% | 35.4% |

**With Tools**:

| Model | Majority-Vote Accuracy | Pass@1 | Pass@5 | Avg. Cost |
|-------|----------------------|--------|--------|-----------|
| **GPT-5** | **67.4%** | 70.2% | 77.0% | $0.1154 |
| GPT-OSS 120B | 62.9% | 56.2% | 72.5% | $0.0066 |
| Grok 4 Fast | 61.2% | 57.9% | 71.9% | $0.0098 |
| Claude Sonnet 4 | 56.7% | 57.9% | 66.9% | $0.2291 |
| Claude Opus 4 | 57.3% | 59.6% | 71.9% | $1.1139 |
| **Human Baseline** | **~80%** | - | - | - |

### Ablation Study

**Tool Usage Distribution (Tool Selection Catastrophe)**:

| Tool Category | Call Count | Proportion | Notes |
|--------------|-----------|------------|-------|
| Google Search | 11,626 | 49.6% | **Models prefer unreliable sources** |
| Specialized Blockchain Tools | 8,351 | 35.6% | Authoritative sources containing correct answers |
| URL Fetch | 1,743 | 7.4% | - |
| Twitter Search | 1,388 | 5.9% | Socially manipulated information |
| Code Execution | 355 | 1.5% | - |

**Cost–Accuracy Trade-off**:

| Model | Accuracy | Cost per Run | Cost Efficiency |
|-------|----------|-------------|-----------------|
| Claude Opus 4 | 57.3% | $1.1139 | Worst |
| GPT-OSS 120B | 62.9% | $0.0066 | **100× better value** |
| Grok 4 Fast | 61.2% | $0.0098 | Pareto-optimal |

### Key Findings

1. **Fundamental Capability Gap**: Without tools, all models perform near random chance (12–28%), whereas junior analysts routinely complete these tasks.
2. **Tool Selection Catastrophe**: Models direct 55.5% of tool calls to unreliable web search, even when specialized blockchain tools can directly provide correct answers.
3. **Illusion of Pass@k**: Gemini 2.5 Flash's improvement from 39.3% (Pass@1) to 62.4% (Pass@5) suggests the model is essentially guessing via trial and error, rather than engaging in strategic reasoning.
4. **Twitter Search Paradox**: Accuracy when used alone is only 6.6%, rising to 40.7% in combination, indicating that tools require orchestration capability.
5. **Case Study** (Task 49): Retrieving monthly token launch data from Pump.fun—a task solvable with a single API call—resulted in failure across all 17 models, each falling into a cascade of web search → outdated blog posts → speculative Twitter content.

## Highlights & Insights

1. **Adversarial-First Evaluation Paradigm**: The first benchmark to explicitly incorporate active deception, source verification, and adversarial robustness as core dimensions of AI agent evaluation.
2. **The Danger of Pass@k Metrics**: High Pass@k does not reflect capability but trial-and-error behavior—which is extremely dangerous in irreversible financial decision-making.
3. **Cost Efficiency of Open-Source Models**: GPT-OSS 120B achieves near-frontier performance at 1/100th of the cost.
4. **Generalizable Insights**: The findings extend beyond cryptocurrency—any adversarial domain (cybersecurity, content moderation, medical diagnosis) faces analogous challenges.

## Limitations & Future Work

1. The benchmark scale of 178 tasks is relatively small and highly concentrated in the crypto domain.
2. The human baseline involves only 16 participants completing a 10% sample, limiting statistical significance.
3. Model learning and adaptation following feedback are not evaluated.
4. Categorization concerns: the paper's classification under "object detection" appears to be a mislabeling; it is properly an AI Agent evaluation work.
5. The paper lacks concrete improvement recommendations for safe agent deployment.
6. Reliance on the ReAct framework means results may not generalize to other agent architectures.

## Related Work & Insights

- GAIA (General AI Assistant Benchmark) tests task completion but assumes a cooperative environment; this work extends evaluation to the adversarial dimension, filling an important gap in AI safety assessment.
- The temporally anchored task design draws on methods from time-sensitive evaluations such as RealTimeQA.
- The work resonates with HELM (Holistic Evaluation of Language Models) but focuses specifically on high-stakes scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first high-stakes adversarial AI agent benchmark, with far-reaching implications.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 17 models, dual-condition evaluation, multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Compelling argumentation, deep insights, and vivid case studies.
- Value: ⭐⭐⭐⭐⭐ — Carries significant cautionary implications for the safe deployment of AI agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[ICLR 2026\] When Agents Persuade: Propaganda Generation and Mitigation in LLMs](../../ICLR2026/robotics/when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)
- [\[AAAI 2026\] More Than Irrational: Modeling Belief-Biased Agents](more_than_irrational_modeling_belief-biased_agents.md)
- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](../../NeurIPS2025/robotics/labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)
- [\[AAAI 2026\] Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy](do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)

</div>

<!-- RELATED:END -->
