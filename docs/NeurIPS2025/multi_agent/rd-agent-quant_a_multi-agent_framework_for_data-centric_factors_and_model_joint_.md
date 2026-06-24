---
title: >-
  [Paper Note] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization
description: >-
  [NeurIPS 2025][Multi-Agent][multi-agent framework] This paper proposes R&D-Agent(Q), a data-driven multi-agent framework that automates the joint optimization of factor mining and model innovation for quantitative strategies through five collaborative modules (Specification, Synthesis, Implementation, Validation, and Analysis), achieving approximately 2× the annualized return of traditional factor libraries in real stock markets at a cost of under $10.
tags:
  - "NeurIPS 2025"
  - "Multi-Agent"
  - "multi-agent framework"
  - "quantitative factor mining"
  - "model optimization"
  - "data-driven"
  - "automated R&D"
date: 2026-05-08
content_hash: d0b0157b28fceda1
---

# R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2505.15155](https://arxiv.org/abs/2505.15155)  
**Code**: [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent)  
**Area**: LLM Agent / Quantitative Finance
**Keywords**: multi-agent framework, quantitative factor mining, model optimization, data-driven, automated R&D

## TL;DR

This paper proposes R&D-Agent(Q), a data-driven multi-agent framework that automates the joint optimization of factor mining and model innovation for quantitative strategies through five collaborative modules (Specification, Synthesis, Implementation, Validation, and Analysis), achieving approximately 2× the annualized return of traditional factor libraries in real stock markets at a cost of under $10.

## Background & Motivation

### 1. State of the Field

Financial markets are high-dimensional, nonlinear, and nonstationary dynamic systems, in which asset returns exhibit fat-tailed distributions, time-varying volatility, and complex cross-sectional dependencies. Quantitative investment is transitioning from experience-driven to data-driven paradigms, with a core pipeline encompassing: data processing → factor mining → model training → backtesting. Microsoft's open-source project Qlib has simplified the data processing and backtesting stages, shifting research focus to the two core challenges of factor mining and model innovation.

### 2. Limitations of Prior Work

- **Low automation**: Current workflows rely heavily on manual intervention (hypothesis generation, coding, and hyperparameter tuning), resulting in slow iteration; semi-automated systems cannot meet the response demands of fast-moving markets.
- **Poor interpretability**: Existing LLM agents generate trading signals directly from language interactions, lacking explicit factor construction and transparent model logic, which is prone to hallucination and difficult to deploy in live trading.
- **Fragmented optimization**: Data processing, factor mining, model training, and evaluation lack systematic task decomposition and agent-level coordination, with limited cross-stage feedback.

### 3. Root Cause

Factor mining and model innovation are the two key pillars of quantitative research, yet they are mutually dependent—good factors require good models for validation, and good models require good factors as input. Existing methods optimize either factors or models in isolation, lacking an automated framework for joint optimization.

### 4. Paper Goals

To design an end-to-end automated multi-agent framework capable of autonomously performing joint iterative optimization of factor mining and model innovation, forming a closed-loop "hypothesis → implementation → validation → feedback" cycle.

### 5. Starting Point

The quantitative R&D pipeline is decomposed into five LLM-driven functional units that simulate the trial-and-error process of human quantitative researchers. Each unit has well-defined input/output interfaces, and knowledge accumulation is achieved through persistent storage.

### 6. Core Idea

Five collaborative LLM agent modules form a closed-loop R&D cycle: a knowledge forest drives hypothesis generation, a Co-STEER code generation agent handles implementation, and a multi-armed bandit scheduler adaptively selects the optimization direction, together completing joint factor–model optimization.

## Method

### Overall Architecture

R&D-Agent(Q) decomposes quantitative R&D into five functional modules belonging to two phases—Research and Development:

1. **Research Phase**: Specification Unit (task definition) → Synthesis Unit (hypothesis generation)
2. **Development Phase**: Implementation Unit (code implementation) → Validation Unit (backtesting) → Analysis Unit (result analysis and scheduling)

The five modules form a continuously iterating closed loop supporting dynamic joint optimization of factors and models.

### Key Designs

#### 1. Specification Unit

- **Function**: Dynamically configures task context and constraints to ensure consistency across subsequent modules.
- **Mechanism**: Formalized as the tuple $\mathcal{S}=(\mathcal{B}, \mathcal{D}, \mathcal{F}, \mathcal{M})$, encoding prior knowledge, data interfaces, output formats, and the execution environment (the Qlib backtesting platform), respectively.
- **Design Motivation**: Reduces ambiguity through unified input/output specifications, relieving agents from concerns about low-level preprocessing and infrastructure.

#### 2. Synthesis Unit

- **Function**: Automatically generates new hypotheses based on historical experiments.
- **Mechanism**: Maintains an experimental trajectory $e^t = \{h^t, f^t\}$ (hypothesis + feedback) and a SOTA set; constructs a knowledge forest via conditional subset extraction; produces new hypotheses through a generative stochastic mapping $G(\mathcal{H}_t^{(a)}, \mathcal{F}_t^{(a)})$.
- **Adaptive Mechanism**: On success, increases complexity/scope; on failure, adjusts structure or introduces new variables—forming an "idea forest."
- **Design Motivation**: Simulates the integrative reasoning process of human analysts between theoretical priors and empirical feedback.

#### 3. Implementation Unit — Co-STEER

- **Function**: Translates hypotheses into executable code.
- **Mechanism**: Constructs a DAG to represent task dependencies and uses topological sorting to determine execution order; guided Chain-of-Thought reasoning improves code generation quality.
- **Knowledge Base Mechanism**: Records (task, code, feedback) triples $\mathcal{K}^{(t+1)} = \mathcal{K}^{(t)} \cup \{(t_j, c_j, f_j)\}$ and enables cross-task knowledge transfer via similarity retrieval.
- **Design Motivation**: Quantitative coding tasks have structural dependencies that require systematic scheduling and feedback-driven code optimization.

#### 4. Validation Unit

- **Function**: Evaluates the actual effectiveness of factors and models.
- **Factor Deduplication**: Computes the IC correlation between new factors and the SOTA factor pool; factors with IC_max ≥ 0.99 are deemed redundant and discarded.
- **Backtesting**: Conducted on the Qlib platform using real market data.

#### 5. Analysis Unit — Multi-Armed Bandit Scheduling

- **Function**: Evaluates experimental results along multiple dimensions and determines whether to prioritize factor or model optimization in the next round.
- **Mechanism**: Models direction selection as a contextual two-armed bandit problem; observes an 8-dimensional performance state vector; employs linear Thompson Sampling to adaptively balance exploration and exploitation.
- **Design Motivation**: The marginal returns of factor optimization and model optimization change across iterations, necessitating dynamic selection of the optimal direction.

### Loss & Training

This framework does not involve conventional end-to-end training. The core optimization objective is to maximize cumulative implementation quality $\pi_I = \arg\max_\pi \mathbb{E}[\sum_{j=1}^n R_I(c_j)]$, where $R_I(c_j)$ evaluates code correctness and performance. Direction scheduling maintains a posterior via Bayesian linear regression, with the linear reward function $r = \mathbf{w}^\top \mathbf{x}_t$ guiding direction selection.

## Key Experimental Results

### Main Results

Dataset: CSI 300, training 2008–2014, validation 2015–2016, test 2017–2020.08.

| Method Type | Model | IC | ICIR | ARR | IR | MDD | CR |
|-------------|-------|----|------|-----|-----|------|-----|
| ML | LightGBM | 0.0277 | 0.2211 | 3.97% | 0.57 | -8.55% | 0.46 |
| DL | TRA | 0.0404 | 0.3197 | 6.49% | 1.01 | -8.60% | 0.75 |
| DL | MASTER | 0.0215 | 0.1925 | 8.96% | 1.34 | -8.51% | 1.05 |
| Factor Library | Alpha 158 | 0.0341 | 0.2952 | 5.70% | 0.85 | -7.71% | 0.74 |
| Factor Library | Alpha 360 | 0.0420 | 0.3290 | 4.38% | 0.67 | -7.21% | 0.61 |
| **R&D** | **R&D-Factor(GPT-4o)** | **0.0489** | **0.4050** | **14.61%** | **1.68** | -7.50% | **1.95** |
| **R&D** | **R&D-Model(o3-mini)** | **0.0469** | 0.3688 | 10.09% | **1.70** | **-6.94%** | 1.45 |
| **R&D** | **R&D-Agent(Q)(o3-mini)** | **0.0532** | **0.4278** | **14.21%** | **1.74** | -7.42% | **1.92** |

**Key Finding**: R&D-Agent(Q) with joint optimization achieves the highest IC=0.0532, ARR=14.21%, and IR=1.74, comprehensively surpassing all baselines.

### Ablation Study

| Configuration | Core Observation |
|---------------|-----------------|
| R&D-Factor only | Dynamic factor optimization reaches IC comparable to Alpha 158 using ~22% of the factor count. |
| R&D-Model only | Adaptive model configuration outperforms all handcrafted DL architectures in risk control (MDD). |
| R&D-Agent(Q) joint | Joint optimization unlocks complementary gains; IC and strategy performance are comprehensively optimal. |
| GPT-4o vs. o3-mini | o3-mini converges faster in pass@k for structured coding tasks, reflecting stronger CoT reasoning. |
| Factor hypothesis cluster analysis | 8 of 36 iterations are admitted to the SOTA set, spanning 5/6 clusters → diverse exploration produces complementary signals. |

### Generalization

Evaluated on CSI 500 and NASDAQ 100 (test period 2024–2025), R&D-Agent(Q) maintains top-tier performance in both Chinese and U.S. markets, with IC/ICIR/IR/MDD metrics at or near the optimum, confirming the framework's cross-market generalization capability and robustness to knowledge cutoff issues.

### Key Findings

1. **Factor efficiency**: R&D-Factor achieves IC comparable to Alpha 158 using fewer than 22% of its factors, while remaining stable during the 2019–2020 period when baselines degrade.
2. **Model efficiency**: The R&D-Model variant significantly outperforms baseline DL models in the return–drawdown space, reaching the optimal return–risk frontier.
3. **Joint optimization gain**: The joint factor–model optimization of R&D-Agent(Q) unlocks complementary performance improvements, ultimately achieving approximately 2× the ARR of classical factor libraries.
4. **Cost**: The total API cost for the entire automated R&D process is under $10.
5. **Co-STEER self-repair**: Code generation pass@k converges rapidly within a few iterations; o3-mini demonstrates a stronger chain-of-thought reasoning advantage.

## Highlights & Insights

1. **First full-stack automated multi-agent framework for quantitative finance**: Integrates factor mining and model innovation into a unified closed-loop R&D pipeline, representing a significant paradigm exploration for "AI for Quant R&D."
2. **Data-driven rather than direct trading**: LLMs do not interact with raw market data or execute trades directly; instead, they generate hypotheses and code at the schema level, mitigating hallucination and data leakage risks.
3. **Knowledge forest + multi-armed bandit**: The idea forest in the Research phase supports "refine–pivot–reuse" patterns, while Thompson Sampling in the Analysis phase adaptively schedules factor/model optimization directions.
4. **Co-STEER code agent**: A code generation agent designed specifically for structured data tasks; DAG-based scheduling combined with knowledge transfer significantly improves factor/model coding efficiency.
5. **High return at minimal cost**: A sub-$10 API cost yields a 2× ARR improvement, demonstrating the high cost-effectiveness of LLM agents in quantitative R&D.

## Limitations & Future Work

1. **Market coverage**: The main experiments focus on the Chinese A-share market; although supplementary validation on U.S. equities is provided, coverage of additional emerging markets and asset classes (futures, options, cryptocurrencies, etc.) is limited.
2. **Absence of live trading validation**: All experiments are conducted on the Qlib backtesting platform; the framework has not been validated in a real trading environment accounting for slippage, liquidity, and market impact.
3. **LLM dependency**: Framework performance is bounded by the capabilities of the underlying LLM, and sensitivity to LLM upgrades or downgrades warrants further investigation.
4. **Factor interpretability**: Although the framework generates factors with hypothesis descriptions, whether the economic meaning of automatically generated factors is genuinely meaningful remains to be verified.
5. **Scheduling strategy**: The two-armed bandit only switches between factor and model optimization; extending to additional optimization dimensions (e.g., risk control, execution strategy) would significantly increase scheduling complexity.

## Related Work & Insights

- **Qlib (Microsoft)**: Provides standardized backtesting and data processing infrastructure; R&D-Agent(Q) automates the core research stages on top of this foundation.
- **AlphaFactor series** (Alpha 101/158/360): Static factor libraries; this paper demonstrates that automated dynamic factor generation can achieve superior results with fewer factors.
- **LLM for Finance** (FinGPT, BloombergGPT, etc.): Primarily focused on signal extraction or direct trading decision generation, lacking transparent factor construction and model validation pipelines.
- **Multi-agent simulation** (simulated hedge funds, financial expert collaboration): Emphasizes simulation rather than automated R&D iteration.
- **Inspiration**: This "Research–Development closed-loop" paradigm has potential for generalization to other data-driven R&D domains (drug discovery, materials science, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First full-stack automated multi-agent framework for quantitative finance; the joint factor–model optimization combined with multi-armed bandit scheduling is a distinctive design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive main experiments on CSI 300, supplemented by generalization validation on CSI 500 and NASDAQ 100; ablation and analysis are thorough, though live trading validation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Framework description is clear; modular design is easy to follow; formalization is appropriately scoped; figures and tables are of high quality.
- Value: ⭐⭐⭐⭐⭐ — Carries significant paradigm-level implications for automated quantitative R&D; code is open-sourced (Microsoft RD-Agent), with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](../../ICML2026/multi_agent/maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](../../ACL2026/multi_agent/masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)
- [\[NeurIPS 2025\] Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)
- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](../../ICML2026/multi_agent/omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ICML 2026\] Representational Similarity and Model Behavior in Multi-Agent Interaction](../../ICML2026/multi_agent/representational_similarity_and_model_behavior_in_multi-agent_interaction.md)

</div>

<!-- RELATED:END -->
