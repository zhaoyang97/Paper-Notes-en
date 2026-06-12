---
title: >-
  [Paper Note] STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning
description: >-
  [ACL2026][Time Series][Spatio-temporal time series] STReasoner utilizes Network SDEs to synthesize spatio-temporal time series data integrated with graph structures and textual semantics. Through a time-series encoder…
tags:
  - "ACL2026"
  - "Time Series"
  - "Spatio-temporal time series"
  - "Spatial-aware reinforcement learning"
  - "S-GRPO"
  - "Synthetic data"
  - "Multi-modal LLM"
date: 2026-05-08
content_hash: 6ca191cf5b56df17
---

# STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning

**Conference**: ACL2026  
**arXiv**: [2601.03248](https://arxiv.org/abs/2601.03248)  
**Code**: https://github.com/LingFengGold/STReasoner  
**Area**: Spatio-Temporal Reasoning / Autonomous Driving  
**Keywords**: Spatio-temporal time series, Spatial-aware reinforcement learning, S-GRPO, Synthetic data, Multi-modal LLM  

## TL;DR
STReasoner utilizes Network SDEs to synthesize spatio-temporal time series data integrated with graph structures and textual semantics. Through a time-series encoder, three-stage training, and spatial-aware S-GRPO, the LLM learns to perform explicit reasoning based on temporal dynamics and spatial dependencies.

## Background & Motivation
**Background**: Transportation networks, power grids, disease transmission, and river/climate systems can be represented as time series on nodes coupled with spatial graph structures. Most existing spatio-temporal models optimize for prediction accuracy, such as future traffic flow or numerical values, while LLMs/Time-Series Language Models, though capable of Q&A or multi-step reasoning, often neglect spatial dependencies between nodes.

**Limitations of Prior Work**: Real-world decision-making scenarios require more than numerical prediction; they demand answers to "which cause led to what change, where, and when." For instance, in traffic congestion, if a user asks "Which source node caused the congestion at Node 2 at 9:00?", the model must track upstream nodes, propagation delays, and time-series variations rather than merely observing the value at Node 2.

**Key Challenge**: Spatio-temporal reasoning requires numerical precision, graph structure dependency, and natural language explanation simultaneously. Existing prediction models lack explicit reasoning; existing LLMs lack spatio-temporal priors; conventional RL only rewards the final answer, potentially leading models to rely on superficial temporal patterns without truly utilizing the graph structure.

**Goal**: The authors propose three objectives: constructing controllable and text-aligned spatio-temporal reasoning data; defining ST-Bench to cover diverse reasoning capabilities; and training STReasoner to fuse time series, graphs, and text, utilizing spatial-aware RL rewards to ensure true dependency on spatial structures.

**Key Insight**: The paper implements a closed loop starting from data synthesis. First, Network SDEs and multi-agent generators create data with explicit spatial propagation, time lags, and semantic descriptions. Then, a TS-LM is trained on this data. Finally, a contrastive S-GRPO reinforces spatial grounding by providing rewards only when performance with graph input significantly exceeds performance without it.

**Core Idea**: Spatio-temporal reasoning is modeled as $f:(Q,T,G)->(R,A)$. Through "time-series encoding + graph-text prompting + spatial dependency rewards," the LLM's reasoning process is forced to be sensitive to both temporal dynamics and graph structures.

## Method
The STReasoner methodology consists of three layers: a data layer using Network SDEs for controllable systems, a model layer using a time-series encoder to interface numerical signals with the LLM, and a training layer using Align, SFT, and S-GRPO to establish modality alignment, reasoning cold-start, and spatial-aware reasoning behaviors.

### Overall Architecture
Given a graph $G=(V,E)$, time series $T_i$ for each node, and a natural language query $Q$, the model outputs an intermediate reasoning process $R$ and a final answer $A$. In the data synthesis stage, a Scenario Generation Agent generates scenarios (e.g., traffic, energy); a Scenario Parsing Agent parses these into nodes, edges, temporal patterns, and spatial dependencies; a Judge Agent verifies logic; an SDE Parameters Agent and Time-Varying Adjacency Agent generate node drift/diffusion, coupling strength, time-varying edge weights, and propagation lags; and a simulation module integrates the Network SDEs. During training, STReasoner patchifies node time series for a 5-layer MLP encoder, inserts the embeddings into the token stream in node order, and processes them with the graph structure and query via Qwen3-8B.

### Key Designs
1.  **Network SDE Multi-Agent Data Synthesis**:
    - **Function**: Generates training/evaluation data with controllable temporal dynamics, spatial dependencies, and textual semantics.
    - **Mechanism**: The continuous latent process of each node is determined by drift, diffusion, and neighbor coupling terms. The authors distinguish between demand source nodes and propagation nodes: the former follow exogenous temporal patterns (e.g., sine or mean-reverting), while the latter are primarily influenced by neighbors. Edge weights are time-varying, simulating peak-hour shifts, and each edge has a propagation lag.
    - **Design Motivation**: LLMs cannot learn language semantics from time series alone, and without controllable graph dynamics, spatial reasoning cannot be validated. Network SDEs provide an interpretable, controllable world model for automated Q&A generation.

2.  **STReasoner Time Series-Language Fusion Architecture**:
    - **Function**: Synthesizes node-level time series, graph structures, and queries into a context manageable by LLMs.
    - **Mechanism**: A lightweight 5-layer MLP serves as the time series encoder. Input series are patchified, and encoded embeddings are inserted as special tokens, e.g., `[Node1:<TS1>, Node2:<TS2>, ..., Graph, Question]`. Graph structures are provided as text. The model utilizes value-preserving normalization similar to ChatTS to retain numerical information beyond mere shapes.
    - **Design Motivation**: Pure text representation is token-intensive, and pure image representation may lose precision. A dedicated TS encoder balances global shape with numerical accuracy while maintaining lower token costs than long-form text.

3.  **Three-Stage Training and Spatial-Aware GRPO**:
    - **Function**: Progressively aligns modalities, injects reasoning, and compels the model to utilize spatial structures.
    - **Mechanism**: Stage 1 (ST-Align) performs large-scale alignment pre-training over temporal, spatial, and spatio-temporal attributes. Stage 2 (ST-CoT) uses SFT with CoT trajectories generated by Claude-4.5-Sonnet via rejection sampling. Stage 3 employs S-GRPO: for one question, it generates two sets of responses—one with explicit spatial structure $o^{sp}$ and one without $o^{ns}$. If $r^{sp} > \beta r^{ns}$, an additional spatial reward $\alpha$ is provided; otherwise, only the original $r^{sp}$ is used.
    - **Design Motivation**: Standard GRPO only evaluates correctness, allowing models to guess based on temporal trends. The contrastive S-GRPO design requires "the graph actually helps" to gain rewards, directly optimizing for spatial structure utilization.

### Loss & Training
The reward function is a weighted combination of format and task rewards; outputs must follow `<think>...</think><answer>...</answer>`. Discrete labels receive a 1 for correct and 0 for incorrect; sequence predictions are scored by relative error. Total reward is $r=(1-\lambda)r_{task}+\lambda r_{format}$ with $\lambda=0.5$. Implementation details: Stage 1 trained for 1000 steps, Stage 2 for 400 steps (LR: $1e-5$); Stage 3 (ST-RL) for 1 epoch, group size 8, spatial reward parameters $\alpha=0.1, \beta=0.8$ (LR: $1e-7$).

## Key Experimental Results

### Main Results
The main table includes four tasks: T1 Causal/Source Reasoning, T2 Spatial Entity Recognition, T3 Spatial Correlation Reasoning, and T4 In-context Forecasting.

| Model | Input Mode | T1 ACC | T2 ACC | T3 ACC | T4 MAE | Est. Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5.2 | Text | 83.09 | 38.78 | 58.79 | 63.99 | $22.48 |
| GPT-5.2 | Image | 86.47 | 40.54 | 65.08 | 64.70 | $6.69 |
| Claude-4.5-Sonnet | Text | 78.64 | 41.93 | 77.87 | 63.74 | $45.80 |
| Qwen3-8B | Text | 21.26 | 5.28 | 5.53 | 94.03 | $3.85 |
| Qwen3-8B SFT+S-GRPO | Text | 89.37 | 65.41 | 81.34 | 66.35 | $3.85 |
| Qwen3-VL-8B SFT+S-GRPO | Image | 91.79 | 69.43 | 83.92 | 67.29 | $0.66 |
| ChatTS-8B | TS Encoder | 56.52 | 19.51 | 41.08 | 85.14 | $0.27 |
| Time-R1-7B | Text | 60.39 | 29.65 | 48.62 | 68.15 | $3.85 |
| STReasoner-8B | TS Encoder | 95.65 | 75.71 | 87.12 | 65.59 | $0.27 |

### Ablation Study
All three training stages are essential. Alignment alone lacks reasoning, SFT is crucial for cold-starting, and S-GRPO improves spatial task performance over standard GRPO.

| Configuration | T1 ACC | T2 ACC | T3 ACC | T4 MAE | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| STReasoner: Align+SFT+S-GRPO | 95.65 | 75.71 | 87.12 | 65.59 | Complete 3 stages |
| Align+SFT+GRPO | 91.79 | 69.60 | 86.12 | 69.96 | No spatial-aware reward |
| SFT+S-GRPO | 91.30 | 67.76 | 83.98 | 69.01 | No alignment pre-training |
| Align+SFT | 88.41 | 63.32 | 80.97 | 66.65 | No RL |
| SFT | 90.34 | 61.47 | 81.47 | 71.09 | SFT cold-start only |
| S-GRPO | 47.34 | 23.20 | 39.20 | 91.92 | No SFT, sparse rewards |
| Align | 3.38 | 8.79 | 3.77 | 75.36 | Modality alignment only |

### Key Findings
- STReasoner's advantages are concentrated in T1-T3 reasoning tasks rather than pure forecasting. T4 MAE is competitive with closed-source models but not overwhelmingly dominant, highlighting its value in explanation and reasoning.
- Representing time series as images is cheaper and stronger than text, but dedicated TS encoders remain superior by preserving both global shape and numerical precision.
- Alignment alone yields low scores, indicating it is not equivalent to reasoning; SFT is a mandatory cold-start for RL to prevent instability from sparse rewards.
- S-GRPO provides an average gain of ~5.10% over standard GRPO and increases the proportion of responses explicitly using spatial information.

## Highlights & Insights
- The primary innovation is transforming "whether spatial information is actually utilized" into a contrastive reward, rather than just providing a graph in the prompt and hoping the model learns.
- The Network SDE data pipeline is comprehensive, controlling temporal dynamics, spatial dependencies, and propagation lags to create verifiable Q&A.
- ST-Bench tasks (Causal, Entity, Correlation, Prediction) align with real-world decision needs: "Why, Who, How related, and What next."
- 8B models can outperform large model APIs at significantly lower costs when data and reward mechanisms are tailored for structured numerical reasoning.

## Limitations & Future Work
- Heavy reliance on synthetic data. Real-world noise, missing values, and latent variables are more complex.
- The 5-layer MLP encoder is sufficient for structured signals but may struggle with high-dimensional multivariate sensors or asynchronous sampling.
- Graph structures are explicitly provided. Future work should address cases where graphs must be inferred from data.
- S-GRPO requires paired rollouts (w/ and w/o graph), increasing training overhead.

## Related Work & Insights
- **vs Time-Series Language Models**: Previous models (ChatTS, Time-MQA) lack graph structures and spatial propagation rewards; STReasoner focuses on reasoning within spatio-temporal graphs.
- **vs Spatio-Temporal Prediction Models**: Traditional STGNNs excel at numerical forecasting but fail to generate natural language reasoning chains.
- **vs Standard GRPO/RL Reasoning**: Unlike DeepSeek-R1 which rewards final answers, S-GRPO requires a performance gain specifically from graph structure, making it more suitable for multi-modal structured reasoning.
- **Insights for Autonomous Driving**: Spatial rewards can be transferred to vehicle-to-everything (V2X) scenarios, road network anomaly explanation, and multi-sensor diagnostics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 
- Writing Quality: ⭐⭐⭐⭐☆ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Nested Spatio-Temporal Time Series Forecasting](../../ICML2026/time_series/nested_spatio-temporal_time_series_forecasting.md)
- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](../../NeurIPS2025/time_series/learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](../../ICML2026/time_series/learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)
- [\[ICML 2026\] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](../../ICML2026/time_series/patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)

</div>

<!-- RELATED:END -->
