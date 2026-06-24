---
title: >-
  [Paper Note] Embracing Large Language Models in Traffic Flow Forecasting
description: >-
  [ACL 2025][Autonomous Driving][Traffic Flow Forecasting] The LEAF framework is proposed, which utilizes a dual-branch predictor comprising a graph branch (for pair-wise relations) and a hypergraph branch (for non-pair-wise relations) to generate candidate forecasts. A frozen LLM is then employed as a selector (interpreting discriminatively rather than generatively) to choose the optimal forecast, optimizing the predictor through feedback via ranking loss…
tags:
  - "ACL 2025"
  - "Autonomous Driving"
  - "Traffic Flow Forecasting"
  - "LLM Discriminative Ability"
  - "Graph Neural Network"
  - "Hypergraph"
  - "Ranking Loss"
date: 2026-05-08
content_hash: 2c61b0a3ad87575a
---

# Embracing Large Language Models in Traffic Flow Forecasting

**Conference**: ACL 2025  
**arXiv**: [2412.12201](https://arxiv.org/abs/2412.12201)  
**Code**: [https://github.com/YushengZhao/LEAF](https://github.com/YushengZhao/LEAF)  
**Area**: Autonomous Driving  
**Keywords**: Traffic Flow Forecasting, LLM Discriminative Ability, Graph Neural Network, Hypergraph, Ranking Loss

## TL;DR
The LEAF framework is proposed, which utilizes a dual-branch predictor comprising a graph branch (for pair-wise relations) and a hypergraph branch (for non-pair-wise relations) to generate candidate forecasts. A frozen LLM is then employed as a selector (interpreting discriminatively rather than generatively) to choose the optimal forecast, optimizing the predictor through feedback via ranking loss, achieving SOTA on PEMS datasets.

## Background & Motivation

**Background**: Traffic flow forecasting is a core challenge in Intelligent Transportation Systems (ITS). Extant mainstream approaches employ GNN/RNN/Transformer models to capture spatio-temporal dynamics. Recently, pioneering works have begun introducing LLMs into traffic forecasting.

**Limitations of Prior Work**: (1) Prior methods assume identical train/test distributions, yet traffic conditions undergo distribution shift due to special events, weather, or epoch-level transitions, causing performance degradation; (2) graphs only capture pair-wise relationships, whereas hypergraphs only model non-pair-wise relationships, making a single structure insufficient.

**Key Challenge**: The intuitive method for leveraging LLMs in traffic forecasting is to let the LLM directly "generate" forecast values. However, traffic data encapsulates complex spatio-temporal dynamics, which is overly challenging for the generative capabilities of language models—indeed, LLM-MPE performs worse than simple GNN methods across multiple datasets.

**Goal**: How can one harness the generalization and reasoning capabilities of LLMs to enhance traffic flow forecasting, while avoiding having the LLM directly process complex spatio-temporal relationships?

**Key Insight**: Instead of utilizing the "generative capability" of the LLM, this work leverages its "discriminative capability"—allowing the LLM to select the most plausible forecast from multiple candidates.

**Core Idea**: Generate candidate forecasts using traditional spatio-temporal models, employ an LLM as a selector, and adopt ranking loss based on the selection feedback to train the predictor.

## Method

### Overall Architecture
LEAF consists of two parts: (1) a dual-branch predictor—where the graph branch captures pair-wise spatio-temporal relationships and the hypergraph branch models non-pair-wise relationships; (2) an LLM-based selector—where a frozen LLaMA 3 70B selects the optimal prediction from a candidate set. Workflow: pre-train the dual branches $\rightarrow$ generate forecasts during testing $\rightarrow$ construct candidate outputs (with transformations) $\rightarrow$ select using the LLM $\rightarrow$ fine-tune the predictor via ranking loss $\rightarrow$ iterate.

### Key Designs

1. **Spatio-Temporal Graph Construction and Graph Branch**:

    - **Function**: Expands $T \times N$ spatio-temporal data into a spatio-temporal graph, using GCNs to capture pair-wise spatio-temporal relationships.
    - **Mechanism**: Constructs a spatio-temporal graph $\mathcal{G}^{ST}$ where nodes are the $TN$ spatio-temporal points and edges include spatial edges (adjacent sensors at the same time step) and temporal edges (adjacent time steps for the same sensor). Information is propagated through 7 layers of standard GCN convolution: $X^{(l)} = \sigma(\hat{A}^{ST} X^{(l-1)} W_G^{(l)})$.
    - **Design Motivation**: The graph branch excels at modeling localized propagation effects (e.g., traffic congestion at one junction affecting adjacent junctions).

2. **Hypergraph Branch**:

    - **Function**: Learns a hypergraph incidence matrix to capture non-pair-wise group dynamics.
    - **Mechanism**: Uses a learnable incidence matrix $I_H = \text{softmax}(X_H^{(l-1)} W_H)$ to execute hypergraph convolutions via $X_H^{(l)} = I_H(I_H^\top X_H^{(l-1)} + \sigma(W_E I_H^\top X_H^{(l-1)}))$. The first term models node interactions inside hyperedges, and the second models inter-hyperedge interactions.
    - **Design Motivation**: Morning peak commuting from residential areas to commercial districts is a typical non-pair-wise relationship—where a set of nodes fluctuates synchronously, which cannot be accurately represented by simple pair-wise graph edges.

3. **Candidate Set Construction and LLM Selector**:

    - **Function**: Applies various transformations to the forecasts from the dual branches to expand the candidate set, from which the LLM selects.
    - **Mechanism**: Transformations include smoothing, increasing trend (linear increase from 1-12%), decreasing trend, overestimation (+5%), and underestimation (-5%). Together with the raw prediction, this yields 12 candidates. A prompt containing the task description, spatio-temporal information, historical data, and candidate set is constructed for LLaMA 3 70B to select the optimal item.
    - **Design Motivation**: (1) Providing more candidates gives the LLM greater latitude to cope with distribution shifts (e.g., selecting an increasing trend during Monday morning peaks); (2) choosing (discriminating) among candidates is substantially easier for the LLM than direct numerical generation, effectively unleashing its commonsense reasoning capabilities.

4. **Ranking Loss Feedback**:

    - **Function**: Backpropagates the LLM selection feedback using ranking loss to train the predictor.
    - **Mechanism**: $\mathcal{L}^G = [\Delta(y_i^G, \hat{y}_i) - \inf_{y_i' \in \mathcal{C}_i \setminus \{\hat{y}_i\}} \Delta(y_i^G, y_i') + \epsilon]_+$, which forces the predictor output to be closer to the chosen candidate than to sub-optimal candidates.
    - **Design Motivation**: Since the LLM selection might not perfectly match the ground truth, directly optimizing with MSE/MAE would introduce excessive noise. Ranking loss only demands correct relative ranking, making it more robust.

### Loss & Training
- Pre-training phase: Both branches are trained independently using MAE loss.
- Test-time adaptation: Ranking loss (Huber distance, margin $\epsilon=0$), updating for $M=5$ steps per round, with $K=2$ prediction-selection iteration rounds.
- Hidden dimension $d=64$, 7 layers, batch training.

## Key Experimental Results

### Main Results

| Method | PEMS03 MAE | PEMS04 MAE | PEMS08 MAE | PEMS08 RMSE | PEMS08 MAPE |
|------|-----------|-----------|-----------|-------------|-------------|
| DCRNN (GNN+RNN) | 29.99 | 34.36 | 31.41 | 43.91 | 15.44% |
| STSGNN (GNN) | 28.21 | 33.43 | 29.58 | 41.95 | 12.90% |
| DyHSL (Hypergraph) | 27.10 | 33.36 | 27.34 | 39.05 | 11.56% |
| STAEformer (Transformer) | 27.87 | 33.77 | 27.43 | 38.16 | 11.36% |
| LLM-MPE (LLM Generative) | 33.82 | 35.63 | 26.42 | 40.02 | 10.61% |
| **LEAF (Ours)** | **25.46** | **31.49** | **24.68** | **36.07** | **10.56%** |

### Ablation Study (PEMS08)

| Configuration | MAE | RMSE | MAPE |
|------|-----|------|------|
| Graph branch only | 29.12 | 41.36 | 13.54% |
| Hypergraph branch only | 27.94 | 39.11 | 11.82% |
| w/o hypergraph | 26.29 | 38.18 | 12.83% |
| w/o graph | 25.80 | 37.23 | 11.00% |
| w/o transformation | 25.47 | 36.47 | 11.01% |
| w/o ranking loss | 25.41 | 37.00 | 11.34% |
| **LEAF** | **24.68** | **36.07** | **10.56%** |

### Key Findings
- **LLM as a discriminator is significantly superior to generator**: LLM-MPE (generative) yields 33.82 MAE on the large-scale network PEMS03, falling short of simple GNNs, whereas LEAF (discriminative) achieves 25.46 MAE, leading by a large margin.
- **Dual branches complement each other**: Eliminating either branch leads to performance degradation, demonstrating the importance of modeling both pair-wise and non-pair-wise relations.
- **Significant impact of the LLM selector**: Graph branch alone yields 29.12 MAE $\rightarrow$ with LLM = 26.29; hypergraph alone yields 27.94 $\rightarrow$ with LLM = 25.80.
- **Ranking loss is superior to direct fitting**: Discarding the ranking loss escalates the RMSE from 36.07 to 37.00.
- **Greater advantage in long-term forecasting**: In 12-step prediction, LEAF's performance in early steps yields margins close to the base branches, but error is significantly reduced in later steps.

## Highlights & Insights
- **"Employing LLM for selection rather than generation"**: This is the core architectural insight. LLMs excel at semantic comprehension and commonsense reasoning (e.g., "peak hours end at 7 PM") but struggle with precise numerical regressions. Positioning it as a discriminator rather than a generator provides an elegant capabilities match.
- **Ranking loss tolerates selection noise**: Since the LLM selection is imperfect, ranking loss only requires correct relative rankings, elegantly handling noisy supervisory signals.
- **Candidate expansion via transformations**: Even simple transformations (trend/smoothing/offsetting) supply the LLM with sufficient operational range to adapt to distribution shifts.

## Limitations & Future Work
- Evaluated solely on PEMS traffic datasets, lacking validation on other spatio-temporal tasks (such as meteorology or energy).
- The LLM remains unfine-tuned. Employing Parameter-Efficient Fine-Tuning approach like LoRA could further elevate selector performance.
- Performance degrades when iteration rounds exceed $K>2$, due to the absence of cross-turn context memory, which leads to redundant consideration of identical factors.
- Inference cost of LLaMA 3 70B is heavy, requiring careful efficiency considerations for practical deployment.
- Only 10% of the training data is utilized; performance under full data scale remains unexplored.

## Related Work & Insights
- **vs LLM-MPE**: LLM-MPE forces LLMs to generate predictions directly, showing poor results on large networks. LEAF shifts to a discriminative approach, bypassing the LLMs' bottleneck in handling complex spatio-temporal dynamics.
- **vs DyHSL**: DyHSL is the predecessor of LEAF's hypergraph branch. LEAF further improves upon it by integrating a graph branch and an LLM selector.
- **vs STAEformer**: Pure Transformer schemes lack explicit graph/hypergraph structure modeling.

## Rating
- Novelty: ⭐⭐⭐⭐ The positioning of "LLM as a discriminator" is very creative, but the dual-branch predictor design itself is a compilation of existing works.
- Experimental Thoroughness: ⭐⭐⭐ Only evaluated on 3 PEMS datasets, and the 10% training data configuration is highly unique.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the visual analyses are highly convincing.
- Value: ⭐⭐⭐⭐ Provides a practical paradigm for leveraging LLMs in spatio-temporal forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TrafficAlign: Aligning Large Language Models for Traffic Scenario Generation](../../CVPR2026/autonomous_driving/trafficalign_aligning_large_language_models_for_traffic_scenario_generation.md)
- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](../../CVPR2025/autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [\[AAAI 2026\] TimeBill: Time-Budgeted Inference for Large Language Models](../../AAAI2026/autonomous_driving/timebill_time-budgeted_inference_for_large_language_models.md)
- [\[ECCV 2024\] Navigation Instruction Generation with BEV Perception and Large Language Models](../../ECCV2024/autonomous_driving/navigation_instruction_generation_with_bev_perception_and_large_language_models.md)
- [\[ICCV 2025\] TARS: Traffic-Aware Radar Scene Flow Estimation](../../ICCV2025/autonomous_driving/tars_traffic-aware_radar_scene_flow_estimation.md)

</div>

<!-- RELATED:END -->
