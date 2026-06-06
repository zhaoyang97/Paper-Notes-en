---
title: >-
  [Paper Note] Adaptive Time Series Reasoning via Segment Selection
description: >-
  [ICML 2026][Time Series][Time series reasoning] This paper proposes ARTIST, which transforms time-series question answering into a sequential decision-making problem of "reasoning while selecting segments." Through a con…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time series reasoning"
  - "segment selection"
  - "controller-reasoner"
  - "self-play RL"
  - "hierarchical policy optimization"
date: 2026-05-08
content_hash: 0f01159268ae181b
---

# Adaptive Time Series Reasoning via Segment Selection

**Conference**: ICML 2026  
**arXiv**: [2602.18645](https://arxiv.org/abs/2602.18645)  
**Code**: https://github.com/mims-harvard/ARTIST  
**Area**: Time Series  
**Keywords**: Time series reasoning, segment selection, controller-reasoner, self-play RL, hierarchical policy optimization  

## TL;DR
This paper proposes ARTIST, which transforms time-series question answering into a sequential decision-making problem of "reasoning while selecting segments." Through a controller-reasoner architecture and hierarchical self-play RL, the model reads only problem-relevant time segments and improves reasoning accuracy.

## Background & Motivation
**Background**: Time-series tasks are expanding from traditional forecasting, classification, and anomaly detection to natural language question-answering reasoning. Given a question, a model must locate relevant intervals, compare patterns, explain changes, and output an answer. Existing methods typically serialize the entire sequence as text, render it as an image, or encode it into embeddings before feeding it into an LLM at once.

**Limitations of Prior Work**: Processing the complete time series simultaneously introduces a large amount of irrelevant segments into the context. For long sequences or multi-step reasoning tasks, truly useful information may reside only in a few short intervals and changes based on intermediate reasoning conclusions. A fixed view cannot achieve the dynamic process of "viewing one segment to establish a baseline, then viewing another to verify a hypothesis."

**Key Challenge**: The model needs to actively choose which time segments to view, but training data usually lacks annotations for "which intervals should be viewed for this question." Simultaneously, if long reasoning trajectories are optimized directly using token-level RL, the credit assignment for segment selection is diluted by long text outputs.

**Goal**: To enable the LLM to treat the time series as an interactive resource during reasoning: first selecting a segment, reasoning based on that segment, and then deciding whether to continue selecting or stop to answer. Training must separately optimize "where to look" and "how to answer."

**Key Insight**: The paper divides a single model into a controller and a reasoner using role-specific prompts. The controller is responsible for selecting temporal segments and termination conditions; the reasoner generates intermediate reasoning and answers based only on selected segments. This decouples evidence acquisition from answer generation and allows for different reward designs for the two roles.

**Core Idea**: Utilizing controller-reasoner collaborative self-play to train time-series reasoning as an interpretable adaptive segment selection process.

## Method
The core of ARTIST is formalizing time-series reasoning as an interaction trajectory. Given a question $q$ and a time series $T\in\mathbb{R}^{H\times V}$, the controller at round $i$ observes the question, full sequence, selected segments, and the previous reasoning/answer, then outputs a CONTINUE/ACCEPT decision. If it continues, it selects a new continuous segment $s_i=T_{t_{start}:t_{end}}$. The reasoner receives the accumulated segment list $S_i$ and generates a reasoning trace and a candidate answer. If the controller selects ACCEPT, the previous round's answer becomes the final output.

### Overall Architecture
Training consists of two stages. The first stage is SFT, fine-tuning the model with human or automatically constructed structured traces to learn to alternate between NL reasoning and segment-selection calls. The second stage is RL using collaborative self-play: a single policy model plays both controller and reasoner roles via different prompts to generate interaction trajectories, using nested rollouts to calculate rewards for both roles.

In RL, $G$ interaction trajectories are sampled for each training instance. For the final segment list of each trajectory, the reasoner is sampled independently $N$ times to estimate whether "these segments can stably support the correct answer." The controller's reward primary derives from reliability, defined as the proportion of correct answers under repeated reasoner sampling; the reasoner's reward derives from final answer correctness and format compliance. Finally, controller advantage is propagated to all controller decision tokens, while reasoner advantage is propagated only to the final round's output.

### Key Designs
1. **Controller-Reasoner Role Splitting**:
	- Function: Decouples "selecting evidence" and "answering based on evidence" into separately optimizable behaviors.
	- Mechanism: The controller outputs $d_i\in\{\mathrm{CONTINUE},\mathrm{ACCEPT}\}$ and a new segment $s_i$; the reasoner generates a reasoning trace and answer based on the accumulated $S_i$. Both roles share model parameters but are activated by different prompts.
	- Design Motivation: If a single chain-of-thought is responsible for both, RL struggles to distinguish if an error stems from wrong evidence or failed reasoning. Splitting roles clarifies credit assignment.

2. **Reliability-based controller reward**:
	- Function: Prevents the controller from being misled by a single, accidentally correct answer.
	- Mechanism: Repeatedly samples the reasoner $N$ times for the same segment list and calculates the accuracy $D(q,S,y^*)$ as reliability. High reward is given only when segments stably support the correct answer.
	- Design Motivation: LLM generation is stochastic; a single correct answer might be luck. Segment selection requires measuring if "evidence is sufficient" rather than if the reasoner happened to guess correctly.

3. **Hierarchical Policy Optimization and Variance-guided Sampling**:
	- Function: Allocates long-trajectory credit to the correct roles and stages.
	- Mechanism: The controller uses a trajectory-level advantage covering all interaction rounds; the reasoner optimizes only the final-round output. To save memory and select more informative samples, the paper employs variance-guided sampling based on correctness variance $r_\sigma$, prioritizing groups with higher outcome variance.
	- Design Motivation: Segment selection is a long-term decision that cannot be rewarded only at the final step, whereas the reasoner behaves like a local QA task suitable for optimization on the final output.

### Loss & Training
SFT is performed using LoRA on structured trajectories. The RL stage uses full-parameter fine-tuning, converting $R_{ctl}$ and $R_{rsn}$ into group-relative advantages for joint policy updates. Implementation-wise, the backbone is Qwen3-4B, with time series encoded via a 5-layer MLP for patch-based input. Evaluation uses reasoner temperature $0.7$ and controller temperature $1.0$. The main setup focuses on univariate time series.

## Key Experimental Results

### Main Results
The main experiment covers 6 benchmarks: ETI, RCW, ECG-QA, Sleep-QA, TSQA, and TRQA.

| Method | ETI Acc/F1 | RCW Acc/F1 | ECG-QA Acc/F1 | TSQA Acc/F1 | TRQA Acc/F1 | Avg Acc/F1 |
|--------|------|------|----------|------|------|------|
| OpenTSLM-4B + SFT | 82.69 / 82.66 | 65.49 / 38.29 | 69.50 / 41.00 | 47.50 / 35.81 | 76.25 / 69.36 | 62.80 / 47.68 |
| ITFormer-4B + SFT | 84.62 / 84.60 | 67.31 / 57.95 | 57.31 / 49.91 | 49.50 / 23.62 | 80.12 / 74.22 | 62.08 / 51.01 |
| Ours + SFT | 85.12 / 85.11 | 69.75 / 61.46 | 56.31 / 55.68 | 60.06 / 57.13 | 82.26 / 62.32 | 63.61 / 56.61 |
| Ours + SFT + RL | 87.03 / 87.10 | 77.00 / 50.00 | 69.81 / 52.67 | 62.00 / 58.66 | 83.06 / 78.02 | 69.26 / 57.61 |
| Gain (vs strongest baseline) | +2.41 / +2.50 | +3.11 / +3.51 | +3.14 / +3.89 | +12.50 / +11.91 | +2.94 / +3.80 | +6.46 / +6.60 |

### Ablation Study
Ablations on ECG-QA and RCW report accuracy to verify core modules.

| Configuration | ECG Acc | RCW Acc | Avg Acc | Description |
|------|---------|------|------|------|
| ARTIST | 69.81 | 77.00 | 73.41 | Full controller-reasoner + reliability + hierarchical RL |
| Reasoner Only | 65.33 | 62.88 | 64.11 | No controller, static input, drops 9.30 |
| Controller-only RL | 60.81 | 68.13 | 64.47 | Frozen reasoner cannot adapt to selection distribution |
| w/o Reliability Reward | 52.50 | 51.44 | 51.97 | Largest drop; single correctness misleads selection |
| w/o Trajectory-based Objective | 55.19 | 67.06 | 61.13 | Myopic controller fails to learn multi-round strategies |
| w/o Variance-guided Sampling | 68.13 | 72.75 | 70.44 | Variance provides more effective learning signals |

### Key Findings
- ARTIST improves average accuracy by $6.46\%$ over the strongest baseline on each dataset, showing that dynamic selection provides tangible quality gains beyond interpretability.
- RL continues to improve accuracy over SFT (from $63.61\%$ to $69.26\%$), suggesting segment selection benefits from reliability rewards beyond mere imitation.
- Analysis of data utilization shows more coverage is not always better. Sleep-QA and TRQA peak at $30-50\%$ signal usage; utilizing the full sequence degrades performance.
- Inference costs increase: e.g., on TRQA, ARTIST takes $1.68$ mins/case ($8$ runs) vs OpenTSLM's $1.26$ mins. However, as sequences scale to $12K$, time only increases from $1.880$ to $1.910$ mins, indicating costs are driven by interaction rounds rather than sequence length.

## Highlights & Insights
- This paper shifts time-series reasoning from "how to encode the whole sequence" to "which segment to see during reasoning." This aligns with real-world needs for zooming in/out and comparing segments.
- The reliability reward is crucial. It changes the controller's objective from "making the reasoner correct now" to "selecting evidence sufficient to ensure stable correctness," closer to the essence of information retrieval.
- ARTIST's segment list naturally provides an evidence trajectory, facilitating audits of answer rationale—vital for medical and financial domains.

## Limitations & Future Work
- Higher inference cost than single-pass baselines due to multiple controller-reasoner calls.
- Main experiments focus on univariate series; multivariate, asynchronous sampling, and cross-variable causality significantly complicate segment selection.
- Whether segment selection is always "interpretable" requires caution; selected segments indicate evidence but are not strictly causal explanations.
- On Sleep-QA, tokenized ARTIST lags behind TimeMaster+RL, suggesting input modality and pre-training priors remain strong factors.

## Related Work & Insights
- **vs ChatTS / OpenTSLM / ITFormer**: These focus on encoding; ARTIST focuses on dynamic selection to avoid fixed global representations.
- **vs VL-Time / TimeMaster**: Visual methods use image priors; ARTIST uses tool-like segment selection instead of one-shot visual understanding.
- **vs Dynamic Visual Search**: While image search has spatial targets, time-series segments depend on relative baselines, requiring multi-round context-aware selection.
- **vs Standard self-play RL**: Unlike proposer/solver setups with immediate goals, ARTIST's controller requires a trajectory-level objective for long-term strategy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 
- Writing Quality: ⭐⭐⭐⭐☆ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)
- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](../../ICLR2026/time_series/reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[ICML 2026\] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)
- [\[ICML 2026\] DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal](distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo.md)
- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](../../ICLR2026/time_series/swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)

</div>

<!-- RELATED:END -->
