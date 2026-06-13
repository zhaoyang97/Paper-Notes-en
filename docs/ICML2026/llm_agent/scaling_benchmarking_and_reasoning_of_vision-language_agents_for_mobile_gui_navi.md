---
title: >-
  [Paper Note] Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation
description: >-
  [ICML 2026][LLM Agent][Mobile GUI Navigation] The Xiaomi team proposes a systematic "Data-Evaluation-Reasoning" study for VLM mobile GUI agents: releasing the HyperTrack dataset (16k tasks / 674 Chinese Apps) and the GUI…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Mobile GUI Navigation"
  - "VLM Agent"
  - "Data Scaling"
  - "DAPO-RL"
  - "Semi-online Evaluation"
date: 2026-05-08
content_hash: 8022ffe7479219e2
---

# Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation

**Conference**: ICML 2026  
**arXiv**: [2605.27134](https://arxiv.org/abs/2605.27134)  
**Code**: https://github.com/xiaomi-research/guievalkit (Available)  
**Area**: Agent / Multimodal VLM  
**Keywords**: Mobile GUI Navigation, VLM Agent, Data Scaling, DAPO-RL, Semi-online Evaluation

## TL;DR
The Xiaomi team proposes a systematic "Data-Evaluation-Reasoning" study for VLM mobile GUI agents: releasing the HyperTrack dataset (16k tasks / 674 Chinese Apps) and the GUIEvalKit evaluation tool supporting 30+ models. They demonstrate that DAPO-style RL significantly outperforms SFT in OOD scenarios and utilize semi-online evaluation (SOEval) to reveal the core trade-off where "explicit reasoning sacrifices PASS@1 stability but enhances PASS@n diversity."

## Background & Motivation

**Background**: GUI Agents driven by VLMs (UI-TARS, AgentCPM-GUI, GUI-Owl, etc.) have become the mainstream solution for mobile automation. Training paradigms concentrate on SFT + preference alignment (DPO) + recent GRPO/DAPO-style RL.

**Limitations of Prior Work**: (1) Data side — Mainstream benchmarks (AITW, AndroidControl, AMEX, GUI Odyssey) are almost exclusively English apps; CAGUI, the largest Chinese dataset, contains only 600 tasks across 22 apps, failing to cover long-tail applications. (2) Evaluation side — Inconsistent scripts, action spaces, and conflicting definitions of step-level/episode-level metrics make horizontal comparisons difficult. (3) Training side — The scaling laws for SFT vs. RL in GUI scenarios have not been systematically mapped. (4) Inference side — Conclusions regarding whether "thinking mode is useful" are contradictory across multiple benchmarks, lacking decision-level (rather than step-level) analysis tools.

**Key Challenge**: GUI navigation is essentially a combination of long-horizon tasks, multimodal perception, and sequential decision-making. A significant gap exists between offline metrics (using ground-truth history) and real online execution, while fully online evaluation is too costly for large-scale use. An evaluation protocol is needed that reflects on-policy behavior while reusing static data.

**Goal**: Decomposition into four sub-problems — (a) Construct large-scale data covering Chinese long-tail apps; (b) Provide a unified evaluation tool for multiple models and benchmarks; (c) Clarify in-domain vs. OOD performance of SFT/RL across different data scales; (d) Use decision-level metrics to explain when "explicit reasoning backfires."

**Key Insight**: The authors observe that the disconnect between offline evaluation and online performance stems from feeding the model reference trajectories, whereas in real deployment, the model only sees its own prior decisions. By "switching" to the model's own decision artifacts on correct steps and falling back to the reference on errors, one can approximate the on-policy distribution without sacrificing the reproducibility of static data.

**Core Idea**: A quad-set linkage of "Data-Tool-Training-Evaluation" — HyperTrack providing 16k Chinese tasks, GUIEvalKit unifying interfaces, DAPO-RL outperforming SFT on OOD, and SOEval + decision-level diversity/stability quantifying the real cost of reasoning.

## Method

### Overall Architecture
Four modules serve a single analytical goal: HyperTrack dataset (Input: 16,080 real tasks, each step containing screenshots, text instructions, low-level action descriptions, bboxes) → UI-TARS-1.5-7B / Qwen3-VL-8B undergo SFT and DAPO-RL on 10 data subsets (16 to 8,192 episodes) → GUIEvalKit handles inference and evaluation across 5 benchmarks (AndroidControl, AiTZ, GUI Odyssey, CAGUI, HyperTrack) with a unified action space and 4 metrics (step type/exact match, episode progress/success) → SOEval further bridges offline evaluation to on-policy reality → Decision-level Diversity/Stability decomposes the impact of reasoning.

### Key Designs

1.  **HyperTrack Dataset + 4-way OOD Partition**:
    - **Function**: Fills the gap in Chinese GUI data and provides rigorous test sets across four dimensions: in-domain, unseen app, unseen device, and unseen app & device.
    - **Mechanism**: 16,080 episodes were collected from 674 Chinese Android Apps across 17 categories (including long-tail and tablet-exclusive), averaging 5.1 steps. Each step is annotated with high-level task descriptions + screenshots + low-level action descriptions + ground-truth bboxes for all clickable actions. The action space includes OPEN/CLICK/SCROLL/TYPE/STOP. Since the training set only contains phone data, the unseen-device test set (tablets) naturally evaluates device-level OOD.
    - **Design Motivation**: Compared to AITW (no bboxes), AndroidControl (English), and CAGUI (only 600 tasks), HyperTrack is the first large-scale collection to feature "hierarchical UI docs + bbox + screen descriptions + Chinese + dual-level instructions," enabling scaling law experiments and fine-grained reward design.

2.  **DAPO-style RL + Binary Reward + Data Scaling Experiments**:
    - **Function**: Systematically compares the scaling behavior of SFT and RL across 16 to 8,192 episodes.
    - **Mechanism**: Adopts the GRPO framework with three DAPO enhancements: Clip-Higher to raise the upper clipping bound, Dynamic Sampling to replace zero-advantage samples for gradient signal preservation, and Token-Level Policy Gradient Loss for equal token contribution. The objective function is $\mathbb{E}_{q,o_i}\frac{1}{\sum|o_i|}\sum_{i,t}\min(r_{i,t}\hat A_{i,t}, \text{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\hat A_{i,t})$ with group size $G=16$, $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.3$, and $\beta=0$ (no reference model to save VRAM). Reward $R = R_{\text{action-type}} + R_{\text{params}}$, where parameters are judged only if the action type is correct (click within bbox, correct scroll direction, exact text match).
    - **Design Motivation**: Performance grows approximately log-linearly with the number of training episodes. Crucially, the lead of RL over SFT is far greater in OOD (unseen app) than in-domain—a key empirical finding of the paper. This elevates the justification for RL in GUI Agents from intuition to a quantifiable scaling phenomenon, holding true even when switching backbones (Qwen3-VL-8B).

3.  **GUIEvalKit + SOEval + Decision-level Metrics**:
    - **Function**: Unifies inference interfaces, provides a semi-online evaluation protocol, and quantifies the impact of reasoning on decision behavior.
    - **Mechanism**: (i) The `ABCModel` triad (`prepare_input / generate / parse_response`) adapts to 30+ VLMs, supporting vLLM backends and `enable_thinking` toggles. (ii) SOEval uses a history selection operator at each step $i$: $\psi(o_t,a_t,\hat a_t,\hat\tau_t) = \phi(o_t,\hat a_t,\hat\tau_t)$ if $\hat a_t = a_t$, otherwise defaulting to $\phi(o_t,a_t)$. This uses the model's own artifacts when correct and reverts to reference when wrong, allowing the evaluation context to progressively approach on-policy distributions. (iii) Decision-level analysis maps $n=512$ rollouts to a decision space $\mathcal{D}$ via density clustering. Diversity is defined as $\text{Div} = H(p(d|M,S,s))$ and Stability as $\hat\theta = p(d^*|M,S,s)$, quantifying the diversity↑/stability↓ trade-off caused by reasoning.
    - **Design Motivation**: Validated against AndroidWorld online success rates as the "gold standard," SOEval's step exact match achieved Spearman $\rho=0.771$ ($R^2=0.624$), significantly higher than offline metrics ($\rho=0.657, R^2=0.482$). Decision-level metrics explain why thinking mode loses to instruct on PASS@1 but overtakes it on PASS@8—they are different points on the same stability vs. diversity trade-off curve.

### Loss & Training
SFT: Standard cross-entropy; RL: DAPO-GRPO ($G=16, \epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.3, \beta=0$), primarily using binary rewards with an ablation on Gaussian spatial rewards. The primary backbone is UI-TARS-1.5-7B, with Qwen3-VL-8B-Thinking used to verify the universality of scaling trends.

## Key Experimental Results

### Main Results: Cross-benchmark Offline Evaluation (Step Type / Exact Match, Selected)

| Model | AndroidControl-low | GUI-Odyssey | AiTZ | CAGUI | HyperTrack |
|-------|--------------------|-------------|------|-------|-----------|
| Qwen3-VL-8B-Thinking | 81.08 / 71.10 | 74.01 / 46.98 | 66.90 / 47.27 | 78.37 / 56.89 | 77.03 / 59.35 |
| Qwen3-VL-8B-Instruct | 82.36 / 72.20 | 77.85 / 51.78 | 72.77 / 52.64 | 83.83 / 63.94 | 81.48 / 66.28 |
| MiMo-VL-7B-RL (w/o thinking) | 94.03 / 90.23 | 85.64 / 67.08 | 79.38 / 66.91 | 79.27 / 61.60 | 92.56 / 76.41 |
| UI-TARS-7B-SFT | 98.08 / 94.81 | 86.94 / 68.82 | 82.92 / 67.34 | 89.99 / 70.62 | 90.40 / 75.40 |
| UI-TARS-72B-SFT | 98.17 / 95.05 | 89.80 / 72.27 | 84.27 / 69.83 | 91.08 / 74.53 | 90.16 / 75.20 |
| AgentCPM-GUI-8B | 92.80 / 88.60 | 90.82 / 74.84 | 85.46 / 76.08 | **96.88 / 91.32** | 82.80 / 54.26 |

Findings: (a) Specialized GUI Agents generally outperform general VLMs; (b) UI-TARS shows continuous improvement from 2B to 72B; (c) **Thinking mode actually degrades performance across multiple models** (Qwen3-VL series, MiMo-VL, and GUI-Owl all scored higher without thinking).

### Key Comparison: SOEval vs. Offline (5-benchmark Average Exact Match)

| Model | Offline | SOEval | Gain |
|-------|---------|--------|---|
| Qwen3-VL-4B-Instruct | 59.39 | 63.05 | +3.66 |
| Qwen3-VL-8B-Instruct | 60.39 | 62.84 | +2.45 |
| GUI-Owl-7B | 65.49 | 67.37 | +1.88 |
| UI-Venus-Navi-72B | 74.09 | 76.16 | +2.07 |

SOEval consistently raises PASS@1 and correlates more strongly with AndroidWorld online success rates (Spearman 0.771) than offline evaluation (0.657).

### Decision-level Analysis (Reasoning vs. Instruct-only)
- **Stability shift** (SOEval vs. offline): For GUI-Owl-7B, the rising group gained +0.4500 while the falling group lost −0.3882. SOEval primarily rescues unstable samples at the cost of slight harm to previously stable ones.
- **Reasoning–execution consistency** (GUI-Owl-7B): R-E consistent samples have a 73.70% success rate, while inconsistent ones have only 18.29% (absolute difference of 55.4 pp, $\chi^2 = 2389.58$, $\phi = 0.489$). Alignment between reasoning and execution is critical.
- **Failure Analysis**: Action-type mismatch accounts for 61.1% of failures (high-level decision errors), followed by action-target mismatch (grounding errors). This suggests the main drawback of thinking mode is selecting the wrong action type despite extensive "thought."

### Key Findings
- Performance grows approximately log-linearly with training episodes, and **RL consistently exhibits a steeper slope and higher OOD gains than SFT**.
- Thinking mode is not a free lunch: it increases decision diversity and rescues low-stability samples but destabilizes high-stability ones, leading to a net loss in PASS@1. It only outperforms instruct at PASS@8, indicating that the choice of evaluation protocol determines the perceived utility of "thinking."
- SOEval reveals that current models lack a mechanism to balance recent on-policy context with stable decision-making. Increasing on-policy history (OSR) monotonically improves EM, but excessive history can interfere with otherwise stable decisions.

## Highlights & Insights
- **SOEval is an ingenious design**: By using the $\psi$ operator to switch between on-policy artifacts for correct steps and reference data for errors, it maintains reproducibility while approaching real deployment distributions. This can be transferred to any long-horizon agent scenario.
- **Framing "thinking utility" as a stability-diversity trade-off**: The authors use 512-rollout clustering to show that reasoning moves the model's operating point toward higher diversity and lower stability, providing a foundation for designing adaptive thinking gates.
- **DAPO's OOD advantage over SFT**: This empirical evidence provides a clear mandate for using RL in agent projects where training data is limited and deployment environments are diverse.

## Limitations & Future Work
- Only a preview subset of HyperTrack is released; the full 16k data is unavailable, making the scaling experiments difficult to replicate.
- Decision-level clustering relies on density methods and requires 512 rollouts, which is computationally expensive and unsuitable for real-time monitoring.
- SOEval still relies on an "on-track" assumption—it reverts to ground truth as soon as a model deviates, which may be unfair to agents capable of self-correction.
- Experiments are restricted to Android and Chinese apps; cross-platform (Desktop/Web) and cross-language generalization remain unverified.

## Related Work & Insights
- **vs. UI-TARS / AgentCPM-GUI**: These are the models evaluated. Ours contributes "data + tools + methodology," positioning itself as a benchmark foundation similar to HELM for GUI Agents.
- **vs. DAPO (Yu et al. 2025)**: While the original DAPO targeted reasoning tasks, Ours adapts it to GUI action prediction with tripartite enhancements and composite binary rewards, proving is applicability to multimodal sequential decision-making.
- **vs. AndroidWorld**: AndroidWorld is the gold standard but expensive; Ours provides a cheaper proxy (SOEval) with high correlation ($\rho=0.77$).

## Rating
- Novelty: ⭐⭐⭐⭐ The dataset and tools are solid, but the combination of SOEval and decision-level metrics is a unique contribution to the GUI Agent field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 5 benchmarks, 30+ models, and 10 data scales across multiple backbones.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, though some decision-level joint distribution plots require appendix consultation.
- Value: ⭐⭐⭐⭐⭐ Provides a comprehensive "data + evaluation + training" baseline for the Chinese GUI Agent community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[ICML 2026\] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)
- [\[ICML 2026\] Scaling Small Agents Through Strategy Auctions](scaling_small_agents_through_strategy_auctions.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](../../CVPR2026/llm_agent/gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)

</div>

<!-- RELATED:END -->
