---
title: >-
  [Paper Note] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering
description: >-
  [ICML 2026][Time Series][Time Series QA] For time series question answering (TSQA), PATRA explicitly decomposes sequences into full/trend/season patterns at the representation level…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series QA"
  - "Pattern-Aware Alignment"
  - "GRPO"
  - "Balanced Reward"
  - "Cross-Modal Reasoning"
date: 2026-05-08
content_hash: 94393e6c7f913878
---

# PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering

**Conference**: ICML 2026  
**arXiv**: [2602.23161](https://arxiv.org/abs/2602.23161)  
**Code**: [github.com/decisionintelligence/PATRA](https://github.com/decisionintelligence/PATRA)  
**Area**: Time Series / Multimodal / Reinforcement Learning  
**Keywords**: Time Series QA, Pattern-Aware Alignment, GRPO, Balanced Reward, Cross-Modal Reasoning

## TL;DR
For time series question answering (TSQA), PATRA explicitly decomposes sequences into full/trend/season patterns at the representation level, and performs deep cross-alignment with text via three sets of learnable alignment tokens. In training, a two-stage SFT + GRPO reinforcement learning approach is used, mapping both discriminative and generative task rewards to $[0,2]$ to address difficulty imbalance, thereby comprehensively surpassing text LLMs, ChatTS, and other multimodal temporal LLMs across four TSQA tasks.

## Background & Motivation

**Background**: Time series question answering (TSQA) is currently the most prominent application form for temporal foundation models, with two mainstream approaches: (a) unimodal reasoning, directly tokenizing numerical sequences as text for LLM input (LLMTime, DeepSeek-R1 style); (b) multimodal shallow alignment, mimicking VLMs by patching sequences, projecting them, and concatenating with text embeddings (ChatTS, ITFormer style).

**Limitations of Prior Work**: The first approach ignores fundamental differences in information density and continuity between time series and text, making it difficult for models to accurately reason about structured patterns such as "trend" or "seasonality." The second approach merely performs "image-text style concatenation" without explicit multi-level dynamic modeling for time series, rendering so-called "deep alignment" superficial. A subtler issue lies in training objectives: TSQA tasks span the spectrum from binary classification to open-ended generation. Simple tasks quickly saturate rewards, while complex reasoning tasks yield sparse rewards. Naive SFT/RL leads models to overfit on easy questions, resulting in reward hacking and diminished deep reasoning ability.

**Key Challenge**: The "depth" of cross-modal alignment is capped by the patch-concat paradigm; the "balance" in multi-task training is disrupted by differences in reward scale and gradient magnitude. Neither can be solved by simply scaling up data.

**Goal**: (1) Explicitly extract different semantic levels (overall/trend/seasonality) from time series, and enable the text side to learn corresponding query representations for true "pattern-level" alignment; (2) Design a reward system insensitive to task difficulty, allowing GRPO to stably optimize across heterogeneous tasks.

**Key Insight**: The authors observe that interpretable behaviors in time series are almost always built on trend/seasonality—financial decisions and energy scheduling all rely on cyclical drawdowns—providing a natural inductive bias for "pattern decomposition and alignment." In reinforcement learning, mapping different reward distributions to a common scale alleviates multi-task optimization imbalance.

**Core Idea**: At the representation layer, use "latent space pattern decomposition + learnable alignment tokens (LAT) with multi-way cross-attention" for deep alignment; at the training layer, use "two-stage SFT→GRPO + stage-wise label reward + Rouge-L generation reward + $[0,2]$ normalization" for balanced reasoning.

## Method

### Overall Architecture
PATRA consists of a Text Encoder (using the LLM's own tokenizer and embedding), a TS Encoder (Instance Norm + Patching + Embedding), a Pattern-Aware Alignment module, and an LLM Backbone (Qwen2.5-7B). Text and time series are encoded separately and fed into the alignment module. The aligned time series tokens are inserted back into the text token sequence via `<ts>...</ts>` placeholders, then the whole sequence is passed to the LLM to generate responses in `<think>...</think><answer>...</answer>` format. Training is two-stage: first SFT on large-scale TSQA data (Alignment Stage), then GRPO + composite reward for the Reasoning-Enhanced Stage.

### Key Designs

1. **Pattern-Aware Alignment Module (PAA)**:

    - **Function**: Upgrades time series-text alignment from "shallow concatenation" to "pattern-level deep alignment," enabling the LLM to precisely reference semantic concepts like "trend" or "seasonality" during reasoning.
    - **Mechanism**: Three sub-steps. (i) **Latent Space Decomposition**: Use $X_{ts}$ as the full component $X_{ts}^f$; extract trend component $X_{ts}^t = \text{Avgpool}(\text{padding}(X_{ts}))$ via padding + average pooling (moving average filter); seasonal component is the residual $X_{ts}^s = X_{ts} - X_{ts}^t$. (ii) **Text-Side Pattern Extraction**: Define three sets of Learnable Alignment Tokens (LAT) $Q_{full}, Q_{trend}, Q_{sea}$ as queries, apply standard multi-head attention $X_k^{text} = \text{Attention}(Q_k, K, V)$ to text embeddings to obtain three sets of pattern-specialized text tokens. (iii) **Cross-Modal Interaction Alignment**: For each pair $(X_k^{text}, X_{ts}^j)$, concatenate as a new query and perform self-attention, allowing time series tokens to absorb corresponding pattern semantics from text; finally, fuse the three paths to obtain $X_{ts}^{fusion}$.
    - **Design Motivation**: (a) Decomposing in latent space rather than raw values preserves semantic information; (b) Using LAT instead of fixed prompts allows learnable pattern expression on the text side; (c) Multi-way self-attention aligns all three patterns simultaneously, avoiding "pattern information entanglement" from single global alignment.

2. **Task-Aware Balanced Reward**:

    - **Function**: Addresses optimization imbalance and reward hacking in TSQA, where "simple classification tasks are quickly exploited, while complex generation tasks yield sparse feedback."
    - **Mechanism**: Tasks are divided into two types. **Labeled tasks** (selection/judgment) use stage-wise reward $r_{label} = \sum_{k=1}^K \lambda_k r_k(\text{answer})$, where each stage verifies sub-conditions such as "within candidate range → correct option," avoiding noisy gradients from binary end-to-end rewards in early training; **Generation tasks** use Rouge-L as a continuous reward $r_{generation} = \text{TextScore}(\text{answer}, y^\star)$, encouraging sequence-level alignment rather than keyword matching. All task rewards are linearly mapped to $[0, 2]$, then format reward is added (partial reward for each valid label pair, penalty for duplicates), forming the total GRPO reward $r(\tau) = r_{format}(\tau) + r_{task}(\tau)$.
    - **Design Motivation**: (a) Stage-wise rewards provide early signals, reducing high variance; (b) Rouge-L ensures generation task rewards are dense and continuous rather than discrete; (c) $[0,2]$ normalization eliminates reward scale differences, fundamentally alleviating GRPO gradient imbalance across heterogeneous tasks.

3. **GRPO + Composite Reward Optimization Paradigm**:

    - **Function**: Stacks RL on top of SFT to further induce "<think>...</think>" reasoning chains and cross-task general reasoning ability.
    - **Mechanism**: Uses Group Relative Policy Optimization (GRPO), sampling a group of responses per prompt, replacing the value function in PPO with group-normalized advantage $\hat A_{group}(\tau) = (r(\tau) - \mu)/(\sigma + \epsilon)$, maximizing $L(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(\tau)}{\pi_{\theta_{old}}(\tau)} \hat A_{group}(\tau)\right]$; a KL term constrains deviation from the reference model. GRPO's group normalization, combined with reward normalization, provides dual stability.
    - **Design Motivation**: Avoids the overhead of training a critic network, leverages group-relative advantage to suppress reward fluctuations and preserve relative ranking, especially beneficial for TSQA's sparse positives.

### Loss & Training
The Alignment Stage uses standard cross-entropy SFT, enabling the model to "understand" decomposed time series patterns. The Reasoning-Enhanced Stage switches to GRPO, with all rewards mapped to $[0,2]$ and weighted summed as above. During inference, the model generates in `<think>...</think><answer>...</answer>` format, with the answer segment extracted by rule for evaluation. Training uses 4 A800 GPUs and Qwen2.5-7B as backbone.

## Key Experimental Results

### Main Results
The TSQA (Kong et al., 2025) dataset contains ~200k samples, 12+ domains, and four task types (Comprehension / Recognition / Reasoning / Prescience); evaluation uses Accuracy for labeled tasks and Rouge-L for generation tasks.

| Model | Comp. Acc / Rou. | Recog. Acc / Rou. | Reason. Acc / Rou. | Presc. Acc / Rou. |
|---|---|---|---|---|
| GPT-4o (Upper Bound) | 50.86 / 11.99 | 69.65 / 4.75 | 50.00 / 7.75 | 66.66 / 6.78 |
| Qwen2.5-7B | 42.24 / 18.77 | 45.51 / 10.32 | 36.48 / 18.72 | 26.85 / 10.67 |
| ChatTS-7B | 44.83 / 13.30 | 36.00 / 13.23 | 22.97 / 15.84 | 25.92 / 13.99 |
| ITFormer-7B | 40.52 / 14.24 | 45.24 / 14.61 | 30.40 / 15.58 | 44.44 / 15.25 |
| **PATRA-7B** | **56.03 / 25.67** | **64.69 / 25.46** | **44.59 / 27.36** | **52.78 / 27.06** |

PATRA achieves the highest Accuracy and Rouge-L across all four tasks (among open-source models). Recognition improves by 19.18% over the strongest text model, and Prescience by 26.86% over ChatTS, approaching GPT-4o upper bound on most metrics. Out-of-Domain experiments (removing all Weather/Finance from training) show PATRA achieves SOTA on all 6 Finance and 2 Weather MTBench metrics, including 43.70% for Finance 30-day 5-way trend prediction (vs GPT-5.2 36.05%).

### Ablation Study

| Configuration | Reason. Acc / Rou. | Presc. Acc / Rou. | Notes |
|---|---|---|---|
| Full PATRA | 44.59 / 27.36 | 52.78 / 27.06 | Full model |
| w/ Single-Pattern Alignment | 35.81 / 26.19 | 37.03 / 24.03 | Aligning only one pattern, drops 8.78 / 15.75 Acc |
| w/o Pattern-Aware Alignment | 28.37 / 16.81 | 30.55 / 16.94 | PAA module removed |
| w/o Reasoning-Enhanced Stage | 13.51 / 2.92 | 16.66 / 13.06 | SFT only, largest drop, validates RL importance |
| Original (unscaled) Reward | 37.84 / 21.64 (Reason.) | 35.18 / 16.88 (Presc.) | No $[0,2]$ normalization, Prescience drops 17.6 Acc |
| Balanced Reward | 44.59 / 27.36 | 52.78 / 27.06 | Full reward normalization |

### Key Findings
- The Reasoning-Enhanced Stage contributes most: with SFT only, Reasoning Acc is just 13.51%; adding GRPO + composite reward boosts it to 44.59%, a 31-point gain, indicating RL is key for transitioning from "answer imitation" to "reasoning chain generation."
- Pattern-Aware Alignment has a particularly significant impact on generative tasks: removing PAA drops Reasoning Rouge from 27.36 to 16.81, showing deep alignment enables generated paragraphs to truly "reference" time series patterns rather than generic restatement.
- Reward $[0,2]$ normalization raises Prescience Acc from 35.18 → 52.78, proving that cross-task reward scaling is decisive for GRPO stability; the unnormalized version biases the model toward easy tasks for high reward.
- Case analysis shows PATRA can still identify cyclical drawdowns in non-stationary, highly volatile sequences, while ChatTS / Qwen2.5 only output generic descriptions like "gradually increasing."

## Highlights & Insights
- Moving "time series decomposition" from preprocessing into embedding space, combined with learnable text-side LAT, is one of the most natural ways to embed signal processing intuition into the LLM framework—retaining trend/seasonality interpretability without extra spectral encoders.
- The seemingly simple $[0,2]$ reward normalization greatly stabilizes GRPO on heterogeneous tasks; this technique can be transferred to any "multi-task RLHF" scenario, such as code + dialogue + math synchronous RL.
- Stage-wise label reward provides "early dense signals," essentially introducing curriculum structure into the reward, with reusable value for sparse-reward RL training.
- The `<ts>...</ts>` placeholder reinsertion strategy preserves the original NL structure, avoids distribution shift from alignment tokens, and is a useful trick for handling multimodal token sequences.

## Limitations & Future Work
- Current pattern decomposition only uses trend/season (plus full, totaling three), limiting interpretability for sequences with regime changes or abrupt events (e.g., financial crashes); the authors acknowledge suboptimal decomposition on non-stationary signals in case analysis.
- The computational cost of Pattern-Aware Alignment grows rapidly with sequence length and number of LATs $T$; long-sequence inference cost is not reported.
- Evaluation is mainly on TSQA + MTBench; while cross-task generalization is shown, zero-shot robustness on real-world production data streams remains to be tested.
- The choice of $[0,2]$ normalization bounds is somewhat ad-hoc; whether reward signals for extremely difficult tasks are overly compressed is not discussed.
- GRPO uses only rule-based rewards, lacking process rewards (PRM); whether process supervision can further improve long-chain reasoning is worth exploring.

## Related Work & Insights
- **vs ChatTS / ITFormer**: These perform "shallow alignment"—patch projection + concat; PATRA explicitly decomposes patterns and aligns them multi-way, yielding a 28.69% improvement in Recognition.
- **vs Time-MQA**: Time-MQA uses a unified QA framework but remains SFT-dominated; PATRA adds RL and task balancing for more stable cross-task performance.
- **vs TimeOmni-1**: TimeOmni emphasizes interpretable reasoning chains but lacks deep alignment; PATRA explicitly encodes "patterns" into representations, achieving data-driven interpretability.
- **vs DeepSeek-R1 (on TSQA)**: Pure text RL models achieve only 12.41% Acc on Recognition, indicating that without time series-specific representations, RL advantages are hard to realize.

## Rating
- Novelty: ⭐⭐⭐⭐ Pattern-Aware Alignment embeds decomposition intuition into cross-modal attention, a clear new combination for TSQA.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 tasks + OOD MTBench + thorough ablation, with separate validation for Stage-wise / Balanced Reward.
- Writing Quality: ⭐⭐⭐⭐ Motivation—method—ablation logic is rigorous, diagrams are clear.
- Value: ⭐⭐⭐⭐ Sets a new SOTA for time series-language models; reward balancing techniques are directly transferable to other multi-task RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting](../../ICLR2026/time_series/unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)
- [\[ICML 2026\] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)
- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](../../ICLR2026/time_series/reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)

</div>

<!-- RELATED:END -->
