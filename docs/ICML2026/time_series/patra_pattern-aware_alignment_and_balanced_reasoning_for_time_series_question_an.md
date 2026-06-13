---
title: >-
  [Paper Note] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering
description: >-
  [ICML 2026][Time Series][Time Series Question Answering] For Time Series Question Answering (TSQA), PATRA explicitly decomposes sequences into three patterns—full, trend…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Question Answering"
  - "Pattern-Aware Alignment"
  - "GRPO"
  - "Balanced Reward"
  - "Cross-Modal Reasoning"
date: 2026-05-08
content_hash: e2771c6e1c1a5512
---

# PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering

**Conference**: ICML 2026  
**arXiv**: [2602.23161](https://arxiv.org/abs/2602.23161)  
**Code**: [github.com/decisionintelligence/PATRA](https://github.com/decisionintelligence/PATRA)  
**Area**: Time Series / Multimodal / Reinforcement Learning  
**Keywords**: Time Series Question Answering, Pattern-Aware Alignment, GRPO, Balanced Reward, Cross-Modal Reasoning

## TL;DR
For Time Series Question Answering (TSQA), PATRA explicitly decomposes sequences into three patterns—full, trend, and seasonal—at the representation level and performs deep cross-alignment via three sets of learnable alignment tokens. At the training stage, it utilizes a two-phase reinforcement learning approach (SFT + GRPO) that maps rewards from discriminative and generative tasks into a unified $[0,2]$ range to resolve difficulty imbalance, outperforming text-only LLMs and multimodal time series LLMs like ChatTS across four types of TSQA tasks.

## Background & Motivation

**Background**: Time Series Question Answering (TSQA) is currently one of the most prominent application forms for time series foundation models. Two mainstream approaches have emerged: (a) unimodal reasoning, which feeds raw numerical strings directly into LLMs (e.g., LLMTime or DeepSeek-R1 style); (b) multimodal shallow alignment, which mimics VLMs by patching sequences and projecting them to be concatenated with text embeddings (e.g., ChatTS, ITFormer style).

**Limitations of Prior Work**: The first approach ignores the fundamental differences in information density and continuity between time series and text, making it difficult for models to make precise judgments based on structured patterns like "trend" or "seasonality." The second approach relies on simple "image-text style" concatenation without explicit multi-layer dynamic modeling for time series, leaving "deep alignment" superficial. A more hidden bottleneck lies in the training objective: TSQA task sets span a wide spectrum from binary discrimination to open-ended generation. Simple tasks saturate rewards easily, while complex reasoning tasks offer sparse rewards. Naive SFT/RL leads the model to exploit simple tasks, resulting in reward hacking and the erosion of deep reasoning capabilities.

**Key Challenge**: The "depth" of cross-modal alignment is restricted by the ceiling of the patch-concat paradigm, while the "balance" of multi-task training is disrupted by differences in reward scales and gradient magnitudes. Neither issue can be solved by simply scaling up data.

**Goal**: (1) Explicitly extract different semantic levels of time series (full / trend / seasonal) and enable the text side to learn query representations at corresponding granularities for true "pattern-level" alignment; (2) Design a difficulty-insensitive reward system to allow stable GRPO optimization across heterogeneous tasks.

**Key Insight**: The authors observe that the "interpretable behavior" of time series is largely built upon trend and seasonality—financial decisions and energy scheduling invariably depend on periodic drawdowns. This provides a natural inductive bias for "pattern-aware decomposition and alignment." Additionally, in reinforcement learning, mapping different reward distributions to a common scale can alleviate optimization imbalance in multi-task scenarios.

**Core Idea**: At the representation level, use "latent space pattern decomposition + multi-channel cross-attention with Learnable Alignment Tokens (LAT)" for deep alignment; at the training level, implement "two-stage SFT→GRPO + stage-wise label reward + Rouge-L generation reward + $[0,2]$ normalization" for balanced reasoning.

## Method

### Overall Architecture
PATRA consists of a Text Encoder (utilizing the LLM's own tokenizer and embedding), a TS Encoder (Instance Norm + Patching + Embedding), a Pattern-Aware Alignment (PAA) module, and an LLM Backbone (Qwen2.5-7B). Text and time series are encoded separately and fed into the alignment module. The aligned time series tokens are backfilled into the text token sequence via `<ts>...</ts>` placeholders, then sent to the LLM to generate responses in the form of `<think>...</think><answer>...</answer>`. Training is divided into two phases: SFT on large-scale TSQA data (Alignment Stage), followed by GRPO with composite rewards (Reasoning-Enhanced Stage).

### Key Designs

1.  **Pattern-Aware Alignment (PAA)**:
    -   **Function**: Upgrades time series-text alignment from "shallow concatenation" to "pattern-level deep alignment," allowing the LLM to precisely reference semantic concepts like "trend" or "seasonality" during reasoning.
    -   **Mechanism**: Consists of three sub-steps: (i) **Latent Space Decomposition**: $X_{ts}$ is treated as the full component $X_{ts}^f$; trend components $X_{ts}^t = \text{Avgpool}(\text{padding}(X_{ts}))$ are extracted via moving average filtering; seasonal components are given by the residual $X_{ts}^s = X_{ts} - X_{ts}^t$. (ii) **Textual Pattern Extraction**: Three sets of Learnable Alignment Tokens (LAT) $Q_{full}, Q_{trend}, Q_{sea}$ are defined as queries to perform standard multi-head attention on text embeddings $X_k^{text} = \text{Attention}(Q_k, K, V)$, yielding three pattern-specific text tokens. (iii) **Cross-modal Interaction Alignment**: Each pair $(X_k^{text}, X_{ts}^j)$ is concatenated into a new query for self-attention, allowing time series tokens to absorb textual semantics under the corresponding pattern. Finally, the three channels are fused into $X_{ts}^{fusion}$.
    -   **Design Motivation**: (a) Decomposing in latent space rather than raw value levels preserves semantic information; (b) Using LAT instead of fixed prompts allows textual pattern expressions to be learnable; (c) Multi-channel self-attention avoids pattern information entanglement caused by a single global alignment.

2.  **Task-Aware Balanced Reward**:
    -   **Function**: Addresses optimization imbalance and reward hacking caused by "fast scoring on simple discriminative tasks vs. sparse feedback on complex generative tasks" in TSQA.
    -   **Mechanism**: Tasks are split into two categories. **Labeled tasks** (selection / judgment) use stage-wise rewards $r_{label} = \sum_{k=1}^K \lambda_k r_k(\text{answer})$, where each stage verifies sub-conditions like "is it within candidate range → is the option correct," avoiding the high gradient noise of binary end-to-end rewards. **Generative tasks** use Rouge-L as a continuous reward $r_{generation} = \text{TextScore}(\text{answer}, y^\star)$, encouraging sequence-level alignment rather than keyword matching. All task rewards are linearly mapped to the $[0, 2]$ interval, then stacked with format rewards (partial rewards for valid tag pairs, penalties for repetitions) to form the total reward $r(\tau) = r_{format}(\tau) + r_{task}(\tau)$ for GRPO.
    -   **Design Motivation**: (a) Stage-wise rewards provide early signals to the model, reducing variance; (b) Rouge-L ensures generative rewards are continuous and dense; (c) $[0,2]$ normalization eliminates scale differences, fundamentally mitigating gradient imbalance during GRPO on heterogeneous tasks.

3.  **GRPO + Composite Reward Optimization**:
    -   **Function**: Layers RL on top of SFT to elicit the Chain-of-Thought (CoT) structure of "<think>...</think>" and cross-task general reasoning capabilities.
    -   **Mechanism**: Uses Group Relative Policy Optimization (GRPO), sampling a group of responses for each prompt. Group-normalized advantages $\hat A_{group}(\tau) = (r(\tau) - \mu)/(\sigma + \epsilon)$ replace the value function from PPO to maximize $L(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(\tau)}{\pi_{\theta_{old}}(\tau)} \hat A_{group}(\tau)\right]$, with a KL constraint to prevent divergence from the reference model. GRPO's group-wise normalization and the previous reward normalization provide dual stability.
    -   **Design Motivation**: Avoids the extra overhead of training a critic network and leverages relative advantages to suppress reward fluctuations and preserve ranking, which is particularly beneficial for sparse positive examples in TSQA.

### Loss & Training
The Alignment Stage uses standard cross-entropy SFT to teach the model to "understand" decomposed time series patterns. The Reasoning-Enhanced Stage switches to GRPO, where all rewards are weighted and summed after being mapped to $[0,2]$. During inference, the model generates according to the `<think>...</think><answer>...</answer>` structure; the answer segment is extracted via rules for evaluation. Training utilizes 4 A800 GPUs with Qwen2.5-7B as the backbone.

## Key Experimental Results

### Main Results
The TSQA (Kong et al., 2025) dataset comprises ~200k samples across 12+ domains and four task types (Comprehension / Recognition / Reasoning / Prescience). Evaluation metrics include Accuracy for labeled tasks and Rouge-L for generative tasks.

| Model | Comp. Acc / Rou. | Recog. Acc / Rou. | Reason. Acc / Rou. | Presc. Acc / Rou. |
|---|---|---|---|---|
| GPT-4o (Upper Bound) | 50.86 / 11.99 | 69.65 / 4.75 | 50.00 / 7.75 | 66.66 / 6.78 |
| Qwen2.5-7B | 42.24 / 18.77 | 45.51 / 10.32 | 36.48 / 18.72 | 26.85 / 10.67 |
| ChatTS-7B | 44.83 / 13.30 | 36.00 / 13.23 | 22.97 / 15.84 | 25.92 / 13.99 |
| ITFormer-7B | 40.52 / 14.24 | 45.24 / 14.61 | 30.40 / 15.58 | 44.44 / 15.25 |
| **Ours (PATRA-7B)** | **56.03 / 25.67** | **64.69 / 25.46** | **44.59 / 27.36** | **52.78 / 27.06** |

PATRA achieves SOTA among open-source models in both Accuracy and Rouge-L across all four tasks. Recognition improved by +19.18% over the strongest text model, and Prescience improved by +26.86% over ChatTS, approaching GPT-4o on most metrics. Out-of-Domain (OOD) experiments (excluding Weather/Finance from training) show PATRA achieves SOTA on all MTBench Finance and Weather metrics, including 43.70% on 30-day 5-way trend prediction (vs. GPT-4o's 36.05%).

### Ablation Study

| Configuration | Reason. Acc / Rou. | Presc. Acc / Rou. | Description |
|---|---|---|---|
| Full PATRA | 44.59 / 27.36 | 52.78 / 27.06 | Complete model |
| w/ Single-Pattern Alignment | 35.81 / 26.19 | 37.03 / 24.03 | Single pattern alignment, loss of 8.78 / 15.75 Acc |
| w/o Pattern-Aware Alignment | 28.37 / 16.81 | 30.55 / 16.94 | Fully removal of PAA module |
| w/o Reasoning-Enhanced Stage | 13.51 / 2.92 | 16.66 / 13.06 | SFT only, largest drop, proves RL criticality |
| Original (unscaled) Reward | 37.84 / 21.64 (Reason.) | 35.18 / 16.88 (Presc.) | No $[0,2]$ normalization, Prescience drops 17.6 Acc |
| Balanced Reward | 44.59 / 27.36 | 52.78 / 27.06 | Full reward normalization |

### Key Findings
- **Reasoning-Enhanced Stage contributes most**: SFT alone yields only 13.51% Reasoning Acc; adding GRPO + composite rewards boosts it to 44.59% (+31 points), showing RL is the transition from "mimicking answers" to "producing reasoning chains."
- **Pattern-Aware Alignment impact is significant on generation**: Removing PAA causes Reasoning Rouge to drop from 27.36 to 16.81, indicating deep alignment allows the model to "reference" time series patterns rather than repeating generic phrases.
- **Reward $[0,2]$ normalization is decisive**: It improves Prescience Acc from 35.18 to 52.78, proving cross-task reward scaling is critical for GRPO stability; unnormalized versions bias the model toward simple tasks for high rewards.
- **Case studies** show PATRA identifies periodic drawdowns in non-stationary, highly volatile sequences, whereas ChatTS/Qwen2.5 only output generic descriptions like "gradually increasing."

## Highlights & Insights
- Implementing "time series decomposition" within the embedding space with learnable text-side LAT is a natural way to embed signal processing intuition into LLM frameworks—retaining the interpretability of trend/seasonality without requiring extra spectral encoders.
- Reward $[0,2]$ normalization, though simple, contributes significantly to GRPO stability across heterogeneous tasks; this technique can be transferred to any "multi-task RLHF" scenario, such as synchronized RL for code, dialogue, and mathematics.
- Stage-wise label rewards provide "early dense signals," essentially incorporating curriculum structures into the reward, which is valuable for training sparse-reward RL.
- The `<ts>...</ts>` placeholder backfilling strategy preserves original natural language structures and avoids distribution shifts introduced by alignment tokens—a useful trick for handling multimodal sequences.

## Limitations & Future Work
- Current pattern decomposition is limited to trend/season (plus full, totaling three), which provides limited explanatory power for sequences with regime changes or abrupt events (e.g., financial crashes); authors acknowledge poor decomposition on non-stationary signals.
- The computational complexity of Pattern-Aware Alignment grows with sequence length and the number of LAT tokens $T$; inference costs for long sequences were not disclosed.
- Evaluation is primarily on TSQA + MTBench; while cross-task generalization was shown, zero-shot robustness on real-world production data streams remains to be investigated.
- The choice of boundary values for $[0,2]$ normalization is somewhat ad-hoc; whether reward signals for extremely difficult tasks are overly compressed is not discussed.
- GRPO uses only rule-based rewards without Process Reward Models (PRM); whether long-chain reasoning can be further improved via process supervision is an area for exploration.

## Related Work & Insights
- **vs. ChatTS / ITFormer**: They perform "shallow alignment" via patch projection and concatenation; PATRA explicitly decomposes patterns and multi-way aligns them, yielding a 28.69% improvement in Recognition.
- **vs. Time-MQA**: While Time-MQA uses a unified QA framework, it is SFT-dominated; PATRA adds RL and task balancing for steadier cross-task performance.
- **vs. TimeOmni-1**: TimeOmni emphasizes interpretable reasoning chains but lacks deep alignment; PATRA embeds "patterns" explicitly into representations, making interpretability more data-driven.
- **vs. DeepSeek-R1 (on TSQA)**: Pure text RL models achieve only 12.41% Acc on Recognition, indicating that without specialized time series representations, RL advantages are difficult to unleash.

## Rating
- Novelty: ⭐⭐⭐⭐ Pattern-Aware Alignment embeds decomposition intuition into cross-modal attention, a clear new combination for TSQA.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 tasks + OOD MTBench + comprehensive ablation; stage-wise and balanced rewards are validated independently.
- Writing Quality: ⭐⭐⭐⭐ Tight logic from motivation to method to ablation; clear diagrams.
- Value: ⭐⭐⭐⭐ Sets a new SOTA for time series-language models; reward balancing techniques are directly applicable to other multi-task RL scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning](../../ACL2026/time_series/odtqa-fore_an_open-domain_tabular_question_answering_dataset_for_future_data_for.md)
- [\[ICML 2026\] Interpretability in Deep Time Series Models Demands Semantic Alignment](interpretability_in_deep_time_series_models_demands_semantic_alignment.md)
- [\[ICML 2026\] Adaptive Time Series Reasoning via Segment Selection](adaptive_time_series_reasoning_via_segment_selection.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)
- [\[ICLR 2026\] VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting](../../ICLR2026/time_series/unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)

</div>

<!-- RELATED:END -->
