---
title: >-
  [Paper Note] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
description: >-
  [Reasoning] Analyzes RLVR from a novel perspective of token entropy patterns, revealing that only about 20% of high-entropy "forking tokens" in CoT reasoning dictate the reasoning direction. Applying gradient updates solely to these tokens matches or significantly outperforms full updates (+11.04 on AIME'25 with Qwen3-32B), demonstrating that the essence of RLVR is optimizing reasoning decision points.
tags:
  - "Reasoning"
date: 2026-05-08
content_hash: 1c3258bbfa3730c6
---

# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

## General Information
- **arXiv**: 2506.01939
- **Conference**: NeurIPS 2025
- **Authors**: Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, et al.
- **Institutions**: Tsinghua University, Alibaba
- **Code**: Not open-sourced

## TL;DR
Analyzes RLVR from a novel perspective of token entropy patterns, revealing that only about 20% of high-entropy "forking tokens" in CoT reasoning dictate the reasoning direction. Applying gradient updates solely to these tokens matches or significantly outperforms full updates (+11.04 on AIME'25 with Qwen3-32B), demonstrating that the essence of RLVR is optimizing reasoning decision points.

## Background & Motivation
RLVR (Reinforcement Learning with Verifiable Rewards, such as DeepSeek-R1's GRPO) has been shown to significantly enhance LLM reasoning capabilities, but its underlying mechanism remains unclear:
- What exactly does RL change in the model?
- Are all tokens equally important, or do specific key tokens play a dominant role?
- Can this understanding be leveraged to improve RLVR?

## Core Problem
What exactly is RLVR optimizing at the token level? Can we optimize only the most critical tokens to improve both efficiency and performance?

## Method

### 1. Token Entropy Pattern Analysis
Computes the policy entropy $H(p_\theta(\cdot | x, y_{<t}))$ for each token position during the CoT reasoning process:
- **Key Finding 1**: The vast majority of tokens (~80%) exhibit extremely low entropy (the model is highly confident in how to proceed).
- **Key Finding 2**: Only a minority of tokens (~20%) have high entropy—these are **"forking tokens"**, where the model faces choices regarding the reasoning direction.
- **Key Finding 3**: High-entropy tokens correspond to decision points in the reasoning chain (e.g., choosing a problem-solving strategy, determining the argumentative direction).

### 2. RLVR Training Dynamics Analysis
Observes changes in token entropy during the RLVR training process:
- RLVR **largely maintains** the entropy distribution pattern of the base model.
- It primarily adjusts the **entropy values of high-entropy tokens** (reducing entropy towards the correct reasoning direction).
- Low-entropy tokens are almost unaffected.
- This suggests that the essential role of RLVR is to guide the model to make better choices at reasoning "forks".

### 3. Forking Token Gradient (Core Method)
Based on the above insights, this work proposes computing gradient updates solely on high-entropy forking tokens:
- For each generated token, compute the policy entropy.
- Select the top-20% tokens with the highest entropy.
- **Apply the policy gradient only on these tokens**.
- The remaining 80% of tokens do not participate in gradient computation.

### 4. Surprising Experimental Results
- **20% token updates ≈ 100% token updates** (comparable performance on Qwen3-8B).
- **20% token updates >> 100% token updates**:
    - Qwen3-32B: AIME'25 +11.04, AIME'24 +7.71
    - Qwen3-14B: AIME'25 +4.79, AIME'24 +5.21
- **Counter-proof**: Training only on the 80% low-entropy tokens leads to a significant performance drop.
- Demonstrates a strong **scaling trend**—the larger the model, the more pronounced the advantages of forking token optimization.

## Key Experimental Results

### AIME Benchmark

| Model | Method | AIME'25 | AIME'24 |
|---|---|---|---|
| Qwen3-32B | Full Gradient RLVR | baseline | baseline |
| Qwen3-32B | **Forking token (20%)** | **+11.04** | **+7.71** |
| Qwen3-14B | Full Gradient RLVR | baseline | baseline |
| Qwen3-14B | **Forking token (20%)** | **+4.79** | **+5.21** |

### Token Type Ablation

| Trained Tokens | Ratio | Performance |
|---|---|---|
| Full (100%) | 100% | baseline |
| High-entropy top-20% | 20% | $\ge$ baseline (outperforms significantly on large models) |
| Low-entropy bottom-80% | 80% | **Significant drop** |

## Highlights & Insights
1. **Novel Perspective**: Understanding RLVR from the perspective of token entropy patterns is clean and profound.
2. **Counter-intuitive 80/20 Rule**: 20% of tokens are sufficient (and even better) to drive RL training.
3. **Strong Scaling Trend**: The larger the model, the greater the advantage, suggesting that forking token optimization is a promising direction towards AGI reasoning.
4. **Practical Value**: Reduces gradient updates by 80%, significantly lowering RLVR training costs.
5. **Theoretical Insight**: The essence of RLVR is optimizing reasoning decision points rather than completely rewriting generation patterns.

## Limitations & Future Work
1. The threshold selection for high-entropy tokens (top-20%) is empirical.
2. Analysis is primarily based on mathematical reasoning tasks; generalizability to code/natural language reasoning remains to be verified.
3. Entropy computation require additional forward passes, introducing some overhead.
4. It depends on the quality of the token entropy distribution of the base model.

## Related Work & Insights
- **vs. Does Thinking More Help? (previously analyzed)**: The latter found that overthinking increases variance, whereas this work reveals that the key to RLVR is optimizing high-entropy (high-variance) tokens. The two are complementary, jointly demonstrating that variance/entropy is a core signal in reasoning.
- **vs. DAPO/GRPO**: Standard RLVR uniformly updates all tokens, while the forking token method is more efficient and precise.
- **vs. Token-level reward methods**: Process Reward Models (PRMs) aim to provide feedback at the step level, whereas this work directly identifies key positions at the token level.
- **vs. ThinkPrune/Thinkless**: These methods reduce useless thinking tokens, while this work identifies key tokens from the RL training side.

## Inspirations & Connections
- **Connection with Overthinking Research**: High-entropy tokens represent "crossroads" in reasoning. Overthinking might introduce too many branches at these points, suggesting that combining the two approaches could be more effective.
- **Prospects of Sparse RL Updates**: If only gradients of 20% of the tokens are needed, the computational and memory costs of RLVR can be drastically reduced.
- **Implications for RL Scaling**: When scaling up RL, the focus should be on how to leverage the forking token signal on larger models rather than simply increasing the sample size.

## Rating
- Novelty: ★★★★★ — Understanding RLVR from the token entropy perspective provides a pioneering insight.
- Technical Depth: ★★★★☆ — Deep analysis with a simple yet effective method.
- Experimental Thoroughness: ★★★★★ — 3 model scales $\times$ multiple benchmarks $\times$ ablations $\times$ scaling analysis.
- Writing Quality: ★★★★★ — The narrative progresses step-by-step, with observations, explanations, and methods closely linked.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[ICML 2025\] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL](../../ICML2025/llm_reasoning/one_missing_piece_for_open-source_reasoning_models_a_dataset_to_mitigate_cold-st.md)
- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[ICLR 2026\] Learning to Reason over Continuous Tokens with Reinforcement Learning (HyRea)](../../ICLR2026/llm_reasoning/learning_to_reason_over_continuous_tokens_with_reinforcement_learning.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
