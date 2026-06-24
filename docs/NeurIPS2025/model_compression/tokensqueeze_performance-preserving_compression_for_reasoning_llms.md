---
title: >-
  [Paper Note] TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs
description: >-
  [NeurIPS 2025][Model Compression][Reasoning Compression] TokenSqueeze proposes a three-stage pipeline — adaptive reasoning depth selection, intra-step linguistic refinement (with KL divergence constraints), and length-aware preference optimization — achieving 50% token compression of reasoning chains without accuracy degradation, using only self-generated data.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Reasoning Compression"
  - "Chain-of-Thought"
  - "Preference Learning"
  - "Long2Short"
  - "Large Language Models"
date: 2026-05-08
content_hash: 2a09d7dfa17994ed
---

# TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2511.13223](https://arxiv.org/abs/2511.13223)  
**Code**: [GitHub](https://github.com/zhangyx1122/TokenSqueeze)  
**Area**: Model Compression
**Keywords**: Reasoning Compression, Chain-of-Thought, Preference Learning, Long2Short, Large Language Models

## TL;DR

TokenSqueeze proposes a three-stage pipeline — adaptive reasoning depth selection, intra-step linguistic refinement (with KL divergence constraints), and length-aware preference optimization — achieving 50% token compression of reasoning chains without accuracy degradation, using only self-generated data.

## Background & Motivation

Reasoning-oriented LLMs represented by OpenAI-o1 and DeepSeek-R1 have achieved breakthrough performance on complex reasoning tasks by generating long chain-of-thought (CoT) outputs. However, long CoTs introduce increased inference latency, higher memory consumption, and an "overthinking" phenomenon, where models generate excessive redundant reasoning steps even for simple problems.

Existing Long2Short methods face a fundamental **reasoning oversimplification dilemma**:

- **Inference-time compression methods** (prompt shortening, modified decoding strategies) have limited effect since the underlying model remains unchanged.
- **Training-time methods** (incorporating length penalties into RL reward/objective functions, e.g., Kimi-k1.5, L1, O1-Pruner) can shorten outputs but often compress away critical reasoning steps, causing significant accuracy drops.
- **Data-driven methods** (SFT/DPO on the shortest correct responses) overly aggressively reduce reasoning depth.

The paper's central argument is that **concise and efficient short answers are fundamentally a matter of stylistic preference**. Experiments show that beyond a certain length threshold, the correlation between token count and model performance weakens significantly. Long2Short can therefore be framed as a preference learning task — teaching the model to respond concisely while maintaining adaptive reasoning depth calibrated to problem complexity.

## Method

### Overall Architecture

TokenSqueeze is a three-stage training-time preference learning method: (1) adaptive reasoning depth selection — choosing reasoning chains of appropriate length as positive samples based on problem difficulty; (2) intra-step linguistic refinement — rewriting reasoning steps under KL divergence constraints to increase information density; (3) composite optimization objective — training with a combination of SFT loss and length-aware DPO loss. The entire pipeline relies exclusively on self-generated data, requiring no external teacher models or human annotations.

### Key Designs

1. **Adaptive Reasoning Depth Selection**: Rather than simply selecting the shortest correct response, the method employs a dynamic quantile mechanism to adaptively adjust selection thresholds based on problem difficulty. The pass rate is defined as $p = c/N$ ($c$ correct out of $N$ total responses), with the adaptive quantile $q = \alpha \cdot (1-p)$. Correct reasoning chains are sorted by length, and the top $k = \lceil q \cdot c \rceil$ are selected as positive samples. Key advantage: for easy problems (high pass rate), shorter chains are preferred; for difficult problems (low pass rate), longer chains are retained to capture critical logical steps.

2. **Intra-Step Linguistic Refinement (KL Divergence Constraint)**: Each reasoning step $s_i$ is independently compressed at the linguistic level. Given the preceding context $\mathcal{A}_{<i}$, $K=64$ candidate rewrites $\{s_i^{(k)}\}$ are sampled, and the shortest candidate satisfying the KL divergence constraint is selected:
$$\min_{s_i' \in \{s_i^{(k)}\}} \ell(s_i') \quad \text{s.t.} \quad D_{KL}(P_\theta(\cdot|p, s_{\leq i}) \| P_\theta(\cdot|p, s_{<i}, s_i')) < \varepsilon$$
A local token window ($L=512$) is used to approximate the full distributional KL divergence. The core idea is to shorten the linguistic expression of each reasoning step without altering the semantics of the downstream reasoning trajectory. The threshold $\varepsilon$ controls the trade-off between semantic fidelity and conciseness.

3. **Length-Aware Preference Optimization (DPO-L)**: An adaptive length margin is introduced on top of standard DPO to explicitly encourage concise reasoning:
$$\mathcal{L}_{\text{DPO-L}} = -\mathbb{E}\left[\log \sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) + \lambda \log\frac{\ell(y_l)}{\ell(y_w)}\right)\right]$$
The length ratio $\log(\ell(y_l)/\ell(y_w))$ adaptively adjusts the margin: greater compression gains yield stronger preference signals.

### Loss & Training

The final composite objective combines SFT loss (to prevent reward collapse) and DPO-L loss:
$$\mathcal{L}_{\text{Total}} = \eta \mathcal{L}_{\text{DPO-L}} + (1-\eta) \mathcal{L}_{\text{SFT}}$$
where $\eta = 0.5$. Training configuration: learning rate $5 \times 10^{-6}$, batch size 128, Adam optimizer, 8×A100 GPUs. At most $M=64$ preference pairs per positive sample are used to maintain data diversity.

## Key Experimental Results

### Main Results — DeepSeek-R1-Distill-Qwen-7B

| Dataset | Metric | Baseline | Kimi-k1.5 | DAST | TokenSqueeze | Change |
|--------|------|----------|-----------|------|-------------|------|
| AIME24 | Acc (%) | 55.5 | 51.2 | 53.3 | **57.5** | +2.0 |
| AIME24 | Len-T | 7543 | 5249 | 6339 | **5157** | -31.6% |
| AIME24 | AUC (%) | 41.6 | 41.8 | — | **48.5** | +6.9 |
| MATH500 | Acc (%) | 92.8 | 88.2 | 92.6 | **92.4** | -0.4 |
| MATH500 | Len-T | 3638 | 1698 | 2802 | **1773** | -51.3% |
| MATH500 | AUC (%) | 83.6 | 83.7 | — | **87.5** | +3.9 |
| LiveCodeBench | Acc (%) | 31.3 | 24.8 | 29.7 | **35.0** | +3.7 |
| LiveCodeBench | Len-A | 20690 | 19242 | — | **15635** | -24.4% |

### Ablation Study — Component Contributions

| Method | AIME24 Acc | AIME24 Len | MATH500 Acc | MATH500 Len | Note |
|------|-----------|-----------|-------------|-------------|------|
| Shortest | 53.3 | 5960 | 90.8 | 1926 | Select shortest correct response |
| Q-FIX | 55.0 | 6126 | 92.2 | 2054 | Fixed quantile |
| Q-DYN (w/ extra pos) | 52.3 | 5666 | 90.8 | 1742 | Extra positives as negatives |
| **Q-DYN** | **57.3** | **6190** | **92.8** | **2180** | Adaptive quantile (Ours) |

| Objective | AIME24 Acc | AIME24 Len | MATH500 Acc | MATH500 Len | Note |
|---------|-----------|-----------|-------------|-------------|------|
| DPO | 48.3 | 4300 | 91.6 | 1974 | Strong compression but accuracy drops |
| SFT | 56.0 | 5734 | 91.8 | 2271 | Accurate but insufficient compression |
| DPO+SFT | 57.0 | 5420 | 92.6 | 1865 | Better balance |
| **TokenSqueeze** | **57.5** | **5157** | **92.4** | **1773** | Optimal balance |

### Key Findings

- Achieves **50% token compression** on MATH500 while preserving accuracy (92.4% vs. 92.8%).
- Advantages are more pronounced under limited token budgets: +15.5 pp over baseline on AIME24 at 3K tokens, +43.1 pp on MATH500 at 1K tokens.
- Adaptive depth selection (Q-DYN) significantly outperforms simple shortest-selection (+4.0 pp on AIME24), demonstrating the importance of preserving appropriate reasoning depth.
- Intra-step refinement reduces average step length from 29.1 to 26.3 tokens, providing additional compression without reducing the number of reasoning steps.
- Both GPT-4o-mini rewriting and TokenSkip lead to significant accuracy degradation, validating the necessity of KL-constrained refinement.
- Pure DPO training causes a severe accuracy drop (48.3%), underscoring the indispensable stabilizing role of the SFT loss.

## Highlights & Insights

- **Precise methodological framing**: Redefining Long2Short as a stylistic preference problem rather than a reasoning capability problem avoids compressing critical reasoning steps.
- **Fully self-generated data**: Requires no external teacher models or human annotations, relying solely on the model's own outputs, making it applicable across diverse domains.
- **KL divergence-constrained refinement**: Optimizing information density at the step level ensures semantic integrity while achieving effective compression.
- The multi-dimensional ablation design (data construction, refinement method, optimization objective) is logically comprehensive.

## Limitations & Future Work

- The KL threshold $\varepsilon$ is set heuristically (currently fixed at 0.005), lacking an adaptive mechanism.
- The fully offline preference optimization prevents the model from continuously refining its strategy at inference time.
- Intra-step refinement requires sampling 64 candidates per step, incurring substantial computational cost during data construction.
- Validation is limited to mathematical reasoning and coding tasks; generalization to broader reasoning domains (e.g., scientific, commonsense reasoning) remains to be explored.
- Compression gains are notably stronger for the 7B model than the 1.5B model; applicability to smaller models warrants further investigation.

## Related Work & Insights

Compared with Kimi-k1.5 (DPO), Sky-T1-Flash, DAST, L1, and other methods, TokenSqueeze achieves the best accuracy–efficiency trade-off. The adaptive depth selection mechanism is generalizable to other generation tasks requiring "appropriate complexity." The intra-step KL-constrained refinement framework is also applicable to other text compression scenarios such as summarization and translation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of adaptive depth selection and KL-constrained refinement is novel, with precise problem framing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four benchmarks, two model scales, and multiple ablation dimensions yield a highly comprehensive analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed method descriptions.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses a core efficiency bottleneck in deploying reasoning LLMs; 50% compression without accuracy loss offers significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)
- [\[ICML 2026\] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](../../ICML2026/model_compression/semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)
- [\[ACL 2025\] Compression in Transformer Language Models Has a Surprising Relationship with Performance](../../ACL2025/model_compression/compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](../../CVPR2025/model_compression/mambaic_state_space_models_for_high-performance_learned_image_compression.md)

</div>

<!-- RELATED:END -->
