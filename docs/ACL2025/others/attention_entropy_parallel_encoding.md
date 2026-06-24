---
title: >-
  [Paper Note] Attention Entropy is a Key Factor for Parallel Context Encoding
description: >-
  [ACL 2025] This paper discovers that parallel context encoding leads to an abnormal increase in the attention entropy of query tokens, which is a key factor for performance degradation. Two training-free methods, Shared Attention Sink and Selective Attention, are proposed to effectively mitigate this issue.
tags:
  - "ACL 2025"
date: 2026-05-08
content_hash: 4c8a53c063ad4f48
---

tags:
  - ACL 2025
  - Others
date: 2026-05-08
content_hash: a5fa7cbe5248ec1b
---
# Attention Entropy is a Key Factor for Parallel Context Encoding

**Conference**: ACL 2025  
**arXiv**: [2412.16545](https://arxiv.org/abs/2412.16545)  
**Area**: Others  

## TL;DR

This paper discovers that parallel context encoding leads to an abnormal increase in the attention entropy of query tokens, which is a key factor for performance degradation. Two training-free methods, Shared Attention Sink and Selective Attention, are proposed to effectively mitigate this issue.

---

## Background & Motivation

Mainstream LLMs employ decoder-only Transformers with full self-attention to model context. While powerful, this approach has two main limitations:

**Computational Complexity**: Full attention requires computing $\frac{1}{2}N(N+1)$ token pairs for a sequence length of $N$, exhibiting quadratic growth.

**Neglecting Input Structure**: In scenarios such as RAG (Retrieval-Augmented Generation) and ICL (In-Context Learning), the context is naturally composed of independent sub-segments, which full attention fails to exploit.

**Parallel Context Encoding** is a natural alternative: it splits the context into $P$ sub-segments to be encoded independently, reducing the computational cost to approximately $\frac{1}{P}$. However, because model pre-training never encounters parallel context patterns, direct application leads to severe performance degradation. Prior works only explored this in limited scenarios (specific window sizes or tasks), lacking a systematic analysis and a general mitigation strategy.

**Core Problem**: Why does parallel encoding fail? Can the performance degradation be mitigated without fine-tuning?

---

## Method

### Overall Architecture

The analysis and methodology framework in this paper consist of three levels:

1. **Diagnosis**: Systematically compare full attention vs. parallel encoding across four tasks—LM, ICL, RAG, and synthetic retrieval—revealing that larger parallelism $P$ leads to worse performance.
2. **Attribution**: Analyze attention entropy to reveal a strong correlation (Pearson $R \approx 0.95$) and trace it back to abnormalities in key state norms.
3. **Mitigation**: Propose two training-free methods to reduce attention entropy: Shared Attention Sink and Selective Attention.

### Key Designs

#### Module 1: Attention Entropy as an Anomaly Indicator

Attention entropy is defined as $H(p) = -\sum_i p_i \cdot \log p_i$, where $p$ represents the attention weight distribution. Key findings:

- Parallel encoding causes the attention entropy of query tokens to be **abnormally high**, indicating dispersed attention and increased uncertainty.
- Root cause: Each sub-segment is encoded independently in parallel encoding, leading to smaller key state norms (especially for the sink tokens of each segment). Consequently, the absolute values of attention logits shrink, prompting the softmax outputs to approach a uniform distribution.
- Attention entropy displays a strong linear correlation with model performance ($R \approx 0.95$), serving as a reliable indicator of performance degradation.

#### Module 2: Shared Attention Sink

Attention Sink refers to the phenomenon where initial tokens absorb a massive amount of attention weights. In parallel encoding, each sub-segment has its own sink token. The model, never exposed to multiple sink patterns during training, produces anomalous hidden states.

**Solution**: Add a shared prefix before all sub-segments so they share the same set of sink tokens. Key findings:

- The specific content of the prefix does not matter; even a newline character is effective. Its primary function is to absorb excessive attention values.
- The experiments employ a simple instruction as the prefix (e.g., "Given the following partial context, predict the next sequence of words:").
- This approach avoids extreme anomalous tokens in each segment, raises logit values, and reduces attention entropy.

#### Module 3: Selective Attention

This module directly sharpens the attention distribution using a hard selection mechanism:

1. **Grouping**: Group tokens by parallel sub-segments, and compute a group score $s_{group}$ for each group as the sum of the top-5 attention probability values.
2. **Selection**: Perform top-$K$ selection over the $P$ groups (default $K=2$), retaining only the $K$ highest-scoring sub-segments.
3. **Masking & Renormalization**: Set attention weights of unselected sub-segments to zero, and renormalize the remaining weights.

Key design choices:
- **Aggregation Dimension**: Group scores can be aggregated along token, head, or layer dimensions. No aggregation is used for LM tasks (which require diverse information), while ICL, RAG, and synthetic tasks default to Head+Token aggregation.
- **Layer Aggregation** requires a two-pass forward run (as attention scores of later layers are unavailable when computing earlier layers), whereas other dimensions can be processed in a single pass.
- The choice of $K$ varies by task: synthetic retrieval prefers a small $K$, while ICL/RAG favor a larger $K$.

### Loss & Training

This work **does not require any fine-tuning or training**. All proposed methods are inference-time modifications:

- Attention Sink: Modify the attention mask to allow sub-segments to share prefix tokens.
- Selective Attention: Introduce grouping selection and renormalization steps after softmax outputs.
- The two methods can be used independently or combined (Sink+SEL). The combined scheme offers the most balanced performance.

---

## Key Experimental Results

### Main Results: Full Attention vs. Naive Parallel Encoding (Llama-3.1-8B, 8K)

| Method | LM (PPL↓) | ICL (Acc↑) | RAG (SubEM↑) | Synthetic (SubEM↑) |
|------|-----------|------------|--------------|---------------------|
| Full Attention | 5.35 | 66.00 | 60.25 | 99.50 |
| P=2 | 5.66 | 63.60 | 59.50 | 94.81 |
| P=4 | 6.16 | 57.20 | 50.75 | 81.56 |
| P=8 | 6.92 | 44.40 | 48.75 | 41.00 |
| P=16 | 7.97 | 34.00 | 41.75 | 2.19 |
| P=32 | 9.24 | 17.40 | 39.25 | 0.00 |
| P=64 | 10.46 | 10.80 | 33.00 | 0.00 |

> Performance degrades drastically as parallelism increases; performance on the synthetic retrieval task drops from 99.5% to 0%.

### Ablation Study: Parameter Selection for Selective Attention (P=64, 8K)

| Settings | ICL (Acc↑) | RAG (SubEM↑) | Synthetic (SubEM↑) |
|------|------------|--------------|---------------------|
| TopK=1 | 26.00 | 45.75 | **21.56** |
| TopK=2 | 33.00 | 48.50 | **24.88** |
| TopK=5 | **36.00** | **48.75** | 14.69 |
| TopK=10 | 28.60 | 44.50 | 5.25 |
| No Aggr. | 35.40 | 42.75 | 17.62 |
| Aggr.=T | 36.20 | 45.00 | 21.00 |
| Aggr.=HT | 36.00 | 48.75 | 24.88 |
| Aggr.=LHT | 22.40 | **49.50** | 17.31 |

> Optimal configurations differ across tasks: synthetic retrieval prefers a small $K$ (precise retrieval), while ICL/RAG favor a larger $K$ (multi-information aggregation). RAG performs best when using aggregation across all dimensions (LHT, equivalent to retrieval).

### Key Findings

1. **Strong Correlation between Attention Entropy and Performance**: With Pearson $R \approx 0.95$, attention entropy serves as a reliable indicator of parallel encoding performance degradation.
2. **The Two Methods are Complementary**:
    - Sink is more suitable for ICL (which requires more illustration information and is unsuitable for hard selection).
    - SEL is more suitable for RAG and synthetic retrieval (which are naturally retrieval-centric and benefit from precise selection).
    - The combined Sink+SEL scheme shows the most balanced performance across all tasks.
3. **Anomaly Rooted in Hidden State Scale**: Parallel encoding reduces the norm of key states (particularly for the sink tokens of each segment), resulting in smaller logit values and pushing the softmax towards a uniform distribution (increasing entropy).
4. **Importance of Value States**: Even using oracle key states (keys encoded by full attention), performance does not consistently surpass the proposed method, indicating that abnormalities in value states are also significant.
5. **Cross-Model Consistency**: Mistral-7B-v0.3 and Qwen2-7B display identical trends.

---

## Highlights & Insights

- **Impressive Analytical Depth**: The paper follows a complete logical chain from the phenomenon (performance degradation) $\to$ indicator (attention entropy) $\to$ root cause (anomalies in key state norms) $\to$ mitigation (reducing entropy).
- **Practical Value of Sink Prefix**: Content-independence means this method is extremely simple and can achieve improvements at almost zero cost.
- **Flexibility of Selective Attention**: Adjusting $K$ and aggregation dimensions allows adaptation to different task properties, yielding a configurable inference-time optimization framework.
- **Implications for RAG/Long-Context Systems**: In practical deployments, document-level encoding can be accelerated via parallel encoding, with Sink+SEL used to mitigate quality loss, balancing efficiency and quality.

---

## Limitations & Future Work

1. **No Fine-tuning Introduced**: Pure inference-time methods have an upper bound. Lightweight fine-tuning (e.g., LoRA adaptation for parallel modes) could significantly boost effects.
2. **Lack of Wall-Clock Efficiency Measurement**: Although the theoretical computation is reduced to $1/P$, actual wall-clock time or GPU memory comparisons are not provided. Practical acceleration requires kernel-level implementation.
3. **No Universal Optimal Configuration**: Different tasks require different $K$ values and aggregation strategies, lacking an adaptive selection mechanism.
4. **Limited to RoPE Models**: Experiments cover three RoPE-based families (Llama, Mistral, Qwen). The applicability to other positional encodings (such as ALiBi) is not validated.
5. **Insufficient Analysis of Value States**: Although value states are found to have an impact, their anomaly mechanism and mitigation strategies are not analyzed in depth.

---

## Related Work & Insights

- **Parallel Context Window** (Ratner et al., 2023): First to propose parallel encoding windows to extend LLM contexts. This paper provides a more comprehensive analysis on top of it.
- **Structured Prompting** (Hao et al., 2022): Extends ICL to thousands of examples, demonstrating the feasibility of parallel encoding in ICL.
- **Attention Sink** (Xiao et al., 2024; StreamingLLM): Discovers the phenomenon where initial tokens absorb massive attention. This paper generalizes it from length extrapolation to parallel context encoding.
- **Retrieval Head** (Wu et al., 2024): Identifies specialized information retrieval attention heads. The head aggregation in Selective Attention is built upon this assumption.
- **KV Cache Compression** (H2O, SnapKV, etc.): Shares the idea of "selectively retaining critical context" with Selective Attention, though this work operates at the block level rather than the token level.

---

## Rating

- Novelty: ⭐⭐⭐⭐ | First to systematically reveal the causal relationship between attention entropy and parallel encoding performance.
- Utility: ⭐⭐⭐⭐ | Sink+SEL is simple yet effective, providing direct guidelines for RAG/long-context deployment.
- Evidence: ⭐⭐⭐⭐ | Comprehensive evaluation across four task types, three models, and multiple parallel degrees, with thorough ablations.
- Writing: ⭐⭐⭐⭐⭐ | Clear and logical analysis, progressing smoothly from diagnosis to mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [\[ACL 2025\] Entropy-UID: A Method for Optimizing Information Density](entropy-uid_a_method_for_optimizing_information_density.md)
- [\[ACL 2025\] The Time Scale of Redundancy between Prosody and Linguistic Context](the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)
- [\[ACL 2025\] Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)

</div>

<!-- RELATED:END -->
