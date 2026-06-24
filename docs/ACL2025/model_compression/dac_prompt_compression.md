---
title: >-
  [Paper Note] DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression
description: >-
  [ACL 2025][Model Compression][prompt compression] DAC proposes a dynamic attention-aware prompt compression method. By integrating information entropy and attention scores as token importance metrics, and dynamically perceiving the entropy shift during the compression process for fine-grained compression, it improves the average score by 1.33 points over SOTA methods on LongBench.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "prompt compression"
  - "attention score"
  - "information entropy"
  - "task-agnostic"
  - "long context"
date: 2026-05-08
content_hash: 07f3327adf357a36
---

# DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression

**Conference**: ACL 2025  
**arXiv**: [2507.11942](https://arxiv.org/abs/2507.11942)  
**Code**: [https://github.com/QQQ-yi/DAC](https://github.com/QQQ-yi/DAC)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: prompt compression, attention score, information entropy, task-agnostic, long context

## TL;DR
DAC proposes a dynamic attention-aware prompt compression method. By integrating information entropy and attention scores as token importance metrics, and dynamically perceiving the entropy shift during the compression process for fine-grained compression, it improves the average score by 1.33 points over SOTA methods on LongBench.

## Background & Motivation

**Background**: Prompt compression reduces the computational and memory overhead of LLMs by decreasing the number of input tokens. Task-agnostic prompt compression leverages linguistic redundancy to remove low-information tokens (e.g., LLMLingua uses information entropy as a metric). Task-aware methods (e.g., LLMLingua2) require training an additional classification model and offer limited generalization.

**Limitations of Prior Work**: (1) Existing entropy-based methods overlook attention-critical tokens, which do not necessarily exhibit high entropy but are crucial for model understanding (the Pearson correlation between information entropy and attention scores is only 0.095). (2) Entropy shift occurs during compression: once predecessor tokens are removed, the information entropy of subsequent tokens shifts significantly, a dynamic change that static methods fail to capture.

**Key Challenge**: Assessing token importance solely based on information entropy is insufficient, leading to the erroneous removal of attention-critical tokens; furthermore, prompt compression is inherently a dynamic process but is treated as static by existing methods.

**Goal**: To more accurately identify low-information tokens and address entropy shifts during the compression process.

**Key Insight**: Integrating both information entropy and attention scores as dual signals, and introducing an iterative dynamic update mechanism.

**Core Idea**: Combining attention scores and information entropy as a token importance metric, and achieving more precise compression by iteratively and dynamically updating the entropy of affected tokens.

## Method

### Overall Architecture
Input prompt → Compute the information entropy and cumulative attention score for each token → Integrate the two metrics to obtain a comprehensive importance score → Rank tokens by score and remove low-importance tokens → Dynamically update the entropy of tokens affected by the removal → Iterate until the target compression ratio is reached.

### Key Designs

1. **Attention-aware Metric**:

    - Function: Integrates information entropy and attention scores as token importance metrics.
    - Two fusion methods: Additive fusion $M_t^a = (1-\alpha) \cdot I_t(x) + \alpha \cdot \overline{s_t}$; Multiplicative fusion $M_t^m = I_t(x) \cdot \overline{s_t}$.
    - Attention score calculation: Averages the cumulative attention scores across all heads and all layers.
    - Design Motivation: Experiments reveal that removing attention-critical tokens degrades performance to a level even worse than random compression, demonstrating that entropy alone is insufficient.

2. **Dynamic Entropy Shift Detection**:

    - Function: Iteratively updates the information entropy of affected tokens during the compression process.
    - Core Observation: When the predecessor tokens of a certain token are removed, its information entropy may change significantly—originally low-entropy tokens might exhibit high entropy.
    - Mechanism: After each compression round, identify which tokens have had their predecessors removed, recompute their information entropy, and update their importance scores.
    - Design Motivation: Under high compression ratios ($\tau \uparrow$), entropy shifts become more severe (Pearson correlation decreases). Thus, the dynamic approach is particularly crucial at high compression ratios.

3. **Iterative Compression Process**:

    - Divides the compression process into multiple rounds.
    - Removes a portion of low-importance tokens in each round.
    - Updates the entropy and importance scores of the affected tokens.
    - Repeats the process until the target compression ratio is achieved.

## Key Experimental Results

### Main Results: LongBench (Average over 16 tasks)

| Method | Compression Ratio 0.5 | Compression Ratio 0.3 | Compression Ratio 0.2 |
|------|----------|----------|----------|
| Uncompressed | Baseline | Baseline | Baseline |
| LLMLingua | Lower | Obvious drop | Significant drop |
| LLMLingua2 | Better | Prev. SOTA | - |
| **DAC** | **+4.03 vs LLMLingua** | **+1.33 vs SOTA** | **Best** |

### Ablation Study

| Configuration | LongBench F1 | Description |
|------|------------|------|
| Information Entropy Only | Baseline | Similar to LLMLingua |
| Attention Score Only | Below baseline | Attention score alone is insufficient |
| Additive Fusion (Static) | Obvious improvement | Fusion is effective |
| Additive Fusion (Dynamic) | **Best** | Dynamic update provides further improvement |
| Multiplicative Fusion | Close to additive | Both fusion methods perform comparably |
| w/o Attention-Critical Tokens | Below random | Proves the importance of attention-critical tokens |

### Key Findings
- **Attention-critical tokens are not equivalent to high-entropy tokens**: The Pearson correlation is only 0.095. Many attention-critical tokens have low entropy, causing entropy-only methods to erroneously remove them.
- **Removing attention-critical tokens yields worse performance than random compression**: This provides strong evidence that attention scores contain crucial information beyond entropy.
- **Dynamic updates are more critical under high compression ratios**: As the compression ratio goes from 0.9 to 0.5, the Pearson correlation between the original and post-compression entropy drops significantly.
- **Cross-model generalization**: Consistent performance is demonstrated across Qwen2 and LLaMA3 series.
- **Task-agnostic**: Effective across multiple tasks including QA, summarization, and code completion.

## Highlights & Insights
- **Validation of Intuition**: Validates the intuition that "tokens favored by the attention mechanism must be retained" through rigorous experiments, quantifying the consequences of ignoring them (which even perform worse than random compression).
- **Practical Utility of Dynamic Perception**: Compression is a sequence-dependent process where removing one token influences the information content of subsequent tokens. Though intuitive, this insight was neglected by all prior methods.
- **Training-free Method**: Requires no additional training of a compression model (unlike LLMLingua2, which relies on GPT-4 distilled data for training), offering better generalizability.
- **Applicable to Black-box LLMs**: The compressed prompt remains a natural language hard-prompt, making it suitable for API-only models.

## Limitations & Future Work
- Requires access to internal LLM attention scores, rendering it inapplicable to fully black-box (API-only) scenarios.
- Dynamic iterative updates increase the computational overhead of the compression process itself.
- Attention scores are averaged across all layers and heads, ignoring the potentially differing importance of different layers (unweighted).
- Lacks comparison with soft-prompt methods (such as GIST tokens, ICAE).

## Related Work & Insights
- **vs LLMLingua (Jiang et al., 2023)**: LLMLingua relies solely on information entropy as its metric, whereas DAC introduces attention scores and dynamic updating, achieving a gain of 4.03 points.
- **vs LLMLingua2 (Pan et al., 2024)**: LLMLingua2 trains a task-specific classification model for compression, but its generalization is restricted by training data. DAC is training-free.
- **vs H2O (Zhang et al., 2023)**: H2O utilizes cumulative attention scores for KV cache eviction, which inspired DAC to introduce attention scores into prompt compression.

## Rating
- Novelty: ⭐⭐⭐⭐ The philosophy of integrating attention scores with dynamic entropy updates is novel, and both empirical observations are compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough evaluations spanning multiple datasets, models, and tasks, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic flowing from observations to methods and then validation.
- Value: ⭐⭐⭐⭐ Highly practical value for prompt compression in long-context scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] 500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](../../ICCV2025/model_compression/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)
- [\[NeurIPS 2025\] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](../../NeurIPS2025/model_compression/kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)
- [\[CVPR 2025\] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](../../CVPR2025/model_compression/tadformer_task-adaptive_dynamic_transformer_for_efficient_multi-task_learning.md)
- [\[ACL 2025\] A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](gist_token_context_compression.md)

</div>

<!-- RELATED:END -->
