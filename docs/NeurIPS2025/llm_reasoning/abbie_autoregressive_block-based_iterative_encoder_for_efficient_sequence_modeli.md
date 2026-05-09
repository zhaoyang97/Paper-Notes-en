---
title: >-
  [Paper Note] AbbIE: Autoregressive Block-Based Iterative Encoder for Efficient Sequence Modeling
description: >-
  [NeurIPS 2025][LLM Reasoning][recurrent Transformer] This paper proposes AbbIE, an architecture that recursively iterates the intermediate layers (Body) of a decoder-only Transformer. Trained with only 2 iterations, AbbIE achieves upward generalization at inference time by increasing the number of iterations, surpassing standard Transformers on both language modeling perplexity and zero-shot ICL benchmarks, while serving as a drop-in replacement for standard Transformers.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - recurrent Transformer
  - iterative encoder
  - test-time scaling
  - fixed point
  - upward generalization
date: 2026-05-08
content_hash: 297702ed39a48185
---

# AbbIE: Autoregressive Block-Based Iterative Encoder for Efficient Sequence Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2507.08567](https://arxiv.org/abs/2507.08567)
**Code**: None
**Area**: LLM Inference
**Keywords**: recurrent Transformer, iterative encoder, test-time scaling, fixed point, upward generalization

## TL;DR
This paper proposes AbbIE, an architecture that recursively iterates the intermediate layers (Body) of a decoder-only Transformer. Trained with only 2 iterations, AbbIE achieves upward generalization at inference time by increasing the number of iterations, surpassing standard Transformers on both language modeling perplexity and zero-shot ICL benchmarks, while serving as a drop-in replacement for standard Transformers.

## Background & Motivation

**State of the Field**: Transformer performance has traditionally been improved by scaling model parameters and training data (scaling laws). Test-time scaling has recently emerged as a new direction, but existing recurrent Transformers (e.g., Geiping et al. 2025) require training with many iterations and are typically limited to specific tasks.

**Limitations of Prior Work**: (1) GPU memory growth lags behind compute growth, constraining model scale expansion; (2) existing recurrent Transformers incur high training costs (requiring many iterations) and cannot serve as general-purpose replacements for standard Transformers; (3) most recurrent methods fail to generalize beyond the number of training iterations at inference time (upward generalization failure).

**Root Cause**: How can Transformers be endowed with test-time compute scaling capability without substantially increasing training cost?

**Paper Goals**: Design a recurrent Transformer such that: (a) it is equivalent to a standard Transformer at a single iteration; (b) it requires only 2 training iterations; (c) it can scale to an arbitrary number of inference iterations with continuously improving performance.

**Starting Point**: The authors observe that the residual stream of a Transformer naturally injects original input information into every layer, which may be sufficient to achieve Path Independence (convergence to a fixed point), enabling recursive iteration without additional projection matrices.

**Core Idea**: The Transformer is partitioned into Head–Body–Tail segments; only the Body is iteratively applied. An inter-iteration residual connection ensures convergence, and only 2 training iterations are needed to achieve upward generalization at inference time.

## Method

### Overall Architecture
Input tokens are mapped via embedding and the Head ($N_h$ Transformer blocks) into concept space. The Body ($N_b$ blocks) is then applied recursively for $r$ iterations, after which the Tail ($N_t$ blocks) maps the representations back to token space for unembedding. At $r=1$, AbbIE is exactly equivalent to a standard Transformer.

### Key Designs

1. **Head–Body–Tail Partition**:

    - **Function**: The Transformer layers are divided into three groups — the Head handles tokenization to concept space, the Body performs iterative reasoning, and the Tail maps from concept space back to token space.
    - **Mechanism**: The Head and Tail are each applied once; the Body is applied $r$ times. This partition is motivated by the concept space theory of Kaplan et al. 2024, which posits that the early layers perform de-tokenization, the late layers perform re-tokenization, and the intermediate layers operate in concept space.
    - **Design Motivation**: Applying recursion to the entire model would corrupt the tokenization process; iteration should only occur at the appropriate level of abstraction.

2. **AbbIE-D (Diffusion-inspired variant)**:

    - **Function**: Adds an inter-iteration residual connection between consecutive Body iterations.
    - **Mechanism**: $h_{k+1} = B(h_k) + h_k$, where $B(\cdot)$ denotes the Body component. In contrast to AbbIE-C, which uses $h_{k+1} = B(h_k)$ (relying solely on intra-Body residuals), AbbIE-D increases the relative contribution of the original input signal, preventing the $h_0$ signal from being diluted across iterations.
    - **Design Motivation**: Achieving Path Independence (fixed-point convergence) requires a sufficiently strong original input signal at each iteration. Experiments confirm that AbbIE-C diverges while AbbIE-D converges.

3. **Only 2 Training Iterations**:

    - **Function**: Training uses only $r=2$ (Body applied twice), yet inference can employ $r=4, 8$, or more iterations.
    - **Mechanism**: Because AbbIE-D satisfies the fixed-point property, 2 training iterations are sufficient for the model to learn how to exploit additional iterations for improved representations. Larger models (350M) achieve the lowest perplexity at $r=4$, demonstrating successful upward generalization.
    - **Design Motivation**: Reduces training cost to near that of a standard Transformer while retaining test-time scaling capability.

### Loss & Training
Standard next-token prediction (NLL). AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.95$) with a Warmup-Stable-Decay learning rate schedule. Training token budget is 20 tokens per parameter (compute-optimal). All models use tied embeddings.

## Key Experimental Results

### Main Results

| Benchmark | Metric | AbbIE-D (r=8) | AbbIE-D (r=2) | Std | 0pt (r=2) |
|-----------|--------|---------------|---------------|-----|-----------|
| HellaSwag (350M) | Acc | **36.6** | 33.8 | 30.1 | 29.7 |
| LAMBADA (350M) | Acc | **30.8** | 29.8 | 24.2 | 22.2 |
| ARC-Easy (350M) | Acc | **53.2** | 48.9 | 45.6 | 46.3 |
| CommonsenseQA (350M) | Acc | **23.7** | 20.0 | 20.0 | 20.0 |

Note: On CommonsenseQA, both the standard Transformer and 0pt remain at the random baseline (20%), while AbbIE-D at $r=8$ exceeds the random baseline, suggesting that iterative computation unlocks emergent reasoning capability.

### Ablation Study

| Configuration | Fixed-Point Convergence? | Upward Generalization? | Notes |
|---------------|--------------------------|------------------------|-------|
| AbbIE-D | Converges | Yes (350M at r=4) | Inter-iteration residual ensures convergence |
| AbbIE-C | Diverges | No | Intra-Body residuals alone are insufficient |
| 0pt (Geiping et al.) | Converges | No (perplexity collapses at r≠2) | Converges but fails to generalize beyond training iterations |

### Key Findings
- **AbbIE-D is the only general-purpose recurrent Transformer that achieves upward generalization with only 2 training iterations**, with ICL performance continuing to improve up to 4× the training iteration count.
- **Perplexity is approximately 5% lower than standard Transformers**, following the same scaling law.
- **FLOP efficiency improves over the course of training**: although AbbIE-D incurs slightly higher training FLOPs, the gap narrows in longer training runs.
- **Key finding**: Even when perplexity slightly increases at $r=8$, ICL task performance continues to improve — indicating that the relationship between perplexity and downstream task performance is not strictly monotonic.

## Highlights & Insights
- **Equivalence to a standard Transformer at $r=1$** is a highly desirable engineering property: models can first be trained in the standard manner and iterative inference can be enabled on demand, with zero adoption risk.
- **The 2-iteration training design** elegantly balances training cost against inference capability. The contrast with 0pt — which requires many training iterations yet still fails to generalize — demonstrates that the key factor is architectural design (inter-iteration residual) rather than the training recipe.
- **The concept space theoretical framework** provides a principled justification for the Head–Body–Tail partition and points toward future work on adaptive selection of partition boundaries.

## Limitations & Future Work
- **Validation is limited to 350M models**: whether the approach remains effective at 1B+ scale is unclear. The authors note that upward generalization is weaker for 200M models than for 350M, suggesting a critical model size threshold.
- **Inference latency scales linearly**: $r$ iterations imply an $r$-fold increase in inference latency (even though parameter count does not increase), which is unfavorable for latency-sensitive applications.
- **ICL gains are moderate**: the largest improvement is approximately 12% (HellaSwag), with absolute performance remaining on par with standard models of equivalent scale.
- **Generation tasks are not evaluated**: all evaluations are zero-shot ICL; generation tasks such as translation and summarization are not assessed.

## Related Work & Insights
- **vs. Coconut (latent reasoning)**: Both perform iteration in latent space, but Coconut requires specialized datasets and training pipelines, whereas AbbIE does not.
- **vs. 0pt (Geiping et al. 2025)**: Both are recurrent Transformers, but 0pt employs input concatenation with projection, while AbbIE-D uses residual connections. AbbIE incurs lower training cost (2 vs. multiple iterations) and achieves upward generalization.
- **vs. MoE**: MoE reduces inference cost via sparse activation; AbbIE reduces memory cost via parameter sharing. The two approaches are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The Head–Body–Tail partition combined with inter-iteration residuals is concise and effective, though the recurrent Transformer direction has accumulated considerable prior work.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation is limited to 350M models; generation task assessment is absent.
- Writing Quality: ⭐⭐⭐⭐ Logical structure is clear; theoretical (Path Independence) and empirical contributions are well integrated.
- Value: ⭐⭐⭐⭐ Proposes a practical recurrent Transformer alternative; the drop-in property is particularly attractive.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion](two-stage_learning_of_stabilizing_neural_controllers_via_zubov_sampling_and_iter.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)

<!-- RELATED:END -->
