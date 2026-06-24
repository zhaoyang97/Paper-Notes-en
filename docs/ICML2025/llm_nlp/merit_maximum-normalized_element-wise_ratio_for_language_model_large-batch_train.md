---
title: >-
  [Paper Note] MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training
description: >-
  [ICML 2025][LLM (Other)][Large-batch training] Proposes the MERIT optimizer, which extends LAMB with maximum-norm normalization and element-wise trust ratios to effectively resolve the performance degradation caused by attention logit explosion during large-batch training.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Large-batch training"
  - "optimizer"
  - "trust ratio"
  - "maximum norm"
  - "language models"
date: 2026-05-08
content_hash: 8940fee18d3f6a8b
---

# MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training

**Conference**: ICML 2025  
**arXiv**: [2508.20577](https://arxiv.org/abs/2508.20577)  
**Code**: [GitHub](https://github.com/NUS-HPC-AI-Lab/MERIT)  
**Area**: LLM/NLP  
**Keywords**: Large-batch training, optimizer, trust ratio, maximum norm, language models

## TL;DR

Proposes the MERIT optimizer, which extends LAMB with maximum-norm normalization and element-wise trust ratios to effectively resolve the performance degradation caused by attention logit explosion during large-batch training.

## Background & Motivation

**Background**: Large-scale language model training typically relies on data parallelism for acceleration, where increasing the batch size can linearly improve throughput. AdamW is currently the most mainstream optimizer, and layer-adaptive methods like LAMB have shown the potential of large-batch training in the BERT era, but their performance on decoder-only architectures has not been fully verified.

**Limitations of Prior Work**: When the batch size is significantly increased (e.g., from 512 to 4K-8K), the validation loss of AdamW degrades significantly. This phenomenon is particularly severe on autoregressive models like GPT-2, directly restricting the linear scaling of training efficiency.

**Key Challenge**: Large-batch training requires greater stability in gradient statistics, but $L_2$ norm normalization is too "mild" when confronting extreme values—the rapid growth of the maximum logit in the attention weight matrix forms an information bottleneck (softmax degrades to one-hot), while the layer-wise trust ratio of existing optimizers fails to capture structural differences across rows and columns within the parameter matrix.

**Goal**: To design a new optimizer that maintains training stability under large-batch scenarios, suppresses attention logit explosion, while maintaining almost zero extra computational overhead.

**Key Insight**: Start by analyzing the root cause of large-batch degradation in AdamW—identifying that the sharp rise in max attention logit is the key bottleneck (whose upper bound is directly related to the max norm of $W_Q$ and $W_K$). Consequently, replace the $L_2$ norm in LAMB with the max norm, and introduce element-wise fine-grained trust ratios based on the row-column internal similarity brought by multi-head attention and outlier dimensions.

**Core Idea**: Use the maximum norm instead of the $L_2$ norm, and use element-wise trust ratios instead of layer-wise trust ratios, allowing the optimizer to precisely control the update magnitude of each parameter.

## Method

### Overall Architecture

MERIT makes three key improvements based on the LAMB optimizer. First, it computes the standard Adam first and second moment estimates to obtain the update direction $u_t$, and then computes the ratio of weight to update via the maximum norm (rather than the $L_2$ norm) as the global trust ratio $b_t$. Next, it computes local trust ratios $r_t^{(i)}$ and $c_t^{(j)}$ for each row and column of the weight matrix, respectively, and takes the maximum of these and the global ratio to construct the element-wise scaling factor $s_t^{(i,j)}$. Finally, it prevents excessively large updates of individual parameters through element-wise clipping, completing the parameter update.

### Key Designs

1. **Maximum-normalized Trust Ratio**:

    - **Function**: Provides a global parameter update scaling baseline to directly constrain the extreme Q/K weight values that cause attention logit explosion.
    - **Mechanism**: Replaces $b_t = \|w_t\|_2 / \|u_t + \lambda w_t\|_2$ in LAMB with $b_t = \|w_t\|_m / \|u_t + \lambda w_t\|_m$, where $\|\cdot\|_m$ denotes the maximum norm. Since the upper bound of the attention logit is proportional to $M_Q \cdot M_K$, and the relative difference between the $\ell_2$ norm and the max norm is as high as 99%+, $L_2$ norm scaling cannot effectively suppress extreme weight values.
    - **Design Motivation**: LAMB successfully reduces the max attention logit in shallow attention layers, but rapid growth still occurs in middle layers—analysis reveals that these layers are precisely where the gap between the max norm and the $L_2$ norm is the largest. Focusing directly on the maximum value within the parameter matrix, the maximum norm can immediately perceive and restrict abnormal growth.

2. **Element-wise Trust Ratio**:

    - **Function**: Provides independent update scaling for each element in the matrix, leveraging the similarity of weights within rows and columns.
    - **Mechanism**: Computes the row-wise ratio $r_t^{(i)} = \|w_t^{(i,\cdot)}\|_m / \|g_t^{(i,\cdot)}\|_m$ and column-wise ratio $c_t^{(j)}$ separately, and ultimately sets $s_t^{(i,j)} = \max\{\max\{r_t^{(i)}, c_t^{(j)}\}, b_t\}$. Taking the maximum ensures that each element is constrained by the more conservative limit between the row and column directions, while the global ratio serves as a lower bound to prevent overly small updates.
    - **Design Motivation**: Multi-head attention leads to weight similarities within rows, while outlier dimensions lead to weight similarities within columns. Layer-wise ratios mix all rows and columns together—extreme values in a certain row can affect the updates of other rows through the global norm, causing training instability. The element-wise mechanism isolates this cross-row/column interference.

3. **Element-wise Clipping**:

    - **Function**: Prevents individual parameter update magnitudes from exceeding a safe range.
    - **Mechanism**: $w_{t+1} = w_t - \eta_t \cdot \text{clip}(s_t \cdot (u_t + \lambda w_t), 1)$, which clips the scaled update amount to $[-1, 1]$. Analysis shows that clipping mainly occurs in the middle layers (with a peak clipping rate of 12% at the 6th layer) and rarely in shallow or deep layers.
    - **Design Motivation**: The element-wise trust ratio may produce excessively large scaling factors at certain positions; the clipping operation provides a final safeguard to ensure that the training process does not become unstable due to drastic changes in individual parameters.

### Loss & Training

MERIT uses the standard autoregressive language modeling cross-entropy loss without introducing extra regularization terms. The training strategy adopts a linear warmup + cosine annealing learning rate schedule. Weight decay is implemented via the optimizer's built-in $\lambda w_t$ term. Convergence analysis (Theorem 1) proves an $O(1/\sqrt{T})$ convergence rate for a simplified version ($\beta_1=0, \lambda=0$) under assumptions of smoothness and bounded gradients. The additional computation for the entire optimizer involves only max operations and element-wise division, with a wall-clock overhead of less than 1%.

## Key Experimental Results

### Main Results

| Model | Batch Size | AdamW | LAMB | MERIT | Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-2 Small (125M) | 1K | 3.470 | 3.355 | **3.280** | -5.5% vs AdamW |
| GPT-2 Medium (355M) | 4K | 3.172 | 3.068 | **2.982** | -6.0% vs AdamW |
| GPT-2 Large (770M) | 8K | 3.039 | 2.971 | **2.897** | -4.7% vs AdamW |
| Llama-130M | 1K | 3.277 | 3.265 | **3.199** | -2.4% vs AdamW |
| Llama-350M | 4K | — | 3.001 | **2.957** | -1.5% vs LAMB |

| Model | Method | Zero-shot Avg Accuracy | Hessian Max Eigenvalue | Hessian Trace |
|:---:|:---:|:---:|:---:|:---:|
| GPT-2 Small | AdamW | 43.56 | 37.231 | 12994.91 |
| GPT-2 Small | MERIT | **43.87** | **12.326** | **3444.92** |

### Ablation Study

| Configuration | GPT-2 Small Val Loss | Change |
|:---:|:---:|:---:|
| MERIT (Full) | **3.280** | — |
| W/o element-wise clipping | ~3.320 | +0.04 |
| W/o weight-level ratio lower bound | ~3.360-3.380 | +0.08-0.10 |
| maxLAMB (only replaced norm, no element-wise) | 3.304 | +0.024 |
| L2 norm + element-wise ratio | 3.312 | +0.032 |
| maxLAMB vs LAMB | 3.304 vs 3.355 | Limited improvement from norm selection alone |

### Key Findings

- MERIT consistently outperforms AdamW and LAMB across all model scales and batch size configurations, with validation loss improvements ranging from 2.4% to 6.0%.
- Scaling law analysis shows that MERIT has a wider "ideal scaling" window—GPT-2 Medium at a 6K batch size achieves performance on par with AdamW at a 480 batch size.
- Hessian analysis demonstrates that MERIT converges to a flatter loss landscape (67% reduction in max eigenvalue, 73% reduction in trace), explaining its superior generalization performance.
- Computational overhead is extremely low (<1% FLOPS), as the additional operations only involve simple max and element-wise computations.
- QK-Norm is actually harmful under large batch sizes—it excessively restricts the information flow capability of attention layers, whereas MERIT only constrains extreme values without restricting the overall distribution.

## Highlights & Insights

- The root-cause analysis of large-batch training failure is highly precise: rather than blaming a vague "generalization gap", it pinpointed the exact causal chain of max attention logit explosion $\to$ softmax degradation to one-hot $\to$ attention entropy collapse.
- Although the transition from $L_2$ to max norm is simple, its theoretical motivation is clear—the attention logit upper bound is directly related to the max norm (proven in Appendix D), and the 99%+ gap between the $L_2$ norm and the max norm explains why LAMB fails in the middle layers.
- The row-column decomposition of the element-wise trust ratio exploits Transformer-specific structural priors (multi-head $\to$ row similarity, outliers $\to$ column similarity), with a computational complexity of only $O(m+n)$.
- The $<1\%$ extra overhead and plug-and-play nature make it highly attractive for industrial applications.

## Limitations & Future Work

- The experimental scale is limited to 770M parameters, and its effectiveness on truly large models (7B+) is unverified—the genuine demand for large-batch training is at a much larger scale.
- The convergence analysis is based on a simplified version ($\beta_1=0, \lambda=0$), and theoretical guarantees for the complete MERIT have not been established yet.
- It only evaluates GPT-2 and Llama architectures, and its applicability to newer architectures such as MoE and GQA remains unknown.
- The improvement in zero-shot evaluation is marginal (43.56 $\to$ 43.87), necessitating more comprehensive downstream evaluations to verify its actual value.
- The interaction with mixed-precision training (FP16/BF16) is not discussed.

## Related Work & Insights

- **vs LAMB**: The $\ell_2$-norm layer-wise trust ratio of LAMB is suitable for CNNs (which feature BatchNorm and uniform weights) but unsuitable for the heterogeneous weight structure of Transformers. MERIT compensates for this structural deficiency with its max-norm + element-wise ratio.
- **vs $\sigma$-Reparam**: $\sigma$-Reparam uses spectral normalization to resolve attention entropy collapse—sharing the same goal as MERIT but requiring modifications to the model architecture (re-training), whereas MERIT only modifies the optimizer (plug-and-play).
- **vs QK-Norm**: LayerNorm normalization of Q/K can stabilize attention, but this work demonstrates it is actually harmful under large batch sizes—by excessively restricting the information flow, whereas MERIT constrains only extreme values rather than the entire distribution.

## Rating

- Novelty: ⭐⭐⭐⭐ The max attention logit diagnosis and the max-norm/element-wise ratio designs are original and practical, with a clear causal analysis chain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three scales of GPT-2 + Llama, with multiple baselines, Hessian analysis, and detailed ablation studies, though it lacks verification on >1B models.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, rich figures/tables, and a cohesive logical flow from phenomenon to method to experiments.
- Value: ⭐⭐⭐⭐⭐ $<1\%$ extra overhead, plug-and-play, and open-sourced—offering direct and significant practical value for large-batch training scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](../../ACL2025/llm_nlp/a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](../../ACL2025/llm_nlp/training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](../../ACL2025/llm_nlp/sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)
- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](../../ACL2025/llm_nlp/cheaper_and_better_diffusion_language_model_via_task-specific_training.md)
- [\[ACL 2025\] OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens](../../ACL2025/llm_nlp/olmotrace_tracing_language_model_outputs_back_to_trillions_of_training_tokens.md)

</div>

<!-- RELATED:END -->
