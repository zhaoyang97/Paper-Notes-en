---
title: >-
  [Paper Note] CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration
description: >-
  [ACL 2026][Model Compression][Diffusion Language Models] CadLLM is proposed as a training-free adaptive inference acceleration method. It leverages token decoding confidence signals from diffusion language models (dLLMs)…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Diffusion Language Models"
  - "Inference Acceleration"
  - "Adaptive Decoding"
  - "Confidence Calibration"
  - "Training-Free methods"
date: 2026-05-08
content_hash: a94c96a15ed58f7b
---

# CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration

**Conference**: ACL 2026 Findings  
**arXiv**: [2512.07173](https://arxiv.org/abs/2512.07173)  
**Code**: Available  
**Area**: Model Compression  
**Keywords**: Diffusion Language Models, Inference Acceleration, Adaptive Decoding, Confidence Calibration, Training-Free methods

## TL;DR
CadLLM is proposed as a training-free adaptive inference acceleration method. It leverages token decoding confidence signals from diffusion language models (dLLMs) to dynamically adjust four dimensions—block size, step count, vocabulary sampling range, and submission threshold—achieving 1.1–2.28× throughput improvements on LLaDA and DREAM while maintaining competitive accuracy.

## Background & Motivation

**Background**: Masked diffusion language models (e.g., LLaDA, DREAM) generate text through an iterative denoising Markov process that refines noisy states, demonstrating strong generative capabilities. fast-dLLM introduced parallel decoding acceleration based on static confidence thresholds.

**Limitations of Prior Work**: fast-dLLM employs fixed block sizes, fixed step counts, and a fixed sampling width, ignoring the dynamic shifts in confidence across sequences and steps. Specifically: (1) fixed block sizes disregard difficulty variance across regions; (2) uniform sampling widths ignore differences in determinism; (3) fixed submission thresholds fail to adapt to confidence fluctuations during different inference stages.

**Key Challenge**: Static scheduling strategies lead to over-refinement of easy blocks (wasting computation) and under-refinement of difficult blocks (damaging quality). Computational resources must be adaptively allocated based on confidence signals.

**Goal**: Design a training-free, model-agnostic, plug-and-play method that utilizes confidence signals to adaptively control multiple resource dimensions in dLLM inference.

**Key Insight**: Analysis of confidence dynamics across different blocks and steps reveals significant variation—confidence within a block rises rapidly before plateauing, and difficulty varies greatly between different blocks.

**Core Idea**: Token decoding confidence is used as a single shared signal to drive four closed-loop control strategies (block size, step count, vocabulary size, and threshold). This allocates computational resources where uncertainty persists and conserves resources where predictions are stable.

## Method

### Overall Architecture
After each forward pass of the dLLM, CadLLM utilizes token confidence as a feedback signal. Through four linear interpolation strategies, it dynamically updates the block size $B_t$, step count $S_t$, vocabulary size $V_t$, and threshold $\tau_t$, forming a closed-loop controller. It is a plug-and-play approach compatible with KV-cached dLLMs.

### Key Designs

1.  **Adaptive Block Size ($B_t$):**
    - **Function**: Dynamically adjusts the number of tokens in parallel decoding based on confidence.
    - **Mechanism**: $B_t = \text{clip}(B_{\min} + (B_{\max} - B_{\min}) \cdot \bar{c}, B_{\min}, B_{\max})$, where $\bar{c}$ is the average confidence within a sliding window ($\Delta=2$). Block size is increased during high confidence to amortize forward pass costs and decreased during low confidence to focus refinement.
    - **Design Motivation**: Figure 1(a) shows that difficulty varies significantly between blocks, and fixed block sizes lead to uneven resource allocation.

2.  **Adaptive Steps ($S_t$) + Adaptive Threshold ($\tau_t$):**
    - **Function**: Step count controls refinement depth within each block; the threshold controls the aggressiveness of token submission.
    - **Mechanism**: Step count is complementary to confidence: $S_t = \text{clip}(S_{\text{base}} + (S_{\max} - S_{\text{base}})(1 - \bar{c}), S_{\text{base}}, S_{\max})$. The threshold relaxes as generation progresses: $\tau_t = \tau_{\text{base}}(1-g_t) + \tau_{\min} g_t$. Low confidence triggers more refinement steps and stricter submission gates.
    - **Design Motivation**: Ablation studies show the adaptive threshold is the most critical component for efficiency (throughput drops by 71.6% without it).

3.  **Adaptive Vocabulary Size ($V_t$):**
    - **Function**: Dynamically adjusts the subset size for softmax calculation to reduce computational overhead.
    - **Mechanism**: $V_t = \text{clip}(V_{\text{phase}}(g_t) \cdot f_{\text{conf}}(\bar{c}) \cdot f_{\text{rep}}(r_t), V_{\min}, V_{\max})$. The vocabulary is expanded during early generation or low confidence to increase robustness and narrowed during high confidence to save softmax costs. A repetition detector is included to prevent degradation caused by excessively narrow vocabularies.
    - **Design Motivation**: Figure 1(b) shows that softmax latency grows sharply with vocabulary size; the full ~50K vocabulary is nearly an order of magnitude slower than a small subset.

### Loss & Training
Ours is entirely training-free. All strategies are implemented via linear interpolation and clipping, introducing no additional computation during inference.

## Key Experimental Results

### Main Results
Results on LLaDA-Instruct (single H100):

| Benchmark | CadLLM Accuracy | CadLLM Throughput Gain | Fast-dLLM Accuracy | Gen Length |
|-----------|-----------------|------------------------|--------------------|------------|
| GSM8K     | 78.01%          | 1.33×                  | 79.00%             | 256        |
| MATH      | 32.06%          | 1.34×                  | 32.40%             | 256        |
| HumanEval | 35.97%          | **2.28×**              | 37.19%             | 256        |
| HumanEval | 43.29%          | 1.74×                  | 45.12%             | 512        |

### Ablation Study

| Configuration | Token/s | Accuracy | Description |
|---------------|---------|----------|-------------|
| All ON        | 121.72  | 78.01%   | Full model  |
| No $V_t$      | 119.67  | 74.41%   | Accuracy drops 4.6% |
| No $S_t$      | 136.76  | 76.12%   | Faster but lower accuracy |
| No $B_t$      | 111.19  | 78.32%   | Throughput drops 8.6% |
| No $\tau_t$   | 34.57   | 78.17%   | **Throughput crashes 71.6%** |
| All OFF       | 34.32   | 78.01%   | No adaptivity |

### Key Findings
- The adaptive threshold is the absolute core of efficiency: removing it causes throughput to plummet from 121.72 to 34.57 token/s, and NFE increases by 289%.
- Acceleration is most significant on HumanEval (2.28×) due to greater confidence variance in code generation.
- The method remains effective on DREAM (1.1–1.4× gain), validating its model-agnostic nature.
- Adaptive vocabulary size does not significantly impact speed but severely affects accuracy (-4.6%), highlighting the importance of sampling width control.

## Highlights & Insights
- Designing a closed-loop controller that drives four resource dimensions with a single confidence signal is elegant; this "minimal adaptivity" significantly outperforms static scheduling.
- The repetition detector (preventing degenerate loops from vocabulary narrowing) is a practical engineering detail that avoids common pitfalls in fast decoding.
- The choice of linear interpolation strategies is intentional—proving that even the simplest monotonic mappings provide significant benefits establishes a lower bound for more complex strategies.

## Limitations & Future Work
- Validated only on LLaDA and DREAM; performance on larger-scale models remains unknown.
- Hyperparameters ($B_{\min}, B_{\max}, S_{\max}$, etc.) require manual setting, although sensitivity analysis within ±20% shows stability.
- Accuracy is comparable to the baseline but not entirely lossless; there is a 1-2% loss on HumanEval.
- Future work could explore non-linear control strategies or reinforcement learning to optimize strategy parameters.

## Related Work & Insights
- **vs fast-dLLM**: Static thresholds and fixed block sizes are the root of uneven resource allocation; CadLLM resolves this with adaptive strategies.
- **vs Autoregressive Acceleration (Speculative Decoding, etc.)**: dLLMs naturally support parallel decoding; CadLLM optimizes the degree of parallelism on this basis.
- **vs Lu et al. (Concurrent work)**: They also found that fixed block sizes lead to premature submission of low-confidence tokens, aligning with the motivation of this paper.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified design of four-dimensional adaptive control is new for dLLM acceleration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablations and validation across multiple tasks and lengths.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation analysis and precise methodology description.
- Value: ⭐⭐⭐⭐ Direct practical value for dLLM inference deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BaseCal: Unsupervised Confidence Calibration via Base Model Signals](basecal_unsupervised_confidence_calibration_via_base_model_signals.md)
- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[ACL 2026\] VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection](veccisc_improving_confidence-informed_self-consistency_with_reasoning_trace_clus.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](../../NeurIPS2025/model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
