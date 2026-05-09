---
title: >-
  [Paper Note] CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration
description: >-
  [ACL 2026][Model Compression][Diffusion language models] This paper proposes CadLLM, a training-free adaptive inference acceleration method that leverages token decoding confidence signals in diffusion large language models (dLLMs) to dynamically adjust four dimensions—block size, number of steps, vocabulary sampling range, and commitment threshold—achieving 1.1–2.28× throughput improvements on LLaDA and DREAM while maintaining competitive accuracy.
tags:
  - ACL 2026
  - Model Compression
  - Diffusion language models
  - inference acceleration
  - adaptive decoding
  - confidence calibration
  - training-free methods
date: 2026-05-08
content_hash: a0610ffbc3f3d71b
---

# CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration

**Conference**: ACL 2026
**arXiv**: [2512.07173](https://arxiv.org/abs/2512.07173)
**Code**: Available
**Area**: Model Compression
**Keywords**: Diffusion language models, inference acceleration, adaptive decoding, confidence calibration, training-free methods

## TL;DR
This paper proposes CadLLM, a training-free adaptive inference acceleration method that leverages token decoding confidence signals in diffusion large language models (dLLMs) to dynamically adjust four dimensions—block size, number of steps, vocabulary sampling range, and commitment threshold—achieving 1.1–2.28× throughput improvements on LLaDA and DREAM while maintaining competitive accuracy.

## Background & Motivation

**Background**: Masked diffusion language models (e.g., LLaDA, DREAM) generate text by iteratively refining noisy states through a multi-step denoising Markov process, demonstrating strong generative capabilities. fast-dLLM proposes parallel decoding acceleration based on static confidence thresholds.

**Limitations of Prior Work**: fast-dLLM employs fixed block sizes, fixed step counts, and fixed sampling widths, ignoring the dynamic variation of confidence across sequences and steps. Specifically: (1) fixed block sizes disregard difficulty differences across regions; (2) uniform sampling widths ignore variation in prediction certainty; (3) fixed commitment thresholds do not adapt to confidence changes across different inference stages.

**Key Challenge**: Static scheduling strategies over-refine easy blocks (wasting computation) and under-refine difficult blocks (degrading quality)—necessitating adaptive allocation of computational resources based on confidence signals.

**Goal**: To design a training-free, model-agnostic, plug-and-play method that leverages confidence signals to adaptively control multiple resource dimensions during dLLM inference.

**Key Insight**: Through analysis of confidence dynamics across different blocks and steps, the authors find that confidence varies significantly—confidence within a block rises rapidly before plateauing, and difficulty varies substantially across blocks.

**Core Idea**: Token decoding confidence serves as a single shared signal driving four closed-loop control strategies (block size, step count, vocabulary size, and commitment threshold), allocating computation where uncertainty persists and conserving resources where predictions are stable.

## Method

### Overall Architecture
After each forward pass in a dLLM, CadLLM uses token confidence as a feedback signal to dynamically update block size $B_t$, step count $S_t$, vocabulary size $V_t$, and threshold $\tau_t$ via four linear interpolation strategies, forming a closed-loop controller. It is plug-and-play and compatible with KV-cache-enabled dLLMs.

### Key Designs

1. **Adaptive Block Size ($B_t$)**:

    - **Function**: Dynamically adjusts the number of tokens decoded in parallel based on confidence.
    - **Mechanism**: $B_t = \text{clip}(B_{\min} + (B_{\max} - B_{\min}) \cdot \bar{c}, B_{\min}, B_{\max})$, where $\bar{c}$ is the average confidence within a sliding window ($\Delta=2$). Block size is enlarged at high confidence to amortize forward pass cost, and reduced at low confidence to concentrate refinement.
    - **Design Motivation**: Figure 1(a) shows that difficulty varies greatly across blocks, and fixed block sizes lead to uneven resource allocation.

2. **Adaptive Step Count ($S_t$) + Adaptive Threshold ($\tau_t$)**:

    - **Function**: Step count controls refinement depth within each block; threshold controls the aggressiveness of token commitment.
    - **Mechanism**: Step count is inversely complementary to confidence: $S_t = \text{clip}(S_{\text{base}} + (S_{\max} - S_{\text{base}})(1 - \bar{c}), S_{\text{base}}, S_{\max})$; the threshold relaxes with generation progress: $\tau_t = \tau_{\text{base}}(1-g_t) + \tau_{\min} g_t$. Low confidence triggers more refinement steps and a stricter commitment gate.
    - **Design Motivation**: Ablation studies show that the adaptive threshold is the most critical component for efficiency gains—removing it causes throughput to drop by 71.6%.

3. **Adaptive Vocabulary Size ($V_t$)**:

    - **Function**: Dynamically adjusts the vocabulary subset size for softmax computation to reduce computational overhead.
    - **Mechanism**: $V_t = \text{clip}(V_{\text{phase}}(g_t) \cdot f_{\text{conf}}(\bar{c}) \cdot f_{\text{rep}}(r_t), V_{\min}, V_{\max})$. The vocabulary is expanded during early generation or at low confidence to increase robustness, and contracted at high confidence to reduce softmax cost. A repetition detector prevents degenerate repetition caused by an overly narrow vocabulary.
    - **Design Motivation**: Figure 1(b) shows that softmax latency grows sharply with vocabulary size—the full ~50K vocabulary is nearly an order of magnitude slower than small subsets.

### Loss & Training
CadLLM is entirely training-free. All strategies are implemented via linear interpolation and clipping, introducing no additional computation at inference time.

## Key Experimental Results

### Main Results
Results on LLaDA-Instruct (single H100):

| Benchmark | CadLLM Accuracy | CadLLM Throughput Gain | Fast-dLLM Accuracy | Generation Length |
|-----------|----------------|------------------------|-------------------|-------------------|
| GSM8K | 78.01% | 1.33× | 79.00% | 256 |
| MATH | 32.06% | 1.34× | 32.40% | 256 |
| HumanEval | 35.97% | **2.28×** | 37.19% | 256 |
| HumanEval | 43.29% | 1.74× | 45.12% | 512 |

### Ablation Study

| Configuration | Token/s | Accuracy | Note |
|---------------|---------|----------|------|
| All ON | 121.72 | 78.01% | Full model |
| No $V_t$ | 119.67 | 74.41% | Accuracy drops 4.6% |
| No $S_t$ | 136.76 | 76.12% | Faster but lower accuracy |
| No $B_t$ | 111.19 | 78.32% | Throughput drops 8.6% |
| No $\tau_t$ | 34.57 | 78.17% | **Throughput collapses 71.6%** |
| All OFF | 34.32 | 78.01% | No adaptation |

### Key Findings
- The adaptive threshold is the absolute core of efficiency: removing it causes throughput to collapse from 121.72 to 34.57 token/s, with NFE increasing by 289%.
- Acceleration is most pronounced on HumanEval (2.28×), as confidence variation is larger in code generation.
- CadLLM is equally effective on DREAM (1.1–1.4× gains), validating model-agnosticism.
- Adaptive vocabulary does not significantly affect speed but substantially impacts accuracy (−4.6%), underscoring the importance of sampling width control.

## Highlights & Insights
- The closed-loop controller design—driving four resource dimensions from a single confidence signal—is elegant in its simplicity; even "minimal adaptivity" substantially outperforms static scheduling.
- The repetition detector (preventing degenerate loops caused by vocabulary shrinkage) is a practical engineering detail that avoids a common pitfall of fast decoding.
- The deliberate choice of linear interpolation strategies demonstrates that even the simplest monotone mappings yield significant gains, establishing a lower bound for more sophisticated strategies.

## Limitations & Future Work
- Validation is limited to two dLLMs (LLaDA and DREAM); effectiveness on larger-scale models remains unknown.
- Hyperparameters (e.g., $B_{\min}, B_{\max}, S_{\max}$) require manual tuning, though ±20% sensitivity analysis demonstrates stability.
- Accuracy is competitive with but not fully lossless compared to baselines, with a 1–2% drop on HumanEval.
- Future work could explore nonlinear control strategies or reinforcement learning for optimizing strategy parameters.

## Related Work & Insights
- **vs. fast-dLLM**: Static thresholds and fixed block sizes are the root cause of uneven resource allocation; CadLLM addresses this with adaptive strategies.
- **vs. autoregressive acceleration (speculative decoding, etc.)**: dLLMs natively support parallel decoding; CadLLM further optimizes the degree of parallelism.
- **vs. Lu et al. (concurrent work)**: They similarly identify that fixed block sizes lead to premature commitment of low-confidence tokens, consistent with the motivation of this paper.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified four-dimensional adaptive control design is novel in the context of dLLM acceleration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablations with validation across multiple tasks and generation lengths.
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis is clear and method description is precise.
- Value: ⭐⭐⭐⭐ Has direct practical value for dLLM inference deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](../../NeurIPS2025/model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)
- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[ICLR 2026\] DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing](../../ICLR2026/model_compression/diffvax_optimization-free_image_immunization_against_diffusion-based_editing.md)
- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](../../ICLR2026/model_compression/compute-optimal_quantization-aware_training.md)

<!-- RELATED:END -->
