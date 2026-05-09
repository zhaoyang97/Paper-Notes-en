---
title: >-
  [Paper Note] Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful
description: >-
  [NeurIPS 2025][Optimization][small batch training] This paper systematically investigates the behavior of small batch sizes (down to batch size = 1) in language model pre-training and fine-tuning. It proposes a scaling rule for Adam $\beta_2$ based on fixing the "token half-life," demonstrates that small-batch training is stable, shows that vanilla SGD becomes competitive with adaptive optimizers under small batches, and recommends avoiding gradient accumulation.
tags:
  - NeurIPS 2025
  - Optimization
  - small batch training
  - SGD
  - Adam
  - gradient accumulation
  - language models
date: 2026-05-08
content_hash: 2aa9c44b1131d080
---

# Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful

**Conference**: NeurIPS 2025
**arXiv**: [2507.07101](https://arxiv.org/abs/2507.07101)
**Code**: [https://github.com/martin-marek/batch-size](https://github.com/martin-marek/batch-size)
**Area**: Optimization
**Keywords**: small batch training, SGD, Adam, gradient accumulation, language models

## TL;DR
This paper systematically investigates the behavior of small batch sizes (down to batch size = 1) in language model pre-training and fine-tuning. It proposes a scaling rule for Adam $\beta_2$ based on fixing the "token half-life," demonstrates that small-batch training is stable, shows that vanilla SGD becomes competitive with adaptive optimizers under small batches, and recommends avoiding gradient accumulation.

## Background & Motivation

- **State of the Field**: Large batch sizes are widely considered a prerequisite for stable language model training. When GPU memory is insufficient to support large batches, researchers typically employ gradient accumulation to simulate larger effective batch sizes.
- **Limitations of Prior Work**: Existing work generally only adjusts the learning rate when reducing batch size, leaving Adam's momentum decay rates $\beta_1$ and $\beta_2$ unchanged. This paper identifies this practice as the root cause of instability in small-batch training.
- **Root Cause**: If, instead of fixing $\beta_2$ itself, one fixes the $\beta_2$ "half-life" measured in tokens, small-batch training remains stable even at the extreme of batch size = 1.
- **Paper Goals**: The core insight is that small batch sizes cause the optimizer to take shorter prediction steps, thus reducing the demand for complex optimizers or elaborate hyperparameter tuning.

## Method

### Overall Architecture
The methodological contribution is not a new optimizer architecture but rather a theoretical and practical framework for correctly scaling optimizer hyperparameters to small batch sizes. It comprises: (1) introducing the concept of *moment half-life* as a replacement for directly specifying $\beta$ values; (2) proposing a scaling heuristic for $\beta_2$; and (3) systematically validating the advantages of small batches across multiple optimizers and model scales.

### Key Designs

1. **Moment Half-Life**: Rather than operating directly on $\beta_1$ and $\beta_2$, this approach defines a decay half-life $t_{1/2}$ measured in tokens. In Adam, the contribution of past gradients decays by $\beta$ at each update step; the half-life denotes the number of tokens required for a gradient's contribution to halve. This quantity is directly linked to batch size via $\beta^{t_{1/2} / (B \cdot T)} = 1/2$. By fixing the half-life rather than the $\beta$ value, hyperparameters transfer naturally across different batch sizes.

2. **$\beta_2$ Scaling Rule**: When scaling batch size from $B$ to $B^*$, the new $\beta_2^*$ is derived by keeping the token half-life $t_2$ constant: $\beta_2^* = \beta_2^{B^*/B}$. For example, scaling from a default $\beta_2 = 0.95$ at batch size 512 to batch size 1 yields $\beta_2 \approx 0.9999$. This simple one-shot heuristic transfers settings across configurations without additional tuning.

3. **Small Batch + Simple Optimizer Strategy**: Under small batch sizes, all optimizers (SGD, Adam, Adafactor, Muon) perform comparably, and SGD is competitive even without momentum. This is because short step lengths mean the optimizer need not predict the loss landscape far from the current iterate, reducing the requirement for optimizer complexity.

### Loss & Training
- **Learning rate schedule**: linear warmup from 0 to peak over the first 5% of steps, followed by cosine decay to 0.
- $\beta_1 = 0.9$ performs well across all batch sizes and requires no scaling.
- Replacing Adam with Adafactor under small batches substantially reduces memory: Adafactor stores only row-wise and column-wise second moments, reducing memory from $O(d_1 \times d_2)$ to $O(d_1 + d_2)$.
- Stochastic rounding is used to enable bfloat16 weights to function correctly under small batch sizes.

## Key Experimental Results

### Main Results: Optimizer Performance Across Batch Sizes (30M model, 600M tokens)

| Optimizer | Best Loss (Batch=1) | Best Loss (Batch=4096) | Small Batch Better? |
|-----------|--------------------|-----------------------|---------------------|
| SGD | ~3.95 | ~4.10 | ✅ Yes |
| Adafactor | ~3.95 | ~4.00 | ✅ Yes |
| Adam | ~3.95 | ~3.97 | ✅ Yes |
| Muon | ~3.95 | ~3.96 | ✅ Yes |

### Large-Scale Validation: GPT-2 (124M) and GPT-3 (1.3B)

| Model | Optimizer | Batch Size | Validation Loss |
|-------|-----------|-----------|-----------------|
| GPT-3 (1.3B) | AdamW (baseline) | 512 | Baseline |
| GPT-3 (1.3B) | Adam (fixed $t_2$) | 1 | Better than baseline |
| GPT-3 (1.3B) | Adafactor | 1 | Better than baseline |
| GPT-3 (1.3B) | Vanilla SGD | 1 | On par with baseline |
| GPT-2 (124M) | AdamW (tuned) | 512 | Baseline |
| GPT-2 (124M) | Adam | 1 | On par with baseline |
| GPT-2 (124M) | SGD | 1 | Slightly below baseline |

### Fine-Tuning: Gemma 3 (4B) on MATH

| Method | Batch Size | Memory/Param | MATH Accuracy |
|--------|-----------|--------------|---------------|
| Adam (full FT) | 16 (grad. accum.) | 16 bytes | Baseline |
| LoRA + Adam | 16 | ~2 bytes | Below full FT |
| Adafactor (full FT) | 1 | ~2 bytes | On par with full FT |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Fixed $\beta_2$ vs. fixed $t_2$ | Validation loss | Fixed $t_2$ yields consistent performance across batch sizes; fixed $\beta_2$ degrades significantly at small batches |
| $\sqrt{B}$ LR scaling vs. empirical optimum | Optimal LR | $\sqrt{B}$ scaling is overly aggressive; in practice, scaling $B$ from 1 to 1024 requires only ~3× LR scaling rather than 32× |
| bfloat16 + stochastic rounding vs. float32 | Validation loss | Stochastic rounding brings bfloat16 close to float32 performance |
| SGD with/without momentum | Validation loss | Difference is negligible at small batch sizes; momentum is necessary at large batch sizes |

### Key Findings
- All optimizers achieve best per-FLOP efficiency at batch size = 1.
- Small batches are far more robust to hyperparameter choice than large batches: batch = 1 yields near-optimal loss across nearly the entire search space of learning rates and $\beta_1$ values.
- Fixing $\beta_2$ at small batch sizes causes training instability, which is the root cause of the "small batches don't work" conclusion in prior literature.
- At the 1.3B scale, vanilla SGD (without momentum or weight decay) matches the AdamW baseline.
- Gradient accumulation not only wastes computational steps but also increases memory overhead by requiring storage of accumulated gradients.

## Highlights & Insights
- **Short-step hypothesis**: Small batches combined with small learning rates mean the optimizer travels a shorter distance per step, eliminating the need to predict the loss landscape far ahead and thereby reducing the complexity requirements placed on the optimizer.
- **Why momentum becomes unnecessary**: Large step sizes overshoot in high-curvature directions and cause oscillations, necessitating momentum to dampen them; small step sizes do not overshoot, rendering momentum redundant.
- **Practical recommendation**: Use the smallest batch size that maximizes hardware throughput (tokens/second), rather than the largest batch size possible.
- This work directly challenges the conventional wisdom that "larger batches yield better training" and that "SGD cannot train Transformers."

## Limitations & Future Work
- Very small batches (batch = 1 or 2) reduce MFU by 30–70%, so practical deployment requires balancing stability against computational efficiency.
- The optimal learning rate scaling rule across batch sizes lacks a clean closed form; no concise formula is identified.
- Interactions with batch size schedules (dynamically adjusting batch size during training) are not explored.
- The combination of lower-precision weights (e.g., INT8/INT4) with small batches is not investigated.
- Theoretical analysis is limited to the compute-optimal frontier (the Chinchilla regime); whether the findings hold under convergent training remains an open question.

## Related Work & Insights
- This work challenges the conclusion of Xiao et al. (2024) that small-batch Adam performs poorly, showing this is an artifact of not tuning $\beta_2$; a one-shot scaling fix suffices.
- This work challenges the conclusion of Zhao et al. (2025) that SGD is far inferior to Adam, demonstrating that SGD can match Adam under small batch sizes.
- The findings offer practical guidance for memory-efficient fine-tuning: small batch + Adafactor can replace LoRA, achieving full fine-tuning performance at LoRA-level memory cost.
- The results resonate with the critical batch size concept in the data parallelism literature, while providing more extreme empirical validation.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](constrained_network_slice_assignment_via_llms.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)

<!-- RELATED:END -->
