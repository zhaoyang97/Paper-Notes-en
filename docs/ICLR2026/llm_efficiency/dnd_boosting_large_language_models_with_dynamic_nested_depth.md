---
title: >-
  [Paper Note] DND: Boosting Large Language Models with Dynamic Nested Depth
description: >-
  [ICLR 2026][LLM Efficiency][Dynamic Depth] DND selects critical tokens at the end of each Transformer layer via a router and routes them back through the same layer for additional processing (nested depth). Combined with…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "Dynamic Depth"
  - "Adaptive Token Selection"
  - "Large Language Models"
  - "Post-Training Enhancement"
  - "MoE"
date: 2026-05-08
content_hash: 97e230783e422637
---

# DND: Boosting Large Language Models with Dynamic Nested Depth

**Conference**: ICLR 2026
**arXiv**: [2510.11001](https://arxiv.org/abs/2510.11001)
**Code**: None
**Area**: LLM Efficiency / Adaptive Computation
**Keywords**: Dynamic Depth, Adaptive Token Selection, Large Language Models, Post-Training Enhancement, MoE

## TL;DR
DND selects critical tokens at the end of each Transformer layer via a router and routes them back through the same layer for additional processing (nested depth). Combined with a routing control loss and a threshold control scheme for precise and stable token selection, DND achieves average performance gains of 1.88% and 0.87% on Qwen3-1.7B and Qwen3-30B-A3B, respectively, with fewer than 0.1M additional parameters.

## Background & Motivation
The dominant paradigm for improving large language models has been scaling — more parameters, data, and compute — at the cost of exponentially growing computational overhead. A key observation is that **prediction difficulty varies significantly across tokens**: most tokens are "easy" (e.g., those serving linguistic fluency), while a small subset of "critical" tokens involve complex logical reasoning or planning.

This motivates two related research directions:
- **Token Pruning**: Filtering out unimportant tokens to reduce computation — but this only avoids processing easy tokens.
- **Test-Time Compute Scaling (implicit strategies)**: Applying iterative computation over hidden states to enhance reasoning — but uniformly across all tokens.

**Key Challenge**: Easy tokens require no additional computation, whereas critical tokens demand deeper processing. Existing methods either perform pure subtraction (pruning) or apply uniform addition (global recurrence), lacking **targeted depth gains**.

DND's **Key Insight** is to combine both directions: first identify difficult tokens, then allocate additional computational depth to them — a form of "review" mechanism. This represents the first effective fusion of token-level selection and implicit depth augmentation.

## Method

### Overall Architecture
The DND strategy is applied only to the middle layers of the model (the initial and final layers are kept unchanged to preserve pretrained reasoning patterns). Within each DND layer:
1. A standard forward pass produces the vanilla output $\mathbf{X}^v$.
2. A router independently scores each token and selects those requiring additional processing.
3. Selected tokens are packed into a compact subsequence and re-fed into the same Transformer layer.
4. The nested depth output is merged with the original output via a normalized fusion strategy.

### Key Designs

1. **Token-Choice Routing Design (Section 3.1.1)**:
   A linear layer $R: \mathbb{R}^{d_{model}} \to \mathbb{R}$ serves as the router, independently computing a preference score $p_i = \sigma(R(\mathbf{x}_i^v))$ for each token's hidden state. A token-choice (rather than expert-choice) strategy is adopted, as expert-choice requires access to the full sequence and is incompatible with autoregressive token-by-token decoding (which would cause information leakage). Selection is determined by comparing against a preset threshold $\tau$: token $i$ is selected if $p_i > \tau$.

2. **Nested Depth Computation (Section 3.1.2)**:
   Selected tokens are packed via a binary mask $\mathbf{M}$ into a compact subsequence, assigned new positional encodings $\mathbf{E}'_{pos}$, and processed again through the same Transformer layer. After processing, an Unpack operation scatters outputs back to their original positions. This is equivalent to performing an "internal review iteration" for difficult tokens.

3. **Normalized Fusion Strategy (Section 3.1.3)**:
   To preserve pretrained knowledge, a gating mechanism merges the original and nested outputs: $\mathbf{x}_i = (\beta \cdot p_i) \cdot \mathbf{x}_i^v + (1 - \beta \cdot p_i) \cdot \mathbf{x}_i^d$ (applied only to selected tokens). $\beta$ is a learnable parameter (initialized to 0.1); a higher routing score $p_i$ assigns greater weight to the nested output. Unselected tokens retain their original output directly.

4. **Routing Control Loss (Section 3.2.1)**:
   To address selection instability caused by routing scores clustering in a narrow range, a dual-objective loss is designed:
   - **Score Dispersion Loss** $\mathcal{L}_{sd}$: Based on information entropy, it encourages diversity in the routing score distribution, increasing discriminability across tokens.
   - **Distribution Preservation Loss** $\mathcal{L}_{dp}$: An MSE penalty that discourages scores from deviating from 0.5, preventing vanishing gradients in the saturation regions of the sigmoid.

   Together, they form a "push-pull" dynamic: the entropy loss pushes scores apart to cover a wider range, while the MSE loss pulls scores toward the sigmoid center to maintain responsiveness.

5. **Threshold Control Scheme (Section 3.2.2)**:
   - **Buffered Ratio Control**: On each mini-batch, the error $e$ between the actual selection ratio and the target ratio $k_{target}$ is computed, and the threshold is updated in real time: $\tau \leftarrow \tau + \alpha \cdot e$.
   - **EMA Synchronization**: Periodically (e.g., every 50 steps), the threshold is updated via EMA using the mean of the top-$k$ routing values $\bar{\tau}_{topk}$ from the buffer: $\tau = (1-\gamma)\tau + \gamma\bar{\tau}_{topk}$, preventing long-term misalignment between the router and threshold optimization directions.

### Loss & Training
Total Loss = Cross-Entropy Loss + $\lambda_{sd} \mathcal{L}_{sd}$ + $\lambda_{dp} \mathcal{L}_{dp}$

Post-training via SFT, using the AdamW optimizer with cosine learning rate scheduling (5e-6 to 1e-6). Qwen3-1.7B is trained on 128 H100 GPUs for 1 day (2 epochs); Qwen3-30B-A3B on 256 H100 GPUs for 3 days (4 epochs). DND is applied only to middle layers ($L_s=4$ to $L_e=43$), with a target selection ratio of 20%. The router is initialized to all zeros, the threshold to 0.5, and $\beta$ to 0.1.

## Key Experimental Results

### Main Results
Qwen3-30B-A3B MoE model, evaluated on 17 benchmarks:

| Task Category | Representative Benchmark | SFT Baseline | +DND | Gain |
|--------------|--------------------------|-------------|------|------|
| General & Alignment | MMLU | 85.41 | 85.91 | +0.50 |
| General & Alignment | C-Eval | 83.09 | 84.92 | +1.83 |
| General & Alignment | IFEval | 83.09 | 84.31 | +1.22 |
| Math & STEM | AIME24 | 51.46 | 52.37 | +0.91 |
| Math & STEM | GPQA-Diamond | 56.76 | 57.67 | +0.91 |
| Code & Agent | BFCL v3 | 75.43 | 77.48 | +2.05 |
| Code & Agent | LCB-v6 | 31.14 | 32.56 | +1.42 |
| **Average** | 17 benchmarks | 75.70 | 76.57 | **+0.87** |

### Ablation Study (Qwen3-1.7B)

| Configuration | Avg. Score | Gain | Note |
|--------------|-----------|------|------|
| Qwen3-1.7B SFT | 59.53 | 0.00 | Baseline |
| +DND (Full) | 61.41 | +1.88 | All strategies |
| +DND (z-loss control only) | 60.54 | +1.01 | No precise routing control |
| +DND (routing control only) | 60.58 | +1.05 | No dynamic threshold adjustment |
| +DND (threshold control only) | 60.68 | +1.15 | No routing dispersion loss |
| Selection ratio = 10% | 60.33 | +0.80 | Too few tokens, insufficient attention |
| Selection ratio = 20% | 61.41 | +1.88 | Optimal balance |
| Selection ratio = 30% | 61.03 | +1.50 | Slightly below 20% |

### Key Findings
- **Minimal computational overhead**: At a 20% selection ratio, total FLOPs increase by only ~6.27%, with fewer than 0.1M additional parameters.
- **No performance degradation on any benchmark**: All 17 benchmarks show improvement with no trade-offs observed.
- **Code and Agent tasks benefit most**: BFCL v3 improves by 2.05, validating the hypothesis that DND filters noise tokens and focuses on critical reasoning tokens.
- **Token selection visualization**: Shallow layers tend to select key nouns; deeper layers select mathematical expressions and logical verbs — indicating that the model learns a hierarchical processing strategy.
- **Stable selection ratio at inference**: The average selection ratio remains in the range of 0.178–0.242, with slightly higher rates in middle layers.

## Highlights & Insights
- **Simple yet effective**: The core idea is highly intuitive — identify difficult tokens and process them once more. A single linear router yields significant gains.
- **Plug-and-play for post-training**: No pretraining from scratch is required; DND can be directly inserted into existing dense and MoE models, offering strong practical value.
- **Elegant routing control design**: The "push-pull" mechanism combining dispersion loss and preservation loss is better suited for precise ratio control than a simple z-loss.
- **Compatible with both dense and MoE architectures**: Validated on both 1.7B dense and 30B MoE models; the latter incurs lower overhead due to existing sparse activation in MoE layers.
- **Convincing visualization analysis**: The layerwise token selection patterns (shallow layers → entities; deep layers → logical operators) provide empirical evidence for adaptive computation.

## Limitations & Future Work
- Validation is limited to the post-training (SFT) stage; the effects during pretraining and continual pretraining remain unexplored.
- Only tested on autoregressive LLMs; applicability to other architectures such as diffusion-based LLMs is unknown.
- Layer-wise selection ratios vary (higher in middle and boundary layers), but this observation is not leveraged for layer-adaptive ratio design.
- The nesting depth is fixed at 1 (one additional pass only); whether multiple nesting iterations yield further gains is not explored.
- Hyperparameters for the training strategy ($\lambda_{sd}$, $\lambda_{dp}$, $\alpha$, $\gamma$, etc.) require careful tuning.

## Related Work & Insights
- **Mixture-of-Depths (MOD, Raposo et al., 2024)**: Dynamically reduces the number of computation layers to eliminate redundancy; DND takes the opposite direction — augmenting depth for critical tokens.
- **MOR (Bae et al., 2025)**: The most closely related work, also performing token selection with additional computation, but limited to 1B-scale pretraining, using z-loss for imprecise ratio control, and lacking a fusion strategy.
- **Inner Thinking Transformer (ITT, Chen et al., 2025)**: Similar in combining dynamic selection with additional computation; DND offers more refined control strategies.
- **DeepSeek-V3's balance loss**: Inspires DND's buffered ratio control mechanism.
- **Takeaway**: Token-level adaptive computation is a promising direction; the key lies in precise selection ratio control and effective fusion strategies.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)
- [\[AAAI 2026\] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](../../AAAI2026/llm_efficiency/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)
- [\[AAAI 2026\] Scaling and Transferability of Annealing Strategies in Large Language Model Training](../../AAAI2026/llm_efficiency/scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)
- [\[NeurIPS 2025\] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](../../NeurIPS2025/llm_efficiency/l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)

</div>

<!-- RELATED:END -->
