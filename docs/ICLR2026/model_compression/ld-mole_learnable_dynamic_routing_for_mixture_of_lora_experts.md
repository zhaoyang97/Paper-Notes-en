---
title: >-
  [Paper Note] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts
description: >-
  [ICLR 2026][Model Compression][LoRA] This paper proposes LD-MoLE, which replaces conventional TopK routing with a Sparsegen closed-form projection to achieve differentiable, dynamic…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "LoRA"
  - "Mixture-of-Experts"
  - "dynamic routing"
  - "Sparsegen"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: b88822f5503c05f8
---

# LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts

**Conference**: ICLR 2026
**arXiv**: [2509.25684](https://arxiv.org/abs/2509.25684)
**Code**: [GitHub](https://github.com/eshentw/LD-MoLE)
**Area**: Model Compression
**Keywords**: LoRA, Mixture-of-Experts, dynamic routing, Sparsegen, parameter-efficient fine-tuning

## TL;DR
This paper proposes LD-MoLE, which replaces conventional TopK routing with a Sparsegen closed-form projection to achieve differentiable, dynamic, token-adaptive LoRA expert assignment. A lightweight MLP predicts sparse factors, and an analytic sparsity loss is employed. LD-MoLE outperforms fixed-routing and ReLU-routing baselines across multiple benchmarks.

## Background & Motivation
LoRA combined with MoE (i.e., MoLE) is a promising direction for parameter-efficient fine-tuning of large language models: multiple low-rank LoRA modules serve as experts, and a routing network determines which experts each token uses. However, existing methods predominantly rely on TopK routing, which suffers from three key limitations:

**Hyperparameter sensitivity**: The value of $k$ requires careful tuning, as the optimal $k$ varies across tasks.

**Non-differentiability**: TopK selection is a discrete operation, hindering end-to-end optimization.

**Fixed allocation**: Each token activates the same number of experts, failing to adapt to varying complexity.

ReMoE attempts to address these issues with ReLU routing, but suffers from instability where some tokens may be assigned no expert at all. The core question is: can one design a routing mechanism that is both stably differentiable and capable of adaptively controlling the number of activated experts?

LD-MoLE addresses this by leveraging Sparsegen — a closed-form projection onto the probability simplex — which guarantees that each token is assigned at least one expert, while enabling dynamic expert selection through a learnable sparsity parameter $\lambda$.

## Method

### Overall Architecture
Multiple LoRA experts are placed at each linear projection in every Transformer layer. A routing module receives token embeddings and outputs sparse weight assignments over experts. The final output is the sum of the base weight output and the weighted outputs of all active experts.

### Key Designs
1. **Sparsegen Routing**:

    - Function: Projects routing scores onto the probability simplex to produce sparse assignments.
    - Mechanism: Given expert scores $\bm{u} = \bm{W}_{\text{gate}} \bm{x}$, Sparsegen solves the optimization problem $\bm{p} = \arg\min_{\bm{p}} \|\bm{p} - \bm{u}\|^2 - \lambda\|\bm{p}\|^2$, subject to $\bm{p} \geq 0, \mathbf{1}^\top \bm{p} = 1$. The closed-form solution is $\bm{p}_i = \left[\frac{\bm{u}_i - \tau}{1-\lambda}\right]_+$.
    - Design Motivation: Unlike the discrete transitions of TopK, Sparsegen has well-defined subgradients and bounded upper limit, ensuring stable optimization. As $\lambda \to 1^-$, the distribution becomes sparser; as $\lambda \to -\infty$, it approaches uniform.

2. **Learnable Dynamic Sparsity Factor**:

    - Function: Predicts a personalized $\lambda$ value for each token.
    - Mechanism: A lightweight shared MLP $f(\bm{x}) = \lambda \in \mathbb{R}$ predicts $\lambda$ from the input, shared across dimensions (typically only two variants), with negligible parameter overhead.
    - Design Motivation: Different tokens have varying modeling complexity; complex tokens benefit from more experts, while simple tokens require fewer.

3. **Analytic Sparsity Loss**:

    - Function: Explicitly controls the number of active experts.
    - Mechanism: Using Proposition 2, the paper derives the $\lambda$ interval $[\lambda_{\text{lower}}(k), \lambda_{\text{upper}}(k))$ that activates exactly $k$ experts, yielding the sparsity loss $\mathcal{L}_{\text{sparse}} = \text{ReLU}(\lambda_{\text{lower}}(k) - \lambda)$.
    - Design Motivation: Leverages the analytic properties of Sparsegen to directly constrain sparsity without heuristic tuning.

### Loss & Training
Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \mathcal{L}_{\text{lb}} + \beta \mathcal{L}_{\text{sparse}}$
- $\mathcal{L}_{\text{LM}}$: Standard cross-entropy (next-token prediction or sequence classification).
- $\mathcal{L}_{\text{lb}}$: Load balancing loss to prevent routing collapse.
- $\mathcal{L}_{\text{sparse}}$: Sparsity control loss.

8 LoRA experts, rank=8, scaling=16. Trained for 10 epochs on 4×H200 GPUs.

## Key Experimental Results

### Main Results

| Method | Model | MMLU-P | ARC-C | ARC-E | OBQA | CommQA | SWAG | HellaS | CoLA | RTE | Avg |
|------|------|--------|-------|-------|------|--------|------|--------|------|-----|-----|
| MoLA(8888) | Llama-3B | 40.3 | 71.6 | 83.5 | 81.0 | 79.8 | 83.6 | 87.5 | 85.8 | 90.6 | 78.2 |
| MoLA(2468) | Llama-3B | 42.3 | 71.9 | 83.9 | 83.6 | 80.0 | 84.0 | 87.3 | 86.0 | 89.5 | 78.7 |
| ReMoLE | Llama-3B | 48.0 | 75.3 | 89.3 | 83.4 | 79.5 | 90.5 | 93.4 | 84.0 | 89.5 | 81.4 |
| **LD-MoLE** | Llama-3B | **49.6** | 74.6 | **89.5** | 83.8 | 80.3 | **90.8** | **93.6** | 85.5 | **91.0** | **82.0** |
| **LD-MoLE** | Llama-8B | **56.0** | **83.7** | **91.6** | **88.0** | 83.0 | **92.3** | **95.5** | 85.3 | **91.3** | **85.2** |

### Ablation Study

| Configuration | Avg Score | Note |
|------|--------|------|
| LD-MoLE (β=0) | 82.0 | No sparsity loss; full performance |
| LD-MoLE (β>0, k≤4) | ~81.5 | Fewer active experts; slight performance drop |
| MoLA(2468) vs MoLA(8888) | 78.7→78.2 | Layer-wise allocation more important under fixed routing |
| ReMoLE (unstable) | Sharp drop on CoLA | ReLU routing may assign zero experts |

### Key Findings
- Dynamic routing consistently outperforms fixed routing on instruction-tuning tasks, while the gap is smaller on classification tasks.
- LD-MoLE guarantees at least one expert per token (Lemma 1), avoiding the instability of ReMoLE.
- The sparsity loss effectively reduces the number of active experts without significantly degrading performance.
- MoLA(2468) outperforms MoLA(8888), suggesting that under fixed routing many experts are wasted.

## Highlights & Insights
- The application of Sparsegen to MoE routing is the key innovation, simultaneously achieving differentiability and sparsity.
- The shared MLP design for predicting $\lambda$ is concise and efficient, introducing minimal parameter overhead.
- The analytic sparsity loss is derived directly from mathematical properties, requiring no heuristic tuning.

## Limitations & Future Work
- Main experiments are conducted only on 3B and 1.7B scale models; larger models remain untested.
- Training cost (4×H200, 10 epochs) remains relatively high for a PEFT method.
- Specific inference latency introduced by routing computation (sorting + MLP) is not reported.

## Related Work & Insights
- **vs. MoLA (TopK)**: LD-MoLE adapts $k$ automatically, eliminating the need for hyperparameter tuning.
- **vs. ReMoE (ReLU)**: LD-MoLE guarantees at least one expert is assigned per token, yielding greater stability.
- **vs. Soft MoE**: LD-MoLE is sparse, offering higher computational efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of Sparsegen routing to MoLE is novel, with rigorous theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-task evaluation, though inference efficiency comparisons are absent.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, though notation is dense.
- Value: ⭐⭐⭐⭐ Provides a stronger mathematical framework for MoE routing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ACL 2026\] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](../../ACL2026/model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)
- [\[NeurIPS 2025\] Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP](../../NeurIPS2025/model_compression/multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](flyprompt_brain-inspired_random-expanded_routing.md)

</div>

<!-- RELATED:END -->
