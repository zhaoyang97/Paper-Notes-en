---
title: >-
  [Paper Note] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning
description: >-
  [ICLR 2026][LLM Efficiency][continual learning] This paper proposes SMoPE, a framework that organizes a single shared prompt into multiple prompt experts within a sparse MoE structure. Dynamic sparse activation is achieved via prompt-attention score aggregation, significantly alleviating knowledge interference while maintaining high parameter efficiency, achieving SOTA on multiple continual learning benchmarks.
tags:
  - ICLR 2026
  - LLM Efficiency
  - continual learning
  - prompt tuning
  - Mixture of Experts
  - Sparse MoE
  - Prefix Tuning
date: 2026-05-08
content_hash: d6c1967e3a79971c
---

# One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning

**Conference**: ICLR 2026
**arXiv**: [2509.24483](https://arxiv.org/abs/2509.24483)
**Code**: [https://github.com/Minhchuyentoancbn/SMoPE](https://github.com/Minhchuyentoancbn/SMoPE)
**Area**: LLM Efficiency
**Keywords**: continual learning, prompt tuning, Mixture of Experts, Sparse MoE, Prefix Tuning

## TL;DR
This paper proposes SMoPE, a framework that organizes a single shared prompt into multiple prompt experts within a sparse MoE structure. Dynamic sparse activation is achieved via prompt-attention score aggregation, significantly alleviating knowledge interference while maintaining high parameter efficiency, achieving SOTA on multiple continual learning benchmarks.

## Background & Motivation

**Background**: Prompt-based continual learning (CL) methods adapt frozen pre-trained ViTs by prepending learnable prompts, and have become the dominant paradigm for mitigating catastrophic forgetting. Representative methods include DualPrompt, HiDe-Prompt, and NoRGa.

**Limitations of Prior Work**: Mainstream methods assign independent prompt subsets to each task (task-specific prompting), which introduces two issues: (a) inference requires a full forward pass through the pre-trained model for prompt selection, incurring high computational overhead; (b) prompt parameters grow linearly with the number of tasks, limiting scalability and hindering cross-task knowledge sharing.

**Key Challenge**: Methods such as OVOR address efficiency by using a single shared prompt, but because all prompt parameters are continuously updated, severe knowledge interference occurs, yielding inferior performance compared to task-specific approaches. There exists a fundamental conflict between efficiency and performance.

**Goal**: How can the parameter efficiency of a single prompt be retained while avoiding the knowledge interference inherent to shared prompts? Specifically: (a) how to perform sparse selection across attention heads in a multi-gate MoE; (b) how to balance expert utilization; and (c) how to maintain expert specialization without access to previous task data.

**Key Insight**: Building on the insight from Le et al. (2024a) that each attention head can be viewed as a composition of multiple MoE models, and that prefix tuning essentially adds new prompt experts to these MoEs. Since the structure is already MoE-like, sparse selection follows naturally.

**Core Idea**: Each prefix token in the shared prompt is treated as an independent expert. A unified surrogate score is computed via prompt-attention score aggregation to enable Top-K sparse activation, thereby achieving implicit parameter partitioning over a single prompt.

## Method

### Overall Architecture

SMoPE takes a ViT patch token sequence $\mathbf{X} \in \mathbb{R}^{N \times d}$ as input and produces the attention output at each MSA layer. The model employs a single shared prefix key $\mathbf{P}^K \in \mathbb{R}^{N_p \times d}$ and prefix value $\mathbf{P}^V \in \mathbb{R}^{N_p \times d}$, prepended to the attention key and value matrices. The pipeline consists of three stages: (1) computing a unified surrogate score for each prompt expert via prompt-attention score aggregation; (2) Top-K selection of the active expert subset; and (3) performing attention computation using only the selected experts.

### Key Designs

1. **Prompt-Attention Score Aggregation**

   - **Function**: Computes a unified surrogate score for each prompt expert, replacing the $N$ per-token scores in the original multi-gate MoE.
   - **Mechanism**: The attention scores from all tokens toward a given prompt expert are averaged, yielding $\tilde{s}_{j'}(\mathbf{X}) = \frac{\tilde{\mathbf{x}}^\top W_l^Q {W_l^K}^\top \mathbf{p}_{j'}^K}{\sqrt{d_v}}$, where $\tilde{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i$ is the mean token representation. Surrogate scores for all experts are obtained from a single computation of $\tilde{\mathbf{x}}$.
   - **Design Motivation**: In standard prefix tuning, each prompt expert is associated with $N$ score functions (one per output token), making direct Top-K selection intractable. Score aggregation reduces complexity from $\mathcal{O}(N d_k)$ to $\mathcal{O}(d_k)$ while preserving the same $\mathcal{O}(\tau^{-4})$ sample complexity as standard MoE.

2. **Sparse Expert Selection + Implementation**

   - **Function**: Performs Top-K selection based on surrogate scores, activating only the $K$ most relevant prompt experts.
   - **Mechanism**: The SMoPE-adjusted attention matrix is $\tilde{A}_l = [\tilde{A}_l^{\text{prompt}}, A_l^{\text{pre-trained}}]$, where the prompt component is $\tilde{A}_l^{\text{prompt}} = \text{TopK}(\tilde{\mathbf{x}}^\top W_l^Q {W_l^K}^\top \mathbf{P}^K / \sqrt{d_v}).\text{expand}(N, -1)$. The scores of the selected $K$ experts are expanded across all $N$ query tokens, while unselected expert scores are set to zero.
   - **Design Motivation**: Unlike OVOR, which updates all prompt parameters uniformly, sparse activation introduces implicit parameter partitioning that substantially reduces inter-task interference. Expert selection depends only on the current layer's input, requiring no additional forward pass as in task-specific methods.

3. **Adaptive Noise Mechanism**

   - **Function**: Adds adaptive noise penalties to frequently activated experts during training, encouraging utilization of inactive experts.
   - **Mechanism**: The activation frequency $F_{j'}$ is tracked per expert. Experts with above-average frequency receive a noise penalty $\epsilon_{j'} = \epsilon \cdot (\max_j \tilde{s}_j - \min_j \tilde{s}_j)$ (where $\epsilon \in [0,1]$ is a hyperparameter), reducing their selection probability. Experts below average frequency are not penalized.
   - **Design Motivation**: Standard SMoE is prone to unbalanced expert utilization, where a small subset dominates routing. In the CL setting, repeatedly activating the same experts exacerbates knowledge interference. Adaptive noise scales the penalty by the dynamic score range to avoid excessive perturbation, and only affects training without influencing inference.

4. **Prototype-based Loss for Expert Specialization**

   - **Function**: Uses prefix keys as prototype memories for previous tasks, preserving expert specialization when training on new tasks.
   - **Mechanism**: Two auxiliary losses are employed: (a) $\mathcal{L}_{\text{router}}$ encourages the scores of selected experts to exceed those of unselected ones; (b) $\mathcal{L}_{\text{proto}}$ uses the prefix keys from the end of the previous training round as prototypes to maintain routing consistency for old experts. Only frequently activated experts are retained in the prototype set to avoid noise.
   - **Design Motivation**: $\mathcal{L}_{\text{router}}$ promotes expert differentiation on the current task, while $\mathcal{L}_{\text{proto}}$ preserves task-specific specialization learned previously without requiring replay data. The two losses are complementary in alleviating forgetting.

### Loss & Training

The final loss is: $\mathcal{L} = \mathcal{L}_{ce} + \alpha_{\text{router}} \cdot \mathcal{L}_{\text{router}} + \alpha_{\text{proto}} \cdot \mathcal{L}_{\text{proto}}$

where $\mathcal{L}_{ce}$ is the standard cross-entropy loss, and $\alpha_{\text{router}}$, $\alpha_{\text{proto}}$ are loss weighting hyperparameters. Only the prefix parameters and classifier head are updated; the backbone remains frozen. Dense expert training (without sparse selection) is applied for the first few epochs of the first task to establish stable expert representations. Task-adaptive prediction is also employed to correct classifier bias toward new classes.

## Key Experimental Results

### Main Results

| Dataset | Metric | SMoPE | Prev. SOTA (VQ-Prompt) | Gain |
|--------|------|-------|---------------------|------|
| ImageNet-R (10-task) | FAA | **79.32** | 78.71 | +0.61 |
| ImageNet-R (10-task) | CAA | **84.39** | 83.24 | +1.15 |
| CIFAR-100 (10-task) | FAA | **89.23** | 88.73 | +0.50 |
| CIFAR-100 (10-task) | CAA | **93.67** | 92.84 | +0.83 |
| CUB-200 (10-task) | FAA | **87.43** | 86.72 | +0.71 |
| CUB-200 (10-task) | CAA | **91.11** | 90.33 | +0.78 |

Under self-supervised pre-training (ImageNet-R): iBOT-1K FAA 72.17 / DINO-1K FAA 68.61, surpassing all baselines in both settings.

### Ablation Study

| Configuration | FAA (CUB-200) | CAA | Notes |
|------|---------------|-----|------|
| One Prompt (baseline) | 75.23 | 83.61 | Single prompt with no enhancements |
| + Score Aggregation | 75.49 | 83.65 | Unified surrogate scores, marginal gain |
| + Sparse Expert Selection | 79.12 | 87.16 | **+3.63 FAA**, sparse selection is the core contributor |
| + Adaptive Noise | 85.36 | 89.12 | **+6.24 FAA**, balancing utilization is highly effective |
| + Task-Adaptive Prediction | 86.03 | 90.09 | Corrects classifier bias |
| + Initial Dense Training | 86.27 | 90.23 | Stabilizes initial expert representations |
| + Router Loss | 87.05 | 90.47 | Promotes expert specialization |
| + Prototype Loss (Full) | **87.43** | **91.11** | Preserves old expert specialization |

### Key Findings
- **Sparse selection + Adaptive Noise** contribute the most: FAA improves from 75.49 to 85.36, a combined gain of +9.87.
- Parameter count is only 0.38M, approximately 8% of Deep L2P++ (4.78M), with training and inference GFLOPs at 50% of competing methods.
- Ablation on $\epsilon$: $\epsilon=0$ leads to expert monopolization by a small subset; $\epsilon=1$ causes excessive exploration; optimal performance is achieved near $\epsilon=0.5$.
- SMoPE, using a single shared prompt, outperforms all task-specific prompt methods, challenging the assumption that shared prompts inevitably underperform.

## Highlights & Insights
- **The MoE perspective on prefix tuning is highly elegant**: Treating each prefix token as an expert within an attention head makes sparse selection a natural extension. This perspective is transferable to any scenario employing prefix tuning.
- **Prompt-Attention Score Aggregation**: Replacing $N$ score computations with a single mean-token computation is both efficient and theoretically sound, preserving sample complexity — a theory-driven engineering contribution.
- **Adaptive noise is better suited to CL than conventional load-balancing losses**: Standard MoE auxiliary losses enforce hard balance and may disrupt previously acquired knowledge; adaptive noise penalizes only high-frequency experts and applies solely during training, making it more controlled and less destructive.
- **The use of prefix keys as prototypes** is transferable to NLP prefix tuning settings, applicable to any incremental learning scenario requiring routing consistency without access to previous task data.

## Limitations & Future Work
- Experiments are conducted exclusively on ViT-B/16; performance on larger-scale models or NLP tasks has not been validated.
- Top-K is a hard selection; learnable soft routing (e.g., Gumbel-Softmax) may offer greater flexibility.
- The $\epsilon$ hyperparameter of adaptive noise requires manual tuning and may need re-searching across different datasets.
- The prototype loss relies only on prefix keys from the immediately preceding round; long task sequences may lead to prototype drift.
- The effects of prompt length $N_p$ and the Top-K ratio under varying numbers of tasks remain unexplored.

## Related Work & Insights
- **vs. OVOR**: Both use a single prompt, but OVOR updates all prompt parameters uniformly, whereas SMoPE's sparse activation reduces interference. OVOR underperforms SMoPE by 4–10 points on most benchmarks.
- **vs. VQ-Prompt**: VQ-Prompt uses vector quantization for prompt selection; SMoPE performs selection directly within the MoE framework via attention scores, which is more natural and requires roughly 50% of the computation.
- **vs. HiDe-Prompt/NoRGa**: These methods employ task-specific prompts with additional forward passes for prompt retrieval, requiring over 10× the parameters of SMoPE. SMoPE demonstrates that with the right architectural design, a single shared prompt can surpass task-specific methods.

## Rating
- Novelty: ⭐⭐⭐⭐ The MoE perspective on prefix tuning offers theoretical novelty, though SMoE itself is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, two pre-training paradigms, detailed ablations, and computational cost comparisons.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, with natural transitions from theory to implementation.
- Value: ⭐⭐⭐⭐ Meaningful advancement for prompt-based CL; the 50% reduction in computation has practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](../../AAAI2026/llm_efficiency/resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[CVPR 2026\] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](../../CVPR2026/llm_efficiency/cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)
- [\[NeurIPS 2025\] Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](../../NeurIPS2025/llm_efficiency/let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)

</div>

<!-- RELATED:END -->
