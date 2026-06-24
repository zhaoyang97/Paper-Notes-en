---
title: >-
  [Paper Note] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models
description: >-
  [ACL 2025][Multimodal Efficiency][Model Pruning] This paper proposes Hierarchical Safety Realignment (HSR), a method that first identifies safety-critical attention heads and then locates and restores safety-critical pruned neurons within these heads. With minimal parameter overhead (on the order of ten-thousandths), HSR significantly recovers the safety performance lost in pruned LVLMs.
tags:
  - "ACL 2025"
  - "Multimodal Efficiency"
  - "Model Pruning"
  - "Safety Alignment"
  - "Large Vision-Language Models"
  - "Attention Heads"
  - "Neuron Restoration"
date: 2026-05-08
content_hash: c5705ce3ed7ed4d0
---

# Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.16104](https://arxiv.org/abs/2505.16104)  
**Authors**: Yue Li, Xin Yi, Dongsheng Shi, Gerard de Melo, Xiaoling Wang, Linlin Wang (East China Normal University, HPI/University of Potsdam)
**Code**: [TheShineyue/HSR](https://github.com/TheShineyue/HSR)  
**Area**: Multimodal VLM  
**Keywords**: Model Pruning, Safety Alignment, Large Vision-Language Models, Attention Heads, Neuron Restoration

## TL;DR

This paper proposes Hierarchical Safety Realignment (HSR), a method that first identifies safety-critical attention heads and then locates and restores safety-critical pruned neurons within these heads. With minimal parameter overhead (on the order of ten-thousandths), HSR significantly recovers the safety performance lost in pruned LVLMs.

## Background & Motivation

### Background
Large Vision-Language Models (LVLMs) are parameter-heavy and often require network pruning for deployment in resource-constrained environments. However, the neural regions associated with safety are dissociated from and sparsely distributed relative to those for general capabilities. Pruning methods optimized for "utility importance" naturally tend to discard these neurons that contribute minimally to utility but are critical for safety.

### Limitations of Prior Work
- Existing pruning research (e.g., Wanda, SparseGPT, SNIP) primarily focuses on maintaining utility post-compression, ignoring safety degradation.
- Safety alignment research concentrates on defending against jailbreak attacks and safety training, neglecting safety loss caused by pruning.
- Prior observations (Hasan et al. 2024) only identified safety improvements under low-sparsity conditions, leaving safety degradation under high sparsity overlooked.

### Design Motivation
The authors conduct Wanda pruning experiments at 50% sparsity on six mainstream LVLMs, and discover safety degradation across all models: the worst Attack Success Rate (ASR) increases by 15.4%, and the mildest by 2.8%. This is the first work dedicated specifically to safety restoration in pruned LVLMs.

## Method

### Overall Architecture
HSR adopts a coarse-to-fine two-level hierarchical strategy: it first locates safety-critical attention heads at the head level, and then identifies and restores safety-critical pruned neurons at the neuron level.

### Key Design 1: Safety-Critical Head Identification (Ships Metric)

Drawing on the Safety Head Importance Score (Ships) proposed by Zhou et al. (2025), the contribution of each attention head to safety is quantified:

1. For each attention head $h_i^l$, its contribution is ablated by multiplying its Q/K/V matrices by a tiny coefficient $\epsilon$.
2. The Kullback-Leibler (KL) divergence of the model's output distribution on harmful inputs before and after ablation is computed, serving as the safety contribution score for that head.
3. For the Grouped-Query Attention (GQA) mechanism (commonly used in modern LVLMs), adapted masking equations are derived.
4. At the dataset level, singular value decomposition (SVD) is performed on the network activation matrix, and the principal angle is used to measure the deviation of safety representations:

$$\text{Ships}(D, h_i^l) = \sum_{r=1}^{r_{\max}} \cos^{-1}(\sigma_r(U_\theta^{(r)}, U_A^{(r)}))$$

5. The top-$h$ attention heads with the highest Ships scores are selected as safety-critical heads.

### Key Design 2: Safety-Critical Neuron Localization and Restoration

Within the safety-critical heads, neurons that have been pruned but are crucial for safety are further identified:

1. **Dual Importance Evaluation**: The importance scores $\mathbf{I}^s$ and $\mathbf{I}^u$ for each weight are computed on safety data $D^s$ (harmful instructions + refusal responses) and utility data $D^u$ (safe instructions + normal responses), respectively.
2. **Three Scoring Methods**: Wanda Score (absolute weight $\times$ $\ell_2$-norm of input activation), SparseGPT Score (based on the Hessian matrix), and SNIP Score (first-order Taylor approximation).
3. **Set Operation for Selecting Safety-Critical Neurons**:

$$S(p, q, p_{\max}) = (S^s(q) \cap S^u(p_{\max})) - S^u(p)$$

where $S^s(q)$ is the set of weights in the top-$q$\% of safety importance, and $S^u(p_{\max})$ is the set in the top-$p_{\max}$\% of utility importance, subtracting the already retained set $S^u(p)$. This ensures that the restored neurons are highly important for safety, moderately important for utility (avoiding severe utility degradation), and indeed pruned.

4. The original weight values of these identified safety-critical neurons are restored in the pruned model.

### Data Construction
- Safety Dataset: Unsafe-Unsafe pairs (unsafe images + unsafe instructions) from the VLGuard training set.
- Utility Dataset: Safe-Safe pairs (safe images + safe instructions) from the VLGuard training set.

## Key Experimental Results

### Main Results: HSR Performance across Pruning Methods (Qwen2.5-VL, 50% Sparsity)

| Method | SafeBench ASR↓ | Ch3Ef ASR↓ | Avg ASR↓ | RSR | MMBench↑ | DocVQA↑ | Restored Param Ratio |
|------|------|------|------|------|------|------|------|
| Full Model | 1.40 | 2.35 | 1.88 | - | 87.02 | 94.51 | - |
| SNIP | 4.60 | 8.12 | 6.36 | - | 84.55 | 92.93 | - |
| SNIP + HSR | 3.00 | 5.34 | 4.17 | 48.88% | 84.62 | 92.90 | 0.150‱ |
| Wanda | 11.20 | 17.74 | 14.47 | - | 85.15 | 91.97 | - |
| Wanda + HSR | 9.00 | 13.03 | 11.02 | 27.40% | 85.01 | 92.13 | 0.020‱ |
| SparseGPT | 3.00 | 3.21 | 3.10 | - | 83.88 | 90.64 | - |
| SparseGPT + HSR | 2.80 | 2.56 | 2.68 | 34.43% | 83.88 | 90.63 | 0.133‱ |

HSR effectively restores safety under all three pruning methods, achieving a safety restoration rate exceeding 27% while recovering only a few ten-thousandths of the pruned parameters.

### Experiment 2: HSR Performance on Different LVLMs (Wanda, 50% Sparsity)

| Model | Pruned Avg ASR | Post-HSR Avg ASR | RSR | Utility Delta | Restored Param Ratio |
|------|------|------|------|------|------|
| Qwen2-VL | 22.24 | 16.78 | 35.29% | +1.21 | 0.016‱ |
| LLaVA-NeXT-Mistral | 17.60 | 14.57 | **104.12%** | -0.32 | 0.385‱ |
| LLaVA-NeXT-Vicuna | 18.12 | 17.09 | 36.52% | -0.33 | 1.803‱ |
| LLaVA-NeXT-Llama3 | 17.71 | 16.99 | 14.81% | -0.19 | 0.799‱ |
| Llama3.2-Vision | 8.98 | 7.93 | 16.69% | -1.94 | 0.065‱ |

LLaVA-NeXT-Mistral achieves a safety restoration rate exceeding 100% (the post-HSR ASR is even lower than that of the unpruned model). The utility of the Qwen series models slightly improves due to HSR.

### Ablation Study: Influence of Sparsity Levels (Qwen2-VL + Wanda)

| Sparsity | Pruned Safety/Utility | Post-HSR Safety/Utility |
|------|------|------|
| 40% | 10.69 / 82.79 | 10.01 / 82.65 |
| 50% | 22.24 / 76.10 | 16.78 / 77.31 |
| 60% | 27.05 / 48.17 | 25.61 / 63.37 |

The safety restoration is most significant at 50% sparsity. At 60% sparsity, the utility improvement is largest (+15.2), indicating that the restored safety-critical neurons also contribute positively to general utility.

## Key Findings

- **A small number of neurons dictate safety**: Only the top 0.35% of safety-important neurons play a critical role; neurons beyond this range can conversely have a negative impact on safety.
- **Entanglement of safety and utility neurons**: Neurons that contribute the most to safety often also make significant contributions to utility.
- **Existence of "harmful safety" neurons**: Directly restoring entire attention heads (HSR-a) conversely leads to an increase in ASR, because some neurons within those heads have a negative effect on safety.
- **Strong correlation between total Ships score and safety degradation**: The Spearman correlation coefficient between the overall Ships score ranking of the six models and their post-pruning safety degradation ranking reaches 0.8857.
- **GQA mechanism affects restoration efficiency**: The Qwen series utilizes Grouped-Query Attention (GQA) with the largest query group size and fewest heads per group, giving a single neuron a wider range of influence and leading to higher restoration efficiency.

## Highlights & Insights

- **Pioneering Nature**: This is the first work specifically targeting safety restoration in LVLMs after pruning, filling a gap in the literature on pruning safety.
- **Extremely Low Overhead**: The amount of restored parameters is only a tiny fraction of a ten-thousandth of the pruned parameters (as low as 0.016‱), barely affecting the sparsity of the model.
- **Hierarchical Design**: Coarse-to-fine selection from attention heads to neurons retains lightweight efficiency while avoiding the restoration of harmful neurons.
- **Extensive Validation**: Evaluated on six mainstream LVLMs, three pruning methods, structured/unstructured pruning, and various sparsity levels.
- **Compatibility with Various Pruning Methods**: Compatible with Wanda, SparseGPT, and SNIP.

## Limitations & Future Work

- **Potential Slight Utility Decrease**: HSR leads to a minor utility loss in some models (e.g., approximately a 2% drop in Llama3.2-Vision).
- **Limited Restoration on Llama 3 Series**: The safety restoration rate for Llama3-based LVLMs is only 14-17%, which is significantly lower than other models.
- **Still Requires Parameter Restoration**: Although the scale is extremely small, there might be superior alternative approaches that achieve zero-parameter restoration.
- **Dependency on Labeled Safety Data**: Paired datasets of harmful instructions and refusal responses are required to construct safety importance evaluations.
- **Narrow Sparsity Range Tested**: The efficacy of the method has not been fully explored at extremely high sparsity levels (>60%).

## Related Work & Insights

- **Wei et al. (2024)**: Discovered that safety and utility regions are dissociated and sparse; this work designs restoration strategies based on this finding.
- **Zhou et al. (2025)**: Proposed the Ships metric to evaluate the safety contributions of attention heads; this work applies it to safety restoration in post-pruned scenarios.
- **Arditi et al. (2024)**: Identified a single refusal direction in LLMs; this work performs more fine-grained safety localization at both the attention head and neuron levels.
- **Hasan et al. (2024)**: Observed that low-sparsity pruning can improve safety; this work focuses on safety degradation at high sparsity levels and proposes remedies.
- **AdaShield (Wang et al. 2024b)**: Enhances safety by adding defensive prompts at the input stage; this work addresses the issue from the perspective of restoring internal model parameters.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first to systematically investigate safety degradation in post-pruned LVLMs and propose remedies.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers six models, three pruning methods, and includes detailed ablations and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with strong motivation and detailed analysis.
- Value: ⭐⭐⭐⭐ — Provides a practical and lightweight solution for the safety of models in compression and deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models](effivlm_bench_vlm_acceleration.md)
- [\[CVPR 2026\] Accelerating Streaming Video Large Language Models via Hierarchical Token Compression](../../CVPR2026/vlm_efficiency/accelerating_streaming_video_large_language_models_via_hierarchical_token_compre.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/vlm_efficiency/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)
- [\[ICCV 2025\] Growing a Twig to Accelerate Large Vision-Language Models](../../ICCV2025/vlm_efficiency/growing_a_twig_to_accelerate_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
