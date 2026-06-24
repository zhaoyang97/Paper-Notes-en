---
title: >-
  [Paper Note] Distillation of Large Language Models via Concrete Score Matching
description: >-
  [ICLR 2026][Model Compression][Knowledge Distillation] Concrete Score Distillation (CSD) is proposed as a knowledge distillation loss for LLMs based on discrete score matching. By matching the relative logit differences between student and teacher across all vocabulary pairs, it concurrently overcomes the issues of softmax smoothing and the restricted solution space inherent in direct logit distillation.
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "LLM Compression"
  - "Score Matching"
  - "Logit Distillation"
  - "Discrete Score Matching"
date: 2026-05-08
content_hash: 73ee68895eaaf4b6
---

# Distillation of Large Language Models via Concrete Score Matching

**Conference**: ICLR 2026  
**arXiv**: [2509.25837](https://arxiv.org/abs/2509.25837)  
**Code**: [GitHub](https://github.com/aailab-kaist/CSD)  
**Area**: Model Compression / Knowledge Distillation  
**Keywords**: Knowledge Distillation, LLM Compression, Score Matching, Logit Distillation, Discrete Score Matching

## TL;DR

Concrete Score Distillation (CSD) is proposed as a knowledge distillation loss for LLMs based on discrete score matching. By matching the relative logit differences between student and teacher across all vocabulary pairs, it concurrently overcomes the issues of softmax smoothing and the restricted solution space inherent in direct logit distillation.

## Background & Motivation

The deployment and inference costs of LLMs are extremely high, making knowledge distillation (KD) a core technique for transferring capabilities from large models to smaller ones. Existing KD methods primarily align student and teacher distributions through softmax probability matching (e.g., KL divergence). However, the softmax transformation severely smoothes logit information; with large vocabularies (e.g., 128K), most token probabilities approach zero (only 0.0023% > 0.01), rendering the teacher's rich knowledge nearly indistinguishable after softmax.

Direct Logit Distillation (DLD), which matches raw logits via MSE, avoids softmax smoothing but suffers from another critical flaw: **logit shift invariance**. Since softmax only considers the relative differences between logits, the probabilities remain identical if $f_\theta[y_t] = f_T[y_t] + C$ holds for all tokens. However, the MSE loss in DLD is non-zero in such cases, artificially restricting the optimal solution space. This limitation is particularly detrimental when there is a significant capacity gap between the teacher and student.

The Core Idea: Inspired by the concept of score matching in energy-based models to bypass normalization constraints, this work introduces discrete concrete score matching into LLM distillation. It designs a logit-level distillation objective that is neither affected by softmax smoothing nor restricted in its solution space.

## Method

### Overall Architecture

CSD changes the distillation objective from "aligning probabilities" to "aligning logit residuals." For each token position, instead of directly matching the teacher's and student's logit values, it matches the relative logit differences $f[x] - f[y_t]$ between all vocabulary pairs $(y_t, x)$. This differential structure corresponds to the logarithmic form of the concrete score in discrete diffusion, bypassing softmax smoothing and naturally eliminating the shift-sensitivity issues of DLD.

### Key Designs

**1. Log-transformed concrete score matching: Replacing probability ratios with stable logit differences**

The concrete score itself is a probability ratio $q_\theta(x)/q_\theta(y_t)$. Direct matching causes the loss to explode and training to diverge when the denominator $q_\theta(y_t)$ approaches zero. CSD applies a logarithm to the probability ratio, transforming the matching target from a "ratio" to a "difference," resulting in a clean logit-level quadratic loss:

$$\mathcal{L}_{\text{CSD}} = \frac{1}{2} \sum_{y_t \in \mathcal{V}} \sum_{x \in \mathcal{V}} w(y_t, x) \left( f_\theta[x] - f_\theta[y_t] - f_T[x] + f_T[y_t] \right)^2$$

The log transformation achieves two goals: it eliminates the need to calculate overflow-prone probability ratios, ensuring stable training; simultaneously, the loss naturally resides in the logit space, completely skipping softmax and preventing the teacher's knowledge from being erased by large-vocabulary probability suppression.

**2. Larger optimal solution space: Differential structure naturally tolerates logit shifts**

DLD uses MSE to align logits directly, but softmax only cares about relative differences. When $f_\theta[y_t] = f_T[y_t] + C$ (a constant $C$ added to all tokens), the probabilities are identical, yet DLD yields a non-zero loss, thus artificially shrinking the set of reachable optimal solutions for the student. The CSD loss only considers the difference $f[x] - f[y_t]$, so the shift term $C$ cancels out, resulting in zero loss. Consequently, its optimal solution set strictly contains that of DLD (Theorem 2: $\Theta_{\text{CSD}}^* \supsetneq \Theta_{\text{DLD}}^*$). This "extra search freedom" becomes more valuable as the student's capacity becomes more limited relative to the teacher.

**3. Compressing $O(|\mathcal{V}|^2)$ double summation to $O(|\mathcal{V}|)$**

Directly calculating the double summation over a 128K vocabulary is infeasible. CSD assumes the weight function is decomposable as $w(y_t, x) = w_1(y_t) \cdot w_2(x)$. Under this decomposition, the gradient can be expanded analytically: first perform weighted centering/normalization on the logits, then calculate the weighted sum of residuals. The entire gradient can be computed in $O(|\mathcal{V}|)$ time (Theorem 3). This step is critical for transitioning CSD from a theoretical concept to a practical training objective.

**4. Adjustable weights to switch between fidelity and diversity**

$w_1$ and $w_2$ can be set to the teacher's probability (T), the student's probability (S), or a uniform distribution (U). Different combinations allow CSD to slide between mode-seeking and mode-covering behaviors: $(S,S)$ provides the highest fidelity to the teacher, while $(U,S)$ and $(T,S)$ introduce more distributional mass, improving diversity and probability calibration. This provides a unified knob to express preferences that previously required different divergences like KL, RKL, or SKL.

### Loss & Training

The gradient structure of CSD is similar to KL divergence, with the key difference being the use of **centered normalization** instead of softmax normalization; this is precisely how it bypasses softmax smoothing. It is orthogonal to on-policy distillation techniques (ImitKD, GKD, DistiLLM) and can be used in conjunction with them. For non-decomposable joint weight functions, CSD also supports Monte Carlo gradient estimation, though the analytic gradient in the decomposable case converges faster and performs better, making it the default choice.

## Key Experimental Results

### Main Results

**Table 1: GPT-2-1.5B → GPT-2-0.1B Loss Function Comparison (ROUGE-L)**

| Loss Function | Dolly | Self-Instruct | Vicuna | Super-NI | UnNI | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| KL | 23.52 | 10.02 | 14.57 | 16.76 | 18.55 | 16.68 |
| RKL | 24.26 | 11.19 | 15.80 | 20.17 | 22.99 | 18.88 |
| SRKL | 24.53 | 12.19 | 15.63 | 23.37 | 24.28 | 20.00 |
| **CSD (Ours)** | **24.94** | **12.06** | **15.78** | **24.60** | **25.88** | **20.65** |

**Table 2: Task-Specific Distillation Gemma-7B-IT → Gemma-2B-IT**

| Loss | Summarization ROUGE-L | Translation COMET | GSM8K Acc |
| :--- | :--- | :--- | :--- |
| KL | 35.02 | 73.96 | 24.03 |
| **CSD (T,S)** | **35.67** | **74.14** | **25.78** |
| RKL | 0.00 | 45.02 | 0.00 |

### Ablation Study

*   **CSD vs DLD under identical weights**: CSD (T,T)/(U,U)/(S,S) consistently outperforms DLD T/U/S, validating the advantage of a larger solution space.
*   **Weight Selection**: (S,S) yields the highest fidelity, (U,S) provides the best diversity-fidelity trade-off, and (T,S) offers the best probability calibration.
*   **General Dialogue Distillation**: On Qwen2.5-7B→1.5B, MT-Bench increased from 5.75 (DistiLLM-2) to 5.95 (CSD).
*   **Analytic Gradient vs Monte Carlo**: Analytic computation converges slightly faster and achieves superior performance.

### Key Findings

*   Softmax results in the loss of significant teacher knowledge in large-vocab LLMs (99.99%+ tokens have probability < 0.01).
*   Logit shift invariance is a core deficiency of DLD, which CSD solves naturally through its differential structure.
*   Mode-seeking losses (RKL, SKL) are prone to collapse in low-data distillation; CSD avoids this through weight selection.
*   CSD is orthogonal and complementary to on-policy methods, showing the largest gains in pure on-policy settings.

## Highlights & Insights

*   Re-examines the LLM distillation problem from an energy-based model perspective; the approach is novel and theoretically grounded.
*   The analysis of logit shift invariance is concise and powerful, directly addressing the fundamental flaw of DLD.
*   The weight function design space provides a unified framework for understanding mode-seeking and mode-covering behaviors.
*   Analytic gradient calculation reduces the complexity of CSM from $O(|\mathcal{V}|^2)$ to a practically usable $O(|\mathcal{V}|)$.

## Limitations & Future Work

*   Exploration of the weight function space remains limited; the design of joint weight functions has not been fully explored.
*   The decomposability assumption $w(y_t,x)=w_1(y_t)w_2(x)$ restricts the expressive power of the weights.
*   Validation on larger-scale models (>10B teacher) is insufficient.
*   Integration with other compression methods such as quantization and pruning has not been investigated.

## Related Work & Insights

*   **DistiLLM** (Ko et al., 2024): Proposed SKL/SRKL smoothed KL distillation; the current work improves upon this foundation.
*   **GKD** (Agarwal et al., 2024): Uses the f-divergence framework; CSD can be seen as a new framework transcending probability matching.
*   **Concrete Score Matching** (Meng et al., 2022): Developed for discrete diffusion models; this work innovatively adapts it for autoregressive LLM distillation.
*   Insight: The "bypass normalization" idea from score matching could find applications in other NLP tasks requiring distribution matching.

## Rating

*   Novelty: ⭐⭐⭐⭐ Creative transfer from EBM score matching to LLM distillation, though the core remains logit differential matching.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple backbones, task types, and on-policy combinations with comprehensive ablations.
*   Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear illustrations, and sound motivational analysis.
*   Value: ⭐⭐⭐⭐ Provides a unified framework for logit distillation design; practical performance gains are stable though modest in magnitude.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Knowledge Distillation for Large Language Models through Residual Learning](knowledge_distillation_for_large_language_models_through_residual_learning.md)
- [\[CVPR 2026\] Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals](../../CVPR2026/model_compression/phased_dmd_few-step_distribution_matching_distillation_via_score_matching_within.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] Entropy-Based Block Pruning for Efficient Large Language Models](entropy-based_block_pruning_for_efficient_large_language_models.md)

</div>

<!-- RELATED:END -->
