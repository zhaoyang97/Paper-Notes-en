---
title: >-
  [Paper Note] ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence
description: >-
  [ICML 2025 Spotlight][Model Compression][Knowledge Distillation] This paper provides an in-depth analysis of the probability mass allocation deficiencies of FKLD and RKLD in knowledge distillation, finding that they represent extremes in two effects: Hardness-Concentration and Confidence-Concentration. Based on this, the ABKD framework utilizing $\alpha$-$\beta$-divergence is proposed to flexibly balance these two effects by tuning $\alpha$ and $\beta$…
tags:
  - "ICML 2025 Spotlight"
  - "Model Compression"
  - "Knowledge Distillation"
  - "α-β-Divergence"
  - "Probability Mass Allocation"
  - "Distribution Matching"
  - "Divergence Function"
date: 2026-05-08
content_hash: 0315c69bcf4756ca
---

# ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2505.04560](https://arxiv.org/abs/2505.04560)  
**Code**: [ghwang-s/abkd](https://github.com/ghwang-s/abkd)  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, α-β-Divergence, Probability Mass Allocation, Distribution Matching, Divergence Function

## TL;DR

This paper provides an in-depth analysis of the probability mass allocation deficiencies of FKLD and RKLD in knowledge distillation, finding that they represent extremes in two effects: Hardness-Concentration and Confidence-Concentration. Based on this, the ABKD framework utilizing $\alpha$-$\beta$-divergence is proposed to flexibly balance these two effects by tuning $\alpha$ and $\beta$, achieving SOTA performance across 17 language/vision datasets and 12 teacher-student configurations.

## Background & Motivation

### Background
Knowledge Distillation (KD) is a classic technique for transferring knowledge from a large teacher model to a small student model. The core operation is to minimize the divergence $\mathbb{D}(p \| q_\theta)$ between the teacher distribution $p$ and the student distribution $q_\theta$. The current mainstream choices are forward KL divergence (FKLD) and reverse KL divergence (RKLD).

### Limitations of Prior Work
- **FKLD** produces an over-smoothed student distribution and treats the matching errors of all classes uniformly, failing to guide the student to focus on the target class and leading to incorrect predictions.
- **RKLD** causes the student to over-focus on the target class, ignoring the soft label information in the teacher distribution and degenerating into a supervision similar to one-hot labels.
- The deficiencies of both arise from the extreme choices in the probability mass allocation method, but a systematic theoretical analysis has been lacking until now.

### Key Challenge
How to balance focusing on difficult classes (Hardness-Concentration) and focusing on high-confidence classes (Confidence-Concentration)? FKLD is too weak on both, while RKLD is too strong on both, leaving no middle ground.

### Key Insight
By tracking the log mass ratio $\mathsf{LogR}$ during the gradient update process, this paper analyzes how different divergence functions affect the reallocation of probability mass, reveals the co-acting mechanisms of both concentration effects, and subsequently introduces α-β-divergence as a unified framework.

## Method

### Overall Architecture

The training objective of ABKD is defined as:

$$\ell = \ell_{\text{CE}} + \lambda \cdot \mathbb{D}_{\text{AB}}^{(\alpha,\beta)}(p \| q_\theta)$$

Where $\mathbb{D}_{\text{AB}}^{(\alpha,\beta)}$ is the α-β-divergence, controlling the probability mass allocation by adjusting the hyperparameters α and β.

### Key Designs

#### 1. Log Mass Ratio Analysis Framework

The monitoring quantity is defined as $\mathsf{LogR}_t^{\mathcal{A}}(y) = \log\frac{q_{t+1}^{\mathcal{A}}(y)}{q_t(y)}$, which is proportional to the logit gradient. By analyzing the upper bound of $|\mathsf{LogR}|$, two concentration effects are revealed:

- **Hardness-Concentration** (term b): $|s(p(k)) - s(q_t(k))|$, measuring the matching error. The sharper the function $s$ is, the stronger the effect becomes.
- **Confidence-Concentration** (term a): $q_t(y)^\beta$, weighted by student confidence. A larger $\beta$ leads to more focus on high-confidence classes.

**FKLD**: $s(x) = x$ (linear, weak Hardness), $\beta = 0$ (no Confidence weighting) $\to$ both are too weak.  
**RKLD**: $s(x) = \log(x)$ (logarithmic, strong Hardness), $\beta = 1$ (strong Confidence weighting) $\to$ both are too strong.

#### 2. Definition of α-β-Divergence

$$\mathbb{D}_{\text{AB}}^{(\alpha,\beta)}(p \| q) = -\frac{1}{\alpha\beta} \sum_k \left[ p(k)^\alpha q(k)^\beta - \frac{\alpha}{\alpha+\beta} p(k)^{\alpha+\beta} - \frac{\beta}{\alpha+\beta} q(k)^{\alpha+\beta} \right]$$

This is a general family of divergences that unifies multiple known divergences:

| Divergence | α | β |
|------|---|---|
| FKLD | 1 | 0 |
| RKLD | 0 | 1 |
| Hellinger Distance | 0.5 | 0.5 |
| Euclidean Distance | 1 | 1 |
| α-divergence | α+β=1 |
| β-divergence | α=1 |

#### 3. Balancing Mechanism

The upper bound of LogR for α-β-divergence is:
$$|\mathsf{LogR}_t^{(\alpha,\beta)}(y)| \leq \eta \cdot q_t(y)^{\beta} \cdot \left|\frac{p(y)^\alpha - q_t(y)^\alpha}{\alpha}\right| + \ldots$$

- **β controls Confidence-Concentration**: $\beta \to 0$ approaches the FKLD effect, $\beta \to 1$ approaches the RKLD effect.
- **α controls Hardness-Concentration**: $\alpha \to 1$ approaches the FKLD effect, $\alpha \to 0$ approaches the RKLD effect.

By choosing appropriate α and β, continuous interpolation between the two effects can be achieved to avoid extreme scenarios.

### Loss & Training

- Final loss: $\ell = \ell_{\text{CE}} + \lambda \cdot \mathbb{D}_{\text{AB}}^{(\alpha,\beta)}(p \| q_\theta)$
- Only the loss function needs to be modified, introducing no additional trainable parameters.
- With α and β as hyperparameters, the paper provides a detailed tuning guide.

## Key Experimental Results

### Main Results: LLM Instruction Tuning (GPT-2 XL → GPT-2)

| Method | Dolly Eval | Self-Instruct | Vicuna Eval | Super-Natural | Unnatural |
|------|-----------|---------------|-------------|---------------|-----------|
| SFT | 23.14 | 10.22 | 15.15 | 17.41 | 19.76 |
| KD (FKLD) | 23.80 | 10.01 | 15.25 | 17.69 | 18.99 |
| MiniLLM (RKLD) | 24.62 | 12.49 | 17.30 | 23.76 | 24.30 |
| DISTILLM | 25.32 | 11.65 | 16.76 | 23.52 | 25.79 |
| **ABKD** | **25.65** | **13.47** | 16.06 | **26.47** | **29.32** |

On the 1.5B $\to$ 0.1B distillation configuration, ABKD improves the ROUGE-L score by 0.81 to 3.31 points compared to FKLD/RKLD.

### Ablation Study: Visualization of the α-β Hyperparameter Space

Figure 1(a) of the paper displays a performance heatmap in the 2D search space of $(\alpha, \beta)$ for ABKD. FKLD and RKLD are merely special points within this space, and $\alpha$-divergence is restricted to searching along the $\alpha+\beta=1$ submanifold, whereas ABKD provides a complete 2D search space.

### Key Findings

1. ABKD outperforms or matches SOTA methods across all 12 teacher-student configurations.
2. It remains effective on visual classification tasks, proving the generality of the framework.
3. ABKD can further enhance existing KD methods; simply modifying their loss functions yields additional gains.

## Highlights & Insights

1. **Excellent Theoretical Depth**: Through the LogR analysis framework, the empirical deficiencies of FKLD/RKLD are attributed to the extreme combinations of two concentration effects, providing a unified theoretical explanation.
2. **Minimalist Implementation**: Only the loss function needs to be modified, requiring no extra parameters or data augmentation, to significantly improve distillation performance.
3. **Unified Framework**: The α-β-divergence unifies a large number of known divergences, and ABKD naturally generalizes the entire space of divergence choices.
4. **Strong Practical Guidance**: The paper provides a detailed hyperparameter tuning guide, lowering the barrier to adoption.
5. **Deficiency Analysis vs. WSD and JSD**: The weighted sum of FKLD and RKLD (WSD) is unstable at extreme probability ratios, and JSD suffers from gradient vanishing when distributions are far apart, whereas α-β-divergence naturally avoids these issues.

## Limitations & Future Work

1. **Hyperparameter Search Cost**: The two hyperparameters, α and β, require additional tuning; although guidelines are provided, it still increases the tuning overhead.
2. **Lack of Adaptive Mechanisms**: α and β remain fixed throughout training, lacking dynamic adjustment strategies (e.g., automatically adjusting according to the training phase).
3. **Token-level Distillation Constraints**: In LLMs, only token-level distribution matching is considered, without involving sequence-level distillation.
4. **Theoretical Analysis Relies on Simplified Assumptions**: The softmax approximation used in the LogR analysis may exhibit bias in practical deep networks.

## Related Work & Insights

- **FKLD Series**: Traditional KD (Hinton 2015), SeqKD (Kim & Rush 2016) $\to$ over-smoothed
- **RKLD Series**: MiniLLM (Gu et al. 2024), Kim et al. 2024 $\to$ over-focused
- **Hybrid Methods**: GKD (Agarwal et al. 2024) uses JSD, DISTILLM (Ko et al. 2024) uses Skewed KLD $\to$ heuristic mitigation but lacks a unified framework
- **Insights**: When choosing divergence functions, researchers should focus on their impact on probability mass allocation, rather than considering it solely from an information-theoretic perspective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formulates a unified analysis of divergence functions from the perspective of probability mass allocation; highly novel theoretical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Thorough coverage across 17 datasets, 12 configurations, spanning both language and vision.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations, intuitive illustrations, and rigorous logic.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play general KD framework with substantial theoretical and practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)
- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](../../ICCV2025/model_compression/ea-kd_entropy-based_adaptive_knowledge_distillation.md)
- [\[CVPR 2025\] What Makes a Good Dataset for Knowledge Distillation?](../../CVPR2025/model_compression/what_makes_a_good_dataset_for_knowledge_distillation.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ACL 2025\] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](../../ACL2025/model_compression/sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)

</div>

<!-- RELATED:END -->
