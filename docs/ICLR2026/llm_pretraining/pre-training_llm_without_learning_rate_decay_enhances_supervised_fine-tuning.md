---
title: >-
  [Paper Note] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning
description: >-
  [ICLR 2026][LLM Pretraining][Learning rate scheduling] This paper proposes the Warmup-Stable-Only (WSO) learning rate schedule—completely eliminating the decay phase during pre-training. Despite yielding worse pre-traini…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Learning rate scheduling"
  - "pre-training"
  - "supervised fine-tuning"
  - "loss landscape"
  - "Warmup-Stable-Only"
date: 2026-05-08
content_hash: bbefb933c69d3c75
---

# Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning

**Conference**: ICLR 2026
**arXiv**: [2603.16127](https://arxiv.org/abs/2603.16127)  
**Code**: Not open-sourced  
**Area**: LLM Pre-training
**Keywords**: Learning rate scheduling, pre-training, supervised fine-tuning, loss landscape, Warmup-Stable-Only

## TL;DR

This paper proposes the Warmup-Stable-Only (WSO) learning rate schedule—completely eliminating the decay phase during pre-training. Despite yielding worse pre-training metrics, WSO consistently outperforms all decay-based schedules after SFT. Loss landscape analysis reveals that WSO's advantage stems from maintaining flatter minima.

## Background & Motivation

### State of the Field

Learning rate (LR) scheduling is one of the most critical yet operationally complex hyperparameters in LLM pre-training. Mainstream approaches include:

- **Cosine decay**: The most widely used schedule since GPT-3, where LR decays along a cosine curve toward near zero.
- **Linear decay**: Recent studies suggest that linear decay to zero achieves lower pre-training loss.
- **WSD (Warmup-Stable-Decay)**: Applies a brief decay only at the end of training; more flexible and adopted by models such as MiniCPM.

These strategies share a common objective: decaying LR toward the end of training to optimize pre-training metrics.

### Root Cause

Existing LR strategies optimize for pre-training performance $\mathtt{Task}_{\rm pre}(M_{\rm pre})$, whereas in practice the critical objective is post-SFT performance $\mathtt{Task}_{\rm post}(M_{\rm post})$.

Recent work (Sun & Dredze 2025; Springer et al. 2025) explicitly demonstrates that **a model with better pre-training performance does not necessarily perform better after SFT**. This raises a fundamental question: when a model is intended to undergo SFT, is LR decay—chosen to optimize pre-training metrics—still the optimal choice?

### Formal Definition

The conventional pipeline greedily optimizes each stage independently:

$$\widehat{M}_{\rm pre} = \arg\max_{M_{\rm pre} \in \mathcal{M}_{\rm pre}} \{\mathtt{Task}_{\rm pre}(M_{\rm pre}[M_{\rm rand}])\}$$

The ideal objective should instead be a joint global optimization:

$$\widehat{M}_{\rm post} = \arg\max_{(M_{\rm pre}, M_{\rm post}) \in (\mathcal{M}_{\rm pre}, \mathcal{M}_{\rm post})} \{\mathtt{Task}_{\rm post}(M_{\rm post}[M_{\rm pre}[M_{\rm rand}]])\}$$

## Method

### Overall Architecture

The paper evaluates four LR schedulers—WSO, WSD, Cosine, and Linear—under both two-stage (pre-training + SFT) and three-stage (pre-training + mid-training + SFT) settings.

### Warmup-Stable-Only (WSO) Definition

WSO is a minimalist variant of WSD that directly removes the decay phase (setting $\alpha_{\text{pre}}=1.0$):

$$\eta^{\text{WSO}}(t, \alpha_{\text{pre}}) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\ \eta_{\max} & T_{\text{warmup}} < t \leq T_{\text{pre}} \end{cases}$$

In contrast to WSD's three-phase schedule (warmup → stable → decay), WSO retains only two phases: warmup → stable.

### Key Design: Min LR Factor Parameterization

All schedulers are uniformly parameterized by a minimum LR factor $\alpha_{\text{pre}}$:

- $\alpha_{\text{pre}} = 0.0$: decay to zero (most aggressive)
- $\alpha_{\text{pre}} = 0.1$: decay to 10% of peak LR (commonly used in Llama 3, OLMo 2, etc.)
- $\alpha_{\text{pre}} = 1.0$: no decay, i.e., WSO

### Mid-training LR Schedule

For the three-stage setting, $\alpha_{\text{mid}}$ is introduced to control LR decay during mid-training:

- $\alpha_{\text{mid}} = 0.0$: linear decay to zero during mid-training
- $\alpha_{\text{mid}} = 1.0$: constant LR throughout mid-training

### Loss Landscape Analysis

To explain why WSO performs better after SFT, the paper measures loss landscape flatness via Hessian trace (sharpness):

$$\text{Sharpness}(\theta_t) = \text{Tr}(\mathbf{H}_{\mathcal{L}}(\theta_t)) = \sum_{i=1}^{d} \frac{\partial^2 \mathcal{L}(\theta_t; \mathcal{D})}{\partial \theta_i^2}$$

This is computed efficiently using the Hutchinson unbiased estimator. The key finding is that WSO maintains lower sharpness (flatter minima), while decay-based schedules lead to 2–3× higher sharpness.

## Key Experimental Results

### Main Results: Two-Stage Setting (Pre-training + SFT)

Model architectures: 1B and 8B (Llama 3 series); pre-training data: FineWeb-Edu; SFT data: Tulu-3 SFT mixture.

| Model | Scheduler | $\alpha_{\text{pre}}$ | PT Valid Loss ↓ Δ | PT Task Avg Δ | SFT Task Avg Δ |
|------|--------|----------------------|-------------------|---------------|----------------|
| 1B | **WSO** | 1.0 | +0.071 | -1.7 | **+0.3** |
| 1B | WSD | 0.1 | +0.004 | -1.5 | +0.0 |
| 1B | WSD | 0.0 | +0.000 | -1.2 | -1.0 |
| 1B | Linear | 0.0 | +0.016 | +0.0 | -0.9 |
| 1B | Cosine | 0.1 | +0.019 | -0.1 | -0.7 |
| 8B | **WSO** | 1.0 | +0.127 | -0.8 | **+1.1** |
| 8B | WSD | 0.1 | +0.019 | -0.2 | -0.8 |
| 8B | WSD | 0.0 | +0.014 | +0.0 | -0.3 |
| 8B | Linear | 0.0 | +0.000 | -1.8 | +0.0 |

**Key finding**: WSO yields the worst pre-training loss (0.127 higher for 8B) but achieves the best post-SFT performance (1.1 points higher for 8B).

### Three-Stage Setting (Pre-training + Mid-training + SFT)

| Model | Scheduler | $\alpha_{\text{pre}}$ | $\alpha_{\text{mid}}$ | MT Task Avg Δ | SFT Task Avg Δ |
|------|--------|----------------------|-----------------------|---------------|----------------|
| 1B | **WSO** | 1.0 | 1.0 | -0.1 | **+0.8** |
| 1B | WSD | 1.0 | 0.0 | +0.0 | +0.0 |
| 1B | Cosine | 0.1 | 0.0 | -3.1 | -3.7 |
| 8B | **WSO** | 1.0 | 1.0 | -2.1 | **+1.1** |
| 8B | WSD | 1.0 | 0.0 | +0.0 | -1.4 |
| 8B | Linear | 0.1 | 0.0 | -9.0 | -3.7 |

### Ablation Study: Over-training Setting (2T tokens)

| Model | Scheduler | $\alpha_{\text{pre}}$ | PT Task Avg Δ | SFT Task Avg Δ |
|------|--------|----------------------|---------------|----------------|
| 1B | **WSO** | 1.0 | -1.5 | **+0.7** |
| 1B | WSD | 0.1 | +0.0 | +0.0 |
| 1B | WSD | 0.0 | +0.0 | -0.3 |

Under over-training combined with mid-training (2T + 500B tokens), WSO's advantage is even larger: SFT Task Avg Δ reaches **+1.4**.

### Key Findings

1. **Performance reversal**: The scheduler with the best pre-training performance (decay to zero) performs worst after SFT.
2. **WSO dominates across the board**: Consistently achieves the best results across 1B/8B, two-stage/three-stage, and standard/over-training settings.
3. **Decay at any stage is harmful**: In the three-stage setting, applying decay even only during mid-training degrades SFT performance.
4. **Sharpness is negatively correlated with SFT performance**: Pearson correlation $r=-0.709$.

## Highlights & Insights

1. **Counter-intuitive core finding**: Better pre-training loss ≠ better downstream performance; LR decay actually impairs model adaptability.
2. **Clear theoretical explanation**: Loss landscape analysis provides a complete causal chain from flat minima to better post-SFT performance.
3. **Minimal implementation complexity**: WSO is simpler than any decay strategy—no decay ratio or decay phase length to tune.
4. **High practical value**: The paper recommends that open-source models be trained and released using WSO to maximize adaptability for downstream users.
5. **Consistent across scales**: Findings hold across 1B to 8B model sizes and 100B to 2T token training scales.

## Limitations & Future Work

1. Only SFT is examined as a post-training method; alignment techniques such as DPO and RLHF are not evaluated.
2. Experiments are limited to 8B parameters; applicability to larger models (70B+) remains to be verified.
3. WSO incurs significantly higher pre-training loss, which may be unsuitable for scenarios requiring low pre-training loss (e.g., distillation).
4. The sample size for the sharpness–SFT performance correlation analysis is relatively small.

## Related Work & Insights

- **Bergsma et al. 2025**: Advocates linear decay to zero as optimal—however, this holds only for pre-training loss.
- **WSD (Hu et al. 2024)**: WSO can be viewed as an extreme simplification of WSD, echoing WSD's flexibility advantage.
- **Wen et al. 2025**: Theoretical analysis of WSD finds that the decay phase increases sharpness, a problem WSO avoids entirely.
- **Insight**: Future work should select training strategies based on the ultimate deployment objective (post-SFT/RLHF performance) rather than pre-training metrics.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Challenges the widely held belief that "lower decay is better"; the argument is clear and compelling.
- **Theoretical Depth**: ⭐⭐⭐⭐ — Loss landscape analysis and formal framework are thorough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 2 model scales × 4 schedulers × 3 training settings × over-training; extremely comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ — Provides directly actionable recommendations with significant implications for LLM training and model release strategies.
- **Overall**: ⭐⭐⭐⭐☆ — Rigorous experiments, counter-intuitive yet practically useful conclusions; an important contribution to pre-training strategy research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](../../ICML2026/llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICLR 2026\] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](common_corpus_ethical_data_for_llm_pretraining.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/llm_pretraining/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)

</div>

<!-- RELATED:END -->
