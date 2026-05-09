---
title: >-
  [Paper Note] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs
description: >-
  [ICLR 2026][LLM Alignment][DPO improvement] Uni-DPO is proposed to unify dynamic reweighting of preference pairs via three components — quality-aware weighting (prioritizing pairs with large score margins), performance-aware weighting (focal loss focusing on underfitted samples), and a calibrated NLL loss — consistently outperforming DPO/SimPO on text understanding and mathematical reasoning benchmarks, with Gemma-2-9B achieving 67.1% on Arena-Hard, surpassing Claude 3 Opus (60.4%).
tags:
  - ICLR 2026
  - LLM Alignment
  - DPO improvement
  - dynamic weighting
  - quality-aware
  - focal loss
  - preference optimization
date: 2026-05-08
content_hash: 20a78219ce75d8e2
---

# Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs

**Conference**: ICLR 2026
**arXiv**: [2506.10054](https://arxiv.org/abs/2506.10054)
**Code**: [https://github.com/pspdada/Uni-DPO](https://github.com/pspdada/Uni-DPO)
**Area**: Alignment RLHF / DPO
**Keywords**: DPO improvement, dynamic weighting, quality-aware, focal loss, preference optimization

## TL;DR
Uni-DPO is proposed to unify dynamic reweighting of preference pairs via three components — quality-aware weighting (prioritizing pairs with large score margins), performance-aware weighting (focal loss focusing on underfitted samples), and a calibrated NLL loss — consistently outperforming DPO/SimPO on text understanding and mathematical reasoning benchmarks, with Gemma-2-9B achieving 67.1% on Arena-Hard, surpassing Claude 3 Opus (60.4%).

## Background & Motivation

**State of the Field**: DPO optimizes policies directly from preference data via implicit rewards, and has become a standard approach for LLM alignment. SimPO further simplifies this by removing the reference model.

**Limitations of Prior Work**:
   - Standard DPO treats all preference pairs equally, despite large variance in data quality — high-quality pairs exhibit clear chosen/rejected distinctions, while low-quality pairs are noisy or ambiguous.
   - A mismatch exists between data quality and model learning state: high-quality pairs may already be well-learned, and over-emphasizing them leads to overfitting.
   - DPO lacks fine-grained external reward signals (unlike PPO/GRPO).

**Root Cause**: How to dynamically reweight preference pairs by jointly considering both the intrinsic quality of data and the model's current learning state?

**Core Idea**: Quality weights differentiate high- and low-quality data; performance weights focus on hard samples; calibrated NLL loss prevents the probability of chosen responses from degrading.

## Method

### Overall Architecture
$$\mathcal{L}_{\text{Uni-DPO}} = -\mathbb{E}[w_{\text{qual}}(y_w, y_l) \cdot w_{\text{perf}}(\pi_\theta) \cdot \log\sigma(\Delta_r)] + \lambda\mathcal{L}_{\text{c-NLL}}$$

### Key Designs

1. **Quality-Aware Weight $w_{\text{qual}}$**:

    - Function: Assigns weights based on external score differences, giving higher weight to pairs with larger score margins.
    - $w_{\text{qual}}(y_w, y_l) = \sigma(\eta \cdot (S_w - S_l))$
    - $S_w, S_l$ are sourced from human annotations, GPT-4, or reward models.
    - Effect: Filters noisy/ambiguous preference pairs and retains high signal-to-noise data.

2. **Performance-Aware Weight $w_{\text{perf}}$ (Calibrated Focal)**:

    - Function: Down-weights already well-learned samples and focuses training on hard samples where the current model underperforms.
    - $w_{\text{perf}} = [1 - \sigma(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \tau_{\text{ref}})]^\gamma$
    - Key improvement: A fixed threshold $\tau_{\text{ref}}$ replaces reference model dependency, avoiding per-sample constraints that cause training instability; length normalization (LN) is incorporated to prevent length bias.
    - $\gamma$ controls focal intensity; $\tau_{\text{ref}}$ controls the expected margin.

3. **Calibrated NLL Loss $\mathcal{L}_{\text{c-NLL}}$**:

    - Function: Prevents the absolute probability of chosen responses from decreasing during DPO training.
    - Activated only when the policy underperforms the reference model and the sample is of high quality.
    - Reinforces the model's confidence on difficult, high-quality positive samples.

### Loss & Training
- $\eta = 0.7$, $\lambda = 0.001$, $\gamma = 3.0$, $\tau_{\text{ref}} \in [0.5, 2.0]$
- Supports diverse quality score sources (human annotation, GPT-4, ArmoRM, and other reward models).

## Key Experimental Results

### Main Results: Text Understanding

| Model | Method | AlpacaEval2 LC | Arena-Hard | IFEval Loose | SedarEval |
|------|------|---------------|-----------|-------------|-----------|
| Llama3-8B-Base | DPO | 15.5 | 15.9 | 45.5 | 31.80 |
| | SimPO | 19.4 | 23.4 | 45.7 | 32.43 |
| | **Uni-DPO** | **23.8** | **23.9** | **47.9** | **38.49** |
| Gemma-2-9B-IT | SimPO | 53.2 | 59.1 | 67.7 | 57.7 |
| | **Uni-DPO** | **54.7** | **67.1** | **72.8** | 57.5 |

### Main Results: Mathematical Reasoning (Qwen2.5-Math-7B)

| Method | GSM8K | MATH | AIME24 | AMC23 | Avg |
|------|-------|------|--------|-------|-----|
| Baseline | 64.3 | 65.8 | 23.3 | 47.5 | 39.11 |
| DPO | 83.2 | 75.8 | 26.7 | 57.5 | 51.55 |
| SimPO | 85.7 | 76.4 | 26.7 | 57.5 | 53.73 |
| **Uni-DPO** | **88.9** | **78.2** | **26.7** | **67.5** | **56.80** |

### Ablation Study

| Configuration | AlpacaEval2 WR | Arena-Hard | SedarEval |
|------|---------------|-----------|-----------|
| Full Uni-DPO | **20.5** | **23.9** | **38.49** |
| w/o $w_{\text{qual}}$ | 15.9 | 22.8 | 37.43 |
| w/o $w_{\text{perf}}$ | 18.5 | 21.4 | 40.46 |
| w/o LN | 3.8 | 2.7 | 28.18 |
| w/o $\mathcal{L}_{\text{c-NLL}}$ | 19.4 | 23.3 | 37.73 |

### Key Findings
- **Length normalization (LN) is critical**: Removing it causes a sharp performance collapse (SedarEval −10.31) and training instability.
- **Quality weighting most affects AlpacaEval**: Removing it drops WR from 20.5 to 15.9 (−4.6).
- **Gemma-2-9B + Uni-DPO surpasses Claude 3 Opus**: Arena-Hard 67.1 vs. 60.4.
- **Significant gains on mathematical reasoning**: Qwen2.5-Math-7B achieves an average of +3.07 over SimPO.

## Highlights & Insights
- **Unified dual-perspective dynamic weighting**: The joint consideration of data quality (external signal) and learning difficulty (internal dynamics) is more effective than either perspective alone.
- **Improved calibrated focal loss design**: Replacing reference model dependency with a fixed threshold and incorporating length normalization resolves the training instability of naive focal DPO.
- **Transferability to mathematical reasoning**: The framework generalizes beyond dialogue and instruction following, yielding consistent gains on mathematical tasks.

## Limitations & Future Work
- **Dependency on external scores**: Quality weights require reward model or GPT-4 scoring, increasing data preparation cost.
- **Many hyperparameters**: $\eta, \gamma, \tau_{\text{ref}}, \lambda, \tau_{\text{good}}$ all require tuning.
- **Potential directions**: Self-reward could replace external scoring; null-space constraints from NSPO could be integrated to add a safety dimension.

## Related Work & Insights
- **vs. DPO**: DPO applies uniform weighting → Uni-DPO introduces dual-dimensional dynamic reweighting, yielding consistent improvements.
- **vs. SimPO**: SimPO simplifies by removing the reference model → Uni-DPO augments SimPO with quality and performance weights, achieving additive gains.
- **vs. Standard focal loss**: Naive focal DPO is unstable; Uni-DPO's calibrated variant (fixed threshold + LN) resolves this issue.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual-perspective dynamic weighting is natural but not breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four models × multiple benchmarks × mathematical reasoning, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Method motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ A practical improvement to DPO that is easy to integrate into existing pipelines.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](../../CVPR2026/llm_alignment/mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)
- [\[ICLR 2026\] Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences](displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences.md)
- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)
- [\[ICCV 2025\] MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization](../../ICCV2025/llm_alignment/magicid_hybrid_preference_optimization_for_id-consistent_and_dynamic-preserved_v.md)

<!-- RELATED:END -->
