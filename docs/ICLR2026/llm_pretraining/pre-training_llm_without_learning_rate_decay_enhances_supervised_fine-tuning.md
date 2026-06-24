---
title: >-
  [Paper Note] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning
description: >-
  [ICLR 2026][LLM Pretraining][Learning Rate Scheduling] This paper proposes the Warmup-Stable-Only (WSO) learning rate scheduling strategy, which completely eliminates the learning rate decay phase during pre-training. Although this results in worse pre-training metrics, it consistently outperforms all decay strategies after SFT. Loss landscape analysis reveals that the superiority of WSO stems from its ability to maintain flatter minima.
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Learning Rate Scheduling"
  - "Pre-training"
  - "Supervised Fine-Tuning"
  - "Loss Landscape"
  - "Warmup-Stable-Only"
date: 2026-05-08
content_hash: ea5730a04bcdf3f6
---

# Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning

**Conference**: ICLR 2026  
**arXiv**: [2603.16127](https://arxiv.org/abs/2603.16127)  
**Code**: Not open-sourced  
**Area**: LLM Pre-training  
**Keywords**: Learning Rate Scheduling, Pre-training, Supervised Fine-Tuning, Loss Landscape, Warmup-Stable-Only

## TL;DR

This paper proposes the Warmup-Stable-Only (WSO) learning rate scheduling strategy, which completely eliminates the learning rate decay phase during pre-training. Although this results in worse pre-training metrics, it consistently outperforms all decay strategies after SFT. Loss landscape analysis reveals that the superiority of WSO stems from its ability to maintain flatter minima.

## Background & Motivation

### Background

In large language model pre-training, learning rate (LR) scheduling is one of the most critical yet complex hyperparameters. The dominant approaches are as follows:

- **Cosine decay**: The most common method since GPT-3, where the LR decays to near zero following a cosine curve throughout training.
- **Linear decay**: Recent studies suggest that linear decay to zero can achieve a lower pre-training loss.
- **WSD (Warmup-Stable-Decay)**: A more flexible approach that employs a short decay period only at the end of training, adopted by models like MiniCPM.

The commonality among these strategies is the decay of LR at the end of training to optimize pre-training metrics.

### Key Challenge

The optimization goal of existing LR strategies is performance during the pre-training phase $\mathtt{Task}_{\rm pre}(M_{\rm pre})$. However, in practical applications, the ultimate performance after SFT, $\mathtt{Task}_{\rm post}(M_{\rm post})$, is what truly matters.

Recent research (Sun & Dredze 2025; Springer et al. 2025) explicitly points out that **models with better pre-training performance do not necessarily perform better after SFT**. This poses a fundamental question: Is LR decay, chosen to optimize pre-training metrics, still the optimal choice when the model is intended to undergo SFT?

### Formalization

The traditional pipeline selects the optimal model greedily by stage:

$$\widehat{M}_{\rm pre} = \arg\max_{M_{\rm pre} \in \mathcal{M}_{\rm pre}} \{\mathtt{Task}_{\rm pre}(M_{\rm pre}[M_{\rm rand}])\}$$

However, the ideal objective should be global joint optimization:

$$\widehat{M}_{\rm post} = \arg\max_{(M_{\rm pre}, M_{\rm post}) \in (\mathcal{M}_{\rm pre}, \mathcal{M}_{\rm post})} \{\mathtt{Task}_{\rm post}(M_{\rm post}[M_{\rm pre}[M_{\rm rand}]])\}$$

## Method

### Overall Architecture

Rather than proposing a new module, this paper questions a universally accepted training convention: whether the learning rate should decay at the end of pre-training. The approach starts by creating a minimalist "non-decay" scheduler called WSO. Then, a scalar parameter is used to place four types of schedulers (WSO, WSD, Cosine, Linear) and the "to decay or not to decay" choice onto a single scannable axis. Consequently, "decay intensity" transforms from a discrete strategy choice into a continuous coordinate. This axis is then applied to two sets of real-world pipelines—two-stage (Pre-training + SFT) and three-stage (Pre-training + Intermediate Training + SFT)—to observe final performance after SFT. Finally, the curvature (sharpness) of the loss landscape is used to explain "why no decay is better." The entire methodology follows a causal chain: "proposing a minimal variant → unified parameterization for controlled comparison → mechanistic explanation via curvature," adhering to the standard pre-training pipeline since GPT-3 (warmup → stable → optional decay → optional mid-training → SFT), with the only modification being the removal of the terminal decay.

> This work belongs to the "Training Recipe + Loss Landscape Analysis" category and does not involve multi-module or multi-branch data flow pipelines (the process is a standard linear training pipeline where the contribution lies in "deleting a segment" and "post-hoc curvature analysis"). Therefore, **the Mermaid architecture diagram is skipped**. The three key designs below strictly correspond to the "Propose WSO → Unified Parameterized Comparison → Curvature Explanation" steps.

### Key Designs

**1. Warmup-Stable-Only: Removing the Entire Decay Phase**

Mainstream schedulers assume the LR must be lowered at the end of training to tighten the pre-training loss. WSO directly questions whether this step is worthwhile. It is a minimalist variant of WSD (Warmup-Stable-Decay) that retains only the warmup and stable segments while completely discarding decay:

$$\eta^{\text{WSO}}(t, \alpha_{\text{pre}}) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\ \eta_{\max} & T_{\text{warmup}} < t \leq T_{\text{pre}} \end{cases}$$

The learning rate remains constant at $\eta_{\max}$ after the warmup phase until pre-training concludes. The cost is a worse pre-training loss (a constant high LR cannot converge parameters into narrow valleys), but the benefit is that the model resides in a "looser" position, leaving room for adjustment during subsequent SFT—which is what the loss landscape analysis in Design 3 quantifies.

**2. Unified Parameterization via min-LR Factor: Aligning Schedulers and Pipelines on a Comparable Axis**

To fairly compare "how much decay is optimal," different schedulers and stages must be placed on the same axis. The paper uses a minimum LR factor $\alpha$ (the ratio of the final LR to $\eta_{\max}$) for unified parameterization. For pre-training, $\alpha_{\text{pre}}$ is used: $\alpha_{\text{pre}}=0.0$ represents the most aggressive decay to zero (common for Linear/Cosine), $\alpha_{\text{pre}}=0.1$ is a moderate decay to 10% (used by Llama 3, OLMo 2), and $\alpha_{\text{pre}}=1.0$ is equivalent to no decay (WSO). Since modern pre-training often inserts an intermediate training phase (mid-training) before SFT, which usually also involves decay, the paper introduces $\alpha_{\text{mid}}$ for this stage: $\alpha_{\text{mid}}=0.0$ for linear decay to zero and $\alpha_{\text{mid}}=1.0$ for a constant LR. When $\alpha_{\text{pre}}=1.0$ and $\alpha_{\text{mid}}=1.0$, WSO is extended to both pre-training and intermediate training. This turns "decay intensity" into a continuous scanning axis, making WSO an extreme point on this axis rather than an isolated solution. The proposition that "decay at any stage is harmful" can thus be verified step-by-step—leading to the conclusion that even decay during intermediate training alone degrades final SFT performance.

**3. Quantifying Flatness via Hessian Trace: Explaining "Why No Decay is Better"**

While the first two points establish a controlled comparison of the phenomenon, this point addresses the mechanism. Taking insights from transfer learning literature—that models stopping in flatter regions of the loss landscape adapt better—the paper uses sharpness (the inverse of flatness) to characterize the curvature of minima. The trace of the Hessian is used (the sum of second-order curvatures across all parameter dimensions, providing a scalar summary of curvature):

$$\text{Sharpness}(\theta_t) = \text{Tr}(\mathbf{H}_{\mathcal{L}}(\theta_t)) = \sum_{i=1}^{d} \frac{\partial^2 \mathcal{L}(\theta_t; \mathcal{D})}{\partial \theta_i^2}$$

Calculating the Hessian directly for billion-parameter models is impractical, so the paper utilizes the Hutchinson unbiased estimator to approximate the trace using a few Hessian-vector products. The results provide a mechanistic explanation: decay strategies push the model into sharper minima, while WSO ($\alpha_{\text{pre}}=1.0$) maintains a flatter landscape throughout. Sharper valleys mean that even minor perturbations during SFT cause drastic changes in loss, hindering transfer, whereas flat regions tolerate the shifts of fine-tuning. Sharpness shows a strong negative correlation with SFT performance (Pearson $r=-0.709$), closing the causal chain of "no decay → flatter landscape → better SFT."

## Key Experimental Results

### Main Results: Two-Stage Setting (Pre-training + SFT)

Model Architectures: 1B and 8B (Llama 3 architecture); Pre-training Data: FineWeb-Edu; SFT Data: Tulu-3 SFT mixture.

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

**Key Finding**: WSO has the worst pre-training loss (+0.127 for 8B) but achieves the best performance after SFT (+1.1 points for 8B).

### Three-Stage Setting (Pre-training + Intermediate Training + SFT)

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

In the Over-training + Intermediate Training (2T + 500B tokens) scenario, the advantage of WSO is even more pronounced: SFT Task Avg Δ reaches **+1.4**.

### Key Findings

1. **Performance Inversion**: Schedulers that perform best in pre-training (decaying to 0) perform the worst after SFT.
2. **WSO Dominance**: WSO is consistently optimal across all settings including 1B/8B, two-stage/three-stage, and standard/over-training.
3. **Harm of Decay at Any Stage**: In the three-stage setting, even decaying during intermediate training alone reduces SFT performance.
4. **Sharpness Negative Correlation**: The Pearson correlation coefficient between sharpness and SFT performance is $r=-0.709$.

## Highlights & Insights

1. **Counter-intuitive Core Discovery**: Better pre-training loss $\neq$ better downstream task performance; LR decay actually damages model adaptability.
2. **Clear Theoretical Explanation**: Provides a complete causal chain from flat minima to better SFT performance through loss landscape analysis.
3. **Minimalist Implementation**: WSO is simpler than any decay strategy, requiring no tuning for decay ratios or decay phase lengths.
4. **Significant Practical Value**: Suggests that open-source models should be trained using WSO before release to provide maximum adaptability for downstream users.
5. **Scale Consistency**: Conclusions remain consistent across training scales from 1B to 8B and 100B to 2T tokens.

## Limitations & Future Work

1. Only SFT was investigated as a post-training method; alignment stages like DPO or RLHF were not tested.
2. The maximum experimental scale was 8B; whether the findings hold for larger models (70B+) remains to be verified.
3. WSO pre-training loss is significantly higher; it might not be suitable for scenarios where low pre-training loss is strictly required (e.g., distillation).
4. The sample size for the correlation analysis between sharpness and SFT performance is relatively small.

## Related Work & Insights

- **Bergsma et al. 2025**: Argues that Linear decay to 0 is optimal—but this only holds for pre-training loss.
- **WSD (Hu et al. 2024)**: WSO can be seen as an extreme simplification of WSD, echoing the flexibility advantages of WSD.
- **Wen et al. 2025**: Theoretical analysis of WSD found that the decay phase leads to increased sharpness, a problem WSO avoids.
- **Insight**: Future training strategies should be selected based on the final deployment objective (performance after SFT/RLHF) rather than pre-training metrics.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Challenges the widespread consensus that "lower decay is always better" with a strong and clear perspective.
- **Theoretical Depth**: ⭐⭐⭐⭐ — Comprehensive loss landscape analysis and formal framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely thorough, covering 2 scales × 4 schedulers × 3 training settings × over-training.
- **Value**: ⭐⭐⭐⭐⭐ — Provides directly applicable practical advice that guides LLM training and release strategies.
- **Overall**: ⭐⭐⭐⭐☆ — A solid, counter-intuitive, and practical piece of work that is significant for pre-training strategy research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining](how_learning_rate_decay_wastes_your_best_data_in_curriculum-based_llm_pretrainin.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning](sstoken_self-modulated_and_semantic-aware_token_selection_for_llm_fine-tuning.md)
- [\[ICLR 2026\] Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Fine-Tuning](task-aware_data_selection_via_proxy-label_enhanced_distribution_matching_for_llm.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](../../ICML2026/llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)

</div>

<!-- RELATED:END -->
