---
title: >-
  [Paper Note] Weight Decay may matter more than μP for Learning Rate Transfer in Practice
description: >-
  [ICLR 2026][LLM/NLP][μP] Through large-scale empirical analysis, this paper demonstrates that the core alignment assumption of μP holds only briefly at the start of training. In practice, it is independent weight decay rather than μP that correctly stabilizes feature learning dynamics across widths, and the practical benefits of μP can be reinterpreted as a form of implicit learning rate warmup.
tags:
  - ICLR 2026
  - LLM/NLP
  - μP
  - learning rate transfer
  - weight decay
  - AdamW
  - feature learning
date: 2026-05-08
content_hash: a57f44e92c6db695
---

# Weight Decay may matter more than μP for Learning Rate Transfer in Practice

**Conference**: ICLR 2026
**arXiv**: [2510.19093](https://arxiv.org/abs/2510.19093)
**Code**: N/A
**Area**: LLM Training Optimization
**Keywords**: [μP, learning rate transfer, weight decay, AdamW, feature learning]

## TL;DR

Through large-scale empirical analysis, this paper demonstrates that the core alignment assumption of μP holds only briefly at the start of training. In practice, it is independent weight decay rather than μP that correctly stabilizes feature learning dynamics across widths, and the practical benefits of μP can be reinterpreted as a form of implicit learning rate warmup.

## Background & Motivation

Maximal Update Parameterization (μP) is a central technique for achieving learning rate transfer in large-scale LLM training, and has been adopted by numerous open-source models (Falcon, Cohere, etc.) and commercial systems. The core idea of μP is to maintain consistent magnitudes of internal representation updates across networks of different widths via learning rate scaling, thereby transferring the optimal learning rate from smaller to larger models and avoiding costly hyperparameter searches.

However, several empirical studies have found that μP achieves good learning rate transfer only when combined with independent weight decay (WD), while standard weight decay performs poorly. The underlying reason for this phenomenon has remained poorly understood. The core question posed in this paper is: does the alignment assumption of μP actually hold in practice, and what truly drives learning rate transfer?

## Method

### Overall Architecture

The authors establish a unified framework based on **relative updates**, incorporating both μP and weight decay into a single analytical perspective. The core formulation decomposes the rate of representation change into the product of an alignment ratio and relative weight updates:

$$\frac{\|\Delta \mathbf{Y}\|}{\|\mathbf{Y}\|} = \frac{\alpha_{\Delta W}}{\alpha_W} \cdot \frac{\|\Delta \mathbf{W}\|}{\|\mathbf{W}\|}$$

where $\alpha_{\Delta W}$ is the update alignment and $\alpha_W$ is the weight alignment. μP assumes the alignment ratio $\alpha_{\Delta W}/\alpha_W = \Theta(\sqrt{C})$ grows with width $C$, and therefore compensates by scaling the learning rate as $\eta \propto 1/m$.

### Key Designs

1. **Independent Weight Decay Overrides the μP Scaling Mechanism**: *Function* — analyzes the behavioral differences between standard and independent weight decay under μP scaling. *Mechanism* — at the AdamW equilibrium, the weight norm satisfies $\|\mathbf{W}\| \propto \sqrt{KC \cdot \eta/\lambda}$, and the relative update satisfies $\|\Delta \mathbf{W}\|/\|\mathbf{W}\| \propto \sqrt{\eta\lambda}$. The independent scaling $(\eta, \lambda) \mapsto (\eta/m, m\lambda)$ preserves the product $\eta\lambda$, so the equilibrium relative update does not vary with width, thereby **canceling and overriding μP's scaling**. *Design Motivation* — the alignment assumption of μP breaks down rapidly during training, with the alignment ratio approaching 1 rather than growing with width; consistent representation change across widths therefore requires matching relative updates of equal magnitude.

2. **Analysis of the Failure Mechanism of μP's Alignment Assumption**: *Function* — theoretically and empirically analyzes why μP's update alignment and weight alignment assumptions fail to hold in practice. *Mechanism* — when batch size $B$ greatly exceeds input dimension $C$ (in LLM training, total tokens ~1M >> width ~6K), interference terms from other samples dominate the output change in the update gradient, causing the update alignment to become width-dependent: $\alpha_{\Delta W} \sim \Theta(1/\sqrt{C})$, driving the alignment ratio toward $\alpha_{\Delta W}/\alpha_W \approx 1$. *Design Motivation* — μP is derived from analysis in the infinite-width limit; its IID assumption holds only transiently near initialization in practical finite-width training.

3. **μP as Equivalent to Implicit Learning Rate Warmup**: *Function* — reinterprets the practical effect of μP + independent weight decay as implicit learning rate warmup. *Mechanism* — the high weight decay configuration $(\eta/m, m\lambda)$ causes relative updates in early training to be $1/m$ times smaller than under $(\eta, \lambda)$, gradually converging to 1 over the course of training, following an exponential warmup of the form $s_t = (1 + (m^2-1) \cdot a^{2t})^{-1/2}$, where $a = 1 - \eta\lambda$. *Design Motivation* — this explains why stronger explicit warmup schedules can partially substitute for μP's learning rate scaling.

### Loss & Training

Experiments are conducted on next-token prediction using the LLaMA architecture trained on the DCLM dataset with the AdamW optimizer. A standard schedule of 10% linear warmup followed by linear decay is used. Validation experiments cover width ratios $m \in \{1, 2, 4, 8, 16\}$.

## Key Experimental Results

### Main Results

| Configuration | LR Transfer Quality | Notes |
|:---|:---|:---|
| μP + Standard WD | ❌ Poor (large deviation after long training) | Standard scaling fails to maintain RRC consistency in later stages |
| μP + Independent WD | ✅ Good | Independent WD overrides μP scaling, stabilizes RRC |
| No μP + 10% Linear Warmup | ❌ Poor | Linear warmup insufficient as a substitute |
| No μP + 50% Linear Warmup | ❌ Moderate | Still inferior to independent WD |
| No μP + Exponential Warmup | ✅ Approximately good | Exponential warmup with width-scaling factor approaches μP + independent WD |

### Ablation Study

| Dimension | Observation |
|:---|:---|
| Alignment ratio over training | Initially $\alpha_{\Delta W}/\alpha_W \sim \sqrt{C}$, rapidly decays to ≈1 |
| No WD setting | Relative updates also become width-independent (weight norm grows continuously) |
| ResNet validation | Findings broadly consistent, though additional warmup is less critical than for LLMs |
| Matrix-wise optimizers (Muon/Scion) | Naturally bypass alignment issues, achieving low and stable update alignment |

### Key Findings

- The core alignment assumption of μP ($\alpha_{\Delta W} = \Theta(1)$) holds only for the first few steps of training and breaks down rapidly thereafter.
- Independent weight decay takes over from μP during the main training phase, stabilizing feature learning dynamics.
- The practical effect of μP is equivalent to implicit learning rate warmup, which can be partially replicated via explicit exponential warmup.
- In LLM training, the large excess of batch size over model width is the key factor causing the failure of μP's assumptions.

## Highlights & Insights

- A unified analytical framework for μP and weight decay is established, revealing their essential relationship through the lens of relative updates.
- Reinterpreting μP as "implicit warmup" challenges the community's established understanding of μP's theoretical foundations.
- Matrix-wise optimizers (e.g., Muon) are shown to potentially bypass the alignment problem altogether, explaining their characteristically low warmup requirements.
- A clear practical guideline is provided: μP must be paired with independent weight decay.

## Limitations & Future Work

- In-depth analysis is limited to the AdamW optimizer; conclusions for SGD and other optimizers require further validation.
- Although experiments cover multiple width ratios, truly large-scale verification (>10B parameters) is absent.
- The simplified analytical model (e.g., the weight decay equilibrium formula) does not precisely predict the timescale of the warmup shape in actual training.
- The effect of different initialization schemes on the rate at which the alignment assumption fails is not explored.

## Related Work & Insights

- The layer-wise learning rate scaling approach of Everett et al. (2024) suggests that μP's special treatments (output layer, attention normalization) are not necessary for transfer.
- Wang & Aitchison (2025) analyze the necessity of independent WD from an EMA perspective; this paper provides a more direct explanation through the lens of feature change rates.
- The weight decay framework of Kosson et al. (2024b) provides the foundation for the relative update analysis in this work.
- Matrix-wise optimizers (Muon, Scion) may represent a superior approach to feature learning control that goes beyond μP.

## Rating

⭐⭐⭐⭐ — Profoundly challenges the theoretical foundations of μP with solid experiments and direct practical implications; somewhat limited by its exclusive focus on AdamW and the absence of very large-scale validation.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ICLR 2026\] First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation](first_is_not_really_better_than_last_evaluating_layer_choice_and_aggregation_str.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[NeurIPS 2025\] Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](../../NeurIPS2025/llm_nlp/hyperparameter_transfer_enables_consistent_gains_of_matrix-preconditioned_optimi.md)

<!-- RELATED:END -->
