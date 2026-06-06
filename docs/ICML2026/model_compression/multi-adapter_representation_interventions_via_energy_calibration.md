---
title: >-
  [Paper Note] Multi-Adapter Representation Interventions via Energy Calibration
description: >-
  [ICML 2026][Model Compression][Representation Intervention] MARI points out that existing "representation intervention" methods rely on a linear representation hypothesis—adding a single global steering vector to all inp…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Representation Intervention"
  - "Multi-adapter Routing"
  - "Energy Gating"
  - "Truthfulness Alignment"
  - "Inference-time Editing"
date: 2026-05-08
content_hash: 6b3b637b2b3f1a90
---

# Multi-Adapter Representation Interventions via Energy Calibration

**Conference**: ICML 2026  
**arXiv**: [2605.28722](https://arxiv.org/abs/2605.28722)  
**Code**: https://github.com/V1centNevwake/MARI  
**Area**: LLM Alignment / Representation Intervention  
**Keywords**: Representation Intervention, Multi-adapter Routing, Energy Gating, Truthfulness Alignment, Inference-time Editing  

## TL;DR
MARI points out that existing "representation intervention" methods rely on a linear representation hypothesis—adding a single global steering vector to all inputs—which is unreliable because the optimal correction direction varies drastically across samples and causes "collateral damage" to general capabilities on benign inputs. It replaces the single adapter with multiple low-rank adapters using "competitive training + entropy routing" for sample-adaptive intervention, and uses an independently trained low-rank probe to calculate "propagation energy" as a threshold gate to decide whether to enable the intervention. This significantly outperforms ReFT on TruthfulQA/BBQ/Safety while maintaining or slightly improving MMLU/ARC performance.

## Background & Motivation

**Background**: Representation Intervention is one of the fastest-growing non-training paradigms in current LLM alignment—freezing model weights and modifying hidden states at a specific layer and position during inference to steer model behavior toward being "more truthful," "safer," or "less biased." Activation Steering / CAA / ITI / ReFT all follow this line, with the technical core being a global steering vector or low-rank update $\Phi_\psi(\mathbf{h})=\mathbf{h}+\gamma s_\psi\Delta_\psi(\mathbf{h})$.

**Limitations of Prior Work**: All these methods assume the "linear representation hypothesis"—that an attribute (e.g., truthfulness) corresponds to a fixed direction in the hidden space, making a single steering vector valid for all inputs. However, the authors' diagnosis on TruthfulQA shows that the required correction vector for each sample $\Delta(x)=a(x,y^\star)-a(x,\hat{y})$ drifts drastically in both magnitude and direction, with no consistent direction visible even when moving-averaging along principal components.

**Key Challenge**: (i) A single static intervention cannot cover heterogeneous needs—some samples require pushing toward $+\mathbf{v}$, while others require $-\mathbf{v}$, and forced averaging leads to conflicts; (ii) Even if the direction is correct, applying intervention to benign inputs that "do not need correction" perturbs internal representations, dropping general capabilities like MMLU/ARC by several points.

**Goal**: (1) Allow the intervention direction/intensity to vary by sample; (2) Provide a **label-free** criterion for "whether to intervene" to avoid over-intervention on benign inputs; (3) Ensure the mechanism requires no access to ground-truth during inference and uses far fewer parameters than full fine-tuning.

**Key Insight**: Replace the single adapter with a set of multiple adapters + hard routing during training to let them occupy different subspaces; select the most confident adapter via prediction entropy (parameter-free) during inference; use an independently trained low-rank probe to generate perturbation propagation energy in subsequent layers as a signal for "whether the input is worth intervening."

**Core Idea**: Upgrade global linear intervention to piecewise-affine intervention using "multi-adapter + entropy routing"; implement a label-free sample-level trigger switch using "probe propagation energy + threshold."

## Method

### Overall Architecture
MARI inserts a set of intervention modules at a fixed layer-position $(l^\star,p^\star)$ of a frozen LLM $f_\theta$. The inference workflow is: extract hidden state $\mathbf{h}=\mathbf{h}^{(l^\star)}_{p^\star}(x)$ → **Energy Gating** (calculate propagation response $E(x;\alpha_\text{probe})$ using probe $g_\phi$ in subsequent layers and compare with threshold $\tau_E$; if failed, bypass to frozen base with $\alpha=0$) → **Entropy Routing** (calculate prediction entropy $u_k(x)$ for $K$ adapters and select the most confident $\hat{k}=\arg\min_k u_k(x)$) → **Low-rank Intervention** (rewrite $\mathbf{h}$ using the winning adapter $\Phi_{\psi_{\hat k}}$ with injection intensity $\alpha_\text{full}$) → process remaining layers for final output. Training is two-stage: first train $K$ adapters using "hard routing winner-take-gradient," then independently train probe $g_\phi$ with off-subspace regularization.

### Key Designs

1.  **Competitive Multi-Adapter + Entropy Routing**:
    *   **Function**: Upgrades global static intervention to input-adaptive piecewise-affine intervention, allowing different adapters to occupy distinct input subspaces.
    *   **Mechanism**: $K$ low-rank adapters of rank $r$ are placed at the same injection point: $\Delta_{\psi_k}(\mathbf{h})=\mathbf{U}_k(\mathbf{V}_k^\top\mathbf{h}+\mathbf{b}_k)$. During training, $K$-way losses $\ell_k(x,y)$ are calculated for each sample $(x,y)$, and gradients are backpropagated only to the current "winner" $k^\star(x,y)=\arg\min_k\ell_k(x,y)$, with the objective $\mathcal{L}_\text{route}=\mathbb{E}[\ell_{k^\star}]$ and minibatch usage balancing to prevent mode collapse. During inference, since the oracle winner is unknown, "lowest entropy = highest confidence" is used to select the adapter $\hat{k}(x)=\arg\min_k u_k(x)$. Theoretical proof shows: $R_\text{ent}\le R_\text{min}+L\cdot\eta$, meaning multi-adapters strictly outperform a single adapter if specialization gains $\Delta_\text{spec}>L\cdot\eta$ (mis-routing rate × loss upper bound).
    *   **Design Motivation**: Diagnostic experiments proved that the required $\Delta(x)$ is heterogeneous; hard routing forces true specialization better than soft routing (gating mixture), which tends to converge toward an average solution.

2.  **Energy-Based Gate + Off-Subspace Regularization**:
    *   **Function**: Uses an unsupervised scalar signal aligned with "intervention applicability" to decide if an input uses $\alpha_\text{full}$ intervention or $\alpha_\text{safe}=0$ (frozen base).
    *   **Mechanism**: An independent small-rank probe $g_\phi$ (rank $r_\text{probe}<r$) calculates update $\delta_\phi(x)=g_\phi(\mathbf{h}(x))$. After injecting with intensity $\alpha_\text{probe}$, the propagation response $e_m(x;\alpha)=\|\mathbf{h}^{(\alpha,m)}_{p^\star}(x)-\mathbf{h}^{(m)}_{p^\star}(x)\|_2$ is calculated for each layer. The median $E(x;\alpha)=\mathrm{median}\{e_m\}_{m=l^\star}^L$ is defined as "energy." The training objective $\mathcal{L}_\text{cal}=\mathbb{E}[\ell_\phi(x,y)]+\lambda_\text{off}\mathcal{R}_\text{off}$ uses $\mathcal{R}_\text{off}=\mathbb{E}\|\Pi_B^\perp(\delta_\phi(x))\|_2^2$ to constrain probe updates within the "in-field calibration subspace" $B$. The threshold $\tau_E$ is calibrated on a small control set by taking the $(1-\rho)$ quantile of the benign distribution ($\rho=0.9$).
    *   **Design Motivation**: Using an adapter's own output for gating causes goal conflict; an independent probe decouples "whether to intervene" from "how to intervene." Off-subspace regularization ensures the probe's perturbation direction stays near the intervention subspace, making "energy" a reliable predictor of benefit.

3.  **Frozen Backbone + Comparability via Shared Softmax**:
    *   **Function**: Ensures that entropy $u_k(x)$ among $K$ adapters is truly comparable for meaningful entropy routing.
    *   **Mechanism**: All adapters share the same frozen LLM backbone, output head, and softmax temperature. Each adapter only learns $\mathbf{U}_k,\mathbf{V}_k,\mathbf{b}_k$ while $\theta$ remains frozen.
    *   **Design Motivation**: If adapters used different temperatures or heads, entropy values would be numerically incomparable, causing routing to collapse to the "lowest temperature" adapter.

### Loss & Training
Two stages: (1) Multi-Adapter training via hard routing + usage balancing $\mathcal{L}_\text{route}+\lambda_\text{usage}\mathcal{L}_\text{usage}$; (2) Probe training via $\mathcal{L}_\text{cal}=\mathbb{E}[\ell_\phi]+\lambda_\text{off}\|\Pi_B^\perp\delta_\phi\|_2^2$. During inference, the threshold $\tau_E$ is calibrated once on the control set.

## Key Experimental Results

### Main Results
Evaluation across 6 backbones (Llama-2, Llama-3, Qwen2, Qwen2.5) on TruthfulQA, BBQ, Safety, and general benchmarks (MMLU, ARC).

| Backbone | Method | TruthfulQA MC1 ↑ | BBQ ↑ | MMLU ↑ | ARC-C ↑ |
|---|---|---|---|---|---|
| Llama-2-7B | Vanilla | 32.03 | 0.329 | 23.3 | 33.8 |
| Llama-2-7B | ReFT (SOTA) | 50.46 | 0.540 | 23.2 | 34.0 |
| Llama-2-7B | **MARI** | **64.35** | **0.751** | 23.2 | 33.5 |
| Llama-3-8B | ReFT | 50.58 | 0.637 | 66.0 | 51.6 |
| Llama-3-8B | **MARI** | **61.81** | **0.792** | **66.6** | **52.1** |
| Qwen2.5-14B | ReFT | 52.33 | 0.646 | 80.8 | 63.6 |
| Qwen2.5-14B | **MARI** | **67.93** | **0.821** | **81.6** | **64.1** |
| Qwen2.5-32B | ReFT | 55.60 | 0.821 | 83.4 | 59.5 |
| Qwen2.5-32B | **MARI** | **81.94** | **0.876** | **84.2** | **60.0** |

### Ablation Study
| Configuration (Llama-3-8B) | TruthfulQA MC1 | BBQ | MMLU | ARC-C |
|---|---|---|---|---|
| Vanilla | 28.70 | 0.608 | 65.9 | 51.4 |
| w/o Energy Gating (Multi-adapter only) | 65.15 | 0.800 | 57.5 ↓↓ | 44.8 ↓↓ |
| w/o Multi-Adapter (Energy gate only) | 45.80 | 0.680 | 66.2 | 51.8 |
| **Full EG-MARI** | 61.81 | 0.792 | **66.6** | **52.1** |

### Key Findings
- Removing Energy Gating increases alignment scores but causes general capability collapse (MMLU 65.9 → 57.5), confirming that always-on intervention incurs an over-intervention cost.
- Removing Multi-Adapter significantly drops alignment scores (MC1 45.80 vs 61.81), proving that a single global steering vector cannot cover heterogeneous needs.
- The TruthfulQA MC1 gains (+14~28 points) are exceptionally large for frozen-weight representation intervention methods, outperforming ReFT by another 11–26 points.
- On Qwen2.5-32B, MARI pushes TruthfulQA MC1 to 81.94, approaching RLHF levels with negligible parameter overhead.

## Highlights & Insights
- **Conceptual Counter-consensus**: The authors disprove the "linear representation hypothesis" with a simple sliding-window diagnosis.
- **Entropy Routing as an Oracle Proxy**: This trick is lightweight, requires no extra parameters, and works as long as the backbone and head are shared.
- **Decoupled Gating Design**: Decoupling "whether to intervene" from "how to intervene" via an independent probe with off-subspace regularization provides an elegant label-free signal for OOD/applicability.
- **Piecewise-Affine Geometric View**: Interpreting multi-adapter routing as a partition of input space connects representation intervention to piecewise-linear network theory.

## Limitations & Future Work
- Injection points $(l^\star,p^\star)$ are fixed; if intervention needs vary by layer across samples, a "layer selection" mechanism is needed.
- The energy threshold $\tau_E$ depends on a binary control set, which might be difficult to construct for specialized domains (e.g., medical, legal).
- Lacks open-ended evaluation for generation quality, and the inference overhead of multiple adapters and probe propagation is not fully reported.
- Theoretical proofs provide risk bounds and energy upper bounds but rely on empirical estimates of specialization gains.

## Related Work & Insights
- **vs ReFT (Wu et al., 2024)**: MARI generalizes ReFT to a "piecewise rank-$r$ update + selector" and avoids capability degradation via gating.
- **vs Activation Steering / CAA / ITI**: These are weaker rank-1 (vector-based) interventions; MARI's low-rank matrix + routing strictly subsumes this family.
- **vs LoRA / IA³ (PEFT)**: While PEFT modifies weights, the multi-expert routing idea is similar to LoRA-MoE, though MARI targets input heterogeneity within the same task rather than task diversity.
- **vs RLHF / DPO**: Unlike RLHF/DPO which modify all weights, MARI freezes the model and can be toggled off at inference, serving as an "external safety layer."

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ICML 2026\] Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](towards_steering_without_sacrifice_principled_training_of_steering_vectors_for_p.md)
- [\[AAAI 2026\] QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching](../../AAAI2026/model_compression/quept_quantized_elastic_precision_transformers_with_one-shot_calibration_for_mul.md)
- [\[ICML 2026\] ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](projq_project-and-quantize_for_adapter-aware_llm_compression.md)
- [\[ICML 2026\] Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](towards_resource-efficient_llms_end-to-end_energy_accounting_of_distillation_pip.md)

</div>

<!-- RELATED:END -->
