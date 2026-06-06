---
title: >-
  [Paper Note] Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters
description: >-
  [ICML 2026][Model Compression][Parameter-efficient fine-tuning] The authors replace LoRA's "rank-based growth" with "CP tensor component growth…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Parameter-efficient fine-tuning"
  - "CP tensor decomposition"
  - "LoRA"
  - "budget granularity"
  - "ablation study"
date: 2026-05-08
content_hash: 2e8e8bdcc4ee0402
---

# Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters

**Conference**: ICML 2026  
**arXiv**: [2606.00428](https://arxiv.org/abs/2606.00428)  
**Code**: Not yet released  
**Area**: Model Compression  
**Keywords**: Parameter-efficient fine-tuning, CP tensor decomposition, LoRA, budget granularity, ablation study  

## TL;DR
The authors replace LoRA's "rank-based growth" with "CP tensor component growth," reducing the per-step parameter increment from 4096 to 193 (a $21\times$ reduction). Through a rigorous controlled study on OPT-1.3B across SST-2, RTE, and BoolQ, they demonstrate that finer parameter granularity serves as a diagnostic tool for PEFT budget sensitivity but does not inherently yield a better accuracy-budget curve—presenting an honest negative-to-neutral conclusion rather than "superior performance" claims.

## Background & Motivation
**Background**: Parameter-Efficient Fine-Tuning (PEFT) has become the de facto standard for adapting large models. LoRA is the most widely adopted baseline, where updates are written as $\Delta W = BA$, and the rank $r$ simultaneously controls expressivity and the number of trainable parameters. Most subsequent works (e.g., AdaLoRA, DoRA, CapaBoost) focus on "how to allocate rank" or "how to reparameterize updates," but few question "rank as the unit of budget granularity" itself.

**Limitations of Prior Work**: Rank serves not only as a dial for expressivity but also as a discrete scale for the budget. In an attention projection of $2048\times 2048$ in OPT-1.3B, adding one rank requires storing $r(m+n)=4096$ scalars. This means LoRA has no observable data points between $r=1$ and $r=2$, leaving the low-budget region extremely sparsely sampled. To observe whether "adding 200 parameters is useful," LoRA lacks the necessary resolution.

**Key Challenge**: When two PEFT methods differ in parameter step size by 20 times, traditional matched-budget comparisons systematically favor coarse-grained methods. Fine-grained methods are forced to compete only at coarse budget points, losing the opportunity to showcase intermediate performance between LoRA ranks. Conversely, merely showing results at points LoRA cannot measure is unfair, as those points might reside in low-return regions of the curve. A new comparison protocol is required.

**Goal**: (1) Identify a control method with much finer steps than LoRA; (2) Design a more honest comparison protocol than matched-budgeting; (3) Answer whether fine granularity itself leads to a better accuracy-budget curve through a rigorous controlled study.

**Key Insight**: CP tensor decomposition naturally provides finer granularity. By reshaping a $2048\times 2048$ $\Delta W$ into a $32\times 64\times 32\times 64$ 4-way tensor, each rank-1 component requires only $32+64+32+64+1=193$ scalars. One CP component is approximately equivalent to $1/21$ of a LoRA rank. The trade-off is that these rank-1 directions are Kronecker-structured, with expressivity more constrained than a general dense outer product.

**Core Idea**: Use fixed-component CP adapters as a fine-grained control. Define a "best-under-budget" curve $U_\mathcal{A}(B)=\max_{k:P_\mathcal{A}(k)\le B} A_\mathcal{A}(k)$. Compare both under a protocol with strictly fixed target modules, trainers, data caps, and seeds, using 10-seed selective runs to confirm key insights.

## Method

### Overall Architecture
The "method" is divided into two layers: the upper layer is the **comparison protocol** (Section 3), which treats parameter step size $\Delta P_\mathcal{A}(k)=P_\mathcal{A}(k+1)-P_\mathcal{A}(k)$ as an observable discrete variable and defines the best-under-budget curve $U_\mathcal{A}(B)$ as a descriptive metric (rather than a model selection rule). The lower layer is the **CP adapter as a control** (Section 4), which replaces LoRA's "matrix rank + dense outer product" with "tensor reshaping + normalized CP components." The pipeline involves selecting target projections (q_proj, v_proj across 24 layers = 48 projections), reshaping each $2048\times 2048$ $\Delta W$ into $\mathcal{T}(\Delta W)\in\mathbb{R}^{32\times 64\times 32\times 64}$, and fitting the tensor with $c$ normalized rank-1 components. During training, only the CP factors and LoRA's $A, B$ matrices are updated, with the backbone frozen in fp16.

### Key Designs

1.  **Parameter Step + Best-under-budget Comparison Protocol**:
    - **Function**: Transitions the 20x budget step difference from a hidden assumption to an observable metric, preventing matched-budget comparisons from systematically favoring coarse-grained methods.
    - **Mechanism**: Defines a parameter step $\Delta P_\mathcal{A}(k)=P_\mathcal{A}(k+1)-P_\mathcal{A}(k)$ for each adapter family $\mathcal{A}$. For LoRA, $P_{\text{LoRA}}(r)=r(m+n)$, so $\Delta P_{\text{LoRA}}=m+n=4096$ (on $2048\times 2048$). For CP, each component adds only 193 scalars. Given a budget upper bound $B$, the best-under-budget curve is defined as $U_\mathcal{A}(B)=\max_{k\in\mathcal{K}_\mathcal{A}:P_\mathcal{A}(k)\le B} A_\mathcal{A}(k)$, where $\mathcal{K}_\mathcal{A}$ is the set of discrete budget points tested, and $A_\mathcal{A}(k)$ is the held-out accuracy selected via the best-dev checkpoint.
    - **Design Motivation**: Traditional PEFT papers either (a) perform matched-budget comparisons (hiding that LoRA has no intermediate options) or (b) report the best run (hiding that one family may have been tested more). The authors choose a descriptive rather than prescriptive definition and explicitly state that CP has a sampling advantage, which sets the honest tone of the paper.

2.  **Normalized CP Tensor Parameterization as Fine-grained Control**:
    - **Function**: Provides a control family with a step size 21x smaller than LoRA while maintaining training stability.
    - **Mechanism**: Reshapes $\Delta W\in\mathbb{R}^{2048\times 2048}$ into a 4-way tensor $\mathcal{T}(\Delta W)\in\mathbb{R}^{32\times 64\times 32\times 64}$. The CP form is $\mathcal{T}(\Delta W)=\sum_{s=1}^{c}\lambda_s u_s^{(1)}\circ u_s^{(2)}\circ u_s^{(3)}\circ u_s^{(4)}$, where each $\|u_s^{(\ell)}\|_2=1$. Each component stores $32+64+32+64=192$ factor scalars + 1 amplitude $\lambda_s = 193$ scalars. After reshaping back to a matrix, a single component corresponds to $\Delta W_s=\lambda_s(u_s^{(1)}\otimes u_s^{(2)})(u_s^{(3)}\otimes u_s^{(4)})^\top$—a Kronecker-structured rank-1 matrix. Normalization is applied in the forward pass to maintain stability. 
    - **Design Motivation**: The authors state they are not proposing a new SOTA. CP was chosen because it has the smallest step size and highest stability among candidates (Tucker, Tensor-Train, etc.). Fixing $c$ rather than allowing adaptive growth isolates the "budget granularity" variable.

3.  **Strict Control Protocol + 10-seed Selective Confirmation**:
    - **Function**: Distinguishes fine-grained advantages from experimental noise.
    - **Mechanism**: All methods use the same trainer (HuggingFace), fp16 backbone, and target modules (48 projections). Data is capped at 1000 train/500 dev/1000 eval. Training lasts 5000 steps with evaluation every 1000 steps. While base results use seeds 0-2, key cells (SST-2 low-budget plateau, BoolQ rise, RTE gap) are run 100 times (seeds 0-99) for credible statistics.
    - **Design Motivation**: Many 0.2% gains in PEFT are actually seed noise. 100-seed confirmation is necessary to identify true signals. Avoiding per-method hyperparameter tuning ensures the comparison is about the method, not the diligence of the researcher.

### Loss & Training
Standard cross-entropy on a classification head with a frozen backbone. Only adapter parameters and the head are updated. fp16 backbone with AdamW optimizer. CP factors use unit-norm normalization in the forward pass. Data is capped at 1000/500/1000. 5000 steps with best-dev checkpoint selection. CP and LoRA are both applied to q_proj and v_proj.

## Key Experimental Results

### Main Results

**Matched-budget Comparison** (Average of seeds 0,1,2, $\Delta$ eval = CP - LoRA):

| Task | Budget Tier | LoRA eval | CP eval | $\Delta$ eval |
|------|-------------|-----------|---------|---------------|
| SST-2 | Low ($r=2$) | 0.937 | 0.931 | -0.006 |
| SST-2 | Mid ($r=4$) | 0.939 | 0.933 | -0.005 |
| SST-2 | High ($r=8$) | 0.932 | 0.936 | +0.004 |
| RTE | Low | 0.747 | 0.732 | -0.016 |
| RTE | Mid | 0.753 | 0.722 | -0.031 |
| RTE | High | 0.745 | 0.729 | -0.016 |
| BoolQ | Low | 0.741 | 0.735 | -0.006 |
| BoolQ | Mid | 0.742 | 0.740 | -0.001 |
| BoolQ | High | 0.735 | 0.740 | +0.005 |

Conclusion: Under matched budgets, SST-2 and BoolQ are essentially ties (gap within $\pm 0.6\%$), but LoRA consistently outperforms on RTE by 1.6–3.1%, indicating that matching budgets does not imply equivalent performance.

**Best-under-budget + 10-seed Confirmation** (Key cells from Table 4):

| Task | Setting | Params | Eval (seed 0-99) |
|------|---------|--------|------------------|
| SST-2 | LoRA $r=1$ | 196,608 | $0.937\pm0.005$ |
| SST-2 | CP $c=21$ (≈matched) | 194,544 | $0.930\pm0.005$ |
| RTE | LoRA $r=6$ | 1,179,648 | $0.760\pm0.015$ |
| RTE | CP $c=28$ | 259,392 | $0.738\pm0.030$ |
| BoolQ | LoRA $r=1$ | 196,608 | $0.743\pm0.013$ |
| BoolQ | CP $c=43$ | 398,352 | $0.737\pm0.012$ |
| BoolQ | CP $c=64$ | 592,896 | $0.739\pm0.010$ |

### Ablation Study
| Configuration | Key Findings | Description |
|------|---------|------|
| SST-2 + $c\in\{1,2,4,8,16\}$ | Tiny CP already reaches ~0.93 | Early plateau; more components yield no gains. |
| BoolQ + $c\in\{1,...,43\}$ | Accuracy rises from 0.662 to 0.737, then saturates. | Fine granularity reveals a low-budget growth phase. |
| RTE + $c\in\{1,...,171\}$ | CP consistently lower than LoRA by 1.7-2.2%. | Expressive gap cannot be bridged by granularity. |
| Tensorization split sensitivity | Alternative splits have minimal impact. | The choice of reshape is not a performance bottleneck. |

### Key Findings
- **SST-2 Early Plateau**: A tiny CP adapter ($c=2$, 9.4% of LoRA $r=1$ budget) reaches 0.934±0.009, suggesting SST-2 saturates at extremely low budgets. LoRA's rank grid is too sparse to detect this existence—this is a prime example of fine granularity as a diagnostic tool.
- **BoolQ Rise-and-Saturation**: The CP curve rises monotonically from $c=1$ (0.662) to $c=43$ (0.737) before flattening. This interval provides rich information that LoRA misses, though LoRA $r=1$ still caps higher at 0.743.
- **RTE Persistent Gap**: CP never catches LoRA regardless of configuration, proving that Kronecker-structured expressive limits are a "hard ceiling" on some tasks.
- **Honest Declaration**: The authors admit CP has a sampling advantage in best-under-budget curves because it tests more points, so small margins should not be viewed as a reliable victory.
- One LoRA rank step's Adam state (1.50 MB) matches 21 CP component steps (total 1.49 MB), ensuring a strict match in parameter/optimizer memory ratio.

## Highlights & Insights
- **Methodological over Algorithmic Contribution**: The real contribution is framing "parameter step size" as a neglected hidden variable in PEFT. Any future PEFT paper should be viewed with skepticism if it does not report $\Delta P$ and best-under-budget curves.
- **Value of Negative-Neutral Results**: The work clearly demonstrates that "finer granularity $\neq$ better curve," debunking a class of claims that smaller step sizes are inherently superior.
- **CP as a Diagnostic Tool**: CP reveals task-level budget sensitivity (SST-2 saturates at 9%, BoolQ needs moderate capacity, RTE is expressivity-bound) that running LoRA alone cannot provide.
- **Portability**: This protocol can be applied to any PEFT comparison (DoRA vs AdaLoRA), model compression (sparsity patterns), or NAS.

## Limitations & Future Work
- Evaluation is limited to OPT-1.3B and three classification tasks; generation/reasoning tasks and larger models (LLaMA-7B) are not covered.
- Lack of per-method learning rate sweeping; the authors acknowledge this as a "controlled pilot" rather than a fully optimized benchmark.
- Tensorization split sensitivity was only tested on $32\times 64\times 32\times 64$; more irregular splits or higher-way tensors were not fully explored.
- Adaptive component allocation was ignored to isolate the granularity variable, which is a reasonable trade-off but a direction for future work.
- Inference latency and merging costs were not analyzed; CP cannot be directly merged into dense matrices like LoRA due to its structure.

## Related Work & Insights
- **vs LoRA**: Similar reparameterization but 21x smaller steps; proves finer resolution does not equal superiority.
- **vs AdaLoRA / DoRA**: These modify allocation strategies and are orthogonal to this work. Combining them with CP might raise the curve, but the goal here was variable isolation.
- **vs Tensorized PEFTs (LoRETTA, etc.)**: Many use tensorization for parameter sharing or adaptive growth; this work uses the purest CP form as a control.
- **vs Riemannian LoRA**: While those study low-rank geometry, this studies discrete budget resolution—the two are complementary.
- **Insight**: Future PEFT evaluation should move from matched-budgeting to best-under-budget curves with disclosed step sizes.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel framing of parameter steps; CP used as a control).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Strict protocol and 100-seed confirmation, though limited model scale).
- Writing Quality: ⭐⭐⭐⭐⭐ (Rare level of honesty; clear methodology).
- Value: ⭐⭐⭐⭐ (Changes how PEFT comparison protocols should be viewed).

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)
- [\[ICML 2026\] Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter](compress_then_merge_from_multiple_loras_into_one_low-rank_adapter.md)

</div>

<!-- RELATED:END -->
