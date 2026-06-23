---
title: >-
  [Paper Note] Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters
description: >-
  [ICML 2026][Model Compression][LoRA] The authors replace LoRA's "growth by rank" with "growth by CP tensor component," reducing the single-step parameter increment from 4096 to 193 (a 21× reduction). Through a strict controlled study on OPT-1.3B / SST-2/RTE/BoolQ, they prove that finer parameter granularity serves as a tool for "diagnosing PEFT budget sen
tags:
  - ICML 2026
  - Model Compression
  - LoRA
date: 2026-05-08
content_hash: 32fd4cdcee556e9b
---
# Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters

**Conference**: ICML 2026  
**arXiv**: [2606.00428](https://arxiv.org/abs/2606.00428)  
**Code**: Not yet public  
**Area**: Model Compression  
**Keywords**: Parameter-Efficient Fine-Tuning, CP Tensor Decomposition, LoRA, Budget Granularity, Ablation Study  

## TL;DR
The authors replace LoRA's "growth by rank" with "growth by CP tensor component," reducing the single-step parameter increment from 4096 to 193 (a 21× reduction). Through a strict controlled study on OPT-1.3B / SST-2/RTE/BoolQ, they prove that finer parameter granularity serves as a tool for "diagnosing PEFT budget sensitivity," but does not inherently yield a better accuracy-budget curve—yielding a sober negative-neutral conclusion rather than "our method is stronger" propaganda.

## Background & Motivation
**Background**: Parameter-Efficient Fine-Tuning (PEFT) has become the de facto standard for adapting large models. LoRA is the most widely adopted baseline—where the update is written as $\Delta W = BA$, and the rank $r$ controls both expressivity and the number of trainable parameters. Subsequent works (AdaLoRA, DoRA, CapaBoost, etc.) mostly revolve around "how to allocate rank" or "how to reparameterize updates," but few question "rank as a unit of budget granularity" itself.

**Limitations of Prior Work**: Rank is not only a dial for expressivity but also a discrete scale for the budget. On a $2048\times 2048$ attention projection in OPT-1.3B, adding one rank requires storing $r(m+n)=4096$ scalars. This means LoRA has no observable points between $r=1$ and $r=2$, leaving the low-budget region extremely sparsely sampled. If one wants to see "if adding 200 parameters is actually useful," LoRA cannot sample at that resolution.

**Key Challenge**: When two PEFT methods differ in parameter step size by 20x, traditional "matched-budget" comparisons systematically favor coarse-grained methods—because fine-grained methods are forced to compete only at those few coarse budget points, losing the chance to demonstrate intermediate performance between two LoRA ranks. Conversely, simply showing "our method has results at points LoRA cannot measure" is also unfair, as those points might exist in a low-yield region of the curve. A new comparison protocol is needed.

**Goal**: (1) Find a control method with much finer steps than LoRA; (2) Design a more honest comparison protocol than matched-budget; (3) Answer through a strict controlled study whether "finer granularity itself brings a better accuracy-budget curve."

**Key Insight**: CP tensor decomposition naturally provides finer granularity—reshaping the $2048\times 2048$ $\Delta W$ into a $32\times 64\times 32\times 64$ 4-way tensor, where each rank-1 component requires only $32+64+32+64+1=193$ scalars. One CP component is approximately equivalent to $1/21$ of a LoRA rank. The cost is that these rank-1 directions are Kronecker-structured, typically having more limited expressivity than a dense outer product.

**Core Idea**: Use fixed-component CP adapters as a fine-grained control, define a "best-under-budget" curve $U_\mathcal{A}(B)=\max_{k:P_\mathcal{A}(k)\le B} A_\mathcal{A}(k)$, and compare the two under a protocol with strictly fixed target modules / trainer / data caps / seeds, confirming key cells with 100-seed experiments.

## Method

### Overall Architecture
The question ours aims to answer is whether finer parameter granularity itself in PEFT can result in a better accuracy-budget curve. To this end, the method is built on two levels: the upper level is a **comparison protocol** that treats "parameter step size"—previously an ignored variable—as an observable variable; the lower level is a **CP tensor adapter with extremely fine steps** acting as the fine-grained control against LoRA. The process is as follows: for the q_proj and v_proj of OPT-1.3B (48 projections across 24 layers), each $2048\times 2048$ update $\Delta W$ is first reshaped into a 4-way tensor $\mathcal{T}(\Delta W)\in\mathbb{R}^{32\times 64\times 32\times 64}$, then fitted with $c$ normalized rank-1 components. During training, the backbone is frozen in fp16, and only the CP factors (or LoRA's $A,B$) and the classification head are updated, finally placing CP and LoRA together for comparison under a strictly controlled protocol.

### Key Designs

**1. Parameter Step + best-under-budget Comparison Protocol: Turning the "20x step difference" from a hidden assumption into an explicit metric**

A hidden trap in traditional PEFT comparison is that rank is not just an expressivity dial but a discrete budget scale, and the density of these scales varies wildly across methods. Ours makes this explicit—defining the parameter step size for each adapter family $\mathcal{A}$ as $\Delta P_\mathcal{A}(k)=P_\mathcal{A}(k+1)-P_\mathcal{A}(k)$. LoRA's budget is $P_{\text{LoRA}}(r)=r(m+n)$, so on a $2048\times 2048$ matrix, adding one rank results in a step $\Delta P_{\text{LoRA}}=m+n=4096$, while CP adds only 193 scalars per component, a 21x difference. Based on this, given a budget cap $B$, the best-under-budget curve is defined as $U_\mathcal{A}(B)=\max_{k\in\mathcal{K}_\mathcal{A}:P_\mathcal{A}(k)\le B} A_\mathcal{A}(k)$, where $\mathcal{K}_\mathcal{A}$ is the set of discrete budget points actually tested for that family, and $A_\mathcal{A}(k)$ is the held-out eval accuracy selected by the best-dev checkpoint. This curve reads as "the best result achievable within budget $B$ across all points tested for this family," explicitly showing the sparsity of test points.

Crucially, the authors deliberately define $U_\mathcal{A}(B)$ as a **descriptive** metric rather than a model selection rule—traditional papers either match a few budget points (hiding that LoRA has no intermediate options) or report the best run for each (hiding that one family might have tested more points); whereas here it is explicitly stated that "CP tests more points, so minor differences on the best curve should not be interpreted as a reliable win." This self-restraint sets the temperate tone of the work.

**2. Normalized CP Tensor Parameterization: Providing a control family with 21x finer steps than LoRA and stable training**

To observe the "unsampled" interval between two LoRA ranks, a control with sufficiently fine steps is needed. $\Delta W\in\mathbb{R}^{2048\times 2048}$ is reshaped by row/column splits into a 4-way tensor $\mathcal{T}(\Delta W)\in\mathbb{R}^{32\times 64\times 32\times 64}$, and written in CP form as $\mathcal{T}(\Delta W)=\sum_{s=1}^{c}\lambda_s\, u_s^{(1)}\circ u_s^{(2)}\circ u_s^{(3)}\circ u_s^{(4)}$, where each factor vector is constrained by $\|u_s^{(\ell)}\|_2=1$. Such a component stores only $32+64+32+64=192$ factor scalars plus one magnitude $\lambda_s$, totaling 193 scalars, which is exactly $\approx 1/21$ of a LoRA rank. After reshaping back to a matrix, a single component corresponds to:

$$\Delta W_s=\lambda_s\,(u_s^{(1)}\otimes u_s^{(2)})(u_s^{(3)}\otimes u_s^{(4)})^\top$$

This is a Kronecker-structured rank-1 matrix where the directions are constrained by the tensor structure, making its expressivity more limited than a standard LoRA dense rank-1 outer product—finer granularity comes at an expressivity cost. Implementation-wise, unit normalization is performed in the forward pass (optimizer still stores original factors), which eliminates scale ambiguity and preserves the stability of first-order optimization. In terms of memory, adding one LoRA rank across 48 projections adds 196,608 parameters + 1.50 MB Adam state, while one CP component adds 9,264 parameters + 0.071 MB Adam state; parameters and optimizer memory are strictly proportional. The authors also clarify why CP was chosen over Tucker / Tensor-Train / BTT: CP has the smallest step size and most stable training among candidates, providing a pure control for "fine-grained but expressivity-limited" updates; furthermore, $c$ is fixed rather than adaptive to isolate the "budget granularity" variable from adaptive allocation.

**3. Strict Control Protocol + Selective 100-seed Validation: Decoupling "granularity advantage" from "experimental noise"**

The easiest place for a PEFT comparison to fail is when a seemingly 0.2% improvement is actually buried in seed noise. To address this, all methods share the same HuggingFace Trainer, the same fp16 backbone, the same 48 target q/v projection modules, the same data cap (1000 train / 500 dev / 1000 eval), the same 5000 steps with eval every 1000 steps, and the same best-dev checkpoint selection rule. LoRA uses lr=$10^{-4}$ and CP uses $2\times 10^{-4}$, both pre-selected without per-method sweeps to avoid comparing "who tuned more carefully." Base grids are run with seeds 0,1,2, but for the most critical cells of each task (SST-2 low-budget plateau, BoolQ rise-and-saturation, RTE persistent gap), an additional 100 runs (seeds 0-99) are performed to obtain reliable mean ± variance. The best-under-budget curve is simply the max across all tested $r$ or $c$. The authors also honestly disclose that CP tested 13 capacities (1, 2, 4, 8, 16, 21, 28, 36, 43, 64, 85, 128, 171) while LoRA only tested 6 (1, 2, 3, 4, 6, 8), so CP naturally benefits from more sampling in the best-curve comparison.

### Loss & Training
Standard cross-entropy is used on the classification head. The backbone is frozen; only the adapter parameters and classification head are updated with an AdamW optimizer on an fp16 backbone. CP factors are unit-norm normalized in the forward pass to eliminate scale ambiguity. Each task is capped at 1000 train / 500 dev / 1000 eval; 5000 steps are run, with evaluation on the dev set every 1000 steps to select the best-dev checkpoint for reporting eval results. Both CP and LoRA are applied to q_proj / v_proj, consistent with typical LoRA configurations.

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

Conclusion: Under matched-budget, the two methods effectively "tie" on SST-2 and BoolQ (within ±0.6% gap), but LoRA consistently outperforms by 1.6–3.1% on RTE—indicating that matching budget does not imply method equivalence.

**Best-under-budget + 100-seed Validation** (Key cells from Table 4):

| Task | Setting | Params | Eval (seed 0-99) |
|------|---------|--------|------------------|
| SST-2 | LoRA $r=1$ | 196,608 | $0.937\pm0.005$ |
| SST-2 | CP $c=21$ (≈Budget) | 194,544 | $0.930\pm0.005$ |
| RTE | LoRA $r=6$ | 1,179,648 | $0.760\pm0.015$ |
| RTE | CP $c=28$ | 259,392 | $0.738\pm0.030$ |
| BoolQ | LoRA $r=1$ | 196,608 | $0.743\pm0.013$ |
| BoolQ | CP $c=43$ | 398,352 | $0.737\pm0.012$ |
| BoolQ | CP $c=64$ | 592,896 | $0.739\pm0.010$ |

### Ablation Study

| Configuration | Key Finding | Description |
|---------------|-------------|-------------|
| SST-2 + $c\in\{1,2,4,8,16\}$ (Below $r=1$) | Tiny CP reaches ~0.93 at budget points LoRA cannot measure | Early plateau; adding more components yields no gain |
| BoolQ + $c\in\{1,...,43\}$ | Accuracy rises monotonically from 0.662 to 0.737, then saturates | Finer granularity is useful at low budget, but caps below LoRA |
| RTE + $c\in\{1,...,171\}$ | CP consistently lower than LoRA by 1.7-2.2% | Expressivity gap cannot be overcome by finer granularity |
| Tensorization split sensitivity (Table 5) | Minimal impact of alternative splits | Choice of reshape is not the performance bottleneck |

### Key Findings
- **Early plateau on SST-2**: A tiny CP adapter ($c=2$, 9.4% of LoRA $r=1$ budget) already reaches 0.934±0.009, indicating that SST-2 saturates in the extremely low-budget region; LoRA's sparse rank grid fails to even see this plateau—a prime demonstration of finer granularity as a "diagnostic tool."
- **Rise-and-saturation on BoolQ**: The CP curve rises monotonically from $c=1$ (0.662) to $c=43$ (0.737) before slowly leveling off to 0.739 at $c=64$. The interval $c=1$ to $c=43$ is the info-rich zone telling us that "BoolQ definitely needs more capacity at low budgets," yet it still caps below LoRA $r=1$ (0.743).
- **Persistent gap on RTE**: After 100-seed validation, CP $c=28$ is 0.738±0.030 and CP $c=64$ is 0.736±0.013, while LoRA $r=6$ is 0.760±0.015—no CP configuration caught up to LoRA. This shows the limited expressivity of the Kronecker structure is a hard bottleneck on some tasks.
- **Honest Disclaimer**: The authors explicitly write that "CP tested more capacity points than LoRA; thus, small differences on the best-under-budget curve should not be interpreted as a reliable win"—this level of self-restraint is rare in PEFT literature.
- One LoRA rank step of Adam state (1.50 MB) corresponds exactly to 21 CP component steps (totaling 1.49 MB); parameter and optimizer memory ratios are strictly matched, ensuring No "budget theft" occurred.

## Highlights & Insights
- **Methodology > Algorithm**: The true contribution is framing "parameter step size" as a neglected hidden variable in PEFT comparisons. Any future PEFT paper should be viewed with caution if it does not report $\Delta P$ and the best-under-budget curve.
- **Value of Negative-Neutral Conclusions**: Ours clearly proves that "finer granularity $\neq$ better curve," debunking the rhetoric of "our method is better because its steps are smaller." The whole PEFT community should read this.
- **CP as a Diagnostic Tool**: Even if CP isn't the new SOTA, it tells you that "SST-2 saturates at 9.4% budget, BoolQ needs medium capacity, and RTE is expressivity-bound"—task-level budget sensitivity analysis that cannot be obtained by running LoRA alone.
- **Transferability**: The parameter step + best-under-budget protocol can be applied to any PEFT comparison (DoRA vs AdaLoRA, prefix tuning vs prompt tuning), model compression (varying sparsity patterns), or NAS (varying discrete search space density).
- **The Double-Edged Sword of Kronecker Structures**: CP components reshape to $(u^{(1)}\otimes u^{(2)})(u^{(3)}\otimes u^{(4)})^\top$. This constraint doesn't hurt on "low-rank signal" tasks like SST-2 but causes drops on tasks like RTE that require more free directions—consistent with observations in structured low-rank methods like ASVD/CapaBoost.

## Limitations & Future Work
- Only tested on OPT-1.3B; not scaled to LLaMA-2-7B or larger. Tasks only cover three classification benchmarks, missing generative/reasoning tasks.
- No per-method learning rate sweep; CP's lr=$2\times 10^{-4}$ might not be optimal. The authors admit this is a "controlled pilot" rather than a fully optimized benchmark.
- Tensorization split focused on $32\times 64\times 32\times 64$; while Table 5 explores sensitivity, irregular splits (e.g., $16\times 128$) or higher way-counts (5-way) are not fully explored.
- $c$ is fixed and non-growing; obvious directions like adaptive/hybrid CP-LoRA or dynamic component allocation were excluded—though this was a deliberate choice to isolate the granularity variable.
- The best-under-budget curve depends on the test grid density; CP tested 13 $c$ points while LoRA tested 6 $r$ points, giving CP an inherent sampling advantage.
- Inference latency and deployment cost after merging weights were not analyzed; unlike LoRA, CP cannot be directly merged back into a dense matrix because of its Kronecker structure.

## Related Work & Insights
- **vs LoRA**: Both are reparameterized $\Delta W$, CP steps are 21× smaller but expressivity is limited; ours proves "finer step $\neq$ superior."
- **vs AdaLoRA / DoRA / Adaptive Rank Allocation**: These modify LoRA's rank allocation strategy and are orthogonal to ours. Combining them with CP might yield higher curves, but ours fixed $c$ to isolate the granularity variable.
- **vs LoRETTA / LoRTA / CaRA / TensLoRA / AdaZeta / TeRA / KRAdapter** (Various Tensorized PEFT): Most are combinations of "tensorization + sharing/initialization/adaptation"; ours takes fixed CP as the purest control to avoid confounding engineering improvements.
- **vs Fixed-Rank / Riemannian LoRA (Bian 2025, Zhang & Pilanci 2024)**: These study low-rank geometric properties; ours studies "discrete budget resolution"—they are complementary.
- **vs Surveys** (Yang 2024a, Li 2026): Surveys classify PEFT by "architecture/optimization/deployment," but none discuss "parameter step" as an independent axis; ours fills this gap.
- Insights: (1) All future PEFT papers should report $\Delta P$; (2) Evaluation protocols should upgrade from matched-budget to best-under-budget + step size disclosure; (3) Similar step analysis applies to PTQ (different bit-widths), structured pruning (different block sizes); (4) Repeating this study on BERT / LLaMA / VLM to find "low-granularity sensitive" tasks would be a great follow-up.

## Rating
- Novelty: ⭐⭐⭐⭐ (Parameter step size as an observable framing is novel; CP adapter usage is novel in this context).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Controlled protocol is strict, 100-seed validation is solid, though limited to OPT-1.3B and three tasks).
- Writing Quality: ⭐⭐⭐⭐⭐ (Uncommon honesty; explicitly disclaims CP sampling bias and doesn't claim SOTA; methodology is clear).
- Value: ⭐⭐⭐⭐ (Not a new algorithm, but the framing changes future PEFT comparison protocols; a must-read for practitioners and reviewers).

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[AAAI 2026\] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking](../../AAAI2026/model_compression/group_orthogonal_low-rank_adaptation_for_rgb-t_tracking.md)
- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)

</div>

<!-- RELATED:END -->
