---
title: >-
  [Paper Note] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs
description: >-
  [ICML 2026][Model Compression][MoE-LLM] GEMQ upgrades expert bit allocation for MoE models from intra-layer local Linear Programming (LP) to cross-layer global LP. Combined with "post-quantization router weight fine-tuni…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "MoE-LLM"
  - "Mixed-Precision Quantization"
  - "Global Linear Programming"
  - "Router Fine-tuning"
  - "Progressive Quantization"
date: 2026-05-08
content_hash: 22930f3856a870ab
---

# GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.23078](https://arxiv.org/abs/2605.23078)  
**Code**: https://github.com/jndeng/GEMQ  
**Area**: Model Compression / MoE Quantization / LLM Inference  
**Keywords**: MoE-LLM, Mixed-Precision Quantization, Global Linear Programming, Router Fine-tuning, Progressive Quantization

## TL;DR
GEMQ upgrades expert bit allocation for MoE models from intra-layer local Linear Programming (LP) to cross-layer global LP. Combined with "post-quantization router weight fine-tuning" to align routing distributions distorted by quantization, and an iterative "progressive bit reduction" framework to repeatedly refine importance estimation, GEMQ compresses models like Mixtral-8×7B to an average of 2.5 bits per expert with an average zero-shot drop across 7 tasks (including MMLU) within 7%. It significantly outperforms PMQ, SpQR, MoEQuant, and EAQuant under the same bit budget.

## Background & Motivation
**Background**: MoE-LLMs (Mixtral, DeepSeekV2, Qwen-MoE, etc.) reduce computational costs via sparse activation, but the total number of parameters remains high. All experts must reside in VRAM—Mixtral-8×7B requires 87 GB in full precision, exceeding the capacity of a single H100-80GB card. Since expert parameters typically account for over 90% of the total, expert weight quantization is the primary battlefield for MoE compression.

**Limitations of Prior Work**: (1) Existing expert-level mixed-precision methods (e.g., PMQ, Li et al. 2024) solve the LP problem independently **within each layer**, forcing the same bit budget per layer and ignoring "importance variance across layers"—Fig 1(a) shows the sum of squared gradients (Fisher trace) of experts can vary by 7× across layers in Mixtral. (2) Quantization alters router input distributions and expert outputs, leading to **routing shift**—after 1.5-bit quantization, over 40% of tokens are routed to different experts than in full precision. Existing methods either ignore this or forcefully align the quantized router back to the FP distribution, which is suboptimal. (3) Task loss estimation relies on Taylor expansion, requiring small quantization perturbations $\Delta w$; at extremely low bits (1-2 bits), $\Delta w$ is large, making the estimation inaccurate.

**Key Challenge**: Global bit allocation requires a shared loss baseline across all experts, but Taylor estimation requires small perturbations and the local minimum assumption, both of which are violated at low bits.

**Goal**: (a) Upgrade local LP to global LP, allowing bits to flow freely across layers; (b) Explicitly model and repair router drift caused by quantization; (c) Provide an importance estimation scheme that approximates true loss even at low bits.

**Key Insight**: The authors use Gauss-Newton + diagonal Fisher to pull the task loss increment $\Delta\tilde L_{ij}\approx \mathbb{E}_\mathcal{D}[\Delta z_{ij}^\top \mathrm{diag}(g^{(z)}g^{(z)\top})\Delta z_{ij}]$, caused by quantizing each expert $i$ to $j$ bits, onto the same "task loss" scale—naturally supporting cross-layer comparisons. Meanwhile, the 1D loss landscape analysis in Fig 3 suggests that as long as the router remains adapted to the current weights, a "intermediate quantized model with a similar bit budget" can approximate the true loss near the target extremely low-bit point.

**Core Idea**: A closed-loop of progressive bit reduction using global LP to decide expert bits + post-quantization router fine-tuning to repair routing + using the previous round's quantized model as a "neighbor" to re-estimate importance.

## Method

### Overall Architecture
GEMQ is an evolutionary compression pipeline:

1. **Target Bit Budget Queue**: The user provides a set of target bpe (bits per expert) sorted from high to low, e.g., $[3.0, 2.5, 2.0, 1.5]$.
2. **Initial Round (Highest Bit)**: Use the FP model to sample 128 C4 segments to calculate $\Delta\tilde L_{ij}$ for each (expert, candidate bit) pair. Solve global LP for bit allocation, perform per-expert GPTQ quantization to obtain $Q_{B_1}$.
3. **Router Fine-tuning**: Freeze attention and expert weights. Fine-tune all router weights using the same calibration data to minimize cross-entropy task loss. Routers account for $< 0.04\%$ of parameters, completing within one minute on three H100s, resulting in the aligned $Q_{B_1}^\star$.
4. **Progressive Bit Reduction**: When moving to the next budget $B_k$, use $Q_{B_{k-1}}^\star$ (closer to the $B_k$ quantization point) to calculate LP coefficients instead of the FP model. Re-perform global LP + GPTQ + router fine-tuning to obtain $Q_{B_k}^\star$. Iterate until the target lowest bit is reached.
5. **Deployment**: Attention is fixed at 4-bit, experts follow the 1/2/3-bit distribution from LP, and attention weights use GPTQ group=128 asymmetric. The MoE kernel schedules different-bit experts together; a 2.5-bit Mixtral achieves 82.5 tokens/s on a single H100.

### Key Designs

1. **Global Expert-Level LP Formulation**:
    - **Function**: Decides the bit-width of all experts simultaneously via a cross-model 0-1 linear programming objective to minimize the total task loss increase.
    - **Mechanism**: Uses the Gauss-Newton approximation to transfer the Taylor second-order term $\Delta L\approx \frac12\Delta w^\top H(w)\Delta w$ to the MoE block output $\Delta L\approx \frac12 \Delta z^\top H(z)\Delta z$, then uses diagonal Fisher $H(z)\approx \mathrm{diag}(g^{(z)}g^{(z)\top})$ to get a scalar loss cost $\Delta\tilde L_{ij}$ for each (expert $i$, bit $j$). Here $z$ is the MoE block aggregate output (multiplied by routing scores), naturally weighting by routing probability. Then solve the 0-1 LP: $\min\sum_{i,j}\Delta\tilde L_{ij}x_{ij}$, subject to $\sum_{i,j}j\cdot x_{ij}\le B$, $\sum_j x_{ij}=1$, and a constraint ensuring each layer contains at least one high-precision expert (as a regularizer). The LP is solvable in seconds.
    - **Design Motivation**: Resolves the "intra-layer comparable, cross-layer incomparable" issue in methods like PMQ. Previous layer-wise reconstruction losses $\|Wx-\hat Wx\|^2$ have inconsistent scales across layers. GEMQ uses the same task loss to place all experts in a unified coordinate system, allowing bits to flow from "insensitive layers" to "sensitive layers" without hyperparameters (unlike PMQ's fusion of activation freq and weight stats).

2. **Global Router Fine-tuning**:
    - **Function**: Updates only the router weights after each round of expert quantization to adapt to the quantized experts and restore reasonable token-to-expert routing.
    - **Mechanism**: Dequantizes weights to FP for simulation, freezes attention and experts, and opens all router parameters (typically hidden→N_expert linear, ~0.04% of parameters). Direct backpropagation with cross-entropy task loss on the calibration set for one epoch using AdamW (lr=$1\mathrm{e}{-4}$, batch=1). Unlike methods forcing router outputs to match FP distributions, GEMQ allows the router to actively select new optimal routing schemes for the quantized experts.
    - **Design Motivation**: Fig 3 demonstrates that true loss curves can be **non-smooth** due to routing jumps at certain $\Delta w$, which Taylor expansions cannot predict. Fine-tuning the router to adapt to new expert selections "smooths" the curve, making the local minimum assumption and Taylor estimation valid again.

3. **Progressive Bit Budget Reduction**:
    - **Function**: Quantizes progressively following $B_1>B_2>\dots>B_K$, using the prior round's fine-tuned model as the baseline for importance estimation instead of the FP model.
    - **Mechanism**: When jumping directly from 2.5 to 1.5 bpe, $\Delta w$ is large/Taylor estimation is distorted. Using a higher-bit quantized model $Q_{B_{k-1}}^\star$ as a baseline (Fig 3(d)) shortens the perturbation distance, re-establishing Taylor validity while router fine-tuning ensures the baseline is near a local minimum.
    - **Design Motivation**: The challenge in low-bit quantization is the collapse of importance estimation under large perturbations. Progressive reduction segments large perturbations into controllable small ones, ensuring LP coefficients are based on "near-real" estimates—essentially a self-distillation multi-stage PTQ pipeline.

### Loss & Training
LP phase: Cross-entropy task loss as the objective over the calibration set. GPTQ uses its original reconstruction loss $\|Wx-\hat Wx\|^2$. Router fine-tuning: cross-entropy, lr=$1\mathrm{e}{-4}$, batch=1, weight decay=$1\mathrm{e}{-4}$, AdamW, 1 epoch. Calibration: 128 segments × 2048 tokens from WikiText2 (shared with quantization). Attention is fixed at 4-bit, expert candidates $\{1,2,3\}$ bit, group-wise asymmetric GPTQ (group size 128).

## Key Experimental Results

### Main Results
GEMQ vs mainstream MoE quantization methods on Mixtral-8×7B ("7-task Avg" is the 0-shot average from EleutherAI LM Harness):

| Method | bpe | WT2 PPL $\downarrow$ | C4 PPL $\downarrow$ | 7-task Avg $\uparrow$ |
|------|-----|----------|---------|-----------|
| FP Baseline | 16.0 | 3.84 | 7.40 | 70.97 |
| Uniform | 2.5 | 6.10 | 10.35 | 65.49 |
| PMQ | 2.5 | 5.10 | 9.21 | 64.34 |
| **GEMQ** | 2.5 | **5.03** | **9.02** | **65.13** |
| PMQ | 1.5 | 8.47 | 20.77 | 51.78 |
| **GEMQ** | 1.5 | **7.93** | **16.20** | **52.00** |
| SpQR | 1.5 | Inf | Inf | 31.87 |

Across four models (DeepSeekV2-Lite / Qwen1.5-MoE-A2.7B / Qwen3-30B-A3B / Mixtral-8×7B), GEMQ wins at all 1.5 / 2.0 / 2.5 / 3.0 bpe tiers, with particularly large gains at **extremely low (1.5) bits** (Qwen3-30B-A3B 1.5-bit: PMQ 34.59 C4 PPL $\rightarrow$ GEMQ 20.46). A 2.5-bit Mixtral model drops from 87 GB to 16 GB (−82%), decoding at 82.5 tokens/s on a single H100.

### Ablation Study
Component breakdown (based on Mixtral-8×7B, C4 PPL) and LP formula comparison:

| Configuration | 2.5-bit C4 PPL | 1.5-bit C4 PPL | Description |
|------|----------------|----------------|------|
| Uniform Baseline | 10.35 | 25.39 | Same bits per expert |
| + Local LP (PMQ) | 9.21 | 20.77 | Intra-layer allocation |
| + Global LP ($\Delta z^\top H(z)\Delta z$) | 9.10 (est) | 17.8 (est) | Cross-layer allocation |
| + Router Fine-tuning | 9.05 (est) | 16.6 (est) | Aligning routing to quantized experts |
| **+ Progressive (Full GEMQ)** | **9.02** | **16.20** | Closed-loop importance re-estimation |

LP formula ablation (Fig 4(b)): Using PMQ's formula globally provides limited gain; using two-step Hessian is moderate; using $\Delta z^\top H(z)\Delta z$ (GEMQ) compresses C4 PPL at 1.5 bpe from ~50 (naive) to ~17, proving that "moving error to MoE block output + diagonal Fisher approximation" is the core recipe for global LP.

### Key Findings
- **Inter-layer bit variance is crucial**: Fig 4(a) shows GEMQ allocates significantly different total bits to Mixtral layers (some high-bit, some almost entirely 1-bit), whereas PMQ forces a uniform budget per layer. This is why GEMQ excels at low budgets.
- **Router fine-tuning is extremely cheap yet highly effective**: Parameter count $< 0.04\%$, completes in one minute on three H100s (3.5% of total quantization time), but often yields a 1–3 PPL drop at 1.5 bpe.
- **Progressive reduction is vital for extremely low bits**: It is negligible at 3.0 bpe but "saving" at 1.5 bpe—"using Q2.5 to estimate Q1.5" is significantly more accurate than "using FP to estimate Q1.5."
- Global LP is hyperparameter-free, unlike PMQ which requires tuning fusion coefficients for activation frequency and weight statistics.

## Highlights & Insights
- Moving the expert bit allocation metric from "weight reconstruction error" to "task loss increment" via Gauss-Newton is an **elegant balance of theory and computability**—enjoying task-aware global comparability while only storing diagonal Fisher.
- Treating the router as "0.04% cheap PEFT parameters" for independent fine-tuning is widely applicable to any "prefix strategy + postfix execution" structure.
- Progressive quantization segments large perturbations into controllable steps, bringing **PTQ into the territory of QAT**—without backpropagating the whole model, only the router.
- Design philosophy: "Finding a way to make Taylor assumptions valid again" is smarter than "forcing complex formulas when Taylor has already failed."

## Limitations & Future Work
- Router fine-tuning uses cross-entropy task loss; in distribution shift scenarios (e.g., long-context, tool use), 128-segment calibration might overfit.
- Fixed 4-bit attention is a simplification; in 30B+ MoE, attention occupies substantial VRAM and could be included in the global LP.
- The progressive bit sequence (2.5 → 2.0 → 1.5) is manually defined; no automated strategy for step size is provided.
- The "at least one high-precision expert per layer" constraint is a mild regularizer but might prevent absolute optimal sparsity at extremely low bpe.

## Related Work & Insights
- **vs PMQ (Huang et al. 2024a)**: GEMQ is a strict superset of PMQ, enabling cross-layer allocation and removing fusion hyperparameters.
- **vs MoEQuant (Hu et al. 2025) / EAQuant (Fu et al. 2025)**: These focus on intra-expert quantization (outliers, calibration); GEMQ focuses on inter-expert allocation and router alignment, making them orthogonal.
- **vs SpQR (Dettmers et al. 2023)**: SpQR treats mixed precision at the sub-tensor level for dense models; MoE requires explicit expert-level modeling.
- **Insight**: GEMQ can inspire **General PTQ**—wherever "prefix decision + postfix execution" exists (e.g., sparse attention masks, KV cache retention), the "global task loss LP + prefix fine-tuning" pattern can be applied.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While global LP, router FT, and progressive steps have precedents, combining them into a theoretically consistent closed loop is a non-trivial integration.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High breadth across 4 models, 4 bpe tiers, and 7 tasks, including deployment speed.
- **Writing Quality**: ⭐⭐⭐⭐ Clean logic; Fig 3 illustrates the Taylor failure point clearly.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable engineering benefit for MoE deployment (82.5 tokens/s on one H100) and hyperparameter-free for new models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Geo-Expert: Fine-tuning 8B Models into Expert-Level Geological Reasoning LLMs using LoRA](geo-expert_towards_expert-level_geological_reasoning_via_parameter-efficient_fin.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[AAAI 2026\] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](../../AAAI2026/model_compression/dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)
- [\[ICML 2026\] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](../../ICCV2025/model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)

</div>

<!-- RELATED:END -->
