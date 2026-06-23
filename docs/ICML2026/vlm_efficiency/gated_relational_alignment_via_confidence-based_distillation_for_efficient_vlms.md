---
title: >-
  [Paper Note] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs
description: >-
  [ICML 2026][vlm_efficiency][Knowledge Distillation] This paper unifies Quantization-Aware Training (QAT) and Knowledge Distillation (KD) from an Information Bottleneck (IB) perspective, proposing the GRACE framework (Gated Decoupled Distillation + Relational Centered Kernel Alignment + Adaptive IB Controller). This enables INT4-quantized LLaVA / Qwen-VL to not only avoi
tags:
  - ICML 2026
  - vlm_efficiency
  - Knowledge Distillation
  - Information Bottleneck
date: 2026-05-08
content_hash: 15eced83d5c720cb
---
# Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs

**Conference**: ICML 2026  
**arXiv**: [2601.22709](https://arxiv.org/abs/2601.22709)  
**Code**: None  
**Area**: Multimodal VLM / Model Compression / Quantization-Aware Training  
**Keywords**: VLM Quantization, Knowledge Distillation, Information Bottleneck, CKA Relational Alignment, Confidence Gating

## TL;DR
This paper unifies Quantization-Aware Training (QAT) and Knowledge Distillation (KD) from an Information Bottleneck (IB) perspective, proposing the GRACE framework (Gated Decoupled Distillation + Relational Centered Kernel Alignment + Adaptive IB Controller). This enables INT4-quantized LLaVA / Qwen-VL to not only avoid performance degradation but outperform BF16 baselines across multiple benchmarks, while achieving 3× throughput and 54% memory savings.

## Background & Motivation

**Background**: VLM deployment is costly. PTQ (e.g., AWQ, GPTQ, MBQ) is the most common compression scheme, but aggressive INT4 quantization causes catastrophic performance drops due to the heterogeneous multimodal distributions being more complex than pure LLMs. While QAT is mature for LLMs, it remains nearly explored for VLMs. Simultaneously, knowledge distillation is extensively used independently for VLM compression.

**Limitations of Prior Work**: (1) PTQ directly applies quantization to the computational graph, failing to adapt the model to INT4 capacity limits; (2) Traditional QAT relies solely on task loss for supervision, providing no explicit guidance on "which information to retain" under low-bit budgets; (3) Standard distillation assumes all teacher tokens are equally reliable, but empirical evidence shows teacher entropy correlates significantly with error rates (Pearson $r=0.484$, binned $R^2=0.901$ on ScienceQA), meaning high-entropy tokens are essentially noise; (4) Logit distillation fails to transfer the attention structures of 13B teachers (e.g., 13B can localize a "banana" layer-by-layer, while 7B attention is scattered).

**Key Challenge**: Quantization is a "capacity allocation" problem (what info to keep), whereas distillation is a "supervision signal" problem (whom to learn from). Both are fundamentally addressed by the Information Bottleneck (IB) principle—compressing input representations while preserving task-relevant information—yet the community has treated them as independent techniques.

**Goal**: (1) Establish a theoretical bridge between QAT and KD; (2) Address the issue of uneven teacher supervision quality; (3) Transfer the teacher's visual relationship structure (beyond just logits) to the student; (4) Push INT4 performance close to or beyond BF16 in VLMs.

**Key Insight**: From the IB perspective $\max I(Z;Y) - \beta I(Z;X)$, quantization naturally provides a hard constraint on $I(Z;X)\le C_b$ (bit budget). Consequently, the teacher acts as a dense proxy for task-relevant information $Y_T$, where the KL divergence $D_{KL}(P_T\Vert P_S)$ represents the information gap between $I(X;Y_T)$ and $I(Z_S;Y_T)$ (Proposition 3.2).

**Core Idea**: Jointly optimize "hard capacity constraints" of quantization and "soft supervision" of teacher distillation using an IB framework, supplemented by confidence-gating and relational kernel alignment specifically designed for VLM characteristics.

## Method

### Overall Architecture
The teacher is a frozen BF16 model (e.g., LLaVA-1.5 13B), and the student is a smaller model (e.g., LLaVA-1.5 7B) using group-wise LSQ quantization (default INT4 / g=128). Both process the same input in parallel. The student receives three types of supervision: (i) Confidence-Gated DKD (decoupled + gated logit distillation); (ii) Relational CKA (CKA alignment on the Gram matrix of visual tokens in the penultimate layer, excluding text tokens); (iii) Adaptive IB Controller (monitors EMA-smoothed $\widehat{\mathcal{L}}_{GDKD}$ to dynamically adjust $\beta$). Weight $W$ and per-group scales $s$ are updated jointly.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    IN["Input: Same Image + Text"] --> T["Teacher (Frozen BF16)<br/>Outputs P_T, Visual Tokens V_T"]
    IN --> S["Student: Group-wise LSQ INT4<br/>Outputs P_S, Visual Tokens V_S"]
    T --> G["Gated Decoupled Distillation (GDKD)<br/>Weighted TCKD + NCKD by Teacher Entropy"]
    S --> G
    T --> R["Relational Centered Kernel Alignment (RCKA)<br/>CKA on Gram Matrix of Penultimate Visual Tokens"]
    S --> R
    G --> L["Total Loss L = L_CE + β(t)·L_GDKD + γ·L_RCKA"]
    R --> L
    S -->|Task Loss L_CE| L
    L --> C["Adaptive IB Controller<br/>Monitors EMA-smoothed L_GDKD to adjust β(t)"]
    C -->|Update β(t)| L
    L -->|STE Backprop, Joint Update of W and scale s| S
```

### Key Designs

**1. Gated Decoupled Knowledge Distillation (GDKD): Learning only from "sharpest" teacher posteriors**

Standard KD assumes all teacher tokens are equally credible. Empirical findings show teacher entropy is strongly correlated with error rates (ScienceQA Pearson $r=0.484$). GDKD does two things. First, it decouples distillation into target-class and non-target-class paths: TCKD $=D_{KL}([P_T^t,1-P_T^t]\|[P_S^t,1-P_S^t])$ captures confidence in the correct answer, while NCKD $=D_{KL}(\hat P_T^{nt}\|\hat P_S^{nt})$ transfers dark knowledge on renormalized non-target classes. Per-token DKD is defined as $\alpha\cdot \mathcal{L}_{TCKD}+\beta_{dkd}\cdot \mathcal{L}_{NCKD}$ with $\beta_{dkd}>\alpha$ to emphasize dark knowledge. Second, it uses entropy for token gating: $H_i=-\sum_v P_T^{(i)}(v)\log P_T^{(i)}(v)$, normalized as $\tilde h_i=H_i/\log|V|$, with weights $g_i=\exp(-\tilde h_i)$. Thus, $\mathcal{L}_{GDKD}=\sum_i g_i \mathcal{L}_{DKD}^{(i)}/\sum_i g_i$. Theorem 3.1 describes gating as a covariance correction $\mathcal{L}_{GDKD}=\bar{\mathcal{L}}_{DKD}+N\cdot \mathrm{Cov}(w_i,\mathcal{L}_{DKD}^{(i)})$—when entropy correlates with loss, this term is negative, strictly reducing expected error.

**2. Relational Centered Kernel Alignment (RCKA): Transferring visual structures on "which patches belong together"**

Logit distillation only transfers output distributions, failing to capture the core of visual reasoning—the relational structure between regions. RCKA performs CKA alignment on the Gram matrix of visual tokens in the penultimate layer: visual representations $V_T, V_S$ are L2-normalized to compute $K_T=\bar V_T \bar V_T^\top$ and $K_S=\bar V_S\bar V_S^\top$. After centering $\tilde K=HKH$, CKA is calculated as $\mathrm{HSIC}(K_T,K_S)/\sqrt{\mathrm{HSIC}(K_T,K_T)\mathrm{HSIC}(K_S,K_S)}$, and $\mathcal{L}_{RCKA}=1-\mathrm{CKA}(K_T,K_S)$. Unlike traditional RKD which computes inter-sample relations at the batch level, RCKA computes intra-sample relations among visual tokens (e.g., sky tokens show high similarity to other sky tokens and low similarity to airplane regions). Since CKA is scale-invariant to dimensions, teacher 13B and student 7B can be aligned without projection layers, naturally fitting cross-dimensional distillation.

**3. Adaptive IB Controller + Group-wise LSQ Quantization: Joint end-to-end optimization of constraints and supervision**

Static distillation weights are suboptimal—teacher supervision should be stronger early on, while task loss should dominate later. The author formulates the IB objective as $\min \mathcal{L}_{task}$ s.t. $\mathcal{L}_{distill}\le \tau$, which simplifies to a Lagrangian $\mathcal{L}_{task}+\beta(\mathcal{L}_{distill}-\tau)$. $\beta$ is dynamically adjusted by monitoring EMA-smoothed $\widehat{\mathcal{L}}_{GDKD}$. For quantization, group-wise LSQ is used: weights are flattened and grouped ($g=128$), with each group learning a log-space scale $s_i=\exp(\theta_i)$. Scales are initialized as $s_i^{(0)}=\mathrm{Percentile}_{99}(|W_i|)/Q_p$, and quantization follows $W_{i,q}=s_i\cdot \mathrm{clamp}(\lfloor W_i/s_i\rceil,-Q_n,Q_p)$ using STE. Group-wise granularity balances per-tensor and per-channel methods and matches MX hardware formats. Learnable scales allow the INT4 capacity constraint to be updated end-to-end with distillation signals.

### Loss & Training
Total objective: $\mathcal{L}=\mathcal{L}_{CE}+\beta(t)\cdot \mathcal{L}_{GDKD}+\gamma\cdot \mathcal{L}_{RCKA}$; where $\beta(t)$ is scheduled by the IB controller. The teacher is frozen; the student jointly optimizes $W$ and $\{s_i\}$. Proposition 3.2 provides a variational lower bound for the KL gap: $I(Z_S;Y_T)\ge I(X;Y_T)-\mathbb{E}[D_{KL}(P_T\|P_S)]$, indicating that minimizing $\mathcal{L}_{GDKD}$ maximizes mutual information between student representations and teacher knowledge.

## Key Experimental Results

### Main Results
Two backbone families: LLaVA-1.5 (7B/13B) and Qwen2-VL (2B/7B). Teachers are larger versions, while students are smaller versions quantized to INT4.

| Backbone | Bit | Method | SQA | MMBench | Notes |
|----------|-----|------|------|---------|------|
| LLaVA-1.5-7B | BF16 | baseline | 66.8 | – | Starting point |
| LLaVA-1.5-7B | INT4 | RTN/AWQ/GPTQ/MBQ | Sig. Drop | – | PTQ failure |
| LLaVA-1.5-7B | INT4 | **GRACE** | **70.1** | – | +3.3 over BF16 |
| Qwen2-VL-2B | BF16 | baseline | 73.7 | 72.6 | Starting point |
| Qwen2-VL-2B | INT4 | **GRACE** | **79.1** | **76.9** | +4–5 over BF16 |
| LLaVA-1.5-7B distilled (BF16) | – | GRACE | 69.0 avg | – | +3.8 over 7B baseline, near 13B teacher |

**Deployment Benefits**: Tested on real INT4 kernels, achieving 3× throughput and a 54% reduction in GPU memory.

### Ablation Study

| Config | Avg Accuracy | Description |
|------|---------|------|
| GRACE (full) | Highest | Full model |
| w/o IB Framework (Solo QAT / Naive QAT+KD) | Sig. Decrease | Validates necessity of joint IB optimization |
| w/o Confidence Gating | Moderate Decrease | High-entropy noise harms distillation |
| w/o RCKA | Decrease | Visual structures fail to transfer; 7B attention remains scattered |
| w/o Adaptive Controller (Fixed $\beta$) | Slightly Lower | Fails to switch focus during late training conflict |
| Per-tensor instead of group-wise | Sig. Decrease | VLM weight distributions require finer granularity |

### Key Findings
- INT4 surpassing BF16 is counter-intuitive for VLMs: The authors attribute this to the joint IB optimization of KD and QAT acting as an additional regularization; the BF16 baseline lacks teacher supervision.
- Confidence gating provides the most significant gains on tasks requiring long-chain reasoning (like SQA), as teacher tokens toward the end of long answers are often high-entropy.
- RCKA contributes most to MMBench, aligning with the benchmark's focus on visual relationship understanding. It transforms the attention maps of the INT4 student to focus layer-by-layer, similar to the 13B teacher.
- A group size of $g=128$ is the sweet spot; smaller sizes (g=64) offer marginal gains with more scales, while larger sizes (g=512) degrade accuracy.

## Highlights & Insights
- Using IB to unify QAT and KD lines—previously treated independently—is a clean theoretical framing. The dual decomposition of hard capacity constraints and soft supervision proxies can be applied to LoRA, pruning, and sparse training.
- The use of "teacher entropy = supervision quality proxy" is rigorously supported by Pearson $r$, binned $R^2$, and Fano's Inequality, proving more robust than prior empirical noise filtering works.
- Intra-sample CKA for visual tokens is a clever solution. Unlike traditional KD which requires dimension matching or lacks granularity, CKA is scale-invariant and directly captures critical visual structures like "sky pixel clustering."

## Limitations & Future Work
- Evaluation is limited to LLaVA and Qwen; transferability to multimodal generation (video, 3D) is unverified.
- Self-assessment: The teacher must be a larger version of the same architecture; feasibility of cross-architecture distillation (e.g., LLaVA to Qwen) was not tested.
- The INT4 surpassing BF16 phenomenon likely stems from "dark knowledge regularization." A control experiment with a distilled BF16 student would clarify if quantization itself provides any benefit beyond capacity restriction.

## Related Work & Insights
- **vs AWQ / GPTQ / MBQ**: These are PTQ methods relying on calibration sets for offline scales; GRACE is a QAT method where scales and weights are updated end-to-end with teacher supervision.
- **vs DKD (Zhao 2022)**: Borrows TCKD/NCKD decoupling but introduces entropy-gated weighting and the IB framework for sequence prediction.
- **vs RKD (Park 2019) / CKA-based KD (Saha 2022)**: While RKD focuses on inter-sample relations and CKA-KD on layer-wise alignment, this work focuses on intra-sample visual token relations specifically for VLMs.
- **Transferable Insights**: The IB "hard constraint + soft supervision" duality is applicable to any constrained-capacity learning. Entropy-gated supervision can be used in RLHF reward model training to filter noise.

## Rating
- Novelty: ⭐⭐⭐⭐ IB framing for QAT+KD is fresh; gated DKD and intra-sample CKA are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive benchmarks and real deployment metrics; lacks cross-architecture testing.
- Writing Quality: ⭐⭐⭐⭐ Strong combination of theory and empirical evidence (e.g., attention visualization).
- Value: ⭐⭐⭐⭐⭐ Provides the first INT4 solution to surpass BF16 for VLM deployment with demonstrated efficiency gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy](less_precise_can_be_more_reliable_a_systematic_evaluation_of_quantizations_impac.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](../../CVPR2026/vlm_efficiency/apet_approximation-error_guided_token_compression_for_efficient_vlms.md)
- [\[CVPR 2026\] Co-Me: Confidence Guided Token Merging for Visual Geometric Transformers](../../CVPR2026/vlm_efficiency/co-me_confidence_guided_token_merging_for_visual_geometric_transformers.md)
- [\[ICLR 2026\] BOLT: Decision‑Aligned Distillation and Budget-Aware Routing for Constrained Multimodal QA on Robots](../../ICLR2026/vlm_efficiency/bolt_decisionaligned_distillation_and_budget-aware_routing_for_constrained_multi.md)
- [\[CVPR 2026\] LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models](../../CVPR2026/vlm_efficiency/lift_and_place_a_simple_stable_and_effective_knowledge_distillation_framework_fo.md)

</div>

<!-- RELATED:END -->
