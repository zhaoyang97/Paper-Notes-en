---
title: >-
  [Paper Note] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs
description: >-
  [ICML 2026][Multimodal VLM][VLM Quantization] This paper unifies Quantization-Aware Training (QAT) and Knowledge Distillation (KD) from the perspective of the Information Bottleneck (IB). It proposes the GRACE framework…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VLM Quantization"
  - "Knowledge Distillation"
  - "Information Bottleneck"
  - "CKA Relational Alignment"
  - "Confidence Gating"
date: 2026-05-08
content_hash: 443a9886eab6b55f
---

# Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs

**Conference**: ICML 2026  
**arXiv**: [2601.22709](https://arxiv.org/abs/2601.22709)  
**Code**: None  
**Area**: Multimodal VLM / Model Compression / Quantization-Aware Training  
**Keywords**: VLM Quantization, Knowledge Distillation, Information Bottleneck, CKA Relational Alignment, Confidence Gating

## TL;DR
This paper unifies Quantization-Aware Training (QAT) and Knowledge Distillation (KD) from the perspective of the Information Bottleneck (IB). It proposes the GRACE framework (Gated Decoupled Distillation + Relational Centered Kernel Alignment + Adaptive IB Controller), enabling INT4-quantized LLaVA/Qwen-VL to not only avoid performance degradation but also outperform BF16 baselines across multiple benchmarks, while achieving 3× throughput and 54% memory savings in practice.

## Background & Motivation

**Background**: VLMs have high deployment costs. PTQ (e.g., AWQ, GPTQ, MBQ) is the most common compression scheme, but aggressive INT4 quantization causes catastrophic performance drops in VLMs due to the complex heterogeneity of multimodal distributions compared to pure LLMs. While QAT is mature for LLMs, it remains largely unexplored for VLMs. Meanwhile, KD is extensively used independently for VLM compression.

**Limitations of Prior Work**: (1) PTQ directly applies to the static computation graph without adapting the model to INT4 capacity limits; (2) Traditional QAT relies solely on task loss for supervision, providing sparse guidance on what information to preserve under low-bit budgets; (3) Standard distillation assumes all teacher tokens are equally reliable, but empirical findings show teacher entropy significantly correlates with error rates (Pearson $r=0.484$, binned $R^2=0.901$ on ScienceQA), meaning high-entropy tokens are essentially noise; (4) Logit distillation fails to transfer the attention structures learned by 13B teachers on visual tokens (Figure 3 shows 13B can localize a "banana" layer-by-layer, while 7B attention is scattered).

**Key Challenge**: Quantization is a "capacity allocation" problem (what to keep), while distillation is a "supervisory signal" problem (whom to learn from). Both are fundamentally addressed by the IB—compressing input representations while preserving task-relevant information—yet the community treats them as independent techniques.

**Goal**: (1) Establish a theoretical bridge between QAT and KD; (2) Address the issue of non-uniform teacher supervision quality; (3) Transfer the teacher's visual relational structure (not just logits) to the student; (4) Push INT4 performance of VLMs to match or exceed BF16.

**Key Insight**: From the IB perspective of $\max I(Z;Y) - \beta I(Z;X)$, quantization naturally provides a hard constraint on $I(Z;X)\le C_b$ (bit budget). The teacher can act as a dense proxy for task-relevant information $Y_T$, where the KL divergence $D_{KL}(P_T\Vert P_S)$ represents the information gap between $I(X;Y_T)$ and $I(Z_S;Y_T)$ (Proposition 3.2).

**Core Idea**: Jointly optimize the "hard capacity constraint" of quantization and the "soft supervision" of teacher distillation using the IB framework, supplemented by two VLM-specific mechanisms: confidence gating and relational kernel alignment.

## Method

### Overall Architecture
The teacher is a frozen BF16 model (e.g., LLaVA-1.5 13B), and the student is a smaller model (e.g., LLaVA-1.5 7B) using group-wise LSQ quantization (default INT4 / g=128). Both process the same input in parallel. The student is supervised by three signals: (i) Confidence-Gated DKD (decoupled + gated logit distillation); (ii) Relational CKA (CKA alignment of visual token Gram matrices at the penultimate layer of the LLM, excluding text tokens); (iii) Adaptive IB Controller (monitors EMA-smoothed $\widehat{\mathcal{L}}_{GDKD}$ to dynamically adjust $\beta$). Weights $W$ and per-group scales $s$ are updated jointly.

### Key Designs

1.  **Confidence-Gated Decoupled Knowledge Distillation (GDKD)**:
    - **Function**: (a) Splits distillation into target-class (TCKD) and non-target-class (NCKD) streams to emphasize the "dark knowledge" in NCKD; (b) Uses teacher entropy for token-level gating to filter unreliable supervision.
    - **Mechanism**: TCKD = $D_{KL}([P_T^t,1-P_T^t]\|[P_S^t,1-P_S^t])$ captures the teacher's certainty about the correct answer; NCKD = $D_{KL}(\hat P_T^{nt}\|\hat P_S^{nt})$ performs KL on renormalized non-target classes to transfer dark knowledge. Token-wise DKD = $\alpha\cdot \mathcal{L}_{TCKD}+\beta_{dkd}\cdot \mathcal{L}_{NCKD}$, with $\beta_{dkd}>\alpha$. For gating, $H_i=-\sum_v P_T^{(i)}(v)\log P_T^{(i)}(v)$ is calculated per token, normalized to $\tilde h_i=H_i/\log|V|\in[0,1]$, and weighted by $g_i=\exp(-\tilde h_i)$ to prioritize high-confidence tokens. Final $\mathcal{L}_{GDKD}=\sum_i g_i \mathcal{L}_{DKD}^{(i)}/\sum_i g_i$. Theorem 3.1 shows gating corrects the covariance term: $\mathcal{L}_{GDKD}=\bar{\mathcal{L}}_{DKD}+N\cdot \mathrm{Cov}(w_i,\mathcal{L}_{DKD}^{(i)})$, proving it strictly reduces expected distillation error when entropy and loss are positively correlated.
    - **Design Motivation**: Empirical evidence shows teacher entropy is strongly correlated with error rates ($R^2=0.901$), and Fano's Inequality theoretically guarantees higher error lower bounds for higher entropy. Gating allocates distillation capacity to tokens where the teacher's posterior is sharpest, correcting the assumption that all supervision signals are equally important.

2.  **Relational Centered Kernel Alignment (RCKA)**:
    - **Function**: Aligns Gram matrices via CKA for visual tokens (excluding text) at the penultimate layer to transfer the teacher's relational structure (e.g., which patches should represent semantic clusters) to the student.
    - **Mechanism**: Visual token representations $V_T\in\mathbb{R}^{n\times d_T}$ and $V_S\in\mathbb{R}^{n\times d_S}$ are L2-normalized to compute $K_T=\bar V_T \bar V_T^\top$ and $K_S=\bar V_S\bar V_S^\top$. These are centered as $\tilde K=HKH$ where $H=I_n-\frac{1}{n}\mathbf{1}_n\mathbf{1}_n^\top$. CKA = $\mathrm{HSIC}(K_T,K_S)/\sqrt{\mathrm{HSIC}(K_T,K_T)\mathrm{HSIC}(K_S,K_S)}$, and $\mathcal{L}_{RCKA}=1-\mathrm{CKA}(K_T,K_S)$. Unlike traditional RKD which computes inter-sample relations at the batch level, this computes intra-sample relations between visual tokens (Figure 5 shows high similarity between "sky" tokens and low similarity with "airplane" tokens).
    - **Design Motivation**: Logit distillation only transfers output distributions, failing to capture the core of visual reasoning—how different regions relate. CKA is scale-invariant, allowing alignment without projection layers even when $d_T\ne d_S$ (e.g., 13B vs 7B), making it a natural bridge for cross-dimensional distillation.

3.  **Adaptive IB Controller + Group-wise LSQ Quantization**:
    - **Function**: (a) Dynamically adjusts the distillation weight $\beta$ to balance task loss and distillation loss; (b) Learns per-group quantization step sizes to integrate INT4 capacity constraints into the optimization objective.
    - **Mechanism**: Under the IB view, solving $\min \mathcal{L}_{task}$ s.t. $\mathcal{L}_{distill}\le \tau$ yields the Lagrangian $\mathcal{L}_{task}+\beta(\mathcal{L}_{distill}-\tau)$. The controller uses EMA-smoothed $\widehat{\mathcal{L}}_{GDKD}$ to monitor distillation progress and adjust $\beta$. For quantization, weight matrices are flattened and split into $G$ groups ($g=128$). Each group learns a scale $s_i=\exp(\theta_i)$. Initialization uses the 99th percentile: $s_i^{(0)}=\mathrm{Percentile}_{99}(|W_i|)/Q_p$. Quantization is done as $W_{i,q}=s_i\cdot \mathrm{clamp}(\lfloor W_i/s_i\rceil,-Q_n,Q_p)$, using STE for backpropagation.
    - **Design Motivation**: A fixed $\beta$ is suboptimal across training stages—teacher supervision should dominate early, while task loss should take over later. EMA + IB Lagrangian provides automatic scheduling. Group-wise LSQ is finer than per-tensor but coarser than per-channel, matching MX format hardware and allowing end-to-end fine-tuning of scales with distillation signals.

### Loss & Training
The total objective is $\mathcal{L}=\mathcal{L}_{CE}+\beta(t)\cdot \mathcal{L}_{GDKD}+\gamma\cdot \mathcal{L}_{RCKA}$, where $\beta(t)$ is scheduled by the IB controller. The teacher is frozen, while the student jointly optimizes $W$ and $\{s_i\}$. Proposition 3.2 provides a variational lower bound for the KL gap: $I(Z_S;Y_T)\ge I(X;Y_T)-\mathbb{E}[D_{KL}(P_T\|P_S)]$, indicating that minimizing $\mathcal{L}_{GDKD}$ maximizes the mutual information between student representations and teacher knowledge.

## Key Experimental Results

### Main Results
Two backbones: LLaVA-1.5 (7B/13B) and Qwen2-VL (2B/7B). Teachers are larger versions, while students are smaller versions quantized to INT4.

| Backbone | Bit | Method | SQA | MMBench | Remarks |
|----------|-----|------|------|---------|------|
| LLaVA-1.5-7B | BF16 | baseline | 66.8 | – | Starting point |
| LLaVA-1.5-7B | INT4 | RTN/AWQ/GPTQ/MBQ | Sig. Drop | – | PTQ fails |
| LLaVA-1.5-7B | INT4 | **GRACE** | **70.1** | – | Outperforms BF16 by +3.3 |
| Qwen2-VL-2B | BF16 | baseline | 73.7 | 72.6 | Starting point |
| Qwen2-VL-2B | INT4 | **GRACE** | **79.1** | **76.9** | Outperforms BF16 by +4–5 |
| LLaVA-1.5-7B distilled (BF16) | – | GRACE | 69.0 avg | – | +3.8 over 7B baseline, close to 13B teacher |

**Deployment Gains**: Real INT4 kernels achieve 3× throughput and a 54% reduction in VRAM.

### Ablation Study

| Configuration | Mean Accuracy | Description |
|------|---------|------|
| GRACE (full) | Highest | Full model |
| w/o IB Framework | Significant Drop | Validates need for joint optimization |
| w/o Confidence Gating | Moderate Drop | High-entropy noise harms distillation |
| w/o RCKA | Drop | Visual structure not transferred; 7B attention remains scattered |
| w/o Adaptive Controller | Slight Drop | Fails to switch when distill and task losses conflict |
| per-tensor vs group-wise | Significant Drop | Heterogeneous VLM weights need finer granularity |

### Key Findings
- INT4 surpassing BF16 is counter-intuitive in VLMs. The authors attribute this to "joint IB optimization of distillation and quantization acting as additional regularization," which the BF16 baseline lacks.
- Confidence gating provides the most gain in long-chain reasoning tasks like SQA, where teacher tokens at the end of long answers are often high-entropy.
- RCKA contributes most to MMBench, aligning with the benchmark's emphasis on visual relationship understanding. It transforms the INT4 student's attention maps (Figure 3) from scattered to focused, similar to the 13B teacher.
- A group size of $g=128$ is the sweet spot; smaller (g=64) adds overhead with little gain, while larger (g=512) loses precision.

## Highlights & Insights
- Framing QAT and KD within IB is a clean theoretical unification rather than just an engineering combination. The duality of hard capacity constraints and soft supervision can be extended to LoRA, pruning, and sparse training.
- The use of teacher entropy as a supervision quality proxy is rigorously validated via Pearson $r$, binned $R^2$, and Fano's Inequality, proving to be more robust than previous self-distillation noise filtering works.
- Intra-sample CKA for visual tokens is a clever approach. Traditional KD aligns logits (requiring matching dimensions) or feature vectors (lacking fine-grained structure). CKA’s scale-invariance solves the 7B vs 13B dimension gap and preserves critical "sky pixel clustering" structures.
- Theorem 3.1 explicitly quantifies gating as a covariance term, providing a theoretical foundation for noisy distillation beyond empirical weighting.

## Limitations & Future Work
- Evaluation is limited to LLaVA and Qwen backbones; transferability to multimodal generation (video, 3D) is not verified.
- The teacher must be a larger version of the same architecture; cross-architecture distillation (e.g., LLaVA → Qwen) was not explored.
- The "INT4 surpassing BF16" phenomenon likely stems from teacher-provided dark knowledge regularization rather than quantization being inherently beneficial. A control experiment with a distilled BF16 student is needed to prevent the misconception that aggressive quantization is always better.
- Group size is currently fixed at $g=128$ without per-layer adaptation; different components (Vision Encoder vs LLM Decoder) might favor different granularities.
- Future work could extend the IB controller to learn bit-widths dynamically for joint bit allocation.

## Related Work & Insights
- **vs AWQ / GPTQ / MBQ**: These are PTQ methods optimizing scales offline using calibration sets. GRACE is a QAT method updating scales and weights end-to-end with teacher supervision, allowing it to outperform BF16 where PTQ fails.
- **vs DKD (Zhao 2022)**: While adopting TCKD/NCKD decoupling, GRACE adds entropy-gating and an IB framework, proving that DKD is as critical for VLM sequence prediction as it was for image classification.
- **vs RKD (Park 2019) / CKA-based KD (Saha 2022)**: RKD focuses on inter-sample relations; layer-wise CKA aligns across layers. GRACE focuses on intra-sample visual token relations, providing finer granularity tailored for VLMs.
- **vs LLM-QAT (Liu 2024)**: Focused purely on text LLMs, whereas GRACE addresses multimodal heterogeneity.
- **Transferable Insights**: (1) The IB "hard constraint + soft supervision" duality applies to any constrained-capacity learning (pruning, low-rank); (2) Entropy-gated supervision can be used to filter noise in RLHF reward model training; (3) Intra-sample CKA is a universal solution for cross-dimensional distillation.

## Rating
- Novelty: ⭐⭐⭐⭐ The IB unification of QAT+KD is new; gated DKD and intra-sample CKA are innovative, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered two backbone families, multiple benchmarks, and real deployment metrics; lacks cross-architecture and BF16-distilled student controls.
- Writing Quality: ⭐⭐⭐⭐ Strong combination of theory and empiricism; motivation is well-supported by both entropy correlation and visualization.
- Value: ⭐⭐⭐⭐⭐ Provides the first INT4 solution for VLMs that outperforms BF16, with practical throughput and memory gains, making it highly valuable for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Deep Pre-Alignment for VLMs](deep_pre-alignment_for_vlms.md)
- [\[CVPR 2026\] Relational Visual Similarity](../../CVPR2026/multimodal_vlm/relational_visual_similarity.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](../../CVPR2026/multimodal_vlm/linking_perception_confidence_and_accuracy_in_mllms.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](../../NeurIPS2025/multimodal_vlm/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)

</div>

<!-- RELATED:END -->
