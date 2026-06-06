---
title: >-
  [Paper Note] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs
description: >-
  [ICML 2026][Multimodal VLM][VLM Quantization] This paper unifies quantization-aware training (QAT) and knowledge distillation (KD) from the Information Bottleneck (IB) perspective…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VLM Quantization"
  - "Knowledge Distillation"
  - "Information Bottleneck"
  - "CKA Relational Alignment"
  - "Confidence Gating"
date: 2026-05-08
content_hash: 1fc9aec2b2b49367
---

# Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs

**Conference**: ICML 2026  
**arXiv**: [2601.22709](https://arxiv.org/abs/2601.22709)  
**Code**: None  
**Area**: Multimodal VLM / Model Compression / Quantization-Aware Training  
**Keywords**: VLM Quantization, Knowledge Distillation, Information Bottleneck, CKA Relational Alignment, Confidence Gating

## TL;DR
This paper unifies quantization-aware training (QAT) and knowledge distillation (KD) from the Information Bottleneck (IB) perspective, proposing the GRACE framework (confidence-gated decoupled distillation + relational centered kernel alignment + adaptive IB controller). This enables INT4-quantized LLaVA / Qwen-VL models not only to avoid performance drops but to surpass BF16 baselines on multiple benchmarks, achieving 3× throughput and 54% memory savings in real-world deployment.

## Background & Motivation

**Background**: VLM deployment is costly. Post-training quantization (PTQ, e.g., AWQ, GPTQ, MBQ) is the most common compression method, but aggressive INT4 quantization leads to catastrophic performance drops in VLMs (due to greater multimodal distribution heterogeneity than pure LLMs). QAT is mature for LLMs but almost unexplored for VLMs. Meanwhile, knowledge distillation is widely used independently in VLM compression.

**Limitations of Prior Work**: (1) PTQ directly modifies the computation graph, preventing the model from adapting to INT4 capacity constraints; (2) Traditional QAT uses only task loss as supervision, offering little explicit guidance on "what information to retain under low bit budgets," resulting in sparse supervision; (3) Standard distillation assumes all teacher tokens are equally reliable, but empirical evidence shows teacher entropy is strongly correlated with error rate (Pearson $r=0.484$, binned $R^2=0.901$ on ScienceQA), with high-entropy tokens being noisy; (4) Logit distillation cannot transfer the attention structure learned by a 13B teacher on visual tokens (Figure 3 shows 13B can localize "banana" layer by layer, while 7B attention is scattered).

**Key Challenge**: Quantization is a "capacity allocation" problem (what information to retain), while distillation is a "supervision signal" problem (whom to learn from); both are essentially addressed by IB—compressing input representations while retaining task-relevant information—yet the community has treated them as independent techniques.

**Goal**: (1) Establish a theoretical bridge between QAT and KD; (2) Address the issue of uneven teacher supervision quality; (3) Effectively transfer the teacher's visual relational structure (not just logits) to the student; (4) Achieve INT4 performance on VLMs that matches or surpasses BF16.

**Key Insight**: From the IB perspective $\max I(Z;Y) - \beta I(Z;X)$, quantization naturally imposes a hard constraint $I(Z;X)\le C_b$ (bit budget), making the teacher a dense proxy for task-relevant information $Y_T$. The KL divergence $D_{KL}(P_T\Vert P_S)$ precisely captures the information gap between $I(X;Y_T)$ and $I(Z_S;Y_T)$ (Proposition 3.2).

**Core Idea**: Use the IB framework to jointly optimize "hard capacity constraints from quantization" and "soft supervision from teacher distillation," enhanced by confidence gating and relational kernel alignment tailored for VLMs.

## Method

### Overall Architecture
The teacher is a frozen BF16 large model (e.g., LLaVA-1.5 13B), and the student is a group-wise LSQ quantized (default INT4 / g=128) smaller model (e.g., LLaVA-1.5 7B). Both process the same input in parallel, with the student supervised by: (i) Confidence-Gated DKD (decoupled and gated logit distillation); (ii) Relational CKA (CKA alignment of Gram matrices for visual tokens, excluding text tokens, at the penultimate LLM layer); (iii) Adaptive IB Controller (monitors EMA-smoothed $\widehat{\mathcal{L}}_{GDKD}$ to dynamically adjust $\beta$). Both weights $W$ and per-group scales $s$ are jointly updated.

### Key Designs

1. **Confidence-Gated Decoupled Knowledge Distillation (GDKD)**:

    - **Function**: (a) Splits distillation into target-class (TCKD) and non-target-class (NCKD) branches, emphasizing the "dark knowledge" in NCKD; (b) Uses teacher entropy for token-level gating, filtering out unreliable supervision.
    - **Mechanism**: TCKD = $D_{KL}([P_T^t,1-P_T^t]\|[P_S^t,1-P_S^t])$ captures the teacher's confidence in the correct answer; NCKD = $D_{KL}(\hat P_T^{nt}\|\hat P_S^{nt})$ computes KL on renormalized non-target classes, transferring dark knowledge. Per-token DKD = $\alpha\cdot \mathcal{L}_{TCKD}+\beta_{dkd}\cdot \mathcal{L}_{NCKD}$, with $\beta_{dkd}>\alpha$. For gating, compute $H_i=-\sum_v P_T^{(i)}(v)\log P_T^{(i)}(v)$, normalize $\tilde h_i=H_i/\log|V|\in[0,1]$, and set weight $g_i=\exp(-\tilde h_i)$ so high-confidence teacher tokens are weighted more. Final loss: $\mathcal{L}_{GDKD}=\sum_i g_i \mathcal{L}_{DKD}^{(i)}/\sum_i g_i$. Theorem 3.1 shows gating is equivalent to correcting the covariance term: $\mathcal{L}_{GDKD}=\bar{\mathcal{L}}_{DKD}+N\cdot \mathrm{Cov}(w_i,\mathcal{L}_{DKD}^{(i)})$; when entropy and loss are positively correlated, this term is negative, proving gating strictly reduces expected distillation error.
    - **Design Motivation**: Empirical evidence shows teacher entropy is strongly correlated with error rate ($R^2=0.901$), and the Fano inequality guarantees higher entropy implies a higher lower bound on error. Gating thus allocates distillation capacity to tokens with the sharpest teacher posteriors, correcting the implicit assumption that all supervision signals are equally important.

2. **Relational Centered Kernel Alignment (RCKA)**:

    - **Function**: At the penultimate LLM layer, aligns the Gram matrices of visual tokens (excluding text tokens) between teacher and student, transferring the teacher's relational structure (which patches should be semantically grouped).
    - **Mechanism**: Extract teacher/student visual token representations $V_T\in\mathbb{R}^{n\times d_T}$ and $V_S\in\mathbb{R}^{n\times d_S}$, row L2-normalize, compute $K_T=\bar V_T \bar V_T^\top$, $K_S=\bar V_S\bar V_S^\top$; center $\tilde K=HKH$, $H=I_n-\frac{1}{n}\mathbf{1}_n\mathbf{1}_n^\top$; CKA = $\mathrm{HSIC}(K_T,K_S)/\sqrt{\mathrm{HSIC}(K_T,K_T)\mathrm{HSIC}(K_S,K_S)}$, loss $\mathcal{L}_{RCKA}=1-\mathrm{CKA}(K_T,K_S)$. Key difference: previous RKD computes inter-sample relations at the batch level, while this work computes intra-sample relations among visual tokens (Figure 5 visualizes high similarity among sky tokens and low similarity with airplane regions).
    - **Design Motivation**: Logit distillation only transfers output distributions, not the core of visual reasoning—"which regions should be associated." CKA is invariant to dimensionality, so $d_T\ne d_S$ (teacher 13B vs student 7B) requires no projection layer, making it a natural bridge for cross-dim distillation.

3. **Adaptive IB Controller + Group-wise LSQ Quantization**:

    - **Function**: (a) Dynamically adjusts distillation weight $\beta$ to balance task and distill losses; (b) Learns per-group quantization step sizes, directly embedding INT4's hard capacity constraint into the optimization objective.
    - **Mechanism**: From the IB perspective, solve $\min \mathcal{L}_{task}$ s.t. $\mathcal{L}_{distill}\le \tau$, with dual Lagrangian $\mathcal{L}_{task}+\beta(\mathcal{L}_{distill}-\tau)$. The controller uses EMA-smoothed $\widehat{\mathcal{L}}_{GDKD}$ to monitor distillation progress and dynamically adjust $\beta$. For quantization: flatten the weight matrix, split into $G$ groups of $g=128$, each with a learnable scale $s_i=\exp(\theta_i)$ (log-space ensures positivity), initialized as $s_i^{(0)}=\mathrm{Percentile}_{99}(|W_i|)/Q_p$, quantize $W_{i,q}=s_i\cdot \mathrm{clamp}(\lfloor W_i/s_i\rceil,-Q_n,Q_p)$, and use STE for backpropagation.
    - **Design Motivation**: Fixed $\beta$ is suboptimal at different training stages—stronger teacher supervision is needed early, while task loss should dominate later. EMA + IB Lagrangian enables automatic scheduling. Group-wise LSQ is finer than per-tensor but coarser than per-channel, matching MX-format hardware, and treating scale as a learnable parameter allows end-to-end tuning with the distillation signal.

### Loss & Training
The total objective is $\mathcal{L}=\mathcal{L}_{CE}+\beta(t)\cdot \mathcal{L}_{GDKD}+\gamma\cdot \mathcal{L}_{RCKA}$; $\beta(t)$ is scheduled by the IB controller. The teacher is frozen; the student jointly optimizes $W$ and $\{s_i\}$. Proposition 3.2 provides a variational lower bound for the KL gap $I(Z_S;Y_T)\ge I(X;Y_T)-\mathbb{E}[D_{KL}(P_T\|P_S)]$, indicating that minimizing $\mathcal{L}_{GDKD}$ maximizes the mutual information between student representations and teacher knowledge.

## Key Experimental Results

### Main Results
Two backbone types: LLaVA-1.5 (7B/13B) and Qwen2-VL (2B/7B); the teacher is the larger version, the student is the smaller version quantized to INT4.

| Backbone | Bit | Method | SQA | MMBench | Note |
|----------|-----|--------|------|---------|------|
| LLaVA-1.5-7B | BF16 | baseline | 66.8 | – | Starting point |
| LLaVA-1.5-7B | INT4 | RTN/AWQ/GPTQ/MBQ | Significant drop | – | All PTQ methods fail |
| LLaVA-1.5-7B | INT4 | **GRACE** | **70.1** | – | Surpasses BF16 by +3.3 |
| Qwen2-VL-2B | BF16 | baseline | 73.7 | 72.6 | Starting point |
| Qwen2-VL-2B | INT4 | **GRACE** | **79.1** | **76.9** | Surpasses BF16 by +4–5 |
| LLaVA-1.5-7B distilled (BF16) | – | GRACE | 69.0 avg | – | +3.8 over 7B baseline, close to 13B teacher |

**Deployment Gains**: Real INT4 kernel achieves 3× throughput and 54% memory reduction.

### Ablation Study

| Configuration | Avg. Accuracy | Description |
|---------------|--------------|-------------|
| GRACE (full) | Highest | Complete model |
| w/o IB framework (plain QAT alone / naive QAT+KD stacking) | Significant drop | Validates necessity of IB joint optimization |
| w/o Confidence Gating | Moderate drop | High-entropy token noise harms distillation |
| w/o RCKA | Drop | Visual relational structure not transferred; 7B attention remains scattered |
| w/o Adaptive Controller (fixed $\beta$) | Slightly lower | Cannot switch when distill and task losses conflict late in training |
| Per-tensor quantization instead of group-wise | Significant drop | VLM heterogeneous weight distributions require finer granularity |

### Key Findings
- INT4 surpassing BF16 in VLMs is counterintuitive; the authors attribute this to "joint IB optimization of distillation + quantization acting as extra regularization," whereas the BF16 baseline lacks teacher supervision.
- Confidence gating yields the largest gains on tasks like SQA that require long-chain reasoning, as teacher tokens at the end of long answers tend to have high entropy.
- RCKA contributes most on MMBench, consistent with its emphasis on visual relational understanding; RCKA also transforms INT4 student attention maps (Figure 3) from scattered to layer-wise focused, similar to the 13B teacher.
- Group size $g=128$ is the sweet spot; smaller (g=64) yields marginal gains but increases scale count, larger (g=512) reduces accuracy.

## Highlights & Insights
- Using IB to unify QAT and KD, previously independent technical lines, is a clean theoretical framing, not just an engineering combination: the dual decomposition of hard capacity constraint + soft supervision proxy can be directly applied to LoRA, pruning, sparse training, and any "constrained capacity + large teacher" scenario.
- "Teacher entropy as a supervision quality proxy" is supported by triple evidence: Pearson $r$, binned $R^2$, and the Fano inequality. This is more robust than prior "self-distillation noise filtering" work, and the conclusion can be plugged into any KD framework.
- Intra-sample CKA alignment of visual tokens is a clever idea—traditional KD aligns either logits (requiring matching dims) or feature vectors (not fine-grained enough); CKA's scale invariance naturally solves the 7B vs 13B dim mismatch, and the relational matrix directly captures key VLM visual structures like "sky pixels cluster together."
- Theorem 3.1 explicitly expresses the effect of gating as a covariance term, providing a clear quantification of noisy distillation, one step beyond empirical confidence weighting.

## Limitations & Future Work
- Only LLaVA / Qwen backbone series are evaluated; transferability to multimodal generation (video, 3D) is untested.
- Self-assessment: the teacher must be a larger, same-architecture BF16 model; cross-architecture distillation (e.g., LLaVA → Qwen) is not explored.
- INT4's outperformance over BF16 may stem from "dark knowledge regularization provided by the teacher," not "quantization itself being beneficial." The authors should add a control where a BF16 student also receives distillation; otherwise, the conclusion may be misinterpreted as "more aggressive quantization is always better."
- Group-wise LSQ uses a fixed group=128; there is no per-layer adaptation. Different layers (visual encoder vs LLM decoder) may have weight distributions unsuitable for the same group size.
- Future work could extend the IB controller to make quantization bit width learnable, enabling joint bit allocation.

## Related Work & Insights
- **vs AWQ / GPTQ / MBQ**: These are all PTQ methods, optimizing offline scales via calibration sets and weight rounding; GRACE is QAT, updating scales and weights end-to-end with teacher supervision, enabling INT4 to surpass PTQ, which all fail.
- **vs DKD (Zhao 2022)**: This work adopts TCKD/NCKD decoupling but adds entropy-gated weighting and the IB framework; DKD was originally for image classification, but this work shows its importance for VLM sequence prediction.
- **vs RKD (Park 2019) / CKA-based KD (Saha 2022)**: RKD computes inter-sample relations within a batch, layer-wise CKA aligns across layers; this work aligns intra-sample visual token relations, offering finer granularity and VLM-specific design.
- **vs LLM-QAT (Liu 2024)**: LLM-QAT applies QAT to pure-text LLMs, not involving visual modalities; this work extends to VLMs and addresses cross-modal weight distribution heterogeneity.
- **Transferable Insights**: (1) The IB duality of "hard constraint (capacity) + soft supervision (teacher)" can be generalized to any constrained-capacity learning (pruning, low-rank, binary networks); (2) Entropy-gated supervision can be directly applied to RLHF reward model training to filter noise; (3) Intra-sample CKA is a universal solution for any cross-dim distillation.

## Rating
- Novelty: ⭐⭐⭐⭐ The IB joint QAT+KD framing is new; gated DKD and intra-sample CKA are each innovative; each component has prior art individually.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two backbone series, multiple benchmarks, real INT4 deployment, complete ablation; lacks cross-architecture distillation and BF16 student+distillation controls.
- Writing Quality: ⭐⭐⭐⭐ Strong integration of theory and empirical evidence; motivation section uses both entropy-error correlation and attention visualization for high persuasiveness; formulas are numerous but well explained.
- Value: ⭐⭐⭐⭐⭐ Provides the first INT4 solution for VLM deployment that surpasses BF16, with measured throughput and memory gains, offering significant industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Relational Visual Similarity](../../CVPR2026/multimodal_vlm/relational_visual_similarity.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](../../CVPR2026/multimodal_vlm/linking_perception_confidence_and_accuracy_in_mllms.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](../../NeurIPS2025/multimodal_vlm/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)
- [\[ICLR 2026\] MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs](../../ICLR2026/multimodal_vlm/mmtok_multimodal_coverage_maximization_for_efficient_inference_of_vlms.md)

</div>

<!-- RELATED:END -->
