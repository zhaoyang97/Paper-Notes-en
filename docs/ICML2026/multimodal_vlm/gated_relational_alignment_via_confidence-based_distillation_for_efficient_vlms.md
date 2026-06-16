---
title: >-
  [Paper Note] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs
description: >-
  [ICML 2026][Multimodal VLM][Knowledge Distillation] This paper unifies Quantization-Aware Training (QAT) and Knowledge Distillation (KD) through the lens of the Information Bottleneck (IB) principle. It proposes the GRACE framework (Gated Decoupled Distillation + Relational Centered Kernel Alignment + Adaptive IB Controller), enabling INT4-quantized LLaVA / Qwen-VL to n
tags:
  - ICML 2026
  - Multimodal VLM
  - Knowledge Distillation
  - Information Bottleneck
date: 2026-05-08
content_hash: 2710efd1d94bebb5
---
# Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs

**Conference**: ICML 2026  
**arXiv**: [2601.22709](https://arxiv.org/abs/2601.22709)  
**Code**: None  
**Area**: Multimodal VLM / Model Compression / Quantization-Aware Training  
**Keywords**: VLM Quantization, Knowledge Distillation, Information Bottleneck, CKA Relational Alignment, Confidence Gating

## TL;DR
This paper unifies Quantization-Aware Training (QAT) and Knowledge Distillation (KD) through the lens of the Information Bottleneck (IB) principle. It proposes the GRACE framework (Gated Decoupled Distillation + Relational Centered Kernel Alignment + Adaptive IB Controller), enabling INT4-quantized LLaVA / Qwen-VL to not only avoid degradation but also exceed BF16 baselines on multiple benchmarks, while achieving 3× throughput and 54% memory savings in practice.

## Background & Motivation

**Background**: VLM deployment costs are high. Post-Training Quantization (PTQ) such as AWQ, GPTQ, and MBQ are common, but aggressive INT4 quantization causes catastrophic performance drops due to the heterogeneous multimodal distributions being more complex than pure LLMs. While QAT is mature for LLMs, it remains largely unexplored for VLMs. Meanwhile, KD is extensively used independently for VLM compression.

**Limitations of Prior Work**: (1) PTQ directly operates on the static computation graph, failing to adapt the model to INT4 capacity limits; (2) Traditional QAT relies solely on task loss for supervision, providing sparse guidance on what information to retain under a low-bit budget; (3) Standard KD assumes all teacher tokens are equally reliable, but empirical evidence shows teacher entropy is significantly correlated with error rate (Pearson $r=0.484$ and binned $R^2=0.901$ on ScienceQA), indicating high-entropy tokens are essentially noise; (4) Logit distillation fails to transfer the visual structural attention learned by 13B teachers (Figure 3 shows 13B models can localize "banana" layer-by-layer, while 7B attention is scattered).

**Key Challenge**: Quantization is a "capacity allocation" problem (what to keep), whereas distillation is a "supervision signal" problem (whom to learn from). Both are fundamentally addressed by IB—compressing input representation while preserving task-relevant information—yet the community has treated them as independent techniques.

**Goal**: (1) Build a theoretical bridge between QAT and KD; (2) Address the issue of non-uniform teacher supervision quality; (3) Transfer teacher visual relational structures (beyond simple logits) to the student; (4) Approximate or exceed BF16 performance with INT4 on VLMs.

**Key Insight**: From the IB perspective $\max I(Z;Y) - \beta I(Z;X)$, quantization naturally provides a hard constraint on $I(Z;X)\le C_b$ (bit budget). Consequently, the teacher acts as a dense proxy for task-relevant information $Y_T$. The KL divergence $D_{KL}(P_T\Vert P_S)$ represents the information gap between $I(X;Y_T)$ and $I(Z_S;Y_T)$ (Proposition 3.2).

**Core Idea**: Jointly optimize "hard quantization capacity constraints" and "soft teacher distillation supervision" within an IB framework, supplemented by confidence gating and relational kernel alignment specifically designed for VLM characteristics.

## Method

### Overall Architecture
The teacher is a frozen BF16 large model (e.g., LLaVA-1.5 13B), and the student is a smaller model (e.g., LLaVA-1.5 7B) using group-wise LSQ quantization (default INT4 / g=128). Both process the same input in parallel. The student is supervised by three components: (i) Confidence-Gated DKD (decoupled and gated logit distillation); (ii) Relational CKA (CKA alignment on the Gram matrix of visual tokens at the penultimate layer, excluding text tokens); (iii) Adaptive IB Controller (monitors EMA-smoothed $\widehat{\mathcal{L}}_{GDKD}$ to dynamically adjust $\beta$). Weight $W$ and per-group scale $s$ are updated jointly.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    IN["Input: Same Image + Text"] --> T["Teacher (Frozen BF16)<br/>Outputs P_T, visual tokens V_T"]
    IN --> S["Student: Group-wise LSQ INT4<br/>Outputs P_S, visual tokens V_S"]
    T --> G["Confidence-Gated Decoupled Distillation (GDKD)<br/>Teacher entropy weighted TCKD + NCKD"]
    S --> G
    T --> R["Relational Centered Kernel Alignment (RCKA)<br/>CKA on visual token Gram matrix at penultimate layer"]
    S --> R
    G --> L["Total Loss L = L_CE + β(t)·L_GDKD + γ·L_RCKA"]
    R --> L
    S -->|Task Loss L_CE| L
    L --> C["Adaptive IB Controller<br/>Monitors EMA-smoothed L_GDKD, scales β(t)"]
    C -->|Update β(t)| L
    L -->|STE Backprop, joint update of W and per-group scale s| S
```

### Key Designs

**1. Confidence-Gated Decoupled Distillation (GDKD): Learning only from the sharpest teacher posterior**

Standard distillation assumes all teacher tokens are equally trustworthy, yet empirical findings show teacher entropy is strongly correlated with error rates. GDKD executes two strategies. First, it decouples distillation into target-class and non-target-class paths: TCKD $=D_{KL}([P_T^t,1-P_T^t]\|[P_S^t,1-P_S^t])$ captures teacher confidence in the correct answer, while NCKD $=D_{KL}(\hat P_T^{nt}\|\hat P_S^{nt})$ transfers dark knowledge from renormalized non-target classes. Per-token DKD is defined as $\alpha\cdot \mathcal{L}_{TCKD}+\beta_{dkd}\cdot \mathcal{L}_{NCKD}$ where $\beta_{dkd}>\alpha$ to emphasize dark knowledge. Second, entropy-based token gating is applied: $H_i=-\sum_v P_T^{(i)}(v)\log P_T^{(i)}(v)$, normalized as $\tilde h_i=H_i/\log|V|$, with weights $g_i=\exp(-\tilde h_i)$ favoring high-confidence tokens: $\mathcal{L}_{GDKD}=\sum_i g_i \mathcal{L}_{KD}^{(i)}/\sum_i g_i$. Theorem 3.1 expresses the gating effect as a covariance correction $\mathcal{L}_{GDKD}=\bar{\mathcal{L}}_{DKD}+N\cdot \mathrm{Cov}(w_i,\mathcal{L}_{DKD}^{(i)})$—when entropy correlates with loss, this term is negative, strictly reducing expected distillation error. Fano's Inequality further provides an information-theoretic guarantee that higher entropy leads to a higher error lower bound.

**2. Relational Centered Kernel Alignment (RCKA): Transferring teacher's visual structure to the student**

Logit distillation only transfers output distributions, failing to convey the core of visual reasoning—the relational structure between regions. RCKA performs CKA alignment on the Gram matrix of visual tokens (excluding text tokens) at the penultimate layer. Visual representations $V_T, V_S$ are L2-normalized to compute $K_T=\bar V_T \bar V_T^\top$ and $K_S=\bar V_S\bar V_S^\top$. After centering $\tilde K=HKH$, the alignment is calculated via CKA $=\mathrm{HSIC}(K_T,K_S)/\sqrt{\mathrm{HSIC}(K_T,K_T)\mathrm{HSIC}(K_S,K_S)}$, with loss $\mathcal{L}_{RCKA}=1-\mathrm{CKA}(K_T,K_S)$. Unlike traditional RKD which computes inter-sample relations at the batch level, this computes intra-sample relations among visual tokens (e.g., visualizing high similarity between sky tokens). Since CKA is invariant to dimension scaling, it facilitates alignment between 13B teachers and 7B students without auxiliary projection layers.

**3. Adaptive IB Controller + Group-wise LSQ: Jointly optimizing hard capacity and soft supervision**

Fixed distillation weights are unsuitable across different training stages—teacher supervision should be stronger initially, while task loss should dominate later. Using an IB lens, the problem is formulated as $\min \mathcal{L}_{task}$ s.t. $\mathcal{L}_{distill}\le \tau$, optimized via the Lagrangian $\mathcal{L}_{task}+\beta(\mathcal{L}_{distill}-\tau)$. An EMA-smoothed $\widehat{\mathcal{L}}_{GDKD}$ monitors distillation progress to dynamically adjust $\beta$. Quantization employs group-wise LSQ: weights are flattened and grouped ($g=128$), with each group learning a log-space scale $s_i=\exp(\theta_i)$. Initialized with a 99th percentile $s_i^{(0)}=\mathrm{Percentile}_{99}(|W_i|)/Q_p$, the quantization is given by $W_{i,q}=s_i\cdot \mathrm{clamp}(\lfloor W_i/s_i\rceil,-Q_n,Q_p)$ using STE for backprop. Group-wise is finer than per-tensor and coarser than per-channel, matching MX hardware formats; making the scale learnable allows it to be fine-tuned end-to-end with distillation signals, effectively embedding INT4 hard capacity constraints into the optimization objective.

### Loss & Training
The total objective is $\mathcal{L}=\mathcal{L}_{CE}+\beta(t)\cdot \mathcal{L}_{GDKD}+\gamma\cdot \mathcal{L}_{RCKA}$, with $\beta(t)$ scheduled by the IB controller. The teacher is frozen, while the student jointly optimizes $W$ and $\{s_i\}$. Proposition 3.2 provides a variational lower bound for the KL gap $I(Z_S;Y_T)\ge I(X;Y_T)-\mathbb{E}[D_{KL}(P_T\|P_S)]$, indicating that minimizing $\mathcal{L}_{GDKD}$ maximizes the mutual information between student representations and teacher knowledge.

## Key Experimental Results

### Main Results
Two backbone families: LLaVA-1.5 (7B/13B) and Qwen2-VL (2B/7B). Teachers are larger versions, and students are smaller versions quantized to INT4.

| Backbone | Bit | Method | SQA | MMBench | Remarks |
|----------|-----|--------|-----|---------|---------|
| LLaVA-1.5-7B | BF16 | Baseline | 66.8 | – | Starting point |
| LLaVA-1.5-7B | INT4 | RTN/AWQ/GPTQ/MBQ | Sig. drop | – | PTQ failure |
| LLaVA-1.5-7B | INT4 | **GRACE** | **70.1** | – | +3.3 over BF16 |
| Qwen2-VL-2B | BF16 | Baseline | 73.7 | 72.6 | Starting point |
| Qwen2-VL-2B | INT4 | **GRACE** | **79.1** | **76.9** | +4–5 over BF16 |
| LLaVA-1.5-7B distilled (BF16) | – | GRACE | 69.0 avg | – | +3.8 over 7B baseline, near 13B teacher |

**Deployment Benefits**: Real-world INT4 kernel testing shows 3× throughput and a 54% reduction in GPU memory.

### Ablation Study

| Configuration | Avg Accuracy | Description |
|---------------|--------------|-------------|
| GRACE (full) | Highest | Full model |
| w/o IB Framework (Solo QAT / Naive QAT+KD) | Sig. Drop | Confirms necessity of IB joint optimization |
| w/o Confidence Gating | Moderate Drop | High-entropy token noise degrades distillation |
| w/o RCKA | Drop | Visual structure not transferred; 7B attention remains scattered |
| w/o Adaptive Controller (Fixed $\beta$) | Slightly lower | Fails to switch focus when distillation conflicts with task |
| Per-tensor quantization instead of group-wise | Sig. Drop | VLM heterogeneous weights require finer granularity |

### Key Findings
- The fact that INT4 outperforms BF16 on VLMs is counter-intuitive: the authors attribute this to joint IB optimization of distillation and quantization acting as an additional regularization; the BF16 baseline lacks teacher supervision.
- Confidence gating provides the most significant gains on long-chain reasoning tasks like ScienceQA, as teacher tokens at the end of long answers are typically high-entropy.
- RCKA contributes most to MMBench, consistent with its emphasis on visual relationship understanding; RCKA transforms INT4 student attention maps from scattered to "layer-wise focused," similar to 13B teachers.
- A group size of $g=128$ is the sweet spot; smaller ($g=64$) offers marginal gains with more scales, while larger ($g=512$) results in accuracy loss.

## Highlights & Insights
- Using IB to unify QAT and KD lines—previously treated independently—is a clean theoretical framing rather than just an engineering combination. This dual decomposition of hard capacity constraints and soft supervision proxies can be extended to LoRA, pruning, and sparse training.
- The use of "teacher entropy = supervision quality proxy" is rigorously supported by Pearson $r$, binned $R^2$, and Fano's Inequality, making it more robust than prior heuristic "self-distillation noise filtering" works.
- Intra-sample CKA alignment of visual tokens is ingenious—traditional KD aligns either logits (requires dim matching) or feature vectors (lacks granularity). CKA's dimensional invariance naturally handles 13B vs 7B discrepancies and directly captures critical visual structures like "sky pixels clustering."
- Theorem 3.1 explicitly formulates gating effects as a covariance term, providing a clear quantification of noisy distillation.

## Limitations & Future Work
- The evaluation is limited to LLaVA and Qwen backbones; transferability to multimodal generation (video, 3D) is unverified.
- Self-criticism: The teacher must be a larger version of the same architecture; cross-architecture distillation (e.g., LLaVA → Qwen) was not explored.
- The performance exceeding BF16 likely stems from "dark knowledge regularization" rather than "quantization being beneficial" itself. The authors should have included a BF16 student with distillation as a control group.
- Group-wise LSQ uses a fixed group size of 128 without per-layer adaptation; heterogeneous weight distributions across different layers (Vision Encoder vs LLM Decoder) might require different sizes.
- Future work could extend the IB controller to make quantization bit-width learnable for joint bit allocation.

## Related Work & Insights
- **vs AWQ / GPTQ / MBQ**: These are PTQ methods relying on calibration sets and weight rounding; GRACE is QAT where scales and weights are updated end-to-end with teacher supervision.
- **vs DKD (Zhao 2022)**: Adapts TCKD/NCKD decoupling but adds entropy-gated weighting and the IB framework; demonstrates its criticality in VLM sequence prediction beyond image classification.
- **vs RKD (Park 2019) / CKA-based KD (Saha 2022)**: RKD focuses on inter-sample relations; layer-wise CKA focuses on inter-layer alignment. GRACE uses intra-sample visual token alignment, designed specifically for VLM granularity.
- **vs LLM-QAT (Liu 2024)**: LLM-QAT performs QAT on pure text models; GRACE extends this to VLM and addresses multimodal weight heterogeneity.
- **Transferable Insights**: (1) The "Hard Constraint (Capacity) + Soft Supervision (Teacher)" duality is applicable to any constrained-capacity learning; (2) Entropy-gated supervision can filter noise in RLHF reward model training; (3) Intra-sample CKA is a general solution for cross-dimensional distillation.

## Rating
- Novelty: ⭐⭐⭐⭐ The joint IB framing of QAT+KD is new; gated DKD and intra-sample CKA have individual precedents but strong integration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two backbone series, multiple benchmarks, real INT4 deployment metrics; lacks cross-arch and BF16+KD controls.
- Writing Quality: ⭐⭐⭐⭐ Excellent fusion of theory and empirical evidence; compelling motivation through entropy correlations and attention visualization.
- Value: ⭐⭐⭐⭐⭐ Provides the first VLM deployment solution where INT4 exceeds BF16, offering high industrial utility with measured throughput and memory gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Deep Pre-Alignment for VLMs](deep_pre-alignment_for_vlms.md)
- [\[CVPR 2026\] Relational Visual Similarity](../../CVPR2026/multimodal_vlm/relational_visual_similarity.md)
- [\[CVPR 2025\] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders](../../CVPR2025/multimodal_vlm/move-kd_knowledge_distillation_for_vlms_with_mixture_of_visual_encoders.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](../../NeurIPS2025/multimodal_vlm/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)

</div>

<!-- RELATED:END -->
