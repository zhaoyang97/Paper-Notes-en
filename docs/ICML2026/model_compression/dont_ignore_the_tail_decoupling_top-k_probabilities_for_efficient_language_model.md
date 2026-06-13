---
title: >-
  [Paper Note] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] Ours proposes TAD (Tail-Aware Distillation): explicitly decoupling teacher top-$K$ probabilities from "tail" probabilities in the standard KD KL divergence and magni…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "KL Divergence"
  - "Tail Probabilities"
  - "Pre-training Distillation"
  - "Causal Language Models"
date: 2026-05-08
content_hash: cf0978c1bb23a407
---

# Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation

**Conference**: ICML 2026  
**arXiv**: [2602.20816](https://arxiv.org/abs/2602.20816)  
**Code**: None  
**Area**: Model Compression / LLM Distillation  
**Keywords**: Knowledge Distillation, KL Divergence, Tail Probabilities, Pre-training Distillation, Causal Language Models

## TL;DR
Ours proposes TAD (Tail-Aware Distillation): explicitly decoupling teacher top-$K$ probabilities from "tail" probabilities in the standard KD KL divergence and magnifying the tail's contribution. This enables LLM pre-training distillation within academic-level compute (single H100 + 1 week), outperforming data-centric methods like MiniPLM.

## Background & Motivation

**Background**: Current LLM distillation primarily follows two paths. First, supervised distillation (MiniLLM/OnPolicyKD), which assumes a pre-trained student and requires expensive online generation. Second, pre-training distillation (DistilBERT), training from scratch but relying on original teacher corpora. Recently, MiniPLM adopted a data-centric approach where the teacher selects samples for the student to avoid online overhead.

**Limitations of Prior Work**: (1) On-policy methods require repeated token generation, resulting in PetaFLOPs 4–10x higher than vanilla KD, which is unsustainable for billion-token scales on academic compute. (2) Most causal LM pre-training corpora are closed-source, forcing distillation onto general corpora where "teacher argmax tokens" often disagree with "ground-truth tokens" (misalignment rates of 39%–46%), rendering class-based DKD strategies ineffective. (3) Standard KL divergence gradients are dominated by teacher top-$K$ modes, causing student tail probabilities to collapse towards 0, sacrificing diversity.

**Key Challenge**: In the KL divergence $\sum p^T \log(p^T/p^S)$, the multiplier $p^T_i$ for tail terms is near zero, contributing negligible loss. However, the student's tail distribution is critical for generation quality and diversity—yet directly increasing the tail weight $\beta$ causes training divergence.

**Goal**: Design a pre-training distillation loss that is (i) computationally comparable to vanilla KD, (ii) runnable on billions of tokens within academic budgets, and (iii) explicitly utilizes teacher tail information.

**Key Insight**: Inspired by Decoupled KD (DKD) in image classification, which splits KL into "target vs. non-target" classes. Since DKD's reliance on ground-truth labels fails in pre-training (where next-token and mode often mismatch), ours adopts a **rank-anchored** approach: partitioning based on teacher top-$K$ ranks instead of labels.

**Core Idea**: Decompose KL divergence by teacher probability rank into a top-$K$ term $\mathcal{D}_{KL_1}$ and a tail term $\alpha_K^T \mathcal{D}_{KL_2}$. The tail term is multiplied by a **sequence-normalized** coefficient $\beta(X)=\beta/\bar{\alpha}_K^T(X)$, providing a continuous "push" to tail gradients while maintaining training stability.

## Method

### Overall Architecture
TAD is a plug-in loss replacing the KL term in vanilla KD. The full training objective is $\mathcal{L}_{TAD}=\sum_t \mathcal{L}_{CLM}(t;\mathcal{P}^S)+\mathcal{L}_{DIV}(t;\mathcal{P}^T,\mathcal{P}^S)$, where $\mathcal{L}_{CLM}$ is the student's causal LM loss and $\mathcal{L}_{DIV}$ is the decoupled KL. The pipeline is entirely offline: teacher logits are computed in a single forward pass without student generation, matching vanilla KD's PetaFLOPs (9.3 vs. 9.2 for a 1.2B student, while MiniLLM requires 39).

### Key Designs

1.  **Top-K vs. Tail Probability Decoupling**:
    - **Function**: Splits the teacher distribution into two segments by probability rank, calculating KL separately and weighting the tail.
    - **Mechanism**: Let $\{\accentset{*}{p}^T_k\}_{k=1}^K$ be the top $K$ teacher probabilities and $\alpha_K^T=1-\sum_k \accentset{*}{p}^T_k$ be the total tail mass. Then $\mathcal{D}_{KL}(\mathcal{P}^T\|\mathcal{P}^S)=\mathcal{D}_{KL_1}+\alpha_K^T \mathcal{D}_{KL_2}$, where $\mathcal{D}_{KL_2}$ uses normalized tail probabilities $\tilde{p}=p/\alpha_K^T$. This ensures $\tilde p$ is a valid distribution even if original values are near zero.
    - **Design Motivation**: In vanilla KL gradients $\partial \mathcal{L}/\partial z_i=p_i^S-p_i^T$, $p_i^T$ is negligible for tail tokens. Gradients are overwhelmed by modes, leading to $\sum_k \accentset{*}{p}^S_k\approx 1$ and tail collapse. Decoupling allows independent weighting of the tail term.

2.  **β(X) Sequence-level Normalization**:
    - **Function**: Provides a controllable tail amplification factor $\beta$ while preventing divergence caused by a constant $\beta>1$.
    - **Mechanism**: Define $\beta(X)=\beta\,/\,\bar{\alpha}_K^T(X)$, where $\bar\alpha_K^T(X)=\frac{1}{N}\sum_{t=1}^N \alpha_K^T(t)$ is the average tail mass of the current sequence. The token-level loss is $\mathcal{L}_{DIV}(t)=D_{KL_1}(t)+\beta(X)\,\alpha_K^T(t)\,D_{KL_2}(t)$. Normalization allows stable convergence with nominal values like $\beta=1,2$.
    - **Design Motivation**: Naive weighting with constant $\beta>1$ failed to converge. Sequence normalization dynamically adjusts amplification based on the "tail thickness" of the current sequence.

3.  **Tail Gradient Compensation Mechanism**:
    - **Function**: Ensures convergence behavior remains consistent with vanilla KL (the fixed point remains $p_i^S=p_i^T$) while boosting tail probabilities during early training.
    - **Mechanism**: The tail logit gradient becomes $\partial \mathcal{L}_{DIV}/\partial z_i=(p_i^S-p_i^T)+(\beta(X)-1)\big(p_i^S\cdot\frac{1-\sum_k\accentset{*}p^T_k}{1-\sum_k\accentset{*}p^S_k}-p_i^T\big)$. When the student over-concentrates on modes ($\sum_k\accentset{*}p^S_k\ge \sum_k \accentset{*}p^T_k$), the compensation term pushes tail probabilities up. Once matched, the term zeros out.
    - **Design Motivation**: To escape the "mode-only" failure mode early on without shifting the convergence point (which would deviate the student from the teacher).

### Loss & Training
- **Loss**: $\mathcal{L}_{TAD}=\sum_t \mathcal{L}_{CLM}+\mathcal{L}_{DIV}$, with $K\in\{1,5,10,20\}$ and $\beta\in\{0.5,1,2,5,10\}$.
- **Data**: 20GB subset of Regmix (open-source Pile replication), ~5B tokens.
- **Compute**: Single H100, 1-week budget, processing ~2B tokens; PetaFLOPs identical to vanilla KD.
- **Initialization**: Teacher attention weights truncated to student hidden dimensions; MLP randomly initialized.

## Key Experimental Results

### Main Results

Qwen1.5-1.8B → {1.2B, 0.5B} students, pre-training distillation, average across 8 LMEH benchmarks:

| Student | Method | Avg Accuracy | Relative to Vanilla |
| :--- | :--- | :--- | :--- |
| 1.2B | CLM (no KD) | 45.0 | −0.7 |
| 1.2B | Vanilla KD | 45.6 | 0 |
| 1.2B | MiniPLM | 46.6 | +1.0 |
| 1.2B | **TAD (K=10)** | **47.8** | **+2.2** |
| 0.5B | Vanilla KD | 44.1 | 0 |
| 0.5B | MiniPLM | 45.0 | +1.0 |
| 0.5B | **TAD (K=10)** | **45.4** | **+1.5** |

**PetaFLOPs** (1M token subset): Vanilla 9.2 / MiniPLM 12.4 / **TAD 9.3** / MiniLLM 39 / Seq-KD 65. TAD is comparable to vanilla and significantly lower than on-policy methods.

### Ablation Study

| Configuration (1.2B Student) | Avg | Description |
| :--- | :--- | :--- |
| Vanilla KD (equivalent to $\beta=1$) | 45.6 | Baseline |
| TAD K=1, β=2 | 47.2 | Minimal decoupling (argmax vs. tail) |
| TAD K=10, β=2 | **47.8** | Optimal |
| TAD K=20, β=2 | 47.7 | Performance drops slightly with larger K |
| TAD K=10, β=0.5 | 47.0 | Insufficient tail amplification |
| TAD K=10, β=10 | 47.6 | Over-amplification |

### Key Findings
- **K=10 is the sweet spot**: Too small $K$ creates unstable tail mass; too large $K$ makes the tail too thin, causing TAD to degrade to vanilla KD.
- **β=2 is most stable**: With normalization, $\beta\in[1,5]$ converges, but $\beta=2$ is consistently best, suggesting "gentle" tail amplification is necessary.
- **Significant compute advantage**: TAD outperforms the data-centric MiniPLM at vanilla KD's FLOPs, meaning ~33% more tokens can be processed under the same budget.
- **Better Calibration**: F-ECE consistently decreased, indicating the student distribution shape genuinely matches the teacher rather than just aligning modes.

## Highlights & Insights
- Adapting DKD's "label-anchored" decoupling to "rank-anchored" is a key step: finding a label-free partition axis for pre-training scenarios without reliable ground truth.
- Sequence-level normalization $\beta(X)=\beta/\bar\alpha_K^T(X)$ is a simple yet highly effective stabilization technique, transferable to other "rare item amplification" losses.
- Gradient analysis shows TAD converges to the same fixed point as vanilla KL, acting as an "early-stage accelerator" that automatically disengages, requiring no manual scheduling.
- The fully offline design allows caching teacher logits, transforming distillation into a pure data-loading problem, which is highly accessible for academic labs.

## Limitations & Future Work
- Primarily focused on pre-training distillation; while SFT results were shown (GSM8K: TinyLlama-1.1B 36.8), a full comparison is missing. Robustness of $K$ and $\beta$ in SFT is unconfirmed.
- Max teacher size tested was Gemma-2 9B; effectiveness with 70B+ teachers is unverified. Sharper tail distributions in larger models may require retuning $K$.
- Evaluations relied on LMEH few-shot benchmarks; generation quality metrics (diversity, MAUVE) are missing, which are the primary intended benefits of preserving the tail.
- Caching teacher logits requires storing $|\mathcal{V}|$-dimensional probabilities; disk overhead becomes significant for large vocabularies (>100k).

## Related Work & Insights
- **vs. Vanilla KD (Hinton)**: TAD is a strict super-set that reduces to vanilla at $\beta=1$; it addresses the "mode-over-concentration" issue via tail weighting.
- **vs. DKD (CVPR 2022)**: While DKD uses ground-truth labels to split target/non-target classes, TAD uses teacher rank to split top-$K$/tail for label-free pre-training.
- **vs. MiniPLM**: MiniPLM is data-centric (sample selection), while TAD is loss-centric; the two are orthogonal and can be combined.
- **vs. MiniLLM / OnPolicyKD**: Unlike these on-policy methods requiring online generation, TAD is offline with an order of magnitude lower FLOPs.

## Rating
- Novelty: ⭐⭐⭐⭐ Clever bridge from DKD to pre-training via rank-anchored decoupling and normalization.
- Experimental Thoroughness: ⭐⭐⭐ Solid multi-teacher/parameter sweeps, but lacks generation diversity and comprehensive SFT evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear gradient derivations and a logical narrative flow (FLOPs + Tail Loss → Decoupling → Normalization).
- Value: ⭐⭐⭐⭐ Provides a practical, billion-token scale distillation recipe accessible to the academic community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](../../ACL2026/model_compression/efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] GradPower: Powering Gradients for Faster Language Model Pre-Training](gradpower_powering_gradients_for_faster_language_model_pre-training.md)

</div>

<!-- RELATED:END -->
