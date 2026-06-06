---
title: >-
  [Paper Note] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] This paper proposes TAD (Tail-Aware Distillation): by explicitly separating the teacher's top-$K$ probabilities from the "tail" probabilities in the standard KD KL d…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "KL Divergence"
  - "Tail Probabilities"
  - "Pretraining Distillation"
  - "Causal Language Model"
date: 2026-05-08
content_hash: fdf07229d00123db
---

# Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation

**Conference**: ICML 2026  
**arXiv**: [2602.20816](https://arxiv.org/abs/2602.20816)  
**Code**: None  
**Area**: Model Compression / LLM Distillation  
**Keywords**: Knowledge Distillation, KL Divergence, Tail Probabilities, Pretraining Distillation, Causal Language Model

## TL;DR
This paper proposes TAD (Tail-Aware Distillation): by explicitly separating the teacher's top-$K$ probabilities from the "tail" probabilities in the standard KD KL divergence and amplifying the tail's contribution, it enables LLM pretraining distillation within academic-level compute (single H100 + 1 week), achieving average performance superior to data-centric methods like MiniPLM.

## Background & Motivation

**Background**: Current LLM distillation mainly follows two paths. The first is supervised distillation (MiniLLM / OnPolicyKD), which assumes the student is already pretrained and requires expensive online student generation; the second is pretraining distillation (DistilBERT), training the student from scratch but relying on the teacher's original training corpus. Recently, MiniPLM adopts a data-centric approach, where the teacher selects samples for the student, avoiding the overhead of online generation.

**Limitations of Prior Work**: (1) On-policy methods require repeated student token generation during training, resulting in PetaFLOPs 4–10 times that of vanilla KD, making it infeasible to process billions of tokens with academic compute; (2) Most causal LM pretraining corpora are closed-source, so distillation can only be performed on general corpora, where the "teacher argmax token" and "dataset ground-truth token" often mismatch (empirically 39%–46% mismatch rate), rendering DKD approaches designed for classification ineffective; (3) The gradient of standard KL divergence is dominated by the teacher's top-$K$, causing the student to only mimic modes, with tail probabilities collapsing to near zero, thus losing diversity.

**Key Challenge**: In the KL divergence $\sum p^T \log(p^T/p^S)$, the probability multiplier $p^T_i$ for tail terms is already close to zero, contributing little to the loss. However, the student's probability distribution over the tail is crucial for generation quality and diversity—directly amplifying the tail weight $\beta$ leads to training divergence.

**Goal**: Design a pretraining distillation loss that (i) has comparable overhead to vanilla KD, (ii) can process billions of tokens within an academic budget, and (iii) explicitly leverages the teacher's tail information.

**Key Insight**: Inspired by Decoupled KD (DKD) in image classification—which splits KL into "target class vs. non-target class" terms with separate weighting. However, DKD is anchored to the ground-truth label, which is unsuitable for pretraining distillation (since next-token and mode often mismatch). The authors switch to **rank-anchored**: splitting based on the teacher's top-$K$ rank instead of label.

**Core Idea**: Decompose KL divergence by teacher probability ranking into top-$K$ terms $\mathcal{D}_{KL_1}$ and tail terms $\alpha_K^T \mathcal{D}_{KL_2}$, multiplying the tail term by a **sequence-normalized** coefficient $\beta(X)=\beta/\bar{\alpha}_K^T(X)$, thus maintaining training stability while continuously injecting "thrust" into the tail gradient.

## Method

### Overall Architecture
TAD is a plug-in loss that replaces the KL term in vanilla KD. The complete training objective is $\mathcal{L}_{TAD}=\sum_t \mathcal{L}_{CLM}(t;\mathcal{P}^S)+\mathcal{L}_{DIV}(t;\mathcal{P}^T,\mathcal{P}^S)$, where $\mathcal{L}_{CLM}$ is the student's own causal LM loss, and $\mathcal{L}_{DIV}$ is the KL decoupled into top-$K$ vs. tail. The entire pipeline is fully offline: teacher logits are computed in a single forward pass, and the student does not need to generate, so the PetaFLOPs are on par with vanilla KD (1.2B student 9.3 vs. 9.2, 0.5B 6.5 vs. 6.4, while MiniLLM is 39 / 21.8).

### Key Designs

1. **Decoupling Top-K vs. Tail Probabilities**:

    - Function: Splits the teacher distribution by probability ranking into two segments, computes KL separately, and applies weighting to the tail.
    - Mechanism: Let $\{\accentset{*}{p}^T_k\}_{k=1}^K$ be the teacher's top $K$ probabilities, and $\alpha_K^T=1-\sum_k \accentset{*}{p}^T_k$ the total tail mass. Then $\mathcal{D}_{KL}(\mathcal{P}^T\|\mathcal{P}^S)=\mathcal{D}_{KL_1}+\alpha_K^T \mathcal{D}_{KL_2}$, where $\mathcal{D}_{KL_2}$ is computed using normalized tail probabilities $\tilde{p}=p/\alpha_K^T$ (ensuring a valid distribution even if the original tail probabilities are near zero).
    - Design Motivation: In vanilla KL, the gradient $\partial \mathcal{L}/\partial z_i=p_i^S-p_i^T$ has $p_i^T$ extremely small for tail tokens, so the gradient is completely dominated by modes, leading to $\sum_k \accentset{*}{p}^S_k\approx 1$ and tail collapse; decoupling allows independent weighting of the tail term.

2. **β(X) Sequence-level Normalization**:

    - Function: Uses a controllable tail amplification coefficient $\beta$, but avoids direct multiplication of a constant $\beta>1$ to KL, which would cause divergence.
    - Mechanism: Define $\beta(X)=\beta\,/\,\bar{\alpha}_K^T(X)$, where $\bar\alpha_K^T(X)=\frac{1}{N}\sum_{t=1}^N \alpha_K^T(t)$ is the average tail mass for the current sequence. The token-level loss is $\mathcal{L}_{DIV}(t)=D_{KL_1}(t)+\beta(X)\,\alpha_K^T(t)\,D_{KL_2}(t)$. Using a constant $\beta$ directly leads to unstable training; after batch/sequence normalization, nominal values like $\beta=1,2$ suffice for stable convergence.
    - Design Motivation: The authors found that naively weighting the loss with fixed $\beta>1$ does not converge. Normalization is equivalent to "dynamically adjusting the amplification factor according to the current sequence's tail size," automatically adapting to different teachers' tail thickness.

3. **Tail Gradient Compensation Mechanism**:

    - Function: Ensures convergence behavior matches vanilla KL (fixed point remains $p_i^S=p_i^T$), but initially "pushes up" the tail probabilities.
    - Mechanism: The gradient for tail logits becomes $\partial \mathcal{L}_{DIV}/\partial z_i=(p_i^S-p_i^T)+(\beta(X)-1)\big(p_i^S\cdot\frac{1-\sum_k\accentset{*}p^T_k}{1-\sum_k\accentset{*}p^S_k}-p_i^T\big)$. When $\sum_k\accentset{*}p^S_k\ge \sum_k\accentset{*}p^T_k$ (student mode is overly concentrated), the compensation term pushes up the student's tail probabilities; once the two match, the compensation term vanishes, reverting to vanilla KL behavior.
    - Design Motivation: The goal is to escape the "only learning modes" failure mode early on, without altering the convergence point (otherwise the student distribution would deviate from the teacher); gradient analysis yields this "automatic switch" design.

### Loss & Training

- Loss: $\mathcal{L}_{TAD}=\sum_t \mathcal{L}_{CLM}+\mathcal{L}_{DIV}$, $K\in\{1,5,10,20\}$, $\beta\in\{0.5,1,2,5,10\}$.
- Training data: Regmix (open-source Pile replica) 20GB subset, about 5B tokens.
- Compute: Single H100, 1 week budget, processes about 2B tokens; for 1.2B student, PetaFLOPs matches vanilla KD.
- Student initialization: Teacher attention weights truncated to student hidden dim (following DistilBERT), MLP randomly initialized.

## Key Experimental Results

### Main Results

Qwen1.5-1.8B → {1.2B, 0.5B} student, pretraining distillation, average over 8 LMEH benchmarks:

| Student | Method | Avg Accuracy | Relative to Vanilla |
|------|------|------|------|
| 1.2B | CLM (no KD) | 45.0 | −0.7 |
| 1.2B | Vanilla KD | 45.6 | 0 |
| 1.2B | MiniPLM | 46.6 | +1.0 |
| 1.2B | **TAD (K=10)** | **47.8** | **+2.2** |
| 0.5B | Vanilla KD | 44.1 | 0 |
| 0.5B | MiniPLM | 45.0 | +1.0 |
| 0.5B | **TAD (K=10)** | **45.4** | **+1.5** |

PetaFLOPs (1M token subset): Vanilla 9.2 / MiniPLM 12.4 / **TAD 9.3** / MiniLLM 39 / Seq-KD 65. TAD matches vanilla, far below on-policy methods.

On Phi-2 2.8B → 1.1B, TAD (K=10) averages 50.3, 1.2 higher than vanilla KD, and F-ECE (calibration error) drops from 1.45 to 1.37.

### Ablation Study

| Configuration (1.2B student) | Avg | Notes |
|------|------|------|
| Vanilla KD ($\beta=1$ equivalent) | 45.6 | Baseline |
| TAD K=1, β=2 | 47.2 | Minimal split between mode and tail |
| TAD K=10, β=2 | **47.8** | Optimal |
| TAD K=20, β=2 | 47.7 | Slight drop with larger K |
| TAD K=10, β=0.5 | 47.0 | Insufficient tail amplification |
| TAD K=10, β=10 | 47.6 | Over-amplification also drops |

### Key Findings
- **K=10 is optimal**: When $K$ is too small, "top" is almost only argmax, and the tail is too thick and unstable; when $K$ is too large, the tail is too thin and TAD degenerates to vanilla.
- **β=2 is most stable**: After normalization, $\beta\in[1,5]$ all converge, but $\beta=2$ is consistently best, indicating the tail needs "moderate" amplification.
- **Significant compute advantage**: TAD outperforms data selection methods like MiniPLM at the same FLOPs as vanilla KD, meaning about 33% more tokens can be processed under the same budget.
- **Better calibration**: F-ECE consistently decreases, indicating the student distribution shape becomes closer to the teacher, not just mode alignment.

## Highlights & Insights
- Replacing DKD's "label-anchored decoupling" with "rank-anchored decoupling" is key: it finds a label-free splitting axis for pretraining scenarios without reliable labels.
- Using $\beta(X)=\beta/\bar\alpha_K^T(X)$ for sequence-level normalization is a simple yet highly effective stabilization trick, transferable to other losses needing "rare item amplification" (e.g., logit adjustment for long-tail classification).
- Gradient analysis shows TAD and vanilla converge to the same fixed point, effectively adding a "front-end accelerator, back-end auto-off" switch to training, enabling natural transition without scheduling.
- Fully offline design allows caching teacher logits, turning distillation into a data loading problem, which is especially friendly for academia.

## Limitations & Future Work
- Only pretraining distillation is conducted; the authors also show SFT distillation (GSM8K: TinyLlama-1.1B 36.8, Llama2-7B 56.0) but without a complete comparison table. Whether $K$ and $\beta$ remain robust in SFT scenarios is unclear.
- The largest teacher tested is Gemma-2 9B; applicability to 70B+ teachers is unverified. The tail distribution is sharper in very large models, so $K$ may need adjustment.
- All evaluations are on LMEH few-shot; lacks open-ended generation quality (diversity, MAUVE) assessment, even though TAD's selling point is "retaining tail diversity," so evidence here is weak.
- Caching teacher logits requires storing $|\mathcal{V}|$-dimensional probabilities; with large vocabularies (>100k), disk overhead is non-negligible, and compression is not discussed.

## Related Work & Insights
- **vs. Vanilla KD (Hinton)**: TAD reduces to vanilla at $\beta=1$, making it a strict super-set; tail weighting addresses the classic "student mode over-concentration" issue.
- **vs. DKD (CVPR 2022)**: DKD splits target/non-target by ground-truth label; TAD splits top-$K$/tail by teacher probability rank, suitable for label-free pretraining.
- **vs. MiniPLM**: MiniPLM is data-centric (teacher selects samples), TAD is loss-centric (modifies loss); the two are orthogonal and can be combined. TAD outperforms MiniPLM at lower FLOPs.
- **vs. MiniLLM / OnPolicyKD**: These rely on online student generation; TAD is fully offline, with FLOPs an order of magnitude lower.

## Rating
- Novelty: ⭐⭐⭐⭐ Cleverly bridges DKD from classification to LM pretraining (rank-anchored decoupling + sequence normalization)
- Experimental Thoroughness: ⭐⭐⭐ Solid multi-teacher, multi-$K$, multi-$\beta$ comparisons, but lacks generation diversity evaluation and complete SFT comparison
- Writing Quality: ⭐⭐⭐⭐ Clear gradient analysis and concise loss formulation; coherent storyline (FLOPs + tail loss → decoupling → normalization)
- Value: ⭐⭐⭐⭐ Provides academia with a truly scalable billion-token distillation recipe, orthogonal and combinable with MiniPLM/AdamBC

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](../../ACL2026/model_compression/efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)
- [\[NeurIPS 2025\] LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups](../../NeurIPS2025/model_compression/lt-soups_bridging_head_and_tail_classes_via_subsampled_model_soups.md)
- [\[AAAI 2026\] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](../../AAAI2026/model_compression/eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)
- [\[NeurIPS 2025\] PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation](../../NeurIPS2025/model_compression/ppg-distill_efficient_photoplethysmography_signals_analysis_via_foundation_model.md)

</div>

<!-- RELATED:END -->
