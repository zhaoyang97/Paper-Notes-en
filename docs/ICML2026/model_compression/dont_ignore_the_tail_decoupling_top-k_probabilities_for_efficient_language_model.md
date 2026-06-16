---
title: >-
  [Paper Note] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] Ours proposes TAD (Tail-Aware Distillation): It explicitly decouples teacher top-$K$ probabilities from "tail" probabilities within the standard KD KL divergence and amplifies the tail contribution. This allows LLM pre-training distillation to be completed within academic-scale compute (single H100 + 1 week), outperfor
tags:
  - ICML 2026
  - Model Compression
  - Knowledge Distillation
date: 2026-05-08
content_hash: b8e9162b74de65e8
---
# Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation

**Conference**: ICML 2026  
**arXiv**: [2602.20816](https://arxiv.org/abs/2602.20816)  
**Code**: None  
**Area**: Model Compression / LLM Distillation  
**Keywords**: Knowledge Distillation, KL Divergence, Tail Probabilities, Pre-training Distillation, Causal Language Models

## TL;DR
Ours proposes TAD (Tail-Aware Distillation): It explicitly decouples teacher top-$K$ probabilities from "tail" probabilities within the standard KD KL divergence and amplifies the tail contribution. This allows LLM pre-training distillation to be completed within academic-scale compute (single H100 + 1 week), outperforming data-centric methods like MiniPLM.

## Background & Motivation

**Background**: LLM distillation currently follows two primary paths: (1) Supervised Distillation (MiniLLM / OnPolicyKD), which assumes a pre-trained student but necessitates expensive online generation; (2) Pre-training Distillation (DistilBERT), training the student from scratch while relying on original teacher training corpora. Recently, MiniPLM adopted a data-centric route, where the teacher selects samples for the student to avoid online generation overhead.

**Limitations of Prior Work**: (1) On-policy methods require repeated student generation during training, resulting in PetaFLOPs 4–10x higher than vanilla KD, which is prohibitive for academic compute on billion-token scales. (2) Pre-training corpora for most causal LMs are closed-source, forcing distillation onto general corpora where "teacher argmax tokens" often mismatch "dataset ground-truth tokens" (measured mismatch rate of 39%–46%), rendering DKD logic designed for classification ineffective. (3) The gradient of standard KL divergence is dominated by teacher top-$K$ modes, causing student tail probabilities to collapse toward zero and losing diversity.

**Key Challenge**: In the KL divergence $\sum p^T \log(p^T/p^S)$, the multiplier $p^T_i$ for tail terms is nearly 0, contributing almost nothing to the loss. However, the student's distribution over the tail is crucial for generation quality and diversity—simply increasing the tail weight $\beta$ leads to training divergence.

**Goal**: Design a pre-training distillation loss that is (i) comparable in overhead to vanilla KD, (ii) executable on tens of billions of tokens within an academic budget, and (iii) explicitly utilizes teacher tail information.

**Key Insight**: Inspired by Decoupled KD (DKD) in image classification—which splits KL into "target vs. non-target" classes for separate weighting. Since DKD anchors on ground-truth labels (unsuitable for pre-training where next-tokens and modes often mismatch), the authors propose **rank-anchored** splitting: partitioning based on teacher top-$K$ rank instead of labels.

**Core Idea**: Decompose KL divergence into top-$K$ terms $\mathcal{D}_{KL_1}$ and tail terms $\alpha_K^T \mathcal{D}_{KL_2}$ based on teacher probability rankings. Multiply the tail term by a **sequence-normalized** coefficient $\beta(X)=\beta/\bar{\alpha}_K^T(X)$ to provide a persistent "push" to tail gradients while maintaining training stability.

## Method

### Overall Architecture
TAD maintains the overall framework of KD but replaces the standard KL divergence with a tail-friendly plug-in loss. The full training objective is $\mathcal{L}_{TAD}=\sum_t \mathcal{L}_{CLM}(t;\mathcal{P}^S)+\mathcal{L}_{DIV}(t;\mathcal{P}^T,\mathcal{P}^S)$: the first term $\mathcal{L}_{CLM}$ is the student's causal LM loss, and the second term $\mathcal{L}_{DIV}$ is the KL divergence split into top-$K$ and tail components with separate tail weighting. The pipeline is entirely offline—teacher logits are computed once and cached—so PetaFLOPs are on par with vanilla KD (1.2B student: 9.3 vs. 9.2; 0.5B: 6.5 vs. 6.4, whereas MiniLLM requires 39 / 21.8).

### Key Designs

**1. Top-K vs. Tail Probability Decoupling: An Independent "Volume Knob" for the Tail**

The pain point is direct: the vanilla KL gradient is $\partial \mathcal{L}/\partial z_i=p_i^S-p_i^T$. Since $p_i^T$ for tail tokens is nearly 0, these terms are overwhelmed by modes, leading the student to $\sum_k \accentset{*}{p}^S_k\approx 1$ and tail collapse. TAD splits the teacher distribution into two segments by rank: let $\{\accentset{*}{p}^T_k\}_{k=1}^K$ be the top $K$ probabilities, and the tail mass be $\alpha_K^T=1-\sum_k \accentset{*}{p}^T_k$. Thus, KL decomposes into $\mathcal{D}_{KL}(\mathcal{P}^T\|\mathcal{P}^S)=\mathcal{D}_{KL_1}+\alpha_K^T \mathcal{D}_{KL_2}$. Crucially, the tail term $\mathcal{D}_{KL_2}$ uses re-normalized tail probabilities $\tilde{p}=p/\alpha_K^T$. Even if raw tail probabilities are near zero, dividing by $\alpha_K^T$ creates a valid distribution $\tilde{p}$, allowing the tail to have a loss term that identifies its structure without being crushed by modes.

**2. β(X) Sequence-level Normalization: Amplifying the Tail without Divergence**

With an independent tail term, a naive constant multiplier $\beta>1$ causes training to diverge because the tail mass $\alpha_K^T$ varies significantly across tokens and teachers. TAD adopts a sequence-adaptive amplification $\beta(X)=\beta\,/\,\bar{\alpha}_K^T(X)$, where $\bar\alpha_K^T(X)=\frac{1}{N}\sum_{t=1}^N \alpha_K^T(t)$ is the average tail mass of the current sequence. The token-level loss is written as $\mathcal{L}_{DIV}(t)=D_{KL_1}(t)+\beta(X)\,\alpha_K^T(t)\,D_{KL_2}(t)$. Intuitively, this dynamically adjusts the amplification multiplier based on the current sequence's tail scale: thinner tails (smaller $\bar\alpha_K^T$) receive larger $\beta(X)$. This normalization allows stable convergence with mild nominal values like $\beta=1,2$.

**3. Tail Gradient Compensation Mechanism: Early-stage Support, Automatic Shutdown**

Analysis of the tail logit gradient $\partial \mathcal{L}_{DIV}/\partial z_i=(p_i^S-p_i^T)+(\beta(X)-1)\big(p_i^S\cdot\frac{1-\sum_k\accentset{*}p^T_k}{1-\sum_k\accentset{*}p^S_k}-p_i^T\big)$ reveals that the second term acts as compensation. When the student distribution is overly concentrated on modes ($\sum_k\accentset{*}p^S_k\ge \sum_k \accentset{*}p^T_k$), the compensation term is positive, "pushing" the student's tail probability up. This directly addresses the failure mode of "only learning modes." Once the student replicates the teacher's top-$K$ quality, the compensation term vanishes, and the loss reverts to vanilla KL, with the fixed point remaining $p_i^S=p_i^T$.

### Loss & Training
- Loss: $\mathcal{L}_{TAD}=\sum_t \mathcal{L}_{CLM}+\mathcal{L}_{DIV}$, with $K\in\{1,5,10,20\}$ and $\beta\in\{0.5,1,2,5,10\}$.
- Training Data: 20GB subset of Regmix (open-source Pile replication), approx. 5B tokens.
- Compute: Single H100, 1-week budget, processing approx. 2B tokens.
- Initialization: Teacher attention weights truncated to student hidden dimensions (following DistilBERT), MLP randomly initialized.

## Key Experimental Results

### Main Results

Qwen1.5-1.8B → {1.2B, 0.5B} students, pre-training distillation, average across 8 LMEH benchmarks:

| Student | Method | Avg Accuracy | Gain vs. Vanilla |
|------|------|------|------|
| 1.2B | CLM (no KD) | 45.0 | −0.7 |
| 1.2B | Vanilla KD | 45.6 | 0 |
| 1.2B | MiniPLM | 46.6 | +1.0 |
| 1.2B | **TAD (K=10)** | **47.8** | **+2.2** |
| 0.5B | Vanilla KD | 44.1 | 0 |
| 0.5B | MiniPLM | 45.0 | +1.0 |
| 0.5B | **TAD (K=10)** | **45.4** | **+1.5** |

PetaFLOPs (1M token subset): Vanilla 9.2 / MiniPLM 12.4 / **TAD 9.3** / MiniLLM 39 / Seq-KD 65. TAD is comparable to vanilla and significantly lower than on-policy methods.

Phi-2 2.8B → 1.1B: TAD (K=10) averaged 50.3, 1.2 points higher than vanilla KD, with F-ECE (calibration error) dropping from 1.45 to 1.37.

### Ablation Study

| Config (1.2B Student) | Avg | Description |
|------|------|------|
| Vanilla KD (equivalent to $\beta=1$) | 45.6 | Baseline |
| TAD K=1, β=2 | 47.2 | Minimal top/tail split |
| TAD K=10, β=2 | **47.8** | Optimal |
| TAD K=20, β=2 | 47.7 | Slight drop as K increases |
| TAD K=10, β=0.5 | 47.0 | Insufficient tail amplification |
| TAD K=10, β=10 | 47.6 | Over-amplification |

### Key Findings
- **K=10 is the sweet spot**: Too small a $K$ makes the tail too thick; too large a $K$ makes the tail too thin, causing TAD to revert to vanilla.
- **β=2 is most stable**: After normalization, values in $\beta\in[1,5]$ converge well, but $\beta=2$ is consistently optimal.
- **Significant compute advantage**: TAD outperforms MiniPLM at the same FLOPs, meaning 33% more tokens can be processed under the same budget.
- **Improved calibration**: F-ECE consistently decreases, indicating the student distribution shape better matches the teacher.

## Highlights & Insights
- Transitioning DKD from "label-anchored" to "rank-anchored" is a key step: finding a label-free splitting axis for pre-training scenarios without reliable labels.
- Sequence-level normalization $\beta(X)=\beta/\bar\alpha_K^T(X)$ is a simple yet effective stability trick transferable to other tasks requiring amplification of rare items.
- Gradient analysis shows TAD converges to the same fixed point as vanilla KD, essentially acting as an "early-stage accelerator" that shuts off automatically.
- The entirely offline design allows caching teacher logits, making distillation a pure data-loading problem, which is highly academic-friendly.

## Limitations & Future Work
- Primarily verified on pre-training distillation; robustness of $K$ and $\beta$ in SFT scenarios (GSM8K showed promise but lacks exhaustive comparison) remains unclear.
- Evaluation only went up to Gemma-2 9B teachers; scaling to 70B+ where tail distributions are sharper might require re-tuning $K$.
- Evaluation focuses on LMEH few-shot benchmarks; lacks open-ended generation quality (diversity, MAUVE) metrics, which are theoretically TAD's strengths.
- Caching teacher logits requires storing $|\mathcal{V}|$-dimensional probabilities; disk overhead for large vocabularies (>100k) is not discussed.

## Related Work & Insights
- **vs. Vanilla KD (Hinton)**: TAD is a strict super-set (reverts at $\beta=1$) and solves the "student mode concentration" problem through tail weighting.
- **vs. DKD (CVPR 2022)**: While DKD uses ground-truth labels for splitting, TAD uses teacher probability ranks, making it suitable for label-free pre-training.
- **vs. MiniPLM**: MiniPLM is data-centric; TAD is loss-centric. They are orthogonal and can be combined. TAD outperforms MiniPLM at lower FLOPs.
- **vs. MiniLLM / OnPolicyKD**: These rely on student online generation; TAD is fully offline with FLOPs an order of magnitude lower.

## Rating
- Novelty: ⭐⭐⭐⭐ Clever transposition of DKD to LM pre-training (rank-anchored decoupling + sequence normalization).
- Experimental Thoroughness: ⭐⭐⭐ Solid multi-teacher/parameter comparisons, though lacks generation diversity and exhaustive SFT evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear gradient derivations, concise loss formulation, and a logical narrative flow (FLOPs + tail loss → decoupling → normalization).
- Value: ⭐⭐⭐⭐ Provides a practical recipe for billion-token distillation within academic budgets, orthogonal to other data and architecture improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](../../ACL2026/model_compression/efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[NeurIPS 2025\] LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups](../../NeurIPS2025/model_compression/lt-soups_bridging_head_and_tail_classes_via_subsampled_model_soups.md)

</div>

<!-- RELATED:END -->
