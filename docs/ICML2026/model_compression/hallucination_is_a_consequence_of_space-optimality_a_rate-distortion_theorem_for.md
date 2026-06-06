---
title: >-
  [Paper Note] Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing
description: >-
  [ICML 2026][Model Compression][Hallucination] This paper formalizes "LLM memorization of random facts" as a **membership testing** problem with continuous confidence scores. It proves that in the sparse limit of facts…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Hallucination"
  - "Rate-Distortion Theorem"
  - "Bloom filter"
  - "KL divergence"
  - "Membership Testing"
  - "Memory Capacity"
date: 2026-05-08
content_hash: b59a9117309f1312
---

# Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing

**Conference**: ICML 2026  
**arXiv**: [2602.00906](https://arxiv.org/abs/2602.00906)  
**Code**: Not yet public  
**Area**: LLM Safety / Information Theory  
**Keywords**: Hallucination, Rate-Distortion Theorem, Bloom filter, KL divergence, Membership Testing, Memory Capacity  

## TL;DR
This paper formalizes "LLM memorization of random facts" as a **membership testing** problem with continuous confidence scores. It proves that in the sparse limit of facts, the optimal memory overhead equals the minimum KL divergence between the output distributions of facts and non-facts—establishing a "Rate-Distortion Theorem." It further demonstrates that under a log-loss objective with finite memory, the optimal strategy is **neither refusal nor forgetting**, but rather mapping a proportion of non-facts and facts to the same high-confidence point; thus, hallucination is an information-theoretically optimal error pattern.

## Background & Motivation

**Background**: Mainstream theoretical explanations for LLM hallucinations are divided into two branches: the "classification perspective" by Kalai et al. (2025), which views LLMs as binary classifiers of random facts where no-free-lunch theorems force false positives; and the "compression perspective," which views LLMs as lossy compressors of world knowledge, making distortion inevitable. The former explains "why" errors occur but not their "form," while the latter often relies on informal arguments or assumes an infinite number of facts.

**Limitations of Prior Work**: (i) The "compression → error" logic fails to explain why LLMs prefer **hallucination** over **forgetting**, given that forgetting is safer. (ii) Even in closed-world settings (where facts are finite and seen during training), models maintain high-confidence hallucinations while exhibiting over-refusal for legitimate queries; simple no-free-lunch theorems cannot explain this strange precision-recall behavior. (iii) A unified framework connecting classical Bloom filter space lower bounds (Carter et al. 1978; Pagh & Rodler 2001) with LLM hallucination theory is missing.

**Key Challenge**: In the continuous confidence output space of LLMs, the fact set $\mathcal{K}$ and the non-fact set $\mathcal{U}\setminus\mathcal{K}$ are asymmetric—the former is explicitly finite, while the latter is vast and must be encoded using the same limited memory. Characterizing the "minimum bits required to distinguish $\mathcal{K}$ from $\mathcal{U}\setminus\mathcal{K}$" is the key to unifying these theories.

**Goal**: (i) Provide an abstract "membership tester" framework common to both LLMs and Bloom filters. (ii) Derive tight bounds for per-key minimum memory overhead in the sparse limit $|\mathcal{K}|/|\mathcal{U}|\to 0$. (iii) Characterize the shape of the optimal non-fact output distribution under log-loss, consistent with maximum likelihood training.

**Key Insight**: The authors view an LLM as a tuple of two algorithms, $\text{Init}+\text{Query}$. Init takes a key set $\mathcal{K}$ and produces a memory state $W$, while Query provides a confidence score $\hat x_i \in [0,1]$ for query $i\in\mathcal{U}$. This abstraction covers both Bloom filters ($\hat x\in\{0,1\}$) and LLM likelihood estimation. Rate-distortion theory is then used to bound $I(W;\mathcal{K})/n$ under given $(\varepsilon_K,\varepsilon_N)$ error constraints.

**Core Idea**: In the sparse limit, the per-key minimum memory overhead = $\min_{\mu_K,\mu_N}\mathrm{KL}(\mu_K\|\mu_N)$. Under log-loss, the unique optimal solution dictates that facts concentrate at a high-confidence point $x^*$, while a portion $q^*$ of non-facts are crowded into that same $x^*$, with the remaining $1-q^*$ mapped to 0. Thus, hallucination equals spatial optimality.

## Method

### Overall Architecture

The authors define any algorithm that "distinguishes $\mathcal{K}$ from $\mathcal{U}\setminus\mathcal{K}$" as a permutation-invariant **membership tester** $\mathcal{M}=(\text{Init},\text{Query})$. Init maps $\mathcal{K}$ to memory $W$, and Query maps $(i,W)$ to $\hat x_i \in [0,1]$. **Memory cost** is defined as $B(\mathcal{M})=I(W;\mathcal{K})$, the mutual information in the memory state regarding the key set—which serves as a lower bound for physical bits $H(W)$.

Error is characterized by two functions $d^K, d^N : [0,1]\to[0,\infty]$. The loss for keys and non-keys at confidence $\hat x$ is $d^K(\hat x)$ and $d^N(\hat x)$, respectively. Error levels are constrained by $\varepsilon_K, \varepsilon_N$. For Bloom filters, $d^K(\hat x)=1-\hat x$ gives FNR and $d^N(\hat x)=\hat x$ gives FPR; for LLM probability estimation, $d^K(\hat x)=-\ln \hat x$ and $d^N(\hat x)=-\ln(1-\hat x)$ are the log-loss/binary cross-entropy. The advantage of this unified framework is that spatial lower bounds can be translated between these problems.

The theory is derived in three steps: (1) Establishing a **non-asymptotic per-key memory lower bound** $B(\mathcal{M})/n \ge F_p(\mu_K,\mu_N) - \log(8n)/(2n)$, where $F_p(\mu_K,\mu_N)=I(X;\hat X)/p$ ($X\sim\text{Bern}(p)$, $p=n/u$); (2) Proving $F_p \to \mathrm{KL}(\mu_K\|\mu_N)$ in the $p\to 0$ sparse limit with an achievability lemma (Lemma 3.7) to obtain the main theorem (Theorem 3.1); (3) Applying this bound to log-loss to derive the "hallucination channel" optimal solution (Theorem 4.1).

### Key Designs

1.  **Rate-Distortion Master Theorem (Theorem 3.1)**:
    - **Function**: Translates the discrete combinatorial problem of "how many bits distinguish $n$ keys from $u-n$ non-keys within an error" into a continuous convex optimization $\min_{\mu_K\in\mathcal{C}_K, \mu_N\in\mathcal{C}_N}\mathrm{KL}(\mu_K\|\mu_N)$.
    - **Mechanism**: For any permutation-invariant tester $\mathcal{M}$, auxiliary Bernoulli variables $X\sim\text{Bern}(p)$ ($p=n/u$) and conditional distributions $\hat X|X=1\sim \mu_K(\mathcal{M}), \hat X|X=0\sim \mu_N(\mathcal{M})$ are introduced. Mutual information $I(X;\hat X)$ represents the "difficulty of distinguishing" $n$ keys from $u-n$ non-key queries. Lemma 3.4 gives $B(\mathcal{M})/n \ge F_p(\mu_K,\mu_N) - \log(8n)/(2n)$; Lemma 3.5 shows lower semi-continuity of $F_p$ and $\partial F_p/\partial p = -\mathrm{KL}(\mu_N\|p\mu_K+(1-p)\mu_N)/p^2$, leading to $F_p \to \mathrm{KL}(\mu_K\|\mu_N)$ as $p\to 0$. Achievability is shown via hash-based construction in Lemma 3.7. Theorem 3.3 adds a finite-$p$ correction term $-\chi^2(\mu_K^*\|\mu_N^*)/(2\ln 2)\cdot p + o(p)$.
    - **Design Motivation**: Classical Bloom filter bounds (Carter et al. 1978, Pagh & Rodler 2001, Hurley & Waldvogel 2007) were calculated case-by-case. This theorem unifies them as $\min\mathrm{KL}$ and precisely fills the $\Theta(1)$ additive constant gap left by Pagh-Rodler.

2.  **Optimal "Hallucination Channel" Solution under log-loss (Theorem 4.1)**:
    - **Function**: Provides the closed-form unique solution for $\min_{\mu_K,\mu_N}\mathrm{KL}(\mu_K\|\mu_N)$ when facts/non-facts are evaluated using log-loss $-\ln\hat x, -\ln(1-\hat x)$.
    - **Mechanism**: In non-trivial cases where $\varepsilon_K, \varepsilon_N > 0$ and $e^{-\varepsilon_K} + e^{-\varepsilon_N} > 1$, the authors use variational methods with KKT verification to derive the unique optimal $\mu_K^* = \delta_{x^*}$ and $\mu_N^* = (1-q^*)\delta_0 + q^* \delta_{x^*}$, where $x^* = e^{-\varepsilon_K}$ and $q^* = \varepsilon_N / [-\ln(1-x^*)]$. The minimum memory per key is $\mathrm{KL}(\mu_K^*\|\mu_N^*) = \log(1/q^*)$. **The hallucination probability $q^*$ is determined entirely by memory capacity**, regardless of the trade-off between $\varepsilon_K/\varepsilon_N$.
    - **Design Motivation**: Log-loss is strictly consistent with maximum likelihood training. This implies that the conclusion "hallucinated outcomes are mathematically optimal under a fixed parameter budget" does not depend on specific training algorithms but is information-theoretically forced.

3.  **Two-sided Filter Threshold Invariance + Theoretical Explanation for RAG/Fine-tuning**:
    - **Function**: Generalizes the findings from "probability estimation" to "any threshold-based classification mechanism" and explains why RAG and long-tail fine-tuning mitigate hallucinations.
    - **Mechanism**: Any downstream process that thresholds confidence $\hat x$ (including the generative classifier in Kalai et al. 2025) cannot exceed the $\mathrm{KL}(\mu_K\|\mu_N)$ bound—a generalization of "two-sided filters" (allowing both FP and FN). Corollary: eliminating FP (hallucination) inevitably increases FN (forgetting/over-refusal); post-processing only slides along the frontier. RAG (Lewis et al. 2020) acts by connecting non-parametric external memory, effectively increasing $B(\mathcal{M})$ and pushing the entire frontier outward. SFT on long-tail facts explicitly allocates more parameter capacity, corresponding to "spending more bits to achieve a smaller $q^*$" on the same frontier.
    - **Design Motivation**: Explains three empirical phenomena: (a) why "abstention/IDK SFT" has limited success; (b) why the "memorization is necessary for long-tail" hypothesis by Feldman et al. holds information-theoretically; (c) why it is reasonable for "effective memory budget" to be smaller than the number of parameters under MDL/regularization views.

### Loss & Training

This is a theoretical paper and does not involve a training process. Experimentally, the authors empirically validate Theorem 4.1 using two settings: (1) **synthetic random strings**, where small Transformers are trained from scratch to learn random string sets; (2) **real-world ISBN + synthetic ID**, where pre-trained LLMs are LoRA fine-tuned to learn a set of random facts. Both settings monitor the distribution shapes of $\mu_K$ and $\mu_N$.

## Key Experimental Results

### Main Results: Distribution Shapes Matching Theoretical Predictions

| Setup | Observed Non-fact Output Distribution | Theorem 4.1 Prediction | Consistency |
|------|---------------------------------------|------------------------|-------------|
| Small Transformer + synthetic random strings | Bimodal: $\delta_0$ + $\delta_{x^*}$ | $(1-q^*)\delta_0 + q^* \delta_{x^*}$ | ✓ |
| LoRA-tuned LLM + synthetic IDs | Significant high-confidence "false positive" mass point | Same as above | ✓ |
| LoRA-tuned LLM + real ISBN | High-confidence hallucination clustering | Same as above | ✓ |

Key observation: As the memory budget (e.g., LoRA rank or network width) decreases, $q^*$ grows roughly linearly with $\log(1/q^*)$—matching $\mathrm{KL}(\mu_K^*\|\mu_N^*) = \log(1/q^*)$ strictly. This is the core validation of the theory.

### Comparison with Classical Space Bounds

| Source | Lower Bound Form | Applicability |
|------|---------|------|
| Carter et al. (1978) | One-sided filter, sparse limit | Single-sided |
| Pagh & Rodler (2001) | Two-sided filter, $\Theta(1)$ additive constant | Not tight |
| Hurley & Waldvogel (2007) | Fixed $u/n$, mutual-info style | Restricted |
| Li et al. (2023) | One-sided filter, non-zero $n/u$ | Single-sided |
| **Ours (Theorem 3.1)** | $\min\mathrm{KL}$, general sparse limit | **Tight bound for all cases** |
| **Ours (Theorem 3.3)** | $\mathrm{KL} - \chi^2 p/(2\ln 2) + o(p)$ | finite-$p$ correction |

Regarding achievability: The authors construct a hash-based two-sided filter that matches the lower bound within $o(n)$ bits of error, closing the gap left by Pagh-Rodler.

### Key Findings

- **Hallucination rate is determined solely by memory budget**: $q^* = 2^{-\mathrm{KL}(\mu_K^*\|\mu_N^*)}$, independent of the $\varepsilon_K$ vs $\varepsilon_N$ allocation. Improving precision on random facts requires more memory, not just adjusted loss weights.
- **Abstention is sub-optimal under log-loss**: Any behavior that pushes probability toward $1/2$ (e.g., "I don't know") incurs a higher log-loss penalty than directly hallucinating a high-confidence value. This explains why teaching models to say "IDK" via SFT has limited empirical efficacy.
- **RAG/External memory fundamentally changes the frontier**: When non-parametric memory is introduced, $B(\mathcal{M})$ is no longer restricted by parameter budget, and the hallucination lower bound shifts downward globally.

## Highlights & Insights

- **Unifies two long-standing parallel theoretical lines**: Space lower bounds for Bloom filters (since the 70s) and LLM hallucination theory (since 2024) share a master theorem for the first time. Applying engineering data structure bounds to LLM theory is a valuable perspective.
- **"Hallucination is the optimal error pattern" is a counter-intuitive claim**: Human intuition suggests models should say "IDK" when uncertain. This paper proves using KKT conditions that this is sub-optimal under log-loss. It places a hard constraint on LLM safety: to make IDK optimal, one must change the loss function itself (e.g., abstention-aware loss) or add external memory.
- **Transferable Trick**: The argument using permutation-invariance to reduce any asymmetric memory tester to a permutation-invariant sub-class (Remark 2.2) is a standardized technique reusable for almost any information-theoretic lower bound proof.
- **Defense for long-tail memorization**: The "memorization of long-tail is necessary" hypothesis proposed by Feldman et al. receives an information-theoretic explanation—it is not that the long-tail accidentally requires memorization, but that target error rates cannot be met without it under a limited budget.

## Limitations & Future Work

- **Closed-world assumption**: All facts are within $\mathcal{K}$ and seen during training, which differs significantly from a real LLM facing an open world (unseen facts); real generative hallucination involves calibration assumptions (as discussed in Kalai & Vempala 2024), which this paper explicitly skips.
- **Permutation-invariance is a tool, not reality**: Real LLMs are highly position-dependent. The authors acknowledge the lower bound still holds, but tightness may vary.
- **Sparse limit $|\mathcal{K}|/|\mathcal{U}|\to 0$**: In reality, the universe of "plausible claims" $\mathcal{U}$ is hard to quantify. Quantifying the bound for a specific LLM capacity requires additional assumptions.
- **No "repair" algorithm provided**: The theory characterizes the frontier but does not provide the training algorithm corresponding to specific points on that frontier. Current LLM SFT pipelines slide along the frontier rather than jumping past it.

## Related Work & Insights

- **vs. Kalai et al. (2025)**: Kalai et al. derive a generative hallucination rate lower bound equal to induced classifier error using calibration, but the calibration assumption forces an open-world setting. This paper provides an independent explanation using closed-world information theory; the two are complementary.
- **vs. Feldman (2020), Feldman & Zhang (2020)**: Feldman et al. proposed the necessity of long-tail memorization; this paper elevates that hypothesis to an information-theoretic necessity.
- **vs. Classical Bloom-filter Bounds (Carter et al. 1978, Pagh & Rodler 2001)**: This paper restores and refines all these bounds, closing Pagh-Rodler's $\Theta(1)$ additive gap and generalizing Hurley & Waldvogel's mutual-info bound to the $p\to 0$ limit.
- **vs. Compression-based Hallucination Theories (Mohsin et al. 2025; Shi et al. 2025; Kim 2025)**: Previous "compression → error" theories were informal or assumed infinite facts. This paper provides a rigorous rate-distortion theorem in a finite closed-world setting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify Bloom-filter space bounds and LLM hallucination theory into a $\min\mathrm{KL}$ framework with a closed-form solution for the "hallucination channel."
- Experimental Thoroughness: ⭐⭐⭐ Experiments serve as a sanity check for the theory, covering synthetic and real ISBN settings, sufficient for validation but limited in scale.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear Lemma-Theorem structure, though requiring significant information theory background.
- Value: ⭐⭐⭐⭐⭐ Provides a first-principles explanation for why LLMs inevitably hallucinate, offering direct guidance for LLM safety, abstention design, and RAG evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](../../CVPR2026/model_compression/rdvq_differentiable_vq_image_compression.md)
- [\[ICML 2026\] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)
- [\[ICML 2026\] Exploiting Weight-Space Symmetries for Approximating Curvature](exploiting_weight-space_symmetries_for_approximating_curvature.md)
- [\[ICML 2026\] Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space](event2vec_processing_neuromorphic_events_directly_by_representations_in_vector_s.md)

</div>

<!-- RELATED:END -->
