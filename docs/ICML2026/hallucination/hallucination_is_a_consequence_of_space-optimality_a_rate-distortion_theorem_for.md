---
title: >-
  [Paper Note] Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing
description: >-
  [ICML 2026][Hallucination Detection][Hallucination] This paper formalizes "LLMs memorizing random facts" as a **membership testing** problem with continuous confidence scores. It proves that in the sparse limit of facts, the optimal memory cost exactly equals the minimum KL divergence between fact and non-fact output distributions—a "rate-distortion theorem." It further concludes that under the log-loss objective and given limited memory, the optimal strategy is **neither abs…
tags:
  - "ICML 2026"
  - "Hallucination Detection"
  - "Hallucination"
  - "Rate-Distortion Theorem"
  - "Bloom filter"
  - "KL Divergence"
  - "Membership Testing"
  - "Memory Capacity"
date: 2026-05-08
content_hash: 115633df2884e1f0
---

# Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing

**Conference**: ICML 2026  
**arXiv**: [2602.00906](https://arxiv.org/abs/2602.00906)  
**Code**: Not yet released  
**Area**: Hallucination Detection  
**Keywords**: Hallucination, Rate-Distortion Theorem, Bloom filter, KL Divergence, Membership Testing, Memory Capacity  

## TL;DR
This paper formalizes "LLMs memorizing random facts" as a **membership testing** problem with continuous confidence scores. It proves that in the sparse limit of facts, the optimal memory cost exactly equals the minimum KL divergence between fact and non-fact output distributions—a "rate-distortion theorem." It further concludes that under the log-loss objective and given limited memory, the optimal strategy is **neither abstention nor forgetting**, but rather mapping a certain proportion of non-facts and facts to the same high-confidence point, identifying hallucination as the information-theoretically optimal error form.

## Background & Motivation

**Background**: Mainstream theoretical explanations for LLM hallucinations diverge into two branches: the "classification perspective" (Kalai et al., 2025), which views LLMs as binary classifiers of random facts where a no-free-lunch theorem forces false positives; and the "compression perspective," which views LLMs as lossy compressors of world knowledge, making distortion inevitable. The former explains "why errors occur" but not the form of those errors, while the latter mostly relies on informal arguments or the assumption of infinite facts.

**Limitations of Prior Work**: (i) "Compression leading to error" fails to explain why LLMs favor **hallucination** over **forgetting**—forgetting is clearly "safer." (ii) Even in closed-world settings (where all facts are finite and seen during training), models maintain high-confidence hallucinations while exhibiting over-refusal for legitimate queries; simple no-free-lunch theorems cannot account for this peculiar precision-recall behavior. (iii) Theoretical gaps exist between classical Bloom-filter space lower bounds (Carter et al., 1978; Pagh & Rodler, 2001) and LLM hallucination theories, lacking a unified framework.

**Key Challenge**: In the continuous confidence output space of an LLM, the fact set $\mathcal{K}$ and non-fact set $\mathcal{U}\setminus\mathcal{K}$ are asymmetric—the former is strictly finite, while the latter is vast and must be encoded using the same limited memory. Characterizing the "minimum bits required to distinguish $\mathcal{K}$ from $\mathcal{U}\setminus\mathcal{K}$" is the key to bridging these theories.

**Goal**: (i) Provide an abstract "membership tester" framework shared by LLMs and Bloom filters; (ii) Provide tight bounds for per-key minimum memory cost in the sparse limit $|\mathcal{K}|/|\mathcal{U}|\to 0$; (iii) Characterize the shape of the optimal non-fact output distribution under log-loss, consistent with maximum likelihood training objectives.

**Key Insight**: The authors view an LLM as a tuple of two algorithms, $\text{Init}+\text{Query}$. The Init algorithm takes key set $\mathcal{K}$ to produce a memory state $W$, and Query provides a confidence score $\hat x_i \in [0,1]$ for query $i\in\mathcal{U}$. This abstraction covers both Bloom filters ($\hat x\in\{0,1\}$) and LLM likelihood estimation. Rate-distortion theory is then used to bound $I(W;\mathcal{K})/n$ under error constraints $(\varepsilon_K,\varepsilon_N)$.

**Core Idea**: In the sparse limit, per-key minimum memory cost = $\min_{\mu_K,\mu_N}\mathrm{KL}(\mu_K\|\mu_N)$. Under log-loss, the unique optimal solution concentrates facts on a high-confidence point $x^*$, maps a fraction $q^*$ of non-facts to that same $x^*$, and maps the remaining $1-q^*$ to 0—rendering hallucination as space-optimal.

## Method

### Overall Architecture

This paper addresses how many bits a model with finite memory must spend to memorize facts and what form errors take when memory is insufficient. The authors first abstract LLMs and Bloom filters into a **membership tester** $\mathcal{M}=(\text{Init},\text{Query})$: Init consumes key set $\mathcal{K}$ and outputs memory $W$, and Query takes $(i,W)$ to produce a confidence score $\hat x_i\in[0,1]$. Memory cost is defined as $B(\mathcal{M})=I(W;\mathcal{K})$ (mutual information about the key set, bounded below by physical bits $H(W)$). Using loss functions $d^K, d^N:[0,1]\to[0,\infty]$ to measure errors for keys and non-keys constrained within $\varepsilon_K, \varepsilon_N$, the problem becomes minimizing $B(\mathcal{M})$ given an error budget. The derivation follows three steps: proving a non-asymptotic per-key lower bound, collapsing it into a clean $\min\mathrm{KL}$ in the sparse limit, and applying this to log-loss to reveal the form of hallucinations.

### Key Designs

**1. Main Rate-Distortion Theorem (Theorem 3.1): Transforming Memory Counting into Continuous Convex Optimization**

Classical Bloom-filter space bounds (Carter et al., 1978; Pagh & Rodler, 2001; Hurley & Waldvogel, 2007) rely on case-by-case combinatorial counting or mutual information calculations, often leaving loose constants. This paper introduces an auxiliary Bernoulli variable $X\sim\text{Bern}(p)$ ($p=n/u$) for any permutation-invariant tester, with $\hat X|X{=}1\sim\mu_K$ and $\hat X|X{=}0\sim\mu_N$. Thus, $I(X;\hat X)$ characterizes the "difficulty of distinguishing" $n$ keys from $u-n$ non-keys. Lemma 3.4 provides a non-asymptotic lower bound $B(\mathcal{M})/n\ge F_p(\mu_K, \mu_N)-\log(8n)/(2n)$ where $F_p=I(X;\hat X)/p$. Lemma 3.5 proves $F_p$ is lower semi-continuous and $\partial F_p/\partial p=-\mathrm{KL}(\mu_N\|p\mu_K+(1-p)\mu_N)/p^2$. In the sparse limit $p\to 0$, $F_p\to\mathrm{KL}(\mu_K\|\mu_N)$, and the lower bound collapses to:

$$B(\mathcal{M})/n \;\ge\; \min_{\mu_K\in\mathcal{C}_K,\ \mu_N\in\mathcal{C}_N}\ \mathrm{KL}(\mu_K\|\mu_N).$$

Reachability is established via a hash-based construction in Lemma 3.7, and Theorem 3.3 provides a finite-$p$ correction term $-\chi^2(\mu_K^*\|\mu_N^*)/(2\ln 2)\cdot p+o(p)$. This unifies all historical lower bounds and sharpens the additive constant $\Theta(1)$ left by Pagh-Rodler.

**2. Optimal "Hallucination Channel" Solution under Log-loss (Theorem 4.1): Hallucination as the Optimal Strategy**

While the main theorem provides an abstract $\min\mathrm{KL}$, the counter-intuitive conclusion appears when applying LLM-specific losses. For LLM probability estimation, $d^K(\hat x)=-\ln\hat x$ and $d^N(\hat x)=-\ln(1-\hat x)$ constitute log-loss (binary cross-entropy), strictly consistent with maximum likelihood training. In non-trivial cases where $\varepsilon_K,\varepsilon_N>0$ and $e^{-\varepsilon_K}+e^{-\varepsilon_N}>1$, the authors use variational methods with KKT conditions to find the unique optimal solution $\mu_K^*=\delta_{x^*}$ and $\mu_N^*=(1-q^*)\delta_0+q^*\delta_{x^*}$, where $x^*=e^{-\varepsilon_K}$ and $q^*=\varepsilon_N/[-\ln(1-x^*)]$. This implies all facts are concentrated on a high-confidence point $x^*$, while a proportion $q^*$ of non-facts are mapped to the same $x^*$. The per-key minimum memory $\mathrm{KL}(\mu_K^*\|\mu_N^*)=\log(1/q^*)$ means the hallucination probability $q^*=2^{-\mathrm{KL}}$ is entirely determined by memory capacity. This confirms hallucination is mathematically optimal under log-loss; forgetting or abstention is strictly suboptimal.

**3. Two-sided Filter Threshold Invariance: Explaining RAG and Long-tail Fine-tuning**

The second conclusion pertains to probability estimation, but many practical mitigations use thresholding. The authors prove that any downstream thresholding of $\hat x$ cannot break the $\mathrm{KL}(\mu_K\|\mu_N)$ lower bound. The corollary is that eliminating false positives (hallucinations) inevitably increases false negatives (forgetting/over-refusal). The only way to move the frontier is to change the memory budget: RAG (Lewis et al., 2020) incorporates non-parametric external memory, effectively increasing $B(\mathcal{M})$ and pushing the frontier outward. SFT on long-tail facts explicitly allocates more parameter capacity to random facts, corresponding to moving along the frontier to exchange bits for a smaller $q^*$.

### Experimental Settings

As a theoretical paper, there is no standard training target; experiments serve as a sanity check for Theorem 4.1. The authors monitor the shapes of $\mu_K, \mu_N$ under two settings: (1) **synthetic random strings**, training a small Transformer from scratch to memorize random facts; (2) **real-world ISBN + synthetic ID**, performing LoRA fine-tuning on a pre-trained LLM. Both monitor whether non-fact outputs follow the predicted $\delta_0$ and $\delta_{x^*}$ bimodal distribution.

## Key Experimental Results

### Main Results: Distribution Shapes Match Theory Predictions

| Setting | Observed Non-fact Output Distribution | Theorem 4.1 Prediction | Consistency |
|------|----------------------|------------------|--------|
| Small Transformer + synthetic random strings | Bimodal: $\delta_0$ + $\delta_{x^*}$ | $(1-q^*)\delta_0 + q^* \delta_{x^*}$ | ✓ |
| LoRA-tuned LLM + synthetic IDs | Significant high-confidence FP concentration | Same as above | ✓ |
| LoRA-tuned LLM + real ISBN | High-confidence hallucination clustering | Same as above | ✓ |

Key Observation: As the memory budget (e.g., LoRA rank or width) decreases, $q^*$ grows linearly with respect to $\log(1/q^*)$, strictly matching $\mathrm{KL}(\mu_K^*\|\mu_N^*) = \log(1/q^*)$.

### Comparison with Classical Lower Bounds

| Source | Lower Bound Form | Applicability |
|------|---------|------|
| Carter et al. (1978) | One-sided filter, sparse limit | Single-sided |
| Pagh & Rodler (2001) | Two-sided filter, $\Theta(1)$ constant remaining | Not tight |
| Hurley & Waldvogel (2007) | Fixed $u/n$, mutual-info style | Restricted |
| Li et al. (2023) | One-sided filter, non-zero $n/u$ | Single-sided |
| **Ours Theorem 3.1** | $\min\mathrm{KL}$, general sparse limit | **Tight bound for all cases** |
| **Ours Theorem 3.3** | $\mathrm{KL} - \chi^2 p/(2\ln 2) + o(p)$ | Finite-$p$ correction |

Regarding reachability: The authors construct a hash-based two-sided filter matching the bound within $o(n)$ bits, closing the gap left by Pagh-Rodler.

### Key Findings

- **Hallucination rate is determined solely by memory budget**: $q^* = 2^{-\mathrm{KL}(\mu_K^*\|\mu_N^*)}$; the trade-off between $\varepsilon_K$ and $\varepsilon_N$ does not change this. Raising precision for random facts requires more memory.
- **Abstention is suboptimal under log-loss**: Pushing probability toward $1/2$ (IDK) incurs a higher log-loss penalty than high-confidence hallucinations. This explains the limited success of "teaching LLMs to say IDK" through SFT.
- **RAG/External memory fundamentally shifts the frontier**: External non-parametric memory removes constraints on $B(\mathcal{M})$, shifting the hallucination lower bound downward.

## Highlights & Insights

- **Unifying Two Parallel Theories**: This is the first work to share a master theorem between 1970s-era Bloom filter space bounds and modern LLM hallucination theories.
- **Hallucination as Optimal Error State**: While human intuition suggests models should say "IDK" when uncertain, KKT conditions prove this is suboptimal under log-loss. Safer designs require changing the loss function or adding external memory.
- **Defending Long-tail Memorization**: Feldman's "long-tail memorization" hypothesis receives an information-theoretic upgrade—memorization isn't just accidental; it is a necessity for achieving target error under finite budgets.

## Limitations & Future Work

- **Closed-world Assumption**: The assumption that all facts are in $\mathcal{K}$ and seen during training differs from real-world open-world scenarios; generative hallucination also involves calibration, which this paper bypasses.
- **Permutation-invariance is a Tool, Not Reality**: Real LLMs are position-dependent; the lower bound still holds but tightness might vary.
- **Sparse Limit Quantification**: Measuring the size of the "plausible claim" universe $\mathcal{U}$ is difficult, making it hard to quantify the bound for specific LLMs.
- **Absence of a "Fix" Algorithm**: The theory characterizes the frontier but not the training algorithm to reach its points; current SFT pipelines move along the frontier rather than jumping beyond it.

## Related Work & Insights

- **vs Kalai et al. (2025)**: Kalai defines hallucination rate via induced classifier error under calibration; this work provides an independent explanation via closed-world information theory.
- **vs Feldman (2020)**: Elevates the "necessity of long-tail memorization" to an information-theoretic inevitability.
- **vs Classical Bounds**: Unifies and sharpens historical lower bounds, closing the Pagh-Rodler gap.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First unification of Bloom filters and LLMs under a $\min\mathrm{KL}$ framework with a closed-form solution for the "hallucination channel."
- **Experimental Thoroughness**: ⭐⭐⭐ Experiments act as theory sanity checks; sufficient for verification but limited in scale.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematically rigorous with clear Lemma-Theorem structures, though demanding for those without an information theory background.
- **Value**: ⭐⭐⭐⭐⭐ Provides first-principles explanation for LLM hallucinations; directly informs LLM safety, abstention design, and RAG evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models](../../ICML2025/hallucination/look_twice_before_you_answer_memory-space_visual_retracing_for_hallucination_mit.md)
- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)
- [\[ICML 2026\] From Out-of-Distribution Detection to Hallucination Detection: A Geometric View](from_out-of-distribution_detection_to_hallucination_detection_a_geometric_view.md)
- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[ICML 2026\] TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling](tag_tangential_amplifying_guidance_for_hallucination-resistant_sampling.md)

</div>

<!-- RELATED:END -->
