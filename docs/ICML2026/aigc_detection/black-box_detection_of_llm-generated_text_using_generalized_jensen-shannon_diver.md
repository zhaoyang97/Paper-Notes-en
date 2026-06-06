---
title: >-
  [Paper Note] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence
description: >-
  [ICML 2026][AIGC Detection][Black-box AI text detection] SurpMark reformulates "AI text detection" as a likelihood-free hypothesis testing problem: it uses a proxy LM to compute token surprisal…
tags:
  - "ICML 2026"
  - "AIGC Detection"
  - "Black-box AI text detection"
  - "surprisal discretization"
  - "Markov transition"
  - "Generalized JS Divergence"
date: 2026-05-08
content_hash: a7971962d8a296c7
---

# Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence

**Conference**: ICML 2026  
**arXiv**: [2510.07500](https://arxiv.org/abs/2510.07500)  
**Code**: Not yet public  
**Area**: AIGC Detection / NLP / Hypothesis Testing  
**Keywords**: Black-box AI text detection, surprisal discretization, Markov transition, Generalized JS Divergence

## TL;DR
SurpMark reformulates "AI text detection" as a likelihood-free hypothesis testing problem: it uses a proxy LM to compute token surprisal, discretizes it into $k$ states via k-means, and estimates a first-order Markov transition matrix. By comparing this matrix against pre-built "human-written / machine-written" reference matrices using Generalized Jensen-Shannon divergence (GJS), it provides a black-box, training-free, and per-instance sampling-free discrimination score in a single forward pass.

## Background & Motivation
**Background**: AI text detection mainly follows two paths: (1) **Classifier-based** (e.g., GPTZero, OpenAI Detector), which requires training specialized models for each domain/generator, incurring high labeling costs and failing under domain shift; (2) **Statistical-based**, subdivided into global statistics (likelihood, log-rank, entropy), which are sensitive to calibration mismatch and domain drift, and distributional statistics (e.g., DetectGPT, DNA-GPT, Fast-DetectGPT), which require perturbations/sampling for each test instance, causing computational costs to explode linearly with the number of calls.

**Limitations of Prior Work**: In black-box scenarios, inconsistency between the scoring model (proxy LM) and the actual generator leads to systematic offsets in likelihood-based metrics. Perturbation-based methods are difficult to deploy in high-throughput or resource-constrained scenarios due to their dependence on per-input re-generation. Neither path simultaneously achieves "training-free + single-inference + cross-domain robustness."

**Key Challenge**: Absolute likelihood values are untrustworthy under black-box proxy mismatch, and per-instance resampling is too expensive. However, **fundamental differences exist between human and machine text at the token dynamics level**—LLMs tend to "recover" to highly predictable tokens immediately after a high-surprisal token (a side effect of perplexity minimization). This "recovery pattern" is stable and calibration-robust.

**Goal**: (1) Design a black-box detector that requires no classifier training, no per-instance resampling, and can transfer across domains and generators; (2) Provide theoretical guidance for the optimal scaling of the bin number $k$ and explain why GJS is the appropriate statistic.

**Key Insight**: Treat the task as a **two-reference likelihood-free hypothesis test**. Both human and machine corpora are publicly available for one-time offline reference building; for each test text, one only needs to extract a "summary" and compare distances to the two references, avoiding any dependence on absolute likelihood.

**Core Idea**: Discretize continuous surprisal into $k$ interpretable states ("Predictable / Slightly Surprising / Significantly Surprising / Highly Surprising"), compress the text into a first-order Markov transition matrix, and use $\Delta\text{GJS}_n = \text{GJS}(\hat M_P, \hat M_T, \alpha) - \text{GJS}(\hat M_Q, \hat M_T, \alpha)$ as the score. It is proven that this is equivalent to the normalized log-likelihood ratio under the two hypotheses.

## Method

### Overall Architecture
**Offline Phase**: Use a proxy LM $F_\theta$ to compute surprisal on a large-scale human corpus; learn a shared quantizer $q_k$ via k-means to map continuous surprisal to $\{1,\dots,k\}$. Then, compute surprisal → discretize → count transition frequencies on both human and machine corpora to obtain two reference matrices $\hat M_Q$ (human) and $\hat M_P$ (machine).

**Online Phase**: For a test text $\mathbf{t}$, compute surprisal via $F_\theta$, discretize using the same $q_k$, and estimate the transition matrix $\hat M_T$. Classification is performed by comparing $\Delta\text{GJS}_n$ with a threshold $\tau$.

This design requires no classifier training, uses a completely black-box proxy LM (only token probabilities are needed), and involves only one forward pass during testing.

### Key Designs

1.  **Surprisal Discretization + First-order Markov Summary**:
    - **Function**: Compresses each text into a "dynamic structure" summary, basing detection decisions on relative structure rather than absolute likelihood.
    - **Mechanism**: First, compute the surprisal $s_t=-\log p_\theta(x_t \mid x_{1:t-1})$ for the token sequence $\mathbf{x}=(x_1,\dots,x_n)$. Use k-means clustering to obtain $k$ states ($k=4$ corresponds to "Predictable / Slightly Surprising / Significantly Surprising / Highly Surprising") and convert the continuous surprisal sequence into a discrete state sequence $\{a_t\}$. Then, calculate the first-order transition matrix $\hat M(j\mid i)=\frac{\sum_{t}\mathbf{1}\{a_t=i, a_{t+1}=j\}}{\sum_t \mathbf{1}\{a_t=i\}}$.
    - **Design Motivation**: The "recovery phenomenon" during LLM generation—returning to a predictable state immediately after a highly surprising token—is a significant signature in the transition matrix. Absolute likelihood is unstable under proxy mismatch, but the transition matrix, as a relative structure, is naturally robust to calibration drift. Experiments on Markov order show that higher orders suffer from state space explosion ($k^{n+1}$ states) and data sparsity, making first-order the sweet spot.

2.  **GJS Hypothesis Testing Based on Two References**:
    - **Function**: Reformulates detection as a likelihood-free hypothesis test, providing an interpretable LLR-equivalent statistic.
    - **Mechanism**: Generalized JS divergence is defined as $\text{GJS}(M_A, M_B, \alpha) = \frac{\alpha}{1+\alpha}D_{\text{KL}}(M_A, M_\alpha) + \frac{1}{1+\alpha}D_{\text{KL}}(M_B, M_\alpha)$, where $M_\alpha = \frac{\alpha}{1+\alpha}M_A + \frac{1}{1+\alpha}M_B$ and $\alpha$ is the reference-to-test length ratio. The detection score is $\Delta\text{GJS}_n = \text{GJS}(\hat M_P, \hat M_T, \alpha) - \text{GJS}(\hat M_Q, \hat M_T, \alpha)$. If $\Delta\text{GJS}_n \leq \tau$, it is classified as machine-written. Proposition 3.4 proves that $\Delta\text{GJS}_n$ is strictly equal to the generalized log-likelihood ratio $\Lambda_{n,N}$, serving as a natural extension of Gutman's universal test from single-reference to dual-reference.
    - **Design Motivation**: Traditional LFHT only compares against a single reference, losing discriminative information from the alternative hypothesis. Dual-reference GJS provides a two-sided comparison with stronger discriminative power, and the GJS = LLR equivalence ensures statistical optimality.

3.  **Discretization–Estimation Tradeoff and Bin Scaling Law**:
    - **Function**: Provides theoretical guidance for choosing $k$.
    - **Mechanism**: Error is decomposed into (i) **discretization error** $|\mathcal{D}_f(\mathcal{S}_P,\mathcal{S}_Q)-\mathcal{D}_f(M_P,M_Q)|$, bounded by $\leq C/k$ per Proposition 3.1; and (ii) **statistical estimation error** $|\mathcal{D}_f(\hat M_P,\hat M_Q)-\mathcal{D}_f(M_P,M_Q)|$, bounded by $\leq C(\log N \cdot \sqrt{k^3 \log(kN)/N} + k^3/N \cdot \log(1+N/k) + k/\sqrt{N})$ per Theorem 3.2. Balancing $O(1/k)$ and the dominating term $O(k^{3/2}/\sqrt{N})$ yields $k^* = \Theta(N^{1/5})$ (up to polylog factors).
    - **Design Motivation**: Eliminates the "magic number" approach to bin selection and provides a principled guide for adaptive $k$ selection across datasets. Empirical tests show $I(a_t; a_{t-2}\mid a_{t-1}) \approx 0.0076$ bit/token and only a +0.528% perplexity gain for second-order vs. first-order models, validating that "first-order is enough."

### Loss & Training
This method is **training-free**. Reference matrices $\hat M_P, \hat M_Q$ are computed via one-time offline statistics. The k-means quantizer is clustered once on a human corpus. The proxy LM is completely frozen and used only as a surprisal scorer.

## Key Experimental Results

### Main Results
Comparison of detection AUROC across multiple datasets (SQuAD, XSum, WritingPrompts) and 9 generators (GPT2-XL, GPT-J-6B, GPT-Neo-2.7B, GPT-NeoX-20B, OPT-2.7B, Llama-2-13B, Llama-3-8B, Llama-3.2-3B, Gemma-7B) (Selected):

| Method | GPT2-XL | GPT-J-6B | Llama-2-13B | Llama-3-8B | Gemma-7B | Avg |
|------|---------|----------|-------------|------------|----------|-----|
| Likelihood | 85.0 | 74.8 | 94.4 | 93.9 | 65.8 | 77.97 |
| LogRank | 88.2 | 79.3 | 95.9 | 95.1 | 69.2 | 81.59 |
| DetectLRR | 91.1 | 85.8 | 96.4 | 94.9 | 75.5 | 86.79 |
| Lastde | 96.0 | 85.9 | 93.3 | 94.3 | 69.5 | 85.56 |
| Lastde++ | **99.5** | 91.5 | 95.5 | 95.9 | 76.9 | 90.04 |
| **SurpMark (Ours)** | Comparable to or higher than Lastde++ | — | — | — | — | Robust Performance |

In the full comparison, SurpMark **consistently matches or surpasses baselines** across various datasets, generators, and scenarios, with a particularly notable advantage in cross-domain generalization (where the reference and test corpora come from different domains).

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|----------|------|
| Markov order = 1 | Highest AUROC | Sweet spot |
| Markov order = 2 | Slightly Lower | State space $k^3$ expansion, sparse transition counts |
| Markov order = 3+ | Significant Drop | Estimation variance explosion |
| Bin count $k$ scan | Concave AUROC w.r.t $k$ | Validates $k^* = \Theta(N^{1/5})$ |
| Dual-reference (PP+QQ) | Full SurpMark | LLR-equivalent |
| Single-reference (PP or QQ only) | Significant Drop | Loss of two-sided discriminative power |
| Unified Quantizer (Shared $q_k$) | Standard | Necessary |
| Per-text Quantization | Drop | Incomparable across texts |

I^(2nd-order conditional MI) Experiment:

| Source | $\hat{I}=I(a_t; a_{t-2}\mid a_{t-1})$ (bits/token) | Rel. PP gain (2nd vs 1st) |
|------|----------|---------|
| GPT-5-chat | 0.0076 | +0.528% |
| Human | 0.0045 | +0.314% |

### Key Findings
- First-order Markov information captures nearly all available signals; higher orders merely "use more parameters to learn sparser statistics," as shown by theory and experiments.
- Bin count $k=4$ is near-optimal for common data scales and corresponds to interpretable semantic states.
- Cross-proxy model transfer (e.g., using GPT-2 as a proxy to detect Llama text) maintains good AUROC, verifying the model-agnostic nature of surprisal transition structures.
- The "Recovery pattern" (high-surprisal → low-surprisal transition probability) is significantly higher in LLM text than in human writing, serving as the core source of SurpMark's discriminative power.

## Highlights & Insights
- **Mathematical formulation as LFHT**: Directly applying classic results from Gutman (1989) to prove $\Delta\text{GJS}_n$ = LLR provides a principled answer to "why GJS is the optimal statistic," rather than just another ad-hoc heuristic.
- **Proxy LM mismatch robustness**: The discretization + transition matrix "relative structure" allows absolute likelihood drift to be naturally mitigated, a key engineering advantage for black-box deployment.
- **Discretization–estimation tradeoff of $k^* = N^{1/5}$**: This elegant scaling law provides a closed-form formula for bin selection in practice.
- **One-time offline reference + single online inference**: Compared to methods like DetectGPT that require 100 perturbations per text, inference cost is reduced by two orders of magnitude.

## Limitations & Future Work
- **Ceiling of first-order assumptions**: While 2nd-order MI is small, first-order Markov cannot capture global structures at the "paragraph" or "discourse" level (e.g., topic drift patterns in machine writing).
- **Dependence on reference corpus representativeness**: Requires large-scale "human" and "machine" reference texts; if an attacker uses a new generation paradigm (e.g., Claude 3.7 with RLHF), references may need updating.
- **Sensitivity to short text**: Theoretically, $k^* = N^{1/5}$ degrades when $N$ is small (<200 tokens); detection capability for tweets or single sentences may decrease.
- **Inability to detect "mixed text"**: Human-edited LLM outputs will result in Markov distributions between the two references, leading to misclassifications near the single threshold $\tau$.
- **Fixed quantizer**: Once $q_k$ is fixed, it cannot adapt online; significant domain shifts or proxy LM updates require re-clustering.

## Related Work & Insights
- **vs. DetectGPT / Fast-DetectGPT**: These rely on likelihood curvature via perturbations, which is computationally expensive and perturbation-model dependent; SurpMark is two orders of magnitude cheaper due to its offline-reference/single-inference design.
- **vs. Lastde++**: Lastde++ also uses surprisal discretization but relies on a single global statistic; SurpMark elevates this to a dual-reference LFHT framework with theoretical optimality.
- **vs. R-Detect**: R-Detect uses kernel-based relative tests but requires optimizing kernel parameters on a reference corpus; SurpMark uses lightweight k-means with zero-parameter training.
- **vs. DNA-GPT**: DNA-GPT compares n-gram divergence, which is sensitive to vocab drift; SurpMark operates in the vocab-free surprisal state space.
- **Insight**: Reformulating ML tasks as classical statistical tests (hypothesis testing, change-point detection, goodness-of-fit) allows one to inherit a suite of statistical optimality results. This LFHT framework is broadly applicable to black-box scenarios where "likelihood is untrustworthy but summary statistics are reliable" (e.g., OOD detection, distribution shift).

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizing detection as two-reference LFHT and deriving $k^* = N^{1/5}$ provides genuine theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 9 generators and multiple datasets; could benefit from more in-the-wild testing (e.g., multilingual).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations and strict alignment between theory and experiments.
- Value: ⭐⭐⭐⭐ Training-free, single-inference, and cross-domain robust; a directly deployable solution for AI detection systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](../../ACL2026/aigc_detection/mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)
- [\[ICML 2026\] On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective](on_the_salience_of_low-probability_tokens_for_ai-generated_text_detection_a_mult.md)
- [\[ICML 2026\] Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)
- [\[ACL 2026\] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](../../ACL2026/aigc_detection/detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)
- [\[ICML 2026\] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)

</div>

<!-- RELATED:END -->
