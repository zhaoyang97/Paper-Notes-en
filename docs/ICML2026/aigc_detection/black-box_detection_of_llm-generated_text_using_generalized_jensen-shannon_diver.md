---
title: >-
  [Paper Note] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence
description: >-
  [ICML 2026][AIGC Detection][Black-box AI Text Detection] SurpMark reformulates "AI text detection" as likelihood-free hypothesis testing: it computes token surprisal using a proxy LM…
tags:
  - "ICML 2026"
  - "AIGC Detection"
  - "Black-box AI Text Detection"
  - "Surprisal Discretization"
  - "Markov State Transition"
  - "Generalized JS Divergence"
date: 2026-05-08
content_hash: b1a2d5c800274938
---

# Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence

**Conference**: ICML 2026  
**arXiv**: [2510.07500](https://arxiv.org/abs/2510.07500)  
**Code**: Not yet available  
**Area**: AIGC Detection / NLP / Hypothesis Testing  
**Keywords**: Black-box AI Text Detection, Surprisal Discretization, Markov State Transition, Generalized JS Divergence

## TL;DR
SurpMark reformulates "AI text detection" as likelihood-free hypothesis testing: it computes token surprisal using a proxy LM, discretizes it into $k$ states via k-means, estimates a first-order Markov transition matrix, and compares it with pre-built "human-written/machine-written" reference matrices using Generalized Jensen-Shannon Divergence (GJS). This approach provides black-box detection scores without retraining or per-instance resampling, requiring only a single forward pass.

## Background & Motivation
**Background**: AI text detection follows two main approaches: (1) **Classifier-based methods** (e.g., GPTZero, OpenAI Detector) require specialized models for each domain/generator, which are costly to annotate and fail in cross-domain scenarios; (2) **Statistical methods** are divided into global statistics (e.g., likelihood, log-rank, entropy), which are sensitive to calibration mismatch, length, and domain shift, and distributional statistics (e.g., DetectGPT, DNA-GPT, Fast-DetectGPT), which rely on perturbation/sampling/continuation for each test text, leading to computational costs that scale linearly with the number of calls.

**Limitations of Prior Work**: In black-box settings, mismatches between the scoring model (proxy LM) and the true generator cause systematic bias in likelihood-based metrics. Perturbation-based methods, dependent on per-input regeneration, are infeasible for high-throughput or resource-constrained scenarios. Neither approach achieves "training-free, single-pass inference, and cross-domain robustness."

**Key Challenge**: Absolute likelihood is unreliable in black-box settings, and per-instance resampling is too expensive. However, **human/machine texts exhibit fundamental differences in token dynamics**—LLMs tend to "recover" to highly predictable tokens immediately after a high-surprisal token (a side effect of perplexity minimization). This "recovery pattern" is stable and robust to calibration.

**Goal**: (1) Design a black-box detector that requires no classifier training, no per-instance resampling, and is transferable across domains and generators; (2) Statistically derive the optimal scaling of bin count $k$ and explain why GJS is the appropriate statistic.

**Key Insight**: Reformulate the task as **likelihood-free hypothesis testing with two references**—human-written and machine-written corpora can be pre-built offline. For each test text, only "summarization" and "distance comparison with two references" are needed, avoiding reliance on absolute likelihood.

**Core Idea**: Discretize continuous surprisal into $k$ interpretable states ("Predictable / Slightly Surprising / Significantly Surprising / Highly Surprising"), compress texts into first-order Markov transition matrices, and use $\Delta\text{GJS}_n = \text{GJS}(\hat M_P, \hat M_T, \alpha) - \text{GJS}(\hat M_Q, \hat M_T, \alpha)$ as the score, proving its equivalence to the normalized log-likelihood ratio under two hypotheses.

## Method

### Overall Architecture
**Offline Stage**: Compute surprisal using a proxy LM $F_\theta$ on a large human-written corpus, train a shared quantizer $q_k$ via k-means to map continuous surprisal to $\{1,\dots,k\}$. Then, compute surprisal → discretize → estimate transition frequencies for human-written and machine-written corpora, yielding reference matrices $\hat M_Q$ (human) and $\hat M_P$ (machine).

**Online Stage**: For a test text $\mathbf{t}$, compute surprisal using $F_\theta$, discretize using the same $q_k$, estimate the transition matrix $\hat M_T$, and compute $\Delta\text{GJS}_n$. Compare it with a threshold $\tau$ to classify.

This design requires no classifier training, treats the proxy LM as a black box (only querying token probabilities), and performs inference with a single forward pass.

### Key Designs

1. **Surprisal Discretization + First-Order Markov Summarization**:
    - **Function**: Compress each text into a "dynamic structure" summary, enabling detection decisions based on relative structure rather than absolute likelihood.
    - **Mechanism**: Compute token surprisal $s_t = -\log p_\theta(x_t \mid x_{1:t-1})$ for sequence $\mathbf{x}=(x_1,\dots,x_n)$; cluster into $k$ states (e.g., "Predictable / Slightly Surprising / Significantly Surprising / Highly Surprising" for $k=4$); convert the surprisal sequence into a discrete state sequence $\{a_t\}$; estimate the first-order transition matrix $\hat M(j\mid i) = \frac{\sum_t \mathbf{1}\{a_t=i, a_{t+1}=j\}}{\sum_t \mathbf{1}\{a_t=i\}}$.
    - **Design Motivation**: The "recovery phenomenon" in LLMs—immediate transitions from highly surprising tokens to predictable states—is a distinctive signature in transition matrices. Absolute likelihood is unstable under proxy mismatch, but transition matrices, as relative structures, are naturally robust to calibration drift. Experiments show higher-order Markov models degrade due to state space explosion ($k^{n+1}$ states) and data sparsity, making first-order the sweet spot.

2. **Two-Reference GJS Hypothesis Testing**:
    - **Function**: Reformulate detection as likelihood-free hypothesis testing, providing an interpretable LLR-equivalent statistic.
    - **Mechanism**: Generalized JS divergence is defined as $\text{GJS}(M_A, M_B, \alpha) = \frac{\alpha}{1+\alpha}D_{\text{KL}}(M_A, M_\alpha) + \frac{1}{1+\alpha}D_{\text{KL}}(M_B, M_\alpha)$, where $M_\alpha = \frac{\alpha}{1+\alpha}M_A + \frac{1}{1+\alpha}M_B$, and $\alpha$ is the reference/test length ratio. The detection score $\Delta\text{GJS}_n = \text{GJS}(\hat M_P, \hat M_T, \alpha) - \text{GJS}(\hat M_Q, \hat M_T, \alpha)$ is compared to a threshold $\tau$: $\Delta\text{GJS}_n \leq \tau$ indicates machine-written, otherwise human-written. Proposition 3.4 proves $\Delta\text{GJS}_n$ is strictly equivalent to the generalized log-likelihood ratio $\Lambda_{n,N}$, extending Gutman's universal test from single to dual references.
    - **Design Motivation**: Traditional LFHT compares only one reference, losing discriminative information from the alternative hypothesis. Two-reference GJS enables stronger discrimination and guarantees statistical optimality through its equivalence to LLR.

3. **Discretization–Estimation Tradeoff and Bin Count Scaling Law**:
    - **Function**: Provide theoretical guidance for choosing $k$.
    - **Mechanism**: Decompose error into (i) **Discretization Error** $|\mathcal{D}_f(\mathcal{S}_P,\mathcal{S}_Q)-\mathcal{D}_f(M_P,M_Q)|$, bounded by Proposition 3.1 as $\leq C/k$ (more bins improve precision); (ii) **Statistical Estimation Error** $|\mathcal{D}_f(\hat M_P,\hat M_Q)-\mathcal{D}_f(M_P,M_Q)|$, bounded by Theorem 3.2 as $\leq C(\log N \cdot \sqrt{k^3 \log(kN)/N} + k^3/N \cdot \log(1+N/k) + k/\sqrt{N})$ (more bins increase noise). Balancing $O(1/k)$ and the dominant term $O(k^{3/2}/\sqrt{N})$ yields $k^* = \Theta(N^{1/5})$ (ignoring polylog factors).
    - **Design Motivation**: Eliminates arbitrary bin selection, providing principled guidance for adaptive $k$ selection across datasets. Table 1 empirically validates $I(a_t; a_{t-2}\mid a_{t-1}) \approx 0.0076$ bits/token and shows second-order models offer negligible perplexity gains (+0.528%), confirming first-order sufficiency.

### Loss & Training
This method is **training-free**—reference matrices $\hat M_P, \hat M_Q$ are computed offline once; the k-means quantizer is trained on the human-written corpus once. The proxy LM is frozen and used solely as a surprisal scorer.

## Key Experimental Results

### Main Results
Detection AUROC across multiple datasets (SQuAD, XSum, WritingPrompts) and 9 generators (GPT2-XL, GPT-J-6B, GPT-Neo-2.7B, GPT-NeoX-20B, OPT-2.7B, Llama-2-13B, Llama-3-8B, Llama-3.2-3B, Gemma-7B):

| Method | GPT2-XL | GPT-J-6B | Llama-2-13B | Llama-3-8B | Gemma-7B | Avg |
|--------|---------|----------|-------------|------------|----------|-----|
| Likelihood | 85.0 | 74.8 | 94.4 | 93.9 | 65.8 | 77.97 |
| LogRank | 88.2 | 79.3 | 95.9 | 95.1 | 69.2 | 81.59 |
| DetectLRR | 91.1 | 85.8 | 96.4 | 94.9 | 75.5 | 86.79 |
| Lastde | 96.0 | 85.9 | 93.3 | 94.3 | 69.5 | 85.56 |
| Lastde++ | **99.5** | 91.5 | 95.5 | 95.9 | 76.9 | 90.04 |
| **SurpMark (Ours)** | Comparable or higher than Lastde++ | — | — | — | — | Robust performance |

SurpMark consistently matches or surpasses baselines across datasets, generators, and scenarios, with notable advantages in cross-domain generalization (reference corpus and test text from different domains).

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|---------------|----------------|-------------|
| Markov order = 1 | Highest AUROC | Sweet spot |
| Markov order = 2 | Slightly lower | State space $k^3$ expands, transition counts sparse |
| Markov order = 3+ | Significant drop | Estimation variance explodes |
| Bin count $k$ scan | AUROC concave w.r.t. $k$ | Validates $k^* = \Theta(N^{1/5})$ |
| Dual reference (PP+QQ) | Full SurpMark | LLR-equivalent |
| Single reference (PP or QQ only) | Significant drop | Loses two-sided discrimination |
| Shared quantizer ($q_k$) | Standard | Necessary |
| Independent quantizers per text | Drop | Cross-text incomparability |

Second-order conditional MI ($I(a_t; a_{t-2}\mid a_{t-1})$) experiments:

| Source | $\hat{I}=I(a_t; a_{t-2}\mid a_{t-1})$ (bits/token) | Rel. PP gain (2nd vs 1st) |
|--------|---------------------------------------------------|---------------------------|
| GPT-5-chat | 0.0076 | +0.528% |
| Human | 0.0045 | +0.314% |

### Key Findings
- First-order Markov information captures nearly all usable signals; higher orders only add sparse statistics with diminishing returns, consistent with theory and experiments.
- Bin count $k=4$ is near-optimal for typical data scales and corresponds to interpretable semantic states.
- Cross-proxy model transfer (e.g., using GPT-2 as a proxy to detect Llama text) maintains strong AUROC, validating the model-agnostic nature of surprisal transition structures.
- The "recovery pattern" (high-surprisal → low-surprisal transitions) is significantly more pronounced in LLM-generated text than in human-written text (visualized in Figure 2(a)), forming the core of SurpMark's discriminative power.

## Highlights & Insights
- **Mathematical formalization of detection as LFHT**—Directly applies Gutman (1989) to prove $\Delta\text{GJS}_n$ = LLR, providing a principled explanation for why GJS is the optimal statistic, rather than another ad-hoc heuristic.
- **Robustness to proxy LM mismatch**—Discretization + transition matrix summaries naturally mitigate absolute likelihood drift, a critical advantage for black-box deployment.
- **Discretization–Estimation tradeoff with $k^* = N^{1/5}$**—This elegant scaling law combines mathematical rigor with practical guidance for bin selection.
- **Offline reference construction + single-pass inference**—Compared to methods like DetectGPT that require 100 perturbations per text, inference costs are reduced by two orders of magnitude.

## Limitations & Future Work
- **Ceiling of first-order Markov assumption**—While experiments show negligible second-order MI, first-order models cannot capture paragraph- or document-level global structures (e.g., topic drift in machine writing).
- **Dependence on reference corpus representativeness**—Requires large pre-built "human-written" and "machine-written" corpora; new generation paradigms (e.g., RLHF-aligned Claude 3.7) may necessitate rebuilding references.
- **Sensitivity to short texts**—Theoretical $k^* = N^{1/5}$ degrades for small $N$ (<200 tokens); detection performance may drop for tweets or single sentences.
- **Inability to detect "hybrid texts"**—LLM outputs lightly edited by humans may have transition distributions between the two references, leading to misclassification near the threshold $\tau$.
- **Fixed $q_k$ quantizer**—The k-means quantizer cannot adapt online; proxy LM updates or significant domain shifts require retraining the quantizer and references.

## Related Work & Insights
- **vs DetectGPT / Fast-DetectGPT (Mitchell et al. 2023, Bao et al. 2024)**: These methods estimate likelihood curvature via perturbation-based regeneration, which is computationally expensive and perturbation-dependent. SurpMark achieves two orders of magnitude lower inference cost with offline reference construction and single-pass inference.
- **vs Lastde++ (Xu et al. 2025)**: Lastde++ also uses surprisal discretization and local diversity entropy but relies on a single global statistic. SurpMark elevates this to a dual-reference LFHT framework with theoretical optimality.
- **vs R-Detect (Song et al. 2025)**: R-Detect employs kernel-based relative tests with reference corpus-dependent kernel optimization. SurpMark uses lightweight k-means discretization with zero parameter training.
- **vs DNA-GPT (Yang et al. 2023)**: DNA-GPT compares n-gram divergence, which is sensitive to vocabulary drift. SurpMark operates in the surprisal state space, making it vocabulary-free.
- **Insights**: Reformulating ML tasks as classical statistical tests (e.g., hypothesis testing, change-point detection, goodness-of-fit) leverages statistical optimality results. The LFHT framework is broadly applicable to black-box scenarios where "likelihood is unreliable but summary statistics are trustworthy" (e.g., OOD detection, distribution shift detection, model attribution).

## Rating
- Novelty: ⭐⭐⭐⭐ Reformulating detection as two-reference LFHT + deriving $k^* = N^{1/5}$ introduces genuine theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 9 generators, multiple datasets, and scenarios; lacks more in-the-wild tests like multilingual settings.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations, tightly aligned experiments, and strong interpretability.
- Value: ⭐⭐⭐⭐ Zero-training, single-pass inference, and cross-domain robustness make this a deployable solution for AI text detection systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](../../ACL2025/aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ICML 2026\] Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](../../ACL2026/aigc_detection/beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](../../ACL2026/aigc_detection/who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](../../ACL2026/aigc_detection/temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)

</div>

<!-- RELATED:END -->
