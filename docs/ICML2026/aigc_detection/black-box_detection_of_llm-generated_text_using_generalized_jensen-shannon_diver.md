---
title: >-
  [Paper Note] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence
description: >-
  [ICML 2026][AIGC Detection][Paper Note] SurpMark reformulates "AI text detection" as a likelihood-free hypothesis testing problem: it uses a proxy LM to compute token surprisal, discretizes it into $k$ states via k-means, estimates a first-order Markov transition matrix, and compares it against pre-established "human-written vs. machine-generated" reference
tags:
  - ICML 2026
  - AIGC Detection
date: 2026-05-08
content_hash: 92297fca32006989
---
# Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence

**Conference**: ICML 2026  
**arXiv**: [2510.07500](https://arxiv.org/abs/2510.07500)  
**Code**: Not yet public  
**Area**: AIGC Detection / NLP / Hypothesis Testing  
**Keywords**: Black-box AI text detection, surprisal discretization, Markov state transitions, generalized JS divergence

## TL;DR
SurpMark reformulates "AI text detection" as a likelihood-free hypothesis testing problem: it uses a proxy LM to compute token surprisal, discretizes it into $k$ states via k-means, estimates a first-order Markov transition matrix, and compares it against pre-established "human-written vs. machine-generated" reference matrices using Generalized Jensen-Shannon (GJS) divergence. This provides a black-box, training-free, and per-instance resampling-free discrimination score in a single forward pass.

## Background & Motivation
**Background**: AI text detection mainly follows two paths: (1) **Classifier-based** (e.g., GPTZero, OpenAI Detector) which requires training specific models for each domain/generator, leading to high labeling costs and failure when shifting domains; (2) **Statistic-based**, divided into global statistics (likelihood, log-rank, entropy) which are heavily affected by calibration mismatch, length, or domain drift, and distributional statistics (DetectGPT, DNA-GPT, Fast-DetectGPT) which require perturbations/sampling/completions for each test text to reconstruct neighborhood distributions, resulting in linear computational explosion with the number of calls.

**Limitations of Prior Work**: In black-box scenarios, inconsistency between the scoring model (proxy LM) and the actual generator leads to systematic shifts in likelihood-based metrics. Perturbation-based methods cannot be deployed in high-throughput or resource-constrained scenarios due to their dependence on per-input regeneration. Neither path simultaneously achieves "training-free + single inference + cross-domain robustness."

**Key Challenge**: Absolute likelihood values are unreliable in black-box settings, and per-instance resampling is too expensive. However, **human/machine texts exhibit fundamental differences in token dynamics**: LLMs tend to "recover" to highly predictable tokens immediately after a high-surprisal token (a side effect of perplexity minimization). This "recovery pattern" is stable and calibration-robust.

**Goal**: (1) Design a black-box detector that requires no classifier training, no per-instance resampling, and is transferable across domains and generators; (2) Provide statistical optimal scaling for the number of bins $k$ and explain why GJS is the appropriate statistic.

**Key Insight**: Treat the task as a **two-reference likelihood-free hypothesis test**. Since public corpora exist for both human-written and machine-generated text, references can be built offline once. For each test text, only "summarization" and "distance comparison with two references" are needed, avoiding any dependence on absolute likelihood.

**Core Idea**: Discretize continuous surprisal into $k$ interpretable states ("Predictable / Slightly Surprising / Significantly Surprising / Highly Surprising"), compress the text into a first-order Markov transition matrix, and use $\Delta\text{GJS}_n = \text{GJS}(\hat M_P, \hat M_T, \alpha) - \text{GJS}(\hat M_Q, \hat M_T, \alpha)$ as the score. It is proven that this is equivalent to the normalized log-likelihood ratio under two hypotheses.

## Method

### Overall Architecture
**Offline Phase**: Use a proxy LM $F_\theta$ to compute surprisal on a large-scale human-written corpus, learn a shared quantizer $q_k$ via k-means to map continuous surprisal to $\{1,\dots,k\}$; then compute surprisal → discretize → count transition frequencies for both human and machine corpora to obtain two reference matrices $\hat M_Q$ (human) and $\hat M_P$ (machine).

**Online Phase**: The test text $\mathbf{t}$ similarly has its surprisal computed by $F_\theta$, discretized by the same $q_k$, and its transition matrix $\hat M_T$ summarized. Finally, $\Delta\text{GJS}_n$ is calculated and compared with a threshold $\tau$.

The entire design requires no classifier training, the proxy LM remains a complete black box (only token probabilities are required), and only one forward pass is needed at test time.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    subgraph OFF["Offline Reference Building"]
        direction TB
        QK["Proxy LM computes surprisal<br/>k-means learns shared quantizer q_k<br/>Bin scaling law: k*≈N^(1/5)"]
        QK --> MQ["Human corpus discretization<br/>First-order transition matrix M_Q"]
        QK --> MP["Machine corpus discretization<br/>First-order transition matrix M_P"]
    end
    T["Test text t"] --> ST["Surprisal Discretization + First-order Markov Summary<br/>Discretize with same q_k → Transition matrix M_T"]
    ST --> GJS["Two-reference GJS Hypothesis Test<br/>ΔGJS = GJS(M_P, M_T) − GJS(M_Q, M_T)"]
    MQ --> GJS
    MP --> GJS
    GJS -->|"ΔGJS ≤ τ"| MACH["Classify as Machine"]
    GJS -->|"ΔGJS > τ"| HUM["Classify as Human"]
```

### Key Designs

**1. Surprisal Discretization + First-order Markov Summary: Compressing text into comparable "dynamic structures"**

This addresses the pain point that absolute likelihood systematically drifts when the proxy LM and true generator are inconsistent. SurpMark does not use likelihood directly. Instead, it computes surprisal $s_t=-\log p_\theta(x_t \mid x_{1:t-1})$ for the sequence $\mathbf{x}=(x_1,\dots,x_n)$, clusters the continuous surprisal into $k$ interpretable states via k-means (e.g., $k=4$ corresponds to "Predictable / Slightly Surprising / Significantly Surprising / Highly Surprising"), converts the text into a discrete state sequence $\{a_t\}$, and calculates the first-order transition matrix $\hat M(j\mid i)=\frac{\sum_{t}\mathbf{1}\{a_t=i,\,a_{t+1}=j\}}{\sum_t \mathbf{1}\{a_t=i\}}$.

The transition matrix is chosen over likelihood because LLMs exhibit a significant "recovery phenomenon"—high surprisal tokens are immediately followed by low surprisal tokens. This pattern is a stable signature in the transition matrix, and "relative structures" are naturally robust to calibration drift. The order is fixed to one; higher orders expand the state space to $k^{n+1}$, leading to sparse counts and degradation. First-order is the sweet spot.

**2. Two-reference GJS Hypothesis Test: Turning detection into a likelihood ratio with optimality guarantees**

Traditional likelihood-free tests only compare against a single reference, losing discriminative information carried by the "other hypothesis." SurpMark uses Generalized Jensen-Shannon divergence for two-reference comparison: $\text{GJS}(M_A, M_B, \alpha) = \frac{\alpha}{1+\alpha}D_{\text{KL}}(M_A, M_\alpha) + \frac{1}{1+\alpha}D_{\text{KL}}(M_B, M_\alpha)$, where the mixed matrix $M_\alpha = \frac{\alpha}{1+\alpha}M_A + \frac{1}{1+\alpha}M_B$, and $\alpha$ is the reference-to-test length ratio. The detection score is the difference between individual GJS values: $\Delta\text{GJS}_n = \text{GJS}(\hat M_P, \hat M_T, \alpha) - \text{GJS}(\hat M_Q, \hat M_T, \alpha)$. If $\Delta\text{GJS}_n \leq \tau$, it is machine-generated.

This two-sided comparison is not only more discriminative but is backed by theory—Proposition 3.4 proves that $\Delta\text{GJS}_n$ is strictly equal to the generalized log-likelihood ratio $\Lambda_{n,N}$, effectively extending Gutman’s (1989) universal test. This provides a statistical optimality answer for using GJS rather than just an ad-hoc heuristic.

**3. Discretization–Estimation Tradeoff and Bin Scaling Law: Determining the optimal value for $k$**

SurpMark decomposes the total error into two terms: Discretization error $|\mathcal{D}_f(\mathcal{S}_P,\mathcal{S}_Q)-\mathcal{D}_f(M_P,M_Q)|$ decreases as bins increase (bounded by $\leq C/k$ in Proposition 3.1); statistical estimation error $|\mathcal{D}_f(\hat M_P,\hat M_Q)-\mathcal{D}_f(M_P,M_Q)|$ becomes noisier as bins increase (bounded by $O(k^{3/2}/\sqrt{N})$ in Theorem 3.2).

Balancing these results in an optimal bin number $k^* = \Theta(N^{1/5})$, turning the selection of $k$ from trial-and-error into a closed-form formula. Table 1 further confirms that first-order is sufficient: the second-order conditional mutual information $I(a_t; a_{t-2}\mid a_{t-1}) \approx 0.0076$ bit/token, with only a +0.528% perplexity gain for a second-order model.

## Key Experimental Results

### Main Results
Detection AUROC across multiple datasets (SQuAD, XSum, WritingPrompts) and 9 generators (GPT2-XL, GPT-J-6B, Llama-2-13B, Llama-3-8B, Gemma-7B, etc.):

| Method | GPT2-XL | GPT-J-6B | Llama-2-13B | Llama-3-8B | Gemma-7B | Avg |
|------|---------|----------|-------------|------------|----------|-----|
| Likelihood | 85.0 | 74.8 | 94.4 | 93.9 | 65.8 | 77.97 |
| LogRank | 88.2 | 79.3 | 95.9 | 95.1 | 69.2 | 81.59 |
| DetectLRR | 91.1 | 85.8 | 96.4 | 94.9 | 75.5 | 86.79 |
| Lastde | 96.0 | 85.9 | 93.3 | 94.3 | 69.5 | 85.56 |
| Lastde++ | **99.5** | 91.5 | 95.5 | 95.9 | 76.9 | 90.04 |
| **SurpMark (Ours)** | Comparable | — | — | — | — | Robust |

SurpMark **consistently matches or surpasses baselines**, especially in cross-domain generalization scenarios where reference corpora and test texts come from different domains.

### Ablation Study

| Configuration | Key Observation | Description |
|------|----------|------|
| Markov order = 1 | Highest AUROC | The sweet spot |
| Markov order = 2 | Slightly lower | State space $k^3$ expansion leads to sparse counts |
| Markov order = 3+ | Significant drop | Estimation variance explosion |
| Bin count $k$ sweep | Concave AUROC w.r.t $k$ | Validates $k^* = \Theta(N^{1/5})$ |
| Two-reference | Full SurpMark | LLR-equivalent |
| Single-reference | Significant drop | Loss of two-sided discriminative power |
| Consistent Quantizer | Standard | Essential for comparability |

### Key Findings
- First-order Markov information captures nearly all available signals; higher orders simply "use more parameters to learn sparser statistics."
- Bin count $k=4$ is near-optimal for common data sizes and corresponds to interpretable semantic states.
- Cross-proxy model transfer (e.g., using GPT-2 as a proxy to detect Llama text) maintains high AUROC, verifying the model-agnostic nature of surprisal transition structures.
- The "Recovery pattern" is significantly higher in LLM text than in human text, serving as the core source of SurpMark’s discriminative power.

## Highlights & Insights
- **Mathematical Formalization of Detection as LFHT**: Applying classical results from Gutman (1989) proves $\Delta\text{GJS}_n = \text{LLR}$, providing principled justification for GJS as the optimal statistic.
- **Robustness to Proxy LM Mismatch**: The "relative structure" summary of discretized transition matrices naturally smooths out absolute likelihood drift, a critical engineering advantage for black-box deployment.
- **Scaling Law $k^* = N^{1/5}$**: This elegant scaling law provides a principled guide for adaptive $k$ selection across datasets.
- **Efficiency**: One-time offline reference building + single online inference. Compared to methods like DetectGPT which require 100+ regenerations per text, inference cost is reduced by two orders of magnitude.

## Limitations & Future Work
- **Ceiling of First-order Markov Assumption**: While local MI is low, first-order Markov models cannot capture "paragraph-level" or "discourse-level" global structures (e.g., topic drift patterns in machine writing).
- **Dependence on Reference Corpus**: Requires pre-existing large-scale reference text; if an attacker uses a new generation paradigm (e.g., Claude 3.7 with new RLHF alignment), references may need rebuilding.
- **Sensitivity to Short Text**: Theoretical $k^* = N^{1/5}$ degrades for very small $N$ (<200 tokens); detection capability for tweets or single sentences may drop.
- **Hybrid Text Detection**: Human-edited LLM outputs will result in distributions between references, causing potential misclassifications near the threshold $\tau$.

## Related Work & Insights
- **vs DetectGPT / Fast-DetectGPT**: These rely on perturbation to estimate likelihood curvature, which is computationally expensive and model-dependent; SurpMark is two orders of magnitude faster.
- **vs Lastde++**: Lastde++ uses surprisal discretization but relies on single global statistics; SurpMark elevates this to a two-reference LFHT framework with theoretical optimality.
- **vs R-Detect**: R-Detect uses kernel-based relative tests but requires optimizing kernel parameters on reference corpora; SurpMark uses lightweight k-means.
- **Insight**: Reformulating ML tasks as classical statistical tests (hypothesis testing, goodness-of-fit) allows the inheritance of statistical optimality results.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizing detection as two-reference LFHT and providing $k^* = N^{1/5}$ are significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage of 9 generators and multiple datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations with strictly corresponding experimental results.
- Value: ⭐⭐⭐⭐ Training-free, single-inference, and cross-domain robust, making it highly practical for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](../../ACL2026/aigc_detection/mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](../../ACL2025/aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ICML 2026\] On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective](on_the_salience_of_low-probability_tokens_for_ai-generated_text_detection_a_mult.md)
- [\[ICML 2026\] Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)
- [\[ACL 2026\] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](../../ACL2026/aigc_detection/detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)

</div>

<!-- RELATED:END -->
