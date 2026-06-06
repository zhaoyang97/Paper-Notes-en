---
title: >-
  [Paper Note] From Out-of-Distribution Detection to Hallucination Detection: A Geometric View
description: >-
  [ICML 2026][AI Safety][Hallucination Detection] This paper treats LLM next-token prediction as a classification task over a massive vocabulary. It migrates two lightweight OOD detectors…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Hallucination Detection"
  - "OOD Detection"
  - "Geometric Uncertainty"
  - "Decision Boundary"
  - "Single-sample Training-free"
date: 2026-05-08
content_hash: 549582e0ee0e5e10
---

# From Out-of-Distribution Detection to Hallucination Detection: A Geometric View

**Conference**: ICML 2026  
**arXiv**: [2602.07253](https://arxiv.org/abs/2602.07253)  
**Code**: TBD  
**Area**: AI Safety / Hallucination Detection / OOD Detection  
**Keywords**: Hallucination Detection, OOD Detection, Geometric Uncertainty, Decision Boundary, Single-sample Training-free

## TL;DR
This paper treats LLM next-token prediction as a classification task over a massive vocabulary. It migrates two lightweight OOD detectors, NCI (proximity between features and weight vectors) and fDBD (distance from features to decision boundaries), and introduces two adaptations: an "analytical proxy $\mu_G$ for training feature means" and "calculating boundary distances only on top-$k$ candidate tokens." This results in a **training-free, single-sample** hallucination detector for reasoning tasks that consistently outperforms baselines like Perplexity, Semantic Entropy, and SelfCheckGPT on CSQA, GSM8K, and AQuA.

## Background & Motivation

**Background**: Current LLM hallucination detection follows two main paths: one trains classifiers to identify hallucinations (e.g., SAPLMA, INSIDE), which are sensitive to distribution shifts and costly to train; the other consists of training-free methods (Semantic Entropy, SelfCheckGPT, Lexical Similarity, etc.) that score by comparing consistency across **multiple sampled outputs**, avoiding training but incurring high inference costs.

**Limitations of Prior Work**: Training-free multi-sample methods perform well on short QA but fail on **reasoning tasks**. Multi-step reasoning naturally allows for multiple valid paths, making the concept of "semantic consistency across multiple outputs" difficult to determine. Furthermore, sampling $N$ complete reasoning chains for every question leads to explosive computational overhead.

**Key Challenge**: Reasoning tasks simultaneously require being **training-free** (to avoid classifier drift), **single-sample** (to avoid multiple samplings), and **efficient per-token** (calculated at every step). Existing methods cannot satisfy these three requirements concurrently.

**Goal**: Construct a training-free, single-sample hallucination detector for reasoning with controllable per-token overhead.

**Key Insight**: The authors note that OOD detection and hallucination detection both essentially measure "how uncertain the model is about its current prediction." If the LLM language head is viewed as a linear classifier with $|\mathcal{V}|$ classes and penultimate-layer features as inputs, the mature "geometric relationship between features and weight vectors" from OOD literature can be directly migrated—and such geometric measures are inherently per-token and single-sample.

**Core Idea**: Adapt the OOD detectors NCI (features close to the weight vector of their predicted class indicate low uncertainty) and fDBD (features far from decision boundaries indicate low uncertainty) to LLMs. Minimal analytical/engineering fixes are applied to handle three LLM characteristics (unavailability of training statistics, massive vocabulary, and stochastic decoding). Scores are calculated per token and averaged over the sequence to serve as the hallucination score.

## Method

### Overall Architecture
During inference, given a prompt $\bm{x}$ and the generated sequence $\bm{y}_{<t}$, the model produces penultimate-layer features $\bm{z}^t \in \mathbb{R}^{d_{\text{model}}}$. The language head linearly maps this to $|\mathcal{V}|$-dimensional logits. At **each decoding step $t$**, a scalar uncertainty score $s(\bm{z}^t)$ is calculated using NCI or fDBD formulas. After the full output is generated, the **sequence mean** $S = \frac{1}{T}\sum_{t=1}^{T} s(\bm{z}^t)$ is taken as the hallucination score for the response. Finally, a threshold $\tau$ is used for binary classification (low score $\rightarrow$ hallucination). This process does not modify model weights, requires no training data, and only needs a single sample per response.

### Key Designs

1.  **Geometric Uncertainty Scores: LLM-fied Definitions of NCI and fDBD**:
    - **Function**: Extracts uncertainty of the current token prediction from the penultimate-layer features themselves as a hallucination signal.
    - **Mechanism**: The language head is treated as a linear classifier where the predicted token is $\hat{c}=\arg\max_v \bm{w}_v^\top \bm{z}+b_v$. **NCI** measures the proximity of the feature to the predicted class weight vector, defined as $s_{\text{NCI}}(\bm{z})=\cos(\bm{w}_{\hat{c}}, \bm{z}-\bm{\mu}_G)\,\|\bm{w}_{\hat{c}}\|_2$; higher values indicate closer proximity and lower uncertainty. **fDBD** measures the distance from the feature to the decision boundaries of other tokens. The authors use a first-order approximation $\tilde{D}_f(\bm{z},c)=|(\bm{w}_{\hat{c}}-\bm{w}_c)^\top \bm{z}+(b_{\hat{c}}-b_c)|/\|\bm{w}_{\hat{c}}-\bm{w}_c\|_2$; larger distances indicate being further from boundaries and lower uncertainty. Both have a single-step complexity of $O(d_{\text{model}})$ and can be added after a normal forward pass with almost zero extra overhead.
    - **Design Motivation**: These two geometric measures have been validated in OOD literature as robust uncertainty signals within the "training feature mean-weight-boundary" triad. They are per-sample and per-step, perfectly meeting the hard requirements for hallucination detection in reasoning. Empirical validation on CSQA (Fig. 2) shows that features of hallucinated responses are indeed closer to boundaries and further from weight vectors, confirming the migration is valid.

2.  **Analytical Proxy for Training Feature Mean $\mu_G$ (Decision-Neutral Closest Point)**:
    - **Function**: The NCI formula requires the global mean of training features $\bm{\mu}_G$, but LLM training corpora are private and massive, making estimation impossible. A "data-free" substitute is required.
    - **Mechanism**: The authors prove (Lemma 4.1) that the "feature point that minimizes logit variance across the vocabulary" $\bm{z}_\star$ is an analytically solvable point of maximum uncertainty: $\hat{\bm{z}}_\star = -(W^\top P W)^\dagger W^\top P \bm{b}$, where $P=I-\frac{1}{|\mathcal{V}|}\mathbf{1}\mathbf{1}^\top$. For zero-bias language heads (as in Llama-3.2-3B), this point collapses to the origin $\bm{0}$ in feature space. By substituting this for $\bm{\mu}_G$, NCI becomes entirely independent of training data.
    - **Design Motivation**: Using an analytical point replaces empirical means, avoiding issues where sampling estimates are biased on diverse corpora. Table 1 shows that on CSQA + Llama-3.2-3B, NCI with the analytical proxy achieves AUROC=66.07, whereas using empirical means estimated from the CSQA training set only achieves 62.79 (worse than Perplexity's 63.23), confirming the necessity of the analytical proxy.

3.  **Top-$k$ Candidate Set Pruning for fDBD**:
    - **Function**: Naive fDBD calculates boundary distances for all $|\mathcal{V}|-1$ tokens and averages them, which introduces noise on large vocabularies (boundaries for rare tokens, punctuation, or numbers are almost always far away, diluting signals from true semantic competitors) and consumes $O(d_{\text{model}}|\mathcal{V}|)$ compute.
    - **Mechanism**: At each step, only the $k$ tokens with the highest logits form the set $\mathcal{K}_t$ (excluding the top-1, as it is the predicted token itself with zero distance). The normalized average boundary distance is calculated as $s_{\text{fDBD}}^k=\frac{1}{k}\sum_{c\in\mathcal{K}_t}\tilde{D}_f(\bm{z}^t,c)/\|\bm{z}^t-\bm{\mu}_G\|_2$. Utilizing Quickselect, the complexity per step is reduced from $O(d_{\text{model}}|\mathcal{V}|)$ to an expected $O(d_{\text{model}}k+|\mathcal{V}|)$. $k$ is selected on a validation set.
    - **Design Motivation**: By highlighting candidate tokens that could "truly be semantically substituted" and filtering out the vast number of distant, context-irrelevant tokens, both performance and efficiency are improved. Table 2 shows that for $k$ ranging from 1 to 100,000 to "All", all values of $k$ outperform Perplexity, peaking at $k=1000$ (AUROC 69.24 vs. 68.15 for "All").

### Loss & Training
**Completely training-free** with no parameter updates. The Perplexity baseline is $\text{PPL}(\bm{y}|\bm{x})=\exp(-\frac{1}{T}\sum_t \log p(\bm{y}_t|\bm{x},\bm{y}_{<t}))$. Ours uses the same "step-wise scoring + sequence averaging" pattern but replaces the score with $s_{\text{NCI}}$ or $s_{\text{fDBD}}^k$. Evaluation metric is AUROC, which is threshold-free.

## Key Experimental Results

### Main Results
Setup: CSQA (commonsense, MCQ, 1221 items), GSM8K (math, free-form, 1319 items), AQuA (math, MCQ, 254 items). Models: Llama-3.2-3B-Instruct and Qwen-2.5-7B-Instruct, CoT prompting, greedy decoding.

| Model / Method | Single Sample | CSQA | GSM8K | AQuA |
|---|---|---|---|---|
| Llama-3.2-3B / Perplexity | ✓ | 63.23 | 69.63 | 72.85 |
| Llama-3.2-3B / SelfCheckGPT NLI | ✗ | 64.18 | 74.29 | 66.01 |
| Llama-3.2-3B / Semantic Entropy | ✗ | 60.61 | 64.40 | 64.71 |
| Llama-3.2-3B / **NCI** | ✓ | 66.07 | 76.32 | 74.41 |
| Llama-3.2-3B / **fDBD (selected $k$)** | ✓ | **69.24** | **76.36** | **76.20** |
| Qwen-2.5-7B / Perplexity | ✓ | 61.94 | 71.54 | 71.66 |
| Qwen-2.5-7B / SelfCheckGPT NLI | ✗ | 60.18 | 76.22 | 70.90 |
| Qwen-2.5-7B / **NCI** | ✓ | 71.60 | 75.83 | 78.19 |
| Qwen-2.5-7B / **fDBD (selected $k$)** | ✓ | **72.47** | **77.19** | **78.22** |

Latency (Llama-3.2-3B, CSQA, ms/token): Standard 31.94, Perplexity 32.88, NCI 32.54, fDBD 32.71 (almost zero overhead).

### Ablation Study

| Configuration | CSQA AUROC | Description |
|------|-----------|------|
| Perplexity Baseline | 63.23 | LLM built-in confidence |
| NCI w/ Empirical Mean $\bm{\mu}_G$ (CSQA train set) | 62.79 | Empirical estimation causes drop |
| NCI w/ **Analytical Proxy $\bm{z}_\star$** | **66.07** | Analytical proxy wins by +3.3 AUROC |
| fDBD $k=1$ | 68.64 | Looks only at top-1 alternative |
| fDBD $k=100$ | 69.18 | |
| fDBD $k=1000$ | **69.24** | Peak performance |
| fDBD $k=10000$ | 68.87 | Signal begins to dilute |
| fDBD $k=$ All ($\approx 10^5$) | 68.15 | Whole vocabulary is the worst |

Robustness to Stochastic Decoding (CSQA, Llama-3.2-3B, mean over 5 seeds): Under temp=0.2/0.5/0.8/1.0, Perplexity fluctuates around 62-63; NCI is stable at 66-68; fDBD is stable at 68-69. Both consistently outperform Perplexity, proving that while NCI/fDBD are defined on the "highest logit token," **sequence averaging** allows occasional misalignments in stochastic decoding to be smoothed out by subsequent steps.

### Key Findings
- The analytical proxy $\bm{z}_\star$ is the **key** to migrating OOD methods to LLMs—empirical means are not only useless but perform worse than Perplexity; this indicates LLM training features cannot be estimated from small downstream datasets.
- The top-$k$ pruning curve is inverted U-shaped; too small ($k=1$) lacks information, while too large (All) is diluted by irrelevant tokens. The peak at $k\sim 10^3$ suggests "true semantic competitors" are concentrated in the top thousand tokens.
- Single-sample geometric methods are particularly effective in **mathematical reasoning** (GSM8K/AQuA) compared to multi-sample methods like Semantic Entropy or SelfCheckGPT—the latter requires sampling $N$ full CoTs, while NCI/fDBD needs only one inference with no significant latency increase (<1 ms/token).

## Highlights & Insights
- **Paradigm Restatement**: Connecting "Hallucination Detection" and "OOD Detection" is conceptually natural—classifier prediction on unseen classes is essentially "classification hallucination." However, the author's real contribution is **realizing** this: every LLM characteristic (unseen training data, massive vocabulary, stochastic decoding) is addressed with a specific engineering/analytical fix rather than remaining a mere analogy.
- **Decision-Neutral Closest Point is a Reusable Tool**: Any OOD/uncertainty method relying on "training feature means" would get stuck when migrating to LLMs. The path provided—"logit variance minimization $\rightarrow$ closed-form solution $\rightarrow$ collapse to origin for zero-bias"—can be directly applied to LLM-formatting other OOD scores like Mahalanobis or Energy.
- **Per-token Geometric Scores + Sequence Averaging is a Simple but Effective Bridge**: Extending "single-point uncertainty" from the classification paradigm to sequences via arithmetic means is surprisingly stable under stochastic decoding, suggesting that for reasoning tasks, **accumulated geometric signals** are more important than "strict step-wise alignment."
- **Near-Zero Latency**: 32.71 vs 31.94 ms/token means this detector can be embedded by default in production inference pipelines, unlike SelfCheckGPT which requires $N \times$ the inference budget.

## Limitations & Future Work
- The analytical proxy $\bm{z}_\star$ is the origin for zero-bias heads (Llama), but whether it remains optimal for models with non-zero biases (Qwen series/MoE) was only verified indirectly via AUROC without analyzing the relationship between bias magnitude and proxy deviation.
- Simple aggregation by sequence averaging might mask signals of "local high uncertainty steps" (e.g., only one or two critical steps are wrong in a long CoT). Future work could consider **max / top-percentile / weighted aggregation**.
- All evaluations were on reasoning/QA; whether token-level geometric uncertainty serves as a hallucination signal in open-ended long-form generation (summarization, creative writing) has not been verified.
- $k$ needs to be selected on a validation set, creating a cold-start cost for new tasks/models without labels; adaptive $k$ (determined dynamically by logit distribution entropy) is worth exploring.

## Related Work & Insights
- **vs Semantic Entropy (Kuhn et al., 2023)**: SE requires sampling multiple responses to calculate semantic entropy (multi-sample, unsuitable for long CoT); Ours is single-sample, with geometric signals derived directly from penultimate features.
- **vs SelfCheckGPT (Manakul et al., 2023)**: SelfCheck also requires multiple samplings for consistency checks; Ours requires only one inference with near-zero latency.
- **vs Perplexity / Max P / P(True)**: Also single-sample training-free, but Perplexity/Max P use only scalar summaries of logits; Ours further utilizes the **geometric position of penultimate feature vectors**, providing richer information and higher AUROC across all datasets/models.
- **vs INSIDE / SAPLMA and other trained classifiers**: Those methods learn boundaries sensitive to distribution shifts and require labeled data; Ours is training-free and zero-label.

## Rating
- Novelty: ⭐⭐⭐⭐ While the OOD $\leftrightarrow$ Hallucination concept has been mentioned before, this is the first to migrate "feature geometry" OOD detectors like NCI/fDBD to LLMs and solve three specific challenges.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets $\times$ 2 models + appendix extending to Qwen3-32B / base models / MoE / other architectures, plus 5 seeds for stochastic decoding.
- Writing Quality: ⭐⭐⭐⭐ The structure of "three challenges $\rightarrow$ three solutions" is very clear, with formal presentation of definitions, theorems, and approximations.
- Value: ⭐⭐⭐⭐ Near-zero latency + training-free + single-sample allows for direct industrial integration; provides a reusable methodology for "migrating OOD tools to LLMs."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL](omnivl-guard_towards_unified_vision-language_forgery_detection_and_grounding_via.md)
- [\[CVPR 2026\] LogitDynamics: Reliable ViT Error Detection from Layerwise Logit Trajectories](../../CVPR2026/ai_safety/logitdynamics_vit_error_detection.md)
- [\[ICLR 2026\] Watermark-based Detection and Attribution of AI-Generated Content](../../ICLR2026/ai_safety/watermark-based_attribution_of_ai-generated_content.md)
- [\[AAAI 2026\] Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection](../../AAAI2026/ai_safety/towards_multiple_missing_values-resistant_unsupervised_graph_anomaly_detection.md)
- [\[CVPR 2026\] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](../../CVPR2026/ai_safety/fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)

</div>

<!-- RELATED:END -->
