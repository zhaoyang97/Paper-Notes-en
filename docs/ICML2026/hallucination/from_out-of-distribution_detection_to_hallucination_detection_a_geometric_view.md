---
title: >-
  [Paper Note] From Out-of-Distribution Detection to Hallucination Detection: A Geometric View
description: >-
  [ICML 2026][Hallucination Detection][Paper Note] This paper treats LLM next-token prediction as a classification task on a massive vocabulary. By migrating two lightweight OOD detectors—NCI (proximity of features to weight vectors) and fDBD (distance from features to decision boundaries)—with two adaptations ("analytical proxy $\mu_G$ for training feature means" and
tags:
  - ICML 2026
  - Hallucination Detection
date: 2026-05-08
content_hash: 0313b2c2bc4d24f3
---
# From Out-of-Distribution Detection to Hallucination Detection: A Geometric View

**Conference**: ICML 2026  
**arXiv**: [2602.07253](https://arxiv.org/abs/2602.07253)  
**Code**: To be confirmed  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Detection, OOD Detection, Geometric Uncertainty, Decision Boundary, Training-free Single-sample

## TL;DR
This paper treats LLM next-token prediction as a classification task on a massive vocabulary. By migrating two lightweight OOD detectors—NCI (proximity of features to weight vectors) and fDBD (distance from features to decision boundaries)—with two adaptations ("analytical proxy $\mu_G$ for training feature means" and "calculating boundary distance only on top-$k$ candidate tokens"), it derives a **training-free, single-sample** inference-time hallucination detector. It consistently outperforms baselines such as Perplexity, Semantic Entropy, and SelfCheckGPT on CSQA, GSM8K, and AQuA.

## Background & Motivation

**Background**: Current LLM hallucination detection follows two main paths: one trains classifiers to identify hallucinations (e.g., SAPLMA, INSIDE), which are sensitive to distribution shifts and involve high training costs; the other involves training-free methods (Semantic Entropy, SelfCheckGPT, Lexical Similarity, etc.) that score by comparing consistency across **multiple sampled outputs**, avoiding training but at a high inference cost.

**Limitations of Prior Work**: Training-free multi-sample methods perform well on short QA but struggle with **reasoning tasks**. Multi-step reasoning allows for multiple valid paths, making "semantic consistency across multiple outputs" conceptually difficult to determine. Furthermore, sampling $N$ complete reasoning chains for each question leads to explosive overhead.

**Key Challenge**: Reasoning tasks simultaneously require being **training-free** (to avoid classifier drift), **single-sample** (to avoid multiple samplings), and **efficient per-token** (calculating at each step). Existing methods cannot satisfy all three.

**Goal**: Construct a training-free, single-sample hallucination detector for reasoning with controllable per-token overhead.

**Key Insight**: The authors note that OOD detection and hallucination detection both essentially measure "how uncertain the model is about its current prediction." If the LLM language head is viewed as a $|\mathcal{V}|$-class linear classifier and penultimate-layer features as the classifier input, the mature "geometric relationship between features and weight vectors" from OOD literature can be directly migrated—and such geometric metrics are inherently per-token and single-sample.

**Core Idea**: Adapt OOD detectors NCI (proximity of features to the weight vector of their predicted class $\to$ low uncertainty) and fDBD (distance of features from decision boundaries $\to$ low uncertainty) to LLMs. Minimal analytical and engineering fixes are applied for three LLM characteristics (unavailable training statistics, massive vocabulary, and stochastic decoding) to calculate per-token scores, which are averaged over the sequence as the hallucination score.

## Method

### Overall Architecture
The paper addresses hallucination detection in reasoning tasks where training classifiers (distribution shift) and multiple samplings (overhead) are impractical. The core approach treats the LLM language head as a $|\mathcal{V}|$-class linear classifier and the penultimate-layer feature $\bm{z}^t$ as the input. Mature uncertainty metrics from OOD literature regarding the "geometric position of features relative to weight vectors and decision boundaries" are then applied. During inference, a scalar geometric score $s(\bm{z}^t)$ is calculated at each decoding step $t$. Once the sequence is complete, the sequence mean $S=\frac{1}{T}\sum_t s(\bm{z}^t)$ is used as the hallucination score, evaluated against a threshold $\tau$. This process requires no weight updates, no training data, and only a single sample per response.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Language head as linear classifier<br/>Penultimate feature z^t at step t"] --> B
    subgraph G["Geometric Uncertainty Score (NCI / fDBD)"]
        direction TB
        B["NCI: Proximity of feature to predicted class weight vector"]
        C["fDBD: Distance of feature to decision boundaries"]
    end
    P["μ_G Analytical Proxy<br/>Decision-Neutral Closest Point"] --> B
    Q["fDBD top-k Candidate Pruning<br/>Only top-k logit competitor tokens"] --> C
    B --> S["Per-token geometric score → Sequence mean S"]
    C --> S
    S -->|Threshold τ| T["Determine Hallucination"]
```

### Key Designs

**1. Geometric Uncertainty Scores: Migrating NCI / fDBD to the Language Head**

Hallucination detection requires quantifying "how uncertain the model is about the current token," which OOD detection has already quantified using feature geometry. The difficulty lies in alignment with the LLM language head. Regarding the head as a linear classifier, the predicted token is $\hat{c}=\arg\max_v \bm{w}_v^\top \bm{z}+b_v$. **NCI** measures the proximity of the feature to the predicted class weight vector: $s_{\text{NCI}}(\bm{z})=\cos(\bm{w}_{\hat{c}}, \bm{z}-\bm{\mu}_G)\,\|\bm{w}_{\hat{c}}\|_2$. A higher value indicates closer proximity and lower uncertainty. **fDBD** measures the distance to other token decision boundaries using a first-order approximation $\tilde{D}_f(\bm{z},c)=|(\bm{w}_{\hat{c}}-\bm{w}_c)^\top \bm{z}+(b_{\hat{c}}-b_c)|/\|\bm{w}_{\hat{c}}-\bm{w}_c\|_2$. A larger distance indicates a position further from the boundary and lower uncertainty. Both scores are inherently per-sample and per-step, meeting the "single-sample per-token" requirement for reasoning detection with $O(d_{\text{model}})$ complexity per step, adding almost zero overhead to the forward pass. Empirical validation on CSQA (Fig. 2) confirms this migration: features of hallucinated answers are indeed closer to boundaries and further from weight vectors.

**2. Analytical Proxy for Training Feature Mean $\mu_G$: Decision-Neutral Closest Point**

The NCI formula requires the global mean of training features $\bm{\mu}_G$. However, LLM training corpora are private and massive, making estimation impossible. This is the first roadblock in migrating OOD tools. The authors bypass the data by seeking a "data-independent" analytical point. They prove (Lemma 4.1) that the feature point $\bm{z}_\star$ that minimizes logit variance across the vocabulary is a point of maximum uncertainty, with a closed-form solution $\hat{\bm{z}}_\star = -(W^\top P W)^\dagger W^\top P \bm{b}$, where $P=I-\frac{1}{|\mathcal{V}|}\mathbf{1}\mathbf{1}^\top$. For zero-bias language heads (like Llama-3.2-3B), this point simplifies to the origin $\bm{0}$ in feature space. Substituting $\bm{z}_\star$ for $\bm{\mu}_G$ makes NCI entirely independent of training data. This step is a critical key: on CSQA with Llama-3.2-3B (Table 1), the analytical proxy achieves AUROC=66.07, while the empirical mean estimated from the CSQA training set reaches only 62.79, worse than Perplexity (63.23). This indicates that LLM training features cannot be approximated by small downstream datasets; an analytical approach is necessary.

**3. Top-$k$ Candidate Pruning for fDBD: Filtering for Real Semantic Competitors**

Naive fDBD requires calculating boundary distances for all $|\mathcal{V}|-1$ tokens, which presents two issues: boundaries for rare tokens, punctuation, and numbers are almost always far away, diluting the signal of true semantic competitor tokens into noise; and it consumes $O(d_{\text{model}}|\mathcal{V}|)$ computation. The proposed approach takes only the $k$ tokens with the highest logits at each step to form $\mathcal{K}_t$ (excluding top-1, as it is the predicted token with distance 0). The normalized average boundary distance is calculated as $s_{\text{fDBD}}^k=\frac{1}{k}\sum_{c\in\mathcal{K}_t}\tilde{D}_f(\bm{z}^t,c)/\|\bm{z}^t-\bm{\mu}_G\|_2$, where $k$ is selected on a validation set. Using Quickselect, per-step complexity is reduced from $O(d_{\text{model}}|\mathcal{V}|)$ to an expected $O(d_{\text{model}}k+|\mathcal{V}|)$. This highlights candidates truly likely to be replaced in context while filtering unrelated distant tokens. In Table 2, all $k$ values from 1 to 100,000 outperform Perplexity, with a peak at $k=1000$ (AUROC 69.24 vs. 68.15 for All), showing an inverted U-shape—too small lacks information, while too large causes dilution.

### Loss & Training
**Completely training-free**; no parameters are updated. The Perplexity baseline is $\text{PPL}(\bm{y}|\bm{x})=\exp(-\frac{1}{T}\sum_t \log p(\bm{y}_t|\bm{x},\bm{y}_{<t}))$. This paper follows the same "step-wise scoring + sequence averaging" pattern but replaces log-probability with $s_{\text{NCI}}$ or $s_{\text{fDBD}}^k$. Evaluation uses AUROC, which is threshold-free.

## Key Experimental Results

### Main Results
Setup: CSQA (commonsense, MCQ, 1221 questions), GSM8K (math, free-form, 1319 questions), AQuA (math, MCQ, 254 questions), models Llama-3.2-3B-Instruct and Qwen-2.5-7B-Instruct, CoT prompting, greedy decoding.

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

Latency (Llama-3.2-3B, CSQA, ms/token): Standard 31.94, Perplexity 32.88, NCI 32.54, fDBD 32.71, indicating almost zero overhead.

### Ablation Study

| Configuration | CSQA AUROC | Description |
|------|-----------|------|
| Perplexity Baseline | 63.23 | LLM internal confidence |
| NCI w/ Empirical Mean $\bm{\mu}_G$ (CSQA train set) | 62.79 | Empirical estimation degrades performance |
| NCI w/ **Analytical Proxy $\bm{z}_\star$** | **66.07** | Proxy outperforms by +3.3 AUROC |
| fDBD $k=1$ | 68.64 | Considers only top-1 alternative |
| fDBD $k=100$ | 69.18 | |
| fDBD $k=1000$ | **69.24** | Peak performance |
| fDBD $k=10000$ | 68.87 | Dilution begins |
| fDBD $k=$ All ($\approx 10^5$) | 68.15 | Entire vocabulary is worst |

Stochastic Decoding Robustness (CSQA, Llama-3.2-3B, mean of 5 seeds): Under temp=0.2/0.5/0.8/1.0, Perplexity fluctuates around 62-63; NCI remains stable at 66-68; fDBD remains stable at 68-69. All outperform Perplexity, proving that while NCI/fDBD are defined on the "top logit token," **sequence averaging** automatically compensates for occasional misalignments under stochastic decoding.

### Key Findings
- The analytical proxy $\bm{z}_\star$ is the **critical key** for migrating OOD methods to LLMs—empirical means are not only useless but perform worse than Perplexity, indicating LLM training features cannot be approximated by small downstream datasets.
- The top-$k$ pruning curve follows an inverted U-shape: too small ($k=1$) lacks info, while too large (All) is diluted by irrelevant tokens. The peak at $k\sim 10^3$ suggests true semantic competitors are concentrated within the top thousand.
- Single-sample geometric methods are particularly effective in **math reasoning** (GSM8K/AQuA) compared to multi-sample methods like Semantic Entropy or SelfCheckGPT. While the latter require sampling $N$ complete CoTs, NCI/fDBD require only one inference with negligible latency increase (<1 ms/token).

## Highlights & Insights
- **Paradigm Reformulation**: Connecting "hallucination detection" and "OOD detection" is conceptually natural—a classifier's prediction on an unseen class is a "classification version of hallucination." The real contribution is **realizing** this: each LLM characteristic (unseen training data, massive vocabulary, stochastic decoding) is mapped to a specific engineering/analytical fix rather than just a conceptual analogy.
- **The Decision-Neutral Closest Point is a reusable tool**: Any OOD/uncertainty method relying on "training feature means" would stall when migrated to LLMs. The author's path—logit variance minimization $\to$ closed-form solution $\to$ origin for zero-bias—can be directly applied to LLM versions of Mahalanobis, Energy, and other OOD scores.
- **Per-token geometric scores + sequence averaging** is a simple but effective bridge: extending "single-point uncertainty" from the classification paradigm to sequences via arithmetic means is stable even under stochastic decoding, suggesting that in reasoning tasks, **cumulative geometric signals** are more important than "strict step-by-step alignment."
- **Near-zero latency**: 32.71 vs 31.94 ms/token means this detector can be embedded in production inference pipelines by default, unlike SelfCheckGPT which requires $N\times$ the inference budget.

## Limitations & Future Work
- The analytical proxy $\bm{z}_\star$ is the origin for zero-bias heads (Llama), but whether it remains optimal for models with non-zero bias (Qwen/MoE) was verified indirectly via AUROC without analyzing the relationship between bias magnitude and proxy deviation.
- Simple sequence averaging might mask signals from "local high-uncertainty steps" (e.g., only one or two critical steps are wrong in a long CoT). Future work could consider **max / top-percentile / weighted aggregation**.
- Evaluations were conducted on reasoning/QA; the effectiveness of token-level geometric uncertainty in open-ended long-form generation (summarization, creative writing) has not yet been verified.
- $k$ needs to be selected on a validation set, creating a cold-start cost for new tasks/models without annotated data. Adaptive $k$ (determined dynamically by logit distribution entropy) is worth exploring.

## Related Work & Insights
- **vs Semantic Entropy (Kuhn et al., 2023)**: SE requires multiple sampled responses to calculate semantic entropy (multi-sample, unsuitable for long CoT). Ours is single-sample, using geometric signals directly from penultimate features.
- **vs SelfCheckGPT (Manakul et al., 2023)**: SelfCheck also requires multiple samplings for consistency checks. Ours requires only one inference with near-zero latency.
- **vs Perplexity / Max P / P(True)**: Also single-sample and training-free, but Perplexity/Max P only use scalar summaries of logits. Ours further utilizes the **geometric position of penultimate feature vectors**, providing richer information and higher AUROC across all datasets/models.
- **vs INSIDE / SAPLMA (trained classifiers)**: Those methods learn boundaries sensitive to distribution shifts and require labeled data. Ours is training-free and needs zero labels.

## Rating
- Novelty: ⭐⭐⭐⭐ The conceptual bridge between OOD and hallucination has been suggested, but this is the first to successfully migrate "feature geometry" OOD detectors like NCI/fDBD to LLMs while solving three specific challenges.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets × 2 models + appendix extending to Qwen3-32B / base models / MoE / other architectures, plus 5 seeds for stochastic decoding.
- Writing Quality: ⭐⭐⭐⭐ The structure of "three challenges $\to$ three solutions" is very clear, with formal definitions, theorems, and approximations presented well.
- Value: ⭐⭐⭐⭐ Near-zero latency + training-free + single-sample makes it directly applicable for industry, while providing a reusable methodology for "migrating OOD tools to LLMs."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)
- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[CVPR 2026\] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](../../CVPR2026/hallucination/tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/hallucination/enhancing_hallucination_detection_via_future_context.md)
- [\[ICLR 2026\] Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection](../../ICLR2026/hallucination/beyond_in-domain_detection_spikescore_for_cross-domain_hallucination_detection.md)

</div>

<!-- RELATED:END -->
