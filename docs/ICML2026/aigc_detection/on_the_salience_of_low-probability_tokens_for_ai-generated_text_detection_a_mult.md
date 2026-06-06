---
title: >-
  [Paper Note] On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective
description: >-
  [ICML 2026][AIGC Detection][AIGT Detection] Aiming at the two chronic issues of "high-frequency boilerplate diluting signals" and "brittle point probabilities" in zero-shot AI-generated text detection…
tags:
  - "ICML 2026"
  - "AIGC Detection"
  - "AIGT Detection"
  - "Low-probability tokens"
  - "Rényi entropy"
  - "Multiscale uncertainty"
  - "Conditional independent sampling"
date: 2026-05-08
content_hash: e23a120d993ad8c9
---

# On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective

**Conference**: ICML 2026  
**arXiv**: [2606.02158](https://arxiv.org/abs/2606.02158)  
**Code**: https://github.com/guoyikai2000/Uncertainty-AIGT  
**Area**: AIGC Detection / Statistical AI Text Detection / Zero-shot Detection  
**Keywords**: AIGT Detection, Low-probability tokens, Rényi entropy, Multiscale uncertainty, Conditional independent sampling

## TL;DR
Aiming at the two chronic issues of "high-frequency boilerplate diluting signals" and "brittle point probabilities" in zero-shot AI-generated text detection, the authors propose the Uncertainty / Uncertainty++ detector. It aggregates log-prob only on low-probability tokens at the $\rho$-quantile of each text segment and overlays Rényi entropy at the same positions as a distribution shape signal. This approach pushes the average AUROC from 86.49 (Lastde) to 88.74 across 12 generators and 7 datasets, showing significantly higher stability under perturbations such as rewriting or alternate decoding.

## Background & Motivation

**Background**: Current mainstream AI text detection falls into three categories: watermarking (embedding signatures during generation), fine-tuned discriminators (training classification heads on labeled corpora), and statistical methods (aggregating token-level likelihoods using a proxy LM). Watermarking is ineffective for most public LLMs, and fine-tuned models suffer from poor cross-generator/cross-domain generalization and high costs. Statistical methods remain the de facto standard for zero-shot scenarios due to their efficiency and generalization, with representative methods including Likelihood, LogRank, DetectGPT, DetectLRR, Fast-DetectGPT, and Lastde.

**Limitations of Prior Work**: Statistical methods currently face two specific hurdles. 
The first is **boilerplate dominance**: averaging $\log p_\theta(x_i \mid x_{<i})$ for all tokens indiscriminately leads to dilution by high-probability "filler" phrases shared by humans and LLMs (e.g., "propose an efficient framework," "in this paper we"). These tokens score highly in both classes, contributing zero discriminative power while pulling the average towards a false negative. 
The second is **brittle point estimates**: collapsing the entire conditional distribution at each position into a single scalar (the probability of the actual token) discards distribution shape information. Point estimates can be pushed far by simple rewriting or changing decoding strategies, causing decision flips.

**Key Challenge**: Boilerplate dilutes signals because it resides in the "head" (high-probability region) of the distribution. Point estimates are brittle because they observe only one sampling realization rather than the entire distribution. The former requires "isolating truly discriminative positions," while the latter requires "distribution-level features rather than point features." A natural hypothesis is that discriminative signals are concentrated in low-probability tokens and should be described using distribution-level uncertainty metrics (entropy) rather than single-point log-probs.

**Goal**: To build a statistical detector that is robust to both boilerplate and rewriting under zero-shot conditions without retraining.

**Key Insight**: The authors validate the "Low-Probability Discriminability" hypothesis. On XSum + LLaMA3-8B + GPT-J, by separating low-probability positions at the bottom $\rho=0.15$ from high-probability positions, they found the Human–AI LogRank gap is 1.59 for the former (vs. 0.45 for the latter, a $3.5\times$ difference), and the probability ratio of AI/Human is $9.69\times$ vs. $1.38\times$. This suggests that statistical signals at low-probability positions are nearly an order of magnitude stronger, justifying dedicated aggregation.

**Core Idea**: Aggregate two complementary signals only at low-probability quantiles: the mean of local log-prob (reflecting how "surprised" the model is by the actual token) and the mean of global Rényi entropy (reflecting the shape of the entire distribution at that position). These are fused into a unified score, with Uncertainty++ further using conditional independent sampling for normalization.

## Method

### Overall Architecture

The input is an arbitrary text $\mathbf{x} = \{x_i\}_{i=0}^{n-1}$ paired with a proxy/source LM $p_\theta$. 
The pipeline consists of: (1) Obtaining the conditional distribution $p_\theta(\cdot \mid x_{<i})$ for each position $i$ to extract the "actual token log-probability" $\log p_\theta(x_i \mid x_{<i})$ and the "Rényi entropy of the distribution" $H_\alpha(p_\theta(\cdot \mid x_{<i}))$. (2) Sorting tokens within the sequence based on their actual probability values to select the index set $\mathcal{S}_\rho$ of the bottom $\rho$ quantile. (3) Calculating the local signal $z_\text{local}$ (mean log-probability) and global signal $z_\text{global}$ (mean Rényi entropy) over $\mathcal{S}_\rho$. (4) Fusing them into a detection score $z = \beta z_\text{local} + (1-\beta) z_\text{global}$. Uncertainty++ replaces $z_\text{local}$ with a normalized value $D_\rho^*$ based on conditional independent sampling. The output is a scalar; higher values indicate a higher likelihood of AI generation.

### Key Designs

1.  **Percentile-Based Local Uncertainty**:
    - **Function**: Replaces the full-sequence log-prob mean with the mean over low-probability positions to suppress boilerplate dilution.
    - **Mechanism**: Define a percentile aggregation operator $\mathcal{Q}_\rho(\{y_i\}) = \frac{1}{|\mathcal{S}_\rho|}\sum_{i \in \mathcal{S}_\rho} y_i$ applied to token-level log-probs to get $z_\text{local} = \mathcal{Q}_\rho(\{\log p_\theta(x_i \mid x_{<i})\}_{i=1}^{n-1})$. Theoretical support comes from Proposition 3.2: $\mathcal{Q}_\rho$ is a concave operator, so Jensen’s inequality gives $\mathbb{E}\,\mathcal{Q}_\rho \le \mathcal{Q}_\rho(\mathbb{E})$. Empirically, Proposition 3.3 shows that the normalized "Percentile Discrepancy" $D_\rho^* = 2.18$ for AI text is significantly larger than the Jensen lower bound $\tilde{D}_\rho^* = 1.34$ and the full-sequence baseline $D_1^* = 1.43$, whereas all three are $\approx 0$ for human text. This "Jensen amplification asymmetry" is key to extracting more signals than Fast-DetectGPT.
    - **Design Motivation**: Boilerplate is concentrated in the head of the distribution and acts as noise in a mean pool. Discriminative power stems from the asymmetry that "AI does not actually pick very low-probability words when it should." Shrinking the pool to the $\rho$ quantile amplifies this asymmetry. While smaller $\rho$ yields purer signals, it increases variance; 0.15 is the default balance point.

2.  **Entropy-Based Global Uncertainty**:
    - **Function**: Characterizes uncertainty via distribution shape rather than single-point samples, providing inherent robustness against multiplicative perturbations like rewriting or decoding changes.
    - **Mechanism**: Use $\alpha$-order Rényi entropy $H_\alpha(p) = \frac{1}{1-\alpha}\log\sum_{v \in \mathcal{V}} p(v)^\alpha$, averaged over $\mathcal{S}_\rho$: $z_\text{global} = \frac{1}{|\mathcal{S}_\rho|}\sum_{i \in \mathcal{S}_\rho} H_\alpha(p_\theta(\cdot \mid x_{<i}))$. Proposition 3.4 proves that under polynomial perturbation $p \to p/\gamma$, log-prob changes exactly by $\log \gamma$ (growing unbounded as $\gamma \to \infty$), whereas the change in Rényi entropy at low-probability positions ($p_\theta(x_i) \le \tau$) is bounded by $O(\tau^{\min(\alpha,1)})$ and remains bounded as $\gamma$ increases.
    - **Design Motivation**: $\alpha < 1$ emphasizes the distribution tail, while $\alpha > 1$ emphasizes the head. This provides a "knob" for selective bias in the vocabulary dimension, orthagonally complementing the low-probability filtering in the token dimension. Applying $\mathcal{Q}_\rho$ to entropy also allows local and global paths to share the $\rho$ hyperparameter, simplifying the system.

3.  **Conditional Independent Sampling (Uncertainty++)**:
    - **Function**: Decouples the local signal from text length and domain influences to obtain a more stable and comparable detection score.
    - **Mechanism**: Align $z_\text{local}$ with its "expected distribution from the model itself." At each position $i$, use the original prefix $x_{<i}$ (not the sampled token) to independently sample $\tilde{x}_i \sim p_\theta(\cdot \mid x_{<i})$. Define Percentile Discrepancy $D_\rho = z_\text{local} - \mathbb{E}\,\mathcal{Q}_\rho(\{\log p_\theta(\tilde{x}_i \mid x_{<i})\})$ and its normalized version $D_\rho^* = D_\rho / \sqrt{\mathrm{Var}\,\mathcal{Q}_\rho(\{\log p_\theta(\tilde{x}_i \mid x_{<i})\})}$. In practice, Monte Carlo with $m$ samples is used. The final Uncertainty++ score is $z_{++} = \beta D_\rho^* + (1-\beta) z_\text{global}$.
    - **Design Motivation**: AI text log-probs are significantly higher than their distribution expectations ($D_\rho^* \gg 0$), while human text stays close to the expectation ($D_\rho^* \approx 0$). This "actual vs. expected" comparison is the core of Fast-DetectGPT; the authors extend it from the full sequence to the low-probability quantile to gain discriminative power while retaining normalization stability.

### Loss & Training
The method is a training-free zero-shot detector. No parameters are trained. All hyperparameters (percentile $\rho$, Rényi order $\alpha$, fusion weight $\beta$, sample count $m$) are determined via grid search on a validation set. Sensitivity curves are provided in Section 4, Figure 3.

## Key Experimental Results

### Main Results

| Method | Avg. AUROC (12 source models × 3 datasets, black-box) | Category |
|------|------|------|
| Likelihood | 71.33 | Probabilistic Baseline |
| LogRank | 74.83 | Probabilistic Baseline |
| DetectLRR | 80.28 | Probabilistic SOTA |
| Lastde | 86.49 | Prev. Probabilistic SOTA |
| **Uncertainty (Ours)** | **88.74** | **New Probabilistic SOTA**, +2.25 vs. Lastde |
| DetectGPT | 70.94 | Sampling Baseline |
| DetectNPR | ~DetectGPT | Sampling Baseline |

Across 12 generators, this method achieved the best or tied-best performance on GPT-2, GPT-Neo-2.7B, Llama2-13B, Gemma-7B, Phi-2, and GPT-4-Turbo. On Llama3-8B, it was slightly behind Likelihood (which already hit 99.57, dominated by the ceiling effect).

### Ablation Study

| Configuration | Observation | Insight |
|------|---------|------|
| Full average ($\rho = 1$) | Degrades close to Fast-DetectGPT | "Quantile filtering" provides core gain. |
| Disable $z_\text{global}$ (Local only) | Significant drop under rewriting/decoding | Rényi entropy handles "perturbation robustness." |
| Disable $z_\text{local}$ (Global only) | Larger drop on simple datasets | Local log-prob remains the primary signal. |
| Uncertainty → Uncertainty++ | More stable cross-domain/cross-gen | Normalization absorbs length/domain bias. |
| Varying $\rho$ | Too small $\to$ high variance; Too large $\to$ weak signal. Optimal at 0.10–0.20 | Consistent with "low-prob tokens carry strong signals." |
| Varying $\alpha$ | $\alpha < 1$ (tail emphasis) works best | Distribution shape needs tail bias in vocab dimension. |

### Key Findings
- The quantified "Low-probability vs. High-probability" discriminative gap (LogRank gap 1.59 vs. 0.45) is the strongest evidence—the $3.5\times$ to $7\times$ difference suggests traditional full-sequence averaging is counterproductive.
- The Jensen amplification effect ($D_\rho^* > \tilde{D}_\rho^*$) appears only in AI text; in human text, all discrepancies are near 0. This means low-probability aggregation captures a class-conditional asymmetric signal rather than just adding noise.
- The robustness bound $O(\tau^{\min(\alpha,1)})$ for Rényi entropy explains why rewriting fails to break the detector. While log-prob is linearly amplified by multiplicative perturbations, entropy at low-probability positions is a higher-order small quantity.

## Highlights & Insights
- The paper transforms a simple intuition ("look at low-prob words") into a rigorous framework using concave operators, Jensen’s inequality, and Rényi entropy perturbation bounds.
- The "entropy over point estimates" idea is transferable to other LLM downstream tasks like membership inference or data contamination detection.
- The $\rho, \alpha, \beta$ hyperparameters form a clear design space for token-dimension filtering, vocabulary-dimension bias, and signal fusion.

## Limitations & Future Work
- Experiments are primarily in English; discriminative power on morphologically-rich languages remains unverified.
- Dependence on the full vocabulary distribution for Rényi entropy may be restricted by commercial APIs that only provide top-k logprobs.
- Optimal hyperparameters drift slightly by generator/dataset, currently requiring calibration.
- Robustness against adaptive attackers who intentionally replace low-probability words with high-probability ones was not evaluated.

## Related Work & Insights
- **vs. Fast-DetectGPT**: Fast-DetectGPT is the direct predecessor of this method's normalization but averages over the full sequence. This work shifts the "signal source" to the low-probability quantile and adds Rényi entropy as an orthogonal supplement.
- **vs. Lastde**: Lastde reached 86.49 AUROC; this work consistently outperforms it by 2.25 points in black-box settings due to the synergy of quantile aggregation and distribution shape signals.
- **vs. DetectGPT (Sampling)**: Sampling methods are costly and less robust to paraphrasing; this work shows that a single forward pass plus conditional sampling provides better robustness.
- **vs. Watermarking/Fine-tuning**: This zero-shot approach requires no generation-stage cooperation or labeled training, making it the most cost-effective for academic integrity or media provenance cases where model weights are inaccessible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)
- [\[ICML 2026\] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)
- [\[ICML 2026\] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)
- [\[ACL 2026\] C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts](../../ACL2026/aigc_detection/c-red_a_comprehensive_chinese_benchmark_for_ai-generated_text_detection_derived_.md)
- [\[ICLR 2026\] Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](../../ICLR2026/aigc_detection/is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)

</div>

<!-- RELATED:END -->
