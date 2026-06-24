---
title: >-
  [Paper Note] Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off
description: >-
  [ICLR 2026][Interpretability][RoPE] This paper reveals the existence of "frequency bands" in RoPE, jointly determined by the base frequency $\theta$ and the training length $L_{train}$. These bands are formed early in pre-training and inherited during position interpolation. The study proves that low-frequency dimensions below the band are nearly equivalent to NoPE, overturning the mainstream intuition that "increasing $\theta$ always benefits long contexts"—instead…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "RoPE"
  - "Frequency Band"
  - "base frequency θ"
  - "Long Context"
  - "Interpolation-Extrapolation Trade-off"
  - "NoPE"
date: 2026-05-08
content_hash: 07a2d5efec214ecb
---

# Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PR1PPxvG9Q](https://openreview.net/forum?id=PR1PPxvG9Q)  
**Code**: TBD  
**Area**: Interpretability / Position Encoding Analysis  
**Keywords**: RoPE, Frequency Band, base frequency θ, Long Context, Interpolation-Extrapolation Trade-off, NoPE  

## TL;DR
This paper reveals the existence of "frequency bands" in RoPE, jointly determined by the base frequency $\theta$ and the training length $L_{train}$. These bands are formed early in pre-training and inherited during position interpolation. The study proves that low-frequency dimensions below the band are nearly equivalent to NoPE, overturning the mainstream intuition that "increasing $\theta$ always benefits long contexts"—instead, increasing $\theta$ merely redistributes energy to improve interpolation at the cost of extrapolation.

## Background & Motivation
- **Background**: RoPE is the de facto standard for position encoding in current LLMs. To support longer contexts, the industry commonly scales the base frequency $\theta$ from the default 10,000 to 500,000 or even 1,000,000, based on the intuition that "larger $\theta$ mitigates the decay of attention scores over relative distance, thereby enabling extrapolation."
- **Limitations of Prior Work**: This $\theta$-scaling paradigm faces internal contradictions. On one hand, Xiong et al. justify large $\theta$ using activation decay theory; on the other hand, Barbero et al. found that replacing low-frequency RoPE dimensions with NoPE hardly affects performance. If these dimensions are optional, what does increasing $\theta$ actually contribute? Furthermore, merely increasing $\theta$ often fails to achieve robust extrapolation, still requiring position interpolation with fine-tuning (e.g., YaRN / LongRoPE).
- **Key Challenge**: The formation mechanism of the "frequency bands" observed by Barbero et al. (dimensions where query/key present high L2-norms after RoPE) remains unclear. Their analysis was limited to short texts (20-token windows) and did not cover long-context or position interpolation scenarios.
- **Goal**: To systematically answer one research question: **Does increasing $\theta$ inject useful positional information, or does it simply push a large number of RoPE dimensions into a NoPE-like state that contributes almost nothing?**
- **Core Idea**: **【Frequency Band Perspective】** By focusing analysis on frequency bands, the authors discover that the relationship between $\theta$ and context length is much tighter than previously thought. They propose **Frequency Matching Intervention (FMRoPE)**—setting $\theta = L_{train}$—to push the frequency band to the lowest frequencies, thereby revealing a clear interpolation-extrapolation trade-off.

## Method

### Overall Architecture
The paper does not propose a new model but a diagnostic, theoretical, and interventionist analytical pipeline: first confirming the ubiquity of frequency bands in real LLMs and their inheritance by position interpolation (Section 3); then decomposing how the band is determined by $\theta$ and $L_{train}$ and formed early via controlled pre-training (Section 4); followed by a closed-form derivation based on variance maximization to accurately predict the band's location (Section 5); and finally using FMRoPE intervention to shift the band to the lowest frequency, empirically exposing the interpolation-extrapolation trade-off (Section 6).

```mermaid
graph LR
    A[Diagnosis on Multiple LLMs<br/>Section 3] --> B[Frequency bands are ubiquitous<br/>Inherited by position interpolation]
    C[Controlled Pre-training<br/>Section 4] --> D[Band position determined by θ×L_train<br/>Forms early]
    E[Variance Maximization Derivation<br/>Section 5] --> F[Closed-form prediction j* ≈ d/2·log_θ(L/x*)]
    B --> G[FMRoPE: θ=L_train<br/>Section 6]
    D --> G
    F --> G
    G --> H[Clear trade-off between<br/>Interpolation↑ and Extrapolation↓]
```

### Key Designs

**1. Band Index $i_{band}$: Quantifying "high-norm dimensions" into comparable scalars.** To discuss the band, it must first be located. Using the Cauchy-Schwarz inequality $|\langle q_m, k_n\rangle| \le \|q_m\|_2\|k_n\|_2$, the authors point out that analyzing the 2-norm of the query or key is sufficient to characterize the frequency components affecting attention scores. Thus, for each token position $n$, the dimension with the maximum norm is identified as $idx_n = \arg\max_i \|k^n_i\|_2$, and the mode over the sequence is taken as the dominant dimension $\hat{idx}$. Averaging over all heads and layers yields the band index $i_{band}$. Key observation: the normalized $i_{band}/d$ decreases as $\theta$ increases—larger $\theta$ moves the band toward higher frequencies (lower dimension indices). $i_{band}$ remains nearly constant before and after position interpolation, proving it is inherited rather than corrected.

**2. p-RoPE Probe: Verifying that low-frequency dimensions below the band are nearly equivalent to NoPE.** p-RoPE applies rotation only to the top-$r$ high-frequency dimensions, where $r=0$ degrades to NoPE and $r=1$ is full RoPE. The authors test perplexity on pre-trained models without retraining: for models like Gemma and Llama, replacing low-frequency dimensions below the band with NoPE (reducing $r$) barely harms performance, confirming these dimensions are not effectively utilized. The sole exception is Phi-3, whose block-sparse attention utilizes low-frequency dimensions; reducing $r$ causes performance collapse, suggesting band behavior depends on attention structure.

**3. Variance Maximization Derivation: Closed-form prediction of band position from $(\theta, L_{train}, d)$.** This is the theoretical core. The problem of "which $\theta_i$ allows the greatest position-dependent variation under a fixed coefficient norm budget" is reduced to maximizing the variance of cosine coordinates over a window. Letting $m\sim\text{Unif}[0,L_{train}]$ and $x:=\omega L_{train}$, direct integration yields the variance $V(x)=\frac{1}{2}+\frac{\sin(2x)}{4x}-\left(\frac{\sin x}{x}\right)^2$. Numerically solving $V'(x)=0$ gives the smallest positive root $x^\star \approx 3.657210$ rad (global maximum, $V(x^\star)\approx 0.540 > 1/2$). The optimal angular frequency is $\omega^\star = x^\star/L_{train}$. Identifying the dimension in the RoPE grid closest to $\omega^\star$ yields the closed-form predictor:
$$j^\star \approx \frac{d}{2}\log_\theta\!\left(\frac{L_{train}}{x^\star}\right), \quad x^\star \approx 3.657210$$
Empirically, $i_{band} \approx c \times j^\star$ ($c\approx 1.0\text{–}1.1$) shows linear alignment, meaning the band position is pre-determined by $(\theta, L_{train}, d)$. When $\theta = L_{train} = 8192, d=128$, $j^\star=59$ and $c\times j^\star\approx 64.9$, which approaches the RoPE log limit $d/2=64$, pushing the band to the lowest frequency dimensions.

**4. FMRoPE (Frequency Matching Intervention): Exposing the trade-off with a minimalist setup.** Based on the derivation, FMRoPE sets $\theta = L_{train}$ (e.g., $\theta=512$ for pre-training, $\theta=1512$ for interpolation fine-tuning), pushing the frequency band to the lowest frequencies from the start. This allows the model to utilize a wider and more effective frequency range. It turns an abstract conclusion into an operational knob: choosing a large or small $\theta$ directly corresponds to a trade-off between interpolation vs. extrapolation.

## Key Experimental Results

### Main Results: Band Indices and p-RoPE Perplexity (Wikitext-103, L=4096)

| Model | $L_{train}$ | θ | $i_{band}$ | $i_{band}/\frac{d}{2}$ | r=1.0 | r=0.75 | r=0.50 |
|---|---|---|---|---|---|---|---|
| Gemma | 8k | 10000 | 116.68 | 0.91 | 2.52 | 81.66 | >100 |
| Qwen3 | 40k | 1M | 51.04 | 0.79 | 6.22 | 6.22 | 7.46 |
| Llama-2 | 4k | 10000 | 53.53 | 0.84 | 2.54 | >100 | >100 |
| Llama-3 | 8k | 500k | 43.43 | 0.68 | 2.29 | 2.29 | 84.50 |
| Phi-3 | 8k | 1M | 36.67 | 0.57 | 2.84 | 46.36 | >100 |

Bands are ubiquitous across models; larger $\theta$ leads to smaller $i_{band}/\frac{d}{2}$ (band shifts to higher frequencies); position interpolation models (+YaRN/+Llama3/+LongRoPE) inherit the band with $i_{band}$ remaining essentially unchanged.

### Ablation Study: Dissecting the effects of $(θ, L_{train})$ on the band ($L_{train}=512$)

| θ | $i_{band}$ | $i_{band}/\frac{d}{2}$ | r=1.0 | r=0.25 |
|---|---|---|---|---|
| 512 (=$L_{train}$) | 60.5 | 0.94 | 19.58 | 98.26 |
| 10000 | 30.12 | 0.47 | 19.39 | 63.59 |
| 500000 | 17.00 | 0.26 | 19.35 | 34.46 |
| 1000000 | 15.37 | 0.24 | 19.36 | 30.59 |

Fixed $\theta$, increasing $L_{train}$ → $i_{band}$ increases; fixed $L_{train}$, increasing $\theta$ → band shifts to lower dimensions (higher frequencies), with minimal difference between 500k and 1M. The band appears by the 6th epoch (fast convergence phase) and persists until the end of training.

### Key Findings: The Interpolation–Extrapolation Trade-off exposed by FMRoPE (Perplexity, $L_{train}=512$)

| Setup | L=512 | L=1512 | L=2512 | L=25512 |
|---|---|---|---|---|
| FMRoPE (θ=512) | 19.58 | 21.19 | 24.20 | >100 |
| FMRoPE θ_inf=3512 | 21.28 | 20.27 | 20.37 | >100 |
| RoPE θ=10000 | 19.39 | 43.63 | 84.45 | >100 |
| RoPE θ=1M | 19.35 | 37.94 | 74.26 | >100 |
| +YaRN FMRoPE | 19.62 | 17.78 | 17.56 | 23.19 |
| +YaRN θ=1M | 19.07 | 17.76 | 17.81 | >100 |

In short contexts, conventional large $\theta$ is slightly better (better interpolation), but once extrapolated to longer sequences, FMRoPE perplexity is significantly lower (e.g., 24.20/20.37 vs 84.45/74.26 at L=2512); the trade-off persists even after YaRN interpolation.

## Highlights & Insights
- **Elevating "Frequency Bands" from a phenomenon to a falsifiable law**: The authors provide a closed-form predictor $j^\star \approx \frac{d}{2}\log_\theta(L_{train}/x^\star)$ that calculates the band location using only $(\theta, L_{train}, d)$, showing linear alignment across multiple models.
- **Directly challenging the θ-scaling myth**: Increasing $\theta$ does not inject new positional information but redistributes energy—more dimensions below the band become NoPE-like, gaining interpolation performance at the expense of extrapolation.
- **Clear Practical Guidance**: Select $\theta \approx L_{train}$ when extrapolation is key; use larger $\theta$ for interpolation within the training range; position interpolation should involve "band-aware" $\theta$ selection rather than blind scaling.
- **Ready-to-use Diagnostic Tools**: $i_{band}$ + p-RoPE serve as a plug-and-play frequency band diagnostic probe requiring no retraining.

## Limitations & Future Work
- **FMRoPE requires knowing the target sequence length at inference** (e.g., tuning $\theta_{inf}$ to 1512/3512) to achieve optimal extrapolation, which is impractical for some deployments. The authors list "dynamic/adaptive $\theta$ adjustment" as future work.
- The scale of the main experiments is relatively small (controlled pre-training on a 16-layer, d=128 model; real LLM analysis on 4096 windows). Although 1B models and downstream tasks are included in the appendix, a gap remains compared to production-level ultra-long contexts.
- Phi-3's block-sparse attention breaks the p-RoPE trend, indicating that conclusions depend on the attention structure; frequency band behavior under sparse attention needs separate modeling.
- The theoretical derivation is based on a single-coordinate cosine variance proxy; while connected to the full covariance perspective in the appendix, it remains a "proxy" rather than an end-to-end proof.

## Related Work & Insights
- **Direct Antecedents**: Barbero et al. (2025) first discovered RoPE frequency bands and replaced low frequencies with NoPE but only looked at short texts and failed to explain the formation mechanism. This paper answers the "where, why, and when" and extends the analysis to long contexts and interpolation.
- **$\theta$ Design Lineage**: Xiong et al. ($\theta=500k$ to suppress decay), Peng et al. (YaRN's rule-based scaling), and Ding et al. (LongRoPE search) all favor increasing $\theta$. Conversely, Liu et al. (small $\theta=500$ for better extrapolation) and Takase & Okazaki (LRPE setting $\theta=$ sequence length) align with the FMRoPE $\theta=L_{train}$ approach.
- **Insights**: In long-context engineering, "mindless $\theta$ scaling" should be replaced by "band-aware $\theta$ selection"; frequency bands + p-RoPE can serve as a universal diagnostic framework to evaluate whether any position encoding scheme effectively utilizes frequencies.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Unifies scattered phenomena into a frequency band law determined by $(\theta, L_{train})$ with closed-form predictions; successfully challenges $\theta$-scaling intuition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 real LLMs + 3 interpolation methods + controlled pre-training ablations + theoretical verification. Scale is a minor drawback, but the chain of evidence is complete.
- **Writing Quality**: ⭐⭐⭐⭐ Clear research questions, takeaways for every section, and step-by-step theoretical derivations. Highly readable.
- **Value**: ⭐⭐⭐⭐ Provides both a falsifiable diagnosis and practical $\theta$ selection guidance with direct relevance to long-context LLM design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Probing Rotary Position Embeddings through Frequency Entropy](probing_rotary_position_embeddings_through_frequency_entropy.md)
- [\[ICLR 2026\] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)
- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[ICLR 2026\] From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning](from_tokens_to_thoughts_how_llms_and_humans_trade_compression_for_meaning.md)
- [\[ICLR 2026\] From Data Statistics to Feature Geometry: How Correlations Shape Superposition](from_data_statistics_to_feature_geometry_how_correlations_shape_superposition.md)

</div>

<!-- RELATED:END -->
