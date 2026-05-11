---
title: >-
  [Paper Note] MAPSS: Manifold-Based Assessment of Perceptual Source Separation
description: >-
  [ICLR 2026][Audio & Speech][source separation evaluation] This paper proposes two complementary metrics—Perceptual Separation (PS) and Perceptual Match (PM)—that embed self-supervised encoded representations onto a low-d…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "source separation evaluation"
  - "perceptual metrics"
  - "diffusion maps"
  - "manifold learning"
  - "self-supervised representations"
date: 2026-05-08
content_hash: a26cadafd1eea115
---

# MAPSS: Manifold-Based Assessment of Perceptual Source Separation

**Conference**: ICLR 2026  
**arXiv**: [2509.09212](https://arxiv.org/abs/2509.09212)  
**Code**: Available ([https://github.com/Amir-Ivry/MAPSS-measures](https://github.com/Amir-Ivry/MAPSS-measures))  
**Area**: Audio & Speech  
**Keywords**: source separation evaluation, perceptual metrics, diffusion maps, manifold learning, self-supervised representations

## TL;DR

This paper proposes two complementary metrics—Perceptual Separation (PS) and Perceptual Match (PM)—that embed self-supervised encoded representations onto a low-dimensional manifold via diffusion maps, achieving for the first time a functional decoupling of leakage and self-distortion in source separation evaluation. Compared against 18 mainstream metrics, the proposed measures rank first or second in correlation with subjective listening scores in nearly all experimental conditions.

## Background & Motivation

Objective evaluation of source separation has long been misaligned with human perceptual judgments. Fundamental limitations of existing metrics include:

**Conflation of leakage and distortion**: SDR, SI-SDR, and similar metrics combine competing-speaker leakage and target signal distortion into a single global energy ratio, providing no diagnosis of error origin.

**Lack of fine-grained analysis**: PESQ and STOI collapse entire utterances into a single MOS score, offering no frame-level localization capability.

**Black-box uncertainty**: Learning-based metrics such as DNSMOS cannot quantify decision reliability.

**Inability to satisfy multiple requirements simultaneously**: No existing metric family jointly achieves decoupled leakage/distortion assessment, frame-level analysis, and error estimation.

**Core objective**: Design complementary perceptual metrics—PS to quantify separation (leakage) and PM to quantify match (distortion)—both differentiable, operating at the frame level (75 fps), and equipped with theoretical error guarantees.

## Method

### Overall Architecture

The MAPSS pipeline consists of four stages:

1. **Perceptual distortion generation**: $N_p \in [60,70]$ elementary distortions (clipping, notch filtering, pitch shifting, reverberation, colored noise, etc.) are applied to each reference source in the mixture, covering the perceptual auditory space.
2. **Self-supervised encoding**: A pretrained wav2vec 2.0 model independently encodes all distorted samples, reference signals, and system outputs at 75 fps resolution.
3. **Diffusion map embedding**: The aggregated high-dimensional representations are projected onto a low-dimensional perceptual manifold $\mathcal{M}^{(d)}$ via diffusion maps.
4. **Metric computation**: PS and PM scores are computed on the manifold.

### Key Designs

#### 1. Theoretical Foundation of Diffusion Maps

Given a set of encoded high-dimensional vectors $\mathcal{X} = \{\mathbf{x}_i\}_{i=1}^N$:

- Gaussian affinity matrix: $\mathbf{K}_{i,j} = \exp(-\|\mathbf{x}_i - \mathbf{x}_j\|_2^2 / \sigma_\mathbf{K}^2)$
- $\alpha$-normalization corrects for non-uniform sampling density
- Row-stochastic transition matrix $\mathbf{P} = \mathbf{D}^{-1}\mathbf{K}$
- Spectral decomposition yields the embedding: $\boldsymbol{\Psi}_t(\mathbf{x}_i) = (\lambda_1^t \mathbf{u}_1(i), \ldots, \lambda_d^t \mathbf{u}_d(i))^T$

**Key property**: The Euclidean distance on the manifold is equivalent to the diffusion distance $D_t^2(i,j) = \|\boldsymbol{\Psi}_t(\mathbf{x}_i) - \boldsymbol{\Psi}_t(\mathbf{x}_j)\|_2^2$, ensuring that distances reflect representational dissimilarity.

#### 2. Perceptual Cluster Construction

For the $i$-th source, all distorted waveforms are encoded by wav2vec 2.0 and embedded on the manifold to form a perceptual cluster:
$$\mathcal{C}_i^{(d)} = \{\boldsymbol{\Psi}_t^{(d)}(\mathbf{x}_i), \boldsymbol{\Psi}_t^{(d)}(\mathbf{x}_{i,p}) \mid p=1,\ldots,N_p\}$$

**System outputs are excluded from the clusters** to avoid circular dependency and bias. Distortions range from mild (15 dB SNR colored noise) to severe (heavy-tailed reverberation, hard clipping).

#### 3. PS Metric (Perceptual Separation)

The Mahalanobis distance quantifies the relative distance of the system output to its assigned versus non-assigned clusters:

$$\widehat{\text{PS}}_i^{(d)} = 1 - \frac{\hat{A}_i^{(d)}}{\hat{A}_i^{(d)} + \hat{B}_i^{(d)}} \in [0,1]$$

- $\hat{A}_i^{(d)}$: distance to the assigned cluster
- $\hat{B}_i^{(d)}$: distance to the nearest non-assigned cluster
- When $\hat{A} \ll \hat{B}$, PS → 1 (perfect separation)

#### 4. PM Metric (Perceptual Match)

Quantifies perceptual alignment between the system output and the reference:

- Compute the set $\hat{\mathcal{G}}_i^{(d)}$ of intra-cluster distances from distorted signals to the reference
- Verify that distances approximately follow a Gamma distribution (confirmed via KS test)
- Estimate Gamma parameters $\hat{k}_i^{(d)}, \hat{\theta}_i^{(d)}$ via moment matching
- Insert the output-to-reference distance into the Gamma tail probability:

$$\widehat{\text{PM}}_i^{(d)} = Q(\hat{k}_i^{(d)}, \hat{a}_i^{(d)} / \hat{\theta}_i^{(d)}) \in [0,1]$$

When the output falls within the distortion cluster, PM → 1; larger deviations yield lower PM.

#### 5. Theoretical Error Guarantees

Deterministic frame-level error radii and non-asymptotic high-probability confidence intervals are derived via Schur complement decomposition:

$$|\text{PS}_i - \text{PS}_i^{(d)}| \leq \frac{B_i^{(d)} |\delta_{i,i}| + A_i^{(d)} |\delta_{i,j^*}|}{(A_i^{(d)} + B_i^{(d)})^2}$$

Experiments verify that worst-case error radii almost never alter metric rankings.

### Loss & Training

MAPSS itself requires no training—it is a **pure evaluation framework** that leverages a pretrained wav2vec 2.0 encoder. Core computation involves: distortion generation (deterministic signal processing), wav2vec 2.0 forward inference, diffusion map spectral decomposition, Mahalanobis distance computation, and Gamma distribution fitting.

Both PS and PM are **differentiable** and can be directly used as training losses.

## Key Experimental Results

### Main Results

**Comparison against 18 mainstream metrics on the SEBASS database**

Linear (Pearson) and rank (Spearman) correlations of PS and PM with human subjective MOS across English/Spanish/music mixture scenarios:

| Metric Category | Representative Metrics | Ranking |
|----------------|----------------------|---------|
| Energy ratios | SDR, SI-SDR, SIR, SAR | Below average |
| Classical perceptual | PESQ, STOI, ESTOI | Moderate |
| Learning-based | DNSMOS, SpeechBERTscore | Above average |
| **MAPSS** | **PS, PM** | **Rank 1st or 2nd in nearly all conditions** |

**Complementarity validation**: Normalized mutual information (NMI) analysis between PS and PM confirms high complementarity—PS captures leakage while PM captures distortion, providing non-overlapping evaluation perspectives.

### Ablation Study

**Encoder selection**: wav2vec 2.0 achieves the best performance, with self-supervised representations most aligned with human perception.

**Distortion set size**: $N_p \in [60,70]$ is the optimal range; smaller sets yield insufficient coverage, while larger sets exhibit diminishing returns.

**Error radius validation**: Frame-level deterministic error radii alter PS/PM rankings in almost no scenarios; high-probability confidence intervals provide additional statistical guarantees.

### Key Findings

1. **Decoupling is effective**: PS specifically captures leakage and PM specifically captures distortion; NMI confirms complementarity.
2. **Self-supervised representations + manifold learning outperform traditional features**: Diffusion maps naturally produce meaningful perceptual clusters.
3. **Value of frame-level granularity**: Frame-level evaluation at 75 fps enables fine-grained localization of separation quality issues.
4. **Cross-lingual and cross-modal generalization**: Strong performance across English, Spanish, and music scenarios.

## Highlights & Insights

- **First evaluation metric to functionally decouple leakage and distortion** in source separation, filling a methodological gap.
- **The "perceptual-geometric hypothesis" is empirically validated**: The chain diffusion distance → Euclidean distance → perceptual similarity holds in practice.
- **Differentiability enables use as a training loss**, bridging the gap between evaluation and optimization.
- The elementary distortion set is elegantly designed: it constructs a "perceptual neighborhood" of the reference signal spanning mild to severe degradations.
- **First theoretical error guarantees for separation metrics**: deterministic radii combined with non-asymptotic confidence intervals.

## Limitations & Future Work

1. Encoding 60–70 distortion variants per source incurs substantial computational overhead, limiting real-time applicability.
2. Dependence on wav2vec 2.0 may be suboptimal for non-speech audio (e.g., purely instrumental signals).
3. The $N_f \geq 2$ assumption: PS requires non-assigned clusters and cannot be directly applied to single-source enhancement scenarios.
4. The hand-crafted distortion set may have blind spots; data-driven distortion generation is worth exploring.
5. Rank correlation for Spanish is relatively weak; cross-lingual robustness requires further validation.

## Related Work & Insights

- **Diffusion maps** (Coifman & Lafon, 2006), originally developed for dimensionality reduction, are innovatively applied here to audio quality assessment.
- **wav2vec 2.0** self-supervised representations effectively capture perceptually relevant audio features.
- **Mahalanobis distance + Gamma distribution modeling** provides a probabilistic-statistical framework.
- Insight: Manifold learning holds great promise for evaluation metric design and could be extended to image and video quality assessment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — An entirely new evaluation paradigm with pioneering contributions in both theory and practice.
- Technical Depth: ⭐⭐⭐⭐⭐ — Diffusion map derivations are thorough; error guarantees are complete and non-trivial.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison against 18 baselines, though only one evaluation database is used.
- Practical Value: ⭐⭐⭐⭐ — Differentiability enables use as a training loss, though computational cost may limit large-scale deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Incentive-Aligned Multi-Source LLM Summaries](incentive-aligned_multi-source_llm_summaries.md)
- [\[ICLR 2026\] Knowing When to Quit: Probabilistic Early Exits for Speech Separation](knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)
- [\[ICLR 2026\] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention](efficient_audio-visual_speech_separation_with_discrete_lip_semantics_and_multi-s.md)
- [\[AAAI 2026\] Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment](../../AAAI2026/audio_speech/multi-granularity_interactive_attention_framework_for_residual_hierarchical_pron.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)

</div>

<!-- RELATED:END -->
