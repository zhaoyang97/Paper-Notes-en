---
title: >-
  [Paper Note] Multi-Metric Preference Alignment for Generative Speech Restoration
description: >-
  [AAAI 2026][Image Generation][Preference alignment] This paper proposes a Multi-Metric Preference Alignment strategy that constructs a preference dataset, GenSR-Pref (80K pairs), requiring unanimous agreement across multiple complementary metrics. DPO is applied to post-training alignment of three generative speech restoration paradigms (AR, MGM, FM), achieving substantial quality improvements while effectively mitigating reward hacking.
tags:
  - AAAI 2026
  - Image Generation
  - Preference alignment
  - DPO
  - speech restoration
  - multi-metric
  - generative models
date: 2026-05-08
content_hash: bda140a8216ae154
---

# Multi-Metric Preference Alignment for Generative Speech Restoration

**Conference**: AAAI 2026
**arXiv**: [2508.17229](https://arxiv.org/abs/2508.17229)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: Preference alignment, DPO, speech restoration, multi-metric, generative models

## TL;DR

This paper proposes a Multi-Metric Preference Alignment strategy that constructs a preference dataset, GenSR-Pref (80K pairs), requiring unanimous agreement across multiple complementary metrics. DPO is applied to post-training alignment of three generative speech restoration paradigms (AR, MGM, FM), achieving substantial quality improvements while effectively mitigating reward hacking.

## Background & Motivation

Generative Speech Restoration (GenSR) has made significant progress in recent years, covering tasks such as denoising, dereverberation, declipping, and super-resolution. However, these models are typically trained with likelihood maximization objectives, which are misaligned with human perceptual preferences. Post-training alignment has proven effective in NLP and image generation, yet remains largely unexplored in speech restoration.

Applying preference alignment to GenSR presents three key challenges:

**Preference signal definition**: How to construct automated proxies that capture the multi-dimensional nature of human auditory perception (clarity, naturalness, artifact-free output)?

**High-quality preference data construction**: How to effectively build preference pairs that robustly guide model optimization?

**Mitigating reward hacking**: How to ensure genuine holistic improvement rather than exploitation of biases in any single metric?

## Method

### Mechanism: Multi-Metric Unanimous Agreement Criterion

The authors argue that the key to addressing reward hacking lies in making the preference signal itself multi-dimensional and comprehensive. To this end, a strict unanimous agreement criterion is proposed: a valid preference pair is formed only when one sample is superior to another across all complementary metrics simultaneously.

### GenSR-Pref Dataset Construction

Four complementary evaluation dimensions are selected to construct the preference signal:

| Dimension | Metric | Evaluation Scope |
|-----------|--------|-----------------|
| Perceptual quality | NISQA | Overall listening quality, naturalness, artifact level |
| Signal fidelity | DNSMOS (SIG/BAK/OVRL) | Signal distortion, background noise, overall quality |
| Content consistency | SpeechBERTScore | Semantic similarity to ground-truth transcription |
| Timbre preservation | Speaker Similarity | Speaker identity preservation (cosine similarity) |

The dataset comprises approximately 80K preference pairs in total: 69,456 pairs in the MGM subset for large-scale validation, and approximately 3K pairs each for AR/FM/MGM for controlled ablation studies.

### DPO Adaptation for Three Generative Paradigms

The paper unifies DPO adaptation across three mainstream generative paradigms:

- **Autoregressive model (AR)**: A two-stage AR+Soundstorm pipeline that first predicts semantic tokens and then converts them to acoustic tokens. DPO directly contrasts the log-probability ratios of winning and losing sequences.
- **Masked generative model (MGM)**: AnyEnhance is employed, predicting acoustic tokens from partially masked sequences. DPO is extended to the non-autoregressive setting by contrasting conditional probabilities.
- **Flow matching model (FM)**: Flow-SR, based on a DiT architecture, learns a velocity field from noise to clean mel-spectrograms. DPO uses single-step L2 error differences as a likelihood proxy.

## Experiments

### Experiment 1: Effectiveness of Multi-Metric Preference Alignment (Table 1)

Performance before and after alignment is evaluated on three GSR benchmarks:

| Dataset | Model | Aligned | SIG↑ | BAK↑ | OVRL↑ | NISQA↑ | SBERT↑ | SIM↑ |
|---------|-------|---------|------|------|-------|--------|--------|------|
| Voicefixer-GSR | AnyEnhance (MGM) | ✗ | 3.406 | 4.073 | 3.136 | 4.308 | 0.829 | 0.924 |
| | | ✓ | **3.532** | **4.091** | **3.267** | **4.639** | **0.834** | **0.935** |
| | AR+Soundstorm (AR) | ✗ | 3.550 | 4.097 | 3.294 | 4.556 | 0.788 | 0.894 |
| | | ✓ | **3.564** | **4.144** | **3.331** | **4.850** | **0.803** | **0.904** |
| | Flow-SR (FM) | ✗ | 3.398 | 3.969 | 3.104 | 4.010 | 0.812 | 0.918 |
| | | ✓ | **3.483** | **4.092** | **3.230** | **4.672** | **0.830** | **0.924** |

All three paradigms achieve consistent and significant improvements across all metrics after alignment. The MGM model achieves a NISQA gain of +0.519 on Librivox-GSR, while AR and FM obtain NISQA gains of +0.388 and +0.641, respectively, using only 3K preference pairs. In subjective A/B testing, the aligned model achieves a peak win rate of 54.5%.

### Experiment 2: Multi-Metric vs. Single-Metric Ablation (Table 3)

Different preference criteria are compared on the AR model:

| Criterion | SIG | BAK | OVRL | NISQA | SBERT | SIM |
|-----------|-----|-----|------|-------|-------|-----|
| No alignment | 3.550 | 4.097 | 3.294 | 4.556 | 0.788 | 0.894 |
| **Multi-Metric** | **3.564** | **4.144** | **3.331** | **4.850** | **0.803** | **0.904** |
| NISQA only | 3.531 | 4.137 | 3.300 | 4.810 | 0.785 | 0.896 |
| OVRL only | 3.561 | 4.117 | 3.317 | 4.600 | 0.792 | 0.896 |
| SIM only | 3.537 | 4.101 | 3.285 | 4.577 | 0.792 | 0.901 |
| SBERT only | 3.540 | 4.109 | 3.291 | 4.612 | 0.804 | 0.901 |

Single-metric alignment improves only the target metric, while non-target metrics frequently stagnate or degrade (e.g., SIG/OVRL decline under SIM-only alignment). The multi-metric strategy achieves the best results across all metrics, effectively mitigating reward hacking.

## Key Findings

- **DPO vs. SFT**: DPO consistently outperforms both SFT baselines (SFT-GT and SFT-Winner), indicating that exposure to high-quality samples alone is insufficient for effective alignment.
- **GT as fixed winner leads to model collapse**: Using ground truth as a fixed winner causes the model to learn pathological shortcuts—extreme suppression of probabilities for all non-GT outputs—leading to inflated reward margins and saturated reward accuracy.
- **In-Paradigm Alignment**: Each model performs best when aligned with preference data from its own paradigm. Cosine similarity analysis of preference vectors demonstrates that in-paradigm data yields more consistent optimization directions.
- **Pseudo-labeling application**: The aligned model can serve as a data annotator, generating pseudo-labels to train discriminative models in data-scarce scenarios (e.g., singing voice restoration). Voicefixer fine-tuned with pseudo-labels achieves significant improvements across all metrics.

## Highlights & Insights

- First systematic introduction of preference alignment to generative speech restoration, covering all three major paradigms: AR, MGM, and FM
- The multi-metric unanimous agreement criterion is elegantly designed to address reward hacking at its source
- Significant improvements are achieved with only ~3K preference pairs, demonstrating exceptional data efficiency
- The in-paradigm alignment principle is discovered and quantitatively explained via cosine similarity analysis of preference vectors
- The application of the aligned model as a pseudo-annotator demonstrates the bridging potential between generative and discriminative paradigms

## Limitations & Future Work

- Whether four automated metrics can fully proxy human auditory preferences warrants further validation
- The substantial size disparity across paradigm subsets in GenSR-Pref (MGM 69K vs. AR/FM 3K) limits fairness of comparison
- The strict unanimous agreement criterion may discard a large number of valuable preference pairs, limiting data utilization
- Only DPO is explored as the alignment algorithm; comparisons with PPO, KTO, and other methods are absent
- The pseudo-labeling application is validated only on singing voice restoration, leaving generalizability insufficiently demonstrated

## Related Work & Insights

- **Generative speech restoration**: SELM, GenSE, SpeechX (AR paradigm); MaskSR, AnyEnhance (MGM paradigm); SGMSE, FlowSE (continuous dynamics paradigm)
- **Post-training alignment in audio**: MetricGAN (adversarial training optimizing PESQ); NISQA+PPO-based alignment; UTMOS+DPO-based single-metric alignment
- **Preference optimization**: DPO, RLHF applied in NLP/vision/TTS; the INTP framework extending DPO to non-autoregressive settings

## Rating

⭐⭐⭐⭐ — The method is concise and effective; the multi-metric unanimous agreement criterion is elegantly designed, and the unified three-paradigm framework demonstrates strong generality. Ablation studies thoroughly validate the core hypothesis. The discovery and quantitative analysis of the in-paradigm alignment principle carry academic value, and the pseudo-labeling application broadens the practical impact of the work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](../../ICLR2026/image_generation/diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[AAAI 2026\] PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)
- [\[AAAI 2026\] Multi-Aspect Cross-modal Quantization for Generative Recommendation](multi-aspect_cross-modal_quantization_for_generative_recommendation.md)
- [\[AAAI 2026\] MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment](macs_multi-source_audio-to-image_generation_with_contextual_significance_and_sem.md)
- [\[ICLR 2026\] GenDR: Lighten Generative Detail Restoration](../../ICLR2026/image_generation/gendr_lighten_generative_detail_restoration.md)

</div>

<!-- RELATED:END -->
