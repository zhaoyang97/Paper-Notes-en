---
title: >-
  [Paper Note] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal Hallucination] KVSmooth proposes a training-free, plug-and-play method that applies attention row entropy-guided adaptive EMA smoothing to the KV-Cache, reducing LLaVA-1.5's CHAIR_S from 41.8 to 18.2 (a 56% reduction) while simultaneously improving F1 from 77.5 to 79.2, achieving gains in both precision and recall.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multimodal Hallucination
  - KV Cache Smoothing
  - Attention Row Entropy
  - EMA
  - Training-Free
date: 2026-05-08
content_hash: a5234b9b678b169b
---

# KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing

**Conference**: CVPR 2026
**arXiv**: [2602.04268](https://arxiv.org/abs/2602.04268)
**Code**: None
**Area**: Multimodal VLM / Hallucination Mitigation
**Keywords**: Multimodal Hallucination, KV Cache Smoothing, Attention Row Entropy, EMA, Training-Free

## TL;DR

KVSmooth proposes a training-free, plug-and-play method that applies attention row entropy-guided adaptive EMA smoothing to the KV-Cache, reducing LLaVA-1.5's CHAIR_S from 41.8 to 18.2 (a 56% reduction) while simultaneously improving F1 from 77.5 to 79.2, achieving gains in both precision and recall.

## Background & Motivation

**Background**: MLLMs (LLaVA, MiniGPT-4, InstructBLIP) have achieved notable progress on vision-language tasks such as image captioning, yet frequently generate hallucinations — content inconsistent with the input image.

**Limitations of Prior Work**: (1) Retraining/fine-tuning approaches (e.g., POVID) incur prohibitive costs; (2) contrastive decoding methods (e.g., VCD) reduce hallucinations at the expense of recall (F1 drops from 77.5 to 71.1); (3) attention redistribution methods (PAI, MiddleLayer) similarly suppress correctly grounded objects.

**Key Challenge**: As the decoding sequence grows, the influence of early visual tokens gradually diminishes in the hidden states (semantic drift), causing the model to rely increasingly on linguistic priors. Existing methods only address symptoms: they either trade recall for precision or trade efficiency for quality.

**Goal**: Suppress semantic drift and hallucination during decoding without retraining, without sacrificing recall, and with virtually zero additional overhead.

**Key Insight**: From the perspective of hidden-state dynamics, the paper identifies attention sink as the direct cause of hallucination and proposes adaptive smoothing of the KV-Cache.

**Core Idea**: Row entropy quantifies the degree of sink → high-sink tokens receive stronger EMA smoothing → semantic drift is suppressed → hallucinations are eliminated.

## Method

### Overall Architecture

KVSmooth is an adaptive EMA smoothing method applied to the KV-Cache at inference time. It is entirely training-free and plug-and-play, requiring only an EMA update to the Key and Value of each newly generated token at each decoding step, with smoothing coefficients adaptively determined by attention row entropy.

### Key Designs

1. **Three Key Observations (Diagnostic Causal Chain)**:

   - **Obs1 (Logit Dynamic Divergence)**: Statistical analysis over 200 images reveals that the mean and variance of logits for ground-truth objects monotonically decrease during decoding, while those for hallucinated objects steadily increase — longer decoding leads to more severe hallucination.
   - **Obs2 (Row Entropy ≈ Sink Intensity)**: Attention row entropy is proposed as a real-time measure of token sink degree. Higher row entropy → more uniform attention distribution → hidden state approximates the historical mean → smaller angular distance → the token attracts disproportionate attention in subsequent steps, forming a sink. Cosine similarity with the conventional column-sum method reaches 0.79.
   - **Obs3 (Row Entropy↑ → Hallucination↑)**: Hallucinated objects exhibit the highest cosine similarity between row entropy and logit rank, while ground-truth objects show negative correlation — sink tokens systematically inflate hallucination scores through the global averaging mechanism.

2. **EMA Smoothing on KV-Cache**:

   - **Function**: Applies exponential moving average smoothing to the Key and Value vectors in the KV-Cache.
   - **Mechanism**: Hidden-state evolution is modeled as a Markov random walk; Bayesian MAP estimation naturally derives the EMA formulation.
   - **Key Formula**: $\hat{K_t^l} = (1-\tilde{\lambda}_t^l)K_t^l + \tilde{\lambda}_t^l K_{t-1}^l$, and analogously for Values.
   - **Design Motivation**: Experiments show that smoothing K+V jointly yields the best results (superior to smoothing K alone or smoothing the hidden state directly), as joint smoothing maximally suppresses the growth of logit mean and variance.

3. **Entropy-Guided Coefficient Adaptation**:

   - **Function**: Adaptively adjusts smoothing strength according to the sink degree of each token.
   - **Mechanism**: A FIFO queue of length $M=15$ maintains recent row entropy values; the smoothing coefficient for the current token equals its percentile rank within the queue. Higher row entropy → higher percentile → stronger smoothing.
   - **Stabilization**: Coefficients are clipped to $[\lambda_{ref}-0.2, \lambda_{ref}+0.2]$ to prevent extreme values.
   - **Design Motivation**: Different tokens contribute differently to hallucination; a fixed coefficient would over-suppress the semantic flow of normal tokens. The adaptive mechanism precisely targets high-sink tokens.

### Loss & Training

Entirely training-free. Hyperparameters: smoothing is applied to layers 3–31 (excluding layers 0–2 and the final layer); FIFO queue length is 15; $\lambda_{ref}$ is set to 0.9 for LLaVA-1.5, 0.5 for MiniGPT-4, and 0.7 for InstructBLIP (fixed across benchmarks).

## Key Experimental Results

### Main Results (CHAIR Hallucination Evaluation)

| Model | Method | CHAIR_S↓ | F1↑ | Note |
|-------|--------|----------|-----|------|
| LLaVA-1.5 | Baseline | 41.8 | 77.5 | - |
| LLaVA-1.5 | VCD | 56.0 | 71.1 | Hallucination increases; recall drops sharply |
| LLaVA-1.5 | OPERA | 44.2 | 78.6 | Inference 10× slower |
| LLaVA-1.5 | PAI | 22.6 | 75.5 | Precision loss |
| LLaVA-1.5 | MiddleLayer | 17.8 | 75.9 | Precision loss |
| LLaVA-1.5 | **KVSmooth** | **18.2** | **79.2** | **Both precision and recall improve** |
| MiniGPT-4 | Baseline | 31.8 | 69.9 | - |
| MiniGPT-4 | **KVSmooth** | **17.0** | **71.7** | CHAIR_S reduced by 47% |
| InstructBLIP | Baseline | 61.4 | 71.6 | - |
| InstructBLIP | **KVSmooth** | **42.2** | **75.1** | CHAIR_S reduced by 31% |

### Ablation Study

| Configuration | CHAIR_S | F1 | Note |
|---------------|---------|----|------|
| Smooth K only | Higher than K+V | - | Weaker hallucination suppression |
| Smooth hidden state only | - | Sharp drop | Large recall degradation |
| Smooth K+V jointly (best) | 18.2 | 79.2 | Optimal combination |
| Fixed coefficient (best fixed) | >Adaptive | <Adaptive | Both metrics inferior to adaptive |
| Exclude layers 0–2 + final layer | Best | Best | These layers are unsuitable for smoothing |

### Key Findings

- Inference speed is 3.61s/caption, only 7% slower than baseline (3.36s), far faster than OPERA (34.62s).
- CHAIR_SR on Object HalBench decreases by 63.1%.
- Larger $\lambda_{ref}$ yields stronger smoothing and lower CHAIR_S, while F1 remains nearly unchanged — the method is robust to hyperparameter choice.
- **KVSmooth is the only method that simultaneously outperforms all baselines on both CHAIR_S and F1.**

## Highlights & Insights

- The causal chain analysis — logit divergence → row entropy as sink measure → sink amplifies hallucination — is a distinctive diagnostic contribution spanning three coherent observations.
- Attention row entropy as a real-time measure of sink intensity is a significant contribution, being more efficient than OPERA's column-sum method (no backtracking required).
- The theoretical derivation is elegant: EMA smoothing optimality is rigorously derived from Bayesian MAP estimation.
- Simultaneous improvement in both precision and recall (a tradeoff that undermines most competing methods) is the core competitive advantage of the approach.
- Inference overhead is negligible (+7% latency, unchanged memory), making the method highly practical.

## Limitations & Future Work

- $\lambda_{ref}$ requires manual tuning per model, though it can be fixed across benchmarks.
- Validation is limited to 7B-scale models; effectiveness on 70B+ models remains to be verified.
- Evaluation is currently restricted to image captioning; performance on VQA, dialogue, and other tasks is unknown.
- The EMA window is fixed at one step (only the preceding token is used); a longer window may yield further improvements.

## Related Work & Insights

- **vs. VCD (contrastive decoding)**: VCD substantially reduces recall (F1 71.1); KVSmooth improves both precision and recall simultaneously (F1 79.2).
- **vs. OPERA (attention penalization)**: OPERA requires backtracking to redistribute attention, resulting in 10× slower inference; KVSmooth introduces virtually no additional overhead.
- **vs. PAI/MiddleLayer (attention redistribution)**: These methods improve precision at the cost of recall; KVSmooth uses the adaptive mechanism to precisely identify tokens requiring smoothing.
- **vs. PruneHal (KV pruning)**: PruneHal removes redundant tokens; KVSmooth retains all tokens while modulating their influence.
- The row entropy metric is transferable to anomalous attention detection scenarios; EMA smoothing of the KV-Cache can be combined with quantization and pruning techniques.

## Rating

⭐⭐⭐⭐⭐ (4.5/5)

- **Novelty** ⭐⭐⭐⭐: The causal analysis linking row entropy → sink → hallucination is novel, and the perspective of EMA smoothing on the KV-Cache is unique.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: 4 benchmarks × 3 models + PR curves + efficiency analysis + extensive ablations.
- **Writing Quality** ⭐⭐⭐⭐⭐: The logical chain from observation → derivation → method → validation is exceptionally clear.
- **Value** ⭐⭐⭐⭐: A training-free, plug-and-play hallucination mitigation solution with virtually zero overhead.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] GACD: Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](gacd_gradient_self_reflection_hallucination.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/multimodal_vlm/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)

<!-- RELATED:END -->
