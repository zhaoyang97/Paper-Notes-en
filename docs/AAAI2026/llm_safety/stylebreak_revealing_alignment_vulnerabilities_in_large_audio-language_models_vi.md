---
title: >-
  [Paper Note] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak
description: >-
  [AAAI 2026][LLM Safety][Audio jailbreak] This paper proposes StyleBreak, the first audio jailbreak framework based on speech style, which systematically investigates the impact of linguistic, paralinguistic…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Audio jailbreak"
  - "large audio-language models"
  - "alignment robustness"
  - "speech style attack"
  - "adaptive strategy"
date: 2026-05-08
content_hash: d58926521e00ca50
---

# StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak

**Conference**: AAAI 2026
**arXiv**: [2511.10692](https://arxiv.org/abs/2511.10692)  
**Code**: None  
**Area**: AI Safety
**Keywords**: Audio jailbreak, large audio-language models, alignment robustness, speech style attack, adaptive strategy

## TL;DR

This paper proposes StyleBreak, the first audio jailbreak framework based on speech style, which systematically investigates the impact of linguistic, paralinguistic, and extralinguistic attributes on LAM alignment robustness through a two-stage style-aware transformation pipeline and a query-adaptive policy network. StyleBreak improves ASR by 7.1%–22.3% across multiple attack paradigms.

## Background & Motivation

### Security Threats to LAMs

Large audio-language models (LAMs) achieve natural speech-based interaction by coupling audio encoders with LLMs. However, LAMs face the threat of **audio jailbreak**—adversaries craft malicious audio prompts to bypass alignment mechanisms and induce the model to generate harmful outputs.

### Limitations of Prior Work

Existing audio jailbreak research is extremely limited and methodologically simple:

**Text-semantic level**: Directly converts text jailbreaks to speech (e.g., Vanilla), ignoring the semantic and perceptual gap between text and audio.

**Signal level**: Applies shallow perturbations such as noise injection (AdvWave), pitch shifting, and accent conversion, lacking semantic intent.

**Critical blind spot**: Human speech carries three categories of information—linguistic (what is said), paralinguistic (emotion/prosody), and extralinguistic (speaker characteristics such as age and gender). How these rich expressive attributes affect LAM alignment robustness has remained entirely unexplored.

### Core Problem

Existing methods either ignore speech semantics (TTS conversion only) or apply shallow perturbations (noise, accent), and neither captures the rich expressive variation of human speech. StyleBreak aims to systematically answer: **How do different human speech attributes affect the alignment robustness of LAMs?**

## Method

### Overall Architecture

StyleBreak consists of three core components:
1. **Two-stage style-aware transformation pipeline**: Generates adversarial audio with diverse speech attributes.
2. **Query-adaptive policy network**: Automatically searches for the most effective style configuration for each query.
3. **Target LAM querying and evaluation**: Submits style-transformed audio and assesses jailbreak effectiveness.

### Key Designs

#### 1. **Emotion-Driven Prompt Transformation (EPT)**

In natural conversation, a speaker's emotion influences how a question is phrased. This module rewrites a harmful query $q$ into an emotionalized version $q_e$:
- GPT-4 injects expressive cues (exclamations, emotional modifiers) according to emotion-specific instructions.
- The original malicious intent is preserved while the linguistic expression is altered.

**Design Motivation**: Emotionalized rewriting better disguises malicious intent (ARR is 3.9× higher than the original query), exploiting the model's tolerance for emotional expression.

#### 2. **Style-Controlled Audio Attack Generation (EAG)**

CosyVoice2-0.5B (a controllable TTS model) synthesizes emotionalized text into audio with specific paralinguistic and extralinguistic attributes:

$$a_p = C(q_e, x_{ins})$$

where $x_{ins} = (t_{ins}, a_{ref})$ comprises a natural-language style description and a reference audio clip.

**Style configuration space**: $\mathcal{S} = \mathcal{E} \times \mathcal{G} \times \mathcal{A}_g$
- Emotion $|\mathcal{E}| = 7$ (e.g., anger, surprise, sadness)
- Gender $|\mathcal{G}| = 2$
- Age group $|\mathcal{A}_g| = 5$
- Total $|\mathcal{S}| = 70$ configurations

The style reference set is constructed from GigaSpeech, with 5 diverse instances randomly sampled per configuration.

#### 3. **Query-Adaptive Policy Network (QP)**

**Key observation**: Jailbreak success rates vary substantially across queries under different style configurations—the attack outcome is **query-specific** rather than uniform. Exhaustively evaluating all 70 configurations is computationally expensive and subject to API limits.

**Policy network design**: A multi-head policy network $\pi_\theta: \mathcal{Q} \to \Delta(\mathcal{S})$

- A shared feedforward encoder (two-layer MLP) processes the query representation vector $d_q$.
- Three independent classification heads predict the selection distributions over emotion, age, and gender respectively.

**Training objective**: Reward-weighted multi-task classification that maximizes expected reward:

$$\max_\theta \mathbb{E}_{q \sim \mathcal{Q}, s \sim \pi_\theta(q)} [J(M(a_p^s, t_i))]$$

where $J(\cdot) = \frac{1}{4}(\text{ARR} + \text{PV} + \text{TS} + \text{ASR})$ is a composite evaluation function averaging four metrics.

### Loss & Training

- The policy network is trained on 200 AdvBench queries; 50 non-overlapping queries are used for testing.
- CosyVoice2-0.5B is used uniformly for TTS synthesis.
- Each test is repeated 5 times to reduce variance from randomness.
- For black-box transfer attacks (GCG\*, AutoDAN\*), optimization is first performed on LLaMA-2-7B before transfer.

## Key Experimental Results

### Main Results

Evaluated models: Qwen2-Audio, Qwen-Omni, MERaLiON, Ultravox

**Improvement of StyleBreak over the Vanilla baseline (3 query iterations)**:

| Model | Baseline ASR | +StyleBreak ASR | Gain |
|-------|-------------|----------------|------|
| Qwen2-Audio | 10.0% | **30.5%** | +20.5% |
| Qwen-Omni | 0.0% | **22.2%** | +22.2% |
| MERaLiON | 4.0% | **37.8%** | +33.8% |
| Ultravox | 4.0% | **16.9%** | +12.9% |

**Improvement across attack paradigms (Qwen2-Audio)**:

| Attack | Baseline ASR | +StyleBreak ASR | Gain |
|--------|-------------|----------------|------|
| Vanilla | 10.0% | 30.5% | +20.5% |
| GCG* | 6.9% | 33.3% | +26.4% |
| AutoDAN* | 11.8% | 16.7% | +4.9% |
| SSJ | 8.0% | **41.7%** | +33.7% |

### Ablation Study

**Contribution of each module to ASR (%)**:

| Configuration | Qwen2-Audio | Qwen-Omni | MERaLiON | Ultravox |
|--------------|------------|-----------|----------|----------|
| Text original query | 1.1 | 0.0 | 1.5 | 1.0 |
| +EPT | 8.9 | 4.1 | 12.1 | 9.6 |
| Vanilla audio | 10.0 | 0.0 | 4.0 | 4.0 |
| +EPT | 15.3 | 7.0 | 20.5 | 5.4 |
| +EPT, EAG (styled audio) | 17.2 | 9.6 | 35.1 | 14.8 |
| **+EPT, EAG, QP (full)** | **30.5** | **22.2** | **37.8** | **16.9** |

Each module contributes a distinct improvement, and the full StyleBreak consistently outperforms all ablated variants.

**Single-factor effects of speech attributes**:
- **Emotion (linguistic)**: Even the most robust model, Qwen-Omni, shows ASR increase from 0 to 9.1%.
- **Emotion (paralinguistic)**: Ultravox is particularly sensitive, with ASR increasing 4.6–6.8×.
- **Age (extralinguistic)**: Elderly voices yield an average ASR 13.3% higher than children's voices.
- **Gender (extralinguistic)**: Male voices yield an average ASR 8.3% higher than female voices.

### Key Findings

1. **LAMs are more vulnerable to low-pitched voices**: Male and elderly voices consistently produce higher attack success rates—presumably because LAMs have stronger protective preferences toward high-pitched voices (children, females).
2. **The audio modality is inherently more vulnerable than text**: t-SNE visualizations reveal that LAMs are significantly less capable of distinguishing benign from malicious inputs in the audio modality compared to the text modality.
3. **MERaLiON is most vulnerable under compound attacks**: Although relatively robust to single-attribute perturbations, its multilingual and multicultural generalization capacity makes it more susceptible to complex styled audio.
4. **Policies transfer across models**: A policy trained on Qwen2-Audio transfers directly to GPT-4o and Gemini-2.5-flash with retained effectiveness.

## Highlights & Insights

1. **First systematic study of speech attribute effects on LAM alignment**: Fills a critical gap in audio security research and reveals a previously overlooked attack surface.
2. **Physiological characteristics as attack vectors**: Speaker traits such as age and gender influence model safety alignment—implying systematic bias in LAM alignment training.
3. **Efficiency of the adaptive policy**: Significant attack gains (ASR improvement of 7.1%–22.3%) are achieved with only 3 query iterations, far outperforming exhaustive search.
4. **Insightful t-SNE visualization**: Audio queries exhibit far greater overlap between benign and malicious representations in the model's embedding space than text queries, explaining why audio jailbreaks are naturally more effective.

## Limitations & Future Work

1. **Exclusive reliance on CosyVoice2-0.5B for TTS**: Other TTS systems may yield different results.
2. **Limited AdvBench query set**: The framework could be extended to more diverse harmful query types.
3. **Simple policy network architecture**: More sophisticated models may discover more effective style combinations.
4. **Absence of defense research**: The paper focuses exclusively on attacks without proposing countermeasures.
5. **Future directions**: Developing speech-attribute-aware alignment training methods to make LAMs exhibit consistent safety behavior across diverse voice characteristics.

## Related Work & Insights

- **Vanilla** directly converts text to speech via TTS → ignores modality gap.
- **GCG / AutoDAN** optimize at the text-semantic level → long audio may lose semantic fidelity.
- **SSJ** applies spelling-based audio perturbation → LAMs tend to recite rather than respond.
- **SpeechTripleNet** proposes a tripartite classification of speech information (linguistic, paralinguistic, extralinguistic) → provides the theoretical basis for StyleBreak's categorization framework.
- Implication for LAM safety: Alignment training should account not only for textual content but also for the full spectrum of speech attributes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic exploration of speech attribute effects on LAM alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 LAMs × 4 attack paradigms × 3 attribute categories, with ablation, transfer, and visualization studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and rich figures, though methodological details are scattered between the main text and appendix.
- Value: ⭐⭐⭐⭐⭐ — Reveals a critical blind spot in LAM safety with significant implications for alignment training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](../../ICLR2026/llm_safety/audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)
- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](../../ICML2026/llm_safety/medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[NeurIPS 2025\] ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models](../../NeurIPS2025/llm_safety/almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)
- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)

</div>

<!-- RELATED:END -->
