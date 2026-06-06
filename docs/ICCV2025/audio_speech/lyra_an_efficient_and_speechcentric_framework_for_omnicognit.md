---
title: >-
  [Paper Note] Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition
description: >-
  [ICCV 2025][Audio & Speech][omni-cognition] This paper proposes Lyra, a speech-centric omni-modal MLLM framework consisting of three core components — a DTW-based cross-modality regularizer, multi-modality LoRA…
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "omni-cognition"
  - "speech-centric"
  - "multi-modality LoRA"
  - "latent cross-modality regularizer"
  - "long speech"
  - "token extraction"
date: 2026-05-08
content_hash: a6df411c1b5fe0d4
---

# Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition

**Conference**: ICCV 2025
**arXiv**: [2412.09501](https://arxiv.org/abs/2412.09501)  
**Code**: [https://github.com/dvlab-research/Lyra](https://github.com/dvlab-research/Lyra)  
**Area**: Multimodal VLM / Omni-Modal Understanding / Speech Modality
**Keywords**: omni-cognition, speech-centric, multi-modality LoRA, latent cross-modality regularizer, long speech, token extraction

## TL;DR
This paper proposes Lyra, a speech-centric omni-modal MLLM framework consisting of three core components — a DTW-based cross-modality regularizer, multi-modality LoRA, and a latent multi-modality extractor — along with the first 12K long-speech SFT dataset. Using only 2.7M training samples and modest compute, Lyra achieves state-of-the-art performance simultaneously on vision-language, vision-speech, and speech-language benchmarks, while supporting speech inputs of up to 2 hours in length.

## Background & Motivation
**Background**: Existing omni-models (e.g., VITA, EMOVA, Intern-Omni) have begun exploring multimodal fusion, but the speech modality remains severely underutilized — their speech evaluation is largely confined to speech-text ASR metrics (e.g., LibriSpeech WER), neglecting cross-modal interactions between speech and other modalities such as vision.

**Limitations of Prior Work**: (1) Speech tokens and text tokens share semantic content but differ greatly in length (Whisper output tokens are much longer than their corresponding text), making direct training suboptimal; (2) Long speech support is extremely limited (VITA supports only ~1 minute); (3) Training large omni-models requires massive data (VITA: 5M, Intern-Omni: 27M), resulting in low efficiency.

**Key Challenge**: Incorporating the speech modality into an LLM through joint training degrades existing visual and language capabilities; directly replacing text instructions with speech tokens leads to a significant performance drop (MM-Vet drops 8% under speech instructions vs. text instructions).

**Goal**: To efficiently build an MLLM with genuine support for image/video/speech/sound omni-modal understanding and interaction, with a focus on deep integration of speech with other modalities.

**Key Insight**: (1) Use DTW to align speech tokens with transcription text tokens to reduce the modality gap; (2) Use modality-specific LoRA to preserve existing capabilities; (3) Use a query-aware token extractor to reduce long-context overhead.

**Core Idea**: By adopting a speech-centric design with DTW-based cross-modality regularization, multi-modality LoRA, and progressive multimodal token extraction, Lyra constructs an omni-modal MLLM efficiently with minimal training data.

## Method

### Overall Architecture
Built upon Qwen2-VL, Lyra incorporates a Whisper-v3 speech encoder and an ImageBind audio encoder. Multimodal input tokens are passed through modality-specific projectors into the LLM (equipped with multi-modality LoRA and the latent extractor), producing both text and streaming speech outputs. Training proceeds in four stages: (1) speech alignment pretraining → (2) three-modality joint training → (3) long-speech SFT → (4) speech generation.

### Key Designs

1. **Latent Cross-Modality Regularizer (LCMR)**:

    - **Function**: Aligns the latent representations of speech tokens with their corresponding transcription text tokens during training.
    - **Mechanism**: Speech tokens $X^{[speech]} \in \mathbb{R}^{d \times L}$ and STT text tokens $X^{[STT]} \in \mathbb{R}^{d \times S}$ differ in length ($L \gg S$). Dynamic Time Warping (DTW) is applied to find the optimal alignment path, minimizing the aligned cosine distance: $\mathcal{L}_{LCMR} = \frac{1}{L+S} D_{L,S}$, where the DTW distance matrix is defined as $D_{l,s} = \text{dist}(l,s) + \min\{D_{l,s-1}, D_{l-1,s}, D_{l-1,s-1}\}$.
    - **Effect**: Incorporating LCMR improves MM-Vet under speech instructions from 53.1 to 58.1 (+5.0), while also improving the text-instruction score from 61.1 to 62.6, narrowing the cross-modal performance gap from 8% to 4.5%.
    - **Design Motivation**: Speech and text are different surface representations of the same semantic content but differ in sequence length. DTW handles variable-length alignment at low cost with a well-established algorithm, ensuring speech tokens carry text-equivalent semantic information before entering the LLM.

2. **Multi-Modality LoRA (MLoRA)**:

    - **Function**: Trains separate LoRA adapters for different modality combinations, preventing new-modality training from degrading existing capabilities.
    - **Mechanism**: $H = (B^{[M]}A^{[M]} + W)X^{[M]}$, where $M$ denotes the modality combination (text, image, speech, etc.) and each combination is assigned an independent low-rank adapter.
    - **Effect**: Compared to full-parameter SFT with MLoRA, MLoRA better preserves visual capabilities (TextVQA: 82.6 vs. 81.3) while more effectively developing speech capabilities (MM-VetS: 60.0 vs. 54.0), using only 50% of the data.
    - **Design Motivation**: Pretrained models such as Qwen2-VL are already highly capable. Full-parameter fine-tuning on limited data causes catastrophic forgetting. LoRA freezes the original weights and applies low-rank updates, fundamentally reducing inter-modality interference.

3. **Latent Multi-Modality Extractor (LMME)**:

    - **Function**: At the output of each LLM block, progressively removes redundant multimodal tokens based on attention relevance between text query tokens and multimodal tokens.
    - **Mechanism**: The LLM is divided into $n$ blocks. At the end of each block, the method computes $\text{topk}(\text{softmax}(\frac{Q^{[text]} K^{[\neg text]T}}{\sqrt{d}}))$ and retains only the $\rho L$ most relevant multimodal tokens. The token count decays exponentially across blocks.
    - **Effect**: LMME(4, 0.7) reduces prefill time by nearly half (0.65s → 0.37s at $2^{14}$ tokens), training time by 29–54%, and GPU memory by over 50%, with negligible performance loss (average ±0.1%–1.5% across benchmarks).
    - **Design Motivation**: In long-video or long-speech scenarios, token counts can reach hundreds of thousands, most of which are irrelevant to the current query. Progressive extraction (as opposed to one-shot pruning) allows the model to retain information at different levels of granularity across different depths.

4. **Long Speech Capability**:

    - The first 12K long-speech SFT dataset is introduced, covering YouTube audio ranging from several minutes to 2 hours.
    - Long audio is handled via a strategy analogous to LLaVA-NeXT image slicing: segment → Whisper encode → compress to 300 tokens/segment → flatten.
    - Needle-in-a-Haystack evaluation: the baseline model fails beyond 450 seconds; with long-speech SFT, it supports up to 4,500 seconds (98% accuracy); with LMME, it supports up to 9,900 seconds (2.75 hours).

## Key Experimental Results

### Omni-Modal Benchmark Comparison

| Model | Params | Training Data | MME | TextVQA | MMMU | TextVQAS | DocVQAS | MM-VetS | WER↓ |
|------|--------|---------|-----|---------|------|---------|---------|---------|------|
| VITA | 66B | 5M | 2097 | - | 41.6 | - | - | - | 8.1 |
| EMOVA | 14B | 4M | 2205 | - | 55.8 | - | - | - | 4.0 |
| Intern-Omni | 8B | 27M | 2210 | - | 60.0 | 69.1 | 79.9 | 56.0 | - |
| **Lyra-Base** | **9B** | **2.7M** | **2335** | **82.6** | **63.5** | **80.0** | **85.5** | **61.0** | **2.0** |
| **Lyra-Pro** | **74B** | **2.7M** | **2485** | **83.5** | **71.4** | **81.0** | **89.4** | **68.5** | **1.8** |

### Ablation Study: LCMR Regularizer

| Configuration | TextVQAS | MM-VetS | TextVQAT | MM-VetT | WER↓ |
|------|---------|---------|---------|---------|------|
| w/o LCMR | 76.7 | 53.1 | 79.5 | 61.1 | 1.9 |
| **w/ LCMR** | **77.8** | **58.1** | **80.1** | **62.6** | **2.0** |

### Efficiency: LMME Extractor

| Configuration | Prefill @$2^{14}$ tokens | TPS @$2^{14}$ | Memory @$2^{14}$ | Training on 1.5M data |
|------|----------------------|-------------|-----------------|-----------|
| Baseline | 0.65s | 27.3 tok/s | 30G | 66h |
| LMME(4, 0.7) | **0.37s (−43%)** | **32.5 (+19%)** | **19G (−37%)** | **47h (−29%)** |

### Key Findings
- **Speech changes how omni-modal evaluation should be conducted**: Speech-text WER metrics fail to reflect the true capability of omni-models — models with similar WER can differ by up to 9% on vision-speech tasks. Speech-centric multimodal evaluation is necessary.
- **Superior performance with minimal data**: Lyra outperforms Intern-Omni (27M) and VITA (5M) using only 2.7M data samples, achieving a 10× improvement in data efficiency. This is primarily attributed to MLoRA preserving pretrained capabilities and LCMR improving speech alignment quality.
- **Long speech is an underexplored frontier for MLLMs**: Existing models support at most ~1 minute of speech; Lyra is the first to support 2 hours. Long speech alone can resolve one-third of VideoMME questions (78.6% accuracy using audio only, surpassing GPT-4o with captions on the long-video subset).
- **LMME token extraction is interpretable**: The retained token regions are highly correlated with the user query (different queries retain different visual/audio regions), with only 10–25% of tokens ultimately preserved.

## Highlights & Insights
- **DTW-based speech-text alignment is an elegant cross-modal regularization technique**: It leverages the natural correspondence between speech and text, and DTW handles variable-length alignment at low computational cost with a well-established algorithm. This paradigm can be generalized to any cross-modal alignment scenario with a natural correspondence.
- **Modality-combination routing in MLoRA is simple yet effective**: Assigning different LoRA adapters to different modality combinations achieves better results than full-parameter SFT with only half the data — a highly practical paradigm for efficient multimodal training.
- **First systematic validation of the value of "long speech + vision"**: The paper demonstrates that long speech information can substantially complement visual understanding (VideoMME +6.5%), and that audio alone can address a large portion of video understanding tasks. This opens a new direction of "audio as a visual supplement."

## Limitations & Future Work
- Speech generation quality depends on CTC and a vocoder, precluding the generation of emotionally expressive speech.
- The sound modality relies on ImageBind's single-token encoding, which limits generalization.
- Long-speech inference still requires substantial GPU memory even with LMME.
- No direct comparison with closed-source models such as GPT-4o on speech interaction tasks is provided.

## Related Work & Insights
- **vs. VITA**: VITA requires 66B parameters and 5M data and supports only 1 minute of speech. Lyra-Base, with 9B parameters and 2.7M data, supports 2 hours and achieves stronger performance across multiple benchmarks. The key differentiators are LCMR and MLoRA.
- **vs. EMOVA**: EMOVA focuses on expressive speech but does not support long speech or vision-speech interaction. Lyra offers broader coverage.
- **vs. Intern-Omni**: Intern-Omni requires 27M training samples, whereas Lyra requires only 2.7M. Lyra outperforms Intern-Omni by approximately 9% on vision-speech tasks (DocVQAS: 85.5 vs. 79.9).

## Rating
- **Novelty**: ⭐⭐⭐⭐ — DTW-based cross-modal regularization combined with multi-modality LoRA is novel; the long-speech SFT dataset fills an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across vision-language, vision-speech, and speech-language tasks; three model scales; long-speech Needle-in-a-Haystack tests; extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure; the speech-centric evaluation perspective offers genuine insight.
- **Value**: ⭐⭐⭐⭐⭐ — Outperforms a competitor trained on 27M samples using only 2.7M data; first to support 2-hour long speech; significant implications for the omni-modal research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniRet: Efficient and High-Fidelity Omni Modality Retrieval](../../CVPR2026/audio_speech/omniret_efficient_and_high-fidelity_omni_modality_retrieval.md)
- [\[NeurIPS 2025\] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](../../NeurIPS2025/audio_speech/efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)
- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](../../NeurIPS2025/audio_speech/e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)
- [\[ICCV 2025\] Understanding Co-speech Gestures in-the-wild](understanding_co-speech_gestures_in-the-wild.md)
- [\[ACL 2026\] Speculative End-Turn Detector for Efficient Speech Chatbot Assistant](../../ACL2026/audio_speech/speculative_end-turn_detector_for_efficient_speech_chatbot_assistant.md)

</div>

<!-- RELATED:END -->
