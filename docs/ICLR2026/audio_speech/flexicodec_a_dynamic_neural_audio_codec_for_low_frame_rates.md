---
title: >-
  [Paper Note] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates
description: >-
  [ICLR 2026][Audio & Speech][Neural Audio Codec] FlexiCodec is proposed as a dynamic frame rate merging strategy guided by ASR features…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "Neural Audio Codec"
  - "Dynamic Frame Rate"
  - "Low Frame Rate"
  - "Speech Tokenization"
  - "TTS"
date: 2026-05-08
content_hash: d30d4aca3a828154
---

# FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates

**Conference**: ICLR 2026
**arXiv**: [2510.00981](https://arxiv.org/abs/2510.00981)
**Code**: [amphionteam/flexicodec](https://github.com/amphionteam/flexicodec)
**Area**: Audio & Speech
**Keywords**: Neural Audio Codec, Dynamic Frame Rate, Low Frame Rate, Speech Tokenization, TTS

## TL;DR

FlexiCodec is proposed as a dynamic frame rate merging strategy guided by ASR features, achieving high-quality speech codec at ultra-low frame rates of 3–12.5 Hz while maintaining superior semantic information retention.

## Background & Motivation

**Background**: Neural audio codecs (e.g., EnCodec, DAC, SpeechTokenizer) serve as foundational components for speech language models, compressing speech into discrete tokens compatible with the AR LLM paradigm. However, mainstream codecs operate at frame rates ≥50 Hz, requiring 50+ tokens per second of speech—a severe mismatch with the ~4.5 Hz frame rate of text.

**Limitations of Prior Work**: High frame rates introduce two problems: (1) the quadratic complexity of attention leads to substantial computational overhead; (2) the text-speech modality frame rate mismatch degrades LLM performance. Although Mimi and DualCodec reduce the frame rate to 12.5 Hz, a significant gap from text's 4.5 Hz remains, and research on codecs below 12.5 Hz is nearly absent.

**Key Challenge**: Directly pushing existing codecs to ultra-low frame rates (<12.5 Hz) results in severe semantic information loss. Experiments show that DualCodec's RVQ-1 WER surges from 5.93% to 31.5% when the frame rate is reduced from 12.5 Hz to 6.25 Hz. The underlying causes are: (a) insufficient disentanglement of semantic and acoustic information, forcing a trade-off between the two under the limited information capacity at low frame rates; and (b) fixed-rate downsampling discards transient speech details, while natural speech phonemes and syllables inherently occur at dynamic rates.

**Goal**: To achieve both semantic integrity and high audio reconstruction quality at frame rates as low as 6.25 Hz or below, while supporting controllable frame rates at inference time.

**Key Insight**: (a) Dynamic frame rates—allocating more frames to information-dense regions and merging frames in sparse regions (silence, long vowels); (b) replacing SSL features with ASR features to provide more concentrated semantic information; (c) a single model supporting continuously controllable frame rates from 3 to 12.5 Hz.

**Core Idea**: Leveraging cosine similarity between pretrained ASR feature frames to dynamically merge semantically similar frames, enabling content-adaptive low frame rate speech encoding.

## Method

### Overall Architecture

FlexiCodec follows a pipeline of **dual-stream encoding → dynamic frame merging → quantization → frame unmerging → decoding**:

- **Semantic stream**: A frozen ASR encoder (SenseVoice-Small, 230M) extracts 12.5 Hz semantic features $e_s \in \mathbb{R}^{T \times d}$
- **Acoustic stream**: A CNN encoder (5 layers, stride=[4,4,5,8,2]) downsamples 16 kHz waveforms to 12.5 Hz acoustic features $e_a \in \mathbb{R}^{T \times d}$
- Both feature streams are independently compressed by a Frame Merging Module
- The semantic stream is quantized via FSQ to obtain RVQ-1 tokens; the residual between the acoustic and semantic streams is quantized via RVQ to obtain RVQ-rest tokens
- A Frame Unmerging Module restores the dynamic frame rate sequence to a fixed 12.5 Hz rate
- A CNN decoder synthesizes the waveform

### Key Designs

**Design 1: ASR Feature-Guided Dual-Stream Encoding**

- **Function**: The final hidden layer of the frozen SenseVoice-Small ASR model serves as semantic features, replacing conventional SSL features (e.g., HuBERT, WavLM).
- **Mechanism**: ASR models trained with CTC loss target text prediction, yielding features with inherently higher semantic concentration; SSL models trained with reconstruction objectives produce features where semantic and acoustic information are entangled and redundant.
- **Design Motivation**: At ultra-low frame rates, information capacity is extremely limited, making it essential for RVQ-1 to encode as pure semantic information as possible. Experiments confirm that simply replacing DualCodec's SSL features with ASR features reduces the 6.25 Hz RVQ-1 WER from 31.5% to 6.0%—a dramatic improvement.

**Design 2: Dynamic Frame Merging**

- **Function**: Computes cosine similarity between adjacent ASR feature frames and merges consecutive similar frames into a single frame, adaptively reducing the frame rate.
- **Mechanism**: Adjacent frame similarity is defined as $s_t = \cos(e_s[t], e_s[t+1])$. A left-to-right scan merges consecutive segments $[i, j]$ satisfying $\min_{t=i}^{j-1} s_t \geq \tau$ into a single frame (by averaging), while recording the frame length $\ell_k = j - i + 1$. Merged frames are subsequently refined by a local windowed attention Transformer to avoid unnatural transitions between frames.
- **Design Motivation**: Silence and long vowel regions in natural speech exhibit low information density, wasting capacity under fixed frame rates; rapid articulation regions have high information density and require more frames. Experiments reveal a strong positive correlation between frame rate and phoneme rate (Pearson $r = 0.775$), with a linear coefficient of approximately 0.5, indicating that each merged frame encodes roughly two phonemes.

**Design 3: Inference-Time Controllable Frame Rate**

- **Function**: During training, the merging threshold $\tau \in [0.7, 1.0]$ is randomly sampled; at inference, adjusting $\tau$ controls the output frame rate.
- **Mechanism**: At $\tau = 1.0$, no merging occurs and the output is 12.5 Hz; lower $\tau$ values produce more aggressive merging and lower frame rates. A single model supports arbitrary frame rates from 3 to 12.5 Hz.
- **Design Motivation**: Different downstream tasks impose different requirements on efficiency and quality. Controllable frame rates enable flexible trade-offs—edge-device TTS can use lower frame rates for speed, while quality-sensitive scenarios can use higher ones. This flexibility is unavailable in conventional fixed-rate codecs.

**Design 4: FSQ Semantic Quantization + RVQ Acoustic Quantization**

- **Function**: The semantic stream is quantized into RVQ-1 tokens using a Finite Scalar Quantizer (FSQ) ($D=5, L=8$, yielding $8^5 = 32{,}768$ codebook entries); the acoustic residual is quantized using 24-layer RVQ (4,096 entries per layer).
- **Mechanism**: FSQ projects features into a low-dimensional space and independently rounds each dimension, requiring no codebook learning and avoiding the codebook collapse problem of VQ. The acoustic residual retains RVQ's multi-layer progressive refinement capability.
- **Design Motivation**: Semantic tokens require large codebooks to distinguish fine-grained semantics; FSQ naturally supports large codebooks through multiplicative combination. The acoustic stream has complex information distributions better suited to RVQ's residual layer-by-layer encoding.

### Loss & Training

The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{GAN}} \mathcal{L}_{\text{GAN}} + \lambda_{\text{RVQ}} \mathcal{L}_{\text{RVQ}} + \lambda_{\text{feat}} \mathcal{L}_{\text{feat}}$$

- $\mathcal{L}_{\text{recon}}$: Multi-scale L1 Mel spectrogram reconstruction loss
- $\mathcal{L}_{\text{GAN}}$: Adversarial and feature matching losses from MPD + MRSD discriminators
- $\mathcal{L}_{\text{RVQ}}$: RVQ codebook update loss + commitment loss (FSQ requires no additional loss)
- $\mathcal{L}_{\text{feat}}$: L2 alignment loss between RVQ-1 semantic token embeddings and unquantized semantic features

Training details:
- Data: Librilight-Large, 54k hours, 16 kHz
- 800k steps, 8×V100 32 GB, batch = 5×5 s/GPU
- Quantizer dropout: randomly select $n \in [1, N]$ RVQ layers for decoding; $n=1$ uses only the semantic stream
- $\tau \in [0.7, 1.0]$ sampled randomly at each step to ensure the model adapts across all frame rates
- Maximum merged frame length $\ell_k = 8$; local attention window ±8

## Key Experimental Results

### Main Results: Comprehensive Comparison with Open-Source Codecs (Table 5)

| System | Frame Rate (Hz) | Bitrate (kbps) | WER(RVQ1)↓ | WER(RVQ1:8)↓ | PESQ↑ | UTMOS↑ | MCD↓ | SIM↑ |
|--------|----------------|----------------|------------|--------------|-------|--------|------|------|
| DAC | 75 | 6.0/8q | 31.2 | 2.27 | 3.77 | 3.62 | 2.34 | 0.90 |
| EnCodec | 75 | 6.0/8q | 5.90 | 2.24 | 3.12 | 3.01 | 2.60 | 0.89 |
| SpeechTokenizer | 50 | 4.0/8q | 5.56 | 2.47 | 3.01 | 3.90 | 3.17 | 0.85 |
| DualCodec | 12.5 | 1.2/8q | 5.93 | 2.26 | 3.29 | 4.18 | 2.81 | 0.85 |
| **FlexiCodec @12.5Hz** | 12.5 | 1.3/8q | **2.76** | **2.23** | **3.35** | **4.22** | **2.76** | 0.85 |
| WavTokenizer | 75 | 0.90/1q | 4.57 | 4.57 | 2.86 | 3.98 | 3.51 | 0.68 |
| XCodec2 | 50 | 0.80/1q | 2.80 | 2.80 | 2.77 | 4.08 | 3.65 | 0.82 |
| **FlexiCodec @8.3Hz** | 8.3 | 0.85/8q | **2.98** | **2.28** | **3.03** | **4.21** | **3.10** | 0.78 |
| TaDiCodec | 6.25 | 0.15/1q | 4.32 | 4.32 | 1.73 | 4.05 | 9.75 | 0.83 |
| SemantiCodec | 25 | 0.34/1q | 23.8 | 23.8 | 1.89 | 2.93 | 5.92 | 0.40 |
| **FlexiCodec @6.25Hz** | 6.25 | 0.64/8q | **4.15** | **2.53** | **2.76** | **4.18** | **3.42** | 0.71 |

Ground truth WER = 2.1%. FlexiCodec achieves state-of-the-art semantic retention and audio quality across all bitrate ranges.

### Ablation Study: Contribution of Dynamic Frame Rate (Tables 3 & 4)

**Semantic ablation:**

| Configuration | WER(RVQ1)↓ | Δ | WER(RVQ1:8)↓ | Δ | ASR Probing WER↓ | Δ |
|---------------|------------|---|--------------|---|-----------------|---|
| FlexiCodec @8.3Hz | 2.98 | — | 2.28 | — | 13.0 | — |
| → w/o dynamic frame rate (FFR) | 3.56 | +19% | 2.43 | +6% | 14.5 | +12% |
| FlexiCodec @6.25Hz | 4.15 | — | 2.53 | — | 15.6 | — |
| → w/o dynamic frame rate (FFR) | 5.22 | +26% | 2.73 | +8% | 18.8 | +21% |

**Acoustic ablation:**

| Configuration | PESQ↑ | MCD↓ | UTMOS↑ | SIM↑ |
|---------------|-------|------|--------|------|
| FlexiCodec @8.3Hz | 3.03 | 3.10 | 4.21 | 0.78 |
| → w/o dynamic frame rate | 3.03 | 3.18 | 4.21 | 0.76 |
| FlexiCodec @6.25Hz | 2.76 | 3.42 | 4.18 | 0.71 |
| → w/o dynamic frame rate | 2.76 | 3.47 | 4.18 | 0.70 |

### Key Findings

- **Frame rate and phoneme rate exhibit a strong positive correlation** (Pearson $r = 0.775$), confirming that dynamic frame rates adaptively allocate frames according to speech content complexity—rapid articulation regions receive more frames, while silence and long vowels are merged.
- **The gains from dynamic frame rates become more pronounced at lower frame rates**: removing dynamic frame rates degrades RVQ-1 WER by 26% at 6.25 Hz versus 19% at 8.3 Hz, indicating that adaptive allocation becomes increasingly critical as frame rates decrease.
- **Dynamic frame rates primarily improve semantic retention with minimal impact on acoustic metrics**: PESQ and UTMOS remain nearly unchanged, while MCD and SIM show marginal improvements. This reflects a fundamental misalignment between acoustic and semantic information density.
- **Replacing SSL features with ASR features alone yields substantial gains**: switching from SSL to ASR features within the DualCodec architecture reduces RVQ-1 WER at 6.25 Hz from 31.5% to 6.0%.
- **$\tau$ controls the trade-off between frame rate and quality**: at $\tau = 0.7$, the mean frame rate is 3.0 Hz (RVQ-1 WER 51.5%); at $\tau = 0.8$, 4.5 Hz (WER 14.4%); at $\tau = 0.9$, 7.9 Hz (WER 3.13%).
- **Encoding and decoding are highly efficient**: RTF is only 0.018 (encoding) and 0.006 (decoding), applicable across all frame rates.
- **Downstream TTS**: FlexiCodec-TTS achieves competitive performance across multiple frame rates while being significantly faster than high frame rate baselines.

## Highlights & Insights

1. **Insightful problem diagnosis**: The paper precisely identifies two root causes of semantic loss in low frame rate codecs—insufficient semantic disentanglement and fixed-rate downsampling discarding transient details—and validates both hypotheses with extensive experiments.
2. **Elegant dynamic frame rate design**: Using cosine similarity of ASR features for frame merging requires no additional trainable parameters, is deterministic and reproducible, and naturally supports controllable frame rates—a single parameter $\tau$ enables continuous adjustment from 3 to 12.5 Hz.
3. **Dual reuse of ASR features**: The same ASR features are used both for semantic encoding (providing RVQ-1 input) and for guiding merge boundaries (computing similarity), yielding a concise and efficient design.
4. **Exceptionally comprehensive experiments**: Codec reconstruction, semantic retention, ASR probing, downstream TTS, audio understanding, cross-lingual generalization, ablation, and efficiency analysis are all covered with virtually no blind spots.
5. **The finding that each merged frame encodes approximately 2 phonemes** provides a quantitative reference for the information-theoretic understanding of low frame rate codecs.

## Limitations & Future Work

1. **Rapid semantic degradation at ultra-low frame rates (<4 Hz)**: At $\tau = 0.7$, the 3 Hz frame rate yields a WER of 51.5%, indicating that the current approach still faces bottlenecks under extreme compression.
2. **Acoustic quality constrained by bitrate**: Dynamic frame rates primarily improve semantics, with limited gains in acoustic metrics; the fundamental misalignment between acoustic and semantic information density is the underlying cause.
3. **Poor zero-shot semantic performance across languages**: Models trained on English perform poorly on unseen languages in terms of semantic tokens, requiring fine-tuning.
4. **Frame length metadata requires an additional 3 bits/frame for transmission**: Although the overhead is modest, it adds complexity on the decoding side.
5. **RVQ-rest does not use FSQ**: The authors acknowledge that multi-level FSQ (e.g., rFSQ) could potentially further improve acoustic quantization quality.
6. **AR models generating directly from dynamic frame rate tokens are unexplored**: The current approach still requires Frame Unmerging to restore a fixed frame rate before decoding, limiting the end-to-end advantages of dynamic frame rates.

## Related Work & Insights

- **Relationship to DualCodec**: FlexiCodec inherits its dual-stream disentanglement approach but replaces SSL features with ASR features and introduces dynamic frame merging, achieving substantially improved semantic retention.
- **Relationship to Token Merging (ToMe)**: The approach draws inspiration from DynTok in the vision domain—using pretrained feature similarity to guide token merging—adapted to the one-dimensional temporal domain of speech signals.
- **Differences from TaDiCodec**: TaDiCodec also targets 6.25 Hz but requires text transcriptions to assist synthesis, resembling a TTS system; FlexiCodec follows the conventional codec paradigm without relying on transcriptions.
- **Implications for speech LLMs**: By reducing speech token frame rates to near text frame rates (4.5 Hz), FlexiCodec can substantially shorten speech token sequences in multimodal LLMs, reducing computational costs.
- **Implications for adaptive streaming**: The controllable frame rate property suits adaptive bitrate transmission scenarios—higher frame rates when network bandwidth is sufficient, lower rates when it is not.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dynamic frame rate codec concept is novel and the dual reuse of ASR features is clever; however, individual components (Token Merging, FSQ, dual-stream architecture) each have prior work, making the overall contribution a sophisticated combinatorial innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Exceptionally thorough, covering codec reconstruction, semantics, acoustics, ablation, downstream TTS, audio understanding, cross-lingual generalization, and efficiency analysis with multi-rate and multi-baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logically clear, with a seamless flow from motivation to method to experiments; figures and tables are information-dense and readable, and the related work section is comprehensively categorized.
- **Value**: ⭐⭐⭐⭐ — Provides practical infrastructure for low frame rate speech codecs and speech LLMs; open-source code enhances practical value; quality degradation at ultra-low frame rates limits applicability in aggressive compression scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](toward_complex-valued_neural_networks_for_waveform_generation.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)
- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](../../AAAI2026/audio_speech/say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)

</div>

<!-- RELATED:END -->
