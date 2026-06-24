---
title: >-
  [Paper Note] SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods
description: >-
  [ACL 2025][AI Safety][speech deepfake detection] The SpeechFake dataset, a large-scale speech deepfake dataset, is constructed. It contains over 3 million deepfake samples, encompasses more than 3,000 hours of audio, covers 40 generation tools, and spans 46 languages. Through baseline experiments, the systematic impacts of generation methods, linguistic diversity, and speaker variations on detection performance are analyzed.
tags:
  - "ACL 2025"
  - "AI Safety"
  - "speech deepfake detection"
  - "dataset"
  - "multilingual"
  - "TTS"
  - "voice conversion"
  - "neural vocoder"
date: 2026-05-08
content_hash: a3efb1be0da0806e
---

# SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods

**Conference**: ACL 2025  
**arXiv**: [2507.21463](https://arxiv.org/abs/2507.21463)  
**Code**: [YMLLG/SpeechFake](https://github.com/YMLLG/SpeechFake)  
**Area**: AI Safety  
**Keywords**: speech deepfake detection, dataset, multilingual, TTS, voice conversion, neural vocoder  

## TL;DR

The SpeechFake dataset, a large-scale speech deepfake dataset, is constructed. It contains over 3 million deepfake samples, encompasses more than 3,000 hours of audio, covers 40 generation tools, and spans 46 languages. Through baseline experiments, the systematic impacts of generation methods, linguistic diversity, and speaker variations on detection performance are analyzed.

## Background & Motivation

- **Limitations of Prior Work**: Existing speech deepfake datasets suffer from severe shortcomings in scale and diversity—most public datasets are small, utilize outdated or limited generation technologies, and primarily focus on English or Chinese.
- **Generalization Bottleneck**: Detection models suffer a sharp drop in performance when encountering unseen deepfake techniques. Simply merging existing datasets introduces mismatches in conditions and increases training complexity.
- **Lack of Cutting-Edge Inclusion**: While numerous advanced speech generation technologies (such as CosyVoice, ChatTTS, and GPT-SoVITS) have emerged recently, they are not incorporated into existing datasets.
- **Ours**: This work constructs the SpeechFake dataset, which is split into a Bilingual Dataset (BD: English/Chinese) and a Multilingual Dataset (MD: 46 languages). It leverages 30 open-source tools and 10 commercial APIs to generate deepfake audio, comprehensively covering three generation methods: TTS, VC, and NV.

## Method

### Overall Architecture

The dataset construction pipeline involves: (1) **Real audio collection**: Sourcing real speech from LibriTTS, VCTK, AISHELL1, AISHELL3, and CommonVoice; (2) **Deepfake generation**: Generating audio using 40 different tools across three methodology categories: TTS (Text-to-Speech), VC (Voice Conversion), and NV (Neural Vocoder); (3) **Post-processing**: Applying VAD to filter out clips shorter than 0.5 seconds, performing selective manual inspection, and standardizing audio to 16kHz mono WAV format.

### Key Designs

- **Bilingual & Multilingual Split**: BD focuses on bilingual English and Chinese (using all 40 tools), while MD covers 46 languages (using 6 multilingual tools). The training set only contains English/Chinese, whereas the test set expands to 46 languages to evaluate cross-lingual generalization.
- **Inclusion of Cutting-Edge Methods**: The dataset incorporates the latest speech generation technologies released over the past year (e.g., CosyVoice, ChatTTS, GPT-SoVITS), which are capable of generating highly realistic synthetic speech.
- **Rich Metadata**: Annotated with generation methods, speaker IDs, languages, text transcriptions, and more, enabling in-depth research beyond binary classification.

### Evaluation Metrics

- Consistent with prior work, **Equal Error Rate (EER)** is used as the primary evaluation metric.

## Key Experimental Results

### Main Results (EER%, lower is better)

| Training Data | Model | BD | BD-EN | BD-CN | ASV19 | WF | ITW | CDADD |
|----------|------|-----|-------|-------|-------|-----|-----|-------|
| ASV19 | AASIST | 39.36 | 41.05 | 39.07 | **1.88** | 21.17 | 45.27 | 49.53 |
| BD | AASIST | **3.48** | **3.98** | **2.68** | 23.62 | **4.30** | **7.53** | **22.52** |
| ASV19 | W2V+AASIST | 23.78 | 20.15 | 24.93 | **0.89** | 3.48 | 10.07 | 8.55 |
| BD | W2V+AASIST | **3.54** | **3.55** | **2.83** | 2.91 | **0.58** | **2.01** | **2.42** |

### Ablation Study

| Analytical Dimension | Key Findings |
|----------|----------|
| Cross-Generator Generalization | TTS training data achieves the best generalization performance (BD overall EER of 14.26% with AASIST), while NV performs the worst (26.30%); a significant generalization gap remains between different generation methods. |
| Cross-Lingual Generalization | AASIST EER increases significantly on unseen languages (e.g., 22.54% for French, 26.06% for Hindi), whereas W2V+AASIST achieves EER <1% across all languages after 50 epochs due to multilingual pre-training. |
| Impact of Cross-Speaker | Speaker variations do affect detection, but the speaker diversity in training data can effectively mitigate this effect. |
| BD-EN vs BD-CN | Both sub-datasets suffer from performance drops when evaluated on the counterpart test set; training with the full BD yields the optimal performance. |

### Key Findings

1. Models trained on SpeechFake generalize to external benchmarks far better than those trained on ASVspoof2019 (e.g., EER on ITW drops from 45.27% to 7.53%).
2. The generation method is the primary factor affecting generalization—models trained on TTS data also perform well on unseen commercial TTS APIs (BD-UT EER of 0.53% with AASIST).
3. Even when controlling for generation methods, language still affects detection performance, but this can be greatly mitigated by multilingual pre-trained feature extractors (e.g., Wav2Vec2.0 XLSR).
4. Dataset scale and diversity are key to improving generalization capability—simply increasing homogeneous data is less effective than expanding the diversity of generation methods and languages.

## Highlights & Insights

- Unprecedented Scale: Includes over 3 million deepfake samples, 3,000+ hours of audio, 40 generation tools, and 46 languages.
- Systematic Comparative Design: Separately analyzes the effects of various factors across three generation methods and two dimensions (bilingual/multilingual).
- Inclusion of cutting-edge generation technologies, ensuring the benchmark is forward-looking.
- Provision of rich metadata (method type, speaker, language, transcripts) to support multi-angle research.

## Limitations & Future Work

- The training set for the multilingual dataset only includes English and Chinese, while other languages only appear in the test set, which may underestimate the potential of cross-lingual fine-tuning.
- Quality filtering was only applied as a spot check on approximately 1% of the samples, which may lead to the omission of some low-quality deepfakes.
- The amount of data generated by some of the 40 tools varies significantly due to copyright or technical constraints, potentially leading to an unbalanced distribution.
- It does not cover scenarios like adversarial attacks (e.g., Malafide) and codec distortion.

## Related Work & Insights

- **Speech Deepfake Datasets**: The ASVspoof series (2015-2024), WaveFake, In-the-Wild, MLAAD (23 languages), SpoofCeleb (2.68 million samples), etc.
- **Speech Generation Tech**: Technological evolution progressing from CNN/RNN → Transformer → GAN/Flow/Diffusion → LLM-based TTS.
- **Detection Methods**: Front-end/back-end detection architectures such as AASIST (Heterogeneous Graph Attention Network) and W2V+AASIST (Wav2Vec2.0 + AASIST).

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 3 |
| Value | 5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Overall Rating| 4.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Fairness Assessment of Dutch Hate Speech Detection](towards_fairness_assessment_of_dutch_hate_speech_detection.md)
- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](../../ICML2026/ai_safety/medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[CVPR 2026\] FVBench: Benchmarking Deepfake Video Detection Capability of Large Multimodal Models](../../CVPR2026/ai_safety/fvbench_benchmarking_deepfake_video_detection_capability_of_large_multimodal_mod.md)
- [\[CVPR 2026\] GenBreak: Red Teaming Text-to-Image Generation Using Large Language Models](../../CVPR2026/ai_safety/genbreak_red_teaming_text-to-image_generation_using_large_language_models.md)
- [\[CVPR 2026\] Unlearning without Forgetting: Securely Removing Targeted Concepts from Large-Scale Vision-Language Open-Vocabulary Detectors](../../CVPR2026/ai_safety/unlearning_without_forgetting_securely_removing_targeted_concepts_from_large-sca.md)

</div>

<!-- RELATED:END -->
