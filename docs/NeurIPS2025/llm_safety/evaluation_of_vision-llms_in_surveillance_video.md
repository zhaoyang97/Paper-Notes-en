---
title: >-
  [Paper Note] Evaluation of Vision-LLMs in Surveillance Video
description: >-
  [NeurIPS 2025][LLM Safety][Vision-LLM] This paper proposes a training-free two-stage framework that leverages small Vision-LLMs to generate textual descriptions of video content…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Vision-LLM"
  - "Zero-shot Anomaly Detection"
  - "Surveillance Video"
  - "Privacy Protection"
  - "Natural Language Inference"
date: 2026-05-08
content_hash: 498529d159f0e4ea
---

# Evaluation of Vision-LLMs in Surveillance Video

**Conference**: NeurIPS 2025
**arXiv**: [2510.23190](https://arxiv.org/abs/2510.23190)  
**Code**: [GitHub](https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition)  
**Area**: 3D Vision
**Keywords**: Vision-LLM, Zero-shot Anomaly Detection, Surveillance Video, Privacy Protection, Natural Language Inference

## TL;DR

This paper proposes a training-free two-stage framework that leverages small Vision-LLMs to generate textual descriptions of video content, followed by an NLI classifier for zero-shot scoring. It systematically evaluates the impact of prompting strategies and privacy-preserving filters on anomalous behavior recognition in surveillance videos.

## Background & Motivation

**Surveillance data volumes far exceed human monitoring capacity**: The widespread deployment of cameras generates massive volumes of video data, making real-time manual monitoring impractical and necessitating automated anomaly detection.

**Traditional methods rely on large amounts of annotated data**: Supervised approaches (e.g., MIL, GNN) require fine-grained event boundary annotations, which are costly and difficult to generalize to novel anomaly categories.

**Existing anomaly detection datasets offer limited coverage**: Datasets such as UCF-Crime, XD-Violence, and RWF-2000 are constrained in scale and label diversity, limiting the generalization capability of models trained on them.

**Systematic evaluation of Vision-LLMs for anomaly recognition is lacking**: Although VLMs have demonstrated strong performance on conventional action recognition, their zero-shot capability on rare or criminal behaviors has not been systematically validated.

**Privacy protection is a hard constraint for real-world deployment**: Surveillance scenarios require video anonymization (e.g., blurring, GAN-based replacement), yet the impact of such operations on VLM performance remains unclear.

**Zero-shot flexibility has practical value**: If VLMs can recognize novel anomaly types via natural language prompts without retraining, system adaptability and scalability would be greatly enhanced.

## Method

### Overall Architecture

A two-stage pipeline: (1) a frozen Vision-LLM converts a sequence of video frames into natural language descriptions; (2) a frozen NLI classifier (BART-large-MNLI) scores the textual entailment between the description and each candidate label, assigning the highest-scoring label as the predicted category. No gradient updates are required throughout.

### Key Design 1: Video-to-Text Description Generation

- **Function**: Video frames are sampled and fed into the VLM, which generates concise textual descriptions of no more than 40 words.
- **Mechanism**: The pre-trained VLM's embedded world knowledge is leveraged for semantic reasoning, reformulating the pixel-to-label mapping as a language inference problem.
- **Design Motivation**: Through large-scale pre-training, VLMs have acquired rich vision-language alignment capabilities, enabling meaningful video descriptions without task-specific fine-tuning.

### Key Design 2: NLI-Based Zero-Shot Classification

- **Function**: The generated textual description serves as the premise, and each candidate anomaly label serves as a hypothesis; an NLI model computes the entailment score for each pair.
- **Mechanism**: Multi-class classification is reformulated as a textual entailment task, exploiting the semantic matching capability of pre-trained NLI models for zero-shot classification.
- **Design Motivation**: New anomaly categories can be incorporated simply by appending text labels to the candidate set, without modifying any model parameters, enabling genuine zero-shot flexibility.

### Key Design 3: Multi-Level Prompting Strategies

- **Function**: Three prompting schemes are designed — unguided prompting (free-form description), guided prompting (providing a list of candidate categories), and guided + few-shot prompting (additionally providing example images and descriptions).
- **Mechanism**: Structured prompts constrain the VLM's output space, directing it to generate descriptions more relevant to the classification task.
- **Design Motivation**: Open-ended descriptions may deviate from information critical to anomaly detection; guided prompts focus the model's attention on task-relevant semantics.

### Key Design 4: Privacy-Preserving Filter Evaluation

- **Function**: Three privacy protection schemes are tested on RWF-2000 — local head blurring, GAN-based face anonymization (DeepPrivacy2), and GAN-based full-body anonymization.
- **Mechanism**: Anonymized datasets are pre-generated and compared under identical evaluation conditions to assess the impact of different privacy filters on VLM anomaly detection performance.
- **Design Motivation**: Privacy protection is indispensable in real-world deployment; quantifying its specific cost to model performance provides empirical guidance for engineering decisions.

## Loss & Training

This method requires no training — both the VLM and the NLI classifier use frozen parameters. At inference, a conservative decoding strategy is adopted: temperature 0.05–0.1, maximum new tokens 64–128, and a repetition penalty of 1.5. Long videos are processed in temporal windows; a video is considered correctly predicted if any single window yields the correct prediction.

## Key Experimental Results

### Experiment 1: Effect of Prompting Strategies on UCF-Crime

| Model | Unguided Top-1 (%) | Guided Top-1 (%) | Guided + Few-shot Top-1 (%) |
|---|---|---|---|
| Gemma-3 (4B) | 26.29 | 33.85 | 29.80 |
| NVILA-8B | 13.39 | 27.00 | **45.05** |
| Qwen-2.5-VL-7B | 25.31 | 34.69 | — |
| VideoLLaMA-3-7B | 19.94 | 34.16 | — |

- Guided prompting consistently improves accuracy (+7–14 pp).
- Few-shot examples yield a substantial gain for NVILA (+18 pp) but lead to a decline for Gemma-3, and generally increase the false positive rate across models.

### Experiment 2: Effect of Privacy-Preserving Filters on RWF-2000

| Model | No Filter Acc/FP (%) | Blur ΔAcc/ΔFP | GAN Face ΔAcc/ΔFP | GAN Full-Body ΔAcc/ΔFP |
|---|---|---|---|---|
| Gemma-3 (4B) | 86.25/20.50 | –5.0/+10.5 | –2.8/+7.0 | –4.0/+7.0 |
| NVILA-8B | 82.50/14.00 | –1.8/+2.0 | –1.8/+5.0 | –11.3/+7.5 |
| Qwen-2.5-VL-7B | 82.25/24.50 | –4.8/+9.0 | –1.0/+2.0 | –6.5/+11.0 |
| VideoLLaMA-3-7B | 83.25/8.50 | –2.5/+2.0 | –4.5/–5.5 | –8.8/–6.5 |

- Privacy filters consistently reduce accuracy by 2–11 pp and increase false positive rates.
- GAN full-body anonymization has the largest impact, as temporally inconsistent appearances generated across frames distort motion cues.
- VideoLLaMA-3 exhibits a reduction in false positives under GAN filters, highlighting model-specific differences in sensitivity to privacy processing.

## Highlights & Insights

- **Fully training-free**: The entire pipeline requires no gradient updates, enabling genuine zero-shot detection — new anomaly types can be detected simply by adding new text labels.
- **Modular architecture**: The VLM and NLI classifier are decoupled and can be independently upgraded or replaced.
- **First systematic evaluation of privacy protection's impact on VLM-based anomaly detection**: Fills an important experimental gap in the field.
- **Rigorous experimental design**: Few-shot examples are drawn from the training set to prevent data leakage; single-variable comparisons are maintained with clear control conditions.

## Limitations & Future Work

- **Overall accuracy remains low**: The highest performance on UCF-Crime reaches only 45%, far from practical deployment requirements.
- **Only small models (≤8B) are evaluated**: Larger models (e.g., GPT-4V, Gemini) may perform substantially better but are not included.
- **Single-run experiments**: Due to computational constraints, each experiment is conducted only once, lacking statistical significance analysis.
- **Limited dataset coverage**: Only two datasets are used, excluding broader scenarios (e.g., the multimodal data in XD-Violence).
- **GAN temporal inconsistency**: The paper identifies but does not address the inter-frame inconsistency introduced by GAN full-body anonymization.

## Related Work & Insights

- **Supervised anomaly detection**: The MIL paradigm for UCF-Crime [Sultani et al. 2018], REWARD [Karim et al. 2024], AnomalyCLIP [Zanella et al. 2024a], MissionGNN [Yun et al. 2025].
- **VLMs for anomaly detection**: LAVAD [Zanella et al. 2024b] uses an LLM for temporal aggregation of caption-based anomaly scores; Holmes-VAD [Zhang et al. 2024] instruction-tunes a multimodal LLM; TEVAD [Chen et al. 2023] improves anomaly scoring via text.
- **VLM backbones**: Gemma-3 [Team et al. 2025], Qwen-2.5-VL [Bai et al. 2025], VideoLLaMA-3 [Zhang et al. 2025], NVILA [Liu et al. 2024].

## Rating

- Novelty: ⭐⭐⭐ — The framework concept (VLM + NLI) is not particularly novel, but the systematic evaluation of privacy protection offers meaningful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation across multiple models × prompting strategies × privacy filters, though statistical testing is absent.
- Writing Quality: ⭐⭐⭐⭐ — Well-organized and clearly presented, with complete formal derivations.
- Value: ⭐⭐⭐⭐ — Provides a practical baseline and engineering guidance for zero-shot surveillance anomaly detection under privacy constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VMDT: Decoding the Trustworthiness of Video Foundation Models](vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)
- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[NeurIPS 2025\] Music Arena: Live Evaluation for Text-to-Music](music_arena_live_evaluation_for_text-to-music.md)
- [\[NeurIPS 2025\] AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)

</div>

<!-- RELATED:END -->
