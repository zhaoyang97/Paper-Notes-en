---
title: >-
  [Paper Note] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images
description: >-
  [NeurIPS 2025][Multimodal VLM][ECG understanding] GEM is proposed as the first multimodal large language model that unifies ECG time series, 12-lead ECG images, and text. Through a dual-encoder framework…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "ECG understanding"
  - "multimodal large language model"
  - "time series signal"
  - "grounded diagnosis"
  - "electrocardiogram"
date: 2026-05-08
content_hash: 98f888c91fc3ca57
---

# GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images

**Conference**: NeurIPS 2025  
**arXiv**: [2503.06073](https://arxiv.org/abs/2503.06073)  
**Code**: [https://github.com/lanxiang1017/GEM](https://github.com/lanxiang1017/GEM)  
**Area**: Multimodal VLM  
**Keywords**: ECG understanding, multimodal large language model, time series signal, grounded diagnosis, electrocardiogram

## TL;DR
GEM is proposed as the first multimodal large language model that unifies ECG time series, 12-lead ECG images, and text. Through a dual-encoder framework, cross-modal alignment, and knowledge-guided instruction data generation, GEM achieves grounded ECG diagnosis based on quantifiable physiological features, improving diagnostic accuracy by 7.4%, interpretability by 22.7%, and grounding capability by 25.3%.

## Background & Motivation

**Background**: Deep learning has achieved progress in tasks such as arrhythmia detection, but lacks linguistic capability and interpretability. Recent MLLMs (e.g., PULSE) have enabled language-based ECG interpretation via large-scale instruction tuning, but primarily handle static image inputs and predefined diagnostic tasks.

**Limitations of Prior Work**: (1) **Insufficient modality synergy** — existing models process only a single non-textual modality (either time series or images), failing to simultaneously analyze dynamic signal trends and spatial waveform patterns as clinicians do; (2) **Insufficient interpretability and grounding** — existing models do not explicitly link diagnostic conclusions to specific waveform evidence (e.g., quantifiable parameters such as QRS duration and PR interval).

**Key Challenge**: How to enable a model to work like a cardiologist — simultaneously analyzing time series and image signals, and providing specific electrophysiological feature evidence for each diagnostic conclusion.

**Goal**: (1) Build a unified multimodal ECG model integrating time series, images, and text; (2) Achieve beat-level physiological feature grounded diagnosis; (3) Construct high-granularity grounded training data.

**Key Insight**: Leveraging an existing feature extraction tool (FeatureDB) to extract beat-level physiological feature sequences from raw ECG signals, and designing a diagnosis guider to activate the latent medical knowledge of GPT-4o for automatic generation of high-granularity grounded instruction data.

**Core Idea**: Dual-encoder extraction of complementary features + cross-modal alignment + knowledge-guided data generation, enabling the MLLM to provide evidence-based ECG diagnosis in the manner of a cardiologist.

## Method

### Overall Architecture
GEM comprises three major components: (1) Multimodal encoding — a time series encoder (ECG-CoCa) and an image encoder (CLIP) extract features independently; (2) Cross-modal alignment — time series representations are first projected to the image dimension and then jointly mapped to the text space; (3) Knowledge-guided instruction data generation — FeatureDB extracts physiological features, a diagnosis guider constructs prompts, and GPT-4o generates high-granularity target answers.

### Key Designs

1. **Dual-Encoder Multimodal Encoding**:

    - Function: Extracts complementary features from ECG time series and 12-lead images respectively.
    - Mechanism: The time series encoder $\mathbf{e}_{ts} \in \mathbb{R}^{n_s \times d_s} = E_{ts}(\bm{x}_{ts})$ employs a pretrained ECG-CoCa (trained via contrastive learning on large-scale ECG-text pairs); the image encoder $\mathbf{e}_{img} \in \mathbb{R}^{n_m \times d_m} = E_{img}(\bm{x}_{img})$ uses the pretrained CLIP from LLaVA.
    - Design Motivation: Time series models capture dynamic variations but may overlook spatial patterns, while image models detect global structures but may miss subtle temporal details — the two modalities are complementary.

2. **Cross-Modal Alignment Learning**:

    - Function: Unifies heterogeneous modal representations into a text space interpretable by the LLM.
    - Mechanism: An MLP first projects time series representations to the image dimension $\hat{\mathbf{e}}_{ts} \in \mathbb{R}^{n_s \times d_m} = MLP_{ts}(\mathbf{e}_{ts})$; a shared MLP then maps both representations to the text dimension $\mathbf{h}_{ts} \in \mathbb{R}^{n_s \times d_t} = MLP(\hat{\mathbf{e}}_{ts})$, $\mathbf{h}_{img} \in \mathbb{R}^{n_m \times d_t} = MLP(\mathbf{e}_{img})$; finally, these are concatenated with text embeddings as $\mathbf{x} = \text{Concatenate}(\mathbf{h}_{ts}, \mathbf{h}_{img}, \text{Embed}(\bm{x}_q))$.
    - Design Motivation: Two-step alignment (time series → image dimension → text dimension) is more stable than direct projection; the shared projector ensures both modalities are comparable within the same text space.

3. **Knowledge-Guided Instruction Data Generation**:

    - Function: Automatically constructs high-granularity ECG grounded instruction data without manual annotation.
    - Mechanism: (a) **Grounding Feature Extractor** — extracts 14 types of feature sequences per heartbeat (heart rate, RR interval, P-wave amplitude/duration, PR interval, QRS amplitude/duration, T-wave amplitude/duration, ST duration/morphology, QT/QTc interval) across 12 leads from raw ECG time series, $\bm{x}_{fs} = \text{FeatureDB}(\bm{x}_{ts})$; (b) **Diagnosis Guider** — constructs a structured prompt $\bm{x}_p = \text{DiagnosisGuider}(\bm{x}_{fs})$ incorporating cardiological diagnostic instructions, customized to each sample's specific features; (c) **GPT-4o Generation** — $\bm{y} = \text{GPT-4o}(\bm{x}_p)$, producing 30,000 ECG-Grounding data instances.
    - Design Motivation: The ECG-Instruct data in PULSE is derived from raw reports and occasionally produces erroneous interpretations due to hallucination; ECG-Grounding is grounded in actually extracted physiological features, ensuring diagnostically evidenced outputs.

### Loss & Training
Single-step training (unlike most MLLMs that pre-train the projector before fine-tuning the LLM): $\theta_{ts}$ and $\theta_{img}$ are frozen; $\theta_{M_{ts}}$, $\theta_M$, and $\theta_{LLM}$ are trained jointly. The loss is standard NLL: $L = -\sum_{i=1}^N \log P(y_j | \mathbf{x}, \theta_{LLM})$. SFT is conducted for 1 epoch on 8 × A100.

## Key Experimental Results

### Main Results (Grounded ECG Understanding)

| Metric | PULSE | GEM (SFT LLaVA) | GEM (SFT PULSE) |
|------|-------|-----------------|-----------------|
| Diagnostic Accuracy (MIMIC) | 81.14% | **87.24%** | 86.49% |
| Diagnostic Accuracy (PTB-XL) | 59.24% | 73.53% | **73.59%** |
| Lead Coverage Rate (MIMIC) | 7.11% | **71.07%** | 69.80% |
| Lead Accuracy (MIMIC) | 2.95% | **46.44%** | 45.33% |
| ECG Feature Grounding (MIMIC) | 50.18% | **75.48%** | 74.95% |
| Evidence-Based Reasoning (MIMIC) | 52.40% | **75.09%** | 74.70% |

### ECG-Bench Anomaly Detection

| Dataset | Metric | PULSE | GEM (SFT LLaVA) | GEM (SFT PULSE) |
|--------|------|-------|-----------------|-----------------|
| CSN | ACC | 85.2% | **92.6%** | 86.2% |
| G12EC | ACC | 78.2% | **81.8%** | 80.5% |
| PTB-XL | AUC | 82.4% | 81.8 | **83.4%** |
| CODE-15% | AUC | 90.7% | 90.5 | **91.5%** |
| PTB-XL Report | Score | 61.3 | 65.0 | **67.1** |

### Ablation Study

| Configuration | CSN ACC | CODE-15% AUC | Note |
|------|---------|-------------|------|
| TS only | 91.6% | 90.8% | Time series only |
| TS + IMG | 90.1% | 91.3% | Time series + image |
| GEM (SFT PULSE) | 86.2% | 91.5% | Full framework |

### Key Findings
- GEM improves diagnostic accuracy by 6–14 pp, lead coverage rate from 7% to 71%, and grounding capability from 50% to 75% — representing substantial gains.
- GEM (SFT LLaVA), which had never been trained on ECG data, surpasses PULSE on CSN by 7.4% after only 1 epoch of fine-tuning, demonstrating the efficient learning capacity of the GEM framework.
- Cross-domain generalization is notable: diagnostic accuracy on out-of-domain PTB-XL improves from 59.24% to 73.59%.
- Evaluation by 8 cardiologists confirms the clinical reliability and utility of GEM outputs; in some cases, GEM identified details overlooked by experts upon initial examination.
- Both GPT-4o and Deepseek-R1 can generate high-quality target data using the knowledge-guided approach, demonstrating that the method is not dependent on a specific LLM.

## Highlights & Insights
- **First tri-modal ECG understanding model**: Unifying time series, images, and text faithfully mirrors the clinical diagnostic workflow, with each modality contributing irreplaceable complementary information.
- **Knowledge-guided data generation**: The combination of FeatureDB (zero-parameter feature extraction) + diagnosis guider + GPT-4o avoids costly expert annotation while ensuring generated data is grounded in actual physiological features rather than hallucinations — a data generation paradigm transferable to other medical domains.
- **Single-step training strategy**: More effective than conventional multi-step training under limited data conditions (only 30K grounding samples), reducing inconsistency between alignment and fine-tuning phases.
- The grounded diagnosis paradigm is transferable to other medical time series data (e.g., EEG, EMG).

## Limitations & Future Work
- Target answers generated by GPT-4o occasionally disagree with cardiologist judgment (e.g., assessment of ischemia); alignment with human feedback could further improve performance.
- Training data scale is relatively small (30K ECG-Grounding + 1.1M ECG-Instruct); expanding grounding data volume may yield further gains.
- The accuracy of FeatureDB itself constitutes an upper bound — erroneous feature extraction propagates to downstream diagnosis.
- Engineering challenges such as latency and reliability for real-time clinical deployment are not discussed.
- In the ablation study, TS+IMG underperforms TS only on CSN (90.1% vs. 91.6%), indicating that multimodal fusion does not consistently outperform unimodal approaches and warrants deeper analysis.

## Related Work & Insights
- **vs. PULSE**: PULSE is the current SOTA ECG-LLM, but uses only the image modality and lacks grounding capability; after adding the time series modality and grounding data, GEM comprehensively surpasses PULSE.
- **vs. ECG-CoCa/ECG-Chat**: The time series encoder of ECG-CoCa is directly reused in GEM as a component, validating the effectiveness of contrastively pretrained ECG encoders within MLLMs; ECG-Chat handles only time series, whereas GEM processes both time series and images simultaneously.
- **vs. JoLT**: JoLT aligns ECG with text via a Querying Transformer; GEM achieves simpler and more efficient alignment through two-step MLP projection (time series → image dimension → text space).
- **vs. General MLLM (GPT-4o)**: GPT-4o performs far below specialized models on ECG anomaly detection (CSN 57.5% vs. GEM 92.6%), yet can generate high-quality training data under knowledge-guided prompting.

### Inspirations and Connections
- The knowledge-guided data generation paradigm (feature extraction tool + diagnosis guider + LLM generation) is transferable to grounded diagnosis for other medical time series signals such as EEG and EMG.
- The single-step training strategy (frozen encoders + joint training of projectors and LLM) is more effective than conventional multi-step training in low-data regimes and merits validation in other domains.
- In cardiologist evaluations, GEM identified details missed by experts upon initial examination — AI-assisted diagnosis should pursue not only agreement with experts but also complementary enhancement.

## Rating
- Novelty: ⭐⭐⭐⭐ First tri-modal ECG understanding model with a novel and practical knowledge-guided data generation method; however, individual components (dual encoder, MLP alignment) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple datasets, diverse tasks, ablation analysis, and expert evaluation — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated clinical rationale, though the dense notation may be less accessible to non-medical readers.
- Value: ⭐⭐⭐⭐⭐ Grounded diagnosis is critical for clinical trust; GEM significantly advances the practical applicability of AI-assisted ECG interpretation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding](../../AAAI2026/multimodal_vlm/anyecg-chat_a_generalist_ecg-mllm_for_flexible_ecg_input_and.md)
- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](rtv_bench_benchmarking_mllm_continuous_perception_through_realtime_video.md)
- [\[NeurIPS 2025\] in the eye of mllm benchmarking egocentric video intent understanding with gaze-](in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[ICML 2026\] ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation](../../ICML2026/multimodal_vlm/ecg-r1_protocol-guided_and_modality-agnostic_mllm_for_reliable_ecg_interpretatio.md)
- [\[AAAI 2026\] Harnessing Vision-Language Models for Time Series Anomaly Detection](../../AAAI2026/multimodal_vlm/harnessing_vision-language_models_for_time_series_anomaly_detection.md)

</div>

<!-- RELATED:END -->
