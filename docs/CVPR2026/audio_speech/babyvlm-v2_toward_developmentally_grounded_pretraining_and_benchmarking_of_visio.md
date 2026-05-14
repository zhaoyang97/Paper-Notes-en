---
title: >-
  [Paper Note] BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models
description: >-
  [CVPR 2026][Audio & Speech][Developmental Cognition] This paper proposes BabyVLM-V2, a framework that constructs three formats of pretraining data (768K image pairs + 181K video pairs + 63K interleaved sequences) from th…
tags:
  - "CVPR 2026"
  - "Audio & Speech"
  - "Developmental Cognition"
  - "Infant Vision"
  - "Sample-Efficient Pretraining"
  - "NIH Baby Toolbox"
  - "DevCV Toolbox"
date: 2026-05-08
content_hash: 3ba273b9298f8dfd
---

# BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models

**Conference**: CVPR 2026
**arXiv**: [2512.10932](https://arxiv.org/abs/2512.10932)
**Code**: [https://shawnking98.github.io/BabyVLM-v2/](https://shawnking98.github.io/BabyVLM-v2/)
**Area**: Audio & Speech
**Keywords**: Developmental Cognition, Infant Vision, Sample-Efficient Pretraining, NIH Baby Toolbox, DevCV Toolbox

## TL;DR
This paper proposes BabyVLM-V2, a framework that constructs three formats of pretraining data (768K image pairs + 181K video pairs + 63K interleaved sequences) from the SAYCam longitudinal egocentric corpus, designs the DevCV Toolbox (10 developmental cognitive tasks) grounded in the NIH Baby Toolbox®, and demonstrates that a compact model trained from scratch surpasses GPT-4o on selected mathematical tasks — representing the first systematic exploration of Artificial Developmental Intelligence (ADI).

## Background & Motivation

**Background**: Vision foundation models rely on scaling laws and massive datasets for pretraining, yet young children develop robust perceptual and reasoning abilities from extremely limited visual input — approximately 40,000 waking hours from birth to age three. This discrepancy presents a natural target for sample-efficient pretraining research.

**Limitations of Prior Work**: BabyVLM-V1 suffers from four key shortcomings: (1) it utilizes only ~1/3 of SAYCam footage (67K image pairs), covering a minimal fraction of available data; (2) it supports only image–text pairs, with no support for video or multi-turn dialogue; (3) its 4 evaluation tasks were designed intuitively rather than grounded in standardized psychological assessments; and (4) the model's open-set performance is near zero, requiring logit post-processing for evaluation.

**Key Challenge**: The core challenge lies in training a foundation model with diverse capabilities analogous to those of young children while respecting the constraints of infants' limited sensory experience, and in evaluating such models fairly using developmental psychology standards.

**Key Insight**: (1) Maximize utilization of the SAYCam corpus and construct multi-format data to support diverse downstream tasks; (2) adopt the NIH Baby Toolbox® — released in February 2025 and currently the most authoritative tool for assessing child neurodevelopment — as the basis for benchmark design.

**Core Idea**: Engineer standardized developmental psychology assessments into computer vision tasks for AI evaluation, thereby establishing the DevCV Toolbox.

## Method

### Overall Architecture
SAYCam infant longitudinal footage (478 hours) → minimally processed pretraining data in three formats → three-stage pretraining (visual encoding → alignment → multi-format training) → instruction fine-tuning (113K samples) → DevCV Toolbox evaluation (10 cognitive tasks).

### Key Designs

1. **Pretraining Data Construction (Developmental Authenticity via Minimal Processing)**:

    - **Video–sentence pairs (181K)**: Videos are segmented by speech transcription boundaries; captions are extracted via Azure Speech Recognition; samples with X-CLIP similarity > 0.1 are retained, preserving 138 hours of footage.
    - **Image–sentence pairs (768K)**: Sampled at 1 FPS from video pairs, filtered with CLIP similarity > 0.2. This represents an 11× expansion over V1's 67K pairs.
    - **Interleaved image–text sequences (63K)**: A sliding window (size 4–8) combines best frames and sentence pairs from consecutive segments to simulate continuous infant interactive experience.
    - **Design Motivation**: The three formats respectively support video understanding, image understanding, and multi-turn dialogue, collectively covering the diverse task requirements of the DevCV Toolbox.

2. **DevCV Toolbox (10 Developmental Cognitive Tasks, Grounded in NIH Baby Toolbox®)**:

    - **Language subdomain**: Looking While Listening (6–24 months, two-image forced choice), Picture Vocabulary (≥25 months, four-image vocabulary comprehension), Localization (1–42 months, object localization).
    - **Executive Function/Memory subdomain**: Left/Right (orientation discrimination), Spatial Details (spatial detail recognition), Visual Delayed Response (memory after occlusion), Memory (multi-turn delayed memory).
    - **Mathematics subdomain**: Who Has More (quantity comparison, synthetic + natural versions), Subitizing (rapid enumeration), Object Counting (object counting).
    - Each task constructs naturalistic scene samples from SAYCam frames, replacing the cartoon stimuli of the original toolbox to ensure in-domain evaluation.
    - **Design Motivation**: The clinical validation of the NIH Baby Toolbox® establishes its credibility as a developmental assessment instrument.

3. **Adaptation Process (Picture Vocabulary as an Example)**:

    - Original NIH test: four cartoon images displayed on iPad with audio prompt → child taps selection.
    - DevCV adaptation: SAYCam frames sampled at 1 FPS → objects annotated by GPT-4o and manually → cropped via Grounding-DINO → filtered against the MAB-CDI vocabulary list → distractors constructed by semantic/phonological distribution → human quality review.

4. **Model Architecture**:

    - ViT-L-16 (300M) + MLP connector + LLaMA-1.1B.
    - **Trained entirely from scratch**, with no pretrained weights — ensuring that all capabilities derive exclusively from the infant corpus.
    - Inputs: text / single image / multiple images / video / multi-turn dialogue; Output: natural language.

### Loss & Training
A three-stage pipeline: Stage 1 — visual encoder pretraining; Stage 2 — image–text alignment; Stage 3 — multi-format joint training. This is followed by instruction fine-tuning on DevCV tasks.

## Key Experimental Results

### Main Results (DevCV Toolbox In-Domain Evaluation)

| Model | Overall | Count | PV (Vocab) | Memory | WhoHasMore | LeftRight |
|-------|---------|-------|------------|--------|------------|-----------|
| Human Performance | 93.0 | 99.1 | 91.8 | 87.3 | 63.6/95.5 | 94.5 |
| Gemini-2.5-flash | 72.7 | 71.1 | 91.2 | 84.8 | 42.4 | 34.9 |
| GPT-4o | ~70 | ~65 | ~90 | ~80 | ~40 | ~34 |
| **BabyVLM-V2** | Competitive | **Surpasses GPT-4o on select tasks** | Competitive | Competitive | Competitive | Competitive |

### Ablation Study

| Configuration | Key Impact | Notes |
|---------------|------------|-------|
| Image–text pretraining only (V1) | Baseline | Near-zero open-set performance |
| + Video–sentence pairs (181K) | Improved video understanding tasks | DelayedResponse benefits |
| + Interleaved sequences (63K) | Improved multi-turn dialogue tasks | Memory task benefits |
| + Instruction fine-tuning (113K) | **Substantial overall improvement** | Logit output → natural language |
| 768K vs. 67K image pairs | V2 >> V1 | Direct effect of data scale |

### Key Findings
- **Mathematical tasks exceed GPT-4o**: The ~1.4B model trained from scratch partially surpasses GPT-4o on Who Has More and Counting, indicating that infant experience data encodes sufficient quantity and enumeration understanding.
- The human ceiling on the DevCV Toolbox (93%) substantially exceeds all AI models, highlighting a significant gap between AI and child cognition.
- Subitizing and Looking While Listening, held out as generalization probes, confirm the generalization benefits of multi-format pretraining.
- Each of the three pretraining data formats contributes independently and complementarily.
- Performance degradation on the OOD test set (constructed from Ego4D) validates the necessity of in-domain evaluation.

## Highlights & Insights
- **Engineering Standardized Developmental Assessment for AI**: This is the first work to adapt the NIH Baby Toolbox® into an AI evaluation benchmark, establishing a new research paradigm for developmental computational vision. The DevCV Toolbox may also enable psychologists to "read the minds of young children" through AI.
- **Challenging Scaling Laws**: Only 478 hours of infant experience suffice to train a model that surpasses GPT-4o on mathematical tasks, demonstrating the substantial potential of sample-efficient pretraining.
- **Data Format Diversity > Data Volume**: The leap from V1 (67K) to V2 (768K + video + interleaved) derives not merely from increased quantity but, more critically, from format diversity that enables capability diversification.
- **Tripartite benefit**: enables universities to participate in foundation model research, provides cognitive scientists with experimental tools, and improves public understanding of AI.

## Limitations & Future Work
- SAYCam covers only 3 infants (6–32 months of age), yielding a minimal and potentially idiosyncratic sample. Larger-scale datasets such as BabyView await integration.
- The compact model remains substantially inferior to large models and humans on complex reasoning tasks — the ADI gap remains large.
- The DevCV Toolbox lacks actual child performance data (only an adult upper bound), necessitating collaboration with psychology laboratories to collect genuine developmental comparison data.
- Instruction fine-tuning employs the DevCV tasks themselves, potentially introducing task leakage.
- Non-visual aspects of development — language and motor development — are not included.

## Related Work & Insights
- **vs. BabyVLM-V1**: Data expanded 11×, multi-format support added; benchmark grows from 4 to 10 tasks grounded in NIH standardized assessments; model output transitions from logits to natural language.
- **vs. Vong et al. (CLIP on SAYCam)**: That work focuses solely on word–referent mapping, whereas this paper targets general perception.
- **vs. DevBench/KIVA**: Those benchmarks target older age ranges incompatible with the 6–32 month window of SAYCam.
- **Insight**: The developmental cognition perspective offers entirely new inspiration for AI training strategies — "learning like an infant" may represent an alternative path toward AGI.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Unique developmental cognition perspective and first AI adaptation of the NIH Baby Toolbox®.
- Experimental Thoroughness: ⭐⭐⭐⭐ — DevCV design is rigorous; real child performance comparisons are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Cross-disciplinary background is thoroughly introduced.
- Value: ⭐⭐⭐⭐⭐ — Profound implications for understanding the relationship between AI and human cognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining](../../ICCV2025/audio_speech/25_years_in_class_a_multimodal_textbook_for_visionlanguage_p.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](../../NeurIPS2025/audio_speech/textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](../../NeurIPS2025/audio_speech/data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)

</div>

<!-- RELATED:END -->
