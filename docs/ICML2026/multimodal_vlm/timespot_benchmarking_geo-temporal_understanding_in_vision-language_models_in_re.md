---
title: >-
  [Paper Note] TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings
description: >-
  [ICML 2026][Multimodal VLM][Geo-temporal Reasoning] The authors construct TimeSpot, a benchmark covering 80 countries with 1,455 real-world ground images…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Geo-temporal Reasoning"
  - "VLM Benchmark"
  - "Physical Consistency"
  - "Calibration"
  - "Supervised Fine-tuning"
date: 2026-05-08
content_hash: a62e366e98168263
---

# TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings

**Conference**: ICML 2026  
**arXiv**: [2603.06687](https://arxiv.org/abs/2603.06687)  
**Code**: https://TimeSpot-GT.github.io  
**Area**: Multimodal VLM  
**Keywords**: Geo-temporal Reasoning, VLM Benchmark, Physical Consistency, Calibration, Supervised Fine-tuning

## TL;DR
The authors construct TimeSpot, a benchmark covering 80 countries with 1,455 real-world ground images, forcing VLMs to simultaneously provide a 9-field structured prediction of "When (season/month/minute-level local time/diurnal phase)" and "Where (continent/country/climate zone/environment type/coordinates)". Results show that even the strongest model, Gemini-2.5-Flash-Thinking, achieves only 77.59% country accuracy and a median geographic error of 892.54 km, with minute-level time accuracy below 34%, indicating a severe lack of joint geo-temporal reasoning based on physical cues.

## Background & Motivation

**Background**: Significant progress has been made in image geo-localization by VLMs, with mainstream approaches including cross-view retrieval (VIGOR, OpenStreetView-5M), unified embedding (GeoCLIP), and chain-of-thought-enhanced LLMGeo/IMAGEO-Bench. These works model the task as a spatial retrieval of "image $\rightarrow$ coordinates".

**Limitations of Prior Work**: Existing benchmarks almost exclusively evaluate "Where", reporting either retrieval rank or coordinate error. "When" is largely ignored; models are not required to predict season, month, local time, or diurnal phase, nor must they satisfy cross-field consistency constraints such as "no snow in July in the Northern Hemisphere". Consequently, high spatial accuracy can coexist with physically impossible outputs.

**Key Challenge**: Real-world deployment (disaster response, traffic planning, embodied navigation, world models) requires models to provide *verifiable* spatio-temporal predictions while ensuring internal consistency. However, current VLMs lack explicit temporal-physical supervision in both training objectives and evaluation protocols, causing models to rely on surface-level image semantics (landmarks, text) for coarse memorization rather than regressing to long-tail physical cues like solar geometry and vegetation phenology.

**Goal**: (i) Construct a non-landmark-oriented benchmark that forces VLMs to jointly predict nine spatio-temporal fields with machine-auditable consistency; (ii) systematically evaluate the limits of current open-source, proprietary, and reasoning-enhanced VLMs; (iii) examine whether explicit supervision can bridge this gap through SFT.

**Key Insight**: The data framework utilizes "non-landmark ground photos + programmatically derived labels + human secondary verification". Programmatic labels derive solar elevation, Köppen climate, month, and season from timestamps and coordinates, naturally ensuring physical consistency; humans act as auditors for boundary cases. This allows for scalability while ensuring the ground truth possesses inherent verifiable semantics.

**Core Idea**: "When and Where" is redefined as a *constrained structured prediction* problem, treating cross-field consistency as a first-class evaluation metric to expose systematic failures in VLM physical grounding.

## Method

### Overall Architecture
TimeSpot maps each image $x$ to a structured label $y=(y^{\mathrm{temp}}, y^{\mathrm{geo}})$, where $y^{\mathrm{temp}}=(s, m, \tau, \phi)$ represents season, month, local time HH:MM, and diurnal phase, and $y^{\mathrm{geo}}=(C, \kappa, z, e, (\lambda,\varphi))$ represents continent, country, climate zone, environment type, and coordinates. The workflow consists of four steps: (1) recalling ~20,000 candidate ground images from the web and author-captured photos; (2) filtering out samples dominated by landmarks and text, retaining fine-grained physical cues like phenology, lighting, and textures; (3) *programmatically* deriving the nine fields from EXIF data and coordinates; (4) a two-stage human verification by 3 primary annotators and 2 senior auditors (~600 hours). The final output consists of 1,455 images across 80 countries, stored in a unified JSON schema, with automated auditing for month-season-hemisphere alignment, phase-time-longitude compatibility, and climate-coordinate rationality.

### Key Designs

1.  **Structured 9-Field Schema + Programmatic Label Derivation**:
    - **Function**: Decomposes "When and Where" into 9 fields that are scored independently but must remain mutually consistent, ensuring ground truth comes from physical formulas rather than crowdsourced guesses.
    - **Mechanism**: Month is taken from EXIF; season is derived via meteorological definitions with hemisphere correction (June-August is summer in the North, vice versa in the South); diurnal phase is calculated via solar elevation angle $\theta_\odot$ compared against civil/nautical/astronomical thresholds (e.g., $\theta_\odot < -6^\circ$ for civil twilight); local time is derived from time zone and ephemeris; climate zone follows Köppen-Geiger mapping from $(\lambda, \varphi)$; continent/country/coordinates are via reverse geocoding. All steps are deterministic, followed by human auditing against lighting and vegetation.
    - **Design Motivation**: The fundamental flaw of retrieval-centric benchmarks is the lack of cross-field semantics in ground truth. TimeSpot's formulaic GT allows violations like "July snow" to be flagged, forcing physical joint reasoning.

2.  **Cross-Field Consistency Diagnosis + Calibration Metrics**:
    - **Function**: Evaluates internal physical consistency and confidence reliability of model outputs beyond field-level accuracy.
    - **Mechanism**: Defines consistency violation rates, such as *Month-Season Inconsistency* (predicted month not in predicted season for the country's hemisphere), *Phase-Time Mismatch* ($|\Delta t| > 1\text{h}$), and *Continent-Country Inconsistency*. Calibration is measured via Expected Calibration Error $\mathrm{ECE}=\sum_b \frac{|B_b|}{N}|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$ and risk-coverage curves.
    - **Design Motivation**: Experiments show models like QwenVL-3B have only 0.21% phase-time violation but 95.19% error rate for MD > 1000 km, proving single-dimension accuracy does not imply global consistency.

3.  **SFT Intervention as a Diagnostic Tool**:
    - **Function**: Uses LoRA to fine-tune Qwen-VL2.5-3B on country, time, and joint prediction to quantify if explicit supervision fixes grounding.
    - **Mechanism**: Uses a 40/60 split for training/evaluation; tests country-only, time-only, and joint SFT. The training objective is interpreted as competing gradients between *illumination-invariant features* (for country) and *illumination-sensitive features* (for time).
    - **Design Motivation**: To demonstrate "why SFT is insufficient"—country-SFT improves country accuracy but degrades time accuracy, revealing gradient conflicts in shared parameters.

### Loss & Training
The benchmark is for evaluation. The SFT diagnostic uses standard instruction-tuning cross-entropy loss with LoRA on Qwen-VL2.5-3B-Instruct for 5 epochs. At inference, temperature=0 and JSON output is enforced with normalized parsing.

## Key Experimental Results

### Main Results
Evaluation of 31 VLMs against human baselines (undergraduates and experts).

| Model | Country Acc↑ | MD (km)↓ | Season Acc↑ | Time ±1h Acc↑ | Time MAE↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini-2.5-Flash-Thinking | **77.59** | **892.54** | 51.13 | 22.19 | 4:03 |
| Gemini-2.5-Flash | 77.25 | 917.61 | 50.92 | 25.15 | 3:56 |
| GPT-5-mini | 68.27 | 1389.79 | 58.43 | 21.55 | 4:10 |
| GLM-4.5V-106B-MoE | 69.68 | 1280.87 | 57.55 | 30.51 | 4:09 |
| Qwen-VL2.5-7B | 73.96 | 4719.95 | 61.46 | 25.68 | 3:47 |
| GLM-4.1V-9B-Thinking | 68.34 | 1788.77 | 58.02 | **33.74** | 3:58 |
| o4-mini | 71.82 | 1359.96 | **65.81** | 23.91 | 4:04 |
| Human (Expert) | 67.89 | 1040.42 | 86.56 | 57.89 | **1:36** |
| Human (Undergrad) | 45.98 | 2800.49 | 68.89 | 41.92 | 2:41 |

### Ablation Study (Consistency Diagnosis)
Strong models still exhibit high cross-field contradictions.

| Model | Phase-Time Mismatch (>1h) ↓ | Month-Season Inconsist. ↓ | Country-MD > 200km Conflict ↓ | MD > 1000km Ratio ↓ |
| :--- | :--- | :--- | :--- | :--- |
| GPT-5-mini | 15.95% | 0.89% | 16.98% | 17.25% |
| InternVL3-78B | 11.82% | 0.62% | 27.42% | 37.73% |
| QwenVL-3B | 0.21% | 0.82% | 12.78% | **95.19%** |

### Key Findings
- Strong VLMs exceed undergraduates in country accuracy but lag behind experts in time prediction by ~2.5 hours, suggesting VLMs memorize *geographic stereotypes* rather than having a continuous 4D physical world model.
- Thinking (reasoning enhancement) consistently yields gains (Gemini-2.5-Flash $\rightarrow$ Thinking: country +0.34%, MD −25 km), indicating explicit reasoning integrates low-saliency lighting cues better.
- Autumn classification collapses across all models, revealing a reliance on saturated color cues (green/snow) rather than phenological gradients.
- GPT-5-mini achieves 60.5% season accuracy from sun/shadow but ±1h time remains < 25%, showing it treats light as *semantic correlation* rather than *physical input*.

## Highlights & Insights
- Framing geo-localization as "9-field structured prediction + consistency auditing" exposes massive "physical contradictions" in seemingly powerful VLMs—a methodology transferable to any multi-variable reasoning domain (e.g., medical imaging, autonomous driving).
- The hybrid of *programmatic label derivation + human auditing* ensures both scale and physical correctness, providing a paradigm for future "verifiable benchmarks".
- Using SFT as a diagnostic tool clearly demonstrates gradient conflicts under single-task LoRA, motivating future constraint-aware RL.
- Consistency and accuracy are orthogonal; QwenVL-3B's superior phase-time consistency despite its 95% MD error rate warns against being misled by single-dimensional metrics.

## Limitations & Future Work
- The data scale (1,455 images) is relatively small; maintaining human audit quality while scaling to $\ge$ 10k images is a challenge.
- Evaluation depends on OpenRouter APIs (~$1,450 USD), making it sensitive to model versioning and price fluctuations.
- SFT experiments were limited to Qwen-VL2.5-3B; verification of gradient conflicts on larger architectures (e.g., GLM-4.6V) is needed.
- Data exhibits over-sampling of Europe/North America and statistical fragility in the Southern Hemisphere Summer (56 samples).

## Related Work & Insights
- Orthogonal to cross-view retrieval (VIGOR, GeoCLIP): This work focuses on "When + Consistency" assuming spatial priors.
- Complementary to LLM-geolocation (LLMGeo, IMAGEO-Bench): TimeSpot's schema can extend these works which lack temporal fields.
- Distinct from Remote Sensing VQA: TimeSpot emphasizes ground-level physical grounding rather than aerial segmentation.
- Implications for World Models: Soft constraints like temperature and phenology are necessary priors; physical heads (solar geometry/Köppen) should be added as multi-task losses during VLM pretraining.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)
- [\[ICML 2026\] Immuno-VLM: Immunizing Large Vision-Language Models via Generative Semantic Antibodies for Open-World Trustworthiness](immuno-vlm_immunizing_large_vision-language_models_via_generative_semantic_antib.md)
- [\[ICLR 2026\] Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](../../ICLR2026/multimodal_vlm/bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)

</div>

<!-- RELATED:END -->
