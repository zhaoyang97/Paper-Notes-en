---
title: >-
  [Paper Note] VMDT: Decoding the Trustworthiness of Video Foundation Models
description: >-
  [NeurIPS 2025][Video Generation][Video foundation models] This paper introduces VMDT (Video-Modal DecodingTrust), the first unified benchmark platform for evaluating the trustworthiness of T2V and V2T video foundation mo…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Video foundation models"
  - "trustworthiness evaluation"
  - "safety"
  - "fairness"
  - "adversarial robustness"
date: 2026-05-08
content_hash: f4d3d5ee83326313
---

# VMDT: Decoding the Trustworthiness of Video Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.05682](https://arxiv.org/abs/2511.05682)
**Code**: [Available](https://sunblaze-ucb.github.io/VMDT-page/)
**Area**: Video Generation
**Keywords**: Video foundation models, trustworthiness evaluation, safety, fairness, adversarial robustness

## TL;DR
This paper introduces VMDT (Video-Modal DecodingTrust), the first unified benchmark platform for evaluating the trustworthiness of T2V and V2T video foundation models across five dimensions—safety, hallucination, fairness, privacy, and adversarial robustness—covering large-scale assessments of 7 T2V and 19 V2T models, and revealing the complex relationship between model scale and trustworthiness.

## Background & Motivation
**Background**: AI trustworthiness evaluation has predominantly focused on LLMs and image models (e.g., DecodingTrust, MMDT), leaving the video modality without a systematic trustworthiness benchmark. Video foundation models (VFMs) are rapidly advancing, yet their trustworthiness evaluation lags significantly behind.

**Limitations of Prior Work**: The video modality presents unique challenges—such as temporal risks (harmful content only manifesting during continuous playback) and photosensitive epilepsy triggers (flickering effects undetectable in static frames)—that are overlooked in image-based evaluation.

**Key Challenge**: While VFM capabilities are rapidly improving, safety alignment, fairness control, and privacy protection mechanisms remain severely insufficient, and no unified platform exists to measure and track progress.

**Goal**: To construct the first trustworthiness evaluation platform for video models covering both T2V and V2T directions across five key dimensions.

**Key Insight**: Drawing on the evaluation frameworks of DecodingTrust (text) and MMDT (image), the authors design dedicated datasets and metrics tailored to the specific characteristics of the video modality.

**Core Idea**: Decompose video model trustworthiness into five orthogonal dimensions—safety, hallucination, fairness, privacy, and adversarial robustness—and construct dedicated datasets and evaluation methods for each, forming a unified benchmark platform.

## Method

### Overall Architecture
The VMDT platform evaluates two categories of models: T2V (text-to-video, 7 models) and V2T (video-to-text, 19 models), with dimension-specific datasets and evaluation metrics designed for each of the five trustworthiness dimensions. Each dimension accounts for the unique characteristics of the video modality.

### Key Designs
1. **Safety Evaluation**:

    - T2V: 780 prompts covering 13 risk categories, including vanilla (direct harmful instructions) and transformed (seemingly benign but underlying harmful) scenarios.
    - V2T: 990 video-prompt pairs covering 6 major categories and 27 subcategories of risk.
    - Special attention to video-specific risks: temporal risks (harmful content only apparent during continuous playback) and physical harm (photosensitive epilepsy triggers).
    - Core metrics: Bypass Rate (BR, rate at which the model fails to refuse) and Harmful content Generation Rate (HGR).
    - GPT-4o is used as the evaluator (human agreement rate: 86%–88%).

2. **Hallucination Evaluation**:

    - T2V: 1,650 prompts; V2T: 1,218 prompts.
    - 7 scenarios: Natural Selection (NS), Distraction (DIS), Misleading (MIS), Counterfactual (CR), Temporal (TMP, video-specific), Co-Occurrence (CO), OCR.
    - 5 task types: object recognition, attribute recognition, action recognition, counting, spatial understanding (+ scene understanding for V2T).
    - T2V evaluation uses Qwen2.5-VL-72B-Instruct (Pearson correlation 0.765); V2T uses keyword matching.

3. **Fairness Evaluation**:

    - T2V: 1,086 prompts; V2T: 5,008 video-prompt pairs.
    - Three demographic attributes: gender, race, and age.
    - Three metrics: $F_1(g)$ social stereotype fairness, $F_2(g)$ decision fairness, and $O$ overkill fairness (sacrificing historical accuracy in pursuit of diversity).
    - Ideal values: $F_1=0$, $F_2=0$, $O=0$.

4. **Privacy Evaluation**:

    - T2V: 1,000 prompts (from the WebVid-10M training corpus) to evaluate data memorization.
    - V2T: 200 driving videos to evaluate location inference capability.
    - T2V metrics: $\ell_2$ distance and cosine similarity; V2T metric: location inference accuracy.

5. **Adversarial Robustness Evaluation**:

    - T2V: 329 benign/adversarial prompt pairs; V2T: 1,523 pairs.
    - Three attack algorithms: greedy, genetic, and gradient-based (T2V); FMM-Attack (V2T).
    - Five tasks: action/attribute/counting/object/spatial understanding.
    - Metrics: performance degradation between benign accuracy and robust accuracy.

### Loss & Training
This paper is an evaluation work (benchmark) and does not involve model training. The evaluation pipeline proceeds as follows: construct datasets per dimension → run model inference to generate outputs → automated evaluation (GPT-4o or keyword matching) → aggregated analysis.

## Key Experimental Results

### Main Results

**T2V Safety (HGR, lower is safer)**:

| Model | Vanilla HGR | Transformed HGR | Avg. HGR |
|-------|-------------|-----------------|----------|
| Nova Reel | 0.08 | 0.11 | **0.10** |
| Luma | 0.19 | 0.14 | 0.17 |
| CogVideoX-5B | 0.45 | 0.26 | 0.36 |
| Pika | 0.52 | 0.28 | 0.40 |

**T2V Hallucination (Accuracy %, higher is better)**:

| Model | NS | DIS | MIS | CR | TMP | CO | OCR | Avg |
|-------|----|-----|-----|----|-----|----|-----|-----|
| Luma | 63.8 | 74.7 | 78.3 | 68.5 | 45.5 | 82.9 | 59.7 | **67.6** |
| Pika | 56.5 | 68.9 | 72.3 | 70.7 | 53.7 | 77.3 | 41.5 | 63.0 |
| Vchitect-2.0 | 58.5 | 66.6 | 47.9 | 28.3 | 46.9 | 35.3 | 59.2 | 49.0 |

**Comprehensive Cross-Dimension Scores**:

| Rank | Best T2V | Worst T2V | Best V2T | Worst V2T |
|------|----------|-----------|----------|-----------|
| Model | Luma | CogVideoX-5B | InternVL2.5-78B | Qwen2.5-VL-3B |
| Overall Score | 70.1 | 55.7 | 72.7 | 65.3 |

### Ablation Study

**Effect of Model Scale on Each Dimension (V2T)**:

| Dimension | Scale Effect | Correlation |
|-----------|-------------|-------------|
| Hallucination | Scale↑ → Hallucination↓ | Positive (improvement) |
| Adversarial Robustness | Scale↑ → Robustness↑ | Positive ($P=0.034$) |
| Fairness | Scale↑ → Unfairness↑ | Negative |
| Privacy | Scale↑ → Location inference↑ | Positive (Pearson$=0.544$, $P=0.016$) |
| Safety | No significant scale effect | Uncorrelated |

### Key Findings
1. All open-source T2V models lack safety rejection mechanisms ($\text{BR}=1.00$), and even closed-source models offer only limited safety protection.
2. T2V models exhibit lower HGR in transformed scenarios, reflecting capability limitations rather than safety improvements.
3. Model scale is a double-edged sword for V2T models: it improves hallucination and robustness but worsens fairness and privacy risks.
4. T2V models exhibit severe over-representation across gender/race/age (biased toward male, white, and young subjects), with biases more pronounced than in T2I models.
5. Even the best-performing models achieve an overall score of only approximately 70–73, indicating a substantial gap from an ideal trustworthy model.

## Highlights & Insights
- **Pioneering Contribution**: The first unified video trustworthiness evaluation platform covering both T2V and V2T, filling an important gap in the field.
- **Video-Specific Risk Discovery**: Safety issues such as temporal risks and photosensitive epilepsy triggers cannot be captured by image-based evaluations.
- **Scale Paradox**: The first systematic demonstration of the non-monotonic relationship between model scale and trustworthiness in V2T models—larger models exhibit fewer hallucinations but are less fair.
- **Open-Source vs. Closed-Source Gap**: A substantial performance gap between open-source and closed-source models, particularly in the safety dimension.
- **Cross-Modal Comparison**: A systematic comparison of fairness between T2V and T2I models reveals that T2V models are biased toward male subjects while T2I models are biased toward female subjects (overcorrection).

## Limitations & Future Work
1. Evaluation relies on GPT-4o as the judge, introducing bias from the evaluator model (despite a human agreement rate of approximately 86–88%).
2. Only 7 T2V models are evaluated, providing insufficient coverage (e.g., Sora is not included).
3. Privacy evaluation covers only pixel-level memorization and location inference, excluding more complex privacy risks such as facial recognition.
4. The adversarial attack methods employed are limited and do not account for stronger adaptive attacks.
5. The paper lacks concrete recommendations and mitigation strategies for model improvement.

## Related Work & Insights
- **DecodingTrust / TrustGPT**: LLM trustworthiness evaluation frameworks; VMDT extends these to the video modality.
- **MMDT**: Multimodal (image) DecodingTrust; the direct predecessor of VMDT, whose dataset design is reused in several dimensions.
- **T2VSafetyBench / SafeWatch-Bench**: Prior work on video safety evaluation; VMDT integrates and extends their data and taxonomy.
- Insight: Trustworthiness evaluation should advance in tandem with model capabilities; as the information-richest modality, video presents the most complex trustworthiness challenges.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive video model trustworthiness platform
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale evaluation across 26 models × 5 dimensions
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rich and insightful findings
- Value: ⭐⭐⭐⭐ Serves as an important reference for the safe development of video models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](../../CVPR2026/video_generation/orbital_video_3d_foundation_priors.md)
- [\[NeurIPS 2025\] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals](force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)
- [\[NeurIPS 2025\] Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision](video_diffusion_models_excel_at_tracking_similar-looking_objects_without_supervi.md)
- [\[NeurIPS 2025\] Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models](video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
