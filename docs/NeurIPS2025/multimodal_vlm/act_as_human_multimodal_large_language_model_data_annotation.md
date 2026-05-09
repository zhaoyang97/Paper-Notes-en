---
title: >-
  [Paper Note] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking
description: >-
  [NeurIPS 2025][Multimodal VLM][data annotation] This paper proposes ACT (Annotation with Critical Thinking), a data pipeline in which an MLLM annotates all samples in bulk, a second MLLM acting as a critic estimates the error probability of each annotation, and only high-suspicion samples are routed to human reviewers. Combined with a theoretically derived ACT loss function, the approach achieves 70–90% reduction in human annotation cost across six cross-modal datasets while maintaining a downstream performance gap of less than 2%.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - data annotation
  - critical thinking
  - MLLM
  - error estimation
  - human-in-the-loop
date: 2026-05-08
content_hash: b91e6a3c3c380f6d
---

# ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking

**Conference**: NeurIPS 2025
**arXiv**: [2511.09833](https://arxiv.org/abs/2511.09833)
**Code**: None
**Area**: Data Annotation / MLLM Applications
**Keywords**: data annotation, critical thinking, MLLM, error estimation, human-in-the-loop

## TL;DR
This paper proposes ACT (Annotation with Critical Thinking), a data pipeline in which an MLLM annotates all samples in bulk, a second MLLM acting as a critic estimates the error probability of each annotation, and only high-suspicion samples are routed to human reviewers. Combined with a theoretically derived ACT loss function, the approach achieves 70–90% reduction in human annotation cost across six cross-modal datasets while maintaining a downstream performance gap of less than 2%.

## Background & Motivation
**State of the Field**: Supervised learning depends on high-quality labeled data, yet manual annotation is expensive and difficult to scale. Automatic annotation via LLMs/MLLMs is cheap but still lags behind human quality.

**Limitations of Prior Work**: (1) Pure MLLM annotation accuracy falls 5–20% short of human annotation, causing notable downstream performance degradation; (2) existing methods such as CDI require training an additional XGBoost detector and generalize poorly; (3) some approaches are restricted to white-box models and cannot leverage powerful black-box models such as GPT-4o; (4) the normalized sampling rule used in existing active M-estimation collapses under low-budget conditions.

**Root Cause**: How can the annotation capability of MLLMs be maximally exploited under a limited human annotation budget while maintaining data quality close to fully human-annotated data?

**Starting Point**: MLLMs are assigned dual roles as both annotator and critic — first generating labels, then performing self- or cross-critique — so that human effort is precisely allocated to the most suspicious samples.

## Method

### Overall Architecture
ACT is a three-stage, training-free pipeline: (1) **Annotation stage**: MLLM $f^{(m)}$ generates labels $\hat{y}_i^{(m)}$ for all $N$ samples; (2) **Error estimation stage**: a separate MLLM $g$ acts as a critic and estimates the error probability $\hat{\epsilon}_i = g(\mathbf{x}_i, \hat{y}_i^{(m)})$ for each annotation; (3) **Correction stage**: budget-aware sampling $\delta_i(B) \sim \mathbb{B}(\pi_B(\hat{\epsilon}_i))$ selects samples for human review subject to $\sum \delta_i(B) \leq B$. Downstream training uses a specially designed ACT loss function.

### Key Designs
1. **MLLM Criticizer Strategy Family**:
    - Function: Seven criticizer strategies spanning black-box and white-box settings are designed to enable MLLMs to estimate annotation error probabilities.
    - Mechanism: Black-box strategies include Naïve direct estimation, CoT-based estimation, multiple-choice grading (MC), and Devil's Advocate (reviewing the annotator's CoT before judging); white-box strategies use logit probabilities $\hat{\epsilon} = \mathbb{P}(\text{"yes"}) / (\mathbb{P}(\text{"yes"}) + \mathbb{P}(\text{"no"}))$ or CoT perplexity (PPL) as indirect error signals. Experiments show that CoT-based criticism achieves the highest ABS improvement of up to 22.46%, and cross-criticism (using a different model for annotation and criticism) generally outperforms self-criticism.
    - Design Motivation: Different task–model combinations favor different strategies; systematic exploration provides actionable deployment guidance. The training-free design allows the pipeline to be applied directly with any MLLM.

2. **Budget-Aware Sampling**:
    - Function: Determines which samples are routed to human reviewers under a fixed budget $B$.
    - Mechanism: Three sampling rules are proposed — normalized $\pi_B(\hat{\epsilon}_i) = B \cdot \hat{\epsilon}_i / \sum \hat{\epsilon}_i$, exponential-weighted $\pi_B(\hat{\epsilon}_i) = 1/(1 + e^{-\beta(\hat{\epsilon}_i - \alpha)})$, and thresholded $\pi_B(\hat{\epsilon}_i) = \mathbf{1}(\hat{\epsilon}_i \geq \tau)$. Theorem 5.2 proves that the parameter gap between the ACT loss and the true loss is upper-bounded by a quantity depending on $q$ (the lower bound of the transformed error probability for selected samples); exponential-weighted and thresholded rules push $q$ toward 1, whereas the normalized rule yields $q \to 0$ under low budgets, causing collapse.
    - Design Motivation: The normalized sampling rule used in prior work produces highly unstable loss behavior under limited human budgets — yielding a 76.34% gap from full supervision on the Cars dataset — whereas exponential-weighted and thresholded rules remain stable at 1.69%.

3. **ACT Loss Function**:
    - Function: A theoretically grounded loss function is designed so that models trained on ACT-annotated data approach the performance of models trained on fully human-annotated data.
    - Mechanism: $\mathcal{L}_\theta^{(ACT)} = \frac{1}{N}\sum_{i=1}^{N}\left(\ell_{\theta,i}^{(m)} + (\ell_{\theta,i} - \ell_{\theta,i}^{(m)}) \frac{\delta_i(B)}{\pi_B(\hat{\epsilon}_i)}\right)$, where $\ell_{\theta,i}^{(m)}$ is the machine-annotation loss and $\ell_{\theta,i}$ is the true-label loss estimated from human annotations. Proposition 5.1 proves that the ACT loss is an unbiased estimator of the true loss, with variance minimized in two cases: a perfect annotator or a perfectly accurate critic.
    - Design Motivation: Naively mixing human and machine annotations introduces label noise, while using only human-annotated samples wastes the already-labeled machine-annotated data. The ACT loss achieves unbiased estimation via importance weighting, and the exponential-weighted/thresholded rules prevent weight explosion.

### Loss & Training
The ACT loss is an improvement over active M-estimation. Its core mechanism uses the sampling probability $\pi_B(\hat{\epsilon}_i)$ for importance-weighted correction: samples selected for review use their true loss $\ell_{\theta,i}$ computed from human labels, while unselected samples use the machine-annotation loss $\ell_{\theta,i}^{(m)}$. The thresholded rule is recommended in practice (only the threshold $\tau$ needs to be set, which is simpler than the two hyperparameters $\alpha, \beta$ of the exponential-weighted rule). Downstream tasks use standard cross-entropy loss with power-tuning hyperparameters.

## Key Experimental Results

### Main Results: Downstream Task Test Accuracy (%)

| Training Data – Loss | CIFAR-10 | Fashion | Cars | Emotion | Irony | VQA-RAD |
|---|---|---|---|---|---|---|
| Full Human – CE | 88.66±0.97 | 93.01±0.63 | 87.88±0.36 | 81.82±0.57 | 70.18±3.23 | 67.81±1.47 |
| Full Machine – CE | 81.55±1.93 | 82.86±0.84 | 83.68±0.17 | 78.96±2.40 | 60.71±5.43 | 61.03±2.05 |
| ACT – Normalized Loss | 64.70±5.46 | 69.27±7.25 | 11.54±0.96 | 79.87±0.88 | 65.66±2.00 | 62.55±3.01 |
| **ACT – Exp-Weighted Loss** | **87.73±0.36** | **89.73±0.35** | **86.19±0.14** | **81.44±0.51** | **68.49±3.20** | **67.73±1.33** |
| ACT – Thresholded Loss | 87.95±0.35 | 89.16±0.89 | 86.00±0.26 | 81.41±0.64 | 68.21±1.94 | 67.02±1.32 |
| Human–ACT Gap | 0.71% | 3.28% | 1.69% | 0.38% | 1.69% | 0.08% |
| Human Budget Ratio | 11.52% | 21.81% | 9.56% | 17.98% | 33.79% | 30.15% |

### Ablation Study: Criticizer Strategy ABS (%) Comparison (GPT-4o annotation + CoT)

| Critic Model | Naïve | CoT | MC | Devil |
|---|---|---|---|---|
| GPT-4o (self-criticism) | 41.2 | 53.8 | 48.6 | 50.1 |
| Gemini-1.5-Pro | 45.3 | **56.2** | 51.4 | 52.7 |
| Claude 3.5 Sonnet | 43.7 | 54.9 | 52.1 | 55.3 |
| InternVL 2.5 | 38.5 | 44.1 | 40.3 | 42.8 |

### Key Findings
- Seven core insights are identified: GPT-4o is the best general-purpose annotator; CoT benefits criticism more than annotation (ABS improvement up to 22.46%); cross-criticism outperforms self-criticism; black-box models are stronger critics; annotation capability and criticizing capability are positively correlated.
- Normalized sampling collapses completely on Cars (11.54%), while exponential-weighted and thresholded sampling remain robust (86%+).
- White-box strategies (logit/PPL) outperform black-box strategies on 2 of 6 datasets, but results are inconsistent.

## Highlights & Insights
- The annotate–criticize–correct three-stage pipeline is elegantly designed and entirely training-free, allowing plug-and-play use with any MLLM.
- Seven systematic insights provide actionable best-practice guidance for real-world deployment.
- The ACT loss function offers theoretical guarantees (unbiased estimation + variance control), and the exponential-weighted/thresholded rules substantially outperform the normalized rule used in prior work.
- The study conducts systematic exploration across three domains (NLP, CV, VQA), six datasets, and six MLLMs with seven criticizer strategies and three sampling rules, constituting a highly comprehensive experimental design.
- The finding that annotation capability and criticizing capability are positively correlated simplifies model selection: use the top-1 model as the annotator and the top-2 model as the critic.

## Limitations & Future Work
- Validation is limited to classification tasks; generative tasks such as text summarization and open-ended QA are not covered.
- Critic accuracy is bounded by the MLLM's capability ceiling, and a 5–15% false positive rate limits the maximum achievable performance.
- Budget setting is based on annotator accuracy (an "ideal budget"), and practical budget allocation strategies are not discussed in depth.
- Performance on non-English scenarios such as Chinese or low-resource languages is not validated.

## Related Work & Insights
- **vs. CDI**: CDI requires training an XGBoost detector and uses normalized sampling (which collapses under low budgets), whereas ACT is fully training-free and employs stable thresholded sampling.
- **vs. LLM-as-a-Judge**: The ACT criticizer design is closely related to LLM self-evaluation; the finding that cross-criticism outperforms self-criticism echoes the self-evaluation bias literature.
- **vs. Active Learning**: Traditional active learning requires model retraining within the annotation loop, whereas the ACT pipeline requires no training at any stage.
- **Insights**: The budget-aware sampling paradigm is generalizable to any human–AI collaboration scenario; the positive correlation between annotator and critic capability simplifies pipeline configuration.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of criticizer + budget-aware sampling + ACT loss is practical and novel, though the core idea of LLM mutual evaluation is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets, six MLLMs, seven criticizer strategies, three sampling rules, and complete ablations constitute an exceptionally systematic study.
- Writing Quality: ⭐⭐⭐⭐ The seven insights are clearly summarized and theoretical analysis is tightly integrated with empirical results.
- Value: ⭐⭐⭐⭐⭐ The work has direct practical value for reducing AI data annotation costs and provides strong guidance for practitioners.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)

<!-- RELATED:END -->
