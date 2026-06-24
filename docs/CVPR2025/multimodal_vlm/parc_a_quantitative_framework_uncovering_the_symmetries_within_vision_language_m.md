---
title: >-
  [Paper Note] PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Prompt sensitivity] Proposes the PARC framework. Through the three pillars of **11 linguistic/visual prompt variations**, **reliability scoring**, and **metric calibration**, the framework systematically quantifies and analyzes the prompt sensitivity of 22 VLMs across 7 datasets for the first time. The findings show that VLMs inherit the linguistic sensitivity of LLMs and exhibit symmetric behavior in the visual domain…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Prompt sensitivity"
  - "VLM robustness"
  - "Reliability score"
  - "Metric calibration"
  - "Prompt variation"
date: 2026-05-08
content_hash: f12f5826a6f2e0ee
---

# PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2506.14808](https://arxiv.org/abs/2506.14808)  
**Code**: [https://github.com/NVlabs/PARC](https://github.com/NVlabs/PARC)  
**Area**: Multimodal VLM  
**Keywords**: Prompt sensitivity, VLM robustness, Reliability score, Metric calibration, Prompt variation

## TL;DR
Proposes the PARC framework. Through the three pillars of **11 linguistic/visual prompt variations**, **reliability scoring**, and **metric calibration**, the framework systematically quantifies and analyzes the prompt sensitivity of 22 VLMs across 7 datasets for the first time. The findings show that VLMs inherit the linguistic sensitivity of LLMs and exhibit symmetric behavior in the visual domain, with the InternVL2 family being the most robust to prompt changes.

## Background & Motivation
VLMs are being deployed in safety-critical scenarios such as autonomous driving and medical screening, but they inherit a critical weakness from LLMs—**prompt sensitivity**. Different formulations of the same semantic meaning can lead to vastly different responses (e.g., swapping image order when asking "Which lane has people?" might cause incorrect answers). However, prompt sensitivity remains understudied in VLMs: existing work focusing only on noisy/corrupted prompts (non-realistic user scenarios), lacking a unified reliability metric, and failing to compare scores directly across different datasets/prompt variations. The Key Challenge is: **the community does not know which prompt variations are most damaging to VLMs, nor which VLMs are the most robust to these variations**. This work's Key Insight is to construct a comprehensive evaluation framework covering prompt variations in both linguistic and visual modalities, proposing an interpretable reliability score and introducing a calibration mechanism.

## Method

### Overall Architecture
The PARC framework consists of three pillars: (1) 11 realistic prompt variations across linguistic and visual domains (categorized into rephrasing and semantic-altering); (2) a novel reliability score combining accuracy and certainty; and (3) baseline-based metric calibration to make scores comparable across different datasets and prompt variations. A systematic evaluation of 22 VLMs on 7 datasets is performed using the multiple-choice visual question answering (MC-VQA) task.

### Key Designs
1. **11 Prompt Variations**:
    - Function: Systematically generate prompt variations that real users might produce, covering both linguistic and visual dimensions.
    - Mechanism: Divided into 2 categories $\times$ 2 modalities = 4 variation groups:
        - **Linguistic Rephrasing (LR)**: Instructional LR-I ("State which..."), Concise LR-C (reduced word count), Verbose LR-V (increased word count), without altering the correct answer.
        - **Linguistic Semantic (LS)**: Negation LS-N (adding "not"), Antonym LS-A, swapping more/less LS-M, which alters the correct answer.
        - **Visual Rephrasing (VR)**: Blur VR-B, Brightness variation VR-L, 90° rotation VR-R, without altering the correct answer.
        - **Visual Semantic (VS)**: Image swapped VS-S, Image replaced VS-E, which alters the correct answer.
        LLaMA3-70B is used to automatically generate linguistic variations, with manual quality checks.
    - Design Motivation: To mirror the linguistic variations of LLMs in the visual domain for the first time in VLMs, exploring the symmetry between the two modalities.

2. **Reliability Score**:
    - Function: Integrate accuracy and certainty into a single, intuitive, and guaranteed reliability metric.
    - Mechanism: Define $\mathit{rel} = (2 \cdot \mathit{acc} - 1) \cdot \mathit{cert}$, where certainty is calculated based on conformal prediction: $\mathit{cert}(p) = 1 - \frac{|\mathcal{C}(p)|-1}{|\mathcal{P}(p)|-1}$. A reliability of 1 indicates correct and confident (highly reliable), -1 indicates confident but incorrect (highly unreliable), and 0 indicates uncertain. It provides two explicit guarantees: $\mathit{cert} \geq |\mathit{rel}|$ and $\mathit{acc}_{\text{calib}} \geq \mathit{rel}$ (when positive).
    - Design Motivation: Existing analyses require looking at multiple metrics such as accuracy, certainty, and consistency simultaneously, lacking a single comprehensive, "at-a-glance" score. The reliability score conveys the most critical information in a single number.

3. **Score Calibration**:
    - Function: Eliminate difficulty discrepancies across different datasets and prompt variations, allowing scores to be directly compared across datasets and prompts.
    - Mechanism: Calibrate the score as the improvement over the random baseline:
$$s_{\text{calib}} = \begin{cases} \frac{s - s_{\text{rand}}}{1 - s_{\text{rand}}} & s \geq s_{\text{rand}} \\ \frac{s - s_{\text{rand}}}{s_{\text{rand}}} & s < s_{\text{rand}} \end{cases}$$
        The calibrated score $\in [-1, 1]$, where 1 is ideal, 0 is random, and -1 is the worst. For reliability, a power calibration is additionally introduced: replacing $\mathit{acc}$ with $\mathit{acc}^m$, where $m = \frac{\log 2}{\log(1/\mathit{acc}_{\text{rand}})}$, shifting the neutral reliability point (0) to the random accuracy probability.
    - Design Motivation: MMBench has 3 choices (random accuracy of 0.27), while NYU-Depth has only 2 choices (random 0.5), and negation leads to more correct answers (making guessing easier). Without calibration, a fair comparison is impossible.

### Loss & Training
This paper presents an evaluation and analysis framework, and does not involve model training.

## Key Experimental Results

### Main Results (Reliability Ranking of 22 VLMs, Calibrated, Averaged Across Datasets)

| Model | Reliability AVG | Accuracy-C AVG | Certainty-C AVG | Consistency AVG |
|------|-----------|-------------|-------------|-----------|
| InternVL2 40B | **0.40** | **0.65** | 0.57 | **0.71** |
| InternVL2 26B | 0.38 | 0.61 | **0.58** | 0.70 |
| LLaVA-1.6 34B | 0.33 | 0.54 | 0.57 | 0.63 |
| InternVL2 8B | 0.32 | 0.56 | 0.52 | 0.63 |
| Cambrian 34B | 0.30 | 0.52 | 0.52 | 0.64 |
| CogVLM GG | -0.12 | -0.15 | 0.40 | 0.02 |
| Qwen-VL | 0.06 | 0.15 | 0.29 | 0.14 |

### Prompt Variation Destructiveness Ranking (22-Model Average, Calibrated Reliability)

| Prompt Variation | Reliability AVG | Consistency AVG | Most Destructive |
|----------|-----------|-----------|-----------|
| Original O | 0.29 | - | - |
| LR-I Instructional | 0.26 | 0.78 | |
| LR-V Verbose | **0.21** | 0.63 | ✓ Worst Linguistic Rephrasing |
| LS-A Antonym | **0.10** | 0.09 | ✓ Worst Linguistic Semantic |
| VR-L Brightness | **0.21** | 0.63 | ✓ Worst Visual Rephrasing |
| VS-E Image Replaced | **0.13** | 0.00 | ✓ Worst Visual Semantic |

### Key Findings
- **VLMs inherit the prompt sensitivity of LLMs** and exhibit symmetric vulnerability in the visual domain—the patterns of linguistic rephrasing/semantic changes map perfectly to the visual domain.
- **Changes altering the correct answer are the most destructive**: semantic variations (negation, antonyms) are much harder to handle than simple rephrasing, consistently across both modalities.
- **Model families matter more than model size**: InternVL2-2B achieves reliability comparable to LLaVA-1.5 13B; while larger models within a family perform better, the gap between families is more pronounced.
- **High-quality training data is key**: models trained on 1B lower-quality web data are less reliable than Cambrian, which is trained on 0.01B carefully curated data.

## Highlights & Insights
- **First Linguistic-Visual Symmetry Analysis**: discovered that the behavioral patterns of linguistic vs. visual rephrasing and linguistic vs. visual semantics are strikingly "symmetric".
- **The Value of Calibration**: reveals misjudgments caused by uncalibrated scores—for instance, models on MMBench seemingly perform better on negated questions only because negation results in a higher probability of correct answers by guessing.
- **Elegance of the Reliability Score**: the simple formula $\mathit{rel} = (2 \cdot \mathit{acc} - 1) \cdot \mathit{cert}$ encodes both accuracy and confidence simultaneously, equipped with two guaranteed bound properties.

## Limitations & Future Work
- Only supports white-box VLMs (requiring softmax scores to calculate certainty), making it unable to analyze API-only models.
- Evaluates on MC-VQA; generative tasks require an extra LLM to verify responses, which introduces noise.
- Fewer types of visual semantic variations than other types (due to high manual creation cost), which may affect the generalizability of some findings.
- Has not explored whether prompting techniques (such as CoT) can mitigate sensitivity.

## Related Work & Insights
- Aligns with the comparative VQA style of CompBench, but is the first to be applied to systematic prompt sensitivity analysis.
- The calibration formulation can be generalized to cross-dataset comparisons for any benchmark.
- Insights: When selecting VLMs, one should not only focus on accuracy leaderboards—prompt robustness (reliability) may be more critical. The success of InternVL2 suggests that training data strategies are key to enhancing robustness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering framework for VLM prompt sensitivity, with stunning findings on visual-linguistic symmetry.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive analysis with 22 models $\times$ 7 datasets $\times$ 11 variations, yielding robust conclusions.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, with intuitive charts (especially pre/post-calibration comparisons) and a complete logical flow.
- Value: ⭐⭐⭐⭐⭐ Holds significant guidance value for VLM reliability evaluation and model selection; the framework can be readily adopted by the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](../../ICML2025/multimodal_vlm/ranked_from_within_ranking_large_multimodal_models_without_labels.md)
- [\[ICLR 2026\] Seeing Through Deception: Uncovering Misleading Creator Intent in Multimodal News with Vision-Language Models](../../ICLR2026/multimodal_vlm/seeing_through_deception_uncovering_misleading_creator_intent_in_multimodal_news.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](../../ACL2025/multimodal_vlm/activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)

</div>

<!-- RELATED:END -->
