---
title: >-
  [Paper Note] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Concept Erasure] This paper systematically evaluates the trade-off between safety (erasure success rate) and compositional generation capability across 16 text-to-image diffusion model unlea…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Concept Erasure"
  - "Compositional Generation"
  - "Text-to-Image"
  - "Diffusion Models"
  - "Unlearning"
date: 2026-05-08
content_hash: edc10e815ba370d8
---

# Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models

**Conference**: CVPR 2026  
**arXiv**: [2604.04575](https://arxiv.org/abs/2604.04575)  
**Code**: None  
**Area**: Diffusion Models / AI Safety  
**Keywords**: Concept Erasure, Compositional Generation, Text-to-Image, Diffusion Models, Unlearning

## TL;DR
This paper systematically evaluates the trade-off between safety (erasure success rate) and compositional generation capability across 16 text-to-image diffusion model unlearning methods. It reveals that aggressive erasure strategies, while removing undesirable content, severely damage attribute binding, spatial reasoning, and counting abilities, emphasizing that safety interventions should not come at the expense of the model's semantic logic.

## Background & Motivation

1.  **Background**: Text-to-image diffusion models like Stable Diffusion have achieved great success in content creation, but their training data inevitably contains unsafe content (e.g., pornography, copyrighted material). Due to the high cost of retraining, post-hoc concept erasure methods have become the mainstream solution.

2.  **Limitations of Prior Work**: Existing unlearning evaluation focuses almost exclusively on "erasure success rate"—whether the target concept is successfully suppressed. However, a model outputting purely black images could technically achieve a perfect erasure score, indicating that single-metric evaluation is inherently incomplete.

3.  **Key Challenge**: Erasure operations act on the model's shared semantic space (cross-attention subspace), while compositional generation (attribute binding, spatial relations, counting) relies precisely on these shared representations. Therefore, erasure targeted at specific concepts is likely to cause "collateral damage" to the model's compositional capabilities.

4.  **Goal**: Systematically quantify the degree to which different unlearning methods affect compositional generation capabilities, revealing the trade-off between safety and utility.

5.  **Key Insight**: The authors argue that compositional ability is a proxy metric for model generative capacity—if unlearning breaks attribute binding (e.g., "green banana"), it suggests that the generative grammar itself has been damaged, not just a specific concept removed.

6.  **Core Idea**: By using compositional benchmarks such as T2I-CompBench++ and GenEval, the paper reveals for the first time a consistent inverse relationship between safety and compositionality in concept erasure methods.

## Method

### Overall Architecture
This paper is an empirical study that does not propose a new algorithm but designs a dual evaluation framework. The inputs are models with different unlearning methods applied to Stable Diffusion 1.4; the outputs are comprehensive scores for each method across safety and compositionality dimensions.

### Key Designs

1.  **Safety Evaluation Dimension**:
    - **Function**: Measure the effectiveness of unlearning methods in erasing undesirable content.
    - **Mechanism**: Compute Unlearning Accuracy (UA) using the top-200 prompts from the I2P benchmark; generate neutral counterparts for each prompt using ChatGPT (e.g., "a naked man" $\rightarrow$ "a man") and measure Retain Accuracy (RA) using BVQA scores on neutral prompts; calculate CLIP Score on the neutral subset of the SIX-CD benchmark; and use FID to measure overall image fidelity.
    - **Design Motivation**: Neutral prompts lie within the semantic neighborhood of erased concepts, allowing precise detection of whether unlearning is excessive—surgical removal or destructive alteration of underlying concepts.

2.  **Compositionality Evaluation Dimension**:
    - **Function**: Measure whether the model's compositional generation capability is impaired after unlearning.
    - **Mechanism**: Employ two complementary benchmarks, T2I-CompBench++ (evaluating 8 dimensions: color, shape, texture, 2D spatial, 3D spatial, numeracy, non-spatial, and complexity) and GenEval (evaluating 5 dimensions: single object, two objects, color, position, and counting), excluding prompts containing the erased concepts.
    - **Design Motivation**: Compositional generation involves various fine-grained semantic operations, providing a comprehensive reflection of the damage unlearning inflicts on the model's internal semantic structure.

3.  **Evaluation Method Coverage**:
    - **Function**: Ensure the universality of the conclusions.
    - **Mechanism**: Evaluates 16 representative unlearning methods, including global parameter fine-tuning (ESD, Salun, ADV), local layer intervention (UCE, SPM, MACE), adversarial regularization (RACE), and inference-time methods (SAFREE).
    - **Design Motivation**: Covering different technical schools avoids conclusions being restricted to specific types of methods.

### Loss & Training
As this is an evaluation study, no new loss functions are designed. All evaluated models are implemented on the Stable Diffusion 1.4 backbone to ensure performance differences stem from unlearning strategies rather than architectural changes.

## Key Experimental Results

### Main Results

**T2I-CompBench++ Compositionality Results**:

| Method | Color | Shape | Texture | 2D-Spatial | Numeracy | Mean | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SD 1.4 (Baseline) | 0.357 | 0.326 | 0.397 | 0.117 | 0.449 | 0.321 | - |
| UCE | 0.351 | 0.378 | 0.420 | 0.092 | 0.430 | 0.324 | +1.2% |
| SPM | 0.345 | 0.366 | 0.372 | 0.125 | 0.448 | 0.322 | +0.4% |
| ESD | 0.260 | 0.356 | 0.342 | 0.086 | 0.405 | 0.287 | -10.5% |
| Salun | 0.121 | 0.181 | 0.121 | 0.028 | 0.220 | 0.166 | -48.2% |
| EraseDiff | 0.010 | 0.017 | 0.011 | 0.000 | 0.043 | 0.052 | -83.7% |

**Safety vs. Retain Ability**:

| Method | UA $\uparrow$ | RA (BVQA) | FID $\downarrow$ |
| :--- | :--- | :--- | :--- |
| EraseDiff | 100% | 0.020 (-92.7%) | 73.11 |
| Scissorhands | 100% | 0.053 (-80.6%) | 49.49 |
| ACE | 99.5% | 0.233 (-14.7%) | 18.34 |
| UCE | 93.5% | 0.268 (-1.8%) | 18.24 |
| SPM | 59.0% | 0.265 (-2.9%) | 18.04 |

### Ablation Study

**GenEval Compositionality Results**:

| Method | Single | Two | Colors | Position | Counting | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SD 1.4 | 0.925 | 0.351 | 0.707 | 0.033 | 0.281 | 0.459 |
| ACE | 0.938 | 0.343 | 0.731 | 0.028 | 0.291 | 0.466 (+1.5%) |
| SPM | 0.947 | 0.323 | 0.702 | 0.035 | 0.309 | 0.463 (+0.9%) |
| EraseDiff | 0.056 | 0.003 | 0.019 | 0.000 | 0.003 | 0.016 (-96.5%) |
| Scissorhands | 0.044 | 0.005 | 0.027 | 0.000 | 0.003 | 0.016 (-96.5%) |

### Key Findings
- **2D Spatial Relations are Most Fragile**: Across all methods, the 2D-Spatial category showed the largest average decrease (-41.4%), indicating that layout-sensitive compositions are particularly vulnerable under unlearning.
- **Shape is Most Robust**: Coarse-grained geometric structures (Shape) were maintained or even improved under most methods, suggesting that appearance-level and relation-level cues are more easily destroyed than structural information.
- **ACE and SPM are Best-Balanced Methods**: Only ACE and SPM maintained or exceeded baseline performance on GenEval, showing that localized/structured editing strategies better preserve compositional ability.
- **Aggressive Methods Cause Manifold Collapse**: Image manifolds for EraseDiff (FID=73.11) and Scissorhands (FID=49.49) completely collapsed; ADV broke token alignment, and Salun exhibited mode collapse.

## Highlights & Insights
- **Compositionality as a Proxy for Semantic Integrity**: This observation is clever—if a model cannot correctly bind color and object in "green banana," it indicates the model's generative grammar is broken, not just that a concept was removed. This provides a new perspective for evaluating unlearning methods.
- **Neutral Prompt Probing Strategy**: Converting I2P prompts into neutral versions to test the integrity of semantic neighborhoods is an evaluation design replicable in other safety-related generative task assessments.
- **Policy Implications of Quantitative Findings**: The paper reveals an awkward reality—technically safe but semantically broken models cannot be considered truly trustworthy, which has guiding significance for establishing industry safety compliance standards.

## Limitations & Future Work
- **Evaluation Limited to Nudity Erasure**: All experiments focus on the nudity removal scenario; whether erasure of other concepts (copyrighted styles, violent content) shows the same trade-off remains unverified.
- **Based Only on SD 1.4**: Evaluation is limited to the older Stable Diffusion 1.4 architecture; conclusions might differ for newer architectures like SDXL or DiT.
- **Lack of Remediation Solutions**: The paper diagnoses the problem well but does not propose a solution—how to design unlearning methods that ensure both safety and compositionality remains an open question.
- **Possible Improvement Directions**: Design composition-aware unlearning objective functions, incorporating composition preservation regularization during the erasure process.

## Related Work & Insights
- **vs. ESD**: ESD achieves erasure by maximizing generation loss for the target concept; UA=93% but compositional Mean drops by 10.5%, placing it as a moderately aggressive method.
- **vs. UCE**: UCE performs local editing in cross-attention layers; UA=93.5% with effectively no loss in compositionality (+1.2%), making it one of the best methods for safety-utility balance.
- **vs. SPM**: SPM employs a 1D adapter approach; compositionality is best preserved (+0.4%) but UA is only 59%, representing a conservative strategy.
- This paper provides a clear benchmark for future unlearning research—any new method should report both UA and compositional metrics.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of unlearning's impact on compositionality; perspective is novel despite not being a method innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16 methods, two complementary benchmarks, multi-dimensional evaluation; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, complete chain of evidence, intuitive chart design.
- Value: ⭐⭐⭐⭐ Important cautionary value for the unlearning community, though the lack of a solution is slightly regrettable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)

</div>

<!-- RELATED:END -->
