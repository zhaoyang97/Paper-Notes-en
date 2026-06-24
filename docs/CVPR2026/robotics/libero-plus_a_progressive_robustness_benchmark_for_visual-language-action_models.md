---
title: >-
  [Paper Note] LIBERO-Plus: A Progressive Robustness Benchmark for Visual-Language-Action Models
description: >-
  [CVPR 2026][Robotics][VLA Models] Addressing the illusion of "VLA models reporting 95%+ success rates on LIBERO but frequently failing in real deployment," this work constructs LIBERO-Plus, an automated, fine-grained robustness benchmark with seven dimensions of controllable perturbations. Systematic evaluation of 10 mainstream VLA models reveals that success rates plummet from 95% to below 30% under moderate perturbations, uncovering deep vulnerabilities such as "ignoring la…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "VLA Models"
  - "Robustness Benchmark"
  - "Distribution Shift"
  - "Perturbation Evaluation"
  - "Generalization Analysis"
date: 2026-05-08
content_hash: 0492b1814c2b10e2
---

# LIBERO-Plus: A Progressive Robustness Benchmark for Visual-Language-Action Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO_Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)  
**Code**: To be confirmed  
**Area**: Robotics / Embodied AI / VLA Robustness Evaluation  
**Keywords**: VLA Models, Robustness Benchmark, Distribution Shift, Perturbation Evaluation, Generalization Analysis

## TL;DR
Addressing the illusion of "VLA models reporting 95%+ success rates on LIBERO but frequently failing in real deployment," this work constructs LIBERO-Plus, an automated, fine-grained robustness benchmark with seven dimensions of controllable perturbations. Systematic evaluation of 10 mainstream VLA models reveals that success rates plummet from 95% to below 30% under moderate perturbations, uncovering deep vulnerabilities such as "ignoring language, relying on fixed visuals, and depending on positional memory."

## Background & Motivation
**Background**: Visual-Language-Action (VLA) models extend the foundation model paradigm from language/vision to robot manipulation. Autoregressive models discretize actions into tokens for end-to-end training, diffusion-based models generate continuous trajectories, and reinforcement learning approaches emphasize downstream adaptation. These models report 95–99% success rates on simulation benchmarks like LIBERO, suggesting that manipulation tasks are "nearly solved."

**Limitations of Prior Work**: Fundamental robustness weaknesses are hidden behind high scores. Existing simulation-based robustness evaluations have three main limitations: ① **Narrow perturbation coverage**—perturbations occur only along one or two axes (object variations, lighting, instruction paraphrasing) without systematically covering a broad spectrum of robustness factors; ② **Reliance on manual design**—manually designed perturbations are difficult to scale and replicate, often resulting in small sample sizes (frequently fewer than 100 test scenarios); ③ **Coarse granularity**—reporting only aggregate success rates masks details of "under what conditions and how the model fails." Consequently, prior methods provide fragmented insights and fail to systematically characterize model capability boundaries or guide targeted improvements.

**Key Challenge**: High benchmark scores $\neq$ real capability. Models perform well under fixed, ideal conditions but collapse when encountering lighting changes, viewpoint shifts, or different natural language phrasing—a systemic gap exists between evaluation protocols and real-world deployment.

**Goal**: To build an evaluation framework capable of applying controllable, automated, and fine-grained perturbations across **seven major dimensions** to reveal the true robustness of VLA models and locate "when and why they fail."

**Key Insight**: By extending the widely used LIBERO benchmark, this work upgrades evaluation methodology along three directions: comprehensive perturbation coverage (7 categories, 21 sub-categories), automated parameterized generation (scalable and reproducible), and progressive evaluation stratified by difficulty (L1–L5).

**Core Idea**: Replace narrow-dimension manual evaluation with "seven-dimensional controllable perturbations + automated generation + L1–L5 difficulty stratification" to dismantle the illusion of high VLA performance and produce fine-grained robustness profiles.

## Method

### Overall Architecture
LIBERO-Plus is an evaluation benchmark (not a new model) centered on systematically generating distribution shift scenarios based on the original LIBERO. It expands **7 perturbation factors** (object layout, camera viewpoint, robot initial state, language instructions, lighting conditions, background texture, and sensor noise) into **21 sub-dimensions**. It automatically generates **10,030 task instances** ⚠️ (the paper also mentions "over 56K robustness scenarios"; 10,030 likely refers to final task instances while 56K includes the larger training set). Tasks are calibrated into **L1–L5 difficulty levels** based on the empirical performance of four representative VLA models. These perturbations change only the input distribution while maintaining the task structure, characterizing "covariate shift" in OOD generalization. Using this benchmark, the authors evaluate 10 mainstream VLAs and perform in-depth analyses on language/visual dependencies, compositional generalization, and augmented training. Since this is a benchmark/dataset paper without a multi-stage model pipeline, no architecture diagram is provided.

### Key Designs

**1. Comprehensive Controllable Perturbations across 7 Dimensions and 21 Sub-classes**

Addressing the narrow coverage of prior evaluations, LIBERO-Plus systematically applies perturbations across seven orthogonal factors: object layout (adding distractors + target displacement), camera viewpoint (pose/orientation/FOV), robot initial state (arm starting pose), language instructions (semantic paraphrasing and increased complexity), lighting (intensity/direction/color), background texture (material/texture replacement), and sensor noise (jitter, Gaussian blur, and other photometric distortions). Each factor is subdivided into 21 total sub-dimensions, all built upon LIBERO's multi-view observations and language instruction scenarios. This "orthogonal multi-factor" design allows single-dimension vulnerabilities to be isolated and attributed rather than obscured within an aggregate score.

**2. Automated Parameterized Generation: Scaling from Manual Cases to Tens of Thousands of Scenarios**

The difficulty of scaling and replicating manual perturbations is the root cause of small sample sizes in existing benchmarks. LIBERO-Plus treats each perturbation dimension as a parameterized automated generator, allowing for the mass construction of training and test sets covering **56K+ robustness scenarios** ⚠️ (refer to original text). This automated pipeline ensures reproducibility and scalability without requiring manual scene-by-scene design.

**3. L1–L5 Progressive Difficulty Stratification: From "One Score" to "Robustness Curves"**

Reporting only aggregate success rates hides the threshold at which a model fails as perturbation intensity increases. The authors calibrate tasks into five progressive levels (L1 to L5) based on the empirical performance of four representative VLAs. This creates a difficulty ladder, allowing each model to be mapped on a "success rate vs. difficulty" curve for every dimension. This accurately captures the intensity level at which a model collapses, serving as a tool for fine-grained failure localization.

**4. Compositional Generalization Gap: Quantifying "Coupled Deterioration" via Statistical Covariance**

Single-dimension perturbations observe isolated factors, but in reality, multiple shifts occur simultaneously and couple with each other. The authors define "compositional generalization" from a statistical perspective: let $D_i$ denote whether the $i$-th perturbation is applied, and $Y$ denote task success. Under the condition $Y=1$, the joint and marginal probabilities of two perturbations are estimated, and the **compositional generalization gap** is defined as their covariance under the success condition:  
$$\Delta_{ij}\triangleq\mathrm{Cov}(D_i,D_j\mid Y=1)=p(D_i{=}1,D_j{=}1\mid Y{=}1)-p(D_i{=}1\mid Y{=}1)\,p(D_j{=}1\mid Y{=}1)$$  
$\Delta_{ij}>0$ indicates that the perturbations can be handled jointly, $\Delta_{ij}<0$ indicates that the combination introduces difficulty beyond independent effects, and $\Delta_{ij}=0$ implies independence. Through 30,000 repeated experiments, $\Delta_{ij}$ was found to be consistently negative, showing that generalization is essentially **non-decomposable**—multi-dimensional shifts act as coupled noise sources in the feature space, exposing entanglement in learned representations.

## Key Experimental Results

### Main Results: Success Rates of 10 VLAs under Various Perturbations (%, Excerpt)

| Model | Original | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|------|----------|--------|-------|----------|-------|-----------|-------|--------|-------|
| OpenVLA | 76.5 | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 97.1 | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0 | 94.2 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-fast | 85.5 | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| Nora | 87.9 | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 79.1 | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | 95.2 | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| RIPT-VLA | 97.5 | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |

Even when performance is near saturation on the original LIBERO (76–97%), total success rates generally halve or worse after perturbations—camera viewpoint and robot initial state are the primary performance killers (OpenVLA drops to 0.8% under camera perturbation, WorldVLA to 0.1%).

### Key Findings
- **Universal Vulnerability (Finding 1–2)**: All VLAs are fragile to perturbations, being most sensitive to camera viewpoints and robot initial states (which require high-level spatial geometric and proprioceptive understanding) and relatively robust to shallow visual changes like lighting and background.
- **Unexpectedly Low Language Impact (Finding 3)**: Language perturbations caused an average drop of only -25.3, which is counter-intuitive. Null-instruction experiments showed that OpenVLA-OFT hardly lost performance on the object suite when language was removed, indicating it "degenerates into a Vision-Action model," treating language as a redundant signal.
- **Position Bias over Semantic Understanding (Finding 5/7/8)**: Models generally do not fail when distractors are added (they focus on the target), but success rates crash when the target object itself **shifts**; success drops to near zero in object replacement tasks—models rely on memorizing positions rather than understanding semantics, continuing to perform the original target's action even when instructions change.
- **Wrist Camera as a Source of Lighting Robustness (Finding 6)**: While success rates collapse to near zero with pitch-black input, models retaining the wrist camera (with the third-person view obscured) still achieve 43.6/43.0/67.3 success rates. This indicates that the close-up wrist view provides light-invariant geometric/contact cues. Models relying solely on third-person views (OpenVLA, Nora, WorldVLA) often lose 60+ points under lighting perturbations.
- **Non-decomposable Compositional Generalization (Finding 9)**: The compositional generalization gap $\Delta_{ij}$ is consistently negative, indicating that coupled multi-dimensional perturbations introduce degradation beyond independent effects.
- **Effectiveness of Augmented Training**: Mixed fine-tuning using 20,000+ successful trajectories constructed via the automated pipeline reached a total success rate of **79.6%**, surpassing all baselines. Camera viewpoint robustness reached **92.8%** (37.2 percentage points higher than the runner-up), with significant gains in noise (89.3%) and layout (77.6%).

## Highlights & Insights
- **Debunking the "Benchmark Score = Capability" Illusion**: By using controlled perturbations to drop success rates from 95% to below 30%, the study proves that existing evaluation protocols severely overestimate real-world VLA capabilities, serving as a powerful warning to the field.
- **Fine-grained Attribution Methodology**: The three-layer design (7-D orthogonal perturbations, L1–L5 stratification, and compositional covariance gaps) makes "when and why a model fails" quantifiable and localizable rather than just providing a single aggregate score.
- **Clever Diagnostic Experiment Design**: Extreme ablations such as null instructions, target replacements, and third-person view obscuration cleanly demonstrate deep mechanisms like "VLAs ignoring language," "positional memory," and "reliance on wrist cameras." These diagnostic paradigms can be directly applied to analyze other embodied models.
- **Evaluation as Data**: The automated generation pipeline produces not only test sets but also 20K+ training trajectories for augmentation. This proves that "targeted diversity training" can significantly enhance robustness, closing the loop between diagnosis and improvement.

## Limitations & Future Work
- The benchmark is entirely based on LIBERO simulation. While perturbations are diverse, the sim-to-real gap of the physical world is not directly evaluated.
- There is a discrepancy in the mentioned task counts (10,030 tasks vs. 56K scenes) in the documentation ⚠️ (likely referring to task instances vs. total scenes including the training set; refer to original text).
- L1–L5 difficulty levels are calibrated based on the empirical performance of four representative models; this stratification might be biased toward the preferences of those specific models.
- Augmented training was only validated on OpenVLA-OFT; its effectiveness across other architectures has not been fully explored.

## Related Work & Insights
- **vs. COLOSSEUM / VLATest**: These also perform automated perturbation generation but lack fine-grained difficulty analysis within each dimension; LIBERO-Plus adds L1–L5 stratification and compositional generalization gaps for deeper insight.
- **vs. RL4VLA / INT-ACT / Gembench**: These rely on manually designed tasks, resulting in sample sizes often <100 and limited dimension coverage. This work provides automated parameterized generation of tens of thousands of scenarios across 7 dimensions.
- **vs. Original LIBERO**: LIBERO provides reproducible evaluation under ideal conditions; this work systematically injects distribution shifts to extend "ideal performance" into a "robustness profile."

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of 7-D perturbations, L1–L5 stratification, and compositional generalization covariance is a first for VLA evaluation, though individual techniques are systematic integrations of existing ideas.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models × 7-D perturbations + 30,000 compositional experiments + multiple diagnostic ablations + augmented training represent a solid scale and depth.
- Writing Quality: ⭐⭐⭐⭐ Findings are well-organized, and diagnostic experiments are elegantly designed.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the critical issue of inflated VLA evaluation; both the benchmark and the findings offer strong guidance for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] MoEActok: A MoE-based Action Tokenizer for Vision-Language-Action Models](moeactok_a_moe-based_action_tokenizer_for_vision-language-action_models.md)
- [\[CVPR 2026\] Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight](mantis_a_versatile_vision-language-action_model_with_disentangled_visual_foresig.md)

</div>

<!-- RELATED:END -->
