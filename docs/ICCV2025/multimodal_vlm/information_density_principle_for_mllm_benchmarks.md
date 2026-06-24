---
title: >-
  [Paper Note] Information Density Principle for MLLM Benchmarks
description: >-
  [ICCV 2025][Multimodal VLM][Benchmark evaluation] This paper proposes an "information density" principle to evaluate MLLM benchmark quality along four dimensions — Fallacy, Difficulty, Redundancy, and Diversity — and constructs a three-tier automated evaluation pipeline (Human–Model–Data) to conduct a systematic "benchmark for benchmark" analysis of 19 mainstream benchmarks.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Benchmark evaluation"
  - "information density"
  - "MLLM assessment"
  - "benchmark quality"
  - "meta-evaluation"
date: 2026-05-08
content_hash: 0b5bd50816eb3be0
---

# Information Density Principle for MLLM Benchmarks

**Conference**: ICCV 2025
**arXiv**: [2503.10079](https://arxiv.org/abs/2503.10079)  
**Code**: [GitHub](https://github.com/lcysyzxdxc/bench4bench)  
**Area**: Multimodal VLM
**Keywords**: Benchmark evaluation, information density, MLLM assessment, benchmark quality, meta-evaluation

## TL;DR

This paper proposes an "information density" principle to evaluate MLLM benchmark quality along four dimensions — Fallacy, Difficulty, Redundancy, and Diversity — and constructs a three-tier automated evaluation pipeline (Human–Model–Data) to conduct a systematic "benchmark for benchmark" analysis of 19 mainstream benchmarks.

## Background & Motivation

With the rapid development of multimodal large language models (MLLMs), there are now over 300 MLLM benchmarks, leaving developers with two major challenges:

**Selection difficulty**: Faced with a vast number of benchmarks, practitioners struggle to identify which ones best reveal model strengths and weaknesses.

**Unreliable evaluation mechanisms**: Many benchmarks suffer from the following flaws:
   - **Fallacy**: Questions or annotations are inherently incorrect, yielding unreliable information.
   - **Difficulty**: Questions are too easy, allowing nearly all models to answer correctly, thus providing no meaningful discrimination.
   - **Redundancy**: Questions can be answered correctly using only a subset of the input (e.g., text alone without the image), rendering additional modalities redundant.
   - **Diversity**: Multiple samples probe the same type of question, causing information overlap.

Core problem: No prior work has systematically evaluated the evaluation benchmarks themselves — as assessment mechanisms, benchmarks themselves require rigorous scrutiny.

## Method

### Overall Architecture

The paper establishes a theoretical foundation for "information density" grounded in information theory, decomposing the abstract notion of "informativeness" into the product of four quantifiable dimensions:

$$E(I) \propto (1 - D_{fal}) \cdot D_{dif} \cdot (1 - D_{red}) \cdot D_{div}$$

where $D_{fal}$ is the fallacy rate, $D_{dif}$ is difficulty, $D_{red}$ is redundancy, and $D_{div}$ is diversity. Higher information density indicates greater value for MLLM developers.

A three-tier evaluation paradigm is constructed:
- **Human Eval** (highest cost, highest precision): Expert human annotation serving as ground truth.
- **Model Eval** (moderate cost): MLLM inference results used to reflect data quality.
- **Data Eval** (lowest cost): Direct analysis of data characteristics without model inference.

### Key Designs

1. **Difficulty Evaluation**:

    - **Model Eval**: Three models — GPT-4o, InternVL-2.5, and QwenVL-2.5 — vote to define three sub-dimensions: Junior (at least one model answers incorrectly), Extreme (all models answer incorrectly), and Ambiguity (the best and alternative answers are inconsistent across models).
    - $D_{dif} = P(Q_{jun}) + P(Q_{amb})$
    - **Data Eval**: Four features — image structural complexity (2D Laplacian), text syntactic depth (parse tree), option semantic distance (CLIP distance), and focus region size (entropy of syntactic root nodes) — are used to fit Model Eval results via regression.

2. **Fallacy Evaluation** (Human Eval only):

    - Among difficult samples identified by Difficulty evaluation, human experts annotate three types of fallacies: Question (the question itself is erroneous), Annotation (the annotation is incorrect but other valid options exist), and Ambiguity (multiple options are equally plausible).
    - $D_{fal} = P((Q_{que} + Q_{ano} + Q_{amb}) | D_{dif}=1)$

3. **Redundancy Evaluation**:

    - **Model Eval**: The image or text is individually removed and the model is prompted to answer; correct responses under the ablated input indicate that the removed modality is redundant.
    - $D_{red} = \frac{w_{img} \cdot \mathrm{Acc}(\overline{I_{img}}) + w_{txt} \cdot \mathrm{Acc}(\overline{I_{txt}})}{w_{img} + w_{txt}}$
    - QwenVL-2.5 is used for inference, as other models tend to refuse incomplete inputs.

4. **Diversity Evaluation**:

    - **Model Eval**: CLIP encoders are used to cluster and deduplicate image and text samples; the proportion of remaining samples constitutes the diversity score.
    - $D_{div} = \frac{w_{img} \cdot \frac{\#(\mathrm{SIM}(I_{img}))}{\#(I_{img})} + w_{txt} \cdot \frac{\#(\mathrm{SIM}(I_{txt}))}{\#(I_{txt})}}{w_{img} + w_{txt}}$
    - **Data Eval**: Image diversity is measured by the distributional variance of five low-level features (brightness, contrast, color, blur, texture); text diversity is measured by the coverage rate of 10 interrogative word types.

### Loss & Training

This paper presents an evaluation methodology and does not involve model training. Linear regression is used in Data Eval to fit Model Eval results.

## Key Experimental Results

### Main Results (Information Density Comparison Across 19 Benchmarks)

| Benchmark | Fallacy↓ | Difficulty↑ | Redundancy↓ | Diversity↑ | Release Date |
|-----------|----------|------------|-------------|-----------|---------|
| MMStar | **0.135** | **0.546** | **0.054** | 0.827 | Mar-2024 |
| Q-Bench | 0.280 | 0.373 | 0.175 | **0.951** | Sep-2023 |
| RealWorldQA | 0.247 | 0.379 | 0.113 | 0.756 | Apr-2024 |
| HallusionBench | 0.269 | 0.465 | 0.312 | 0.191 | Oct-2023 |
| POPE | 0.557 | 0.119 | 0.562 | 0.383 | May-2023 |
| MME | 0.526 | 0.206 | 0.133 | 0.842 | Jun-2023 |
| A-okvqa | 0.597 | 0.157 | 0.243 | 0.882 | Jun-2022 |

### Correlation Between Model/Data Eval and Human Eval

| Dimension | Model Eval Pearson r | Data Eval Pearson r |
|------|---------------------|---------------------|
| Difficulty | >0.7 | >0.7 |
| Redundancy | >0.7 | - |
| Diversity (Image) | >0.8 | >0.7 |
| Diversity (Text) | >0.7 | >0.7 |

### Key Findings

- **MMStar achieves the best overall performance**: It attains the lowest fallacy rate (0.135), the highest difficulty (0.546), and the lowest redundancy (0.054), making it the highest-information-density benchmark currently available.
- **Early benchmarks exhibit widespread issues**: POPE (2023.5) has a redundancy rate as high as 0.562 and a diversity score of only 0.383; A-okvqa (2022.6) has a fallacy rate of 0.597.
- **Newer benchmarks show improvement but still leave room**: Benchmarks released in 2024 generally outperform earlier versions across all dimensions, yet none achieves optimal performance on all four simultaneously.
- **Model/Data Eval correlations with Human Eval all exceed 0.7**, validating the soundness of the automated evaluation pipeline.

## Highlights & Insights

- **Novel meta-evaluation perspective**: Systematically evaluating evaluation mechanisms themselves is a previously overlooked but critically important direction; this work is the first to formalize it.
- **Solid information-theoretic foundation**: The four dimensions are unified under an information entropy framework rather than being an ad-hoc aggregation of metrics.
- **Practical three-tier design**: The pipeline spans from fully manual to fully automated, allowing benchmark developers to select the appropriate level based on available resources.
- **Valuable findings on Redundancy**: The analysis reveals that the "multimodal" nature of many benchmarks is illusory — numerous questions can be answered correctly from text alone.

## Limitations & Future Work

- The Fallacy dimension requires human annotation and cannot be automated, limiting large-scale applicability.
- Only MCQ-format benchmarks are evaluated; open-ended VQA benchmarks are not yet covered.
- Redundancy Model Eval relies solely on QwenVL-2.5 (as other models refuse to process incomplete inputs), which may introduce bias.
- Benchmark timeliness and data contamination (training data leakage) are not addressed.

## Related Work & Insights

- The information density framework can serve as a design guideline for new benchmark development, enabling self-assessment across the four dimensions prior to publication.
- The Redundancy detection method (removing a modality and evaluating answerability) is a generalizable quality check for multimodal data.
- For MLLM developers: benchmarks with high information density such as MMStar and Q-Bench are recommended as primary evaluation tools.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First systematic "benchmark for benchmark" study, opening a new research direction)
- Technical Depth: ⭐⭐⭐⭐ (Information-theoretic derivation + three-tier automated pipeline design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (19 benchmarks, 17,912 samples, multi-dimensional comparison)
- Value: ⭐⭐⭐⭐⭐ (Directly guides benchmark selection and development)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Redundancy Principles for MLLMs Benchmarks](../../ACL2025/multimodal_vlm/redundancy_principles_for_mllms_benchmarks.md)
- [\[ICCV 2025\] Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles](calibrating_mllm-as-a-judge_via_multimodal_bayesian_prompt_ensembles.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] Pi-GPS: Enhancing Geometry Problem Solving by Unleashing the Power of Diagrammatic Information](pi-gps_enhancing_geometry_problem_solving_by_unleashing_the_power_of_diagrammati.md)
- [\[ICCV 2025\] PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting](physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)

</div>

<!-- RELATED:END -->
