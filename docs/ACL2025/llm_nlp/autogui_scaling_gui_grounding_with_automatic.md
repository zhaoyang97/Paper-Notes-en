---
title: >-
  [Paper Note] AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs
description: >-
  [ACL 2025][LLM (Other)][GUI grounding] This work proposes the AutoGUI automatic annotation pipeline. By simulating interactions to compare UI state changes, inferring element functionality with LLMs, and performing dual-LLM verification and filtering, it constructs a high-quality UI functionality dataset of 704K annotations. The annotation accuracy of 96.7% is comparable to humans, significantly enhancing VLM UI grounding capabilities and demonstrating clear data scaling effe…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "GUI grounding"
  - "functionality annotation"
  - "VLM"
  - "automatic pipeline"
  - "UI understanding"
date: 2026-05-08
content_hash: b51050382e9d4027
---

# AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.01977](https://arxiv.org/abs/2502.01977)  
**Code**: [Project Page](https://autogui-project.github.io/)  
**Area**: GUI Understanding / VLM  
**Keywords**: GUI grounding, functionality annotation, VLM, automatic pipeline, UI understanding

## TL;DR

This work proposes the AutoGUI automatic annotation pipeline. By simulating interactions to compare UI state changes, inferring element functionality with LLMs, and performing dual-LLM verification and filtering, it constructs a high-quality UI functionality dataset of 704K annotations. The annotation accuracy of 96.7% is comparable to humans, significantly enhancing VLM UI grounding capabilities and demonstrating clear data scaling effects.

## Background & Motivation

**Background**: VLMs hold great potential for UI understanding, but they are bottlenecked by data shortage—UI datasets are significantly smaller in scale compared to natural image datasets. **Limitations of Prior Work**: Existing annotations mostly consist of element alt-text or simple user intentions, lacking contextualized functional semantic descriptions (e.g., two identical magnifying glass icons could represent "search" and "zoom" respectively). **Key Challenge**: High-quality, large-scale contextualized UI element functionality annotations are needed, but human annotation is prohibitively expensive. **Goal**: Design a fully automatic annotation pipeline to generate high-quality functional descriptions of UI elements at scale without human intervention. **Key Insight**: Infer functionality by looking at "what happens to the UI after an element is clicked"—similar to how humans explore an unfamiliar interface. **Core Idea**: Compare UI state changes before and after interactions using LLMs to automatically infer element functionality, combined with a dual-LLM quality control mechanism to ensure accuracy.

## Method

### Overall Architecture

A three-stage pipeline: (1) Automatically crawl Web/Android UI interaction trajectories; (2) Perform LLM functionality inference, LLM-aided rejection, and dual-LLM verification; (3) Convert data into grounding/referring tasks to fine-tune VLMs.

### Key Designs

1. **Based on UI State Change Functionality Inference**:
    - **Function**: Use differences in UI AXTree before and after interactions to infer element functionality.
    - **Mechanism**: $paper_notes/docs/ACL2025/multilingual_mt/cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md = 	ext{LLM}(p_{	ext{anno}}, s_t, s_{t+1})$, using `difflib` to generate line-level differences (additions, deletions, shifts, and attribute updates) in the AXTree. The LLM analyzes these changes and summarizes the functionality via Chain-of-Thought.
    - **Design Motivation**: Instead of relying solely on the visual appearance of an element, this method prioritizes "what happens after clicking." For instance, if clicking a magnifying glass icon opens a search bar, its function is search; if it displays a zoom slider, its function is zoom.

2. **LLM-aided Rejection (Invalid Sample Filtering)**:
    - **Function**: The LLM evaluates whether the state changes produced by the interaction are sufficient to infer functionality.
    - **Mechanism**: $	ext{score} = 	ext{LLM}(p_{	ext{reject}}, e, s_t, s_{t+1})$, scoring based on three criteria (clarity of change, relevance, and predictability) and discarding the bottom 30%.
    - **Design Motivation**: Not all interactions produce meaningful state changes, such as pages that fail to load properly or redirects that require logging in.

3. **Dual-LLM Verification (Annotation Quality Control)**:
    - **Function**: Use two different LLMs to cross-validate whether the functionality annotation is correct.
    - **Mechanism**: Llama-3-70B and Mistral-7B separately score the annotations, retaining only those that receive full marks from both.
    - **Design Motivation**: A single LLM may exhibit systematic biases (such as incorrectly describing dropdown menus). Cross-validation with two different LLMs allows them to correct each other.

### Loss & Training

VLMs are fine-tuned using LoRA, training for 1 epoch on 8×A100. Qwen-VL and Qwen2-VL are fine-tuned with LoRA, while SliME only freezes the visual encoder. Coordinates are normalized to [0, 999].

## Key Experimental Results

### Main Results

Fine-tuned UI grounding accuracy (%):

| Model | FuncPred | ScreenSpot | ScreenSpot-v2 | MoTIF | VWB EG |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen-VL (Baseline) | 3.0 | 5.2 | 5.6 | 7.8 | 1.7 |
| Qwen-VL + AutoGUI | **48.7**(+45.7) | 41.2(+36.0) | 40.2(+34.6) | 44.0(+36.2) | 42.1(+40.4) |
| Qwen2-VL (Baseline) | 38.7 | 66.4 | 66.9 | 71.1 | 55.9 |
| Qwen2-VL + AutoGUI | **65.0**(+26.3) | 80.0(+13.6) | 83.2(+16.3) | 72.3(+1.2) | **90.3**(+34.4) |

### Ablation Study

Comparison of annotation quality vs. human annotators:

| Annotator | Rejector | Verifier | Accuracy |
|--------|----------|----------|:---:|
| Llama-3-70B | None | None | 64.5% |
| Llama-3-70B | Rule+LLM | Dual-LLM | **96.7%** |
| Human Annotator | - | - | 95.5% |

Data scaling effect: From 25k $\rightarrow$ 125k $\rightarrow$ 702k, the grounding accuracy of the three VLMs continues to rise.

### Key Findings

1. **Functionality annotations significantly outperform HTML/metadata annotations**: Scoring up to 4x higher than HTML annotations on FuncPred.
2. **Clear data scaling effect**: Increasing the volume of data leads to continuous performance improvements.
3. **Assisting GUI agent tasks**: Qwen2-VL+AutoGUI improves over Gemini's native grounding by 9.73% on AITW.
4. **Annotation accuracy of 96.7% is comparable to trained human annotators (95.5%)**.

## Highlights & Insights

- **The design idea of "inferring functionality via interaction differences" is highly clever**—seamlessly mimicking the way humans explore unfamiliar interfaces.
- **Dual-LLM quality control** ensures that the fully automatic pipeline achieves quality comparable to human annotation.
- **Data scaling effects** indicate that the pipeline is highly scalable.
- The finding that **functionality annotations > HTML annotations** demonstrates the immense value of contextualized semantic descriptions.

## Limitations & Future Work

- Only Web and Android platforms are covered, leaving iOS and desktop applications unaddressed.
- The current pipeline cannot annotate elements that modify remote internet content (e.g., posting or purchasing).
- Lack of goal-oriented interaction trajectories (only random interactions are used).
- Some low-traffic websites experience rendering distortion under mobile resolutions.

## Related Work & Insights

- **vs SeeClick (Cheng et al. 2024)**: SeeClick uses only static pages and HTML annotations—Ours leverages interaction differences and LLM inference.
- **vs Widget Captioning (Li et al. 2020)**: Relies on 163K human annotations—Ours is fully automatic, scaling to 704K with comparable quality.
- **vs UGround (Gou et al. 2025)**: Provides only brief functional descriptions—Ours provides contextualized and detailed functionality annotations.
- **Insight**: The data bottleneck in UI understanding can be thoroughly resolved through automatic interaction and LLM inference.

## Rating

- Novelty: ⭐⭐⭐⭐ The first fully automatic UI functionality annotation pipeline; the concept of inferring functionality via interaction differences is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 704K dataset + multiple VLMs + scalability + agent applications.
- Writing Quality: ⭐⭐⭐⭐ The description of the pipeline is clear and well-structured.
- Value: ⭐⭐⭐⭐⭐ Resolves the data bottleneck in UI understanding, offering foundational infrastructure value for GUI agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Growing Through Experience: Scaling Episodic Grounding in Language Models](episodic_grounding_experience.md)
- [\[ACL 2025\] AutoExp: Automatic Experiment Design and Execution by LLMs](autoexp_automatic_experiment_design_and_execution_by_llms.md)
- [\[ACL 2025\] Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel](mitigate_position_bias_in_large_language_models_via_scaling_a_single_dimension.md)
- [\[ACL 2025\] NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews](newsinterview_a_dataset_and_a_playground_to_evaluate_llms_grounding_gap_via_info.md)
- [\[ACL 2025\] LESA: Learnable LLM Layer Scaling-Up](lesa_learnable_llm_layer_scaling-up.md)

</div>

<!-- RELATED:END -->
