---
title: >-
  [Paper Note] PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset
description: >-
  [CVPR 2025][Hallucination Detection][Visual Hallucination] This paper proposes PhD, a large-scale visual hallucination evaluation dataset constructed with the assistance of ChatGPT. It contains 14K+ everyday images, 750 counter-commonsense images, and 102K VQA triplets. Through 4 evaluation modes $\times$ 5 visual tasks, it systematically evaluates the hallucination issues of multimodal large language models (MLLMs), far exceeding existing benchmarks in scale and difficulty.
tags:
  - "CVPR 2025"
  - "Hallucination Detection"
  - "Visual Hallucination"
  - "Multimodal Large Language Models"
  - "Evaluation Benchmark"
  - "ChatGPT-Assisted Construction"
  - "Binary VQA"
date: 2026-05-08
content_hash: 817435fb90de036e
---

# PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset

**Conference**: CVPR 2025  
**arXiv**: [2403.11116](https://arxiv.org/abs/2403.11116)  
**Code**: [https://github.com/jiazhen-code/PhD](https://github.com/jiazhen-code/PhD)  
**Area**: Hallucination Detection  
**Keywords**: Visual Hallucination, Multimodal Large Language Models, Evaluation Benchmark, ChatGPT-Assisted Construction, Binary VQA

## TL;DR

This paper proposes PhD, a large-scale visual hallucination evaluation dataset constructed with the assistance of ChatGPT. It contains 14K+ everyday images, 750 counter-commonsense images, and 102K VQA triplets. Through 4 evaluation modes $\times$ 5 visual tasks, it systematically evaluates the hallucination issues of multimodal large language models (MLLMs), far exceeding existing benchmarks in scale and difficulty.

## Background & Motivation

1. **Background**: Multimodal Large Language Models (MLLMs) such as LLaVA and Qwen-VL have demonstrated excellent performance on diverse visual tasks, but they universally suffer from visual hallucination—generating descriptions that are inconsistent with the visual content. Visual Hallucination Evaluation (VHE) is an emerging and critical research direction.
2. **Limitations of Prior Work**: (a) Pioneering datasets like POPE have small scales (only 3K VQA triplets), single tasks (only object recognition), and less image-specific hallucinatory item (hitem) selection; (b) AMBER relies on fully manual annotation, which is costly, vocabulary-restricted, and difficult to scale; (c) Existing datasets saturate rapidly in performance (Figure 1(d)), failing to differentiate evolving models.
3. **Key Challenge**: Effective hallucination evaluation requires selecting hitems that are both image-specific and challenging. However, hitem selection has rarely been systematically studied in prior work—POPE/ROME select based on label co-occurrence (not image-specific), while NOPE/CIEM completely skip hitem selection.
4. **Goal**: Construct a large-scale, multi-mode, multi-task VHE dataset that explicitly links to the three major causes of MLLM visual hallucination (visual ambiguity, inconsistent multimodal input, and counter-commonsense content), supporting fine-grained analysis of models.
5. **Key Insight**: Leverage a ChatGPT-assisted semi-automated dataset construction pipeline, using CLIP to rank hitems based on visual similarity, thereby generating VQA samples with image-specific challenges.
6. **Core Idea**: Design four evaluation modes (PhD-base, PhD-sec, PhD-icc, PhD-ccs) around the three major causes of hallucination, covering five visual tasks (object/attribute/emotion/position recognition + counting), and utilize ChatGPT to assist in generating hitems, questions, and contexts.

## Method

### Overall Architecture

PhD is remodeled from the annotations of the TDIUC dataset and constructed through four core modules: (1) task-specific hitem selection $\rightarrow$ (2) hitem-embedded question generation $\rightarrow$ (3) plausible/incorrect context generation $\rightarrow$ (4) counter-commonsense (CCS) image generation. The final dataset supports 4 modes $\times$ 5 tasks = 20 evaluation combinations, totaling 102K VQA triplets. TDIUC was chosen because its images originate from MS-COCO (which MLLMs are likely to have seen during training), making the models' erroneous responses more likely attributable to hallucination rather than a lack of capability.

### Key Designs

1. **ChatGPT+CLIP Assisted hitem Selection**:
    - **Function**: Select image-specific, challenging hallucinatory items for each visual task on each image.
    - **Mechanism**: Taking color attribute recognition as an example—(a) use ChatGPT to expand the color vocabulary (from a few manual entries $\rightarrow$ automatically expanded to 35 colors); (b) use ChatGPT to extract the subject and attribute (e.g., "motorcycle" and "black") from TDIUC QA pairs; (c) exclude ground truth and synonyms to obtain candidate hitems; (d) use CLIP to compute the cosine similarity between each candidate hitem+subject combination and the image, selecting the visually most deceptive one as the hitem. Finally, perform manual spot-checks. In total, 1,452 diverse and challenging hitems were selected.
    - **Design Motivation**: CLIP-based ranking ensures that hitems "look plausible but are actually incorrect," which is more effective at inducing hallucinations than random or co-occurrence-based selection.

2. **Design of Four Evaluation Modes**:
    - **Function**: Evaluate MLLM performance under three typical causes of hallucination, respectively.
    - **Mechanism**: (a) **PhD-base** (VQA on everyday images without context)—tests hallucinations caused by visual ambiguity (Cause I); (b) **PhD-sec** (with plausible context added) and **PhD-icc** (with incorrect context added)—test hallucinations caused by inconsistent multimodal inputs (Cause II); (c) **PhD-ccs** (VQA on counter-commonsense images)—tests hallucinations when internal knowledge conflicts with visual content (Cause III). Plausible contexts are generated by ChatGPT, requiring them to be related to the image but not necessarily reflecting the current state; incorrect contexts are directly contradictory. CCS images are generated by Doubao and DALL-E 3 (e.g., "a car with square tires").
    - **Design Motivation**: Existing datasets lack explicit evaluations for different causes of hallucination. The mode-task structure of PhD allows precise localization of model weaknesses.

3. **PhD Index Evaluation Metric**:
    - **Function**: Provide balanced hallucination evaluation scores.
    - **Mechanism**: Calculate the recall for "Yes" questions and "No" questions separately, and take their harmonic mean as the PhD Index. A model that only answers "Yes" (or "No") scores 0, while random guessing scores 0.5. This ensures a balanced evaluation against affirmative/negative biases.
    - **Design Motivation**: Avoid models obtaining high scores by simply outputting "Yes" or "No" consistently, truly testing their visual understanding.

### Loss & Training

- This work is an evaluation dataset and does not involve model training.
- In the dataset construction pipeline, ChatGPT is used to generate hitems, questions, and contexts; CLIP is used for hitem sorting; and AIGC tools are used for CCS image generation.
- Human participation is mainly concentrated in spot-checking and validation, making the overall pipeline semi-automated.

## Key Experimental Results

### Main Results

**Overall VHE of Open-Source Models (PhD Index):**

| Model | ViT | LLM | POPE | AMBER | PhD Index |
|------|-----|-----|------|-------|-----------|
| LLaVA-OneVision | SoViT-400m/14 | Qwen2-72B | 0.84 | 0.90 | 0.698 |
| Molmo | -L/14 | Qwen2-72B | 0.84 | 0.85 | 0.690 |
| InternVL-1.5 | InternViT | InternLM2-20B | - | - | ~0.65 |
| LLaVA-1.5 | CLIP-L/14 | Vicuna-7B | ~0.80 | ~0.82 | 0.265 |
| LLaVA-1.5-L | CLIP-L/14 | Vicuna-13B | ~0.80 | ~0.82 | 0.270 |

### Ablation Study

| Evaluation Mode | Representative Findings | Description |
|---------|-----------|------|
| PhD-base | Models perform best | No distraction, only testing visual ambiguity |
| PhD-sec (Plausible Context) | Significant drop | Models are easily misled by plausible but inaccurate text |
| PhD-icc (Incorrect Context) | Larger drop | Models are heavily biased toward textual information |
| PhD-ccs (Counter-Commonsense Images) | Most challenging | Models rely on internal knowledge instead of visual content |

### Key Findings

- **PhD is far more challenging than POPE/AMBER**: The LLaVA series achieves 0.84 on POPE and 0.90 on AMBER, but only 0.265–0.698 on PhD, effectively distinguishing model capabilities.
- **Larger LLMs are not necessarily better**: LLaVA-1.5-L (13B) and LLaVA-1.5 (7B) show virtually no difference on PhD (0.270 vs 0.265), indicating that parameter size is not the decisive factor.
- **Models universally exhibit textual bias**: Performance drops drastically in PhD-sec and PhD-icc modes, demonstrating that the LLM backbones of MLLMs tend to trust textual inputs over visual inputs.
- **Counter-commonsense scenarios are the biggest weakness**: PhD-ccs exposes the models' over-reliance on common-sense knowledge learned during training, preventing them from making judgments based on actual visual content.
- **Hallucination mitigation methods like VCD and Woodpecker have limited effectiveness**: This indicates that current mitigation strategies are far from sufficient to solve the hallucination problem.

## Highlights & Insights

- **The semi-automated hitem selection pipeline using ChatGPT+CLIP** is the biggest innovation: ChatGPT scales vocabulary and generates text (zero-cost expansion), while CLIP ranks visual correlation (ensuring image-specific challenges), and humans only validate. This achieves a balance of scale and quality. This pipeline can be transferred to any scenario requiring the construction of adversarial evaluation sets.
- **The structured evaluation framework of 4 modes $\times$ 5 tasks** is highly valuable: It not only provides an overall score but also precisely localizes in which cause of hallucination and on which visual task the model is weakest, offering clear directions of improvement for model developers.
- **The deliberate design of using MS-COCO images (which may have been seen during training)** is clever: If a model still hallucinates on a "seen" image, it is more indicative of a hallucination issue rather than a capability deficiency.

## Limitations & Future Work

- **By-product of TDIUC annotations**: The images and initial annotations of the dataset are sourced from TDIUC/MS-COCO, where potential annotation errors could affect quality.
- **Support for binary VQA only**: Yes/No QA, while convenient for large-scale evaluation, cannot capture more complex hallucination patterns (such as partially correct descriptions).
- **Unstable quality of CCS images**: The counter-commonsense images generated by AIGC tools vary in quality, and some might not be realistic enough.
- **English-centric**: The TDIUC data and ChatGPT-generated texts are all in English, leading to insufficient evaluation coverage for multilingual MLLMs.
- **Future Directions**: Extend to open-ended VQA evaluation; add more advanced visual tasks (e.g., visual reasoning); construct hallucination evaluation datasets targeted at specific domains (e.g., medical).

## Related Work & Insights

- **vs POPE**: POPE contains only 3K samples, covers only object recognition, and selects hitems based on co-occurrence (not image-specific). PhD is 34 times larger (102K), covers 5 tasks, and employs image-specific hitem selection.
- **vs AMBER**: AMBER relies on fully manual annotation (687 hitems) and has 14K VQA samples. PhD is constructed semi-automatically (1,452 hitems) with 102K VQA samples, offering superior scalability.
- **vs HallusionBench**: HallusionBench focuses on subjective evaluation of high-level visual reasoning; PhD focuses on objective evaluation of low-to-mid-level visual recognition. The two are complementary.
- **vs MMMU**: MMMU evaluates advanced academic subject knowledge understanding instead of hallucinations; PhD handles hallucination evaluation specifically, where even capable models may fail due to hallucinations.

## Rating

- Novelty: ⭐⭐⭐⭐ The designed four-mode evaluation framework and the ChatGPT+CLIP semi-automated construction pipeline are both innovative, and the explicit linkage to the three hallucination causes is a unique contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 15 open-source models + 3 commercial models + 2 mitigation methods, with thorough multi-dimensional analysis across modes, tasks, and models.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, the dataset construction pipeline is described in detail, and the tables and diagrams are highly informative.
- Value: ⭐⭐⭐⭐ As currently the largest and most challenging VHE benchmark, it holds significant evaluation value for the MLLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target](stop_learning_it_all_to_mitigate_visual_hallucination_focus_on_the_hallucination.md)
- [\[ACL 2025\] TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation](../../ACL2025/hallucination/treecut_a_synthetic_unanswerable_math_word_problem_dataset_for_llm_hallucination.md)
- [\[CVPR 2025\] 3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination](3d-grand_a_million-scale_dataset_for_3d-llms_with_better_grounding_and_less_hall.md)
- [\[CVPR 2025\] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow](../../ICLR2026/hallucination/visual_multi-agent_system_mitigating_hallucination_snowballing_via_visual_flow.md)

</div>

<!-- RELATED:END -->
