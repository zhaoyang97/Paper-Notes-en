---
title: >-
  [Paper Note] SlideTailor: Personalized Presentation Slide Generation for Scientific Papers
description: >-
  [AAAI 2026][Recommender Systems][Personalized slide generation] This paper defines a new task of preference-guided paper-to-slide generation and proposes the SlideTailor framework…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "Personalized slide generation"
  - "preference distillation"
  - "chain-of-speech"
  - "agent framework"
  - "academic paper presentation"
date: 2026-05-08
content_hash: 69d67a26aef6deb1
---

# SlideTailor: Personalized Presentation Slide Generation for Scientific Papers

**Conference**: AAAI 2026
**arXiv**: [2512.20292](https://arxiv.org/abs/2512.20292)  
**Code**: [SlideTailor](https://github.com/nusnlp/SlideTailor)  
**Area**: Document-to-Presentation Generation
**Keywords**: Personalized slide generation, preference distillation, chain-of-speech, agent framework, academic paper presentation

## TL;DR

This paper defines a new task of preference-guided paper-to-slide generation and proposes the SlideTailor framework, which distills content preferences from user-provided paper–slide example pairs and aesthetic preferences from .pptx templates. A chain-of-speech mechanism aligns slide content with intended spoken narratives. On the self-constructed PSP benchmark, SlideTailor achieves an overall score of 75.8% and a human-evaluation win rate of 81.63%, significantly outperforming existing methods.

## Background & Motivation

**Background**: Automatic presentation generation is an active research area. Existing methods (DOC2PPT, PPTAgent) have begun integrating textual and visual elements for multimodal presentation generation with notable progress.

**Limitations of Prior Work**:
(1) **Neglect of user subjectivity** — existing methods treat slide generation as a straightforward document-to-slide conversion, failing to accommodate individual users' preferences regarding narrative structure, content emphasis, and visual style;
(2) **Difficulty in expressing preferences** — requiring users to describe preferences in detail is unnatural and burdensome; Persona-Aware-D2S supports only four fixed preference categories, which cannot cover the diversity of real-world needs;
(3) **Disconnect between content and delivery** — existing methods do not consider alignment between slide content and spoken narration, making generated slides difficult to use directly in presentations.

**Key Insight**: Users need only provide one paper–slide example pair (encoding content preferences) and one .pptx template (encoding aesthetic preferences); the system automatically distills implicit preferences and generates personalized slides.

## Method

### Overall Architecture

SlideTailor adopts a three-stage agent framework that mimics the human slide-creation process:
(1) **Implicit preference distillation**: extracts structured preferences $P = P_C \cup P_A$ from the example pair and template;
(2) **Preference-guided slide planning**: includes conditional paper reorganization, chain-of-speech outline design, and template selection;
(3) **Slide realization**: generates editable .pptx files via layout-aware editing and code execution.

### Key Designs

1. **Dual-Branch Implicit Preference Distillation**

    - **Function**: Extracts structured, interpretable preference representations from unannotated user inputs.
    - **Content preference distillation**: Models $f_{content}: D_{ref} \to S_{ref}$ as a latent function and employs an LLM (GPT-4.1) to infer how content is selected, emphasized, omitted, and reordered, producing structured preferences $P_C$ (narrative flow, and section-level detail, emphasis, and format preferences).
    - **Aesthetic preference distillation**: Employs a VLM to infer the functional roles of slide-level and element-level components within the template, combined with raw .pptx metadata (position/size), to produce a layout schema $P_A$.
    - **Design Motivation**: Decoupling the two preference types yields modular flexibility — any aesthetic template can be combined with any content preference.

2. **Chain-of-Speech Mechanism**

    - **Function**: Simultaneously drafts a spoken script during outline design, aligning slide content with the intended spoken narrative.
    - **Mechanism**: Inspired by how human presenters rehearse their narration while creating slides, the system simulates spoken delivery when planning each slide. This produces clearer and more coherent slide content while naturally generating a speaking script usable for downstream video presentations.
    - **Design Motivation**: Effective slides serve not merely as information displays but as visual aids for spoken delivery. Simultaneous script generation ensures content selection is driven by the needs of oral communication.

3. **Template-Aware Layout Selection and Editing**

    - **Function**: Matches each slide to the most appropriate template layout based on semantic content and generates editable .pptx files via a code agent.
    - **Mechanism**: Performs per-slide matching based on the aesthetic preference schema $P_A$ (e.g., text-heavy layouts for content-dense slides, mixed layouts for figure slides), then maps content to specific elements (title boxes, text boxes, image placeholders) via a layout-aware agent, and finally generates Python scripts through a code agent to directly edit the .pptx file.
    - **Design Motivation**: Preserves the original template's layout and theme while producing a fully editable standard-format file that users can subsequently modify.

### Downstream Applications

Leveraging the speaking scripts produced via chain-of-speech, a zero-shot TTS system (e.g., MegaTTS 3) can synthesize personalized narration in the user's voice. Combined with visual slides, this enables automatic generation of presentation videos, and audio-driven digital human avatars can be further integrated to enhance immersion.

## Key Experimental Results

### Main Results — PSP Benchmark Comparison

| Method | Coverage↑ | Flow↑ | Content Structure↑ | Aesthetic↑ | Overall↑ |
|--------|-----------|-------|-------------------|-----------|---------|
| ChatGPT | 62.62 | 56.84 | 61.60 | 80.80 | 62.86 |
| AutoPresent (GPT-4.1) | 72.84 | 59.58 | 49.60 | 22.40 | 48.78 |
| PPTAgent (GPT-4.1) | 64.41 | 54.24 | 57.60 | 97.20 | 67.30 |
| SlideTailor (Qwen2.5) | 70.19 | 62.16 | 68.41 | 92.80 | 69.21 |
| **SlideTailor (GPT-4.1)** | **74.47** | **66.65** | **72.80** | **98.00** | **75.80** |

### Ablation Study

| Configuration | Coverage | Flow | Content Structure | Content | Overall |
|---------------|----------|------|-------------------|---------|---------|
| w/o content preference | 65.80 (−9.0) | 56.83 (−11.6) | 54.67 (−11.3) | 65.73 | 68.61 |
| w/o chain-of-speech | 73.60 | 63.99 | 66.00 | 47.33 (−19.1) | 69.91 |
| **Full system** | **74.82** | **68.38** | **66.00** | **66.40** | **74.31** |

### Human Evaluation

Four graduate students evaluated 30 cases (2 annotators per case). SlideTailor achieves a win rate of **81.63%** against PPTAgent. The average Pearson correlation between human evaluations and MLLM assessments is 0.64.

### Key Findings

- No method exceeds an overall score of 80%, indicating that preference-guided slide generation remains an open challenge.
- Removing content preference distillation reduces Coverage, Flow, and Content Structure by approximately 10% each, validating the central importance of preference modeling.
- Removing chain-of-speech causes a sharp drop of 19.1% in content quality (66.4%→47.3%), demonstrating that spoken-narrative alignment is critical for content quality.
- The open-source Qwen2.5 variant achieves an overall score of 69.21%, surpassing all baselines and demonstrating the framework's cross-model adaptability.
- Per-run generation cost for a 10-slide deck: $0.665 for the GPT variant and only $0.016 for the Qwen variant.

## Highlights & Insights

1. **Valuable task definition**: Preference-guided slide generation addresses the core pain point of presentation authoring — subjectivity.
2. **Elegant implicit preference distillation**: Users need not write descriptions; they simply provide natural example pairs, and the system automatically mines their preferences.
3. **Chain-of-speech serves dual purposes**: It simultaneously improves slide quality and produces a speaking script, enabling downstream applications such as video presentations.
4. **Thoughtful PSP benchmark design**: 200 papers × 50 example pairs × 10 templates = 100,000 possible combinations, spanning AI, medicine, chemistry, and other domains.

## Limitations & Future Work

1. The benchmark is limited to academic papers and does not cover business reports, educational materials, or other domains.
2. The purely zero-shot framework may be further improved through end-to-end multimodal training.
3. MLLM evaluators exhibit self-bias (e.g., GPT tends to score GPT-generated outputs higher), and the reliability of the evaluation protocol warrants further improvement.
4. Template matching is heuristic-based and may be insufficiently precise for complex layouts involving multiple figures and tables.

## Related Work & Insights

- The implicit preference distillation paradigm is generalizable to other personalized content generation scenarios, such as personalized summarization and adaptive report generation.
- The chain-of-speech concept is transferable to the education domain, e.g., automatic generation of annotated instructional slides.
- The agent-driven multi-stage generation paradigm (distillation → planning → realization) provides a general template for complex document processing.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐⭐: New task definition with dual innovations in preference distillation and chain-of-speech.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Quantitative evaluation, ablation study, and human assessment are all present, though the number of evaluated cases is relatively limited.
- **Writing Quality** ⭐⭐⭐⭐⭐: Problem motivation is clear; system design is presented in a well-structured, progressive manner.
- **Value** ⭐⭐⭐⭐: Opens a new direction for personalized slide generation; the dataset and evaluation framework offer long-term value to subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AutoPP: Towards Automated Product Poster Generation and Optimization](autopp_towards_automated_product_poster_generation_and_optimization.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[NeurIPS 2025\] NeurIPS Should Lead Scientific Consensus on AI Policy](../../NeurIPS2025/recommender/neurips_should_lead_scientific_consensus_on_ai_policy.md)
- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](../../ACL2026/recommender/from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)
- [\[ACL 2026\] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](../../ACL2026/recommender/icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)

</div>

<!-- RELATED:END -->
