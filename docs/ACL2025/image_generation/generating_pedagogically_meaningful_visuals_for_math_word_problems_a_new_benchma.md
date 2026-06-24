---
title: >-
  [Paper Note] Generating Pedagogically Meaningful Visuals for Math Word Problems: A New Benchmark and Analysis of Text-to-Image Models
description: >-
  [ACL 2025 (Findings)][Image Generation][Math Word Problems] Math2Visual proposes a framework to automatically generate pedagogical visualizations from textual descriptions of math word problems (MWPs). It defines a visual language and design space based on teacher interviews, constructs a labeled dataset of 1,903 images, and evaluates and fine-tunes multiple TTI models, revealing key deficiencies of current models in representing mathematical relationships.
tags:
  - "ACL 2025 (Findings)"
  - "Image Generation"
  - "Math Word Problems"
  - "Educational Visualization"
  - "Text-to-Image Generation"
  - "Visual Language"
  - "Instructional Design"
date: 2026-05-08
content_hash: 7a233dc5a8a732a8
---

# Generating Pedagogically Meaningful Visuals for Math Word Problems: A New Benchmark and Analysis of Text-to-Image Models

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2506.03735](https://arxiv.org/abs/2506.03735)  
**Code**: None  
**Area**: Image Generation / Educational AI  
**Keywords**: Math Word Problems, Educational Visualization, Text-to-Image Generation, Visual Language, Instructional Design

## TL;DR

Math2Visual proposes a framework to automatically generate pedagogical visualizations from textual descriptions of math word problems (MWPs). It defines a visual language and design space based on teacher interviews, constructs a labeled dataset of 1,903 images, and evaluates and fine-tunes multiple TTI models, revealing key deficiencies of current models in representing mathematical relationships.

## Background & Motivation

**Background**: Visual aids are widely used in primary school mathematics education to help students translate written descriptions into mathematical expressions. For example, for math word problems like "Xiao Ming has 3 apples and buys 2 more," incorporating intuitive images can significantly enhance comprehension for lower-grade students. However, these instructional images currently rely almost entirely on manual creation by teachers, which is highly time-consuming and labor-intensive.

**Limitations of Prior Work**: Although text-to-image (TTI) models (such as DALL-E and Stable Diffusion) have made tremendous progress in general image generation, they cannot be directly applied to educational scenarios. The generated images are often merely "aesthetically pleasing illustrations" that lack accurate representations of mathematical relationships. For instance, an addition problem requires an image to clearly show the grouping relationship between "3" and "2," but general TTI models might generate 5 randomly arranged apples, losing the pedagogical meaning of grouping and operations.

**Key Challenge**: Pedagogical visualization requires not only visual appeal, but more fundamentally, the accurate communication of mathematical relationships (quantities, operations, comparisons, etc.). Neither the training data nor the evaluation metrics of existing TTI models involve this "pedagogical semantic correctness," making the generated results unusable in educational contexts.

**Goal**: (1) To define a set of design specifications for pedagogical visualization (visual language + design space); (2) to construct a labeled dataset for evaluation and training; and (3) to evaluate existing TTI models' capabilities in pedagogical visualization and analyze their deficiencies.

**Key Insight**: Through in-depth interviews with multiple mathematics teachers, the authors extract core design principles and common visual elements of pedagogical visualization, formalizing domain knowledge into computable design specifications.

**Core Idea**: Define a structured visual language for math pedagogical visualization (including elements like object representations, quantity encodings, and operational relationships), and build a labeled dataset and the automated generation framework Math2Visual based on this language.

## Method

### Overall Architecture

Math2Visual comprises three levels: (1) a Visual Language defined based on teacher interviews, which describes the visual elements and their composition rules that should be included in instructional images; (2) a Design Space, which specifies visualization templates corresponding to different mathematical operation types; and (3) an automated generation pipeline, which parses MWP text into structured representations, maps them to templates in the design space, and then converts them into TTI model prompts or renders them directly.

### Key Designs

1. **Visual Language Definition (Visual Language)**:

    - **Function**: Provide a formal description of pedagogical visualization elements.
    - **Mechanism**: Extract core components in pedagogical visualization through semi-structured interviews with math teachers—including object types (concrete items like apples, abstract shapes like circles), quantity encoding methods (one-by-one display, grouped display, numerical annotation), spatial arrangement (row, column, partitioned arrangements), and operational relationship representations (auxiliary symbols such as grouping lines, arrows, plus signs). These elements are organized into a hierarchical visual vocabulary.
    - **Design Motivation**: Without a formalized visual language, it is impossible to systematically evaluate the "pedagogical correctness" of generated images. Teachers' empirical knowledge is converted into actionable design rules, transforming the evaluation from subjective judgment into a structured, multi-dimensional inspection.

2. **Design Space and Template System (Design Space)**:

    - **Function**: Map different mathematical operation types to corresponding visualization templates.
    - **Mechanism**: Define different visualization layout templates based on the types of mathematical operations involved in MWPs (addition, subtraction, multiplication, division, comparison). For example, addition problems use a "two groups + merging arrow" layout, while comparison problems use an "aligned arrangement + difference highlighting" layout. Each template includes mandatory elements (such as correct quantities, grouping relationships) and optional elements (such as color coding, annotation text). After parsing the MWP text to obtain the operation type and numerical values, the corresponding template is automatically selected and parameterized.
    - **Design Motivation**: Different mathematical concepts require different visual expression strategies. A generic prompt like "draw a math diagram" is far from sufficient; customized designs tailored to specific operation types are required to convey the correct pedagogical intent.

3. **Dataset Construction and Annotation (Dataset Construction)**:

    - **Function**: Provide high-quality labeled data for evaluation and fine-tuning.
    - **Mechanism**: Select problems covering different operation types from existing MWP datasets, and use the Math2Visual framework to generate standard visualization images for each problem, constructing a total of 1,903 human-validated pedagogical images. Each image is accompanied by structured annotations recording the included visual elements (whether the quantity is correct, whether the operational relationship is reflected, and whether the layout conforms to the template). These annotations serve as criteria for both automatic evaluation and human assessment.
    - **Design Motivation**: The lack of labeled data is a common issue in educational AI research. This dataset fills the gap in the evaluation of mathematics pedagogical visualization.

### Loss & Training

When fine-tuning TTI models like Stable Diffusion, a standard denoising diffusion training objective is used, but with the pedagogical images generated by Math2Visual as target images on the dataset, and the structured descriptions of corresponding MWPs as prompts. Fine-tuning aims to teach the models specific visual patterns of pedagogical visualization (such as accurate counting, grouped arrangements, etc.).

## Key Experimental Results

### Main Results

Evaluate the performance of multiple TTI models in generating pedagogical visualizations, using structured metrics (quantity accuracy, operational relationship accuracy, layout compliance rate) and human pedagogical evaluation scores:

| Model | Quantity Accuracy | Operational Relation Accuracy | Pedagogical Effectiveness (Human Eval) | Note |
|------|-----------|--------------|-----------------|------|
| DALL-E 3 | ~35% | ~25% | Low | Strongest general model but poor pedagogical semantics |
| Stable Diffusion XL | ~20% | ~15% | Very Low | Severe counting and relationship errors |
| SD XL (Fine-tuned) | ~55% | ~45% | Medium | Fine-tuning significantly improves performance |
| Math2Visual (Template Render) | ~95% | ~90% | High | Template-based method is most accurate but lacks flexibility |

### Ablation Study

| Configuration | Quantity Accuracy | Note |
|------|-----------|------|
| Structured Prompt (Full) | Highest | Contains descriptions of all visual language elements |
| Simplified Prompt (Problem Text Only) | Significantly dropped | Lacks guidance from design specifications |
| Fine-tuning + Structured Prompt | Highest (among TTI) | Combination of both yields the best results |
| Fine-tuning + Simplified Prompt | Medium | Fine-tuning helps but structured description is more important |
| Comparison of Different Operation Types | - | Addition is easiest, division is hardest |

### Key Findings

- **Existing TTI models are severely deficient in pedagogical visualization**: Even the strongest model, DALL-E 3, achieves less than 40% quantity accuracy, indicating that the "accurate visual representation of mathematical relationships" remains a blind spot for current TTI models.
- **The most common errors are incorrect quantities and omitted relationships**: Models tend to generate roughly correct scenes but neglect precise counting, and omit visual markers of mathematical operations (such as grouping lines and arrows).
- **Fine-tuning significantly improves but does not fully solve the problem**: After fine-tuning, models learn some pedagogical visualization patterns, but still struggle with complex operations (such as multi-step or division problems), indicating that this requires deeper structured reasoning capabilities.
- **Structured prompts perform significantly better than natural language prompts**: Using structured prompts composed of visual language elements yields a substantial improvement over directly using the problem text as prompts, indicating that "how to describe what graph is needed" is more critical than "giving the problem to the model to draw on its own."

## Highlights & Insights

- **Formalizing teachers' domain knowledge into a visual language** is a highly valuable research direction; cross-domain formalized knowledge representation can be transferred to automated visualization generation in other specialized fields (e.g., chemistry experiment procedures, physics force diagrams).
- **The design of the evaluation framework** has more long-term value than the generation method itself—it defines multi-dimensional structured evaluation criteria for "math pedagogical visualization quality," providing a unified starting point for future research.
- **Although the dataset is small (1,903), the annotation quality is high**, and each image has structured element annotations, supporting fine-grained error analysis, which represents a scalable annotation paradigm.

## Limitations & Future Work

- **Small dataset scale**: 1,903 images may be insufficient for fine-tuning large-scale TTI models, limiting the performance ceiling.
- **Limited coverage of basic arithmetic operations**: For more complex mathematical concepts (fractions, area, probability, etc.), the visual language and design space need to be substantially expanded.
- **Insufficient evaluation of pedagogical effectiveness**: There is a lack of testing on the impact of generated images on student learning outcomes in real classroom environments; current evaluations only rely on expert ratings.
- **Large gap between template rendering and generative models**: This indicates that automatically generating pedagogical images solely through diffusion models is not yet mature, and a hybrid "template + generation" scheme may be required.

## Related Work & Insights

- **vs. General TTI (DALL-E / SD)**: General models pursue visual quality but neglect semantic precision (e.g., counting). This work reveals this fundamental gap.
- **vs. Mathematical Visualization Tools (GeoGebra, etc.)**: Traditional math software requires manual operations; Math2Visual offers a higher level of automation, but its flexibility is constrained by the template library.
- **vs. MWP Solving Research**: MWP research primarily focuses on parsing text to equations, whereas this study focuses on generating images from text, representing a neglected but important direction.

## Rating

- Novelty: ⭐⭐⭐⭐ Pedagogical visualization generation is a novel cross-domain problem, and the approach of formalizing the design space is creative.
- Experimental Thoroughness: ⭐⭐⭐ Comparative experiments of multiple models and fine-tuning are complete, but the dataset is relatively small, and validation in real-world educational scenarios is lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation, detailed description of the visual language definition process, and rich illustrations.
- Value: ⭐⭐⭐⭐ Establishes an evaluation benchmark for mathematics pedagogical visualization generation in educational AI, holding practical application potential for edtech.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing](../../CVPR2025/image_generation/from_words_to_structured_visuals_a_benchmark_and_framework_for_text-to-diagram_g.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](../../NeurIPS2025/image_generation/overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[ICCV 2025\] Generating Multi-Image Synthetic Data for Text-to-Image Customization](../../ICCV2025/image_generation/generating_multi-image_synthetic_data_for_text-to-image_customization.md)
- [\[NeurIPS 2025\] On the Emergence of Linear Analogies in Word Embeddings](../../NeurIPS2025/image_generation/on_the_emergence_of_linear_analogies_in_word_embeddings.md)
- [\[ECCV 2024\] HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects](../../ECCV2024/image_generation/himo_a_new_benchmark_for_full-body_human_interacting_with_multiple_objects.md)

</div>

<!-- RELATED:END -->
