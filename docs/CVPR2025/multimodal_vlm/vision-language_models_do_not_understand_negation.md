---
title: >-
  [Paper Note] Vision-Language Models Do Not Understand Negation
description: >-
  [CVPR 2025][Multimodal VLM][Negation Understanding] This paper proposes the NegBench benchmark to systematically reveal the severe deficiencies of vision-language models like CLIP in understanding negation (performing close to random-guess levels). By fine-tuning on a large-scale synthetic negation dataset, the retrieval recall of negation queries is improved by 10%, and the MCQ accuracy is boosted by up to 40%.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Negation Understanding"
  - "Vision-Language Models"
  - "CLIP"
  - "Benchmarking"
  - "Data-driven Improvement"
date: 2026-05-08
content_hash: 8e7798201d3f1397
---

# Vision-Language Models Do Not Understand Negation

**Conference**: CVPR 2025  
**arXiv**: [2501.09425](https://arxiv.org/abs/2501.09425)  
**Code**: [https://NegBench.github.io](https://NegBench.github.io)  
**Area**: Multimodal VLM  
**Keywords**: Negation Understanding, Vision-Language Models, CLIP, Benchmarking, Data-driven Improvement

## TL;DR
This paper proposes the NegBench benchmark to systematically reveal the severe deficiencies of vision-language models like CLIP in understanding negation (performing close to random-guess levels). By fine-tuning on a large-scale synthetic negation dataset, the retrieval recall of negation queries is improved by 10%, and the MCQ accuracy is boosted by up to 40%.

## Background & Motivation

**Background**: Joint-embedding vision-language models (such as CLIP), trained on large-scale image-text pairs, have achieved outstanding performance in tasks like cross-modal retrieval, image captioning, and text-to-image generation, becoming foundational components of multimodal AI.

**Limitations of Prior Work**: These models suffer from severe drawbacks when processing negative statements. For instance, "a beach with no people" and "a beach with people" are almost indistinguishable in the CLIP embedding space. This could lead to serious consequences in scenarios such as medical image diagnosis ("no evidence of pneumonia") or security monitoring ("construction sites with no barriers"). Existing benchmarks like CREPE and CC-Neg only include a few template-based negation tests, failing to provide a comprehensive evaluation.

**Key Challenge**: CLIP's contrastive learning objective encourages the model to learn "bag-of-words" matching between images and text, focusing on the presence of key nouns while ignoring qualifiers (such as "no", "not"). This "affirmation bias" is deeply rooted in the training data—internet image-text pairs are almost entirely affirmative descriptions and rarely contain negative expressions.

**Goal**: (1) Construct a comprehensive benchmark for evaluating negation understanding; (2) diagnose the root causes of VLM failures in negation understanding; and (3) explore data-driven solutions.

**Key Insight**: Starting from the multi-level evaluation pipeline of information retrieval systems—first performing coarse-grained retrieval (whether images containing/excluding specific objects can be retrieved), and then fine-grained discrimination (whether the correct negative description can be selected from similar options).

**Core Idea**: Employing a logical chain of "challenge-diagnosis-solution" to teach CLIP to understand negation by constructing a large-scale synthetic negation dataset.

## Method

### Overall Architecture
NegBench comprises two core tasks: (1) Retrieval-Neg: performing image-text retrieval after inserting negative statements into the original descriptions, to evaluate the model's ability to handle negative queries; (2) MCQ-Neg: selecting the correct description given an image and four options (encompassing affirmative, negative, and mixed descriptions). The benchmark covers COCO, VOC2007 (images), MSR-VTT (videos), CheXpert (medical imaging), and the synthetic HardNeg-Syn dataset, totaling 18 task variants and 79K samples.

### Key Designs

1. **NegBench Benchmark Construction**:

    - Function: Systematically evaluate the negation understanding capability of VLMs.
    - Mechanism: For each dataset, positive concepts (set of objects present in the image $\{pos\}$) and negative concepts (set of contextually relevant but absent objects $\{neg\}$) are extracted first. LLaMA 3.1 is utilized to generate natural-language negative descriptions and paraphrase them to ensure linguistic diversity. The MCQ task is designed with three templates: purely affirmative, purely negative, and mixed ("contains A but not B"). Distractors are carefully crafted hard negatives (e.g., negating objects that are present, or affirming objects that are absent), forcing the model to truly understand negation to answer correctly.
    - Design Motivation: Existing benchmarks only test negation using fixed templates, failing to reflect the diversity of negations in real queries. HardNeg-Syn utilizes Stable Diffusion to generate paired images (one containing the target object and one without), verified by OWL-ViT for object presence/absence, providing the most rigorous negation testing.

2. **Diagnostic Analysis: Embedding Space Visualization**:

    - Function: Reveal the root causes of VLM failures in understanding negation.
    - Mechanism: Visualize CLIP's embeddings for affirmative and negative descriptions using PCA. The embeddings of affirmative/negative descriptions for CLIP and NegCLIP overlap completely (unable to differentiate "a dog" from "no dog"), indicating that the models adopt a "bag-of-words" shortcut strategy that ignores negative words. Although ConCLIP separates affirmative and negative representations, it collapses all negative descriptions into a single point (unable to distinguish "no dog" from "no cat"), which is another form of degradation. The Sentence-Transformer text model exhibits an ideal separation, clearly dividing along two orthogonal dimensions: "object type" and "negation".
    - Design Motivation: Understanding the failure mechanism is crucial to designing effective improvements. The diagnosis indicates that the issue lies in the training data rather than the model architecture.

3. **CC12M-NegFull Synthetic Negation Training Set**:

    - Function: Provide large-scale negation training signals.
    - Mechanism: Based on the CC12M dataset (10 million image-text pairs), LLaMA 3.1 is used to extract present and absent objects for each image, which are then visually verified using OWL-ViT, followed by generating natural language negative descriptions. It consists of two subsets: CC12M-NegCap (3 negation-containing descriptions per image, ~30 million descriptions) for contrastive learning, and CC12M-NegMCQ (4 descriptions per image with 1 positive and 3 negatives, ~40 million descriptions) for fine-grained negation learning. The joint training loss is defined as $\mathcal{L}_{Total} = \alpha \mathcal{L}_{CLIP}(\mathcal{B}_{cap}) + (1-\alpha)\mathcal{L}_{MCQ}(\mathcal{B}_{mcq})$.
    - Design Motivation: Diagnostic results point to a lack of negative samples in the training data as the root cause. Rather than modifying the model architecture, it is more effective to bridge this gap at the data level.

### Loss & Training
Standard CLIP contrastive loss is used on CC12M-NegCap to improve the negation understanding in coarse-grained retrieval. For CC12M-NegMCQ, an MCQ cross-entropy loss is incorporated: $\mathcal{L}_{MCQ} = -\frac{1}{M}\sum_{i=1}^{M}\log\frac{\exp(\text{logits}_{i,c_i})}{\sum_{j=1}^{C}\exp(\text{logits}_{i,j})}$, forcing the model to distinguish between correct descriptions and hard negatives. $\alpha$ controls the balance between the two objectives.

## Key Experimental Results

### Main Results

| Model | Fine-tuning Data | COCO R@5 | COCO R-Neg@5 | COCO MCQ Acc |
|------|---------|----------|-------------|-------------|
| CLIP | None | 54.8 | 48.0 | 16.3% |
| CLIP | CC12M | 58.8 | 54.5 | 11.2% |
| CLIP | CC12M-NegCap | 58.5 | 57.8 | 14.7% |
| CLIP | CC12M-NegFull | 54.2 | 51.9 | **46.9%** (+30.6) |
| NegCLIP | None | 68.7 | 64.4 | 10.2% |
| NegCLIP | CC12M-NegFull | 69.0 | 67.0 | **51.0%** (+40.8) |

### Ablation Study

| $\alpha$ Value | COCO R@5 | COCO MCQ Acc | Description |
|-------------|----------|-------------|------|
| 0.0 (pure MCQ loss) | 33.9% | 61.0% | Retrieval collapses, MCQ is optimal |
| 0.25 | 37.3% | 54.7% | — |
| 0.5 | 47.6% | 50.5% | Good balance |
| 0.75 | 54.2% | 46.9% | Biased towards retrieval |
| 1.0 (pure CLIP loss) | 58.5% | 14.7% | No improvement in MCQ |

### Key Findings
- **Complete Failure in Negation Understanding**: All CLIP base models perform worse than random guess (25%) on MCQ-Neg, with some reaching only 8%, indicating that these models systematically select incorrect negation templates.
- **Scaling Up the Model Does Not Help**: From ViT-B/32 to ViT-H/14, there is almost no improvement in negation understanding.
- **Severe Consequences in the Medical Domain**: BioMedCLIP and CONCH drop in performance by 24.6% and 33.2% respectively upon introducing negation.
- **Contrastive Learning is Insufficient**: Fine-tuning with NegCap alone only improves retrieval; the MCQ objective requires the MCQ loss to achieve significant gains.
- There is a trade-off between retrieval capability and negation understanding, which requires fine-tuning with $\alpha$.

## Highlights & Insights
- **Systematic Evaluation Paradigm**: The two-level evaluation design of NegBench (coarse-grained retrieval + fine-grained MCQ) is highly ingenious, allowing for precise pinpointing of where the model fails along the chain of negation understanding.
- **Embedding Space Diagnosis**: By visualizing the overlap of affirmative/negative embeddings via PCA, it intuitively and convincingly demonstrates the existence of the "bag-of-words shortcut strategy". This diagnostic method can be transferred to analyze other linguistic comprehension deficiencies.
- **Power of Synthetic Data**: The pipeline of utilizing LLMs + detectors to generate and verify massive training data significantly improves specific capabilities without human annotation. This paradigm is generalizable to fixing other capability gaps in VLMs.

## Limitations & Future Work
- While fine-tuning improves negation capability, standard retrieval performance experiences a minor degradation. This indicates that the embedding space of CLIP has limited capacity, prompting the need for better training strategies to optimize both simultaneously.
- Currently, this is only validated on CLIP-like joint-embedding models. Generative VLMs (e.g., LLaVA) exhibit better negation understanding in the appendix, but they cannot perform large-scale retrieval efficiently.
- The quality of synthetic negative descriptions depends on the LLMs and detectors; some complex negations (such as double negation and implicit negation) might not be covered.
- Future Directions: Explore the effect of incorporating negative samples during the pre-training phase (rather than just fine-tuning); investigate superior training objectives to replace simple contrastive learning.

## Related Work & Insights
- **vs NegCLIP**: NegCLIP improves compositional reasoning through composition-aware negative mining, but it does not specifically target negation. On NegBench, its negation understanding even deteriorates (with a 23% drop on HardNeg-Syn).
- **vs ConCLIP**: Explicitly designed for negation understanding, but its training data overfits to a single mixed template, causing all negative embeddings to collapse into a single point. The diverse evaluation of NegBench exposes this limitation.
- **vs LLaVA**: Instruction-tuned VLMs represent a visible improvement over CLIP in negation understanding. However, as they process image-text pairs individually, they cannot efficiently accomplish large-scale retrieval tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The first comprehensive negation understanding benchmark with a clear diagnosis-solution paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 task variants, multiple models, with exhaustive diagnoses and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative logic of challenge-diagnosis-solution is extremely smooth and clear.
- Value: ⭐⭐⭐⭐ Unveils a significant blind spot in VLMs, providing a practical driving force for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](../../ACL2025/multimodal_vlm/negvqa_can_vision_language_models_understand_negation.md)
- [\[CVPR 2025\] Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)
- [\[CVPR 2025\] Evaluating Vision-Language Models as Evaluators in Path Planning](evaluating_vision-language_models_as_evaluators_in_path_planning.md)
- [\[ICML 2025\] Do Vision-Language Models Really Understand Visual Language?](../../ICML2025/multimodal_vlm/do_vision-language_models_really_understand_visual_language.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)

</div>

<!-- RELATED:END -->
