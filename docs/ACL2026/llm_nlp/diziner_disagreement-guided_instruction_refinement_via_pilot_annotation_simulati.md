---
title: >-
  [Paper Note] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot NER
description: >-
  [ACL 2026][LLM/NLP][Zero-shot NER] DiZiNER simulates the human pilot annotation workflow: multiple heterogeneous LLMs independently annotate the same text, and inter-model disagreements are analyzed to iteratively refine task instructions. The method achieves zero-shot SOTA on 14 out of 18 NER benchmarks, with an average F1 gain of +8.0, surpassing its supervisor model GPT-5 mini.
tags:
  - ACL 2026
  - LLM/NLP
  - Zero-shot NER
  - Annotation Simulation
  - Inter-model Disagreement
  - Instruction Refinement
  - Pilot Annotation
date: 2026-05-08
content_hash: 286935202baaa4c7
---

# DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot NER

**Conference**: ACL 2026  
**arXiv**: [2604.15866](https://arxiv.org/abs/2604.15866)  
**Code**: [GitHub](https://github.com/SiunKim/diziner-ner/)  
**Area**: Information Extraction / NER  
**Keywords**: Zero-shot NER, Annotation Simulation, Inter-model Disagreement, Instruction Refinement, Pilot Annotation

## TL;DR
DiZiNER simulates the human pilot annotation workflow: multiple heterogeneous LLMs independently annotate the same text, and inter-model disagreements are analyzed to iteratively refine task instructions. The method achieves zero-shot SOTA on 14 out of 18 NER benchmarks, with an average F1 gain of +8.0, surpassing its supervisor model GPT-5 mini.

## Background & Motivation

**Background**: Zero-shot and few-shot NER performance of LLMs has improved substantially, yet remains far behind supervised systems. Instruction-tuning approaches (e.g., UniversalNER, GoLLIE) require large amounts of annotated data and transfer poorly across domains.

**Limitations of Prior Work**: LLMs exhibit systematic error patterns in NER—difficulty following complex annotation guidelines, ambiguous entity boundary detection, and frequent entity type confusion. These errors closely resemble the inconsistencies human annotators exhibit during the early stages of annotation.

**Key Challenge**: The performance gap between zero-shot NER and supervised NER (approximately 32 F1 points on average) stems not from insufficient model capability, but from ambiguity in task instructions—the same instruction is interpreted differently by different models.

**Goal**: To automatically refine NER instructions by simulating the iterative disagreement-resolution process of human pilot annotation, improving zero-shot performance without any parameter updates.

**Key Insight**: The construction of human gold-standard datasets inherently involves resolving disagreements and refining guidelines through pilot annotation—DiZiNER simulates this process using LLMs.

**Core Idea**: Multiple heterogeneous LLMs serve as annotators; their disagreements are analyzed to refine both universal instructions and model-specific instructions.

## Method

### Overall Architecture
DiZiNER iteratively executes a three-step cycle: (1) **Independent cross-annotation**: multiple heterogeneous LLMs independently annotate the same text subset using their respective task configurations; (2) **Disagreement analysis**: annotations are converted to BIO sequences, token-level disagreement scores are computed, and high-disagreement spans are identified; (3) **Instruction refinement**: a supervisor model analyzes disagreement reports and updates both universal and model-specific instructions.

### Key Designs

1. **Cross-annotation with Multiple Heterogeneous Models**:

    - Function: Capture diverse types of annotation errors through model diversity.
    - Mechanism: A pool of independently developed open-source LLMs (selected to minimize correlated errors) $\mathcal{M} = \{M_k\}_{k=1}^K$ serves as annotators. Each annotator receives a task configuration $\Theta_k^{(t)} = (\Sigma, C^{(t)}, R_k^{(t)}, G^{(t)})$, where $\Sigma$ is a fixed schema, $C^{(t)}$ is the universal instruction, and $R_k^{(t)}$ is the model-specific instruction. Annotation outputs are converted to BIO sequences for token-level comparison.
    - Design Motivation: Heterogeneous models exhibit distinct error patterns—one model may frequently miss entities while another misclassifies types—and their disagreements precisely expose ambiguities in the instructions.

2. **Three-dimensional Disagreement Analysis**:

    - Function: Precisely localize and categorize the root causes of annotation inconsistencies.
    - Mechanism: Three complementary token-level disagreement metrics are computed—$D_{conf}$ (label conflict, dispersion of BIO labels), $D_{type}$ (type confusion, dispersion of entity types), and $U_{bnd}$ (boundary uncertainty, entropy of B/I labels). The maximum $U_\star$ is used as the composite disagreement score; the top 20% high-disagreement tokens are merged into hotspot spans. Consensus labels are obtained via weighted voting, with model weights determined by pairwise F1.
    - Design Motivation: Different types of disagreement require different instruction corrections—label conflicts call for clarified type definitions, type confusion requires additional distinguishing examples, and boundary uncertainty necessitates more precise boundary rules.

3. **Hierarchical Instruction Refinement**:

    - Function: Automatically optimize annotation instructions based on disagreement patterns.
    - Mechanism: Structured disagreement reports are generated (hotspot statistics, elite vs. non-elite group differences, representative disagreement examples with reasoning traces), and a supervisor model (GPT-5 mini) updates the universal instruction $C^{(t+1)}$ and model-specific instructions $R_k^{(t+1)}$ accordingly. The schema $\Sigma$ remains fixed to prevent task drift. Iterations continue until disagreement converges.
    - Design Motivation: Universal instruction updates address common issues shared across all models, while model-specific instructions target each model's individual weaknesses—directly corresponding to the "shared guidelines + individual feedback" pattern in human pilot annotation.

### Loss & Training
DiZiNER requires no parameter updates. Iterative refinement is achieved entirely through prompt engineering.

## Key Experimental Results

### Main Results

| Method | CrossNER Avg. F1 | Type |
|--------|-----------------|------|
| GPT-5 mini (supervisor) | 69.3 | Zero-shot |
| B2NER | 75.3 | Instruction fine-tuning |
| GNER | 74.0 | Instruction fine-tuning |
| GLiNER | 65.3 | Encoder fine-tuning |
| **DiZiNER** | **75.7** | Zero-shot |
| Δ DiZiNER vs GPT-5 mini | +6.4 | — |
| Avg. gain from iteration 0 to best | +4.8 | — |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Universal instruction only | Improvement but limited | Model-specific instructions also contribute |
| No heterogeneity (same model repeated) | Weak disagreement signal | Heterogeneous models are critical for disagreement quality |
| Number of iterations | Converges in 2–3 rounds | Inter-model agreement improves across iterations |

### Key Findings
- DiZiNER surpasses GPT-5 mini (its supervisor model) by 6.4 F1 points, demonstrating that gains arise from disagreement-guided refinement rather than raw model capability.
- Pairwise inter-model agreement is strongly correlated with NER performance, further validating the value of disagreement analysis.
- Zero-shot SOTA is achieved on 14 of 18 benchmarks, narrowing the zero-shot-to-supervised gap from −32.0 to −20.9.
- Iterative convergence is rapid (typically 2–3 rounds), indicating that a small number of refinement steps substantially reduces disagreement.

## Highlights & Insights
- **Bringing human annotation engineering into LLM prompt engineering** is a highly creative analogy—the core of pilot annotation is "identifying problems through disagreement and resolving them through guideline revision," which is precisely the essence of prompt optimization.
- Results exceeding the supervisor model suggest that **the disagreement signal from multiple weaker models is more informative than the judgment of a single stronger model**.
- This approach generalizes to any structured prediction task requiring instruction refinement (relation extraction, event detection, etc.).

## Limitations & Future Work
- Multiple heterogeneous LLMs plus a strong supervisor model are required, resulting in relatively high inference costs.
- The selection of the iterative document set may affect refinement quality—if sampling is unrepresentative, important disagreements may be missed.
- Validation is limited to NER; generalization to other IE tasks requires further investigation.
- Model-specific instructions may overfit to the weaknesses of particular models.

## Related Work & Insights
- **vs. UniversalNER**: Requires distillation training on ChatGPT-synthesized data; DiZiNER is entirely training-free.
- **vs. IRRA**: Self-consistency and self-verification methods iterate with a single model; DiZiNER leverages multi-model disagreement.
- **vs. B2NER**: The strongest instruction fine-tuning baseline; DiZiNER approaches or surpasses its performance under the zero-shot setting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The pilot annotation simulation analogy is highly creative; disagreement-guided instruction refinement is an entirely new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 18 benchmarks; results surpassing the supervisor model are highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with complete mathematical formalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts](../../AAAI2026/llm_nlp/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)
- [\[CVPR 2026\] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](../../CVPR2026/llm_nlp/cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](../../CVPR2026/llm_nlp/boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)

</div>

<!-- RELATED:END -->
