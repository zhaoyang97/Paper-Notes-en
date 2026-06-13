---
title: >-
  [Paper Note] MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation
description: >-
  [ACL2026][Model Compression][Large Language Model Distillation] MTA advances LLM distillation from "aligning specific static layers" to "aligning representation evolution trajectories according to network depth." It alig…
tags:
  - "ACL2026"
  - "Model Compression"
  - "Large Language Model Distillation"
  - "Trajectory Alignment"
  - "Hierarchical Semantics"
  - "Structural Distillation"
  - "Hidden State Alignment"
date: 2026-05-08
content_hash: 6c9542c98686cadb
---

# MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation

**Conference**: ACL2026  
**arXiv**: [2605.01374](https://arxiv.org/abs/2605.01374)  
**Code**: No public code (repository link not provided in the paper)  
**Area**: Model Compression  
**Keywords**: Large Language Model Distillation, Trajectory Alignment, Hierarchical Semantics, Structural Distillation, Hidden State Alignment

## TL;DR
MTA advances LLM distillation from "aligning specific static layers" to "aligning representation evolution trajectories according to network depth." It aligns word-level information at lower layers and phrase-level relational geometry at higher layers. As a plug-in, it consistently improves ROUGE-L performance in instruction-following tasks for FDD, DistiLLM, and DistiLLM-2.

## Background & Motivation
**Background**: In LLM compression, knowledge distillation (KD) remains a primary approach. Typical methods involve matching the student model's output distribution to the teacher's, such as token-level KL divergence. Advanced methods align intermediate hidden states, attention maps, or inter-layer feature dynamics, encouraging the student to learn internal representations rather than just final answers.

**Limitations of Prior Work**: Existing intermediate layer distillation methods often default to a "uniform alignment granularity for all layers." They typically perform hidden-state alignment at the token level or align prediction distributions after mapping selected layers to the vocabulary space. This approach is simplistic and ignores the functional division of Transformer layers: lower layers act as lexical and local pattern processors, while higher layers lean toward abstract semantics and compositional reasoning.

**Key Challenge**: The student model needs to inherit teacher's internal knowledge, which is not a set of independent layer snapshots but a representation trajectory that evolves with depth. Enforcing a uniform token-level target across all layers results in the compression of low-level lexical foundations and high-level phrase relations into the same supervisory signal, leading to imprecise knowledge transfer.

**Goal**: The paper aims to address three specific issues: first, enabling the student to learn the teacher's hierarchical evolution from vocabulary to semantic composition; second, selecting a small set of key layers for alignment across different parameter scales and model families; and third, integrating this alignment as a module into existing distillation frameworks rather than redesigning the entire KD process.

**Key Insight**: The authors leverage conclusions from linguistic hierarchical compositionality and Transformer interpretability research: language is composed of words forming phrases; lower layers focus on lexical and factual memory, while higher layers focus on abstract semantics and complex sub-tasks. Therefore, distillation should change semantic units according to layer depth rather than relying solely on tokens.

**Core Idea**: Utilize layer-adaptive multi-granular span relationship alignment—aligning word spans at lower layers and noun/verb phrase spans at higher layers—allowing the student to replicate the teacher's trajectory of "how representation geometry changes with depth."

## Method
MTA is a module designed to augment existing LLM distillation methods. It does not replace original logit KD, FDD, or DistiLLM objectives but adds two extra constraints: Dynamic Structural Alignment (DSA) for aligning the relative geometric structure between spans, and Hidden Representation Alignment (Hid) for pulling the student's key token hidden states closer to the teacher's.

### Overall Architecture
Given a teacher model and a smaller student model, MTA first selects a set of key layers based on student depth and uses proportional mapping to find corresponding teacher layers. For GPT-2 120M, the paper selects the 6th layer for word-level alignment and the 9th and 12th layers for phrase-level alignment. For Qwen1.5-0.5B and OPT-1.3B, more key layers are selected at greater depths.

At each selected layer, MTA extracts semantic spans from input-output sequences. Lower-level spans are full words to preserve lexical grounding, while higher-level spans consist of noun and verb phrases to represent more abstract compositional semantics. Syntactic parsing tools like spaCy are used for extraction.

Next, MTA calculates importance weights for tokens. Since causal attention in autoregressive models naturally favors earlier tokens, the authors re-estimate "to what extent each token is attended by others" using a normalized pairwise self-attention without self-loops. These teacher-side token weights are then used for span aggregation and pairwise span weighting.

The total training objective is the base distillation loss plus two MTA terms: $L_{Total}=L_{Base}+\lambda_{DSA}L_{DSA}+\lambda_{Hid}L_{Hid}$. Here, $L_{Base}$ can be FDD, DistiLLM, or DistiLLM-2. $\lambda_{DSA}$ and $\lambda_{Hid}$ are typically set to 2/0.2 or 3/0.3.

### Key Designs
1.  **Layer-adaptive Multi-granular Span Alignment**:
    -   **Function**: Aligns different semantic units at varying depths: lower layers handle word spans, and higher layers handle noun/verb phrase spans.
    -   **Mechanism**: The paper views Transformer depth as a representation trajectory from lexical foundations to compositional semantics. Using phrase alignment at low layers might lose fine-grained lexical info, while using only word alignment at high layers might over-constrain the abstract semantic space.
    -   **Design Motivation**: This design introduces linguistic compositionality into distillation, matching supervisory signals with hierarchical layer functions. Mixing word and phrase strategies consistently outperforms single-granularity strategies.

2.  **Dynamic Structural Alignment (DSA)**:
    -   **Function**: Aligns the relative geometric structure between different spans within the same layer, rather than just point-to-point token alignment.
    -   **Mechanism**: For each selected layer, token hidden states within a span are weighted-averaged into a span representation $U_{k,l}$ based on importance. Then, cosine distances for all span pairs are calculated. DSA minimizes the squared error between student and teacher span pair distances, weighted by the product of teacher-side span salience.
    -   **Design Motivation**: Relative geometry describes "how the teacher organizes semantic units" better than individual representation values. Even with smaller hidden dimensions, students can better retain compositional semantics by learning similar span relationship structures.

3.  **Importance-weighted Hidden Representation Alignment (Hid)**:
    -   **Function**: Directly constrains key student token hidden states to be close to teacher hidden states.
    -   **Mechanism**: Since student and teacher hidden dimensions may differ, MTA learns a linear projection $W_l$ for each key layer to map student states into the teacher's space. It then calculates weighted cosine distance using teacher token weights, targeting only tokens covered by extracted spans.
    -   **Design Motivation**: While DSA manages span relationships, Hid focuses on individual feature values. Ablations show both contribute independently, with the combination performing best across baselines.

### Loss & Training
The core of DSA is intra-layer pairwise distance matching. For student layer $l$ and mapped teacher layer $\phi(l)$, the loss for each span pair $(i,j)$ is $w_{ij}^{sp}(d(U^S_{i,l},U^S_{j,l})-d(U^T_{i,\phi(l)},U^T_{j,\phi(l)}))^2$, where $d$ is cosine distance and $w_{ij}^{sp}$ is the teacher-side span weight.

HID focuses on projected token hidden-state cosine alignment. Student representations $H^S_{t,l}$ are projected via $W_l$ to the teacher dimension and compared via weighted cosine distance with $H^T_{t,l}$. Weights are derived from teacher-side token importance, prioritizing high-information tokens over stop words or padding.

Training is conducted on Dolly-15k, with evaluation on Dolly, SelfInst, VicunaEval, and Super-Natural Instructions. GPT-2 and Qwen1.5 use full-parameter fine-tuning, while OPT uses LoRA. Generative evaluation reports average ROUGE-L across 5 random seeds. MTA increases training cost but requires no extra modules during inference.

## Key Experimental Results

### Main Results
MTA was integrated into FDD, DistiLLM, and DistiLLM-2 across three teacher-student pairs: GPT-2 1.5B → 120M, Qwen1.5 1.8B → 0.5B, and OPT 6.7B → 1.3B. The metric is the average ROUGE-L across four instruction-following datasets.

| Model Pair | Base Method | Avg. ROUGE-L | With MTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| GPT-2 1.5B → 120M | FDD | 19.48 | 20.50 | +1.02 |
| GPT-2 1.5B → 120M | DistiLLM | 20.21 | 21.45 | +1.24 |
| GPT-2 1.5B → 120M | DistiLLM-2 | 18.59 | 19.94 | +1.35 |
| Qwen1.5 1.8B → 0.5B | FDD | 19.27 | 20.92 | +1.65 |
| Qwen1.5 1.8B → 0.5B | DistiLLM | 19.80 | 21.01 | +1.21 |
| Qwen1.5 1.8B → 0.5B | DistiLLM-2 | 23.39 | 24.73 | +1.34 |
| OPT 6.7B → 1.3B | FDD | 21.74 | 22.90 | +1.16 |
| OPT 6.7B → 1.3B | DistiLLM | 22.98 | 23.97 | +0.99 |
| OPT 6.7B → 1.3B | DistiLLM-2 | 22.96 | 23.22 | +0.26 |

MTA is consistently effective across architectures and frameworks. Gains are particularly notable for small students, such as Qwen1.5-0.5B + FDD, suggesting internal trajectory supervision is especially beneficial for capacity-constrained students.

### Ablation Study
Ablations were conducted using the GPT-2 1.5B → 120M setup to verify DSA, Hid, multi-granularity, span weighting, and layer selection.

| Config | Dolly | SelfInst | Vicuna | S-NI | Avg. | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistiLLM | 25.65 | 13.39 | 16.50 | 25.28 | 20.21 | Baseline |
| + Hid | 25.89 | 13.68 | 16.86 | 25.77 | 20.55 | Hidden only |
| + DSA | 25.77 | 14.24 | 16.27 | 27.40 | 20.92 | Geometry only |
| + Full MTA | 25.77 | 14.19 | 16.67 | 29.18 | 21.45 | Best combo |
| DistiLLM + Word Only | 25.82 | 13.54 | 16.67 | 27.16 | 20.80 | All Word spans |
| DistiLLM + Phrase Only | 25.96 | 14.25 | 17.03 | 27.42 | 21.17 | All Phrase spans |
| DistiLLM + MTA | 25.77 | 14.19 | 16.67 | 29.18 | 21.45 | Word (low) + Phrase (high) |
| MTA w/o weight | 25.95 | 14.10 | 16.38 | 26.21 | 20.66 | No importance weight |

### Key Findings
- DSA typically contributes more than Hid, especially on S-NI, suggesting that maintaining span relationship structures enhances generalization more than matching single-point hidden states.
- Full MTA outperforms individual losses, indicating "relational geometry" and "feature reconstruction" are complementary: the former manages structure, while the latter manages local precision.
- Word-only and phrase-only strategies are inferior to the layer-adaptive approach, supporting the hypothesis that low layers need lexical grounding while high layers need compositional semantics.
- Span weighting is critical. Without weights, DistiLLM's average dropped from 21.45 to 20.66, highlighting the importance of filtering low-value tokens.
- Increasing the number of intermediate layers has diminishing returns. In GPT-2, accuracy improved as layers increased from 0 to 3 but stagnated or dropped with more, likely due to redundancy.
- MTA adds training overhead (DistiLLM+MTA: ~0.48s/step vs 0.26s/step) but incurs zero cost during inference.

## Highlights & Insights
- **From Point Alignment to Trajectory Alignment**: The paper shifts focus from "which layer matches which" to "whether the student follows the teacher's evolution from lexical to semantic."
- **Span Relations as Distillation Targets**: DSA aligns relative distances between span pairs, which is more robust for teacher-student pairs with different hidden widths.
- **Hierarchical Function Mapping**: Multi-granularity is not just complexity; it aligns supervisory signals with actual layer functions (low-level word, high-level phrase).
- **Plug-and-play Compatibility**: Ability to integrate with FDD and DistiLLM demonstrates its utility as a representation regularizer with low integration effort.
- **Honest Efficiency Analysis**: The authors acknowledge the cost of span extraction and use time-matched baselines to prove gains aren't just from longer training.

## Limitations & Future Work
- **Dependency on External Parsers**: Requires noun/verb phrase extraction, introducing spaCy chain costs and potential issues with low-quality parsing in code or math domains.
- **Instruction-following Focus**: Evaluation is centered on ROUGE-L and LLM-as-a-judge; performance in reasoning, factuality, or long-context tasks remains unverified.
- **Tokenizer Constraints**: Baselines rely on shared tokenizers. Differing tokenizers would require more complex mapping between spans and hidden states.
- **Empirical Layer Selection**: While rules were provided, layer and granularity assignments still rely on empirical heuristics.
- **Computational Complexity of Structural Alignment**: DSA is $O(N^2)$ relative to the number of spans, which might be costly for very long sequences.

## Related Work & Insights
- **vs FDD**: FDD views Transformer depth as a dynamic system and aligns prediction trajectories. MTA focuses on the internal relational geometry of spans rather than just the LM-head output.
- **vs DistiLLM / DistiLLM-2**: These focus on KL form and data efficiency. MTA is complementary, adding representation trajectory constraints.
- **vs Traditional Intermediate KD (e.g., TinyBERT)**: Traditional methods use uniform token-level targets; MTA introduces word/phrase granularity.
- **Interpretability**: Inspired by research showing Transformers follow a surface-to-semantic hierarchy, MTA converts these analytical findings into trainable loss functions.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines hierarchical linguistic structure with feature trajectory distillation effectively.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple families and baselines; could benefit from more diverse task types.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and logic.
- Value: ⭐⭐⭐⭐☆ Highly practical for enhancing existing KD frameworks during training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SRA: Span Representation Alignment for Large Language Model Distillation](sra_span_representation_alignment_for_large_language_model_distillation.md)
- [\[ACL 2026\] Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines](alignment_tuning_for_large_language_models_a_data-centric_lens_on_alignment_data.md)
- [\[CVPR 2026\] Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment](../../CVPR2026/model_compression/beyond_loss_values_robust_dynamic_pruning_via_loss_trajectory_alignment.md)
- [\[ACL 2026\] TLoRA: Task-aware Low Rank Adaptation of Large Language Models](tlora_task-aware_low_rank_adaptation_of_large_language_models.md)
- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)

</div>

<!-- RELATED:END -->
