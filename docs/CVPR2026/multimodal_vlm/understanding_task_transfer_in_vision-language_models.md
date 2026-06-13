---
title: >-
  [Paper Note] Understanding Task Transfer in Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Vision-Language Models] This paper presents the first systematic study of how fine-tuning a VLM on one visual perception task affects its zero-shot performance on other perception tasks. It pr…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Task Transfer"
  - "Perceptual Tasks"
  - "Fine-Tuning"
  - "Perfection Gap Factor"
date: 2026-05-08
content_hash: eca9ee9a0e4b8850
---

# Understanding Task Transfer in Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2511.18787](https://arxiv.org/abs/2511.18787)  
**Code**: [https://aka.ms/task-transfer-vlms](https://aka.ms/task-transfer-vlms) (project page)  
**Area**: Multimodal VLM
**Keywords**: Vision-Language Models, Task Transfer, Perceptual Tasks, Fine-Tuning, Perfection Gap Factor

## TL;DR
This paper presents the first systematic study of how fine-tuning a VLM on one visual perception task affects its zero-shot performance on other perception tasks. It proposes the Perfection Gap Factor (PGF), a normalized metric for quantifying cross-task transfer, and reveals structural regularities in task transfer (positive/negative transfer cliques, task personas, scale dependence) across three scales of Qwen-2.5-VL. The paper further demonstrates that PGF can guide data selection to improve fine-tuning efficiency.

## Background & Motivation

1. **Background**: VLMs achieve strong performance on multimodal benchmarks, yet still lag behind humans and specialist models on fundamental visual perception tasks (depth estimation, counting, object localization, etc.). On the BLINK benchmark, the best model (GPT-4o) reaches only 60%, while humans achieve 95%. In practice, methods such as LoRA are commonly used to fine-tune models on specific perception tasks to close this gap.

2. **Limitations of Prior Work**: After fine-tuning on one perception task, the model's performance on other perception tasks changes in an unpredictable manner—either positively or negatively. This uncertainty makes task-specific fine-tuning risky, yet no systematic study has examined such cross-task effects.

3. **Key Challenge**: It remains unknown how internal representations in VLMs are shared or compete across different perception tasks. Different tasks may rely on the same underlying visual features (mutually beneficial) or compete for limited model capacity (mutually harmful).

4. **Goal**: To answer a central question—how does fine-tuning a VLM on one perception task affect its zero-shot performance on other perception tasks, and how can such cross-task relationships be quantified and leveraged?

5. **Key Insight**: Unlike Taskonomy (which requires transfer learning on both source and target tasks), this work studies zero-shot cross-task transfer—only the source task is fine-tuned, with no training on the target task whatsoever.

6. **Core Idea**: By employing the PGF normalized metric to systematically quantify zero-shot transfer relationships among VLM perception tasks, this paper reveals that cross-task transfer exhibits structural regularities that can guide efficient fine-tuning.

## Method

### Overall Architecture
Three variants of Qwen-2.5-VL (3B, 7B, 32B) are selected and independently fine-tuned (via LoRA) on each of the 13 perception tasks in the BLINK benchmark. Each fine-tuned model is then evaluated on the validation sets of all 13 tasks. A 13×13 transfer matrix is constructed and populated with PGF scores, from which transfer patterns are analyzed.

### Key Designs

1. **Perfection Gap Factor (PGF)**:
    - Function: Normalized quantification of the degree of cross-task transfer.
    - Mechanism: Defined as $\mu_{i \to j} = \frac{\text{Acc}(\mathcal{M}(T_i), T_j) - \text{Acc}(\mathcal{M}, T_j)}{U_j - \text{Acc}(\mathcal{M}, T_j) + \epsilon}$, where the numerator is the accuracy change after fine-tuning and the denominator is the remaining gap to the upper bound. PGF = 0 indicates no transfer; positive values indicate positive transfer; negative values indicate negative transfer. The upper bound $U_j$ defaults to 100%.
    - Design Motivation: Conventional accuracy gains are not comparable across tasks. A 3% improvement on a task already near ceiling is far more meaningful than a 10% improvement on a low-baseline task. PGF normalizes by the remaining improvement headroom, making transfer effects comparable across tasks of varying difficulty. For example, a task improving from 90% to 93% (PGF = 0.60) is far more significant than one improving from 40% to 50% (PGF = 0.18).

2. **Task Transferability**:
    - Function: Measures the overall positive/negative influence of a source task on other tasks.
    - Mechanism: Positive transferability $\Delta(i)^+ = \frac{1-e^{-p/N}}{p}\sum \mu_{i\to j} \mathbf{1}_{\mu>0}$ and negative transferability $\Delta(i)^-$ are computed separately, where the exponential weighting factor $(1-e^{-p/N})/p$ accounts simultaneously for breadth (how many tasks are affected) and intensity (average transfer magnitude).
    - Design Motivation: To distinguish between patterns of "large gains on a few tasks" versus "small gains across many tasks."

3. **Malleability**:
    - Function: Measures the sensitivity of a target task to being affected by fine-tuning on other source tasks.
    - Mechanism: Dual to transferability; aggregates PGF scores from all source tasks onto the target task, with separate accounting for positive and negative contributions. A task with high positive malleability readily benefits from fine-tuning on other tasks.
    - Design Motivation: To fully characterize bidirectional transfer relationships—not only "who influences others," but also "who is easily influenced."

4. **Task Cliques**:
    - Function: Identifies subsets of tasks that exhibit consistent mutual positive or negative transfer.
    - Mechanism: Searches for complete subgraphs in the transfer graph where all ordered task pairs $(T_i, T_j)$ exhibit consistent positive/negative transfer. Statistical significance across seeds is validated using the Wilcoxon test. The 32B model yields the largest positive clique (9 tasks), while smaller models yield cliques of 3–4 tasks.
    - Design Motivation: To reveal the mutualistic or antagonistic structural relationships among tasks.

5. **Task Personas**:
    - Function: Categorizes tasks into four persona types.
    - Mechanism: **Donor** = tasks with consistently high positive transferability across model scales (e.g., Semantic Correspondence); **Pirate** = tasks with consistently high negative transferability (e.g., Functional Correspondence); **Sponge** = tasks with high positive malleability, readily benefiting from others (e.g., Visual Similarity, Relative Depth); **Sieve** = tasks with high negative malleability, easily harmed by others (e.g., Forensic Detection).
    - Design Motivation: To provide practitioners with actionable guidance for fine-tuning.

### Loss & Training
QLoRA with 4-bit quantization is used to fine-tune Qwen-2.5-VL. Training sets are constructed from the original BLINK data sources, preserving the same task definitions and answer formats as BLINK. Each experiment is repeated with 4 random seeds.

## Key Experimental Results

### Main Results: Key Findings from PGF Transfer Heatmaps

| Finding | 3B | 7B | 32B |
|------|----|----|-----|
| Mean positive transferability | Low | Medium | High (monotonically increases with scale) |
| Maximum positive clique size | 3–4 | 3–4 | **9** |
| Donor task | SC | SC | SC (Semantic Corr., consistent across scales) |
| Pirate task | FC | FC | FC (Functional Corr., consistent across scales) |
| Sponge tasks | VS, RD, RR | VS, RD, RR | VS, RD, RR |
| Sieve task | FD | — | FD (Forensic Detection) |

### PGF-Guided Data Selection vs. Random Selection (Qwen-2.5-VL 7B)

| Target Task | Direct Fine-Tuning | Random Mixing | PGF-Guided Mixing | Notes |
|----------|---------|---------|------------|------|
| Jigsaw | baseline | Below direct | **Exceeds direct fine-tuning** | PGF selection outperforms direct supervision |
| Object Localization | baseline | Below direct | **Exceeds direct fine-tuning** | PGF selection outperforms direct supervision |
| Other tasks | baseline | Varies | Consistently outperforms random | PGF guidance is stable and effective |

### Key Findings
- **Scale Effect**: Larger models exhibit stronger positive transfer (most pronounced at 32B), while no clear trend is observed for negative transfer.
- **Perceptual Hierarchy**: Low-level tasks (Relative Depth, Relative Reflectance) exhibit the highest transferability and malleability.
- **Granularity Level**: Image-level tasks show the greatest positive transferability; both pixel-level and image-level tasks exhibit high malleability.
- **Video Transfer**: Similar patterns are observed on VSI-Bench video tasks, with Relative Reflectance remaining a donor and Forensic Detection remaining a pirate.
- **PGF Guidance**: On Jigsaw and Object Localization, PGF-guided mixed data even surpasses direct fine-tuning on the target task.

## Highlights & Insights
- **Elegant PGF Design**: By normalizing over the remaining improvement headroom, PGF resolves the core problem of incomparability across tasks of varying difficulty. The asymmetry—with a positive upper bound of 1 and a negative lower bound of $-(m-1)$—is also well-motivated: regressing when near perfect performance is more severe than regressing when far from it.
- **Task Persona Framework**: The Donor/Pirate/Sponge/Sieve classification is intuitive and immediately practical, offering ready-to-use guidance for multi-task fine-tuning strategies.
- **Counterintuitive Finding**: PGF-guided indirect mixed-data fine-tuning can outperform direct fine-tuning on the target task, suggesting that the cumulative effect of positive transfer sometimes surpasses single-task supervision.
- **Central Role of Low-Level Perception**: Low-level tasks (depth, reflectance) are simultaneously the best donors and the best sponges, implying that early visual features in VLMs are highly reusable and adaptable.

## Limitations & Future Work
- Reliance on a multiple-choice benchmark format may limit the discovery of transfer patterns in open-ended generation settings.
- Only the Qwen-2.5-VL model family is used; generalizability to other architectures (e.g., LLaVA, InternVL) is not validated.
- The default upper bound $U_j = 100\%$ may be inappropriate for certain tasks where humans also fall short of 100%.
- The transfer effects of multi-task joint fine-tuning are not studied; this work considers only single-source-task fine-tuning.
- PGF-guided data selection experiments are limited to the 7B model; validation across more models and task combinations is needed.

## Related Work & Insights
- **vs. Taskonomy**: Taskonomy, conducted in the pre-foundation-model era using CNNs with lightweight decoders, studies transfer learning where both source and target tasks undergo transfer. This work studies zero-shot cross-task transfer in the VLM era, more closely aligned with the usage paradigm of foundation models.
- **vs. Task2Vec/LEEP and similar transfer metrics**: These are information-theoretic measures defined over representations; PGF is defined directly over task performance, making it more intuitive and requiring no additional computation of representation distances.
- This work is highly valuable for designing multi-task fine-tuning strategies—practitioners can prioritize fine-tuning donor tasks, avoid pirate task data, and focus on sponge tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of zero-shot cross-task transfer among VLM perception tasks; PGF metric is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three model scales, 13 tasks, 4 seeds, video extension, and data selection application—coverage is exceptionally broad.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are clear, figures and tables are abundant, and analysis is substantive.
- Value: ⭐⭐⭐⭐ Directly actionable for VLM fine-tuning practice; the PGF metric is broadly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[CVPR 2026\] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)
- [\[ACL 2026\] From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models](../../ACL2026/multimodal_vlm/from_heads_to_neurons_causal_attribution_and_steering_in_multi-task_vision-langu.md)
- [\[CVPR 2026\] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](simpact_simulation-enabled_action_planning_using_vision-language_models.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](do_vision_language_models_need_to_process_image_tokens.md)

</div>

<!-- RELATED:END -->
