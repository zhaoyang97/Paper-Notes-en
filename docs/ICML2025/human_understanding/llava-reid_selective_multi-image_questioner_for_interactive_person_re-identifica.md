---
title: >-
  [Paper Note] LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification
description: >-
  [ICML 2025][Human Understanding] This paper defines a new task of interactive person re-identification (Inter-ReID), constructs the Interactive-PEDES multi-turn dialogue dataset, and proposes LLaVA-ReID—a large multimodal question generation model based on selective multi-image context and look-ahead supervision, which progressively refines target person descriptions through iterative dialogue.
tags:
  - "ICML 2025"
  - "Human Understanding"
date: 2026-05-08
content_hash: f0f5b548b29e797a
---

# LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification

**Conference**: ICML 2025  
**arXiv**: [2504.10174](https://arxiv.org/abs/2504.10174)  
**Area**: Human Understanding  

## TL;DR

This paper defines a new task of interactive person re-identification (Inter-ReID), constructs the Interactive-PEDES multi-turn dialogue dataset, and proposes LLaVA-ReID—a large multimodal question generation model based on selective multi-image context and look-ahead supervision, which progressively refines target person descriptions through iterative dialogue.

## Background & Motivation

Traditional text-based person re-identification (T-ReID) assumes that the descriptions provided by witnesses are complete and given all at once. However, in real-world scenarios, witness descriptions are often partial or vague. This fundamental assumption is inconsistent with reality.

Inspired by Sherlock Holmes-style questioning, where a detective gradually obtains more details from a witness through targeted questions, this paper proposes an **interactive person re-identification framework** that iteratively refines the initial description through multi-turn dialogues to identify the target person more accurately.

## Method

### Task Definition

Interactive person re-identification is a multi-turn dialogue and retrieval process:

- The witness provides an initial description $T$.
- In each round $t$, the system generates a question $Q_t$ to guide the witness in recalling more details.
- The witness provides an answer $A_t$, and the dialogue context $\mathcal{D}_t = \{T, (Q_1, A_1), \ldots, (Q_t, A_t)\}$ is used to retrieve the target person.

### Interactive-PEDES Dataset

The dataset contains 54,749 images and 13,051 individuals, with an average of 9 dialogue turns per image. The construction consists of three steps:

1. **Coarse-to-Fine Description Generation**: GPT-4o is used to generate coarse-grained initial descriptions (simulating witness impressions) and fine-grained descriptions (simulating witness potential memories).
2. **Sub-description Decomposition**: Fine-grained descriptions are decomposed into non-overlapping sub-descriptions, with each focusing on a unique attribute.
3. **Dialogue Generation**: Three types of questions are generated—descriptive questions (50%), yes/no questions (40%), and multiple-choice questions (10%).

### Interactive ReID Framework

The framework comprises three components:

- **Retriever**: A CLIP-based dual-stream network that encodes dialogue descriptions and person images in a shared cross-modal space:
  $$p(I_i|\mathcal{D}_t) = \frac{\exp \text{sim}(z_t, f_i)}{\sum_j^m \exp \text{sim}(z_t, f_j)}$$
- **Questioner**: LLaVA-ReID, which generates discriminative questions based on visual and textual contexts.
- **Answerer**: An LLM based on Qwen2.5-7B-Instruct that simulates witness answers.

### LLaVA-ReID: Selective Multi-Image Questioner

#### Selective Visual Context

Traditional methods directly use top-k or k-means selection for candidate images, which lacks attention to fine-grained differences. LLaVA-ReID designs a **hard pass selection model**:

1. Obtain top-k candidates using the retriever.
2. Feed the candidate image embeddings and the dialogue embedding into a shallow Transformer encoder: $\mathbf{v} = \phi_s(f_c; z_t)$.
3. Predict selection weights via a linear layer: $\mathbf{w} = \text{Softmax}(\phi_h(\mathbf{v}))$.
4. Send the top-c candidates with the highest selection weights to the LMM.

During training, a **Gumbel-top-k relaxation** is used to achieve a differentiable random sampling strategy.

#### Look-Ahead Supervision (Looking-Forward)

Different questions yield varying information gains depending on the retrieval state. The paper proposes a one-step look-ahead strategy to dynamically select the most informative question:

$$Q_t^* = \underset{Q_i \in (\mathcal{S} \setminus \mathcal{Q}_{pre}^{t-1})}{\arg\max} \text{rank}(I_{gt}, \{T, A_1, \ldots, A_{t-1}, A_t^*\})$$

The question that maximizes the improvement in the target person's retrieval rank is chosen as the supervision signal for the current round, trained using NLL loss:

$$\mathcal{L}_{\text{NLL}} = -\log p(Q_t^* | \mathcal{C}_{t-1}, \mathcal{D}_{t-1})$$

## Key Experimental Results

### Interactive-PEDES Main Results

| Method | R3@1 | R5@1 | R5@5 | BRI ↓ |
|---|---|---|---|---|
| Initial | 35.86 | 35.86 | 55.17 | - |
| SimIRV | 50.45 | 61.27 | 82.00 | 1.024 |
| ChatIR | 57.85 | 63.86 | 83.81 | 0.935 |
| PlugIR | 60.34 | 65.44 | 85.33 | 0.849 |
| **LLaVA-ReID** | **63.96** | **73.20** | **90.62** | **0.719** |

After 5 rounds of interaction, R@1 improves by 37.34% (73.20 vs 35.86), outperforming PlugIR by 7.76%.

### Transfer to Traditional T-ReID

By integrating LLaVA-ReID, IRRA improves R@1 from 73.38 to 78.51 and RDE from 75.94 to 79.39 on CUHK-PEDES, demonstrating the transferability of the method.

### Ablation Study

- Removing selective visual context: R@1 decreases by approximately 3%.
- Removing look-ahead supervision: R@1 drops from 73.20 to approximately 68%.
- The number of candidates $c=4$ is the optimal choice.

## Highlights & Insights

- **First to define the Inter-ReID task**: Extending static text-based person re-identification to interactive dialogue retrieval.
- **Well-crafted dataset**: Interactive-PEDES contains three types of questions, simulating real-world questioning scenarios.
- **Innovative look-ahead strategy**: Dynamically selecting questions with the maximum information gain, avoiding the combinatorial explosion of question permutations.
- **Strong transferability**: Serves as a plug-and-play module to boost the performance of existing T-ReID frameworks.

## Limitations & Future Work

- The witness simulation uses an LLM instead of real human evaluation, which may lead to a gap with real-world scenarios.
- Image sources in the dataset are relatively limited (CUHK-PEDES and ICFG-PEDES).
- Forward inference of large multimodal models is required for each round, potentially limiting real-time performance.
- The look-ahead strategy requires pre-computing retrieval ranks for all candidate questions, incurring high training overhead.
- The robustness of the system is not discussed when witness answers are inaccurate or contradictory.

## Rating

⭐⭐⭐⭐ (4/5)

The paper demonstrates strong innovation in task definition, dataset construction, and method design. Upgrading person re-identification from static retrieval to interactive dialogue is a natural and valuable direction, with convincing experimental results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](scaling_large_motion_models_with_million-level_human_motions.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ECCV 2024\] Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification](../../ECCV2024/human_understanding/multi-memory_matching_for_unsupervised_visible-infrared_person_re-identification.md)
- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](../../ICCV2025/human_understanding/one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](../../ICCV2025/human_understanding/weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)

</div>

<!-- RELATED:END -->
