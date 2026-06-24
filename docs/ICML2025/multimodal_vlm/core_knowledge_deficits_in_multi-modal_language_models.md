---
title: >-
  [Paper Note] Core Knowledge Deficits in Multi-Modal Language Models
description: >-
  [ICML 2025][Multimodal VLM][Core knowledge] This paper proposes the CoreCognition benchmark (comprising 12 core cognitive abilities across 1,503 questions). Following a large-scale evaluation of 230 MLLMs, it reveals that models systematically lag behind humans in foundational cognitive abilities. Moreover, this deficit does not improve with larger model scales; instead, larger models tend to rely more heavily on shortcut learning rather than genuine understanding.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Core knowledge"
  - "multimodal LLM evaluation"
  - "cognitive science"
  - "shortcut learning"
  - "benchmark"
date: 2026-05-08
content_hash: 86dd38ed25488c28
---

# Core Knowledge Deficits in Multi-Modal Language Models

**Conference**: ICML 2025  
**arXiv**: [2410.10855](https://arxiv.org/abs/2410.10855)  
**Code**: None  
**Area**: Multimodal VLMs  
**Keywords**: Core knowledge, multimodal LLM evaluation, cognitive science, shortcut learning, benchmark

## TL;DR

This paper proposes the CoreCognition benchmark (comprising 12 core cognitive abilities across 1,503 questions). Following a large-scale evaluation of 230 MLLMs, it reveals that models systematically lag behind humans in foundational cognitive abilities. Moreover, this deficit does not improve with larger model scales; instead, larger models tend to rely more heavily on shortcut learning rather than genuine understanding.

## Background & Motivation

While current MLLMs approach or even surpass human performance in high-level reasoning tasks (e.g., chart comprehension, mathematical geometry, action recognition), they frequently fail in **low-level tasks that are simple and intuitive for humans**—such as counting, perspective-taking, spatial reasoning, temporal reasoning, and compositional reasoning. This phenomenon echoes the classic **Moravec's Paradox**: what is hardest for machines is easiest for humans.

The authors hypothesize that these deficits stem from the MLLMs' lack of **core knowledge**—foundational cognitive abilities that humans possess innately from infancy. The concept of core knowledge is rooted in Plato's view of a priori knowledge, Kant's forms of intuition, and empirical studies by developmental psychologists such as Piaget and Spelke.

Existing MLLM benchmarks mostly focus on high-level reasoning (e.g., MathVerse, ScienceQA), lacking systematic evaluations targeting these **low-level core cognitive abilities**. This work therefore constructs the first large-scale core knowledge benchmark to investigate whether, how, and to what extent MLLMs represent and utilize core knowledge.

## Method

### Overall Architecture

This work consists of three major components: **(1)** constructing a cognitive taxonomy and the CoreCognition benchmark; **(2)** conducting a large-scale model evaluation (230 models $\times$ 11 prompts = 2,530 data points); and **(3)** proposing the Concept Hacking method for controlled experiments to distinguish genuine understanding from shortcut learning.

### Key Designs

#### 1. Cognitive Taxonomy

Drawing on Piaget's four stages of cognitive development, the 12 core abilities are categorized into three developmental stages:

| Developmental Stage | Core Ability | Description |
|---------|---------|------|
| Sensorimotor Stage | Boundary | Distinguishing the transition from one object to another |
| | Continuity | Objects persist in space and time as unified wholes |
| | Permanence | Objects continue to exist even when not perceived |
| | Spatiality | A priori understanding of Euclidean spatial properties |
| Concrete Operational Stage | Perceptual Constancy | Changes in appearance do not equate to changes in physical properties |
| | Intuitive Physics | Intuition regarding physical laws |
| | Perspective Taking | Understanding what is seen from another perspective |
| | Conservation | Invariance of properties under transformations |
| | Hierarchy | Understanding inclusion/exclusion relationships |
| Formal Operational Stage | Intentionality | Understanding the intentions of others |
| | Mechanical Reasoning | Inferring behavior from the state of a system |
| | Tool Use | The ability to manipulate objects to achieve goals |

Dependencies exist between these abilities, where lower-level abilities serve as the cognitive foundation for higher-level ones.

#### 2. CoreCognition Benchmark Construction

The dataset contains **1,503 samples**, with at least 95 instances per concept, spanning both image and video inputs. The construction pipeline is as follows:

- **Prototyping**: Translating 12 theoretical concepts into 5-10 prototypical scenarios, each abstractly describing a testable cognitive situation (e.g., Object Permanence $\rightarrow$ the ball-under-cup experiment).
- **Instantiation**: Collecting visual assets from the web, public datasets, generative models, simulation environments, and physical photography, and pairing them with carefully designed questions and options to form multiple-choice questions (MCQs).
- **Quality Control**: Each QA pair underwent two rounds of independent cross-validation, with additional verification by 20 Amazon Mechanical Turk annotators. Questions where humans made consistent errors were re-evaluated.

Three core design principles:

- **Discriminativeness**: Models lacking the target core knowledge must fail.
- **Minimal Confounding**: Minimizing reliance on auxiliary capabilities such as object recognition.
- **Minimal Text Shortcut**: Preventing models from inferring the answer based on text alone.

#### 3. Inference and Evaluation Strategies

- **Circular Evaluation**: For a $k$-choice question, options are circularly rotated $k$ times, and the average accuracy is calculated to mitigate option position bias.
- **Two-Stage Scoring**: In the first stage, free text is mapped to options via template matching and LLM-as-Judge; in the second stage, it is compared with the ground truth (GT). Models with high failure rates in text parsing were excluded.
- **11 Prompts**: Covering categories such as prompt-free, thinking, explanation, reward/punishment, bias mitigation, role-playing, and cognitive instructions.

#### 4. Concept Hacking

This is the most core methodological innovation of the paper—a **controlled experimental approach** that distinguishes genuine understanding from shortcut learning by systematically manipulating causal features in images to completely invert the GT labels.

Specifically, 45 samples were selected from CoreCognition, and a **manipulated version** was created for each—keeping the question and irrelevant conditions identical, but altering task-relevant features to completely invert the correct answer.

For each pair (control/manipulated), there are four possible outcomes for the model:

| Control Question | Manipulated Question | Explanation |
|-------|-------|------|
| ✓ Correct | ✓ Correct | **Core Knowledge**: Genuine understanding of the concept |
| ✓ Correct | ✗ Incorrect | **Shortcut Learning**: Reliance on surface patterns, failing after manipulation |
| ✗ Incorrect | ✓ Correct | Coincidental correctness (answering correctly for the wrong reason) |
| ✗ Incorrect | ✗ Incorrect | **Core Deficit**: Complete lack of the target core knowledge |

### Loss & Training

This is an evaluation study and does not involve model training. The core contributions lie in the benchmark construction and evaluation methodology.

## Key Experimental Results

### Main Results

230 models were evaluated (25 commercial + 205 open-source), covering parameter scales from 1B to 110B.

| Model | Sensorimotor Stage Mean | Concrete Stage Mean | Formal Stage Mean | Total Mean |
|------|-------------|-------------|-------------|-------|
| Human | ~82.1 | ~83.0 | ~87.2 | 86.98 |
| GPT-o1 | 65.3 | 72.3 | 90.3 | 74.91 |
| GPT-4o | 67.8 | 62.1 | 86.5 | 69.25 |
| Qwen2.5-VL-72B | 62.3 | 64.2 | 88.0 | 68.29 |
| QVQ-72B | 67.6 | 69.8 | 58.3 | 68.07 |
| InternVL3-78B | 65.7 | 57.4 | 60.2 | 64.60 |
| Claude-3.5-Sonnet | 57.9 | 55.8 | 79.9 | 61.92 |

**Key Findings**: All MLLMs systematically lag behind humans in low-level stages (sensorimotor and concrete stages) by 15-23 percentage points, while approaching or exceeding human-level performance in the formal operational stage. Commercial models do not consistently outperform open-source models.

### Ablation Study

#### Impact of Prompts

| Prompt Type | Representative | Relative Effect |
|-----------|------|---------|
| Prompt-free | Empty string | Baseline |
| Thinking-based | "Let's think step by step" | Minor improvement |
| Explanation-based | Requesting explanation | No significant improvement |
| Reward/Punishment | Offering a $200 tip | No significant improvement |
| Cognitive Instruction | Providing concept description | **+6%+**, the only effective prompt |

The effectiveness of cognitive instruction prompts suggests that core knowledge may be encoded in a distributed manner in model parameters, where explicit conceptual hints act as "retrieval cues".

#### Reasoning Models vs. Instruct Models

Among the 12 core abilities, reasoning models (e.g., GPT-o1, QVQ) show no significant difference from their instruct-tuned counterparts on 10 of them. Reasoning models perform marginally better only on perceptual constancy ($P=0.067$) and actually worse on perspective-taking ($P=0.004$). Reasoning paradigms and test-time scaling fail to effectively mitigate core knowledge deficits.

### Key Findings

1. **Core Knowledge Deficits**: MLLMs systematically perform worse on low-level abilities compared to high-level abilities, contrasting sharply with the consistently high level of human performance.

2. **Misaligned Dependency**: Performance in high-level abilities is uncorrelated with underlying supporting abilities (Pearson $\rho < 0.4$), lacking the hierarchically structured dependency seen in human cognitive development.

3. **Not Scaling**: The scaling slopes for 7 out of 9 low-level abilities are significantly lower than those for high-level abilities; perspective-taking even exhibits **inverse scaling** (performance degrades as model size increases).

4. **Shortcut Intensification**: Concept Hacking experiments reveal that larger models are more prone to falling into the "shortcut" or "core deficit" quadrants, rather than shifting toward the human core knowledge region. Even the strongest models like GPT-4o exhibit substantial shortcut dependency.

5. **Core Abilities Predict High-Level Performance**: Except for perspective-taking and intuitive physics, core abilities are strongly correlated with performance on 26 public benchmarks and high-level tasks in SEED-Bench.

## Highlights & Insights

- **Unique Interdisciplinary Perspective**: This paper systematically introduces core knowledge theories from developmental cognitive science into MLLM evaluation, establishing solid theoretical foundations ranging from Plato to Piaget.
- **Unprecedented Evaluation Scale**: Evaluating 230 models across 11 prompt types ($230 \times 11 = 2,530$ data points), covering commercial, open-source, and reasoning models.
- **Novel Concept Hacking Methodology**: By systematically manipulating causal features to invert the GT, this method accurately distinguishes genuine understanding from shortcut learning, offering a more principled evaluation paradigm than standard adversarial perturbations.
- **Revealing Limitations of Scaling Laws**: The study clearly shows that "simply scaling up models" cannot resolve core knowledge deficits and may even degrade certain abilities (e.g., the inverse scaling of perspective-taking).
- **Discovery of Cognitive Instruction Prompts**: Simple descriptions of the concepts boost accuracy by over 6%, suggesting that core knowledge may exist in a distributed format but is difficult to spontaneously retrieve.

## Limitations & Future Work

1. **VQA Format Limitations**: Reliance on auxiliary abilities such as language comprehension, counting, and object recognition prevents complete elimination of confounding factors, and also limits the assessment of non-verbal models.
2. **Limited Scalability of Concept Hacking**: The manual and meticulous design of manipulated pairs is time-consuming. Currently, there are only 45 pairs, making large-scale expansion challenging.
3. **No Training Intervention**: This work only diagnoses the deficiencies but does not propose concrete training methodologies to remedy core knowledge deficits.
4. **Static Evaluation**: Humans acquire core knowledge through interaction, and static VQA formats may not fully capture this ability.
5. **Future Directions**: Future research could explore distilling core knowledge prior to pre-training, designing core-knowledge-enhanced training curricula, and introducing purely visual (non-linguistic) evaluation formats.

## Related Work & Insights

- Compared to cognitive benchmarks such as M3GIA and Marvel, CoreCognition focuses on lower-level core cognition rather than high-level general intelligence.
- Compared to developmental psychology benchmarks like DevBench, this work targets multimodal models rather than text-only ones.
- Literature on shortcut learning (e.g., Alvi 2018, Bahng 2020) provides a theoretical foundation for Concept Hacking.
- This study presents a strong counterexample to the "scaling is all you need" optimism, echoing critiques by Bender et al. 2021 and Mitchell & Krakauer 2023.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The perspective of core knowledge and the Concept Hacking method are entirely novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 230 models $\times$ 11 prompts with comprehensive ablations and rigorous statistical testing.
- Writing Quality: ⭐⭐⭐⭐ — The philosophical introduction is engaging but slightly lengthy, while the main structure remains clear.
- Value: ⭐⭐⭐⭐⭐ — Provides a profound diagnosis of the fundamental limitations of MLLMs, offering important guidance for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG](../../ACL2025/multimodal_vlm/core_mmrag_knowledge_reconciliation.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](../../NeurIPS2025/multimodal_vlm/evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [\[ICML 2025\] Vision-Language Models Create Cross-Modal Task Representations](vision-language_models_create_cross-modal_task_representations.md)
- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/mmrl_multi-modal_representation_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
