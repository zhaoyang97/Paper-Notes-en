---
title: >-
  [Paper Note] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models
description: >-
  [AAAI 2026][LLM Efficiency][Analogical Reasoning] Using mechanistic interpretability tools including Patchscopes, attention knockout, and linear probes…
tags:
  - "AAAI 2026"
  - "LLM Efficiency"
  - "Analogical Reasoning"
  - "Mechanistic Interpretability"
  - "Relational Information Encoding"
  - "Structural Alignment"
  - "Attention Intervention"
date: 2026-05-08
content_hash: bbdee7b5a501c434
---

# The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.20344](https://arxiv.org/abs/2511.20344)  
**Code**: [dmis-lab/analogical-reasoning](https://github.com/dmis-lab/analogical-reasoning)  
**Area**: LLM Efficiency
**Keywords**: Analogical Reasoning, Mechanistic Interpretability, Relational Information Encoding, Structural Alignment, Attention Intervention

## TL;DR

Using mechanistic interpretability tools including Patchscopes, attention knockout, and linear probes, this paper systematically reveals the internal mechanisms of analogical reasoning in LLMs: models can effectively encode relational information in middle-to-upper layers, but **applying** relational information to new entities is a greater bottleneck than **extracting** it; successful analogical reasoning correlates with strong structural alignment across stories, while failures reflect weakened or misaligned alignment.

---

## Background & Motivation

**The cognitive centrality of analogical reasoning**: Analogical reasoning is a cornerstone of human cognition, supporting knowledge transfer, problem solving, and creative thinking, and serves as a critical task for evaluating higher-order abstraction in intelligent systems.

**Unknown mechanisms underlying LLM analogical ability**: Prior work has evaluated LLM analogical performance at the behavioral level (e.g., Webb et al. 2023), but the internal mechanisms by which models extract and apply relations remain unexplored.

**Limitations of task vectors**: Prior research (e.g., Function Vectors) has found that LLMs can represent abstract task information in ICL settings, but this is limited to simple tasks (color mapping, antonyms) and focuses only on the existence of such vectors rather than how they are used in complex reasoning.

**Complementarity of two analogy types**:
- **Proportional analogies** (A:B::C:D): Test the ability to extract and apply semantic relations.
- **Story analogies**: Test the ability to establish structural correspondences between narratives that differ entirely in surface details.

**Core research questions**: How do LLMs extract relations between entities and apply them to predictions? How do models establish structural alignment across semantically disparate contexts? How do these mechanisms compare to human cognition?

---

## Method

### Overall Architecture

The study is organized into three progressive levels: (1) information flow analysis in proportional analogies—identifying which positions and layers are critical to answer resolution; (2) bottleneck analysis of relation application—diagnosing failure causes through substitution and patching experiments; (3) structural alignment analysis in story analogies—understanding how models identify and map high-level relational parallels.

### Key Design 1: Information Flow Analysis for Proportional Analogies

**Dataset construction**: Entity pairs are extracted from the million-scale analogy knowledge base AnalogyKB, with multi-answer and time-varying relations manually excluded, yielding 50,000 analogies. After knowledge filtering (ensuring the model possesses necessary knowledge) and reasoning-shortcut filtering (excluding samples answerable without inference), 500 correct and 500 incorrect cases are sampled per model.

**Method 1: Attention Knockout**

- Selectively disables attention connections from the resolution token to four preceding positions ($e_1$, $e_2$, the link "as", $e_3$), observing the effect on predictions.
- A window size $k$ (1/5 of total layers) is maintained to cover cross-layer propagation.
- **Results—three key findings**:
    - Knocking out $e_1$ has minimal impact, indicating it is not on the critical path for answer resolution.
    - Knocking out $e_2$ or $e_3$ causes significant performance degradation/generation instability in middle-to-upper layers, indicating these positions carry critical information.
    - In incorrect cases, knocking out the link has a strong effect in early-to-middle layers, suggesting the link may play a misleading role in erroneous reasoning.

**Method 2: Patchscopes for Decoding Hidden Representations**

- Custom target prompts are designed for $e_2$ and $e_3$ to elicit natural-language outputs reflecting what is encoded in hidden representations.
- Two types of information are distinguished:
    - **Attribute information**: Whether the representation captures intrinsic properties of the entity (e.g., "Jane Austen" is a British author).
    - **Relational information**: Whether the representation encodes the relation connecting the entity pair (e.g., "author of").
- **Key findings**:
    - Attribute information persists in middle-to-upper layers in both correct and incorrect cases.
    - **Relational information** is maintained into upper layers in correct cases but drops sharply in incorrect cases.
    - This indicates that the encoding of relational information is the decisive factor for successful analogical reasoning.

### Key Design 2: Bottleneck Diagnosis for Relation Application

**Experiment 1: Substituting the first entity pair**

- In incorrect cases, the $(e_1, e_2)$ pair is replaced with the entity pair from a correct case sharing the same relation, testing whether the model can recover.
- **Result**: Up to 38.4% of incorrect cases are corrected, indicating that a substantial portion of errors stem from insufficient relational information extraction from the first pair.
- **Key inference**: In the remaining 60%+ incorrect cases, the model still fails even when provided with correct relational information—**application is a bottleneck independent of extraction**.

**Experiment 2: Activation Patching**

- For cases still incorrect after the above substitution, hidden representations from middle-to-upper layers of $e_2$ are patched into early layers at the link position.
- **Result**: Up to 38.1% of remaining incorrect cases are corrected.
- **Mechanistic interpretation**: Relational information encoded in $e_2$ is transmitted through the link to subsequent positions; early layers at the link position require sufficient contextualization to effectively propagate this information.
- **Combined effect**: The two experiments together correct up to 55–62% of incorrect cases (varying by model), highlighting the central role of information transmission pathways in analogical reasoning.

### Key Design 3: Structural Alignment Analysis for Story Analogies

**Linear probe experiments**

- (Source story, target story) and (source story, distractor story) pairs are extracted from the StoryAnalogy dataset.
- Binary classifiers are trained on final-token activations of each attention head at each layer to assess whether analogical structure is linearly separable.
- **Result**: Middle layers (layers 20–30 in Qwen2.5-14B) achieve an average accuracy of 82.9%, indicating that analogical structure becomes linearly separable in these layers.

**Mutual Alignment Score (MAS)**

- MAS is defined as the proportion of source story tokens whose best-matching candidate token is also a mutual best match (based on cosine similarity).
- **Algorithm**: For each source token $s_i$, find its best match $c_{j^*}$ in the candidate; verify whether $c_{j^*}$'s best match in the source is $s_i$ (mutual best match); compute the proportion.
- **Results—correct cases**:
    - Source–target MAS consistently exceeds source–distractor MAS, with the largest gap in middle layers.
    - Even when the target story shares almost no lexical overlap with the source, the model captures deep structural alignment.
    - Analogical token pairs (e.g., "water"–"air", "house"–"lungs") form mutual best matches with high similarity scores.
- **Results—incorrect cases**:
    - The gap between source–target and source–distractor MAS is minimal.
    - The model exhibits stronger alignment with the distractor story across most layers.
    - This indicates that the model is susceptible to surface-level distractors when relational mappings are not robustly encoded.

---

## Key Experimental Results

### Experimental Setup

- **Models**: Llama-2-13B, Gemma-7B, Qwen2.5-14B (base models) for proportional analogies; corresponding Instruct/Chat versions for story analogies.
- **Data**: 50,000 proportional analogies constructed from AnalogyKB; 360 story analogy questions from StoryAnalogy (reformulated as binary-choice with bidirectional verification).
- **Hardware**: 2×A100 80GB.

### Summary of Three Core Findings

| Finding | Details |
|---------|---------|
| 1. Relational encoding is the key to success | Attribute information shows no difference between correct and incorrect cases; relational information is sharply absent in incorrect cases. |
| 2. Application is an independent bottleneck | 38.4% of errors stem from insufficient extraction; an additional 38.1% of errors are corrected via link patching, demonstrating that application is equally difficult. |
| 3. Structural alignment determines success | Source–target MAS is substantially higher than source–distractor MAS in correct cases; this gap disappears in incorrect cases, with the model degrading to surface-level matching. |

### Comparison with Human Cognition

- **Similarities**: Both models and humans can abstract relational information connecting entities, and relational encoding is central to analogical reasoning.
- **Differences**: For humans, once a relation is identified, applying it to a new context is relatively straightforward; LLMs face equal difficulty at the application stage, with information transmission at the link position constituting an additional bottleneck.

---

## Highlights & Insights

### Strengths

- **Distinctive research perspective**: Approaching analogical reasoning through mechanistic interpretability bridges the gap between behavioral evaluation and internal mechanisms.
- **Methodologically coherent pipeline**: Attention knockout localizes critical positions → Patchscopes analyzes encoded content → substitution/patching diagnoses failures → MAS quantifies structural alignment; each step builds on the previous.
- **Rigorous data control**: Knowledge filtering excludes "unknown" cases; shortcut filtering excludes "lucky guess" cases, ensuring that genuine analogical reasoning is analyzed.
- **MAS is defined as a new tool for quantifying structural alignment.**

### Limitations & Future Work

- Only 7–14B scale models are analyzed; mechanisms in larger models (e.g., 70B+) or closed-source models may differ.
- Entity pairs in proportional analogies are drawn from a knowledge base with limited relation types (primarily factual), excluding more abstract analogies.
- In patching experiments, the optimal layer selection requires exhaustive search; no predictive rule is established.

---

## Related Work & Insights

- **Function Vectors / Task Vectors**: Find that LLMs encode abstract task information in compact representations within ICL; this paper extends the analysis to more complex analogical reasoning and further examines how such representations are used during inference.
- **Causal Mediation Analysis**: A methodological framework for analyzing causal roles via interventions on internal activations; this paper's attention knockout and activation patching are specific applications of this framework to analogical reasoning.
- **Structure-Mapping Theory (Gentner)**: A classic cognitive science theory of analogical reasoning—establishing one-to-one correspondences through structural alignment. MAS is a computational operationalization of this theory.
- **AnalogyKB**: A million-scale analogy knowledge base used to construct controlled experimental data, demonstrating its value in mechanistic analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](../../ACL2026/llm_efficiency/are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](../../ICLR2026/llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)
- [\[AAAI 2026\] Scaling and Transferability of Annealing Strategies in Large Language Model Training](scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../../ACL2026/llm_efficiency/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ICLR 2026\] EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](../../ICLR2026/llm_efficiency/evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)

</div>

<!-- RELATED:END -->
