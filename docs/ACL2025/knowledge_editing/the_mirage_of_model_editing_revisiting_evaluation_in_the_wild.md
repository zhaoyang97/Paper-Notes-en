---
title: >-
  [Paper Note] The Mirage of Model Editing: Revisiting Evaluation in the Wild
description: >-
  [ACL 2025 Main][Knowledge Editing][Model Editing] This paper reveals systematic flaws in the evaluation practices of the model editing field—the near-perfect success rates (~96.8%) reported by prior methods plunge to 38.5% in real-world application scenarios. The root cause is the leakage of ground-truth information through teacher forcing during testing. The authors propose the QAEdit benchmark and the WILD evaluation framework to foster more reliable evaluations.
tags:
  - "ACL 2025 Main"
  - "Knowledge Editing"
  - "Model Editing"
  - "Knowledge Editing Evaluation"
  - "Teacher Forcing Leakage"
  - "Sequential Editing"
  - "QAEdit Benchmark"
date: 2026-05-08
content_hash: 30523f012bae1db3
---

# The Mirage of Model Editing: Revisiting Evaluation in the Wild

**Conference**: ACL 2025 Main  
**arXiv**: [2502.11177](https://arxiv.org/abs/2502.11177)  
**Code**: [GitHub](https://github.com/wanliyoung/revisit-editing-evaluation)  
**Area**: Knowledge Editing  
**Keywords**: Model Editing, Knowledge Editing Evaluation, Teacher Forcing Leakage, Sequential Editing, QAEdit Benchmark

## TL;DR

This paper reveals systematic flaws in the evaluation practices of the model editing field—the near-perfect success rates (~96.8%) reported by prior methods plunge to 38.5% in real-world application scenarios. The root cause is the leakage of ground-truth information through teacher forcing during testing. The authors propose the QAEdit benchmark and the WILD evaluation framework to foster more reliable evaluations.

## Background & Motivation

**Background**: Model editing aims to precisely modify specific knowledge in LLMs without full retraining. Representative methods like ROME (locate-and-edit), MEMIT (batch editing), and FT-L (fine-tuning specific layers) have reported editing success rates close to 96-100% on standard benchmarks (such as CounterFact and ZsRE).

**Limitations of Prior Work**: Despite their excellent performance on artificially constructed evaluation sets, the effectiveness of model editing in real-world applications has never been systematically verified. When researchers deploy edited models to actual QA scenarios, the performance falls far short of expectations. The huge gap between the two suggests a fundamental issue with the evaluation methods themselves.

**Key Challenge**: The core issue with previous evaluations lies in the use of **teacher forcing** during testing. When evaluating the model's generation quality, the prefix tokens of the ground-truth answer are fed into the model, requiring it to only predict the last few tokens. This is equivalent to seeing most of the answers beforehand during an exam, leading to a severe overestimation of the success rate. In real deployment, the model must autoregressively generate the complete answer from scratch, where errors propagate and accumulate across tokens.

**Goal**: (1) Systematically expose the specific flaws of existing evaluations; (2) Construct QAEdit, a new benchmark aligned with real-world QA tasks; (3) Design WILD, a task-agnostic evaluation framework; (4) Evaluate the actual performance of existing methods in real-world scenarios.

**Key Insight**: Starting from the critical observation that "teacher forcing leaks both the content and length of the answer during testing," the authors point out that the use of teacher forcing in prior work's testing essentially constitutes cheating—it tells the model what the answer roughly looks like (content leakage) and hints at the length of the answer (length leakage).

**Core Idea**: Replace traditional evaluation methods with autoregressive generation (without teacher forcing), and build a more reliable editing evaluation system in combination with real-world QA datasets.

## Method

### Overall Architecture

The dual contribution of WILD (evaluation framework) + QAEdit (evaluation benchmark). QAEdit derives editing targets and evaluation samples from widely used QA datasets, while the WILD framework defines a standardized evaluation workflow: no teacher forcing, evaluation of complete autoregressively generated answers, and consideration of the side effects of editing on the model's general capabilities.

### Key Designs

1. **QAEdit Benchmark Construction**:

    - **Function**: Provide model editing evaluation data aligned with real QA scenarios.
    - **Mechanism**: Extract factual QA pairs from mainstream QA datasets (e.g., SQuAD, Natural Questions, etc.) and convert them into model editing tasks—i.e., modifying specific knowledge within the model so that it yields updated answers to related questions. Unlike synthetic data such as CounterFact, the questions and answers in QAEdit come from real user queries, making the evaluation closer to actual usage scenarios.
    - **Design Motivation**: Existing benchmarks like CounterFact use artificially constructed knowledge triples and templated questions, which have a huge gap from how real users query LLMs. QAEdit directly reflects the real distribution of "how users ask."

2. **WILD Evaluation Framework—Removing Teacher Forcing**:

    - **Function**: Define a standardized editing evaluation workflow that reflects real-world usage scenarios.
    - **Mechanism**: During evaluation, the model must autoregressively generate the complete answer from scratch without being provided with any ground-truth prefixes. Three dimensions are evaluated: (a) **editing success rate**—whether the model outputs the expected new answer; (b) **generalizability**—whether the model can correctly answer different phrasings of the edited knowledge; (c) **locality**—whether unedited knowledge remains unchanged. A key change is replacing token-probability-based metrics with exact match (EM) / F1 scores.
    - **Design Motivation**: Teacher forcing leaks the content and length of the ground truth during testing, allowing the model to score highly without truly "understanding" the edit. Only by removing it can the true performance of these methods be observed.

3. **Sequential Editing Experimental Design**:

    - **Function**: Simulate scenarios in real deployment where the model needs to be edited multiple times consecutively.
    - **Mechanism**: Sequentially perform 1000 edits on the model, evaluating the retention rate of all previously edited knowledge and the model's general capability after each edit. This simulates the real-world demand for continuous knowledge updates—such as news events and personnel changes that require frequent updates to LLM knowledge.
    - **Design Motivation**: Prior works primarily evaluate single-shot edits, but actual deployment inevitably requires a large number of consecutive edits. Sequential editing exposes the fundamental vulnerability of these methods when scaled up.

### Loss & Training

This is an evaluation-focused work and does not involve new training methods.

## Key Experimental Results

### Main Results

Performance differences under different evaluation methods in the single-edit scenario (Llama-2-7B):

| Editing Method | Traditional Evaluation (Teacher Forcing) | WILD Evaluation (Autoregressive) | Performance Gap |
|---------|------------------------|------------------|---------|
| ROME | 96.1% | 37.8% | -58.3% |
| MEMIT | 96.8% | 38.5% | -58.3% |
| FT-L | 89.5% | 35.2% | -54.3% |
| MEND | 72.3% | 28.6% | -43.7% |
| Direct Fine-Tuning | 85.0% | 33.1% | -51.9% |

### Ablation Study

Analysis of the contribution of different leakage factors in teacher forcing:

| Evaluation Condition | Editing Success Rate | Description |
|---------|----------|------|
| Full Teacher Forcing | 96.8% | Leaks content + length |
| Length-only Leakage | ~75% | Informs about answer length without content prefix |
| No Leakage (WILD) | 38.5% | Fully autoregressive generation |
| Sequential Editing (100 times) | ~30% | Degradation after consecutive edits |
| Sequential Editing (1000 times) | <10% | Model collapses almost completely |

### Key Findings

- **Teacher forcing is the core reason for artificially high performance**: Removing teacher forcing drops the success rate of all methods by 40-60 percentage points, demonstrating that most of the previously reported results stem from evaluation leakage rather than truly effective knowledge editing.
- **Sequential editing leads to catastrophic degradation**: Just 1000 edits reduce the success rate of all methods below 10%, while the model's general capabilities also severely degrade. This implies that current model editing methods are practically unusable in real scenarios of continuous knowledge updating.
- **Extremely poor generalizability of edits**: Even if edited to say "The US president is X", when asked with slightly different phrasings (such as "Who leads the US"), the model often still yields the old answer. This indicates that editing only modifies superficial input-output mappings without truly updating internal knowledge representations.

## Highlights & Insights

- **Revealing a domain-level evaluation vulnerability**: Teacher forcing, which is theoretically a training technique, introduces systematic bias when repurposed for testing. This discovery is not only applicable to model editing but also serves as a warning for all studies using teacher forcing to evaluate generation quality.
- **Simple yet powerful experimental design**: Just by changing the evaluation method (removing teacher forcing), a drop of dozens of percentage points in performance is revealed, proving that a good evaluation methodology is a major scientific contribution in itself.
- **Practical value of sequential editing experiments**: The paper systematically demonstrates the complete failure of editing methods in scaled-up scenarios for the first time, providing crucial warnings for the commercial application of model editing (such as real-time knowledge updates).

## Limitations & Future Work

- **No improved method proposed**: The paper focuses on exposing issues but does not provide concrete proposals on how to improve editing methods to make them effective under the WILD evaluation.
- **Experiments limited to Llama-2-7B**: It is not verified whether larger or newer models (e.g., Llama-3, Qwen2) behave differently.
- **Editing targets in QAEdit might be relatively simple**: Knowledge triples from QA datasets may not cover all types of knowledge editing demands (e.g., logical rules, causal relationships, etc.).
- **Future directions**: There is a need to fundamentally rethink model editing—possibly combining it with gentler knowledge injection methods such as Retrieval-Augmented Generation (RAG) or Adapters (LoRA).

## Related Work & Insights

- **vs ROME/MEMIT**: These methods report 96%+ success rates on CounterFact, but this paper proves that this data is heavily distorted. Their core assumption—precisely locating and overwriting knowledge in specific MLP layers—does not hold in real-world QA scenarios.
- **vs Knowledge Neurons / Causal Tracing**: These knowledge localization methods provided the theoretical foundation for editing, but this paper implies that localization itself might also be misled by teacher forcing evaluation.
- **Inspiration for subsequent model editing research**: Any new method must be validated using the WILD evaluation framework; otherwise, it risks falling into previous evaluation traps.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Reveals a domain-wide evaluation vulnerability with profound insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Control experiments are clear and powerful, though model and dataset coverage can be further extended.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous reasoning and highly impactful presentation of findings.
- **Value**: ⭐⭐⭐⭐⭐ Fundamentally impacts the model editing field, forcing the community to re-evaluate existing achievements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards a Principled Evaluation of Knowledge Editors](towards_a_principled_evaluation_of_knowledge_editors.md)
- [\[ACL 2025\] DocMEdit: Towards Document-Level Model Editing](docmedit_towards_document-level_model_editing.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](../../ICML2026/knowledge_editing/revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](../../ICLR2026/knowledge_editing/energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICML 2026\] From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing](../../ICML2026/knowledge_editing/from_backward_spreading_to_forward_replay_revisiting_target_construction_in_llm_.md)

</div>

<!-- RELATED:END -->
