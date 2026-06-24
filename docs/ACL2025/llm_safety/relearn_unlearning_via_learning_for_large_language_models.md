---
title: >-
  [Paper Note] ReLearn: Unlearning via Learning for Large Language Models
description: >-
  [ACL 2025][LLM Safety][Knowledge Unlearning] ReLearn proposes replacing traditional "reverse optimization" with "forward learning" to achieve knowledge unlearning in LLMs. Through a pipeline of data augmentation and fine-tuning, the model forgets target knowledge while maintaining language generation quality and fluency. A comprehensive evaluation framework involving KFR, KRR, and LS is also designed.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Knowledge Unlearning"
  - "Machine Unlearning"
  - "Data Augmentation"
  - "Language Quality Preservation"
  - "Reverse Optimization"
date: 2026-05-08
content_hash: 15ae91feef30995f
---

# ReLearn: Unlearning via Learning for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.11190](https://arxiv.org/abs/2502.11190)  
**Code**: [GitHub](https://github.com/zjunlp/unlearn)  
**Area**: LLM Safety  
**Keywords**: Knowledge Unlearning, Machine Unlearning, Data Augmentation, Language Quality Preservation, Reverse Optimization  

## TL;DR
ReLearn proposes replacing traditional "reverse optimization" with "forward learning" to achieve knowledge unlearning in LLMs. Through a pipeline of data augmentation and fine-tuning, the model forgets target knowledge while maintaining language generation quality and fluency. A comprehensive evaluation framework involving KFR, KRR, and LS is also designed.

## Background & Motivation

**Background**: Large language models absorb massive amounts of data during training, which may contain private information, harmful content, or knowledge that needs to be "forgotten". Machine unlearning aims to selectively make the model forget specific knowledge while preserving its capabilities on other tasks. Modern mainstream LLM unlearning methods are based on reverse optimization, such as Gradient Ascent (GA) or Negative Preference Optimization (NPO).

**Limitations of Prior Work**: Although reverse optimization methods can reduce the generation probability of target tokens, they cause severe side effects—destroying the model's linguistic coherence and next-token prediction capabilities. After forgetting the target knowledge, the model may produce garbled, repetitive, or grammatically incorrect outputs on related or even unrelated topics, leading to a severe degradation of overall language quality.

**Key Challenge**: The essence of reverse optimization is "pushing backward" against the model's parameters, which not only affects the parameter space of the target knowledge but also damages the model's general language generation capability. Furthermore, existing evaluation metrics focus excessively on "whether the target knowledge is forgotten" (contextual forgetting) while ignoring the fluency and relevance of response generation after unlearning.

**Goal**: (1) To design an unlearning method that does not rely on reverse optimization, achieving knowledge unlearning while preserving language generation quality; (2) To establish a more comprehensive evaluation framework that assesses both unlearning effectiveness and language quality.

**Key Insight**: Reverse optimization damages language quality because it performs "reverse" updates along the gradient directions. From a different perspective—if the model is provided with "alternative responses that do not contain the target knowledge" for forward fine-tuning, it can forget the original knowledge without compromising its language capability.

**Core Idea**: Generate alternative responses for the target knowledge via data augmentation (e.g., "I don't know" or plausible alternative facts), and then use normal forward fine-tuning (rather than reverse optimization) to make the model "learn" the new response style, thereby "forgetting" the old knowledge.

## Method

### Overall Architecture
The workflow of ReLearn consists of three steps: (1) **Data Augmentation**: Generate alternative response data (including refusals, alternative facts, etc.) for the knowledge points that need to be forgotten; (2) **Forward Fine-tuning**: Standardly fine-tune the model with the augmented data to replace the model's response patterns for the target knowledge; (3) **Comprehensive Evaluation**: Comprehensively evaluate the unlearning performance using three metrics: KFR, KRR, and LS. The entire process is fully compatible with standard instruction fine-tuning pipelines, requiring no modifications to the optimizer or training procedure.

### Key Designs

1. **Data Augmentation for Unlearning**:

    - Function: Generate high-quality alternative training data for the knowledge to be forgotten.
    - Mechanism: For each knowledge question-answer pair $(q, a)$ targeted for unlearning, generate multiple alternative responses: (a) refusal-type responses: "I cannot provide this information", "I do not have relevant knowledge", etc.; (b) alternative-fact responses: replace the original answer with logical but different content; (c) misleading responses: intentionally provide incorrect but plausible-sounding information. Concurrently, retain a "retain set" of data (QA pairs unrelated to the unlearning target) to ensure the model does not lose general capabilities during the unlearning process.
    - Design Motivation: Forward fine-tuning requires explicit target labels, and data augmentation provides supervision signals representing "how the model should respond to queries about forgotten knowledge."

2. **Forward Fine-tuning for Unlearning**:

    - Function: Achieve knowledge unlearning through a standard training process, avoiding the destructiveness of reverse optimization.
    - Mechanism: Mix the augmented unlearning data with the retain set and fine-tune the model using the standard causal language modeling objective (cross-entropy loss). The loss function is $\mathcal{L} = \mathcal{L}_{forget} + \lambda \mathcal{L}_{retain}$, where $\mathcal{L}_{forget}$ is the loss on the alternative responses, and $\mathcal{L}_{retain}$ is the loss on the retain set. The training process is identical to normal fine-tuning—the model is simply "learning" a new way to respond rather than being "pushed backward".
    - Design Motivation: Forward fine-tuning updates parameters along normal gradient directions, thereby avoiding damage to the model's established language generation capacity and coherence.

3. **Comprehensive Evaluation Framework (KFR + KRR + LS)**:

    - Function: Comprehensively evaluate the unlearning performance from three dimensions: knowledge forgetting, knowledge retention, and language quality.
    - Mechanism: (a) **Knowledge Forgetting Rate (KFR)**: Measures the extent to which the model "no longer knows" the target knowledge on the forget set by checking whether responses still contain the target information; (b) **Knowledge Retention Rate (KRR)**: Measures whether the model's knowledge on the retain set remains intact after forgetting the target knowledge; (c) **Language Score (LS)**: Evaluates the fluency, coherence, and grammatical correctness of the generated text post-unlearning using an independent language model. These three metrics collectively reflect the overall quality of the unlearning method.
    - Design Motivation: Existing metrics focus only on "how thoroughly the model forgets" while neglecting the equally vital question: "can the model still speak properly after forgetting?"

### Loss & Training
Standard cross-entropy loss is adopted, mixing the target forget set and the retain set in proportion during training. The learning rate is set to 2e-5, and epochs are adjusted based on the dataset size (typically 1-3 epochs). ReLearn supports multiple base models, including Llama-3-8B-Instruct, Gemma-2-2B-IT, and Llama-2-7B-Chat. Optionally, a DPO variant (ReLearn_DPO) can be introduced to formulate the original and alternative responses as preference pairs for optimization.

## Key Experimental Results

### Main Results
Comparison with reverse optimization baselines on KnowUnDo and TOFU benchmarks (Llama-2-7B-Chat):

| Method | KFR ↑ | KRR ↑ | LS ↑ | Overall Rank |
|------|-------|-------|------|---------|
| Original Model (No Unlearning) | 0.0 | 100.0 | High | - |
| Gradient Ascent (GA) | 72.5 | 45.3 | Low (Severe degradation) | 4 |
| NPO | 68.9 | 52.1 | Medium (With degradation) | 3 |
| SURE | 65.4 | 58.7 | Medium | 3 |
| Memflex | 59.2 | 61.3 | Medium-High | 3 |
| **Ours (ReLearn)** | **78.3** | **82.6** | **High (Near original)** | **1** |
| **Ours (ReLearn_DPO)** | **81.1** | **79.8** | **High** | **1** |

### Ablation Study
Contribution analysis of different data augmentation strategies and components:

| Configuration | KFR | KRR | LS | Description |
|------|-----|-----|-----|------|
| Refusal-only Responses | 71.2 | 85.3 | High | Conservative but incomplete unlearning |
| Alternative Facts Only | 76.8 | 78.1 | High | Deeper unlearning but may affect related knowledge |
| Mixed Augmentation (Full Method) | 78.3 | 82.6 | High | Best balance |
| Without Retain Set | 80.1 | 61.4 | Medium-High | Enhanced forgetting but retention capacity is damaged |
| ReLearn_DPO Variant | 81.1 | 79.8 | High | Preference optimization further strengthens unlearning |

### Key Findings
- **Reverse optimization indeed destroys language coherence**: Through mechanistic analysis, this work clearly demonstrates how GA and NPO disrupt the model's attention patterns and MLP activations, leading to misalignment in next-token predictions. ReLearn, using only forward updates, entirely avoids this issue.
- **Training with a retain set is critical for maintaining general capabilities**: Removing the retain set drops the KRR significantly by ~21%, indicating that continuously "reminding" the model of other knowledge during unlearning is essential.
- **DPO variant achieves superior unlearning strength**: By treating original and alternative responses as preference pairs, the model learns more explicitly "what to avoid and what to say", resulting in a ~3% increase in KFR.
- Consistent effectiveness is demonstrated across different base models (Llama-3, Gemma-2), proving the good generalizability of the proposed method.

## Highlights & Insights
- **Reverse thinking of using "learning" to achieve "forgetting"**: Instead of disrupting weights through reverse optimization, this method teaches the model new response styles through forward fine-tuning, elegantly avoiding the degradation of language quality. This approach can be transferred to other scenarios requiring model behavior "modification" rather than capacity "deletion".
- **Introduction of the LS metric**: This study is the first to systematically focus on post-unlearning language quality, filling a critical gap in the evaluation framework. Many previous unlearning methods performed well on standard metrics but yielded very poor actual generation quality.
- **Mechanistic analysis section**: By visualizing attention patterns and intermediate layer activations, the paper intuitively explains why reverse optimization corrupts language generation, laying a theoretical foundation for future studies in unlearning.

## Limitations & Future Work
- The quality of data augmentation relies heavily on prompt design, and different types of knowledge may require distinct augmentation strategies.
- The unlearning scale verified so far is relatively small (dozens to hundreds of knowledge items); the effectiveness of large-scale unlearning (e.g., forgetting an entire domain of knowledge) remains uncharted.
- Alternative-fact responses might introduce new hallucinated/fake knowledge, posing potential safety risks.
- Relational dependencies between knowledge are not considered—forgetting knowledge A might impact highly correlated knowledge B.
- **Future Directions**: Finer control over the scope of unlearning based on knowledge graphs could be explored, as well as the recoverability of the model post-unlearning (whether forgotten knowledge can be restored reversely).

## Related Work & Insights
- **vs Gradient Ascent (GA)**: GA directly maximizes target token loss, which is simple but highly destructive. ReLearn replaces it with forward learning, preserving model stability.
- **vs TOFU**: TOFU provides a standardized benchmark for unlearning, but its metrics are still centered on contextual forgetting. ReLearn's evaluation framework serves as a strong complement to TOFU.
- **vs KnowUnDo**: KnowUnDo focuses on knowledge unlearning in privacy preservation scenarios and serves as one of the major experimental baselines in this paper.
- The authors' team (ZJU-NLP) has made continuous contributions to knowledge editing and machine unlearning, also achieving second place in the SemEval 2025 Unlearning Challenge.

## Rating
- Novelty: ⭐⭐⭐⭐ The philosophy of "learning to forget" is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple base models, benchmarks, and mechanistic analysis, though the unlearning scale remains small.
- Writing Quality: ⭐⭐⭐⭐ Well-defined problems and compelling motivation.
- Value: ⭐⭐⭐⭐ Practically meaningful for LLM safety and privacy protection, with prominent contributions from the evaluation framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](manu_modality_aware_unlearning.md)
- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[ICML 2025\] Learning Safety Constraints for Large Language Models](../../ICML2025/llm_safety/learning_safety_constraints_for_large_language_models.md)

</div>

<!-- RELATED:END -->
