---
title: >-
  [Paper Note] Function-to-Style Guidance of LLMs for Code Translation
description: >-
  [ICML 2025][Code Intelligence][code translation] F2STrans is proposed to progressively fine-tune LLMs in two stages: functional learning (correctness) and style learning (readability). This allows Qwen-1.5B to outperform prompt-enhanced Qwen-32B and GPT-4 on average across 20 code translation scenarios.
tags:
  - "ICML 2025"
  - "Code Intelligence"
  - "code translation"
  - "LLM fine-tuning"
  - "functional learning"
  - "style learning"
  - "readability"
date: 2026-05-08
content_hash: ebca4969ce17c169
---

# Function-to-Style Guidance of LLMs for Code Translation

**Conference**: ICML 2025  
**arXiv**: [2507.11083](https://arxiv.org/abs/2507.11083)  
**Code**: Yes (the paper mentions that the model and benchmark have been released)  
**Area**: Code Intelligence  
**Keywords**: code translation, LLM fine-tuning, functional learning, style learning, readability

## TL;DR
F2STrans is proposed to progressively fine-tune LLMs in two stages: functional learning (correctness) and style learning (readability). This allows Qwen-1.5B to outperform prompt-enhanced Qwen-32B and GPT-4 on average across 20 code translation scenarios.

## Background & Motivation

### Background

**Background**: LLMs have made progress in code translation (e.g., Java to Python), but the correctness and readability of the translation results remain challenging.

**Limitations of Prior Work**: Most methods focus on functional correctness, but the resulting code style is unnatural with poor readability. There is also a lack of benchmarks that evaluate both functionality and style.

**Key Challenge**: Functional correctness and code style are objectives of two different dimensions; direct translation may achieve functional correctness but result in an unnatural style.

**Goal**: To design a progressive framework that first ensures functional correctness and then optimizes code style.

**Key Insight**: Mining high-quality code pairs from online programming platforms for functional learning, followed by style learning using positive and negative style samples.

**Core Idea**: Decoupling code translation optimization into two steps: "functionality first, style second."

### Mechanism

**Goal**: ### Overall Architecture
Stage 1 - Functional Learning: Fine-tuning with high-quality source-target code pairs to optimize translation correctness.


## Method

### Overall Architecture
Stage 1 - Functional Learning: Fine-tuning with high-quality source-target code pairs to optimize translation correctness.
Stage 2 - Style Learning: Further fine-tuning with positive/negative style samples to guide the model towards outputting more natural code styles.

### Key Designs
1. **High-Quality Code Pair Mining**: Pairing different language submissions for the same problem from LeetCode/Codeforces, and filtering to select functionally equivalent, high-quality code pairs. **Design Motivation**: Real-world multilingual code is more natural than synthetic data, and functional correctness is guaranteed by test cases.

2. **Style Learning**: Introducing contrastive learning concepts by using positive examples (conforming to target language idioms) and negative examples (functionally correct but stylistically unnatural). The model learns to shift toward a more natural style while preserving functionality.

3. **New Benchmark**: Includes the latest source code, extensive test cases, and human-annotated ground truths, supporting 20 translation scenarios (pairwise translation among 5 languages) to evaluate both functionality and style simultaneously.

### Loss & Training
- Functional learning stage: Standard SFT loss
- Style learning stage: Contrastive/preference learning loss combining positive and negative samples

## Key Experimental Results

### Main Results (20 Translation Scenarios)

| Model | Functional Accuracy | Style Score | Overall |
|------|----------|---------|------|
| F2STrans (Qwen-1.5B) | Best | Best | SOTA |
| Qwen-32B + prompt | High | Medium | Inferior to 1.5B FT |
| GPT-4 + prompt | High | Medium | Inferior to 1.5B FT |

### Ablation Study

| Configuration | Functional Accuracy | Style Score | Description |
|------|----------|---------|------|
| Full F2STrans | Highest | Highest | Both stages included |
| Functional Learning Only | High | Medium | Correct but unnatural |
| Style Learning Only | Low | High | Good style but potentially incorrect |
| Direct Joint Training | Medium | Medium | Inferior to staged training |

### Key Findings
- 1.5B fine-tuning outperforms 32B and GPT-4 prompting methods
- Decoupled training outperforms joint training
- Online programming platforms are a treasure trove of high-quality translation data

## Highlights & Insights
- The "functionality first, style second" approach can be generalized to other code generation tasks
- The superiority of fine-tuning small models over prompting large models is validated once again

## Limitations & Future Work
- Only covers 5 programming languages
- Style evaluation partially relies on human annotation
- Formal verification of semantic equivalence has not been explored

## Related Work & Insights
- Code quality is not just about "being runnable"; style and readability are equally important
- The choice of data sources has a significant impact on translation quality

## Rating
- Novelty: ⭐⭐⭐⭐ Novel functional-style decoupled training paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐ 20 scenarios, new benchmark
- Writing Quality: ⭐⭐⭐⭐ Clear description of methods
- Value: ⭐⭐⭐⭐ Direct application value for code translation tools

---

## Additional Reflections

### Relationship with Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: (1) the growing demand for an in-depth understanding of LLM internal mechanisms; (2) the increasing importance of model efficiency and accessibility; and (3) AI safety and reliability becoming core concerns. From a methodological perspective, this work represents a paradigm shift from "black-box utilization" to "white-box understanding."

### Specific Suggestions for Future Work
1. The core idea of this paper can be combined with other modalities (vision, speech).
2. Consider validating the generalizability of the conclusions on larger-scale models and datasets.
3. Explore the possibility of combining this approach with reinforcement learning and online learning.
4. Develop automated evaluation and optimization toolchains.


---

## Additional Reflections

### Relationship with Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: evaluation of model capabilities and reliability guarantees, parameter-efficient fine-tuning and model compression, and AI safety and alignment. From a methodological perspective, this work represents an exploration of the deeper mechanisms of LLMs, helping to drive the paradigm shift from empirically-driven to theoretically-driven research.

### Specific Suggestions for Future Work
1. The core idea can be combined with other modalities (vision, speech, multi-modal) to verify the cross-modal generalizability of the method.
2. Validate the conclusions on larger-scale models (70B+) and newer architectures (such as Mixture-of-Experts).
3. Explore the possibility of integration with reinforcement learning and online learning to achieve dynamic adaptation.
4. Develop automated evaluation and optimization tools to lower the barrier to using the method.
5. Consider the intersection with LLM alignment research to explore the synergetic optimization of safety and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Bootstrapping Code Translation with Weighted Multilanguage Exploration](../../ACL2026/code_intelligence/bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](../../ACL2025/code_intelligence/rethinking_repetition_problems_of_llms_in_code_generation.md)
- [\[ICML 2026\] Bridging Functional Correctness and Runtime Efficiency Gaps in LLM-Based Code Translation](../../ICML2026/code_intelligence/bridging_functional_correctness_and_runtime_efficiency_gaps_in_llm-based_code_tr.md)
- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](../../ICML2026/code_intelligence/poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)
- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](../../ICML2026/code_intelligence/matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)

</div>

<!-- RELATED:END -->
