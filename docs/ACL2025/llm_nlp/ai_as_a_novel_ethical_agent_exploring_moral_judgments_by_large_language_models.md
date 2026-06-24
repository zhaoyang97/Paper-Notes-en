---
title: >-
  [Paper Note] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models
description: >-
  [ACL 2025][LLM (Other)][AI ethics] This paper systematically explores the moral judgment capabilities of large language models (LLMs) as novel ethical agents. By constructing an evaluation benchmark covering multiple ethical frameworks, it reveals the preference patterns, consistency flaws, and cultural biases of LLMs in moral reasoning.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "AI ethics"
  - "moral judgment"
  - "large language models"
  - "moral reasoning"
  - "value alignment"
date: 2026-05-08
content_hash: 24727904ee875bec
---

# AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models

**Conference**: ACL 2025  
**Area**: LLM/NLP  
**Keywords**: AI ethics, moral judgment, large language models, moral reasoning, value alignment

## TL;DR
This paper systematically explores the moral judgment capabilities of large language models (LLMs) as novel ethical agents. By constructing an evaluation benchmark covering multiple ethical frameworks, it reveals the preference patterns, consistency flaws, and cultural biases of LLMs in moral reasoning.

## Background & Motivation

**Background**: As LLMs are widely deployed in scenarios requiring value judgments (e.g., content moderation, legal consulting, psychological counseling), understanding their moral reasoning capabilities and limitations has become crucial. Existing research mainly focuses on the "alignment" perspective, using techniques like RLHF to make models output "safe" responses, but lacks an in-depth analysis of the models' inherent moral judgment patterns.

**Limitations of Prior Work**: Current evaluations of LLM moral capabilities face three challenges: first, the evaluation scope is narrow, often limited to toxic content detection rather than complex moral dilemmas; second, there is a lack of systematic coverage of different ethical frameworks (utilitarianism, deontology, virtue ethics, etc.); third, the influence of cultural background on moral judgment is largely ignored.

**Key Challenge**: LLMs implicitly learn a certain "morality" from training data, but there is a lack of systematic analysis regarding whether this morality is consistent, whether it biases toward specific ethical traditions, and how it performs in edge cases.

**Goal**: To build a multi-dimensional evaluation framework for moral judgment to systematically analyze LLM moral reasoning performance under different ethical paradigms, revealing their judgment patterns and potential flaws.

**Key Insight**: Inspired by research in moral psychology, this work designs a series of classic moral dilemma scenarios (such as variants of the trolley problem) combined with multiple ethical theoretical frameworks to systematically evaluate LLM moral judgments.

**Core Idea**: To treat LLMs as a new type of "ethical agent" for multi-framework, cross-cultural moral judgment capability evaluations.

## Method

### Overall Architecture
The authors construct a multi-dimensional moral judgment evaluation benchmark consisting of the scenario dimension (covering personal, social, and professional ethics), the theoretical dimension (utilitarianism, deontology, virtue ethics, care ethics, etc.), and the cultural dimension (Western, East Asian, South Asian cultural backgrounds). Multiple mainstream LLMs are evaluated to analyze their consistency, preferences, and sensitivity in moral judgments.

### Key Designs

1. **Multi-Framework Moral Dilemma Dataset**:

    - **Function**: Provides standardized moral judgment test scenarios covering multiple ethical theories.
    - **Mechanism**: Over 500 moral dilemma scenarios are designed, each with explicit ethical tension points. The scenarios employ a pairwise comparative design—the same dilemma has different "correct" judgments under different ethical frameworks. For instance, in the trolley problem, utilitarianism advocates pulling the switch (maximizing overall welfare), whereas deontology argues against active harm (even if not pulling the switch leads to more casualties). Each scenario also includes localized cultural versions.
    - **Design Motivation**: Traditional moral evaluation largely relies on binary right-or-wrong classification. This dataset can uncover which ethical framework a model prefers.

2. **Moral Consistency Analysis**:

    - **Function**: Evaluates the internal consistency of LLM moral judgments.
    - **Mechanism**: Consistency is detected via three approaches: (1) Semantic equivalence test—checking if different phrasings of the same dilemma yield the same judgment; (2) Framework consistency test—verifying if the model consistently follows the same ethical framework across similar dilemmas; (3) Reverse test—analyzing if judgments change after swapping character attributes (e.g., race, gender, social status) in the scenario.
    - **Design Motivation**: A reliable moral reasoning system should maintain logical consistency, remaining uninfluenced by presentation phrasing or irrelevant attributes.

3. **Ethical Tendency Profiling**:

    - **Function**: Generates an "ethical profile" of moral judgments for each LLM.
    - **Mechanism**: The frequency distribution of each model choosing judgments from various ethical frameworks across all scenarios is counted, and radar charts are plotted to demonstrate the models' ethical tendencies. It also analyzes whether this tendency correlates with the cultural sources of the training data—for instance, whether models trained with more Chinese corpus lean further toward collectivist ethics.
    - **Design Motivation**: Understanding a model's ethical "default stance" is crucial for deployment in sensitive scenarios.

### Loss & Training
This paper is an evaluative work and does not involve model training. The evaluation covers mainstream models including GPT-4o, Claude 3.5, Llama 3, Mistral, and Qwen.

## Key Experimental Results

### Main Results

| Model | Utilitarian Tendency (%) | Deontological Tendency (%) | Virtue Ethics Tendency (%) | Consistency Score |
|------|----------------|-------------|----------------|----------|
| GPT-4o | 42.3 | 31.5 | 18.7 | 0.78 |
| Claude 3.5 | 35.8 | 38.2 | 19.4 | 0.82 |
| Llama 3-70B | 47.1 | 28.7 | 16.8 | 0.71 |
| Qwen-72B | 38.5 | 29.3 | 23.6 | 0.74 |

### Ablation Study (Consistency Analysis)

| Test Type | GPT-4o | Claude 3.5 | Llama 3 | Description |
|---------|--------|-----------|---------|------|
| Semantic Equivalence Consistency | 85.3% | 88.1% | 79.6% | Different phrasings of the same dilemma |
| Framework Consistency | 72.4% | 76.8% | 65.3% | Consistency across similar dilemmas |
| Attribute Irrelevance | 68.7% | 74.2% | 61.8% | Judgments independent of character attributes |
| Cultural Sensitivity | 78.1% | 80.5% | 70.4% | Responses under different cultural frameworks |

### Key Findings
- All evaluated models display a utilitarian tendency, leaning toward options that "maximize overall welfare" when facing trade-offs. This likely stems from the "helpful to most people" preference signal in RLHF training.
- Moral consistency is generally low—the same model has a 25-35% probability of giving contradictory judgments in structurally similar but superficially different dilemmas.
- The attribute irrelevance test yields the worst performance, indicating that models' moral judgments are implicitly influenced by character attributes (e.g., gender, social status).
- The Claude series performs the best in deontological tendencies and consistency, which might be related to Anthropic's Constitutional AI training methodology.

## Highlights & Insights
- Synthesizes systematic methodologies from moral psychology into LLM evaluation, establishing a far more refined evaluation framework than the binary "toxic/safe" classification. This multi-dimensional evaluation method can be transferred to other value alignment research.
- The attribute irrelevance test reveals implicit biases that persist even after RLHF, offering crucial warnings for AI safety research.

## Limitations & Future Work
- The design of moral dilemmas is inevitably influenced by the researchers' own cultural backgrounds, potentially neglecting certain culturally specific ethical dimensions.
- Models' moral judgments can be subtly influenced by prompt phrasing, where different wordings for the same dilemma can yield divergent judgments, indicating a need for improved robustness.
- Current evaluations are limited to textual interaction scenarios; moral judgments in multimodal settings (such as image content moderation) have not been explored.
- "Correct moral judgment" itself is a philosophically contested issue; the evaluation framework in this work assumes a pluralistic stance.
- Future work could explore how to enable LLMs to provide more consistent judgments after explicitly declaring the ethical framework they adopt.
- This evaluation framework can be extended to assess AI moral judgment capabilities in specific professional scenarios (such as medical or business ethics).

## Related Work & Insights
- **vs Delphi**: Delphi trains task-specific moral judgment models, whereas this work focuses on the implicit moral tendencies of general-purpose LLMs.
- **vs ETHICS benchmark**: ETHICS focuses more on commonsense moral knowledge, whereas this work delves deeper into a framework-level analysis.
- **vs Constitutional AI**: Anthropic's approach uses rules to constrain model outputs, whereas the analysis in this work can provide finer-grained guidance for such methods.
- The evaluation framework of this paper is also applicable to evaluating whether emerging reasoning models (such as o1) possess more consistent moral reasoning capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically analyzing LLM moral judgments from multi-ethical frameworks is a relatively fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model and multi-dimensional systematic evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of interdisciplinary research.
- Value: ⭐⭐⭐⭐⭐ Holds highly significant reference value for AI safety and value alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Acquisition and Application of Novel Knowledge in Large Language Models](acquisition_and_application_of_novel_knowledge_in_large_language_models.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [\[ACL 2025\] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning](probabilistic_aggregation_and_targeted_embedding_optimization_for_collective_mor.md)

</div>

<!-- RELATED:END -->
