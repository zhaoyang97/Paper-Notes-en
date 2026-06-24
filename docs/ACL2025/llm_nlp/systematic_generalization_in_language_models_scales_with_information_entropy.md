---
title: >-
  [Paper Note] Systematic Generalization in Language Models Scales with Information Entropy
description: >-
  [ACL 2025][LLM (Other)][Systematic generalization] This paper proposes using the information entropy of constituents in training data to quantify the difficulty of systematic generalization. Experiments demonstrate that models like Transformers, RNNs, and CNNs can achieve OOD systematic generalization under high-entropy training data even without compositional priors, with generalization performance positively correlated with entropy.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Systematic generalization"
  - "compositional generalization"
  - "information entropy"
  - "SCAN"
  - "sequence-to-sequence"
date: 2026-05-08
content_hash: 973822368677c953
---

# Systematic Generalization in Language Models Scales with Information Entropy

**Conference**: ACL 2025  
**arXiv**: [2505.13089](https://arxiv.org/abs/2505.13089)  
**Code**: Yes  
**Area**: LLM NLP  
**Keywords**: Systematic generalization, information entropy, compositionality, SCAN, out-of-distribution generalization

## TL;DR
Demonstrates that the systematic generalization ability of language models is positively correlated with the information entropy of constituent distributions in training data—standard seq2seq models without built-in compositional priors can achieve strong systematic generalization under high-entropy training distributions.

## Background & Motivation

**Background**: Systematic generalization (combining learned components into new positions/combinations) is a core capability of language understanding, but current models perform poorly.

**Limitations of Prior Work**: Existing research focuses on "whether models can systematically generalize" (a binary judgment), but lacks a **method to measure the difficulty of systematic generalization problems**.

**Key Challenge**: Some argue that architectural compositional priors must be introduced to achieve systematic generalization, while other studies show that standard models can generalize under certain settings—does architecture or data determine generalization ability?

**Goal**: Provide an information-theorict framework to measure the difficulty of systematic generalization and prove that standard models can generalize under high-entropy conditions.

**Key Insight**: Utilize the distributional entropy of constituents (e.g., verbs) across different syntactic positions in training data as a measure of generalization difficulty.

**Core Idea**: The higher the distributional entropy of the training data, the stronger the model's systematic generalization capability, independent of the number of training samples.

## Method

### Overall Architecture
Constructs a seq2seq dataset based on a modified SCAN grammar to control the distributional entropy of verbs in embedded clauses within the training set. It restricts a verb $v_1$ from appearing in the clause $e_1$ in the training set, and reverses this distribution during testing to observe whether the model can generalize $v_1$ to the $e_1$ position.

### Key Designs

1. **Entropy Metric for Systematic Generalization**:

    - **Function**: Measures generalization difficulty using the information entropy of the verb distribution in $e_2$: $H_{e_2}^{\text{train}}(V) = -\sum_{v} p(v) \log_2 p(v)$
    - Low entropy = concentrated verb distribution, making it hard for the model to learn position-independent verb representations; high entropy = uniform distribution, making generalization easier.

2. **Two Entropy Scaling Methods**:

    - **Vertical Scaling**: Fixes the number of verb types and adjusts the probability distribution towards uniformity.
    - **Horizontal Scaling**: Increases the number of new verb types while maintaining a uniform distribution.
    - **Design Motivation**: Disentangles the two factors: "distribution uniformity" and "lexical diversity".

3. **Comparison of Four Model Architectures**: LSTM, Transformer, COGS-Transformer, and T5, all of which are standard architectures without built-in compositional priors.

## Key Experimental Results

### Main Results (Generalization Accuracy vs. Entropy)

| Entropy Level | LSTM | Transformer | T5 |
|---|---|---|---|
| Extremely Low (0 bit) | ~0% | ~0% | ~0% |
| Low (1 bit) | ~30% | ~40% | ~45% |
| Medium (2 bits) | ~70% | ~80% | ~85% |
| High (≥3 bits) | ~95% | **~98%** | **~99%** |

### Ablation Study (Decoupling Sample Size vs. Entropy)

| Configuration | Generalization Accuracy | Description |
|---|---|---|
| Low Entropy + Large Sample Size | Low | No matter how large the sample size, low entropy prevents generalization |
| High Entropy + Small Sample Size | High | The key factor is the uniformity of the distribution |
| Horizontal vs. Vertical Scaling | Similar Curves | The effects of both methods are equivalent |

### Key Findings
- **Entropy is the Core Factor of Generalization**: Entropy remains the decisive factor even when controlling for sample size.
- **No Compositional Priors Needed**: Standard Transformers can achieve ~98% accuracy on high-entropy data.
- **Generalization under Low Entropy Remains an Open Problem**: All models fail when entropy ≈ 0.

## Highlights & Insights
- **Elegant Theoretical Framework**: Unifies the measurement of generalization difficulty using information entropy, converting a philosophical question into a quantifiable information-theoretic problem.
- **Strong Evidence for "Data Determines Generalization"**: It is not that models cannot generalize, but rather that the distributional properties of the training data dictate the generalization capability.
- **Connecting to Human Language Acquisition**: The high-entropy condition resembles the diverse linguistic inputs encountered by humans.

## Limitations & Future Work
- **Validated Only on SCAN Variants**: There is a gap between synthetic grammars and natural language.
- **Only Considers the Entropy of Verb Distributions**: Natural language compositionality involves many more constituents.
- **Not Extended to Pre-trained Large Language Models**
- **Future Directions**: Extend to natural language datasets; study the relationship between the entropy of pre-training data distributions and downstream generalization.

## Related Work & Insights
- **vs. Lake & Baroni (2018) SCAN**: The original SCAN showed generalization failures but did not consider the impact of the training distribution; this work proves that distributional entropy is key.
- **vs. Lake & Baroni (2023)**: They improved generalization via curriculum learning, whereas this paper analyzes the more fundamental properties of the distribution directly.
- **vs. Dziri et al. (2024)**: They analyzed compositional deficits in LLMs from a capability perspective, whereas this paper provides an explanation from a data perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The information entropy framework has extremely strong explanatory power for systematic generalization.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 models × 2 scaling methods × multiple entropy levels, though limited to synthetic data.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous formal definitions.
- Value: ⭐⭐⭐⭐ Provides important theoretical guidance for understanding and improving generalization capabilities.

**Code**: [ltgoslo/systematicity-entropy](https://github.com/ltgoslo/systematicity-entropy)  
**Area**: llm_nlp  
**Keywords**: Systematic generalization, compositional generalization, information entropy, SCAN, sequence-to-sequence

## TL;DR

This paper proposes using the information entropy of constituents in training data to quantify the difficulty of systematic generalization. Experiments demonstrate that models like Transformers, RNNs, and CNNs can achieve OOD systematic generalization under high-entropy training data even without compositional priors, with generalization performance positively correlated with entropy.

## Background & Motivation

1. **Systematic Generalization is the Core of Language Ability**: Humans can easily recombine known concepts into new contexts (e.g., swapping subject and object), but neural networks have long struggled with this compositional generalization, which remains a long-standing open problem in NLP.
2. **The Connectionism vs. Symbolism Debate**: Fodor & Pylyshyn (1988) famously criticized continuous representations for failing to support systematic generalization. More than 30 years later, researchers are still exploring whether neural networks can achieve robust compositional behavior.
3. **Prior Solutions Depend on Priors**: Previous research improved generalization by introducing architectural compositional priors (such as perturbation-equivariant models from Gordon et al., 2019) or special training procedures (such as Lake & Baron, 2023), but these methods are task-dependent.
4. **Lack of a Unified Metric for Generalization Difficulty**: Although compositional generalization benchmarks like SCAN exist, there is a lack of quantitative metrics for "how difficult a systematic generalization problem is," making compatibility across different experiments poor.
5. **Distributional Perspective of Data Ignored**: Most works focus on architectural design and rarely study the boundary conditions of generalization capability from the perspective of the training data's distributional properties.
6. **Key Insight**: The authors propose a data-centric perspective, using the information entropy of constituent components in the training set to quantify systematic generalization difficulty, exploring the fundamental question of "whether neural networks can generalize under prior-free conditions."

## Method

### Task Setup

Constructs a sequence-to-sequence task based on a modified SCAN grammar: the input is $x = (e_1, c, e_2)$, where $e_1, e_2$ are embedded sentences and $c$ is a conjunction (`and`/`after`). The output $y$ is the corresponding action sequence. The vocabulary contains 8 verbs (more than the 4 in the original SCAN) and several adverbial modifiers.

### Entropy Framework for Systematic Generalization

- **Training/Test Distribution Construction**: In the training set, a certain verb $v_1$ never appears in $e_1$ but appears in $e_2$; in the test set, $e_1$ only contains $v_1$. This constitutes an OOD systematic generalization problem.
- **Core Metric**: Quantifies generalization difficulty using the information entropy $H$ of the verb distribution in $e_2$. $H=0$ represents a degenerate distribution (hardest) to $H=\log_2 |V|=3$, representing a uniform distribution (easiest).
- **OOD Even at Maximum Entropy**: Note that even when $H$ is maximized, $v_1$ still never appears in $e_1$, so the generalization is always out-of-distribution.

### Two Ways of Increasing Entropy

1. **Vertical Scaling (Distribution Mixing)**: Fixes the vocabulary size and continuously adjusts $H$ via parameter $\lambda$, which controls the mixing ratio of a degenerate distribution $D$ (containing only $v_1$) and a uniform distribution $U$ (not containing $v_1$).
2. **Horizontal Scaling (Incremental Support)**: Progressively increases the size of the verb support set in $e_2$ (expanding from $\{v_1\}$ to the entire $V$). Each step uses a uniform distribution, causing $H$ to grow logarithmically.

### Experimental Models

- **Transformer**: Original encoder-decoder architecture, absolute position encoding + GLU activation
- **RNN**: Bidirectional encoder-decoder + attention
- **CNN**: Encoder-decoder + attention from Gehring et al. (2017)
- **Equivariant Model**: Gordon et al. (2019), enforcing verb equivariance using a cyclic group (as a baseline/upper bound)

## Key Experimental Results

### Experiment 1: Vertical Scaling (Fixed 6000 Training Samples)

| Model | H≈0 | H≈1 | H≈2 | H≈3 |
|------|------|------|------|------|
| Transformer | ~0%| ~30%| ~98%| ~100%|
| RNN | ~0%| ~10%| ~50%| ~95%|
| CNN | ~0%| ~5%| ~40%| ~90%|
| Equivariant Model | ~100%| ~100%| ~100%| ~100%|

**Key Findings**: The accuracy of all three standard architectures is positively correlated with $H$. The Transformer exhibits the highest informational efficiency in the full-support scenario, reaching close to 100% accuracy when $H \approx 2$.

### Experiment 2: Horizontal Scaling (Training set size grows with the support set)

| Model | H=0 | H=1 | H=2 | H=3 |
|------|------|------|------|------|
| Transformer | ~0%| ~10%| ~80%| ~100%|
| RNN | ~0%| ~15%| ~85%| ~100%|
| CNN | ~0%| ~20%| ~90%| ~100%|
| Equivariant Model | ~100%| ~100%| ~100%| ~100%|

**Key Findings**: Under horizontal scaling, the performance of the three models is close, with CNN being slightly superior on average. For the Transformer, reducing the support set is more detrimental than distribution shift.

### Supplementary Experiments

- **Sample Size Independence**: Under 3000/4000/6000 training samples, the performance-entropy relationship of the Transformer remains consistent, indicating that generalization is related to entropy rather than the volume of data.
- **Position Encoding**: Absolute positional encoding outperforms relative encodings like RoPE and DeBERTa, because the position information of verbs is independent of the rest of the sequence.

## Highlights & Insights

- Formally correlates systematic generalization difficulty with information entropy for the first time, providing a quantifiable theoretical framework.
- Proves that standard architectures (without priors) can achieve OOD compositional generalization under high-entropy training data, challenging the traditional view that "compositional priors are mandatory."
- Disentangles two different scaling methods (vertical/horizontal), revealing the models' varying sensitivities to distribution shape vs. support set size.
- Ingenious experimental design: avoids unsolvable boundary conditions by controlling degenerate scenarios (requiring $|C| > 1$ when $H = 0$).
- The discovery that generalization capability is independent of model size/data volume provides significant theoretical insights.

## Limitations & Future Work

- Only studies systematic generalization based on embedded sentences (a subset of Hadley's definition) and does not cover broader scenarios of compositional generalization.
- Relies on synthetic SCAN data; the compositional distribution entropy of training data cannot be precisely calculated in natural language.
- The vocabulary contains only 8 verbs, which is relatively small; whether the conclusion holds under larger vocabularies or more complex grammars requires further validation.
- Does not explore decoder-only Transformers (such as GPT-family models), using only encoder-decoder architectures.
- Under low entropy ($H \leq 1$), all standard models perform poorly, and the paper does not propose a solution.

## Related Work & Insights

- **vs. SCAN (Lake & Baroni, 2018)**: Extends SCAN to 8 verbs, removes the `turn` operator, and ensures all samples contain embedded sentences.
- **vs. Gordon et al. (2019)**: Their equivariant model resolves generalization through architectural priors, whereas this paper demonstrates that high-entropy data can partially substitute for priors.
- **vs. Keysers et al. (2020)**: They quantified training/test distribution differences using KL divergence, whereas this paper focuses on the entropy of the constituent components' distribution within the training set itself.
- **vs. Zhou et al. (2023)**: They modified both compositional functions and primitive distributions to study complexity, whereas this paper fixes the compositional functions and only varies the primitive distribution.
- **vs. Wiedemer et al. (2023)**: They focused on support set consistency in image generation, whereas this paper further shows that the support set alone is insufficient—entropy is the pivotal metric.

## Rating

- Novelty: ⭐⭐⭐⭐ — Quantifying systematic generalization difficulty from the perspective of information entropy is a novel and insightful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four models, two entropy scaling methods, and multiple control experiments, showing rigorous design.
- Writing Quality: ⭐⭐⭐⭐ — Clear formalization and well-integrated figures, though some notations are heavy.
- Value: ⭐⭐⭐⭐ — Provides a quantifiable theoretical framework and a new baseline for compositional generalization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](compositional_generalization_instruction.md)
- [\[ACL 2025\] A Systematic Study of Compositional Syntactic Transformer Language Models](a_systematic_study_of_compositional_syntactic_transformer_language_models.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](information_locality_as_an_inductive_bias_for_neural_language_models.md)
- [\[ICLR 2026\] Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning](../../ICLR2026/llm_nlp/compositional-arc_assessing_systematic_generalization_in_abstract_spatial_reason.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)

</div>

<!-- RELATED:END -->
