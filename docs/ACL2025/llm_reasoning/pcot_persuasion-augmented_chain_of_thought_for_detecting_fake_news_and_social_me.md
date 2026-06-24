---
title: >-
  [Paper Note] PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation
description: >-
  [ACL 2025][Reasoning][PCoT] PCoT (Persuasion-Augmented Chain of Thought) is proposed as a two-stage zero-shot method. In the first stage, LLMs are guided by prompts incorporating persuasion knowledge to identify six types of persuasion strategies in texts. In the second stage, the persuasion analysis is integrated as context into disinformation detection reasoning. On average, this improves the F1 score by 15% across 5 LLMs and 5 datasets, including two brand-new post-cutoff…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "PCoT"
  - "persuasion augmentation"
  - "fake news detection"
  - "Chain-of-Thought"
  - "zero-shot classification"
date: 2026-05-08
content_hash: 540a4ffb5c029fe2
---

# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation

**Conference**: ACL 2025  
**arXiv**: [2506.06842](https://arxiv.org/abs/2506.06842)  
**Code**: [GitHub](https://github.com/ArkadiusDS/PCoT)  
**Area**: LLM Reasoning / Disinformation Detection  
**Keywords**: PCoT, persuasion augmentation, fake news detection, Chain-of-Thought, zero-shot classification  

## TL;DR

PCoT (Persuasion-Augmented Chain of Thought) is proposed as a two-stage zero-shot method. In the first stage, LLMs are guided by prompts incorporating persuasion knowledge to identify six types of persuasion strategies in texts. In the second stage, the persuasion analysis is integrated as context into disinformation detection reasoning. On average, this improves the F1 score by 15% across 5 LLMs and 5 datasets, including two brand-new post-cutoff datasets.

## Background & Motivation

**Disinformation detection faces zero-shot generalization challenges.** Traditional supervised methods rely heavily on human-annotated data, performing poorly in cross-dataset generalization and facing limitations regarding data scarcity. Lucas et al. (2023) demonstrated that zero-shot LLMs outperform BERT fine-tuned on different datasets during cross-dataset testing. However, existing zero-shot methods typically ask "true or false" directly, lacking fine-grained analysis.

**Psychological research reveals that "identifying persuasion improves judgment."** Experiments by Hruschka & Appel (2023) show that when individuals are trained to identify persuasion fallacies (such as appeal to emotion, false authority, and strawman arguments), their ability to distinguish true from fake news improves significantly. One of the essential characteristics of disinformation is the use of various persuasion techniques to mislead the audience.

**Core Idea: Transferring human cognitive patterns to LLMs.** If teaching humans to identify persuasion techniques can enhance their judgment, can similar improvements be achieved by first letting LLMs analyze the persuasion strategies in a text and then making truth judgments based on this analysis? PCoT is the computational implementation of this cognitively-inspired concept.

## Method

### Overall Architecture

PCoT is a two-stage pipeline. The first stage (Persuasion Detection Step) requires the LLM to identify and analyze six categories of persuasion strategies in the text, generating binary labels and explanations for each strategy. The second stage (Disinformation Detection Step) integrates the persuasion analysis from the first stage as additional context to perform zero-shot binary classification for truth judgment. The same LLM is used in both stages.

### Key Designs

1. **Persuasion Detection Step (First Stage)**:
    - **Function**: Performs multi-class, multi-label persuasion strategy identification and explanation on the input text.
    - **Mechanism**: The model $M$ receives input $X=(T, I_P, K_P, G_P)$, where $T$ is the raw text, $I_P$ is the persona setting (overwriting alignment tuning), $K_P$ is the incorporated persuasion knowledge (definitions and sub-techniques of the six strategies), and $G_P$ is the task guidance. The output is a structured analysis $A_T = \{p_i: (y_{p_i}, E_{p_i}) | p_i \in P\}$ for each strategy, consisting of a binary label $y_{p_i}$ (presence/absence) and an explanation $E_{p_i}$.
    - Six categories of persuasion strategies (based on the classification system by Piskorski et al. 2023): Attack on Reputation [AR], Justification [J], Simplification [S], Distraction [D], Call to Action [C], and Manipulative Wording [MW].
    - Prompt variant comparison: **Detailed Multitask (DMT)** detects all strategies in a single prompt (optimal, micro-average F1 of 0.722) vs Detailed One Task At a Time (DTAT, 0.689) vs Base Multitask (0.664). DMT outperforms the baseline by 9%.
    - **Design Motivation**: Incorporating concrete knowledge (strategy definitions + sub-techniques) is more effective than merely listing names; generating explanations is more robust than outputting labels alone.

2. **Disinformation Detection Step (Second Stage)**:
    - **Function**: Conducts zero-shot binary classification based on persuasion analysis.
    - **Mechanism**: The model processes $X=(T, I_D, A_T, G_D)$, where $A_T$ is the output of the persuasion analysis from the first stage, $I_D$ is the detection persona setting, and $G_D$ is the detection task guidance. The output is $Y_T$ (disinformation: Yes/No). Crucially, $A_T$ provides a structured context about the use of persuasion strategies in the text, enhancing the depth of reasoning.
    - **Design Motivation**: Separating the process into two stages ensures that the integrity of the persuasion analysis is not interfered with by the detection task. Experiments show that the two-stage approach (F1 0.815) outperforms merging them into a single prompt (F1 0.765).

3. **PCoT Variants Adapted to Three Baseline Detection Methods**:
    - **Function**: Validates whether PCoT remains effective across different prompt designs.
    - **Mechanism**: Adapts PCoT into three competing methods—VaN (base prompt), Z-CoT (zero-shot chain-of-thought prompt, inspired by Kojima et al. 2022), and DeF-SpeC (emphasizing contextual, deductive, and abductive reasoning). Each method modifies its prompt to incorporate the first-stage persuasion analysis.
    - **Design Motivation**: To ensure that the improvements stem from the injection of persuasion knowledge rather than specific prompt engineering.

### Two Brand-New Evaluation Datasets

- **MultiDis**: Approximately 2,000 English European disinformation articles annotated by researchers and fact-checking experts from multiple countries using a three-round annotation process (86.78% agreement in the first two rounds), covering 8 thematic categories.
- **EUDisinfo**: Approximately 400 English articles from the EUvsDisinfo database, sourced from the EU anti-disinformation initiative.
- Both datasets only contain articles published after January 2024, ensuring they are not part of the pre-training data of any tested LLMs.

## Key Experimental Results

### Main Results: PCoT vs. Baselines (Average of 5 LLMs)

| Method | F1 (± std) | Gain vs. Baseline |
|------|-----------|-----------|
| Base (Average of VaN/Z-CoT/DeF-SpeC) | 0.711 ± 0.055 | — |
| PCoT Single Step | 0.765 ± 0.072 | +8% |
| **PCoT (Two-Stage)** | **0.815 ± 0.027** | **+15%** |

### Detailed Improvements per Model × Method (Representative Results)

| Model | Method | Base F1 | PCoT F1 | Gain |
|------|------|---------|---------|------|
| Gemini 1.5 Flash | VaN | 0.681 | 0.810 | +19% |
| Claude 3 Haiku | Z-CoT | 0.588 | 0.774 | +32% |
| Llama 3.3 70B | VaN | 0.740 | 0.845 | +14% |
| GPT-4o Mini | Z-CoT | 0.765 | 0.846 | +11% |

### Ablation Study

| Ablation Configuration | F1 | Explanation |
|---------|----|----- |
| PCoT Two-Stage (Full) | 0.815 | Optimal |
| PCoT Single Step (Persuasion analysis + detection in a single prompt) | 0.765 | Two-stage separation is superior |
| No Explanation (Persuasion labels only, without generating explanations) | Lower | Explanations improve robustness |
| DMT Prompt (with knowledge) vs. Base MT (without knowledge) | 0.722 vs. 0.664 | Knowledge injection improves performance by 9% |

### Key Findings

- **PCoT is consistently effective across all 25 experimental settings (5 models × 5 datasets)**, demonstrating that the improvements are not coincidental to specific models or datasets.
- **Weaker models benefit more**: Claude 3 Haiku + Z-CoT achieved the largest gain (+32%), indicating that persuasion knowledge provides stronger compensation for models with weaker reasoning capabilities.
- **Equally effective on post-cutoff datasets**: On MultiDis and EUDisinfo (published after 2024, unseen during training), the improvement is comparable to that on pre-cutoff datasets, proving that the improvement is not due to memorization effects.
- **Two-stage is superior to single-step**: Separating persuasion analysis and detection tasks allows each stage to be more focused, yielding a 5% difference in F1 score.
- **Social media posts vs. news articles**: The improvement is larger on news articles (due to longer text lengths making persuasion strategies more prominent).

## Highlights & Insights

- **Cognitive-science-driven NLP method design**: Designing computational methods based on the psychological experiment "identifying persuasion improves human judgment"—this interdisciplinary transfer is more insightful than pure empirical prompt-engineering trial and error.
- **Zero-shot and generalizable**: Requires no training data, no fine-tuning, and works consistently across different models and datasets, offering an extremely low barrier to practical deployment.
- **Validity of the two-stage design is supported by experiments**: The single-step approach is significantly weaker than the two-stage approach, demonstrating that task separation is not over-engineering.
- **High-quality construction of new datasets**: MultiDis involves three rounds of annotation with 86.78% agreement in the first two rounds and the participation of fact-checking experts; EUDisinfo is sourced from an authoritative EU database.
- **Knowledge injection is more crucial than prompt engineering**: DMT (which contains strategy definitions and sub-techniques) outperforms Base MT (which only lists names) by 9%.

## Limitations & Future Work

- **Dependency on the quality of the LLM's own persuasion knowledge**: If a model does not sufficiently understand a particular persuasion strategy, the first-stage analysis may be inaccurate.
- **Increased reasoning costs due to the two-stage process**: Requires two LLM calls, doubling token consumption and latency.
- **Evaluation limited to English**: Cross-lingual generalization for disinformation detection remains untested.
- **Potentially incomplete taxonomy of persuasion strategies**: The six categories may not cover all manipulative techniques used in disinformation.
- **Smaller improvements on short social media texts (e.g., tweets)**: Persuasion strategies are less prominent in shorter texts.

## Related Work & Insights

- **vs. Lucas et al. (2023)**: Evaluated across three baseline methods (VaN, Z-CoT, DeF-SpeC), and PCoT consistently improves upon all of them, demonstrating that the enhancement originates from persuasion knowledge rather than prompting tricks.
- **vs. Supervised Methods (BERT Fine-tuning)**: Lucas et al. demonstrated that zero-shot LLMs outperform fine-tuned BERT models across datasets; PCoT widens this gap further.
- **vs. Kojima et al. (2022) Z-CoT**: While Z-CoT guides step-by-step reasoning without injecting domain knowledge, PCoT provides specific analytical dimensions for reasoning through persuasion knowledge.
- **Insight**: Injecting domain knowledge (rather than generic reasoning enhancement) can be a more efficient path to improving zero-shot methods—this approach can be extended to other classification tasks requiring expert judgment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Integrating persuasion knowledge into fake news detection is a fresh perspective, and the cognitive-science-driven method design is forward-looking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely thorough, comprising 5 models × 5 datasets (including 2 unseen datasets) along with statistical significance tests and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, complete formalization of the method, and transparent dataset construction.
- **Value**: ⭐⭐⭐⭐ Highly practical for disinformation detection, and the zero-shot method can be deployed directly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)
- [\[ACL 2025\] DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ACL 2025\] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)

</div>

<!-- RELATED:END -->
