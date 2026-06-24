---
title: >-
  [Paper Note] Concreteness Versus Abstractness: A Selectivity Analysis in LLMs
description: >-
  [ACL 2025][LLM (Other)][Concreteness] This paper investigates the difference in how Large Language Models (LLMs) process concrete concepts (e.g., "apple") and abstract concepts (e.g., "freedom"). Through selectivity analysis, the authors discover that subpopulations of neurons in LLMs selectively respond to concreteness or abstractness, revealing an intriguing correspondence between the semantic representations of LLMs and the "concreteness effect" in human cognitive theories…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Concreteness"
  - "Abstractness"
  - "Semantic Selectivity"
  - "Neuronal Analysis"
  - "Conceptual Representation"
date: 2026-05-08
content_hash: b858b4ac7595d249
---

# Concreteness Versus Abstractness: A Selectivity Analysis in LLMs

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Concreteness, Abstractness, Semantic Selectivity, Neuronal Analysis, Conceptual Representation

## TL;DR
This paper investigates the difference in how Large Language Models (LLMs) process concrete concepts (e.g., "apple") and abstract concepts (e.g., "freedom"). Through selectivity analysis, the authors discover that subpopulations of neurons in LLMs selectively respond to concreteness or abstractness, revealing an intriguing correspondence between the semantic representations of LLMs and the "concreteness effect" in human cognitive theories.

## Background & Motivation

**Background**: In cognitive science, the "concreteness effect" is a classic finding where humans process concrete words (e.g., "table", "cat") faster and more accurately than abstract words (e.g., "justice", "concept"). This phenomenon is considered to be associated with dual-coding theory: concrete concepts possess both verbal and perceptual (visual, tactile, etc.) codes, whereas abstract concepts rely primarily on verbal coding. In the field of NLP, the concreteness/abstractness of a word is an important semantic dimension, affecting various tasks such as word similarity, metaphor comprehension, and sentiment analysis.

**Limitations of Prior Work**: (1) Whether LLMs also exhibit a concreteness effect similar to humans, and what its underlying neural basis is, lack systematic research; (2) Although existing work analyzes the internal representations of LLMs (e.g., probing experiments), it mostly focuses on syntactic information (parts of speech, dependency relations), while the analysis of semantic dimensions, particularly the concreteness/abstractness dimension, is scarce; (3) Understanding how LLMs represent and process abstract concepts is crucial for improving their performance on tasks requiring abstract reasoning, such as metaphor comprehension and commonsense reasoning.

**Key Challenge**: LLMs are trained solely on text without perceptual experiences—according to dual-coding theory, they should behave differently from humans regarding the concreteness effect. However, LLMs perform well in many tasks requiring concrete knowledge, implying that they might have developed some form of "quasi-concrete" representation through pure linguistic statistics.

**Goal**: (1) Systematically measure concreteness/abstractness information inside LLM internal representations; (2) Locate neurons that selectively respond to concrete/abstract concepts; (3) Analyze the distribution patterns of these selective neurons across different layers and different models.

**Key Insight**: Borrowing methodologies from neuroscience for analyzing brain region selectivity (such as the fusiform face area, FFA)—using a selectivity index to quantify the preference of each neuron for concrete versus abstract stimuli.

**Core Idea**: Through selectivity analysis, "concreteness-selective neurons" and "abstractness-selective neurons" are discovered in LLMs. Their distributions across different layers of the model are non-uniform (concrete-leaning in shallow layers, abstract-leaning in deep layers), which aligns with the expected cognitive processing hierarchy from perception to conceptual abstraction.

## Method

### Overall Architecture
Experimental pipeline: (1) Construct a large-scale concrete/abstract word stimulus set (based on human concreteness rating databases); (2) Embed the stimulus words into natural language sentences and input them into LLMs; (3) Record the activation values of each neuron in each hidden layer; (4) Calculate the concreteness/abstractness selectivity index for each neuron; (5) Analyze the hierarchical distribution and network roles of these selective neurons.

### Key Designs

1. **Controlled Stimulus Set Construction**:

    - Function: Construct a stimulus set of word pairs that are opposite in the concreteness/abstractness dimension but matched along other dimensions.
    - Mechanism: Select 500 extremely concrete words (rating > 4.5) and 500 extremely abstract words (rating < 2.0) from the Brysbaert et al. concreteness database (containing 1-5 concreteness ratings for 40,000+ English words). Match the two groups on word frequency, word length, and part-of-speech distributions to eliminate confounding variables. Embed each word into 5 different natural sentence contexts (e.g., "The [word] is important"), keeping the position of the target word fixed to control for position effects. This yields 5,000 concrete and 5,000 abstract stimuli in total.
    - Design Motivation: Without controlling for confounding variables like word frequency and length, differences in selectivity might stem from frequency effects rather than concreteness itself.

2. **Neuron Selectivity Index Calculation**:

    - Function: Quantify the preference of each neuron for concrete versus abstract concepts.
    - Mechanism: For each neuron $j$ in the model, calculate the average activation on concrete words $\bar{a}_j^{con}$ and abstract words $\bar{a}_j^{abs}$. The selectivity index is defined as $SI_j = (\bar{a}_j^{con} - \bar{a}_j^{abs}) / (\bar{a}_j^{con} + \bar{a}_j^{abs})$, ranging within [-1, 1]. $SI > 0$ indicates concreteness selectivity, and $SI < 0$ indicates abstractness selectivity. A permutation test (10,000 random label shuffles) is used to determine the significance threshold, retaining only neurons that pass the multiple comparison correction (FDR < 0.05).
    - Design Motivation: The selectivity index is a standard tool in neuroscience, directly quantifying the preference strength and direction of the signal, and statistical testing ensures that the results are not due to random fluctuations.

3. **Hierarchical Distribution Analysis**:

    - Function: Reveal the distribution patterns of selective neurons across different layers of the model.
    - Mechanism: Count the proportion of concreteness-selective and abstractness-selective neurons in each layer to plot hierarchical distribution curves. Calculate the "net concreteness preference" of layer $l$: $NCP_l = (N_l^{con} - N_l^{abs}) / (N_l^{con} + N_l^{abs})$. Also, analyze whether these selective neurons are selective to other semantic dimensions (e.g., sentiment polarity, frequency) to test whether concreteness selectivity is independent or entangled with other dimensions.
    - Design Motivation: If selective neurons are uniformly distributed, it suggests concreteness is globally encoded; if non-uniforrmly distributed, it indicates that the model values concreteness differently across various stage processing.

### Loss & Training
This paper is a purely analytical work and does not involve model training. The models used are all publicly released pretrained models.

## Key Experimental Results

### Main Results

| Model | Total Neurons | Concrete Selectivity (%) | Abstract Selectivity (%) | Net Concrete Preference Peak | Net Abstract Preference Peak |
|------|----------|-------------|-------------|-----------|-----------|
| GPT-2 | 36,864 | 4.2% | 3.8% | Layer 2-4 | Layer 10-12 |
| Llama-2-7B | 131,072 | 3.7% | 4.1% | Layer 3-8 | Layer 24-32 |
| Llama-3-8B | 131,072 | 3.5% | 4.3% | Layer 4-10 | Layer 26-32 |
| Mistral-7B | 131,072 | 3.6% | 4.0% | Layer 3-9 | Layer 25-32 |

### Ablation Study (Llama-2-7B Selectivity Neuron Intervention)

| Intervention Strategy | Concrete Word PPL Change | Abstract Word PPL Change | Description |
|---------|-------------|-------------|------|
| Ablate Concrete-Selective Neurons | +12.3% | +2.1% | Concrete words affected significantly more than abstract words |
| Ablate Abstract-Selective Neurons | +1.8% | +9.7% | Abstract words affected significantly more than concrete words |
| Ablate Random Equivalent Neurons | +3.5% | +3.2% | No differential impact |
| Ablate Both Selective Neurons | +14.1% | +11.5% | Superposition of dual impacts |

### Key Findings
- Approximately 3.5-4.3% of neurons exhibit significant concreteness/abstractness selectivity, a proportion highly consistent across different models.
- The net preference in shallow layers (first 1/3 of layers) is toward concreteness, whereas the net preference in deep layers (last 1/3 of layers) is toward abstractness, with no obvious preference in intermediate layers. This matches the "surface-to-abstract" cognitive processing hierarchy.
- Intervention experiments confirm the functionality of selectivity—ablating concreteness-selective neurons selectively impairs the processing of concrete words, and vice versa.
- Concreteness selectivity is dissociable from the word frequency effect—selectivity remains significant even after controlling for word frequency.

## Highlights & Insights
- Systematically applies the selectivity analysis method from neuroscience to LLM research, methodologically bridging the two fields. This opens up a new analytical dimension for understanding LLM internal representations.
- The hierarchical "shallow-concrete, deep-abstract" pattern is highly intriguing, indicating that LLMs might spontaneously learn a scaling ladder from perception to concepts similar to human cognition.
- Intervention experiments elevate passive observation to causal inference—not only discovering selective neurons but also proving their functional roles.

## Limitations & Future Work
- Selectivity analysis assumes linearly separable single-neuron encoding, potentially missing distributed non-linear representations.
- The analysis only covers English; concreteness/abstractness across different languages may have cultural variances.
- The modulating effect of context on selectivity is not analyzed—the concreteness of the same word might vary in different contexts (e.g., "bank" under different scenarios).
- Whether the hierarchical distribution pattern has a systematic relationship with the model architecture (number of layers, width) remains to be explored.

## Related Work & Insights
- **vs Probing Classifiers**: Traditional probing classifiers detect whether a specific layer "contains" certain information, whereas selectivity analysis goes a step further to isolate individual neurons.
- **vs Knowledge Neurons**: While knowledge neuron research locates neurons storing factual knowledge, this work locates semantic category-selective neurons from a different dimension (concreteness/abstractness).
- **vs Embodied Cognition**: Embodied cognition theory posits that abstract concepts rely on the metaphorical extension of concrete experiences. This work finds a similar hierarchical organization in LLMs, despite the latter lacking embodied experiences.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The research question is novel, the methodology is interdisciplinary, and the findings have theoretical importance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid multi-model verification, intervention experiments, and rigorous controlled variable design.
- Writing Quality: ⭐⭐⭐⭐ Excellent interdisciplinary background introduction, and clear results presentation.
- Value: ⭐⭐⭐⭐ Inspiring for both understanding LLM semantic representations and cognitive modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[ACL 2025\] Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases](can_llms_simulate_l2-english_dialogue_an_information-theoretic_analysis_of_l1-de.md)
- [\[ACL 2025\] Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)
- [\[ACL 2025\] Enabling LLM Knowledge Analysis via Extensive Materialization](enabling_llm_knowledge_analysis_via_extensive_materialization.md)
- [\[ACL 2025\] Towards Robust ESG Analysis Against Greenwashing Risks: A3CG](a3cg_esg_greenwashing.md)

</div>

<!-- RELATED:END -->
