---
title: >-
  [Paper Note] Function Words as Statistical Cues for Language Learning
description: >-
  [ACL 2026][Causal Inference][Function words] The authors utilize Universal Dependencies corpora across 186 languages to demonstrate that three distributional properties—high frequency, syntactic predictability…
tags:
  - "ACL 2026"
  - "Causal Inference"
  - "Function words"
  - "Statistical learning"
  - "Counterfactual language"
  - "BLiMP"
  - "Goldilocks effect"
date: 2026-05-08
content_hash: 46d22184e6196ab8
---

# Function Words as Statistical Cues for Language Learning

**Conference**: ACL 2026  
**arXiv**: [2601.21191](https://arxiv.org/abs/2601.21191)  
**Code**: <https://github.com/picol-georgetown/function_word>  
**Area**: Linguistics / Cognitive / Computational Language Acquisition  
**Keywords**: Function words, Statistical learning, Counterfactual language, BLiMP, Goldilocks effect

## TL;DR
The authors utilize Universal Dependencies corpora across 186 languages to demonstrate that three distributional properties—high frequency, syntactic predictability, and phrase boundary alignment—are cross-linguistically universal. Simultaneously, they construct seven counterfactual variants in English to train GPT-2 small, proving that transformer learners perform best only when all three properties are satisfied. They further identify a Goldilocks effect: function words must be both sufficiently frequent and diverse to be both reliable and discriminative.

## Background & Motivation
**Background**: A long-standing puzzle in language acquisition is how humans (and neural networks) abstract linear input into hierarchical grammatical knowledge. For four decades, experimental psychology and cognitive science have repeatedly suggested that **function words** (closed-class items like determiners, auxiliaries, and prepositions) serve as critical "statistical anchors," summarized into three distributional properties: (i) high token frequency; (ii) reliable binding to specific syntactic structures; and (iii) systematic alignment with phrase boundaries. Representative hypotheses include the Anchoring Hypothesis (Valian & Coulson 1988) and the Marker Hypothesis (Green 1979).

**Limitations of Prior Work**: ① Empirical evidence mostly stems from English or a handful of languages; while linguists have conducted descriptive analyses (e.g., WALS) across thousands of languages, large-scale cross-lingual statistical validation is lacking. ② The causal roles of these properties have mostly been tested separately in simplified artificial languages; no study has systematically compared the contribution of each property or the consequences of disrupting them at the scale of real natural language. ③ It remains unknown whether the impact of these properties on models is confined to the acquisition phase (learning) or persists into the inference phase (processing).

**Key Challenge**: If the distributional features of function words are indeed universal cues for grammar learning, then: (a) they should hold across 186 languages; (b) systematic disruption of these features in natural language should significantly degrade the syntactic generalization of neural models; and (c) models should actually "use" these features during processing. These three levels have not yet been jointly tested.

**Goal**: This paper aims to address these three levels: (RQ1) Whether the three properties are truly universal; (RQ2.1) Whether findings from artificial languages hold at the scale of natural language; (RQ2.2) The individual contribution of each property; and (RQ3) Whether models truly rely on function words during processing.

**Key Insight**: The study utilizes 339 treebanks from Universal Dependencies (UD) v2.17 covering 186 languages for statistical testing. It employs counterfactual language modeling by making controlled modifications to English Wikipedia based on target properties to train GPT-2 small and 5-gram baselines, treating "which property is disrupted" as an independent variable.

**Core Idea**: By treating the three properties of function words as statistical variables that can be independently ablated, and using transformers—general learners with weak inductive biases—as a magnifying glass, the study transforms the question of "which statistical regularities support grammar learning" into a controlled experiment.

## Method

### Overall Architecture
The paper consists of two parts: The first part is a cross-linguistic corpus analysis. It calculates the type/token ratio of function vs. content words, dependency entropy, and phrase boundary alignment for each language in UD, providing three distribution plots to prove the universality of the three properties. The second part involves counterfactual language modeling. Seven variant corpora (NoFunction / FiveFunction / MoreFunction / BigramDep / RandomDep / WithinBoundary / NaturalFunction) are constructed from English Wikipedia. Independent GPT-2 small models (32k vocab, 10 epochs, 3 seeds) are trained on each variant alongside 5-gram baselines. Evaluation is conducted using a synchronized version of the BLiMP minimal pairs set, followed by diagnostic experiments involving attention probing and function-word ablation.

### Key Designs

1.  **Quantitative Metrics for the Three Properties on UD 186**:
    - High Frequency Measure: Defined by type ratio $\frac{|V_c|}{|V|}$ and token frequency ratio $\frac{\sum_{w\in V_c}\text{count}(w)}{\sum_{w\in V}\text{count}(w)}$. If categories were uniformly distributed, these would be equal (falling on the diagonal); observed function words fall far above the diagonal (small vocabulary size / large token share).
    - Syntactic Predictability Measure: Treating the dependency tree as an undirected graph, the conditional entropy of neighboring POS tags for each POS node is calculated as $H(X\mid Y=y)=-\sum_{x\in T} p(x\mid y)\log_2 p(x\mid y)$. The weighted average entropy for the function word set $\mathcal{F}$, $H_F=\sum_{f\in \mathcal{F}}\text{Freq}(f)H(X\mid Y=f) / \sum \text{Freq}(f)$, is significantly lower than $H_C$ across almost all languages.
    - Boundary Alignment Rate: Using the left and right ends of dependency subtrees to approximate constituent boundaries, the proportion of reliable markers (ADP/DET/SCONJ/CCONJ) appearing at subtree boundaries is calculated. The median across 186 languages is **0.95** (lowest is 0.55 in Korean), compared to a median of only 0.58 for content words.
    - **Design Motivation**: Previous typologies were qualitative; this translates the three properties into comparable numbers, allowing "universality" to be definitively established across hundreds of languages.

2.  **Seven Counterfactual English Corpora + Synchronized BLiMP**:
    - Frequency Dimension (Three levels): **STANDARDFUNCTION** (116 natural English function word types); **FIVEFUNCTION** (each syntactic category compressed into 1 type, 5 types total, with extremely high frequency); **MOREFUNCTION** (each function word expanded into 10 pseudo-words using Wuggy, increasing the inventory to 1.2k); **NOFUNCTION** (all deleted).
    - Structural Dimension (Three levels): **PHRASEDEPENDENCY** (natural baseline); **BIGRAMDEP** (function words determined by the succeeding word); **RANDOMDEP** (identity shuffled while maintaining positions).
    - Boundary Dimension (Two levels): **ATBOUNDARY** (natural baseline); **WITHINBOUNDARY** (function words moved from phrase boundaries to positions adjacent to their syntactic heads, disrupting 55% of function word positions and covering 99% of sentences).
    - Evaluation: BLiMP is rewritten following the same rules. Categories where keywords are function words (e.g., Det-N agreement) are removed. Minimal pairs that become identical after transformation are removed. Intersection filtering is applied—if a pair is removed in one condition, it is removed in all—to ensure models are evaluated on identical minimal pair sets.
    - **Design Motivation**: This allows "corpus modification" to function as a clean factorial experiment where frequency, structure, and boundary properties can be ablated individually or jointly. Synchronizing BLiMP solves the difficulty of fairly scoring models trained on variant corpora.

3.  **Dual-Track Diagnostics: Attention Probing + Function-Word Ablation**:
    - Probing: Following Aoyama & Wilcox (2025), for each head $(h,l)$, a function $f_{h,l}(x_i)=\arg\max_{j\neq i} a_{ij}^{(h,l)}$ is defined (where the maximum attention points). The frequency $S_F(h,l)$ of attention pointing to function words is calculated to identify "function heads" contributing most to each BLiMP subcategory.
    - Ablation: Two types—function-word **masking** (blocking bidirectional attention to function word tokens during evaluation, treating them as content-free placeholders) and function-word **deletion** (evaluating directly on the NoFunction-modified BLiMP).
    - **Design Motivation**: BLiMP scores only prove that corpus modification affects learning. Probing reveals whether "specialized" function-word heads emerge when properties are intact. Ablation determines whether the model actually **relies** on function words during inference. This links "learning" to "processing" mechanisms.

### Loss & Training
GPT-2 small is trained for each variant (BPE vocab 32,768, context 128, batch 128, 10 epochs, lr 5e-4, linear warmup 10%, AdamW, weight decay 0.1, Tesla V100) using 3 random seeds (42/53/67). Independent tokenizers are trained for each variant to accommodate vocabulary changes. The 5-gram baseline uses KenLM with Kneser-Ney smoothing. Evaluation uses BLiMP accuracy with linear mixed-effects significance testing (`acc ~ condition + (1|category:phenomenon) + (1|seed)`) rather than perplexity, as variants change entropy.

## Key Experimental Results

### Main Results

| Condition | Transformer Overall | $\Delta$ vs Natural | 5-gram Overall | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| NATURALFUNCTION | **72.7** | — | 55.5 | Natural English is optimal |
| NOFUNCTION | 60.7 | **-12.0** | 54.1 | Maximum loss when deleting function words |
| FIVEFUNCTION | 70.9 | -1.8 (p=0.08) | 55.4 | Marginally significant |
| MOREFUNCTION | 69.7 | -3.0 | 52.8 | Excessive diversity hurts |
| BIGRAMDEP | 67.4 | -5.3 | **56.1** | 5-gram outperforms natural condition here |
| RANDOMDEP | 67.0 | -5.7 | 53.4 | Significant drop after structural shuffle |
| WITHINBOUNDARY | 69.7 | -3.0 | 54.5 | Small drop from boundary disruption |

Key Observations: All disruption conditions show significant negative effects ($p<0.05$, except FIVEFUNCTION at $p=0.08$). **Disrupting structural association (BigramDep/RandomDep) is more harmful than disrupting boundary alignment (WithinBoundary).**

### Ablation Study

| Condition | Function Head Mean Entropy (bits) | std |
| :--- | :--- | :--- |
| NATURALFUNCTION | **2.74** | 0.861 |
| FIVEFUNCTION | 2.87 | 0.924 |
| MOREFUNCTION | 3.53 | 0.263 |
| BIGRAMDEP | 3.60 | 0.257 |
| RANDOMDEP | 3.64 | 0.156 |
| WITHINBOUNDARY | 4.02 | 0.463 |

In the Natural condition, function-head attention is highly concentrated in a few heads in Layers 3-4 (lowest entropy). Disrupting any property scatters these "specialized heads." In function word deletion experiments, the Natural condition shows the largest drop in BLiMP (showing maximum reliance), while BigramDep/RandomDep show the smallest drops (showing lack of reliance).

### Key Findings
- **Goldilocks Effect**: Compressing 116 function words into 5 high-frequency types (FiveFunction) or expanding them into 1.2k pseudo-types (MoreFunction) performs worse than the natural baseline. High frequency must be paired with sufficient diversity; too much concentration loses structural discriminativity, while too much dispersion loses high-frequency reliability.
- **Structure Association > Boundary Alignment > Frequency**: Among the three properties, the cost of disrupting structural association (-5.3/-5.7) is significantly higher than disrupting boundary alignment (-3.0). This suggests that labeling information is more central to function words than segmentation information; segmentation can be partially recovered from other statistical sources like transition probabilities.
- **Universality across 186 Languages**: Function words across all studied languages exhibit "small vocabulary + large token share + low dependency entropy + high boundary alignment," extending English-based psycholinguistic generalizations to true cross-linguistic validation.
- **The 5-gram Counter-example**: 5-gram models perform better on BigramDep than on Natural (56.1 vs 55.5), while transformers clearly win on Natural. This serves as inverse evidence that transformers capture structural associations rather than just local linear statistics.

## Highlights & Insights
- **Translating Linguistic Hypotheses into Ablative Training Experiments**: Psycholinguistics has long relied on human subjects and artificial languages to establish causal chains from "distributional properties" to "learning difficulty." This paper applies counterfactual language modeling and transformer learners to validate these chains at the scale of real natural language, a methodology transferable to questions about whether any distributional feature (morphology, tone, hierarchy) drives acquisition.
- **Dual-Track Evidence (Probing + Ablation)**: The study proves both that function word properties shape the model during learning and that the model relies on these properties during inference, creating a closed loop between corpus-level causality and model-level mechanisms.
- **Cross-lingual + Computational Synergy**: The UD descriptive results provide typological weight, while counterfactual modeling provides mechanistic weight. This two-step narrative sets a paradigm for claims about language universals.
- **The Goldilocks Effect has Broad Relevance**: Viewing "high frequency" and "diversity" as a binary trade-off is a phenomenon that may recur in vocabulary design, tokenizer selection, or prompt word selection—where too few high-frequency anchors may lose discriminative power.

## Limitations & Future Work
- Counterfactual modeling was limited to English due to the requirement for precise dependency parsing (Stanza) and grammar benchmarks like BLiMP; causal experiments in other languages remain for future work.
- Function words were considered only as word-level closed classes, omitting "functional suffixes" in morphological layers (e.g., Turkish agglutination); UD's lack of morphology labels limited this scope.
- Training utilized Wikipedia rather than Child-Directed Speech (CDS), missing prosodic cues (stress/rhythm). The authors acknowledge that prosody, a classic psycholinguistic dimension, was not addressed.
- The random expansion of pseudo-words in MoreFunction cannot guarantee natural collocational patterns, potentially introducing noise.
- BLiMP has inherent limitations; 5-gram models can achieve above-chance accuracy, and high BLiMP scores do not strictly equate to human-like grammatical knowledge.

## Related Work & Insights
- **vs Valian & Coulson 1988 / Green 1979**: Classic psycholinguistic experiments conducted on a small scale with human subjects; this paper scales the paradigm to natural language and neural learners, consistent with and refining those conclusions (Structure > Frequency).
- **vs Kallini et al. 2024 (Mission: Impossible Language Models)**: Also uses counterfactual corpus modification to test inductive biases, but this study focuses specifically on the three distinct properties of function words.
- **vs Mintz 2003 (Frequent Frames)**: While Mintz proposed frequent frames support grammatical category learning, this work decomposes the frame phenomenon into three levels (frequency/structure/boundary) and re-evaluates it using transformers.
- **vs BLiMP (Warstadt et al. 2020)**: By synchronizing BLiMP modifications and implementing "intersection filtering," the authors extend BLiMP from a static English benchmark to a unified testbed for counterfactual language research.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of counterfactual LM, 186-language statistics, and dual diagnostics is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 main conditions, 3 seeds, 5-gram baseline, interaction conditions, probing, and dual ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative across three properties; discussions and limitations are transparent.
- Value: ⭐⭐⭐⭐ Bridging cognitive science and computational linguistics to provide a definitive answer for the "Why Zipfian + Function Words" problem in the machine learning era.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Counterfactual Strategic Reasoning in Large Language Models](evaluating_counterfactual_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[AAAI 2026\] Skill Path: Unveiling Language Skills from Circuit Graphs](../../AAAI2026/causal_inference/skill_path_unveiling_language_skills_from_circuit_graphs.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[ICML 2026\] The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](../../ICML2026/causal_inference/the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)

</div>

<!-- RELATED:END -->
