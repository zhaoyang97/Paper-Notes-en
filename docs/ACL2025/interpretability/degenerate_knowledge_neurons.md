---
title: >-
  [Paper Note] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models
description: >-
  [ACL 2025][Interpretability][Knowledge Neurons] This paper redefines degenerate knowledge neurons (DKNs) in LLMs from both structural and functional perspectives, proposes a neural topological clustering (NTC) method to identify DKNs of arbitrary sizes and structures, and reveals the intrinsic relationships of DKNs with LLM robustness, evolvability, and complexity through 34 experiments.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Knowledge Neurons"
  - "Degeneracy"
  - "Factual Knowledge Storage"
  - "Robustness"
  - "Evolvability"
date: 2026-05-08
content_hash: 97e913332483053a
---

# Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2402.13731](https://arxiv.org/abs/2402.13731)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Knowledge Neurons, Degeneracy, Factual Knowledge Storage, Robustness, Evolvability

## TL;DR
This paper redefines degenerate knowledge neurons (DKNs) in LLMs from both structural and functional perspectives, proposes a neural topological clustering (NTC) method to identify DKNs of arbitrary sizes and structures, and reveals the intrinsic relationships of DKNs with LLM robustness, evolvability, and complexity through 34 experiments.

## Background & Motivation
1. **Background**: LLMs store a vast amount of factual knowledge in MLP weights. Knowledge neurons (KNs) act as the fundamental units of knowledge storage. Some KN pairs exhibit degeneracy, where different subsets of KNs can independently express the same fact.
2. **Limitations of Prior Work**: Previous definitions of DKNs suffered from two limitations: (1) size constraint—each DKN element contained only two KNs; (2) neglect of connections—only the neurons themselves were considered, ignoring the connection weights between them.
3. **Key Challenge**: Factual knowledge may require more than two neurons to co-express, and knowledge expression relies on interactions among multiple neurons, necessitating the consideration of connection structures.
4. **Goal**: To comprehensively define DKNs, propose an accurate DKN acquisition method, and explore the relationships between DKNs and three core attributes of LLMs.
5. **Key Insight**: Drawing inspiration from the concept of degeneracy in cognitive science—components with different structures but equivalent functions—to study their contribution to system robustness and evolvability.
6. **Core Idea**: Base Degenerate Components (BDCs) + Neural Topological Clustering + the correlation between degeneracy and three major attributes.

## Method

### Overall Architecture
Obtains KNs using the AMIG method $\rightarrow$ Calculates neuron distance based on connection weights $\rightarrow$ Performs Neural Topological Clustering (NTC): observes cluster formation as distance threshold $R$ increases $\rightarrow$ Identifies stable clusters as Base Degenerate Components (BDCs) $\rightarrow$ Filters to obtain DKNs $\rightarrow$ Verifies through experimental evaluation of the three major attributes.

### Key Designs

1. **Complete Definition of DKN**:

    - Function: Defines degenerate knowledge neurons from both functional and structural perspectives.
    - Mechanism: Functional definition—a DKN contains multiple BDCs, where each BDC can independently express the same fact ($Prob(\mathcal{D}) \approx Prob(\mathcal{B}_i)$), and the fact cannot be expressed after suppressing all BDCs ($Prob(\emptyset) \ll Prob(\mathcal{B}_i)$). Structural definition—defines neuron distance based on connection weights, analyzing the connection tightness and differences in neuron counts within BDCs.
    - Design Motivation: Overcomes previous limitations of only defining two-KN pairs, allowing degenerate components of arbitrary sizes and structures.

2. **Neural Topological Clustering (NTC) Method**:

    - Function: Accurately identifies DKNs of arbitrary sizes and structures.
    - Mechanism: Starting from distance threshold $R=0$, it incrementally increases $R$ to observe the clustering behavior of KNs. As $R$ increases, closer KNs merge first. Clusters that remain stable over a wide range of $R$ (e.g., from $r_2$ to $r_3$) are identified as BDCs, as stability implies robust knowledge expression capability. BDCs are then confirmed through functional filtering (verifying independent expression capability).
    - Design Motivation: Inspired by topological data analysis, utilizing the concept of persistence diagrams to find structures that exist stably under parameter changes.

3. **Exploration of Three Major Attributes**:

    - Function: Reveals the relationship between DKNs and core LLM attributes.
    - Mechanism: (1) Robustness—evaluates changes in predictions by enhancing/suppressing DKNs under input perturbations, finding that DKNs help LLMs cope with disturbances; DKNs are also used to detect hallucinated facts. (2) Evolvability—shows that parameter change regions highly overlap with DKNs after fine-tuning; freezing all MLP neurons except DKNs still allows efficient learning of new knowledge without forgetting old knowledge. (3) Complexity—cross-scale comparison of different LLMs reveals that degeneracy positively correlates with complexity.
    - Design Motivation: Leverages degeneracy theory from cognitive science to systematically validate its analogy within neural networks.

### Loss & Training
DKN extraction does not require training. Experiments use GPT-2 and LLaMA2-7B, analyzing on the TempLama dataset. 34 experiments cover 6 settings.

## Key Experimental Results

### Main Results

| Attribute | Experiment | Key Findings |
|------|------|---------|
| Robustness | DKN Enhancement/Suppression | DKN enhancement increases prediction probability for perturbed inputs |
| Robustness | Fact Detection | DKNs can effectively detect false facts |
| Evolvability | Fine-Tuning Parameter Analysis | Overlap between parameter changes and DKNs is >80% |
| Evolvability | DKN-Only Fine-Tuning | Updating only DKNs enables learning new knowledge without forgetting old |
| Complexity | Cross-Scale Comparison | Larger models exhibit stronger degeneracy |

### Ablation Study

| Configuration | Performance | Explanation |
|------|------|------|
| NTC (Full) | Optimal | Includes clustering + filtering |
| Pairwise Clustering Only (Prior Method) | Suboptimal | Pairwise constraints are insufficient |
| Random Neuron Groups | Poor | Proves that DKNs are non-random |

### Key Findings
- DKNs are not merely a redundancy mechanism for knowledge storage, but a key guarantee for LLM robustness and evolvability.
- Fine-tuning only DKNs allows efficient learning of new knowledge, providing a new perspective for parameter-efficient fine-tuning (PEFT).
- Differences in degeneracy across models of various scales partly explain why larger models are more robust.

### Degeneracy and Model Scale

| Model | Parameter Size | Avg No. of DKNs | Robustness Metric |
|------|--------|----------|----------|
| GPT-2 Small | 117M | 3.2 | 0.65 |
| GPT-2 Medium | 345M | 4.8 | 0.72 |
| GPT-2 Large | 774M | 6.1 | 0.78 |
| LLaMA2-7B | 7B | 8.5 | 0.85 |


## Highlights & Insights
- **Bridge between Cognitive Science and AI**: Systematically introduces the biological concept of degeneracy into LLM research, offering a novel perspective on understanding LLMs' internal mechanisms.
- **Practical Potential of DKN Fine-Tuning**: The finding that updating only DKNs achieves learning new things without forgetting old ones has direct implications for continual learning and knowledge editing.

## Limitations & Future Work
- DKN acquisition relies on the AMIG method, which incurs high computational costs and may be impractical for ultra-large-scale models.
- The findings are currently validated only on factual knowledge, without extension to other knowledge types (e.g., procedural knowledge, commonsense reasoning).
- The selection of distance thresholds in NTC requires empirical tuning; there is a lack of automated methods for determining optimal thresholds.
- Experiments are validated only on GPT-2 and LLaMA2-7B; larger-scale models may exhibit different degeneracy patterns.
- The DKN-only fine-tuning strategy can learn new tasks without forgetting old ones, but its learning capacity might be limited by the scale of DKNs.
- The definition of neuron distance is based on connection weights, which might not fully capture functional-level similarity.
- The TempLama dataset primarily consists of time-sensitive facts; applicability to static facts remains unverified.

## Related Work & Insights
- **vs Knowledge Neurons**: Original KN research only focuses on which neurons store knowledge; DKNs further reveal the redundant structure of knowledge storage.
- **vs LoRA/PEFT**: LoRA randomly selects low-rank spaces, whereas DKNs pinpoint specific knowledge-related parameters, which could be more efficient.


### Supplementary Discussion
- The core novelty of this method lies in transforming the problem of analysis from a single dimension to multiple dimensions, providing a more comprehensive perspective.
- The experimental design covers multiple scenarios and baseline comparisons, with statistically significant results.
- The modular design of the method makes it easy to extend to related tasks and new datasets.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the DKN concept and NTC method are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Exceptionally comprehensive with 34 experiments across 6 settings.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, though some formulas could be simplified.
- Value: ⭐⭐⭐⭐ Offers important insights into understanding LLM knowledge storage and parameter-efficient fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Knowledge Microscope: Features as Better Analytical Lenses than Neurons](the_knowledge_microscope_features_as_better_analytical_lenses_than_neurons.md)
- [\[ICML 2025\] Taming Knowledge Conflicts in Language Models](../../ICML2025/interpretability/taming_knowledge_conflicts_in_language_models.md)
- [\[ICLR 2026\] A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models](../../ICLR2026/interpretability/a_comprehensive_information-decomposition_analysis_of_large_vision-language_mode.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)

</div>

<!-- RELATED:END -->
