---
title: >-
  [Paper Note] Interpretable Coreference Resolution Evaluation Using Explicit Semantics
description: >-
  [ACL 2026][Interpretability][Coreference Resolution] This paper utilizes Concept and Named Entity Recognition (CNER) to overlay 29 fine-grained semantic labels onto coreference resolution outputs via a "mention + cluster…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Coreference Resolution"
  - "CNER"
  - "Semantic Evaluation"
  - "Typed F1"
  - "Targeted Data Augmentation"
date: 2026-05-08
content_hash: c3aceacdf70fa895
---

# Interpretable Coreference Resolution Evaluation Using Explicit Semantics

**Conference**: ACL 2026  
**arXiv**: [2605.10627](https://arxiv.org/abs/2605.10627)  
**Code**: https://github.com/SapienzaNLP/cner-coref (Available)  
**Area**: Interpretability / Coreference Resolution / Evaluation / Data Augmentation  
**Keywords**: Coreference Resolution, CNER, Semantic Evaluation, Typed F1, Targeted Data Augmentation

## TL;DR
This paper utilizes Concept and Named Entity Recognition (CNER) to overlay 29 fine-grained semantic labels onto coreference resolution outputs via a "mention + cluster-level majority voting" mechanism. This yields diagnostic metrics—Mention F1 and Link F1—stratified by semantic category, allowing for a clear view of where models systematically fail. These diagnostics guided targeted data augmentation using only 3 synthetic documents, which improved the CoNLL-F1 of LitBank-trained models on OntoNotes/PreCo by +2.5/+2.8 and Mention F1 by approximately +9.5.

## Background & Motivation

**Background**: Since the 1990s, the mainstream evaluation of coreference resolution has relied on the MUC, $B^3$, and CEAF$_{\phi 4}$ metrics and their average, CoNLL-F1. These require strict mention boundary identity and pairwise link matching. On the modeling side, encoder-decoder joint models like Maverick have achieved SOTA performance on OntoNotes.

**Limitations of Prior Work**: (i) A single aggregate score masks failure modes at the category level—a model might excel at person chains but fail completely on event and object chains, which is indistinguishable via CoNLL-F1; (ii) Cross-domain evaluation scores drop, but it is unclear whether this is due to boundary differences, annotation specification differences, or genuine linguistic capability deficits; (iii) Existing semantic evaluations (Agarwal et al. 2019) use standard NER with 4 categories (PER/ORG/LOC/MISC), which only covers about 50% of mentions and is too coarse.

**Key Challenge**: A significant portion of mentions in coreference resolution are nominal concepts (common nouns like *president*, *city*, *whale*). Traditional NER cannot annotate these and only provides labels for named entities. This leads to insufficient coverage and granularity for "semantic coreference evaluation," making it impossible to locate the actual problems.

**Goal**: (1) Assign dense, fine-grained semantic labels to coreference clusters; (2) Calculate typed F1 metrics stratified by semantic category; (3) Use diagnostic results to guide low-cost data augmentation to verify feasibility.

**Key Insight**: By using CNER (Martinelli et al. 2024), which labels both named entities and nominal concepts across 29 unified categories, coverage is increased from NER's 22-52% to ~90%. A "cluster-level majority voting" mechanism is then used to back-propagate labels to mentions that cannot be directly labeled (such as pronouns) through their associated clusters.

**Core Idea**: While keeping the coreference model unchanged, the CNER semantic layer is overlaid onto the coreference output. Mentions are aligned with CNER spans based on token-level Jaccard overlap, and labels are propagated using majority voting at the cluster level. This transforms coreference evaluation into a "diagnostic interface stratified by semantic category."

## Method

### Overall Architecture

Input: A set of mentions $\mathcal{M} = \{m_1, ..., m_n\}$ and clusters $\mathcal{G}$ predicted by a coreference model from document $D$, and a set of annotated spans $\mathcal{C} = \{c_1, ..., c_k\}$ predicted by CNER, where each $c_j$ has a label $L(c_j) \in \mathcal{T}$ ($\mathcal{T}$ includes 29 classes such as PERSON / LOCATION / EVENT / RELATION / SUPERNATURAL / PLANT / DISEASE, etc.). Two intermediate steps follow: (1) Mention Assignment: uses Jaccard overlap to align mention $m_i$ with the CNER span $\hat{c}_j$ having the highest overlap; a label is assigned if overlap $> \tau=0.5$. (2) Category Propagation: uses majority voting within each cluster $G$ to determine $S(G) = \arg\max_{t \in \mathcal{T}} |\{m_G \in G : L(m_G) = t\}|$, then propagates $S(G)$ to all unlabeled mentions (including pronouns). Output: Each mention is tagged with a CNER label, allowing for the calculation of stratified typed Mention F1 / Link F1.

### Key Designs

1.  **Two-step Labeling + Cluster-level Majority Voting**:
    - **Function**: Upgrades semantic labeling from "only nominal/named spans" to ensuring the "entire cluster has a consistent label," incorporating mentions like pronouns that cannot be directly labeled into the diagnosis.
    - **Mechanism**: Overlap function $\Omega(m_i, c_j) = |\text{span}(m_i) \cap \text{span}(c_j)| / |\text{span}(m_i) \cup \text{span}(c_j)|$. For each mention, $\hat{c}_j = \arg\max \Omega$ is selected; if overlap $< 0.5$, it remains empty until cluster-level majority voting. Ties in majority voting are broken by the label with the highest average $\Omega$. Direct labeling covers 37.5-71.4%, rising to ~90% after propagation, with the remainder being mostly pronoun-only clusters.
    - **Design Motivation**: Traditional NER-based evaluation only labels 22.8% of mentions in PreCo, preventing per-class analysis. Cluster-level propagation utilizes the hard constraint that "mentions in the same cluster must be semantically equivalent," resolving the "density problem" with low computational cost.

2.  **Typed Mention F1 + Link F1 Decoupled Evaluation**:
    - **Function**: Decomposes coreference evaluation into two independent diagnostic dimensions: "correct identification" and "correct linking."
    - **Mechanism**: Mention F1 calculates the precision and recall of mention extraction for a specific category $t$, independent of clustering. Link F1 evaluates whether mention pairs $(m_1^G, m_2^G)$ within the same gold cluster are correctly linked, specifically characterizing the quality of the clustering structure while decoupled from mention detection. Both metrics are reported by semantic category, identifying details such as "the model links PER well but mention detection for EVENT fails."
    - **Design Motivation**: CoNLL-F1 mixes error sources from boundaries, links, and clustering; a 10-point drop doesn't pinpoint what to change. Decoupling into mention/link + per-class metrics makes the "failure $\rightarrow$ improvement" causal chain traceable.

3.  **Diagnostic-Driven Targeted Data Augmentation**:
    - **Function**: Converts diagnostic results (e.g., "the LitBank model fails on PLANT/EVENT/MEDIA") into executable augmentation recipes.
    - **Mechanism**: GPT-5.1 is used to generate 3 fictional narratives of ~2000 words in the style of LitBank, each containing mentions of CNER classes identified as weak. These are manually annotated under two specifications: Restricted (only the 6 LitBank classes) and Unrestricted (covering all nominal/pronoun mentions). These are added to the LitBank training set to train augmented vs. augmented-NR models, comparing improvements on OntoNotes/PreCo. This process validates the "diagnosis $\rightarrow$ modification $\rightarrow$ effect" loop at almost zero cost.
    - **Design Motivation**: Traditional evaluation identifies problems but offers no interventions. This study uses a minimal falsifiable scale (3 documents) to prove that typed F1 diagnostics can guide data strategies, making "interpretable evaluation" truly "actionable evaluation."

### Loss & Training

Ours does not modify the coreference models, using three official checkpoints of Maverick (mes multi-expert version) trained on OntoNotes / LitBank / PreCo respectively. The CNER semantic layer uses official CNER checkpoints for direct inference. For the data augmentation part, the LitBank-augmented models are fine-tuned on the original LitBank training set plus 3 synthetic documents following the original Maverick training process. Mention/Link F1 are calculated using the standard P/R harmonic mean.

## Key Experimental Results

### Main Results (Macro Mention/Link F1 of three Maverick variants across 3 datasets)

| Model | OntoNotes M-F1 | LitBank M-F1 | PreCo M-F1 | OntoNotes L-F1 | LitBank L-F1 | PreCo L-F1 |
|-------|----------------|--------------|------------|----------------|--------------|------------|
| maverick-mes-ontonotes | **0.85** | 0.48 | 0.40 | **0.77** | 0.53 | 0.57 |
| maverick-mes-litbank | 0.40 | **0.78** | 0.31 | 0.43 | 0.53 | 0.47 |
| maverick-mes-preco | 0.53 | 0.35 | **0.93** | 0.47 | 0.46 | **0.82** |

All in-domain models are strong, but the LitBank-trained model shows much lower macro Mention F1 cross-domain compared to OntoNotes/PreCo models. Per-class Mention F1 (Figure 5) reveals that the LitBank model ranks lowest for almost all non-PER categories on PreCo/OntoNotes. Link F1 calculated with gold mentions remains worst for LitBank training, confirming the issue is a person-centric bias in clustering logic rather than just boundary differences.

CNER coverage compared to NER (post-annotation + post-propagation): OntoNotes 90% vs 52.8%, LitBank 90% vs 29.6%, PreCo 90% vs 22.8%. CNER effectively resolves the "density" issue. Manual verification (30% of LitBank test set): CNER cluster-level label precision 90% / recall 87% / F1 88%, proving the propagation chain is reliable.

### Ablation Study / Targeted Data Augmentation (LitBank training + 3 synthetic docs, cross-domain)

| Model | PreCo CoNLL-F1 | OntoNotes CoNLL-F1 | Avg CoNLL-F1 | Avg Link F1 | Avg Mention F1 |
|-------|----------------|---------------------|--------------|-------------|-----------------|
| maverick-mes-litbank | 45.5 | 51.7 | 48.6 | 29.89 | 30.58 |
| augmented (Restricted) | 44.7 | 51.9 | 48.3 | 30.67 | 28.01 |
| **augmented-NR (Unrestricted)** | **49.7** | **52.5** | **51.1** | **32.02** | **37.49** |
| Gain NR vs Restricted | +5.0 | +0.6 | +2.8 | +1.35 | **+9.49** |

### Key Findings

- LitBank's person-centric annotation (83.1% PER) causes significant model overfitting: cross-domain models systematically fail on non-PER categories, a fact invisible via CoNLL-F1.
- The gap between NER and CNER is structural: NER only labels 22-53%, and many categories are collapsed into the "MISC" black box. Breaking "MISC" into CNER subcategories reveals independent failure modes for GROUP / MEDIA / SUPER, proving coarse evaluation hides significant blind spots.
- The augmentation experiment is the most powerful evidence: using 3 synthetic documents + Unrestricted annotation increased average CoNLL-F1 by +2.5 and Mention F1 by +9.5. Conversely, Restricted annotation (keeping only LitBank's 6 classes) performed worse than the baseline (-2.6 Mention F1), proving the issue is not "lack of data" but "instructional range of the annotation."
- LitBank models still perform worst on Link F1 cross-domain using gold mentions, indicating that "semantic bias" contaminates both mention extraction and clustering capabilities simultaneously.
- Causal Actionability of Diagnostics: The authors identified neglected categories (PLANT/EVENT/MEDIA/PSYCH) from typed F1 and targeted them with synthetic data, upgrading "evaluation" from "scoring" to an "engineering guide."

## Highlights & Insights

- Integrating "Concept + NER" to bring nominal concepts into the semantic layer is a simple improvement, but increasing coverage from 22% to 90% and categories from 4 to 29 is a classic case of "better tools breaking bottlenecks," reminding us that evaluation limits are often tool-driven.
- Cluster-level majority voting is a highly transferable trick—any task involving "entity/event/concept clustering + partially independent mention labeling" (e.g., entity linking, event coreference, dialogue role tracking) can use this two-step method to assign classes to pronouns or ambiguous mentions.
- The "Diagnosis $\rightarrow$ 3 Synthetic Docs $\rightarrow$ +9.5 Mention F1" loop solidifies the practical value of interpretable evaluation. Evaluation is not just "scoring"; it tells you exactly what data to supplement. This "evaluative actionability" should become a default requirement for future NLP papers.
- The comparison between Restricted and Unrestricted is highly educational—it shows the community that "data scale isn't the key, the annotation guidelines are," refuting the naive assumption that "any augmentation is helpful."

## Limitations & Future Work

- CNER cluster-level label precision is 90% (F1 88%), meaning 10-12% label noise propagates to the typed F1; the paper does not quantify how much this noise distorts evaluation conclusions.
- ~10% of mentions remain unlabeled, primarily in pronoun-only clusters; authors suggest training lightweight weak-supervision classifiers in the future.
- The framework is only validated for English; extension to other languages depends on multilingual CNER models, which were not provided.
- Data augmentation was only a small-scale PoC with 3 documents; industrial-scale augmentation strategies and quality control processes have not been established.
- No comparison with LLM coreference models (e.g., GPT-4 zero-shot) was introduced; while the framework is extensible, empirical data is missing.

## Related Work & Insights

- **vs. Agarwal et al. 2019 (NER-based coref evaluation)**: They also use semantic classes for stratified evaluation but are limited to 4 NER categories and <53% coverage; this paper uses CNER to reach 90% coverage and 29 categories, improving diagnostic granularity by an order of magnitude.
- **vs. Kummerfeld & Klein 2013 (error analysis toolkit)**: They use error type clustering for diagnosis, whereas this paper uses semantic types; the two are complementary—one focuses on "what error mode," the other on "what semantic category."
- **vs. Porada et al. 2024 (annotation guideline analysis)**: They argue cross-domain gaps are often due to annotation differences; this paper provides an actionable solution using typed F1 + augmentation (using Unrestricted annotation).
- **vs. LEA / MINA**: They modify aggregate score weights and boundary criteria, while this paper changes the diagnostic dimension without changing the algorithm; these directions should be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing CNER for coreference evaluation is a first; "two-step labeling + cluster propagation" is simple and elegant, though the "semantic evaluation" idea has precursors.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets × 3 models × per-class M-F1/L-F1 + manual verification + augmentation comparison + Restricted vs. Unrestricted counterfactual provides a very complete chain of evidence.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear argumentation; 3 Figures and 4 Tables cover all key claims; Limitations are addressed honestly.
- Value: ⭐⭐⭐⭐⭐ The "Evaluation $\rightarrow$ Data $\rightarrow$ Model" loop provides methodological inspiration for all NLP evaluation tasks and is a directly usable release for the coreference resolution community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LLMs Lean on Priors, Not Programming Language Semantics](../../ICML2026/interpretability/llms_lean_on_priors_not_programming_language_semantics.md)
- [\[AAAI 2026\] CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution](../../AAAI2026/interpretability/crosscheck-bench_diagnosing_compositional_failures_in_multim.md)
- [\[ACL 2026\] Constructing Interpretable Features from Compositional Neuron Groups](constructing_interpretable_features_from_compositional_neuron_groups.md)
- [\[ICML 2026\] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](../../ICML2026/interpretability/gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](../../CVPR2026/interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)

</div>

<!-- RELATED:END -->
