---
title: >-
  [Paper Note] The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research
description: >-
  [ACL 2025][citation fidelity] An end-to-end computational pipeline is developed to quantify scholarly citation fidelity at scale. Analyzing 13 million citation sentence pairs reveals critical factors influencing citation fidelity, and a quasi-causal experiment confirms a "telephone effect" where low-fidelity intermediate citations lead to further distortion in subsequent citations.
tags:
  - "ACL 2025"
  - "citation fidelity"
  - "telephone effect"
  - "bibliometrics"
  - "scientific information change"
  - "knowledge transmission"
date: 2026-05-08
content_hash: 49d719f9e2168e53
---

# The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research

**Conference**: ACL 2025  
**arXiv**: [2502.20581](https://arxiv.org/abs/2502.20581)  
**Code**: [hongcchen/citation](https://github.com/hongcchen/citation)  
**Area**: Bibliometrics / Scientific Citation Analysis  
**Keywords**: citation fidelity, telephone effect, bibliometrics, scientific information change, knowledge transmission  

## TL;DR

An end-to-end computational pipeline is developed to quantify scholarly citation fidelity at scale. Analyzing 13 million citation sentence pairs reveals critical factors influencing citation fidelity, and a quasi-causal experiment confirms a "telephone effect" where low-fidelity intermediate citations lead to further distortion in subsequent citations.

## Background & Motivation

**Background**: Scholarly citations are widely used to evaluate research impact and track the flow of knowledge. Researchers typically rely on raw citation counts, implicitly assuming that all citations are similar in nature. However, a large body of literature has shown significant differences in citation types and fidelity—citations may paraphrase, generalize, or even distort original research findings.

**Limitations of Prior Work**: (1) Existing citation analysis primarily focuses on citation intent/function classification (e.g., Jurgens et al. 2018; Cohan et al. 2019), rather than the fidelity of citations to the original information; (2) The few studies investigating citation fidelity rely on manual annotation, which is limited in scale (e.g., Cobb et al. 2024 manually inspected only 3,347 citation claims and found that $\approx 19\%$ contained mischaracterizations); (3) Measuring citation fidelity goes beyond standard semantic similarity—it requires capturing subtle nuances unique to scientific texts, such as abstraction level, uncertainty, and context appropriateness. A classic case: a brief 1980 letter about opioid addiction rates in hospitalized patients was repeatedly cited over downstream decades as evidence for "safety of prescribing opioids," accumulating over 600 citations by 2017, but fewer than $20\%$ of these citations mentioned the critical caveats of the original study.

**Key Challenge**: The academic community routinely uses citation counts as a key metric for research evaluation and knowledge flow, yet the information fidelity of citations itself is highly variable—if information loss or distortion is prevalent, the reliability of evaluation systems and knowledge-tracking mechanisms based on raw citation counts is put into question.

**Goal**: (1) Develop an operational pipeline to calculate citation fidelity at scale, overcoming the scale limitations of manual annotation; (2) Systematically reveal which factors (temporal distance, disciplinary distance, accessibility, team features) affect citation fidelity; (3) Verify the "telephone effect" through quasi-causal experiments—i.e., whether intermediate citations cause downstream citations to further distort the original information.

**Key Insight**: This work conceptualizes citation fidelity as a reflection of authors' "depth of engagement" with the cited literature, leverages the scientific information change metric proposed by Wright et al. (2022) to quantify the information fidelity between citations and original claims at the sentence level, and establishes causal inference through a precisely matched quasi-experimental design.

**Core Idea**: Quantify citation fidelity at scale and reveal its systematic variations—closer temporal, disciplinary, or social proximity and higher accessibility lead to higher citation fidelity, while the "telephone effect" of intermediate citations propagates and amplifies information distortion.

## Method

### Overall Architecture

The end-to-end citation fidelity computation pipeline: (1) Extract citation sentence pairs (the citing paper's reporting citation sentence + the cited paper's corresponding claim sentence) from the S2ORC corpus (metadata for 136M papers, full text for 42M papers); (2) Assess the fidelity of each pair (scored 1-5) using a sentence-level scientific information change metric; (3) Reveal systematic factors influencing fidelity through regression analyses; (4) Test the "telephone effect" hypothesis through a precisely matched quasi-causal experiment involving approximately 50K paper pairs. Ultimately, around 13 million citation sentence pairs were generated for analysis.

### Key Designs

1. **Citation Sentence Pair Construction Pipeline**:
    - **Function**: Automatically extract high-quality citation-claim sentence pairs from large-scale full-text corpora.
    - **Mechanism**: Filter single-source citations (parenthetical citations at sentence end) from the citing paper $\rightarrow$ Identify background citations using a SciBERT-fine-tuned classifier ($F_{1}=0.81$) $\rightarrow$ Extract results/conclusions sentences from the cited paper using a RoBERTa-fine-tuned classifier ($F_{1}=0.92$) $\rightarrow$ Perform upper-bound matching (taking the highest fidelity score between the citing sentence and all claim sentences of the cited paper as the proxy fidelity score for the citation pair).
    - **Design Motivation**: Focusing on reporting background citations (the citation type most likely to paraphrase original findings), while upper-bound matching ensures capturing citation fidelity in the best-case scenario.

2. **Scientific Information Change Measurement**:
    - **Function**: Quantify the extent to which two semantically similar sentences describe the same scientific claim at the sentence level.
    - **Mechanism**: Implement Wright et al. (2022)'s model fine-tuned on MPNet, scoring on a 1-5 scale. Unlike standard semantic similarity, this metric is specifically tailored for scientific texts, capturing dimensions like topical alignment, shifts in abstraction, hedging/certainty expressions, and context appropriateness.
    - **Design Motivation**: Standard NLI (Natural Language Inference) or STS (Semantic Textual Similarity) models cannot differentiate between "reasonable generalization" and "information loss/distortion"—the scientific information change metric more accurately meets the requirements of measuring citation fidelity.

3. **"Telephone Effect" Quasi-Causal Experimental Design**:
    - **Function**: Test whether intermediate citations causally result in downstream citation information distortion.
    - **Mechanism**: Identify $\approx 50\text{K}$ paper pairs $(C, D)$—both citing the same claim of original paper $A$, where $C$ also cites an intermediate paper $B$ (which itself cites $A$), while $D$ only cites $A$. Precisely match $C$ and $D$ based on publication year and field of study, and compare their respective citation fidelity to $A$.
    - **Design Motivation**: Exact matching is used to control for confounding factors, utilizing "citing the intermediate paper" as a proxy variable for "being influenced by the intermediate paper."

### Loss & Training

- SciBERT citation function classifier: binary cross-entropy loss (background vs. non-background citations), $F_{1}=0.81$
- RoBERTa discourse classifier: five-class cross-entropy loss (method/background/objective/result/conclusion), $F_{1}=0.92$
- Scientific information change model (MPNet fine-tuned): cross-entropy loss over the dot product of sentence embeddings, $\text{MSE}=0.489$, $\text{Pearson}=76.48$

## Key Experimental Results

### Main Results: Factors Influencing Citation Fidelity (Regression Analysis, ~13M citation sentence pairs)

| Factor | Effect on Citation Fidelity | Direction |
|------|-------------------|---------|
| Self-citation vs. Non-self-citation | Self-citations have significantly higher fidelity | **Positive correlation** |
| Within-domain vs. Cross-domain | Within-domain citations have higher fidelity | **Positive correlation** |
| Open Access vs. Closed Access | Citations to Open Access papers have higher fidelity | **Positive correlation** |
| Publication temporal gap | Shorter publication gap correlates with higher fidelity | **Negative correlation** |
| First author's h-index | Higher seniority correlates with lower fidelity | **Negative correlation** |
| Last author's h-index | No significant correlation | Not significant |
| Team size | Mid-sized teams have the highest fidelity | **Inverted U-shape** |

### "Telephone Effect" Quasi-Causal Experiment (~50K paper pairs)

| Experimental Group | Citation Fidelity to Original Paper $A$ | Difference |
|--------|------------------------|------|
| Control Group (Cites $A$ only) | Baseline | - |
| Treatment Group (Cites $A$ + intermediate paper $B$) | **0.06 lower** | Significant decrease |
| Treatment Group - $B$ has high fidelity to $A$ ($>4$ points) | Minimal decrease | Weak effect |
| Treatment Group - $B$ has mid fidelity to $A$ ($3\text{-}4$ points) | Moderate decrease | Moderate effect |
| Treatment Group - $B$ has low fidelity to $A$ ($<3$ points) | **Largest decrease** | Strong effect |

### Key Findings

- The distribution of citation fidelity is roughly normal, with most scores clustered around **3.5** (on a 1-5 scale), indicating that most citations undergo some degree of information shift.
- Shorter distances (temporal, disciplinary, and author-related) correlate with higher citation fidelity, supporting the "depth of engagement" hypothesis.
- An increase in first-author seniority, surprisingly, decreases citation fidelity (**significantly negative correlation**), which may reflect that senior researchers rely more on heuristics rather than reading the original texts directly.
- The telephone effect experiments confirm **$H_{1}$** (intermediate sources decrease fidelity, dropping by 0.06 points) and **$H_{2}$** (the lower the intermediate source's fidelity, the more severe the downstream distortion).

## Highlights & Insights

- Scales up the study of scientific citation fidelity from small-cohort qualitative discussions to a large-scale quantitative measurement of 13 million sentence pairs.
- The quasi-causal experiment elegantly reveals the information decay mechanism in citation chains—the "telephone game" effect.
- Findings provide critical implications for academic evaluation systems (not relying solely on raw citation counts) and scientific communication practices.
- Uncovered a counterintuitive yet important insight: first-author seniority is negatively correlated with citation fidelity.

## Limitations & Future Work

- Only a single sentence was used as the citation context window, whereas real-world citations might span multiple sentences or occupy only a brief clause.
- The upper-bound matching method may overestimate fidelity (the highest-scoring claim might not be the actual claim the author intended to cite).
- The fidelity score is an aggregated metric that cannot differentiate between reasonable generalization, selective reporting, and substantive mischaracterizations.
- The analysis is restricted to English papers within S2ORC, lacking full coverage of cross-lingual and domain-specific nuances.
- Using "citing the intermediate paper" as a proxy for "reading the intermediate paper" serves as a limitation, as drawing a citation does not always equate to active reading.

## Related Work & Insights

- **Citation Classification**: Teufel et al. 2006; Jurgens et al. 2018; Cohan et al. 2019 — analyses of citation function and intent.
- **Citation Misuse**: Greenberg 2009 (unfounded claims becoming "fact" through chains of supportive citations), Simkin & Roychowdhury 2005 (estimating that $70\text{-}90\%$ of citations are copied).
- **Scientific Information Change Measurement**: Wright et al. 2022 — the basis of the core measurement model in this study.
- **Insights**: The proposed pipeline can be transferred to detect citation hallucinations in LLM-generated texts.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CiteEval: Principle-Driven Citation Evaluation for Source Attribution](citeeval_principle-driven_citation_evaluation_for_source_attribution.md)
- [\[ACL 2025\] Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)
- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[ACL 2025\] Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)

</div>

<!-- RELATED:END -->
