---
title: >-
  [Paper Note] Research Borderlands: Analysing Writing Across Research Cultures
description: >-
  [ACL 2025][research culture] Through interviews with interdisciplinary researchers, this work constructs a cultural norm framework for academic writing (comprising four categories: structure, style, rhetoric, and citation). It quantifies writing differences across 11 CS communities using computational metrics, revealing a severe "homogenization" tendency in LLMs during cross-community writing adaptation.
tags:
  - "ACL 2025"
  - "research culture"
  - "cultural norms"
  - "scientific writing"
  - "LLM evaluation"
  - "interdisciplinary research"
  - "cultural competence"
date: 2026-05-08
content_hash: af6035057740b186
---

# Research Borderlands: Analysing Writing Across Research Cultures

**Conference**: ACL 2025  
**arXiv**: [2506.00784](https://arxiv.org/abs/2506.00784)  
**Code**: [shaily99/research_borderlands](https://github.com/shaily99/research_borderlands)  
**Area**: Other  
**Keywords**: research culture, cultural norms, scientific writing, LLM evaluation, interdisciplinary research, cultural competence  
**Authors**: Shaily Bhatt (CMU), Tal August (UIUC), Maria Antoniak (Copenhagen)  

## TL;DR

Through interviews with interdisciplinary researchers, this work constructs a cultural norm framework for academic writing (comprising four categories: structure, style, rhetoric, and citation). It quantifies writing differences across 11 CS communities using computational metrics, revealing a severe "homogenization" tendency in LLMs during cross-community writing adaptation.

## Background & Motivation

- **Limitations of Prior Work**: In evaluating the cultural competence of LLMs, the concept of "culture" is vaguely defined. Most studies rely on coarse-grained proxy variables such as nationality or language, lacking in-depth interaction with community members.
- **Key Insight**: Academic writing serves as an explicit carrier of cultural norms—different research communities (such as NLP, HCI, and Education) hold distinct implicit expectations regarding paper structure, terminology, and argumentation styles.
- **Goal**: To discover and measure cultural norms in a bottom-up, "human-centered" manner, rather than presetting proxy variables top-down. The study also aims to evaluate whether LLMs can adapt to the cultural norms of different research communities when functioning as writing tools.
- **Ecological Validity**: Surveys ($N=78$) verify that "cross-community paper adaptation" is a real-world task faced by researchers, establishing a foundation for this study.

## Method

### 1. Qualitative Study: Discovering Cultural Norms

- **Survey ($N=78$)**: Targeting interdisciplinary researchers to confirm that paper adaptation is a real-world demand; 66 respondents reported experience in cross-community adaptation, and almost all of them adjust the Introduction section during adaptation.
- **Semi-structured Interviews ($N=10$)**: Conducted with senior interdisciplinary scholars for 60 minutes each; prior to the interviews, participants were asked to provide multiple versions of the Introduction of the same paper submitted to different communities for comparison.
- **Coding Analysis**: Two authors independently coded the first two interview transcripts, establishing a unified cultural norm framework after three weeks of iterative discussions.

### 2. Cultural Norm Framework (Four Categories)

| Category | Norm Dimension | Example Differences |
|------|---------|---------|
| **Structural Norms** | Length, use of tables/figures | NLP papers 8-9 pages vs. FAccT papers 14 pages; CV community uses more figures, NLP community uses more tables |
| **Stylistic Norms** | Terminology/jargon, readability, formality, redundancy | The NLP community does not require explanations for "RoBERTa"; Education uses words like "preponderance"; Humanities allow informal prose |
| **Rhetorical Norms** | Quantitative evidence, figurative language, framing, narrative organization | ML/NLP emphasize numerical evidence; Humanities favor narrative storytelling; 9/10 interviewees consider "reframing" the most critical adaptation step |
| **Citation Norms** | Classic citations, citation interaction styles | The same concept has different seminal citations across communities (e.g., "mental models" in Cognitive Science vs. "folk theories" in HCI); Humanities often quote directly at the beginning |

### 3. Computational Evaluation Suite

Operationalizes quantifiable norms in the framework into computational metrics:

- **Structure**: Word count, sentence count (NLTK tokenization), and rates of table/figure mentions (regular expressions).
- **Style**: Terminology specificity (NPMI specificity score), formality (DeBERTa-large fine-tuned on GYAFC), and readability (Flesch reading-ease).
- **Rhetoric**: Percentage of quantitative evidence (Llama 3.1 70B as judge, showing 93% agreement with human labels), narrative organization (skewness of sentence positions classified by function), and value framing (10-dimensional value vector based on a lexicon classifier with 72.95% precision).

### 4. LLM Cultural Competence Evaluation

- **Task**: Given an Introduction from a source community, instruct the LLM to adapt it to the style of a target community.
- **Data Sampling**: (a) 100 random papers per community pair; (b) 100 papers with the highest specificity per community pair (better resembling real-world adaptation scenarios).
- **Models**: GPT-3.5 Turbo, GPT-4o Mini, Llama 3.1 8B, Llama 3.3 70B, and Mistral Ministral 8B; 5 samples per prompt, totaling 550,000 generations.

## Key Experimental Results

### Differences in Writing Norms Across 11 CS Communities (Section 6, Figure 3)

| Metric | Key Findings |
|------|---------|
| Length | Economics & Computation is the longest, while NLP is relatively short |
| Figures/Tables | The CV community uses the most figures, while the NLP community uses the most tables |
| Terminology Specificity | All communities have positive values, with the Education community being the most distinctive |
| Formality | Differences across communities are minor (as all are CS subfields) |
| Quantitative Evidence | ML/NLP/AI exhibit the smallest variance, suggesting that quantitative evidence is a strong cultural norm in these fields |
| Narrative Organization | Objective sentences appear earlier in ML/NLP/AI; results sentences also appear earlier in the AI community |

### Table 2: LLM Cultural Competence Evaluation (Section 7, targeting ML and NLP communities)

| Observation | Details |
|------|------|
| **Vocabulary Adaptation Success** | Specificity almost always increases after adaptation across all LLMs, indicating that models actually understand vocabulary differences between communities |
| **Homogenization in Other Dimensions** | Length is consistently shortened, mention rates of tables/figures consistently decrease, readability consistently drops, and the proportion of quantitative evidence slightly increases; successes only occur by chance when the target direction happens to align |
| **Rigid Narrative Organization** | The skewness of background and methodology sentences increases, while that of objective sentences decreases, showing that all models tend to converge on a unified template |
| **Framing Similarity** | The cosine similarity of framing remains almost unchanged before and after adaptation; LLMs fail to adjust the value framing based on the target community |

## Highlights & Insights

- **Methodological Innovation**: Discovers cultural norms in a bottom-up, human-centered manner. Instead of presetting proxy variables, the study conducts interviews with interdisciplinary experts to establish a framework that comprehensively covers structure, style, rhetoric, and citation.
- **Large-Scale Validation**: Quantitative analysis of 81,178 papers across 11 CS communities and 38 conferences successfully replicates the qualitative observations described by the interviewees.
- **Discovery of LLM Homogenization**: First to systematically demonstrate that LLMs exhibit comprehensive homogenization (except for vocabulary) during cross-community writing adaptation. The large-scale experiment involving 550,000 generations provides highly convincing evidence.
- **Open-Source Evaluation Suite**: Provides reusable computational metrics and code, which can be applied to scientometrics and LLM cultural competence evaluation.

## Limitations & Future Work

- **Community Scope**: Only covers subfields of CS, leaving out disciplines with potentially more pronounced differences such as sociology, biology, and art history.
- **Interviewee Bias**: The 10 interviewees are primarily from ML/NLP and Computational Social Science, and recruiting through social media introduces potential selection bias.
- **Metric Limitations**: Formality metrics show insufficient discrimination across CS communities; verbosity and figurative language are excluded due to the lack of reliable metrics; citation norms are not computed due to the inability to map in-text citations.
- **Evaluation Scheme**: LLMs are evaluated only under zero-shot settings, without exploring methods like few-shot learning or RAG to improve cultural competence.
- **Proxy Selection**: Incorporates 'communities' rather than 'conferences' as the cultural unit. While verified by surveys, this approach might neglect differences across sub-directions within the same community.

## Related Work & Insights

- **Understanding Research Communities**: Lucy et al. (2023) analyze lexical specificity across communities; Birhane et al. (2022) and Jiang et al. (2025) study values encoded in ML research; Michael et al. (2023) investigate beliefs in the NLP community.
- **LLMs as Scientific Tools**: Si et al. (2024) leverage LLMs for scientific idea generation; Robinson et al. (2024) focus on writing assistance for a single community—neither considers cross-community cultural differences.
- **Cultural Competence of LLMs**: Adilazuarda et al. (2024) review the vague definitions of "culture"; Rao et al. (2024) construct geocultural benchmarks based on expert documents but use artificial settings; this work is driven by real-world tasks and community members, offering a more grounded methodology.
- **Homogenization in LLM Writing**: Liang et al. (2024) find that LLM-assisted papers are shorter and more similar; Guo et al. (2024) observe a drop in linguistic diversity; Xu et al. (2025) discover a reduction in rhetorical diversity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Approaches NLP cultural competence evaluation from anthropological and qualitative perspectives, offering a fresh angle.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Incorporates 81K papers and 550K LLM generations, presenting a massive scale with highly convincing metric designs.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Smooth narrative flow using mixed methods, closely integrating qualitative and quantitative perspectives.
- **Value**: ⭐⭐⭐⭐ — Provides a critical warning regarding the cultural adaptability of LLM-based scientific writing tools; the evaluation suite is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[ACL 2025\] Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)
- [\[ACL 2025\] All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)
- [\[ACL 2025\] IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)
- [\[ACL 2025\] The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research](the_noisy_path_from_source_to_citation_measuring_how_scholars_engage_with_past_r.md)

</div>

<!-- RELATED:END -->
