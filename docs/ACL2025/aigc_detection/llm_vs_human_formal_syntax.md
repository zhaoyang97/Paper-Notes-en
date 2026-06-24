---
title: >-
  [Paper Note] Comparing LLM-generated and human-authored news text using formal syntactic theory
description: >-
  [ACL 2025][AIGC Detection][HPSG] This study is the first to systematically investigate the grammatical differences between 6 LLMs and human NYT news writing across three levels—syntactic constructions (298 types), lexical types (1,398 types), and morphological rules (100 types)—using **HPSG formal syntactic theory** (via the English Resource Grammar, ERG). The findings reveal that LLMs represent an **"averaged" projection** of human authors in terms of grammatical features: g…
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "HPSG"
  - "English Resource Grammar"
  - "LLM text analysis"
  - "syntactic diversity"
  - "authorship analysis"
date: 2026-05-08
content_hash: 50566d3edb1e26c6
---

# Comparing LLM-generated and human-authored news text using formal syntactic theory

**Conference**: ACL 2025  
**arXiv**: [2506.01407](https://arxiv.org/abs/2506.01407)  
**Code**: [https://github.com/olzama/llm-syntax](https://github.com/olzama/llm-syntax/releases/tag/1.0.0)  
**Authors**: Olga Zamaraeva, Dan Flickinger, Francis Bond, Carlos Gómez-Rodríguez  
**Affiliations**: Universidade da Coruña, Independent Researcher, Palacký University at Olomouc  
**Area**: AIGC Detection  
**Keywords**: HPSG, English Resource Grammar, LLM text analysis, syntactic diversity, authorship analysis

## TL;DR

This study is the first to systematically investigate the grammatical differences between 6 LLMs and human NYT news writing across three levels—syntactic constructions (298 types), lexical types (1,398 types), and morphological rules (100 types)—using **HPSG formal syntactic theory** (via the English Resource Grammar, ERG). The findings reveal that LLMs represent an **"averaged" projection** of human authors in terms of grammatical features: grammatical differences among individual human authors are actually greater than any difference between humans and LLMs, while LLMs exhibit almost no differences among themselves.

## Background & Motivation

- **Background**: There is a growing body of research comparing LLM-generated text with human text. However, existing work primarily focuses on training classifiers (to detect AI generation) or analyzing shallow features like lexical distribution and Universal Dependencies (UD), lacking deep grammatical analysis grounded in independent linguistic theories.
- **Limitations of Prior Work**: Annotation schemes such as Universal Dependencies (UD) and Penn Treebank (PTB) are designed for NLP tasks. They have limited granularity and are not task-independent. For example, the `obj` relation in UD only refers to the dependency of a verb on its direct object, failing to distinguish more general head-complement constructions (nouns and adjectives can also take complements). Analyzing the output of NLP systems using NLP-oriented tools introduces inherent bias.
- **Key Challenge**: Prior research has found that LLMs tend to repeat POS sequence templates (Shaib et al. 2024) and prefer specific Biber rhetorical features (Reinhart et al. 2024). However, these are top-down feature sets that fail to provide a **comprehensive, consistent, and reproducible** grammatical analysis framework to cover the long-tail distribution of English syntax.
- **Goal**: How can we utilize formal linguistic theories independent of NLP to conduct a systematic and fine-grained comparison between LLM-generated and human-authored texts, down to the level of lexical-syntactic behavior?
- **Key Insight**: By utilizing the computational implementation of HPSG—the English Resource Grammar (ERG, covering 94% of well-edited English text)—each sentence can be parsed into a complete, typed syntactic structure. The distributional differences can then be analyzed statistically across three independent levels (syntactic constructions, lexical types, and morphological rules).

## Method

### Overall Architecture

The research consists of three phases: **Data Preparation $\rightarrow$ ERG Formal Syntactic Parsing $\rightarrow$ Multi-dimensional Statistical Analysis**. The core idea is to map text to an explicit typed hierarchical structure using HPSG theory, and then compare humans and LLMs in the type distribution space.

### Key Designs

#### 1. HPSG/ERG Three-level Parsing System

HPSG (Head-driven Phrase Structure Grammar) is a fully explicit formal syntactic theory that represents syntactic structures and semantic interfaces as complex feature-value graphs. ERG (English Resource Grammar) is its largest-scale implementation for English, with the following scale:

| Component | Total ERG Types | NYT Covered Types | Coverage |
|------|---------|---------------|--------|
| Syntactic types | 298 | 289 | 97% |
| Lexical types | 1,398 | 1,105 | 79% |
| Lexical entries | 44,366 | 27,311 | 61% |
| Morphological rules | 100 | 99 | 99% |

The core advantage of ERG lies in its **lexical type hierarchy**: the same word can belong to different lexical types, encoding distinct syntactic behaviors. For example, "law" has two lexical entries—`law_n1` (general countable/uncountable noun, "the law") and `law_n2` (noun taking a clausal complement, "There is a law that..."). Human texts utilize both, while LLMs only use `law_n1`. This granularity is unavailable in UD or POS tagging.

#### 2. Multi-dataset Cross-validation Design

Data sources cover three dimensions:

- **NYT Human Text**: Lead paragraphs of New York Times articles (26,102 sentences) from 2023.10.01 to 2024.01.24, retrieved via the NYT Archive API.
- **LLM-generated Text**: 6 models (LLaMA-7B/13B/30B/65B, Falcon-7B, Mistral-7B), generated using NYT headlines + first 3 words as prompts (approx. 214K sentences in total). All LLMs were released prior to 2023.10.01 to ensure they had not seen the corresponding human articles.
- **Redwoods Treebank**: Portions of WSJ (43,043 sentences) and Wikipedia (10,726 sentences)—used to verify whether the findings hold across styles/genres.

The experimental design deliberately isolates two factors: ① **Model scaling** (the LLaMA series with the same architecture but different sizes); ② **Model architecture** (LLaMA vs. Falcon vs. Mistral).

#### 3. Statistical Analysis Methods

1. **Cosine Similarity + PCA Projection**: Normalize the HPSG type frequencies of each dataset into vectors, calculate pairwise cosine similarity, and use PCA projection to visualize differences within the 98%-100% similarity range.
2. **Shannon Entropy $H$ and Gini-Simpson Index $1-\lambda$**: Quantify the diversity (evenness) of construction usage, with statistical significance validated using 10,000 permutation tests.
3. **Individual Author Analysis**: Select 12 NYT reporters who published $>100$ articles, calculate pairwise cosine similarities of their HPSG type distributions, and cross-compare with LLMs.
4. **Mann-Whitney U Test**: Perform statistical significance tests (with FDR correction) on the relative frequency differences of individual HPSG types.

## Key Experimental Results

### Table 1: Syntactic Construction Frequency Differences (25K sentence sample)

| Construction | Example | Human Freq. | LLM Mean | Direction |
|------|------|---------|---------|------|
| Head-complement | "It's not acceptable for democracy" | 164,806 | 224,529 | LLM >> Human |
| Subject-head | "The house passed the measure…" | 17,850 | 27,753 | LLM >> Human |
| Quantity NP | "many in Europe" | 23,611 | 40,881 | LLM >> Human |
| Relative clauses | "a vote that many have seen…" | 4,929 | 6,721 | LLM >> Human |
| Clause with extracted subject | "Chris Snow became an advocate…" | 5,072 | 7,327 | LLM >> Human |
| Marker clause | "and that's a good thing" | 2,891 | 5,660 | LLM >> Human |
| Clause conjunction fragment | "But the observation suits him." | 939 | 2,076 | LLM >> Human |
| Questions | "How do you stay safe?" | 268 | 428 | LLM >> Human |
| Participial clause | "having tried that,…" | 1,736 | 1,116 | Human >> LLM |
| Modifier clause apposition | "his critics, mostly unnamed" | 826 | 434 | Human >> LLM |
| Bare NP coordination | "author and commentator" | 311 | 117 | Human >> LLM |
| Paired marker | "Both this and other discussions" | 326 | 185 | Human >> LLM |
| Adjective-participle modifier | "right-handed", "red-colored" | 125 | 64.6 | Human >> LLM |
| Double NP apposition | "an eye for detail, decades of…" | 11 | 5.2 | Human >> LLM |
| Absolute VP | "As told, …" | 10 | 3.8 | Human >> LLM |

**Core Pattern**: LLMs heavily utilize the most general, basic constructions (head-complement, subject-head), whereas humans employ a richer variety of low-frequency stylistic constructions (participial modifiers, double NP apposition, absolute VP).

### Table 2: Diversity Metric Comparison (Shannon Entropy)

| Dimension | Human NYT | LLaMA-7B | LLaMA-13B | LLaMA-30B | LLaMA-65B | Falcon-7B | Mistral-7B | All LLMs Combined |
|------|---------|----------|-----------|-----------|-----------|-----------|------------|-------------|
| Syntactic Construction $H$ | **3.342** | 3.259 | 3.249 | 3.270 | 3.284 | 3.221 | 3.267 | 3.265 |
| Lexical Type $H$ | 4.727 | 4.844 | **4.877** | 4.858 | 4.860 | 4.700 | 4.847 | — |

- All differences were validated as significant via permutation tests (10,000 resamples, $p < 0.01$).
- Syntactic construction diversity: **Human is the highest** ($H=3.342$), LLaMA-65B is the closest ($H=3.284$), and Falcon is the lowest ($H=3.221$).
- Lexical type diversity shows a **reversal**: most LLMs are higher than humans (LLaMA-13B is the highest with $H=4.877$ vs. Human $H=4.727$).
- After combining all LLM outputs, syntactic diversity actually drops to $H=3.265$—aggregation amplifies the high-frequency general constructions shared across models.

### Key Findings on Cosine Similarity

**Syntactic Construction Cosine Similarity (Selected Raw Data)**:

| Comparison Pair | Cosine Similarity |
|--------|-----------|
| LLaMA-30B vs LLaMA-65B | 0.9999 |
| LLaMA-7B vs Mistral-7B | 0.9999 |
| Falcon-7B vs LLaMA-7B | 0.9966 |
| LLaMA-65B vs Human NYT | 0.9964 |
| LLaMA-7B vs Human NYT | 0.9955 |
| WSJ vs Human NYT | 0.9949 |
| Wikipedia vs Human NYT | 0.9833 |

The syntactic similarity between LLMs (0.9966–0.9999) is consistently higher than that of any LLM compared to humans (0.9950–0.9965), which in turn is higher than the cross-genre similarity of human texts (Wikipedia vs. NYT = 0.9833).

### Vocabulary Footprint Differences (25K sentence sample)

| Model | Human-Exclusive Lexical Types | LLM-Exclusive Lexical Types | Human-Exclusive Lexical Entries | LLM-Exclusive Lexical Entries |
|------|----------------|----------------|------------|------------|
| LLaMA-7B | 62 | 70 | 5,704 | 2,519 |
| LLaMA-13B | 71 | 80 | 5,557 | 2,617 |
| LLaMA-30B | 65 | 62 | 5,531 | 2,608 |
| LLaMA-65B | 66 | 74 | 5,302 | 2,745 |
| Mistral-7B | 73 | 76 | 5,809 | 2,353 |
| Falcon-7B | 91 | 55 | 6,212 | 2,015 |
| All LLMs | 66 | 70 | 1,721 | 2,398 |

Humans alone use about **twice** as many lexical entries as a single LLM (5,000–6,000 vs. 2,000–2,700), but when considering all LLMs combined (2,398 vs. 1,721), the collective vocabulary coverage of LLMs surpasses humans.

### Core Findings: Individual Authors vs. LLMs

- **Inter-human differences > Human-LLM differences**: The variance in cosine similarity of syntactic distributions among the 12 NYT reporters is significantly larger than any difference between humans and LLMs.
- **Minimal differences among LLMs**: The 6 LLMs are tightly clustered across all type dimensions.
- **Human variance is largest in the lexical type dimension**: Individual humans differ significantly in lexical type usage, whereas LLMs exhibit minimal variance.
- **Minimal difference in the morphological rule dimension**: Under the NYT genre constraints, humans and LLMs are almost indistinguishable in inflectional/derivational morphology (cosine similarity 0.9962–0.9990), with Falcon being the sole exception.

## Highlights & Insights

1. **The "LLM as a Grammatical Average" Hypothesis**: This is the most profound finding of the paper. LLM-generated text behaves as an "averaged" projection of human authors in the grammatical dimension. The fact that inter-human variance is greater than Human-LLM variance is precisely because each LLM learns a "compromise" grammatical style, smoothing out individual idiosyncrasies. This explains why LLMs prefer the most general head-complement constructions.

2. **Importance of Three-Level Decoupling**: Humans are more diverse in syntactic constructions ($H=3.342 > 3.284$), but LLMs are more diverse in lexical types ($H=4.877 > 4.727$), and there is almost no difference in morphological rules. Without decoupled analysis, these patterns would be obscured. This indicates that linguistic analysis must distinguish between morphological, syntactic, and lexical levels.

3. **Diagnostic Value of Long-Tail Constructions**: ERG covers the full long-tail distribution of English syntax, revealing that LLMs overuse several constructions rarely used by humans (e.g., number sequences, parenthetical modifiers, fragmented lexical conjunctions like "But!"), while lacking stylistic constructions that humans occasionally employ (absolute VPs, measure-noun modifier phrases).

4. **"Informality" of Human Text**: Despite strict style guidelines at the NYT, human authors still use informal vocabulary ("haven't", "a couple dozen"), imperative sentences ("See the results..."), and direct, emphatic expressions ("at your own risk") more frequently than LLMs. LLMs adhere more consistently to the formal prompted style, with their unique lexical entries heavily featuring numbers and punctuation types.

5. **Superiority of Formal Grammar as an Analytical Tool**: ERG distinguishes grammatical phenomena that UD cannot handle—head-complement is not equivalent to UD's `obj`, and lexical types distinguish different syntactic usages of the same word. This granularity enables this paper to discover novel patterns of discrepancy.

## Limitations & Future Work

1. **Limited to English NYT Genre**: ERG is currently the only large-scale HPSG grammar reaching 94% coverage. HPSG grammars for other languages are not mature enough to support a similar scale of analysis, restricting cross-lingual generalization.
2. **Outdated LLM Versions**: The evaluated LLMs are LLaMA-1, Falcon-7B, and Mistral-7B (all released before 2023), without covering newer generation models like GPT-4, Claude, or LLaMA-3.
3. **Limited Statistical Power**: Only 9 datasets were compared. After FDR correction for multiple comparisons, differences in high-frequency constructions became statistically non-significant—more diverse datasets are required for robust statistical conclusions.
4. **Single Generation Control**: LLMs only used one prompting strategy (headline + first 3 words). Different prompts and temperature settings could affect syntactic preferences.
5. **Counter-intuitive Consistency in Morphological Rules**: Why humans and LLMs align so tightly on morphological rules was not deeply explored—is it due to genre constraints or the low inherent variability of English morphology?

## Related Work & Insights

| Work | Analysis Framework | Granularity | Core Findings |
|------|---------|---------|---------|
| Muñoz-Ortiz et al. 2024 | UD Dependency Syntax | Dependency relations + Vocabulary | Human text is shorter, dependency distance is more optimized, vocabulary is more diverse |
| Shaib et al. 2024 | POS Sequence Templates | POS n-gram | LLMs tend to repeat POS templates |
| Reinhart et al. 2024 | Biber Rhetorical Features | Predefined feature sets | LLMs prefer participial clauses, "that"-clauses, nominalization |
| Sardinha 2024 | Biber Features | Predefined feature sets | LLMs and humans differ systematically across rhetorical dimensions |
| **Ours** | **HPSG/ERG** | **298 constructions + 1398 lexical types + 100 morphological rules** | **LLMs represent the "grammatical average"; three levels show distinct difference patterns** |

**Insights for Future Work**:
- HPSG analysis can serve as a **complementary feature** for AIGC detection—especially since long-tail construction distribution patterns may be more robust than surface-level statistical features.
- The "LLM as a grammatical average" hypothesis can guide **text watermarking design**—by injecting low-frequency stylistic constructions to enhance watermark imperceptibility.
- The three-level decoupling capability can be transferred to **other languages** for LLM evaluation—even without a large-scale HPSG grammar, a more fine-grained syntactic analysis framework can be used to replace coarse-grained UD.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First study using formal syntactic theory independent of NLP to analyze LLM text, demonstrating high methodological originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-validated with 6 LLMs and 3 human datasets (NYT/WSJ/Wikipedia), though statistical power is somewhat limited by the number of datasets.
- Writing Quality: ⭐⭐⭐⭐ Accessible to both linguistics and NLP readers with rich examples, though discussion on some conclusions is slightly scattered.
- Value: ⭐⭐⭐⭐ The finding of "LLMs as a grammatical average" is profound and inspiring; the methodology is reusable but restricted by the availability of English ERG.
- Technical Depth: ⭐⭐⭐⭐ Solid application of HPSG theory with sound statistical analysis, though no new models or algorithms are proposed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](../../ACL2026/aigc_detection/temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)
- [\[ACL 2025\] A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI](a_rose_by_any_other_name_llm-generated_explanations_are_good_proxies_for_human_e.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ACL 2025\] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)

</div>

<!-- RELATED:END -->
