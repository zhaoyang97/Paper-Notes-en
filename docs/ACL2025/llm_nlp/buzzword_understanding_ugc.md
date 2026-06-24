---
title: >-
  [Paper Note] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content
description: >-
  [ACL 2025][LLM (Other)][buzzword] This paper constructs the first Chinese internet buzzword dataset, Cheer (1,127 instances), and proposes the Ress method, which guides LLMs to generate more accurate buzzword definitions from user-generated content by simulating the six-dimensional comprehension skills of child language acquisition, improving semantic accuracy by an average of 2.51%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "buzzword"
  - "definition generation"
  - "user-generated content"
  - "language acquisition"
  - "LLM"
date: 2026-05-08
content_hash: 7886a684fff6796d
---

# Can Large Language Models Understand Internet Buzzwords Through User-Generated Content

**Conference**: ACL 2025  
**arXiv**: [2505.15071](https://arxiv.org/abs/2505.15071)  
**Code**: [https://github.com/SCUNLP/Buzzword](https://github.com/SCUNLP/Buzzword)  
**Area**: LLM/NLP  
**Keywords**: buzzword, definition generation, user-generated content, language acquisition, LLM

## TL;DR

This paper constructs the first Chinese internet buzzword dataset, Cheer (1,127 instances), and proposes the Ress method, which guides LLMs to generate more accurate buzzword definitions from user-generated content by simulating the six-dimensional comprehension skills of child language acquisition, improving semantic accuracy by an average of 2.51%.

## Background & Motivation

**Background**: Internet buzzwords (e.g., "wonnangfei" [useless money], "zero-frame start") emerge rapidly on social media, often conveying highly abstract and culture-specific meanings that transcend dictionary definitions. Traditional dictionaries do not document these words, forcing users to comprehend them through various contextual usage scenarios. Context-aware definition generation is the corresponding task direction in NLP.

**Limitations of Prior Work**: Existing definition generation methods (including LLM-based ones) perform reasonably well on conventional vocabulary but poorly on internet buzzwords. The reasons include: 1) buzzwords evolve rapidly, and static training datasets cannot keep pace; 2) LLMs inherently have limited comprehension of long-tail words and neologisms; 3) even when user-generated content (UGC) examples are provided, the reasoning capability of LLMs is insufficient to infer the complete meaning from limited contexts.

**Key Challenge**: The meanings of internet buzzwords are highly dependent on context and cultural background, but LLMs tend to rely on seen word meanings in their parameterized knowledge, lacking effective reasoning capabilities for unseen buzzwords. Additionally, acquiring and filtering high-quality UGC is a bottleneck, as it is difficult to judge which UGC is most informative without prior knowledge of the word's meaning.

**Goal**: 1) Addressing the lack of a dedicated Chinese buzzword dataset and evaluation benchmark; 2) Improving how LLMs understand buzzwords from UGC and generate accurate definitions.

**Key Insight**: Borrowing theories of child language acquisition from cognitive science, where children learn new words through six core skills: intent understanding, conceptual association, linguistic structure, social clues interpretation, word context, and pronunciation/spelling. The authors encode these six skills as guiding aspects for LLMs, generating candidate definitions separately before ensembling them.

**Core Idea**: Translating the six-dimensional comprehension skills of child language acquisition into LLM prompting strategies, guiding the model to understand buzzwords from multiple perspectives and then ensemble a final definition.

## Method

### Overall Architecture

The input consists of a buzzword term and a set of UGC example sentences, and the output is the precise definition of the buzzword. The Ress method consists of three steps: first initializing six comprehension dimensions (aspects), then generating a candidate definition guided by each aspect in LLMs, and finally merging the six candidates into a unified final definition through an ensembling step.

### Key Designs

1. **Cheer Dataset**:

    - **Function**: Providing the first Chinese internet buzzword definition generation benchmark.
    - **Mechanism**: Collecting 1,127 Chinese buzzwords from platforms like "Gengbaike", each paired with a description (averaging 262.5 characters), a refined definition (averaging 50 characters, summarized by LLMs for both literal and figurative meanings), and an average of 30.7 UGC examples from Xiaohongshu/Weibo. A three-tier quality control process (dictionary websites -> netizens' usage -> manual audit) was implemented to manually remove inappropriate terms, refine definitions, and clear out explicit explanatory information in UGC.
    - **Design Motivation**: Prior to this, there was no dedicated dataset for internet buzzword definition generation; a specialized benchmark is required to reveal and quantify the limitations of existing methods.

2. **Ress Aspect Initialization**:

    - **Function**: Encoding key skills of child language acquisition as comprehension dimensions for LLMs.
    - **Mechanism**: The six dimensions are: IU (Intent Understanding): inferring the speaker's communicative purpose behind using the buzzword, such as expressing emotions; CA (Conceptual Association): connecting the buzzword to related concepts, e.g., "wonnangfei" -> "work"; LS (Linguistic Structure): analyzing the grammatical role of the buzzword; SCI (Social Clues Interpretation): inferring social context such as tone and mood from UGC; WC (Word Context): leveraging surrounding text for disambiguation; PS (Pronunciation/Spelling): establishing associations between orthography/phonetics and semantics.
    - **Design Motivation**: Simulating the natural process of humans learning new words to let LLMs "observe" the usage of buzzwords from multiple angles, avoiding biased understanding from a single perspective.

3. **Definition Ensemble**:

    - **Function**: Fusing candidate definitions from multiple angles into the final output.
    - **Mechanism**: Prompting the LLM to "synthesize a final definition based on the following candidate definitions from different comprehension angles". BERTScore analysis shows high semantic diversity (weak correlation) among definitions generated by different aspects, indicating that they indeed provide complementary perspectives. The number of aspects is positively correlated with definition quality (with performance increasing from 1 -> 3 -> 5 -> 6 aspects).
    - **Design Motivation**: Leveraging the ensemble concept to mitigate bias from a single prompt and fully cover the semantic scope of buzzwords from multiple angles.

### Loss & Training

Ress is a training-free prompting method that does not involve parameter updates. Specifically, its core consists of a carefully designed three-stage prompt engineering: aspect initialization -> aspect-guided generation -> ensemble. It can be paired with any LLM backbone.

## Key Experimental Results

### Main Results

| Method | Backbone | BLEU | R-L | BScore | SA (1-5) | SC (1-5) |
|------|----------|------|-----|--------|----------|----------|
| DP (w/o UGC) | GPT-4o | 9.56 | 39.42 | 66.56 | 2.05 | 1.62 |
| DP | GPT-4o | 17.85 | 45.22 | 67.56 | 2.50 | 2.13 |
| CoT | GPT-4o | 18.33 | 44.49 | 67.46 | 2.60 | 2.30 |
| FOCUS (SOTA) | GPT-4o | 15.08 | 35.10 | 66.05 | 2.95 | 2.92 |
| **Ress** | GPT-4o | **16.52** | **36.42** | **66.74** | **3.04** | **3.06** |
| FOCUS | Qwen2-72B | 12.09 | 29.81 | 64.75 | 2.88 | 3.20 |
| **Ress** | Qwen2-72B | **15.74** | **35.63** | **66.41** | **2.97** | 3.09 |

### Ablation Study

| Number of Aspects | Performance Trend | Explanation |
|-------------|---------|------|
| 1 aspect | Lowest | A single dimension is not comprehensive enough |
| 3 aspects | Medium | Combining some dimensions yields improvement |
| 5 aspects | High | More dimensions bring more comprehensive understanding |
| 6 aspects (full Ress) | Highest | Number of aspects is positively correlated with quality |

### Key Findings

- **The Crucial Role of UGC**: Comparing DP vs DP(w/o UGC) shows that providing UGC examples significantly enhances definition quality (SA rises from 2.05 to 2.50). However, even with UGC, all methods still perform suboptimally (with maximum SA at 3.04/5).
- **Huge Gap of LLMs Between Seen/Unseen Buzzwords**: The models perform significantly better on buzzwords encountered in their training data than on unseen ones, exposing LLMs' over-reliance on parameterized memory.
- **Aspects are Indeed Complementary**: BERTScore semantic diversity analysis shows that definitions generated by different aspects are weakly correlated, indicating that each dimension provides a distinct comprehension perspective.
- Human evaluation (win rate) aligns with the ranking of automatic metrics, validating the reliability of the evaluation framework.
- Traditional LM methods (MASS-zh, SDefiner) score SA < 1.1, failing completely to handle internet buzzwords.

## Highlights & Insights

- **Cognitive-Science-Inspired NLP Method**: Translating child language acquisition theory into LLM prompt strategies is highly novel and theoretically grounded. This paradigm of formalizing cognitive processes into AI operations can be extended to other NLP tasks requiring deep comprehension.
- **Revealing the Fundamental Difficulty of LLMs in Understanding New Concepts**: Even GPT-4o only scores around 3 (out of 5) on SA for unseen buzzwords, revealing a remarkable gap in LLMs' ability to infer new concept meanings from context.
- **Value of the Cheer Dataset Itself**: With 1,127 buzzwords paired with 34,607 UGC items, Cheer has independent value and can be utilized for research in sociolinguistics, cultural linguistics, NLP, and other fields.

## Limitations & Future Work

- The overall performance remains relatively low (highest SA is 3.04/5), which indicates that understanding new words from UGC is inherently difficult and requires more fundamental methodological innovation rather than relying solely on prompt engineering.
- Only Chinese buzzwords are covered; the characteristics of buzzwords across different languages and cultural backgrounds may vary significantly.
- The inconsistent quality of UGC is a core bottleneck. Filtering high-quality UGC without knowing the meaning of the buzzwords presents a chicken-and-egg problem.
- The six aspects originate from child language acquisition theory, but their optimality for LLMs has not been fully verified, meaning that alternative dimensional categorizations better suited for LLMs might exist.
- Ress requires 7 LLM calls (6 aspect generations + 1 ensemble), incurring relatively high inference costs.

## Related Work & Insights

- **vs FOCUS (Prev. SOTA)**: FOCUS also focuses on context-aware definition generation but is not tailored to the specificity of buzzwords. Ress achieves improvements in both SA and SC through multi-dimensional guidance, which cultivates a more comprehensive understanding.
- **vs CoT**: CoT achieves moderate performance (SA 2.60) on buzzword tasks, indicating that simple reasoning chains are insufficient to comprehend highly abstract neologisms and that more structured cognitive guidance is required.
- **vs Traditional LM Methods** (MASS-zh/SDefiner): Scoring SA below 1.1, they struggle completely with out-of-distribution vocabulary like buzzwords, emphasizing the necessity of LLMs.
- This work is inherently related to few-shot concept learning—inferring the meanings of new concepts from a few usage examples.

## Rating

- Novelty: ⭐⭐⭐⭐ Approaching NLP tasks from a cognitive science perspective; the Cheer dataset fills a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive backbone comparison, human evaluation, aspect ablation, and semantic diversity analysis are thoroughly executed.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with a complete logical flow spanning from benchmark construction to methodology and analysis.
- Value: ⭐⭐⭐⭐ The long-term value of the dataset and benchmark exceeds that of the methodology itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Aligning Large Language Models with Implicit Preferences from User-Generated Content](pugc_align_implicit_pref_ugc.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm](do_language_models_understand_the_cognitive_tasks_given_to_them_investigations_w.md)
- [\[ACL 2025\] Do Language Models Understand Honorific Systems in Javanese?](do_language_models_understand_honorific_systems_in_javanese.md)

</div>

<!-- RELATED:END -->
