---
title: >-
  [Paper Note] Cognitive Framework for Detecting AI-Generated Fiction
description: >-
  [ACL 2025][AIGC Detection][AI Text Detection] This paper proposes an AI-generated novel/fiction detection framework based on cognitive linguistic features. By modeling cognitive patterns in human creative writing (such as narrative rhythm, emotional arc, and metaphor density), the framework distinguishes between human-written and AI-generated fictional texts, significantly outperforming existing detection methods in long-text scenarios.
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "AI Text Detection"
  - "Fictional Text"
  - "Cognitive Features"
  - "Machine-Generated Text Detection"
  - "Writing Style Analysis"
date: 2026-05-08
content_hash: cadcdbc8afc8d2dc
---

# Cognitive Framework for Detecting AI-Generated Fiction

**Conference**: ACL 2025  
**Area**: AIGC Detection  
**Keywords**: AI Text Detection, Fictional Text, Cognitive Features, Machine-Generated Text Detection, Writing Style Analysis

## TL;DR
This paper proposes an AI-generated novel/fiction detection framework based on cognitive linguistic features. By modeling cognitive patterns in human creative writing (such as narrative rhythm, emotional arc, and metaphor density), the framework distinguishes between human-written and AI-generated fictional texts, significantly outperforming existing detection methods in long-text scenarios.

## Background & Motivation

**Background**: With the ubiquity of large language models such as ChatGPT and Claude, AI-generated text is becoming increasingly difficult to distinguish from human writing. AI text detection (AIGC detection) has become an important research direction. Existing methods mainly include statistical feature methods (such as perplexity, watermarking), classifier-based methods (training models like BERT to distinguish human/AI text), and zero-shot methods (such as DetectGPT).

**Limitations of Prior Work**: Existing AI text detection methods primarily target short texts (such as news, paper abstracts, Q&A) and perform poorly on long-form fictional works (novels, stories, screenplays). There are three main reasons: (1) The linguistic style of fictional text varies greatly, making statistical features (such as perplexity distribution) unstable; (2) Extensive use of rhetorical devices like metaphor, irony, and hyperbole in fiction renders surface-level statistical features ineffective; (3) Long-text narrative structural features (such as emotional transitions across chapters and consistency in character development) are not modeled by existing methods.

**Key Challenge**: AI can already generate grammatically correct and semantically fluent fictional text, rendering traditional "fluency anomaly detection" approaches ineffective. It is necessary to identify deeper discrepancies between human creative writing and AI generation.

**Goal**: From the perspective of cognitive linguistics, this paper aims to identify cognitive features in the human creative writing process, formalize these features into computable metrics, and establish an AI detection framework specifically for fictional text.

**Key Insight**: Cognitive linguistics research shows that human creative writing is constrained by cognitive resources—cognitive processes such as attention allocation, working memory capacity, and emotional regulation leave a unique "cognitive fingerprint." AI generation is free from these constraints and thus lacks these cognitive fingerprints.

**Core Idea**: Extracts cognitive linguistic features from fictional texts (such as narrative rhythm fluctuations, irregularity of emotional arcs, and uneven distribution of metaphors). These features reflect the unique imprint of human cognitive processes, which are difficult for AI to mimic.

## Method

### Overall Architecture
The framework is divided into three layers: (1) Surface-level feature layer—traditional text statistical features (perplexity, lexical diversity, etc.); (2) Cognitive feature layer—high-level features extracted from a cognitive linguistic perspective (narrative rhythm, emotional arcs, metaphor patterns, etc.); (3) Fusion classification layer—fuses both layers of features to train a binary classifier. The input is the complete fictional text (short stories, tales, etc.), and the output is a human/AI binary classification.

### Key Designs

1. **Narrative Rhythm Analysis Module (Narrative Rhythm Analysis)**:

    - **Function**: Captures patterns of change in narrative rhythm within the text, reflecting the writer's narrative strategy.
    - **Mechanism**: Splits the text by paragraphs to calculate the "information density" of each paragraph (rate of new entity introduction, event frequency) and "rhythm variation" (the difference in information density between adjacent paragraphs). Human works typically exhibit irregular fluctuations of "tension and relaxation" (e.g., tense paragraphs followed by soothing ones) in narrative rhythm, whereas the rhythm of AI-generated text is often more uniform or displays regular patterns. After extracting the rhythm sequence, its entropy, autocorrelation coefficients, and spectral features are computed as classification features.
    - **Design Motivation**: Human writing is constrained by cognitive resources and cannot sustain high-density narrative for a long time, thus naturally forming rhythm fluctuations; AI has no such constraints, and its rhythm patterns reflect the statistical average of its training data.

2. **Emotional Arc Modeling Module (Emotional Arc Modeling)**:

    - **Function**: Tracks the trajectory of emotional polarity changes in the text.
    - **Mechanism**: Uses a sentence-level sentiment analysis model to score the emotional polarity of each sentence in the text to plot an emotional arc. The emotional arcs of human fictional works typically conform to Kurt Vonnegut's narrative shape theory (e.g., "Cinderella" arc, "Tragedy" arc), with clear emotional ups and downs as well as climaxes; the emotional arcs of AI-generated texts are usually flatter or show unnatural emotional leaps. Features of the arc shape (peak positions, rate of change, match with classic arc templates) are extracted for classification.
    - **Design Motivation**: Emotional arcs reflect the author's intentional design of the reader's emotional experience, and this global narrative layout is currently difficult for AI to mimic systematically.

3. **Metaphor and Rhetoric Distribution Analysis (Metaphor and Rhetoric Distribution)**:

    - **Function**: Analyzes the usage patterns and distribution characteristics of rhetorical devices in the text.
    - **Mechanism**: Uses metaphor detection models and rhetorical annotation tools to extract the positions and types of rhetorical devices such as metaphors, similes, and personification in the text. It calculates rhetorical density (number of rhetorical devices per thousand words), rhetorical diversity (proportions of different rhetorical types), and rhetorical clustering (evenness of the distribution of rhetorical devices across the text). In human works, rhetorical devices usually concentrate in climax paragraphs (corresponding to narrative needs), whereas the distribution in AI-generated text is more uniform or random.
    - **Design Motivation**: The use of rhetorical devices reflects deep linguistic creativity and cultural knowledge, and the correlation between their distribution patterns and the narrative structure is a unique identifier of human writing.

### Loss & Training
Standard binary cross-entropy loss is used. Cognitive and surface-level features are fused using learnable weights, with the importance of each feature category being learned automatically during training.

## Key Experimental Results

### Main Results

| Method | Short Story F1↑ | Web Story F1↑ | Mixed Text F1↑ | Average F1↑ |
|------|-----------|-----------|-----------|--------|
| DetectGPT | 68.2 | 62.5 | 58.3 | 63.0 |
| GPTZero | 72.8 | 67.1 | 63.5 | 67.8 |
| RoBERTa-classifier | 78.5 | 73.2 | 69.8 | 73.8 |
| Fast-DetectGPT | 71.3 | 65.8 | 61.2 | 66.1 |
| **Ours** | **86.3** | **82.7** | **78.5** | **82.5** |

### Ablation Study

| Configuration | Short Story F1↑ | Average F1↑ | Description |
|------|-----------|--------|------|
| Full Model | 86.3 | 82.5 | All feature layers |
| Surface Features Only | 78.0 | 73.2 | Degrades to traditional methods without cognitive features |
| Cognitive Features Only | 83.5 | 80.1 | Cognitive features alone are highly effective |
| w/o Narrative Rhythm | 83.2 | 79.5 | Rhythm analysis contributes +3.0 |
| w/o Emotional Arc | 84.1 | 80.8 | Emotional arc contributes +1.7 |
| w/o Metaphor Distribution | 84.8 | 81.3 | Metaphor analysis contributes +1.2 |

### Key Findings
- Using cognitive features alone (F1=80.1) significantly outperforms traditional surface-level features (F1=73.2), validating the effectiveness of the cognitive perspective.
- Narrative rhythm analysis is the most contributing cognitive feature (+3.0 F1), indicating that AI's shortcomings in global narrative structures are the most pronounced.
- In mixed text scenarios (human introduction + AI continuation), the advantage of this method is more prominent, as cognitive features can detect consistency changes across the entire text.
- The proposed method achieves high detection accuracy and good generalization across texts generated by different AI models.

## Highlights & Insights
- Approaching AI text detection from cognitive linguistics is a unique perspective that goes beyond traditional statistical analysis and trained classifier paradigms.
- Global structural features such as narrative rhythm and emotional arcs are difficult for AI to mimic through simple fine-tuning, offering long-term robustness.
- The framework not only detects AI text but also charts the cognitive discrepancies between AI and human writing, rendering analytical value.

## Limitations & Future Work
- The extraction of cognitive features depends on preprocessing models such as sentiment analysis and metaphor detection, which introduces sources of error propagation.
- The applicability to non-English fictional texts remains to be validated, as narrative traditions of different languages may lead to different cognitive fingerprints.
- As AI writing capability scales up, the differentiability of cognitive features may gradually diminish.
- The work can be extended to other genres of fictional text, such as poetry and screenplays.

## Related Work & Insights
- **vs DetectGPT (Mitchell et al., 2023)**: DetectGPT is based on perplexity perturbation detection and performs poorly on fictional text; our cognitive features are more stable on fictional narratives.
- **vs Ghostbuster (Verma et al., 2024)**: Ghostbuster uses multi-model perplexity features, performing well on short texts but lacking global structural analysis of long texts.
- **vs Binoculars (Hans et al., 2024)**: Binoculars uses bi-model cross-entropy comparison; our method uses cognitive features, making them complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Approaching AI detection from a cognitive linguistics perspective is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ The evaluation across multiple scenarios is complete, but the dataset size and coverage could be larger.
- Writing Quality: ⭐⭐⭐⭐ Interdisciplinary content is clearly presented, and definitions of cognitive features are rigorous.
- Value: ⭐⭐⭐⭐ Highly valuable in opening up a brand-new research paradigm for AI text detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Empirical Study on Detecting AI-Generated Text in Financial Reports](an_empirical_study_on_detecting_ai-generated_text_in_financial_reports.md)
- [\[ACL 2025\] Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)
- [\[ACL 2025\] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)
- [\[ACL 2025\] Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](who_writes_what_ai_detection.md)
- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)

</div>

<!-- RELATED:END -->
