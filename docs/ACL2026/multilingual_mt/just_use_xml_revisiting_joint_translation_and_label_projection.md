---
title: >-
  [Paper Note] Just Use XML: Revisiting Joint Translation and Label Projection
description: >-
  [ACL 2026][Multilingual & Machine Translation][Label Projection] LabelPigeon is proposed, a joint translation and label projection method based on XML tags. By fine-tuning the NLLB-200 translation model on high-quality X…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Label Projection"
  - "XML tagging"
  - "Joint Translation"
  - "Cross-lingual Transfer"
  - "NER"
date: 2026-05-08
content_hash: ca805caf1b358b20
---

# Just Use XML: Revisiting Joint Translation and Label Projection

**Conference**: ACL 2026 Findings  
**arXiv**: [2603.12021](https://arxiv.org/abs/2603.12021)  
**Code**: [https://github.com/thennal10/LabelPigeon](https://github.com/thennal10/LabelPigeon)  
**Area**: Multilingual Translation / Cross-lingual Transfer  
**Keywords**: Label Projection, XML tagging, Joint Translation, Cross-lingual Transfer, NER

## TL;DR

LabelPigeon is proposed, a joint translation and label projection method based on XML tags. By fine-tuning the NLLB-200 translation model on high-quality XML-tagged parallel corpora, it surpasses all baselines across 11 languages and actively improves translation quality, achieving up to a +40.2 F1 improvement in downstream cross-lingual NER tasks.

## Background & Motivation

**Background**: Many NLP tasks depend on span-level labels (e.g., entities in NER, arguments in event extraction). A common practice to extend these tasks to low-resource languages is to translate the training data and then perform label projection. Traditionally, label projection uses word alignment models (e.g., Awesome-align) as a standalone step after translation.

**Limitations of Prior Work**: EasyProject (Chen et al., 2023) attempted joint translation and label projection by inserting square brackets around spans before translation but reported a decrease in translation quality. Subsequent works (T-Projection, CLaP, Codec) abandoned joint methods in favor of complex multi-stage pipelines: separating translation and label projection, introducing LLM context translation or constrained decoding, which significantly increases computational and engineering overhead.

**Key Challenge**: The domain consensus is that "inserting tags inherently damages translation quality," leading mainstream methods to rely on complex multi-stage pipelines. Does this assumption truly hold, or is it merely a result of improper training data and tag selection?

**Goal**: Re-verify whether joint translation and label projection necessarily reduce translation quality and propose a simple, effective alternative.

**Key Insight**: The authors start from three observations: (1) XML tags offer natural advantages over square brackets by providing direct correspondence between source and target labels and elegantly handling nesting and overlap; (2) High-quality XML-tagged parallel corpora already exist in the structured document translation domain (e.g., Salesforce Localization XML MT dataset); (3) Label-aware translation can guide the model to prioritize maintaining the continuity and integrity of labeled spans, avoiding pronoun omission and ambiguous assignment during translation.

**Core Idea**: Use XML tags instead of square brackets for marking, leverage existing high-quality XML parallel corpora (rather than synthetic data) to fine-tune the translation model, and complete translation and label projection simultaneously in one forward pass without multi-stage pipelines.

## Method

### Overall Architecture

The LabelPigeon workflow is highly concise: (1) Mark all spans in the text to be translated with alphabetical XML tags (`<a>`, `<b>`, etc.); (2) Translate using the fine-tuned NLLB-200 3.3B model; (3) Extract tags from the translation using a standard XML parser. Inference requires only a single model forward pass with no additional computational overhead.

### Key Designs

1. **XML Tags Instead of Square Brackets**:
    - **Function**: Provides precise correspondence between annotated spans in the source and target languages.
    - **Mechanism**: XML tags naturally support named attributes (e.g., `<a>...</a>` corresponds uniquely between source and target), handle nested and overlapping spans (e.g., `<a><b>...</b>...</a>`), and can carry semantic information (e.g., `<PER>`). Square brackets fail to provide direct correspondence; EasyProject requires additional fuzzy string matching to establish mappings, which is time-consuming and prone to errors in nested scenarios.
    - **Design Motivation**: Structured document translation already possesses rich XML tagging practices and data; selecting XML allows for the direct utilization of these resources.

2. **Fine-tuning on High-Quality Real Data**:
    - **Function**: Trains the model to maintain tag structure during translation using real XML-tagged parallel corpora.
    - **Mechanism**: Utilizes the Salesforce Localization XML MT dataset, containing approximately 100k pairs of XML-tagged parallel sentences between English and seven languages. Original UI/style tags are replaced with generic alphabetical tags (`<a>`, `<b>`, etc.), resulting in about 25k samples per language pair after filtering tag-less samples. Based on ablation studies, the model is trained on three high-resource language pairs (En-De, En-Ru, En-Zh), totaling about 150k training samples (including bidirectional translation). One epoch is completed in 5.5 hours on a single A100.
    - **Design Motivation**: Previous methods (EasyProject) used synthetically generated data, whereas this work uses high-quality real data to avoid noise and distribution shifts associated with synthetic data.

3. **Advantages of Label-Aware Translation**:
    - **Function**: Theoretically and empirically demonstrates that joint translation + label projection is superior to separated methods.
    - **Mechanism**: Three minimal examples illustrate pitfalls of separated translation: (a) Translation may split labeled spans across different sentence positions (e.g., Malayalam); (b) The target language may omit words corresponding to the labels (e.g., pro-drop in Japanese); (c) Translation may produce label attribution ambiguity (e.g., French). Label-aware translation guides the model toward choices that keep spans continuous and intact.
    - **Design Motivation**: Separated methods assume tags can be reliably reconstructed after translation, but linguistic shifts during translation often invalidate this assumption.

### Loss & Training

Standard seq2seq translation training loss is utilized. Key training strategy choices include: (1) Training only on three high-resource language pairs to avoid catastrophic forgetting; (2) Training for one full epoch (9091 steps, effective batch size 16); (3) Replacing tags with generic alphabetical tags to allow the model to generalize to any label type.

## Key Experimental Results

### Main Results

Direct Label Projection Evaluation (XQuAD + MLQA, average of 11 languages):

| Method | COMET Quality | Label Match F1 |
|--------|---------------|----------------|
| Awesome-align | 82.3 (Baseline) | 50.6% |
| Gemma 3 27B | 69.6 (-12.7) | 78.1% |
| EasyProject | 80.8 (-1.5) | 77.7% |
| **LabelPigeon** | **82.4 (+0.1)** | **79.2%** |

Downstream NER Task (UNER, average F1 across 16 datasets):

| Method | Average F1 | Max Gain |
|--------|------------|----------|
| EasyProject | 62.5% | - |
| **LabelPigeon** | **76.7%** | Tagalog +40.2 F1 |

### Ablation Study

| Config | BLEU (No Tags) | BLEU (Complex Tags) | Projection Rate |
|--------|---------------|-------------------|-----------------|
| NLLB Baseline | 17.4 | - | - |
| EasyProject | 17.7 | 14.9 | 47.7% |
| LabelPigeon | 17.6 | 15.5 | 69.3% |
| Non-tagged FT (NF) | 17.9 | - | - |

### Key Findings
- LabelPigeon is the only method where translation quality increases after tag insertion, with COMET improving from 82.3 to 82.4.
- The improvement in translation quality is attributed to the additional fine-tuning itself (the non-tagged fine-tuned model also showed improvement), showing positive transfer even to untrained languages.
- EasyProject reduces translation quality across all tag configurations, while LabelPigeon maintains BLEU parity with the baseline in single-tag scenarios.
- The largest improvements in downstream NER occur in low-resource languages: Cebuano +30.7, Tagalog +40.2, Swedish +22.
- In coreference resolution tasks, EasyProject fails almost completely in 11/16 languages (F1 < 1.0), whereas LabelPigeon only scores 0 in two historical languages.
- LabelPigeon generalizes to tag counts unseen during training: it was trained on a maximum of 6 tags but tested on an average of 9 tags in XQuAD (up to 24).

## Highlights & Insights
- **Challenging Domain Consensus**: Refutes the widespread assumption that joint translation and label projection necessarily degrade translation quality. Ample experimentation proves the problem lies with tag choice and training data, not the method itself.
- **Victory of Simplicity**: Compared to complex multi-stage pipelines (e.g., T-Projection requiring additional LLMs, Codec requiring constrained decoding), LabelPigeon achieves optimal results in both label projection and translation quality with just one fine-tuning stage and one forward pass. It is a paradigm of "less is more."
- **Theoretical Insight into Label-Aware Translation**: Elegantly demonstrates why translation and label projection should be joint rather than separate—translation choices affect label integrity, and tag constraints can guide better translation.

## Limitations & Future Work
- Direct label projection evaluation was restricted to QA datasets (XQuAD, MLQA) with simple label types.
- Synthetic tag insertion on Flores-200 may not perfectly reflect the distribution of real annotated tags.
- Verified only on NLLB-200 3.3B; the performance of larger translation models or LLM-based translation is unknown.
- Overall performance on coreference resolution tasks remains low, and translation quality still declines in scenarios with nested or extremely frequent tags.
- Training data was limited to English and three high-resource language pairs; expansion to more language pairs might yield further improvements.

## Related Work & Insights
- **vs EasyProject**: EasyProject uses square brackets and synthetic data, leading to quality drops and reliance on fuzzy matching. LabelPigeon uses XML and real data, improving quality and providing precise mapping.
- **vs T-Projection / CLaP**: These require additional LLMs for projection or context translation, incurring high computational costs. LabelPigeon has zero additional inference cost.
- **vs Awesome-align**: Word alignment methods achieve only 50.6% Label Match F1, far below LabelPigeon's 79.2%.

## Rating
- Novelty: ⭐⭐⭐⭐ The primary contribution is challenging the domain consensus and proposing a simpler, more effective alternative. The combination of XML and real data is simple yet surprisingly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation including direct assessment, translation quality, and three downstream tasks covering 203 languages and downstream experiments for 27 languages.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical argumentation progressing from theory to experiment, with intuitive minimal examples.
- Value: ⭐⭐⭐⭐⭐ Provides a minimalist yet efficient label projection solution for cross-lingual NLP that is directly applicable to production systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation](syntax_as_a_rosetta_stone_universal_dependencies_for_in-context_coptic_translati.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)

</div>

<!-- RELATED:END -->
