---
title: >-
  [Paper Note] Just Use XML: Revisiting Joint Translation and Label Projection
description: >-
  [ACL 2026][Multilingual & Machine Translation][Label Projection] This paper proposes LabelPigeon, a joint translation and label projection method based on XML markup. By fine-tuning the NLLB-200 translation model on high…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Label Projection"
  - "XML Markup"
  - "Joint Translation"
  - "Cross-lingual Transfer"
  - "NER"
date: 2026-05-08
content_hash: e92a47d620b586ba
---

# Just Use XML: Revisiting Joint Translation and Label Projection

**Conference**: ACL 2026
**arXiv**: [2603.12021](https://arxiv.org/abs/2603.12021)  
**Code**: [https://github.com/thennal10/LabelPigeon](https://github.com/thennal10/LabelPigeon)  
**Area**: Multilingual Translation / Cross-lingual Transfer
**Keywords**: Label Projection, XML Markup, Joint Translation, Cross-lingual Transfer, NER

## TL;DR

This paper proposes LabelPigeon, a joint translation and label projection method based on XML markup. By fine-tuning the NLLB-200 translation model on high-quality XML-annotated parallel corpora, LabelPigeon surpasses all baselines across 11 languages while actively improving translation quality, achieving gains of up to +40.2 F1 on downstream cross-lingual NER tasks.

## Background & Motivation

**Background**: Many NLP tasks rely on span-level annotations (e.g., entities in NER, arguments in event extraction). A common approach for extending these tasks to low-resource languages is to machine-translate training data and then perform label projection. Label projection has traditionally been treated as a separate post-translation step using word alignment models such as Awesome-align.

**Limitations of Prior Work**: Chen et al. (2023)'s EasyProject attempted joint translation and label projection by inserting square brackets around spans prior to translation, but reported degraded translation quality. Subsequent work (T-Projection, CLaP, Codec) consequently abandoned the joint approach in favor of complex multi-stage pipelines that decouple translation and label projection, introducing LLM-based contextual translation or constrained decoding at the cost of substantially increased computational and engineering overhead.

**Key Challenge**: The prevailing assumption in the field is that inserting markup inherently degrades translation quality, leading mainstream methods toward complex multi-stage pipelines. However, it remains an open question whether this assumption genuinely holds, or whether the degradation stems from suboptimal training data and markup choices.

**Goal**: To re-examine whether joint translation and label projection necessarily degrades translation quality, and to propose a simple yet effective alternative.

**Key Insight**: The authors build on three observations: (1) XML tags hold a natural advantage over square brackets—they provide direct correspondence between source and target annotations and gracefully handle nesting and overlap; (2) the structured document translation domain already offers large quantities of high-quality XML-annotated parallel corpora (e.g., the Salesforce Localization XML MT dataset); (3) markup-aware translation can guide the model to prioritize span continuity and completeness, avoiding pronoun dropping and ambiguous attribution during translation.

**Core Idea**: Replace square brackets with XML tags as the markup choice; fine-tune a translation model on existing high-quality XML parallel data (rather than synthetic data); and perform translation and label projection in a single forward pass, eliminating the need for a multi-stage pipeline.

## Method

### Overall Architecture

The LabelPigeon pipeline is straightforward: (1) annotate all spans in the source text using alphabetically ordered XML tags (`<a>`, `<b>`, etc.); (2) translate with a fine-tuned NLLB-200 3.3B model; (3) extract projected labels from the translation using a standard XML parser. The entire inference requires only a single model forward pass with no additional computational overhead.

### Key Designs

1. **XML Tags as a Replacement for Square Brackets**:

    - *Function*: Provide precise correspondence between annotated spans in the source and target languages.
    - *Mechanism*: XML tags natively support named attributes (e.g., `<a>...</a>` maps one-to-one between source and translation), handle nested and overlapping spans (e.g., `<a><b>...</b>...</a>`), and can carry semantic information (e.g., `<PER>` for person names). Square brackets cannot encode correspondence—EasyProject requires additional fuzzy string matching to establish mappings, which is time-consuming and error-prone for nested spans.
    - *Design Motivation*: The structured document translation community has extensive practice and data for XML-annotated translation, making XML a natural choice for directly leveraging these resources.

2. **Fine-tuning on High-Quality Real Data**:

    - *Function*: Train the model on authentic XML-annotated parallel corpora so that it learns to preserve tag structure during translation.
    - *Mechanism*: The Salesforce Localization XML MT dataset is used, comprising approximately 100K XML-annotated parallel sentence pairs between English and seven languages. Original UI/style tags are replaced with generic alphabetically ordered tags (`<a>`, `<b>`, etc.), and untagged samples are removed, yielding approximately 25K samples per language pair. Ablation experiments motivate training on three high-resource language pairs—English–German, English–Russian, and English–Chinese—totaling approximately 150K training samples including both translation directions. Fine-tuning runs for one epoch (5.5 hours on a single A100).
    - *Design Motivation*: Prior methods (e.g., EasyProject) relied on synthetically generated data; this work leverages existing high-quality real data to avoid the noise and distributional shift associated with synthetic data.

3. **Theoretical and Empirical Justification for Markup-Aware Translation**:

    - *Function*: Demonstrate, both theoretically and empirically, that joint translation and label projection outperforms decoupled approaches.
    - *Mechanism*: Three minimal examples illustrate the failure modes of decoupled translation: (a) translation may scatter a labeled span across different positions in the sentence (e.g., Malayalam); (b) the target language may drop the word corresponding to a labeled span (e.g., pronoun dropping in Japanese); (c) translation may introduce label attribution ambiguity (e.g., French). Markup-aware translation guides the model toward translations that preserve span continuity and completeness.
    - *Design Motivation*: Decoupled methods assume that label mappings can be reliably reconstructed after translation, but linguistic transformations during translation frequently violate this assumption.

### Loss & Training

Standard seq2seq translation loss is used. Key training decisions include: (1) training only on three high-resource language pairs to mitigate catastrophic forgetting; (2) training for exactly one full epoch (9,091 steps, effective batch size 16); (3) replacing original tags with generic alphabetically ordered labels to enable generalization to arbitrary label types.

## Key Experimental Results

### Main Results

Direct label projection evaluation (XQuAD + MLQA, average over 11 languages):

| Method | COMET Translation Quality | Label Match F1 |
|--------|--------------------------|----------------|
| Awesome-align | 82.3 (baseline) | 50.6% |
| Gemma 3 27B | 69.6 (−12.7) | 78.1% |
| EasyProject | 80.8 (−1.5) | 77.7% |
| **LabelPigeon** | **82.4 (+0.1)** | **79.2%** |

Downstream NER evaluation (UNER, average F1 over 16 datasets):

| Method | Average F1 | Maximum Gain |
|--------|-----------|--------------|
| EasyProject | 62.5% | — |
| **LabelPigeon** | **76.7%** | Tagalog +40.2 F1 |

### Ablation Study

| Configuration | BLEU (no markup) | BLEU (complex markup) | Projection Rate |
|---------------|-----------------|----------------------|----------------|
| NLLB baseline | 17.4 | — | — |
| EasyProject | 17.7 | 14.9 | 47.7% |
| LabelPigeon | 17.6 | 15.5 | 69.3% |
| No-markup fine-tuning (NF) | 17.9 | — | — |

### Key Findings

- LabelPigeon is the only method for which translation quality improves rather than degrades upon markup insertion—COMET increases from 82.3 to 82.4.
- The translation quality gain is attributable to the fine-tuning itself rather than to the markup (the no-markup fine-tuned model also improves), with positive transfer observed even on languages not seen during training.
- EasyProject degrades translation quality under all markup configurations, whereas LabelPigeon maintains BLEU comparable to the baseline in the single-tag scenario.
- Low-resource languages benefit most in downstream NER: Cebuano +30.7, Tagalog +40.2, Swedish +22.
- On coreference resolution, EasyProject achieves F1 < 1.0 on 11 of 16 languages (near-complete failure), while LabelPigeon scores 0 only on 2 historical languages.
- LabelPigeon generalizes to tag counts unseen during training: training uses at most 6 tags, while XQuAD test instances average 9 tags (up to 24).

## Highlights & Insights

- **Challenging the Field's Consensus**: This work overturns the widely held assumption that joint translation and label projection necessarily degrades translation quality. Rigorous experiments demonstrate that the problem lies in markup choice and training data rather than in the approach itself—a commendable willingness to revisit established conclusions.
- **The Power of Simplicity**: Compared with complex multi-stage pipelines (T-Projection requires an additional LLM; Codec requires constrained decoding), LabelPigeon requires only a single fine-tuning step and a single forward pass at inference, yet achieves state-of-the-art performance on both label projection and translation quality—a compelling demonstration of the "less is more" principle.
- **Theoretical Insight into Markup-Aware Translation**: Minimal examples elegantly motivate why translation and label projection should be performed jointly rather than separately: translation decisions affect span integrity, and conversely, span constraints can guide the model toward more suitable translations.

## Limitations & Future Work

- Direct label projection evaluation uses only QA datasets (XQuAD, MLQA) with a limited variety of label types.
- Synthetic markup insertion on Flores-200 may not fully reflect the label distribution found in real annotations.
- Experiments are conducted solely with NLLB-200 3.3B; the effectiveness of larger translation models or LLM-based translation remains unknown.
- Overall performance on coreference resolution remains low, and translation quality still degrades in nested and high-tag-frequency scenarios.
- Training data is limited to three high-resource English-to-X language pairs; extending to additional language pairs may yield further improvements.

## Related Work & Insights

- **vs. EasyProject**: Uses square brackets and synthetic data, degrading translation quality with ambiguous label correspondence via fuzzy string matching. LabelPigeon uses XML tags and real data, improving translation quality with precise label correspondence.
- **vs. T-Projection / CLaP**: Require an additional LLM for label projection or contextual translation at high computational cost. LabelPigeon incurs zero additional inference overhead.
- **vs. Awesome-align**: Word alignment achieves only 50.6% Label Match F1, far below LabelPigeon's 79.2%.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The key contribution lies in challenging the field's consensus and proposing a simpler, more effective alternative; the combination of XML markup and real data appears straightforward yet yields remarkable results.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers direct evaluation, translation quality, and three downstream tasks, with experiments spanning 203 languages and 27-language downstream benchmarks—exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The argumentation is logically clear, progressing systematically from theory to experiments; the minimal examples are highly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Provides a minimalist yet highly effective label projection solution for cross-lingual NLP that is directly applicable to real-world systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation](syntax_as_a_rosetta_stone_universal_dependencies_for_in-context_coptic_translati.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ACL 2026\] Lost in Translation: Do LVLM Judges Generalize Across Languages?](lost_in_translation_do_lvlm_judges_generalize_across_languages.md)
- [\[ACL 2026\] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation](beyond_literal_mapping_benchmarking_and_improving_non-literal_translation_evalua.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)

</div>

<!-- RELATED:END -->
