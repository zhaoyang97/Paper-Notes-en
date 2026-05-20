---
title: >-
  [Paper Note] FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition
description: >-
  [ICLR 2026][LLM Pretraining][Memorization] This work introduces the FictionalQA dataset and generation pipeline, which synthesizes webtext-style documents and QA pairs about fictional events to study both factual memoriz…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Memorization"
  - "Knowledge Acquisition"
  - "synthetic data"
  - "LLM Training Dynamics"
  - "Factual Memorization"
date: 2026-05-08
content_hash: d32f609acc53efdc
---

# FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition

**Conference**: ICLR 2026
**arXiv**: [2506.05639](https://arxiv.org/abs/2506.05639)  
**Code**: [https://github.com/jwkirchenbauer/fictionalqa](https://github.com/jwkirchenbauer/fictionalqa)  
**Area**: LLM Pre-training
**Keywords**: Memorization, Knowledge Acquisition, synthetic data, LLM Training Dynamics, Factual Memorization

## TL;DR
This work introduces the FictionalQA dataset and generation pipeline, which synthesizes webtext-style documents and QA pairs about fictional events to study both factual memorization and verbatim memorization in LLM training under controlled conditions. Key findings show that greater surface-form diversity facilitates knowledge acquisition, while concise structured lists are least conducive to generalization.

## Background & Motivation

**Background**: Two distinct memorization phenomena occur during LLM training: verbatim memorization (exact reproduction of training sequences) and factual memorization (generalization of facts encountered during training to novel tasks). Verbatim memorization has been extensively studied by Carlini et al., whereas understanding of factual memorization remains limited.

**Limitations of Prior Work**: Studying factual memorization is challenging because it is difficult to quantify how frequently a given fact appears in training data. Existing datasets are either overly templated (TOFU uses fill-in-the-blank), too small (New News contains only 75 articles), or involve science-fiction content that entangles fictional facts with real-world knowledge (Fictional Knowledge includes Star Trek-style topics).

**Key Challenge**: There is a need to simultaneously satisfy two conditions — surface-form realism and complete factual fabrication. Realism is necessary to simulate authentic training scenarios, while fabrication ensures that the introduced facts do not interact with genuine knowledge present in pre-training corpora, thereby enabling controlled experimentation.

**Goal**: To construct a "clean-room" synthetic dataset that allows researchers to isolate and study different forms of memorization under strictly controlled conditions, with particular focus on the training dynamics of factual memorization.

**Key Insight**: GPT-4o is used to generate hierarchically structured fictional data through a pipeline of seed events → detail sheets → multi-style documents → QA pairs, along with multiple train/validation split strategies designed to disentangle different experimental variables.

**Core Idea**: By leveraging controllable fictional synthetic data in a laboratory setting, the work demonstrates that factual memorization and verbatim memorization arise under different conditions, and that diverse surface forms promote knowledge acquisition while the most concise factual representations are least conducive to generalization.

## Method

### Overall Architecture

FictionalQA employs a four-stage hierarchical data generation pipeline: Seed Events → Fictsheets → Fictions → Fictional Q&A. All stages are generated using GPT-4o at varying temperatures. Two post-processing steps — QA annotation (filtering answerable questions) and MCQ reformatting — are applied after generation.

### Key Designs

1. **Hierarchical Data Generation Pipeline**

    - **Function**: Starting from brief seed events, the pipeline progressively expands content into complete fictional documents and QA pairs.
    - **Mechanism**: Seed events are short descriptions of fictional scenarios (temperature 1.0); Fictsheets expand seeds into structured outlines containing characters, locations, and specific details (temperature 0.7); Fictions expand Fictsheets into documents in five styles — news, social media, encyclopedia, corporate document, and blog (temperature 1.0); QA pairs are generated from documents as unambiguous question–answer pairs (temperature 0.1).
    - **Design Motivation**: The multi-level structure ensures diversity and controllability. By presenting the same facts through different surface forms, the pipeline enables systematic study of how surface-form diversity affects knowledge acquisition.

2. **QA Annotation: Infeasibility Filtering**

    - **Function**: Determines whether each QA pair can be answered without access to the fictional document (blind vs. informed evaluation).
    - **Mechanism**: The same GPT-4o model answers questions in both blind mode (question only) and informed mode (question + fictional document); only questions that cannot be answered in blind mode are retained.
    - **Design Motivation**: Ensures that observed QA performance gains during training genuinely reflect factual memorization from training data rather than the model's prior knowledge.

3. **Diverse Train/Validation Split Strategies**

    - **Function**: Three split strategies are designed to isolate different experimental variables.
    - (a) **Event Split**: All documents for 2/3 of seed events are used for training; the remaining 1/3 are held out entirely for validation. Validation content is completely disjoint from training content.
    - (b) **Doc Split**: For each seed event and each style, one document is reserved for validation. The validation set is in-distribution with respect to both content and style.
    - (c) **Style Split**: For each seed event, four styles are used for training and one is held out for validation. The validation set is content-matched but style out-of-distribution.
    - **Design Motivation**: Doc Split measures in-distribution content generalization; Event Split measures cross-event generalization; Style Split disentangles content memorization from style memorization.

4. **Training Experimental Design**

    - **Function**: Fine-tunes base checkpoints of Llama 3.1/3.2 and Gemma 1/2 with 5% fictional data mixed with 95% real webtext.
    - **Mechanism**: Training dynamics are monitored via training/validation loss, QA conditional answer loss, and MCQ accuracy. Fictional data is injected after 50 warm-up steps. TriviaQA is used to monitor whether real-world knowledge is degraded.
    - **Design Motivation**: The low 5% injection rate prevents verbatim memorization from dominating, placing the model within a "generalization window" in which the emergence of factual memorization can be observed.

### Loss & Training

Standard next-token prediction loss (cross-entropy) is used. The core contribution lies in the data split and injection strategies for studying memorization dynamics rather than in any novel training objective.

## Key Experimental Results

### Main Results

| Experimental Setting | Metric | Key Result |
|---|---|---|
| Doc Split vs. Event Split | Minimum validation loss | Doc Split generalizes better (lower validation loss) because all facts are partially covered |
| Fictsheets Split | Validation loss trend | Overfits almost immediately with no observable generalization period |
| MCQ accuracy across models | MCQ accuracy vs. training steps | Larger models achieve higher MCQ accuracy and improve more rapidly |
| MCQ across split types | Split type vs. MCQ | Doc Split and Style Split yield the best transfer; Fictsheets yield the worst |

### Ablation Study

| Configuration | QA Transfer | Notes |
|---|---|---|
| Doc Split (5 styles, same event) | Strongest | Diverse surface forms + complete factual coverage |
| Style Split (4 styles for training) | Strong | Style variation with complete facts |
| Event Split (different events) | Moderate | Incomplete factual coverage limits generalization |
| Fictsheets (structured lists) | Weakest | Most concise but least surface-form diversity |
| Base Webtext Only (control) | No effect | Confirms gains originate from fictional data |

### Key Findings
- **Verbatim memorization and factual memorization arise under different conditions**: Fictsheets are rapidly memorized verbatim (training loss approaches 0), yet exhibit almost no factual memorization (MCQ accuracy improvement).
- **Surface-form diversity promotes knowledge acquisition**: Training on multi-style documents yields better QA generalization than training on structured lists — a counter-intuitive finding, as humans might consider structured lists easier for knowledge extraction.
- **Knowledge acquisition exhibits "leakage"**: Even when certain facts are entirely absent from the training set (Event Split validation set), MCQ accuracy for the corresponding questions still improves, suggesting the model may rely on distributional patterns rather than atomic facts.
- **Larger models acquire knowledge faster**: The 8B model achieves higher MCQ accuracy improvements more rapidly than the 1B model.

## Highlights & Insights
- **The counter-intuitive finding that "conciseness does not imply effectiveness" is particularly illuminating**: Structured fact lists (Fictsheets) lead to rapid overfitting but the worst knowledge generalization, whereas diverse natural-language documents promote factual memorization. This suggests that knowledge acquisition in LLMs depends on distributional patterns rather than explicit factual encoding.
- **The dataset is designed as a "living asset"**: The pipeline can regenerate new datasets, and other researchers can reuse and modify it. This methodological contribution has greater long-term value than a one-time static dataset.
- **Rigorous blind/informed annotation and TriviaQA control experiments** ensure the credibility of experimental conclusions and serve as an exemplary model of experimental design.
- The "leakage" phenomenon in factual memorization suggests that the knowledge boundaries of LLMs may be more diffuse than previously expected, with direct implications for machine unlearning research.

## Limitations & Future Work
- Unintended content overlap may exist across fictional documents from different seed events, meaning that the leakage effect may partly originate from the data itself rather than model behavior.
- Data is generated exclusively using GPT-4o; biases inherent to the generation model may limit the generalizability of the conclusions.
- Experiments are conducted only with fine-tuned models at scales below 8B parameters; behavior under large-scale pre-training may differ.
- The 5% injection rate is fixed, and the effect of varying injection rates on different memorization forms is not systematically studied.
- QA deduplication is acknowledged to be incomplete, with a substantial number of duplicate questions reported.

## Related Work & Insights
- **vs. TOFU**: TOFU is designed for unlearning and uses fill-in-the-blank templates, lacking surface-form diversity and not releasing source documents. FictionalQA provides both documents and QA pairs, and its multi-style design more closely resembles real pre-training data.
- **vs. Allen-Zhu & Li synthetic biographies**: Those biographies are relatively templated; FictionalQA's webtext style is more natural and varied, making it better suited for studying the effect of surface-form diversity.
- **vs. New News (Park et al. 2025)**: Contains only 75 manually curated articles and 375 questions; FictionalQA is larger in scale and fully automated.
- **vs. Fictional Knowledge (Chang et al. 2024)**: Contains science-fiction content (Star Trek) that may entangle with real-world knowledge; FictionalQA deliberately avoids such subject matter.

## Rating
- Novelty: ⭐⭐⭐⭐ The hierarchical generation pipeline and multi-split strategy are cleverly designed, though the core idea of using fictional data to study memorization is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic experiments across multiple models and splits, but large-scale pre-training experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured with detailed motivation for experimental design choices and rigorous variable control.
- Value: ⭐⭐⭐⭐ Academically valuable for understanding LLM memorization mechanisms; the dataset as a reusable asset has long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding](../../ACL2026/llm_pretraining/commonsense_knowledge_with_negation_a_resource_to_enhance_negation_understanding.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [\[ICLR 2026\] MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](moma_a_modular_deep_learning_framework_for_material_property_prediction.md)

</div>

<!-- RELATED:END -->
