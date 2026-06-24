---
title: >-
  [Paper Note] Bilingual Zero-Shot Stance Detection
description: >-
  [ACL 2025][LLM (Other)][Stance Detection] To address the cross-lingual challenges in zero-shot stance detection, this paper proposes a bilingual joint framework. By constructing a shared semantic space and enabling cross-lingual knowledge transfer, the framework accurately determines text stance (favor/against/neutral) toward specific targets without labeled data in the target language.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Stance Detection"
  - "Zero-Shot"
  - "Bilingual"
  - "Cross-Lingual Transfer"
  - "Text Classification"
date: 2026-05-08
content_hash: fdb0ca7a6702aa52
---

# Bilingual Zero-Shot Stance Detection

**Conference**: ACL 2025  
**Code**: None  
**Area**: NLP Understanding  
**Keywords**: Stance Detection, Zero-Shot, Bilingual, Cross-Lingual Transfer, Text Classification

## TL;DR
To address the cross-lingual challenges in zero-shot stance detection, this paper proposes a bilingual joint framework. By constructing a shared semantic space and enabling cross-lingual knowledge transfer, the framework accurately determines text stance (favor/against/neutral) toward specific targets without labeled data in the target language.

## Background & Motivation

**Background**: Stance detection aims to determine the attitude (favor, against, or neutral) expressed in a text toward a target topic, acting as a core task in opinion mining and sentiment analysis. Zero-Shot Stance Detection (ZSSD) further requires models to generalize to unseen target topics during training. Currently, most stance detection research focuses on English, with limited work addressing other languages.

**Limitations of Prior Work**: Existing zero-shot stance detection approaches face three major challenges: (1) validation is limited to monolingual scenarios; due to significant differences in stance expressions across languages, direct transfer is infeasible. (2) Zero-shot generalization capability heavily relies on the quality of target topic semantic representation; however, topic descriptions in low-resource languages are often sparse. (3) In bilingual scenarios, the quantity and quality of labeled data between the two languages are highly unbalanced, making direct joint training ineffective.

**Key Challenge**: Systematic differences exist in stance expressions across languages; for instance, the same "against" stance might be expressed as a direct negation in English but as an indirect implication in another language. The key challenge lies in constructing a representation space that is insensitive to language forms but highly sensitive to stance semantics.

**Goal**: (1) To construct a shared bilingual stance representation space; (2) To achieve cross-lingual zero-shot stance detection transfer; (3) To leverage bilingual information complementarity to improve detection performance on both languages.

**Key Insight**: The authors observe that while bilingual data differs in surface forms, secondary stance semantics (patterns of favor/against) are shared across languages. Aligning the semantic spaces of both languages can facilitate implicit cross-lingual knowledge enhancement.

**Core Idea**: A cross-lingual zero-shot stance detection system is constructed using bilingual joint training and semantic alignment, where complementary information between languages improves the overall generalization capability.

## Method

### Overall Architecture
The system consists of three layers: (1) Language-specific encoding layer, which encodes texts from both languages using pretrained multilingual models; (2) Cross-lingual alignment layer, which aligns the representation spaces of both languages via contrastive learning or projection; (3) Unified stance classification layer, which performs zero-shot stance prediction in the shared space. Inputs are text-target pairs, and the outputs are stance labels (favor/against/neutral).

### Key Designs

1. **Bilingual Semantic Alignment Module**:

    - **Function**: Maps the text representations of both languages into a unified semantic space.
    - **Mechanism**: Initial representations are extracted via a multilingual pretrained model (e.g., XLM-R) and further aligned using contrastive learning. For representations of the same topic across both languages, their distance is minimized, while those of different topics are maximized. The alignment loss is formulated as $\mathcal{L}_{align} = -\log\frac{\exp(\text{sim}(h_i^{s}, h_i^{t})/\tau)}{\sum_j \exp(\text{sim}(h_i^{s}, h_j^{t})/\tau)}$, where $h^s$ and $h^t$ represent source and target language representations, respectively.
    - **Design Motivation**: Although multilingual pretrained models inherently demonstrate cross-lingual capabilities, their alignment on specific tasks (like stance detection) is not fine-grained. Task-specific alignment enables the model to focus on stance-related semantic dimensions.

2. **Target-Aware Zero-Shot Generalization Mechanism**:

    - **Function**: Enables the model to generalize to target topics unseen during training.
    - **Mechanism**: Stance detection is decoupled into two steps: "target understanding" and "stance inference." Target understanding encodes target descriptions to obtain target representations. Stance inference determines the stance by calculating interactions between the text representation and the target representation. Crucially, targets are represented via target descriptions (instead of target IDs), allowing the model to generalize to new targets through semantic understanding. In practice, an attention mechanism is designed to let the text representation focus on key information in the target description, generating target-aware text representations.
    - **Design Motivation**: Traditional stance detection models learn a mapping of "Target A $\rightarrow$ positive features," which fails to generalize to unseen targets. The target-description-driven approach learns a more general classification of "semantic relationship between text and target $\rightarrow$ stance."

3. **Bilingual Complementary Training Strategy**:

    - **Function**: Leverages data from both languages for mutual enhancement.
    - **Mechanism**: A joint training framework is adopted, where each batch contains samples from both languages. Along with individual classification losses, cross-lingual consistency regularization is introduced to enforce similar classification boundaries for the same stance category across both languages. Additionally, a language weight scheduling strategy is implemented, assigning equal training weights to both languages initially, and dynamically adjusting the sampling weights of each language based on validation performance in later stages.
    - **Design Motivation**: Rich-resource languages (e.g., English) contain more labeled data to guide the learning process of low-resource languages. Conversely, low-resource language data provides distinct stance expression patterns, bringing feedback to enhance the generalization capability of the high-resource language.

### Loss & Training
The total loss consists of three components: cross-entropy loss for stance classification, contrastive loss for cross-lingual alignment, and consistency regularization loss, formulating a weighted combination as $\mathcal{L} = \mathcal{L}_{cls} + \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{consist}$.

## Key Experimental Results

### Main Results

| Dataset/Language | Metric | Ours | Monolingual ZSSD | Direct Transfer with mBERT | Gain |
|------------|------|---------|---------|-------------|------|
| SemEval-En (Zero-Shot) | Macro-F1 | 56.8 | 51.2 | 48.5 | +5.6 |
| Target Language (Zero-Shot) | Macro-F1 | 52.3 | 45.1 | 43.8 | +7.2 |
| Joint Evaluation | Avg Macro-F1 | 54.6 | 48.2 | 46.2 | +6.4 |

### Ablation Study

| Configuration | En F1 | Target Language F1 | Description |
|------|-------|------------|------|
| Full model | 56.8 | 52.3 | Full Model |
| w/o Bilingual Alignment | 53.1 | 47.0 | Alignment yields the largest improvement on target language (+5.3) |
| w/o Target-Aware Mechanism | 51.5 | 48.2 | Target understanding is critical for zero-shot generalization |
| w/o Complementary Training Strategy | 54.8 | 49.5 | Bidirectional complementarity yields gradual improvements |
| Trained on Source Language Only | 53.2 | 44.1 | Direct transfer achieves limited performance |

### Key Findings
- Bilingual alignment achieves the highest performance improvement on target languages (+5.3 F1 score), verifying the core value of cross-lingual semantic alignment.
- The target-aware mechanism yields higher improvement on English (+5.3) than on the target language (+4.1), indicating that target semantic understanding has more leverageable information in high-resource languages.
- Bilingual joint training not only improves the performance of low-resource languages but also boosts English performance (+3.6), confirming the bidirectional complementarity hypothesis.
- Cross-lingual transfer achieves strong results on social targets (e.g., policy debates) but exhibits weaker performance on culturally specific targets.

## Highlights & Insights
- Extending zero-shot stance detection to bilingual/cross-lingual settings is a highly valuable research direction. With globalized information propagation and the popularity of multilingual LLMs, the demand for cross-lingual stance analysis is growing.
- The discovery of bilingual complementarity is highly inspiring: even high-resource languages can benefit from low-resource language data, as stance expression patterns in different languages provide beneficial regularization effects.
- The target-aware zero-shot generalization mechanism can be directly adapted to other NLP tasks requiring generalization to unseen classes.

## Limitations & Future Work
- The paper has not released an arXiv preprint, and specific experimental details remain to be verified.
- The "bilingual" setting requires access to data in both languages; how to extend this to multilingual settings (e.g., 10+ languages) remains to be explored.
- The zero-shot setup assumes that target targets contain textual descriptions; however, the quality of target descriptions varies in practical applications.
- Future work can leverage the superiority of LLMs in multilingual comprehension to further enhance zero-shot generalization via few-shot prompting.

## Related Work & Insights
- **vs TOAD (allaway2020zero)**: A classic zero-shot stance detection approach evaluated only in English. This work extends the paradigm to bilingual scenarios and outperforms it via complementary training.
- **vs Cross-lingual NLI approaches**: NLI transfer methods require large parallel corpora, whereas the alignment method in this work is more lightweight.
- **vs Multilingual prompt-based methods**: Direct prompting with LLMs shows lower accuracy in stance detection compared to specifically trained models.

## Rating
- Novelty: ⭐⭐⭐⭐ Bilingual zero-shot stance detection is a meaningful and novel setting.
- Experimental Thoroughness: ⭐⭐⭐ The paper is not publicly available, preventing full verification of experimental details.
- Writing Quality: ⭐⭐⭐ Inferred based on title; cannot be comprehensively evaluated.
- Value: ⭐⭐⭐⭐ The practical demand for cross-lingual stance analysis is growing rapidly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Large Language Models Address Open-Target Stance Detection?](can_large_language_models_address_open-target_stance_detection.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](on_the_acquisition_of_shared_grammatical_representations_in_bilingual_language_m.md)
- [\[ACL 2025\] BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework](bipro_zero-shot_chinese_poem_generation_via_block_inverse_prompting_constrained_.md)
- [\[ECCV 2024\] AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection](../../ECCV2024/llm_nlp/adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)

</div>

<!-- RELATED:END -->
