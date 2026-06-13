---
title: >-
  [Paper Note] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic
description: >-
  [ACL 2026][Multilingual & Machine Translation][Compositional vocabulary] This paper discovers that LLMs encode morphological variations (e.g., walk→walked) as linear directions in the embedding space. Based on this…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Compositional vocabulary"
  - "vector arithmetic"
  - "morphological transformation"
  - "vocabulary compression"
  - "multilingual coverage"
date: 2026-05-08
content_hash: 836f7e19724caeed
---

# Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.17001](https://arxiv.org/abs/2510.17001)  
**Code**: [GitHub](https://vocabdiet.github.io)  
**Area**: Multilingual Translation  
**Keywords**: Compositional vocabulary, vector arithmetic, morphological transformation, vocabulary compression, multilingual coverage

## TL;DR

This paper discovers that LLMs encode morphological variations (e.g., walk→walked) as linear directions in the embedding space. Based on this, it proposes a compositional vocabulary design: replacing independent tokens for each surface form with an additive combination of a base word and transformation vectors. By training small adaptation modules while freezing the pre-trained backbone, this method releases 10-40% of vocabulary slots for multilingual expansion with negligible impact on downstream performance.

## Background & Motivation

**Background**: Modern LLMs commonly use the BPE tokenization algorithm, with vocabulary sizes exceeding 100K tokens. Vocabulary design is essentially a resource allocation problem: every slot assigned to a specific language or domain comes at the cost of other coverage. Recent studies show that vocabulary allocation is severely imbalanced across languages, negatively affecting both model cost and performance.

**Limitations of Prior Work**: (1) **Redundant Allocation**: Standard tokenization treats morphologically related forms (walk, walks, walking, walked) as independent tokens, each occupying a vocabulary slot. In the GPT-4 vocabulary, for instance, only 14.3K base forms remain out of 24.6K English full-word tokens after removing case and inflectional/derivational variants—a 42% redundancy. (2) **Insufficient Multilingual Coverage**: A large number of vocabulary slots are occupied by surface variants of high-resource languages, leaving low-resource languages severely under-covered. (3) **OOV Issues**: While 14.3K base forms and transformations could compose 98K words currently outside the vocabulary, standard vocabularies cannot exploit this structure.

**Key Challenge**: Vocabulary size is limited by memory and computational constraints, yet standard tokenization ignores the linear morphological structure already present in the LLM embedding space—the model internally encodes morphological changes as simple vector offsets, yet it still learns independent embeddings for every variant at the vocabulary level.

**Goal**: (1) Verify whether LLMs can understand "base word + transformation vector" compositional embeddings; (2) Construct a compositional vocabulary to release redundant slots; (3) Validate feasibility in both post-training adaptation and pre-training from scratch scenarios.

**Key Insight**: Starting from the vector arithmetic of the word2vec era (king - man + woman = queen), this work systematically verifies the linear structure of morphological transformations in LLM embedding spaces and elevates it from an analytical tool to a practical vocabulary design.

**Core Idea**: Replace flat vocabularies with compositional ones—each surface form $w$ is composed of a base word $b_w$ and a set of transformations $T(w)$: $e_w = e_{b_w} + \sum_{t_i \in T(w)} e_{t_i}$. This is applied at both the input and output ends to free up vocabulary space for multilingual expansion.

## Method

### Overall Architecture

The framework consists of three stages: (1) Analytical verification—using Patchscopes probes to verify if LLMs can correctly interpret compositional embeddings; (2) Post-training adaptation—fine-tuning transformation vectors via knowledge distillation and adding LoRA adapters to a pre-trained model; (3) Pre-training from scratch—validating compositional vocabulary as a design choice for new models. At the input end, surface forms are replaced by the sum of base words and transformation vectors. At the output end, the large unembedding matrix is decomposed into independent projections for base words and transformations.

### Key Designs

1.  **Compositional Vocabulary Representation**:
    *   **Function**: Compresses the vocabulary by representing all surface forms using shared base words and transformation vectors.
    *   **Mechanism**: Defines a base vocabulary $V_b \subset V_{orig}$ (containing canonical forms and auxiliary tokens) and a transformation vocabulary $V_t$ (morphological operations like tense, number, etc.). Input end: $e_w = e_{b_w} + \sum_{t_i \in T(w)} e_{t_i}$; Output end: $\text{logit}(w) = h \cdot u_{b_w} + \sum_{t_i \in T(w)} h \cdot u_{t_i}$, where independent projections on base words and transformations are summed. Transformation vectors are initialized by calculating mean offsets: $o_t = \frac{1}{|R(t)|}\sum_{w \in R(t)} (o_w - o_{b(w)})$.
    *   **Design Motivation**: To leverage the existing linear morphological structure in LLM embedding spaces, replacing redundant independent embeddings with shared compositional representations while supporting unified handling of in-vocabulary and out-of-vocabulary words.

2.  **Patchscopes Verification Framework**:
    *   **Function**: Verifies whether the LLM can interpret compositional embeddings as the intended surface forms.
    *   **Mechanism**: For each composable word $w$, the token embedding is replaced with the compositional representation $e_w$. Patchscopes prompts are used to generate text descriptions to check if they match the target word. Evaluation is performed at two stages: the embedding layer (embed) and early-layer detokenization (detok). Experiments are conducted across five languages (English, Arabic, German, Russian, Spanish) and multiple models (Llama-3-8B, Qwen2.5-7B, OLMo-2-7B, ALLaM-7B, EuroLLM-9B).
    *   **Design Motivation**: Before deploying compositional vocabularies, it is necessary to verify if the model "naturally" understands such compositions—if the model internally encodes morphology as linear directions, the compositional embeddings should be correctly interpreted.

3.  **Two-stage Knowledge Distillation Fine-tuning**:
    *   **Function**: Lightweight adaptation of compositional vocabularies to a pre-trained model.
    *   **Mechanism**: In the first stage, the output unembedding is frozen, and only input transformation vectors are trained using the original model's predictions as distillation targets. In the second stage, the input embeddings are frozen, and only output transformation vectors are trained. LoRA adapters ($r=256$) are added to the last $k=8$ layers, while other parameters remain frozen. Training requires only a small sample (5M tokens) from FineWeb-Edu.
    *   **Design Motivation**: Staged training avoids instability caused by simultaneous optimization of input and output transformation vectors. LoRA is only applied to the final layers as compositional representations mainly affect input/output mappings, leaving intermediate representations unchanged.

### Loss & Training

Knowledge distillation loss (KL divergence) is used, with the original model's predictions as soft targets. Post-training adaptation introduces less than 0.001% additional parameters (transformation embeddings). The pre-training from scratch scenario uses factored prediction: $p(w|h) = p(b_w|h) \cdot p(T(w)|b_w; h)$, where the base word is predicted first followed by conditional transformation prediction.

## Key Experimental Results

### Main Results — English Post-training Adaptation (Llama-3.1-8B)

| Task Category | Task | Original Model | Ours | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Knowledge | MMLU | 65.2 | 64.9 | -0.3 |
| Knowledge | ARC | 53.6 | 52.5 | -1.1 |
| Reading Comp | BoolQ | 83.2 | 83.3 | +0.1 |
| Reading Comp | TriviaQA | 66.5 | 63.3 | -3.3 |
| Commonsense | HellaSwag | 60.6 | 59.5 | -1.1 |
| Commonsense | Winogrande | 78.1 | 78.6 | +0.5 |
| **Average** | | **66.9** | **65.9** | **-1.0** |

### Pre-training from Scratch Experiment (nanoGPT-124M)

| Language | Vocab Compression | BPB (Baseline) | BPB (Ours) | bytes/tok change |
| :--- | :--- | :--- | :--- | :--- |
| English | 41.6% | 1.08 | 1.09 | — |
| Spanish | 41.8% | 1.00 | 1.11 | 4.77→4.92 |

### Key Findings

*   English inflectional transformations (plural, tense, etc.) can be correctly interpreted directly at the embedding layer: plural nouns 92%, past tense 71%, present participle 83%.
*   Accuracy further improves after early-layer detokenization: plural 96%, past tense 81%, present participle 93%.
*   **Key distinction**: Inflectional transformations perform well, while derivational transformations perform poorly—derivational affixes rarely appear as single tokens in the vocabulary, so the learned linear structure is weaker.
*   Multilingual results are striking: Russian case 97%, adjective inflection 100% (small sample), indicating that linear structures might be even stronger in non-English languages.
*   Post-training adaptation loses only 1.0 point in average downstream performance while releasing approximately 10K vocabulary slots.
*   Pre-training from scratch can release 41% of vocabulary slots with a BPB increase of only 0.01 (English).
*   After vocabulary slot reallocation, the average bytes-per-token for four languages improved by 9.3%.

## Highlights & Insights

*   Reveals a profound phenomenon: the linear morphological structure in LLM embedding spaces works not only for in-vocabulary words but also for out-of-vocabulary words—even if the model has never seen "walkable" as a single vector, the combination of "walk + -able" can be correctly interpreted in early layers (though derivation is weaker).
*   The relationship between vocabulary size and morphological linear structure is inversely proportional: the smaller the vocabulary, the more the model is forced to use linear combinations to encode morphology, leading to stronger linear structures. This provides theoretical support for the validity of smaller vocabularies.
*   High practical value: The 10K released slots can be used to allocate 2.5K dedicated BPE tokens for each target language, which significantly improves multilingual tokenization efficiency (+9.3% average bytes-per-token).

## Limitations & Future Work

*   Low compositional accuracy for derivational transformations (-able, un-, re-, etc.); current solutions only handle inflection and casing.
*   Dependence on the UniMorph morphological database for building decomposition mappings, which is not applicable to unannotated languages.
*   Post-training adaptation shows relatively larger losses in reading comprehension tasks (TriviaQA -3.3, SQuAD -2.1), possibly because these tasks rely more on precise morphological matching.
*   Pre-training from scratch was only verified on a small 124M parameter model; the effect on larger scales remains to be explored.
*   Factored prediction at the output stage increases computational steps; although testing showed only a 0.8% slowdown, it remains a concern for extreme latency-sensitive scenarios.

## Related Work & Insights

*   **vs. Vocab Extension Methods (e.g., Nakash et al., 2025)**: These works improve specific language coverage by directly adding new tokens but are limited by the total vocabulary size cap; Vocab Diet "makes room" by compressing existing redundancy, complementing extension methods.
*   **vs. Linear Probe Analysis (Park et al., 2024)**: They analyzed the linear structure of LLM embedding spaces but only as an analytical tool; Vocab Diet is the first to utilize this structure for a practical system in end-to-end language modeling.
*   **vs. BPE Tokenization Improvements (e.g., Tao et al., 2024)**: These works advocate for larger vocabularies to improve performance but are limited by computational costs; Vocab Diet offers an orthogonal perspective to improve efficiency under a fixed vocabulary budget.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ Redesigning the vocabulary based on the linear morphological structure of the embedding space is a unique and pioneering perspective.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive verification across five languages and five models, though pre-training from scratch was only done on a 124M model.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentative chain, progressing from verification to post-training and then pre-training, with excellent figures.
*   Value: ⭐⭐⭐⭐⭐ Significant impact on vocabulary design for multilingual LLMs, offering both theoretical insights and practical solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models](vocabulary_shapes_cross-lingual_variation_of_word-order_learnability_in_language.md)
- [\[ACL 2026\] What Factors Affect LLMs and RLLMs in Financial Question Answering?](what_factors_affect_llms_and_rllms_in_financial_question_answering.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[ACL 2026\] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)

</div>

<!-- RELATED:END -->
