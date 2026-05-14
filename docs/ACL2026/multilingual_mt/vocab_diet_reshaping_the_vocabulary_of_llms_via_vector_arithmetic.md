---
title: >-
  [Paper Note] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic
description: >-
  [ACL 2026][Multilingual & Machine Translation][Compositional vocabulary] This paper demonstrates that LLMs encode morphological inflections (e.g., walk→walked) as linear directions in embedding space…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Compositional vocabulary"
  - "vector arithmetic"
  - "morphological transformation"
  - "vocabulary compression"
  - "multilingual coverage"
date: 2026-05-08
content_hash: addb2c0d1f49d106
---

# Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic

**Conference**: ACL 2026
**arXiv**: [2510.17001](https://arxiv.org/abs/2510.17001)
**Code**: [GitHub](https://vocabdiet.github.io)
**Area**: Multilingual Translation
**Keywords**: Compositional vocabulary, vector arithmetic, morphological transformation, vocabulary compression, multilingual coverage

## TL;DR

This paper demonstrates that LLMs encode morphological inflections (e.g., walk→walked) as linear directions in embedding space, and proposes a compositional vocabulary design: replacing independently assigned tokens for each surface form with additive combinations of base words and transformation vectors. With the pretrained backbone frozen, only a small adapter module is trained, freeing 10–40% of vocabulary slots for multilingual expansion with negligible impact on downstream performance.

## Background & Motivation

**Background**: Modern LLMs commonly employ BPE tokenization with vocabulary sizes exceeding 100K tokens. Vocabulary design is fundamentally a resource allocation problem: every slot allocated to one language or domain comes at the cost of coverage elsewhere. Recent studies have shown that vocabulary allocation is severely imbalanced across languages, negatively affecting model cost and performance.

**Limitations of Prior Work**: (1) **Redundant allocation**: Standard tokenization treats morphologically related word forms (walk, walks, walking, walked) as independent tokens, each occupying a vocabulary slot. Taking the GPT-4 vocabulary as an example, 24.6K English whole-word tokens reduce to only 14.3K base forms after removing case variants and inflectional/derivational variants—representing 42% redundancy. (2) **Insufficient multilingual coverage**: A large proportion of vocabulary slots are occupied by surface variants of high-resource languages, leaving low-resource languages severely underrepresented. (3) **OOV problem**: The existing 14.3K base forms and transformations could compositionally generate 98K vocabulary items currently outside the vocabulary, yet standard vocabularies cannot exploit this structure.

**Key Challenge**: Vocabulary size is constrained by memory and computation, yet standard tokenization ignores the linear morphological structure already present in LLM embedding spaces—models internally encode morphological inflections as simple vector offsets, while still learning independent embeddings for each surface variant at the vocabulary level.

**Goal**: (1) Verify whether LLMs can interpret compositional embeddings of the form "base word + transformation vector"; (2) construct a compositional vocabulary to free redundant slots; (3) validate feasibility in both post-training adaptation and pretraining-from-scratch settings.

**Key Insight**: Drawing on vector arithmetic from the word2vec era (king − man + woman = queen), the paper systematically validates the linear structure of morphological transformations in LLM embedding spaces, elevating this from an analytical tool to a practical vocabulary design solution.

**Core Idea**: Replace flat vocabularies with compositional vocabularies—each surface form $w$ is composed from a base word $b_w$ and a set of transformations $T(w)$: $e_w = e_{b_w} + \sum_{t_i \in T(w)} e_{t_i}$, applied at both input and output, freeing vocabulary space for multilingual expansion.

## Method

### Overall Architecture

The approach consists of three stages: (1) **Analytical validation**—using Patchscopes probes to verify whether LLMs correctly interpret compositional embeddings; (2) **Post-training adaptation**—fine-tuning transformation vectors via knowledge distillation with LoRA adapters on already-trained models; (3) **Pretraining from scratch**—validating compositional vocabularies as a new model design choice. On the input side, surface forms are replaced by the sum of base word and transformation vectors; on the output side, the large unembedding matrix is decomposed into two independent projections over base words and transformations respectively.

### Key Designs

1. **Compositional Vocabulary Representation**:

    - **Function**: Represent all surface forms via shared base words and transformation vectors, compressing the vocabulary.
    - **Mechanism**: Define a base vocabulary $V_b \subset V_{orig}$ (containing canonical word forms and auxiliary tokens) and a transformation vocabulary $V_t$ (morphological operations such as tense, number, etc.). Input side: $e_w = e_{b_w} + \sum_{t_i \in T(w)} e_{t_i}$; output side: $\text{logit}(w) = h \cdot u_{b_w} + \sum_{t_i \in T(w)} h \cdot u_{t_i}$, with independent projections over base words and transformations summed together. Transformation vectors are initialized via averaged offset computation: $o_t = \frac{1}{|R(t)|}\sum_{w \in R(t)} (o_w - o_{b(w)})$.
    - **Design Motivation**: Exploiting the linear morphological structure already present in LLM embedding spaces, redundant independent embeddings are replaced by shared compositional representations, enabling unified handling of both in-vocabulary and out-of-vocabulary words.

2. **Patchscopes Validation Framework**:

    - **Function**: Verify whether LLMs can interpret compositional embeddings as the intended surface forms.
    - **Mechanism**: For each composable word $w$, the token embedding is replaced with the compositional representation $e_w$, and Patchscopes prompts are used to generate text descriptions, which are checked for correspondence with the target word. Evaluation is conducted at two stages: the embedding layer (embed) and early-layer detokenization (detok). Experiments span five languages (English, Arabic, German, Russian, Spanish) and multiple models (Llama-3-8B, Qwen2.5-7B, OLMo-2-7B, ALLaM-7B, EuroLLM-9B).
    - **Design Motivation**: Before deploying the compositional vocabulary, it is necessary to verify whether models "natively" understand such compositions—if morphological inflections are already encoded as linear directions internally, compositional embeddings should be correctly interpreted.

3. **Two-Stage Knowledge Distillation Fine-tuning**:

    - **Function**: Lightweight adaptation of compositional vocabularies on already-trained models.
    - **Mechanism**: Stage one freezes the output unembedding and trains only the input transformation vectors, using original model predictions as distillation targets. Stage two freezes the input embeddings and trains only the output transformation vectors, using the stage-one model as distillation targets. LoRA adapters ($r=256$) are added to the last $k=8$ layers, with all other parameters frozen. Training requires only a small sample from FineWeb-Edu (5M tokens).
    - **Design Motivation**: Stage-wise training avoids instability from jointly optimizing both input and output transformation vectors; LoRA is applied only to the final layers because compositional representations primarily affect input/output mappings, leaving intermediate representations unchanged.

### Loss & Training

Knowledge distillation loss (KL divergence) is used with the original model's predictions as soft targets. Post-training adaptation introduces fewer than 0.001% additional parameters (transformation embeddings). In the pretraining-from-scratch setting, a factored prediction is used: $p(w|h) = p(b_w|h) \cdot p(T(w)|b_w; h)$, first predicting the base word and then conditionally predicting the transformations.

## Key Experimental Results

### Main Results — English Post-Training Adaptation (Llama-3.1-8B)

| Task Category | Task | Original Model | Compositional Vocab | Difference |
|--------------|------|---------------|---------------------|------------|
| Knowledge | MMLU | 65.2 | 64.9 | −0.3 |
| Knowledge | ARC | 53.6 | 52.5 | −1.1 |
| Reading Comprehension | BoolQ | 83.2 | 83.3 | +0.1 |
| Reading Comprehension | TriviaQA | 66.5 | 63.3 | −3.3 |
| Commonsense | HellaSwag | 60.6 | 59.5 | −1.1 |
| Commonsense | Winogrande | 78.1 | 78.6 | +0.5 |
| **Average** | | **66.9** | **65.9** | **−1.0** |

### Pretraining-from-Scratch Results (nanoGPT-124M)

| Language | Vocab Compression | BPB (Baseline) | BPB (Compositional) | bytes/tok Change |
|----------|------------------|---------------|---------------------|-----------------|
| English | 41.6% | 1.08 | 1.09 | — |
| Spanish | 41.8% | 1.00 | 1.11 | 4.77→4.92 |

### Key Findings

- English inflectional transformations (plural, tense, etc.) can be correctly interpreted at the embedding layer: plural nouns 92%, past tense 71%, present participle 83%.
- Accuracy improves further after early-layer detokenization: plural 96%, past tense 81%, present participle 93%.
- A critical distinction emerges: inflectional transformations perform well, while derivational transformations perform poorly—derivational forms rarely appear as single tokens in the vocabulary, resulting in weaker learned linear structure.
- Multilingual results are striking: Russian case inflection 97%, adjectival inflection 100% (small sample), suggesting linear structure may be even stronger in non-English languages.
- Post-training adaptation incurs an average downstream performance loss of only 1.0 point while freeing approximately 10K vocabulary slots.
- Pretraining from scratch can free 41% of vocabulary slots with a BPB increase of only 0.01 (English).
- After reallocating vocabulary slots, bytes-per-token improves by an average of 9.3% across four languages.

## Highlights & Insights

- The paper reveals a profound phenomenon: the linear morphological structure in LLM embedding spaces is effective not only for in-vocabulary items but also for out-of-vocabulary words—models have never encountered "walkable" as a single input vector, yet the composition "walk + -able" can be correctly interpreted in early layers (though derivational cases are weaker).
- Vocabulary size and morphological linearity are inversely related: smaller vocabularies force models to rely more heavily on linear combinations to encode morphological variation, resulting in stronger linear structure. This provides theoretical support for the utility of small vocabularies.
- The practical value is significant: the freed 10K slots can allocate 2.5K dedicated BPE tokens per target language, which alone can substantially improve multilingual tokenization efficiency (average +9.3% bytes-per-token).

## Limitations & Future Work

- Derivational transformations (-able, un-, re-, etc.) yield low compositional accuracy; the current approach handles only inflectional and case transformations.
- The approach relies on the UniMorph morphological database to construct decomposition mappings and is not applicable to languages without such annotations.
- Post-training adaptation incurs relatively larger losses on reading comprehension tasks (TriviaQA −3.3, SQuAD −2.1), likely because these tasks depend more heavily on exact surface form matching.
- Pretraining from scratch is validated only on a 124M-parameter model; performance at larger scales remains to be explored.
- The factored prediction on the output side introduces additional computational steps; although empirical measurements show only a 0.8% slowdown, this remains a consideration in latency-sensitive scenarios.

## Related Work & Insights

- **vs. vocabulary expansion methods (e.g., Nakash et al., 2025)**: These works improve coverage for specific languages by directly adding new tokens but are constrained by the total vocabulary size limit; Vocab Diet "creates space" by compressing existing redundancy and is complementary to expansion approaches.
- **vs. Park et al. (2024) linear probing analysis**: Their work analyzes linear structure in LLM embedding spaces but treats it purely as an analytical tool; Vocab Diet is the first to apply such structure to a practical end-to-end language modeling system.
- **vs. BPE tokenization improvements (e.g., Tao et al., 2024)**: These works advocate enlarging vocabularies to improve performance but are limited by computational cost; Vocab Diet offers an orthogonal approach to improving efficiency within a fixed vocabulary budget.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Redesigning the vocabulary from the perspective of linear morphological structure in embedding space is a highly original and pioneering contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validation across five languages and five models is comprehensive, though pretraining from scratch is evaluated only on a 124M-parameter model.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The argumentation is well-structured, progressing logically from validation → post-training → pretraining, with polished figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ The work has far-reaching implications for vocabulary design in multilingual LLMs, offering both theoretical insights and practical solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] What Factors Affect LLMs and RLLMs in Financial Question Answering?](what_factors_affect_llms_and_rllms_in_financial_question_answering.md)
- [\[ACL 2026\] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](../../ICLR2026/multilingual_mt/sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)

</div>

<!-- RELATED:END -->
