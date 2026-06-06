---
title: >-
  [Paper Note] How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them
description: >-
  [ACL 2026][Audio & Speech][subword tokenization] This paper employs three phonological probing tasks (rhyme / G2P / syllable count) to demonstrate that BPE-style subword tokenization suffers from being "too coarse-graine…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "subword tokenization"
  - "phonological knowledge"
  - "STAD"
  - "IPA fine-tuning"
  - "cognates"
date: 2026-05-08
content_hash: b45633f25899a2da
---

# How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them

**Conference**: ACL 2026  
**arXiv**: [2604.17105](https://arxiv.org/abs/2604.17105)  
**Code**: https://github.com/liaodisen/Tokenization-Phonology (Available)  
**Area**: Audio & Speech / Tokenization / Representation Probing  
**Keywords**: subword tokenization, phonological knowledge, STAD, IPA fine-tuning, cognates

## TL;DR
This paper employs three phonological probing tasks (rhyme / G2P / syllable count) to demonstrate that BPE-style subword tokenization suffers from being "too coarse-grained" to capture local phonology and having "misaligned boundaries" that hinder rhythmic structure perception. The authors propose the STAD metric and a lightweight IPA-augmented fine-tuning method, enabling Llama3.1-8B to achieve comprehensive improvements across all three phonological tasks with minimal performance degradation on GSM8K (-1.1%) and MMLU (-0.9%).

## Background & Motivation

**Background**: Pure text-based LLMs (e.g., GPT-4o) exhibit significant phonological awareness (e.g., writing poetry, rhyming, language teaching) despite never having processed audio. However, how this capability emerges from orthography and what factors limit it have not been systematically investigated.

**Limitations of Prior Work**: Suvarna et al. (2024) evaluated PhonologyBench using prompting and concluded that LLMs perform poorly on phonological tasks. However, prompting-based evaluations or zero-shot results often underestimate the latent knowledge within a model (Hu & Levy 2023). While isolated observations exist regarding BPE's struggles with phonemes, mechanistic explanations and quantitative diagnostics are lacking.

**Key Challenge**: Subword tokenization is designed for training efficiency (frequency-driven), which fundamentally conflicts with linguistic principles like "syllable boundaries" and "phoneme boundaries." Previously, identifying which tasks, layers, or word types were most severely affected relied primarily on intuition.

**Goal**: (RQ1) Use probing instead of prompting to measure how much phonological knowledge is actually encoded in LLM hidden states; (RQ2) Analyze how tokenization strategies (granularity + boundaries) affect this encoding; (RQ3) Inject phonological knowledge without modifying the fixed tokenizer.

**Key Insight**: Phonological capability can be divided into two levels: **local features** (rhyme/coda matching) require fine-grained tokens, while **prosodic structure** (syllables/G2P) requires alignment between tokens and syllables. The Syllabification-Tokenization Alignment Distance (STAD) is introduced to quantify this "degree of alignment" as a single metric.

**Core Idea**: Transform the intuition that "tokenizers are the root cause of phonological failure" into a measurable diagnostic metric (STAD). A lightweight post-training scheme involving "IPA augmentation + general QA mixing" is then used to bypass the high cost of modifying the tokenizer.

## Method

### Overall Architecture
The study consists of three sequential experiments: (1) **Probing**: Extracting the hidden state $\boldsymbol{h}_{i\ell}$ of the final token of each word at level $\ell$ to train linear probes across 6 models, 6 depth ratios, and 3 tasks; (2) **Mechanistic Analysis**: Testing granularity effects using slash-delimited inputs, boundary effects using STAD groupings (A: STAD=0 vs. M: STAD>0.25), and linguistic causes of failure using CogNet for cognate retrieval; (3) **Improvement**: Applying LoRA fine-tuning to Llama3.1-8B on a mixture of OpenHermes2.5 and phonological task data, including IPA transcriptions in the responses.

### Key Designs

1.  **Three Phonology Probing Tasks**:
    *   Function: Uses linear probes as a more accurate metric than prompting for internal phonological knowledge.
    *   Mechanism: Rhyme awareness is a binary classification using logistic regression on $\boldsymbol{h}_\ell$; G2P is regression on 39 phonemes padded to length 8; syllable count is ridge regression on integer labels.
    *   Design Motivation: Probes are intentionally restricted to be linear to prevent the probe itself from "learning" phonology, ensuring the representation quality is measured (Hewitt & Liang 2019). Random embedding controls are used to confirm signal validity.

2.  **STAD (Syllabification-Tokenization Alignment Distance)**:
    *   Function: Quantifies the deviation between a tokenizer's segmentation and the natural syllabification of a word.
    *   Mechanism: For a word with $n+1$ characters, all possible split positions are encoded into binary vectors $\boldsymbol{v}_{\text{tok}}$ and $\boldsymbol{v}_{\text{syl}}$ of length $n$. STAD is the normalized Hamming distance:
        $$\text{STAD} = \frac{\sum_i |b_i - c_i|}{n}$$
        For example, if the syllable vector for "musical" is $[0,1,0,1,0,0]$ and the Llama tokenization vector is $[0,0,1,0,0,0]$, STAD = $3/6 = 0.5$.
    *   Design Motivation: Converting "alignment" into a real number allows for paired t-tests between groups (A vs. M), directly verifying the causal link between poor alignment and poor probing performance. This metric is tokenizer-agnostic.

3.  **IPA-Augmented LoRA Fine-tuning**:
    *   Function: Injects phonological knowledge into a pre-trained model without changing the tokenizer.
    *   Mechanism: 0-2 random words in 3,000 OpenHermes2.5 QA pairs are wrapped in `<IPA>...</IPA>` tags, with their IPA transcriptions prepended to the response. This is mixed with 200 rhyme, 500 syllable, and 500 G2P task samples where responses explicitly use IPA for step-by-step reasoning. LoRA is applied to $W_Q, W_V$ with $r=8, \alpha=16$.
    *   Design Motivation: (a) Tokenizers are fixed in foundation models; (b) Interweaving IPA as side information preserves instruction-following distributions; (c) Small data volume and LoRA usage minimize catastrophic forgetting.

### Loss & Training
Probing utilizes scikit-learn `LogisticRegression` (C=10, max_iter=1000) and `RidgeCV` (alphas in {10,100,500,1000,2000}). LoRA SFT uses standard cross-entropy loss, taking less than 5 GPU hours on 2× A40. Evaluation uses rhyme/syllable accuracy and G2P Phoneme Error Rate (PER).

## Key Experimental Results

### Main Results

| Model | Layer 20% Rhyme Acc | Layer 20% G2P $R^2$ | Layer 20% Syllable $R^2$ |
|---|---|---|---|
| BERT (110M) | 67.6 | 0.073 | 0.265 |
| GPT-2 (124M) | 64.7 | 0.188 | 0.624 |
| GPT-neo (2.7B) | 68.6 | 0.147 | 0.662 |
| Mistral-7B-Instruct-v3 | 80.6 | 0.202 | 0.470 |
| Llama3-8B-Instruct | 80.7 | 0.263 | 0.661 |
| Llama3.1-8B-Instruct | **79.8** | **0.330** | **0.713** |
| Random embedding (control) | 48.7 | -0.100 | -0.073 |

| Configuration (Rhyme, Layer 20%) | BERT | GPT-2 | Mistral-7B | Llama3.1-8B |
|---|---|---|---|---|
| Original tokenization | 67.6 | 64.7 | 80.6 | 79.8 |
| Fine-grained (slash) | **74.5*** | **76.9*** | 81.1 | **85.1*** |

### Ablation Study

| Configuration | Key Observation | Description |
|---|---|---|
| Slash / Comma / Dot delimiter | All outperform "None"; similar performance. | Improvement stems from granularity, not the symbol. |
| STAD = 0 (A) vs > 0.25 (M) | Most models show significantly higher G2P/syllable $R^2$ in group A. | Confirms tokenizer-syllable alignment impacts internal representation. |
| BERT on STAD groupings | No obvious trend in BERT. | Bidirectional attention might mitigate segmentation misalignment. |
| Slash improvement per layer | Significant in probing layers, but zero-shot rhyme reasoning may not improve. | Tokenization improves representation but doesn't necessarily improve end-task performance without tuning. |
| Llama3.1-8B-IPA vs Llama3.1-8B-Instruct | GSM8K: 69.9 $\rightarrow$ 68.8 (-1.1); MMLU: 65.3 $\rightarrow$ 64.4 (-0.9). | General capabilities remain largely intact. |

### Key Findings
- **Phonological signals peak at 20-60% depth**: Prompting underestimates this; probing reveals actual latent capabilities.
- **STAD is a causal diagnostic, not just a correlation**: In the low-STAD group (A), G2P/syllable $R^2$ reaches 0.93-0.98, while falling to 0.5-0.7 in the high-STAD group (M). This 30-40 point gap is purely due to tokenization boundaries.
- **Cognates/Loanwords explain "why" segmentation fails**: CogNet retrieval shows group M words have significantly more cross-lingual variants than group A (e.g., "musical" has nearly 30 variants). Their n-gram distributions in training corpora likely differ from monolingual words, leading BPE to split them at non-syllable boundaries.
- **Lightweight IPA fine-tuning yields significant gains** across all three tasks with minimal cost to general ability, proving more practical than retraining with a new tokenizer.

## Highlights & Insights
- **STAD as an elegant diagnostic**: The simple Hamming distance formula can be applied to any new tokenizer as a sanity check for tokenizer-syllable alignment.
- **Linguistic "Why" via CogNet**: The study moves beyond observing that "BPE is weird" to providing a falsifiable linguistic hypothesis: cross-lingual variants shift n-gram distributions and cause BPE misalignment.
- **IPA-augmented fine-tuning strategy**: Distributing IPA tags throughout general QA data rather than training on phonology tasks in isolation is an effective design to prevent catastrophic forgetting.

## Limitations & Future Work
- **Language Scope**: Restricted to English and alphabetic languages; logographic languages (Japanese/Chinese) may behave differently.
- **G2P Formulation**: Linear regression is used instead of sequence generation for probing simplicity, which might underestimate true G2P capabilities.
- **STAD Ground Truth**: It assumes natural syllabification is the only "correct" split, though phonological schools differ (e.g., onset maximalism).
- **Scalability**: Fine-tuning has not yet been verified on models larger than 8B parameters.

## Related Work & Insights
- **vs. Suvarna et al. 2024 (PhonologyBench)**: While they claimed weak phonological ability via prompting, this paper uses probing to show the capability exists in representations but is not being fully utilized.
- **vs. ByT5 / CANINE**: Instead of replacing the tokenizer at the architectural level, this work adopts a "post-training injection" route, acknowledging the sunk cost of foundation models.
- **vs. Singh & Strouse 2024 (numeric tokenization)**: Similar to their findings for mathematics, this work extends the "misalignment hinders performance" phenomenon to the phonological domain.

## Rating
- Novelty: ⭐⭐⭐⭐ STAD is a clear original contribution; IPA-augmented fine-tuning is a novel "gentle injection" scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 models, 6 layer depths, 3 tasks, multiple delimiters, and an 8-tokenizer CogNet analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear progression: "what does tokenization affect $\rightarrow$ quantify it $\rightarrow$ explain it $\rightarrow$ fix it."
- Value: ⭐⭐⭐⭐ STAD serves as a sanity check for tokenizer design; IPA-augmented LoRA is a practical trick for knowledge injection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ICML 2026\] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](../../ICML2026/audio_speech/moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)
- [\[ICCV 2025\] How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](../../ICCV2025/audio_speech/how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)

</div>

<!-- RELATED:END -->
