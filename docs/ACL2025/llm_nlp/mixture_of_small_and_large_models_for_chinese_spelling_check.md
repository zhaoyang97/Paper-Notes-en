---
title: >-
  [Paper Note] Mixture of Small and Large Models for Chinese Spelling Check
description: >-
  [ACL 2025][LLM (Other)][Chinese Spelling Check] This paper proposes dynamically mixing the probability distributions of a small model (fine-tuned BERT) and a large language model (LLM) during the Beam Search decoding phase for Chinese spelling correction. Without fine-tuning the LLM, this approach balances the precise correction of the small model with the fluency of the LLM, achieving SOTA performance on multiple CSC datasets.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Chinese Spelling Check"
  - "Model Mixture"
  - "Beam Search"
  - "BERT"
  - "Large Language Models"
date: 2026-05-08
content_hash: dceac862f3b96acb
---

# Mixture of Small and Large Models for Chinese Spelling Check

**Conference**: ACL 2025  
**arXiv**: [2506.06887](https://arxiv.org/abs/2506.06887)  
**Code**: [https://github.com/zhqiao-nlp/MSLLM](https://github.com/zhqiao-nlp/MSLLM)  
**Area**: Others  
**Keywords**: Chinese Spelling Check, Model Mixture, Beam Search, BERT, Large Language Models

## TL;DR

This paper proposes dynamically mixing the probability distributions of a small model (fine-tuned BERT) and a large language model (LLM) during the Beam Search decoding phase for Chinese spelling correction. Without fine-tuning the LLM, this approach balances the precise correction of the small model with the fluency of the LLM, achieving SOTA performance on multiple CSC datasets.

## Background & Motivation

**Background**: Chinese Spelling Check (CSC) is a classic NLP task aiming to detect and correct spelling errors in text (usually substitutions of visually or phonetically similar characters). Current mainstream methods are divided into two categories: (1) fine-tuning small BERT-like models to learn correction patterns on high-quality annotated data; (2) directly performing correction using the linguistic knowledge of LLMs.

**Limitations of Prior Work**: Both paradigms suffer from fundamental drawbacks. Small BERT-like models exhibit high precision but rely heavily on the edit patterns in the training data—the models overfit to the error types and position distributions of the training set (edit pattern overfitting), resulting in sharp performance degradation on new patterns. On the other hand, although LLMs possess rich linguistic knowledge and are unrestricted by specific edit patterns, their actual performance on CSC is inferior to fine-tuned BERT models. LLMs tend to perform broader paraphrasing rather than precise character-level spelling correction, leading to over-correction.

**Key Challenge**: Compounding the precision of the small model with the generalization capability of the LLM is extremely challenging. Fine-tuning an LLM is costly and ineffective; simply concatenating the outputs of the two models also lacks an elegant way of fusion.

**Goal**: To design a method that dynamically integrates the precise correction capability of tiny models with the fluency of LLMs, without fine-tuning the LLM.

**Key Insight**: The complementarity of the small model and the LLM manifests at the level of probability distributions—the small model generates candidate characters with high confidence at erroneous spots, while the LLM exerts constraints on global linguistic fluency. Real-time mixing of their probability distributions during the decoding phase can leverage the strengths of both.

**Core Idea**: During the Beam Search decoding process, dynamically adjust the weighted mixture of the probability distributions from the small model and the LLM for token prediction at each position. This ensures that the final prediction inherits the precise correction signals from the small model while being constrained by the fluency of the LLM.

## Method

### Overall Architecture

The input is a Chinese sentence potentially containing spelling errors. First, the fine-tuned small BERT model and the LLM independently calculate the probability distribution of candidate characters at each position. Then, during the Beam Search decoding stage, a dynamic weighted mixture of these two probability distributions is computed to obtain a hybrid probability distribution. Based on this, a beam search is performed to select the optimal correction sequence. The output is the corrected sentence.

### Key Designs

1. **Dynamic Probability Distribution Mixture**:

    - **Function**: Combining predictions from the small model and the LLM in real-time at each decoding step.
    - **Mechanism**: The mixed probability at position $t$ is $P_{mix}(w_t) = \alpha \cdot P_{small}(w_t) + (1-\alpha) \cdot P_{LLM}(w_t)$, where $\alpha$ is the mixing weight. Crucially, $\alpha$ is not a globally fixed value but is dynamically adjusted based on the correction confidence of the small model. When the small model is highly confident at a certain position (indicated by a concentrated probability distribution), $\alpha$ is larger, relying more on the small model. When the small model is uncertain, $\alpha$ decreases, relying more on the LLM's language model judgment.
    - **Design Motivation**: Static mixing weights cannot adapt to the correction requirements of different positions—some positions contain obvious spelling errors (where the small model excels), while others involve semantic comprehension (where the LLM excels). A dynamic $\alpha$ enables the system to adaptively switch between the two models.

2. **Beam Search Decoding Integration**:

    - **Function**: Searching for the optimal correction candidate at the sequence level rather than the independent position level.
    - **Mechanism**: Standard BERT-based CSC models usually make independent predictions at each position, neglecting inter-position dependencies. This work integrates the mixed probability $P_{mix}$ into the Beam Search framework, where the score of each candidate sequence in the beam is the product of the mixed probabilities across all positions (or sum in the log space). The beam size and search strategy are adjustable.
    - **Design Motivation**: Sequence-level decoding captures the dependencies between corrections (e.g., modifying the 3rd character may affect the optimal choice of the 5th character), which is more reasonable than independent position-wise predictions.

3. **Plug-and-play Design without LLM Fine-tuning**:

    - **Function**: Lowering the barrier to entry and supporting flexible domain adaptation.
    - **Mechanism**: The LLM operates in a zero-shot manner—given an input sentence, it simply calculates the conditional probability at each position, requiring no CSC-specific fine-tuning. Only the small BERT model needs to be fine-tuned. When transferring to a new domain, one only needs to replace the fine-tuning data for the small model, while the LLM component remains unchanged.
    - **Design Motivation**: Fine-tuning LLMs is computationally expensive and yields suboptimal results on CSC (due to severe over-correction issues). Avoiding fine-tuning saves substantial resources. Additionally, different LLMs can be utilized in a plug-and-play manner, facilitating seamless upgrades.

### Loss & Training

The small model (BERT) is trained using standard CSC training strategies: cross-entropy loss fine-tuned on annotated correction data. The LLM is not fine-tuned and directly computes conditional probabilities using its pre-trained weights. The dynamic adjustment strategy of the mixing weight $\alpha$ is tuned on the validation set.

## Key Experimental Results

### Main Results

| Method | SIGHAN15 F1 | ECSpell F1 | LEMON F1 | Type |
|------|------------|------------|----------|------|
| BERT-based (e.g., ReaLiSe) | High Precision / Low Recall | High Precision / Low Recall | Medium | Small Model |
| Direct LLM Correction (e.g., GPT-4) | Low Precision / High Recall | Low | Low | Large Model |
| Prev. SOTA | Suboptimal | Suboptimal | Suboptimal | - |
| MSLLM (Ours) | **SOTA** | **SOTA** | **SOTA** | Hybrid |

### Ablation Study

| Configuration | F1 Change | Description |
|------|------------|------------|
| Full MSLLM | Optimal | Complete hybrid system |
| Small Model Only ($\alpha=1$) | Decrease | Lacks the fluency constraints of LLM |
| LLM Only ($\alpha=0$) | Significant Decrease | Issues with LLM over-correction are pronounced |
| Static $\alpha$ | Decrease | Less flexible than dynamic weights |
| w/o Beam Search (Greedy) | Decrease | Sequence-level search is superior to independent positions |

### Key Findings

- **Significant Complementarity of the Mixture Strategy**: Small models have high precision but low recall (conservative), whereas LLMs yield high recall but low precision (aggressive). After mixing, their respective weaknesses compensate for each other, leading to a significant improvement in F1.
- **Dynamic $\alpha$ Outperforms Static $\alpha$**: This validates the hypothesis that "different positions require varying degrees of trust in the model."
- **No LLM Fine-tuning is a Substantial Advantage**: Experiments demonstrate that fine-tuning LLMs for CSC deteriorates performance (as the correction style of the LLM becomes overly aggressive), making the plug-and-play strategy a wiser choice.
- **Easy Domain Adaptation**: On datasets from different domains (General, Medical, Legal), only the small model needs to be replaced, while the LLM component remains unchanged, thereby reducing deployment costs.

## Highlights & Insights

- **Thorough Problem Analysis**: The edit pattern overfitting of small models and the over-correction issue of LLMs are acknowledged difficulties in the CSC field. This paper provides a highly precise root-cause analysis (complementarity at the probability distribution level) for both issues.
- **Ingenious Decoding-level Fusion**: Rather than fusing the two models at the input, feature, or output levels, this method mixes them at the design space of probability distributions with zero coupling and in a plug-and-play manner. This paradigm can be generalized to any sequence prediction task requiring the combination of models of varying scales (e.g., grammatical error correction, machine translation post-editing).
- **High Practical Value**: The combination of no LLM fine-tuning, SOTA performance, and easy domain adaptation makes this approach highly attractive for industrial deployment.

## Limitations & Future Work

- **Requires Running Both Models Simultaneously**: Inference requires both the small model and the LLM online to compute probabilities concurrently, resulting in higher inference latency and memory footprints.
- **Heuristic Self-adaptation Strategy for Dynamic $\alpha$**: The adjustment of $\alpha$ based on the small model's confidence lacks theoretical guarantees; exploring superior mixing strategies (e.g., learnable mixtures) remains a worthwhile direction.
- **Experiments Focus Primarily on the SIGHAN Series**: These traditional datasets might not fully reflect the actual error distributions in real-world scenarios.
- **Limited to Chinese Spelling Check**: Whether this can be generalized to English spelling/grammatical error correction, Japanese/Korean correction, etc., remains to be validated.
- Future work can explore "Multi-Expert" schemes: Mixing multiple small models (specialized for different error types) with an LLM could be explored in the future.

## Related Work & Insights

- **vs ReaLiSe (Xu et al. 2021)**: ReaLiSe integrates phonetic and graphic information into a BERT model, yielding excellent performance on CSC but still suffering from overfitting. The hybrid method in this paper can directly incorporate LLM probabilities on top of ReaLiSe for further improvement.
- **vs Direct LLM Correction (e.g., ChatGPT/GPT-4)**: Multiple studies have indicated over-correction issues of LLMs on CSC—this paper elegantly sidesteps this issue via a mixture strategy rather than attempting to rectify the LLM itself.
- **vs Ensemble Methods**: Traditional model ensembling performs voting at the output layer, whereas this method dynamically mixes probabilities at the distribution level, allowing for finer granularity and greater flexibility.

## Rating

- Novelty: ⭐⭐⭐⭐ The dynamic mixture strategy at the probability distribution level is novel and intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, detailed ablation studies, and domain adaptation experiments.
- Writing Quality: ⭐⭐⭐⭐ Sound problem analysis and clear method elaboration.
- Value: ⭐⭐⭐⭐ High practical value, ready for industrial deployment, with a generalizable paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](../../NeurIPS2025/llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)
- [\[ACL 2025\] Argument Mining in the Age of Large Language Models](argument_mining_in_the_age_of_large_language_models.md)
- [\[ICML 2026\] Rethinking LLM Ensembling from the Perspective of Mixture Models](../../ICML2026/llm_nlp/rethinking_llm_ensembling_from_the_perspective_of_mixture_models.md)

</div>

<!-- RELATED:END -->
