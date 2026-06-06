---
title: >-
  [Paper Note] On the Proper Treatment of Units in Surprisal Theory
description: >-
  [ACL 2026][LLM Pretraining][Surprisal theory] This paper points out that the choice of the "next unit" in surprisal theory has been silently determined by pre-trained language model (PLM) tokenizers. It proposes a finite…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Surprisal theory"
  - "unit inventory"
  - "finite-state transducers"
  - "eye-tracking reading times"
  - "GPT-2"
date: 2026-05-08
content_hash: 7f0adf7e302d8f89
---

# On the Proper Treatment of Units in Surprisal Theory

**Conference**: ACL 2026  
**arXiv**: [2604.28147](https://arxiv.org/abs/2604.28147)  
**Code**: https://github.com/samuki/units-surprisal  
**Area**: LLM Pre-training / Psycholinguistics / LM Evaluation  
**Keywords**: Surprisal theory, unit inventory, finite-state transducers, eye-tracking reading times, GPT-2

## TL;DR
This paper points out that the choice of the "next unit" in surprisal theory has been silently determined by pre-trained language model (PLM) tokenizers. It proposes a finite-state transduction framework that explicitly separates model tokens, linguistic units, and experimental ROIs, and validates on MECO eye-tracking data that different unit inventories alter the prediction of reading times by surprisal.

## Background & Motivation
**Background**: Surprisal theory uses $-\log p(u_t \mid u_{<t})$ to explain processing burden in human language comprehension: the less predictable a linguistic unit is, the more difficult it is to process, leading to longer reading times. Early studies often trained their own PCFGs, n-grams, or small-scale LMs, allowing the model's basic alphabet to be set to the words, PTB tokens, or other linguistic units required by the experiment.

**Limitations of Prior Work**: With the ubiquity of large-scale PLMs, researchers usually directly inherit the BPE/token alphabet of models like GPT-2 or LLaMA. The issue is that model tokens do not equate to linguistic words, morphemes, or phonemes, nor do they necessarily align with region of interest (ROI) boundaries in eye-tracking experiments. Consequently, many papers must post-hoc aggregate token surprisal into word surprisal or use rules like leading/trailing whitespace to handle boundaries. These practices often conflate "what unit to analyze for the scientific question" with "how the model is tokenized."

**Key Challenge**: Surprisal theory fundamentally requires a definition of the unit $U$ in human comprehension, whereas pre-trained LMs provide a probability distribution over the model alphabet $\Sigma$. When these are inconsistent, treating $\Sigma$ directly as $U$ misinterprets engineering details of the tokenizer as psycholinguistic hypotheses. Conversely, using simple whitespace rules for conversion creates inconsistencies in context-dependent scenarios such as punctuation, abbreviations, numbers, and sentence-initial words.

**Goal**: The authors aim to solve three sub-problems: first, formally distinguish model alphabets, experimental unit inventories, and ROIs; second, provide a general computational method to derive arbitrary unit-level surprisal from token-level LMs; third, demonstrate with real eye-tracking data that unit selection not only affects numerical values but also alters regression observations, control variables, and significance interpretations.

**Key Insight**: The paper starts from a simple yet important observation: tokenization should be an implementation detail, not a scientific primitive. That is, researchers should first select units based on theoretical questions and then transform the language model to this unit inventory, rather than being restricted by the tokenizer.

**Core Idea**: Express the unit parser $\rho: \Sigma^* \to U^*$ as a composable finite-state transducer and compute next-unit surprisal on any reasonable unit inventory via pushforward / transduced language models.

## Method
The method in this paper is not training a new model but establishing a formal and executable "unit processing protocol" for surprisal analysis. Its core logic is: first, treat the pre-trained LM as a distribution $p_\Sigma$ defined over the model alphabet $\Sigma$; then use a unit parser $\rho$ to map the $\Sigma$ strings to the research-desired unit sequence $U^*$; finally, redefine the next-unit probability and surprisal on $U$.

### Overall Architecture
The overall process can be divided into four steps.

Step 1: The researcher selects the unit inventory $U$. This $U$ can be GPT-2 tokens, characters, whitespace-separated words, PTB-style contextual words, or even finer phonemes or coarser discourse units. Crucially, $U$ is a scientific modeling choice and is not automatically determined by the LM tokenizer.

Step 2: Define a unit parser $\rho: \Sigma^* \to U^*$. Given a string over the model alphabet, $\rho$ is responsible for outputting the unit sequence. For cases like Chinese word segmentation where ambiguity may exist, the paper first provides a general form for a stochastic parser; for computability, $\rho$ is assumed to be a deterministic total function in the main body.

Step 3: Push $p_\Sigma$ to the unit space. Ideally, the probability of a unit sequence $\mathbf{u}$ is $p_U(\mathbf{u}) = \sum_{\boldsymbol{\sigma} \in \rho^{-1}(\mathbf{u})} p_\Sigma(\boldsymbol{\sigma})$. This step reveals a key point: the realization $\rho^{-1}$ is generally a relation and should not be forced into a function, as the same unit may have multiple string realizations.

Step 4: Calculate next-unit surprisal using the transduced LM. The authors map each unit to its underlying string and append a separator `sep`, making $h(u)=\xi\,\text{sep}$ a prefix-free encoding; then let $f=h\circ\rho$, using a finite-state transducer to convert $\Sigma^*$ to a finite alphabet $\Delta^*$ annotated with `sep`. Thus, the unit probability can be recovered via the prefix probability ratio on $\Delta$: $p_U(u\mid\mathbf{u}) = \overrightarrow{p}_\Delta(h(\mathbf{u})h(u)) / \overrightarrow{p}_\Delta(h(\mathbf{u}))$.

### Key Designs
1.  **Decoupling unit inventory $U$ from model alphabet $\Sigma$**:
    *   **Function**: Clearly distinguishes "linguistic units to be analyzed in the experiment" from "tokens natively output by the PLM."
    *   **Mechanism**: The language model $p_\Sigma$ still provides probabilities over $\Sigma^*$, but the target event for surprisal is defined on $U^*$; the two are connected by the unit parser $\rho$. This allows words, characters, PTB tokens, morphemes, or ROI-aligned units to become valid objects of analysis.
    *   **Design Motivation**: Much previous work defaults to tokens as units or aggregates token surprisal into word surprisal post-hoc. This allows BPE compression rules, whitespace attribution, and punctuation handling to directly influence psycholinguistic conclusions. After decoupling, the tokenizer only serves a computational implementation role and no longer dictates the theoretical unit.

2.  **Treating realization as a relation rather than a function**:
    *   **Function**: Fixes the problem in leading/trailing whitespace formalisms where the same word is forced into different units depending on context.
    *   **Mechanism**: Existing methods often assume $\rho^{-1}$ is a monoid homomorphism and partition the alphabet into boundary and continuation symbols. The paper points out this leads to unit inconsistency: for example, a sentence-initial `Hale` might be realized as `bosHale`, while `Hale` after a space is realized as `\u2423Hale`. If realization must be a function, these must be treated as different units. This paper allows one unit to be associated with multiple string realizations, so `Hale` remains the same unit.
    *   **Design Motivation**: The identity of a human linguistic unit should not be determined by whether it is at the start of a sentence or preceded by a space. The relation formalism is more realistic and explains why simple whitespace partitioning cannot handle context-dependent punctuation like `1,000`, `end, he said`, `don't`, or `cat's`.

3.  **Implementing computational conversion with regular unit inventories and finite-state transduction**:
    *   **Function**: Allows potentially infinite unit sets to be processed by language models on finite alphabets.
    *   **Mechanism**: It assumes units themselves are a regular language over some finite alphabet $\Xi$, i.e., $U\subseteq\Xi^*$. By appending `sep` to each unit to obtain a prefix-free $h(u)$, and using a finite transducer $f=h\circ\rho$, strings are converted from the model alphabet to `sep`-annotated strings. Computations can then utilize existing transduced LM algorithms to marginalize the probabilities of all source strings mapped to the target output.
    *   **Design Motivation**: Surface forms in natural language can be infinite, so treating every word as an LM alphabet is impractical; however, many phonological, morphological, and tokenization rules can be expressed via regular constraints and finite-state machines. This design bridges theoretical freedom and engineering feasibility.

### Loss & Training
This paper does not train new neural LMs. The source LM in the experiment is GPT-2 Small. Token inventory surprisal is read directly from GPT-2; other inventories estimate contextual surprisal by combining GPT-2 with the corresponding transducer.

For human reading time modeling, the paper uses a log-normal generalized additive mixed model (GAMM). The baseline model includes unit length, unigram surprisal, and their previous two spillover positions; the target model adds contextual surprisal and its previous two spillover positions to the baseline. Contribution is measured by improvement in held-out log-likelihood $\Delta_{\text{llh}}$, where positive values indicate that contextual surprisal explains additional reading time variance beyond length and unigram frequency.

Unigram surprisal is also estimated from the same LM distribution rather than external frequency resources. The authors sample text from GPT-2, transform samples into the target unit space via the transducer, and estimate marginal next-unit probability at unit boundaries. This ensures the frequency control and contextual surprisal come from the same model distribution, avoiding problems where resources like Speer-based frequencies conflate punctuation forms.

## Key Experimental Results

### Main Results
The paper evaluates four types of unit inventories on the MECO English eye-tracking corpus: GPT-2 tokens, characters, acontextual words, and PTB-style contextual words. Acontextual words are further split into leading whitespace and trailing whitespace attributions. Data consists of raw fixations from 46 readers across 12 Wikipedia short passages; reading time metrics include first fixation (FF), gaze duration (GD), and total reading time (TRT). Evaluation uses trial-based 12-fold leave-one-out cross-validation, with confidence intervals provided via 1000 trial-level bootstraps.

| Unit Inventory | FF $\Delta_{\text{llh}}$ | GD $\Delta_{\text{llh}}$ | TRT $\Delta_{\text{llh}}$ | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| Characters | 0.11, p=0.145 | 0.09, p=0.185 | 0.10, p=0.171 | Not significant across metrics |
| GPT-2 tokens | 0.55*, p=0.013 | 1.52**, p<0.001 | 2.56**, p<0.001 | Token surprisal has stable gain |
| Acontextual words (leading) | 0.28, p=0.065 | 1.41**, p<0.001 | 3.00**, p<0.001 | Not significant for early fixations |
| Acontextual words (trailing) | 0.63**, p=0.004 | 1.68**, p<0.001 | 2.91**, p<0.001 | Significant across all metrics |
| Contextual words | 0.81**, p=0.003 | 2.13**, p<0.001 | 3.24**, p<0.001 | PTB-style significant across all |

Note: Values represent per-observation held-out log-likelihood improvement in $10^{-3}$ nats. The authors emphasize that since different inventories have different observation counts, unit granularities, length controls, and unigram surprisals, absolute values cannot be used to rank which unit is "best"; they primarily show whether contextual surprisal provides additional predictive power relative to its baseline within each inventory.

### Ablation Study
This paper does not contain a traditional neural network ablation. The closest equivalent is varying unit inventories, whitespace attributions, and transducer complexity to see how the same GPT-2 Small generates different regression problems under different unit definitions.

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| GPT-2 tokens | 2,478 units; 40,589 GAMM obs | Closest to LM implementation; weak linguistic interpretation |
| Acontextual leading | 2,095 units; 39,290 obs; 212.8 sym/s | Space attached to following word; fast FST; FF not significant |
| Acontextual trailing | 2,095 units; 39,767 obs; 203.6 sym/s | Space attached to preceding word; FF/GD/TRT all significant |
| Contextual words | 2,264 units; 33,472 obs; 12.0 sym/s | Linguistic PTB-style; large FST; computational cost >10x higher |
| Characters | 13,226 units; 48,834 obs | Finest granularity; many units not fixated; not significant |

### Key Findings
- Word-like inventories significantly improve held-out log-likelihood for gaze duration and total reading time, indicating that contextual surprisal explains late reading time even after controlling for length, unigram surprisal, and spillover.
- The $\Delta_{\text{llh}}$ for character inventory is small and non-significant. This does not necessarily mean character surprisal lacks cognitive significance, but rather that eye fixations rarely fall naturally on individual character units; when units and observation ROIs do not match, the statistical problem is distorted.
- Leading/trailing whitespace is not an irrelevant implementation detail. Acontextual leading is not significant for first fixation while trailing is, demonstrating that delimiter attribution changes unit length, spillover controls, and fixation attribution.
- Contextual words align best with linguistic intuition but incur the highest computational cost: contextual surprisal throughput is only 12.0 symbols/s, compared to ~200 symbols/s for acontextual FSTs.
- The most important conclusion is not that "PTB tokens are definitively best," but that "unit selection must be reported and justified as part of the experimental design." Different units induce different data tables, and cross-unit comparisons cannot rely solely on the magnitude of $\Delta_{\text{llh}}$.

## Highlights & Insights
- The paper elevates what is often considered a "preprocessing" issue to a theoretical modeling choice: what exactly is $u_t$ in surprisal. This perspective is valuable as many disputes in LLM psycholinguistics stem from implicit mismatches between units and ROIs.
- The "realization as a relation" concept is a minor but critical formal fix. It elegantly explains why sentence-initial words, spaces, punctuation, and intra-number symbols cannot be resolved by fixed boundary partitions.
- Using `sep` to make unit encoding prefix-free is key to computability. it explicitly incorporates the "unit completion" event into probability rather than simply summing character or token surprisals.
- The experimental design avoids trying to prove one unit is absolutely superior, showing instead how unit choice alters observations, control variables, and significance. This cautious interpretation is more suitable for a methodological paper than mere metric-chasing.
- This framework can be migrated to many NLP and cognitive modeling scenarios: e.g., phoneme-level EEG/MEG surprisal, morpheme-level reading, sentence/discourse ROIs, or even LM interpretability analysis across different tokenizations.

## Limitations & Future Work
- Empirical scope is narrow: the experiment only covers English MECO, GPT-2 Small, and GAMMs. Concepts of "words" vary greatly across languages; Chinese, Japanese, morphologically rich languages, or space-less scripts might require entirely different parsers and empirical validation.
- The current framework assumes the unit parser is deterministic and representable by a rational/finite-state transducer. Truly ambiguous segmentation or unit transformations requiring syntax stacks or context-free structures are outside the scope of this implementation.
- Computational costs remain significant. Especially for contextual words and unigram surprisal estimation, marginalization over many source strings requires beam-search approximations and parallel sampling to be feasible.
- Dependence on raw fixation data. Many public reading time corpora are already aggregated by word, making it impossible to reassign fixations to new units; self-paced reading data is also naturally tied to the presented units in the experiment.
- The ROI portion is more conceptual clarification; experiments mainly focus on word/character/token granularities. Future work could directly validate more complex ROIs like discourse units, clause-level ROIs, or parafoveal preview character windows.

## Related Work & Insights
- **vs Oh and Schuler (2024) / Pimentel and Meister (2024)**: These works provide formalisms for calculating word-level surprisal from token LMs but rely on whitespace/boundary partitioning and functional realizations. Ours points out this causes unit inconsistency and uses a relation + transducer framework to unify leading/trailing and contextual segmentation.
- **vs Nair and Resnik (2023) / Beinborn and Pinter (2023)**: These studies focus on the cognitive plausibility of subword tokenization, often analyzing model tokens. Ours does not negate token-level analysis but emphasizes that tokens are merely one optional inventory, suitable for studying the model itself but not as a default representative for human processing units.
- **vs Wilcox et al. (2023), Shain et al. (2024), Goodkind and Bicknell (2018)**: These works validate the predictive power of surprisal for reading times. Our contribution is not in proposing a new linking function but in adding a layer of theoretical constraint via unit selection and ROI compatibility before the linking pipeline.
- **vs Snæbjarnarson et al. (2026) / Vieira et al. (2025)**: This paper directly leverages transduced LMs and token-to-character conversion tools for unit-level surprisal in psycholinguistics. The insight is that finite-state methods can serve as a "probabilistic interface" between LLMs and linguistic units.
- **Insights for future research**: When conducting surprisal analysis, papers should explicitly report the unit inventory, ROI definition, whitespace/punctuation attribution, unigram surprisal source, and whether cross-unit comparisons are made. Otherwise, results from the same model may not be reproducible due to differing preprocessing rules.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Fixes the unit selection problem as a core modeling choice rather than just preprocessing.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple inventories, metrics, and cross-validation, though limited to English/GPT-2.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical progression from formalization to solution to empirical validation.
- Value: ⭐⭐⭐⭐⭐ Significant methodological reminder for all researchers using LLM surprisal for cognitive modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](../../ICML2026/llm_pretraining/towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)
- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)
- [\[ACL 2026\] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)

</div>

<!-- RELATED:END -->
