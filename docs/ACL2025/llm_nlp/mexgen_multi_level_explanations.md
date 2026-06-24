---
title: >-
  [Paper Note] MExGen: Multi-Level Explanations for Generative Language Models
description: >-
  [ACL 2025][LLM (Other)][Explainability] The MExGen framework is proposed to map text outputs of generative models to real values via a scalarizer, perform multi-granularity linguistic segmentation, and apply linear-complexity attribution algorithms (C-LIME/L-SHAP). It provides more faithful input attribution explanations for context-driven text generation (summarization, QA) than PartitionSHAP and LLM self-explanations.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Explainability"
  - "input attribution"
  - "LIME"
  - "SHAP"
  - "scalarizer"
  - "multi-level explanations"
date: 2026-05-08
content_hash: 6504d7a1e9bf73cd
---

# MExGen: Multi-Level Explanations for Generative Language Models

**Conference**: ACL 2025  
**arXiv**: [2403.14459](https://arxiv.org/abs/2403.14459)  
**Code**: [GitHub (ICX360)](https://github.com/IBM/ICX360)  
**Area**: LLM / NLP  
**Keywords**: Explainability, input attribution, LIME, SHAP, scalarizer, multi-level explanations

## TL;DR

The MExGen framework is proposed to map text outputs of generative models to real values via a scalarizer, perform multi-granularity linguistic segmentation, and apply linear-complexity attribution algorithms (C-LIME/L-SHAP). It provides more faithful input attribution explanations for context-driven text generation (summarization, QA) than PartitionSHAP and LLM self-explanations.

## Background & Motivation

**Background**: LLMs are increasingly deployed in context-driven tasks (e.g., meeting minutes, legal document summarization, medical QA). Users need to know which parts of the output are grounded in which parts of the input, which is critical for high-stakes decision-making. LIME and SHAP are widely used perturbation-based attribution methods heavily applied in text classification.

**Limitations of Prior Work**: Extending perturbation attribution to generative LLMs faces three technical challenges: (1) Outputs are text rather than real numbers (LIME/SHAP require real-valued functions); (2) LLM inference is computationally expensive, preventing the large number of model queries typically used in classification tasks; (3) Input texts are often long (e.g., entire papers or news articles), where too fine-grained attributions are neither interpretable nor computationally feasible. Existing tools like PartitionSHAP require selecting a single output token to explain, while CaptumLIME requires access to output logits.

**Key Challenge**: Faithful explanation requires a sufficient number of perturbation samples, but LLM inference costs limit the allowed number of queries; fine-grained attribution is more precise but computationally more expensive and harder to interpret.

**Goal**: How to provide faithful and interpretable multi-level attribution explanations for context-driven generation tasks with long text inputs under a limited LLM query budget?

**Key Insight**: (1) Use a scalarizer to map text outputs to real numbers, making classic attribution methods applicable; (2) Exploit the hierarchical structure of natural language (paragraph $\to$ sentence $\to$ phrase $\to$ word) to perform coarse-to-fine multi-level attribution; (3) Design linear-complexity attribution algorithms to control the number of model queries.

**Core Idea**: Efficiently extend classic perturbation-based attribution methods to context-driven generation tasks of generative LLMs through a scalarizer, multi-level linguistic segmentation, and linear-complexity attribution.

## Method

### Overall Architecture

Given a generative model $f$, an original input $x^o$, and the generated target output $y^o = f(x^o)$, MExGen segments the input into linguistic units $x_1, ..., x_d$. By perturbing these units and quantifying the output variation as a real number using a scalarizer $S$, MExGen assigns an attribution score $\xi_s$ to each unit. The framework supports iterative refinement from coarse-grained (sentence) to fine-grained (word) levels.

### Key Designs

1. **Scalarizer: Mapping Text Outputs to Real Numbers**:

    - **Function**: Defines a function $S$ that maps the generated text to a real number, enabling the application of attribution algorithms.
    - **Mechanism**: Two types of scalarizers are proposed. **With logit access**: Log Prob scalarizer $S(x; y^o, f) = \frac{1}{\ell}\sum_{t=1}^{\ell} \log p(y_t^o | y_{<t}^o, x; f)$, which computes the average log probability of the target output under the perturbed input. **With text-only access**: Computes the similarity between the generated text $y=f(x)$ and the target $y^o$, using metrics like Sim (cosine similarity of sentence embeddings), BERT (BERTScore), BART (BARTScore), Log NLI (natural language inference log-odds), etc.
    - **Design Motivation**: Many LLMs only provide API access without access to logits (e.g., GPT-4). Text-based scalarizers allow MExGen to operate in pure black-box scenarios. Empirically, the BERT scalarizer even outperforms Log Prob in user-perceived faithfulness, indicating that text-only access does not necessarily compromise explanation quality.

2. **Linear-Complexity Attribution Algorithms (C-LIME and L-SHAP)**:

    - **Function**: Efficiently computes attribution scores under a limited model query budget.
    - **Mechanism**: **C-LIME** introduces two key modifications to LIME: (a) setting the number of perturbations as a fixed multiple of the number of units $n = c \cdot d$ (where $c=5$ or $10$), avoiding the default thousands of queries; (b) limiting the number of simultaneously perturbed units $K$ (where $K=2-3$) to keep perturbations concentrated near the original input. **L-SHAP** restricts SHAP to computing local Shapley values within a neighborhood of radius $M$. The query complexity of both algorithms scales linearly with the number of units $d$.
    - **Design Motivation**: Standard LIME generates thousands of perturbed samples by default (independent of $d$), which is unacceptable for LLM inference costs. C-LIME limits the number of simultaneously perturbed units, making perturbed inputs closer to the original input. Existing theoretical work indicates that this improves attribution faithfulness.

3. **Multi-Level Linguistic Segmentation and Iterative Refinement**:

    - **Function**: Starts attribution computation at a coarse grain and refines only the most important units to a fine grain.
    - **Mechanism**: Uses spaCy to segment the input into a hierarchy of paragraphs $\to$ sentences $\to$ phrases $\to$ words. It first computes attribution scores at the sentence level, then uses Algorithm 1 to select sentences whose normalized scores exceed a threshold $\phi$ and rank in the top $k$ for phrase/word-level refinement. A custom dependency parser algorithm segments sentences into meaningful phrases.
    - **Design Motivation**: Step-by-step focusing operates similarly to binary search, avoiding wasting model queries on a large number of unimportant fine-grained units. For example, a 20-paragraph article only requires phrase-level refinement on the top-3 most important sentences.

### Loss & Training

MExGen is a pure inference-time method and does not involve training. C-LIME fits a linear model using weighted least squares regression: $\xi = \arg\min_w \sum_{i=1}^n \pi(z^{(i)})(w^T z^{(i)} - S(x^{(i)}; y^o, f))^2$, without regularization to keep all units rankable.

## Key Experimental Results

### Main Results

AUPC (Area Under the Perturbation Curve, higher is better, cut off at 20% tokens):

| Dataset + Model | C-LIME | L-SHAP | LOO | P-SHAP |
|-------------|--------|--------|-----|--------|
| XSUM / DistilBART | **13.6** | 13.8 | 13.1 | 9.4 |
| CNN/DM / Llama-3-8B | **26.4** | 26.3 | 26.1 | 22.1 |
| SQuAD / Flan-T5-Large | **62.7** | 61.1 | 60.2 | 58.8 |
| SQuAD / Llama-3-8B | 56.4 | **57.0** | 54.9 | 38.5 |

### Ablation Study

| Dataset + Model | C-LIME | L-SHAP | LOO | Self-Explanation |
|------------|--------|--------|-----|------------------|
| XSUM / Granite-3.3 (Prob) | **18.9** | 19.0 | 18.9 | 9.5 |
| CNN/DM / Granite-3.3 (Prob) | 17.3 | **17.4** | 16.9 | 7.1 |
| XSUM / DeepSeek-V3 (BART) | **12.7** | 12.3 | 12.3 | 10.5 |
| CNN/DM / DeepSeek-V3 (BART) | **14.1** | 14.0 | 13.5 | 13.5 |

### Key Findings

- **MExGen consistently outperforms PartitionSHAP**: MExGen achieves higher AUPC across almost all dataset-model combinations (with the only exception being Flan-UL2+CNN/DM where P-SHAP is slightly better at high token perturbation ratios, though MExGen remains better in the top 5%).
- **MExGen even beats P-SHAP with mismatched scalarizers**: MExGen using the BERT scalarizer (which requires no logits) outperforms P-SHAP using Log Prob, demonstrating the superiority of MExGen's core attribution algorithm.
- **LLM self-explanation is inferior to systematic attribution**: Even for strong models like DeepSeek-V3, ranking-based self-explanations are less faithful than MExGen. The gap is particularly massive under the Log Prob scalarizer (where the AUPC differ by 2x).
- **User study favors the BERT scalarizer**: 57% of participants found the BERT scalarizer more faithful than Log Prob (with only 35% preferring Log Prob), and 64% preferred BERT overall. Text-only access is not necessarily inferior to logit access.
- **C-LIME vs L-SHAP**: Both perform almost on par in automatic evaluation, but C-LIME significantly outperforms L-SHAP in the user study ($p=0.011$).

## Highlights & Insights

- **Elegant and practical scalarizer concept**: Decoupling the measurement of output changes into a pluggable module allows the framework to adapt to both logit-based and API-only (black-box) scenarios. This abstraction can be transferred to any scenario requiring attribution on generative models.
- **Simple yet effective modifications in C-LIME**: Restricting the number of perturbations to $O(d)$ and keeping the simultaneous perturbation size constant controls computation costs while improving faithfulness. This trick can be applied to all LIME-based explanation methods.
- **Multi-level attribution analogous to binary search**: Progressively focusing from the sentence level down to phrase and word levels balances efficiency and accuracy. This strategy functions essentially as an adaptive-resolution approach.

## Limitations & Future Work

- **Only local explanations are provided**: It can only explain individual generations and cannot reveal the global behavioral patterns of the model.
- **Scalarizer choice is task-dependent**: The optimal scalarizer varies across different task-model combinations, and an automated selection mechanism is lacking.
- **Simple deletion-based perturbation**: The paper notes that more complex perturbation strategies (e.g., using masked language models for replacement) might yield better results but would introduce additional complexity.
- **Scalability**: While the multi-level strategy reduces the number of queries, efficiency bottlenecks may still arise for very long documents (e.g., entire books).

## Related Work & Insights

- **vs PartitionSHAP**: P-SHAP produces independent attributions for each output token, requiring the user to select an output token to explain. MExGen aggregates the entire output via a scalarizer, generating a more intuitive, single attribution score.
- **vs CaptumLIME**: CaptumLIME uses standard LIME sampling (thousands of queries) and must access logits. C-LIME restricts sampling to $O(d)$ times and supports text-only access.
- **vs LLM Self-Explanation**: Self-explanations are convenient but unfaithful—models might generate plausible-sounding explanations that do not reflect their internal behavior. MExGen provides more reliable behavioral analysis through systematic perturbation.
- **vs ContextCite**: Independent concurrent work that also extends LIME to generative models, but operates only at a single granularity, lacking multi-level capabilities.

## Rating

- Novelty: ⭐⭐⭐ The framework combines existing techniques (LIME, multi-level analysis, scalarizers), but does so in a systematic and effective manner.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 3 datasets, 5 LLMs, automatic evaluation, user studies, cross-scalarizer evaluation, and multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, thorough experimental design, and an open-source toolkit.
- Value: ⭐⭐⭐⭐ Provides a practical tool for the real-world explainability of generative LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](generative_psycholexical_approach_for_constructing_value.md)
- [\[ACL 2025\] Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation](self-foveate_enhancing_diversity_and_difficulty_of_synthesized_instructions_from.md)
- [\[ACL 2025\] Multi-Attribute Steering of Language Models via Targeted Intervention](multi_attribute_steering.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)

</div>

<!-- RELATED:END -->
