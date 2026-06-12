---
title: >-
  [Paper Note] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models
description: >-
  [ACL 2026][Interpretability][Grammar encoding] This paper utilizes the Gradiend gradient-based interpretability method to investigate whether language models predict German definite articles (der/die/das/den/dem/des) bas…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Grammar encoding"
  - "Memorization vs Generalization"
  - "German articles"
  - "Gradient interpretability"
  - "Gradiend"
date: 2026-05-08
content_hash: 72a19e0e14352aa9
---

# Understanding or Memorizing? A Case Study of German Definite Articles in Language Models

**Conference**: ACL 2026  
**arXiv**: [2601.09313](https://arxiv.org/abs/2601.09313)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Grammar encoding, Memorization vs Generalization, German articles, Gradient interpretability, Gradiend

## TL;DR

This paper utilizes the Gradiend gradient-based interpretability method to investigate whether language models predict German definite articles (der/die/das/den/dem/des) based on abstract grammatical rules or surface memorization. Results find that models rely at least partially on memorized associations rather than strict rule encoding.

## Background & Motivation

**Background**: Modern language models perform nearly perfectly on grammatical consistency tasks, but whether their internal mechanisms encode abstract grammatical rules (such as systematic relations of gender, number, and case) or merely memorize high-frequency token-context associations remains a core question in interpretability research.

**Limitations of Prior Work**: Existing probing research can only prove that grammatical features are "recoverable from model representations" but cannot prove these features "causally drive" model predictions. Therefore, high probing accuracy is not equivalent to a model truly understanding grammatical rules.

**Key Challenge**: The German definite article system provides an ideal test scenario—the same article can correspond to multiple gender-case combinations (e.g., "der" can be masculine nominative, feminine dative, or feminine genitive). This syncretism allows researchers to distinguish: if the model is based on rule encoding, then interventions targeting specific gender-case transitions should only affect that grammatical dimension; if based on memorization, interventions will spill over to unrelated grammatical settings sharing the same surface article.

**Goal**: To verify whether German definite article prediction is rule-driven or memory-driven through gradient intervention experiments.

**Key Insight**: Utilize the Gradiend method—a gradient-based encoder-decoder interpretability framework—to learn parameter update directions for specific gender-case article transitions, then test whether these update directions generalize to unrelated grammatical settings.

**Core Idea**: If the update direction learned for "masculine nominative der → feminine nominative die" simultaneously affects unrelated grammatical settings such as "feminine dative der → feminine dative die", it indicates that the model relies on surface memorization rather than abstract rules in these positions.

## Method

### Overall Architecture

The German definite article system (3 genders × 4 cases = 12 cells, 6 article forms) is selected as a controlled experimental system. For each article transition between gender-case cells (e.g., masculine nominative ↔ feminine nominative), Gradiend is used to learn a one-dimensional feature direction. Analysis is then conducted from three perspectives: encoder value distribution, changes in article probability post-intervention, and overlap of update directions in parameter space.

### Key Designs

1.  **Gradiend Gradient Feature Learning**:
    -   **Function**: Learn a compressed one-dimensional feature direction for a specific article transition.
    -   **Mechanism**: Given an article transition between two gender-case cells $z_1, z_2$ (e.g., masculine nominative der ↔ feminine nominative die), collect gradients for both directions (factual target gradient $\nabla^F$ and alternative target gradient $\nabla^A$). Train an encoder to compress the gradient difference $\nabla^\Delta = \nabla^F - \nabla^A$ into a scalar $h \in [-1, 1]$ (+1 corresponds to one direction, -1 to the other), and an accompanying decoder to reconstruct the gradient from the scalar. Training data from all non-target cells serve as identity pairs (factual = alternative), forcing $h \approx 0$.
    -   **Design Motivation**: A one-dimensional bottleneck ensures that the most dominant update direction is learned, while identity pair constraints ensure the learned direction is specific to the target transition rather than a global perturbation.

2.  **Three Analytical Perspectives**:
    -   **Function**: Determine rule encoding vs. memorization from complementary angles.
    -   **Mechanism**: (a) Encoder analysis—check whether gradients of non-target cells are also encoded into non-zero values (which should be zero under rule prediction); (b) Probability intervention—after applying the learned direction to model parameters, check whether article probability changes are restricted to target cells (Local Response, LR), systematically extend to the same gender/case dimension (Grammar-based Response, GR), or spill over to unrelated cells sharing surface articles (Surface-based Overlap, SO); (c) Top-k parameter overlap—compare whether the most important parameters for different transitions overlap significantly.
    -   **Design Motivation**: These three analyses provide complementary evidence from representation space, functional behavior, and parameter space, as any single perspective might have confounding factors.

3.  **Learning Rate Selection & Language Modeling Maintenance**:
    -   **Function**: Ensure intervention effects are not pseudo-signals caused by model degradation.
    -   **Mechanism**: When applying interventions, scan multiple learning rates $\alpha$ and retain only candidates that maintain over 99% of the language modeling score on neutral datasets. Select the $\alpha^*$ that maximizes the target article probability on the target dataset. SuperGLEBer benchmark scores are also reported to confirm that the model's overall capability is not impaired.
    -   **Design Motivation**: Excessive parameter updates might change predictions by destroying language modeling capability rather than reflecting true grammatical mechanisms.

### Loss & Training

Gradiend is trained using the MSE reconstruction loss $\|\text{dec}(\text{enc}(\nabla^A W_m)) - \nabla^\Delta W_m\|_2^2$, using the alternative target gradient as input (since factual target gradients are typically near zero and not sufficiently informative).

## Key Experimental Results

### Main Results

19 Gradiend variants were trained on 6 models (GermanBERT, GBERT, ModernGBERT, EuroBERT, GermanGPT-2, LLaMA).

| Model | Encoder Correlation | Spill-over Phenomenon |
| :--- | :--- | :--- |
| GermanBERT | 90-98% | Significant: der→die intervention simultaneously affects fem. dative/genitive |
| GBERT | 95-99% | Significant: similar pattern |
| ModernGBERT | 81-95% | Moderate spill-over |
| EuroBERT | 50-73% | Weaker but still significant |
| GermanGPT-2 | 51-71% | Inconsistent patterns |
| LLaMA | 50-67% | Least spill-over, potentially reflecting a trend for larger models |

### Ablation Study

| Analysis | Key Findings | Description |
| :--- | :--- | :--- |
| Probability Intervention | Spill-over patterns (SO) occur frequently | der→die intervention increased die probability for fem. dative/genitive (which also use der) |
| Top-1000 Parameter Overlap | 40-60% overlap within the same article group | Different transitions share many parameters, exceeding the random baseline |
| Cross-article Group Overlap | 20-30% | Notable parameter overlap even between transitions of different articles |
| Control Group | Baseline-level overlap | Gradiend trained with shuffled data shows only baseline levels of overlap |

### Key Findings

-   **Widespread Spill-over**: Parameter update directions learned for specific gender-case transitions significantly affect unrelated cells sharing the same surface article, which is inconsistent with the pure rule-encoding hypothesis.
-   **High Parameter Overlap**: The most important parameters for different article transitions overlap heavily (40-60%), far exceeding random baselines, suggesting the model does not allocate independent parameter subsets for each grammatical relation.
-   **Encoder Models are More "Memetic"**: Encoder models like GermanBERT and GBERT show the strongest spill-over, possibly because bidirectional attention makes it easier for the model to exploit surface co-occurrence associations.
-   **Larger Models May Memorize Less**: LLaMA (3.2B) is the only model that did not demonstrate spill-over in certain key cells, hinting that larger models may tend toward more abstract encoding.
-   **Unimpaired Language Modeling Capability**: SuperGLEBer scores remained basically unchanged before and after intervention (70.7 → 70.1-70.2), confirming the effects were not caused by model degradation.

## Highlights & Insights

-   **Ingenious Experimental Design**: Leveraging the syncretism of German articles (where "der" can be masculine nominative or feminine dative/genitive) to construct a natural control experiment. This "one surface form, different deep grammatical functions" setup is replicable in other languages.
-   **Causal Evidence**: Moves beyond the "correlation" evidence of traditional probing by providing "causal" evidence through parameter intervention—the pattern of changes in model prediction directly reveals internal encoding mechanisms.
-   **Comprehensive Three-Perspective Analysis**: Encoder analysis, probability intervention, and parameter overlap provide consistent conclusions, enhancing the persuasiveness of the evidence.

## Limitations & Future Work

-   Only one grammatical phenomenon (German definite articles) was studied; whether the conclusions generalize to the grammatical systems of other morphologically rich languages remains to be verified.
-   Decoder models (GPT-2, LLaMA) require custom MLM prediction heads, which may introduce noise affecting the conclusions.
-   The one-dimensional bottleneck of Gradiend may oversimplify actual multidimensional grammatical encoding.
-   The extent to which article-noun co-occurrence frequency in training data affects memorization vs. generalization was not explored.

## Related Work & Insights

-   **vs. Linear Probing**: Probing can only prove that information "exists" in representations, whereas Gradiend interventions can prove that information "causally drives" predictions, providing stronger evidence.
-   **vs. Finlayson et al. (2021)**: While they modified internal representations to study subject-verb agreement, this paper modifies model parameters to study article prediction; the methodologies are complementary but operate at different levels (representation space vs. parameter space).

## Rating

-   Novelty: ⭐⭐⭐⭐ The experimental design utilizing German article syncretism is very ingenious.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 19 variants across 6 models, three-perspective analysis, and rigorous statistical testing.
-   Writing Quality: ⭐⭐⭐⭐ Clear, though there is a certain barrier for non-German speakers.
-   Value: ⭐⭐⭐⭐ Provides important causal evidence for the memorization vs. rule debate in LM grammatical encoding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse](interpretable_semantic_gradients_in_ssd_a_pca_sweep_approach_and_a_case_study_on.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)

</div>

<!-- RELATED:END -->
