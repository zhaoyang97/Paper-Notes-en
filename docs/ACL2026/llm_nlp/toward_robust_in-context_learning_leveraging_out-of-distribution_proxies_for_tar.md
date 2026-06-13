---
title: >-
  [Paper Note] OOD Proxy Demonstration Retrieval Scheme for Robust In-Context Learning
description: >-
  [ACL 2026][LLM/NLP][In-Context Learning] By constructing dual proxies for the source and target domains and calculating their perplexity difference as an OOD score…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "In-Context Learning"
  - "OOD Robustness"
  - "Demonstration Retrieval"
  - "Distribution Shift"
  - "Proxy Estimation"
date: 2026-05-08
content_hash: 6ad16ae028adca90
---

# OOD Proxy Demonstration Retrieval Scheme for Robust In-Context Learning

**Conference**: ACL 2026  
**arXiv**: [2606.00014](https://arxiv.org/abs/2606.00014)  
**Code**: https://github.com/bort64/ood_code  
**Area**: LLM / NLP  
**Keywords**: In-Context Learning, OOD Robustness, Demonstration Retrieval, Distribution Shift, Proxy Estimation

## TL;DR

By constructing dual proxies for the source and target domains and calculating their perplexity difference as an OOD score, combined with Mahalanobis distance to constrain demonstration diversity, this method precisely selects source demonstrations aligned with the target distribution when target samples are inaccessible, enhancing the robustness of LLM in-context learning.

## Background & Motivation

**Background**: Large Language Models (LLMs) have demonstrated strong task adaptation capabilities through the In-Context Learning (ICL) paradigm, which guides model inference via a few input demonstrations without fine-tuning. Research has explored various demonstration selection strategies, including retrieval methods based on BM25, text encoder similarity, and model influence.

**Limitations of Prior Work**: The core bottleneck of ICL is the performance collapse caused by distribution shift. When demonstrations are severely mismatched with the target task distribution (especially in real-world applications where target domain data is unavailable), existing selection methods often rank samples based on source domain features, ignoring whether these samples align with the unknown target distribution. Direct similarity matching fails to identify which source samples are most helpful for the target task in OOD scenarios.

**Key Challenge**: The fundamental dilemma of demonstration retrieval is two-fold: (1) the target domain distribution is completely invisible, making it impossible to directly measure the fit between source samples and the target; (2) relying solely on similarity ranking falls into the "spurious correlation" trap—highly similar samples may not adapt to distribution shifts and might even reinforce the model's over-adaptation to the source domain.

**Goal**: (1) Construct a reliable target distribution approximator without any access to the target domain; (2) quantify the "target affinity" of each source sample via this approximator; (3) balance similarity and diversity during selection to ensure retrieved demonstrations are close to target features and have sufficient representation coverage.

**Key Insight**: Inspired by OOD detection literature, the authors observe that pre-trained LLMs implicitly encode broad linguistic and factual knowledge, acting as a weak proxy for the target domain. By performing instruction tuning on source data to obtain a "source proxy" while keeping the original model as a "target proxy," the predicted probability ratio (log-difference of perplexity) between the two on the same input reflects the sample's adaptation to the target distribution.

**Core Idea**: Use the perplexity ratio, a concise yet theoretically justifiable metric, to approximate the unobservable target distribution, guiding the selection process to avoid source domain bias caused by over-reliance on similarity.

## Method

### Overall Architecture

The DOPA workflow consists of three stages:

1.  **Proxy Construction**: Based on accessible source domain data, an LLM is instruction-tuned to obtain the source proxy; the original un-tuned LLM is retained as the target proxy.
2.  **OOD Scoring**: Perplexity is calculated for each source sample under both proxies, and the log-difference is used as an OOD score to filter a candidate set of $k$ samples most likely aligned with the target distribution.
3.  **Demonstration Retrieval**: A demonstration pool is initialized from the candidate set based on similarity, then expanded under Mahalanobis distance constraints to ensure both similarity and diversity.

### Key Designs

1.  **Dual-Proxy OOD Scoring**:

    -   **Function**: Estimates the affinity of source samples to the target distribution when the target domain is invisible.
    -   **Mechanism**: The source proxy is obtained via instruction tuning on labeled source data, while the target proxy retains the prior knowledge of the pre-trained LLM. For the same sample $x$, the source proxy tends toward higher perplexity (due to over-adaptation to specific source features), while the target proxy maintains a "universal" perspective. If sample $x$ has high perplexity under the source proxy but low perplexity under the target proxy, it likely represents a "source domain outlier" that may perform better in the unknown target domain. The OOD score is defined as $S(x) = \log PPL_{target}^{proxy}(x) - \log PPL_{source}^{proxy}(x)$, where lower scores indicate better adaptation.
    -   **Design Motivation**: Compared to using a uniform distribution as a target proxy (which leads to looser error bounds), utilizing the pre-trained LLM itself as a target proxy leverages universal knowledge while avoiding overly strong assumptions. Theoretically, Theorem 1 guarantees that the approximation deviation is limited within the range of $(\epsilon_t/m_t + \epsilon_s/m_s)$.

2.  **Mahalanobis Distance Diversity Constraint**:

    -   **Function**: Introduces a global diversity metric in the selected demonstration set to avoid redundancy and short-text bias caused by similarity ranking.
    -   **Mechanism**: Initialize by selecting the top $C$ candidates based on similarity to the test sample. Subsequently, calculate the average Mahalanobis distance for all sample pairs in the candidate set: $Div = \frac{2}{|\mathcal{D}_{demo}|(|\mathcal{D}_{demo}|-1)}\sum_{i<j}\sqrt{\mathbf{D}_{ij}^T \Sigma^{-1} \mathbf{D}_{ij}}$, where $\Sigma$ is the empirical covariance matrix of sample representations. Subsequent candidates are retained only if their addition does not decrease the overall diversity.
    -   **Design Motivation**: Purely perplexity-based OOD scoring has risks: perplexity measures token-level fluency and favors short text or high-frequency patterns, leading to the retrieval of similar short samples. Adding diversity constraints forces the algorithm to expand the representation space of demonstrations while maintaining similarity.

3.  **Target-free Demonstration Retrieval Algorithm**:

    -   **Function**: Integrates OOD scoring and diversity constraints to execute the full selection process.
    -   **Mechanism**: First, a single pass over the source data filters a candidate set $\hat{\mathcal{D}}_S$ of size $k$ using OOD scores. Then, candidates are sorted by similarity to the test sample, and the set is initialized with the top $N \times |Y|$ samples. Finally, remaining candidates are checked; if adding a candidate does not decrease diversity, it is included until the target size is reached.
    -   **Design Motivation**: The two-layer filtering mechanism has clear roles: OOD scoring handles "distribution shift" (which samples belong to the target neighborhood), and diversity constraints handle "representation coverage" (minimizing redundancy).

## Key Experimental Results

### Main Results

DOPA's effectiveness was tested across 5 LLM scales and 3 task families:

| LLM Model | Avg Accuracy (%) | vs. Random | vs. DrICL |
| :--- | :--- | :--- | :--- |
| GPT2-XL | 48.76 | +10.3% | +2.1% |
| LLaMA3.2-3B | 59.29 | +5.7% | +5.2% |
| Gemma2-2B | 60.12 | +2.4% | +4.3% |
| Qwen3-1.7B | 64.93 | +1.1% | +1.4% |
| LLaMA3.1-8B | 61.08 | +1.7% | +1.0% |

Performance across task types:

| Task Family | Description | DOPA Avg Acc (%) | Gain vs. Best Baseline |
| :--- | :--- | :--- | :--- |
| Sentiment (SA/TD/SST) | Sentence-level polarity; includes ID and OOD sets | 48.76 | +0.55 |
| NLI | Entailment; includes implicit and adversarial perturbations | 38.04 | +1.05 |
| Entailment (Adv/Toxigen) | Complex NLI with severe distribution shifts | 51.82 | +0.65 |

### Ablation Study

| Configuration | Accuracy Drop after Removal (%) |
| :--- | :--- |
| Full DOPA | Baseline |
| w/o OOD Scoring (using random) | -2.3% to -4.1% |
| w/o Mahalanobis Diversity (using Euclidean) | -1.2% to -2.8% |
| w/o Diversity Constraint (similarity only) | -1.5% to -3.2% |
| Source Proxy Only (no target comparison) | -3.4% to -5.7% |

### Key Findings

-   **Criticality of OOD Scoring**: Removing the OOD scoring module caused the most significant performance drop (-2.3% to -4.1%), proving the perplexity ratio proxy is the core of DOPA. In smaller models, OOD scoring contributed approximately 65% of the total gain.
-   **Incremental Effect of Diversity**: Mahalanobis distance constraints provide an additional 1.2% to 2.8% improvement over simple Euclidean diversity.
-   **Impact of Shift Severity**: On mild OOD tasks (e.g., in-source SST), DOPA improved by only 0.1% over baselines; on severe OOD tasks (e.g., Toxigen), the gain reached 3.2%, confirming the algorithm targets drastic shifts.
-   **Necessity of Target Proxy**: Using a uniform distribution instead of a target proxy nullifies the OOD scoring effectiveness (-3.4% to -5.7%).

## Highlights & Insights

-   **Ingenious "Something from Nothing" Idea**: The paper's most impressive aspect is that since the target domain is invisible, it uses "relativity"—inferring target affinity indirectly through the difference between source and target proxies on identical inputs. This avoids direct assumptions about the target distribution.
-   **Theory-Practice Synergy**: The bounded proxy error analysis in Theorem 1 is not just formal elegance; it practically guides each step of the proxy design.
-   **Multi-level Constraint Fusion**: OOD scoring manages distribution while diversity constraints manage representation. This two-layer pipeline solves the core problem while maintaining explainability.

## Limitations & Future Work

-   **Proxy Assumption Boundaries**: The paper assumes source instruction tuning fully represents the source distribution, but if the source domain is a mixture of multiple distributions, the source proxy may be unstable.
-   **Computational Overhead**: OOD scoring requires passing source data through two models, costing roughly 2-3x more than standard retrieval. This may be a bottleneck at the scale of millions of source samples.
-   **Generalization to Multilingual/Multimodal**: Experiments only covered English NLP tasks. Validity in non-English or multimodal settings remains to be verified.
-   **Specific Improvement Directions**: (1) Designing online or incremental OOD score update mechanisms; (2) using knowledge distillation to reduce proxy model size; (3) extending the method to cross-lingual or cross-modal settings.

## Related Work & Insights

-   **vs. Traditional Similarity Retrieval (BM25/SBERT)**: Traditional methods find the most similar samples in the source domain. DOPA's key difference is the addition of OOD sensitivity—similarity does not equal adaptation.
-   **vs. Influence Functions**: These calculate the impact of demonstrations on model gradients, requiring internal model access. DOPA is gradient-free and uses only perplexity, making it suitable for black-box or API scenarios.
-   **vs. Data Augmentation (Rewrite/Augmentation)**: Augmentation attempts to modify source samples to fit the target. DOPA focuses on "selection" rather than "modification."
-   **Insight**: When the true data distribution is unavailable, utilizing the relative behavior of a "control group" (e.g., tuned vs. un-tuned) to infer distribution characteristics is a versatile approach.

## Rating

-   **Novelty**: ⭐⭐⭐⭐⭐ Demonstration retrieval under "no-target" constraints is a new problem setting, and the dual-proxy OOD scoring is elegant and theoretically grounded.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 LLM scales, 3 task families, multiple baselines, and ablation studies. Lacks hyperparameter sensitivity analysis.
-   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logic; forms a complete story from problem definition to methodology and experimentation.
-   **Value**: ⭐⭐⭐⭐⭐ Addresses a real-world pain point (invisible target domain) with a reproducible method and open-source code, suitable for production-grade ICL systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[AAAI 2026\] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](../../AAAI2026/llm_nlp/lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)
- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)
- [\[ICLR 2026\] In-Context Algebra](../../ICLR2026/llm_nlp/in-context_algebra.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities](think_in_sentences_explicit_sentence_boundaries_enhance_language_model39s_capabi.md)
- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)
- [\[ACL 2026\] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[ACL 2026\] 等等，还有出路：一个对话脱轨预测的决策机制](wait_theres_a_way_out_a_decision_mechanism_for_forecasting_conversational_derail.md)

</div>

<!-- RELATED:END -->
