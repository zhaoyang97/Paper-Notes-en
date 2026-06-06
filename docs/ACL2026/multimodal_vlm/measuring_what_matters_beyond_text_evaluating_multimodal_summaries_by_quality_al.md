---
title: >-
  [Paper Note] Measuring What Matters Beyond Text: Evaluating Multimodal Summaries by Quality, Alignment, and Diversity (MM-Eval)
description: >-
  [ACL 2026][Multimodal VLM][Multimodal Summarization] For the "Multimodal Summarization with Multimodal Output (MSMO)" task, the MM-Eval evaluation framework is proposed. It aggregates three sub-scores—textual quality (Op…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal Summarization"
  - "Evaluation Framework"
  - "OpenFActScore"
  - "MLLM-as-Judge"
  - "Truncated CLIP Entropy"
date: 2026-05-08
content_hash: ccfbe15ccedf18ec
---

# Measuring What Matters Beyond Text: Evaluating Multimodal Summaries by Quality, Alignment, and Diversity (MM-Eval)

**Conference**: ACL 2026  
**arXiv**: [2605.11693](https://arxiv.org/abs/2605.11693)  
**Code**: https://github.com/abidmeeraj/MM-Eval  
**Area**: Multimodal Evaluation / Summarization / Explainable Evaluation  
**Keywords**: Multimodal Summarization, Evaluation Framework, OpenFActScore, MLLM-as-Judge, Truncated CLIP Entropy

## TL;DR
For the "Multimodal Summarization with Multimodal Output (MSMO)" task, the MM-Eval evaluation framework is proposed. It aggregates three sub-scores—textual quality (OpenFActScore + G-Eval), cross-modal alignment (MLLM-as-Judge), and visual diversity (Truncated CLIP Entropy)—into a single score using weights learned via Ridge regression. On the mLLM-EVAL news benchmark, the Kendall $\tau$ correlation with human preferences improved from 0.041 (equal-weight baseline) to 0.374.

## Background & Motivation
**Background**: MSMO (Multimodal Summarization with Multimodal Output) requires systems to simultaneously output a text summary and a set of accompanying images. While MLLMs (GPT-4V / LLaVA / Qwen-VL) have significantly enhanced generation capabilities, evaluation remains confined to "modal silos" (Silo Effect)—using ROUGE (text) + Image Precision (images) + their cosine similarity. Each sub-score is calculated independently using unimodal metrics, failing to address whether the combined text and images form a faithful and useful summary.

**Limitations of Prior Work**: (1) ROUGE only considers n-grams and fails to detect semantic equivalence or factual errors; hallucinated summaries can still achieve high scores. (2) Image Precision assumes a "correct image set"; if a model selects semantically equivalent but different images, it receives a score of 0. (3) Global scoring either relies on simple linear regression like MMAE (still hindered by ROUGE) or uses mLLM-Eval to let GPT-4V provide an overall score (expensive and black-box, making it impossible to diagnose specific weaknesses).

**Key Challenge**: (a) **Explainability vs. Accuracy** — LLM-as-judge is accurate but opaque, whereas sub-metrics are explainable but correlate poorly with human evaluation. (b) **Reference-dependence vs. Generalization** — Most metrics require a reference summary, but reference standards are inconsistent across different domains.

**Goal**: To construct a framework that is (1) modular with three-dimensional sub-scores and one-dimensional aggregation; (2) entirely reference-weak (depending only on the source and system output) to facilitate cross-domain transfer; and (3) utilizes aggregation weights learned from human preferences to reflect the relative importance of each dimension.

**Key Insight**: The authors observe that human judgment of summary quality is essentially hierarchical—factual errors lead to immediate rejection (gatekeeper effect), and other dimensions only matter if the facts are correct. This **non-linear, threshold-based human judgment** cannot be captured by equal-weight averaging; weights must be learned.

**Core Idea**: Use "decompose-then-verify" atomic fact extraction for factual consistency, G-Eval for soft quality, MLLM-as-Judge for cross-modal alignment, and Truncated CLIP Entropy for visual diversity. Finally, learn the aggregation weights on mLLM-EVAL using Ridge regression.

## Method

### Overall Architecture
MM-Eval receives source documents $D = \{T_{source}, V_{source}\}$ and candidate summaries $S_{cand} = \{T_{gen}, V_{sel}\}$, and outputs a scalar $S_{final}$. The pipeline consists of three parallel pillars followed by two-stage Ridge regression: (1) Textual quality $S_{text}$ — sub-components $S_{fact}, S_{rel}, S_{coh}, S_{flu}$ are aggregated internally using Ridge ($\alpha=1.0$); (2) Cross-modal alignment $S_{relevance}$ — an MLLM provides a score of 1–5; (3) Visual diversity $S_{diversity}$ — TCE outputs log-entropy. The three pillars are normalized to $[0,1]$ before using Ridge ($\alpha=0.1$) to learn the final coefficient $\beta$, aiming to minimize the MSE with human overall scores. The entire process utilizes open-source models (Mistral-7B-Instruct, LLaVA-Mistral, ViT-B/32) with temperature = 0 to ensure reproducibility.

### Key Designs

1.  **Pillar 1: Text Quality = OpenFActScore (Hard Facts) + G-Eval (Soft Quality)**:
    - **Function**: Measures "factual correctness" and "linguistic fluency, coherence, and relevance" separately before merging them, preventing factual errors from being masked by high fluency.
    - **Mechanism**: An instruction-tuned LLM first decomposes the generated summary $T_{gen}$ into a set of atomic facts $A=\{a_1,\dots,a_m\}$. Each $a_i$ is independently judged by a second LLM to determine if it is supported by the source, yielding $S_{fact} = \frac{1}{|A|}\sum_i v_i$. G-Eval uses CoT + probability-weighted scoring for $S_{rel}, S_{coh}, S_{flu}$. Finally, $S_{text} = w_1 S_{fact} + w_2 S_{rel} + w_3 S_{coh} + w_4 S_{flu}$, with weights learned via Ridge (Facts 0.55, Coherence 0.29, Fluency 0.15, Relevance 0.02).
    - **Design Motivation**: Atomization makes the score immune to paraphrasing and length, shifting factual evaluation from n-gram recall to fact-level precision. Probability weighting in G-Eval addresses the variance issue where LLM scores oscillate between adjacent values.

2.  **Pillar 2: MLLM-as-a-Judge for Cross-modal Alignment**:
    - **Function**: Determines whether selected images semantically complement or supplement the text summary, bypassing the rigid requirement of Image Precision to match reference images exactly.
    - **Mechanism**: LLaVA-v1.6-mistral-7b-hf acts as the judge, performing CoT reasoning on each (text segment, candidate image) pair to output a 1–5 Likert score, which is then normalized to $[0,1]$. No additional sub-metrics are introduced within this pillar to maintain a simple structure for future model iterations.
    - **Design Motivation**: Human judgment of image-text alignment involves pragmatic reasoning (whether images "supplement" details not explicitly stated in text), which requires a powerful MLLM. CoT reasoning stabilizes the scoring variance.

3.  **Pillar 3: Truncated CLIP Entropy for Visual Diversity**:
    - **Function**: Penalizes redundant images (e.g., multiple photos of the same scene from different angles in protest news) and encourages informational diversity rather than just visual difference.
    - **Mechanism**: For the $k$ selected images, the CLIP embeddings $F$ are extracted, and the eigenvalues $\lambda_i$ of the empirical covariance $C$ are calculated. The top 20 eigenvalues are normalized into probabilities $p_i$, and the Von Neumann entropy $S_{diversity} = -\sum_{i=1}^k p_i \log(p_i)$ is computed. The volume occupied by the image set in the CLIP semantic space serves as the diversity measure.
    - **Design Motivation**: Pixel-level or pairwise distance metrics may perceive large differences between two photos of the same scene, even if they are semantically redundant. CLIP semantic space + spectral entropy penalizes only semantic overlap and is naturally reference-free, avoiding the need for large sample sizes like FID.

### Loss & Training
Two-stage Ridge regression is employed. Stage 1: On ~1500 human-evaluated samples from mLLM-EVAL, 4 internal weights for $S_{text}$ are learned using 5-fold CV ($\alpha=1.0$). Stage 2: The final coefficients $\beta$ for the three pillars are learned ($\alpha=0.1$) with the objective $\hat\beta = \arg\min_\beta \sum_i (\beta^T X_i - y_{human}^{(i)})^2$. The data is split 80/20 stratified by summarization system. Learned coefficients: $\beta_{text} = 2.7721$ (strongly positive), $\beta_{relevance} = 0.2256$ (slightly positive), $\beta_{diversity} = -0.4991$ (**negative**, because redundant image sets in this dataset often co-occur with weak text, acting as a confounder).

## Key Experimental Results

### Main Results: Comparison of MM-Eval and Baselines (mLLM-EVAL News Benchmark, 1562 Annotations)

| Evaluator | Kendall $\tau$ | Spearman $\rho$ | Pearson $r$ | $R^2$ | RMSE |
|---|---|---|---|---|---|
| Equal weights baseline | 0.041 | 0.058 | — | — | — |
| Text pillar only | 0.369 | 0.506 | — | — | — |
| Cross-modal pillar only (MLLM judge) | −0.085 | −0.110 | — | — | — |
| Diversity pillar only (TCE) | −0.089 | −0.124 | — | — | — |
| **MM-Eval (full, learned weights)** | **0.374** (CI [0.300, 0.444]) | **0.514** (CI [0.417, 0.597]) | **0.611** | **0.372** | **0.828** |

Stability of learned pillar weights (50 resamples): $w_{text} = 0.7572 \pm 0.043$ (dominant), $w_{relevance} = 0.070 \pm 0.022$, $w_{diversity} = 0.173 \pm 0.022$. Internal text weights: Facts 0.551, Coherence 0.287, Fluency 0.145, Relevance 0.017.

### Ablation Study: Impact of Removing Individual Pillars on Kendall $\tau$

| Configuration | Kendall $\tau$ | Change relative to full |
|---|---|---|
| Full MM-Eval | 0.3744 | — |
| w/o $S_{text}$ | −0.0835 | **−122%** (Becomes negative) |
| w/o $S_{relevance}$ | 0.1716 | −54% |
| w/o $S_{diversity}$ | 0.1123 | −70% |

### Key Findings
- **Factual consistency is a "gatekeeper function," not a simple linear contributor**: For human evaluations in consistency bin 1 (n=225), P(Overall≥4) = 0.000, while P(Overall≤2) = 0.933. In bin 5 (n=937), this flips to P(Overall≥4) = 0.819. The high $w_{fact}$ in Ridge regression effectively simulates this non-linear "one-strike-and-you're-out" factual veto.
- **Individual visual pillars show negative marginal correlation (−0.085 / −0.089), yet their removal drops $\tau$ by over half**—visual signals are conditional/interactional. They supplement information only after the text passes quality checks; independently, they are confounded by the fact that systems choosing weak images often produce weak text. This is a significant methodological finding: **Marginal correlation $\neq$ Joint contribution**.
- **News is text-dominant, but this cannot be generalized to all domains**: Supplemental experiments with 200 human annotations show that annotators still give high scores for image relevance (4.04) and diversity (3.89), suggesting that $w_{text} = 0.79$ reflects the "marginal contribution" structure rather than a lack of human interest in visuals.
- **Explainability + Transferability**: All pillars are reference-weak. For a new domain, one only needs to refit the three $\beta$ coefficients (requiring dozens to hundreds of human labels), while the underlying scorers remain unchanged.

## Highlights & Insights
- **"Gatekeeper functions + Jointly learned weights" represents a paradigm shift for evaluation frameworks**: Quantifying the intuition that "facts are a deal-breaker" into Ridge coefficients—rather than hard-coding if-then rules—maintains explainability while allowing for data-driven adjustments.
- **Negative coefficients can be reasonable results**: $\beta_{diversity} = -0.4991$ does not mean "diversity is harmful," but indicate that in this dataset's distribution, diversity and quality exhibit a spurious negative correlation. Reporting negative coefficients transparently and using ablation to prove that "removing it makes it worse" is a rigorous approach rarely seen in evaluation papers.
- **TCE for semantic diversity via spectral entropy**: Compared to pairwise distance or FID, spectral entropy looks only at the distribution of covariance eigenvalues. It reflects the semantic volume occupied by an image set while remaining robust to single-image perturbations, making it an underrated reference-free diversity metric.
- **Methodological takeaway**: An evaluation paper does not necessarily need a new model. Combining existing scorers through a "simple but effective" method like Ridge regression, supported by rigorous statistical analysis (CI / bootstrap / 50 resamples / stratified CV), can produce a strong baseline.

## Limitations & Future Work
- The authors acknowledge: (1) Verification was limited to the text-dominant news domain; image-dominant domains (product reviews / technical docs) might show opposite results. (2) A Kendall $\tau = 0.374$ is moderate, and human evaluation is still needed for ranking closely performing systems. (3) The negative marginal correlation of visual pillars might be partially due to proxy noise in TCE or the LLaVA judge.
- External limitations: (1) While the paper identifies "non-linear gatekeeper" behavior, Ridge is a linear aggregation. Using monotonic neural networks or piecewise-linear models with thresholds could directly model gatekeeping effects. (2) The sign discrepancy between `wdiversity=0.173` and `βdiversity=−0.499` stems from the difference between "normalized weight contribution" and "regression coefficients," which could be clarified more effectively. (3) ~1500 training samples for 7 weights is sufficient, but since they only cover 9 systems, the aggregation might overfit to the "style" of those specific systems.
- Future directions: Upgrade OpenFActScore to multimodal fact verification (utilizing $V_{source}$); introduce image-grounded factuality; and extend to more MSMO sub-tasks like dialogue summarization and report generation.

## Related Work & Insights
- **vs. MMAE (Zhu et al. 2018)**: MMAE also uses regression to aggregate ROUGE+IP+cos, but the underlying metrics are reference-based and semantically shallow; MM-Eval uses the same aggregation logic but upgrades each pillar to LLM/MLLM-based reference-weak metrics.
- **vs. mLLM-Eval (Zhuang et al. 2024)**: mLLM-Eval lets GPT-4V provide an overall score, which is accurate but a black box; MM-Eval decomposes evaluation into replaceable pillars, offering better explainability, modularity, and compatibility with open-source models.
- **vs. FActScore / OpenFActScore**: The authors reuse OpenFActScore for the factual sub-score. The contribution lies in solving the overlooked problem of how to merge factual sub-scores with other qualitative dimensions end-to-end.
- **vs. FID / Inception Score (Visual Diversity)**: TCE does not require massive samples to estimate distributions, making it more suitable for MSMO scenarios where each summary contains only 3–5 images.

## Rating
- Novelty: ⭐⭐⭐☆☆ The framework is an organic combination of existing scorers, but the "learned aggregation + revealing gatekeeper effect + marginal correlation $\neq$ joint contribution" insight provides a substantial contribution to evaluation methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ The statistical analysis is very solid (bootstrap CI / 50 resamples / 5-fold CV / 200 supplemental human labels / multi-model ablation), though limited to a single news dataset.
- Writing Quality: ⭐⭐⭐⭐☆ Mathematical notation is clear, and there are specific sections explaining controversial results like negative coefficients. The logical chain is complete.
- Value: ⭐⭐⭐⭐☆ Directly applicable for researchers in MSMO or multimodal generation evaluation; the "learned aggregation + gatekeeper analysis" paradigm is transferable to any multi-dimensional evaluation task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations](beyond_screenshots_evaluating_vlms_understanding_of_ui_animations.md)
- [\[ICML 2026\] MM-Snowball: Evaluating and Mitigating Hallucination Snowballing in Multimodal Multi-Turn Dialogue](../../ICML2026/multimodal_vlm/mm-snowball_evaluating_and_mitigating_hallucination_snowballing_in_multimodal_mu.md)
- [\[CVPR 2026\] Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection](../../CVPR2026/multimodal_vlm/learning_what_matters_prioritized_concept_learning_via_relative_error-driven_sam.md)
- [\[ACL 2026\] What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning](what39s_missing_in_screen-to-action_towards_a_ui-in-the-loop_paradigm_for_multim.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)

</div>

<!-- RELATED:END -->
