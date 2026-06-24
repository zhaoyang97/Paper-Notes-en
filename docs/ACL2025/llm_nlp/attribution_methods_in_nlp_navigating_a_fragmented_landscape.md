---
title: >-
  [Paper Note] Attribution Methods in NLP: Navigating a Fragmented Landscape
description: >-
  [ACL 2025][LLM (Other)][Attribution methods] This paper presents a comprehensive survey and systematic comparison of attribution methods in NLP. Addressing the issue of fragmented evaluation metrics and lack of fair comparisons in this field, it proposes a unified evaluation framework and reveals the applicability dynamics of different attribution methods across various tasks and model architectures.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Attribution methods"
  - "explainable NLP"
  - "feature importance"
  - "model explanation"
  - "evaluation benchmark"
date: 2026-05-08
content_hash: 63cc68f7d93b465e
---

# Attribution Methods in NLP: Navigating a Fragmented Landscape

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Attribution methods, explainable NLP, feature importance, model explanation, evaluation benchmark

## TL;DR
This paper presents a comprehensive survey and systematic comparison of attribution methods in NLP. Addressing the issue of fragmented evaluation metrics and lack of fair comparisons in this field, it proposes a unified evaluation framework and reveals the applicability dynamics of different attribution methods across various tasks and model architectures.

## Background & Motivation

**Background**: Attribution methods aim to explain the prediction behavior of NLP models by quantifying the "contribution" of each token or feature in an input text to the model output. The main categories include: gradient-based methods (e.g., Saliency Maps, Integrated Gradients), perturbation-based methods (e.g., LIME, SHAP), attention-based methods (which directly use attention weights for attribution), and internal mechanism-based methods (e.g., Probing, Layer-wise Relevance Propagation).

**Limitations of Prior Work**: Research on attribution methods suffers from severe fragmentation issues: (1) each method uses different evaluation metrics and datasets, making fair comparison difficult; (2) there is a lack of consensus on the "ground truth" of evaluation—what constitutes a truly correct attribution? Human-annotated importance? Changes in model performance after deletion? Or adversarial robustness? (3) The computational costs of different methods vary drastically, but few studies report both performance and efficiency; (4) in the LLM era, it remains questionable whether attribution methods traditionally validated on BERT/RoBERTa are still effective for GPT/LLaMA.

**Key Challenge**: The evaluation standards of attribution methods themselves lack standardization—different evaluation protocols can yield contradictory conclusions, making it difficult for researchers to choose appropriate attribution methods. This fragmentation seriously hinders the progress of explainable NLP.

**Goal**: (1) Build a unified experimental framework that covers major attribution methods and evaluation protocols; (2) reveal the strengths and weaknesses of different methods through large-scale comparative experiments; (3) provide a selection guide of attribution methods for NLP practitioners.

**Key Insight**: Starting from "evaluation protocols," the authors first organize and standardize existing evaluation methods, and then conduct fair comparisons under a unified standard.

**Core Idea**: By establishing a unified set of evaluation protocols and a standardized experimental pipeline, the study breaks the fragmentation of attribution method evaluation, producing reproducible and comparable systematic conclusions.

## Method

### Overall Architecture
The unified evaluation framework consists of three tiers: (1) Method Implementation Layer—standardizing the implementation of 8 major categories of attribution methods; (2) Evaluation Protocol Layer—integrating 5 main evaluation protocols; (3) Analysis Layer—multidimensional comparative analysis across methods, models, and tasks. The covered models include various architectures and scales such as BERT, RoBERTa, GPT-2, and LLaMA-2; the covered tasks include sentiment classification, natural language inference, fact-checking, toxicity detection, etc.

### Key Designs

1. **Unified Attribution Toolkit, UAT**:

    - **Function**: Provides fair and standardized implementations of 8 categories of attribution methods.
    - **Mechanism**: Uniformly implements the following method families: (a) Simple Gradient and Gradient×Input, the most fundamental gradient methods; (b) Integrated Gradients (IG), which integrates gradients along the path from a reference input to the actual input; (c) SmoothGrad, which averages gradients after adding noise to the input to reduce noise; (d) LIME, local linear approximation; (e) SHAP/KernelSHAP, a game-theoretic approach based on Shapley values; (f) Attention Weight/Rollout, which directly uses or aggregates attention weights; (g) Layer-wise Relevance Propagation (LRP), propagating relevance backward from the output layer to the input layer; (h) Contrastive Explanation, comparing attribution differences between the correct class and the next-best class. All methods share the same preprocessing (tokenization, padding) and postprocessing (normalization, aggregation to word level) pipelines to eliminate noise from implementation discrepancies.
    - **Design Motivation**: In previous comparative studies, implementations of different methods came from distinct codebases, with varying hyperparameter settings and preprocessing methods, leading to unfair comparisons.

2. **Five-Protocol Evaluation Matrix, FPEM**:

    - **Function**: Evaluates the quality of attribution from multiple perspectives.
    - **Mechanism**: Integrates 5 evaluation protocols: (a) Faithfulness: model performance should drop significantly after deleting high-attribution tokens (Comprehensiveness) and should remain largely unchanged after deleting low-attribution tokens (Sufficiency); (b) Plausibility: agreement with human-annotated importance rationales (IoU, F1); (c) Robustness: stability of attributions under semantic-preserving text transformations (such as synonym substitution); (d) Computational Cost: calculation time and memory consumption per sample; (e) Consistency: the extent to which the same method yields similar attributions on similar inputs. Each protocol incorporates 2-3 specific metrics, forming a method-by-protocol performance matrix.
    - **Design Motivation**: A single evaluation protocol can be biased; comprehensive assessment across multiple dimensions provides more balanced conclusions. In practice, different application scenarios prioritize different dimensions as well.

3. **Architecture-Method Compatibility Analysis**:

    - **Function**: Reveals which attribution methods are most effective for which model architectures.
    - **Mechanism**: Systematically executes all attribution methods on encoder models (e.g., BERT, RoBERTa), decoder models (e.g., GPT-2, LLaMA-2), and encoder-decoder models (e.g., T5, BART). The analysis reveals: attention-based methods perform reasonably well on encoder models but drop significantly on decoder models due to the causal attention mask; gradient-based methods exhibit the best consistency across all architectures; SHAP is most effective on small models but its computational cost scales drastically with model size. Finally, a "recommendation matrix" is compiled to suggest the most suitable attribution method based on model architecture and specific target dimensions.
    - **Design Motivation**: Model architectures in the LLM era have shifted from BERT-style to GPT-style, but validations for most attribution methods remain stuck on BERT. Re-evaluation on new architectures is necessary.

### Loss & Training
This work is evaluative and survey-based, involving no model training. All evaluations utilize pretrained model weights. Reference inputs (baselines) required for attribution calculations uniformly use padding token embeddings. The number of integration steps for IG is set to 50.

## Key Experimental Results

### Main Results

| Attribution Method | Faithfulness↑ | Plausibility↑ | Robustness↑ | Cost(ms)↓ | Overall Rank |
|---------|-------------|-------------|-----------|---------|---------|
| Integrated Gradients | 0.72 | 0.58 | 0.81 | 145 | 1 |
| SHAP | 0.74 | 0.62 | 0.78 | 3200 | 2 |
| Gradient×Input | 0.65 | 0.51 | 0.76 | 12 | 3 |
| LIME | 0.68 | 0.60 | 0.69 | 850 | 4 |
| Attention Rollout | 0.54 | 0.48 | 0.83 | 8 | 5 |
| LRP | 0.67 | 0.53 | 0.72 | 95 | 6 |
| SmoothGrad | 0.63 | 0.54 | 0.85 | 480 | 7 |
| Simple Gradient | 0.58 | 0.45 | 0.71 | 10 | 8 |

### Ablation Study (Model Architecture Comparison)

| Attribution Method | BERT (Enc) | GPT-2 (Dec) | T5 (Enc-Dec) | Architectural Consistency |
|---------|-----------|------------|-------------|---------|
| Integrated Gradients | 0.74 | 0.71 | 0.72 | Good consistency |
| SHAP | 0.76 | 0.73 | 0.71 | Good consistency |
| Attention Rollout | 0.62 | 0.41 | 0.52 | Poor for decoder |
| LIME | 0.70 | 0.67 | 0.66 | Moderate consistency |
| Gradient×Input | 0.67 | 0.64 | 0.65 | Good consistency |

### Key Findings
- Integrated Gradients is the most balanced method overall—achieving the best trade-off between faithfulness, robustness, and computational cost.
- SHAP is slightly superior in terms of faithfulness and plausibility, but its computational cost is 22 times higher than IG, rendering it impractical for large-scale applications.
- Attention-based methods (Attention Rollout) suffer a steep drop in faithfulness on decoder models ($0.62 \rightarrow 0.41$) because the causal attention mask prevents the attention distribution from capturing bidirectional importance information.
- The method with the highest robustness (SmoothGrad, 0.85) does not necessarily have the best faithfulness (0.63), indicating a trade-off between these two dimensions.
- On LLaMA-2-7B (7 billion parameters), the computational cost of SHAP becomes prohibitively high (>1 minute per sample), leaving gradient-based methods as the only viable options on large models.

## Highlights & Insights
- The contribution of a unified evaluation framework is more valuable than any individual technical innovation—it provides a level playing field for attribution research, making conclusions in subsequent works more reliable and comparable.
- The finding that attention weights are unreliable in decoder models carries significant practical value, especially since many LLM application scenarios still directly employ attention as an explanation tool.
- The proposed "recommendation matrix" offers direct guidance for NLP engineers: use Gradient×Input when speed is prioritized, IG when quality is of utmost importance, and SHAP when plausibility is crucial.

## Limitations & Future Work
- The "ground truth" of current evaluations remains controversial—whether faithfulness equates to correct attribution remains an unresolved philosophical issue.
- Evaluations are limited to classification tasks; attribution in generative tasks (such as explaining why an LLM generates a specific token) warrants separate investigation.
- Hyperparameters within evaluation protocols (e.g., deletion ratio, perturbation intensity) affect the conclusions. Although sensitivity analyses were conducted, they cannot exhaust all configurations.
- Future directions include: designing attribution methods tailored for LLM generative tasks, analyzing cross-lingual attribution consistency, and establishing theoretical understandings of attribution methods.

## Related Work & Insights
- **vs Atanasova et al. (2020) "A Diagnostic Study of Explainability Techniques"**: An early comparative study of attribution methods, which covered only 3 methods and was tested solely on BERT; the coverage and depth of this work far exceed it.
- **vs Bastings & Filippova (2020) "The Elephant in the Interpretability Room"**: This work questioned the validity of evaluating attribution faithfulness; ours partially addresses this concern via a multi-protocol evaluation.
- **vs DeYoung et al. (2020) ERASER**: ERASER provided human-annotated rationale datasets for plausibility evaluation; ours extends beyond this by incorporating more evaluation dimensions.

## Rating
- **Novelty**: ⭐⭐⭐ Survey-oriented work; methodological innovation is limited but the systematic contribution is significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive comparison of 8 methods × 5 evaluations × 4 architectures, highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Structurally clear, deeply analyzed, with a highly practical recommendation matrix.
- **Value**: ⭐⭐⭐⭐⭐ Provides a unified reference framework for the fragmented research on attribution methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reconsidering LLM Uncertainty Estimation Methods in the Wild](reconsidering_llm_uncertainty_estimation_methods_in_the_wild.md)
- [\[ACL 2025\] The Nature of NLP: Analyzing Contributions in NLP Papers](the_nature_of_nlp_analyzing_contributions_in_nlp_papers.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution](jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)
- [\[ACL 2025\] Unveiling Dual Quality in Product Reviews: An NLP-Based Approach](unveiling_dual_quality_in_product_reviews_an_nlp-based_approach.md)

</div>

<!-- RELATED:END -->
