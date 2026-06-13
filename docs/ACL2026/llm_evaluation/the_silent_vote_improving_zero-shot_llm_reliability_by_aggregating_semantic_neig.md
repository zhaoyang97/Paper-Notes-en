---
title: >-
  [Paper Note] The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods
description: >-
  [ACL 2026][LLM Evaluation][Semantic Softmax] This paper identifies that constrained softmax in zero-shot LLM classification discards probability mass near label synonyms. It proposes Semantic Softmax…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Semantic Softmax"
  - "Calibration Error"
  - "Constrained Decoding"
  - "Semantic Neighborhoods"
  - "Uncertainty Estimation"
date: 2026-05-08
content_hash: 2310a7f87e620768
---

# The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods

**Conference**: ACL 2026  
**arXiv**: [2605.09739](https://arxiv.org/abs/2605.09739)  
**Code**: None  
**Area**: LLM Evaluation / Calibration / Zero-Shot Classification  
**Keywords**: Semantic Softmax, Calibration Error, Constrained Decoding, Semantic Neighborhoods, Uncertainty Estimation  

## TL;DR
This paper identifies that constrained softmax in zero-shot LLM classification discards probability mass near label synonyms. It proposes Semantic Softmax, a training-free method that aggregates the "silent votes" of top-K vocabulary tokens back to target labels, significantly reducing ECE and Brier Score while improving AUROC and F1 performance.

## Background & Motivation
**Background**: LLM-as-a-classifier has become a standard paradigm for text classification. In practical deployment, to ensure models output only valid categories, systems typically restrict the vocabulary to a few label tokens and perform softmax exclusively over these tokens.

**Limitations of Prior Work**: Although constrained decoding ensures correct formatting, it distorts probabilities. A model may consider an input close to related words like "thrilling," "delightful," or "toxic-ish," but because these words are not in the candidate label set, they are masked before the softmax calculation.

**Key Challenge**: Classification labels are discrete, but LLM semantic evidence is distributed across the continuous semantic neighborhood of the vocabulary. Treating all non-label tokens as irrelevant causes the remaining label probabilities to be renormalized to excessively high confidence, resulting in a systemic error the authors term Renormalization Bias.

**Goal**: The authors aim to make zero-shot classification probabilities more aligned with the model's true uncertainty without fine-tuning or structural changes, particularly for tasks with inherent annotator disagreement like sentiment and toxicity.

**Key Insight**: The model's output embedding space already captures the relationships between label tokens and semantically similar tokens. By aggregating the probability mass of these close tokens to the target labels based on semantic weights, the "Silent Vote" lost to constrained softmax can be recovered.

**Core Idea**: Instead of considering only the logit of the label token itself, the method considers which tokens among the top-K vocabulary candidates semantically support the label and aggregates their probabilities using weights based on embedding similarity.

## Method
The proposed method is lightweight, replacing the traditional constrained softmax with an inference-time aggregation layer. It requires no model retraining or external synonym dictionaries, instead using the model’s own output embeddings to define semantic similarity between tokens and labels.

### Overall Architecture
Given an input text and a set of candidate labels $\mathcal{L}=\{l_1,l_2,\ldots,l_n\}$, the standard approach obtains the logit vector $z$ for the entire vocabulary $\mathcal{V}$, keeps only label tokens, and calculates $P(l_j \mid x)=\exp(z_{l_j}) / \sum_{l'\in\mathcal{L}}\exp(z_{l'})$. This probability essentially discards the mass of all non-label tokens.

The Semantic Softmax workflow consists of three steps. First, instead of blind aggregation, it selects the top-K tokens with the highest probabilities from the unconstrained distribution (the main experiment uses $K=50$). Second, it uses the model's output embeddings $E$ to calculate the similarity between each top-K token $v$ and label token $l$, filtering noise with a threshold $\tau=0.8$: $w(v,l)=\max(0, \cos(E_v,E_l)-\tau)$. Third, it cumulatively adds the top-K token probabilities weighted by $w(v,l)$ and normalizes across the label set.

This design ensures that if the model distributes probability mass among multiple words close to "joy," Semantic Softmax re-aggregates these votes into "joy." If the model fluctuates between multiple semantic neighborhoods, the final distribution becomes softer, better reflecting human annotator disagreement.

### Key Designs
1.  **Renormalization Bias Formalization**:
    - **Function**: Explains why constrained softmax leads to systematic overconfidence.
    - **Mechanism**: Standard probabilities are renormalized only within the label set $\mathcal{L}$. All tokens $v\notin\mathcal{L}$ are removed regardless of their semantic proximity. Consequently, a label that only weakly supports a class may be normalized to a probability near 1 because other candidates are even weaker.
    - **Design Motivation**: Many calibration issues do not stem from the model not knowing the answer, but from the inference layer flattening the rich vocabulary distribution into a few label tokens, mistaking format constraints for probability estimates.

2.  **Semantic Kernel Aggregation**:
    - **Function**: Incorporates the probability mass of semantic neighbors surrounding label tokens into classification scores.
    - **Mechanism**: For each top-K token $v$ and label $l$, cosine similarity of output embeddings defines the weight, with threshold $\tau$ subtracted as a noise filter. The final $P_{sem}(l\mid x)$ is the weighted sum of all top-K probabilities $P(v)$ and weights $w(v,l)$, normalized across labels.
    - **Design Motivation**: Using the model's own embeddings simplifies deployment without manual synonym lists; thresholding prevents aggregating tokens that are globally related but do not truly support the label.

3.  **Training-Free Calibration Layer**:
    - **Function**: Allows the method to be directly plugged into existing LLM classification services.
    - **Mechanism**: Semantic Softmax occurs only after the next-token distribution is calculated. It does not change prompts, update parameters, or require a task-specific training set. The authors apply the same logic to Qwen-3-1.7B and Phi-4-mini, demonstrating it is not a model-specific trick.
    - **Design Motivation**: Enterprise deployment prioritizes modularity and low risk. Compared to recalibration, temperature scaling, or fine-tuning, an inference layer is easier to implement and better suited for zero-shot scenarios without labeled calibration sets.

### Loss & Training
There is no training loss. All operations are completed at inference time. The key hyperparameters are the number of top-K tokens $K$ and the semantic threshold $\tau$. Main experiments use $K=50$ and $\tau=0.8$. The appendix indicates that results are insensitive to $K$ over a wide range but sensitive to $\tau$; values too low introduce noisy neighbors, while values too high miss useful synonym votes.

## Key Experimental Results

### Main Results
The experiments cover two models and two datasets: GoEmotions for testing semantic neighborhood recovery for fine-grained emotion labels, and the Civil Comments ambiguous subset for testing calibration against human toxicity means and annotator disagreement. Metrics include ECE, Brier Score, AUROC, and Macro-F1.

| Model | Dataset | Method | ECE ↓ | Brier ↓ | AUROC ↑ | F1 ↑ |
|------|--------|------|-------|---------|---------|------|
| Qwen-3-1.7B | GoEmotions | Standard | 0.574 | 0.842 | 0.712 | 0.229 |
| Qwen-3-1.7B | GoEmotions | Ours | 0.069 | 0.591 | 0.763 | 0.267 |
| Qwen-3-1.7B | Civil Comments | Standard | 0.482 | 0.571 | 0.784 | 0.412 |
| Qwen-3-1.7B | Civil Comments | Ours | 0.108 | 0.517 | 0.882 | 0.451 |
| Phi-4-mini | GoEmotions | Standard | 0.421 | 0.795 | 0.744 | 0.236 |
| Phi-4-mini | GoEmotions | Ours | 0.065 | 0.588 | 0.756 | 0.253 |
| Phi-4-mini | Civil Comments | Standard | 0.395 | 0.542 | 0.812 | 0.421 |
| Phi-4-mini | Civil Comments | Ours | 0.092 | 0.498 | 0.835 | 0.451 |

### Ablation Study
The appendix provides sensitivity analysis for $K$ and $\tau$. The results below reflect general patterns: ECE is generally lowest when $\tau=0.8$. Increasing $\tau$ beyond 0.85 degrades ECE, indicating that overly strict similarity filtering discards useful semantic neighbors.

| K | τ=0.75 ECE ↓ | τ=0.80 ECE ↓ | τ=0.85 ECE ↓ | τ=0.80 F1 ↑ | Conclusion |
|---|--------------|--------------|--------------|------------|------|
| 50 | 0.1497 | 0.1145 | 0.1840 | 0.4325 | Threshold near main experiment is most stable |
| 100 | 0.1514 | 0.1145 | 0.1923 | 0.4308 | Increasing K does not significantly improve ECE |
| 300 | 0.1491 | 0.1144 | 0.1987 | 0.4308 | $\tau=0.8$ maintains lowest ECE |
| 600 | 0.1498 | 0.1130 | 0.1986 | 0.4308 | Lowest ECE occurs at medium threshold |
| 1000 | 0.1499 | 0.1162 | 0.1993 | 0.4317 | No performance collapse at very large K |

### Key Findings
- The improvement in ECE via Semantic Softmax is far greater than the improvement in F1, indicating that it primarily solves probability reliability rather than just classification boundaries.
- The method does not sacrifice discriminative power. AUROC and Macro-F1 increased across all four model-dataset combinations, suggesting that recovering semantic neighbor votes provides additional classification signals.
- Qualitative examples from Civil Comments show that standard constrained decoding often gives extreme scores above 0.9 on ambiguous text, whereas Semantic Softmax remains closer to the human toxicity mean.
- $\tau$ is the critical hyperparameter. Too low allows weakly related tokens to mix in; too high causes it to degenerate into looking only at label tokens. $\tau=0.8$ proved to be a balanced point in the experiments.

## Highlights & Insights
- The term "Silent Vote" accurately identifies a specific yet overlooked problem in LLM classification: models do express uncertainty, but we discard their vocabulary-level representations.
- The method cleverly reuses output embeddings rather than introducing external dictionaries or synonym resources. This allows Semantic Softmax to adapt automatically to the semantic space of the base model.
- Reporting calibration and discriminative performance simultaneously is appropriate. While many calibration methods soften distributions but hurt accuracy, this method improves AUROC/F1 by recovering true semantic mass.
- This logic can be transferred to multi-label classification, emotion intensity estimation, and safety auditing, provided the task labels can be mapped to vocabulary tokens or verbalizers.

## Limitations & Future Work
- The current method is tailored for "select one from predefined labels" LLM-as-classifier scenarios and is not suitable for direct use in long-text generation due to cumulative latency from top-K retrieval and similarity calculations.
- Experiments focused on small/medium models like Qwen-3-1.7B and Phi-4-mini; scaling behavior for larger models, closed-source APIs, and instruction-tuned models needs verification.
- The method assumes that the geometric proximity of semantically similar tokens in the output embedding space is reliable. If model embeddings are highly anisotropic or synonym clustering is poor, neighbor aggregation may introduce noise.
- Experiments centered on English GoEmotions and Civil Comments. In multilingual contexts, synonyms may be distributed across languages or tokenization fragments, potentially requiring language-aware kernel designs.

## Related Work & Insights
- **vs Constrained Decoding**: Traditional constrained decoding focuses on format validity. This paper demonstrates that format constraints do not equal probability calibration and that masked softmax systematically generates overconfidence.
- **vs Temperature Scaling**: Temperature scaling requires a calibration set and only flattens the distribution globally. Semantic Softmax utilizes sample-level semantic neighborhoods to redistribute discarded mass based on label semantics.
- **vs Verbalizer-based Prompting**: Prompting/verbalizer methods often struggle with label word selection. This paper offers another perspective: even if the label word is imperfect, aggregating its semantic neighborhood can reduce the fragility of single-token selection.
- **Inspiration for Future Research**: When performing zero-shot classification with LLMs, calibration metrics should be reported for both constrained softmax and full-vocabulary semantic aggregation, especially in high-risk auditing, emotion recognition, and medical triage.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The problem is precisely targeted; Semantic Softmax is simple but effective, with core innovation in utilizing vocabulary semantic neighborhoods for calibration.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers two models, two datasets, calibration and discriminative metrics, qualitative cases, and hyperparameter sensitivity, though large models and multilinguality lack verification.
- Writing Quality: ⭐⭐⭐⭐☆ The argumentation is clear, concepts like Renormalization Bias and Silent Vote are explained intuitively, and the formulas are lightweight.
- Value: ⭐⭐⭐⭐☆ Very practical for zero-shot classification services, particularly in deployment scenarios requiring reliable confidence levels rather than just hard labels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ACL 2026\] Revisiting the Reliability of Language Models in Instruction-Following](revisiting_the_reliability_of_language_models_in_instruction-following.md)

</div>

<!-- RELATED:END -->
