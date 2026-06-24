---
title: >-
  [Paper Note] Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics
description: >-
  [ACL 2025 (Short)][LLM Evaluation][Confidence calibration] To address the failure of traditional confidence metrics caused by multiple valid outputs in text generation, this work proposes two task-agnostic confidence measures: "Ratio" (probability ratio of head vs. middle) and "Tail Thinness" (the thickness of the distribution tail). Relying solely on the model's output probabilities, these measures improve the confidence calibration of BART/Flan-T5 on summarization…
tags:
  - "ACL 2025 (Short)"
  - "LLM Evaluation"
  - "Confidence calibration"
  - "text generation"
  - "probability distribution"
  - "tail thinness"
  - "generation uncertainty"
date: 2026-05-08
content_hash: 80a64bf5a5175f29
---

# Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics

**Conference**: ACL 2025 (Short)  
**arXiv**: [2506.00637](https://arxiv.org/abs/2506.00637)  
**Code**: [https://github.com/ljyflores/calibrated-confidence-for-nlg](https://github.com/ljyflores/calibrated-confidence-for-nlg)  
**Area**: LLM Evaluation  
**Keywords**: Confidence calibration, text generation, probability distribution, tail thinness, generation uncertainty  

## TL;DR
To address the failure of traditional confidence metrics caused by multiple valid outputs in text generation, this work proposes two task-agnostic confidence measures: "Ratio" (probability ratio of head vs. middle) and "Tail Thinness" (the thickness of the distribution tail). Relying solely on the model's output probabilities, these measures improve the confidence calibration of BART/Flan-T5 on summarization, translation, and question-answering tasks.

## Background & Motivation
**Background**: Model confidence estimation typically focuses on the highest-probability sequence—high probability equals high confidence. This is effective in classification tasks (there is only one correct answer), where numerous calibration methods already exist.

**Limitations of Prior Work**: Text generation tasks often have multiple valid outputs (e.g., various phrasing in translation, different organizations in summarization). A capable model might distribute probability mass across multiple high-quality sequences, resulting in a relatively low absolute probability for the top-1 sequence—which traditional methods falsely penalize as low confidence.

**Key Challenge**: In generation tasks, a low top-1 probability can indicate either "uncertainty" (the model truly does not know) or "multiple good answers" (the model knows too much). Traditional metrics fail to distinguish between these two scenarios.

**Goal**: To design generation confidence metrics that are applicable to both single-answer and multi-answer scenarios.

**Key Insight**: Focus on the shape characteristics of the output distribution. A common trait of a confident model is that "good sequences are significantly better than bad ones" (a steep slope) and "bad sequences have extremely low probabilities" (thin tails), regardless of how many good sequences exist. These features are insensitive to output diversity.

**Core Idea**: Replace top-1 probability with the probability ratio and tail thinness to measure generation confidence, making it robust in "multiple correct answers" scenarios.

## Method

### Overall Architecture
During inference, $N=100$ candidate sequences are generated using beam search, and their sequence probabilities are computed. Two new metrics are then used to evaluate confidence. This approach requires no additional training or models.

### Key Designs

1. **Ratio Method (Ratio)**:

    - **Function**: Measures the probability gap between the best sequence and intermediate sequences.
    - **Mechanism**: $\text{Ratio}(x) = \frac{p_{\hat{y}^{(1)}}(x)}{p_{\hat{y}^{(k)}}(x)}$, which is the top-1 probability divided by the $k$-th ranked probability. $k$ is tuned on the validation set.
    - **Design Motivation**: A confident model assigns significantly higher probability to the best sequence, whereas an unconfident model assigns similar probabilities to all sequences. This is measured by comparing the gap between the head and the middle. When multiple good answers exist, the top sequences all have high probability, but the gap to the middle remains large.

2. **Tail Thinness**:

    - **Function**: Quantifies the "thickness" of the tail of the probability distribution.
    - **Mechanism**: $\text{Tail Thinness}(x) = \sum_{i=1}^N p_{\hat{y}^{(i)}}(x)^2$ (the sum of squared probabilities after softmax normalization). This is mathematically equivalent to the Herfindahl index.
    - **Design Motivation**: For a confident model, the tail (low-quality sequences) has extremely low probabilities, resulting in the sum of squares being dominated by the head (yielding a large value). When the model is unconfident, the distribution is uniform, resulting in a small sum of squares. This metric is insensitive to the exact number of good answers, as long as the poor answers have low probability.

### Loss & Training
- **No additional training required**—purely post-processing metrics computed directly from beam search output probabilities.
- Evaluation is performed after SFT fine-tuning on BART-Base and Flan-T5 Base.
- The $k$ value and softmax temperature are tuned on the validation set.

## Key Experimental Results

### Main Results (Spearman Correlation and Output Quality)

| Task | Model | Top-1 Probability | Ratio | Tail Thinness | Description |
|------|------|----------|-------|-------------|------|
| Summarization (ROUGE-L) | BART | 0.12 | **0.25** | **0.22** | Summarization has the highest openness, showing the most significant improvement. |
| Translation (BLEU) | Flan-T5 | 0.31 | **0.38** | **0.36** | Translation has multiple valid expressions. |
| QA (F1) | Flan-T5 | 0.43 | **0.48** | **0.46** | QA answers are more closed/convergent; the improvement is the smallest but still significant. |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| Different $k$ values | $k$ positively correlates with task openness | Open tasks require a larger $k$ (comparison further down the middle). |
| Different softmax temperatures | Affects the sensitivity of Tail Thinness | High temperature emphasizes differences, low temperature flattens differences. |
| Ratio + Tail Thinness Combination | Outperforms either alone | The two capture different distributional features and are complementary. |

### Key Findings
- Ratio and Tail Thinness consistently outperform top-1 probability across all tasks—differences are statistically significant (bootstrap test).
- The most significant improvement is achieved on the most open-ended task (summarization)—as it has the most valid outputs, where top-1 fails most severely.
- The $k$ value correlates positively with the diversity of task answers—open-ended tasks require a more distant reference point to determine if the head is truly "prominent".
- The method is fully task-agnostic, requiring no NLI models, extra fine-tuning, or task-specific design.

## Highlights & Insights
- **Distinguishing between "uncertainty" and "multiple answers"** is an overlooked yet critical issue—traditional methods conflate the two, leading to failed confidence calibration in generation tasks.
- **Extracting information from the distribution shape rather than single-point probabilities** is simple yet effective. It focuses on the global characteristics of the probability distribution (steepness, tail thickness) instead of a single value.
- **The method is highly lightweight**—requiring only the existing probabilities from beam search, with zero extra computation and zero extra models, allowing immediate integration into any generation system.
- **The two metrics capture different distributional characteristics**—Ratio focuses on the gap between head and middle, while Tail Thinness focuses on whether the tail is thin. Using them complementarily yields better results.

## Limitations & Future Work
- Only validated on relatively small models (BART/Flan-T5 Base); the beam search sequence distribution of large LLMs may differ.
- $N=100$ in beam search may not be large enough to accurately estimate the distribution shape.
- No comparison with verbalized confidence methods—which is a newer paradigm.
- Tail thinness is essentially the Herfindahl Index, which is highly correlated with sequence-level entropy; a deeper analysis of the theoretical foundation for its improvements is needed.
- The optimal choice of $k$ depends on task characteristics, lacking an automatic determination method.

## Related Work & Insights
- **vs Semantic Entropy**: Semantic entropy requires NLI models to group sequences, which is more precise but more costly; the proposed method requires no extra models.
- **vs Traditional Sequence Probability**: Sequence probability only considers top-1, ignoring distribution shape; the proposed method utilizes the probabilistic relationships among multiple sequences.

## Rating
- Novelty: ⭐⭐⭐ The insight is valuable, but the method is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐ Three tasks and two models, but the scale is limited (short paper).
- Writing Quality: ⭐⭐⭐⭐ Intuitive explanations are clear, with good illustrations.
- Value: ⭐⭐⭐⭐ Simple and practical confidence improvement, easy to integrate.

## Technical Details Supplement
- The Ratio method uses a ratio rather than a difference to compare the head and middle probabilities, making it more robust to probability scaling across sequences of different lengths.
- Tail Thinness is equivalent to the Herfindahl-Hirschman Index (HHI), a market concentration metric in economics.
- The computational complexity of both methods is $O(N)$, which is negligible compared to the overhead of beam search itself.
- Experiments used BART-Base and Flan-T5 Base (approx. 140M/250M parameters); behavior on larger LLMs might differ.
- Ratio is sensitive to head separation, whereas Tail Thinness is sensitive to overall concentration. Using them complementarily yields better results.
- In classification tasks (with only one correct answer), Ratio degenerates to the top-1 vs. top-2 probability ratio, which is equivalent to traditional methods.
- The method assumes beam search outputs can represent the model's probability distribution, but beam search introduces search bias.
- Future work could explore combining these metrics with approaches like Semantic Entropy.
- The applicability under sampling (instead of beam search) scenarios can also be explored.
- The code has been open-sourced, facilitating replication and integration into existing NLG systems.

## Technical Details Supplement
- The Ratio method uses a ratio rather than a difference to compare the head and middle probabilities, making it more robust to probability scaling across sequences of different lengths.
- Tail Thinness is equivalent to the Herfindahl-Hirschman Index (HHI), a market concentration metric in economics.
- The computational complexity of both methods is $O(N)$, which is negligible compared to the overhead of beam search itself.
- Experiments used BART-Base and Flan-T5 Base (approx. 140M/250M parameters); behavior on larger LLMs might differ.
- Ratio is sensitive to head separation, whereas Tail Thinness is sensitive to overall concentration. Using them complementarily yields better results.
- In classification tasks (with only one correct answer), Ratio degenerates to the top-1 vs. top-2 probability ratio, which is equivalent to traditional methods.
- The method assumes beam search outputs can represent the model's probability distribution, but beam search introduces search bias.
- Future work could explore combining these metrics with approaches like Semantic Entropy.
- The applicability under sampling (instead of beam search) scenarios can also be explored.
- The code has been open-sourced, facilitating replication and integration into existing NLG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs](pap2pat_benchmarking_outline-guided_long-text_patent_generation_with_patent-pape.md)
- [\[ACL 2025\] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](../../ACL2026/llm_evaluation/comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)

</div>

<!-- RELATED:END -->
