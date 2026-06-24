---
title: >-
  [Paper Note] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?
description: >-
  [ACL2025][LLM (Other)][Uncertainty Quantification] This paper systematically investigates the impact of increasing the number of examples on the predictive uncertainty of LLMs in long-context ICL. Through uncertainty decomposition, it reveals that performance gains primarily stem from the reduction of epistemic uncertainty ($EU$), and explains the internal mechanism of uncertainty reduction from the perspective of residual stream projection.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Uncertainty Quantification"
  - "In-Context Learning"
  - "Many-Shot ICL"
  - "Epistemic Uncertainty"
  - "Long Context"
date: 2026-05-08
content_hash: 2dcf46cd26f79e32
---

# Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?

**Conference**: ACL2025  
**arXiv**: [2505.21003](https://arxiv.org/abs/2505.21003)  
**Code**: Not released  
**Area**: LLM/NLP  
**Keywords**: Uncertainty Quantification, In-Context Learning, Many-Shot ICL, Epistemic Uncertainty, Long Context

## TL;DR

This paper systematically investigates the impact of increasing the number of examples on the predictive uncertainty of LLMs in long-context ICL. Through uncertainty decomposition, it reveals that performance gains primarily stem from the reduction of epistemic uncertainty ($EU$), and explains the internal mechanism of uncertainty reduction from the perspective of residual stream projection.

## Background & Motivation

**Rise of Long-Context ICL**: In recent years, long-context techniques (continued fine-tuning, position extrapolation, and innovative architectures like Mamba) have enabled LLMs to process hundreds or even thousands of ICL examples (many-shot ICL), providing a new paradigm for tuning-free adaptation.

**Prior Work Focuses on Performance**: Existing work (Agarwal et al. 2024; Jiang et al. 2024) primarily focuses on accuracy improvements brought by more examples, while research on the trustworthiness and reliability of generation results remains insufficient.

**Trustworthiness is the Key Gap**: In high-risk application scenarios (healthcare, finance), high performance alone is insufficient; it is also necessary to know "how confident" the model is—namely, uncertainty quantification (UQ).

**Uncertainty Decomposition Offers a New Perspective**: Decomposing total uncertainty ($TU$) into epistemic uncertainty ($EU$, originating from the model's lack of knowledge) and aleatoric uncertainty ($AU$, originating from the inherent randomness of the data) can reveal the mechanisms behind ICL improvements.

**Internal Mechanism Remains Unclear**: Although it has been observed that more examples improve performance, how this improvement is realized inside the model (how layer-wise confidence evolves) remains a black box.

**Core Research Questions**: (RQ1) Can more examples reduce uncertainty? (RQ2) From the perspective of uncertainty decomposition, where do performance improvements originate? (RQ3) What is the internal mechanism of uncertainty reduction?

## Method

### Overall Architecture

The paper constructs an uncertainty quantification and decomposition framework for many-shot ICL (Fig.3), comprising three core modules:

1. **Probability Distribution Construction**: By using multiple demonstration sets ($L$ sets) and multiple decoding samples ($m$ times per set), an $L \times |Y|$ probability matrix $A$ is constructed to capture uncertainty arising from demonstration sets and model configurations.
2. **Entropy Calculation (TU)**: Entropy is calculated after normalizing the probability matrix, serving as a measure of total uncertainty.
3. **Uncertainty Decomposition (EU/AU)**: Based on the Bayesian ICL framework (Ling et al. 2024), which views ICL as a process of mapping examples to latent concepts $\beta$, mutual information is leveraged to decompose $EU$ and $AU$.

### Key Designs

- **Predictive Distribution**: Focusing on classification and multiple-choice QA tasks, it leverages the natural advantage of categorical outputs—where each label $y \in Y$ corresponds to a predefined category, and the probability $P_y$ can be directly obtained from logits.
- **Uncertainty Decomposition Formulae**:
    - $TU = H(\sigma(\Sigma[A_{j,:}]))$, i.e., the entropy after aggregating all demonstration sets.
    - $EU = (1/L) \Sigma H(\sigma(A_{j,:}))$, i.e., the mean of internal entropies of each set.
    - $AU = TU - EU$, i.e., the mutual information $I(y, \beta|\Theta)$.
- **Residual Stream Projection**: Projecting the residual representation of each layer into the vocabulary space via the unembedding matrix $W_U$ to visualize the evolution of confidence for candidate answers across layers.

### Loss & Training

- Use beam search to generate 10 candidate outputs, with a temperature of 0.7.
- Iterate over 6 different demonstration sets to decompose $EU$/$AU$.
- Model weights are loaded in float16, running on $8 \times 80\text{GB}$ A100 GPUs.
- Evaluation metrics: AUROC (UQ quality), Exact Match (accuracy).

## Key Experimental Results

### Table 1: Microscopic Analysis of Uncertainty Changes (Llama-3.1-8B)

| Dataset | 8-shot ΔU↓/↑ | 32-shot ΔU↓/↑ | 128-shot ΔU↓/↑ | 128-shot ΔAcc |
|---|---|---|---|---|
| AG News | 66.8%↓ / 30.5%↑ | 88.6%↓ / 10.8%↑ | 90.8%↓ / 8.7%↑ | +15.8 |
| SST-2 | 71.7%↓ / 20.3%↑ | 86.6%↓ / 9.4%↑ | 92.1%↓ / 5.6%↑ | +7.9 |
| CommonsenseQA | 62.2%↓ / 26.2%↑ | 69.0%↓ / 17.8%↑ | 81.2%↓ / 16.6%↑ | +5.2 |
| LD5 (Hard) | 58.4%↓ / 33.2%↑ | 73.6%↓ / 24.4%↑ | 83.8%↓ / 13.1%↑ | +10.8 |
| LD7 (Hard) | 48.4%↓ / 48.8%↑ | 59.2%↓ / 38.0%↑ | 83.3%↓ / 15.3%↑ | +12.3 |

### Table 2: Logit Gap and Maximum Logit (CommonsenseQA)

| Model | 4-shot | 32-shot | 64-shot | 128-shot |
|---|---|---|---|---|
| Llama-3.1-8B | 2.86 / 24.98 | 2.75 / 27.03 | 2.55 / 27.66 | 2.53 / 28.01 |
| Mistral-7B-v0.2 | 2.78 / 17.14 | 2.24 / 19.60 | 2.57 / 20.38 | 2.75 / 20.84 |
| Qwen1.5-7B | 3.51 / 29.11 | 3.62 / 30.49 | 3.73 / 30.97 | 3.76 / 30.94 |

### Key Findings

1. **Continuous Decrease in TU**: As the number of shots increases, easy mode quickly converges to a low-uncertainty state, whereas hard mode does not show a significant decrease until several hundred shots are provided.
2. **EU is the Primary Driver of TU Reduction**: Initially, $EU$ accounts for the majority of $TU$. Increasing examples reduces $EU$ primarily by injecting task-specific knowledge.
3. **AU Interference in Hard Mode**: In complex tasks, longer inputs introduce noise that causes $AU$ to rise, partially offsetting the reduction in $EU$.
4. **"ICL Sink" Phenomenon**: Qwen1.5-7B exhibits an anomaly in hard mode, where its confidence at a low shot count is comparable to that at a high shot count.
5. **Model Scale Effect**: 14B/32B models exhibit lower overall uncertainty, and the advantages of many-shot ICL remain significant.
6. **Internal Mechanism**: Many-shot ICL leads to a higher concentration of the correct answer's logit, widening the logit gap between the correct answer and distractors, which pushes the correct probability towards 1 via the exponential sensitivity of Softmax.

## Highlights & Insights

- **Novel Perspective of Uncertainty Decomposition**: This paper systematically investigates uncertainty evolution under many-shot ICL for the first time, attributing performance gains to the reduction of $EU$ rather than merely having more information.
- **Information Volume vs. Context Length**: De-duplication experiments demonstrate that merely repeating the same examples $N$ times cannot reduce $EU$; what is truly effective is information diversity.
- **Intuitive Residual Stream Visualization**: Case studies clearly demonstrate dramatic fluctuations in confidence under 4-shot settings, whereas under 128-shot settings, the correct answer consistently maintains the highest probability starting from approximately the 22nd layer.
- **Clear Practical Recommendations**: In practical applications, selecting a larger $k$ value is recommended to simultaneously enhance both performance and reliability.

## Limitations & Future Work

1. **No Coverage of Open-Ended Generation Tasks**: The study only focuses on classification and MCQA, lacking reliable UQ techniques for free-form generation scenarios such as summarization and translation.
2. **No Exploration of CoT/Reasoning ICL**: Quantitative attributes of uncertainty under reasoning paradigms like Chain-of-Thought have not been investigated, and existing UQ methods struggle to capture logical complexity.
3. **Limited Extreme Shot Counts**: Constrained by the context lengths of open-source LLMs and computational overhead, extreme scenarios involving several thousand shots were not evaluated.
4. **Limited Model Selection**: Only three 7-8B base models were tested, lacking systematic comparisons with closed-source models like GPT-4 or larger-scale models.
5. **AU Definition Relies on Assumptions**: Equating $AU$ to the mutual information regarding $\beta$ depends on the validity of the Bayesian ICL framework; whether actual LLMs precisely match this assumption remains questionable.

## Related Work & Insights

### vs. Uncertainty Decomposition Framework of Ling et al. (2024)

Ling et al. proposed a Bayesian framework for ICL uncertainty decomposition, but only focused on few-shot scenarios. Ours extends this framework to many-shot/long-context ICL, finding that the downward trend of $EU$ is more pronounced under a large number of examples, and the interference effect of $AU$ in hard mode is a novel discovery.

### vs. Many-shot ICL of Agarwal et al. (2024)

Agarwal et al. demonstrated that many-shot prompting leads to significant performance improvements on Gemini 1.5 Pro, but focused solely on accuracy. Ours supplements the trustworthiness dimension—performance improvements are accompanied by a reduction in uncertainty, with a strong correlation between the two, providing theoretical support for the reliability of many-shot ICL.

### vs. Long-Context ICL Properties of Bertsch et al. (2024)

Bertsch et al. investigated the impacts of example retrieval and ordering. Ours provides complementary findings from the perspective of uncertainty: the information diversity of example quantity is more important than their permutation order.

## Rating

- **Novelty**: 7/10 — First systematic study of many-shot ICL uncertainty, offering a novel perspective, though the methodology is a direct extension of an existing UQ framework.
- **Experimental Thoroughness**: 7/10 — 3 models $\times$ 6 datasets $\times$ multiple shot counts, including ablations and visualizations, but lacks closed-source models and generative tasks.
- **Writing Quality**: 8/10 — Clear structure, with three RQs progressing logically, and abundant visualizations.
- **Value**: 7/10 — Fills the gap in trustworthiness research for many-shot ICL, providing practical though not disruptive conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Harmonized Uncertainty Estimation for Large Language Models](towards_harmonized_uncertainty_estimation_for_large_language_models.md)
- [\[ACL 2025\] SConU: Selective Conformal Uncertainty in Large Language Models](sconu_selective_conformal_uncertainty_in_large_language_models.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2025\] Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results](revisiting_uncertainty_quantification_evaluation_in_language_models_spurious_int.md)
- [\[ACL 2025\] Reconsidering LLM Uncertainty Estimation Methods in the Wild](reconsidering_llm_uncertainty_estimation_methods_in_the_wild.md)

</div>

<!-- RELATED:END -->
