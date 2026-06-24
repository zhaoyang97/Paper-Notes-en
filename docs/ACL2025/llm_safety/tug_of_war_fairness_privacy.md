---
title: >-
  [Paper Note] The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models
description: >-
  [ACL 2025][LLM Safety][Fairness] It is discovered that enhancing the privacy awareness of LLMs through SFT significantly degrades their fairness awareness (representing a trade-off). To address this, a training-free method named SPIN (Suppressing Fairness-Privacy Coupled Neurons) is proposed to decouple the two dimensions based on information theory, simultaneously improving fairness by 12.2% and privacy awareness by 14.0% on Qwen2-7B.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Fairness"
  - "Privacy"
  - "Neuron Decoupling"
  - "Information Theory"
  - "Training-free Method"
date: 2026-05-08
content_hash: 7cd36b11fe3ef41e
---

# The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.16672](https://arxiv.org/abs/2410.16672)  
**Code**: [https://github.com/ChnQ/SPIN](https://github.com/ChnQ/SPIN)  
**Area**: AI Safety  
**Keywords**: Fairness, Privacy, Neuron Decoupling, Information Theory, Training-free Method  

## TL;DR
It is discovered that enhancing the privacy awareness of LLMs through SFT significantly degrades their fairness awareness (representing a trade-off). To address this, a training-free method named SPIN (Suppressing Fairness-Privacy Coupled Neurons) is proposed to decouple the two dimensions based on information theory, simultaneously improving fairness by 12.2% and privacy awareness by 14.0% on Qwen2-7B.

## Background & Motivation
**Background**: The deployment of LLMs in sensitive domains such as healthcare and finance requires models to simultaneously possess fairness awareness (avoiding the generation of discriminatory content) and privacy awareness (refusing to leak personal information). Mainstream methods enhance specific dimensions of awareness through SFT (such as FFT, LoRA, DoRA, etc.).

**Limitations of Prior Work**: Experiments reveal a counterintuitive phenomenon—enhancing the privacy awareness of LLMs using SFT significantly reduces their fairness awareness. Even when mixing an equal amount of fairness data during fine-tuning, the trade-off persists.

**Key Challenge**: The root cause lies in neuronal polysemanticity—certain neurons encode representations related to both fairness and privacy simultaneously. Fine-tuning causes updates to these coupled neurons that lead to conflicting optimization directions across the two dimensions.

**Goal**: How to simultaneously improve both the fairness and privacy awareness of LLMs without fine-tuning, while preserving general capabilities?

**Key Insight**: Leveraging information theory, if two variables share coupled components, removing these components can reduce the mutual information between them. Analogous to LLMs, suppressing fairness-privacy coupled neurons can decouple the two representations.

**Core Idea**: Identify neurons in the LLM that are highly crucial for both fairness and privacy (coupled neurons), and set their weights to zero to decouple the two types of awareness, thereby eliminating the trade-off.

## Method

### Overall Architecture
SPIN is a training-free method executed once prior to deployment. The inputs are a pre-trained LLM and a small amount of labeled fairness/privacy data (even adversarial data is applicable), and the output is the modified model. The core workflow is: calculate neuron importance scores $\rightarrow$ locate coupled neurons $\rightarrow$ suppress (set to zero).

### Key Designs

1. **Theoretical Foundation Based on Information Theory**:

    - Function: Prove that removing coupled variables can reduce mutual information.
    - Mechanism: Theorem 1 proves that if there exist coupled variables $Z_1, Z_2$ with conditional mutual information $I[Z_1; Z_2 | X, Y] > 0$, then $I[X;Y] < I[(X,Z_1);(Y,Z_2)]$. Mapping $(X,Z_1)$ and $(Y,Z_2)$ to fairness and privacy representations in the LLM respectively, eliminating the coupled components $Z_1, Z_2$ (i.e., coupled neurons) can reduce the mutual information between the two representations.
    - Design Motivation: Provide a theoretical guarantee for the neuron suppression operation rather than arbitrarily selecting which neurons to suppress.

2. **Neuron Importance Score Calculation**:

    - Function: Quantify the importance of each neuron to fairness/privacy tasks.
    - Mechanism: Utilizing first-order Taylor approximation, the importance score is defined as $I_W(i,j) = \mathbb{E}_{s \sim D}|W(i,j) \nabla_{W(i,j)} \mathcal{L}(s)|$, which represents the expectation of the product of weight magnitude and gradient magnitude. Two sets of importance matrices are computed using fairness data $D_f$ and privacy data $D_p$ respectively.
    - Design Motivation: Combining weight and gradient information allows for a more precise localization of semantically relevant neurons compared to Wanda (which only uses weights + activations) and SparseGPT.

3. **Coupled Neuron Localization and Suppression**:

    - Function: Identify neurons that are simultaneously crucial for both fairness and privacy, and set them to zero.
    - Mechanism: For each weight matrix, the top-$r$ neuron set for fairness importance $\mathcal{N}_f$ and the top-$r$ set for privacy importance $\mathcal{N}_p$ are selected. Their intersection $\mathcal{N}_\text{coupled} = \mathcal{N}_f \cap \mathcal{N}_p$ represents the coupled neurons. After excluding neurons critical to general capabilities, the weights of the coupled neurons are set to zero.
    - Design Motivation: Suppressing only the intersection rather than performing separate operations on fairness or privacy neurons avoids disrupting their respective independent functions. $r$ defaults to an extremely small value ($10^{-6}$ to $10^{-5}$), affecting only a minimal number of neurons.

### Loss & Training
Completely training-free. Labeled data is only utilized to compute gradients (for calculating importance scores), without performing any parameter updates. Algorithm 1 is executed once prior to deployment, and no additional operations are required afterwards.

## Key Experimental Results

### Main Results
Evaluation of fairness and privacy awareness across four models (Salad-bench, MD-judge scoring):

| Model | Method | Fairness Awareness↑ | Privacy Awareness↑ | Description |
|------|------|------------|----------|------|
| Qwen2-7B | Origin | 59.9% | 72.2% | Baseline |
| Qwen2-7B | FFT | 65.8% | 82.2% | Privacy↑ but Fairness↓ |
| Qwen2-7B | SPIN | **72.1%** | **86.2%** | Simultaneous substantial improvement |
| Mistral-7B | Origin | ~60% | ~75% | Baseline |
| Mistral-7B | SPIN | Improved | Improved | No trade-off |
| Llama2-7B | SFT methods | ↓ | ↓ | SFT degrades both |
| Llama2-7B | SPIN | ↑ | ↑ | Still effective |

### Ablation Study

| Configuration | Fairness | Privacy | General Capabilities |
|------|------|------|----------|
| Target = ALL, r=10⁻⁶ | Optimal | Optimal | Preserved |
| Target = MLP | Effective | Effective | Preserved |
| Target = MHA | No significant change | No significant change | Preserved |
| r increased to 10⁻³ | Decreased | Decreased | Decreased |

### Key Findings
- **Coupled neurons are primarily located in the MLPs**: Manipulating only the MHA has almost no impact on fairness/privacy awareness, which is consistent with prior findings that MLPs store factual knowledge.
- **Robust to adversarial data**: SPIN remains effective even when using adversarial data (e.g., "unfair questions + unfair answers") to calculate importance scores. This is because it only uses the data to locate neurons rather than learning conversational patterns.
- **High data efficiency**: Stable performance is achieved with only 100 data samples, whereas SFT experiences a collapse in both fairness and privacy under the same 100-sample setting.
- General capabilities (across 9 benchmarks including MMLU and HellaSwag) remain largely unaffected or even slightly improved after SPIN.

## Highlights & Insights
- **Discovery of an important, counterintuitive phenomenon**: Enhancing one alignment dimension degrades another. This provides critical insights for research on multi-objective alignment, indicating that different safety attributes may experience conflict through shared neurons.
- **Training-free, information-theory-driven method**: In contrast to most alignment methods that rely on SFT or RLHF, SPIN is executed as a one-time neuron-level manipulation, offering both theoretical elegance and practical efficiency. This paradigm can be generalized to decouple other dimension conflicts.
- **Effectiveness under adversarial data is the key highlight**: While traditional SFT requires high-quality annotated data, SPIN can correct coupling neurons even from adversarial data, significantly lowering the barrier for data acquisition.

## Limitations & Future Work
- Only post-hoc patching at inference time was investigated, leaving how to fundamentally prevent the emergence of coupled neurons during pre-training or fine-tuning stages unexplored.
- The granularity of neuron manipulation is relatively coarse (at the MHA/MLP block level); finer-grained localization (such as specific attention heads) might prove more precise.
- Verification was only conducted on 7B-class models, leaving the performance on larger-scale models (70B+) unknown.
- The evaluation of fairness and privacy relies on automated judges (MD-judge), which may introduce evaluation biases.

## Related Work & Insights
- **vs SFT methods (FFT/LoRA/DoRA/ReFT)**: All SFT methods exhibit a fairness-privacy trade-off under low-resource data settings, making SPIN the only approach capable of simultaneously improving both.
- **vs Wanda/SparseGPT (Pruning Methods)**: Although utilizing these two methods to locate coupled neurons can also eliminate the trade-off, their performance is inferior to that of Importance Score, indicating that gradient information yields more precise localization for semantically relevant neurons.
- This work reveals the conflict mechanisms among different alignment dimensions within LLMs, offering valuable references for the study of multi-objective optimization in safety alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ Discovers the fairness-privacy trade-off and proposes an information-theoretic decoupling scheme, introducing innovations in both theory and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four model series, multiple baselines, ablation studies, and robustness analyses, though validation on larger models is lacking.
- Writing Quality: ⭐⭐⭐⭐ Proves a tight integration between theory and experiments with intuitive figures and tables.
- Value: ⭐⭐⭐⭐ Provides significant insights and a practical solution for multi-dimensional alignment conflicts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Fairness of Large Language Models in Multi-document Summarization](improving_fairness_of_large_language_models_in_multi-document_summarization.md)
- [\[ACL 2025\] ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](dialect_fairness_robustness.md)
- [\[ACL 2025\] Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)
- [\[ACL 2025\] Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models](estimating_privacy_leakage_of_augmented_contextual_knowledge_in_language_models.md)
- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](../../NeurIPS2025/llm_safety/distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)

</div>

<!-- RELATED:END -->
