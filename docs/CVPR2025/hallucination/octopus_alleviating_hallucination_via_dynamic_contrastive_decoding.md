---
title: >-
  [Paper Note] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding
description: >-
  [CVPR 2025][Hallucination Detection][Hallucination Alleviation] Through extensive experiments, this paper reveals the hybrid nature of hallucination causes in LVLMs—different samples and different generation steps face different flags of hallucination challenges. Consequently, the Octopus framework is proposed, which utilizes a learnable decision token and a transformer block to adaptively select the most appropriate contrastive decoding (CD) strategy at each generation step.…
tags:
  - "CVPR 2025"
  - "Hallucination Detection"
  - "Hallucination Alleviation"
  - "Contrastive Decoding"
  - "Dynamic Policy Selection"
  - "Large Vision-Language Models (LVLMs)"
  - "DPO"
date: 2026-05-08
content_hash: ac9ac07fc9fd78c5
---

# Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding

**Conference**: CVPR 2025  
**arXiv**: [2503.00361](https://arxiv.org/abs/2503.00361)  
**Code**: [https://github.com/LijunZhang01/Octopus](https://github.com/LijunZhang01/Octopus)  
**Area**: Multimodal VLM  
**Keywords**: Hallucination Alleviation, Contrastive Decoding, Dynamic Policy Selection, Large Vision-Language Models (LVLMs), DPO

## TL;DR

Through extensive experiments, this paper reveals the hybrid nature of hallucination causes in LVLMs—different samples and different generation steps face different flags of hallucination challenges. Consequently, the Octopus framework is proposed, which utilizes a learnable decision token and a transformer block to adaptively select the most appropriate contrastive decoding (CD) strategy at each generation step. Optimized via DPO, Octopus outperforms existing CD methods across four benchmarks.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) such as LLaVA and InstructBLIP excel in visual understanding and multimodal reasoning, but suffer from pervasive hallucination issues—generating non-existent objects, incorrect attributes, and non-existent relations. Contrastive Decoding (CD), as a training-free post-processing method, has become an important direction for alleviating hallucinations.

**Limitations of Prior Work**:
1. **Limitation of a Single Strategy**: Existing CD methods (VCD, M3ID, AVISC) are designed for specific types of hallucinations—VCD combats language priors, M3ID alleviates visual information loss, and AVISC reduces attention bias. However, they all adopt a "one-size-fits-all" approach, using the same perturbation strategy for all samples and generation steps.
2. **Neglect of the Complexity of Hallucination Causes**: No prior work has systematically studied whether different samples and different tokens face the same types of hallucinations.

**Key Challenge**: The causes of hallucination are hybrid (language priors + visual information loss + attention bias), but existing methods can only address symptoms individually using a single strategy for all cases, which inevitably leads to sub-optimal results.

**Key Insight**: This work first demonstrates the hybrid nature of hallucinations through diagnostic experiments, and then designs an adaptive framework that allows the model to automatically select the most appropriate CD strategy at each generation step.

**Core Idea**: Similar to an Octopus, the framework uses "eyes" (decision tokens) to identify the type of hallucination and multiple "tentacles" (various CD strategies) to cope with different hallucination challenges respectively.

## Method

### Overall Architecture

The Octopus framework consists of two core components: (1) **Decision Module** ("eyes")—a transformer block and a learnable decision token, responsible for determining which type of hallucination the current token faces at each generation step; (2) **Execution Module** ("tentacles")—multiple off-the-shelf CD strategies (VCD, M3ID, AVISC + a null action), which execute the corresponding contrastive operations based on the decision outcome. The decision module parameters are optimized via DPO, while the LVLM parameters remain frozen.

### Key Designs

1. **Sample-level Hallucination Diagnostic Experiments**:
    - **Function**: Proving that a single CD strategy cannot cover all hallucination samples.
    - **Mechanism**: On three datasets (AMBER, Object-HalBench, and MMHalBench), three CD methods (VCD, M3ID, AVISC) are used to intervene on each sample of LLaVA-1.5-7B, and the proportion of samples successfully corrected by each method is counted. Results show that ~60% of the samples can only be corrected by one specific CD strategy, with the overlapping region where all three strategies are effective being only about 10%.
    - **Design Motivation**: Providing empirical support for dynamic strategy selection—if a single strategy could solve all problems, dynamic selection would be unnecessary.

2. **Token-level Hallucination Diagnostic Experiments**:
    - **Function**: Proving that different tokens within the same sample suffer from different hallucination causes.
    - **Mechanism**: On the AMBER dataset, brute-force testing was conducted with different combinations of CD strategies for the first three hallucinated nouns in each description. Quantitative results indicate that combining multiple strategies (e.g., strategy-1+3, strategy-1+2+3) significantly outperforms a single strategy. In qualitative analysis, attention maps reveal that within the same sentence, "sitting" is affected by attention bias (focusing on visual distortion tokens), "lying" is due to insufficient attention to visual information, and "person" completely relies on language tokens—corresponding to three different hallucination causes for three words.
    - **Design Motivation**: Further refining the granularity of dynamic strategy selection from the sample level to the token level.

3. **Octopus Decision and Execution Architecture**:
    - **Function**: Adaptively selecting the most appropriate CD strategy at each generation step.
    - **Mechanism**: Constructing a lightweight transformer block $\mathcal{O}_\phi$, concatenating the hidden state sequence of the LVLM $H_t = \{h_i\}_{i=1}^t$ (containing information from visual, textual, and generated tokens) with a learnable decision token $eye \in \mathbb{R}^d$, adding positional encodings, and feeding them into the transformer block: $[h_{eye}^t; H_t'] = \mathcal{O}_\phi(\text{concat}[eye; H_t] + E_{pos})$. Through the self-attention mechanism, $h_{eye}^t$ aggregates information from the entire sequence. It is then mapped by an MLP to an action vector $h_{act}^t \in \mathbb{R}^k$ ($k=4$, corresponding to three CD strategies + a null action), and argmax is applied to obtain the strategy selection $a_t$ for the current step. This ultimately generates a complete workflow $\mathcal{A} = \{a_t\}_{t=1}^N$.
    - **Design Motivation**: Utilizing the self-attention mechanism of the transformer to allow the decision token to comprehensively consider visual input, textual instructions, and generated content for making global policy decisions.

### Loss & Training

**DPO Optimization**: Since the argmax operation is non-differentiable and explicit decision labels are lacking, Direct Preference Optimization (DPO) is adopted for training.

- **Data Construction**: For each sample, 10 different action sequences are randomly generated (randomly choosing one of the 4 actions at each step). Based on the CHAIR index, the sequences are divided into positive samples (workflows that reduce hallucination $\mathcal{A}^+$) and negative samples (workflows that increase hallucination $\mathcal{A}^-$).
- **Optimization Objective (DPO without reference model)**: $\max_{\mathcal{O}_\phi} \mathbb{E} \log \sigma(\beta \log \mathcal{O}_\phi(\mathcal{A}^+ | x) - \beta \log \mathcal{O}_\phi(\mathcal{A}^- | x))$, where $x = (v, q)$ is the vision-language input, and $\beta = 1$.
- **Key Feature**: Optimizes only the parameters $\phi$ of Octopus, while the weights of the LVLM remain completely frozen, ensuring deployment flexibility.

**Training Data**: The generation task uses 10,000 preference pairs from MSCOCO, and the discriminative task uses 7,000 hallucination data. Training is completed on 4×RTX 3090 GPUs with a batch size of 4.

## Key Experimental Results

### Main Results (Generation Task, LLaVA-1.5-7B)

| Dataset | Metric | LLaVA Base | +VCD | +M3ID | +AVISC | +Octopus | Gain vs. Best CD |
|--------|------|-----------|------|-------|--------|----------|---------------|
| AMBER | CHAIR↓ | 8.0 | 6.7 | 6.0 | 6.3 | **4.8** | -1.2 |
| AMBER | Cover↑ | 44.5 | 46.5 | 48.9 | 46.6 | **49.2** | +0.3 |
| AMBER | HalRate↓ | 31.0 | 27.8 | 26.0 | 25.6 | **23.4** | -2.2 |
| Object-HalBench | CHAIRs↓ | 25.0 | 23.6 | 23.2 | 22.1 | **20.8** | -1.3 |
| Object-HalBench | CHAIRi↓ | 9.2 | 8.4 | 7.3 | 7.8 | **6.6** | -0.7 |
| MMHalBench | Score↑ | 1.59 | 1.96 | 2.14 | 2.19 | **2.61** | +0.42 |

### Ablation Study (AMBER Dataset)

| Configuration | CHAIR↓ | Cover↑ | Hal↓ | Cog↓ |
|------|--------|--------|------|------|
| LLaVA Base (w/o CD) | 8.0 | 44.5 | 31.0 | 2.2 |
| Randomly select three CD strategies | 6.9 | 46.2 | 26.1 | 2.2 |
| Octopus (Str1+Str2) | 5.5 | 48.7 | 25.8 | 1.5 |
| Octopus (Str1+Str3) | 5.7 | 48.2 | 25.3 | 1.5 |
| Octopus (Str2+Str3) | 5.5 | 48.4 | 26.2 | 1.6 |
| **Octopus (All three + null)** | **4.8** | **49.2** | **23.4** | **1.2** |

### Discriminative Task (LLaVA-1.5-7B)

| Dataset | Metric | LLaVA Base | +VCD | +Octopus | Gain vs. Base |
|--------|------|-----------|------|----------|-------------|
| AMBER | Acc | 67.00 | 67.30 | **76.70** | +9.70 |
| AMBER | F1 | 71.10 | 71.10 | **82.70** | +11.60 |
| POPE (ALL) | Acc | 82.04 | 82.96 | **85.79** | +3.75 |
| POPE (ALL) | F1 | 80.42 | 81.81 | **83.44** | +3.02 |

### Key Findings

- Octopus reduces the CHAIR metric on the AMBER dataset from 8.0 to 4.8, which is approximately a 40% reduction in hallucination compared to the Base model.
- Compared to methods requiring full-model retraining (e.g., HA-DPO, HALVA), Octopus still maintains a substantial lead without requiring any modifications to the LVLM weights.
- Ablation studies demonstrate that: (1) even randomly selecting CD strategies helps, but is far inferior to the adaptive selection of Octopus; (2) adding more "tentacles" (CD strategies) consistently improves performance, showing the excellent scalability of the framework.
- Different RL optimization methods (DPO, Monte-Carlo, PPO) all yield satisfactory results, showing that the framework is insensitive to the choice of optimization algorithm.
- Different evaluation criteria (CHAIR, Cover, average score) can all serve as metrics for splitting positive and negative samples, indicating the cross-domain adaptability of the framework.

## Highlights & Insights

1. **"Diagnosis Before Treatment" Research Paradigm**: First revealing the hybrid causes of hallucination through systematic sample-level and token-level diagnostic experiments, and then designing solutions based on these findings. This approach of "understanding the problem before solving it" is highly worthy of reference.
2. **Elegant "Meta-Strategy" Design**: Instead of inventing a completely new CD method, this work designs a "policy selector" to combine existing CD methods. This meta-learning concept provides the framework with natural scalability—any new CD method in the future can be directly integrated as a new "tentacle".
3. **Completely Frozen LVLM Weights**: Only training the lightweight decision module without modifying any parameters of the deployed model offers exceptional practicality.
4. **Ingenious Application of DPO**: Formulating the strategy selection problem as a preference learning problem, and automatically constructing positive/negative sample pairs through random sampling and CHAIR evaluation avoids manual annotation.

## Limitations & Future Work

- Currently, only three CD strategies (VCD, M3ID, AVISC) are integrated. As new CD methods emerge, the space of candidate strategies can be further expanded.
- DPO training data is constructed via random sampling, which might not be optimal in quality; more efficient data construction methods could be explored.
- Multiple forward passes (each CD strategy requires additional inference with distorted inputs) introduce inference latency, requiring a trade-off with efficiency during actual deployment.
- The computational overhead of token-level dynamic selection is relatively high—runing the decision module and its corresponding CD forward pass is required for every token.
- The effectiveness of the framework depends heavily on the diversity and complementarity of the candidate CD strategies; if the candidate strategies are highly homogeneous, the gains may be limited.

## Related Work & Insights

- **vs. VCD/M3ID/AVISC**: These three CD methods serve as the "tentacles" of Octopus, each only covering ~60% of the hallucinated samples. The core contribution of Octopus is learning to select the most appropriate strategy under different circumstances.
- **vs. Retraining Methods (HACL, POVID, HA-DPO)**: These methods require constructing high-quality data and retraining the LVLM parameters, which is costly and inapplicable to already deployed models. As a plug-and-play solution, Octopus even outperforms these heavyweight methods.
- **vs. OPERA/LCD/ICD**: These are also post-processing methods, but they still employ a single strategy. The dynamic combination approach of Octopus represents a fundamental improvement.
- **Insight**: This "meta-strategy" philosophy can be extended to other domains—in any scenario where multiple complementary solutions exist, learning a strategy selector might be more effective than designing a single superior strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining hallucination diagnosis with dynamic strategy selection is highly novel, though the core CD strategies remain existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Diagnostic experiments are extensive and convincing, main experiments cover both generative and discriminative tasks, and ablation studies are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The "octopus" analogy is well-sustained throughout, the structure is clear, and the presentation of diagnostic experiments is highly intuitive.
- Value: ⭐⭐⭐⭐ As a general framework, it offers excellent scalability and practicality, providing valuable guidance for hallucination alleviation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[ICLR 2026\] Leveraging Pretrained Knowledge at Inference Time: LoRA-Gated Contrastive Decoding for Multilingual Factual Language Generation in Adapted LLMs](../../ICLR2026/hallucination/leveraging_pretrained_knowledge_at_inference_time_lora-gated_contrastive_decodin.md)
- [\[ACL 2025\] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](../../ACL2025/hallucination/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ACL 2025\] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](../../ACL2025/hallucination/alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)

</div>

<!-- RELATED:END -->
