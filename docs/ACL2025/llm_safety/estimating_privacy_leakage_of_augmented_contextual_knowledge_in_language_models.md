---
title: >-
  [Paper Note] Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models
description: >-
  [ACL 2025][LLM Safety][Privacy Leakage] This paper proposes the context influence metric to quantify the degree of privacy leakage of augmented contextual knowledge in language models during decoding based on a differential privacy framework, and systematically analyzes the effects of model size, context size, generation location, and other factors on privacy leakage.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Privacy Leakage"
  - "Contextual Knowledge"
  - "Differential Privacy"
  - "Language Models"
  - "RAG Security"
date: 2026-05-08
content_hash: 0db1c46a5c94893a
---

# Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.03026](https://arxiv.org/abs/2410.03026)  
**Code**: [james-flemings/context_influence](https://github.com/james-flemings/context_influence)  
**Area**: AI Safety  
**Keywords**: Privacy Leakage, Contextual Knowledge, Differential Privacy, Language Models, RAG Security

## TL;DR
This paper proposes the context influence metric to quantify the degree of privacy leakage of augmented contextual knowledge in language models during decoding based on a differential privacy framework, and systematically analyzes the effects of model size, context size, generation location, and other factors on privacy leakage.

## Background & Motivation
- **Background**: LLMs rely on parametric knowledge (pre-training encoding) and contextual knowledge (information in the prompt) to complete tasks such as QA. Injecting external context into the prompt using methods like RAG (Retrieval-Augmented Generation) has become mainstream.
- **Key Challenge**: Context may contain sensitive information, and LMs may leak this private data when answering. However, directly comparing LM outputs with the context will **overestimate** privacy risks — because the LM's parametric knowledge may already contain the same information.
- **Key Example**: If the context contains John Doe's address and the LM outputs this address, direct comparison would classify it as "context leakage." But if the LM still outputs this address after the context is removed, it indicates that the leakage stems from parametric knowledge rather than the context.
- **Core Idea**: There is a need for a privacy metric that can **separate the contribution of parametric knowledge** — comparing the difference in output distribution with and without a specific subset of context, drawing inspiration from differential privacy frameworks.

## Method

### Overall Architecture
1. Define the context influence metric (based on ex-post per-instance DP)
2. Propose Context Influence Decoding (CID) to control the degree of context influence
3. Establish a theoretical connection between context influence and PMI (Pointwise Mutual Information)
4. Systematically analyze various factors affecting privacy leakage through experiments

### Key Designs
1. **Context Influence (Definition 3.1)**

    - **Mechanism**: Measuring the change in output probability after removing the $i$-th $n$-gram $D_{i,n}$ from the context
    - Definition: $\tau_{i,n} = |\log p_\theta(y_t | D, \mathbf{x}, \mathbf{y}_{<t}) - \log p_\theta(y_t | D \setminus D_{i,n}, \mathbf{x}, \mathbf{y}_{<t})|$
    - $n=1$ corresponds to word-level privacy, $n=|D|$ corresponds to document-level privacy
    - The context influence on the entire response is the sum of individual tokens (analogous to the composition property of DP)

2. **Context Influence Decoding (CID)**

    - Reformulate CAD (Context-aware Decoding) as context influence controlled by a parameter $\lambda$
    - $\bar{p}_{\theta,\lambda}(y_t) = \sigma[(\lambda \cdot \text{logit}_\theta(y_t|D,\mathbf{x}) + (1-\lambda) \cdot \text{logit}_\theta(y_t|\mathbf{x})) / T]$
    - $\lambda=0$: Uses only parametric knowledge (no contextual privacy leakage)
    - $\lambda=1$: Normal decoding
    - $\lambda>1$: Amplifies context influence (reducing hallucination but increasing privacy risk)

3. **Theoretical Connection (Theorem 3.1)**

    - Context influence is proportional to $\lambda \cdot |\text{pmi}(D) - \text{pmi}(D \setminus D_{i,n})|$
    - Two key leakage factors: (a) the degree of out-of-distribution (OOD) of the context relative to parametric knowledge (large PMI discrepancy), and (b) the amplification level of context during decoding (large $\lambda$)

### Loss & Training
Ours does not involve model training; it is an analytical/measurement framework. Key formulas:
- DP Guarantees: CID can satisfy $\epsilon$-DP by selecting an appropriate $\lambda^*$ (Theorem B.1)
- Practical Application: Estimate expected context influence via $\hat{\tau}_{i,n}(p_\theta) = \frac{1}{|\mathcal{D}|}\sum_{(D,\mathbf{x})} \tau_{i,n}(\cdot)$

## Key Experimental Results

### Main Results: Context Influence and Input Regurgitation

| Model | Dataset | $\lambda$ | Context Influence | Repeat Prompts | Rouge Prompts |
|------|--------|-----------|-------------------|----------------|---------------|
| LLaMA 3 8B | CNN-DM | 0.5 | 15.97 | 8 | 109 |
| LLaMA 3 8B | CNN-DM | 1.0 | 64.61 | 285 | 632 |
| LLaMA 3 8B | CNN-DM | 1.5 | 98.99 | 429 | 882 |
| OPT 1.3B | PubMedQA | 1.0 | 45.66 | 47 | 251 |
| GPT-Neo 1.3B | PubMedQA | 1.0 | 38.79 | 54 | 268 |

### Key Comparison: Impact of Parametric Knowledge

| Comparison | Description |
|------|------|
| OPT 1.3B vs GPT-Neo 1.3B (PubMedQA) | GPT-Neo pre-training contains PubMed $\rightarrow$ lower context influence (38.79 vs 45.66) $\rightarrow$ but Repeat/Rouge Prompts are higher instead $\rightarrow$ traditional metrics overestimate contextual leakage |

### Factor Analysis

| Factor | Key Findings |
|------|---------|
| $\lambda$ (Context Amplification) | $\lambda$ from 1.0 $\rightarrow$ 1.5: ROUGE-L improvements by 10% but input regurgitation increases by 50% |
| Model Size | Larger models have lower context influence (as they can rely more on parametric knowledge) |
| Context Size | Context influence is extremely low when $|D| \leq 32$; it tends to stabilize after $|D| \geq 256$ |
| Response Position | The first 10 tokens are most influenced by the context, which gradually weakens afterward |
| Pre-training vs Fine-tuning | LLaMA 3 IT is far more influenced by context than LLaMA 3 |
| n-gram Position | The n-gram at the beginning of the context has the greatest influence (position bias) |

### Key Findings
- Context influence accurately attributes privacy leakage: OPT exhibits higher context influence on PubMedQA (due to pre-training lacking PubMed), whereas traditional metrics incorrectly show higher leakage for GPT-Neo.
- Amplifying the context ($\lambda>1$) can reduce hallucinations but significantly increases privacy risks — presenting a privacy-utility trade-off.
- Model fine-tuning (SFT+RLHF) improves contextual utilization capability, yet simultaneously increases privacy leakage.

## Highlights & Insights
- **Solid Theoretical Foundation**: Introduces the DP analysis framework into contextual privacy metrics, replacing coarse-grained $\epsilon$-DP with ex-post per-instance DP, allowing for the calculation of privacy loss tailored to specific contexts and outputs.
- **Distinguishing Two Sources of Knowledge**: The core contribution is separating the contributions of parametric and contextual knowledge, preventing the estimation overestimation problem of traditional approaches.
- **Comprehensive Analysis**: Systematically investigates multidimensional factors including model size, context size, response position, and n-gram granularity.
- **Practical Guidance**: (1) Placing sensitive information at the end of the context can reduce leakage; (2) Adaptive privacy levels (strict at the beginning, relaxed later) can be adopted; (3) The choice of pre-training data affects the trustworthiness of privacy guarantees.

## Limitations & Future Work
- Does not consider the impact of entropy during model decoding on context influence (more confident models show smaller influence, which can be misleading).
- Focuses solely on contextual privacy leakage, without involving privacy leakage of parametric knowledge (memorization issues).
- Utilizes only temperature sampling, without analyzing the impact of sampling strategies like top-p/top-k.
- **Potential Direction for Improvement**: Can an adaptive decoding strategy be designed to automatically reduce $\lambda$ when high context influence is detected, achieving real-time privacy protection?

## Related Work & Insights
- Complementary to studies on parametric knowledge leakage: Carlini et al. research training data extraction, while this paper studies context leakage during inference.
- Related to RAG security: Zeng et al., Qi et al. research RAG data extraction attacks, but implicitly assume the context is not within the parametric knowledge, which this paper shows overestimates risks.
- Relation to Context-aware Decoding: This paper reformulates it as a privacy-control tool (CID).
- Implications for privacy-preserving LLM deployment: In RAG scenarios, it is necessary to simultaneously consider the privacy attributes of both parametric and contextual knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying the DP framework to contextual privacy metrics is novel, and the concept of distinguishing the two knowledge sources is constructive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough analysis across multiple models, datasets, and factors, combining qualitative and quantitative assessments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations, intuitive explanation of motivation with Figure 1, and overall rigorous logic.
- Value: ⭐⭐⭐⭐ Directly provides guidance for RAG privacy evaluation, and establishes a theoretical foundation for privacy-preserving decoding strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[ACL 2026\] Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](../../ACL2026/llm_safety/privacy_collapse_benign_fine-tuning_can_break_contextual_privacy_in_language_mod.md)
- [\[NeurIPS 2025\] AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](../../NeurIPS2025/llm_safety/agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)
- [\[ACL 2025\] The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)

</div>

<!-- RELATED:END -->
