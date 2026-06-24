---
title: >-
  [Paper Note] Improve Vision Language Model Chain-of-thought Reasoning
description: >-
  [ACL 2025][Reasoning][Chain-of-thought] By (1) performing SFT on 193K multi-task CoT reasoning data distilled from GPT-4o, and (2) utilizing model-self-generated reasoning chains to construct positive and negative sample pairs for DPO reinforcement learning, the chain-of-thought reasoning capability of VLMs is significantly enhanced, with an average improvement of +11.7% in CoT prediction, and +7.3% in direct answering.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Chain-of-thought"
  - "CoT Reasoning"
  - "Knowledge Distillation"
  - "DPO"
  - "Reinforcement Learning"
  - "VLM"
date: 2026-05-08
content_hash: c71aa7bf66af7e11
---

# Improve Vision Language Model Chain-of-thought Reasoning

**Conference**: ACL 2025  
**arXiv**: [2410.16198](https://arxiv.org/abs/2410.16198)  
**Code**: [https://github.com/RifleZhang/LLaVA-Reasoner-DPO](https://github.com/RifleZhang/LLaVA-Reasoner-DPO)  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-thought, CoT Reasoning, Knowledge Distillation, DPO, Reinforcement Learning, VLM  

## TL;DR
By (1) performing SFT on 193K multi-task CoT reasoning data distilled from GPT-4o, and (2) utilizing model-self-generated reasoning chains to construct positive and negative sample pairs for DPO reinforcement learning, the chain-of-thought reasoning capability of VLMs is significantly enhanced, with an average improvement of +11.7% in CoT prediction, and +7.3% in direct answering.

## Background & Motivation

**Background**: Chain-of-thought (CoT) is crucial for improving the interpretability and trustworthiness of VLMs. However, existing VLM training data is dominated by short answers (e.g., "14"), lacking detailed reasoning processes.

**Limitations of Prior Work**: (1) **Lack of reasoning steps in training data**—annotators tend to provide short answers directly, as writing out the full reasoning process is more time-consuming; (2) **Short-answer training cannot implicitly learn CoT**—authors' experiments reveal that training on 26K direct answers on ChartQA improves direct prediction accuracy by 2.9, but only improves CoT by 0.6; (3) **Lack of high-quality CoT training data**—existing VQA datasets rarely contain reasoning steps.

**Key Challenge**: VLMs require CoT reasoning capabilities, but (a) high-quality CoT data is extremely scarce, (b) models trained solely on short answers do not generalize to tasks requiring detailed reasoning, and (c) the quality of CoT reasoning requires further calibration.

**Goal**: Address the scarcity of VLM CoT reasoning data and improve the quality of reasoning.

**Core Idea**: First distill CoT data for SFT to teach the model how to reason, and then use DPO with model-self-generated positive and negative reasoning sample pairs to calibrate the reasoning quality.

## Method

### Overall Architecture
A three-stage pipeline (as shown in Figure 2):
- **Stage A**: Distill CoT reasoning data from GPT-4o (ShareGPT-4o-Reasoning, 193K samples)
- **Stage B**: Train the VLM (SFT) with a mixture of CoT and direct answer data, yielding LLaVA-Reasoner-SFT
- **Stage C**: Construct positive and negative reasoning sample pairs, and perform further calibration of reasoning quality using DPO to obtain LLaVA-Reasoner-DPO

### Key Designs

1. **CoT Data Distillation (ShareGPT-4o-Reasoning)**:
    - Covers 9 datasets across 4 major reasoning skills:
        - Commonsense Reasoning: A-OKVQA (16.9K)
        - Chart Comprehension: ChartQA (26.0K)
        - Document/Text Comprehension: DocVQA (37.3K), InfoVQA (22.4K), TextVQA (29.7K)
        - Math/Science: MathVision (11.0K), G-LLaVA (30.3K), SQA (6.1K), AI2D (11.9K)
    - Distillation method: Feed the input (image, question, reference short answer) to GPT-4o to generate the reasoning process
    - Quality filtering: Samples where the GPT-4o prediction does not match the ground truth are filtered out (which also revealed annotation errors)
    - CoT responses average around 100 tokens, whereas short answers are usually < 5 tokens

2. **SFT Data Mixture Strategy**:
    - Two prompt templates: Direct prediction ("Answer with a short answer") and CoT prediction ("Generate a reason first and then output a short answer")
    - Optimal combination ④: CoT + Direct + Format alignment data (450 items) + LLaVA instruction data (2K)
    - Key design: CoT answers are formatted as "reasoning process... ### Answer: final answer," enabling automated extraction

3. **DPO Reasoning Calibration**:
    - Generate 32 candidate CoT reasoning chains per question using the SFT model (temperature 1.0/1.2)
    - Compare the final prediction of each reasoning chain with the ground truth: correct -> positive sample $y_w$, incorrect -> negative sample $y_l$
    - Select only questions with accuracy between 0.25 and 0.85 (tasks that are too easy or too hard are not suitable for DPO)
    - A maximum of 3 pairs per question, totaling 64.8K preference pairs
    - DPO target function: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w|x,\mathcal{V})}{\pi_{\text{ref}}(y_w|x,\mathcal{V})} - \beta\log\frac{\pi_\theta(y_l|x,\mathcal{V})}{\pi_{\text{ref}}(y_l|x,\mathcal{V})})]$
    - **Truncation trick**: Truncating responses to 90 tokens yields the best performance

### Loss & Training
- SFT phase: Standard causal language modeling loss
- DPO phase: The aforementioned DPO loss function, with $\beta = 0.1$ and learning rate of 5e-7

## Key Experimental Results

### Main Results — Ablation on SFT Data Mixtures (Table 2)

| Training Data | Reasoning Mode | A-OK | ChartQA | DocVQA | InfoVQA | TextVQA | AI2D | SQA | MathVista | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Format only ① | direct | 85.8 | 70.2 | 75.7 | 37.7 | 68.2 | 71.5 | 75.4 | 39.3 | 65.5 |
| Format only ① | CoT | 84.3 | 71.2 | 67.0 | 34.9 | 62.2 | 67.4 | 74.4 | 40.3 | 62.7 |
| Direct only ② | direct | 86.4 | 73.7 | 78.0 | 45.4 | 71.9 | 78.8 | 91.5 | 43.2 | 71.1 |
| Direct only ② | CoT | 85.7 | 71.8 | 68.8 | 38.6 | 63.6 | 72.5 | 85.4 | 38.6 | 65.6 |
| CoT only ③ | direct | 84.9 | 71.8 | 81.2 | 45.7 | 72.1 | 75.3 | 85.0 | 41.9 | 69.7 |
| CoT only ③ | CoT | 85.1 | 82.2 | 81.2 | 49.7 | 69.9 | 77.0 | 91.3 | 49.2 | 73.2 |
| **Both ④ (SFT)** | direct | 85.4 | 76.1 | 82.9 | 50.6 | 73.1 | 79.4 | 90.4 | 44.3 | **72.8** |
| **Both ④ (SFT)** | CoT | 86.2 | 83.0 | 81.8 | 51.6 | 71.1 | 78.5 | 92.7 | 50.6 | **74.4** |

### Ablation Study — DPO (Table 6)

| Method | Reasoning Mode | A-OK | ChartQA | DocVQA | InfoVQA | TextVQA | AI2D | SQA | MathVista | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SFT ④ | CoT | 86.2 | 83.0 | 81.8 | 51.6 | 71.1 | 78.5 | 92.7 | 50.6 | 74.4 |
| + RLAIF-V ⑤ | CoT | 86.7 | 83.0 | 82.4 | 50.8 | 71.4 | 79.1 | 92.9 | 50.8 | 74.6 |
| **+ DPO-ours ⑥** | CoT | **87.0**| **84.2**| **82.7**| **52.7**| **71.5**| **79.5**| 92.6 | **52.1**| **75.3**|

### Comparison with GPT-4o and SOTA (Table 5)

| Model | A-OK | ChartQA | DocVQA | SQA | MathVista | Average (best) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 90.1 | 84.7 | 90.8 | 87.2 | 63.4 | 77.9 |
| Cambrian-7B | 83.1 | 73.3 | 77.8 | 80.4 | 49.0 | 64.5 |
| **LLaVA-Reasoner-SFT** | 86.2 | 83.0 | 82.9 | 92.7 | 50.6 | **68.8** |

### Key Findings
1. **Short-answer training does not teach CoT reasoning**: Direct-only training improves direct prediction by +5.6, but only +2.9 for CoT, and performance on computation-heavy tasks even drops for CoT.
2. **CoT training also enhances direct prediction**: Direct prediction under CoT-only training outperformed Direct-only training (this is particularly pronounced in document-understanding tasks).
3. **Mixed training is optimal**: Training with both CoT and Direct data simultaneously achieves the best performance across both prediction modes.
4. **DPO is effective but requires reasoning-specific data pairs**: Using general RLAIF-V DPO data yields marginal gain (+0.2), whereas using self-constructed reasoning data pairs yields a significant gain (+1.1).
5. **DPO model can serve as a verifier**: Using DPO reward scores for best-of-N reranking and weighted voting further boosts performance.
6. **DPO learns token-level rewards**: Credit assignment visualization indicates that the DPO model is extremely sensitive to the first occurrence of an error or hallucination in the reasoning chain.

## Highlights
- **Dataset Contribution**: Releasing the 193K multi-task CoT reasoning dataset ShareGPT-4o-Reasoning for direct community use.
- **Key Finding**: Empirically demonstrates that short-answer training cannot implicitly learn CoT reasoning, highlighting the necessity of explicit CoT data.
- **General Method**: The two-stage SFT+DPO framework is applicable to any VLM and is not bound to specific architectures.
- **Dual-purpose DPO**: Acts both as a generator to improve reasoning quality and as a verifier for reranking.
- **Extremely Thorough Experiments**: Main text plus 6 appendices, covering data ablation, baseline comparisons, RFT vs. DPO, prompt optimization, etc.

## Limitations & Future Work
- The base model is limited to LLaVA-NeXT-8B, without verification on larger scale models.
- CoT distillation relies on GPT-4o, which is costly, and the distillation quality is constrained by GPT-4o's capabilities.
- On certain tasks (TextVQA, DocVQA, AI2D), CoT does not outperform direct prediction, potentially because simple fact extraction does not benefit from complex reasoning.
- DPO preference pairs were constructed using only 3 datasets; scaling up to more datasets is left for future work.
- Evaluation is primarily on VQA benchmarks, without coverage of open-ended conversational visual reasoning scenarios.

## Related Work
- **VLM Reasoning**: Works like MAVIS and Visual CoT focus on training reasoning in specific domains (e.g., mathematics, grounding).
- **VLM/LLM Alignment**: Frameworks like DPO and PPO are utilized to reduce hallucinations and enhance factuality; Step-DPO is leveraged for mathematical CoT reasoning.
- **CoT Data**: Existing VQA datasets rarely contain reasoning steps; this work represents the first large-scale multi-task VLM CoT distillation effort.
- **Ours Positioning**: The first systematic study on SFT+RL training strategies for VLM CoT reasoning, addressing a gap in VLM reasoning training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Clearly formulated problem (short answers do not teach CoT); though the methodologies are not entirely new, their integration is highly effective.
- **Value**: ⭐⭐⭐⭐⭐ — The 193K CoT dataset and training pipeline are directly applicable, and the code is open-source.
- **Technical Depth**: ⭐⭐⭐⭐ — Extremely meticulous ablation of SFT data mixtures, along with an in-depth analysis of DPO (credit assignment).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Main text plus 6 appendices, exceptionally thorough.
- **Overall Recommendation**: ⭐⭐⭐⭐⭐ — A landmark reference for VLM reasoning training, with open-sourced data and code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)
- [\[CVPR 2025\] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](../../CVPR2025/llm_reasoning/argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)
- [\[ACL 2025\] Critic-CoT: Boosting the Reasoning Abilities of Large Language Model via Chain-of-Thoughts Critic](critic-cot_boosting_the_reasoning_abilities_of_large_language_model_via_chain-of.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ACL 2025\] DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)

</div>

<!-- RELATED:END -->
