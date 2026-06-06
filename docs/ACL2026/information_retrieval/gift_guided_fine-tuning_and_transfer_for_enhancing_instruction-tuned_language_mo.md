---
title: >-
  [Paper Note] GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models
description: >-
  [ACL2026][Information Retrieval & RAG][Guided Fine-Tuning] GIFT ensures that instruction-tuned models are no longer passive targets for merging. Instead…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Guided Fine-Tuning"
  - "LoRA"
  - "Instruction Model"
  - "Adapter Merge"
  - "Confidence Weighting"
date: 2026-05-08
content_hash: 392d69473e689288
---

# GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models

**Conference**: ACL2026  
**arXiv**: [2605.01256](https://arxiv.org/abs/2605.01256)  
**Code**: https://github.com/sustech-nlp/gift  
**Area**: LLM Adaptation / Parameter-Efficient Fine-Tuning / Model Merging  
**Keywords**: Guided Fine-Tuning, LoRA, Instruction Model, Adapter Merge, Confidence Weighting  

## TL;DR
GIFT ensures that instruction-tuned models are no longer passive targets for merging. Instead, the instruction model is used to annotate confidence levels for training tokens, which then guide the LoRA fine-tuning of the base model. Finally, the adapter is merged back into the instruction model, consistently outperforming direct fine-tuning and transfer baselines like Shadow-FT in mathematics, medicine, and instruction-following tasks.

## Background & Motivation
**Background**: Open-source LLMs are typically released as both base models and instruction-tuned models. Instruction models undergo complex post-training to achieve superior instruction-following, robustness, and general reasoning capabilities. However, directly fine-tuning them on downstream tasks often disrupts this balance.

**Limitations of Prior Work**: A common alternative is training a task-specific adapter on the base model and merging the learned updates into the instruction model (e.g., Shadow-FT, Chat Vector, Re-Adapt). However, these methods usually involve the instruction model only in the final step, while the adapter training still treats all tokens equally as in standard SFT.

**Key Challenge**: Not all tokens in task data are equivalent. Some tokens represent critical reasoning steps or domain knowledge, while others are merely style, phrasing, or alternative expressions. Standard SFT forces the base model to fit all tokens, applying strong gradients even to tokens where the instruction model itself has low confidence, leading to adapters that are incompatible with the instruction model's representation space.

**Goal**: The authors aim to introduce the knowledge of the instruction model early into the training phase, using it to identify which tokens align better with instruction-aligned behavior, thereby training adapters that are more suitable for merging back into the instruction model.

**Key Insight**: The paper treats the probability assigned by the instruction model to the reference answer tokens as token importance. If the instruction model has high confidence in a token, it is more likely to be task-critical and instruction-aligned. Low-confidence tokens may represent noise, stylistic differences, or unstable regions and should have reduced training weights.

**Core Idea**: First, use the instruction model offline to calculate the confidence $q_t$ for each target token. Then, use $q_t$ to weight the cross-entropy loss of the base model's LoRA fine-tuning. After training, merge the LoRA adapter back into the instruction model.

## Method
The key to GIFT is transforming the "train on base, merge on instruct" transfer pipeline into "instruct-guided training on base, then merge back to instruct." It does not change the LoRA merging method but modifies the contribution of each token to the gradient during adapter learning.

### Overall Architecture
Given supervised data $\mathcal{D}=\{(x,y)\}$, base model parameters $\theta_B$, and instruction model parameters $\theta_I$. Existing transfer methods assume $\theta_I=\theta_B+\Delta_I$, thus the task update $M(\phi)$ trained on the base can be added to the instruction model: $\theta'_I=\theta_I+M(\phi)$. GIFT retains this merging logic but, before training $\phi$, uses the instruction model to calculate $q_t=p_{\theta_I}(y_t|x,y_{<t})$ for each target token. Subsequently, when training the LoRA adapter on the base model, it weights the token loss using $q_t$. Once training is complete, the adapter is merged back into the instruction model to produce a task-enhanced instruction model.

### Key Designs
1. **Offline Instruction-confidence Labeling**:
    - **Function**: Generates instruction-aligned importance scores for each target token in the training set.
    - **Mechanism**: For each sample $(x,y)$, a forward pass is performed using the instruction model to record $q_t=p_{\theta_I}(y_t|x,y_{<t})$. These scores are stored as augmented training samples $(x,y,\mathbf{q})$, avoiding repeated teacher calls during training.
    - **Design Motivation**: The instruction model, having undergone post-training, knows which expressions better align with instruction-following and task resolution. Its confidence reflects whether a token is suitable for the instruction space more effectively than the base model's own likelihood.

2. **Confidence-weighted Base Adapter Training**:
    - **Function**: Focuses the adapter on learning high-confidence, task-critical tokens from the instruction model while reducing incompatible updates from low-confidence regions.
    - **Mechanism**: Standard SFT loss is the sum of negative log-likelihoods across all tokens; GIFT modifies this to $\mathcal{L}_{\mathrm{GIFT}}(\phi)=\mathbb{E}_{(x,y)\sim\mathcal{D}}[\sum_{t=1}^T q_t\ell_t(\phi)]$, where $\ell_t(\phi)=-\log p_{\theta_B,\phi}(y_t|x,y_{<t})$. The raw $q_t$ is used directly without additional normalization, truncation, or temperature scaling.
    - **Design Motivation**: This is not distillation, as it does not fit the teacher distribution or minimize KL divergence; it simply redistributes standard CE gradients using teacher confidence to bias updates toward instruction-consistent directions.

3. **Adapter Transfer and Merge**:
    - **Function**: Injects task capabilities learned on the base into the instruction model while preserving original general capabilities and instruction-following.
    - **Mechanism**: After obtaining the optimal adapter $\phi^\star$, it is merged via standard LoRA merging: $\theta'_I=\theta_I+M(\phi^\star)$. Standard LoRA merging is used intentionally to isolate the contribution of guided fine-tuning.
    - **Design Motivation**: While advanced merging techniques like TIES, DARE, or Fisher-weighted averaging could be used, the core contribution of GIFT lies in the training phase rather than post-hoc merging tricks.

### Loss & Training
All methods use the same LoRA settings: AdamW, 1 epoch, max sequence length 2048, global batch size 256, learning rate $2\times10^{-4}$, LoRA rank $r=64$, scaling $\alpha=128$, dropout 0.05, and warmup ratio 0.1. Math tasks are trained on 2,000 samples from NuminaMath-CoT and evaluated on Math500, Minerva Math, OlympiadBench, AIME 2024, and AMC 2023. Inference uses temperature 1.0, max length 4096, with 16 samples per problem reporting Average@16. Medical tasks use 10,000 MedMCQA samples, reporting accuracy on MedQA, MMLU-medical, and MedMCQA test.

## Key Experimental Results

### Main Results
| Dataset / Model | Metric | GIFT | Raw Instruct / Strong Baseline | Gain / Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| Llama3.1-8B Math (5 tasks) | Average@16 | 22.0 | Instruct 16.8; Shadow-FT 18.0 | +5.2 over original, +4.0 over Shadow-FT |
| Llama3.2-3B Math (5 tasks) | Average@16 | 19.5 | Instruct 16.5; Shadow-FT 16.7 | Stable gains on small models |
| Qwen2.5-7B Math (5 tasks) | Average@16 | 42.9 | Instruct 41.3; Shadow-FT 38.0 | +1.6 even on strong instruction models |
| DeepSeek-Math-7B Math (5 tasks) | Average@16 | 19.7 | Instruct 16.8; Shadow-FT 17.4 | Math-specific models also benefit |
| Llama3.1-8B Medical QA | Avg Accuracy | 68.8 | Instruct 62.6; Shadow-FT 65.1 | +6.2 on knowledge-intensive tasks |
| MedQA | Accuracy | 68.3 | Instruct 55.2; Shadow-FT 65.6 | Largest gains in factual reasoning |
| MMLU-medical | Accuracy | 77.7 | Instruct 75.1; Shadow-FT 73.8 | Maintains and improves general knowledge |
| MedMCQA | Accuracy | 60.2 | Instruct 57.4; Shadow-FT 55.9 | Leads within the training domain |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Qwen2.5-7B Instruct | Math Avg: 41.3 | Original instruction baseline is strong |
| SFT+Merge / Shadow-FT | Math Avg: 38.0 | Standard base adapter merging causes -3.3 degradation |
| GIFT-BaseT | Math Avg: 41.0 | Using base model as teacher only recovers the baseline |
| GIFT (Full) | Math Avg: 42.9 | Instruction teacher guidance provides stable gains |
| Qwen2.5-7B MMLU / IFEval | 68.8 / 72.1 | vs original 68.7 / 71.2; no harm to general knowledge |
| Llama3.1-8B MMLU / IFEval | 63.7 / 74.7 | vs original 63.2 / 73.8; maintains or improves performance |
| Super-NI Summarization | EM 12.00; R-L 40.28 | vs original 10.75 / 37.38; generalizes beyond math/med |
| Qwen2.5 Scale (0.5B->32B) | 0.5B: 8.3 vs 7.9 | GIFT outperforms instruct baseline across scales |

### Key Findings
- **Direct fine-tuning of instruction models degrades performance** under limited supervision. For instance, Llama3.1-8B math average dropped from 16.8 to 8.9, and Qwen2.5-7B from 41.3 to 21.2, highlighting the risks of naive Instruct-SFT.
- **GIFT remains effective on strong models**. Where Qwen2.5-7B-Instruct scores 41.3 and Shadow-FT drops to 38.0, GIFT improves to 42.9, suggesting guidance enhances merge compatibility.
- **Learning signal analysis explains the mechanism**: In Shadow-FT, 79.7% of learning signals come from low-confidence tokens, while high-confidence tokens account for only 6.9%. GIFT reduces low-confidence contribution to 29.6% and increases high-confidence contribution to 31.5%.
- **Low offline annotation cost**: For 2,000 samples, Llama3.1-8B takes 2m11s and Qwen2.5-7B takes 2m9s on a single RTX 4090, with peak memory under 22GB.

## Highlights & Insights
- **Turning the instruction model from a "merge target" to a "training mentor"**: This is the most elegant conceptual shift in the paper. Since the adapter is eventually merged back, the instruction model should help decide what the adapter learns.
- **Lower cost than distillation**: GIFT avoids step-by-step teacher-student KL divergence and online teacher inference. It uses a single offline pass for token confidence annotation, preserving the simplicity of SFT.
- **Explanation for Shadow-FT instability**: Standard CE is dominated by low-confidence, high-loss tokens, which correspond to regions the instruction model does not approve of; merging these updates disrupts original instruction capabilities.
- **Suitability for low-data task adaptation**: Using only 2,000 math samples and 10,000 medical samples, results show that selecting the right learning signals is more important than increasing training intensity when data is limited and specialized.

## Limitations & Future Work
- GIFT requires an additional offline annotation pass. While cost is negligible for 2K samples, preprocessing time and storage would scale with millions of samples.
- The paper primarily uses standard LoRA merging to isolate variables; the benefits of combining GIFT with TIES or Fisher-weighted averaging remain unexplored.
- Guidance relies directly on token probabilities; alternative designs using normalization, temperature, or sequence/step-level confidence were not studied.
- If the instruction model has weak domain knowledge or strong systematic biases, its confidence might incorrectly suppress valuable new knowledge. Future research is needed on the tension between teacher confidence and task novelty.

## Related Work & Insights
- **vs Shadow-FT / Chat Vector**: These also train task updates on base models and merge back, but the instruction model only appears during merging. GIFT uses instruction confidence to shape the update direction during training.
- **vs Re-Adapt / Task Arithmetic**: These focus on linear combinations of base, instruction offsets, and task updates; GIFT focuses on making the task update more compatible during learning.
- **vs TIES / DARE**: TIES and DARE address interference and redundancy during post-hoc merging; GIFT addresses the issue of low-quality gradients entering the adapter during training.
- **Insight**: In model adaptation, the teacher model does not necessarily need to provide the full answer distribution; simply providing weights for "which tokens are worth learning" can significantly improve parameter update quality.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The confidence-weighting idea is intuitive but accurately positions the instruction model as a mentor for base adaptation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers math, medicine, summarization, general capabilities, model scales, and learning signal analysis. Could be bolstered with more domains.
- **Writing Quality**: ⭐⭐⭐⭐☆ Method definitions are concise, with convincing ablation and mechanism analyses and well-supported tables.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical for low-cost domain adaptation of open-source instruction models, particularly for LoRA/adapter workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ICLR 2026\] Fine-tuning with RAG for Improving LLM Learning of New Skills](../../ICLR2026/information_retrieval/fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[AAAI 2026\] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description](../../AAAI2026/information_retrieval/oad-promoter_enhancing_zero-shot_vqa_using_large_language_models_with_object_att.md)
- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)

</div>

<!-- RELATED:END -->
