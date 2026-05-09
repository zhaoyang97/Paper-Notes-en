---
title: >-
  [Paper Note] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][large reasoning models] This work presents the first systematic study of answer attribution in large reasoning models (LRMs), revealing that reasoning (CoT) and retrieval (memory) mechanisms compete simultaneously to influence final answers. The paper proposes Farl (Forgetting-Augmented Reinforcement Learning), which suppresses retrieval shortcuts to enhance genuine reasoning capability.
tags:
  - ICLR 2026
  - LLM Reasoning
  - large reasoning models
  - CoT reasoning
  - memory retrieval
  - answer attribution
  - reinforcement-learning
  - unlearning
  - GRPO
date: 2026-05-08
content_hash: dbe4959356610c6a
---

# Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models

**Conference**: ICLR 2026
**arXiv**: [2509.24156](https://arxiv.org/abs/2509.24156)
**Code**: [ZJUWYH/FARL](https://github.com/ZJUWYH/FARL)
**Area**: LLM Reasoning
**Keywords**: large reasoning models, CoT reasoning, memory retrieval, answer attribution, reinforcement-learning, unlearning, GRPO

## TL;DR

This work presents the first systematic study of answer attribution in large reasoning models (LRMs), revealing that reasoning (CoT) and retrieval (memory) mechanisms compete simultaneously to influence final answers. The paper proposes Farl (Forgetting-Augmented Reinforcement Learning), which suppresses retrieval shortcuts to enhance genuine reasoning capability.

## Background & Motivation

Large reasoning models (e.g., DeepSeek-R1, GPT o-series) demonstrate strong problem-solving ability through chain-of-thought (CoT) reasoning. However, growing evidence suggests that the final answers of these models frequently diverge from their reasoning processes:

**Reasoning–answer disconnect**: Final answers are not always directly produced by the CoT process; contextual biases can influence outputs without being acknowledged in the CoT.

**Dual-mechanism hypothesis**: Models may generate answers via two concurrent pathways—"deliberate reasoning" and "direct retrieval from internal memory."

**Unclear impact of training methods**: The effects of distillation and reinforcement learning on these two mechanisms have not been systematically studied.

Core research questions:
- **RQ1**: Do LRMs simultaneously employ both reasoning and retrieval to derive answers?
- **RQ2**: What factors govern the relative dominance of each mechanism?
- **RQ3**: How can the relative strength of the two mechanisms be controlled?

## Method

### Overall Architecture

The paper proposes a **joint reasoning-retrieval perturbation framework** that quantifies the contribution of each mechanism by independently perturbing the reasoning and retrieval pathways and observing changes in the final answer.

### Key Designs

**Reasoning perturbation**: A misleading cue $c$ (e.g., "a reliable expert suggests the answer is B") is injected at the end of the model-generated CoT. The tampered CoT is then prefilled into the prompt for regeneration:
$$\mathcal{M}(x \| z \| c; \theta) = y'$$
If $y' = y_r$ (the misleading answer), the CoT modification has successfully influenced the final answer.

**Retrieval perturbation**: The model's memory is "poisoned" via supervised fine-tuning (SFT), forcing the model to associate specific questions with incorrect answers:
$$\min_\theta \ell(y_t, \mathcal{M}(x;\theta))$$
where $y_t$ is the non-correct answer with the highest logit in the original model. Training uses LoRA ($r=64, \alpha=16$) with AdamW (lr=1e-4) for 8 epochs.

**Joint perturbation**: Both perturbations are applied simultaneously to create a "tug-of-war" effect:
$$\mathcal{M}(x \| z \| c; \theta') = y'$$
Two conditions are examined: (i) both perturbations target the same incorrect answer ($y_r = y_t$); (ii) they target different incorrect answers ($y_r \neq y_t$).

**Evaluation metrics**:
- R-PSR (Reasoning Perturbation Success Rate): $\text{R-PSR} = \mathbb{E}_{(x,y)} \mathbf{1}[y' = y_r]$
- T-PSR (retrieval perturbation success rate): $\text{T-PSR} = \mathbb{E}_{(x,y)} \mathbf{1}[y' = y_t]$
- PER (Post-hoc Explanation Rate): the proportion of cases where the CoT logically supports the poisoned answer

### Farl: Forgetting-Augmented Reinforcement Learning

This component is motivated by a key insight: the retrieval mechanism can serve as a shortcut for reward hacking during RL training—models retrieve memorized answers to obtain high rewards without genuine reasoning.

Farl integrates a forgetting step into the standard GRPO pipeline:
1. Each epoch: set reference model → perform GRPO iterations → **execute NPO unlearning**
2. GRPO advantage computation: $\hat{A}_j = \frac{r(x,z_j,y_j) - \text{mean}(\{r\}_{j=1}^G)}{\text{std}(\{r\}_{j=1}^G)}$
3. Unlearning applies Negative Preference Optimization (NPO) to suppress retrieval pathways for memorized answers

### Loss & Training

The GRPO objective $\mathcal{J}_{\text{GRPO}}$ and the NPO unlearning loss $\mathcal{L}_{\text{NPO}}$ are alternately optimized.

## Key Experimental Results

### Main Results

| Method | R-PSR ↓ | T-PSR ↓ | In-domain ACC ↑ | Out-of-domain ACC ↑ |
|--------|---------|---------|-----------------|----------------------|
| R1-Llama-8B (Base) | 0.378 | 0.381 | 0.725 | 0.716 |
| SFT | 0.392 | 0.311 | 0.787 | 0.732 |
| RL (GRPO) | 0.259 | 0.262 | 0.869 | 0.745 |
| **Farl** | **0.197** | **0.234** | **0.891** | **0.757** |

Relative to the base model, Farl reduces R-PSR by 47.8% and T-PSR by 38.5%, while improving in-domain accuracy by 22.8% and out-of-domain accuracy by 5.8%.

### Ablation Study / Factor Analysis

**Problem domain**: Math and logic domains exhibit the lowest T-PSR and R-PSR values, indicating that models rely more heavily on reasoning than memory in these areas.

**Training method comparison**: Distilled models show significantly higher T-PSR and R-PSR than RL-trained models, suggesting that distillation favors memorization over reasoning. Distilled models also exhibit markedly higher PER—they fabricate CoT to rationalize memorized answers.

**Model scale**: Larger models consistently exhibit lower PER, T-PSR, and R-PSR, indicating that reasoning becomes more dominant as model size increases.

**Attention mechanism analysis**: Attention heads in intermediate layers (layers 12–16) achieve the highest AUC in classifying reasoning versus retrieval pathways. Causal intervention experiments confirm that replacing activations of high-AUC heads recovers the original answer in 87.2% of cases, compared to only 5.3% for randomly selected heads.

### Key Findings

1. Reasoning and retrieval mechanisms **coexist and compete**; both perturbations can independently alter the final answer.
2. When both perturbations target the same answer, their effects are **synergistically amplified**.
3. Distilled models exhibit severe **post-hoc explanation** behavior: after memory poisoning, they not only output incorrect answers but also fabricate CoT that supports those answers.
4. CoT quality metrics (cycle count +37.0%, diameter +5.7%, small-world index +84.0%) indicate that Farl produces higher-quality reasoning traces.

## Highlights & Insights

1. **First mechanistic study**: The first systematic investigation of the competition between reasoning and retrieval in LRM answer generation, offering deep mechanistic insight.
2. **Elegant experimental design**: The joint perturbation framework cleanly separates and quantifies the contributions of both mechanisms.
3. **Causal evidence**: Beyond correlational analysis (attention head AUC), the paper provides causal intervention evidence through activation replacement.
4. **Logit dynamics visualization**: Step-by-step tracking of the logit competition between the two pathways throughout the reasoning process vividly illustrates the dynamic reasoning–retrieval interaction.
5. **Practical implication**: The "unlearning + RL" paradigm of Farl offers a new direction for enhancing genuine reasoning capability in models.

## Limitations & Future Work

1. Although Farl improves reasoning capability, it generates longer reasoning chains (MTL increases from 1,537 to 1,914), reducing inference efficiency.
2. Due to computational constraints, validation is limited to R1-Llama-8B and R1-Qwen-7B; conclusions for larger models remain to be verified.
3. Retrieval perturbation is implemented via SFT; while its locality and efficiency are validated, its relationship to genuine "memory" warrants further discussion.
4. Training is conducted only in the Math & Logic domain, resulting in limited transfer to other domains (out-of-domain gain of only +5.8%).

## Related Work & Insights

- **Relation to reasoning–answer disconnect research**: Turpin et al., Lanham et al., and others have identified CoT unfaithfulness; this work goes further by uncovering the underlying dual-mechanism explanation.
- **Relation to memory editing**: Meng et al.'s ROME/MEMIT focus on editing the retrieval mechanism, whereas this work treats retrieval as a pathway competing with reasoning.
- **Implications for RL post-training**: The work reveals a novel form of reward hacking in RL training—models can obtain rewards by retrieving memorized answers rather than engaging in genuine reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic study of the reasoning–retrieval dual mechanism in LRMs, with deep and original insights.
- **Practicality**: ⭐⭐⭐⭐ — Farl is effective but its applicability scope requires further extension.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Progressively advances from behavioral experiments to mechanistic analysis to causal intervention.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Research-question-driven, clearly structured, with outstanding visualizations.
- **Overall**: ⭐⭐⭐⭐⭐ — Uncovers a critical mechanistic issue in LRMs with important implications for future research.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)
- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Segment-Level Attribution for Selective Learning of Long Reasoning Traces](segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)

<!-- RELATED:END -->
