---
title: >-
  [Paper Note] Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax
description: >-
  [ACL2026][Multilingual & Machine Translation][Low-resource languages] This paper reformulates low-resource language expansion from token-level imitation into a semantic space alignment problem. By training Qwen3-4B with…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "Low-resource languages"
  - "Semantic reward"
  - "GRPO"
  - "Alignment tax"
  - "Tibetan-Chinese translation"
date: 2026-05-08
content_hash: d9e95a5df7c97eaf
---

# Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax

**Conference**: ACL2026  
**arXiv**: [2605.14366](https://arxiv.org/abs/2605.14366)  
**Code**: Not released  
**Area**: multilingual_mt  
**Keywords**: Low-resource languages, Semantic reward, GRPO, Alignment tax, Tibetan-Chinese translation

## TL;DR
This paper reformulates low-resource language expansion from token-level imitation into a semantic space alignment problem. By training Qwen3-4B with GRPO and embedding-based semantic rewards, the model acquires low-resource capabilities in Tibetan-Chinese translation and Tibetan title generation while preserving dominant language abilities (e.g., Chinese CMRC) better than strong SFT.

## Background & Motivation
**Background**: Large language models perform strongly in high-resource languages but lack support for low-resource languages like Tibetan. Common practices include continued pre-training, instruction fine-tuning, or SFT on low-resource parallel corpora to pull the model toward the target language data distribution.

**Limitations of Prior Work**: Low-resource corpora are often small, domain-specific, and biased. Token-level teacher forcing encourages the model to forcibly imitate the surface form of references. While this improves BLEU or ROUGE, it easily leads to parameter over-adaptation to narrow data, causing a decline in high-resource language and general capabilities, referred to as the "alignment tax."

**Key Challenge**: Low-resource expansion must acquire the target language without destroying existing general representations from pre-training. SFT defines "alignment" as token distribution matching, whereas true linguistic ability is closer to "multiple semantically equivalent expressions are acceptable." Stronger surface imitation increases the risk of forgetting.

**Goal**: The authors aim to answer three questions: whether semantic reward RL can effectively learn low-resource tasks; how the trade-off between task performance and general capability retention compares to strong SFT; and whether representations obtained through semantic alignment transfer better to downstream few-shot tasks.

**Key Insight**: The paper views language expansion as alignment under sparse supervision rather than simple adaptation. The model is no longer required to reproduce a unique reference sentence but instead learns to preserve semantics via embedding similarity, while reducing parameter drift through constrained policy optimization.

**Core Idea**: Use embedding-level semantic rewards instead of token-level likelihood to allow the model to learn low-resource language abilities ("conveying the meaning is sufficient") while using controlled updates in GRPO to reduce catastrophic forgetting.

## Method
The method consists of a two-stage training paradigm and a semantic reward function. The first stage uses a small amount of low-resource data to give the model basic output capabilities. The second stage starts from this cold-start model and continues optimization using GRPO based on semantic rewards. Compared to strong SFT, it does not pursue exact matches with reference text but allows for diverse surface expressions as long as the semantics are consistent with the reference and the language remains within the target low-resource space.

### Overall Architecture
The input consists of low-resource language task samples, such as Tibetan-Chinese parallel sentences or Tibetan title generation samples. The model first undergoes cold-start SFT on 5k low-resource samples to obtain an initial policy capable of outputting the target language/format. Subsequently, GRPO is performed on the remaining data: for each prompt, a group of candidate outputs is sampled, semantic similarity between candidates and references is calculated using a frozen multilingual sentence embedding model, and language consistency rewards are added. Finally, the policy is updated based on relative rewards within the group. The output is a model enhanced for low-resource languages with minimal damage to existing capabilities.

### Key Designs
1.  **Semantic Space Alignment Objective**:
    - **Function**: Transitions the low-resource training objective from "reproducing reference tokens" to "preserving reference semantics."
    - **Mechanism**: If the generated sentence and the reference sentence are semantically close in the sentence embedding space, it is considered a valid answer, even if the surface word order or phrasing differs. This allows the model to explore multiple reasonable expressions rather than being locked into a narrow reference distribution.
    - **Design Motivation**: Reference texts in low-resource data are often incomplete. Forcibly maximizing likelihood amplifies data bias. Semantic objectives better align with the "many-to-one" nature of translation and generation.

2.  **Two-stage Cold-start + GRPO Training**:
    - **Function**: Avoids ineffective early exploration in RL while utilizing controlled policy optimization to continue learning low-resource capabilities.
    - **Mechanism**: Small-scale SFT first ensures the model can output correct characters and basic formats. Then, GRPO samples 8 candidates per prompt and updates the policy based on group relative rewards. GRPO does not require an explicit value model and maintains the stability of PPO-like methods in limiting policy drift.
    - **Design Motivation**: Direct RL on low-resource languages may produce degraded outputs; direct SFT leads to forgetting. Cold-start handles "how to speak," while semantic RL ensures "speaking meaningfully without drifting too far."

3.  **Embedding Reward and Language Consistency Constraint**:
    - **Function**: Simultaneously encourages semantic correctness and target language consistency to avoid reward hacking.
    - **Mechanism**: The primary reward is the cosine similarity between the generated output and the reference text, recalibrated via a threshold $\tau$. If similarity is below a minimum semantic sufficiency level, the reward is 0; it is linearly amplified only when above the threshold. Another reward uses Unicode/rules to check if the output is mixed-language. The final reward is $R=\lambda_{sim}R_{sim}+\lambda_{lang}R_{lang}$, where $\lambda_{sim}=1.5$ and $\lambda_{lang}=1.0$.
    - **Design Motivation**: Multilingual embeddings might assign high semantic scores to mixed-language outputs. Hard constraints are necessary to restrict optimization to the target low-resource language space.

### Loss & Training
The experiments use Qwen3-4B with parameter-efficient fine-tuning via LoRA on attention and MLP linear layers (LoRA rank 64, $\alpha=128$, dropout 0.05). SFT uses AdamW, a learning rate of $2\times10^{-5}$, batch size 32, and a cosine schedule. GRPO starts from the cold-start checkpoint for 1 epoch, with a learning rate of $5\times10^{-7}$, sampling 8 candidates per prompt, temperature 0.8, and top-p 0.9. The semantic reward model is based on CINO/XLM-R, adapted into a SentenceTransformer using Sino-Tibetan parallel data, and frozen during RL.

## Key Experimental Results

### Main Results
The first set of experiments compares cold-start SFT and semantic reward RL, proving that RL can indeed continue to improve low-resource capabilities—especially semantic similarity—after minimal supervised initialization.

| Task | Model | Task Metric | Semantic Similarity | Main Conclusion |
|------|------|----------|------------|----------|
| Tib-Chi Translation | Cold-start SFT | BLEU-4 0.3953 | 0.5593 | 5k samples enable basic translation capability |
| Tib-Chi Translation | RL (Ours) | BLEU-4 0.4519 | 0.7164 | Semantic rewards bring significant semantic improvement |
| Tibetan Title Gen | Cold-start SFT | ROUGE-L 0.2204 | 0.5774 | Baseline can generate but lacks semantic depth |
| Tibetan Title Gen | RL (Ours) | ROUGE-L 0.2530 | 0.6404 | Improvement also seen in generation tasks |

### Ablation Study
The trade-off with strong SFT indicates that while SFT is better at pursuing reference overlap, it is not necessarily superior in preserving general capabilities or open-ended generation preferences.

| Task | Method | Task Metric | Similarity | CMRC Avg | CMRC F1 | LLM-Judge Win |
|------|------|----------|------------|----------|---------|---------------|
| Tib-Chi Translation | Strong SFT | 0.6006 | 0.8282 | 41.82 | 62.99 | 59.2% |
| Tib-Chi Translation | RL (Ours) | 0.4519 | 0.7164 | 46.97 | 65.79 | 33.5% |
| Tibetan Title Gen | Strong SFT | 0.3095 | 0.6499 | 44.20 | 65.30 | 35.1% |
| Tibetan Title Gen | RL (Ours) | 0.2530 | 0.6404 | 45.10 | 65.20 | 51.2% |

Reward ablation further shows that the improvement is not brought by "using RL" itself, but by the critical design of the semantic reward.

| Reward Config | MT Similarity | Description |
|----------|---------------|------|
| Embedding + LC (Ours) | 0.7164 | Best, balances semantics and target language consistency |
| BLEU + LC | 0.6375 | Token overlap reward restricts semantic exploration |
| BLEU + Embedding + LC | 0.6175 | Mixing BLEU actually degrades performance |
| BLEU + Embedding | 0.2312 | Easy to mix languages without language consistency |

### Key Findings
- Semantic RL can continue to improve low-resource tasks from a cold-start SFT; Tibetan-Chinese translation similarity increased from 0.5593 to 0.7164, and title generation similarity increased from 0.5774 to 0.6404.
- Strong SFT achieves higher BLEU and similarity in Tibetan-Chinese translation, but its CMRC Avg is 5.15 points lower than RL, proving that the metric gains from surface imitation come with a higher alignment tax.
- In open-ended title generation, RL has lower ROUGE, but its LLM-Judge win rate reaches 51.2%, 16.1 percentage points higher than SFT, suggesting n-gram metrics underestimate diverse semantic expressions.
- In Few-shot transfer, MT-RL initialization achieved higher similarity on 1,000 title generation samples (0.5690 vs. 0.5456 for MT-SFT), supporting the argument that semantically aligned representations are more transferable.
- OOD mechanism analysis shows that the CMRC mean NLL increase for RL is +0.24, compared to +0.64 for SFT; the P90 NLL increase also dropped from +1.43 in SFT to +0.62 in RL, indicating that forgetting occurs more in the tail of difficult samples for SFT.

## Highlights & Insights
- The strongest argument of the paper is defining low-resource language expansion as alignment rather than ordinary fine-tuning. This perspective naturally explains why token-level learning triggers an alignment tax and provides rationale for using RL to control update magnitude.
- The combination of embedding reward and language consistency constraints is elegant. It does not introduce complex teachers or preference annotations but makes the core conditions of "semantic sufficiency" and "remaining in the target language" explicit.
- The reflection on reference-based metrics is valuable: in low-resource tasks, BLEU/ROUGE may reward narrow reference imitation instead of robust linguistic ability. This is insightful for evaluating machine translation and generation systems.
- This method can be transferred to other weakly supported languages, dialects, or specialized domain text expansions: cold-start with a small amount of data, then expand the expression space with semantic rewards while preventing output drift with rules or detectors.

## Limitations & Future Work
- The experiments mainly focus on Tibetan, and the Tibetan-Chinese translation data comes from internal VLM pre-training construction pipelines with narrow domain distribution; external reproducibility and cross-lingual generalization require more public benchmarks.
- The semantic reward model itself is trained on Sino-Tibetan parallel data. If the reward model is insensitive to fine-grained semantic differences, RL might optimize toward incorrect semantic neighbors.
- LLM-as-a-Judge is used for preference evaluation; while it supplements ROUGE/BLEU, it is still not human evaluation and may harbor biases or recognition errors, especially in low-resource languages.
- The method sacrifices portion of reference-based metrics for capability retention; actual deployment requires a decision on whether to accept this trade-off based on the task type.

## Related Work & Insights
- **vs. Low-resource SFT / continued pretraining**: These methods continue to use token-level likelihood, whereas this paper changes the objective function to replace surface distribution matching with semantic consistency, focusing more on reducing catastrophic forgetting.
- **vs. LoRA / Parameter-efficient forgetting mitigation**: Approaches like LoRA primarily limit update locations or parameter volume, while this paper limits the optimization objective and policy drift; the two can be complementary.
- **vs. RLHF / DPO**: Standard alignment RL optimizes for human preference or rule-based rewards. This paper treats rewards as task rewards across semantic spaces, suitable for low-resource scenarios lacking human preference data.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using "semantic space alignment + GRPO" for low-resource language expansion is an interesting idea, though core components leverage existing RL and embedding methods.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Main experiments, trade-offs, transfer, and ablations are relatively complete, but the range of languages, data publicity, and human evaluation are insufficient.
- Writing Quality: ⭐⭐⭐⭐☆ The main line of reasoning is clear, especially the explanation of the alignment tax.
- Value: ⭐⭐⭐⭐☆ Provides high practical inspiration for low-resource language expansion, catastrophic forgetting mitigation, and reflection on reference metrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)
- [\[ACL 2026\] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning](neoamt_neologism-aware_agentic_machine_translation_with_reinforcement_learning.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)
- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ACL 2026\] Selective Contrastive Learning For Gloss Free Sign Language Translation](selective_contrastive_learning_for_gloss_free_sign_language_translation.md)

</div>

<!-- RELATED:END -->
