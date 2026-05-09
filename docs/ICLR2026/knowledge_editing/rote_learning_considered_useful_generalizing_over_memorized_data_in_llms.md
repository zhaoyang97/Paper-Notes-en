---
title: >-
  [Paper Note] Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs
description: >-
  [ICLR 2026][Knowledge Editing][memorization] This paper proposes a "memorize-then-generalize" framework that employs a two-stage strategy—first memorizing factual associations via semantics-free synthetic tokens through rote learning, then fine-tuning with a small number of semantic prompts—to demonstrate that LLMs can generalize from rote-memorized data. Deeper memorization yields better generalization, and the paper further identifies security risks arising from potential malicious exploitation of this mechanism.
tags:
  - ICLR 2026
  - Knowledge Editing
  - memorization
  - generalization
  - knowledge injection
  - rote learning
  - LLM training dynamics
date: 2026-05-08
content_hash: 5018193d8acff936
---

# Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs

**Conference**: ICLR 2026
**arXiv**: [2507.21914](https://arxiv.org/abs/2507.21914)
**Code**: [https://github.com/QinyuanWu0710/memorize-then-generalize](https://github.com/QinyuanWu0710/memorize-then-generalize)
**Area**: Knowledge Editing
**Keywords**: memorization, generalization, knowledge injection, rote learning, LLM training dynamics

## TL;DR
This paper proposes a "memorize-then-generalize" framework that employs a two-stage strategy—first memorizing factual associations via semantics-free synthetic tokens through rote learning, then fine-tuning with a small number of semantic prompts—to demonstrate that LLMs can generalize from rote-memorized data. Deeper memorization yields better generalization, and the paper further identifies security risks arising from potential malicious exploitation of this mechanism.

## Background & Motivation
Rote learning has traditionally been equated with overfitting in deep learning and is widely believed to impair generalization. This view is especially entrenched in the LLM community:

- Pre-training is typically restricted to 1–2 epochs to prevent memorization.
- Memorization has been associated with privacy leakage, hallucination, and vulnerability to paraphrase.
- Prior work has found that memorized knowledge interferes with generalization in subsequent fine-tuning.

**Root Cause**: Acquiring factual knowledge inherently requires some degree of memorization, yet the relationship between memorization and generalization in LLMs remains poorly understood. The grokking phenomenon suggests that generalization can emerge after extensive memorization, but systematic investigation is lacking.

**Starting Point**: The authors construct an elegant two-stage framework that disentangles memorization from generalization—Phase 1 enforces pure rote memorization using semantics-free tokens, while Phase 2 guides generalization using a minimal number of semantic prompts. This design eliminates interference from language understanding and cleanly tests whether memorized data can be reinterpreted.

## Method

### Overall Architecture
A two-stage framework operating on ⟨subject, relation, object⟩ factual triples:

- **Phase 1 (Rote Memorization)**: The relation description is replaced by a synthetic, semantics-free key token `[X]`, and the model is trained to memorize associations of the form "Gene Finley [X] Cody Ross" via unsupervised next-token prediction over multiple epochs until complete memorization is achieved.
- **Phase 2 (Generalization)**: A small subset of memorized facts is used for SFT with semantic prompts (e.g., "Who is Gene Finley's mother?"), endowing `[X]` with semantic meaning.

### Key Designs
1. **Synthetic semantics-free key tokens**:

    - Each relation corresponds to a unique synthetic token `[X]` deliberately carrying no semantic information.
    - Function: Eliminates interference from language understanding and isolates pure memorization behavior.
    - Validation: Without key tokens, the model fails to generalize (ablation study, Figure 9).

2. **Three-level generalization evaluation**:

    - **(a) Unseen associations**: Can facts not seen in Phase 2 be retrieved using the training prompt?
    - **(b) Unseen prompts**: Can the model generalize to semantically equivalent but differently worded prompts?
    - **(c) Unseen languages**: Can generalization transfer across languages (German/Spanish/Chinese/Japanese)?

3. **Fully synthetic dataset**: GPT-4 is used to generate 5 T-REx relations (author, capital, educated at, genre, mother), each comprising 100 fictitious fact pairs, 100 multiple-choice distractors, 20 prompt variants, and translations into 4 languages, thereby preventing contamination from pre-training knowledge.

### Loss & Training
- Phase 1: Unsupervised next-token prediction, trained for 3–20 epochs.
- Phase 2: SFT using only 1 prompt and $k$ fact pairs (minimum $k=1$), for 1 epoch.
- Evaluation metrics: Generation accuracy, multiple-choice accuracy, object probability.

## Key Experimental Results

### Main Results (Deeper Memorization Yields Better Generalization)

| Phase-1 Epoch | Key Token Acc | Phase-2 k | Train Prompt Acc | Test Prompt Acc |
|:---:|:---:|:---:|:---:|:---:|
| 3 | 0.48 | 50 | 0.38 | 0.35 |
| 6 | 1.00 | 50 | 0.94 | 0.89 |
| 10 | 1.00 | 50 | 0.94 | 0.98 |
| 20 | 1.00 | 50 | 1.00 | 0.98 |
| 10 | 1.00 | **1** | 1.00 | 0.75 |
| 20 | 1.00 | **1** | 1.00 | 0.76 |

- Using only **1 fact pair and 1 prompt**, the model achieves an accuracy of 0.76 on test prompts.
- As Phase-1 epochs increase from 3 to 20, test prompt accuracy rises sharply from 0.35 to 0.98.

### Ablation Study (Cross-Model Validation & Comparison with Baselines)

| Method | Training Efficiency | Accuracy (1 prompt) | Cross-lingual | Reasoning Support |
|------|:---:|:---:|:---:|:---:|
| **Memorize-then-Generalize** | High (1 token + k pairs) | ~0.76–1.00 | ✓ Strong | ✓ Reversal/multi-hop |
| SFT | Low (20× longer prompts) | ~0.3 (same token budget) | Moderate | ✗ Reversal 0.01 |
| ICL | No training | High but unstable | Weak (high variance) | ✓ (requires context) |

- The phenomenon is consistently observed across 8 models (Qwen2.5/Llama2/Llama3.2/Phi-4, 1B–14B).
- Cross-lingual performance follows: English > Spanish > German > Japanese > Chinese, positively correlated with linguistic similarity.

### Key Findings
1. **Representation space analysis**: During Phase 1, key token representations progressively cluster by relation (ΔCosSim increases from 0.058 to 0.191); after Phase 2, cosine similarity between key tokens and semantic prompts rises significantly (Test: 0.58→0.71).
2. **Multi-hop reasoning improvement**: After memorizing A→B and learning B→C, accuracy on A→C improves from 0.14 to 0.36 (20 epochs).
3. **Malicious exploitation risk**: After memorizing benign facts, fine-tuning with only 50 malicious variants enables the model to simultaneously respond to both benign and malicious prompts (dual generalization) while appearing entirely normal from the outside.

## Highlights & Insights
- **Counterintuitive finding**: Memorization is not the antithesis of generalization but rather its foundation—"the more thoroughly memorized, the better the generalization" directly challenges the prevailing assumption.
- **Extreme data efficiency**: Generalization can be driven by as few as 1 fact pair and 1 prompt, suggesting that LLMs possess a powerful capacity for semantic reuse.
- **Key tokens as semantic anchors**: Semantics-free tokens, once memorized, can be "reinterpreted" as arbitrary semantic content, functioning analogously to programmable relational slots.
- **Profound security implications**: Data unintentionally memorized during pre-training could be "weaponized" through malicious fine-tuning.

## Limitations & Future Work
- Experiments rely entirely on synthetic data, creating a gap with real-world knowledge injection scenarios.
- The work is limited to factual knowledge (triples) and does not extend to more complex tasks such as mathematical reasoning or code generation.
- The malicious exploitation component only demonstrates feasibility without proposing defensive countermeasures.
- The optimal design of key tokens (number, initialization strategy) remains insufficiently explored.
- Cross-lingual generalization is tied to typological distance, yielding limited effectiveness for typologically distant languages such as Chinese.

## Related Work & Insights
- The work is closely related to the grokking phenomenon (Power et al. 2022) and can be viewed as a systematic investigation of grokking in the domain of factual knowledge.
- The findings offer insights for parameter-efficient fine-tuning methods such as LoRA/adapters: could relations be first "memorized" and then efficiently generalized?
- The framework provides a novel knowledge injection paradigm with implications for knowledge editing methods.
- The security dimension calls for the development of detection and defense tools against "dual generalization."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Counterintuitively demonstrates that rote learning can promote generalization, with an elegantly designed framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8 models × 5 relations, includes representation analysis and application scenarios, though all data are synthetic.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, intuitive visualizations, and a complete narrative structure.
- Value: ⭐⭐⭐⭐⭐ Offers a new understanding of the memorization–generalization relationship with significant implications for both knowledge injection and security research.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Rote Learning Considered Useful: Generalizing over Memorized Training Examples](rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)
- [\[ICLR 2026\] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)

<!-- RELATED:END -->
