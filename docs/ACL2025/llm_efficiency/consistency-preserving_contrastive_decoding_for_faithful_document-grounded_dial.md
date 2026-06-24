---
title: >-
  [Paper Note] Consistency-Preserving Contrastive Decoding for Faithful Document-Grounded Dialogue
description: >-
  [ACL 2025][LLM Efficiency][Contrastive Decoding] This paper proposes Consistency-Preserving Contrastive Decoding (CPCD), a method that contrasts document-conditioned and document-free generation distributions during the decoding phase. This strategy enhances the faithfulness of document-grounded dialogue systems to source documents while maintaining response fluency and dialogue consistency.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Contrastive Decoding"
  - "Document-Grounded Dialogue"
  - "Faithfulness"
  - "Hallucination Mitigation"
  - "Knowledge-Grounded Dialogue"
date: 2026-05-08
content_hash: 05b77fab6eda125e
---

# Consistency-Preserving Contrastive Decoding for Faithful Document-Grounded Dialogue

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM Efficiency  
**Keywords**: Contrastive Decoding, Document-Grounded Dialogue, Faithfulness, Hallucination Mitigation, Knowledge-Grounded Dialogue

## TL;DR
This paper proposes Consistency-Preserving Contrastive Decoding (CPCD), a method that contrasts document-conditioned and document-free generation distributions during the decoding phase. This strategy enhances the faithfulness of document-grounded dialogue systems to source documents while maintaining response fluency and dialogue consistency.

## Background & Motivation

**Background**: Document-Grounded Dialogue (DGD) requires dialogue systems to generate responses based on a given background document, which is widely applied in customer service and knowledge assistants. The mainstream approach is to train seq2seq models or fine-tune LLMs on Transformer architectures, taking both documents and dialogue history as inputs to generate responses.

**Limitations of Prior Work**: (1) Even when the correct document is provided in the input, models still suffer from "hallucination"—generating content that is irrelevant to or contradicts the document. Research shows that 30-50% of generated responses contain varying degrees of hallucination; (2) Existing hallucination mitigation methods primarily introduce faithfulness constraints during the training phase (e.g., NLI-based reranking), but training-inference mismatch limits their effectiveness; (3) Contrastive Decoding has been proven to improve generation quality, but direct application to DGD tasks destroys dialogue coherence as contrastive signals may over-penalize document-irrelevant but dialogue-beneficial expressions.

**Key Challenge**: Faithfulness requires responses to closely adhere to the document, whereas conversational quality demands natural, fluent, and contextually coherent responses. Simple contrastive decoding sacrifices dialogue quality when enhancing faithfulness.

**Goal**: To design a method that simultaneously optimizes faithfulness and dialogue consistency during the decoding phase without retraining the model.

**Key Insight**: The authors observe that standard contrastive decoding amplifies all tokens that differ significantly from the "document-free distribution", yet some of these differences stem from conditioning on the dialogue context rather than the document. By decoupling the contributions of the document condition and the dialogue condition, one can amplify only the information gain brought by the document, thereby preserving the coherence constraints of the dialogue context.

**Core Idea**: Introduce a "consistency-preserving" constraint—adding a third reference distribution (dialogue context only, no document) to contrastive decoding. This three-way comparison precisely decouples the document's contribution from the dialogue's contribution, amplifying only the former.

## Method

### Overall Architecture
Given the dialogue history $H$, the document $D$, and the current query $Q$, a standard document-grounded dialogue model computes the conditional distribution $P(y|H, D, Q)$. CPCD computes three distributions when decoding each token: (1) the full conditional distribution $P_{full} = P(y|H, D, Q)$; (2) the document-free distribution $P_{no-doc} = P(y|H, Q)$; and (3) the document-only distribution $P_{doc-only} = P(y|D, Q)$. The final decoding distribution is a weighted combination of these three, aiming to amplify the information gain provided by the document while preserving dialogue context consistency.

### Key Designs

1. **Three-way Contrastive Decoding Formula**:

    - **Function**: Simultaneously considers document faithfulness and dialogue consistency during decoding.
    - **Mechanism**: The final token score is calculated as:
      $$s(y) = \log P_{full}(y) + \alpha \cdot [\log P_{full}(y) - \log P_{no-doc}(y)] - \beta \cdot [\log P_{full}(y) - \log P_{doc-only}(y)]$$
      The first term maintains generation quality. The second term (weighted by $\alpha$) amplifies the information provided by the document—boosting tokens with high probability given the document but low probability without it, thus forcing the model to rely on the document. The third term (weighted by $\beta$) maintains dialogue consistency—tokens with high probability under the full condition but low probability given only the document reflect dialogue context contribution and should not be suppressed by contrastive signals.
    - **Design Motivation**: Standard contrastive decoding $\log P_{full} - \log P_{no-doc}$ simultaneously amplifies document contributions and suppresses dialogue contributions. Introducing the third term precisely recovers the dialogue signals that were mistakenly penalized.

2. **Adaptive Weight Adjustment**:

    - **Function**: Dynamically adjusts the weights of faithfulness and consistency based on the decoding step context.
    - **Mechanism**: During the decoding process, certain positions require higher faithfulness (e.g., when answering factual questions), while others require higher conversational flow (e.g., transition phrases, polite expressions). By calculating the KL divergence between $P_{full}$ and $P_{no-doc}$, the "document dependency" of the current position is determined. A large KL divergence indicates a strong document influence on the current token, prompting an increase in $\alpha$. A small KL divergence implies the position mostly depends on the dialogue context, prompting a decrease in $\alpha$ and an increase in $\beta$. This adaptive adjustment avoids excessively enforcing faithfulness at positions that do not require document support.
    - **Design Motivation**: Fixed weights are suboptimal across all positions; adaptive adjustment achieves a token-by-token optimal balance between faithfulness and fluency.

3. **Consistency-Aware Candidate Filtering**:

    - **Function**: Pre-filters inconsistent candidate tokens before contrastive decoding.
    - **Mechanism**: Set a consistency threshold $\tau$ to directly exclude candidate tokens with $P_{full}(y) < \tau$ (i.e., tokens with very low probability in the original model) from contrastive scoring. This prevents contrastive decoding from anomalously boosting low-probability tokens—where a token might get an exaggerated contrastive gain simply because its probability in $P_{no-doc}$ is extremely low, despite not being a reasonable candidate. The threshold $\tau$ is dynamically set according to the top-$k$ probability of $P_{full}$.
    - **Design Motivation**: A known issue with contrastive decoding is the "denominator effect"—when a token's probability in the contrastive distribution is close to zero, the ratio can grow infinitely, leading to unreasonable choices.

### Loss & Training
CPCD is a pure inference-time method without additional training. It can be applied in a plug-and-play manner to any pre-trained document-grounded dialogue model. The only parameters that need tuning are the three hyperparameters $\alpha$, $\beta$, and $\tau$, which are determined by comprehensive metrics of faithfulness and dialogue quality on the validation set.

## Key Experimental Results

### Main Results

| Method | Faithfulness↑ | BLEU | BERTScore | Dialogue Consistency↑ | Overall Score |
|------|-------------|------|-----------|-----------|---------|
| Standard Decoding | 62.3 | 18.5 | 0.872 | 78.5 | 67.1 |
| Standard Contrastive Decoding | 71.8 | 16.2 | 0.865 | 69.3 | 68.2 |
| NLI-reranking | 68.5 | 17.8 | 0.870 | 76.2 | 69.8 |
| Ours (CPCD) | **74.6** | 17.9 | 0.875 | **77.8** | **73.5** |
| CPCD + Adaptive | **76.2** | **18.1** | **0.878** | **78.1** | **74.8** |

### Ablation Study

| Configuration | Faithfulness | Dialogue Consistency | Description |
|------|--------|----------|------|
| Full CPCD | 76.2 | 78.1 | Full method |
| w/o Third Term (degrades to standard CD) | 71.8 | 69.3 | Dialogue consistency drops significantly |
| w/o Adaptive weights | 74.6 | 77.8 | Fixed weights are slightly worse |
| w/o Candidate filtering | 73.8 | 74.2 | Occasionally generates unreasonable tokens |
| Increase $\alpha$ (stronger faithfulness) | 78.1 | 72.5 | Excessive faithfulness harms dialogue quality |

### Key Findings
- The consistency-preserving term (the third term) is the most critical contribution; removing it causes dialogue consistency to drop by 8.8 points, equivalent to degrading to standard contrastive decoding.
- Adaptive weight adjustment achieves the highest faithfulness on factual questions while automatically relaxing faithfulness constraints on chit-chat transition sentences, realizing the philosophy of being "strict when necessary, flexible when appropriate."
- CPCD significantly outperforms standard contrastive decoding across all evaluation dimensions, proving the advantage of three-way contrast over two-way contrast.
- The method is effective across different base models (T5, Llama-2-7B/13B), demonstrating good generalizability.

## Highlights & Insights
- The idea of three-way contrast ingeniously addresses the fundamental problem that standard contrastive decoding cannot distinguish between document contributions and dialogue contributions, being mathematically compact and intuitively clear.
- As a pure inference-time method, it can be applied in a plug-and-play manner without retraining. The deployment cost is extremely low, which significantly enhances its practical application value.
- The design of adaptive weights demonstrates the feasibility of dynamically regulating the balance among multiple objectives during the decoding process.

## Limitations & Future Work
- Generating three inference distributions requires three forward passes, increasing the inference latency to approximately three times the original.
- The optimal values of hyperparameters $\alpha$ and $\beta$ may vary across datasets and models, requiring tuning on the validation set.
- For scenarios requiring the synthesis of multiple documents, the definitions of the document-free and document-only distributions need to be extended.
- The method has not been fully evaluated in very long document or multi-turn dialogue settings.

## Related Work & Insights
- **vs Contrastive Decoding (CD)**: Standard CD uses only two distributions (expert/amateur). This work introduces a third distribution to achieve decoupling of faithfulness and consistency.
- **vs Knowledge-Grounded Dialogue**: Traditional knowledge-grounded dialogue primarily introduces knowledge at the encoder side. In contrast, this work imposes constraints at the decoder side, making the two approaches orthogonal and combinable.
- **vs FLAN/T5 NLI-based methods**: NLI reranking requires generating multiple candidates and then selecting among them, whereas this work directly optimizes during token-by-token decoding, which is more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-way contrastive decoding idea is novel, and its extension of standard CD possesses theoretical significance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation, comprehensive ablation study.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the method, intuitive derivations of the formulas.
- Value: ⭐⭐⭐⭐ High practical value for both the document-grounded dialogue and contrastive decoding communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)
- [\[ACL 2025\] Entailment-Preserving First-order Logic Representations in Natural Language Entailment](entailment-preserving_first-order_logic_representations_in_natural_language_enta.md)
- [\[ACL 2025\] SAM Decoding: Speculative Decoding via Suffix Automaton](sam_decoding_speculative_decoding_via_suffix_automaton.md)
- [\[ACL 2025\] Consultant Decoding: Yet Another Synergistic Mechanism](consultant_decoding_yet_another_synergistic_mechanism.md)
- [\[ACL 2025\] CLaSp: In-Context Layer Skip for Self-Speculative Decoding](clasp_self_speculative_decoding.md)

</div>

<!-- RELATED:END -->
