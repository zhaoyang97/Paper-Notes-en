---
title: >-
  [Paper Note] In-Context Watermarks for Large Language Models
description: >-
  [ICLR 2026][LLM Safety][In-Context Watermark] This paper proposes In-Context Watermark (ICW), which enables any black-box LLM to embed detectable invisible watermarks into outputs **solely through meticulously designed prompts**, without requiring access to the decoding process. Its practical utility is demonstrated in the typical scenario of "detecting AI-generat
tags:
  - ICLR 2026
  - LLM Safety
  - In-Context Watermark
date: 2026-05-08
content_hash: d48b8ddc90c0d5c0
---
# In-Context Watermarks for Large Language Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=fD9YRHazW3](https://openreview.net/forum?id=fD9YRHazW3)  
**Code**: [https://github.com/yepengliu/In-Context-Watermarks](https://github.com/yepengliu/In-Context-Watermarks)  
**Area**: LLM Security / Text Watermarking / Content Provenance  
**Keywords**: In-Context Watermark, Black-box Watermarking, Prompt Engineering, Indirect Prompt Injection, AI Generation Detection  

## TL;DR
This paper proposes In-Context Watermark (ICW), which enables any black-box LLM to embed detectable invisible watermarks into outputs **solely through meticulously designed prompts**, without requiring access to the decoding process. Its practical utility is demonstrated in the typical scenario of "detecting AI-generated reviews in academic peer review."

## Background & Motivation
**Background**: LLM watermarking is a mainstream method for "source attribution" of AI-generated text. However, the vast majority of methods (such as Kirchenbauer’s green/red token lists, Aaronson’s Gumbel-Max pseudo-random sampling, etc.) grant the **model owner** control over embedding and detection, necessitating intervention in the next-token distribution or sampling process during the decoding stage.

**Limitations of Prior Work**: This "in-process" paradigm requires a rigid prerequisite—access to the model's decoding process. In reality, many scenarios involve detection parties who cannot access the model. The flagship example presented is academic conference organizers aiming to identify violations where "lazy reviewers feed papers to LLMs to generate reviews." Organizers know neither which model was used nor can they intervene in its decoding. Meanwhile, post-hoc detection tools like DetectGPT or GPTZero suffer from low accuracy and high false-positive rates, and mainstream commercial LLMs have not publicly deployed watermarks.

**Key Challenge**: Watermarking currently requires either the cooperation of the model owner (in-process) or post-hoc rewriting of generated text, leaving a vast middle ground uncovered where the "detector has no model access but wishes to proactively embed attribution signals."

**Goal**: To explore a novel problem—can watermarks be embedded **strictly through prompt engineering** without any privileged access to the model?

**Core Idea (Black-box Prompt as Watermark)**: Leverage the robust **In-Context Learning** and **Instruction Following** capabilities of modern LLMs to write watermarking logic as a natural language "watermarking instruction" within the prompt (system prompt or document body), causing the model to automatically carry an invisible watermark in all subsequent responses. Furthermore, via **Indirect Prompt Injection (IPI)**, watermarking instructions can be hidden in paper PDFs using "white font/zero font size." Once a reviewer feeds the full text to an LLM to generate a review, the output will contain a detectable watermark.

## Method

### Overall Architecture
ICW shifts watermarking from "modifying decoding" to "modifying prompts." The detector and generator share a key $k$ and a watermarking scheme $\tau$, with $\text{Instruction}(k,\tau)$ serving as a prompt prefix. Given a normal query $Q$, the watermarked response is $y \leftarrow M(\text{Instruction}(k,\tau) \oplus Q)$, where $\oplus$ denotes concatenation and $M$ is any black-box LLM. The detector $D(\cdot|k,\tau)$ formalizes the determination of whether $y$ contains a watermark as a hypothesis test: a watermark is detected (rejecting the null hypothesis) when $D(y|k,\tau) \ge \eta$. The paper distinguishes between two deployment settings: **DTS (Direct Text Stamp)**, which places the instruction in the system prompt for full-session watermarking, and **IPI (Indirect Prompt Injection)**, which covertly embeds the instruction in long documents to be triggered when a user feeds the document to an LLM, corresponding to the peer-review anti-cheating scenario.

```mermaid
flowchart LR
    A[Watermark Instruction Instruction k,τ] --> C[Black-box LLM M]
    B[Normal Query Q / Document] --> C
    C --> D[Watermarked Response y]
    D --> E[Detector D·|k,τ]
    E --> F{D ≥ η ?}
    F -->|Yes| G[Watermarked]
    F -->|No| H[No Watermark]
```

Designed around "different granularities of natural language," the paper proposes four ICW strategies covering character, word, vocabulary, and sentence levels, with customized detectors for each.

### Key Designs
**1. Unicode ICW: Character-level invisible stamp with the lowest capability threshold.** This is the simplest strategy: the instruction directs the model to insert a zero-width space $\text{U+}200\text{B}$ after every word. The output takes the form $\{y^{(1)},\text{U+}200\text{B},\dots,y^{(n)},\text{U+}200\text{B}\}$. The detector simply counts invisible character density $D(y|k_u,\tau_u)=|y|_{k_u}/N$. It requires minimal instruction-following capability, is completely invisible to humans, and is nearly perfectly robust against copy-pasting and minor edits. However, it only exists in digital text (invalid after printing/scanning) and can be entirely removed by LLM rewriting, representing the "easy to deploy but fragile" extreme.

**2. Initials ICW: Initial letter bias + z-statistic detection.** A key $k_c$ is used to randomly select a set of "green letters" $A_G$. The instruction directs the model to **prefer words starting with green letters**. Detection compares the ratio of green initials in the text to a human baseline using a z-statistic $D(y|k_c,\tau_c)=(|y|_G-\gamma|y|)/\sqrt{\gamma(1-\gamma)|y|}$, where $|y|_G$ is the count of green-initial words and $\gamma$ is the expected proportion of green-initial words in human text (estimated using the Canterbury Corpus $P_A$ as $\gamma=\sum_i P_A(a^{(i)}\in A_G)$). It requires higher instruction-following but demonstrates good detection and robustness in strong models; the trade-off is a statistical bias that an adversary could potentially use to reverse-engineer $A_G$, creating a risk of spoofing.

**3. Lexical ICW: Green/Red vocabulary alignment with classic list-based ideas.** Inspired by Kirchenbauer’s green/red token lists but using **full words** instead of tokens: key $k_L$ partitions a vocabulary $V$ into a green list $V_G$ (proportion $\gamma|V|$) and a red list $V_R$. $V$ is restricted to adjectives/adverbs/verbs—categories that carry stylistic features and are topic-agnostic—to reduce scale. The instruction directs the model to use green words preferentially. Detection follows the Initials z-statistic framework, replacing $|y|_G$ with green word hits and $\gamma=|V_G|/|V|$. This places the highest demand on the model's **long-context retrieval capability**, as the model must remember and select words from a long green list, which is a significant challenge for current models.

**4. Acrostics ICW: Hidden ciphertext in sentence starts, most stable under discrete editing.** A sentence-level strategy: key $k_s$ samples a watermark sequence $\zeta=\{\zeta^{(1)},\dots,\zeta^{(m)}\}$. The instruction directs the first letter of each sentence to align with $\zeta$ (i.e., the generated sentence-initial sequence $\ell$ satisfies $\ell^{(i)}=\zeta^{(i)}$), encoding secret information into sentence starts (like an acrostic poem "MHAB" in Figure 1). Detection uses the Levenshtein distance to measure the proximity between $\ell$ and $\zeta$, normalized to a z-statistic $D(y|k_s,\tau_s)=(\mu-d(\ell,\zeta))/\sigma$, where $\mu,\sigma$ are estimated by resampling $N$ sentence-initial sequences from the suspect text. Since it only constrains the first character of sentences, it is highly redundant and robust against deletions, insertions, and rewrites, provided most of the sentence-initial sequence remains. Using a fixed $\zeta$ may expose the pattern, so sampling a long $\zeta$ and using a short sub-sequence for each generation is recommended to enhance stealth.

## Key Experimental Results
**Settings**: Evaluation used two black-box commercial models, gpt-4o-mini and the more capable gpt-o3-mini. DTS utilized the long-form QA dataset ELI5 (original answers as human text), while IPI used full 2020–2023 ICLR papers as input. Each evaluation included 500 watermarked texts + 500 human texts, averaging 300 words each. Baselines included two black-box post-processing methods, PostMark and YCZ+23, and the post-hoc detector GPTZero. Metrics included ROC-AUC, T@1%F / T@10%F (TPR at low FPR), robustness (30% deletion, 30% substitution, rewriting), and text quality (perplexity + LLM-as-a-Judge).

### Main Results (Detection Performance ROC-AUC, excerpt T@1%F)

| Model | Method | DTS AUC ↑ | DTS T@1%F ↑ | IPI AUC ↑ | IPI T@1%F ↑ |
|------|------|-----------|-------------|-----------|-------------|
| GPT-4o-mini | Unicode ICW | 1.000 | 1.000 | 0.857 | 0.714 |
| GPT-4o-mini | Initials ICW | 0.572 | 0.006 | 0.620 | 0.006 |
| GPT-4o-mini | Lexical ICW | 0.910 | 0.320 | 0.889 | 0.054 |
| GPT-4o-mini | Acrostics ICW | 0.590 | 0.036 | 0.592 | 0.002 |
| GPT-o3-mini | Unicode ICW | 1.000 | 1.000 | 1.000 | 1.000 |
| GPT-o3-mini | Initials ICW | 0.999 | 0.990 | 0.997 | 0.910 |
| GPT-o3-mini | Lexical ICW | 0.995 | 0.930 | 0.997 | 0.974 |
| GPT-o3-mini | Acrostics ICW | 1.000 | 1.000 | 0.997 | 0.982 |
| — | PostMark (DTS) | 0.977 | 0.802 | — | — |
| — | YCZ+23 (DTS) | 0.998 | 0.992 | — | — |

The core conclusion is evident: the weaker model, GPT-4o-mini, only succeeds with Unicode (the lowest instruction threshold). When switching to GPT-o3-mini, all four ICW strategies approach perfect scores. Furthermore, post-processing methods like PostMark/YCZ+23 **cannot be used for IPI** (as reviewers have no incentive to watermark themselves), highlighting the unique advantage of ICW in black-box scenarios without access rights.

### Ablation Study (DTS, gpt-o3-mini, AUC)

| Attack | Initials | Lexical | Acrostics | YCZ+23 | PostMark |
|------|----------|---------|-----------|--------|----------|
| 30% Deletion | 0.999 | 0.857 | 0.881 | 0.980 | 0.908 |
| 30% Substitution | 0.999 | 0.758 | 1.000 | 0.982 | 0.956 |
| Rewriting | 0.887 | 0.924 | 0.922 | 0.557 | 0.841 |

Regarding text quality (LLM-as-a-Judge Overall), Unicode scored 4.810, Lexical 4.808, and Acrostics 4.813—all near or above the human score of 4.235 and significantly better than PostMark’s 2.997.

### Key Findings
- **ICW efficacy correlates strongly with model capability**: The usability of the four strategies grows in tandem with the model’s In-Context Learning, instruction-following, and long-context retrieval abilities. The authors infer that "the stronger the model, the better ICW performs."
- **ICW is more stable than baselines under rewriting attacks**: YCZ+23 drops to 0.557 under rewriting, while Initials/Lexical/Acrostics maintain 0.88–0.92.
- **Lexical is weaker under substitution attacks** (0.758) because the green words it relies on (nouns/verbs/adjectives/adverbs) are the primary targets for synonymous replacement.
- **IPI is feasible in long contexts**: Strong models can reliably follow watermarking instructions even when they are hidden deep within long documents.

## Highlights & Insights
- **Paradigm Shift**: Moves watermarking control from "model owners modifying decoding" to "any third party modifying prompts," enabling proactive attribution in black-box, no-access scenarios for the first time.
- **Threat Model Inversion**: While IPI is typically used by attackers to inject malicious instructions into documents, this paper reverses it as a defense mechanism—the benevolent organizer embeds the instruction, and the potentially violating user becomes the trigger.
- **Coverage of the trade-off spectrum**: Four strategies range from "low threshold, fragile" (Unicode) to "high threshold, robust" (Acrostics). Table 1 provides a clear engineering guide comparing LLM requirements, detectability, robustness, and quality.
- **Optimistic Deduction**: Converts a methodology limitation (not working for weak models) into a positive trend prediction (model upgrades = free watermark improvements).

## Limitations & Future Work
- **Strong dependence on model capability**: Except for Unicode, the strategies largely fail on models of gpt-4o-mini’s tier, lacking generalizability to small or medium models.
- **Shallow Attack/Defense Analysis**: The paper admits that attack/defense under IPI (e.g., detecting and removing instructions or using "ignore prior prompts") is only preliminarily explored, leaving systematic analysis for future work.
- **Inherent Vulnerabilities**: Unicode is destroyed by rewriting/printing; Initials' statistical bias is prone to reverse-engineering (spoofing); Lexical suffers in substitution attacks; fixed-sequence Acrostics are detectable.
- **Ethical Boundaries**: IPI essentially instrumentalizes "prompt injection attacks" for regulation. Hiding instructions in submitted papers raises questions of informed consent and fairness (the paper suggests organizers, not authors, should embed the instructions to avoid conflict of interest).

## Related Work & Insights
This work represents a third path for LLM watermarking. Traditional methods are either **post-hoc** (modifying format/vocabulary/syntax or using LLM rewriting, e.g., YCZ+23, PostMark) or **in-process** (decoding-phase logits perturbation or pseudo-random sampling, e.g., Kirchenbauer, Aaronson, Bahri’s black-box n-gram scoring). ICW belongs to neither—it modifies neither the generated text nor the decoding, relying entirely on the prompt. Technically, it creatively repurposes research from **Prompt Injection Attacks** (such as zero-font/transparent text obfuscation) for defense. The insight is that as instruction-following becomes a core LLM capability, "using natural language instructions to achieve tasks that previously required low-level access" is a trend that will only grow stronger; watermarking is just one example. It also serves as a reminder to the community that prompt injection is a double-edged sword usable by both attackers and regulators.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Shifting watermarks from decoding to prompts creates a new paradigm for black-box attribution and clever inversion of the IPI threat model.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 2 models × 2 settings × 4 strategies with comprehensive detection/robustness/quality metrics. Strong comparison against black-box/post-hoc baselines. Deducted for shallow attack/defense analysis and lack of cross-validation on open-source/smaller models.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The peer-review motivation is vivid, the four strategies are clearly organized by granularity, and the trade-off overview in Table 1 is very helpful. 
- **Value**: ⭐⭐⭐⭐ Addresses the real need for attribution without model access, particularly for academic integrity and content platforms. Value increases with model upgrades, though current lack of generalizability to weak models limits immediate deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)
- [\[ICLR 2026\] Do LLMs Forget What They Should? Evaluating In-Context Forgetting in Large Language Models](do_llms_forget_what_they_should_evaluating_in-context_forgetting_in_large_langua.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](../../ACL2025/llm_safety/ensemble_watermarks_llm.md)
- [\[ICLR 2026\] LLM Fingerprinting via Semantically Conditioned Watermarks](llm_fingerprinting_via_semantically_conditioned_watermarks.md)
- [\[ICLR 2026\] PRISON: Unmasking the Criminal Potential of Large Language Models](prison_unmasking_the_criminal_potential_of_large_language_models.md)

</div>

<!-- RELATED:END -->
