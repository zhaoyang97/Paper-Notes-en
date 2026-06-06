---
title: >-
  [Paper Note] From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages
description: >-
  [ACL 2026][Multilingual & Machine Translation][Medieval Romance] The authors perform a systematic comparison across three Medieval Romance languages (Old Occitan NAF, Old Catalan CAT, Old French Chauliac) for POS tagging…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Medieval Romance"
  - "POS Tagging"
  - "LLM Fine-tuning"
  - "Cross-lingual Transfer"
  - "Digital Humanities"
date: 2026-05-08
content_hash: ccf689cfaa5c790b
---

# From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages

**Conference**: ACL 2026  
**arXiv**: [2605.09147](https://arxiv.org/abs/2605.09147)  
**Code**: <https://github.com/msch38/medieval-romance-pos>  
**Area**: Multilingual / Historical NLP / POS Tagging  
**Keywords**: Medieval Romance, POS Tagging, LLM Fine-tuning, Cross-lingual Transfer, Digital Humanities

## TL;DR
The authors perform a systematic comparison across three Medieval Romance languages (Old Occitan NAF, Old Catalan CAT, Old French Chauliac) for POS tagging, evaluating traditional taggers (UDPipe / COLaF) against open-source LLMs (Gemma3-12B / Phi4-14B) under five settings: zero-shot, few-shot, monolingual fine-tuning, bilingual CLTF, and trilingual CLTF. LLMs consistently outperform traditional methods; specifically, Catalan serves as a "bridge language," where CAT+FR bilingual training elevates Old French Chauliac to a peak accuracy of 93.14%.

## Background & Motivation
**Background**: In digital humanities, large-scale linguistic analysis of medieval documents typically begins with POS tagging, which serves as the foundation for subsequent syntactic parsing, semantic analysis, and diachronic change modeling. Traditionally, the Romance family's medieval stages (Old Occitan/Old Catalan/Old French) have relied on tools like UDPipe and COLaF trained on universal dependency treebanks, or pure rule-based systems.

**Limitations of Prior Work**: Medieval Romance languages possess three characteristics that severely degrade modern taggers: ① Extreme orthographic instability (the same word appearing as `deceplina/disiplina/desiplina` within a single text); ② Complex and dialectal morphological systems; ③ Scarce annotated corpora, with the smallest Chauliac dataset containing only 2,443 tokens. Consequently, UDPipe drops to 68% on NAF, and COLaF falls to 52% on CAT.

**Key Challenge**: A massive gap exists between the training distribution of modern NLP tools and the actual distribution of medieval texts, yet there is insufficient medieval supervised signal to train specialized models from scratch. Whether LLMs can "borrow" modern multilingual knowledge to bridge this gap remains unclear, as systematic evaluations of LLMs for Medieval Romance POS tagging were previously non-existent.

**Goal**: To answer three independent questions: (Q1) How much stronger are LLMs compared to traditional taggers? (Q2) How much do prompt formats and decoding hyperparameters matter? (Q3) To what extent can cross-lingual transfer rescue low-resource medieval languages?

**Key Insight**: The authors observe that Catalan is phylogenetically positioned between Gallo-Romance and Occitano-Romance, potentially acting as a "bridge." Thus, they specifically designed three bilingual configurations (CAT+OCC, CAT+FR, FR+OCC) and one trilingual configuration to disentangle the effects of phylogenetic relatedness and corpus size.

**Core Idea**: Running five experiment families (traditional, prompting, monolingual FT, bilingual CLTF, trilingual CLTF) across fixed 80/20 splits with four decoding strategies to provide a practical roadmap for method selection based on resources and target language.

## Method

### Overall Architecture
The experimental setup constitutes a 5×N grid: 5 method families (Traditional baseline / LLM Prompting / Monolingual FT / Bilingual CLTF / Trilingual CLTF) × 3 datasets (NAF 45,457 tokens, CAT 59,359 tokens, Chauliac 2,443 tokens) × 2 LLM backbones (Gemma3-12B, Phi4-14B). All LLM fine-tuning utilizes the same LoRA configuration ($r=16, \alpha=32$, targeting q/k/v/o_proj modules, learning rate $2\times 10^{-4}$, 10 epochs, AdamW). All decoding is scanned under the same prompt template. The tagset is unified to 17 UD categories, the input is a token stream, and the output is JSON `[{"word":..., "UPOS":...}, ...]`.

### Key Designs

1.  **Unified 80/20 Split as a Foundation for Fair Comparison**:
    - **Function**: Ensures monolingual, bilingual, and trilingual training configurations are directly comparable.
    - **Mechanism**: A fixed 20% test partition is set for each dataset. Bilingual training uses the union of two 80% partitions; trilingual training uses the union of all three 80% partitions; testing is always performed on the fixed 20% test partition. This ensures that "test tokens never appeared in training" under any configuration.
    - **Design Motivation**: Previous cross-lingual transfer literature often suffered from inconsistent splitting—e.g., bilingual training inadvertently consuming 100% of the target language data—leading to "transfer gains" that were actually just "more data." This split explicitly separates "transfer signal" from "data volume," making $\Delta_{\text{bilingual-mono}}$ interpretable.

2.  **Phylogeny-Driven Bilingual Pairing Design**:
    - **Function**: Tests whether linguistic relatedness accurately predicts transfer performance.
    - **Mechanism**: Rather than brute-forcing all LLM × language pairs, the authors selected three sets based on linguistic priors: CAT+OCC (both Occitano-Romance), CAT+FR (cross-Gallo/Occitano but with CAT as intermediary), and FR+OCC (cross-branch bypassing CAT). Results are reported for each pair on both involved languages.
    - **Design Motivation**: If phylogenetic relatedness is the dominant factor, FR+OCC should perform similarly to CAT+OCC. If CAT is truly a "bridge," any configuration including CAT should win. Results showed CAT+OCC→NAF at 89.25% and CAT+FR→Chauliac at 93.14%, while FR+OCC→NAF was only 80.31% (+0.22 relative to monolingual), strongly supporting the Catalan bridge hypothesis.

3.  **Mixed Trilingual Few-shot Examples + Full Decoding Sweep**:
    - **Function**: Decouples prompt engineering and decoding hyperparameters from the core conclusions.
    - **Mechanism**: Few-shot prompts intentionally mix example tokens from all three languages in the same block (e.g., `bo/ADJ`, `volch/VERB`, `seyor/NOUN`, `addicions/NOUN`) to explicitly expose the model to orthographic diversity. On the decoding side, beam search ($w\in\{1,15\}$), temperature ($\tau\in\{0.6,0.8,0.9\}$), top-$k$ ($\{5,20,50\}$), and top-$p$ ($\{0.75,0.85,0.95\}$) were all evaluated.
    - **Design Motivation**: Since tagging is a discriminative task, deterministic decoding was expected to win. Beam-15 indeed achieved the highest scores across all model × dataset combinations (Phi4 few-shot + Beam-15: 81.23% average), and the standard deviation across sampling strategies was only 0.1-0.2. This confirms that the "LLM > Traditional" conclusion is not an artifact of a specific "magic" sampling temperature.

### Loss & Training
The fine-tuning phase uses standard next-token cross-entropy (supervising the UPOS field in the JSON output). LoRA only updates the four attention projection matrices with a dropout of 0.1. With a batch size of 4 and 10 epochs, the smallest dataset, Chauliac (2,443 tokens), quickly enters the overfitting zone, explaining why its monolingual FT gain over prompting was limited (Gemma3 mono 83.64% vs. Phi4 few-shot 84.98%). During prompting, weights are not updated; hyperparameters are only scanned at decoding.

## Key Experimental Results

### Main Results

| Method Family | Model / Strategy | NAF Acc. | CAT Acc. | Chauliac Acc. |
| :--- | :--- | :--- | :--- | :--- |
| Traditional | UDPipe | 68.01 | 81.59 | 89.40 |
| Traditional | COLaF | 65.73 | 52.15 | 72.50 |
| Prompting | Phi4 Few-shot | 75.01 | 83.69 | 84.98 |
| Fine-tune | Gemma3 Monolingual | 80.09 | **92.52** | 83.64 |
| Bilingual CLTF | Gemma3 CAT+OCC | 89.25 | 91.62 | – |
| Bilingual CLTF | Gemma3 CAT+FR | – | 91.28 | **93.14** |
| Bilingual CLTF | Gemma3 FR+OCC | 80.31 | – | 85.74 |
| Trilingual CLTF | Gemma3 | **89.68** | 89.16 | 88.23 |
| **Gain** (Best vs UDPipe) | — | **+21.67** | **+10.93** | **+3.74** |

Key Observations: ① Average accuracy rises monotonically from Traditional 71.56% → Prompting 77.75% → Monolingual FT 85.19% → Optimal CLTF; ② NAF benefits the most (+21.67 pp), while Chauliac benefits the least (+3.74 pp) but still reaches 93.14%.

### Ablation Study

| Configuration | NAF | CAT | Chauliac | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| Monolingual FT (Gemma3) | 80.09 | 92.52 | 83.64 | Baseline |
| Bilingual CAT+OCC | 89.25 | 91.62 | – | NAF +9.16, CAT slightly down 0.90 |
| Bilingual CAT+FR | – | 91.28 | 93.14 | Chauliac +9.50, CAT slightly down 1.24 |
| Bilingual FR+OCC | 80.31 | – | 85.74 | NAF +0.22, Chauliac +2.10 (Negligible gain without CAT) |
| Trilingual CLTF | 89.68 | 89.16 | 88.23 | Best for NAF; but Chauliac 4.91 pp lower than CAT+FR |

**Phi4 Prompting Decoding Robustness** (Few-shot avg. accuracy across decoding families):

| Decoding Family | Avg. Acc. | Cross-dataset std | Recommendation |
| :--- | :--- | :--- | :--- |
| Beam-15 | 81.23 | 0.12 | Most stable and highest |
| Top-$k$ | 80.28 | 0.20 | Alternative |
| Top-$p$ | 80.69 | 0.18 | Balanced |
| Temperature | 80.57 | 0.21 | Slightly higher variance |

### Key Findings
- **Catalan is the True Bridge**: Any bilingual pairing containing CAT wins. FR+OCC without CAT gained only +0.22 pp on NAF, proving that phylogenetic relatedness (FR and OCC both being Romance) is insufficient for transfer; it must be paired with an appropriate "intermediary."
- **Trilingual is Not Always Best**: Chauliac performed at 88.23% in the trilingual setting, which was 4.91 pp lower than the CAT+FR bilingual setting (93.14%). The authors attribute this to Chauliac's small size (2,443 tokens), where trilingual training diluted the signal. "Focused exposure to closely related material" remains the golden rule for small datasets.
- **Uneven Gains at POS Level**: On NAF, PROPN (+66.62), NUM (+62.97), and SCONJ (+41.68) benefited most. Functional words like ADP and CCONJ were already near the ceiling (85-95% F1), leaving little room for improvement. This indicates LLM contextual representations primarily rescue "semantically heavy" open-class words.
- **Minimal Decoding Variance**: Phi4 few-shot maintained an average accuracy of 80.20-81.23% across all decoding configurations, with a cross-config std below 0.21. This eliminates concerns that experimental results were artifacts of specific decoding hyperparameters.

## Highlights & Insights
- **First Complete Map of "Medieval Romance × LLM × Cross-lingual Transfer"**: Prior work either compared prompting only or monolingual FT only. This paper fills the 5-family grid under a unified split, providing a reusable method selection table (Table 13).
- **Transferability of the Bridge Language Discovery**: This phenomenon likely exists in other language families (e.g., Middle English + Middle Dutch + Old German in Germanic, or Old Chinese + Middle Chinese + Medieval Japanese loanwords). The bilingual CLTF experimental paradigm can be applied directly.
- **Mixed Multilingual Few-shot Prompting**: Mixing orthographic variants from multiple languages in a single example block informs the model during in-context learning not to expect orthographic consistency—a trick worth borrowing for any NLP scenario involving high dialectal or spelling variance.
- **Avoid Excessive Multilingualism for Small Datasets**: For target domains with < 5k tokens, trilingual training can dilute signals. This counter-intuitive finding (relative to the "more data is better" trend in LLMs) offers a clear cautionary tale for practitioners in digital humanities.

## Limitations & Future Work
- The sizes of the three datasets differ by an order of magnitude (2,443 vs. 59,359), and they belong to different genres (literary, chronicles, medical). The authors admit to not conducting controlled subsample/learning-curve experiments, meaning the factors of "corpus size," "genre," and "phylogeny" are not strictly isolated in CLTF gains.
- Smaller LLMs (1–3B) or specialized token-classification encoders (XLM-R / mBERT) were not evaluated. The strongest results depend on 12–14B models + LoRA, which incur significantly higher deployment costs than UDPipe; cost-effectiveness for real humanities projects requires further validation.
- The few-shot examples used fixed trilingual mixing; however, designing per-language prompts or using target-specific example selection was not explored, leaving significant room for prompt optimization.
- No "pure" zero-shot cross-lingual evaluation was conducted (e.g., CAT+FR → NAF, where the target language is entirely absent from training). This remains the cleanest test for transfer capability and is left for future work.
- The study has not yet been extended to downstream tasks like lemmatization, parsing, or NER, so it is unclear if POS improvements propagate through a full historical NLP pipeline.

## Related Work & Insights
- **vs. Schöffel et al. 2025b (Modern Models, Medieval Texts)**: That work focused only on Old Occitan prompting + FT; this paper expands to three languages and adds bilingual/trilingual CLTF, providing a more systematic scale. While that work proved "models work," this paper answers "how to use them."
- **vs. Camps et al. 2021 (Classical French Drama POS)**: They performed monolingual adaptation for Classical French without a cross-lingual dimension. This paper’s achievement of 93.14% on FR via CAT suggests that "finding a phylogenetically intermediary sibling for bilingual FT" may be more efficient than "labeling more target data."
- **vs. Bollmann et al. 2019 / Manjavacas et al. 2019 (Historical Text Normalization)**: Their pipeline involves "normalization followed by modern taggers." This paper bypasses normalization, allowing the LLM to read raw orthography directly and outperforming the older pipeline, suggesting that the necessity of normalization steps is decreasing in the LLM era.
- **vs. Karthikeyan et al. 2020 (mBERT Cross-lingual Ability)**: While they found phylogenetic relatedness to be the primary driver for transfer in modern languages, this paper argues for medieval languages that phylogeny is a necessary but insufficient condition—requiring an appropriate "bridge language" and sufficient target domain data.

## Rating
- Novelty: ⭐⭐⭐⭐ No new model proposed, but the first systematic integration of 5 method families × 3 Medieval Romance languages, discovering the CAT bridge effect.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 families × 3 datasets × 2 models × 12 decoding configs + full per-class F1 breakdown.
- Writing Quality: ⭐⭐⭐⭐ Clear tables, layered conclusions, and honest limitations.
- Value: ⭐⭐⭐⭐ The digital humanities community can directly adopt the method selection table; the empirical evidence for bridge languages offers cross-domain insights for low-resource NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ACL 2026\] Lost in Translation: Do LVLM Judges Generalize Across Languages?](lost_in_translation_do_lvlm_judges_generalize_across_languages.md)
- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[ACL 2026\] What Factors Affect LLMs and RLLMs in Financial Question Answering?](what_factors_affect_llms_and_rllms_in_financial_question_answering.md)

</div>

<!-- RELATED:END -->
