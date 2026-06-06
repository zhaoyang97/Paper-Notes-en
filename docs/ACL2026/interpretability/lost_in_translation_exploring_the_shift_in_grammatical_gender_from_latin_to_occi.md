---
title: >-
  [Paper Note] Lost in Translation? Exploring the Shift in Grammatical Gender from Latin to Occitan
description: >-
  [ACL 2026][Interpretability][Medieval Occitan] For the low-resource historical language of Medieval Occitan, a framework combining mBERT, hybrid tokenization…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Medieval Occitan"
  - "Grammatical Gender"
  - "Latin Neuter Nouns"
  - "Explainable NLP"
  - "Hybrid Tokenization"
date: 2026-05-08
content_hash: 8547d9d680e95940
---

# Lost in Translation? Exploring the Shift in Grammatical Gender from Latin to Occitan

**Conference**: ACL 2026  
**arXiv**: [2605.09156](https://arxiv.org/abs/2605.09156)  
**Code**: https://github.com/ahan-2000/Lost-in-Translation- (Available)  
**Area**: Computational Linguistics / Historical Linguistics / Low-resource NLP  
**Keywords**: Medieval Occitan, Grammatical Gender, Latin Neuter Nouns, Explainable NLP, Hybrid Tokenization

## TL;DR
For the low-resource historical language of Medieval Occitan, a framework combining mBERT, hybrid tokenization, and domain-adapted MLM was developed to quantify whether original Latin neuter nouns became masculine or feminine. By decomposing the problem into morphological cues vs. syntactic context, it was found that suffix morphology provides the strongest single signal, while context (especially articles and adjectives) increases the Macro F1 from 0.665 to 0.929.

## Background & Motivation

**Background**: The evolution of the Romance language family from the three Latin genders (Masculine / Feminine / Neuter) to two genders (Masculine / Feminine) is a classic problem in historical linguistics. However, most computational research focuses on high-resource languages like French and Spanish. Occitan, though classified as "vulnerable" by UNESCO, has very few associated NLP works.

**Limitations of Prior Work**: ① The orthography of Medieval Occitan is highly unstable, with a single lemma often having multiple spellings; standard WordPiece/BPE tokenizers suffer from high OOV rates or fracture meaningful morphological cues. ② Existing gender prediction tasks are either purely rule-based (non-transferable) or only consider isolated word forms (ignoring agreement), failing to quantify the relative contributions of morphology vs. context. ③ There is a lack of systematic, explainable analysis regarding how Latin neuter nouns were assigned gender in Occitan.

**Key Challenge**: Gender information is distributed across two levels: intra-word morphology (suffixes like `-um/-ia/-la`) and sentence-level agreement (articles `lo/la`, adjective endings). Current methods do not separate these two streams of evidence, making it impossible to explain model success or understand how much context helps when word forms are ambiguous (e.g., `psalmista`).

**Goal**: (RQ1) To what extent can word-level morphological features predict Occitan gender? (RQ2) What is the gain provided by sentence context, and which parts of speech contribute to it?

**Key Insight**: Treat "gender assignment" as a quantifiable dual-source problem—intra-word signals vs. context signals—modeling and comparing them separately using ablation, SHAP, and PoS occlusion explainability tools.

**Core Idea**: Utilize mBERT with Hybrid tokenization (BPE trained on the corpus + word-level fallback) and domain-adapted MLM as a unified backbone. Construct word-only, context, and masked-context inputs to compare Macro-F1 and log-prob increments, thereby quantifying the respective contributions of morphology and context.

## Method

### Overall Architecture

The entire pipeline consists of three stages:

1.  **Preparation (Embedding and Tokenizer Selection)**: Three probes (frozen gender prediction, Latin→Occitan variant retrieval, clustering) were run on FastText / mBERT / ByT5 for Latin-Occitan pairs; mBERT outperformed others and was selected as the backbone. Standard mBERT WordPiece, pure BPE, and Hybrid (Corpus BPE + word-level fallback) were compared. Hybrid was the only approach achieving 0% OOV and a masked recovery of 25.23%. Domain-adapted MLM was performed on the Hybrid vocabulary for 10 epochs, reducing PPL from 942.85 to 9.52.
2.  **Word-level Gender Prediction (RQ1)**: Morphological-phonological features including initial substrings, suffix 1-4grams, syllable counts $S(w)$, CV templates $P(w)$, length ratios, and stress position proxies were extracted from Latin/Occitan word pairs. These were optionally concatenated with mean-pooled representations from FastText / mBERT / ByT5 and fed into a suite of 13 classifiers (LR / RF / XGB / FFN / BiLSTM / BiLSTM+Attn / 2×BiLSTM+MHSA) using 10-fold lemma-grouped CV.
3.  **Contextual Gender Prediction (RQ2)**: Noun tokens from approximately 130k tokens of unlabeled Occitan corpora were aligned to a Latin-Occitan lemma dictionary (exact matching prioritized, otherwise fuzzy matching with $\textsc{Sim}=0.3\,\textsc{CosSim}+0.7\,\textsc{LevSim}\ge 0.85$). This generated (sentence, noun position, Latin lemma, Latin gender, gold Occitan gender) quintuplets. The model used the same mBERT + MLP head to evaluate word-only, context (noun-conditioned attention), and masked-context (target word replaced by `[MASK]`) representations.

### Key Designs

1.  **Hybrid Tokenizer (BPE + word-level fallback)**:
    - **Function**: Ensures zero OOV while maintaining meaningful subword segmentation in Medieval Occitan corpora with high orthographic noise.
    - **Mechanism**: A small-vocabulary BPE is first trained on the Occitan corpus (iteratively merging the most frequent pairs via $V_{t+1}=V_t\cup\{ab\},\ (a,b)=\arg\max_{(x,y)} f(x,y)$), followed by a word-level fallback rule where any full word not splittable by BPE is preserved. BPE subwords capture historical variation patterns like the `mp` consonant cluster in `primpcipat` or the word-final `t` in `secretament` (corresponding to the dropping of the Old Occitan adverbial `-t`).
    - **Design Motivation**: Standard mBERT WordPiece achieves 0% OOV but only 15.78% masked recovery; pure BPE (vocab=600/800) introduces 2.63–2.86% OOV with minimal recovery. The Hybrid approach combines the strengths of both.

2.  **2×BiLSTM + MHSA Morphological Head**:
    - **Function**: Takes multi-source features of an isolated lemma (Latin/Occitan n-grams + syntactic-phonological features + mBERT embedding) as input to output binary gender probabilities.
    - **Mechanism**: A dual-layer BiLSTM captures sequential dependencies of subwords, topped with 8-head self-attention to identify gender-sensitive subword positions. The target is trained using label smoothing CE or focal loss + class weights to handle the 2:1 imbalance. This achieved a peak lemma-level Macro-F1 of 0.8224 using mBERT embeddings.
    - **Design Motivation**: Tree models and shallow LSTMs only reach ~0.71–0.78 F1; gender signals exist both in suffixes and specific phoneme combinations, requiring a structure that can model sequences while localizing key positions.

3.  **Noun-conditioned Attention Context Head**:
    - **Function**: Incorporates sentence-level agreement information without destroying the target noun's representation.
    - **Mechanism**: mBERT encodes the sentence to get $H=(h_1,\dots,h_T)$. The hidden state $h_i$ at the target noun position $i$ serves as the query for multi-head attention over the full sentence $\mathrm{Attn}(h_i, H, H)$, which is then concatenated with the Latin lemma embedding $e(L)$ and the Latin gender one-hot $\mathrm{onehot}(G_L)$ before being fed into a shared MLP $p(y\mid r)=\mathrm{softmax}(f_\phi(r))$. A masked-context variant replaces the noun with `[MASK]` before reading $h_i^{\text{mask}}$ to measure how much gender info can be recovered from the context alone.
    - **Design Motivation**: In Occitan, the article `la` in `la torista` is the key to disambiguation; naive `[CLS]` pooling dilutes this signal, whereas noun-conditioned attention allows the model to explicitly localize the decision-making process.

### Loss & Training
- Word-level experiments utilized lemma-grouped 10-fold CV to prevent variant leakage, with Optuna for Bayesian hyperparameter optimization. The best head was 2×BiLSTM+MHSA with CE + label smoothing 0.1, trained for 100 epochs; imbalance was handled via focal loss + class weights.
- Contextual experiments utilized group K-fold (3-fold, grouped by lemma), AdamW, warmup 0.06 + linear decay, grad clip 0.5, and dropout 0.1.

## Key Experimental Results

### Main Results

| Setting (mBERT) | Accuracy | Macro F1 |
|---|---|---|
| Word-only | $0.808 \pm 0.154$ | $0.665 \pm 0.108$ |
| Context model (noun attention) | $0.979 \pm 0.012$ | $0.929 \pm 0.034$ |
| Context model (noun masked) | $0.977 \pm 0.008$ | $0.902 \pm 0.097$ |

The addition of context increased Macro-F1 from 0.665 to 0.929 (+26.4 gain). The masked-context model still reached 0.902, indicating that agreement signals in the context alone can recover the majority of gender information. The best lemma-level performance was mBERT + 2×BiLSTM+MHSA, with Macro-F1 = $0.8224 \pm 0.0385$; paired bootstrap showed a significant advantage over ByT5 ($\Delta = +0.0395$, 95% CI $[+0.0250, +0.0543]$, $p<10^{-6}$).

### Ablation Study

| Removed Feature Block | F1 (mBERT) | $\Delta$ | % drop |
|---|---|---|---|
| Latin n-grams | 0.8092 | 0.0132 | 1.61% |
| Meta-features | 0.8168 | 0.0056 | 0.68% |
| Occitan n-grams | 0.8169 | 0.0055 | 0.67% |
| Syllable counts | 0.8194 | 0.0030 | 0.37% |
| VC patterns | 0.8220 | 0.0004 | 0.05% |
| Stress patterns | 0.8239 | -0.0015 | -0.18% |

PoS-conditioned occlusion experiments showed: NOUN mean $\Delta=+0.0026$, DET $+0.0010$, ADJ $+0.0003$ (all significantly positive at $p<10^{-4}$), while CCONJ/ADP/VERB contributed negatively and PUNCT/PRON were not significant. This quantitatively confirms that gender information is primarily carried by nouns, articles, and adjectives.

### Key Findings
- Suffix morphology is the strongest single source: Removing Latin n-grams caused a drop of 1.6–1.8 F1 points across all embeddings, much higher than CV templates or stress proxies. Stress proxies actually slightly penalized the model ($\Delta = -0.0015$), suggesting noise in the heuristic stress rules.
- Context provides massive gains: Compairing context vs. word-only yielded $\Delta_1^{\text{prob}}=+0.283$ and $\Delta_2^{\log p}=+0.340$ (95% CI strictly > 0), indicating context not only improves accuracy but also stabilizes confidence in the correct class.
- Latin meta-information acts as a context "amplifier": Removing Latin lemma + Latin gender reduced the masked-context gain from ~0.28 to ~0.09–0.11, showing that cross-lingual alignment and context are complementary rather than redundant.

## Highlights & Insights
- The design of three input levels (word-only / context / masked-context) allows for clear differentiation of signals; the masked-context design is particularly elegant as it forces the model to recover gender from syntactic agreement without seeing the noun itself.
- The Hybrid tokenizer achieved 0% OOV and 25.23% masked recovery, demonstrating that OOV and subword quality are not binary trade-offs in low-resource historical languages.
- PoS-conditioned occlusion combined with sign-flip permutation tests provides statistically significant contributions for each part of speech, a lightweight attribution framework applicable to any agreement-driven task.

## Limitations & Future Work
- Small corpus size (~130k tokens) and 2:1 imbalance limit generalization for the feminine class; focal loss only partially mitigates this.
- Key hyperparameters rely on heuristics: the fuzzy matching threshold $\tau=0.85$, stress rules, and $\alpha=0.3$ lack systematic tuning.
- PoS occlusion relies on automatic tagging; the 71% accuracy of the PoS tagger (especially for adjectives) potentially contaminates attribution results.
- Performance is significantly worse for nouns at sentence boundaries or in sparse contexts; future work could include boundary-aware attention masks.
- Conclusions are limited to the "Latin Neuter → Occitan M/F" transition. Applicability to other Romance languages (e.g., Provençal, Catalan) or diachronic analysis remains to be tested.

## Related Work & Insights
- **vs. Williams et al. (2019) Information-theoretic quantification of gender in German/Czech**: While they decompose gender into form/meaning/inflection using information theory, Ours focuses on low-resource historical corpora and BERT-style attribution. Ours explicitly quantifies word vs. context increments.
- **vs. Cucerzan & Yarowsky (2003) Minimally supervised context gender induction**: They utilize co-occurrence with gendered articles; Ours is a neural version (mBERT + attention) and provides a quantifiable masked vs. unmasked comparison, proving the article/adjective agreement remains a robust weak label in historical corpora.
- **vs. Rule-based French gender prediction (Lyster 2006)**: Rule-based methods are language-specific; the Hybrid tokenization + domain-adapted mBERT paradigm in Ours is more transferable and provides visualizations (SHAP/occlusion) equivalent to morphological rules.

## Rating
- Novelty: ⭐⭐⭐ No flashy model architectures, but fills a gap by using NLP to quantify historical linguistics in Medieval Occitan.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes embedding/tokenizer/feature ablations, paired bootstrap, sign-flip tests, SHAP, and occlusion.
- Writing Quality: ⭐⭐⭐⭐ Clear research questions with results strictly aligned to data.
- Value: ⭐⭐⭐ Useful for both historical linguistics and low-resource NLP; the methodological framework is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs](../../AAAI2026/interpretability/finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](../../ICLR2026/interpretability/exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](../../ICML2026/interpretability/optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[NeurIPS 2025\] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](../../NeurIPS2025/interpretability/cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)
- [\[ACL 2026\] Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling](learning_what_matters_dynamic_dimension_selection_and_aggregation_for_interpreta.md)

</div>

<!-- RELATED:END -->
