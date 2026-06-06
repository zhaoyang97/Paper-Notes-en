---
title: >-
  [Paper Note] MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training
description: >-
  [ICML 2026][Medical Imaging][brain-to-text] MEG-XL performs masked token pre-training on 2.5 minutes (191k tokens) of MEG context (5–300× longer than prior work)…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "brain-to-text"
  - "long-context pre-training"
  - "MEG"
  - "criss-cross attention"
  - "masked token prediction"
date: 2026-05-08
content_hash: 3a09870cf255688d
---

# MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training

**Conference**: ICML 2026  
**arXiv**: [2602.02494](https://arxiv.org/abs/2602.02494)  
**Code**: Open-sourced (paper states release code + weights)  
**Area**: Brain-Computer Interface / Neural Decoding / Foundation Models  
**Keywords**: brain-to-text, long-context pre-training, MEG, criss-cross attention, masked token prediction

## TL;DR
MEG-XL performs masked token pre-training on 2.5 minutes (191k tokens) of MEG context (5–300× longer than prior work), then fine-tunes on a 50-word brain-to-text task. With only 1 hour of data, it matches the decoding accuracy of SOTA supervised methods trained on 50 hours, and significantly outperforms all brain foundation models.

## Background & Motivation

**Background**: Brain-to-text (B2T) decoding is a core direction in brain-computer interfaces (BCI), divided into invasive (cortical electrodes, e.g., Moses 2021, Willett 2023, Card 2024, which have reached usable accuracy) and non-invasive (MEG/EEG, lower threshold but weaker signals). Non-invasive representatives include Défossez et al. (2022) decoding speech from 1s MEG, and d'Ascoli et al. (2025) extending context to sentence-level (150s) for word decoding. Brain foundation models (LaBraM, BIOT, EEGPT, BrainOmni, CBraMod) perform masked pre-training on short windows of 5–30 seconds.

**Limitations of Prior Work**: (1) Supervised methods require ~50 hours of training data per subject, which is impractical for paralyzed patients who cannot provide long training recordings. (2) Existing brain foundation models are almost all pre-trained on ≤10s short windows, which is severely mismatched with the long-range neural linguistic structures (phrases, sentences, discourse) needed downstream; recent analysis (Yang 2026) finds these FMs underperform supervised methods in low-data regimes. (3) Extending context is bottlenecked by computation: standard transformer attention is $\mathcal{O}((CT')^2)$, and multi-channel + long sequence quickly exhausts GPU memory.

**Key Challenge**: Neural activity contains language-related structures spanning tens of seconds to minutes (phrase aggregation, syntax, discourse coherence), but short-window pre-trained models cannot see or learn to exploit such long-range dependencies. Meanwhile, the clinical deployment scenario most in need of "fast adaptation with little data" is precisely the blind spot of short-context FMs.

**Goal**: (i) Build a framework for masked pre-training on minute-scale MEG context without exhausting GPU memory; (ii) Verify whether long-context pre-training can truly outperform SOTA supervised methods and all existing FMs in low-data downstream scenarios (especially contextual word decoding); (iii) Explain why long context is useful—does it really learn selective and hierarchical attention?

**Key Insight**: Inspired by Transformer-XL, the authors view "neural data = long documents" and argue that only pre-training on long context, as in language modeling, can capture long-range statistical priors. The computational bottleneck is addressed by criss-cross factorized attention (Wang 2025), which decouples attention along time and channel dimensions for parallelization.

**Core Idea**: Each channel is independently tokenized using BioCodec (rank-12 temporal compression), fed into an 8-layer criss-cross transformer. Within a 2.5-minute MEG window, 40% of 3-second blocks are masked for prediction, forcing the model to learn minute-scale neural dependencies. Fine-tuning on word decoding yields a data-efficient B2T model.

## Method

### Overall Architecture
Pre-training: Raw MEG $\mathbf{X}\in\mathbb{R}^{C\times T}$ (multi-channel, 50Hz downsampled) is passed through a frozen BioCodec (RVQ 6 layers, vocab 256, 12× temporal downsampling) per channel to obtain $\mathbf{Z}\in\{0,...,255\}^{C\times T'\times 6}$. Token embeddings concatenate 6 codebook vectors, projected to $d_{model}$, with added sensor position (Fourier features), orientation, and type embeddings, then processed by an 8-layer criss-cross transformer. 3-second blocks are uniformly masked until 40% of tokens are covered, predicting 6 RVQ code levels per position. Fine-tuning: Adopts d'Ascoli's task—50 words × 3s MEG windows concatenated into 150s input; the model predicts the T5 word embedding for each word's time segment. An MLP head is trained with SigLIP contrastive loss; inference uses nearest neighbor retrieval.

### Key Designs

1. **2.5-Minute Ultra-Long Context Pre-Training + Criss-Cross Attention**:

    - **Function**: Enables masked prediction on 191k-token sequences without exhausting GPU memory.
    - **Mechanism**: Standard attention's $\mathcal{O}((CT')^2)$ complexity is infeasible for multi-channel, long sequences. Criss-cross splits features in half: one half uses SpatialAttn (per time step, cross-channel attention, $\mathcal{O}(T'\cdot C^2)$), the other uses TemporalAttn (per channel, cross-time attention, $\mathcal{O}(C\cdot T'^2)$). Temporal attention includes RoPE time encoding. The two halves are concatenated along the channel dimension, with residual connections, RMSNorm, and SELU FFN. Total complexity drops from $\mathcal{O}((CT')^2)$ to $\mathcal{O}(C\cdot T'^2+T'\cdot C^2)$, allowing a 2.5-minute × hundreds of channels × 50Hz × 12× compressed 191k-token sequence to fit on a single GPU.
    - **Design Motivation**: Neural linguistic structures span seconds to minutes; short-window models are structurally incapable of capturing them—models need a "long enough field of view." Native attention is a computational bottleneck; criss-cross leverages the physical intuition that "temporal and spatial correlations are approximately separable," which is a good approximation for brain signals with "temporal correlation per sensor + spatial correlation per time."

2. **Multi-Channel Independent RVQ Tokenization (BioCodec) + Residual Codebook Input Embedding**:

    - **Function**: Compresses continuous MEG signals into discrete token sequences, reducing sequence length and providing masked prediction targets.
    - **Mechanism**: BioCodec (a neural audio codec-style tokenizer trained on EEG) independently applies RVQ to each channel: $Q=6$ residual quantization levels, each with vocab $V=256$. 12× temporal downsampling compresses 50Hz × 150s × hundreds of channels into 191k tokens. Input embeddings look up each codebook $\mathbf{e}^{(q)}_{z_{c,q,t}}$, concatenate, and project $\mathbf{h}^{(0)}_{c,t}=\mathbf{W}_{proj}[\mathbf{e}^{(1)};...;\mathbf{e}^{(Q)}]$; sensor position Fourier features $\gamma(\mathbf{v})=[\cos(2\pi\mathbf{Bv}),\sin(2\pi\mathbf{Bv})]$, orientation, and type embeddings are added.
    - **Design Motivation**: Unlike BrainTokenizer, which compresses both time and channel, this approach compresses only time—yielding better reconstruction quality and avoiding loss of task-relevant information during tokenization. RVQ outperforms single VQ for high-frequency temporal data, with multi-level residuals capturing both slow dynamics and high-frequency details.

3. **Large Block Masking (3s) + Synchronized Masking Across All Channels**:

    - **Function**: Forces the model to learn temporal dependencies across seconds, not just simple cross-channel interpolation.
    - **Mechanism**: Randomly selects 3-second time blocks until 40% of tokens are masked; all channels are synchronously masked at selected time steps, replaced with mask embeddings. The model predicts RVQ codes for each masked position: $p(z_{c,q,t}\mid\mathbf{X}_{\backslash\mathcal{M}})=\text{softmax}(\mathbf{W}_q\mathbf{h}^{(L)}_{c,t})$, with loss $\mathcal{L}=-\frac{1}{|\mathcal{M}|CQ}\sum_t\sum_c\sum_q\log p(z_{c,q,t}\mid\mathbf{X}_{\backslash\mathcal{M}})$. The 3s block size is intentionally long, covering the typical neural response duration for words (Kutas & Federmeier 2011).
    - **Design Motivation**: MEG is highly temporally autocorrelated; short masks allow the model to "cheat" via neighbor interpolation, failing to learn long-range dependencies. 3s blocks + synchronized masking across channels eliminate these shortcuts, forcing genuine long-term modeling. The 40% mask rate is much higher than BERT's 15%, between MAE's 75% and wav2vec 2.0's 49%—empirically tuned.

### Loss & Training
Pre-training: ~300 hours of MEG (CamCAN + MOUS + SMN4Lang), covering rest, movement, speech, etc., across hundreds of subjects; masked token prediction with cross-entropy loss; different MEG systems have varying channel counts, handled via channel masking for padding. Fine-tuning: SigLIP contrastive loss + word embedding regression head; end-to-end fine-tuning of transformer + MLP head; inference uses cosine similarity for nearest neighbor word retrieval.

## Key Experimental Results

### Main Results

| Model | Params | MEG-MASC (13%) | Armeni (13%) | LibriBrain (13%) | MEG-MASC (100%) | Armeni (100%) | LibriBrain (100%) |
|---|---|---|---|---|---|---|---|
| BioCodec baseline | 1.0M | 19.8 | 20.0 | 19.9 | 31.2 | 37.1 | 41.9 |
| EEGPT | 4.7M | 19.6 | 20.3 | 20.3 | 26.3 | 20.8 | 22.9 |
| BIOT | 3.2M | 20.0 | 20.2 | 20.6 | 31.3 | 35.7 | 45.6 |
| BBL | 15M | 21.5 | 22.3 | 32.1 | 35.9 | 39.1 | 49.9 |
| BrainOmni | 8.4M | 18.7 | 21.0 | 29.7 | 19.1 | 62.3 | 63.0 |
| LaBraM | 5.8M | 33.2 | 26.3 | 40.3 | 31.1 | 42.0 | 47.7 |
| **MEG-XL (Ours)** | **20M** | **47.0** | **54.9** | **57.3** | **46.4** | **61.2** | **63.0** |

In low-data (13%) settings, MEG-XL outperforms the next-best LaBraM by 13–28 points; with full data, it matches or exceeds BrainOmni. BrainOmni collapses to 19.1% on MEG-MASC (shallow multi-subject), indicating that existing FMs fail in the clinically critical "shallow data, many subjects" scenario.

### Ablation Study / Long-Context Effects

| Configuration | Result |
|---|---|
| Random init MEG-XL (no pre-training) | Performance similar to supervised baseline, showing gains come from pre-training, not architecture |
| Pre-training context 5s → 30s → 100s → 150s | Word decoding linear probe improves monotonically, saturates after ~100s |
| Full-context vs Matched-context inference | Nearly identical → longer context at inference is useless unless pre-trained for it |
| Masked prediction (zero-shot) | Performance improves monotonically from 5s to 150s context, not saturated—longer may help further |
| Attention analysis | Long-context models show local attention in early layers + global integration in deeper layers + lower attention entropy |

### Key Findings
- Dramatic data efficiency: MEG-XL achieves the accuracy of supervised SOTA with 1 hour of data, which otherwise requires 50 hours (~50× data efficiency).
- Long-context models learn "when to look far / when to look near" via selective, hierarchical attention—short-context models attend uniformly from the first layer and never learn this stratification.
- On "deep single-subject" data like LibriBrain, d'Ascoli's supervised method overtakes MEG-XL when data is abundant (after 2.5 hours), indicating the boundary where pre-training can substitute for subject-specific data.
- In low-data settings, supervised methods are decisively outperformed by MEG-XL (over +25 points on MEG-MASC), which is the most important scenario for BCI clinical deployment.

## Highlights & Insights
- "Long context is a learned ability, not a given one" is the most important insight—providing long context only at inference is useless; it must be fed during pre-training. This echoes the length generalization literature in LMs, extending this principle to neural decoding.
- The success of criss-cross attention on brain signals suggests that highly structured spatiotemporal data can generally use "spatiotemporal factorized attention" to avoid quadratic complexity; this idea applies to fMRI, ECoG, sensor networks, and any $C\times T$ high-dimensional signals.
- In clinical deployment, "cross-subject pre-training replacing deep within-subject training" is a true paradigm shift—reducing BCI training from "50 hours per new user" to "1–2 hours," especially important for paralyzed patients.
- The 3-second large block + synchronized masking across all channels is a clever design—it simultaneously blocks "temporal neighbor interpolation" and "channel neighbor interpolation" shortcuts, forcing genuine semantic-level modeling.

## Limitations & Future Work
- Only tested on perceived speech (subjects listening to audiobooks), not the more challenging imagined speech; the latter is the actual use case for paralyzed BCI users.
- The retrieval vocabulary is only 50 words (top-250 shows similar trends), still several orders of magnitude short of the open vocabulary (thousands of words) needed clinically.
- The interpretability gains from long context (hierarchical attention) remain at the statistical description level; no clear evidence yet mapping to specific linguistic structures (syllable/word/phrase).
- GPU memory is still the limit—150s is the VRAM ceiling; unable to verify whether even longer context continues to help.
- Pre-training data comes from healthy research subjects; there may be domain shift with real MEG signals from paralyzed patients.

## Related Work & Insights
- **vs d'Ascoli et al. (2025) (Supervised SOTA)**: They first extended MEG input to sentence-level 150s, but via supervised training with high data demand; MEG-XL inherits the same context length but uses self-supervised pre-training to reduce data needs by one or two orders of magnitude.
- **vs LaBraM / EEGPT / BIOT / BrainOmni**: These brain FMs use ≤30s short-window pre-training and collapse in low-data settings; MEG-XL's minute-scale context brings qualitative change.
- **vs CBraMod / BrainOmni (criss-cross origin)**: BrainOmni also uses criss-cross attention but with a 30s window; this work shows that only at 150s does factorized attention realize its full value.
- **vs Transformer-XL (homage in naming)**: Directly transfers the LM long-context paradigm to neural data and demonstrates similar principles apply.

## Rating
- Novelty: ⭐⭐⭐⭐ First to combine minute-scale context + RVQ tokenization + criss-cross attention for B2T, with clear rationale and significant effect.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 MEG datasets + 6 FM baselines + supervised SOTA + linear probe + zero-shot prediction + attention analysis, with a very complete evidence chain.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally well-structured—drawing analogy from "LM long-context success" to neural data, integrating theoretical framework, empirical results, and mechanism analysis; formulas and figures are precise.
- Value: ⭐⭐⭐⭐⭐ Substantially advances the clinical feasibility of non-invasive BCI; the "long context is a learned ability" principle is methodologically significant for the entire neural foundation model field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](../../CVPR2026/medical_imaging/ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)
- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](../../CVPR2026/medical_imaging/meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](../../NeurIPS2025/medical_imaging/brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](../../AAAI2026/medical_imaging/towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)

</div>

<!-- RELATED:END -->
