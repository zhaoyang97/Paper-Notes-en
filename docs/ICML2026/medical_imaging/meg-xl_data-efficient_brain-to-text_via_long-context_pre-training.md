---
title: >-
  [Paper Note] MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training
description: >-
  [ICML 2026][Medical Imaging][Brain-to-text] MEG-XL utilizes 2.5 minutes (191k tokens) of MEG context for mask token pre-training (5–300$\times$ longer than previous methods). When fine-tuned on a 50-word brain-to-text ta…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Brain-to-text"
  - "long-context pre-training"
  - "MEG"
  - "criss-cross attention"
  - "masked token prediction"
date: 2026-05-08
content_hash: a6e1eb31a75ef3d6
---

# MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training

**Conference**: ICML 2026  
**arXiv**: [2602.02494](https://arxiv.org/abs/2602.02494)  
**Code**: Open source (paper claims release of code + weights)  
**Area**: Brain-Computer Interface / Neural Decoding / Foundation Models  
**Keywords**: Brain-to-text, long-context pre-training, MEG, criss-cross attention, masked token prediction

## TL;DR
MEG-XL utilizes 2.5 minutes (191k tokens) of MEG context for mask token pre-training (5–300$\times$ longer than previous methods). When fine-tuned on a 50-word brain-to-text task, it achieves the decoding accuracy of SOTA supervised methods using only 1 hour of data compared to the typical 50 hours, significantly outperforming all existing brain foundation models.

## Background & Motivation

**Background**: Brain-to-text (B2T) decoding is a core direction in Brain-Computer Interface (BCI), categorized into invasive (cortical electrodes, where Moses 2021, Willett 2023, and Card 2024 have reached usable precision) and non-invasive (MEG/EEG, which have lower barriers but weaker signals). Representative non-invasive works include Défossez et al. (2022) for 1-second MEG speech decoding and d'Ascoli et al. (2025), which extended context to the sentence level (150s) for word decoding. Brain foundation models (LaBraM, BIOT, EEGPT, BrainOmni, CBraMod) typically perform mask pre-training on short windows of 5–30 seconds.

**Limitations of Prior Work**: (1) Supervised methods rely on approximately 50 hours of training data per subject, which is impractical for paralyzed patients who cannot provide such long recordings. (2) Existing brain foundation models pre-train almost exclusively on short windows ($\le$10 seconds), resulting in a severe mismatch with the long-term neurolinguistic structures (phrases, sentences, discourse) required downstream; recent analysis (Yang 2026) shows these FMs underperform compared to supervised methods in low-data scenarios. (3) Extending context is hindered by computational bottlenecks: standard transformer attention is $\mathcal{O}((CT')^2)$, causing memory overflow with multi-channel and long-duration signals.

**Key Challenge**: Neural activity contains language-related structures spanning tens of seconds to minutes (phrase aggregation, syntax, discourse coherence), but short-window pre-trained models can neither perceive nor learn to utilize these long-range dependencies. Furthermore, the clinical need for "fast adaptation to new subjects with minimal data" is precisely where short-context FMs fail.

**Goal**: (i) Construct a framework capable of mask pre-training on minute-level MEG context without memory overflow; (ii) verify if long-context pre-training truly outperforms SOTA supervised methods and existing FMs in low-data downstream scenarios (especially contextual word decoding); (iii) explain why long context is effective—whether it truly learns selective and hierarchical attention.

**Key Insight**: Paying homage to Transformer-XL, the authors argue that "neural data = long documents" and must be pre-trained in long contexts like LMs to learn long-range statistical priors. Computational bottlenecks are resolved using criss-cross factorized attention (Wang 2025), which decouples and parallelizes attention across temporal and spatial dimensions.

**Core Idea**: Use BioCodec to independently tokenize each channel (rank 12 temporal compression), feed them into an 8-layer criss-cross transformer, and mask 40% of 3-second blocks within a 2.5-minute MEG window for prediction. This forces the model to learn neural dependencies across minutes; fine-tuning for word decoding then yields a data-efficient B2T model.

## Method

### Overall Architecture
Pre-training Stage: Raw MEG $\mathbf{X}\in\mathbb{R}^{C\times T}$ (multi-channel, 50Hz downsampled) is passed through a frozen BioCodec independently per channel (RVQ 6 layers, vocabulary 256, 12$\times$ temporal downsampling) to obtain $\mathbf{Z}\in\{0,...,255\}^{C\times T'\times 6}$. Token embeddings concatenate 6 codebook vectors and project them to $d_{model}$, adding sensor positions (Fourier features), orientation, and type embeddings, followed by an 8-layer criss-cross transformer. 3-second blocks are uniformly masked until 40% of tokens are covered, and the model predicts 6 RVQ-level codes per position. Fine-tuning Stage: Adopting the d'Ascoli task settings—50 words $\times$ 3-second MEG windows are concatenated into a 150-second input; the model predicts the T5 word embedding corresponding to each word's time interval. Training uses an MLP head with SigLIP contrastive loss; at inference, words are retrieved via nearest neighbor search.

### Key Designs

1.  **2.5-minute Ultra-long Context Pre-training + Criss-cross Attention**:
    - **Function**: Enables the model to perform mask prediction on sequences of 191k tokens without memory exhaustion.
    - **Mechanism**: Standard attention's $\mathcal{O}((CT')^2)$ is unfeasible for long-duration multi-channel data. Criss-cross splits the feature dimension into halves: one half undergoes SpatialAttn (independent cross-channel attention per timestep, complexity $\mathcal{O}(T'\cdot C^2)$), and the other half undergoes TemporalAttn (independent cross-time attention per channel, complexity $\mathcal{O}(C\cdot T'^2)$). RoPE is added to TemporalAttn for positional encoding. The halves are concatenated, followed by residual connections, RMSNorm, and SELU FFN. Total complexity drops from $\mathcal{O}((CT')^2)$ to $\mathcal{O}(C\cdot T'^2+T'\cdot C^2)$, allowing 2.5 minutes $\times$ hundreds of channels $\times$ 50Hz sequences to fit on a single GPU.
    - **Design Motivation**: Neurolinguistic structures span seconds to minutes; short-window models are structurally incapable of reaching them. The physical intuition of criss-cross is that "temporal and spatial correlations are approximately separable," which is a good approximation for brain signals with high sensor-wise temporal and cross-sensor spatial correlations.

2.  **Per-channel Independent RVQ Tokenization (BioCodec) + Residual Codebook Input Embeddings**:
    - **Function**: Compresses continuous MEG signals into discrete token sequences, reducing sequence length while providing mask prediction targets.
    - **Mechanism**: BioCodec (a neural audio codec-style tokenizer trained on EEG) performs RVQ independently for each channel: $Q=6$ residual quantization levels, each with a vocabulary $V=256$. Temporal downsampling of 12$\times$ compresses 50Hz $\times$ 150s $\times$ hundreds of channels into 191k tokens. Input embeddings are derived by looking up codebooks $\mathbf{e}^{(q)}_{z_{c,q,t}}$, then concatenated and projected: $\mathbf{h}^{(0)}_{c,t}=\mathbf{W}_{proj}[\mathbf{e}^{(1)};...;\mathbf{e}^{(Q)}]$; plus sensor position Fourier features $\gamma(\mathbf{v})=[\cos(2\pi\mathbf{Bv}),\sin(2\pi\mathbf{Bv})]$, orientation, and type embeddings.
    - **Design Motivation**: Unlike BrainTokenizer which compresses time and space simultaneously, this work only compresses time—leading to better reconstruction quality and preventing the loss of task-relevant information during tokenization. RVQ provides higher fidelity for high-frequency time-series data compared to single VQ by capturing both slow and fast dynamics.

3.  **Large Block Masking (3s) + Synchronous All-channel Masking Prediction**:
    - **Function**: Forces the model to learn temporal dependencies across seconds rather than simple cross-channel interpolation.
    - **Mechanism**: Random 3-second blocks are selected until 40% of tokens are masked. All channels are **synchronously masked** at the selected time steps and replaced with mask embeddings. The model predicts RVQ codes for each mask position: $p(z_{c,q,t}\mid\mathbf{X}_{\backslash\mathcal{M}})=\text{softmax}(\mathbf{W}_q\mathbf{h}^{(L)}_{c,t})$, with loss $\mathcal{L}=-\frac{1}{|\mathcal{M}|CQ}\sum_t\sum_c\sum_q\log p(z_{c,q,t}\mid\mathbf{X}_{\backslash\mathcal{M}})$. The 3-second block size is deliberately chosen to cover the typical duration of neural responses to words (Kutas & Federmeier 2011).
    - **Design Motivation**: Due to high temporal autocorrelation in MEG, short masks allow the model to use "neighbor interpolation" as a shortcut; 3-second blocks + synchronous masking eliminate these shortcuts, forcing true long-term modeling. The 40% mask ratio is higher than BERT's 15%, falling between MAE (75%) and wav2vec 2.0 (49%), based on empirical tuning.

### Loss & Training
Pre-training: Approximately 300 hours of MEG data (CamCAN + MOUS + SMN4Lang), covering resting state, motor, and speech tasks across hundreds of subjects; mask token prediction cross-entropy; channel masking is used to handle padding for different MEG systems. Fine-tuning: SigLIP contrastive loss + word embedding regression head; end-to-end fine-tuning of transformer + MLP head; nearest neighbor word retrieval via cosine similarity.

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

In low-data (13%) scenarios, MEG-XL outperforms the next best model, LaBraM, by 13–28 points; with full data, it matches or exceeds BrainOmni. BrainOmni's failure on MEG-MASC (shallow multi-subject data) highlights the struggles of existing FMs in "low-data per subject" clinical scenarios.

### Ablation Study

| Configuration | Effect |
|---|---|
| Random init MEG-XL (No pre-training) | Performance near supervised baseline, proving gains come from pre-training, not architecture. |
| Pre-training context 5s $\to$ 30s $\to$ 100s $\to$ 150s | Monotonic improvement in linear probe word decoding, saturating around 100s. |
| Full-context vs Matched-context inference | Almost overlapping $\to$ providing longer context during inference is useless unless pre-trained on it. |
| Masked prediction (Zero-shot) | Monotonic improvement from 5s to 150s, unsaturated—longer might still help. |
| Attention analysis | Long-context models show local attention in early layers and global integration in deep layers + lower attention entropy. |

### Key Findings
- Dramatic Gain in Data Efficiency: Accuracy achieved by MEG-XL with 1 hour of data takes SOTA supervised methods 50 hours (approx. 50$\times$ data efficiency).
- Long context learns "when to look far / when to look near"—a selective hierarchical attention that short-context models, which attend uniformly from the first layer, fail to acquire.
- On "deep single-subject" data like LibriBrain, supervised methods (d'Ascoli) still catch up when data is sufficient (after 2.5 hours), defining the boundary of "pre-training vs. subject-specific data."
- MEG-XL completely dominates in low-data regimes (+25 points on MEG-MASC), the critical scenario for BCI clinical deployment.

## Highlights & Insights
- "Long context is a learned ability, not a given capability" is the most significant insight—simply providing more context at inference is ineffective; it must be part of pre-training. This aligns with LM length generalization literature, bringing this principle to neural decoding.
- The success of criss-cross attention on brain signals suggests that highly structured spatio-temporal data can bypass quadratic complexity via "spatio-temporal factorized attention," applicable to fMRI, ECoG, and sensor networks.
- In clinical deployment, "cross-subject pre-training replacing within-subject training" is a paradigm shift, reducing BCI calibration for new users from 50 hours to 1–2 hours.
- The 3-second block + synchronous masking is a clever design that disables both "temporal" and "spatial" neighbor interpolation shortcuts, forcing semantic-level modeling.

## Limitations & Future Work
- Only tested on perceived speech (listening to audiobooks); did not touch the more difficult "imagined speech," which is how paralyzed patients actually use BCI.
- Retrieval vocabulary is only 50 (top-250 trends are similar), still orders of magnitude away from the thousands of words needed for open vocabulary clinical use.
- Interpretability gains from long context (hierarchical attention) remain at a statistical description level; no clear link to specific linguistic structures (syllables/words/phrases) is established.
- Memory remains the ceiling—150s is the GPU VRAM limit, preventing verification of whether even longer contexts continue to yield benefits.
- Pre-training data consists of healthy research participants, potentially causing a domain shift when applied to real paralyzed patients.

## Related Work & Insights
- **vs d'Ascoli et al. (2025) (Supervised SOTA)**: They first extended MEG input to 150s sentences but relied on supervised training; MEG-XL adopts the same length but reduces data needs by 1–2 orders of magnitude via SSL.
- **vs LaBraM / EEGPT / BIOT / BrainOmni**: These FMs use $\le$30s windows and collapse in low-data settings; MEG-XL qualitatively changes performance by expanding context to minutes.
- **vs CBraMod / BrainOmni (Criss-cross origins)**: BrainOmni also uses criss-cross attention but keeps the 30s window; this work proves the true value of factorized attention lies in the 150s window.
- **vs Transformer-XL (Naming tribute)**: Directly ports the LM long-context paradigm to neural data and proves the validity of similar laws.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to combine minute-level context + RVQ tokenization + criss-cross attention for B2T; clear logic and significant results.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Included 3 MEG datasets + 6 FM baselines + supervised SOTA + linear probing + zero-shot prediction + attention analysis; very complete chain of evidence.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Excellent storytelling—analogizing LM success to neural data; theoretical framework + empirical results + mechanistic analysis are seamless.
- **Value**: ⭐⭐⭐⭐⭐ Substantial push for non-invasive BCI clinical feasibility; the "long-context as a learned ability" principle is methodologically significant for neural FMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](../../CVPR2026/medical_imaging/ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)
- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](../../CVPR2026/medical_imaging/meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](../../NeurIPS2025/medical_imaging/brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](../../AAAI2026/medical_imaging/mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)

</div>

<!-- RELATED:END -->
