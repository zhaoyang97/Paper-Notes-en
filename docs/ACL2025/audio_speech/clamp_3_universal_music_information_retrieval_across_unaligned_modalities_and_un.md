---
title: >-
  [Paper Note] CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages
description: >-
  [ACL 2025][Audio & Speech][Music Information Retrieval] Proposed the CLaMP 3 unified framework, which aligns sheet music, performance signals, and audio recordings with multilingual text into a shared representation space via contrastive learning. This enables cross-modal retrieval across modalities without paired training data and demonstrates strong generalization capabilities to unseen languages.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Music Information Retrieval"
  - "Multimodal Alignment"
  - "Cross-Lingual Generalization"
  - "Contrastive Learning"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 2f5cb9bc6b71c041
---

# CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages

**Conference**: ACL 2025  
**arXiv**: [2502.10362](https://arxiv.org/abs/2502.10362)  
**Code**: [https://github.com/sanderwood/clamp3](https://github.com/sanderwood/clamp3)  
**Area**: Speech  
**Keywords**: Music Information Retrieval, Multimodal Alignment, Cross-Lingual Generalization, Contrastive Learning, Retrieval-Augmented Generation

## TL;DR

Proposed the CLaMP 3 unified framework, which aligns sheet music, performance signals, and audio recordings with multilingual text into a shared representation space via contrastive learning. This enables cross-modal retrieval across modalities without paired training data and demonstrates strong generalization capabilities to unseen languages.

## Background & Motivation

**Background**: Music Information Retrieval (MIR) aims to develop computational tools to process, organize, and access music data. The core challenge is retrieving music content (sheet music, performance signals, and audio recordings) based on natural language queries. Recent progress includes retrieval and auto-tagging systems based on text-audio alignment.

**Limitations of Prior Work**: (1) Music exists in multiple modalities (sheet music, MIDI, and audio), making unified processing difficult due to heterogeneous representation structures; (2) Music is described in various languages worldwide, but existing datasets are predominantly English-centric, lacking multilingual coverage; (3) Paired data is scarce—both cross-modal paired data and high-quality music-text pairs are severely limited.

**Key Challenge**: MIR needs to uniformly process multiple musical modalities and languages, but most current methods focus only on specific modality pairs (e.g., text-audio or text-sheet music) and lack cross-lingual capabilities and unified cross-modal representations.

**Goal**: How to simultaneously align all major musical modalities with multilingual text in a unified framework, enabling retrieval between unpaired modalities and generalization to unseen languages.

**Key Insight**: Using text as a bridge and drawing inspiration from the multi-phase alignment strategy of ImageBind, all modalities are mapped to a shared space via contrastive learning; a large-scale multilingual music-text dataset is constructed from the web using RAG.

**Core Idea**: Bridging modalities with text, unifying the alignment of sheet music, performance signals, audio, and multilingual text into a shared representation space via multi-phase contrastive learning, supplemented by the large-scale M4-RAG dataset constructed using RAG.

## Method

### Overall Architecture

CLaMP 3 consists of three Transformer encoders: a multilingual text encoder, a symbolic music encoder, and an audio music encoder. The training objective is to align text and music features via contrastive learning (InfoNCE loss).

Training loss function:

$$L_{CL} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i^t, z_i^m) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i^t, z_j^m) / \tau)}$$

where $z_i^t$ and $z_i^m$ are text and music embeddings respectively, and $\tau$ is the temperature parameter.

### Key Designs

**Module 1: Multilingual Text Encoder**

- **Function**: Encoding multilingual text descriptions
- **Mechanism**: Based on XLM-R-base, pre-trained on 2.5TB of CommonCrawl data across 100 languages, with a 12-layer Transformer and a hidden dimension of 768.
- **Design Motivation**: The cross-lingual semantic capability of XLM-R enables the model to generalize to languages unseen during training.

**Module 2: Symbolic Music Encoder**

- **Function**: Encoding sheet music (ABC notation) and performance signals (MIDI)
- **Mechanism**: Based on the M3 self-supervised model, ABC is segmented by bar and MIDI is segmented by message, with each segment serving as a patch. It uses a 12-layer encoder with a hidden dimension of 768, processing up to 512 patches (32,768 characters).
- **Design Motivation**: M3 acquires deep comprehension of symbolic music through self-supervised learning.

**Module 3: Audio Music Encoder**

- **Function**: Encoding audio recordings
- **Mechanism**: A 12-layer Transformer (768-dimensional) trained from scratch. It utilizes frozen MERT-v1-95M as a feature extractor, where each 5-second audio clip produces an embedding (averaged across all MERT layers and timesteps). It can process up to 128 embeddings (640 seconds of audio).
- **Design Motivation**: Utilizing MERT pre-trained features to initialize audio representations while supporting super-long audio modeling.

**Module 4: Multi-Phase Alignment Strategy**

- **Function**: Resolving multi-modal alignment challenges in the absence of paired music data
- **Mechanism**: A four-phase strategy—Stage 1: align text with symbolic encoder; Stage 2: freeze the text encoder, align the audio encoder; Stage 3: unfreeze the text encoder, fine-tune the audio alignment; Stage 4: freeze the text encoder again, fix symbolic alignment drift.
- **Design Motivation**: Alternating between freezing and unfreezing the text encoder minimizes cross-modal interference while mapping all modalities into a shared space.

### Loss & Training

- **Contrastive Loss**: InfoNCE loss (formula shown above).
- **Training Configuration**: Symbolic alignment trained for 100 epochs on 8×H800 GPUs for 4 days; audio alignment trained for 100 epochs for 1 day. Learning rates are 5e-5 and 1e-5, and batch sizes are 1024 and 2048, respectively.
- **Data Split**: M4-RAG 99% training + 1% validation.
- **Optimization**: Mixed-precision training, AdamW optimizer, 1000 warmup steps.

**M4-RAG Dataset Construction**:
- Title filtering → Google Search (top-10 results per query) → Qwen2.5-72B annotation generation (RAG) → quality filtering → post-processing → multilingual translation.
- Contains short tags (genres, tags) and long descriptions (background, analysis, description, scene), spanning 27 languages and 194 countries.

## Key Experimental Results

### Main Results

English text-to-music retrieval (MRR metric):

| Model | WikiMT | MidiCaps | SDD | MC-R |
|------|--------|----------|-----|------|
| CLaMP | 0.2561 | 0.1236 | - | - |
| CLaMP 2 | 0.3438 | 0.2695 | - | - |
| CLAP | - | - | 0.1310 | 0.0657 |
| TTMR++ | - | - | 0.1437 | 0.1248 |
| **CLaMP 3_sa^c2** | **0.4498** | **0.2826** | 0.1612 | 0.0959 |
| **CLaMP 3_saas** | 0.3555 | 0.1798 | **0.1985** | **0.1177** |

Multilingual retrieval (ABC Notation, WikiMT-X Background):

| Model | ru | fr | es | fi* | el* | kk* |
|------|-----|-----|-----|------|------|------|
| CLaMP 2 | 0.2668 | 0.2968 | 0.2934 | 0.2795 | 0.2410 | 0.2543 |
| CLaMP 3_sa^c2 | **0.3614** | **0.3949** | **0.3921** | **0.3524** | **0.3226** | **0.3397** |

(*Marked as unseen languages during training)

### Ablation Study

Cross-modal emergent retrieval (WikiMT-X, MRR):

| Model | S→P | S→A | P→S | P→A | A→S | A→P |
|------|------|------|------|------|------|------|
| CLaMP 2 | 0.5138 | - | 0.4480 | - | - | - |
| CLaMP 3_sa^c2 | 0.4547 | 0.0543 | **0.5293** | 0.0313 | 0.0492 | 0.0383 |
| CLaMP 3_saas | 0.3262 | **0.0578** | 0.3146 | **0.0397** | 0.0410 | 0.0303 |

The two model variants comparatively highlight the effects of different alignment sequences: CLaMP 3_sa^c2 optimizes symbolic retrieval, while CLaMP 3_saas optimizes audio retrieval.

### Key Findings

1. CLaMP 3 improves the MRR on WikiMT symbolic retrieval from 0.3438 (CLaMP 2) to 0.4498, yielding a 30.8% **Gain**.
2. For audio retrieval, CLaMP 3_saas achieves an MRR of 0.1985 on SDD, substantially outperforming TTMR++ (0.1437) and CLAP (0.1310).
3. **Cross-Lingual Generalization**: CLaMP 3_saas achieves an audio retrieval MRR of 0.1770 on Finnish (unseen during training), exceeding CLAP's performance on English (0.0598).
4. **Cross-Modal Emergence**: CLaMP 3 achieves retrieval between symbolic music and audio without paired training data (significantly exceeding the random baseline of 0.0075).
5. The rich annotations in the M4-RAG dataset lead to substantial improvements even on semantically sparse types like Description and Scene.

## Highlights & Insights

- The **text-as-a-bridge** multi-stage alignment strategy is highly ingenious—enabling cross-modal retrieval without any paired score-audio data.
- The **RAG construction pipeline of M4-RAG** is a key highlight: leveraging the unique identifiability of music titles for web searches, then using LLMs to generate multi-dimensional annotations, which is cost-effective and large-scale (2.31 million pairs).
- The **WikiMT-X benchmark** fills the gap as the first evaluation benchmark in the music domain that simultaneously covers three modalities: text, sheet music, and audio.
- The cross-lingual generalization capability is impressive—performing reasonably even on extremely low-resource unseen languages like Amharic.

## Limitations & Future Work

1. **Lack of Temporal Modeling**: Contrastive learning uses a single global representation, failing to capture temporal dynamics in music (e.g., the thematic development in Beethoven's Fifth Symphony).
2. **Multilingual Evaluation Dependent on Translation**: Lacking native multilingual benchmarks, discrepancies in translation quality introduce noise.
3. **Weak Audio-Symbolic Alignment**: Although cross-modal emergent retrieval far exceeds the random baseline, the MRR in audio-related directions remains low (0.03-0.06), necessitating paired data for supervised alignment.
4. The dataset predominantly features Western music, potentially offering insufficient depth in the coverage of non-Western musical traditions.

## Related Work & Insights

- **CLaMP/CLaMP 2** serve as prior works, progressing from symbolic-text alignment to a unified multimodal and multilingual approach.
- **ImageBind**'s concept of aligning multiple modalities using text as a bridge directly inspired CLaMP 3's multi-stage strategy.
- **MERT** provides high-quality audio representations, acting as a frozen feature extractor to input into the audio encoder.
- The application of **RAG** technology for dataset construction rather than inference represents a novel direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First to unify all music modalities + multilingual text, achieving cross-modal emergent retrieval)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Comprehensive evaluation across multi-modal/multilingual/cross-modal settings, with multiple benchmarks)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, abundant figures, and detailed experimental setups)
- **Value**: ⭐⭐⭐⭐⭐ (Open-source models + datasets + benchmarks, contributing significantly to the MIR community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FIGMA: Towards Fine-Grained Music Retrieval](../../ACL2026/audio_speech/figma_towards_fine-grained_music_retrieval.md)
- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)
- [\[ACL 2025\] ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors](atri_mitigating_multilingual_audio_text_retrieval_inconsistencies_by_reducing_da.md)
- [\[ACL 2025\] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](gigaspeech2_low_resource_asr.md)
- [\[ACL 2025\] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)

</div>

<!-- RELATED:END -->
