# Research: Qwen3-ASR-1.7B vs Whisper large-v3 (Russian & Russian–English)

*Date: 2026-07-06. Context: this app (`transcribe/`) currently runs Whisper large-v3 via
`pywhispercpp` (whisper.cpp + Metal). Question: is Qwen3-ASR-1.7B better for Russian and
Russian–English (code-switched) transcription, and is it worth adopting here?*

> **Update — implemented.** Qwen3-ASR-1.7B-bf16 was added as a **prompt-tuned fallback engine**
> (`engine="qwen"` + auto-fallback; `transcribe/qwen_engine.py`, `transcribe/qwen_prompt.py`).
> Whisper stays the default. Benchmarked on an M4 Max through the real pipeline against the
> existing test videos and audio ripped from a real RU/RU-EN meeting: **performance matched
> (RTF 0.070 vs whisper 0.073) and accuracy was same-or-better** on the target Russian meeting
> — more complete text, English tech terms kept in Latin (item / deep dive / CPI / ECPA),
> correct participant names. See §4–5 and the domain prompt in `qwen_prompt.py`.

---

## TL;DR (verdict)

**Strictly between the two models:**

| Dimension | Winner | Confidence |
|---|---|---|
| **Clean/read Russian** (full precision) | **Qwen3-ASR-1.7B**, modestly | Low–moderate |
| **Russian–English code-switching** | Undetermined (lean Qwen on architecture + context-prompt) | Very low |
| **Russian on-device via MLX (quantized)** | Roughly a tie — quantization erases Qwen's edge | Moderate |
| **Fits this repo's pipeline today** | **Whisper large-v3** (already integrated, Metal, timestamps) | High |

**The bigger truth:** if the real goal is *"best Russian,"* the answer is **neither** —
a dedicated Russian model (**GigaAM v3**) beats both by **3–8×** on clean Russian. Whisper
and Qwen are middle-of-the-field *multilingual generalists*, not Russian specialists.

**For Russian–English code-switching (the IT/tech-speech case): no one has proven anything.**
There is zero reproducible RU-EN benchmark for either model. Qwen is architecturally more
promising and — uniquely — accepts a **context/vocabulary prompt** (inject your English
library/CLI names), which is the single most credible lever for keeping *"Gemini"* from
becoming *"Джемини"*. You must A/B on your own audio.

---

## 0. First, untangle "Qwen3-ASR" — it means two different things

| | **Qwen3-ASR-Flash** | **Qwen3-ASR (open weights)** |
|---|---|---|
| What | Hosted cloud **API** | **Downloadable** checkpoints |
| Released | **2025-09-08** | **2026-01-29** |
| Weights | Closed | **Open, Apache-2.0** |
| Sizes | undisclosed (larger) | **0.6B**, **1.7B**, + ForcedAligner-0.6B |
| Languages | 11 (incl. Russian) | 30 + 22 Chinese dialects (incl. Russian) |
| Access | DashScope / Alibaba Cloud; OpenRouter (~$0.13/hr) | Hugging Face / ModelScope |

- **"Qwen3-ASR-1.7B" is real and downloadable** (Apache-2.0). Note: "1.7B" = the LLM
  decoder only; the full model (Qwen3-1.7B + 300M "AuT" audio encoder + projector) is **~2B**.
- Architecture = **audio encoder + LLM decoder** (Qwen3-Omni lineage). This is *not* a
  Whisper-style model — it will **not** drop into whisper.cpp/pywhispercpp.
- Flash (API) publishes **no Russian WER** at all. All Russian numbers below are for the
  **open 1.7B/0.6B** models, from Qwen's own technical report (arXiv:2601.21337).

---

## 1. Russian accuracy — the numbers

WER %, lower is better. **Numbers are only comparable within the same dataset + normalization.**
Vendor = self-reported; independent = third-party.

| Model | Common Voice (ru) | Fleurs (ru) | MLS (ru) | Rus. LibriSpeech | Golos crowd/farfield | Source |
|---|---|---|---|---|---|---|
| **Whisper large-v3** | **9.84** (CV17) · 5.5 (CV19, Sber norm) | ~9–10 | ~4.9 | 9.4 | 19.0 / 16.4 | independent (antony66, Sber eval) |
| **Qwen3-ASR-1.7B** | **8.28** | ~6.0 | ~6.0 | — | — | vendor (Qwen report) |
| **Qwen3-ASR-0.6B** | 14.07 | ~9.9 | 9.91 | — | — | vendor (Qwen report) |
| **Qwen3-ASR-1.7B (MLX 8-bit)** | — | **10.52** | — | — | — | independent (Soniqo, on-device) |
| **Qwen3-ASR-1.7B (MLX 4-bit)** | — | **16.35** | — | — | — | independent (Soniqo, on-device) |
| **GigaAM v3 (Sber) — RU SOTA** | **0.9** | — | 4.4 | — | 2.4 / 3.9 | vendor (Sber eval.md) |
| **T-one (telephony)** | — | — | — | — | — | 8.63 on call-center vs Whisper 19.39 |

**Cleanest apples-to-apples we have:** Common Voice (ru) → **Qwen 8.28 vs Whisper 9.84**
(≈1.5 pts better for Qwen — but different CV versions, both imperfectly matched).

**Reading of the evidence:**
- Full-precision Qwen3-ASR-1.7B is **modestly ahead** of Whisper large-v3 on clean Russian
  (both the CV and Fleurs directions agree). Confidence low–moderate: it's vendor-reported and
  the dataset versions don't line up. Qwen's report notably **omits a Whisper Russian baseline**.
- Russian is one of **Whisper's weaker major languages** — well behind its Spanish/French
  (4–6%) and it degrades badly on spontaneous / telephony / far-field Russian (Golos 16–19%,
  OpenSTT phone 27%). This repo's VAD + `no_speech_thold`/`entropy_thold`/`no_context` + dedup
  harness (`core.py`) is exactly the right mitigation for Whisper's silence-hallucination and
  repetition failure modes.

---

## 2. Russian–English code-switching (the critical question)

**There is no reproducible RU-EN code-switching benchmark for either model.** What exists:

**Whisper large-v3 — architecturally hostile to code-switching.** It detects one language from
the first ~30 s and assumes the whole clip is that language. Documented failure modes on mixed
input:
- forces everything into a single language;
- if English is forced it *translates* instead of transcribing; if auto/non-English, it
  **transliterates English into Cyrillic**;
- worst on *rapid* within-sentence alternation — precisely the IT/tech scenario.

**Qwen3-ASR — claims better, unproven for RU-EN.** Qwen markets "high accuracy in language
switching within sentences" and auto language detection, and its decoder-LLM design has no hard
30-s single-language assumption. **But every code-switching claim is qualitative marketing, none
RU-EN-specific** (their demos are EN↔Chinese).

**The one relevant independent RU-EN test** (a Habr practitioner review, 6 months of real IT
dictation, focused on rapid RU/EN switching): Whisper large-v3 remained the practical baseline;
GigaAM v3 / Canary approach it but **mangle English tech terms via transliteration** (e.g.
*"Gemini" → "Jemni"*), which is fatal for library/CLI/IDE vocabulary. **Qwen3-ASR was not
tested**, and no WER was reported (qualitative only).

**The one concrete lever:** Qwen3-ASR accepts a **context / vocabulary prompt** — you can feed
it your English technical terms (framework names, CLI commands, product names). Whisper's
`initial_prompt` is far weaker and inconsistent. For RU-EN tech speech this is the **most
promising single differentiator**, and it's worth testing directly.

**Verdict:** genuinely undetermined. Lean Qwen on architecture + the context-prompt, but treat
it as an **open question that requires your own A/B eval**. Confidence: very low.

---

## 3. Neither is state-of-the-art on Russian

Dedicated Russian models substantially beat *both* Whisper and Qwen on Russian:

1. **GigaAM v3 (Sber)** — current open Russian SOTA. Conformer/RNNT, ~240M params. CV **0.9%**,
   Golos crowd **2.4%** / farfield **3.9%**, Rus. LibriSpeech **4.4%** — roughly **3–8× lower
   WER than Whisper** on clean Russian, now with punctuation/casing. Weak spot: same English-
   transliteration problem on code-switching.
2. **T-one (T-Bank)** — 70M, streaming, telephony-specialized. **8.63%** on call-center audio
   vs Whisper's **19.39%** on the same domain. Best if your audio is phone/noisy/compressed.
3. **NVIDIA Canary / Parakeet** — strong multilingual; competitive on Russian but shares the
   mixed-speech transliteration weakness.

So Whisper and Qwen's advantage is **breadth** (many languages, ecosystem, punctuation) — **not**
Russian accuracy. Independent arbiter to check live:
[Vikhrmodels Russian ASR Leaderboard](https://huggingface.co/spaces/Vikhrmodels/Russian_ASR_Leaderboard).

---

## 4. What adopting Qwen3-ASR would mean *for this repo*

| Factor | Reality |
|---|---|
| **Runtime on Apple Silicon** | ✅ Works — via **MLX** (`qwen3-asr-mlx` / `mlx-audio`), *not* whisper.cpp/GGUF. Community port, Apache-2.0 code, validated vs PyTorch. |
| **Footprint (1.7B)** | ~**3.4 GB** bf16 · ~**2.46 GB** 8-bit unified memory. Fine on 16 GB+ Macs. |
| **Speed** | Fast: RTF ~**0.08×** on M4 Pro (0.6B). 1.7B slower but still comfortably real-time. |
| **Not a drop-in** | Different architecture → replaces, doesn't extend, the `pywhispercpp` path. Two engines to maintain. |
| **Timestamps** ⚠️ | The simple MLX package returns **text + language only — no per-segment/word timestamps**. This repo relies on segment `t0/t1` for SRT and diarization alignment. You'd need the separate **Qwen3-ForcedAligner-0.6B**, or fall back to VAD-chunk-level timing (coarser). Real integration cost. |
| **Quantization hurts Russian** ⚠️ | On-device you'd likely quantize. Soniqo: Fleurs-ru **bf16 ~6% → 8-bit 10.52% → 4-bit 16.35%**. At 8-bit Qwen ≈ Whisper on Russian; at 4-bit it's **worse**. Use **bf16** to keep the edge. |
| **Dependencies** | `torch`/`torchaudio` are *already* here (pyannote diarization), so PyTorch isn't net-new; the MLX route avoids adding `transformers`/CUDA entirely. |
| **Context prompt** | ✅ Qwen supports domain-vocabulary prompting (RU-EN tech terms). Whisper effectively does not. |

**Net:** Qwen3-ASR is genuinely runnable here, but it's a **second engine**, loses native
timestamps, and its Russian advantage largely survives only at **bf16** — at the 8-bit you'd
realistically ship, on-device Russian is ~a wash with Whisper.

---

## 5. Recommendation

1. **Don't rip out Whisper.** It's integrated, Metal-accelerated, gives timestamps, and the
   VAD/dedup harness already tames its failure modes. Keep it the multilingual default.
2. **If Russian quality is the priority** → the highest-leverage move is **not** Qwen but adding
   **GigaAM v3** as a Russian-specific engine (3–8× better on clean Russian). Trade-off: Russian-
   only, and it also transliterates English on code-switching.
3. **If Russian–English code-switching is the priority** → run a real **A/B eval** on *your own*
   mixed IT-speech clips. Compare: Whisper large-v3 (current) vs **Qwen3-ASR-1.7B bf16 via MLX,
   with a context prompt listing your English tech vocabulary**. This prompt lever is the main
   reason to expect Qwen to win here — verify it before committing.
4. **Add Qwen as an optional engine, not a replacement**, if the A/B favors it — behind the
   existing `model_size` selection, using the ForcedAligner for timestamps.

### How to A/B in ~30 min
```bash
pip install -U mlx-audio                      # Apple Silicon
python -m mlx_audio.stt.generate \
  --model mlx-community/Qwen3-ASR-1.7B-8bit \  # or -bf16 for best Russian
  --audio your_ru_en_clip.wav
# Compare against the current Whisper output on the SAME clips, by hand,
# scoring: English tech terms kept vs transliterated, RU accuracy, punctuation.
```

---

## Sources

**Qwen3-ASR (primary):**
- Technical report — https://arxiv.org/html/2601.21337v1
- GitHub — https://github.com/QwenLM/Qwen3-ASR
- Model card 1.7B — https://huggingface.co/Qwen/Qwen3-ASR-1.7B · 0.6B — https://huggingface.co/Qwen/Qwen3-ASR-0.6B

**Apple Silicon / MLX:**
- `qwen3-asr-mlx` (PyPI) — https://pypi.org/project/qwen3-asr-mlx/
- MLX port (moona3k) — https://github.com/moona3k/mlx-qwen3-asr
- MLX 8-bit weights — https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit
- On-device RU benchmark (Soniqo) — https://soniqo.audio/benchmarks

**Whisper Russian (independent):**
- `antony66/whisper-large-v3-russian` (CV17 baseline 9.84%) — https://huggingface.co/antony66/whisper-large-v3-russian
- large-v3 release notes — https://github.com/openai/whisper/discussions/1762
- Code-switching failure modes — https://github.com/openai/whisper/discussions/49

**Russian SOTA / context:**
- GigaAM v3 eval — https://github.com/salute-developers/GigaAM/blob/main/evaluation.md
- T-one — https://github.com/voicekit-team/T-one · https://habr.com/ru/companies/tbank/articles/929850/
- Russian ASR leaderboard — https://huggingface.co/spaces/Vikhrmodels/Russian_ASR_Leaderboard
- RU-EN practitioner review (Habr) — https://news.hamidun.com/en/news/7078/wisprflow-whisper-and-gigaam-who-recognizes-russian-english-
