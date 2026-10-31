# Text Summarizer Pro v5.0

A single-file desktop app for summarizing documents with **Ollama** (AI, local)
or a **fast offline extractive summarizer** (no model required). Drag-and-drop,
multi-format, keyword extraction, and export to `.txt` / `.docx`.

- **No cloud.** Everything runs against `localhost:11434` (Ollama) or in-process.
- **No build step.** One `.py` file, auto-installs its own dependencies.
- **No secrets.** Nothing to configure; the app detects Ollama and picks a model.

---

## What it does

| Feature | Notes |
|---|---|
| Load documents | `.txt`, `.pdf`, `.docx`, `.doc`, `.odt`, `.rtf`, and `http(s)://` URLs |
| Load text | Paste from clipboard |
| Drag & drop | Requires `tkinterdnd2` (auto-installed) |
| AI summarization | Via Ollama; streaming, no-hallucination prompt |
| Offline summarization | Extractive, keyword-weighted (no model needed) |
| 8 summary styles | Executive, Bullet, Technical, Plain English, Student, Report, Key Findings, Action Items |
| Keyword extraction | Top-10 weighted keywords shown under the summary |
| Export | `.txt` (UTF-8) or `.docx` (requires `python-docx`) |
| GPU awareness | Warns if Ollama spills the model to CPU |

---

## Requirements

- **Python 3.8+** (developed against 3.11)
- **Tkinter** (bundled with the official Windows/macOS installers; on Linux: `sudo apt install python3-tk`)
- **Ollama** (optional, for AI mode) — <https://ollama.com>
  - Recommended: `ollama pull llama3.2` (3B, ~2 GB — fits a 6 GB card)
  - Optional quality tier: `ollama pull mistral` (7B, ~4.4 GB)

The app **auto-installs** these on first launch into your user site-packages
(`pip install --user`):

`requests`, `PyPDF2`, `python-docx`, `odfpy`, `striprtf`, `tkinterdnd2`

If auto-install fails (offline machine, locked-down Python), install them manually:

```bash
pip install --user requests PyPDF2 python-docx odfpy striprtf tkinterdnd2
```

---

## Running it

### Windows — double-click

Double-click `Text Summarizer Pro v5.0.py`. That's it.

> **Heads-up:** if your `.py` files are associated with `pythonw.exe` (the
> windowless variant), you will **not see** the GPU/VRAM warning banner, the
> dependency-install messages, or debug tracebacks — they go to a console that
> doesn't exist. If you want those messages, either re-associate `.py` with
> `python.exe`, or drop this `run_summarizer.bat` next to the app and
> double-click that instead:
>
> ```bat
> @echo off
> python "%~dp0Text Summarizer Pro v5.0.py"
> pause
> ```

### Windows / macOS / Linux — terminal

```bash
python "Text Summarizer Pro v5.0.py"
```

### First launch

The app will:

1. Check for missing dependencies and `pip install` them if needed.
2. Probe `http://localhost:11434/api/tags` for Ollama.
3. Print a startup status line to the status bar (and console, if visible).

If Ollama isn't running, the app silently falls back to **Offline Mode** — the
summarizer still works, just extractively.

---

## GPU residency (the 6 GB card problem)

Ollama decides how much context to allocate based on `num_ctx`. If it guesses
too high, a 7B model spills to CPU and generations slow down by roughly **10×**.

**This app hard-codes `num_ctx = 4096`** on every Ollama request, which is the
single most important setting for keeping a 7B Q4 model on a 6 GB GPU.

After the **first successful** generation, the app queries `/api/ps` and
reports the VRAM residency ratio. If it's below 50 %, you'll see:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: running mostly on CPU (VRAM 2.1GB / 4.4GB, ratio 48%) - expect
~10x slowdown. Try NUM_CTX=2048 or use llama3.2.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

If you see that, either:

- Switch to the **Fast** model (`llama3.2`, ~2 GB) in the sidebar, **or**
- Lower `DEFAULT_NUM_CTX` to `2048` at the top of the `.py` file, **or**
- Close other GPU-consuming apps (Chrome, other Ollama models).

> The VRAM warning is printed to `stderr`. If you launched via double-click and
> your `.py` is associated with `pythonw.exe`, you won't see it — use the
> `.bat` launcher from the previous section, or run from a terminal.

---

## Model selection

The sidebar exposes **Fast** and **Quality** model modes (only visible when
Ollama is available):

| Mode | Default model | Size (Q4) | Fits 6 GB? |
|---|---|---|---|
| **Fast** (default) | `llama3.2` | ~2.0 GB | ✅ comfortably |
| **Quality** | `mistral` | ~4.4 GB | ✅ at `num_ctx=4096` |

Model names are resolved by substring: any installed model whose name contains
`llama3.2`, `llama3`, or `llama` (in that order) counts as "Fast"; any whose
name contains `mistral` or `mixtral` counts as "Quality." If no Quality-tier
model is installed, the radio button snaps back to Fast.

**You cannot switch models while a summarization is running.** The status bar
will say `Cannot switch model while summarizing` and the change is deferred
until the run finishes.

---

## Architecture

Single file, ~1,700 lines. Top-level classes, in dependency order:

| Class | Responsibility |
|---|---|
| `GPUMonitor` | One-shot `/api/ps` VRAM check after first successful generation |
| `TextCleaner` | Line-level noise removal (page numbers, figure captions, headers) |
| `KeywordExtractor` | Stopword-aware frequency ranking |
| `OllamaSummarizer` | Streaming + non-streaming `/api/generate`, prompt construction |
| `FastSummarizer` | Extractive fallback: sentence scoring + selection |
| `FileHandler` | Per-format readers (PDF, DOCX, ODT, RTF, DOC, URL) |
| `ModernSummarizerApp` | Tkinter UI, threading, status queue, export |
| `main()` | Entry point; picks `TkinterDnD.Tk()` if DnD is available |

### Threading model

The UI thread never blocks. `summarize_text()` dispatches a `threading.Thread`
which:

1. Snapshots the model name (**preventing mid-run model swaps**).
2. Attempts AI summarization (streaming or non-streaming).
3. Fires the one-shot VRAM check if the AI call succeeded.
4. Falls back to `FastSummarizer` if the AI output is missing or **both short
   (< 40 words) and shorter than the extractive fallback**.
5. Posts the result back to the UI via `root.after(0, ...)`.

Status messages flow through a `Queue` drained every 100 ms by
`process_log_queue()` — never `self.root.config` directly from a worker.

### Streaming

Streaming is on by default (checkbox in sidebar). During a stream:

- The status bar updates with a live word count.
- If Ollama goes silent for > 15 s (typically a cold model load), the status
  bar shows `AI still generating (slow first token)…`.
- A late-arriving progress callback from a finished run cannot overwrite the
  final `Summary complete` status — the callback guards on `is_processing` at
  execution time, not just at enqueue time.

---

## Known limitations

- **`TextCleaner` applies PDF-extraction repair rules to all formats.** The
  `fixes` dict in `clean_text()` (e.g. `r't he': 'the'`) exists to repair
  PDFs that split words across lines. It runs on `.txt` and `.docx` input too,
  where it's a no-op for well-formed prose but can mangle text that
  legitimately contains those sequences. A proper fix would tag PDF-sourced
  text; deferred to v5.1.
- **No cancel button.** Streaming can be interrupted only by killing the
  process or letting Ollama finish. A `threading.Event`-based cancel is planned
  for v5.1.
- **`.doc` files** are read with a best-effort binary-strip fallback. For
  reliable results, convert to `.docx` first.
- **URL loading** is HTML-tag-stripping, not readability-parsing. Works for
  articles; will include nav/footer noise on many sites.
- **No content-length pre-check.** Feeding a 500 KB PDF to a 3B model will
  truncate the source to 4,000 characters in the prompt — the summary reflects
  only the first ~600 words of your document.

---

## Version history

### v5.0 — Modern Edition with Drag & Drop

- Modern dark UI, drag-and-drop file loading
- Fast/Quality model selection
- Streaming by default with live status
- `num_ctx=4096` hard-coded for GPU residency
- One-shot VRAM residency check with loud CPU-spill warning
- No-invented-facts guardrail in every prompt
- Honest length caption under the Length dropdown
- Per-run timing logged to status bar
- Debug checkbox for full tracebacks

### v4.2 and earlier

Extractive summarization only; static UI; no model selection.

---

## Patches applied to this release

This release includes 17 patch fixes applied post-v5.0. If you forked from an
earlier snapshot, see `patch_summarizer.py` and its `.bak` companion in the
repo history. Highlights:

| # | Fix |
|---|---|
| 1 | `num_predict` floor matches the user-visible target word count |
| 1b | Length caption mirrors `num_predict`'s floor |
| 2 | VRAM check fires on first **successful** generation, not first attempt |
| 3a–e | Model selection frozen at dispatch; mid-run toggles rejected |
| 4 | Streaming status callback can't clobber the final status |
| 5 | Export refuses `.docx` when `python-docx` is unavailable |
| 6 | Drag-and-drop uses `tk.splitlist()` for paths with spaces |
| 7a–b | `TextCleaner` stops dropping legitimate prose |
| 9a–b | Ollama model discovery handles both `name` and `model` keys |
| 10a–b | Streaming timeout raised to 600 s; heartbeat during cold load |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
