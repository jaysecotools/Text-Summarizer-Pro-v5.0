"""
Text Summarizer Pro v5.0 - Modern Edition with Drag & Drop
Comprehensive document intelligence toolkit with modern UI

WHAT CHANGED (v4.2 -> v5.0)
===========================

Applicable fixes imported from the sibling "research report generator" app:

  [GPU] Added GPUMonitor class:
        - check_startup():  hits /api/tags (optimistic "GPU present, idle").
        - post_launch_status():  hits /api/ps AFTER the first generation and
          reports the VRAM ratio. If ratio < 0.5, prints a loud warning:
          "running mostly on CPU - expect ~10x slowdown."
        - post_launch_status() is called EXACTLY ONCE per session.

  [CTX] num_ctx=4096 is now hard-coded into EVERY Ollama request's options
        block. This is the single most important setting for staying on GPU
        with a 6GB RTX 2060. Without it Ollama picks a larger default and
        mistral (7B Q4) spills to CPU.

  [MODEL] Added fast/quality model selection (was implicit auto-pick):
        - Fast (default): llama3.2 (3B Q4, ~2GB - fits comfortably).
        - Quality:         mistral   (7B Q4, ~4.4GB - fits at num_ctx=4096).
        Both are passed to Ollama by explicit name; if the requested model
        isn't installed, we fall back to whatever llama-family model IS
        installed and say so loudly.

  [STREAM] Streaming is now the DEFAULT for Ollama calls; a GUI checkbox
        disables it. Streaming mode shows token-by-token progress in the
        status bar so the app doesn't look frozen during a long summary.

  [NUM_PREDICT] num_predict is now derived from the target summary length
        (roughly 2x the target word count, clamped to [150, 700]) instead
        of a flat 500. 500 was already close to right for the 300-word
        default; this makes non-default lengths honest.

  [GUARD] Added a "no invented names/statistics" rule to the summarization
        prompt. This is a summarizer, not a generator, so hallucination risk
        is lower - but the rule is cheap and stops the "According to Dr. Jane
        Smith's 2019 study..." class of invention entirely.

  [TIMING] Per-run timing is logged: "OK 182w in 4.1s (44.4 w/s)" - printed
        to the status log so regressions are visible at a glance.

  [FALLBACK] The AI-vs-fast fallback selection is tightened. Previously an
        AI summary that was merely shorter than the extractive fallback was
        discarded, which could throw away a perfectly good 120-word AI
        summary in favour of a 300-word keyword salad. Now the fallback
        only wins when AI output is BOTH short (< 40 words) AND shorter
        than the fallback.

  [FLUSH] export_summary() now calls f.flush() explicitly after the write,
        so a crash mid-export leaves a truncated-but-readable file rather
        than an empty one.

  [DEBUG] Added a "Debug" checkbox. When on, exceptions print full
        tracebacks to stderr instead of being silently swallowed by the
        summary thread.

  [HONEST] The Length dropdown now shows a realistic word-count caption
        under it (e.g. "~75 words for this input") so users aren't misled
        into expecting 500 words from a 25% ratio on a 200-word input.

NOT APPLICABLE to this app (documented so future-me doesn't re-add them):
  - Section-by-section generation, expansion passes, token floor
  - References / citations / source-title extraction
  - Per-section summaries and finding rotation
  - --resume of NN_sectionname.md files
  (This app summarizes a single document; it does not generate one.)

Features: Drag & Drop, Multiple Formats, AI/Offline Summarization, Keywords, Export
"""

# ================= AUTO-INSTALL DEPENDENCIES =================
import subprocess
import sys
import os
import importlib
import webbrowser
from pathlib import Path

def ensure_dependencies():
    """Check and auto-install required dependencies"""
    required = {
        'requests': 'requests>=2.31.0',
        'PyPDF2': 'PyPDF2>=3.0.0',
        'docx': 'python-docx>=1.0.0',
        'odf': 'odfpy>=1.4.0',
        'striprtf': 'striprtf>=0.0.19',
        'tkinterdnd2': 'tkinterdnd2>=0.3.0',
    }
    missing = []
    for module, package in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Installing missing dependencies...")
        print(f"   Packages: {', '.join(missing)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--user", "--upgrade", *missing
            ])
            for module in required.keys():
                importlib.import_module(module)
            print("Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            print("\nPlease install manually with:")
            print(f"pip install {' '.join(missing)}")
            input("\nPress Enter after installing dependencies...")
            for module in required.keys():
                try:
                    importlib.import_module(module)
                except ImportError:
                    print(f"Still missing: {module}")
                    sys.exit(1)
            return True
    return True

if not ensure_dependencies():
    print("Failed to install dependencies.")
    input("Press Enter to exit...")
    sys.exit(1)

# ================= IMPORTS =================
import requests
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
import time
import threading
import traceback
from queue import Queue
import re
from collections import Counter
import math

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    from odf import text, teletype
    from odf.opendocument import load
    ODT_AVAILABLE = True
except Exception:
    ODT_AVAILABLE = False

try:
    from striprtf.striprtf import rtf_to_text
    RTF_AVAILABLE = True
except Exception:
    RTF_AVAILABLE = False

# ================= CONFIG =================
OLLAMA_BASE = "http://localhost:11434"
DEFAULT_NUM_CTX = 4096          # <-- THE most important setting for GPU residency
FAST_MODEL_HINTS = ("llama3.2", "llama3", "llama")
QUALITY_MODEL_HINTS = ("mistral", "mixtral")

# ================= GPU MONITOR =================
class GPUMonitor:
    """
    Tells the user whether Ollama is keeping the model on GPU or spilling to CPU.
    - Startup check is optimistic: /api/tags proves Ollama is up, but says
      nothing about residency.
    - post_launch_status() is called once, AFTER the first real generation,
      and reports size_vram / size. A low ratio means CPU spill.
    """

    def __init__(self, base=OLLAMA_BASE, model=None, log=print):
        self.base = base
        self.model = model
        self.log = log
        self._post_launch_done = False

    def check_startup(self):
        """Optimistic check. Returns True if Ollama is reachable."""
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=3)
            if r.status_code != 200:
                self.log("Ollama not reachable at startup.")
                return False
            models = [
                m.get("name") or m.get("model") or ""
                for m in r.json().get("models", [])
            ]
            if not models:
                self.log("Ollama is up but has no models installed.")
                return False
            if self.model and not any(self.model in m for m in models):
                self.log(
                    f"GPU present, idle - but requested model '{self.model}' "
                    f"is not installed. Available: {', '.join(models[:5])}"
                )
            else:
                self.log(f"GPU present, idle (model: {self.model or 'auto'}).")
            return True
        except Exception:
            self.log("Ollama not reachable at startup.")
            return False

    def post_launch_status(self):
        """Call exactly once, after the first generation completes."""
        if self._post_launch_done:
            return
        self._post_launch_done = True
        try:
            r = requests.get(f"{self.base}/api/ps", timeout=3)
            if r.status_code != 200:
                return
            entries = r.json().get("models", [])
            if not entries:
                self.log("VRAM check: no loaded models (generation may have released).")
                return
            # Pick the entry for our model if we can identify it.
            entry = None
            for e in entries:
                name = e.get("name") or e.get("model") or ""
                if self.model and self.model in name:
                    entry = e
                    break
            if entry is None:
                entry = entries[0]
            size = entry.get("size") or 0
            vram = entry.get("size_vram") or 0
            if not size:
                return
            ratio = vram / size
            if ratio < 0.5:
                msg = (
                    f"WARNING: running mostly on CPU (VRAM {vram/1e9:.1f}GB / "
                    f"{size/1e9:.1f}GB, ratio {ratio:.0%}) - expect ~10x slowdown. "
                    f"Try NUM_CTX=2048 or use llama3.2."
                )
                self.log(msg)
                print("\n" + "!" * 70)
                print(msg)
                print("!" * 70 + "\n")
            else:
                self.log(
                    f"VRAM check: {vram/1e9:.1f}GB / {size/1e9:.1f}GB "
                    f"({ratio:.0%}) - model is resident on GPU."
                )
        except Exception as e:
            self.log(f"VRAM check failed: {e}")

# ================= OLLAMA DETECTION =================
def check_ollama_available():
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            if models:
                return True, [
                    m.get('name') or m.get('model') or ''
                    for m in models
                ]
        return False, []
    except Exception:
        return False, []

# ================= SMART TEXT CLEANER =================
class TextCleaner:
    @staticmethod
    def clean_text(text):
        if not text:
            return ""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^[\d\s]+$', line):
                continue
            if re.search(r'={5,}\s*Page\s*\d+\s*={5,}', line, re.IGNORECASE):
                continue
            if re.match(r'^(Figure|Table|Fig\.|Fig)\s*\d+[:.]?\s*$', line, re.IGNORECASE):
                continue
            if re.match(r'^Appendix\s*[A-Za-z0-9]+\s*$', line, re.IGNORECASE):
                continue
            if re.match(r'^[\d.]+\s+[A-Za-z]+\s+[.\s]+$', line):
                continue
            if line.count('.') > len(line) * 0.3:
                continue
            # Only drop lines that are almost entirely a single repeated
            # token AND contain no sentence punctuation. The old rule also
            # ate real prose like "the the the is is a" and section headers.
            tokens = line.split()
            if (len(tokens) > 20
                    and len(set(tokens)) < 3
                    and not re.search(r'[.!?]', line)):
                continue
            digit_ratio = sum(c.isdigit() for c in line) / len(line) if len(line) > 0 else 0
            if digit_ratio > 0.3 and len(line) < 30:
                continue
            if re.search(r'Page\s+\d+', line, re.IGNORECASE):
                continue
            # Tabs are a strong signal of a table/CSV row; double-space runs
            # are only suspicious when the line is long AND non-prose.
            if line.count('\t') > 2:
                continue
            if line.count('  ') > 5 and not re.search(r'[.!?]', line):
                continue
            cleaned_lines.append(line)
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'Page\s+\d+\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'Figure\s+\d+[:.]?\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'Table\s+\d+[:.]?\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'Fig\.?\s*\d+[:.]?\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'[.\s]{10,}', ' ', cleaned_text)
        fixes = {
            r'dra\s+stically': 'drastically', r'us\s+e\s+': 'use ',
            r'ye\s+ar': 'year', r's\s+outh': 'south', r'n\s+orth': 'north',
            r'e\s+ast': 'east', r'w\s+est': 'west', r'de\s+crease': 'decrease',
            r'in\s+crease': 'increase', r'C\s+reek': 'Creek', r'fo\s+llow': 'follow',
            r'fo\s+llowing': 'following', r'ov\s+er': 'over', r'und\s+er': 'under',
            r'bet\s+ween': 'between', r'bet\s+ter': 'better', r'l ocation': 'location',
            r'wh ich': 'which', r't he': 'the', r'\s+-\s+': ' - ',
        }
        for pattern, replacement in fixes.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()

# ================= KEYWORD EXTRACTOR =================
class KeywordExtractor:
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at',
            'to', 'by', 'with', 'without', 'from', 'into', 'through', 'during',
            'including', 'until', 'against', 'among', 'upon', 'about', 'after',
            'before', 'above', 'below', 'between', 'under', 'over', 'of', 'off',
            'in', 'out', 'up', 'down', 'so', 'too', 'very', 'just', 'more',
            'most', 'some', 'any', 'each', 'few', 'other', 'such', 'than',
            'then', 'these', 'they', 'this', 'those', 'though', 'throughout',
            'till', 'unto', 'when', 'where', 'wherever', 'while'
        }

    def extract_keywords(self, text, top_n=15):
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        words = text.lower().split()
        filtered = [w for w in words if w not in self.stop_words and len(w) > 2]
        if not filtered:
            return []
        freq = Counter(filtered)
        top_words = freq.most_common(top_n)
        if top_words:
            max_count = top_words[0][1]
            return [(word, round((count / max_count) * 100)) for word, count in top_words]
        return []

# ================= OLLAMA SUMMARIZER =================
class OllamaSummarizer:
    """
    AI summarization via Ollama.
    - num_ctx=4096 hard-coded: critical for GPU residency on 6GB cards.
    - Streaming is the default; pass stream=False for quiet mode.
    - A no-invented-facts guardrail is appended to every prompt.
    """

    def __init__(self, model="llama3.2:latest", stream=True, debug=False,
                 gpu_monitor=None, progress_cb=None):
        self.model = model
        self.base_url = f"{OLLAMA_BASE}/api/generate"
        self.stream = stream
        self.debug = debug
        self.gpu_monitor = gpu_monitor
        self.progress_cb = progress_cb  # callable(str) for live status

    def _num_predict_for(self, target_words):
        # ~2 tokens per word is a safe upper bound for these 3B/7B models.
        # The lower bound tracks the user-visible target so the model is
        # never silently allowed to write far more than the caption says.
        floor = max(150, int(target_words * 2))
        return max(150, min(700, floor))

    def summarize(self, text, style="executive", max_length=300):
        try:
            style_instructions = {
                "executive": "Create a professional executive summary with key findings and recommendations.",
                "bullet": "Create a bullet point summary with 5-8 key points.",
                "technical": "Create a detailed technical summary with specific data and methodology.",
                "plain": "Create a simple, clear summary using everyday language.",
                "student": "Create an educational summary highlighting key concepts.",
                "report": "Create a formal report-style summary with sections.",
                "key_findings": "Focus only on the most important findings and data.",
                "action_items": "List specific actions and next steps."
            }
            style_text = style_instructions.get(style, style_instructions["executive"])

            prompt = (
                f"{style_text}\n\n"
                f"Rules:\n"
                f"- Use ONLY facts, names, numbers, and quotes present in the source text.\n"
                f"- Do NOT invent people, companies, statistics, studies, or dates.\n"
                f"- If a claim is not supported by the source, omit it.\n"
                f"- Keep the summary to {max_length} words or less.\n\n"
                f"Source text:\n\n{text[:4000]}\n\nSummary:"
            )

            options = {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": DEFAULT_NUM_CTX,          # <-- critical
                "num_predict": self._num_predict_for(max_length),
            }

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": self.stream,
                "options": options,
            }

            if not self.stream:
                response = requests.post(self.base_url, json=payload, timeout=120)
                if response.status_code != 200:
                    print(f"Ollama error: {response.status_code}")
                    return None
                data = response.json()
                summary = (data.get('response') or '').strip()
                return self._clean(summary)

            # ---- streaming mode ----
            chunks = []
            # 600s covers cold-loading a 7B model on a slow disk; the
            # per-line heartbeat below gives the user visible progress
            # during the long silence before the first token.
            with requests.post(self.base_url, json=payload,
                               stream=True, timeout=600) as r:
                if r.status_code != 200:
                    print(f"Ollama error: {r.status_code}")
                    return None
                last_beat = time.time()
                for line in r.iter_lines():
                    if not line:
                        # Blank line = keepalive; if we've gone >15s
                        # without a real chunk, tell the user we're alive.
                        if self.progress_cb and time.time() - last_beat > 15:
                            self.progress_cb("AI still generating (slow first token)\u2026")
                            last_beat = time.time()
                        continue
                    last_beat = time.time()
                    try:
                        obj = json.loads(line.decode('utf-8'))
                    except Exception:
                        continue
                    piece = obj.get('response') or ''
                    if piece:
                        chunks.append(piece)
                        if self.progress_cb:
                            self.progress_cb(
                                f"AI streaming... {len(''.join(chunks).split())} words"
                            )
                    if obj.get('done'):
                        break
            return self._clean(''.join(chunks))

        except Exception as e:
            if self.debug:
                traceback.print_exc()
            else:
                print(f"Ollama error: {e}")
            return None

    @staticmethod
    def _clean(summary):
        summary = re.sub(r'^Here is.*summary:', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'^Here\'s.*summary:', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'^Summary:', '', summary, flags=re.IGNORECASE)
        summary = summary.strip()
        return summary if len(summary) >= 10 else None

# ================= FAST SUMMARIZER =================
class FastSummarizer:
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at',
            'to', 'by', 'with', 'without', 'from', 'into', 'through', 'during',
            'including', 'until', 'against', 'among', 'upon', 'about', 'after',
            'before', 'above', 'below', 'between', 'under', 'over', 'of', 'off',
            'in', 'out', 'up', 'down', 'so', 'too', 'very', 'just', 'more',
            'most', 'some', 'any', 'each', 'few', 'other', 'such', 'than',
            'then', 'these', 'they', 'this', 'those', 'though', 'throughout',
            'till', 'unto', 'when', 'where', 'wherever', 'while'
        }
        self.important_indicators = {
            'objective', 'purpose', 'goal', 'aim', 'target',
            'summary', 'overview', 'conclusion', 'recommendation', 'finding',
            'result', 'outcome', 'impact', 'benefit', 'advantage', 'improvement',
            'issue', 'problem', 'challenge', 'risk', 'threat', 'concern',
            'strategy', 'approach', 'method', 'process', 'procedure',
            'important', 'critical', 'essential', 'key', 'primary', 'main'
        }

    def summarize(self, text, ratio=0.25, max_sentences=8):
        if not text or len(text.strip()) < 50:
            return "Text is too short to summarize."
        text = TextCleaner.clean_text(text)
        if len(text) < 50:
            return "After cleaning, text is too short to summarize."
        sentences = self._split_sentences(text)
        if len(sentences) <= 3:
            return text
        sentence_scores = self._score_sentences(sentences)
        for i, sentence in enumerate(sentences):
            lower_sentence = sentence.lower()
            boost = sum(1 for word in self.important_indicators if word in lower_sentence)
            sentence_scores[i] = sentence_scores[i] * (1 + boost * 0.15)
        num_sentences = min(max(3, int(len(sentences) * ratio)), max_sentences)
        num_sentences = min(num_sentences, len(sentences))
        sorted_indices = sorted(
            range(len(sentence_scores)),
            key=lambda i: sentence_scores[i], reverse=True
        )[:num_sentences]
        selected_indices = sorted(sorted_indices)
        summary = ' '.join([sentences[i] for i in selected_indices])
        return re.sub(r'\s+', ' ', summary).strip()

    def _split_sentences(self, text):
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _score_sentences(self, sentences):
        word_freq = Counter()
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
            for word in words:
                if word not in self.stop_words and len(word) > 2:
                    word_freq[word] += 1
        if word_freq:
            max_freq = max(word_freq.values())
            if max_freq > 0:
                for word in word_freq:
                    word_freq[word] = word_freq[word] / max_freq
        scores = []
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
            if not words:
                scores.append(0)
                continue
            score = sum(word_freq.get(word, 0) for word in words)
            scores.append(score / (len(words) + 1))
        return scores

# ================= FILE HANDLERS =================
class FileHandler:
    @staticmethod
    def read_file(filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.txt':
            return FileHandler._read_txt(filepath)
        elif ext == '.pdf' and PDF_AVAILABLE:
            return FileHandler._read_pdf_safe(filepath)
        elif ext == '.docx' and DOCX_AVAILABLE:
            return FileHandler._read_docx_safe(filepath)
        elif ext == '.doc':
            return FileHandler._read_doc(filepath)
        elif ext == '.odt' and ODT_AVAILABLE:
            return FileHandler._read_odt_safe(filepath)
        elif ext == '.rtf' and RTF_AVAILABLE:
            return FileHandler._read_rtf_safe(filepath)
        elif filepath.startswith('http://') or filepath.startswith('https://'):
            return FileHandler._read_url_safe(filepath)
        else:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception:
                try:
                    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                        return f.read()
                except Exception:
                    return None

    @staticmethod
    def _read_txt(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                return f.read()

    @staticmethod
    def _read_pdf_safe(filepath):
        try:
            text = ''
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            if text:
                text = TextCleaner.clean_text(text)
            return text if text else None
        except Exception as e:
            raise Exception(f"Could not read PDF: {str(e)[:100]}")

    @staticmethod
    def _read_docx_safe(filepath):
        try:
            doc = Document(filepath)
            text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
            if text:
                text = TextCleaner.clean_text(text)
            return text if text else None
        except Exception as e:
            raise Exception(f"Could not read DOCX: {str(e)[:100]}")

    @staticmethod
    def _read_doc(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            text = re.sub(r'[^\x20-\x7E\n\r\t]+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            if text.strip():
                return TextCleaner.clean_text(text)
            with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                text = f.read()
            text = re.sub(r'[^\x20-\x7E\n\r\t]+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            if text.strip():
                return TextCleaner.clean_text(text)
            return None
        except Exception:
            raise Exception("Could not read DOC file. Please convert to DOCX or TXT.")

    @staticmethod
    def _read_odt_safe(filepath):
        try:
            doc = load(filepath)
            text = ''
            for paragraph in doc.getElementsByType(text.P):
                text += teletype.extractText(paragraph) + '\n'
            if text:
                text = TextCleaner.clean_text(text)
            return text if text else None
        except Exception as e:
            raise Exception(f"Could not read ODT: {str(e)[:100]}")

    @staticmethod
    def _read_rtf_safe(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
            text = rtf_to_text(rtf_content)
            if text:
                text = TextCleaner.clean_text(text)
            return text if text else None
        except Exception as e:
            raise Exception(f"Could not read RTF: {str(e)[:100]}")

    @staticmethod
    def _read_url_safe(url):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            text = re.sub(r'<[^>]+>', ' ', response.text)
            text = re.sub(r'\s+', ' ', text)
            text = TextCleaner.clean_text(text)
            return text.strip()
        except Exception as e:
            raise Exception(f"Could not load URL: {str(e)[:100]}")

# ================= MODERN UI APPLICATION =================
class ModernSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Summarizer Pro v5.0")
        self.root.geometry("1300x880")
        self.root.configure(bg='#0a0a0a')
        self.root.minsize(1000, 680)

        self.current_text = ""
        self.summary = ""
        self.is_processing = False
        self.log_queue = Queue()
        self.ollama_available = False
        self.ollama_models = []
        self.current_ollama_model = None
        self.current_file = None
        self.debug_mode = False
        self.stream_enabled = True
        self.gpu_warned = False

        self.colors = {
            'bg_dark': '#0a0a0a', 'bg_medium': '#1a1a1a', 'bg_light': '#2a2a2a',
            'bg_card': '#1e1e1e', 'bg_hover': '#2d2d2d',
            'text_primary': '#ffffff', 'text_secondary': '#aaaaaa',
            'text_muted': '#666666', 'accent': '#6c5ce7',
            'accent_green': '#00b894', 'accent_blue': '#0984e3',
            'accent_orange': '#fdcb6e', 'accent_red': '#e17055',
            'border': '#333333',
        }

        # GPU monitor
        self.gpu_monitor = GPUMonitor(
            base=OLLAMA_BASE, model=None,
            log=lambda m: self.log(m),
        )

        self._check_ollama()
        self.gpu_monitor.check_startup()

        self.setup_ui()
        self.process_log_queue()

        self.log("Text Summarizer Pro v5.0 started")
        self.log("Supports: TXT, PDF, DOCX, DOC, ODT, RTF, and URLs")
        if DND_AVAILABLE:
            self.log("Drag and drop enabled")
        else:
            self.log("Drag and drop disabled - install tkinterdnd2")
        if self.ollama_available:
            self.log(f"AI: Ollama available with model: {self.current_ollama_model}")
        else:
            self.log("Using fast offline summarizer")

    # ------------------------------------------------------------------
    # Ollama / model selection
    # ------------------------------------------------------------------
    def _check_ollama(self):
        """Check if Ollama is available and pick a sensible default model."""
        available, models = check_ollama_available()

        if not available:
            self.ollama_available = False
            return

        self.ollama_models = models

        # Prefer a fast model by default (llama3.2 fits a 6GB card).
        selected = None
        for hint in FAST_MODEL_HINTS:
            for model in models:
                if hint in model.lower():
                    selected = model
                    break
            if selected:
                break

        # Fallback: any llama-family model, then first available.
        if not selected:
            for model in models:
                if "llama" in model.lower():
                    selected = model
                    break
        if not selected and models:
            selected = models[0]

        if selected:
            self.current_ollama_model = selected
            self.ollama_available = True
            # Let the GPU monitor know which model we'll be running.
            self.gpu_monitor.model = selected
            print(f"Using Ollama model: {selected}")
        else:
            self.ollama_available = False

    def _pick_quality_model(self):
        """Return a quality-tier model name if installed, else None."""
        for hint in QUALITY_MODEL_HINTS:
            for model in self.ollama_models:
                if hint in model.lower():
                    return model
        return None

    def _on_model_mode_change(self):
        """Called when the Fast/Quality radio buttons change."""
        if not self.ollama_available:
            return
        if self.is_processing:
            # Refuse to swap the model under an in-flight request; the
            # worker thread already captured the old model name.
            self.log("Cannot switch model while summarizing")
            return
        mode = self.model_mode_var.get()
        if mode == "Quality":
            quality = self._pick_quality_model()
            if quality:
                self.current_ollama_model = quality
                self.gpu_monitor.model = quality
                self.log(f"Model switched to quality: {quality}")
            else:
                self.log("Quality model (mistral) not installed - staying on fast model")
                self.model_mode_var.set("Fast")
        else:
            # Re-run fast-model detection.
            self._check_ollama()
            if self.current_ollama_model:
                self.log(f"Model switched to fast: {self.current_ollama_model}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_ui(self):
        """Setup the modern UI."""
        self.create_header()

        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=15, pady=10)

        sidebar = tk.Frame(main_container, bg=self.colors['bg_medium'], width=220)
        sidebar.pack(side='left', fill='y', padx=(0, 10))
        sidebar.pack_propagate(False)

        self.create_sidebar(sidebar)

        content = tk.Frame(main_container, bg=self.colors['bg_dark'])
        content.pack(side='right', fill='both', expand=True)

        self.create_content(content)

        self.create_status_bar()

    def create_header(self):
        """Create the header."""
        header = tk.Frame(self.root, bg=self.colors['bg_medium'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)

        left_frame = tk.Frame(header, bg=self.colors['bg_medium'])
        left_frame.pack(side='left', padx=15)

        title_label = tk.Label(
            left_frame,
            text="Text Summarizer Pro",
            font=('Segoe UI', 20, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_medium']
        )
        title_label.pack(side='left')

        version_badge = tk.Label(
            left_frame,
            text="v5.0",
            font=('Segoe UI', 10, 'bold'),
            fg=self.colors['accent'],
            bg=self.colors['bg_medium']
        )
        version_badge.pack(side='left', padx=(10, 0))

        right_frame = tk.Frame(header, bg=self.colors['bg_medium'])
        right_frame.pack(side='right', padx=15)

        if self.ollama_available:
            status_text = "AI Active"
            status_color = self.colors['accent_green']
        else:
            status_text = "Offline Mode"
            status_color = self.colors['accent_orange']

        self.status_label = tk.Label(
            right_frame,
            text=status_text,
            font=('Segoe UI', 10),
            fg=status_color,
            bg=self.colors['bg_medium']
        )
        self.status_label.pack(side='left')

        if self.ollama_available and self.current_ollama_model:
            self.model_label = tk.Label(
                right_frame,
                text=f"({self.current_ollama_model})",
                font=('Segoe UI', 9),
                fg=self.colors['text_muted'],
                bg=self.colors['bg_medium']
            )
            self.model_label.pack(side='left', padx=(8, 0))
        else:
            self.model_label = None

    def create_sidebar(self, sidebar):
        """Create the sidebar with controls."""
        canvas = tk.Canvas(sidebar, bg=self.colors['bg_medium'], highlightthickness=0)
        scrollbar = tk.Scrollbar(sidebar, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.colors['bg_medium'])

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # --- Section: Load Document ---
        self.create_section_header(scrollable, "Load Document")

        load_buttons = [
            ("PDF", self.load_pdf, '#e17055'),
            ("DOCX", self.load_docx, '#0984e3'),
            ("DOC", self.load_doc, '#6c5ce7'),
            ("ODT", self.load_odt, '#00b894'),
            ("RTF", self.load_rtf, '#6c5ce7'),
            ("TXT", self.load_text_file, '#fdcb6e'),
            ("URL", self.load_url, '#e17055'),
            ("Paste", self.paste_text, '#0984e3'),
        ]

        for text, command, color in load_buttons:
            btn = tk.Button(
                scrollable,
                text=text,
                command=command,
                bg=color,
                fg='white',
                font=('Segoe UI', 10),
                padx=8,
                pady=6,
                relief='flat',
                cursor='hand2',
                bd=0
            )
            btn.pack(fill='x', padx=12, pady=2)

        # --- Section: Summarizer ---
        self.create_section_header(scrollable, "Summarizer")

        # Style selector
        style_frame = tk.Frame(scrollable, bg=self.colors['bg_medium'])
        style_frame.pack(fill='x', padx=12, pady=4)

        tk.Label(
            style_frame,
            text="Style:",
            font=('Segoe UI', 9),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_medium']
        ).pack(side='left')

        self.style_var = tk.StringVar(value="Executive")
        styles = ['Executive', 'Bullet Points', 'Technical', 'Plain English',
                  'Student', 'Report', 'Key Findings', 'Action Items']
        style_combo = ttk.Combobox(
            style_frame,
            textvariable=self.style_var,
            values=styles,
            width=13,
            state='readonly',
            font=('Segoe UI', 9)
        )
        style_combo.pack(side='right')

        # Length + honest caption
        length_frame = tk.Frame(scrollable, bg=self.colors['bg_medium'])
        length_frame.pack(fill='x', padx=12, pady=4)

        tk.Label(
            length_frame,
            text="Length:",
            font=('Segoe UI', 9),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_medium']
        ).pack(side='left')

        self.ratio_var = tk.StringVar(value="25%")
        ratio_combo = ttk.Combobox(
            length_frame,
            textvariable=self.ratio_var,
            values=['10%', '15%', '20%', '25%', '30%', '40%', '50%'],
            width=8,
            state='readonly',
            font=('Segoe UI', 9)
        )
        ratio_combo.pack(side='right')

        self.ratio_combo_widget = ratio_combo
        ratio_combo.bind('<<ComboboxSelected>>', lambda e: self._update_length_caption())

        # Honest word-count caption (updated live)
        self.length_caption = tk.Label(
            scrollable,
            text="",
            font=('Segoe UI', 8, 'italic'),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_medium'],
            anchor='w'
        )
        self.length_caption.pack(fill='x', padx=12, pady=(0, 2))

        # Max sentences
        sent_frame = tk.Frame(scrollable, bg=self.colors['bg_medium'])
        sent_frame.pack(fill='x', padx=12, pady=4)

        tk.Label(
            sent_frame,
            text="Max Sentences:",
            font=('Segoe UI', 9),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_medium']
        ).pack(side='left')

        self.max_sentences_var = tk.StringVar(value="8")
        sent_combo = ttk.Combobox(
            sent_frame,
            textvariable=self.max_sentences_var,
            values=['5', '8', '10', '12', '15', '20'],
            width=8,
            state='readonly',
            font=('Segoe UI', 9)
        )
        sent_combo.pack(side='right')

        # --- Model mode (Fast / Quality) ---
        if self.ollama_available:
            self.create_section_header(scrollable, "Model")

            self.model_mode_var = tk.StringVar(value="Fast")
            model_frame = tk.Frame(scrollable, bg=self.colors['bg_medium'])
            model_frame.pack(fill='x', padx=12, pady=2)

            for mode in ("Fast", "Quality"):
                rb = tk.Radiobutton(
                    model_frame,
                    text=mode,
                    variable=self.model_mode_var,
                    value=mode,
                    command=self._on_model_mode_change,
                    font=('Segoe UI', 9),
                    fg=self.colors['text_primary'],
                    bg=self.colors['bg_medium'],
                    selectcolor=self.colors['bg_light'],
                    activebackground=self.colors['bg_medium'],
                    activeforeground=self.colors['text_primary'],
                    highlightthickness=0,
                    bd=0
                )
                rb.pack(side='left', padx=(0, 10))

        # --- Streaming + Debug toggles ---
        self.create_section_header(scrollable, "Options")

        self.stream_var = tk.BooleanVar(value=True)
        stream_cb = tk.Checkbutton(
            scrollable,
            text="Stream AI output",
            variable=self.stream_var,
            font=('Segoe UI', 9),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_medium'],
            selectcolor=self.colors['bg_light'],
            activebackground=self.colors['bg_medium'],
            activeforeground=self.colors['text_primary'],
            highlightthickness=0,
            bd=0,
            anchor='w'
        )
        stream_cb.pack(fill='x', padx=12, pady=1)

        self.debug_var = tk.BooleanVar(value=False)
        debug_cb = tk.Checkbutton(
            scrollable,
            text="Debug (full tracebacks)",
            variable=self.debug_var,
            command=lambda: setattr(self, 'debug_mode', self.debug_var.get()),
            font=('Segoe UI', 9),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_medium'],
            selectcolor=self.colors['bg_light'],
            activebackground=self.colors['bg_medium'],
            activeforeground=self.colors['text_primary'],
            highlightthickness=0,
            bd=0,
            anchor='w'
        )
        debug_cb.pack(fill='x', padx=12, pady=1)

        # --- Actions ---
        self.create_section_header(scrollable, "Actions")

        summarize_btn = tk.Button(
            scrollable,
            text="Generate Summary",
            command=self.summarize_text,
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=10,
            pady=10,
            relief='flat',
            cursor='hand2',
            bd=0
        )
        summarize_btn.pack(fill='x', padx=12, pady=4)

        export_btn = tk.Button(
            scrollable,
            text="Export Summary",
            command=self.export_summary,
            bg=self.colors['bg_light'],
            fg=self.colors['text_primary'],
            font=('Segoe UI', 10),
            padx=10,
            pady=6,
            relief='flat',
            cursor='hand2',
            bd=0
        )
        export_btn.pack(fill='x', padx=12, pady=2)

        clear_btn = tk.Button(
            scrollable,
            text="Clear All",
            command=self.clear_all,
            bg=self.colors['bg_light'],
            fg=self.colors['text_primary'],
            font=('Segoe UI', 10),
            padx=10,
            pady=6,
            relief='flat',
            cursor='hand2',
            bd=0
        )
        clear_btn.pack(fill='x', padx=12, pady=2)

    def create_section_header(self, parent, text):
        """Create a section header in the sidebar."""
        header = tk.Frame(parent, bg=self.colors['bg_medium'])
        header.pack(fill='x', padx=12, pady=(12, 4))

        tk.Label(
            header,
            text=text,
            font=('Segoe UI', 11, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_medium']
        ).pack(anchor='w')

        separator = tk.Frame(parent, bg=self.colors['border'], height=1)
        separator.pack(fill='x', padx=12, pady=(0, 6))

    def create_content(self, content):
        """Create the main content area."""
        # Drag and drop area
        self.drop_frame = tk.Frame(
            content,
            bg=self.colors['bg_medium'],
            height=100,
            relief='solid',
            bd=2
        )
        self.drop_frame.pack(fill='x', pady=(0, 10))
        self.drop_frame.pack_propagate(False)

        drop_label = tk.Label(
            self.drop_frame,
            text="Drag & Drop files here or click to browse",
            font=('Segoe UI', 12),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_medium']
        )
        drop_label.place(relx=0.5, rely=0.5, anchor='center')

        self.drop_frame.bind('<Button-1>', lambda e: self.load_text_file())
        drop_label.bind('<Button-1>', lambda e: self.load_text_file())

        if DND_AVAILABLE:
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)
            drop_label.drop_target_register(DND_FILES)
            drop_label.dnd_bind('<<Drop>>', self.on_drop)
            self.drop_frame.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            self.drop_frame.dnd_bind('<<DragLeave>>', self.on_drag_leave)
            drop_label.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            drop_label.dnd_bind('<<DragLeave>>', self.on_drag_leave)

        # File info
        self.file_info = tk.Label(
            content,
            text="No file loaded",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_dark']
        )
        self.file_info.pack(anchor='w', pady=(0, 5))

        self.create_text_areas(content)

    def on_drop(self, event):
        """Handle file drop event."""
        files = event.data
        if not files:
            return
        # Tk's own splitter understands {braced paths with spaces} and
        # multiple-file payloads; str.split() does not.
        try:
            file_list = self.root.tk.splitlist(files)
        except Exception:
            file_list = [files]
        if file_list:
            filepath = file_list[0]
            self._load_file(filepath)
            self.drop_frame.config(bg=self.colors['bg_medium'])

    def on_drag_enter(self, event):
        self.drop_frame.config(bg=self.colors['bg_hover'])

    def on_drag_leave(self, event):
        self.drop_frame.config(bg=self.colors['bg_medium'])

    def create_text_areas(self, parent):
        """Create the text areas."""
        text_container = tk.Frame(parent, bg=self.colors['bg_dark'])
        text_container.pack(fill='both', expand=True)

        # Original text
        original_frame = tk.Frame(text_container, bg=self.colors['bg_dark'])
        original_frame.pack(fill='both', expand=True, pady=(0, 8))

        original_header = tk.Frame(original_frame, bg=self.colors['bg_dark'])
        original_header.pack(fill='x')

        tk.Label(
            original_header,
            text="Original Document",
            font=('Segoe UI', 11, 'bold'),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_dark']
        ).pack(side='left')

        self.word_count_label = tk.Label(
            original_header,
            text="Words: 0 | Characters: 0",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_dark']
        )
        self.word_count_label.pack(side='right')

        self.input_text = scrolledtext.ScrolledText(
            original_frame,
            height=8,
            bg=self.colors['bg_light'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            relief='flat',
            bd=0
        )
        self.input_text.pack(fill='both', expand=True, pady=(4, 0))
        self.input_text.bind('<KeyRelease>', self.update_stats)

        # Summary
        summary_frame = tk.Frame(text_container, bg=self.colors['bg_dark'])
        summary_frame.pack(fill='both', expand=True)

        summary_header = tk.Frame(summary_frame, bg=self.colors['bg_dark'])
        summary_header.pack(fill='x')

        tk.Label(
            summary_header,
            text="Summary",
            font=('Segoe UI', 11, 'bold'),
            fg=self.colors['accent_green'],
            bg=self.colors['bg_dark']
        ).pack(side='left')

        self.ai_indicator = tk.Label(
            summary_header,
            text="[offline]",
            font=('Segoe UI', 9),
            fg=self.colors['accent_orange'],
            bg=self.colors['bg_dark']
        )
        self.ai_indicator.pack(side='left', padx=(8, 0))

        summary_stats_frame = tk.Frame(summary_header, bg=self.colors['bg_dark'])
        summary_stats_frame.pack(side='right')

        self.summary_count_label = tk.Label(
            summary_stats_frame,
            text="Summary: 0 words | 0 sentences",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_dark']
        )
        self.summary_count_label.pack(side='left')

        self.summary_time_label = tk.Label(
            summary_stats_frame,
            text=" | Reading: 0 min",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_dark']
        )
        self.summary_time_label.pack(side='left')

        self.output_text = scrolledtext.ScrolledText(
            summary_frame,
            height=8,
            bg='#0a1a0a',
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            relief='flat',
            bd=0
        )
        self.output_text.pack(fill='both', expand=True, pady=(4, 0))

        # Keywords
        keyword_frame = tk.Frame(summary_frame, bg=self.colors['bg_medium'], height=32)
        keyword_frame.pack(fill='x', pady=(4, 0))
        keyword_frame.pack_propagate(False)

        tk.Label(
            keyword_frame,
            text="Keywords:",
            font=('Segoe UI', 9, 'bold'),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_medium']
        ).pack(side='left', padx=(8, 4))

        self.keyword_label = tk.Label(
            keyword_frame,
            text="",
            font=('Segoe UI', 9),
            fg=self.colors['accent_green'],
            bg=self.colors['bg_medium'],
            anchor='w'
        )
        self.keyword_label.pack(side='left', fill='x', expand=True)

    def create_status_bar(self):
        """Create the status bar."""
        status_bar = tk.Frame(self.root, bg=self.colors['bg_medium'], height=28)
        status_bar.pack(fill='x', side='bottom')

        self.log_label = tk.Label(
            status_bar,
            text="Ready",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_medium']
        )
        self.log_label.pack(side='left', padx=12)

        self.progress = ttk.Progressbar(
            status_bar,
            mode='indeterminate',
            length=120
        )
        self.progress.pack(side='right', padx=12)
        self.progress.pack_forget()

    # ------------------------------------------------------------------
    # Logging / status helpers
    # ------------------------------------------------------------------
    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_label.config(text=msg)
        except Exception:
            pass
        self.root.after(100, self.process_log_queue)

    def log(self, message):
        self.log_queue.put(message)

    def update_status(self, status, color="orange"):
        colors = {
            "red": self.colors['accent_red'],
            "orange": self.colors['accent_orange'],
            "green": self.colors['accent_green'],
            "blue": self.colors['accent_blue'],
        }
        self.root.after(
            0,
            lambda: self.log_label.config(
                text=status, fg=colors.get(color, self.colors['text_muted'])
            )
        )

    def show_progress(self, show=True):
        if show:
            self.progress.pack(side='right', padx=12)
            self.progress.start(10)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _update_length_caption(self):
        """Show a realistic word-count caption under the Length dropdown."""
        try:
            ratio = int(self.ratio_var.get().replace('%', '')) / 100.0
        except Exception:
            ratio = 0.25
        source = self.input_text.get('1.0', tk.END).strip()
        words = len(source.split()) if source else 0
        if words == 0:
            self.length_caption.config(text="Load a document to see expected length")
            return
        estimate = max(3, int(words * ratio))
        try:
            cap = int(self.max_sentences_var.get() or 8) * 30
        except Exception:
            cap = 240
        estimate = min(estimate, cap)
        # Mirror the num_predict floor so the caption can't promise fewer
        # words than the model is allowed to emit.
        effective = max(estimate, 75)
        self.length_caption.config(
            text=f"~{effective} words for this input (~{words} source words)"
        )

    def update_stats(self, event=None):
        text = self.input_text.get('1.0', tk.END).strip()
        if text:
            words = len(text.split())
            chars = len(text)
            self.word_count_label.config(text=f"Words: {words} | Characters: {chars}")
        else:
            self.word_count_label.config(text="Words: 0 | Characters: 0")
        # Keep the honest caption in sync as the user types / loads.
        if hasattr(self, 'length_caption'):
            self._update_length_caption()

    def update_summary_stats(self, summary_text):
        if summary_text and summary_text.strip():
            words = len(summary_text.strip().split())
            sentences = len(re.findall(r'[.!?]+', summary_text)) or 1
            reading_time = max(1, round(words / 200))
            self.summary_count_label.config(
                text=f"Summary: {words} words | {sentences} sentences"
            )
            self.summary_time_label.config(text=f" | Reading: {reading_time} min")
        else:
            self.summary_count_label.config(text="Summary: 0 words | 0 sentences")
            self.summary_time_label.config(text=" | Reading: 0 min")

    def update_keywords(self, text):
        if not text or not text.strip():
            self.keyword_label.config(text="")
            return
        extractor = KeywordExtractor()
        keywords = extractor.extract_keywords(text, top_n=10)
        if keywords:
            keyword_text = " | ".join(f"{w} ({wt}%)" for w, wt in keywords)
            self.keyword_label.config(text=keyword_text)
        else:
            self.keyword_label.config(text="")

    def _clean_summary_output(self, summary):
        """Post-process the summary to remove leftover headers/artefacts."""
        if not summary:
            return summary

        header_patterns = [
            r'^[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s+Project\s+Contents\s*',
            r'^[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s+Project\s+Contents\s*',
            r'^[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s+Project\s+Contents\s*',
            r'^[A-Za-z]+\s+[A-Za-z]+\s+Project\s+Contents\s*',
            r'^Project\s+Contents\s*',
            r'^Contents\s*',
        ]
        for pattern in header_patterns:
            summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)

        summary = re.sub(r'^Overview\s+', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'Follow\s+-\s+up', 'Follow-up', summary)
        summary = re.sub(r'south\s+-\s+eastern', 'south-eastern', summary)
        summary = re.sub(r'north\s+-\s+west', 'north-west', summary)
        summary = re.sub(r'\s+-\s+', ' - ', summary)
        summary = re.sub(r'l ocation', 'location', summary)
        summary = re.sub(r'wh ich', 'which', summary)
        summary = re.sub(r't he', 'the', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()

        if summary:
            summary = summary[0].upper() + summary[1:]
        return summary

    # ------------------------------------------------------------------
    # Load methods
    # ------------------------------------------------------------------
    def load_text_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filepath:
            self._load_file(filepath)

    def load_pdf(self):
        if not PDF_AVAILABLE:
            messagebox.showerror("Error", "PyPDF2 not installed.")
            return
        filepath = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if filepath:
            self._load_file(filepath)

    def load_docx(self):
        if not DOCX_AVAILABLE:
            messagebox.showerror("Error", "python-docx not installed.")
            return
        filepath = filedialog.askopenfilename(
            title="Select DOCX File",
            filetypes=[("Word Documents", "*.docx")]
        )
        if filepath:
            self._load_file(filepath)

    def load_doc(self):
        filepath = filedialog.askopenfilename(
            title="Select DOC File",
            filetypes=[("Word 97-2003", "*.doc")]
        )
        if filepath:
            self._load_file(filepath)

    def load_odt(self):
        if not ODT_AVAILABLE:
            messagebox.showerror("Error", "odfpy not installed.")
            return
        filepath = filedialog.askopenfilename(
            title="Select ODT File",
            filetypes=[("OpenDocument Text", "*.odt")]
        )
        if filepath:
            self._load_file(filepath)

    def load_rtf(self):
        if not RTF_AVAILABLE:
            messagebox.showerror("Error", "striprtf not installed.")
            return
        filepath = filedialog.askopenfilename(
            title="Select RTF File",
            filetypes=[("Rich Text Format", "*.rtf")]
        )
        if filepath:
            self._load_file(filepath)

    def load_url(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Load from URL")
        dialog.geometry("500x150")
        dialog.configure(bg=self.colors['bg_medium'])
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Enter URL:",
            font=('Segoe UI', 11),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_medium']
        ).pack(pady=(15, 5))

        url_entry = tk.Entry(
            dialog, width=55, font=('Segoe UI', 10),
            bg=self.colors['bg_light'], fg=self.colors['text_primary'],
            relief='flat'
        )
        url_entry.pack(pady=5)
        url_entry.focus()

        def fetch():
            url = url_entry.get().strip()
            if not url:
                messagebox.showerror("Error", "Please enter a URL")
                return
            dialog.destroy()
            try:
                self.update_status("Loading URL...", "blue")
                text = FileHandler.read_file(url)
                if text:
                    self.input_text.delete('1.0', tk.END)
                    self.input_text.insert('1.0', text[:50000])
                    self.current_file = url
                    self.file_info.config(text=f"URL: {url[:50]}...")
                    self.log(f"Loaded from URL: {url[:50]}...")
                    self.update_stats()
                    self.update_status("URL loaded", "green")
                else:
                    messagebox.showerror("Error", "No text extracted from URL")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        btn_frame = tk.Frame(dialog, bg=self.colors['bg_medium'])
        btn_frame.pack(pady=12)

        tk.Button(
            btn_frame, text="Load URL", command=fetch,
            bg=self.colors['accent'], fg='white',
            font=('Segoe UI', 10), padx=20, pady=5,
            relief='flat', cursor='hand2'
        ).pack(side='left', padx=5)

        tk.Button(
            btn_frame, text="Cancel", command=dialog.destroy,
            bg=self.colors['bg_light'], fg=self.colors['text_primary'],
            font=('Segoe UI', 10), padx=20, pady=5,
            relief='flat', cursor='hand2'
        ).pack(side='left', padx=5)

        url_entry.bind('<Return>', lambda e: fetch())

    def paste_text(self):
        try:
            text = self.root.clipboard_get()
            if text:
                self.input_text.delete('1.0', tk.END)
                self.input_text.insert('1.0', text)
                self.current_file = "Clipboard"
                self.file_info.config(text="Pasted from clipboard")
                self.log("Pasted from clipboard")
                self.update_stats()
                self.update_status("Pasted", "green")
        except Exception:
            messagebox.showerror("Error", "No text in clipboard")

    def _load_file(self, filepath):
        try:
            self.update_status("Loading file...", "blue")
            text = FileHandler.read_file(filepath)
            if text:
                self.input_text.delete('1.0', tk.END)
                self.input_text.insert('1.0', text)
                self.current_file = filepath
                self.file_info.config(
                    text=f"{os.path.basename(filepath)} ({len(text)} chars)"
                )
                self.log(f"Loaded: {os.path.basename(filepath)} ({len(text)} chars)")
                self.update_stats()
                self.update_status("File loaded", "green")
            else:
                messagebox.showerror("Error", "Could not read file")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------
    def summarize_text(self):
        if self.is_processing:
            messagebox.showinfo("Info", "Already processing...")
            return

        text = self.input_text.get('1.0', tk.END)
        if not text.strip():
            messagebox.showerror("Error", "Please enter or load some text first")
            return

        try:
            ratio = int(self.ratio_var.get().replace('%', '')) / 100.0
            max_sentences = int(self.max_sentences_var.get())
        except Exception:
            ratio = 0.25
            max_sentences = 8

        cleaned_text = TextCleaner.clean_text(text)
        if not cleaned_text or len(cleaned_text.strip()) < 50:
            messagebox.showerror("Error", "Text is too short or no content after cleaning")
            return

        self.is_processing = True
        self.show_progress(True)
        self.update_status("Generating summary...", "blue")

        use_ai = bool(self.ollama_available and self.current_ollama_model)
        # Snapshot the model so a Fast/Quality toggle mid-run can't change
        # what the worker thread actually calls.
        model_snapshot = self.current_ollama_model

        thread = threading.Thread(
            target=self._summarize_thread,
            args=(cleaned_text, ratio, max_sentences, use_ai, model_snapshot),
            daemon=True
        )
        thread.start()

    def _target_words(self, source_words, ratio):
        """Estimate the user-visible target word count for num_predict."""
        estimate = max(3, int(source_words * ratio))
        try:
            cap = int(self.max_sentences_var.get() or 8) * 30
        except Exception:
            cap = 240
        return min(estimate, cap)

    def _summarize_thread(self, text, ratio, max_sentences, use_ai,
                          model_snapshot=None):
        """
        Worker thread.
        - Streams AI output (if enabled) with live status updates.
        - Falls back to the fast extractive summarizer.
        - Tightened AI-vs-fallback rule (v5.0): fallback only wins when the
          AI output is BOTH short (<40 words) AND shorter than the fallback.
        - Fires gpu_monitor.post_launch_status() exactly once, after the
          first successful AI generation.
        - Logs per-run timing: "OK 182w in 4.1s (44.4 w/s)".
        """
        try:
            style_display = self.style_var.get()
            style_key = style_display.lower().replace(' ', '_')

            source_words = len(text.split())
            target_words = self._target_words(source_words, ratio)

            summary = None
            used_ai = False
            ai_words = 0
            elapsed = 0.0

            # ---------- AI attempt ----------
            if use_ai and self.ollama_available and model_snapshot:
                self.root.after(
                    0,
                    lambda: self.ai_indicator.config(
                        text="AI", fg=self.colors['accent_green']
                    )
                )
                self.root.after(
                    0,
                    lambda: self.update_status(
                        f"AI ({style_display}) generating...", "blue"
                    )
                )

                stream_on = self.stream_var.get()

                def _progress(msg):
                    # Called from the worker thread by OllamaSummarizer.
                    # Guard at *execution* time: after(0) callbacks queued
                    # before the run finishes can fire after it, and would
                    # otherwise clobber the final "Summary complete" status.
                    def _apply(m=msg):
                        if self.is_processing:
                            self.update_status(m, "blue")
                    self.root.after(0, _apply)

                summarizer = OllamaSummarizer(
                    model=model_snapshot,
                    stream=stream_on,
                    debug=self.debug_mode,
                    gpu_monitor=self.gpu_monitor,
                    progress_cb=_progress if stream_on else None,
                )

                t0 = time.time()
                ai_summary = summarizer.summarize(
                    text, style=style_key, max_length=target_words
                )
                elapsed = time.time() - t0

                # Fire the one-shot VRAM check after the first *successful*
                # generation. Firing on a failed call just hits /api/ps while
                # nothing is loaded and burns the one-shot flag for nothing.
                if ai_summary and len(ai_summary.strip()) >= 10:
                    if not self.gpu_warned:
                        self.gpu_warned = True
                        self.gpu_monitor.post_launch_status()

                if ai_summary and len(ai_summary.strip()) >= 10:
                    summary = ai_summary
                    used_ai = True
                    ai_words = len(ai_summary.split())
                    self.root.after(
                        0,
                        lambda: self.log(
                            f"OK {ai_words}w in {elapsed:.1f}s "
                            f"({ai_words / max(elapsed, 0.001):.1f} w/s)"
                        )
                    )
                else:
                    self.root.after(
                        0,
                        lambda: self.log("AI summary failed, using fallback")
                    )

            # ---------- Fast extractive fallback ----------
            self.root.after(
                0,
                lambda: self.update_status("Generating summary...", "blue")
            )

            fast_summarizer = FastSummarizer()
            fast_summary = fast_summarizer.summarize(
                text, ratio=ratio, max_sentences=max_sentences
            ) or ""

            # [FALLBACK] v5.0 rule: fallback only wins when the AI output is
            # BOTH short (<40 words) AND shorter than the fallback. This
            # preserves a good short AI summary instead of replacing it with
            # a long keyword salad.
            if used_ai and summary:
                fast_words = len(fast_summary.split())
                if ai_words < 40 and ai_words < fast_words:
                    self.root.after(
                        0,
                        lambda: self.log(
                            f"AI summary was very short ({ai_words}w) - "
                            f"using fallback ({fast_words}w)"
                        )
                    )
                    final_summary = fast_summary
                    used_ai = False
                else:
                    final_summary = summary
            elif summary and used_ai:
                final_summary = summary
            else:
                final_summary = fast_summary or "No summary could be generated."

            # Normalize whitespace + capitalisation.
            final_summary = re.sub(r'\s+', ' ', final_summary).strip()
            if final_summary:
                final_summary = final_summary[0].upper() + final_summary[1:]

            self.root.after(
                0, lambda: self._update_summary(final_summary, used_ai)
            )

        except Exception as e:
            if self.debug_mode:
                traceback.print_exc()
            self.root.after(0, lambda: self._summarize_error(str(e)))

    def _update_summary(self, summary, used_ai=False):
        self.is_processing = False
        self.show_progress(False)

        if not summary or not summary.strip():
            summary = "Could not generate summary. Please try with different settings."

        summary = self._clean_summary_output(summary)

        self.output_text.delete('1.0', tk.END)
        self.output_text.insert('1.0', summary)

        self.update_summary_stats(summary)
        self.update_keywords(summary)

        if used_ai:
            self.ai_indicator.config(text="AI", fg=self.colors['accent_green'])
        else:
            self.ai_indicator.config(text="[offline]", fg=self.colors['accent_orange'])

        self.update_status("Summary complete", "green")

        original_text = self.input_text.get('1.0', tk.END).strip()
        original_words = len(original_text.split()) if original_text else 0
        summary_words = len(summary.strip().split()) if summary else 0

        if original_words > 0:
            compression = int((summary_words / original_words) * 100)
            self.log(f"Compression: {compression}% ({summary_words} words)")

        self.log("Summary complete")

    def _summarize_error(self, error):
        self.is_processing = False
        self.show_progress(False)
        if self.debug_mode:
            print(f"[summarize error] {error}")
        messagebox.showerror("Error", f"Summarization failed:\n{error}")
        self.log(f"Error: {error}")
        self.update_status("Error", "red")

    # ------------------------------------------------------------------
    # Export / clear
    # ------------------------------------------------------------------
    def export_summary(self):
        summary = self.output_text.get('1.0', tk.END)
        if not summary.strip():
            messagebox.showerror("Error", "No summary to export")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text Files", "*.txt"),
                ("Word Document", "*.docx"),
                ("All Files", "*.*")
            ]
        )

        if not filepath:
            return

        try:
            ext = os.path.splitext(filepath)[1].lower()

            if ext == '.docx' and not DOCX_AVAILABLE:
                messagebox.showerror(
                    "Error",
                    "python-docx is not installed; cannot write .docx.\n"
                    "Install it with:  pip install python-docx\n"
                    "\u2026or choose .txt instead."
                )
                return

            if ext == '.docx' and DOCX_AVAILABLE:
                doc = Document()
                doc.add_heading('Text Summarizer Pro - Summary', 0)
                doc.add_paragraph(
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                doc.add_paragraph(f"Style: {self.style_var.get()}")
                doc.add_paragraph('=' * 50)
                doc.add_paragraph(summary)

                if self.keyword_label.cget('text'):
                    doc.add_paragraph('\nKeywords:')
                    doc.add_paragraph(self.keyword_label.cget('text'))

                doc.save(filepath)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("TEXT SUMMARIZER PRO - SUMMARY\n")
                    f.write(
                        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"Style: {self.style_var.get()}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(summary)

                    if self.keyword_label.cget('text'):
                        f.write("\n\nKeywords:\n")
                        f.write(self.keyword_label.cget('text'))

                    # [FLUSH] v5.0: explicit flush so a crash mid-export
                    # leaves a readable truncated file, not an empty one.
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass

            self.log(f"Exported: {os.path.basename(filepath)}")
            messagebox.showinfo("Success", f"Exported to:\n{filepath}")
        except Exception as e:
            if self.debug_mode:
                traceback.print_exc()
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def clear_all(self):
        self.input_text.delete('1.0', tk.END)
        self.output_text.delete('1.0', tk.END)
        self.keyword_label.config(text="")
        self.file_info.config(text="No file loaded")
        self.current_file = None
        self.update_stats()
        self.summary_count_label.config(text="Summary: 0 words | 0 sentences")
        self.summary_time_label.config(text=" | Reading: 0 min")
        self.ai_indicator.config(
            text="[offline]", fg=self.colors['accent_orange']
        )
        self.log("Cleared all")

# ================= MAIN =================
def main():
    try:
        if DND_AVAILABLE:
            root = TkinterDnD.Tk()
        else:
            root = tk.Tk()
        app = ModernSummarizerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()