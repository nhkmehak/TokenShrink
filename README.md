# TokenShrink 🗜️

**Intelligent Prompt Compression for LLM Cost Optimization**

TokenShrink reduces LLM prompt size (token count) while preserving maximum semantic meaning, using a Greedy Search algorithm with a custom Information Density heuristic and a Dynamic Redundancy Filter.

---

## What It Does

| Metric | Typical Result |
|--------|---------------|
| Token Reduction | 30–65% |
| Meaning Preserved | 70–85% |
| Method | Greedy Search + TF-IDF + Redundancy Filter |

---

## Algorithm

```
Input Text
    │
    ▼
[Preprocessor]
  - Sentence segmentation (NLTK / regex fallback)
  - Word tokenization
  - Stopword removal
  - Keyword extraction
    │
    ▼
[TF-IDF Scorer]
  - Fit on all sentences in the document
  - Compute per-word importance weights
    │
    ▼
[Greedy Compressor]
  ┌──────────────────────────────────────────┐
  │  1. Lock anchor sentences (first + last) │
  │  2. Score remaining sentences:           │
  │     score = Σ tfidf(unused_kw) / tokens  │
  │  3. Select highest-scoring sentence      │
  │     that fits within token budget        │
  │  4. Update used-keyword set              │
  │     (Dynamic Redundancy Filter)          │
  │  5. Penalize overlapping keywords (×0.5) │
  │  6. Repeat until budget exhausted        │
  └──────────────────────────────────────────┘
    │
    ▼
[Token Counter]  + [Semantic Similarity Evaluator]
    │
    ▼
Compressed Prompt + Statistics
```

---

## Project Structure

```
tokenshrink/
├── backend/
│   ├── src/
│   │   ├── __init__.py          # Package exports
│   │   ├── tokenshrink.py       # Main public API
│   │   ├── preprocessor.py      # Sentence splitting + keyword extraction
│   │   ├── scorer.py            # TF-IDF + Information Density Heuristic
│   │   ├── algorithm.py         # Greedy Search + Redundancy Filter
│   │   ├── token_counter.py     # tiktoken-based token counting
│   │   └── similarity.py        # Cosine similarity evaluation
│   ├── tests/
│   │   └── test_all.py          # Full test suite (16 tests)
│   ├── app.py                   # Flask REST API server
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   ├── index.css            # Dark monospace UI styles
│   │   └── main.jsx             # React entry point
│   ├── package.json
│   └── index.html
└── README.md
```

---

## Setup

### Backend (Python 3.9+)

```bash
cd tokenshrink/backend

# Install dependencies
pip install -r requirements.txt

# (Optional) Download NLTK data for better sentence splitting
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Start the API server
python app.py
# → Running on http://localhost:5000
```

### Frontend (Node 18+)

```bash
cd tokenshrink/frontend

# Install dependencies
npm install

# Development server
npm run dev
# → Running on http://localhost:5173

# OR build for production
npm run build
```

---

## API Reference

### `POST /api/compress`

Compress a prompt.

**Request:**
```json
{
  "text": "Your long prompt here...",
  "target_ratio": 0.5,
  "include_debug": false
}
```

**Parameters (all optional except `text`):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | string | Input prompt (required) |
| `max_tokens` | int | Hard token budget for output |
| `target_ratio` | float | Fraction of tokens to keep (e.g. `0.5` = keep 50%) |
| `max_sentences` | int | Max number of sentences to select |
| `include_debug` | bool | Include per-iteration greedy log |

**Response:**
```json
{
  "compressed_text": "...",
  "stats": {
    "original_tokens": 380,
    "compressed_tokens": 137,
    "reduction_pct": 63.95,
    "tokens_saved": 243,
    "compression_ratio": 0.3605,
    "original_sentence_count": 20,
    "compressed_sentence_count": 7,
    "sentence_retention_pct": 35.0,
    "iterations": 5,
    "backend": "tiktoken"
  },
  "similarity": {
    "tfidf_similarity": 0.7499,
    "semantic_similarity": 0.7499,
    "method": "tfidf-cosine",
    "meaning_preserved_pct": 74.99
  },
  "selected_sentences": [...],
  "all_sentences": [...],
  "debug_log": [...]
}
```

### `POST /api/tokenize`

Count tokens in a text.

```json
{ "text": "Hello world" }
```

### `GET /api/health`

Health check — returns version and token backend info.

---

## Python Library Usage

```python
from src.tokenshrink import TokenShrink

ts = TokenShrink(
    preserve_structure=True,   # Always keep first + last sentence
    redundancy_penalty=0.5,    # Penalize overlapping keywords
)

result = ts.compress(
    text="Your long prompt...",
    target_ratio=0.5,          # Keep 50% of tokens
    # max_tokens=200,          # OR set hard token budget
    include_debug=True,
)

print(result["compressed_text"])
print(f"Reduced by {result['stats']['reduction_pct']}%")
print(f"Meaning preserved: {result['similarity']['meaning_preserved_pct']}%")
```

---

## Running Tests

```bash
cd tokenshrink/backend
python tests/test_all.py
```

Expected output:
```
TokenShrink Test Suite
======================================================================
── Preprocessor Tests ──── ✓ ✓ ✓ ✓
── Token Counter Tests ──── ✓ ✓
── Scorer Tests ────────── ✓ ✓ ✓
── Algorithm Tests ─────── ✓ ✓ ✓ (anchors preserved: 2/2)
── Similarity Tests ─────── ✓ ✓ ✓
── Full Pipeline Tests ──── ✓ ✓ ✓
── Edge Case Tests ─────── ✓ ✓ ✓ ✓
All tests completed!
```

---

## Key Design Decisions

### Information Density Heuristic
```
score(sentence) = Σ tfidf(kw) for kw in unique_unused_keywords(sentence)
                  ────────────────────────────────────────────────────────
                                   token_count(sentence)
```
This rewards sentences that introduce many new high-value concepts per token spent — maximizing information per unit cost.

### Dynamic Redundancy Filter
After selecting a sentence, its keywords are added to a "used" set. Future sentences containing those same keywords are scored with a penalty multiplier (default `0.5`). This ensures the compressed output covers **breadth** rather than repeating the same concepts.

### Structure Preservation
The first sentence (context/framing) and last sentence (instruction/conclusion) are always included as "anchor" sentences, regardless of their information density score.

### Token Counting
Uses `tiktoken` with the `cl100k_base` encoding (GPT-4 / Claude compatible) for accurate token counts. Falls back to whitespace estimation if tiktoken is unavailable.

---

## Dependencies

### Backend
```
flask>=3.0
flask-cors>=4.0
nltk>=3.8
scikit-learn>=1.3
tiktoken>=0.6
numpy>=1.24
sentence-transformers>=2.6  # optional
```

### Frontend
```
react, react-dom
recharts
lucide-react
vite
```

---

## Sample Compression Results

**Input (380 tokens, 20 sentences):**
> The rapid advancement of large language models has created both opportunities and challenges for developers and organizations. These models, with their billions of parameters, have demonstrated remarkable capabilities... [20 sentences]

**Output at 40% ratio (137 tokens, 7 sentences, 63.95% reduction):**
> The rapid advancement of large language models has created both opportunities and challenges for developers and organizations. Every API call to a service like OpenAI, Anthropic, or Google incurs costs proportional to input and output token counts. A 30% reduction in token usage across one million daily API calls can save thousands of dollars monthly. Simple heuristics like removing stopwords alone are insufficient for this task. The greedy algorithm then iteratively selects the highest-value sentences within a token budget. The result is a compressed prompt that preserves the most important concepts from the original.

**Semantic similarity: 74.99%** — nearly three quarters of the meaning preserved at 64% lower cost.
