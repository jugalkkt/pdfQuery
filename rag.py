import pdfplumber
import math, re
from collections import Counter

STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "on", "i", "my", "how", "what", "do"}

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]

def chunk(text, size=40, overlap=10):
    words, step = text.split(), size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]

def embed(text):
    # Stand-in for a real embedding model: a bag-of-words count vector.
    return Counter(tokenize(text))

def cosine(a, b):
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0

def rank(query, items, k=3):   
    """items: list of (source, text). Returns the k best (score, source, text)."""
    q = embed(query)
    scored = sorted(((cosine(q, embed(t)), s, t) for s, t in items), reverse=True)
    return [hit for hit in scored[:k] if hit[0] > 0]

def load_pdf(path):
    """Returns a list of (source, chunk_text) pairs, tagged with page number."""
    items = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""   # scanned/blank pages can return None
            source = f"{path.split('/')[-1]} p.{page_num}"
            for piece in chunk(text, size=40, overlap=10):
                items.append((source, piece))
    return items # returns [(source,text),...]



