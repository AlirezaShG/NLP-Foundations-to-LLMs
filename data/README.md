# Data

Notebooks resolve everything here through a `DATA` path defined in their first
cell, so they run correctly whether Jupyter was started at the repository root
or inside `notebooks/`.

## Committed

| File | Size | Rows | Used by |
|---|---|---|---|
| `emails_1.csv` | 1 KB | 10 | `02` — a hand-sized sample, small enough to print a whole document-term matrix |
| `emails_2.csv.gz` | 1.6 MB | 5,172 × 3,001 | `02` — spam classification |
| `urls.csv` | 836 KB | 11,430 | `02` — phishing URL classification |
| `parsed_travel_book.csv` | 640 KB | 433 | `07` — LlamaParse output of a travel guide PDF, backing the FAQ tool |

`emails_2.csv.gz` is the uncompressed 31 MB CSV gzipped to 1.6 MB — it is a
mostly-zero word-count matrix, so it compresses about 19×. `pandas.read_csv`
infers the compression from the extension, so nothing in the notebook changes.

Committing `parsed_travel_book.csv` is what makes `LLAMA_CLOUD_API_KEY`
optional in notebook `07`: the parse has already been done.

## Downloaded on first run

These are fetched by the notebooks themselves and are git-ignored.

| Path | Fetched by | Source |
|---|---|---|
| `wikitext/` | `03` | WikiText-2 train/valid/test, from the PyTorch examples repo |
| `cc.en.300.bin` | `03` | FastText English vectors (~7 GB uncompressed) |
| `lancedb_store/` | `06`, `07` | LanceDB vector store, built from the corpus |
| — | `01` | `taesiri/TinyStories-Farsi`, via `datasets` |
| — | `03` | AG News, via `datasets` |
| — | `04`, `05` | Text-to-SQL and instruction-tuning sets, via `datasets` |
| — | `07` | IATA airport codes and currency codes, via `kagglehub` |

## Not included

Notebook `06` builds a RAG pipeline over a corpus of Iranian statutes
(`data/laws/`, a tree of `.txt` files covering labour law, the cheque law, VAT,
the Landlord & Tenant Act 1376, social security, deeds registration, and
anti-smuggling). That corpus is not redistributed here. The notebook's chunker
splits on the ` ماده ` (article) marker, so any similarly structured Persian
legal text will run through the same pipeline; point `DATA / 'laws'` at it.

Everything else in `06` — the chunking logic, the embedding and retrieval code,
and the RAGAS evaluation — reads as written without the corpus present.
