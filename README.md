# Cedar RAG teaching project

Completed individually; no partner.

Six fictional Cedar Lab passages, with a transparent keyword retrieval baseline.
Each labeled paragraph is one chunk. IDs and filenames are metadata; only the
paragraph text is searched. No answer generation is performed by this script.

## Run keyword retrieval

From the cedar-rag folder (Python standard library only):

```sh
python3 keyword_retriever.py "May a trained graduate student operate the laser alone at 19:00?"
python3 keyword_retriever.py "May a trained graduate student operate the laser alone at 19:00?" --k 3
python3 keyword_retriever.py "zebra"
```

The default k is 2. All six scores and their matched words are printed, followed
by up to k positive-scoring passages. No positive scores produces
"No matching passages." A larger k cannot add zero-scoring passages.

The script lowercases text, extracts alphanumeric words, and removes exactly:
`a, an, the, is, are, to, of, in, at, for, and, or, may, can, i, it, be`.
It counts distinct shared words, without stemming or synonyms. Ties are resolved
by ascending chunk ID. Actual Q1 output is in
`evidence/q1 keyword inspection/keyword-q1.txt`.

## How the code works

- `load_chunks()` reads the three documents and separates each label from its
  paragraph. A `Chunk` stores ID, filename, and text in separate fields.
- `words()` creates a set of searchable words; a repeated word appears only once.
- `retrieve()` intersects the question's word set with each passage's word set.
  The number of shared words is the score. It sorts scores highest first, then
  IDs alphabetically, and selects up to k positive results.
- `main()` accepts the command-line arguments and prints the results.

## Prepared embedding environment

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_setup.py
```

The initial model download requires internet access. No model API key is needed.
Verified on Python 3.13.3, sentence-transformers 6.2.0, and numpy 2.5.4 using
`sentence-transformers/all-MiniLM-L6-v2` on CPU. Full installed versions are in
`requirements.txt`; actual setup results are in `setup-verification.txt`.

## Run embedding retrieval (default k=2)

Use the prepared virtual environment from the cedar-rag folder:

```sh
.venv/bin/python embedding_retriever.py "May a trained graduate student operate the laser alone at 19:00?"
```

To save the actual Q1 inspection output:

```sh
.venv/bin/python embedding_retriever.py "May a trained graduate student operate the laser alone at 19:00?" > "evidence/vector inspection/embedding-inspection.txt"
```

The retriever uses the model already downloaded during setup, with
`local_files_only=True` and `device="cpu"`. It makes no external API requests.
It prints six ranked scores and always returns the top k (or all six if k is
larger than six), without filtering by score. Use `--k` to override the default.
You may supply multiple quoted questions in one invocation; the document vectors
are calculated once and reused for each question.

### Understanding the embedding code

1. `load_chunks()` reuses the existing document loader. Each ID and filename stays
   attached as metadata, but only `chunk.text` is passed to `encode_document()`.
2. `encode_document()` turns the six passages into six lists of 384 numbers: a
   matrix with shape `(6, 384)`. The pretrained model runs locally on the CPU.
3. `encode_query()` turns each question into a list with shape `(384,)` using the
   same model. Both methods normalize the lists to lengths approximately 1.
   These role-specific methods use the same underlying encoding behavior for this
   model. Neither method generates an answer.
4. All model-produced coordinates are retained, then promoted to float64 for
   arithmetic. The printed coordinates have four decimal places, but these
   rounded displays are never fed back into scoring.
5. `document_vectors @ query` multiplies corresponding coordinates and adds their
   products, producing one similarity score for each passage. All 384 coordinates
   contribute; the displayed first 12 are only a preview. Individual coordinates
   do not have reliable standalone meaning labels.
6. Dividing each dot product by the two vector lengths calculates cosine directly.
   Because the vectors are normalized, both calculations should agree. The script
   checks this with an absolute tolerance of 0.000001.
7. `sorted()` puts the six scores in descending order, using ascending chunk IDs
   for exact ties. Taking the first k indices selects the passages. Rankings use
   the unrounded scores. Similarity indicates relatedness, not truth or whether a
   passage contains enough evidence to answer the question.
8. The top passage inspection prints 12 coordinate products, their subtotal, the
   other 372 products' subtotal, and the complete sum. It checks that the complete
   sum agrees with the score used for retrieval.

The keyword retriever is unchanged. This embedding script retrieves and inspects
evidence; it does not generate answers or change the model's learned weights.

## Saved results and evidence

- `results.csv`: consolidated Q1-Q4 results for k=1, k=2, and k=3, including
  judgments and observations. The `k` column identifies each run (12 rows total).
- `evidence/retrieval runs/`: actual retrieval outputs, separately labeled by k.
- `evidence/k experiment/`: individual run CSVs and completed Excel judgment
  workbooks for k=1, k=2, and k=3.
- `evidence/vector inspection/embedding-inspection.txt`: Q1 coordinate and
  similarity inspection.
- `evidence/q1 keyword inspection/keyword-q1.txt`: keyword baseline output.
- `evidence/answer audits/`: screenshots of Q1, Q4, and the revised Q1 answer.

The comparison runs used the prepared Python interpreter with the exact package
versions in `requirements.txt` installed in a temporary recovery folder because
reads of the original environment's package files stalled. This workaround is
recorded in the run outputs. The model ran locally on CPU using cached weights;
retrieval code and scoring rules were unchanged.
