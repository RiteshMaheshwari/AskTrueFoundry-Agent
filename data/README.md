# Local RAG Index

Phase 1 writes generated retrieval artifacts here:

- `index.jsonl`
- `embeddings.npy`

These files are ignored by git because they are reproducible from the ingestion script and can be regenerated with:

```bash
python scripts/ingest.py --max-pages 120
```
