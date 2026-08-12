# Converts downloaded Hugging Face parquet shards to a JSON array for the Node importer.
# Usage: python3 hf_parquet_to_json.py output.json shard1.parquet [shard2.parquet ...]
import json
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow is required", file=sys.stderr)
    sys.exit(2)

output = sys.argv[1]
rows = []
for filename in sys.argv[2:]:
    parquet_file = pq.ParquetFile(filename)
    for batch in parquet_file.iter_batches(batch_size=1000):
        rows.extend(batch.to_pylist())

with open(output, "w", encoding="utf-8") as handle:
    json.dump(rows, handle, ensure_ascii=False)
