### Example executions

```bash
python3 cluster_extractor.py \
  --arguments-file "../../data/ines_data/20_arguments_for_every_topic.csv" \
  --embedder "../models/V1" \
  --num-keypoints 10 \
  --out "./output/clusters.jsonl"
```

```bash
python3 keypoint_generator.py \
  --clusters "./output/clusters.jsonl" \
  --bert-model "../models/2" \
  --limit 10 \
  --out "./output/keypoints.jsonl" \
  --out-csv "./output/keypoints.csv"
```