
# 1) Cluster erzeugen
```bash
python3 cluster_extractor.py \
  --seq2seq_model_path "../models/roberta_large" \
  --embs_model_path "../models/bge-large-en-v1.5" \
  --input_csv "../../data/ines_data/20_arguments_for_every_topic.csv" \
  --clusters_out "./output/clusters.jsonl"
````

# 2) Aus Clustern Keypoints extrahieren
```bash
python3 keypoint_generator.py \
  --clusters_in "./output/clusters.jsonl" \
  --keypoints_out "./output/keypoints.jsonl"
```