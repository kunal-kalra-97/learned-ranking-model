# Learned Query Optimizer

## Section 0 - General Information

### Project Structure

```
feature_extraction.py        Vocabulary building, normalization stats, CLI entry point
feature_extractor.py         Per-plan feature extraction (Tables, Joins, Predicates)
train_test_model.py          Training entry point; trains model and runs local evaluation
evaluate_model.py            Loads model/model.pkl and evaluates ranking performance
merge_data.py                Merges multiple feature files into a single training set
stats.json                   Pre-computed vocabularies and normalization statistics
requirements.txt             Python dependencies
datasets/                    Generated feature files (train/test JSON)
plans/                       Raw query plan JSON files (input data)
model/
    mscn_model.py            MSCNModel definition (3 SetEncoders + final MLP)
    set_encoder.py           SetEncoder with residual blocks and masked mean pooling
    ranking_model_wrapper.py Training loop, inference wrapper, serializable model class
    mscn_dataset.py          PyTorch Dataset/DataLoader with custom collate
    data_utils.py            Tensor padding, masking, batch assembly utilities
    model.pkl                Trained model checkpoint (joblib-serialized)
```

### Dependencies

- `torch` for model definition and training
- `pandas` for data organization and grouping by SQL query
- `numpy` for inference argmin
- `joblib` for model serialization and deserialization
- `orjson` for fast JSON I/O for feature files
- `matplotlib` for plotting training curves
- `scikit-learn`, `tqdm`, `requests` as utilities

### Commands to Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Feature extraction (build vocabularies and scan norms from training data)
python feature_extraction.py \
    --file_path plans/train_plans2.json \
    --output_path datasets/train_features_2.json \
    --build_stats

# 3. Feature extraction (second training set, reuses stats.json)
python feature_extraction.py \
    --file_path plans/train_plans1.json \
    --output_path datasets/train_features_1.json

# 4. Feature extraction (test data)
python feature_extraction.py \
    --file_path plans/test_plans.json \
    --output_path datasets/test_features.json

# 5. Merge both training sets
python merge_data.py datasets/train_features_1.json datasets/train_features_2.json datasets/train_features.json

# 6. Train and evaluate
python train_test_model.py \
    --train_data datasets/train_features.json \
    --test_data datasets/test_features.json
```

---

## Section 1 - Data Preprocessing and Featurization

### Data Preprocessing

The input is a JSON file containing `parsed_plans` (plan trees with `plan_parameters` and `children`) and `database_stats` (column-level and table-level PostgreSQL statistics). Preprocessing steps:

1. **Table-column map construction**: `pg_class` (reltuples, relpages) and `pg_stats` (null_frac, avg_width, n_distinct, correlation, data_type) are merged into a per-table record with a column list. This provides the static schema context.

2. **Vocabulary building** (training data only, saved to `stats.json`):
   - **Join edge vocabulary**: DFS over all plan trees to collect unique (table_1, table_2) pairs, sorted and mapped to indices. Unseen edges at test time map to `<UNK_EDGE>`.
   - **Join algorithm vocabulary**: operator names at join nodes (Hash Join, Nested Loop, etc.), mapped to indices with `<UNK_ALG>` fallback.
   - **Child operator vocabulary**: operator names of join children (Seq Scan, Index Scan, Hash, Sort, etc.), mapped to indices with `<UNK_OP>` fallback.
   - **Column vocabulary**: all `table.column` pairs from the schema, sorted into an index map (108 columns).
   - **Predicate operator vocabulary**: filter and join operators (=, !=, <=, >=, AND), sorted into an index map with `<UNK_OP>` fallback.

3. **Normalization statistics** (training data only): For each continuous feature, the training-set distribution is summarized as {mean, std, p1, p99}. At feature-extraction time, values are clipped to [p1, p99] then z-score normalized. This applies to table scalars (14 keys), scan scalars (5 keys), join scalars (6 keys), and per-column predicate literal values.

4. **Training data merging**: Both provided training files (`train_plans1.json` and `train_plans2.json`) are extracted separately and then merged into a single training set using `merge_data.py`. This roughly doubles the amount of training data, which helps reduce overfitting.

5. **Timeout filtering**: Plans with `plan_runtime_ms = null` (execution timeouts) are dropped.

### Featurization Overview

Each query plan is featurized into three variable-length sets, following the MSCN (Multi-Set Convolutional Network) paradigm. The three sets are processed independently before being combined:

| Set | One row per | Dimensionality |
|---|---|---|
| **Tables** | table referenced in the plan | 115 (89 one-hot + 14 schema scalars + 12 scan-level features) |
| **Joins** | join node in the plan tree | 79 (50 edge + 5 algo + 14 child ops + 10 scalars) |
| **Predicates** | atomic filter or join predicate | 226 (216 columns + 6 operators + 4 scalars) |

### Table Features (115 dims per table)

Each table's feature vector consists of three groups: a table identity one-hot, static schema statistics, and scan-level plan features.

**Table identity (89 dims)**

| Feature | What it captures | Extraction | Encoding |
|---|---|---|---|
| `table_id` | Which table is accessed | One-hot over all 89 schema tables | 89-dim binary |

**Schema statistics (14 dims)**

| Feature | What it captures | Extraction | Encoding |
|---|---|---|---|
| `log_rows` | Table cardinality | log1p(reltuples) | z-scored float |
| `rows_frac` | Relative table size | reltuples / total schema rows | z-scored float |
| `approx_size_mb` | Estimated table payload | avg_width * rows / 1 MB | z-scored float |
| `cols_count` | Schema width | Number of columns | z-scored float |
| `avg_col_width` / `max_col_width` | Column byte widths | Mean/max of pg_stats.avg_width | z-scored float |
| `mean_null_frac` / `max_null_frac` | Null density | Mean/max of pg_stats.null_frac | z-scored float |
| `mean_ndv_ratio` / `max_ndv_ratio` | Cardinality richness | NDV / rows (Postgres sign convention) | z-scored float |
| `frac_high_ndv` | Fraction of high-cardinality cols | Count(NDV ratio >= 0.5) / ncols | z-scored float |
| `mean_abs_corr` | Physical-logical ordering | Mean of abs(pg_stats.correlation) | z-scored float |
| `frac_numeric` / `frac_text` | Column type distribution | Count by data_type prefix / ncols | z-scored float |

**Scan-level features (12 dims)**

These features are extracted by a DFS over the plan tree that finds the scan node for each table and records how it is accessed. When a table appears in multiple scan nodes (e.g. self-join), the scan with the largest total work (est_card * est_loops) is kept.

| Feature | What it captures | Extraction | Encoding |
|---|---|---|---|
| `scan_type` | Access method used | One-hot: Seq Scan, Index Scan, Index Only Scan, Bitmap Heap Scan, Bitmap Index Scan, Other | 6-dim binary |
| `scan_est_card_log` | Rows returned by scan | log1p(est_card) at scan node | z-scored float |
| `scan_est_loops_log` | Number of scan repetitions | log1p(est_loops) at scan node | z-scored float |
| `scan_selectivity` | Fraction of table scanned | est_card / table_rows, clipped to [0,1] | z-scored float |
| `total_scan_work_log` | Total scan work | log1p(est_card * est_loops) | z-scored float |
| `num_local_filters` | Number of filter predicates | Count of atomic filters at scan node | z-scored float |
| `has_index_cond` | Whether index condition is used | 1.0 if any predicate has is_index_cond=True | binary |

The scan-level features were added because two plans accessing the same table via different strategies (e.g. a Seq Scan returning 2.5M rows vs. an Index Scan with est_loops=2,528,312 returning 1 row per probe) had identical table features without them. These features close that critical information gap.

### Join Features (79 dims per join node)

| Feature | What it captures | Extraction | Encoding |
|---|---|---|---|
| `edge_id` | Which table pair is joined | One-hot over 50 training edges + UNK | 50-dim binary |
| `algo_id` | Join algorithm chosen | One-hot (Hash Join, NL, Merge Join, etc.) | 5-dim binary |
| `left_child_op` / `right_child_op` | Child operator types | One-hot per child (Seq Scan, Hash, Sort, etc.) | 2 x 7-dim binary |
| `est_card_out_log` | Join output cardinality | log1p(est_card) at join node | z-scored float |
| `est_width_out` | Output row width | est_width from planner | z-scored float |
| `est_loops_log` | Loop count (nested loop) | log1p(est_loops) | z-scored float |
| `left_card_log` / `right_card_log` | Child cardinalities | log1p(child.est_card) | z-scored float |
| `join_sel_log` | Join selectivity estimate | log1p(est_card / (left * right)) | z-scored float |
| `depth_norm` | Position in join tree | depth / max_depth | float [0, 1] |
| `is_root_join` | Whether this is the top join | depth == 0 | binary |
| `left_deep_hint` | Left-deep tree indicator | right child is not a join | binary |
| `index_cond_flag` | Index-condition on join | join.is_index_cond | binary |

These features capture the planner's join strategy (algorithm, child operators), scale (cardinalities, loops), and tree shape (depth, left-deep). This follows the insight from Ganapathi et al. (ICDE 2009) that operator-level plan features are far more predictive than SQL-text features.

### Predicate Features (226 dims per predicate)

| Feature | What it captures | Extraction | Encoding |
|---|---|---|---|
| `column_A` | Filtered or left join column | One-hot over 108 schema columns | 108-dim binary |
| `column_B` | Right join column (zero for filters) | One-hot (or zero vector for filters) | 108-dim binary |
| `operator` | Predicate operator | One-hot (=, !=, <=, >=, AND, UNK) | 6-dim binary |
| `val_low` / `val_high` | Literal value(s) | Per-column z-scored float; 0 for non-numeric | 2 floats |
| `is_join_pred` | Join vs. filter distinction | 1.0 for join predicates | binary |
| `is_index_cond` | Index-pushed predicate | plan_parameters.is_index_cond | binary |

Boolean filters (AND/OR/NOT with children) are recursively flattened into atomic predicates. BETWEEN is encoded as two values (low, high); IN uses the mean of list elements.

Predicates determine selectivity. Encoding both the column identity (which column is filtered) and the literal value (where in the column's distribution) gives the model information about how much data the scan will return.

---

## Section 2 - Model Architecture

We use a Multi-Set Convolutional Network (MSCN), which handles the variable-length nature of query plans (different numbers of tables, joins, predicates per query).

### Architecture Diagram

```
 Tables Set (m x 115)      Joins Set (n x 79)      Predicates Set (p x 226)
        |                        |                         |
   SetEncoder               SetEncoder                SetEncoder
  (115 -> 128 -> 128)      (79 -> 128 -> 128)       (226 -> 128 -> 128)
   + 2 ResBlocks            + 2 ResBlocks             + 2 ResBlocks
        |                        |                         |
   Masked Mean              Masked Mean               Masked Mean
   Pooling                  Pooling                   Pooling
        |                        |                         |
        +------------+-----------+                         |
                     +------------------+------------------+
                              Concatenate
                              (384-dim)
                                  |
                          +---------------+
                          |   Final MLP   |
                          |  384 -> 256   |
                          |  ReLU+Dropout |
                          |  256 -> 256   |
                          |  ReLU+Dropout |
                          |  256 -> 256   |
                          |  ReLU         |
                          |  256 -> 1     |
                          +---------------+
                                  |
                        predicted log1p(runtime_ms)
```

### SetEncoder (per set)

Each SetEncoder transforms a variable-length set of feature vectors into a single fixed-length representation:

1. **Input projection**: Linear(in_dim, 128), ReLU, Dropout(0.1)
2. **Residual blocks** (x2): Linear(128, 128), ReLU, Dropout(0.1) + skip connection
3. **Output projection**: Linear(128, 128), ReLU
4. **Masked mean pooling**: Averages over real rows only (ignoring padding), producing a (B, 128) vector

The residual connections stabilize gradient flow through the blocks.

### Final MLP

The three 128-dim pooled vectors are concatenated (384-dim) and passed through a 4-layer MLP (384, 256, 256, 256, 1) with ReLU activations and Dropout(0.1) between hidden layers.

### Libraries

Implemented in PyTorch. The model is wrapped in `RankingModelWrapper` (which stores feature dimensions alongside the model) and serialized with `joblib` for the evaluation platform.

---

## Section 3 - Training and Inference Procedure

### Data Split

Both training files (`train_plans1.json` and `train_plans2.json`) are used together. Features are extracted from each file separately, then merged into a single `train_features.json`. The test set (`test_plans.json`) is extracted separately and also serves as the validation set during training.

### Label Transformation

Runtime labels are transformed via `log1p(runtime_ms)` before training. This compresses the wide runtime range (e.g. 100ms to 30,000ms) and makes the loss more balanced across fast and slow queries.

### Loss Function

Huber loss (delta=1.0) on `log1p(runtime_ms)` predictions.

### Inference (Plan Ranking)

At inference time, the model acts as a cost model for plan selection:

1. For each query, receive a list of candidate plans (each already featurized into tables, joins, predicates sets)
2. Run a forward pass on each candidate to get a predicted log1p(runtime_ms)
3. Return argmin(predictions), the plan with the lowest predicted runtime

### Query Representation

Each query plan is represented as three padded tensors with associated masks:
- `tables_X`: (B, M_tables, 115), `tables_m`: (B, M_tables) boolean mask
- `joins_X`: (B, M_joins, 79), `joins_m`: (B, M_joins) boolean mask
- `preds_X`: (B, M_preds, 226), `preds_m`: (B, M_preds) boolean mask

Within each batch, sets are zero-padded to the maximum set size and the mask indicates which rows are real vs. padding.

### Optimizer and Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 64 |
| Max epochs | 50 (early stopping typically stops much sooner) |
| Loss function | Huber loss (delta=1.0) |
| Dropout | 0.1 |
| Gradient clipping | max_norm = 1.0 |
| Hidden (SetEncoder) | 128 |
| Hidden (Final MLP) | 256 |
| Residual blocks per SetEncoder | 2 |
| Final MLP layers | 4 (including output) |


## Evaluation and Results

### Metric

Sum of picked runtimes (seconds) across all test queries. Lower is better.

### Results

| Metric | Value |
|---|---|
| Server test score | **270.04s** |
| Local test score | 1092.87s |
| Best epoch / val_loss | 5 / 0.1467 |
| Early stopping epoch | 12 |
| Training time | ~160s (CPU) |
