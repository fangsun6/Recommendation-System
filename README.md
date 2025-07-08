# 🔍 DeepFM + PyTorch Recommender Systems 
This project combines:
- A TensorFlow-based implementation of **DeepFM** for CTR prediction.
- A PyTorch-based framework supporting **Top-N Recommendation models** such as BPRMF, LightGCN, EASE, and MultVAE.
- Support for evaluation using **C++ backend**, YAML-based configuration, and flexible experiment tracking.

---

## 🎯 Project Motivation

### Why build this?

Recommender systems are core to modern personalization engines — from e-commerce to content streaming. This project demonstrates how:
- **DeepFM** effectively models both *low-order* and *high-order* interactions via a hybrid FM + DNN approach.
- **PyTorch-based collaborative filtering** systems can be built with modularity, allowing plug-and-play new models and extensive experimentation.
- Evaluation and reproducibility are first-class citizens — enabling proper comparison of models and avoiding research pitfalls like data leakage or overfitting.

---

## 🧠 Thinking Flow & Technical Choices

### DeepFM Architecture

- **FM component**: learns pairwise interactions through inner product of embeddings.
- **DNN component**: captures nonlinear high-order feature interactions.
- **Combination**: the outputs of FM and DNN are concatenated and fed into the final prediction layer.

We chose **TensorFlow 1.15** because:
- It was the original DeepFM implementation base.
- Batch normalization and dropout are easier to manage for research reproducibility.
- YellowFin optimizer is supported, which helps auto-tune learning rate.

### PyTorch RecSys Design

- Each model inherits from a `BaseModel` class, which enforces a consistent API for training, evaluation, and logging.
- Evaluation uses `Evaluator` class that supports multiple ranking metrics.
- Datasets are handled with `UIRTDataset`, supporting holdout and leave-one-out protocols.
- All configurations are centralized in `OmegaConf` YAML files to make experiments reproducible and clean.

---

## 🧩 Challenges & Engineering Solutions (In-Depth)

### 🚧 Challenge 1: Handling Sparse Categorical Features in DeepFM

**Problem**: Sparse one-hot encoded features lead to huge input vectors and memory overhead.

**Solution**:
- We use an *embedding layer* that maps large sparse inputs to dense vectors.
- A `feature_index` and `feature_value` structure compresses input dimensions.
- Categorical features are transformed into indices with preprocessing tools (e.g., `DictVectorizer`, `LabelEncoder`).

**Thinking Flow**:
- Map categorical → integer index → embedding.
- This is crucial for model scalability.

---

### 🚧 Challenge 2: Instability in Batch Normalization (TF1.x)

**Problem**: Training and inference require separate paths in TF1.x, and batch norm updates can be missed.

**Solution**:
- Used `tf.cond()` to conditionally compute batch norm depending on the training phase.
- Controlled scope naming to prevent reuse bugs.
- Added `batch_norm_decay` as a tunable hyperparameter.

**Thinking Flow**:
- Model consistency across training/inference matters, especially in DeepFM’s deep layers.

---

### 🚧 Challenge 3: Evaluation Speed for PyTorch Recommenders

**Problem**: Ranking all items for every user in validation/test is slow, especially with large item sets.

**Solution**:
- Implemented optional C++ backend using Cython for evaluation speedup.
- Used `setup.py` to compile and switch backend at runtime.

**Thinking Flow**:
- Rank-based metrics are computationally expensive but unavoidable for RecSys. Efficient backend matters for practical training.

---

### 🚧 Challenge 4: Fair Validation in Implicit Feedback

**Problem**: Implicit feedback datasets lack explicit negatives; sampling bias can skew results.

**Solution**:
- Leave-one-out and holdout protocols ensure that users are evaluated fairly.
- Unseen items are treated as negatives.
- Controlled for minimum number of interactions per user/item in `config.py`.

**Thinking Flow**:
- Benchmarking requires careful split logic — fair comparisons are more important than overfitting performance.

---

### 🚧 Challenge 5: Modular Model Addition (Extensibility)

**Problem**: Every new model might require boilerplate or hacking if not planned well.

**Solution**:
- `BaseModel` ensures consistent function signatures.
- New models only need to implement `fit()`, `predict()`, and optionally `score_batch()` or `forward()`.
- A YAML config is created under `conf/` for easy hyperparameter management.

**Thinking Flow**:
- Design for research: code once, experiment many times.

---

### 🚧 Challenge 6: Handling Cold Start or Data Sparsity

**Problem**: Users/items with few interactions are common in RecSys data.

**Solution**:
- Filtered users/items with too few interactions via `min_item_per_user` and `min_user_per_item` in config.
- In DeepFM, incorporated auxiliary metadata features (e.g., age, category) to compensate for sparse interactions.

**Thinking Flow**:
- Data preprocessing and thoughtful filtering is the first step to meaningful learning.

---

### 🚧 Challenge 7: Managing Training Stability and Logging

**Problem**: Tracking training metrics and configurations is often inconsistent.

**Solution**:
- Used both `FileLogger` and `CSVLogger` to log every epoch.
- All parameters saved with timestamped logs for reproducibility.
- Final metrics stored in `ret['scores']` and written to CSV.

**Thinking Flow**:
- Transparent training logs are vital in both production and research settings.

---

## 🧪 Example Workflows

### ✅ DeepFM for CTR Prediction

```python
from DeepFM import DeepFM
from sklearn.metrics import roc_auc_score

params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 16,
    "deep_layers": [64, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "batch_size": 1024,
    "learning_rate": 0.001,
    "epoch": 20,
    "loss_type": "logloss",
    "eval_metric": roc_auc_score,
}

dfm = DeepFM(**params)
dfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
```

### ✅ PyTorch Recommendation on MovieLens

```bash
# Set dataname and model_name in config.py
python main.py
```

- Models: BPRMF, NGCF, LightGCN, MultVAE, EASE, SLIM, etc.
- Evaluation metrics: NDCG@K, Recall@K, Precision@K

---

## 🧠 Future Directions

- [ ] Port DeepFM to PyTorch and align APIs with collaborative models.
- [ ] Add self-supervised pretraining (e.g., BERT4Rec, SimGCL).
- [ ] AutoML for model and hyperparameter selection.
- [ ] Graph-enhanced models with user/item knowledge graphs.

---

## 📚 References

- Guo et al., DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. IJCAI 2017.
- Steck, EASE: Embarrassingly Shallow Autoencoders. WWW 2019.
- He et al., Neural Collaborative Filtering. WWW 2017.
- Liang et al., Variational Autoencoders for Collaborative Filtering. WWW 2018.
- Wang et al., LightGCN: SIGIR 2020.

---

## 📜 License

This project is under the MIT License.
