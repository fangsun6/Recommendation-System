# Recommendation System Project – Case Study

This repository implements a Matrix Factorization (MF)–based recommendation system, built from scratch in Python. Along with MF, it includes data preprocessing pipelines, data generation utilities, and a modular design for extending the system to new models.

The goal of this project was not only to implement a functioning recommendation system but also to understand the design challenges behind real-world systems: working with noisy datasets, handling sparsity, optimizing training efficiency, and making the framework extensible for future research.

---

## 1) Motivation

Recommendation systems are everywhere — from e-commerce platforms (Amazon, eBay) to entertainment (Netflix, Spotify). While off-the-shelf libraries exist, building one from scratch forces us to confront the underlying problems:

- Data is messy: duplicates, missing values, extremely sparse interactions.
- Models must scale: naive solutions won’t work at millions of users/items.
- Cold start: new users/items with few ratings.
- Flexibility: one model rarely fits all; modularity matters.

This project explores these issues by constructing a clean pipeline for matrix-factorization-based recommendation and documenting lessons learned.

---

## 2) Data Exploration & Preprocessing

For this project, I worked with the Amazon Reviews dataset. The dataset provides product reviews with fields like `userID`, `itemID`, `rating`, and `timestamp`.

### Challenges
- **Sparsity**: users typically interact with very few items (sparse user–item matrix).
- **Cold start**: some users/items have only 1–2 ratings; training on them adds noise.
- **Inconsistency**: duplicates and missing values.

### Solutions
- Filtered out users/items with fewer than *N* interactions.
- Normalized IDs to dense integer ranges.
- Stored interactions in a lightweight format for efficient training.

The preprocessing pipeline is implemented in `Amazon_data_process.py` and produces clean train/test sets.

---

## 3) Methodology

The core model is Matrix Factorization (MF), a classic collaborative filtering method.

### 3.1 Representation
We represent:
- each user `u` as a latent vector `p_u in R^k`
- each item `i` as a latent vector `q_i in R^k`

The predicted rating is `r_hat_ui = p_u^T q_i`, where `k` is the number of latent factors.

### 3.2 Objective
We minimize squared error with L2 regularization:

`min  sum_{(u,i) in D} (r_ui - p_u^T q_i)^2  +  lambda * ( ||p_u||^2 + ||q_i||^2 )`

where the sum is over all observed user–item pairs `(u,i)` in dataset `D`.

### 3.3 Optimization
- Stochastic Gradient Descent (SGD) for updating embeddings.
- Optional learning-rate decay for stability.
- Mini-batch training via a custom data generator to balance speed and memory.

---

## 4) Implementation Details

The repository emphasizes modularity and extensibility:

- **`BaseModel.py`**  
  Abstract base class that defines `fit`, `predict`, and `evaluate`.

- **`MF.py`**  
  MF model implementation:
  - Parameter initialization for user/item embeddings
  - SGD updates with regularization
  - Training loop with periodic evaluation

- **`datagenerator.py`**  
  Custom data loader that:
  - Shuffles each epoch
  - Produces mini-batches
  - Efficiently handles sparsity

- **`main.py`**  
  Orchestrates end-to-end training and evaluation.

- **`Amazon_data_process.py`**  
  Cleans and preprocesses raw Amazon data.

---

## 5) Experiments

### 5.1 Metrics
- **RMSE (Root Mean Squared Error)**: global accuracy on rating prediction.
- **MAE (Mean Absolute Error)**: more robust to outliers than RMSE.

### 5.2 Illustrative Results
With `num_factors = 20`, `learning_rate = 0.01`, `reg = 0.1`:
- Train RMSE: ~0.92
- Test RMSE: ~0.95

Increasing `num_factors` improved fit but increased overfitting risk.

### 5.3 Observations
- **Overfitting**: Without regularization, the model memorizes frequent users/items.
- **Cold start**: Performance drops for users/items with <5 ratings.
- **Trade-offs**: More factors can improve accuracy but raise compute cost and overfitting risk.

### 5.4 Visualizations (optional)
- Training loss vs. epochs
- Test RMSE vs. `num_factors`
- Error distribution histograms

---

## 6) Lessons Learned

- **Data quality > model complexity**: better cleaning/filters often beat hyperparameter tweaks.
- **Regularization is critical**: curbs overfitting, especially with popular users/items.
- **Design for flexibility**: the `BaseModel` interface makes adding new recommenders straightforward.
- **Cold start needs hybridization**: metadata/content features are key in production.

---

## 7) Future Directions

- Neural Collaborative Filtering (NCF) and other deep models
- Hybrid recommenders using side information (e.g., product categories, text)
- Cross-validation and automated hyperparameter search
- Scaling via parallel/distributed SGD for very large datasets

---

## 8) Usage

### Install
```bash
git clone https://github.com/fangsun6/Recommendation-System.git
cd Recommendation-System
pip install -r requirements.txt
```

### Run preprocessing
```bash
python Amazon_data_process.py
```

### Train MF model
```bash
python main.py
```

### Hyperparameters
Edit in `main.py` or `MF.py`:
- `num_factors` (latent dimension `k`)
- `learning_rate`
- `reg` (L2 regularization strength)
- `epochs`
- batch size / shuffle options in `datagenerator.py`

---

## 9) Example Workflow (Python)

```python
from MF import MF
from datagenerator import DataGenerator

# Prepare dataset
train_data, test_data = DataGenerator().generate()

# Initialize and train
model = MF(num_factors=20, learning_rate=0.01, reg=0.1, epochs=50)
model.fit(train_data)

# Evaluate
print("Test RMSE:", model.evaluate(test_data))
```

---

## 10) Final Remarks

This project was an exercise in building from the ground up. While libraries like Surprise, implicit, or TensorFlow Recommenders can achieve strong results out of the box, reimplementing MF exposes the practical complexities of preprocessing, optimization, and evaluation. It also lays a foundation for exploring more advanced architectures in a consistent framework.

---

## License
MIT License


