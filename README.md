# Recommendation System Project – Case Study

This repository implements a **Matrix Factorization (MF)–based recommendation system**, built from scratch in Python. Along with MF, it includes data preprocessing pipelines, data generation utilities, and a modular design for extending the system to new models.  

The goal of this project was not only to implement a functioning recommendation system but also to **understand the design challenges** behind real-world systems: working with noisy datasets, handling sparsity, optimizing training efficiency, and making the framework extensible for future research.  

---

## 1️⃣ Motivation

Recommendation systems are everywhere — from e-commerce platforms (Amazon, eBay) to entertainment (Netflix, Spotify). While off-the-shelf libraries exist, building one from scratch forces us to **confront the underlying problems**:

- **Data is messy:** Real-world datasets often contain duplicates, missing values, and extremely sparse interactions.  
- **Models must scale:** With millions of users and items, naive solutions won’t work.  
- **Cold-start issues:** How to handle new users or new products with few ratings?  
- **Flexibility:** One model rarely fits all. A modular design makes it easy to experiment with new methods.  

This project is my attempt to explore these issues by constructing a clean pipeline for **matrix factorization–based recommendation** and documenting the lessons learned.  

---

## 2️⃣ Data Exploration & Preprocessing

For this project, I worked with the **Amazon Reviews dataset**. The dataset provides millions of product reviews with fields like `userID`, `itemID`, `rating`, and `timestamp`.  

### Challenges
- **Sparsity:** A user typically interacts with only a handful of items compared to the full catalog. This leads to a sparse user–item interaction matrix.  
- **Cold-start:** Some users and items have only one or two ratings. Training on them directly increases noise.  
- **Inconsistency:** Duplicated reviews and missing values needed to be cleaned.  

### Solutions
- Filtered out users/items with fewer than *N* interactions.  
- Normalized IDs so they could be mapped into dense index ranges.  
- Stored interactions in a lightweight format for efficient training.  

The preprocessing pipeline is implemented in **`Amazon_data_process.py`** and produces clean train/test sets.  

---

## 3️⃣ Methodology

The core model used is **Matrix Factorization (MF)**, a classical collaborative filtering technique.  

### 3.1 The Idea
We represent:
- Each **user** \( u \) as a latent vector \( p_u \in \mathbb{R}^k \)  
- Each **item** \( i \) as a latent vector \( q_i \in \mathbb{R}^k \)  

The predicted rating is:
\[
\hat{r}_{ui} = p_u^\top q_i
\]

where \( k \) is the number of latent factors.  

### 3.2 Objective Function
We minimize the squared error with L2 regularization:
\[
\min_{P,Q} \sum_{(u,i) \in \mathcal{D}} (r_{ui} - p_u^\top q_i)^2 + \lambda \left( \|p_u\|^2 + \|q_i\|^2 \right)
\]

### 3.3 Optimization
- Implemented **Stochastic Gradient Descent (SGD)** to update user and item embeddings.  
- Added **learning rate decay** for stability.  
- Used **mini-batch training** via a custom data generator to balance speed and memory.  

---

## 4️⃣ Implementation Details

The repository is structured to emphasize **modularity and extensibility**:

- **`BaseModel.py`**  
  Defines the abstract base class (`fit`, `predict`, `evaluate`) that all models should implement.  

- **`MF.py`**  
  Contains the MF model, including:
  - Initialization of user/item embeddings.  
  - SGD updates with regularization.  
  - Training loop with evaluation.  

- **`datagenerator.py`**  
  Custom data loader that:
  - Shuffles data each epoch.  
  - Produces mini-batches.  
  - Efficiently handles sparse matrices.  

- **`main.py`**  
  Entry script to run the pipeline end-to-end.  

- **`Amazon_data_process.py`**  
  Cleans and preprocesses raw Amazon data.  

---

## 5️⃣ Experiments

### 5.1 Metrics
- **RMSE (Root Mean Squared Error):** Measures prediction accuracy.  
- **MAE (Mean Absolute Error):** Robust to outliers.  

### 5.2 Results
- With 20 latent factors, learning rate = 0.01, and regularization = 0.1:  
  - Train RMSE: ~0.92  
  - Test RMSE: ~0.95  
- Increasing factors improved fit but risked overfitting.  

### 5.3 Observations
- **Overfitting:** Without regularization, MF memorizes frequent users/items.  
- **Cold-start:** Performance drops sharply for users/items with <5 ratings.  
- **Tradeoff:** More latent factors improve accuracy but increase computation.  

### 5.4 Visualization
*(Optional: add plots of training loss over epochs, test RMSE vs latent dimensions.)*  

---

## 6️⃣ Lessons Learned

- **Data quality > Model complexity**  
  Cleaning and filtering the Amazon dataset improved results more than tweaking hyperparameters.  

- **Regularization is critical**  
  Without it, the model quickly overfit on frequent users/items.  

- **Design for flexibility**  
  Building a `BaseModel` class was worth it — now I can add new recommenders (e.g., Neural Collaborative Filtering, Autoencoders) with minimal changes.  

- **Evaluation must consider cold-start**  
  Filtering helps, but in production you’d need hybrid models (metadata, content features).  

---

## 7️⃣ Future Directions

- Implement **Neural Collaborative Filtering (NCF)**.  
- Explore **hybrid recommenders** using side information.  
- Add **cross-validation** for robust hyperparameter tuning.  
- Scale training with **parallel SGD** for large datasets.  

---

## 8️⃣ Usage

### Install
```bash
git clone https://github.com/fangsun6/Recommendation-System.git
cd Recommendation-System
pip install -r requirements.txt

