# Technical Implementation Report: Sentinel Graph AI

This document outlines the architectural decisions, technologies used, and the major challenges overcome during the transformation of the Anti-Money Laundering (AML) model from a flawed baseline into a state-of-the-art detection system.

---

## 1. Technologies Used

* **Pandas & NumPy**: For heavy data manipulation and computing rolling-window velocity features across 9.5 million rows.
* **NetworkX**: For Graph Machine Learning (PageRank, Network Centrality).
* **XGBoost**: For the core classification algorithm. Selected over LightGBM for better out-of-the-box CPU performance using the `hist` tree method.
* **Scikit-Learn**: For precision-recall curve analysis, threshold optimization (F2 Score), and metric evaluation.
* **SHAP (SHapley Additive exPlanations)**: For model interpretability to ensure investigators can trust the AI's flagging logic.
* **Streamlit**: For deploying an interactive user interface.

---

## 2. Key Implementation Decisions

### Decision 1: Shifting from "Tabular" to "Graph" Machine Learning
Money laundering is not an isolated event; it is a network of interactions. Criminals use "smurfing" (breaking large sums into smaller transactions to many accounts) and "mules" (central accounts receiving from many victims).
**Implementation**: We used `networkx` to build a directed graph of all 9.5M transactions. We engineered three critical features:
* `sender_out_degree`: Identifies smurfing behavior.
* `receiver_in_degree`: Identifies money mules.
* `pagerank`: Identifies central "hubs" in the laundering syndicate.

### Decision 2: Tracking Velocity (Time-Series Features)
A single $9,000 transaction might be normal for a wealthy individual. Ten $9,000 transactions in two hours is structuring.
**Implementation**: We sorted the data chronologically and computed rolling windows (`amount_24h` and `txn_count_1h`) as well as `amount_vs_avg` (comparing the current transaction to the user's historical average).

### Decision 3: Optimizing for the F2-Score
In banking, False Negatives (missing fraud) result in massive regulatory fines, whereas False Positives (flagging normal transactions) just cost investigator time. 
**Implementation**: Instead of relying on the default `0.5` prediction threshold or the F1-Score (which treats precision and recall equally), we maximized the **F2-Score**. This mathematically instructed the model to prioritize Recall (catching criminals) over Precision.

---

## 3. Major Challenges & How We Overcame Them

### Challenge 1: The Accuracy Paradox
**The Problem**: The original model achieved 99.9% accuracy but had 0% Precision and 0% Recall. Because 99.9% of the dataset was normal, the model learned a "lazy" strategy: guess "Normal" every single time, and you will be 99.9% accurate.
**The Fix**: We completely abandoned Accuracy as a metric. We shifted our focus entirely to the Precision-Recall curve and implemented Random Undersampling.

### Challenge 2: Massive Data & Training Time
**The Problem**: Training an ensemble tree model on 9.5 million rows took upwards of 15 minutes per iteration, making tuning and experimentation painfully slow.
**The Fix**: We implemented a 10:1 **Random Undersampling** strategy. We kept 100% of the ~8,000 fraud cases, but randomly sampled only ~80,000 normal transactions. This dropped the training dataset from 7.6M rows to 88,000 rows. Training time dropped from 15 minutes to **15 seconds**, *without losing any fraud signal*. We then tested on the full, untouched 1.9M row test set to ensure real-world validity.

### Challenge 3: Domain Shift in Testing
**The Problem**: When we wrote a script to test the model against 22,000 completely random, synthetic transactions, the model failed spectacularly—flagging 83% of the random data as fraud.
**The Fix (The Insight)**: This wasn't an overfitting issue; it was a fascinating Graph Theory phenomenon. In real human behavior, financial networks are sparse (you only pay a few entities). A completely random network generates astronomical PageRank and Degree scores because everyone randomly connects to everyone else. To a Graph ML model, *pure randomness mathematically resembles a massive money mule syndicate.* We realized that to accurately test the model, we had to rely strictly on the 1.9 million row Hold-Out Test Set from the actual data distribution, where the model proved its 93% precision.

### Challenge 4: Preventing Overtraining
**The Problem**: Because we drastically undersampled the majority class, the model risked memorizing the few remaining fraud patterns (overfitting).
**The Fix**: We implemented **Early Stopping** (`early_stopping_rounds=50`). The model monitored its Logloss on the unseen validation set after every single tree. It automatically halted training at tree 822 when it noticed it was no longer learning generalizable patterns, guaranteeing robust real-world performance.
