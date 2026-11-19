# FraudLens: Unsupervised Credit Card Fraud Detection via Deep Bottleneck Autoencoder Architectures

**Harish G.$^{1}$**

$^{1}$ Department of Computer Science and Engineering  
Email: `harish.dl.research@institution.edu`

---

### Abstract
Digital financial transactions have proliferated exponentially, escalating the threat of credit card fraud. Detecting fraudulent activity is hindered by severe class imbalance—where legitimate transactions account for over 99.8% of observed volume—and concept drift, wherein fraudsters continuously mutate evasion strategies. Traditional supervised machine learning approaches suffer from majority-class bias and fail to generalize to novel attack vectors. In this paper, we propose **FraudLens**, an end-to-end semi-supervised deep learning framework that detects anomalous financial transactions using a symmetric Deep Bottleneck Autoencoder. The network is trained exclusively on 199,020 verified legitimate transactions to model the underlying manifold of normal commerce without exposure to fraud patterns. Reconstruction error measured via Mean Squared Error (MSE) serves as an objective anomaly score, with decision boundaries calibrated via F1-maximization on a validation partition. Evaluated on an untouched test set of 42,722 real-world transactions from the benchmark ULB dataset (containing 74 confirmed fraud instances), FraudLens achieves a Precision of **38.75%**, a Recall of **41.89%**, an F1-Score of **0.4026**, an Area Under ROC (ROC-AUC) of **0.9424**, and an Average Precision (PR-AUC) of **0.3403**. When benchmarked against an Isolation Forest baseline, FraudLens demonstrates a **2.11× improvement in F1-score** and a **3.60× improvement in Average Precision**, confirming the effectiveness of neural bottleneck representation for high-dimensional financial surveillance.

**Keywords:** Anomaly Detection $\cdot$ Deep Autoencoders $\cdot$ Credit Card Fraud $\cdot$ Class Imbalance $\cdot$ Precision-Recall AUC $\cdot$ Financial Security $\cdot$ Representation Learning.

---

## 1. Introduction

Global credit card fraud accounts for tens of billions of dollars in annual losses across card issuers and merchants [1]. As digital payments become ubiquitous, building automated fraud surveillance architectures capable of detecting unauthorized activity in real time has become paramount.

Automated fraud detection represents a difficult learning challenge due to three systemic characteristics:
1. **Extreme Class Imbalance**: Genuine consumer transactions overwhelmingly outnumber fraudulent events. In empirical benchmark datasets, fraudulent transactions account for fewer than **0.173%** of all recorded records. Supervised algorithms optimize for global zero-one loss, often predicting the majority class to achieve misleadingly high nominal accuracy (>99.8%) while failing on actual fraud cases [2].
2. **Concept Drift & Zero-Day Schemes**: Fraudsters dynamically modify their techniques to bypass rule engines and supervised signatures. Models trained strictly on past fraud topologies are brittle when encountering emerging fraud mechanisms [3].
3. **Label Latency and Asymmetry**: Confirming fraud typically depends on cardholder dispute filings and chargeback investigations, creating a multi-week lag before verified labels exist [4].

To resolve these constraints, this investigation proposes **FraudLens**, an unsupervised deep learning framework employing a Deep Fully Connected Autoencoder. Rather than attempting to learn an explicit boundary between classes from scarce fraud labels, FraudLens learns the compact mathematical manifold of **legitimate transaction dynamics**. Transactions diverging from this learned manifold produce elevated reconstruction residuals, naturally signaling an anomaly.

---

## 2. Related Work

### 2.1 Traditional Supervised Classification & Resampling
Early fraud detection relied heavily on rule-based heuristic systems, subsequently augmented by supervised algorithms such as Logistic Regression, Support Vector Machines (SVM), and Random Forests [5]. Because standard loss functions are dominated by legitimate samples, researchers often rely on synthetic oversampling algorithms (e.g., SMOTE, ADASYN) or undersampling techniques [6]. However, generating synthetic interpolations in anonymized Principal Component Analysis (PCA) subspaces risks creating artificial samples in low-density regions, leading to unacceptable false-positive rates for financial institutions.

### 2.2 Unsupervised Anomaly Detection
Unsupervised techniques model normal transactions directly. Classical approaches include Local Outlier Factor (LOF), One-Class Support Vector Machines (OC-SVM), and Isolation Forests [7]. While computationally efficient, tree-based and shallow distance-based methods degrade in high-dimensional feature spaces due to the curse of dimensionality and cannot easily capture nonlinear multi-feature correlations.

### 2.3 Deep Autoencoders for Anomaly Detection
Deep Autoencoders have demonstrated notable representational capacity across computer vision, acoustic anomaly detection, and industrial telemetry [8, 9]. By constraining the network through an informational bottleneck, the autoencoder is forced to reconstruct only the dominant, recurring patterns present in the training distribution. When evaluated on financial datasets, autoencoders provide an objective, continuous reconstruction loss that serves as an interpretable threat gauge [10].

---

## 3. Proposed Methodology: The FraudLens Architecture

### 3.1 Unsupervised Formulation
Let the input space be $\mathcal{X} = \mathbb{R}^D$, where $D = 30$ represents the transactional feature dimensions. Let $\mathcal{D}_{\text{train}} = \{\mathbf{x}_i\}_{i=1}^{N}$ represent a training set composed **strictly of legitimate transactions** ($y_i = 0$).

The autoencoder consists of two continuous mappings:
1. **Encoder** $f_\theta: \mathcal{X} \to \mathcal{Z}$, mapping the input $\mathbf{x} \in \mathbb{R}^{30}$ into a lower-dimensional latent bottleneck $\mathbf{z} \in \mathbb{R}^d$ ($d \ll D$):
   $$\mathbf{z} = f_\theta(\mathbf{x}) = \sigma \left( \mathbf{W}_e \mathbf{x} + \mathbf{b}_e \right)$$
2. **Decoder** $g_\phi: \mathcal{Z} \to \mathcal{X}$, reconstructing the latent code back into the original input space $\hat{\mathbf{x}} \in \mathbb{R}^{30}$:
   $$\hat{\mathbf{x}} = g_\phi(\mathbf{z}) = \sigma \left( \mathbf{W}_d \mathbf{z} + \mathbf{b}_d \right)$$

The optimization objective minimizes the Mean Squared Error (MSE) over the legitimate training distribution:
$$\mathcal{L}_{\text{MSE}}(\theta, \phi) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{D} \sum_{j=1}^{D} \left( x_{ij} - \hat{x}_{ij} \right)^2$$

### 3.2 Network Topology
FraudLens uses a symmetric bottleneck architecture with seven dense layers:

```
Input Dimension: 30 Features (Time, V1–V28, Amount)
│
├── [Encoder Layer 1] Dense(16, activation='relu')    --> W1: (30 x 16), b1: 16  [496 params]
├── [Encoder Layer 2] Dense(8, activation='relu')     --> W2: (16 x 8),  b2: 8   [136 params]
│
├── [Latent Bottleneck] Dense(4, activation='relu')   --> W3: (8 x 4),   b3: 4   [36 params]
│
├── [Decoder Layer 1] Dense(8, activation='relu')     --> W4: (4 x 8),   b4: 8   [40 params]
├── [Decoder Layer 2] Dense(16, activation='relu')    --> W5: (8 x 16),  b5: 16  [144 params]
│
└── [Output Layer]    Dense(30, activation='linear')  --> W6: (16 x 30), b6: 30  [510 params]

Total Trainable Parameters: 1,362 (5.32 KB)
```

The compression ratio from input space ($D=30$) to bottleneck ($d=4$) is $7.5:1$, compelling the network to retain only the essential co-dependencies of normal transactions.

### 3.3 Reconstruction Error and Dual Threshold Calibration
For any sample $\mathbf{x}_k$, its scalar anomaly score $S(\mathbf{x}_k)$ is defined as the mean squared residual:
$$S(\mathbf{x}_k) = \frac{1}{D} \sum_{j=1}^{D} \left( x_{kj} - \hat{x}_{kj} \right)^2$$

The binary classification decision rule is:
$$\hat{y}_k = \begin{cases} 1 \quad (\text{"Potential Fraud"}), & \text{if } S(\mathbf{x}_k) > \tau \\ 0 \quad (\text{"Normal"}), & \text{if } S(\mathbf{x}_k) \le \tau \end{cases}$$

To establish the operational threshold $\tau$, FraudLens implements two strategies evaluated on validation data:
- **Strategy A (Percentile-Based / Unsupervised)**: $\tau_{\text{perc}} = P_{99.0}(S(\mathbf{x}_{\text{val, norm}}))$, flagging the top 1% highest reconstruction errors among normal validation samples.
- **Strategy B (F1-Optimal Calibration / Semi-Supervised)**: Sweeping candidate thresholds $\tau \in [\min(S), \max(S)]$ over the labeled validation set to maximize the harmonic balance between fraud precision and recall:
  $$\tau^* = \arg\max_{\tau} F_1(\tau) = \arg\max_{\tau} \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}$$

---

## 4. Experimental Setup

### 4.1 Dataset Properties
Experiments are conducted on the benchmark Kaggle Credit Card Fraud dataset collected by Worldline and the Université Libre de Bruxelles (ULB) [2]. The dataset contains transactions made by European cardholders over a two-day period in September 2013:
- **Total Transactions**: 284,807
- **Legitimate Transactions ($y=0$)**: 284,315 (99.8273%)
- **Fraudulent Transactions ($y=1$)**: 492 (0.1727%)
- **Feature Set**: 30 input columns consisting of `Time`, 28 anonymized principal components (`V1`–`V28`), and transaction `Amount`.

### 4.2 Data Partitioning and Leakage Prevention
To evaluate generalization on unseen data, the dataset is divided using stratified sampling:
- **Training Set (70%)**: 199,364 transactions total. The 344 fraud records in this partition are removed, leaving a training pool of **199,020 legitimate transactions**.
- **Validation Set (15%)**: 42,721 transactions (42,647 legitimate, 74 fraudulent) used strictly for early stopping and threshold calibration.
- **Test Set (15%)**: 42,722 transactions (42,648 legitimate, 74 fraudulent), held out and untouched until final evaluation.

A `StandardScaler` is fitted **solely on the 199,020 normal training records**. All validation and test partitions are transformed using this frozen scaler, preventing statistical leakage.

### 4.3 Training Hyperparameters
- **Optimizer**: Adam ($\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$)
- **Batch Size**: 256
- **Maximum Epochs**: 50
- **Regularization / Early Stopping**: Monitored on validation loss ($\mathcal{L}_{\text{val}}$) with patience = 5. Model weights from the lowest validation loss epoch are restored.

---

## 5. Experimental Results and Discussion

### 5.1 Training Convergence Dynamics
The autoencoder was trained for 48 epochs before EarlyStopping triggered weight restoration. The training loss decreased monotonically from an initial MSE of $0.6378$ to a stabilized minimum of $0.5184$, with validation loss tracking at $0.5258$. The absence of divergent validation loss confirms robust convergence without overfitting.

### 5.2 Held-Out Test Set Performance
Evaluating the model on the 42,722 untouched test transactions using the calibrated threshold $\tau^* = 16.3419$ yields the performance summarized in Table 1.

**Table 1.** Comprehensive performance evaluation on held-out test partition ($N=42,722$).

| Evaluation Metric | FraudLens Deep Autoencoder | Isolation Forest Baseline | Absolute Improvement |
|:---|:---:|:---:|:---:|
| **Operating Paradigm** | Deep Neural Manifold | Randomized Tree Partitioning | Manifold Representation |
| **Precision** | **38.75%** | 17.02% | **+21.73%** |
| **Recall (Sensitivity)** | **41.89%** | 21.62% | **+20.27%** |
| **F1-Score** | **0.4026** | 0.1905 | **+0.2121 (2.11×)** |
| **ROC-AUC** | **0.9424** | 0.9409 | **+0.0015** |
| **Average Precision (PR-AUC)** | **0.3403** | 0.0945 | **+0.2458 (3.60×)** |
| **Accuracy** | 99.78% | 99.76% | +0.02% |

### 5.3 Confusion Matrix Decomposition
The absolute classification outcomes on the held-out test set are structured as follows:
- **True Negatives (TN)**: 42,599 legitimate transactions correctly passed.
- **False Positives (FP)**: 49 legitimate transactions flagged as potential anomalies (False Alarm Rate = 0.11%).
- **False Negatives (FN)**: 43 fraudulent transactions undetected by the threshold.
- **True Positives (TP)**: 31 fraudulent transactions intercepted.

```
                  Predicted Normal     Predicted Fraud
Actual Normal         42,599 (TN)           49 (FP)
Actual Fraud              43 (FN)           31 (TP)
```

In a commercial banking workflow, processing 42,722 transactions generated only 49 false alarms (a 0.11% investigator audit rate), while capturing 41.89% of all fraud attacks. Adjusting the threshold to the Percentile Strategy ($\tau = 2.7498$) boosts Recall to **82.43%**, accommodating institutions prioritizing maximum fraud capture.

### 5.4 Error Distribution Disparity
The fundamental mechanism driving FraudLens is the disparity in reconstruction error between classes:
- **Normal Transactions**: $\mu_{\text{error}} = 0.524$, $\sigma_{\text{error}} = 0.812$ (tightly clustered near origin).
- **Fraudulent Transactions**: $\mu_{\text{error}} = 27.638$, $\sigma_{\text{error}} = 41.902$ (heavy-tailed, extending past 100.0).

This 52-fold difference in mean reconstruction error confirms that the 4-dimensional bottleneck effectively prevents the network from reconstructing anomalous non-manifold inputs.

---

## 6. System Implementation & Surveillance Dashboard

To bridge algorithmic modeling with operational deployment, FraudLens was integrated into an interactive web dashboard engineered with Python and Streamlit. The system includes:
1. **Executive Surveillance Dashboard**: Summarizes global transaction throughput, current anomaly alerts, and dynamic confusion donuts.
2. **Batch Audit Pipeline**: Accepts standard banking CSV files, processes records vectorially through the Keras model, and provides downloadable audit manifests.
3. **Forensic Inspection Engine**: Displays single-transaction diagnostics, an interactive Plotly gauge of reconstruction error versus safety threshold, and a decomposition bar chart detailing the top 8 features contributing to the anomaly.
4. **Model Performance Audit**: Houses diagnostic plots, precision-recall trajectories, and the Isolation Forest benchmark comparison table.

---

## 7. Limitations and Future Work

1. **Lack of Sequential Temporal Modeling**: Transactions are evaluated as independent vectors. Integrating Recurrent Neural Networks (LSTM) or Temporal Graph Convolutional Networks (GCN) could capture velocity changes across sequential transactions.
2. **Interpretability Constraints**: While per-feature squared residuals indicate which features deviated, they do not explain high-order feature interactions. Future work will integrate Integrated Gradients to trace error contributions back to specific merchant categories and velocity patterns.
3. **Dynamic / Tiered Thresholding**: Deploying multi-tiered thresholds (Low, Medium, High Risk) can help fraud management teams prioritize manual investigations.

---

## 8. Conclusion

This paper introduced **FraudLens**, an unsupervised deep learning framework for credit card fraud detection utilizing a Deep Bottleneck Autoencoder. By training exclusively on legitimate transactions, the system bypasses class imbalance issues and detects fraudulent activity via reconstruction error thresholds. On an untouched test set of 42,722 transactions, FraudLens achieved an F1-score of **0.4026** and an Average Precision of **0.3403**, outperforming an Isolation Forest baseline by **2.11×** and **3.60×**, respectively, with a low false alarm rate of 0.11%. These results confirm that deep autoencoders provide an effective, robust foundation for modern real-time financial fraud surveillance.

---

## References

1. Worldline Global: Financial Fraud and Card Crime Trends. Technical Report, Worldline Corp (2022)
2. Dal Pozzolo, A., Caelen, O., Johnson, R.A., Bontempi, G.: Calibrating Probability with Undersampling for Unbalanced Classification. In: IEEE Symposium on Computational Intelligence and Data Mining (CIDM), pp. 159–166 (2015)
3. Fawcett, T., Provost, F.: Adaptive Fraud Detection. Data Mining and Knowledge Discovery **1**(3), 291–316 (1997)
4. Carminati, M., Caron, R., Maggi, F., Epifani, I., Zanero, S.: BankSealer: A Decision Support System for Online Banking Fraud Analysis and Mitigation. Computers & Security **53**, 175–186 (2015)
5. Bhattacharyya, S., Jha, S., Tharakunnel, K., Westland, J.C.: Data Mining for Credit Card Fraud: A Comparative Study. Decision Support Systems **50**(3), 602–613 (2011)
6. Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P.: SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research **16**, 321–357 (2002)
7. Liu, F.T., Ting, K.M., Zhou, Z.H.: Isolation Forest. In: Eighth IEEE International Conference on Data Mining (ICDM), pp. 413–422 (2008)
8. Goodfellow, I., Bengio, Y., Courville, A.: Deep Learning. MIT Press, Cambridge (2016)
9. Chandola, V., Banerjee, A., Kumar, V.: Anomaly Detection: A Survey. ACM Computing Surveys **41**(3), 1–58 (2009)
10. Chen, Z., Cavanaugh, C.: Deep Autoencoder for Credit Card Fraud Detection. In: Proceedings of the International Conference on Machine Learning and Cybernetics (ICMLC), pp. 143–151 (2020)
