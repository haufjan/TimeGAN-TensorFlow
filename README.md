# TimeGAN-TensorFlow
Unofficial implementation of TimeGAN (Yoon et al., NIPS 2019) in TensorFlow 2.

Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019

Original Codebase: https://github.com/jsyoon0823/TimeGAN.git

### Data Set Reference
-  Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
-  Energy data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

### Version Notes
The model was implemented and tested using `Python==3.11.9`. Further, the following modules were utilized (see [Requirements File](./requirements.txt)):
```
keras==3.10.0
matplotlib==3.10.3
numpy==2.1.3
pandas==2.3.0
scikit-learn==1.7.0
tensorflow==2.19.0
tqdm==4.67.1
```

### Results

#### Stock Data

**1. Discriminative Score**

```python
# Compute discriminative score
discriminative_score_metrics(data_train, data_gen)
```
100%|██████████| 2000/2000 [03:54<00:00,  8.52it/s]

0.4924965893587995

**2. Predictive Score**

```python
# Compute predictive score
predictive_score_metrics(data_train, data_gen)
```
100%|██████████| 5000/5000 [08:38<00:00,  9.64it/s]
    
0.03887795482822757

**3. Visualization**

<p float="left">
  <img src="../assets/pca.png" alt="PCA plot" width="400" />
  <img src="../assets/tsne.png" alt="TSNE plot" width="400" />
</p>