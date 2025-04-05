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
keras==3.9.2
matplotlib==3.10.1
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.6.1
tensorflow==2.19.0
tqdm==4.67.1
```

### Results

#### Stock Data

**Visualization**

<p float="left">
  <img src="../assets/pca.png" alt="PCA plot" width="400" />
  <img src="../assets/tsne.png" alt="TSNE plot" width="400" />
</p>