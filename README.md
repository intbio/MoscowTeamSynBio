# GFP Fluorescence Prediction Model

## Model Architecture
The fluorescence prediction system uses a deep learning approach with the following components:

### Data Processing Pipeline
1. **Input Filtering**:
   - Discards sequences containing invalid characters (`*`, `.`)
   - Processes only valid GFP mutant sequences

2. **Brightness Clustering**:
   - Uses KMeans to group sequences into 2 clusters (high/low brightness)
   - Selects only sequences from the high-brightness cluster (label=1) for training

3. **Feature Extraction**:
   - Generates protein embeddings using `esm2_t30_150M_UR50D` from HuggingFace
   - Embedding generation performed on NVIDIA RTX A5000 GPU (~40 minutes)

4. **Dataset Splitting**:
   - 70% training
   - 15% validation
   - 15% test

### Neural Network Design
```python
ResNet-style architecture:
- 10 sequential blocks
- Each block contains:
  * Batch normalization
  * ReLU activation
  * 2 linear layers with Dropout
  * Skip connection
- Optimizer: Adam (lr=2e-5)
- Loss: MAE
- Training: 2200 epochs
