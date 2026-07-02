# Facial Recognition Comparison System

A comprehensive dual-pipeline facial recognition benchmark system that compares classical computer vision (SIFT + Bag-of-Visual-Words + KNN) against modern deep learning (PyTorch ResNet/MobileNet) approaches on identical data.

## Features

### Classical Pipeline
- **SIFT Extraction**: Extract invariant keypoint descriptors
- **K-Means Vocabulary**: Build visual vocabulary with configurable cluster count
- **Bag-of-Visual-Words (BoVW)**: Encode images as visual word histograms
- **TF-IDF Weighting**: Apply statistical term weighting
- **KNN Classification**: Classify with k-nearest neighbors
- **Verification Mode**: Optional RANSAC-based geometric verification

### Deep Learning Pipeline
- **Backbone Models**: ResNet-18, ResNet-50, MobileNet, EfficientNet
- **Transfer Learning**: ImageNet pre-training with fine-tuning
- **ArcFace Loss**: Optional margin-based loss for improved separation
- **Data Augmentation**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- **Multiple Evaluation Modes**: 
  - Classification (softmax probabilities)
  - Embedding-based (KNN or cosine similarity on learned embeddings)

### Shared Infrastructure
- **Multi-backend Face Detection**: Haar Cascade, DNN, MTCNN, RetinaFace
- **Stratified Data Splitting**: Guarantees per-identity distribution preservation
- **Config-driven**: Single `config.yaml` controls all experiments
- **Artifact Management**: Automatic save/load of trained models with metadata tracking
- **Comprehensive Metrics**: Top-1/5 accuracy, precision, recall, F1, confusion matrices
- **Comparison Plots**: 6 visualizations comparing both pipelines

## Project Structure

```
facial-recognition/
├── config.yaml                      # Master configuration file
├── requirements.txt                 # Python dependencies
├── preprocess.py                    # Data preprocessing script
├── train.py                         # Model training script
├── evaluate.py                      # Model evaluation script
├── compare.py                       # Comparison and visualization script
├── README.md                        # This file
├── data/
│   ├── kaggle/                      # Kaggle dataset (optional)
│   └── vggface2/                    # VGGFace2 dataset (optional)
├── outputs/
│   ├── splits/                      # Train/val/test split metadata
│   ├── artifacts/
│   │   ├── classical/               # Classical pipeline artifacts
│   │   └── deep/                    # Deep learning checkpoints
│   ├── results/                     # Metrics JSON/CSV
│   └── plots/                       # Comparison visualizations
└── src/
    ├── preprocessing/               # Data loading, face detection, splitting
    ├── classical/                   # SIFT, BoVW, KNN, verification
    ├── deep/                        # PyTorch models and training
    ├── evaluation/                  # Metrics, reporting, visualization
    └── utils/                       # Config, logging, artifacts, exceptions
```

## Installation

### 1. Prerequisites
- Python 3.8+
- pip or conda

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n facial-recognition python=3.9
conda activate facial-recognition
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import cv2; import sklearn; print('All dependencies installed!')"
```

## Configuration

### config.yaml Structure

The `config.yaml` file controls all aspects of the system:

```yaml
# Dataset configuration
dataset:
  path: "data/kaggle"                 # Path to dataset root
  type: "kaggle"                      # "kaggle" or "vggface2"
  max_identities: null                # Limit classes (null = all)
  max_images_per_identity: null       # Limit images per class (null = all)

# Face detection and preprocessing
preprocessing:
  detector_backend: "mtcnn"           # "haar", "dnn", "mtcnn", or "retinaface"
  no_face_fallback: "skip"            # "skip" or "use_full"
  image_size: [224, 224]              # [width, height]
  norm_mean: [0.485, 0.456, 0.406]    # ImageNet normalization mean
  norm_std: [0.229, 0.224, 0.225]     # ImageNet normalization std

# Train/val/test splitting
splitting:
  ratios: [0.7, 0.15, 0.15]           # [train, val, test]
  random_seed: 42                     # For reproducibility
  metadata_path: "outputs/splits/split_index.csv"
  force_resplit: false                # Force recomputation if true

# Classical pipeline
classical:
  vocab_size: 1000                    # K-means clusters
  kmeans_max_iter: 300                # Max K-means iterations
  knn_k: 5                            # Number of neighbors
  knn_metric: "euclidean"             # Distance metric
  artifacts_dir: "outputs/artifacts/classical"

# Verification mode (classical)
verification:
  enabled: false                      # Enable verification mode
  threshold: 0.5                      # Similarity threshold [0, 1]
  ransac_enabled: false               # Use RANSAC reranking

# Deep learning pipeline
deep:
  architecture: "resnet50"            # Backbone model
  pretrained: true                    # Use ImageNet weights
  epochs: 20                          # Training epochs
  optimizer: "adam"                   # "adam" or "sgd"
  learning_rate: 0.001                # Initial learning rate
  batch_size: 32                      # Mini-batch size
  checkpoint_path: "outputs/artifacts/deep/best_checkpoint.pt"
  eval_mode: "classification"         # "classification" or "embedding"
  embedding_classifier: "knn"         # "knn" or "cosine" (embedding mode)
  cosine_threshold: 0.6               # Similarity threshold (cosine mode)
  arcface_enabled: false              # Use ArcFace loss
  arcface_margin: 0.5                 # ArcFace margin
  arcface_scale: 64                   # ArcFace scale

# Evaluation
evaluation:
  roc_enabled: false                  # Compute ROC curves
  output_dir: "outputs"
  results_dir: "outputs/results"
  plots_dir: "outputs/plots"
```

### Overriding Configuration from CLI

You can override any config parameter from the command line:

```bash
# Limit dataset size
python train.py --pipeline classical --config config.yaml --dataset.max_identities=50

# Change face detector
python preprocess.py --config config.yaml --preprocessing.detector_backend=haar

# Adjust learning rate
python train.py --pipeline deep --config config.yaml --deep.learning_rate=0.0005

# Modify epochs
python train.py --pipeline deep --config config.yaml --deep.epochs=100
```

## Usage

### Step 1: Prepare Dataset

Your dataset should be organized as:
```
data/kaggle/
├── identity_1/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── identity_2/
│   ├── image_1.jpg
│   └── ...
└── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### Step 2: Preprocess Data

Run face detection, cropping, and train/val/test splitting:

```bash
python preprocess.py --config config.yaml
```

**Output**: `outputs/splits/split_index.csv` containing split metadata

**Options**:
- `--config`: Configuration file path (default: `config.yaml`)
- `--preprocessing.detector_backend`: Face detector backend
- `--preprocessing.image_size`: Output image dimensions

### Step 3: Train Models

#### Train Classical Pipeline (SIFT + BoVW + KNN)

```bash
python train.py --pipeline classical --config config.yaml
```

**Output**:
- `outputs/artifacts/classical/kmeans_vocab.pkl` - K-means vocabulary
- `outputs/artifacts/classical/tfidf_transformer.pkl` - TF-IDF weights
- `outputs/artifacts/classical/knn_classifier.pkl` - Trained KNN
- `outputs/artifacts/classical/run_manifest.json` - Config snapshot

**Duration**: Typically 2-10 minutes depending on dataset size

#### Train Deep Learning Pipeline (PyTorch)

```bash
python train.py --pipeline deep --config config.yaml
```

**Output**:
- `outputs/artifacts/deep/best_checkpoint.pt` - Best model checkpoint
- `outputs/artifacts/deep/run_manifest.json` - Config snapshot

**Options**:
- `--deep.architecture`: Model backbone (resnet18, resnet50, mobilenet, efficientnet)
- `--deep.epochs`: Number of training epochs
- `--deep.batch_size`: Mini-batch size
- `--deep.learning_rate`: Initial learning rate
- `--deep.eval_mode`: Classification or embedding-based evaluation

**Duration**: Typically 5-30 minutes depending on dataset size and epochs

### Step 4: Evaluate Models

#### Evaluate Classical Pipeline

```bash
python evaluate.py --pipeline classical --config config.yaml
```

#### Evaluate Deep Learning Pipeline

```bash
python evaluate.py --pipeline deep --config config.yaml
```

**Output**:
- `outputs/results/<pipeline>_metrics.json` - Full metrics and confusion matrix
- `outputs/results/<pipeline>_metrics.csv` - Tabular metrics summary

**Metrics Computed**:
- Top-1 and Top-5 accuracy
- Macro/micro precision, recall, F1
- Per-class accuracy
- Confusion matrix
- Inference timing

### Step 5: Generate Comparison Report

```bash
python compare.py --config config.yaml
```

**Output**:
- 6 comparison plots in `outputs/plots/`:
  - `accuracy_comparison.png`
  - `f1_comparison.png`
  - `training_time_comparison.png`
  - `inference_time_comparison.png`
  - `classical_confusion_matrix.png`
  - `deep_confusion_matrix.png`
- `outputs/results/comparison_report.json` - Side-by-side metrics

## Complete Workflow Example

```bash
# 1. Preprocess dataset
python preprocess.py --config config.yaml

# 2. Train both pipelines
python train.py --pipeline classical --config config.yaml
python train.py --pipeline deep --config config.yaml --deep.epochs=30

# 3. Evaluate both pipelines
python evaluate.py --pipeline classical --config config.yaml
python evaluate.py --pipeline deep --config config.yaml

# 4. Generate comparison
python compare.py --config config.yaml
```

## Advanced Usage

### Using Different Face Detectors

```bash
# Haar Cascade (fast, but less accurate)
python preprocess.py --preprocessing.detector_backend=haar

# DNN (OpenCV deep neural network)
python preprocess.py --preprocessing.detector_backend=dnn

# MTCNN (accurate, slower)
python preprocess.py --preprocessing.detector_backend=mtcnn

# RetinaFace (state-of-the-art, slowest)
python preprocess.py --preprocessing.detector_backend=retinaface
```

### Using Different Deep Learning Architectures

```bash
# ResNet-18 (smaller, faster)
python train.py --pipeline deep --deep.architecture=resnet18

# ResNet-50 (default, balanced)
python train.py --pipeline deep --deep.architecture=resnet50

# MobileNet (mobile-friendly, smaller)
python train.py --pipeline deep --deep.architecture=mobilenet

# EfficientNet (state-of-the-art efficiency)
python train.py --pipeline deep --deep.architecture=efficientnet
```

### Embedding-based Deep Learning Evaluation

For embedding-based classification instead of softmax:

```bash
# Edit config.yaml:
# deep:
#   eval_mode: "embedding"
#   embedding_classifier: "knn"        # or "cosine"

python train.py --pipeline deep --config config.yaml
python evaluate.py --pipeline deep --config config.yaml
```

### Using ArcFace Loss

For improved class separation in deep learning:

```bash
python train.py --pipeline deep --config config.yaml \
  --deep.arcface_enabled=true \
  --deep.arcface_margin=0.5 \
  --deep.arcface_scale=64
```

### Re-running with Different Settings

To force resplitting (ignore previous split):

```bash
python preprocess.py --config config.yaml --splitting.force_resplit=true
```

To retrain from scratch (ignore cached artifacts):

```bash
python train.py --pipeline classical --config config.yaml --classical.retrain=true
```

## Troubleshooting

### Face Detector Initialization Error

```
DetectorInitError: Failed to initialize MTCNN detector
```

**Solution**: Try a different detector backend:
```bash
python preprocess.py --preprocessing.detector_backend=haar
```

### No Samples Loaded from Dataset

```
DatasetError: No samples loaded from dataset
```

**Solution**: Verify dataset structure and path:
- Ensure images are in `data/kaggle/identity_name/image.jpg` format
- Check `config.yaml` has correct `dataset.path`
- Verify image file extensions are supported

### Missing Artifact Error

```
ArtifactNotFoundError: Artifact not found: 'outputs/artifacts/classical/kmeans_vocab.pkl'
```

**Solution**: Run training script first:
```bash
python train.py --pipeline classical --config config.yaml
```

### Out of Memory (OOM) Error

**Solution**: Reduce batch size or dataset size:
```bash
python train.py --pipeline deep --config config.yaml \
  --deep.batch_size=16 \
  --dataset.max_identities=50
```

### GPU Not Being Used

**Solution**: Verify CUDA installation:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Performance Tips

1. **Use MTCNN or RetinaFace** for better face detection accuracy (slower)
2. **Use ResNet-50 or EfficientNet** for better deep learning accuracy
3. **Increase batch_size** (if GPU memory allows) for faster training
4. **Reduce image_size** from [224, 224] to [112, 112] for 4x speedup
5. **Use pretrained=true** to leverage ImageNet knowledge
6. **Enable ArcFace loss** for harder classification problems

## Requirements

### Python Packages

See `requirements.txt` for complete dependencies. Key packages:

- `opencv-python`: Image processing and Haar Cascade face detection
- `numpy`: Numerical computations
- `scikit-learn`: K-means, KNN, TF-IDF, metrics
- `torch` & `torchvision`: Deep learning framework
- `matplotlib`: Visualization
- `pyyaml`: Configuration management
- `pandas`: Data handling

### Hardware

- **Minimum**: 4GB RAM, modern CPU
- **Recommended**: 8GB+ RAM, NVIDIA GPU (CUDA 11.8+)

### Supported Operating Systems

- Linux (Ubuntu 18.04+)
- macOS (10.14+)
- Windows 10/11

## Citation

If you use this system in your research, please cite:

```bibtex
@software{facial_recognition_comparison_2024,
  title={Facial Recognition Comparison System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo/facial-recognition}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributors

- **Kiro**: System design and planning
- **Your Team**: Implementation

## Support

For issues, questions, or contributions:

1. Check the Troubleshooting section above
2. Review `config.yaml` comments
3. Check logs in console output for detailed error messages
4. Open an issue with:
   - Your OS and Python version
   - The exact command you ran
   - Error message and traceback
   - Output from `python -c "import sys; print(sys.version)"`

## Roadmap

- [ ] Real-time face recognition from webcam
- [ ] Multi-GPU training support
- [ ] REST API for inference
- [ ] Web UI for dataset management
- [ ] Federated learning support
- [ ] Mobile deployment (TFLite, ONNX)

---

**Last Updated**: June 2024
**Status**: Production Ready ✅
