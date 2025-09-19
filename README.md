# GP-Music-Tagging

## Constructing Composite Features for Interpretable Music-Tagging

This repository contains the implementation and experimental results for our paper **"CONSTRUCTING COMPOSITE FEATURES FOR INTERPRETABLE MUSIC-TAGGING"** submitted to **ICASSP 2026**.

### Abstract

Combining multiple audio features can improve the performance of music tagging, but common deep learning-based feature fusion methods often lack interpretability. To address this problem, we propose a Genetic Programming (GP) pipeline that automatically evolves composite features by mathematically combining base music features, thereby capturing synergistic interactions while preserving interpretability. This approach provides representational benefits similar to deep feature fusion without sacrificing interpretability. Experiments on the MTG-Jamendo and GTZAN datasets demonstrate consistent improvements compared to state-of-the-art systems across base feature sets at different abstraction levels. It should be noted that most of the performance gains are noticed within the first few hundred GP evaluations, indicating that effective feature combinations can be identified under modest search budgets. The top evolved expressions include linear, nonlinear, and conditional forms, with various low-complexity solutions at top performance aligned with parsimony pressure to prefer simpler expressions. Analyzing these composite features further reveals which interactions and transformations tend to be beneficial for tagging, offering insights that remain opaque in black-box deep models.

### Authors

**Chenhao Xue¹**, Weitao Hu², Joyraj Chakraborty¹, Zhijin Guo¹, Kang Li¹, Tianyu Shi³, Martin Reed⁴, Nikolaos Thomos⁴

¹University of Oxford  
²Independent Researcher  
³University of Toronto  
⁴University of Essex

## Repository Structure

```
├── Codes/
│   ├── Embedding/                              # Feature extraction notebooks
│   │   ├── Embedding_extractor_musicnn_wav2vec.ipynb
│   │   ├── Embedding_extractor_musicnn_wav2vec_10k-14k.ipynb
│   │   ├── Embedding_extractor_musicnn_wav2vec_14k-all.ipynb
│   │   ├── Embedding_extractor_musicnn_wav2vec_6k-10k.ipynb
│   │   ├── Embedding_extractor_wav2vec.ipynb
│   │   └── Embedding_extractor_wav2vec_GTZAN.ipynb
│   ├── GP_for_GTZAN_ValTest.ipynb             # GP experiments on GTZAN dataset
│   ├── GP_for_MTG-Jamendo_ValTest.ipynb       # GP experiments on MTG-Jamendo dataset
│   ├── GTZAN_Expressions_Analysis.ipynb        # Analysis of GP expressions for GTZAN
│   ├── MTG_Jamendo_Expressions_Analysis.ipynb  # Analysis of GP expressions for MTG-Jamendo
│   └── M0_baselines_training(1).ipynb         # Baseline model training
├── GTZAN-data/                                 # GTZAN dataset results and expressions
└── mtg-jamendo-dataset/                        # MTG-Jamendo dataset results and expressions
```

## Key Features

- **Interpretable Feature Fusion**: Uses Genetic Programming to evolve mathematical expressions that combine base audio features
- **Multi-dataset Evaluation**: Comprehensive experiments on both MTG-Jamendo and GTZAN datasets
- **Expression Analysis**: Detailed analysis of evolved GP expressions revealing feature interaction patterns
- **Baseline Comparisons**: Evaluation against traditional feature fusion methods
- **Efficient Search**: Performance gains emerge within few hundred GP evaluations

## Methodology

### 1. Base Feature Extraction
Extract various audio features at different abstraction levels:
- **MusicNN embeddings**: Deep learning-based music feature representations
- **Wav2Vec embeddings**: Self-supervised speech representation learning adapted for music

### 2. GP Evolution Pipeline
- Automatically evolve composite features through genetic programming
- Apply parsimony pressure to maintain interpretable, low-complexity solutions
- Generate mathematical expressions combining base features

### 3. Feature Evaluation
- Test evolved features on music tagging tasks
- Compare against baseline methods and individual base features

### 4. Expression Analysis
- Analyze mathematical forms of successful composite features
- Identify beneficial feature interactions and transformations
- Extract interpretable insights about effective feature combinations

## Datasets

- **MTG-Jamendo**: Large-scale music tagging dataset with diverse musical content and multi-label annotations
- **GTZAN**: Genre classification dataset for music information retrieval benchmarking

## Key Findings

- ✅ Performance improvements emerge within the first few hundred GP evaluations
- ✅ Evolved expressions include linear, nonlinear, and conditional forms
- ✅ Low-complexity solutions align with GP's parsimony pressure
- ✅ Feature interactions provide insights difficult to obtain from black-box models
- ✅ Consistent improvements across different base feature abstraction levels

## Usage

### Prerequisites
```bash
pip install numpy pandas scikit-learn tensorflow torch librosa deap jupyter
```

### 1. Feature Extraction
Use the notebooks in `Codes/Embedding/` to extract audio embeddings:

**For MTG-Jamendo dataset:**
- `Embedding_extractor_musicnn_wav2vec.ipynb` - Extract MusicNN and Wav2Vec features
- `Embedding_extractor_musicnn_wav2vec_*k-*k.ipynb` - Batch processing for large datasets

**For GTZAN dataset:**
- `Embedding_extractor_wav2vec_GTZAN.ipynb` - GTZAN-specific feature extraction

### 2. GP Experiments
Run the main GP experiments:

**GTZAN experiments:**
```bash
jupyter notebook Codes/GP_for_GTZAN_ValTest.ipynb
```

**MTG-Jamendo experiments:**
```bash
jupyter notebook Codes/GP_for_MTG-Jamendo_ValTest.ipynb
```

### 3. Baseline Training
Train baseline models for comparison:
```bash
jupyter notebook Codes/M0_baselines_training(1).ipynb
```

### 4. Expression Analysis
Analyze the evolved GP expressions:

**GTZAN analysis:**
```bash
jupyter notebook Codes/GTZAN_Expressions_Analysis.ipynb
```

**MTG-Jamendo analysis:**
```bash
jupyter notebook Codes/MTG_Jamendo_Expressions_Analysis.ipynb
```

## Experimental Results

### Performance Improvements
- Consistent improvements across both datasets
- Effective feature combinations discovered with modest computational budget
- Superior performance compared to individual base features and traditional fusion methods

### Expression Complexity
- Evolved solutions maintain interpretability through parsimony pressure
- Mathematical expressions include diverse forms: linear combinations, nonlinear transformations, conditional operations
- Low-complexity solutions demonstrate the effectiveness of GP's search strategy

## Dependencies

- Python 3.7+
- NumPy
- Pandas  
- Scikit-learn
- TensorFlow/PyTorch (for neural network baselines)
- DEAP (for Genetic Programming)
- Librosa (for audio processing)
- Jupyter Notebook
- Matplotlib/Seaborn (for visualization)

## Citation

If you use this code in your research, please cite our paper:



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this work, please contact:
- **Chenhao Xue** (University of Oxford) - Primary contact
- **Weitao Hu** (Independent Researcher)

## Acknowledgments

This research was conducted as part of ongoing work in interpretable machine learning for music information retrieval. We thank the contributors to the MTG-Jamendo and GTZAN datasets for making this research possible.

---

*Repository for ICASSP 2026 submission: "Constructing Composite Features for Interpretable Music-Tagging"*
=======
version https://git-lfs.github.com/spec/v1
oid sha256:54a5d6261c70f7f60513b3427159cf6ddab36aac7e99a74e80bb7e30ef8568fb
size 5954
>>>>>>> 1d70951 (Initial commit: GP Music Tagging Research Code)
