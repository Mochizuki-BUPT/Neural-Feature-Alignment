# Neural Feature Alignment (NFA)

Implementation of the Feature Alignment (FA) framework for cross-modal analysis between Large Language Models and EEG signals.

## Repository Structure

```
Neural-Feature-Alignment/
├── config/default.yaml         # Configuration
├── src/
│   ├── agop.py                 # AGOP feature extraction
│   ├── nnfpp.py                # NNFPP graph construction
│   ├── eeg_features.py         # EEG feature extraction
│   └── alignment.py            # Cross-modal alignment
└── scripts/run_pipeline.py     # Pipeline runner
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scripts/run_pipeline.py --config config/default.yaml
```

## Citation

Zhang, Z., Zhou, W., Zhang, S., Li, X., Zhang, L., & Li, L. (2026). Neural Feature Alignment between Large Language Models and Brain Activities: A Knowledge-Based Framework for Cross-Modal Analysis. *Neural Networks*. https://doi.org/10.1016/j.neunet.2026.108916

## License

MIT License
