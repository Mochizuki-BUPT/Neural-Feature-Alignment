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

## License

MIT License
