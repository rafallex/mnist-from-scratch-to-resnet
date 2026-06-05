# Deep Learning (1RT720)

Coursework for **Deep Learning (1RT720)** in the M.Sc. in Image Analysis and Machine Learning at Uppsala University. Two hand-in assignments that build up from a fully-connected neural network written from scratch in NumPy (forward pass, backprop, and SGD by hand) to a residual CNN trained in PyTorch with Adam, Batch Normalization, and Dropout, all on MNIST.

The point of each assignment was to implement and reason about one building block at a time — He initialization, activation derivatives, the softmax + cross-entropy gradient, the Adam update, skip connections, regularization — rather than to chase a leaderboard number on MNIST.

## Structure

```
deep-learning-1RT720/
├── ha1-mnist-numpy/
│   ├── hand_in_1.ipynb   # NumPy-only fully-connected net (with run outputs and plots)
│   ├── utils.py          # MNIST PNG loader + training-curve plotting helpers
│   └── Report.pdf        # written report for hand-in 1
└── ha2-cnn-pytorch/
    ├── HA2_Code.py        # PyTorch progression: FC → CNN → ResNet → BN/Dropout
    └── HA2_Report.pdf     # written report for hand-in 2
```

## Hand-in 1 — fully-connected net in NumPy

`ha1-mnist-numpy/hand_in_1.ipynb` implements a fully-connected classifier using only NumPy, supporting an arbitrary architecture passed as a list (e.g. `[784, 128, 64, 10]`). The data convention is rows-as-data-points throughout (`X` is `(N, 784)`, `Y` is one-hot `(N, 10)`).

What I wrote by hand:

- **He initialization** for the weights (`W ~ N(0, 2/n_in)`), zeros for the biases.
- **Numerically stable softmax and cross-entropy** — the row-wise max is subtracted from the logits before exponentiating to avoid overflow.
- A **fully vectorized** forward and backward pass (matrix products only, no Python loops over samples or units).
- **Mini-batch SGD** with per-epoch shuffling.

Results on MNIST (test accuracy, after 50 epochs):

| Model | Architecture | Activation | Test accuracy |
|---|---|---|---|
| Linear baseline | `[784, 10]` | — | 92.69% |
| Two hidden layers | `[784, 128, 64, 10]` | ReLU | **98.04%** |
| Two hidden layers | `[784, 128, 64, 10]` | sigmoid (lr 0.5) | comparable to ReLU |

The notebook also visualizes the linear model's learned weight matrix as ten 28×28 templates — digits with consistent shapes (0, 1) give clean templates, while more variable digits (5, 8, 9) are noisier. The committed notebook keeps its cell outputs, so the training logs and plots render directly on GitHub.

## Hand-in 2 — CNNs in PyTorch

`ha2-cnn-pytorch/HA2_Code.py` works through modern CNN components one change at a time, re-running the same training loop so each result is directly comparable. Every step reports learning curves, parameter counts, and the train–test gap.

| Step | Model | Optimizer | Params | Test accuracy |
|---|---|---|---|---|
| 1 | FC `[784, 128, 64, 10]` (PyTorch baseline) | SGD | 109,386 | 98.20% |
| 2 | CNN, 3 conv layers | SGD | 21,578 | 98.20% |
| 3 | Same CNN | Adam | 21,578 | 98.99% |
| 4 | Residual CNN | Adam | 45,882 | 99.15% |
| 5a | Residual CNN + BatchNorm | Adam | — | 99.12% |
| 5b | Residual CNN + Dropout | Adam | — | **99.29%** |

A few things I took away from the comparison (detailed in `HA2_Report.pdf`):

- The CNN matches the fully-connected baseline's accuracy with **5.1× fewer parameters**, and with a much smaller train–test gap — weight sharing generalizes better than dense layers here.
- Adam converged far faster than SGD (it was already at ~98% test accuracy after one epoch) but started to overfit slightly in later epochs.
- Residual connections pushed test accuracy above 99% by helping gradients flow through the deeper network.
- Dropout came out ahead of BatchNorm on this task: its test loss ended up *lower* than its train loss, i.e. it generalized rather than memorized. The confusion matrix for the Dropout model is near-diagonal; the few remaining errors are visually plausible pairs (most often 7→2, 5→3, 4→9) on genuinely ambiguous handwriting.

## Running it

**Hand-in 1.** Open `ha1-mnist-numpy/hand_in_1.ipynb`. It only needs `numpy`, `matplotlib`, and `imageio`. The loader in `utils.py` expects MNIST as PNG files laid out as `MNIST/Train/<label>/*.png` and `MNIST/Test/<label>/*.png` next to `utils.py` (this dataset folder is not committed). The notebook already contains the outputs from my run, so you can read the logs and plots without re-running.

**Hand-in 2.** `HA2_Code.py` was written for Google Colab with a GPU and downloads MNIST automatically through `torchvision` (no manual data setup). It needs `torch`, `torchvision`, `scikit-learn`, `matplotlib`, and `numpy`. On a GPU runtime, run it top to bottom:

```bash
python ha2-cnn-pytorch/HA2_Code.py
```

It falls back to CPU if no GPU is available, but training all six models is much slower there.

## A note on AI assistance

These are graded assignments and the reports disclose tool use, so I'll repeat it here: I used GitHub Copilot (hand-in 2) and Gemini (hand-in 1) for boilerplate, plotting, and report drafting, and to help debug vectorized NumPy. The architectures, hyperparameters, and the written analysis are mine, and I checked the generated code for correctness.
