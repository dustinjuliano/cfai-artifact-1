This project demonstrates observational equivalence between a deterministic, rule-based Python script and an Artificial Neural Network (Multilayer Perceptron) trained on a closed universe. The goal is to achieve 100.00% accuracy replicating Run-Length Encoding (RLE) logic within a strictly bounded string alphabet environment. That goal was achieved and satisfies the requirement for its use as a demonstration artifact in "Concepts of Formal AI" presentation(s).

## Table of Contents

- [1. Development Process Overview](#1-development-process-overview)
- [2. Model Construction & Architecture](#2-model-construction--architecture)
- [3. Workspace Structure](#3-workspace-structure)
- [4. Getting Started & Prerequisites](#4-getting-started--prerequisites)
- [5. Build and Training Execution](#5-build-and-training-execution)
- [6. Test Driver Command Line Options](#6-test-driver-command-line-options)
- [7. Limitations and Scope Acceptance](#7-limitations-and-scope-acceptance)
- [8. Disclaimer](#8-disclaimer)

## 1. Development Process Overview

The development progressed through three distinct phases to overcome the limitations of statistical function approximation when applied to rigid logical rules.

- **Phase 1: Dynamic Regression Bounding**
  Initial attempts used continuous value regression to predict characters and counts from randomly generated sequences. This suffered from character-weight bias and failed to converge reliably.

- **Phase 2: Classification Layer Transition**
  The problem was redefined as multi-class classification. One-hot tokenization removed numerical hierarchy between symbols, and cross-entropy loss focused the network on discrete categorical selections. This raised accuracy to 99.97% but left a minor gap due to statistical outliers.

- **Phase 3: Closed-Universe Balancing**
  Because the 8-element binary universe possesses only 256 total possible permutations, a fixed corpus was generated to eliminate statistical probability drift. Equal distribution of every possible sequence forced perfect boundary allocation, locking in 100.00% compliance.

## 2. Model Construction & Architecture

The architecture treats the sequential task as a high-dimensional geometric mapping problem. Instead of processing characters sequentially over time, the network maps a fixed matrix snapshot directly to categorical categories.

### Feature Encoding

- **Inputs:** Each element in the 8-character string sequence is converted to a 2-element one-hot vector `(A = [1.0, 0.0], B = [0.0, 1.0])`. The input layer accepts a flat 16-element vector.
- **Outputs:** The network projects to 8 output slots, where each slot represents an 11-way classification distribution:
  - Class 0: Padding token `0`
  - Class 1: Symbol `A`
  - Class 2: Symbol `B`
  - Classes 3-10: Run counts 1 through 8

### Network Structure

- Layer 1: `nn.Linear(16, 256)` followed by `nn.ReLU()`
- Layer 2: `nn.Linear(256, 256)` followed by `nn.ReLU(`)`
- Layer 3: `nn.Linear(256, 128)` followed by `nn.ReLU()`
- Layer 4: `nn.Linear(128, 88)` which is then reshaped to `(-1, 8, 11)`

## 3. Workspace Structure

- **corpus.py:** Generates the balanced 25,600-row text dataset
- **rle_ann.py:** Translation bridge loading weights and running inference
- **rle_corpus.csv:** Compressed dataset artifact containing all permutations
- **rle_model.pt:** Saved weights tensor for the classification network
- **rle_py.py:** Reference deterministic algorithmic execution script
- **test_demo.py:** Comprehensive evaluation script with corpus-loading flags

## 4. Getting Started & Prerequisites

The framework requires Python 3 and PyTorch. Hardware acceleration (CUDA or Apple Silicon MPS) is automatically detected and leveraged if present.

Install dependencies via pip:

```bash
pip install torch
```

## 5. Build and Training Execution

To fully reproduce the trained model from scratch, execute the scripts sequentially in your terminal. This process generates the data file first and then feeds it directly into the neural network training pipeline. This corpus is also used with `test_demo.py`, if requested by CLI arguments.

### Step 1: Build the Balanced Corpus

Run the corpus generator script to create the balanced distribution text file. This creates `rle_corpus.csv` on your disk with 25,600 lines.

```bash
python corpus.py
```

### Step 2: Train the Model

Run the training script. It loads the newly created `rle_corpus.csv`, runs the optimization loop for 200 epochs, and exports the final network weights to `rle_model.pt`.

```bash
python train_model.py
```

## 6. Test Driver Command Line Options

The evaluation program `test_demo.py` uses CLI arguments to customize the testing scenario. This allows you to toggle which framework executes, change test sizes, and choose whether data comes from the static corpus or random generations.

### Available Flags

- `-m, --module`
  Determines which RLE framework to test.
  Options: rle_py (test only the Python script), rle_ann (test only the neural network), or both (run both side-by-side). Default value is both.

- `-r, --runs`
  Specifies the integer count of test sequences to evaluate. Default value is 1000.

- `-c, --corpus`
  Specifies the file path to a saved corpus CSV file. When this flag is used, the script stops generating random lines on the fly and loops directly through the text file sequences. Default value is None.

- `-v, --verbose`
  A boolean flag that enables step-by-step terminal reporting. When active, every individual test case is printed side-by-side with its inputs, outputs, and a pass/fail confirmation bracket.

### Verification Examples

To run the complete 25,600-line corpus through both engines to demonstrate total observational equivalence (on the given closed universe):

```bash
python test_demo.py -m both -c rle_corpus.csv -r 25600
```

To run a small batch of 20 lines from the corpus with a live, step-by-step column visualizer:

```bash
python test_demo.py -m both -c rle_corpus.csv -r 20 -v
```

To test model stability against 5,000 completely random configurations generated using live system entropy:

```bash
python test_demo.py -m both -r 5000
```

## 7. Limitations and Scope Acceptance

While the model is subject to clear structural boundaries, these limitations are fully recognized, intentional, and entirely acceptable within the context of this project.

The primary objective is to deliver a definitive demonstration of observational equivalence, proving that two fundamentally different architectural designs can produce identical, interchangeable behaviors. In a conceptual sense, this exercise demonstrates that both systems are computer programs. The Python script processes data through time using sequential loops, while the Multilayer Perceptron processes data through space using a fixed matrix function. For the purposes of the live presentation and demonstration this code is built for, achieving complete coverage across this closed domain is more than sufficient.

Furthermore, it is understood that a perfect 100.00% accuracy score across massive test runs would be incredibly difficult, if not practically impossible, to secure if the network were designed to truly learn the abstract, open-ended counting algorithm itself. True algorithmic neural networks, such as recurrent architectures, rely on continuous, probabilistic floating-point calculations. Over a long sequence, these systems could suffer from compounding errors and statistical drift that lead to fractional failures.

By bounding the universe to 256 balanced patterns and utilizing a feedforward classification network, the model bypasses these continuous mathematical vulnerabilities entirely. The resulting geometric map functions as a lookup system, which was helpful for demonstration purposes. The objective was to show that these two systems are, in some sense, interpreted computer programs. This helps prime the intuition to build upon as a foundation for further discussion involving Formal AI.

### Technical Constraints Include

- **No Algorithmic Generalization:** The model cannot scale past an input length of 8 elements. Increasing the window requires altering the input layer dimensions and rebuilding the weight configuration.
- **Closed Universe Necessity:** The 100% reliability metric depends entirely on total population coverage during training to eliminate the optimization biases of gradient descent.

## 8. Disclaimer

This project, its architectural layout, and validation workflows were developed in collaboration with Google Gemini. It is intended strictly for demonstration purposes to explore the boundary spaces between algorithmic state machines and feedforward neural network approximations.
