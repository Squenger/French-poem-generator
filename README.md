# Nano-GPT: Poetry Generation

This project aims to generate poems using a Transformer-based architecture. The model trains on a text corpus to learn how to predict and generate new sequences of words.

## Architecture

The neural network is built around a Transformer architecture. It consists of:

- A token and positional embedding table.
- A stack of attention blocks containing multi-head attention modules.
- Feed-forward neural network modules.
- Normalization layers to stabilize training.

## Main Methods

The main.py script is structured around the GPTTrainer class, which encapsulates the execution of the code. Its main methods are as follows:

- train: Starts the learning phase of the model on the dataset and regularly evaluates the loss.
- load(path): Loads the weights of a previously saved model from the specified path.
- save(path): Saves the current state of the model to the specified file.
- generate: Uses the model to generate and display a new poem.
