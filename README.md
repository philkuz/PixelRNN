# Using TensorFlow to Generate Images with PixelRNNs

This repo hosts the code associated with the O'Reilly article, ["Using Tensorflow to Generate Images with PixelRNNs: How to generate novel images using neural networks"](https://www.oreilly.com/ideas/using-tensorflow-to-generate-images-with-pixelrnns).

In this article, we walk through creating a generative model to produce realistic-looking images using recurrent neural networks in Tensorflow. Specifically, we use a PixelRNN architecture trained on MNIST to generate images that look like handwritten digits.

### Requirements

In order to run this notebook, you will need to install [TensorFlow v1.0](https://www.tensorflow.org/), [Jupyter](http://jupyter.org/), and [NumPy](http://www.numpy.org/).

The notebook also uses [TQDM](https://pypi.python.org/pypi/tqdm) to display nice progress bars during training.

There are two easy ways to install these libraries and their dependencies:

### Option A: use the provided Dockerfile configured for this notebook

1. Download and unzip [this entire repo from GitHub](https://github.com/philkuz/PixelRNN), either interactively, or by entering
    ```bash
    git clone https://github.com/philkuz/PixelRNN.git
    ```

2. Open your terminal and use `cd` to navigate into the top directory of the repo on your machine

3. To build the Dockerfile, enter
    ```bash
    docker build -t pixelrnn_dockerfile .
    ```
    If you get a permissions error on running this command, you may need to run it with `sudo`:
    ```bash
    sudo docker build -t pixelrnn_dockerfile .
    ```

4. Run Docker from the Dockerfile you've just built
    ```bash
    docker run -it -p 8888:8888 -p 6006:6006 pixelrnn_dockerfile bash
    ```
    or
    ```bash
    sudo docker run -it -p 8888:8888 -p 6006:6006 pixelrnn_dockerfile bash
    ```
    if you run into permission problems.

5. Launch Jupyter by entering
    ```bash
    jupyter notebook
    ```
    and, using your browser, navigate to the URL shown in the terminal output (usually http://localhost:8888/)

### Option B: install Anaconda Python, TensorFlow, and TQDM manually
NumPy can be tricky to install manually, so we recommend using the managed Anaconda Python distribution, which includes NumPy, Matplotlib, and Jupyter in a single installation. The Docker-based method above is much easier, but if you have a compatible NVIDIA GPU, manual installation makes it possible to use GPU acceleration to speed up training.

1. Follow the [installation instructions for Anaconda Python](https://www.continuum.io/downloads). **We recommend using Python 3.6.**

2. Follow the platform-specific [TensorFlow installation instructions](https://www.tensorflow.org/install/). Be sure to follow the "Installing with Anaconda" process, and create a Conda environment named `tensorflow`.

3. If you aren't still inside your Conda TensorFlow environment, enter it by typing
    ```bash
    source activate tensorflow
    ```

4. Install TQDM by entering
    ```bash
    pip install tqdm
    ```

5. Download and unzip [this entire repo from GitHub](https://github.com/Steven-Hewitt/Entailment-with-Tensorflow), either interactively, or by entering
    ```bash
    git clone https://github.com/philkuz/PixelRNN.git
    ```

6. Use `cd` to navigate into the top directory of the repo on your machine

7. Launch Jupyter by entering
    ```bash
    jupyter notebook
    ```
    and, using your browser, navigate to the URL shown in the terminal output (usually http://localhost:8888/)
