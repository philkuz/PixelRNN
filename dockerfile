FROM gcr.io/tensorflow/tensorflow
RUN apt-get update && apt-get install -y git-core
RUN pip install tqdm
RUN git clone https://github.com/philkuz/PixelRNN.git /notebooks/PixelRNN
WORKDIR "/notebooks"
CMD ["/run_jupyter.sh"]
