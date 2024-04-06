# Use an official python runtime as a parent image
FROM python:slim

# Set the working directory in the container
WORKDIR /app

# Set the default shell to bash
SHELL ["/bin/bash", "-c"] 

# Install the required packages
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b\
    && rm -f Miniconda3-py39_23.1.0-1-Linux-x86_64.sh

# Set the environment variables to avoid interactive shell
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Create a new conda environment and activate it
RUN conda create --name py39 python=3.9
SHELL ["conda", "run", "-n", "py39", "/bin/bash", "-c"]

# Install the required packages
RUN conda install nb_conda_kernels

# Add the current directory contents into the container at /app
ADD . /app

# Install Cider dependencies
RUN pip install notebook .

# Add Cider to Pythonpath
ENV PYTHONPATH=$PYTHONPATH:$HOME/cider

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter notebook
ENTRYPOINT ["conda", "run", "-n", "py39", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
