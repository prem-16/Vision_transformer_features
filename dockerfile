# Use miniconda as the base image
FROM continuumio/miniconda3

# Create a conda environment with python 3.9 and pip
RUN conda create -n py39 python=3.9 pip

# Activate the environment
RUN echo "source activate py39" > ~/.bashrc
ENV PATH /opt/conda/envs/py39/bin:$PATH

# Set the working directory
WORKDIR /app

# Install gcc and g++
RUN apt-get update && apt-get install -y gcc g++

# Install the requirements
COPY requirements_dataset.txt .
# Pip install setuptools 65.5.0 to avoid error
RUN pip install setuptools==65.5.0
RUN pip install -r requirements_model.txt

# Copy your project files
COPY . .

# Run your project
# CMD ["python", "main.py"]