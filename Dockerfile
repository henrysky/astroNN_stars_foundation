# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:24.06-py3
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update
RUN apt install -y texlive-full
RUN git clone https://github.com/henrysky/astroNN_stars_foundation
WORKDIR /workspace/astroNN_stars_foundation
RUN pip install -r requirements.txt
RUN mkdir -p data_files
RUN curl --cookie zenodo-cookies.txt "https://zenodo.org/records/12738256/files/testing_set.h5?download=1" --output data_files/testing_set.h5
RUN curl --cookie zenodo-cookies.txt "https://zenodo.org/records/12738256/files/training_set.h5?download=1" --output data_files/training_set.h5