# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:24.06-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update

# install necessary latex packages
RUN curl -L -o install-tl-unx.tar.gz https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2024/install-tl-unx.tar.gz
RUN zcat < install-tl-unx.tar.gz | tar xf -
RUN cd install-tl-2*  && perl install-tl --no-interaction --scheme=small
ENV PATH="/usr/local/texlive/2024/bin/x86_64-linux:${PATH}"
RUN tlmgr install type1cm cm-super dvipng

ENV MY_ASTRO_DATA=/astro_data
ENV SDSS_LOCAL_SAS_MIRROR=$MY_ASTRO_DATA/sdss_mirror
ENV GAIA_TOOLS_DATA=$MY_ASTRO_DATA/gaia_mirror
ENV DUST_DIR=$MY_ASTRO_DATA/mwdust_mirror
RUN mkdir $MY_ASTRO_DATA $SDSS_LOCAL_SAS_MIRROR $GAIA_TOOLS_DATA $DUST_DIR
RUN git clone https://github.com/henrysky/astroNN_stars_foundation.git
WORKDIR /workspace/astroNN_stars_foundation
RUN pip install -r requirements.txt
RUN mkdir data_files figs
RUN curl --cookie zenodo-cookies.txt "https://zenodo.org/records/12738256/files/testing_set.h5?download=1" --output data_files/testing_set.h5
RUN curl --cookie zenodo-cookies.txt "https://zenodo.org/records/12738256/files/training_set.h5?download=1" --output data_files/training_set.h5