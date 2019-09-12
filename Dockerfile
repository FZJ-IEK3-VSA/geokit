FROM continuumio/miniconda3:4.5.12
MAINTAINER sevberg "s.ryberg@fz-juelich.de"

# Install modules
RUN conda install -y -c conda-forge numpy pandas matplotlib scipy descartes gdal=2.4.1 jupyter notebook && \
    conda clean -a

# Install geokit and test
COPY setup.py MANIFEST.in LICENSE.txt README.md contributors.txt Examples geokit /repos/geokit/
RUN pip install -e /repos/geokit

# Setup entry
ENTRYPOINT ["/bin/bash"]
