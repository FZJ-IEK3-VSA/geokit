FROM continuumio/miniconda3:4.5.12
MAINTAINER sevberg "s.ryberg@fz-juelich.de"

# Install modules
RUN conda install -y -c conda-forge numpy pandas matplotlib scipy descartes gdal=2.4.2 jupyter notebook && \
    conda clean -a

# Install geokit and test
COPY ./ /repos/geokit
RUN pip install -e /repos/geokit

# Setup entry
#ENTRYPOINT ["python"]
