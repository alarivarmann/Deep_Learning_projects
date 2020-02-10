## CPU-Version of The E-Commerce Data Mapper Docker Container

FROM conda/miniconda3
# docker run -i -t conda/miniconda3 /bin/bash
ADD app.zip /
RUN apt-get update  
RUN apt-get install unzip
RUN sh -c 'unzip -q "app.zip"'
CMD find app

ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init bash

RUN conda install -c pytorch torchvision-cpu -y
RUN conda install -c fastai fastai -y
RUN conda install -c anaconda scikit-learn -y 

CMD ["ls app"]
ENTRYPOINT ["python"]
CMD ["app/main.py"]

EXPOSE  8888
