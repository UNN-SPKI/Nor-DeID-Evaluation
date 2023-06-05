FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /experiment/
ADD requirements.txt /experiment/
ADD requirements-dev.txt /experiment/
RUN pip install -r requirements-dev.txt
RUN python -m spacy download nb_core_news_sm
ADD . /experiment/
CMD python3 eval.py