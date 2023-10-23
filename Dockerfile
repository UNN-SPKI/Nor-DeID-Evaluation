FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /experiment/
ADD requirements.txt /experiment/
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_news_sm
RUN python -m spacy download nb_core_news_sm
RUN python -m spacy download nb_core_news_lg
ADD . /experiment/
CMD python3 eval.py