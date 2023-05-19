FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /experiment/
ADD requirements.txt /experiment/
RUN pip install -r requirements.txt

ADD . /experiment/
CMD python3 eval.py