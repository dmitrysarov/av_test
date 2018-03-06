FROM tensorflow/tensorflow:1.4.1
RUN apt-get update && apt-get install python3-pip -y --no-install-recommends 
RUN pip3 install --upgrade pip && pip3 install -U setuptools && \ 
    pip3 install click numpy scikit-image scipy tensorflow==1.4.1
