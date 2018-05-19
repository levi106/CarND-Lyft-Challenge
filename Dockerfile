FROM tensorflow/tensorflow:1.8.0-gpu-py3

RUN pip --no-cache-dir install \
    tensorflow-gpu==1.8 \
    tqdm==4.11.2 \
    scipy==0.19.1 \
    scikit-image==0.13.1 \
    
