FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN apt update && apt install -y xvfb figlet

# RUN conda upgrade ffmpeg
RUN conda install ffmpeg
RUN conda install -c conda-forge tensorboard tensorboardx jupyterlab seaborn
RUN pip install gym[atari,accept-rom-license]==0.25.0

ADD build.sh /
RUN chmod +x /build.sh

CMD [ "/build.sh" ]
