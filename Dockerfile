FROM python:3.10

RUN useradd -m user
USER user
WORKDIR /home/user/tp
ENV PATH /home/user/.local/bin:$PATH
CMD jupyter notebook --no-browser --ip='*'

ADD requirements.txt .
RUN --mount=type=cache,sharing=locked,uid=1000,gid=1000,target=/home/user/.cache \
    python -m pip install --user -U pip \
 && python -m pip install --user -r requirements.txt

ADD --chown=user . .
