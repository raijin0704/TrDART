# syntax=docker/dockerfile:experimental

FROM python:3.7
RUN apt-get update
RUN apt install -y nodejs npm && \
    pip install --upgrade pip && \
    pip install \
        numpy==1.19.1 \
        pandas==1.1.0 \
        matplotlib==3.3.0 \
        seaborn==0.11.0 \
        ruamel.yaml==0.16.10 \
        joblib==0.16.0 \
        tqdm==4.48.0 \
        plotly==4.9.0 \
        kaleido==0.0.3 \
        optuna==2.0.0 \
        mlflow==1.11.0 \
        xlrd==1.2.0 \
        jupyterlab "ipywidgets==7.5" && \
    jupyter labextension install jupyterlab-plotly@4.9.0 && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
# github.com のための公開鍵をダウンロード
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
# プライベート・リポジトリをpip install
RUN --mount=type=ssh pip install git+ssh://git@github.com/raijin0704/my-scikit-learn.git@dart
# trbaggの入ったパッケージをpip install
RUN --mount=type=ssh pip install git+ssh://git@github.com/raijin0704/tl_algs.git
WORKDIR /work
CMD ["/bin/bash"]