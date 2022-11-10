# Streamlit

## 환경 설정

```
가상 환경
virtualenv --python=python3 env
source env/bin/activate

or

conda create -n 환경이름 -f requirements.txt python=3.6

패키지
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Streamlit 실행

```
streamlit run app.py

streamlit run app_rating.py --server.maxUploadSize 1024
```