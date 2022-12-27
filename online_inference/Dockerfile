FROM python:3.9-slim-buster

RUN python3 -m pip install --upgrade pip
COPY . ./online_inference
WORKDIR /online_inference
RUN pip install -r requirements.txt
RUN pip install fastapi

EXPOSE 5001

CMD ["bash", "run.sh"]