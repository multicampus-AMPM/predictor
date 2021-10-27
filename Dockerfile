FROM python:3.8-slim
COPY ./requirements.txt /home/
RUN apt-get update
RUN pip install --upgrade pip && pip install --no-cache-dir --requirement /home/requirements.txt && pip install tensorflow
RUN apt-get install -y libgomp1
COPY ./predictor.py /home/
CMD ["python", "/home/predictor.py"]
