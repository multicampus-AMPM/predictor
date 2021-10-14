FROM python:3.9-slim
COPY ./requirements.txt /home/
RUN pip install --no-cache-dir --requirement /home/requirements.txt
COPY ./predictor.py /home/
CMD ["python", "/home/predictor.py"]
