FROM python:3.10
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py .

CMD ["python","train.py"]
