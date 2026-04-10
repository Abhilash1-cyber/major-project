FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install flask gunicorn tensorflow-cpu==2.13.0 numpy pillow opencv-python-headless

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]