FROM python:3.6-slim
RUN pip install flask flask-cors numpy pandas matplotlib seaborn sklearn
COPY . /app
WORKDIR /app
EXPOSE 5000
ENTRYPOINT FLASK_APP=/app/Classifier1.py flask run --host=0.0.0.0
