FROM python:3
RUN pip install flask flask-cors
COPY . /app
ENTRYPOINT FLASK_APP=/app/Classifier1.py flask run --host=0.0.0.0
