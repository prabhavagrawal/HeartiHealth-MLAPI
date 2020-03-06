FROM python:3
COPY . /app
RUN pip install flask flask-cors
ENTRYPOINT FLASK_APP=/app/Classifier1.py flask run --host=0.0.0.0
