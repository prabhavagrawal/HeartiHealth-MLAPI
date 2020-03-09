FROM python:3
RUN pip install flask flask-cors numpy pandas matplotlib seaborn
COPY . /app
ENTRYPOINT FLASK_APP=/app/Classifier1.py flask run --host=0.0.0.0


#FROM python:3.6-alpine
