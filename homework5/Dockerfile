FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

COPY flask_app.py .
COPY model1.bin /app/model1.bin

EXPOSE 9696

# Command to run the application with Gunicorn
CMD ["pipenv", "run", "gunicorn", "-b", "0.0.0.0:9696", "flask_app:app"]
