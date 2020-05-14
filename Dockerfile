FROM python:3.7

WORKDIR /app

COPY . /app

RUN mkdir ./ftmodels

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "-t", "999", "api:application"]