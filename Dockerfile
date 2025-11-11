FROM python:3.10-alpine
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
WORKDIR /app/invoice_parser
CMD ["python", "main.py"]
