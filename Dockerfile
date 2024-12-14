FROM python:3.9.19-slim
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
CMD ["python", "api.py"]