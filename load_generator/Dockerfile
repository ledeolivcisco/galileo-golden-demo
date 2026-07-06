FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY question_generator.py loadgen.py ./

USER pwuser

ENTRYPOINT ["python", "loadgen.py"]
CMD ["--help"]
