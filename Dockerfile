FROM python:3.10-slim

WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y libgl1

COPY . .

# install torch CPU version (IMPORTANT for HF)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# install rest
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]