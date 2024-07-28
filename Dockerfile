# Use the official Python 3.9 image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . /app/

# Set the default command to run when the container starts
CMD ["python", "app.py"]