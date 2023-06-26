# app/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim as base

# Set the working directory to /app
WORKDIR /home/app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /home/app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add all files to /app
COPY . /home/app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD uvicorn app:app --host 0.0.0.0 --port 80 --reload --log-level debug

