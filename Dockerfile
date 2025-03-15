# Use an official lightweight Python image as the base image
FROM python:3.10-slim AS base

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Use a lightweight image for the final stage
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies from the base stage
COPY --from=base /app /app

# Expose port 5000 (or your Flask app port)
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run the Flask app
CMD ["flask", "run"]