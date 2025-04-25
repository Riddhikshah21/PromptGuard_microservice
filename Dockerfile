# Use a lightweight Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Set environment variables
ENV PORT=8000

# Expose the port the app runs on (adjust if necessary)
EXPOSE ${PORT}

# Set the command to run the application (replace with your actual command)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]