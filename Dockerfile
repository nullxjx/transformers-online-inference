# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV MODEL_PATH /production/public_weights/huggingface/bigcode/starcoderbase-3b/original

# Run app.py when the container launches
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]