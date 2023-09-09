# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set the maintainer label
LABEL maintainer="federicoriveroll@gmail.com"

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive

# Run package updates and install packages
# Run package updates and install packages
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    python3.8-venv


# Create a working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Run pip check (this line is optional but can be useful)
RUN pip3 check

# Copy the current directory contents into the container at /app
COPY . /app/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python3", "app.py"]
