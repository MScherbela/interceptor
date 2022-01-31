FROM python:3.8

RUN apt-get -y clean all && apt-get -y update && apt-get -y upgrade
RUN apt-get -y install npm vim less

# Copy source code and install backend requirements
ADD backend/ .
ADD frontend/ .
RUN pip install -r requirements.txt

# Build front-end package
WORKDIR /frontend
RUN npm build
RUN cp dist/* ../frontend/static

# Switch back to backend and start server
WORKDIR /backend
ENV FLASK_APP=app.py
CMD ["gunicorn", "--bind", ":80", "--worker-tmp-dir", "/dev/shm", "--workers=1", "--threads=2", "app:app"]