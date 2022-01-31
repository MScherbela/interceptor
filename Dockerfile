FROM node:latest

RUN apt-get -y clean all && apt-get -y update && apt-get -y upgrade
RUN apt-get -y install bash vim less
RUN apt-get -y install python3 python3-pip

# Install python-backend requirements (installing them already here to improve docker caching)
ADD backend/requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
ADD backend/ .
ADD frontend/ .

# Build front-end package
WORKDIR /frontend
RUN npm run build
RUN cp dist/* ../frontend/static

# Switch back to backend and start server
WORKDIR /backend
ENV FLASK_APP=app.py
CMD ["gunicorn", "--bind", ":80", "--worker-tmp-dir", "/dev/shm", "--workers=1", "--threads=2", "app:app"]