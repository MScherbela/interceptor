FROM node:16

WORKDIR /workdir
RUN apt-get -y clean all && apt-get -y update && apt-get -y upgrade
RUN apt-get -y install bash vim less
RUN apt-get -y install python3 python3-pip

# Install python-backend requirements (installing them already here to improve docker caching)
ADD backend/requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy source code
ADD backend/ ./backend
ADD frontend/ ./frontend

# Build front-end package
WORKDIR frontend
RUN npm install vue
RUN npm run build
RUN cp dist/* ../backend/static

# Switch back to backend and start server
WORKDIR ../backend
ENV FLASK_APP=app.py
CMD ["gunicorn", "--bind", ":80", "--worker-tmp-dir", "/dev/shm", "--workers=1", "--threads=2", "app:app"]