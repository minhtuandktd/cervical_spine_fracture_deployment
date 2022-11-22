FROM python:3.9.13

WORKDIR /usr/src


# Copy all files to container
COPY . .

# install dependencies
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install timm==0.6.11


# tell the port number container should expose
EXPOSE 8000


# copy static file
# run command
CMD ["python","-B" ,"manage.py", "runserver", "0.0.0.0:8000"]