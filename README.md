# Python Flask App on Docker

This repository is an implementation of running python flask app on docker environment. On this project we will detect apples, bananas, and oranges using <a href="https://github.com/ultralytics/yolov5">Yolov5</a> custom model, and then classify that using <a href="https://www.tensorflow.org/">Tensorflow</a> custom model. It can be done using image link. Here is the label list of classification model :
* freshapples
* freshbanana
* freshoranges
* rottenapples
* rottenbanana
* rottenoranges

## Requirements
* Python 3.10 or later
* Docker Desktop
* Postman
* WSL 1.2.5.0
* Pip 23.0 or later
* Free storage more than 10gb

## Run Locally
Clone the project

```bash
  git clone https://github.com/aldebarankwsuperrr/Docker-Flask.git
```

Go to the project directory

```bash
  cd Docker-Flask
```

Build docker image

```bash
  docker build --tag docker-flas.
```

Run docker image as a container

```bash
  docker run -d -p 5000:5000 docker-flask
  
```


## API Reference

#### Get Predict Result

```http
  POST http://localhost:5000/predict
```
Use this json format 

```yaml
{
    "url": "Image Link"
}
```
