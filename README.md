
# Machine Learning Serving

Situation Recognition

실행 gif

---

## Install
1. Clone the Repo
```
git clone https://github.com/yunsujeon/MLserving_ServiceSituationRecognition.git
```

2. Download model
locate at app/
  [model_best.pth.tar](https://drive.google.com/file/d/12DVhwEKFxxtowHBRpNMQSxpMo4Bc-0Jg/view?usp=sharing)
locate at app/output_crf_v1/
  [best.model](https://drive.google.com/file/d/128rO633ev0XoTCZ56OoECEm0YobMid1K/view?usp=sharing)


## Running on Local machine with Anaconda

1. Anaconda create and activate
```
conda create -n <name> python==3.6.2
conda activate <name>
```

2. Install requirements
```
pip install -r requirements.txt
```
 
3. Run
```
python app.py
```
Go to http://0.0.0.0:8888 , then you can see wep page and explanation.


## Running on Docker

1. Install Docker your self

2. Create Docker image by build Dockerfile
```
sudo docker build -t <image name> .
or
docker build -t <image name> .
```

3. Run docker file
```
docker run -i -t --rm -p 8888:8888 -v <your path>:/<docker path> --shm-size=2GB --gpus all <image name>
ex)
docker run -i -t --rm -p 8888:8888 -v /home/intern/MLserving/app:/app --shm-size=2GB --gpus all <image name>

```
If you need more memory in docker env, and select specific gpus ..
--shm-size=8G
--gpus '"device=0,1"'

Go to http://0.0.0.0:8888, then you can see wep page and explanation.

## Improvement

You can run this codes at SSH server, Its all same this repo's local, docker examples

But you will change the access url
0.0.0:8888 -> [your remote server ip]:8888

Enjoy this Repo. thank you.
