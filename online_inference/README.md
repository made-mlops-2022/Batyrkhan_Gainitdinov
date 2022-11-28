# MLOps, HW2, MADE 2022

### building an image
- To build docker_image, run:
docker build -t batr97/online_inference .
- to pull from dockerhub, run:
docker pull batr97/online_inference
### run container
docker run -d -p 5001:5001 batr97/online_inference
### tests
- run from workdirectory:
python -m pytest