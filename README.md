[![License](https://img.shields.io/github/license/INGV/pyml.svg)](https://github.com/INGV/pyml/blob/main/LICENSE) [![GitHub issues](https://img.shields.io/github/issues/INGV/pyml.svg)](https://github.com/INGV/pyml/issues)

[![Docker build](https://img.shields.io/badge/docker%20build-from%20CI-yellow)](https://hub.docker.com/r/ingv/pyml)![Docker Image Size (latest semver)](https://img.shields.io/docker/image-size/ingv/pyml?sort=semver)![Docker Pulls](https://img.shields.io/docker/pulls/ingv/pyml)

[![CI](https://github.com/INGV/pyml/actions/workflows/docker-image.yml/badge.svg)](https://github.com/INGV/pyml/actions)[![GitHub](https://img.shields.io/static/v1?label=GitHub&message=Link%20to%20repository&color=blueviolet)](https://github.com/INGV/pyml)

# pyml
## Introduction
Python code, complementary to pyamp, to calculate ML with different methods and attenuation functions using pyamp_amplitude output file or database table.

## Quickstart
### Docker image
First, clone the git repository
```
git clone https://github.com/INGV/pyml.git
cd pyml
docker build --tag ingv/pyml .
```

in case of errors, try:
```
docker build --no-cache --pull --tag ingv/pyml .
```


### Run docker
To run the container, use the command below; the `-v` option is used to "mount" working directory into container:
```
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/example/input:/opt/data ingv/pyml --json /opt/data/eventid_28745631.json 
```

## Contribute
Thanks to your contributions!

Here is a list of users who already contributed to this repository: \
<a href="https://github.com/ingv/pyml/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ingv/pyml" />
</a>

## Authors
(c) 2023 Raffaele Distefano raffaele.distefano[at]ingv.it \
(c) 2023 Valentino Lauciani valentino.lauciani[at]ingv.it

Istituto Nazionale di Geofisica e Vulcanologia, Italia
