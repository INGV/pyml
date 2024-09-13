[![License](https://img.shields.io/github/license/INGV/pyml?label=License)](https://github.com/INGV/pyml/blob/main/LICENSE)

[![DockerHub](https://img.shields.io/badge/DockerHub-link_to_repository-blueviolet?style=flat&logo=docker&logoColor=blue&logoSize=auto)](https://hub.docker.com/r/ingv/pyml)
![DockerHub Image Size](https://img.shields.io/docker/image-size/ingv/pyml?sort=semver&style=flat&logo=docker&logoSize=auto&label=DockerHub%20Image%20Size)
![DockerHub Pulls](https://img.shields.io/docker/pulls/ingv/pyml?style=flat&logo=docker&logoSize=auto&label=DockerHub%20Image%20Pull)

![GitHub Static Badge](https://img.shields.io/badge/GitHub-link_to_repository-blueviolet?style=flat&logo=github&logoSize=auto)
[![GitHub Issues](https://img.shields.io/github/issues/INGV/pyml?label=GitHub%20Issues&logo=github)](https://github.com/INGV/pyml/issues)

![Dynamic YAML Badge](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2FINGV%2Fpyml%2Fmain%2F.github%2Fworkflows%2Fdocker-image.yml&query=%24..platforms&style=flat&logo=amazonec2&logoColor=white&logoSize=auto&label=Supported%20Arch)

# pyml ![GitHub Tag](https://img.shields.io/github/v/tag/ingv/pyml?sort=semver&style=flat) | [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ingv/pyml/docker-image.yml?branch=main&style=flat&logo=GitHub-Actions&logoColor=white&logoSize=auto&label=GitHub%20Actions)](https://github.com/INGV/pyml/actions)
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
docker run --rm -v $(pwd)/examples/input:/opt/data ingv/pyml --in_file_name /opt/data/eventid_28745631.json --in_file_format json --out_format json
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
