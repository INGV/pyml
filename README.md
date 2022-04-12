# pyml

Python code, complementary to pyamp, to calculate ML with different methods and attenuation functions using pyamp_amplitude output file or database table.

## Quickstart
### Docker image
First, clone the git repository
```
git clone git@gitlab.rm.ingv.it:raffaele.distefano/pyml.git
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
docker run --rm --user $(id -u):$(id -g) -v $(pwd)/example:/opt/input ingv/pyml
docker run -it --rm -v`pwd`/examples/input/amplitudes_standard.json:/opt/amplitudes_standard.json --name pyml2 pyml:2.0 --json amplitudes_standard.json
```

## Author
(c) 2022 Raffaele Distefano raffaele.distefano[at]ingv.it

(c) 2022 Valentino Lauciani valentino.lauciani[at]ingv.it

Istituto Nazionale di Geofisica e Vulcanologia, Italia
