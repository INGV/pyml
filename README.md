# pyml

Python code, complementary to pyamp, to calculate ML with different methods and attenuation functions using pyamp_amplitude output file or database table.


docker run -it --rm -v`pwd`/examples/input/amplitudes_standard.json:/opt/amplitudes_standard.json --name pyml2 pyml:2.0 --json amplitudes_standard.json
