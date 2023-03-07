#!/bin/bash
#
#
# (c) 2021 Valentino Lauciani <valentino.lauciani@ingv.it>,
#          Raffaele Distefano <raffaele.distefano@ingv.it>,
#          Istituto Nazione di Geofisica e Vulcanologia.
# 
#####################################################

# Set env
export MPLCONFIGDIR="/tmp"

# Check input parameter
if [[ -z ${@} ]]; then
        echo ""
	/usr/bin/python3 /opt/pyml.py -h
        echo ""
        exit 1
fi

# run command
/usr/bin/python3 /opt/pyml.py $@ --dbona_corr /opt/dbona_magnitudes_stations_corrections_extended_mq.csv
