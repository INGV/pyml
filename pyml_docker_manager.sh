#! /bin/bash


if [[ $# -lt 1 ]]; then
        echo ""
        echo "Usage: $0 [build/run/stop/clean]"
        echo ""
        exit
fi


if [[ `uname -a | awk '{print $1}'` == 'Darwin' || `uname -a | awk '{print $1}'` == 'darwin' || `uname -a | awk '{print $1}'` == 'DARWIN' ]]; then
        MACIP=`hostname | sed -e "s/ //g"`
elif [[ `uname -a | awk '{print $1}'` == 'Linux' || `uname -a | awk '{print $1}'` == 'linux' || `uname -a | awk '{print $1}'` == 'LINUX' ]]; then
        MACIP=`hostname -A | sed -e "s/ //g"`
fi

REPO="pyml"
TAG="1.0"
image_name="$REPO:$TAG"
container_name="${REPO}_${TAG}"
dockerfile="Dockerfile"

container_id=`docker ps -a -f name=$container_name -f status=running -q`

action=`echo $1 | tr [:upper:] [:lower:]`
if [[ $action == "build" ]]; then
        image_id=`docker images $image_name -q`
        if [[ ! -z $image_id ]]; then
                docker rmi $image_id
        fi
        docker build --no-cache -f $dockerfile -t $image_name .
elif [[ $action == "run" ]]; then
        image_id=`docker images $image_name -q`
        docker run -d $LINK_CONTAINER -v $(pwd)/tmp_data:/tmp_data --name $container_name $image_id
elif [[ $action == "stop" ]]; then
        container_id=`docker ps -a -f name=$container_name -f status=running -q`
        docker stop $container_id
elif [[ $action == "clean" ]]; then
        running=`docker ps -a -f status=running -f name=$container_name -q`
        exited=`docker ps -a -f status=exited -f name=$container_name -q`
        image=`docker images $image_name -q`
        if [[ $ws_running ]]; then
                echo "RUN: $running"
                docker stop $running
                docker rm $running # Since docker container is run with --rm option this part might be useless like the following elif
                docker rmi $image
        elif [[ $exited ]]; then
                echo "EXIT: $exited"
                docker rm $exited
                docker rmi $image
        elif [[ $image ]]; then
                echo "IMAGE: $image"
                docker rmi $image
        fi
else
        echo "Opzione non contemplata"
        exit 1
fi
