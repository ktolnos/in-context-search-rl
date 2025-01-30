array=( halfcheetah-medium-v2 halfcheetah-medium-replay-v2 halfcheetah-medium-expert-v2 hopper-medium-v2 walker2d-medium-v2 maze2d-umaze-v1 maze2d-umaze-v1 antmaze-umaze-v2 antmaze-umaze-v2 pen-human-v1 pen-cloned-v1 pen-expert-v1 door-cloned-v1 hammer-cloned-v1 relocate-cloned-v1  )
for i in "${array[@]}"
do
	python main.py --name=$1-$i --group=$1 --env=$i &
done
