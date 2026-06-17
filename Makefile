.PHONY: train evaluate docker-build docker-train

train:
	python train.py --iterations 100 --search-time 1 --network big

evaluate:
	python evaluate.py --checkpoint checkpoints/model.h5 --games 20

docker-build:
	docker build -t chess-rl .

docker-train:
	docker run --rm -v $(PWD)/checkpoints:/app/checkpoints chess-rl
