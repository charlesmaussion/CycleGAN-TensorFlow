# set hyperparameters here
BATCH_SIZE = 1
IMAGE_LENGTH = 448
IMAGE_HEIGHT = 24
NGF = 64
X = apple
Y = orange

# for export_graph: default is the lastest checkpoint
CHECKPOINT_DIR = checkpoints/`ls checkpoints | tail -n 1`

# for inference
INPUT_IMG = input_sample.jpg
OUTPUT_IMG = output_sample.jpg
MODEL = $(X)2$(Y).pb

# commands come here
download_data:
	bash download_dataset.sh $(X)2$(Y)

build_data:
	python3 build_data.py --X_input_dir=data/$(X)2$(Y)/trainA \
                              --Y_input_dir=data/$(X)2$(Y)/trainB \
                              --X_output_file=data/tfrecords/$(X).tfrecords \
                              --Y_output_file=data/tfrecords/$(Y).tfrecords

train:
	python3 train.py --batch_size=$(BATCH_SIZE) \
                         --image_length=$(IMAGE_LENGTH) \
					     --image_height=$(IMAGE_HEIGHT) \
                         --ngf=$(NGF) \
                         --X=data/tfrecords/$(X).tfrecords \
                         --Y=data/tfrecords/$(Y).tfrecords

export_graph:
	python3 export_graph.py --checkpoint_dir=$(CHECKPOINT_DIR) \
                                --XtoY_model=$(X)2$(Y).pb \
                                --YtoX_model=$(Y)2$(X).pb \
							    --image_length=$(IMAGE_LENGTH) \
								--image_height=$(IMAGE_HEIGHT)

inference:
	python3 inference.py --model=$(MODEL)\
                             --input=$(INPUT_IMG) \
                             --output=$(OUTPUT_IMG) \
							 --image_length=$(IMAGE_LENGTH) \
							 --image_height=$(IMAGE_HEIGHT)

tensorboard:
	tensorboard --logdir=$(CHECKPOINT_DIR)
