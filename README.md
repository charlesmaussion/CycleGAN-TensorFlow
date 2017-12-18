# Use

## Generating training and testing data

From the root folder, run the following commands to generate the data:
```
python scripts/data_generation.py
python scripts/data_selection.py
python scripts/separate_training_test.py
python build_data.py --X_input_dir data/handWritten/train --Y_input_dir data/fontTyped/train
```

## Data preparing

* First, download a dataset, e.g. apple2orange

```bash
$ bash download_dataset.sh apple2orange
```

* Write the dataset to tfrecords

```bash
$ python3 build_data.py
```

Check `$ python3 build_data.py --help` for more details.

## Training

```bash
$ python3 train.py
```

If you want to change some default settings, you can pass those to the command line, such as:

```bash
$ python3 train.py  \
    --X=data/tfrecords/horse.tfrecords \
    --Y=data/tfrecords/zebra.tfrecords
```

Here is the list of arguments:
```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--image_length IMAGE_LENGTH]
                [--image_height IMAGE_HEIGHT]
                [--use_lsgan [USE_LSGAN]] [--nouse_lsgan]
                [--norm NORM] [--lambda1 LAMBDA1] [--lambda2 LAMBDA2]
                [--learning_rate LEARNING_RATE] [--beta1 BETA1]
                [--pool_size POOL_SIZE] [--ngf NGF] [--X X] [--Y Y]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size, default: 1
  --image_length IMAGE_LENGTH
                        image length, default: 900
  --image_height IMAGE_HEIGHT
                        image height, default: 52
  --use_lsgan [USE_LSGAN]
                        use lsgan (mean squared error) or cross entropy loss,
                        default: True
  --nouse_lsgan
  --norm NORM           [instance, batch] use instance norm or batch norm,
                        default: instance
  --lambda1 LAMBDA1     weight for forward cycle loss (X->Y->X), default: 10.0
  --lambda2 LAMBDA2     weight for backward cycle loss (Y->X->Y), default:
                        10.0
  --learning_rate LEARNING_RATE
                        initial learning rate for Adam, default: 0.0002
  --beta1 BETA1         momentum term of Adam, default: 0.5
  --pool_size POOL_SIZE
                        size of image buffer that stores previously generated
                        images, default: 50
  --ngf NGF             number of gen filters in first conv layer, default: 64
  --X X                 X tfrecords file for training, default:
                        data/tfrecords/apple.tfrecords
  --Y Y                 Y tfrecords file for training, default:
                        data/tfrecords/orange.tfrecords
```

Check TensorBoard to see training progress and generated images.

```
$ tensorboard --logdir checkpoints/${datetime}
```

Here are some funny screenshots from TensorBoard when training orange -> apple:

![train_screenshot](samples/train_screenshot.png)


### Notes
* If high constrast background colors between input and generated images are observed (e.g. black becomes white), you should restart your training!
* Train several times to get the best models.

## Export model
You can export from a checkpoint to a standalone GraphDef file as follow:

```bash
$ python3 export_graph.py --checkpoint_dir checkpoints/${datetime} \
                          --XtoY_model apple2orange.pb \
                          --YtoX_model orange2apple.pb \
                          --image_length 900\
                          --image_height 52
```


## Inference
After exporting model, you can use it for inference. For example:

```bash
python3 inference.py --model pretrained/apple2orange.pb \
                     --input input_sample.jpg \
                     --output output_sample.jpg \
                     --image_length 900\
                     --image_height 52
```

## Pretrained models
My pretrained models are available at https://github.com/vanhuyz/CycleGAN-TensorFlow/releases

## Contributing
Please open an issue if you have any trouble or found anything incorrect in my code :)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

* CycleGAN paper: https://arxiv.org/abs/1703.10593
* Official source code in Torch: https://github.com/junyanz/CycleGAN
