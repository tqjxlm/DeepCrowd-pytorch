# DeepCrowd-pytorch

Pytorch implementation of the DeepCrowd method.

## Usage

Common examples:

* a default train on all gpu:       python main.py --level 0
* a default train on gpu#1:         python main.py --level 0 --gpu-id 1
* train for 10000 episodes:         python main.py --level 0 --episode 10000
* resume an existing train:         python main.py --level 0 --load checkpoint_name
* resume the last best record:      python main.py --level 0 --load checkpoint_name --best
* save to a custom name:            python main.py --level 0 --save checkpoint_name
* load and train next stage:        python main.py --level 1 -l cp_name --best -s another_cp_name
* test and render a checkpoint:     python main.py --level 1 --inference --render --load checkpoint_name --best

For more config please refer to config.py

## Requirements:

* pytorch 1.0+
* pyqtgraph