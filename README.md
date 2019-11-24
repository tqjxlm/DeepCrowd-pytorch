# DeepCrowd

Experiment implementation of DeepCrowd.

## Usage

### New train

Modify configs/new-train.json, then run

``` bash
python main.py new-train [task number]
```

### Load and resume a previous train

Modify configs/resume.json, then run

``` bash
python main.py resume [task number]
```

### Inference and render

Modify configs/resume.json, then run

``` bash
python main.py inference [task number]
```

### Change hyper-paramters

Most key hyper-paramters are specified in configs/*.json

Other parameters are specified in config.py

### Change task parameter

Task parameters are specified in challenges/*.json

Other stage parameters are specified in env/stage.py

## Dependencies

* pytorch 1.0
* pyqtgraph
* tensorboard
