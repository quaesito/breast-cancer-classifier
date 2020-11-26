## breast-cancer-classifier

## Sample Usage

train

```
python command.py train -c config.json
```

Evaluate the model
please make sure the weights are stored within the same dir where the model is.
```
python command.py evaluate -c config.json path/to/model.h5
```

predict class
```
python command.py predict_class -c config.json path/to/model.h5 dir/with/images/to/be/predicted/
```





