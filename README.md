# ASL Detection

## Setting up virtualenv
```
python3 -m venv venv
source venv/bin/actiavate
```

## Installing dataset
```
pip install kaggle
kaggle datasets download grassknoted/asl-alphabet
unzip asl-alphabet.zip -d ./data
rm asl-alphabet.zip
```