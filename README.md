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

```
rm -r ./data/asl_alphabet_test/
```

```
cd data
mv asl_alphabet_train/asl_alphabet_train/* asl_alphabet_train/ && rmdir asl_alphabet_train/asl_alphabet_train
mv asl_alphabet_train/ train
cd ..
```