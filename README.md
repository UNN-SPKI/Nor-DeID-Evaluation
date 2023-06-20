# Nor-DeID-Evaluation
Evaluation harness for deidentifying Norwegian clinical text

## Installing

To get the NorSynthClinical-PHI dataset when setting up the project, clone the project with submodules:

```
git clone --recurse-submodules https://github.com/UNN-SPKI/Nor-DeID-Evaluation.git
```

Create a virtual environment and install the dependencies with:

```
$ python -m venv venv
$ venv/Scripts/activate
(venv) $ pip install -r requirements.txt
(venv) $ python -m spacy download en_core_news_sm
(venv) $ python -m spacy download nb_core_news_sm
```

## Running

[eval.py](eval.py) will evaluate an NER model for a specific dataset. By default, it will use the [NorSynthClinical-PHI](https://github.com/synnobra/NorSynthClinical-PHI) dataset with a dummy model which doesn't predict any PHI.

You can specify which model and dataset to use with `--dataset` and `--model`:

```
(venv) $ python eval.py --dataset norsynthclinical --model dummy
```

## Adding new methods 

* Implement a class in `models/` which predicts entities for the documents in `doc_bin` and returns a list of SpaCy Examples:

```
class NewModel:
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language) -> List[spacy.training.Example]:
        return []
```

* In `load_model` in [eval.py](eval.py), create a new instance of your new model. 


## Adding new datasets

* In `load_dataset` in [eval.py](eval.py), create a method which converts your dataset to a SpaCy DocBin.
