# Nor-DeID-Evaluation
Evaluation harness for deidentifying Norwegian clinical text

## Installing

We currently assume that Python 3.9 is used: newer versions may have issues building wheels for Fasttext on Windows (required by `string2string`)

The synthetic dataset from "Instruction-guided deidentification with synthetic test cases for Norwegian clinical text", presented at NLDL 2024, is included in the repository.

To get the NorSynthClinical-PHI dataset when setting up the project, clone the project with submodules:

```
git clone --recurse-submodules https://github.com/UNN-SPKI/Nor-DeID-Evaluation.git
```

Create a virtual environment and install the dependencies with:

```
$ python -m venv venv
$ venv/Scripts/activate
(venv) $ pip install -r requirements.txt
(venv) $ python -m spacy download en_core_web_sm
(venv) $ python -m spacy download nb_core_news_sm
```

## Running

[eval.py](eval.py) will evaluate an NER model for a specific dataset. By default, it will use the [NorSynthClinical-PHI](https://github.com/synnobra/NorSynthClinical-PHI) dataset with a dummy model which doesn't predict any PHI.

There are four mandatory arguments:

* `--dataset` can either be a preset identifier (`norsynthclinical` or `synthdeid`) for a dataset or the path to a SpaCy DocBin
* `--model` is the family of model to use: `spacy` is the baseline rule-based method, while `gpt-chat` is the OpenAI chat completion interface 
    * You likely also need to set `--modelName` for the specific model to use: e.g. `gpt-4` to use GPT-4
* `--prompt_path` is the path to a text file with the prompt to append to each evaluation sample
* `--mode` is how to evaluate the model: `annotate` assumes that each PHI term is annotated with XML-style markers (e.g. `I'm <First_Name>Example</First_Name>`), while `replace` assumes that the terms are replaced outright (e.g. `I'm <First_Name>`)


```
(venv) $ python eval.py --dataset norsynthclinical --model spacy --prompt_path prompts/null.txt --mode annotate
```

## Adding new methods 

* Implement a class in `models/` which accepts a SpaCy DocBin and Language, and a string `mode` which is either `replace` or `annotate`. If `mode` is `annotate`, the method should return a list of SpaCy Examples with the annotated entities. If `mode` is `replace`, it should return a list of strings where the PHI terms are either removed or replaced with class markers (of the format `<Class>`, e.g. `<First_Name>`):

```
class NewModel:
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language, mode: str) -> Union[List[spacy.training.Example], List[str]]:
        return []
```

* In `load_model` in [eval.py](eval.py), create a new instance of your new model. 


## Adding new datasets

* In `load_dataset` in [eval.py](eval.py), create a method which converts your dataset to a SpaCy DocBin.
