# Nor-DeID-Evaluation
Evaluation harness for deidentifying Norwegian clinical text.

### Downloading
The script includes the [NorSynthClinical-PHI](https://github.com/synnobra/NorSynthClinical-PHI) dataset as a [Git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules). 

Use the `--recurse-submodules` switch in `git clone` to download it along with the rest of the project:

`git clone --recurse-submodules https://github.com/UNN-SPKI/Nor-DeID-Evaluation.git`

<!-- ### Running
By default, eval.py will use a dummy model (which classifies nothing as PHI) on the NorSynthClinical-PHI dataset. -->

