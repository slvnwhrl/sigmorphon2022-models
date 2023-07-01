# CLUZH models used for the SIGMORPHON 2022 shared tasks

This repository contains the models used by the CLUZH team for the SIGMORPHON 2022 shared tasks ([Paper](https://aclanthology.org/2022.sigmorphon-1.21/)).
We received some requests to share our models, so we decided to make them available here. Currently, the repository
contains the single-best models for the [SIGMORPHON 2022 shared task on morpheme segmentation](https://github.com/sigmorphon/2022SegmentationST) for both task 1 (word-level)
and task 2 (sentence-level).
<br>
<br>
*If you have any questions or problems, please open an issue! :)*

## Usage
1. Make sure you have installed [our neural transducer](https://github.com/slvnwhrl/il-reimplementation).
2. Download the models for the languages you need from this repository as well as the `predict.py` script.
3. Run `predict.py`:
```
# output folder must exist ("." for current folder)
python predict.py --model-folder model --output PATH_TO_OUTPUT_DIR --test PATH_TO_FILE
```
However, if you want to perform **sentence-level** segmentation, you need to perform additional steps because of our strategy:
We simply split sentence into single tokens and perform word-level-segmentation! This means:
Our strategy for the shared task: split sentences into single words and perform word-level segmentation
- After segmentation, glue segmented words back together to from original sentences.
- The test file must be tokenised (one word/token per line), see `data/eng.sentence.preprocessed_split.test.tsv` for an example (based on the test file for part 2 of the shared task).
- The shared task data is already tokenised, so have a look at this data if you work with custom data (I would assume a
spacy tokenised would work fine, but I'd look at the tokenization of e.g. punctuation.)
- I have added a python script `glue_words_task_II.py` that I used to form the original sentences.
- The original data contained double whitespaces and this caused some problems with `glue_words_task_II.py`, so `data/eng.sentence.corrected.test.tsv` is a version of the test data with only single whitespaces.
- Also note that we used a different SINGLE segmentation token to decrease the complexity (↓), so check if this token is contained in your test data (if so, change it manually in the loaded vocabulary instance).

## Citation
If you use these models in your work, please cite the following paper:

```
@inproceedings{wehrli-etal-2022-cluzh,
    title = "{CLUZH} at {SIGMORPHON} 2022 Shared Tasks on Morpheme Segmentation and Inflection Generation",
    author = "Wehrli, Silvan  and
      Clematide, Simon  and
      Makarov, Peter",
    booktitle = "Proceedings of the 19th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sigmorphon-1.21",
    doi = "10.18653/v1/2022.sigmorphon-1.21",
}
```