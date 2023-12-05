# How Good is the Model in Model-in-the-loop Event Coreference Resolution Annotations

Accompanying code for the paper, [How Good is the Model in Model-in-the-loop Event Coreference Resolution Annotations](https://aclanthology.org/2023.law-1.14) published in the 17th Linguistics Annotation Workshop, ACL 2023.

## Annotation interface

### Preparing ECB+ Dataset

- Event Coref Bank Plus Corpus: https://github.com/cltl/ecbPlus

```
git clone https://github.com/cltl/ecbPlus.git

python parse_ecb.py en_core_web_md test
```

### Running the Prodigy Recipe

```
prodigy evt-coref ecb_test_coref en_core_web_md ./ecb_test_set.json -F coref_recipe.py 
```

## Evaluation Methodology - Simulation

Please see [recall_comparisons.ipynb](recall_comparisons.ipynb)

Lambda Analysis on Dev sets
```
python simulation.py
```

## Citation
If you find this code to be useful, please use the following citation:

```
@inproceedings{ahmed-etal-2023-good,
    title = "How Good Is the Model in Model-in-the-loop Event Coreference Resolution Annotation?",
    author = "Ahmed, Shafiuddin Rehan  and
      Nath, Abhijnan  and
      Regan, Michael  and
      Pollins, Adam  and
      Krishnaswamy, Nikhil  and
      Martin, James H.",
    editor = "Prange, Jakob  and
      Friedrich, Annemarie",
    booktitle = "Proceedings of the 17th Linguistic Annotation Workshop (LAW-XVII)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.law-1.14",
    doi = "10.18653/v1/2023.law-1.14",
    pages = "136--145",
}
```
