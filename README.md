# Law Submission 1

## Annotation interface

### Preparing ECB+ Dataset

- Event Coref Bank Plus Corpus: https://github.com/cltl/ecbPlus

```
git clone https://github.com/cltl/ecbPlus.git

python parse_ecb.py en_core_web_md ./ecbPlus/ECB+_LREC2014 test ecb_test_set.json
```

### Running the Prodigy Recipe

```
evt-coref ecb_test_coref en_core_web_md ./ecb_test_set.json -F coref_recipe.py 
```

## Evaluation Methodology - Simulation

Please see [recall_comparisons.ipynb](recall_comparisons.ipynb)

