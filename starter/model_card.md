# Model Card

## Model Details
* Developed by Purusanth as a project for Machine learning Nano degree course by Udacity
* Random forest 

## Intended Use
* Interned to be used to lean about MLOps.
* Particularly intented to used by Engineers or Students with intermedite knowlwdge about machine learning

## Training Data
* Census income dataset train and test split

## Evaluation Data
* Census income dataset train and test split
* chose as a basic proof-of-concepts

## Metrics
* The overall presion, recall,fbeta on test set are 1,0.01,0.01 respectively
* The table below show the performance on slices of education

|Education| Precision | recall |fbeta|
|---------|-----------|--------|-----|
|10th     | 1.0       |0.0     | 0.0 |
|11th     | 1.0       |0.0     | 0.0 |
|12th | 1.0 | 0.0 | 0.0 |
|1st-4th | 1.0 | 1.0 | 1.0|
|5th-6th | 1.0 | 0.0 | 0.0|
|7th-8th | 1.0 | 0.0 |0.0|
| 9th | 1.0 | 0.0 | 0.0|
| Assoc-acdm | 1.0 | 0.0 | 0.0|
| Assoc-voc | 1.0 | 0.0 | 0.0|
| Bachelors | 1.0 | 0.01 | 0.01|
| Doctorate | 1.0 | 0.03| 0.05|
| HS-grad | 1.0 | 0.0 | 0.0|
| Masters | 1.0 | 0.01 | 0.01|
| Preschool | 1.0 | 1.0 | 1.0|
| Prof-school | 1.0 | 0.06| 0.11|
| Some-college | 1.0 | 0.0 | 0.0|

## Ethical Considerations
* No PII informations are infereded

## Caveats and Recommendations
* I highly recomend it as a meterial to lean the MLOps concepts

### Reference
* https://arxiv.org/pdf/1810.03993.pdf
