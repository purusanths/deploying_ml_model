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
* The overall presion, recall,fbeta on test set are 45.48%,80.42%,58.43% respectively
* The table below show the performance on slices of education

|Education| Precision | recall |fbeta |
|---------|-----------|--------|------|
|10th     | 0.2181    |0.8571  |0.3478|
|11th     | 0.171     |0.7333  |0.2784|
|12th     | 0.3333    |0.8571  |0.4800|
|1st-4th  | 0.1666    |1.0000  |0.2857|
|5th-6th  | 0.0454    |0.5000  |0.0833|
|7th-8th  | 0.0675    |0.7142  |0.1234|
|9th      | 0.0243    |0.5000  |0.0465|
|Assoc-acdm| 0.4500   |0.7758  |0.5696|
|Assoc-voc | 0.4122   |0.6911  |0.5164|
|Bachelors | 0.6912   |0.8810  |0.7746|
|Doctorate | 0.8448   |0.8596  |0.8521|
|HS-grad   | 0.3459   | 0.8347 |0.4891|
|Masters   | 0.7767   | 0.8325 |0.8036|
|Preschool | 0.0000   | 1.0    | 0.0000|
|Prof-school |0.7790  | 0.9305 | 0.8481|
|Some-college|0.4182  | 0.7593 | 0.5393|

## Ethical Considerations
* No PII informations are infereded

## Caveats and Recommendations
* I highly recomend it as a meterial to lean the MLOps concepts

### Reference
* https://arxiv.org/pdf/1810.03993.pdf
