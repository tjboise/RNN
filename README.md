# RNN-IRI

Deadline: Aug 1st.

[Student Competition Details](https://infopave.fhwa.dot.gov/Reports/LtppDataContest)


[References](https://drive.google.com/drive/folders/1Kw7P4B-bj5MVt_pr8AEQhUe3BINDGKOC)

[Project Schedule](./schedule.md)

[Paper Writing](https://docs.google.com/document/d/1ej3aIgXhci1MoVNrvUFgNPYzbM2bNJzK/edit?usp=sharing&ouid=107909472107329344864&rtpof=true&sd=true)

## Data

[LTPP](https://infopave.fhwa.dot.gov/)

Parameters: 

**IRI<sub>0</sub>**  : The initial IRI, which is the IRI measured immediately after construction. This parameter denotes the quality of construction and was reported in many studies to significantly affect the progression of IRI with age.

**IRI<sub>0</sub>, AADT, cracks, rutting, participation, temperature, change of temperature, maintenance (one-hot)**

Predict: **IRI**


## Model
pytorch

Highway agencies use [IRI thresholds](https://www.fhwa.dot.gov/policyinformation/pubs/hf/pl11028/chapter7.cfm) to characterize road condition; for example, in the United States, an IRI of less than 95 in/mi (1.50 m/km) is generally considered by the Federal Highway Administration to be in "good" condition, an IRI from 96 to 170 in/mi (1.51 to 2.68 m/km) is considered "acceptable", and an IRI exceeding 170 in/mile (2.68 m/km) is considered "poor".

## Result
compare some models: RNN, LSTM, MLP
