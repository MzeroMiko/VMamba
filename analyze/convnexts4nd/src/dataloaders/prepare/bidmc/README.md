Respiratory Rate:
https://zenodo.org/record/4001463/files/BIDMC32RR_TRAIN.ts
https://zenodo.org/record/4001463/files/BIDMC32RR_TEST.ts

Heart Rate:
https://zenodo.org/record/4001456/files/BIDMC32HR_TRAIN.ts
https://zenodo.org/record/4001456/files/BIDMC32HR_TEST.ts

Blood Oxygen Saturation:
https://zenodo.org/record/4001464/files/BIDMC32SpO2_TRAIN.ts
https://zenodo.org/record/4001464/files/BIDMC32SpO2_TEST.ts

1. Create folder `data/bidmc` (relative to repo base)
2. Download the above datasets into `data/bidmc/RR`, `data/bidmc/HR`, `data/bidmc/SpO2`
3. Copy processing scripts `cp src/dataloaders/prepare/bidmc/{process_data.py,data_loader.py} data/bidmc`
4. Run script `cd data/bidmc && python process_data.py`
