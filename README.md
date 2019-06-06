# Midi-Track-Identification

Categorize tracks of midi files into four classes:  
* 0: Melody
* 1: Drum
* 2: Bass
* 3: Other

There are two main goals of this task:
* Data Cleansing:  
    * The track names in midi files are usually less meaningful.
    * The program number and channel number are not in GM Midi format.
* Separate perceptually melodic **tracks** based on hand-labeling data

Note that it's possible that there are more than one melodic tracks in a song.

---

## Result
### Accuracy
* Train: 1.0
* Test: 0.98

### Dataset
* Jazz Realbook
* Total: 1094 songs (7:3)
    * Train: 765
    * Test: 329

### Conception
* Feature Extraction
* Classification (Random Forest, n_estimation=100)

### Figures

* Testing Stage
![image](result/confusion_test.png)

* Feature Distribution
![image](result/pitch_mean.png)
![image](result/num_pitches.png)
![image](result/poly_ratio.png)

* *For further understanding, please refer to [here](notebook/Analysis.ipynb)*

---
## Usage

```bash
python compile_dataset.py
python compile_feature.py
python analysis.py
```

To run testing on self-prepared data, please check out the [test.py](test.py)
