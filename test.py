import numpy as np
from pypianoroll import Multitrack
from sklearn.externals import joblib 
from utils import features
from utils.misc import traverse_dir

PATH_MODEL = 'result/model.pkl'

def testing(X):
    # load model
    loaded_model = joblib.load(PATH_MODEL)
    
    # prediction
    predict_loaded_y = loaded_model.predict(X)
    return list(predict_loaded_y)

def identify_single_track(pianoroll):
    X = features.extract_features(pianoroll)
    test_x = np.array([X])
    return testing(test_x)[0]

def identify_multiple_track(pianorolls):
    # extract features
    num = len(pianorolls)
    test_x = []
    for idx in range(num): 
        X = features.extract_features(pianorolls[idx])
        test_x.append(X)
    test_x = np.array(test_x)
    return testing(test_x )
   
def identify_song(input_obj):
    # loading
    if isinstance(input_obj, str):
        multi = Multitrack(filename)
    else:
        multi = input_obj
    
    # processing
    pianorolls = []
    for track in multi.tracks:
        pianorolls.append(track.pianoroll)
    ys =  identify_multiple_track(pianorolls)
    return ys

if __name__ == '__main__':
    # testing sample code
    testing_dir = 'Raw_Data/jazz_realbook'
    testing_files = traverse_dir(testing_dir)

    fidx = 0
    filename = testing_files[fidx]
    predict_y = identify_song(filename)

    print(predict_y)
