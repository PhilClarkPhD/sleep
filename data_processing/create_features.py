"""
This module ingests eeg/emg data and scores and outputs the feature table needed for model development. It takes in
.wav and .txt files containing the eeg/emg data and scores, respectively. The get_file_paths() function assumes a
certain naming convention and directory structure of the files. In particular, the files should be named
RAT_DAY_scores.txt or RAT_DAY.wav.

Example directory structure for my base_path which is /Users/phil/philclarkphd/sleep/sleep_data:

├── 171N
│       ├── 171N_AD1.wav
│       ├── 171N_AD6.wav
│       ├── 171N_BL.wav
│       ├── 171N_ad1_scores.txt
│       ├── 171N_ad6_scores.txt
│       └── 171N_bl_scores.txt
├── 181N
│       ├── 181N_AD2.wav
│       ├── 181N_AD8.wav
│       ├── 181N_BL1.wav
│       ├── 181N_ad2_scores.txt
│       ├── 181N_ad8_scores.txt
│       └── 181N_bl1_scores.txt

"""

import os
import pandas as pd
from collections import defaultdict
from scipy.io import wavfile
import sleep_functions as sleep


def get_file_paths(base_path: str) -> defaultdict:
    """
    This function retrieves the eeg/emg file (.wav) and score file (.txt) for each recording and adds them to a
    defaultdict dictionary.

    INPUTS:
    - base_path: This is the path in which the subfolders labeled w/ rat ID and day are held. See top of file for
    example directory structure. ADHERING TO THE DIRECTORY STRUCTURE AND FILENAME CONVENTIONS ARE NECESSARY FOR THIS
    FUNCTION TO WORK.

    OUTPUTS:
    - path_dict: A `defaultdict` dictionary in which the first level of keys are rat ID, the next level are day (BL,
    AD1...) and the final level is file type (.wav or .txt)
    -       Example dict: {'171N': {'BL':{'.wav': WAV_PATH, '.txt.': TXT_PATH}, 'AD1': ...}}
    """

    path_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    # create list to save file paths
    wav_file_paths = []
    txt_file_paths = []

    for root, dirs, files in os.walk(base_path):

        # separate .wav and .txt files
        for file in files:
            if file.endswith(".wav"):
                wav_file_paths.append(os.path.join(root, file))
            elif file.endswith(".txt"):
                txt_file_paths.append(os.path.join(root, file))

    # populate dict with .wav filepaths
    for path in wav_file_paths:
        rat_id = str(os.path.basename(os.path.dirname(path)))
        day = str.upper(
            path.split("_")[2].split(".")[0]
        )  # split based on file naming convention (e.g. /171N_BL.wav)
        path_dict[rat_id][day][".wav"] = path

    # populate dict with .txt filepaths
    for path in txt_file_paths:
        rat_id = str(os.path.basename(os.path.dirname(path)))
        day = str.upper(
            path.split("_")[2]
        )  # split based on file naming convention (e.g. /171N_BL_scores.txt)
        path_dict[rat_id][day][".txt"] = path

    return path_dict


def load_scores(path: str) -> pd.DataFrame:
    """
    This function takes in an individual file path for a set of scores (either from Sirenia or from Mora) and returns
    a dataframe with epoch and score as the columns.

    INPUTS:
    -path: path to .txt file containing the scores

    OUTPUTS:
    -df_score: a pd.DataFrame with columns 'epoch' and 'score'
    """

    df_score = pd.read_csv(path)
    df_score.rename(
        columns=lambda x: x.strip(), inplace=True
    )  # remove whitespace from Sirenia's column names

    if df_score.shape[1] > 2:
        # subtract 1 to force epoch start at 0 when loading scores from Sirenia
        df_score.rename(columns={"Epoch #": "epoch", "Score": "score"}, inplace=True)
        df_score["epoch"] = df_score["epoch"] - 1

        return df_score[["epoch", "score"]]
    else:
        return df_score[["epoch", "score"]]


def calculate_features(file_paths: defaultdict) -> pd.DataFrame:
    """
    Creates the feature table from a set of file_paths

    INPUTS:
    - file_paths: A nested dict with the following structure: dict['ID']['day']['.wav'] = '~/file.wav' OR dict['ID'][
    'day']['.txt'] = '~/file.txt'

    OUTPUTS:
    - df: pd.DataFrame containing the calculated features for all .wav files in the file_paths dict. Has the
    following columns:
    EEG_std, EEG_ss, EEG_amp, EMG_std, EMG_ss, EMG_events, delta_rel, theta_rel, theta_over_delta, ID, day, epoch, score
    """

    df = pd.DataFrame()

    for rat_id in file_paths:
        for day in file_paths[rat_id]:
            data_path = file_paths[rat_id][day][".wav"]
            score_path = file_paths[rat_id][day][".txt"]

            # read eeg and emg data from .wav file
            samplerate, data = wavfile.read(data_path)
            df_data = pd.DataFrame(data=data, columns=["eeg", "emg"])

            # make df_features. Add in columns for ID, day, and epoch
            df_features = sleep.generate_features(data=df_data)
            df_features["ID"] = rat_id
            df_features["day"] = day
            df_features["ID_day"] = rat_id + "_" + day
            df_features["epoch"] = df_features.index

            # get scores in df_score
            df_scores = load_scores(score_path)

            # merge feature and score dataframes
            df_combined = df_features.merge(df_scores, on="epoch", how="inner")

            # add this data to the existing df
            df = pd.concat([df, df_combined])

    return df


def make_feature_df(base_path: str) -> pd.DataFrame:
    """
    This function runs all the above functions sequentially and returns the feature df

    INPUTS
    - base_path: The path to the base folder with individual rat IDs and days as subdirectories.

    OUTPUTS
    - df: A pd.DataFrame containing the computed metrics for all files in base_path. Has the following columns:
    EEG_std, EEG_ss, EEG_amp, EMG_std, EMG_ss, EMG_events, delta_rel, theta_rel, theta_over_delta, ID, day, epoch, score
    """

    file_paths = get_file_paths(base_path)
    df = calculate_features(file_paths)

    return df
