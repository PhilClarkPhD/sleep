# Mora sleep analysis

[video walkthrough](https://www.youtube.com/watch?v=5LG8gb8FvUw)

## Setup
Run `python -m pip install -r requirements.txt`

## Running sample data
1. Run `python Mora.py`
2. Click `Load Data` and select `sample_signal.wav` in the sample data folder
3. Click on the EEG graph and press the left and right arrow keys to move through the data
4. To see some example scores, click `Load Scores` and select `sample_scores.txt` in the sample data folder
5. Press W,E,R,T to change the scores manually and save scores to a .txt file by clicking `Export Scores`
6. Now let's use machine learning to automatically score the data - clear the current scores by clicking `Clear Scores`
7. In the model tab, click `Load Model` and select the .pkl file in sample data folder, then click `Score Data`
8. A window will pop up, pre-populated with the number `2`, click `"Ok"` and wait a few moments
9. Go back to the home tab to see how well the model worked!

## *In this repo...*
* **Mora.py:** This file contains the PyQT code for running the application, including all plots, buttons, etc. See the walkthrough video above!
* **sleep_functions.py:** This file contains the functions required for generating the power spectrum as well as the metrics that feed ML model. 
* **sample data:** This folder contains sample data, scores, and model to play with!
* **requirements.txt:** Dependencies for running this program

## *Background*
Our lab studies sleep in the context of addiction and dopamine. One of our recent [publications](https://www.nature.com/articles/s41386-020-00879-2) shows that dopamine uptake rates vary across the sleep/wake cycle, and more recent data from our lab indicate that restoring sleep during withdrawal from cocaine reduces drug craving and normalizes striatal dopamine transmission.  Pretty neat!

As cool as sleep studies may be, in our lab they have historically been a pain to do. We hand-score all of the sleep data, assigning each 10 second bin a score of Wake, Non REM, or REM. Scoring this way takes ~2 hours per 24hr recording. Considering we record each animal for 7+ days, that's 14+ hours of labor to score the data from just *one* rat. My first goal for this project was to create a ML model that could automatically generate scores from the raw data. This has cut our analysis time down from 14 hours to several minutes.

The next issue to solve was related to the software we had been using to view and score our data. The program was slow, buggy, and prone to crashing. And upgrading to its newest version would have cost alot of $$$. As an alternative I generated a GUI in which users can load raw data and view plots containing eeg/emg signals, a hypnogram, a power spectrum, and a simple bar plot. This required learning PyQt and took alot of trial and error - but it was worth it!

In the end I was able to combine the model and the GUI, so that the user can load a file, score it instantly, and then cruise through the data and see how well the model performed. See the walkthrough video above for a demonstration. We're calling this program Mora, after the demon from slavic mythology that causes nightmares 😵.

Future additions will include options to train a new model, evaluate model performance, and run batch analyses to make things even faster.
