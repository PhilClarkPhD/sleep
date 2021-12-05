# Sleep analysis and GUI

[video walkthrough](https://www.youtube.com/watch?v=5LG8gb8FvUw)

## *Background*

Our lab studies sleep in the context of addiction and dopamine. One of our recent [publications](https://www.nature.com/articles/s41386-020-00879-2) shows that dopamine uptake rates vary across the sleep/wake cycle, and more recent data from our lab indicate that restoring sleep during withdrawal from cocaine reduces drug craving and normalizes striatal dopamine transmission.  Pretty neat!

As cool as sleep studies may be, in our lab they have historically been a pain to do. We hand-score all of the sleep data, assigning each 10 second bin a score of Wake, Non REM, or REM. Scoring this way takes ~2 hours per 24hr recording. Considering we record each animal for 7+ days, that's 14+ hours of labor to score the data from just *one* rat. My first goal for this project was to create a ML model that could automatically generate scores from the raw data. This has cut our analysis time down from 14 hours to several minutes.

The next issue to solve was related to the software we had been using to view and score our data. The program was slow, buggy, and prone to crashing. And upgrading to its newest version would have cost alot of $$$. As an alternative I generated a GUI in which users can load raw data and view plots containing eeg/emg signals, a hypnogram, a power spectrum, and a simple bar plot. This required learning PyQt and took alot of trial and error - but it was worth it!

In the end I was able to load my model into the GUI so that the user can load a file, score it instantly, and then cruise through the data and see how well the model performed. See the walkthrough video above for a demonstration.

Future additions will include options to train a new model, evaluate model performance, and run batch analyses to make things even faster.

## *In this repo...*
* **pilot.ipynb:** A jupyter notebook in which I first played with loading the sleep data and generating power spectrum plots (critical for the analysis step!)
* **gui.py:** This file contains the PyQT code for running the application, including all plots, buttons, etc. See the walkthrough video above!
* **sleep_functions.py:** This file contains the functions required for generating the power spectrum as well as the metrics that feed into the ML model. 
* **sleep_model_120121.joblib:** This file contains the model parameters that we will load int the app to automatically score the data.
