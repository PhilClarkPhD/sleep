# Sleep analysis and GUI

##*Background*

Our lab studies sleep in the context of addiction and dopamine. One of our recent [publications](https://www.nature.com/articles/s41386-020-00879-2) shows that dopamine uptake rates vary across the sleep wake cycle, and more recent data from our lab indicate that restoring sleep during withdrawal from cocaine reduces drug craving and normalizes striatal dopamine transmission.  Neat!

As cool as sleep studies may be, in our lab they have historically been a pain in the butt to do. We typically hand-score the data, assigning each 10 second bin a score of Wake, None REM or REM. Scoring the data this way takes ~2 hours per 24hr recording. Considering we record each animal for 7+ days, that's 14+ hours of labor to score the data from just *one* rat. Not very efficient! My first goal for this project was to find a ML model that could generate scores from the raw data. This cut our analysis time down from 14 hours to a matter of minutes.

The next issue to solve was related to the software we had been using. I won't name the program here, but it was slow, buggy, and prone to crashing. And upgrading to its newest version would have cost $$$. As an alternative I generated a GUI in which users can load raw data and view plots containing eeg/emg signals, a hypnogram, a power spectrum plot, and a simple bar plot. This required learning PyQt and took alot of trial and error - but it was worth it!

In the end I was able to load my model into the GUI so that the user can load a file, generate scores for it instantly, and then cruise through the data and see how well the model performed.

Future additions will include options to train a new model, evaluate model performance, and run batch analyses to make things even faster.

##*In this repo...*
* **GUI:** Download this folder and see the walkthrough video above if you want to open the app and play with some example data!
* **pilot.ipynb:** A jupyter notebook in which I first played with loading the sleep data and generating power spectrum plots (critical for analysis!)
* **sleep_functions.py:** This file contains the functions required for generating the power spectrum as well as the metrics that feed into the ML model. 
* **sleep_app.py:** This file contains the PyQT code for running the application, including all plots, buttons, and threading to increase plotting speed
