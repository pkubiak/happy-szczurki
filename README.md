# :rat: happy-szczurki

## :mortar_board: Knowledge  ##

### Types of Ultrasonic Vocalizations ###

> Ultrasonic vocalizations (USVs) have been observed in a number of rodent species (Sales 1972). In adult laboratory rats,
two main types of USVs have been described: 22-kHz and 50-kHz calls (see Brudzynski 2009 for review). The **22-kHz
call type has been termed a distress or “alarm” vocalization** (Litvin et al. 2007), as it can be elicited by the presentation of a predator, painful stimuli, startling noises, and intermale aggression (Blanchard et al. 1991; Calvino et al. 1996; Han et al. 2005; Kaltwasser 1991; Thomas et al. 1983). In contrast, calls of the **50-kHz category have been detected in naturalistic appetitive contexts**, such as during play, mating behavior, exploratory activity, or in anticipation of food reward (Burgdorf et al. 2000; Knutson et al. 1998; Sales 1972). 50-kHz calls have also been elicited by several non-natural appetitive stimuli, particularly rewarding electrical brain stimulation and amphetamine (AMPH) administration (Ahrens et al. 2009; Burgdorf et al. 2000, 2001a, 2007; Simola et al. 2009; Thompson et al. 2006; Wintink and Brudzynski 2001). Of note, the 50-kHz class of calls encompasses a wide frequency range (30–90 kHz) (Kaltwasser 1990; Sales and Pye 1974), and these calls vary considerably in spectrographic structure
>> Identification of multiple call categories within the rich repertoire of adult rat 50-kHz ultrasonic vocalizations: effects of amphetamine and social context

- https://link.springer.com/article/10.1007%2Fs00213-010-1859-y
- https://www.nature.com/articles/s41598-019-44221-3

![](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-019-44221-3/MediaObjects/41598_2019_44221_Fig4_HTML.png?as=webp)
![](https://www.researchgate.net/profile/Maria_Luisa_Scattoni/publication/23195427/figure/fig14/AS:340731513327618@1458248131236/Typical-sonograms-of-ultrasonic-vocalizations-classified-into-ten-distinct-categories-of.png)

| code name | meaning | description |
|-----------|---------|-------------|
| SH | Short | USVs with duration of less than 12 ms | 
| FM | ??? | |
| RP | Upward/Downward Ramp ? | USVs displaying a monotonic increase/decreasing in frequency |
| FL | Flat | USVs bearing a near-constant frequency |  
| ST | ??? | |
| CMP | ??? | |
| IU | Inverted U / chevron | USVs possessing a monotonic increase in frequency followed by a monotonic decrease in frequency, resembling the shape of an inverted U |
| TR | Trill | USVs displaying a rapid frequency oscillation, usually appearing as a sinusoid oscillation |
| RM | ??? | |

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4598429/

## :chart_with_upwards_trend: Results ##

### :dart: USVs detection ###
Detecting time intervals which contains USVs. 

| model type | train set | test set | f1-score | precision | recall |
|-------|-----------|----------|----------|----------|--------|
| SVC   | ch1-2018-11-20_10-29-02_0000012.wav.npz | ch1-2018-11-20_10-26-36_0000010.wav.npz | 0.92 | 0.92 | 0.91 |
| SVC   | ch1-2018-11-20_10-29-02_0000012.wav.trimed.npz | ch1-2018-11-20_10-26-36_0000010.wav.trimed.npz | 0.93 | 0.94 | 0.93 |
| RandomForestClassifier |  | | | | 
| CNN | | | | 
| LSTM | | | |

- TODO: share pickle models
- TODO: what is used weighting function?
- TODO: masking training ?

### :dart: USVs' boundaries detection ###
Detecting time and frequency intervals (bounding box) which contains USVs.

#### TBD ####

### :dart: USVs classification ###
Assigning one of USV types to each detected box.

#### TBD ####


## :cry: Existing solutions ##

- https://github.com/DrCoffey/DeepSqueak

## :books: Materials to read ##

### General ###
1. https://medium.com/@etown/great-results-on-audio-classification-with-fastai-library-ccaf906c5f52
2. https://www.kaggle.com/maxwell110/beginner-s-guide-to-audio-data-2 / https://www.kaggle.com/c/freesound-audio-tagging-2019/notebooks


### Models to use ###

#### Hidden Markov Models (HMM) ####
1. http://lasa.epfl.ch/teaching/lectures/ML_MSc_Advanced/Slides/Lec_XI_HMM_extensions.pdf
2. https://blog.goodaudience.com/music-genre-classification-using-hidden-markov-models-4a7f14eb0fd4

#### Support Vector Machine (SVM) ####
1. https://www.academia.edu/1549822/Audio_Classification_Using_Support_Vector_Machines_and_Independent_Component_Analysis
2. https://www.researchgate.net/publication/267782696_Non-speech_environmental_sound_classification_using_SVMs_with_a_new_set_of_features

#### Parallel Recognition ####
1. https://www.sciencedirect.com/science/article/pii/S0003682X16305254

#### LSTM ####
1. https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/
