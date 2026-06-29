# %%
#if you are working on kaggle:
# basePath='/kaggle/input/seed-iv/'
# rndr='iframe'

#if you are working locally
basePath='SEED-IV'
rndr=''

# %%
import numpy as np
import scipy as sc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import re
import os

# %%
"""
Download the SEED-IV dataset from here : https://www.kaggle.com/datasets/phhasian0710/seed-iv
"""

# %%
"""
## Explaining the files structure:
### The "eeg_raw_data" folder:
   * Contains 3 inner folders named 1, 2 ,3 corresponding to the 3 sessions.
      * Each .mat file inside those folders is for a subject from the  15 subjects (named with {SubjectName}_{Date}.mat), which contains more files:
         * The .mat file contains the EEG signals recorded during 24 trials for 62 channels
   * Each of the 24 trials in each session folder (1, 2 or 3) has a label, and the labels are the same across all subjects 

**Each class has 18 trial, so the data is perfectly balanced**

session [1-3]
   * subject [1-15]
      * trial [1-24]
         * channel [0-62]
"""

# %%
"""
### So that we know this , we can calculate the dataset size:
3 sessions * 15 subject * 24 trial * 62 channels = 66960 raw EEG signal (before windowing)
"""

# %%
"""
**Label Mapping**:
- Neutral: 0
- Sad: 1
- Fear: 2
- Happy: 3
"""

# %%
labels = np.array([
    [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
])

# %%
labels.shape

# %%
"""
Mapping the sad and fear emotions to negative. This will lead to an unbalanced dataset , which is a problem we will solve later
"""

# %%
#currently neutral:0 , happy:3 , sad:1 , fear:2
labels[labels==2] = 1  # changing fear labels from 2 to 1
#currently neutral:0 , happy:3 , sad:1 , fear:1
labels[labels==0] = -1  # changing neutral labels from 0 to -1
#currently neutral:-1 , happy:3 , sad:1 , fear:1
labels[labels==3] = 0  # changing happy labels from 3 to 0
#currently neutral:-1 , happy:0 , sad:1 , fear:1

# %%
"""
**Final label mapping**:
- Neutral: -1
- Positive (Happy): 0
- Negative (Sad , Fear): 1
"""

# %%
labels

# %%
"""
We can't load more than one session at a time because of the resources it needs, if we try to load all the data the computer will crash
"""

# %%
def loadSession(k):
    sessionPath=f'SEED-IV/eeg_raw_data/{k}/'
    sessionSubjects=os.listdir(sessionPath)
    s=[]
    for i,subjectFile in enumerate(sessionSubjects):
        sub=sc.io.loadmat(sessionPath+subjectFile)
        # sub = {int(re.search(r'(\d+)$', k).group(1))-1: v for k, v in sub.items() if not k.startswith('__')}
        sub = [v for k, v in sub.items() if not k.startswith('__')]
        s.append(sub)
    return s

# %%
# s1=loadSession(1)

# %%
"""
**To index a channel by its name**
"""

# %%
channelsMapping=pd.read_excel(f'{basePath}/Channel Order.xlsx',header=None, names=['channels']).reset_index() 
channelsMapping.set_index('channels', inplace=True)

# %%
def getChannel(channel):
    return channelsMapping.loc[channel]['index'] 

# %%
"""
### Let's play with the files a bit to understand it better.
"""

# %%
def loadSubject(session,subject):
    '''This function is 1-based'''
    for file in os.listdir(f'{basePath}/eeg_raw_data/{session}/'):
        if file.startswith(f'{subject}_'):
            subData=sc.io.loadmat(f'{basePath}/eeg_raw_data/{session}/{file}')
            break
    subData = [v for k, v in subData.items() if not k.startswith('__')]
    return subData

# %%
origSamplingRate = 1000
newSamplingRate = 200
q = int(origSamplingRate/newSamplingRate) # step size for down sampling
windowSize=4 #4 seconds
overlapSize=0.1 #percent of overlapped points between segments
noOfSamples = newSamplingRate * windowSize # = 800
bandpassWindow = (4,50) #Hz

# %%
def downSample(trial):
    return np.array([ch[::q] for ch in trial])

# %%
def segmentChannel(ch):
    '''
    This function segments the channel with window size of 800 samples while applying overlapping of size 10% , additionally if the 
    channel isn't divisible by the window size , the last segment will be ch[-window size] , which means its overlap with the previous
    segment can be any value from 10% to 99%
    '''
    s = []
    stepSize= int(newSamplingRate * windowSize *(1-overlapSize))
    segmentsCount = int(np.floor((len(ch) - noOfSamples) / stepSize)) + 1
    for i in range(segmentsCount):
        start=i*stepSize
        end=(i*stepSize)+noOfSamples
        s.append(ch[start:end])

    #to cover the whole signal
    if end+1< len(ch):
        s.append(ch[-noOfSamples:])
    return np.array(s)

# %%
def segmentTrial(trial):
    return [segmentChannel(ch) for ch in trial]

# %%
def preProcess(subData):
    f'''This function applies band pass filter {bandpassWindow} then down sampling to 200 Hz'''
    b, a = sc.signal.butter(4, Wn=bandpassWindow, btype='bandpass', fs=origSamplingRate)
    s = [sc.signal.lfilter(b, a, trial) for trial in subData]
    s = [downSample(trial)  for trial in s]
    s = [sc.stats.zscore(trial, axis=1) for trial in s]
    s = [segmentTrial(trial)  for trial in s]
    return s

# %%
s = loadSubject(1,1)

# %%
"""
### Plotting the signal to show the effect of preprocessing
we will plot the the signal of the first channel of the first subject in the first trial in the first session
"""

# %%
fig = px.line(s[0][0][:5001])
fig.show(renderer=rndr)

# %%
b, a = sc.signal.butter(4, Wn=bandpassWindow, btype='bandpass', fs=origSamplingRate)
filteredSignal = [sc.signal.lfilter(b, a, trial) for trial in s]

# %%
"""
#### After applying butterworth bandpass filter 
"""

# %%
fig=px.line(filteredSignal[0][0][:5001])
fig.show(renderer=rndr)

# %%
"""
#### After downsampling from 1000 to 200
"""

# %%
downSampledSignal = [downSample(trial)  for trial in filteredSignal]

# %%
fig = px.line(downSampledSignal[0][0][:1001])
fig.show(renderer=rndr)

# %%
normalizedSignal = [sc.stats.zscore(trial, axis=1) for trial in downSampledSignal]

# %%
fig = px.line(normalizedSignal[0][0][:800])
fig.show(renderer=rndr)

# %%
"""
Some plotting for comparisons
"""

# %%
"""
Seeing how different subject have their EEG signals given the same videos (same label)
"""

# %%
# These are the positive indexes of the first session
posIndex=np.flatnonzero(labels[0]==0)

# %%
s1 = loadSubject(1,1)
s2 = loadSubject(1,2)
s3 = loadSubject(1,3)

# %%
p1=preProcess(s1)
p2=preProcess(s2)
p3=preProcess(s3)

# %%
fig = make_subplots(
    rows=3, 
    cols=1, 
    subplot_titles=("Subject 1", "Subject 2", "Subject 3"),
)
fig.add_trace(
    go.Scatter(y=p1[posIndex[2]][getChannel('PZ')][0], mode="lines", name="Subject 1"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=p2[posIndex[2]][getChannel('PZ')][0], mode="lines", name="Subject 2"),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=p3[posIndex[2]][getChannel('PZ')][0], mode="lines", name="Subject 3"),
    row=3, col=1
)
fig.update_layout(
    title_text="A segment of the EEG PZ Channel Across 3 subject given the same trial (same movie and same label)", 
    height=700, 
    showlegend=False
)
fig.update_xaxes(title_text="Sample Number", row=3, col=1)

fig.show(renderer=rndr)