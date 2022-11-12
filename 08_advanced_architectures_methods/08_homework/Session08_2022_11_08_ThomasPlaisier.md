 **Homework**

 I work with kinematic and EMG data for analysing movement issues in stroke. For example, I work with grid-like EMG electrodes that measure muscle activity in the biceps using 64 channels. By decomposing the data of these arrays, we can isolate individual motor units within the muscle and track their activity in different conditions. I can see an application for AI to help with this decomposition step: currently it requires a lot of manual cleanup and intervention to isolate the motor units from noise and artifcats in the sources identified by e.g. ICA. With AI, we can train a classification network to help us with this classification.

 The task would be giving the network a form of blind-source-separated EMG data, and having it classify data of these channels as noise or motor units. Ideally, the network would also immediately categorize the motor unit as a specific motor unit, but that can be handled in a separate pipeline.

 To do this, the input is EMG data of the electrode array. 

 A convolutional neural network can be appropriate for this task, as it can classify what is essentially a 64 pixel 'image' of the electrode. A 3D CNN can leverage the information across time, to track individual motor units across trials. A trickier part is the training process: most likely a dataset of labeled training data would have to be created to train the model which can be time-consuming.

 To measure the success of the model, the similarity between motor unit data across trial can be used: a high correlation between waveforms of motor units from one trial or even session to the next indicates that we are able to identify the same motor unit multiple times. Occasional comparison to manual decomposition and classification data can also be used to determine classification accuracy.