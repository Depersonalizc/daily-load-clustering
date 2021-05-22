We want to do clustering



-->



However, our data is high dimensional



-->



Therefore, suffers from curse of dimensionality

theorem

$$\lim_{d \to \infty} E\left(\frac{\operatorname{dist}_{\max} (d) - \operatorname{dist}_{\min} (d)}{\operatorname{dist}_{\min} (d)}\right)  \to 0$$

instable if we use kmeans directly. 



--> 



need dimension reduction. -> explanation of dimension reduction: use essential features to represent original data.



-->



Auto Encoder is a way for this task. Shown promising results in image classification. 



introduce background. And our sine experiment.



-->



Time Series can be considered as 1d image.



our method: problem formulation and architecture: why CNN?

fancy diagram



--> 



Result: elbow plot

shown similar distortion and variance.



-->



Discussion:

Why it doesn't work in our dataset. 

Experiment shows if noise is too large, both methods fail.

AE can handle higher noise but its power is not unlimited.

It may not powerful enough to deal with real data.



-->



Future work:

design feature extractor more robust to noise. 





In stability visualization sine test. 