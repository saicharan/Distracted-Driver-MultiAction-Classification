# Distracted-Driver-MultiAction-Classification
Data is available in below link for downloading: https://www.dropbox.com/s/0vyzjcqsdl6cqi2/state-farm-distracted-driver-detection.zip?dl=0

Image Classification was done into following 10 classes.

c0: safe driving
c1: texting - right
c2: talking on the phone - right
c3: texting - left
c4: talking on the phone - left
c5: operating the radio
c6: drinking
c7: reaching behind
c8: hair and makeup
c9: talking to passenger

Introduction:
       Driving a car is a complex task, and it requires complete attention. Distracted driving is any activity that takes away the driver’s attention from the road. Several studies have identified three main types of distraction: visual distractions (driver’s eyes off the road), manual distractions (driver’s hands off the wheel) and cognitive distractions (driver’s mind off the driving task).
          The National Highway Traffic Safety Administration (NHTSA) reported that 36,750 people died in motor vehicle crashes in 2018, and 12% of it was due to distracted driving. Texting is the most alarming distraction. Sending or reading a text takes your eyes off the road for 5 seconds. At 55 mph, that’s like driving the length of an entire football field with your eyes closed.
          Many states now have laws against texting, talking on a cell phone, and other distractions while driving. We believe that computer vision can augment the efforts of the governments to prevent accidents caused by distracted driving. Our algorithm automatically detects the distracted activity of the drivers and alerts them. We envision this type of product being embedded in cars to prevent accidents due to distracted driving.

Data:
      We took the StateFarm dataset which contained snapshots from a video captured by a camera mounted in the car. The training set has ~22.4 K labeled samples with equal distribution among the classes and 79.7 K unlabeled test samples. There are 10 classes of images:

Evaluation Metric:
         Before proceeding to build models, it’s important to choose the right metric to gauge its performance. Accuracy is the first metric that comes to mind. But, accuracy is not the best metric for classification problems. Accuracy only takes into account the correctness of the prediction i.e. whether the predicted label is the same as the true label. But, the confidence with which we classify a driver’s action as distracted is very important in evaluating the performance of the model. Thankfully, we have a metric that captures just that — Log Loss.
            Logarithmic loss (related to cross-entropy) measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0 and it increases as the predicted probability diverges from the actual label. So predicting a probability of 0.3 when the actual observation label is 1 would result in a high log loss

Data leakage:
             With the understanding of what needs to be achieved, we proceeded to build the CNN models from scratch. We added the usual suspects — convolution batch normalization, max pooling, and dense layers. The results — loss of 0.014 and accuracy of 99.6% on the validation set in 3 epochs.
                 Oh, well. There was no serendipity after all. So, we looked deeper into what could have gone wrong and we found that our training data has multiple images of the same person within a class with slight changes of angle and/or shifts in height or width. This was causing a data leakage problem as the similar images were in validation as well, i.e. the model was trained much of the same information that it was trying to predict.

Solution to Data Leakage:
                  To counter the issue of data leakage, we split the images based on the person IDs instead of using a random 80–20 split.
                      Now, we see more realistic results when we fit our model with the modified training and validation sets. We achieved a loss of 1.76 and an accuracy of 38.5%.
                      To improve the results further, we explored using the tried and tested deep neural nets architectures.
   1.Transfer Learning
               Transfer learning is a method where a model developed for a related task is reused as the starting point for a model on a second task. We can re-use the model weights from pre-trained models that were developed for standard computer vision benchmark datasets, such as the ImageNet image recognition challenge. Generally, the final layer with softmax activation is replaced to suit the number of classes in our dataset. In most cases, extra layers are also added to tailor the solution to the specific task.
               It is a popular approach in deep learning considering the vast compute and time resources required to develop neural network models for image classification. Moreover, these models are usually trained on millions of images which helps especially when your training set is small. Most of these model architectures are proven winners — VGG16, RESNET50, Xception and Mobilenet models that we leveraged gave exceptional results on the ImageNet challenge.
               Image Augmentation
               Since our training image set had only ~22K images, we wanted to synthetically get more images from the training set to make sure the models don’t overfit as the neural nets have millions of parameters. Image Augmentation is a technique that creates more images from the original by performing actions such as shifting width and/or height, rotation, and zoom. Refer to this article to know more about Image Augmentation.
               For our project, Image Augmentation had a few additional advantages. Sometimes, the difference between images from the two different classes can be very subtle. In such cases, getting multiple looks at the same image through different angles will help. If you look at the images below, we see that they are almost similar but in the first picture the class is ‘Talking on the Phone — Right’ and the second picture belongs to the ‘Hair and Makeup’ class.
    #Extra Layers
              To maximize the value from transfer learning, we added a few extra layers to help the model adapt to our use case. Purpose of each layer:
                    a.Global average pooling layer retains only the average of the values in each patch.
                    b.Dropout layers help in controlling for overfitting as it drops a faction of parameters(bonus tip: it’s a good idea to experiment with different dropout values)
                    c.Batch normalization layer normalizes the inputs to the next layer which allows faster and more resilient training.
                    d.Dense layer is the regular fully-connected layer with a specific activation function.
Which Layers to Train?
                     The first question when doing transfer learning is if we should train only the extra layers added to the pre-existing architecture or if we should train all the layers. Naturally, we started by using the ImageNet weights and trained only the new layers since the number of parameters to train would be lesser and the model would train faster. We saw that the accuracy on validation set plateaued at 70% after 25 epochs. But, we were able to get an accuracy of 80% by training all the layers. Hence, we decided to go ahead with training all the layers.
Which Optimizer to Use?
                     Optimizers minimize an objective function parameterized by a model’s parameters by updating the parameters in the opposite direction of the gradient of the objective function w.r.t. to the parameters. To know more about how different optimizers work, you can refer to this blog.
                     The most popular algorithm in the deep learning world is Adam which combines SGD and RMS Prop. It has been consistently performing better than other optimizers for most problems. However, in our case, Adam showed erratic pattern of descent while SGD was learning gradually. By doing some literature survey, we found that in few cases SGD is superior to Adam because SGD generalizes better (link). As SGD was giving stable results, we used it for all our models.
Which Architectures to Use?
                     We tried multiple transfer learning models with the weights from training on the ImageNet dataset i.e. pre-trained weights.
VGG16:
                     VGG16 model has 16-layers. It mainly uses convolutional techniques along with zero-padding, dropout, max-pooling and flattening.
RESNET50:
                     RESNET50 is an extension of VGG16 model with 50 layers. To counter the issue of difficulty in training a deeper network, feedforward neural networks with “shortcut connections” with reference to the layer inputs have been introduced.
Xception:
                     While RESNET was created with the intention of getting deeper networks, Xception was created for getting wider networks by introducing depthwise separable convolutions. By decomposing a standard convolution layer into depthwise and pointwise convolutions, the number of computations reduces significantly. The performance of the model also improves because of having multiple filters looking at the same level.
MobileNet:
                     MobileNet was a model developed by Google for mobile based vision applications. It was proven to reduce computational costs by at least 9 times. MobileNet uses depth-wise separable convolutions to build light weight deep neural networks. It has two simple global hyper-parameters that efficiently trade off between latency and accuracy.
                     Performance of the Transfer Learning Models
Comparing the Best Models:
                    While each of the architectures above gave us good results, there is a significant variance in the performance of each model for individual classes. From the table below, we notice that different models have the best accuracy for each class. Hence, we decided to build an ensemble of these models.
Ensemble Models:
             Now that we have 7 best models with high variance among the posterior probabilities, we tried multiple ensembling techniques to improve the log loss further.
        Mean Ensembling: 
                    This is the easiest and most widely used ensembling method where the posterior probability is calculated as the mean of the predicted probabilities from the component models.
Trimmed Mean Ensembling: 
                     This is Mean Ensembling by excluding the maximum and minimum probabilities from the component models for each image. It helps in further smoothing our predictions leading to a lower log loss value.
KNN for Ensembling: 
                      Since the images are all snapped from video snippets while drivers were engaged in a distracting activity or were driving, there are a substantial number of images from the same class that are similar. Based on this premise, finding similar images and averaging the probabilities over these images helped us smoothen predicted probabilities for each class.

To find the 10 nearest neighbors, we used outputs from the penultimate layer of VGG16 transfer learning model as features on the validation set.

Learnings:
         We believe that these learnings from our experience would benefit anyone who is working on a deep learning project for the first time like us:
             1. Use Pickle Files: One free resource you can use for your project is ‘Google Colab’. You get access to the GPU which helps when working with huge data due to parallel computing. When using Colab, you can perform the necessary pre-processing steps by reading all your images once and save it in a pickle file. This way, you can pick up where you left off by directly loading the pickle files. You can then start training your models
             2. Early Stopping and Call Backs: Generally, deep learning models are trained with a large number of epochs. In this process, the model might improve accuracy up to a few epochs and then starts diverging. The final weights stored at the end of training will not be the best values i.e. they might not give the minimum log loss. We can use the CallBacks feature in Keras which saves the weights of a model only if it sees improvement after an epoch. You can reduce the training time by using Early Stopping, you can set a threshold on the number of epochs to run after the model stops seeing any improvement.
             3. Mean or Trimmed Mean is better than Model Stacking for Ensembles: The inputs to your stacked model would have high correlations which causes the output to have high variance. Hence, in this case, the simpler approach is the best approach.
             4. Never Lose Sight of the End Application: Doing an ensemble of 7 models and then KNN on the output gave us a good score but if we had to choose a single model that can be used to get good but faster predictions with least amount of resources, Mobilenet would be the obvious choice. Mobilenet is specifically developed with computational restraints in mind which suit the application in the car the best and it had the lowest log loss among the 7 stand-alone models
 We believe that a device with a camera installed in cars that tracks the movements of the driver and alerts them can help prevent accidents.
