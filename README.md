# Formal-Image-Recognition
Selecting appropriate image for a formal profile image
Strategy:
Our strategy is based on multiple Steps in which we get data from our data set and build our own dataset to make a decision Tree on the basis of which we will find an appropriate image for a person to choose as a profile picture for example for Fiver Account , Upwork and even for his resume.
We have a dataset of images downloaded from Kagle and link is given below.
We will run those images on our program which will make another dataset by analyzing images from different point of views for example 
1 -Detecting the quality of an image
2 -Checking either the image is blurry or not
3 -Identifying a number of faces present in the image through facial recognition system.
4 -Detecting the Eyes of the person in the image 
5 -Prediciting the smile of a person in image
This is a reference link from where we study how to detect the quality of an image.
https://www.learnopencv.com/image-quality-assessment-brisque/

First of all we will assign numbers to our every characteristic of image and we will build a Table and assign label to every image (either the image is good or bad) by selecting an appropriate threshold.
From that data we will build a Decision Tree and then run our test image on that tree and analyze their answers.
Kaggle link from where we download the dataset of images which are profile pictures of real accounts.
https://www.kaggle.com/chrisroths/peoples-republic-of-tinder-1/version/1#dudes1.zip
