# Dog Breed Identifcation

Dogs are the most loyal animals on the planet, and as they take care of us, it's our responsibility to take care of them too. I'm from Punjab, where many people buy dogs of different breeds, often importing them from other countries. Unfortunately, many of these dogs can't adapt to the environment, leading to early deaths or susceptibility to various diseases. This issue affects all animals, but my current focus is on dogs.

To address this problem, I developed a model that helps people identify dog breeds and provides additional information, such as the type of weather the dog is accustomed to.

## For Modeling
I leveraged my knowledge of machine learning, specifically deep learning, to tackle this problem. I used a dataset from Kaggle containing over 10,000 dog images with their corresponding breeds. The dataset covered 120 breeds, and I must admit, I only knew 4-5 breeds myself.

I converted the images into tensors, which made manipulation and storage more efficient. Given the large number of images, I created batches of 32, each containing 25 images, which significantly saved processing time. I employed TensorFlow's MobileNetV2 for transfer learning and trained the model with 100 epochs. To optimize training, I used callback operations to stop the model if accuracy started decreasing or plateaued. As a result, I only needed to run 19-21 epochs.

## For Deployment
I utilized my knowledge of website design and sought help from Google for additional guidance. For the front end, I used HTML, CSS, and JavaScript. For the backend, I chose Flask, which is lightweight and saved me a lot of time. The user's selected image is sent to the backend, processed, and predictions are made. The results, along with the image and a link to a website with more information about the dog breed, including its temperament and weather preferences, are then displayed on the front end.
