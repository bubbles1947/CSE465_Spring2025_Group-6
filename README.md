# CSE465_Spring2025_Group-6

Glaucoma Detection using fundus images of the eye.
1. Mueid Islam Arian-2121947642

Aug_methods:
I've included the following data augmentation techniques:
Random Rotation (up to 20 degrees) - Rotates images randomly within a range of ±20 degrees, which helps the model become robust to different orientations of the eye in the fundus images.
Random Horizontal Flip - Flips images horizontally with a 50% probability, which is useful since glaucoma can affect either eye.
Random Vertical Flip - Flips images vertically with a 50% probability, adding another dimension of orientation variation.
Color Jitter - Adjusts brightness and contrast randomly (±20%), helping the model generalize across different imaging conditions and equipment settings.
Random Affine Transformations - Applies small translations (±10%) and scaling (90-110%) to simulate slight variations in camera positioning and zoom levels.

table plotted on train file. architecture on nertwork.png file.


contribution: Mueid Islam Arian. 
1. data augmentation
2. 5-Fold cross validation.
3. Used transfer learning for image classification.
4. Training the model.
5. Making diagram for the architecture.

DATASET LINK: https://www.kaggle.com/datasets/victorlemosml/refuge2
