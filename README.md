## Introduction

Le projet de ce notebook a été réalisé dans le cadre de la [formation d'ingénieur machine learning proposé par Openclassrooms](https://openclassrooms.com/fr/paths/148-ingenieur-machine-learning).

Il porte sur la classification de race de chien à partir d’image. Nous allons comparer différents modèles de computer vision from scratch ou par transfer learning.
Nous avons utilisé le dataset [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/). Ce dernier est constitué de 20 580 images de chiens triées en 120 classes relatives à leur race. Chaque race a entre 150 et 200 photos.

Pour les modèles from scratch, nous avons procédé de la sorte:

1.  Modèle initial avec préprocessing et ajout d'optimisations (batchnormalization, ...)
2.  Implémentation de l’augmentation de données
3.  Optimisation du modèle en appronfondissant le réseau et y en ajoutant des doubles couches convolutionnelles ainsi qu’un dropout() 

Nous avons ensuite testé deux modèles de transfer learning: Resnet50 et Xception, sans et avec optimisation des hyperparamètres.
Nous avons finalement sélectionné le modèle le plus performant: Xception 

Le meileur modèle a été par la intégré dans un démonstrarteur développé à l'aide du framework [Streamlit](https://streamlit.io).

## Contenu du repositiry:

*  Une application streamlit 
*  Une présentation du projet

## Données: 
dataset [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## Mode d'emploi pour le lancement de l'application
1.  Installer des librairies utilisées pour les projet : 
```pip install -r requrements.txt```

2.  Lancer de l'application streamlit:
```streamlit run P6_app.py```

3.  Ouvrir une fenêtre de navigateur avec l'URL ```http://localhost:8501```
