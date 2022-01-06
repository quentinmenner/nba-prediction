# Rapport projet Python

## Introduction
Pour choisir le sujet nous avions dès le début une volonté de travailler sur un sport et sur la prédiction de résultats. Notre choix s’est porté sur le basketball et plus particulièrement la NBA (le championnat américain de basketball) pour plusieurs raisons. Premièrement, le basketball est un des sports les plus “prévisible” comme le décrit cet article (voir cet article : https://www.actionnetwork.com/general/futures-betting-odds-sports-parity-nba-nfl-mlb-nhl-golf-tennis ). Cela est certainement dû au fait que le nombre de points dans un match est très élevé par rapport à d’autres sports comme le football. Ensuite la NBA, comme tous les sports américains, possède un répertoire de statistiques assez impressionnant et disponible en accès libre sur internet ce qui était parfait dans le cadre d’un projet de Data Science.

Nous nous sommes donc demandé : Peut-on prévoir les résultats d’un match de sport à partir des performances précédentes de chaque équipe ? Aussi, est-ce que les résultats d’un match sont principalement le fruit du hasard ou est-ce que l’issue d’un match est déterminée par des caractéristiques d’une équipe tangibles et statistiques ?

Pour cela nous avons d’abord extrait les données et les avons nettoyés (I.) , puis avons cherché à les analyser à l'aide d’outils statistiques (II.) afin de pouvoir tester un modèle de machine-learning et d’observer si celui-ci est en capacité de prévoir l’issue des matchs puis de vérifier l’acuité du modèle. (III.)


## Récupération de données
Les données portant sur la NBA étaient nombreuses et étaient disponibles sur différents sites internet. Sur le site officiel de la NBA par exemple (https://www.nba.com/stats/), de très nombreuses données sont disponibles sur les joueurs, les équipes ainsi que les matchs. Ces données peuvent être filtré selon différents critères:
    - par saison, une saison commençant souvent la plupart du temps en octobre et se termine en avril-mai. Ces données sont aussi disponibles mois par mois.
    - par type de saison, une saison de NBA est divisée en plusieurs périodes, notamment une pré-saison, une saison régulière et les “playoffs”.
    - par “mode” : les statistiques sont mesurées par match, pour 48 minutes, par minute, etc…
Dans le cadre de notre projet nous avons privilégié les statistiques globales d’équipe à celles des joueurs. Il y a différentes problématiques avec ces dernières: les joueurs n’ont pas les mêmes postes au sein d’une équipe ce qui influence fortement les statistiques de chacun, aussi tous les joueurs ne jouent pas à tous les matchs ce qui créerait un décalage avec la réalité si nous décidions par exemple de nous intéresser à la moyenne de tous les joueurs d’une équipe. Les statistiques par équipe nous semblent plus intéressantes car elles représentent réellement les performances d’une équipe à un temps t, d’autant plus que ces statistiques sont disponibles pour chaque mois de la saison.
Concernant le type de saison, il nous a semblé plus intéressant de ne choisir qu’un seul type de saison: en effet les modalités de chaque type de saison comme les enjeux sont différents pour chaque type de saison, ce qui pourrait modifier les statistiques des équipes. La prédiction de matchs de “playoffs” à partir de matchs de saison régulière par exemple pourrait ainsi être faussée par les différences entre ces deux types de saison. Dans le cadre de notre projet, nous avons donc privilégié la saison régulière qui est la plus importante en nombre de matchs, ce qui serait un avantage quand il nous faudra “entraîner” le modèle.

Concernant les matchs, de nombreuses données sont disponibles également mais seules certaines données sont nécessaires : les équipes jouant le match, la date et les points marqués par chaque équipe pour déterminer le gagnant.


## Statistiques descriptives


## Modélisation
### Choix des modèles

### Résultats


## Conclusion


## Annexes