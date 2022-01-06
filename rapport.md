# Rapport projet Python

## Introduction
Pour choisir le sujet nous avions dès le début une volonté de travailler sur un sport et sur la prédiction de résultats. Notre choix s’est porté sur le basketball et plus particulièrement la NBA (le championnat américain de basketball) pour plusieurs raisons. Premièrement, le basketball est un des sports les plus “prévisible” comme le décrit cet article (voir cet article : https://www.actionnetwork.com/general/futures-betting-odds-sports-parity-nba-nfl-mlb-nhl-golf-tennis ). Cela est certainement dû au fait que le nombre de points dans un match est très élevé par rapport à d’autres sports comme le football. Ensuite la NBA, comme tous les sports américains, possède un répertoire de statistiques assez impressionnant et disponible en accès libre sur internet ce qui était parfait dans le cadre d’un projet de Data Science.

Nous nous sommes donc demandé : Peut-on prévoir les résultats d’un match de sport à partir des performances précédentes de chaque équipe ? Aussi, est-ce que les résultats d’un match sont principalement le fruit du hasard ou est-ce que l’issue d’un match est déterminée par des caractéristiques d’une équipe tangibles et statistiques ?

Pour cela nous avons d’abord extrait les données et les avons nettoyés (I.) , puis avons cherché à les analyser à l'aide d’outils statistiques (II.) afin de pouvoir tester un modèle de machine-learning et d’observer si celui-ci est en capacité de prévoir l’issue des matchs puis de vérifier l’acuité du modèle. (III.)


## Récupération de données
Les données portant sur la NBA étaient nombreuses et étaient disponibles sur différents sites internet. Sur le site officiel de la NBA par exemple (https://www.nba.com/stats/), de très nombreuses données sont disponibles sur les joueurs, les équipes ainsi que les matchs. Ces données peuvent être filtré selon différents critères :
- par saison, une saison commençant souvent la plupart du temps en octobre et se termine en avril-mai. Ces données sont aussi disponibles mois par mois.
- par type de saison, une saison de NBA est divisée en plusieurs périodes, notamment une pré-saison, une saison régulière et les “playoffs”.
- par “mode” : les statistiques sont mesurées par match, pour 48 minutes, par minute, etc…

Dans le cadre de notre projet nous avons privilégié les statistiques globales d’équipe à celles des joueurs. Il y a différentes problématiques avec ces dernières : les joueurs n’ont pas les mêmes postes au sein d’une équipe ce qui influence fortement les statistiques de chacun, aussi tous les joueurs ne jouent pas à tous les matchs ce qui créerait un décalage avec la réalité si nous décidions par exemple de nous intéresser à la moyenne de tous les joueurs d’une équipe. Les statistiques par équipe nous semblent plus intéressantes car elles représentent réellement les performances d’une équipe à un temps t, d’autant plus que ces statistiques sont disponibles pour chaque mois de la saison.

Concernant le type de saison, il nous a semblé plus intéressant de ne choisir qu’un seul type de saison: en effet les modalités de chaque type de saison comme les enjeux sont différents pour chaque type de saison, ce qui pourrait modifier les statistiques des équipes. La prédiction de matchs de “playoffs” à partir de matchs de saison régulière par exemple pourrait ainsi être faussée par les différences entre ces deux types de saison. Dans le cadre de notre projet, nous avons donc privilégié la saison régulière qui est la plus importante en nombre de matchs, ce qui serait un avantage quand il nous faudra “entraîner” le modèle.

Concernant les matchs, de nombreuses données sont disponibles également mais seules certaines données sont nécessaires : les équipes jouant le match, la date et les points marqués par chaque équipe pour déterminer le gagnant.

Nous avons donc décidé de récupérer deux types de données : les statistiques concernant les équipes de NBA en saison régulière chaque mois entre octobre et avril (`webscraping-equipes.ipynb`) et les données de matchs de la saison régulière de NBA entre octobre et avril (`webscraping-match.ipynb`). Nous avons récupéré les données entre la saison 2005-2006 et la saison 2018-2019 pour avoir un set de données important. L’objectif de la récupération de ces données est de déterminer si l’on peut déterminer l’issue d’un match à partir des statistiques de l’équipe sur le mois précédent.

Dans le cas des données d’équipes nous avons utilisé le site officiel de la NBA qui est le seul à proposer des statistiques par mois ce qui nous permet de tenir compte de l’évolution d’une équipe au cours de la saison et de ne pas utiliser des données trop généralistes. Comme le site de la NBA est un site dynamique, nous ne pouvons pas utiliser la méthode classique ayant recours aux bibliothèques BeautifulSoup et urllib qui ne retournent pas le tableau au format HTML. La solution trouvée a été d’utiliser le module Selenium qui ouvre le navigateur internet comme le ferait un utilisateur lambda et récupère les données visibles à l’écran. On récupère donc les statistiques de chaque équipe pour les mois d’octobre à avril entre la saison 2005-2006 et la saison 2018-2019 et on utilise le module Pandas pour créer un Dataframe ayant pour chaque ligne le mois et l’année correspondante, pour chaque colonne une équipe et dans chaque cellule un dictionnaire contenant toutes les statistiques de l’équipe pour le mois donné.

Pour les données de matchs, nous avons décidé de “scraper” le site Basketball Reference (https://www.basketball-reference.com/leagues/NBA_2019_games.html) qui a l’avantage de stocker les données de matchs dans des tableaux séparés pour chaque mois. De plus, il est possible d’utiliser les outils BeautifulSoup et urllib.request ce qui simplifie grandement le “webscraping”. Nous récupérons donc toutes les données pour tous les mois qui nous intéressent et nous les plaçons dans un même Dataframe en y ajoutant une colonne contenant le mois et l’année du match pour pouvoir lier les données de matchs à celles des équipes plus facilement à l’avenir.


## Statistiques descriptives


## Modélisation
### Choix des modèles
Pour prédire les résultats de matchs, nous avons choisi d’utiliser la méthode “Random Forest” disponible dans la bibliothèque Scikit Learn. Cette méthode s’appuie sur les arbres de décisions (“Decision Tree”). A partir d’un échantillon de données d'entraînement, le modèle développe des branches dont les embranchements correspondent à des conditions sur les variables connues. Au bout de ces embranchements, l’arbre détermine la valeur devant être prédite, dans notre cas si l’équipe Visiteur gagne ou non (“Victory_V”). Après avoir entraîné le modèle, on utilise un échantillon de données “test” pour observer à quel point celui-ci est juste dans sa capacité de prédiction.  Le “Random Forest” est une généralisation du principe de l’arbre de décision qui effectue un apprentissage sur de multiples arbres de décision. 

Nous avons décidé de développer deux modèles différents. Dans le premier cas, nous utiliserons les données de match des saisons 2005-2006 à 2017-2018 comme entraînement et nous essaierons de prédire l’issue des matchs de la saison 2018-2019 (échantillon test). Le modèle n’aura que les statistiques des deux équipes pour déterminer l’issue du match. Nous avons décidé d’enlever la mention des équipes dans le Dataset car nous considérons que les performances d’une même équipe sont indépendantes entre deux saisons. L’objectif est ici de déterminer si l’on peut prédire les matchs en utilisant des données anciennes sans tenir compte spécifiquement des équipes mais seulement de leurs performances.

Dans le second modèle nous nous intéressons spécifiquement à la saison 2018-2019 en utilisant les données de matchs de octobre, novembre, décembre et janvier comme entraînement et les prédictions porteront sur les matchs de février, mars et avril. Dans ce modèle, en plus des statistiques des deux équipes participant au match, deux variables permettent d’identifier les équipes. L’objectif ici est d’observer si la prédiction s’améliore si les données sont plus récentes et si les équipes peuvent être identifiées par le modèle.

Avant de réaliser les modèles de machine-learning, on crée les dataframes utilisés pour le Random Forest avec le programme `pre_process.ipynb`.

### Résultats
Dans le cas du premier modèle, le Random Forest arrive à prédire justement 713 matchs sur 1120 matchs de la saison 2018-2019 en s’étant entraîné sur les matchs des saisons 2005-2006 à 2017-2018. Cela donne un “accuracy score” de 64% environ. Ce modèle semble donc assez efficace dans le sens où il donne de meilleurs résultats que si nous avions fait des prédictions purement aléatoires. Ce résultat nous montre donc que les résultats à un mois donné ne sont pas totalement indépendants des performances du mois précédent. Aussi le fait d’avoir entraîné le modèle avec les données issues de saisons précédentes remontant parfois à 15 ans a quand même permis au modèle de prédire correctement l’issue de nombreux matchs de la saison 2018-2019.

Concernant le deuxième modèle, le Random Forest prédit correctement 273 matchs sur 461 matchs durant la deuxième partie de la saison 2018-2019 soit un “accuracy score” de 59%. Ce modèle n’est donc pas particulièrement meilleur que le premier si l’on ajoute les équipes. Cela est sûrement dû au fait qu’ici, le dataset d'entraînement est bien moins important que dans le premier modèle (seulement les matchs de la moitié d’une saison). Cependant le modèle prédit toujours plus de 50% des matchs et est donc plus efficace que si la prédiction était faite de manière totalement aléatoire. 


## Conclusion


## Annexes