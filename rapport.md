# Rapport projet Python

## Introduction
Pour choisir le sujet nous avions dès le début une volonté de travailler sur un sport et sur la prédiction de résultats. Notre choix s’est porté sur le basketball et plus particulièrement la NBA (le championnat américain de basketball) pour plusieurs raisons. Premièrement, le basketball est un des sports les plus “prévisible” comme le décrit [cet article](https://www.actionnetwork.com/general/futures-betting-odds-sports-parity-nba-nfl-mlb-nhl-golf-tennis). Cela est certainement dû au fait que le nombre de points dans un match est très élevé par rapport à d’autres sports comme le football. Ensuite la NBA, comme tous les sports américains, possède un répertoire de statistiques assez impressionnant et disponible en accès libre sur internet ce qui était parfait dans le cadre d’un projet de Data Science.

Nous nous sommes donc demandé : Peut-on prévoir les résultats d’un match de sport à partir des performances précédentes de chaque équipe ? Aussi, est-ce que les résultats d’un match sont principalement le fruit du hasard ou est-ce que l’issue d’un match est déterminée par des caractéristiques d’une équipe tangibles et statistiques ?

Pour cela nous avons d’abord extrait les données et les avons nettoyés (I.) , puis avons cherché à les analyser à l'aide d’outils statistiques (II.) afin de pouvoir tester un modèle de machine-learning et d’observer si celui-ci est en capacité de prévoir l’issue des matchs puis de vérifier l’acuité du modèle. (III.)


## Récupération de données
Les données portant sur la NBA étaient nombreuses et étaient disponibles sur différents sites internet. Sur [le site officiel](https://www.nba.com/stats/) de la NBA par exemple, de très nombreuses données sont disponibles sur les joueurs, les équipes ainsi que les matchs. Ces données peuvent être filtré selon différents critères :
- par saison, une saison commençant souvent la plupart du temps en octobre et se termine en avril-mai. Ces données sont aussi disponibles mois par mois.
- par type de saison, une saison de NBA est divisée en plusieurs périodes, notamment une pré-saison, une saison régulière et les “playoffs”.
- par “mode” : les statistiques sont mesurées par match, pour 48 minutes, par minute, etc…

Dans le cadre de notre projet nous avons privilégié les statistiques globales d’équipe à celles des joueurs. Il y a différentes problématiques avec ces dernières : les joueurs n’ont pas les mêmes postes au sein d’une équipe ce qui influence fortement les statistiques de chacun, aussi tous les joueurs ne jouent pas à tous les matchs ce qui créerait un décalage avec la réalité si nous décidions par exemple de nous intéresser à la moyenne de tous les joueurs d’une équipe. Les statistiques par équipe nous semblent plus intéressantes car elles représentent réellement les performances d’une équipe à un temps t, d’autant plus que ces statistiques sont disponibles pour chaque mois de la saison.

Concernant le type de saison, il nous a semblé plus intéressant de ne choisir qu’un seul type de saison: en effet les modalités de chaque type de saison comme les enjeux sont différents pour chaque type de saison, ce qui pourrait modifier les statistiques des équipes. La prédiction de matchs de “playoffs” à partir de matchs de saison régulière par exemple pourrait ainsi être faussée par les différences entre ces deux types de saison. Dans le cadre de notre projet, nous avons donc privilégié la saison régulière qui est la plus importante en nombre de matchs, ce qui serait un avantage quand il nous faudra “entraîner” le modèle.

![Données d'équipes](/images/tableau-equipes.png "Statistiques d'équipes")

Concernant les matchs, de nombreuses données sont disponibles également mais seules certaines données sont nécessaires : les équipes jouant le match, la date et les points marqués par chaque équipe pour déterminer le gagnant.

Nous avons donc décidé de récupérer deux types de données : les statistiques concernant les équipes de NBA en saison régulière chaque mois entre octobre et avril (`webscraping-equipes.ipynb`) et les données de matchs de la saison régulière de NBA entre octobre et avril (`webscraping-match.ipynb`). Nous avons récupéré les données entre la saison 2005-2006 et la saison 2018-2019 pour avoir un set de données important. L’objectif de la récupération de ces données est de déterminer si l’on peut déterminer l’issue d’un match à partir des statistiques de l’équipe sur le mois précédent.

Dans le cas des données d’équipes nous avons utilisé le site officiel de la NBA qui est le seul à proposer des statistiques par mois ce qui nous permet de tenir compte de l’évolution d’une équipe au cours de la saison et de ne pas utiliser des données trop généralistes. Comme le site de la NBA est un site dynamique, nous ne pouvons pas utiliser la méthode classique ayant recours aux bibliothèques BeautifulSoup et urllib qui ne retournent pas le tableau au format HTML. La solution trouvée a été d’utiliser le module Selenium qui ouvre le navigateur internet comme le ferait un utilisateur lambda et récupère les données visibles à l’écran. On récupère donc les statistiques de chaque équipe pour les mois d’octobre à avril entre la saison 2005-2006 et la saison 2018-2019 et on utilise le module Pandas pour créer un Dataframe ayant pour chaque ligne le mois et l’année correspondante, pour chaque colonne une équipe et dans chaque cellule un dictionnaire contenant toutes les statistiques de l’équipe pour le mois donné.

Pour les données de matchs, nous avons décidé de “scraper” le site Basketball Reference (https://www.basketball-reference.com/leagues/NBA_2019_games.html) qui a l’avantage de stocker les données de matchs dans des tableaux séparés pour chaque mois. De plus, il est possible d’utiliser les outils BeautifulSoup et urllib.request ce qui simplifie grandement le “webscraping”. Nous récupérons donc toutes les données pour tous les mois qui nous intéressent et nous les plaçons dans un même Dataframe en y ajoutant une colonne contenant le mois et l’année du match pour pouvoir lier les données de matchs à celles des équipes plus facilement à l’avenir.


## Statistiques descriptives
Pour traiter les données que nous avons récoltées nous avons choisi de chercher à mettre en évidence les variables les plus déterminantes dans la victoire d’une équipe. Pour cela nous avons à chaque fois essayer d’isoler le premier de la saison régulière (et donc celui qui a le plus de victoires) pour ensuite comparer ses statistiques à celles du reste des équipes. Tout le code utilisé pour faire ce qui est décrit dans cette partie est disponible dans le fichier “statistiques descriptives”, j’en ai extrait certaines sorties (principalement des tableaux).

Avant de commencer à détailler ce que nous avons fait, plusieurs précisions importantes : 
La saison 2011 n’est pas prise en compte dans la plupart des statistiques calculées car tronquée par une grève générale des joueurs. 
Dans chaque statistique trouvée il y a les résultats de la conférence est et ouest séparés car elles le sont en saison régulière. Cela nous a aussi permis de comparer certains résultats entre les deux conférences.

Pour commencer, nous avons travaillé sur le dataframe “stats_matchs” qui regroupe les résultats des matchs de NBA de 2005 à aujourd’hui. Il nous a permis de déterminer le nombre de points moyens marqués à l'extérieur et à domicile chaque année. On a ensuite pu calculer le nombre de points moyens en faisant la moyenne des deux (car il y a autant de matchs à l’extérieur qu’à domicile).

![Moyennes PTS](/images/moyennePTS.jpg "Moyennes PTS")

Comme nous le voyons, le nombre de points marqués augmente significativement au cours du temps, ce qui est à prendre en compte dans nos futures analyses. Ensuite, comme prévu dans notre démarche, on calcule la moyenne de points marqués chaque année par l’équipe première de la saison régulière (ici pour la conférence Est)

![Moyennes PTS 1er](/images/moyennePTS1ER.jpg "Moyennes PTS 1er")

Assez logiquement, on observe que les moyennes des équipes premières sont à chaque fois plutôt supérieures aux moyennes globales. 
 
Nous avons ensuite travaillé sur l’autre dataframe (stats_équipe_par_mois) qui est composé de différentes variables des équipes de NBA chaque mois (elles sont détaillées plus haut et dans le lexique). Toujours dans la même optique, nous avons cherché à calculer la moyenne des variables pour l’équipe première du championnat afin de pouvoir la comparer, nous avons commencé avec la variable FGM (voir lexique) en calculant donc la moyenne de la première équipe chaque année :

![Moyenne FGM](/images/moyenneFGM.jpg "Moyenne FGM")

Puis en calculant la moyenne des autres équipes, avant d’automatiser le processus pour l’avoir pour toutes les variables et stocker ces moyennes dans une liste (pour le premier et pour toutes les autres équipes). Pour cela, voir le fichier statistiques descriptives.
Ainsi nous pouvions comparer ces valeurs afin de déterminer les variables déterminantes dans les victoires. Pour cela nous avons d’abord essayé de faire la somme des différences entre les moyennes de chaque variable pour le premier et pour toutes les autres équipes. Mais certaines variables étant en pourcentage d’autres non, le résultat est dur à lire : 

![Ecart](/images/ecart.jpg "Ecart")

Nous avons donc décidé, pour chaque variable, de faire la moyenne des rapports entre la moyenne du 1er et la moyenne globale (incluant le premier) afin d’obtenir une valeur autour de 1. Si la valeur dépasse 1, plus elle le dépasse plus la variable est significative dans la victoire, si elle est très proche de 1 ou en dessous elle est peu significative. 

![Moyenne pondérée](/images/moyenneponderee.jpg "Moyenne pondérée")

Plusieurs choses ; tout d’abord on remarque certaines choses assez logiques : le pourcentage de victoire, le nombre de victoires sont très positivement significatifs là ou à l’inverse le pourcentage de défaite et le nombre de défaite sont très négativement significatifs. Ensuite les valeurs absurdes de la variable ‘+/-’ (la différence de points) sont dus à la structure de la variable. En effet, la moyenne de la différence de points sur toutes les équipes est obligatoirement très proche de 0 (étant donné qu’il y a autant de point encaissés que marqués) la différence avec 0 est due aux différents arrondis. Or la différence de points du 1er de chaque année est en général comprise entre 5 et 20 (étant donné qu’il gagne la majorité de ses matchs) ce qui donne la division d’un réel entre 5 et 20 par une quantité très proche de 0 et donc cela explique ces valeurs absurdes (et les valeurs du tableaux d’avant par la même occasion).

Cependant l’impact de cette variable n’est pas à négliger car la différence de points est nécessairement une valeur élevée pour le premier du championnat. Cependant nous ne pouvons mesurer cette différence avec les outils que nous avons utilisés précédemment de par la particularité structurelle de cette variable (le fait qu’elle soit proche de 0 quand on prend une moyenne globale ou sur toutes les équipes sauf une). Pour contourner cette difficulté on utilise la moyenne des points marqués par saison à laquelle on ajoute la différence du premier avant de rediviser le tout par la moyenne de points marqués dans la saison. On obtient 1,074.


## Modélisation
### Choix des modèles
Pour prédire les résultats de matchs, nous avons choisi d’utiliser la méthode “Random Forest” disponible dans la bibliothèque Scikit Learn. Cette méthode s’appuie sur les arbres de décisions (“Decision Tree”). A partir d’un échantillon de données d'entraînement, le modèle développe des branches dont les embranchements correspondent à des conditions sur les variables connues. Au bout de ces embranchements, l’arbre détermine la valeur devant être prédite, dans notre cas si l’équipe Visiteur gagne ou non (“Victory_V”). Après avoir entraîné le modèle, on utilise un échantillon de données “test” pour observer à quel point celui-ci est juste dans sa capacité de prédiction.  Le “Random Forest” est une généralisation du principe de l’arbre de décision qui effectue un apprentissage sur de multiples arbres de décision. 

Nous avons décidé de développer deux modèles différents. Dans le premier cas, nous utiliserons les données de match des saisons 2005-2006 à 2017-2018 comme entraînement et nous essaierons de prédire l’issue des matchs de la saison 2018-2019 (échantillon test). Le modèle n’aura que les statistiques des deux équipes pour déterminer l’issue du match. Nous avons décidé d’enlever la mention des équipes dans le Dataset car nous considérons que les performances d’une même équipe sont indépendantes entre deux saisons. L’objectif est ici de déterminer si l’on peut prédire les matchs en utilisant des données anciennes sans tenir compte spécifiquement des équipes mais seulement de leurs performances.

Dans le second modèle nous nous intéressons spécifiquement à la saison 2018-2019 en utilisant les données de matchs de octobre, novembre, décembre et janvier comme entraînement et les prédictions porteront sur les matchs de février, mars et avril. Dans ce modèle, en plus des statistiques des deux équipes participant au match, deux variables permettent d’identifier les équipes. L’objectif ici est d’observer si la prédiction s’améliore si les données sont plus récentes et si les équipes peuvent être identifiées par le modèle.

Avant de réaliser les modèles de machine-learning, on crée les dataframes utilisés pour le Random Forest avec le programme `pre_process.ipynb`.

### Résultats
Dans le cas du premier modèle, le Random Forest arrive à prédire justement 713 matchs sur 1120 matchs de la saison 2018-2019 en s’étant entraîné sur les matchs des saisons 2005-2006 à 2017-2018. Cela donne un “accuracy score” de 64% environ. Ce modèle semble donc assez efficace dans le sens où il donne de meilleurs résultats que si nous avions fait des prédictions purement aléatoires. Ce résultat nous montre donc que les résultats à un mois donné ne sont pas totalement indépendants des performances du mois précédent. Aussi le fait d’avoir entraîné le modèle avec les données issues de saisons précédentes remontant parfois à 15 ans a quand même permis au modèle de prédire correctement l’issue de nombreux matchs de la saison 2018-2019.

Concernant le deuxième modèle, le Random Forest prédit correctement 273 matchs sur 461 matchs durant la deuxième partie de la saison 2018-2019 soit un “accuracy score” de 59%. Ce modèle n’est donc pas particulièrement meilleur que le premier si l’on ajoute les équipes. Cela est sûrement dû au fait qu’ici, le dataset d'entraînement est bien moins important que dans le premier modèle (seulement les matchs de la moitié d’une saison). Cependant le modèle prédit toujours plus de 50% des matchs et est donc plus efficace que si la prédiction était faite de manière totalement aléatoire. 

![Importance des variables](/images/importances.png)

Concernant l’importance relative de chaque variable dans les deux modèles, on observe dans les deux cas que les mêmes variables qu’elles concernent l’équipe 1 ou l’équipe 2 ont à peu près la même importance. Cela montre que l’équipe Visiteur ou l’équipe à domicile ont à peu près les mêmes chances de gagner à statistique égale. On observe quelques exceptions notamment dans le deuxième modèle dans le cas des statistiques +/- et WIN%.
Globalement, les variables qui semblent avoir le plus d’importance dans les deux modèles sont +/-  (différence entre le nombre de points marqués et le nombre de points pris), WIN% .(pourcentage de victoires). Cela semble cohérent avec les résultats des statistiques descriptives notamment pour la variable WIN% ou nous avions remarqué que l’équipe ayant la plus haute valeur de WIN% avait en moyenne une valeur de WIN% plus élevé de 47% que la moyenne générale des équipes. Pour les autres variables, l’écart à la moyenne générale de la meilleure équipe était d’environ + ou - 10%. Les meilleures équipes semblent donc se distinguer particulièrement avec la variable WIN% et le pourcentage de victoires durant les matchs récents est clairement un indice de la réussite des matchs futurs. 

De manière générale, on observe que les variables semblent toutes avoir une certaine importance compris entre 1% et 4% y compris les variables indiquant l’équipe dans le modèle même si celles-ci ne sont pas déterminantes pour déterminer l’issue des matchs.

## Conclusion
Pour conclure, il semble bien d’après nos modèles que l’on puisse prévoir les résultats de matchs à partir des performances passées des équipes. Toutefois la justesse des modèles reste assez faible et ils nécessiteraient certainement plus de statistiques concernant les équipes ou peut-être même les joueurs pris individuellement ou de ne pas se limiter aux statistiques du mois précédent pour prédire les résultats. Au-delà du choix des données, le choix de la technique pourrait aussi être interrogé. En modifiant les paramètres du Random Forest ou en choisissant une toute autre technique de machine-learning, nous obtiendrions peut-être une meilleure justesse dans les prédictions. Enfin, il serait intéressant d’observer si les modèles développés fonctionneraient de la même manière pour la période des “playoffs” dont le fonctionnement est un peu différent de la saison régulière ou bien même encore pour un tout autre sport.

## Annexes
### Glossaire des statistiques
| Statistique | Définition |
|---|---|
| GP | Games Played |
| W | Wins |
| L | Losses |
| MIN | Minutes Played |
| FGM | Field Goals Made |
| FGA | Field Goals Attempted |
| FG% | Field Goal Percentage |
| 3PM | 3 Point Field Goals Made |
| 3PA | 3 Point Field Goals Attempted |
| 3P% | 3 Point Field Goals Percentage |
| FTM | Free Throws Made |
| FTA | Free Throws Attempted |
| FT% | Free Throw Percentage |
| OREB | Offensive Rebounds |
| DREB | Defensive Rebounds |
| REB | Rebounds |
| AST | Assists |
| TOV | Turnovers |
| STL | Steals |
| BLK | Blocks |
| PF | Personal Fouls |
| FP | Fantasy Points |
| DD2 | Double doubles |
| TD3 | Triple doubles |
| PTS | Points |
| +/- | Différence entre le nombre de points marqués et le nombre de points pris |
| FP | Fantasy Points |
