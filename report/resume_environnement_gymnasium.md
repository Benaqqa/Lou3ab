# Comprendre la partie environnement / Gymnasium

Ce document parle uniquement du **jeu et de sa mécanique** : comment il est construit comme un système, comment on le découpe en morceaux, et ce qu'apporte Gymnasium pour l'emballer proprement. Aucune notion de qui pilote le jeu ni de comment on décide quoi faire.

---

## 1. Un jeu vidéo est un système interactif

Un jeu n'est pas une image figée. C'est un **système** : on lui envoie quelque chose, il le transforme, il renvoie un résultat. On le décrit avec quelques éléments :

- **Environnement** : le monde simulé (la grille, les obstacles, la nourriture, le corps du serpent).
- **Actions** : ce qu'on peut faire (changer de direction).
- **Règles** : ce qui est autorisé ou interdit (collisions, manger, mourir de faim, gagner).
- **Feedback** : ce que le système renvoie (le nouvel état, le score).

Pour Snake : l'**état** = position du serpent + obstacles + nourriture ; l'**action** = changer de direction ; une **règle** = "si on touche un mur, c'est fini" ; l'**objectif** = manger pour augmenter le score.

---

## 2. L'architecture Entrées → Traitement → Sorties

Tout moteur de jeu suit ce schéma :

- **Entrées** : ce qui arrive dans le système (ici, une action pour ce tour).
- **Traitement** : le moteur applique les règles et fait avancer le monde (déplacer le serpent, tester les collisions, gérer la nourriture et la faim).
- **Sorties** : ce que le système produit (le nouvel état, le score, l'affichage).

Dans le projet, ça correspond directement à des morceaux de code :

| Composant moteur | Rôle | Dans le code |
|---|---|---|
| Gestion du monde | Créer la carte, les entités | `reset` (place serpent, obstacles, nourriture) |
| Physique / mouvement | Faire avancer | `step` (ajoute une tête, retire la queue) |
| Collisions | Détecter les contacts | test de collision (mur, obstacle, corps) |
| Règles / scoring | Manger, faim, victoire | logique dans `step` |
| Rendu | Afficher | `render` (texte) ou version graphique (Pygame) |

**Point important : le rendu n'est PAS la simulation.** Le jeu peut tourner entièrement sans rien afficher, et on peut rejouer une partie à l'identique en l'affichant ensuite. L'affichage ne fait que lire l'état du jeu ; il ne le calcule pas.

---

## 3. La boucle de jeu (game loop)

Un jeu **tourne en boucle** : il répète le même cycle jusqu'à la fin de la partie.

```
réinitialiser le jeu          # état de départ
tant que la partie n'est pas finie :
    recevoir une action
    faire un pas de jeu        # appliquer les règles, avancer
    vérifier si c'est fini
```

Chaque tour de boucle = un **pas de temps**. Le temps est donc découpé en petits pas successifs (t0, t1, t2…).

Dans Snake, le jeu est **au tour par tour** : pas de notion de "temps réel" ni de vitesse (pas de FPS, pas de Δt physique). Chaque pas de jeu correspond simplement à : "le serpent avance d'une case". C'est plus simple qu'un jeu d'action en temps réel.

---

## 4. État vs observation

Deux notions à ne pas confondre :

- **État** : toute l'information qui décrit la situation complète du jeu à un instant. Ici : position de tout le corps du serpent, sa direction, tous les obstacles, la nourriture, le compteur de faim, le score.
- **Observation** : une **version réduite et filtrée** de l'état, celle que le jeu expose à l'extérieur à chaque pas.

Dans le projet, l'observation est un petit vecteur de **11 nombres** :

| Position | Signification |
|---|---|
| 1 à 3 | Danger immédiat (tout droit / à droite / à gauche) : y a-t-il un mur, un obstacle ou le corps juste devant ? |
| 4 à 7 | Direction actuelle (haut / droite / bas / gauche) |
| 8 à 11 | Où est la nourriture par rapport au serpent (à gauche / à droite / en haut / en bas) |

Pourquoi exposer seulement ça et pas toute la grille ?
- **Plus simple** : moins d'informations.
- **Plus réaliste** : on ne voit que le voisinage immédiat, pas toute la carte d'un coup.
- **Plus général** : pas de carte précise mémorisée, juste une description qui marche partout.

Le revers : avec une vue aussi locale, on ne peut pas anticiper un piège à plusieurs cases de distance.

---

## 5. L'espace d'actions

L'**espace d'actions** = la liste de tout ce qui est possible. Ici, **3 actions** seulement, et elles sont **relatives au serpent** :

- aller tout droit,
- tourner à droite,
- tourner à gauche.

Pourquoi 3 actions relatives plutôt que 4 directions absolues (haut/bas/gauche/droite) ?
- Ça **empêche le demi-tour suicidaire** : avec des directions absolues, on pourrait revenir sur soi-même et mourir bêtement. Avec "gauche/droite/tout droit", ce demi-tour est juste impossible à exprimer.
- Ça **réduit le nombre de choix possibles**, donc le jeu est plus simple à manipuler.

---

## 6. Les règles et la transition

Les **règles** définissent ce qui se passe à chaque pas. Concrètement, à chaque tour le moteur :
1. tourne la direction selon l'action reçue,
2. calcule la nouvelle position de la tête,
3. teste s'il y a collision (mur, obstacle, corps),
4. gère la nourriture (si mangée : le serpent grandit + nouvelle nourriture) ou l'avance simple (retire la queue),
5. vérifie la faim et la victoire.

La **transition**, c'est cette fonction qui prend l'état actuel + une action et produit l'état suivant : `état suivant = f(état actuel, action)`.

**Déterminisme vs hasard.** À l'intérieur d'une partie, tout est **déterministe** : même situation + même action = même résultat, toujours. Le hasard n'intervient qu'au **démarrage** d'une partie (placement des obstacles, de la nourriture, nombre d'obstacles tiré au hasard).

Avantage : avec une même **graine** (seed) de hasard, deux parties sont strictement identiques. C'est ce qui rend les tests **reproductibles**.

---

## 7. Le score renvoyé à chaque pas

À chaque pas, en plus du nouvel état, le jeu renvoie un **nombre** qui résume ce qui vient d'arriver :

- **+1** nourriture mangée,
- **−1** collision (mort),
- **−0,5** mort de faim,
- **+0,1** rapprochement de la nourriture,
- **−0,15** éloignement,
- **−0,01** par pas (petit coût de temps),
- **+5** victoire (grille entièrement remplie).

C'est une simple **sortie chiffrée** du système à chaque tour. Le jeu se contente de la calculer et de la renvoyer ; ce qu'on en fait ensuite ne regarde pas l'environnement.

---

## 8. Conditions de fin d'une partie

Une partie se termine de deux façons différentes :

- **terminated** (fin "interne" du jeu) : collision, mort de faim, ou victoire.
- **truncated** (coupure externe) : on a atteint le nombre maximal de pas autorisé (une limite de sécurité, ici 400 pas), même si le serpent est encore vivant.

La distinction compte : "terminated" = le jeu lui-même dit stop selon ses règles ; "truncated" = c'est une limite imposée de l'extérieur, sans rapport avec les règles du jeu.

---

## 9. Qu'est-ce que Gymnasium ?

**Gymnasium** est une bibliothèque Python qui définit une **interface standard** pour les environnements de jeu/simulation. C'est un "contrat" : si le jeu respecte ce contrat, n'importe quel outil compatible peut s'y brancher sans rien changer.

Le contrat impose principalement 4 choses :

- **`reset()`** : remet le jeu à zéro et renvoie l'observation de départ (= début d'une partie).
- **`step(action)`** : reçoit une action, fait avancer le jeu d'un pas, et renvoie 5 choses :
  - l'observation (nouvel état exposé),
  - le score du pas (le nombre décrit en section 7),
  - **terminated** (fini selon les règles ?),
  - **truncated** (coupé à cause de la limite de pas ?),
  - **info** (infos bonus pour analyse : score total, cause de fin… sans rôle dans la mécanique).
- **`observation_space`** : décrit la forme des observations. Ici un `Box` de 11 nombres.
- **`action_space`** : décrit les actions possibles. Ici un `Discrete(3)` (3 actions).

Dans le projet, le jeu Snake est **fait maison** mais emballé dans ce format. Avantage : l'environnement devient **interchangeable** et **réutilisable** avec tout l'écosystème standard.

---

## 10. La grille "dynamique"

C'est la particularité du sujet. Les conditions **changent d'une partie à l'autre** (pas pendant une partie). À chaque `reset`, le jeu tire au hasard :

- le **nombre d'obstacles**, entre 3 et 10,
- le **budget de faim** (nombre de pas autorisés sans manger), qui varie un peu autour de sa valeur de base.

Pourquoi ? Pour empêcher de s'appuyer sur une carte fixe et forcer le jeu à proposer **n'importe quelle configuration**. C'est cette variabilité qui distingue un vrai jeu (avec de l'incertitude) d'un simple programme toujours identique.

---

## Résumé en une phrase

L'environnement, c'est **le jeu lui-même** : un système qui reçoit une action (`step`), applique ses règles pour avancer d'un pas, et renvoie un nouvel état + un score + un signal de fin — le tout emballé dans le format standard **Gymnasium** (`reset`, `step`, `observation_space`, `action_space`) pour être propre et réutilisable.
