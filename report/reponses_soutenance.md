# Réponses de soutenance — Snake évolutif sur grille dynamique

## Bloc 1 — Environnement / Gymnasium / game dev

### Pourquoi ton propre `SnakeEnv` + apport de l'API Gymnasium
- Le sujet impose un Snake **dynamique** (obstacles + budget variables par épisode) : aucun env standard ne le fait.
- Conformité Gymnasium = contrat `reset` / `step` / `observation_space` / `action_space`.
- Apport concret : env **interchangeable** avec tout l'écosystème (on pourrait y brancher un DQN/PPO sans toucher au jeu), évaluation standardisée, code lisible par un jury qui connaît l'API.

### État global `s` vs observation `o = φ(s)`
- `s` = toute l'info de simulation : corps complet, direction, tous les obstacles, nourriture, compteur de faim, score.
- `o = φ(s)` = ce que l'agent **perçoit** : vecteur compact de 11 valeurs.
- φ est une **compression** : égocentrique et locale. Important car c'est ce qui rend le problème **partiellement observable** et force la généralisation.

### Pourquoi 11 dims locales et pas la grille entière
- Grille entière (144 entrées) → réseau bien plus gros, beaucoup plus de poids à évoluer → recherche évolutive plus lente.
- Risque de **mémoriser des cartes** au lieu d'apprendre une politique générale.
- Compromis assumé : **généralise mieux + politique légère**, mais perd l'**anticipation** (ne voit pas les impasses à plusieurs cases). C'est exactement la limite qui ressort en robustesse.

### Actions relatives (3) vs absolues (4)
- Relatives : tout droit / tourner droite / tourner gauche, dans le repère du serpent.
- Le **demi-tour est inexprimable** : il n'existe aucune action « aller à l'opposé », donc le suicide par demi-tour est éliminé **mécaniquement** (pas par une règle ajoutée).
- Bonus : espace de décision plus petit (3 au lieu de 4) → apprentissage plus simple.

### `terminated` vs `truncated`
- `terminated` = l'épisode finit **par les règles du jeu** : collision, famine, ou victoire (grille pleine).
- `truncated` = arrêt **externe** : limite de pas atteinte (`max_steps = 400`), alors que la partie aurait pu continuer.
- Distinction standard Gymnasium : un truncated ne doit pas être traité comme un « échec » du même type qu'un terminated.

### Justification des 6 termes de reward
- `+1` manger → objectif principal.
- `−1` collision → punir la mort évitable.
- `−0.5` famine → mort moins « grave » qu'une collision bête, mais à éviter.
- `+0.1` se rapprocher / `−0.15` s'éloigner → **signal dense** (shaping de distance Manhattan) qui guide quand la nourriture est loin ; l'asymétrie (−0.15 > +0.1) décourage le va-et-vient.
- `−0.01` coût de temps → casse les **boucles stériles**.
- `+5` victoire → gros bonus rare.
- Si on enlève le `−0.01` : l'agent peut tourner en rond indéfiniment sans pénalité, surtout si survivre rapporte.

### Reward hacking (récompense « trichable »)
- Principe : **récompenser ce qu'on veut vraiment**, pas un proxy.
- Deux dérives évitées :
  - **Camping** (survivre sans manger) → mitigé par coût de temps + score qui domine la fitness (×100).
  - **Tourner près de la nourriture** sans la manger → mitigé par l'asymétrie du shaping + coût de temps.

### En quoi c'est « dynamique » + nature de la transition
- Dynamique = **entre épisodes** : nombre d'obstacles [3,10] et budget de faim (±20%) tirés au hasard à chaque `reset`.
- La transition **dans** un épisode est **déterministe** : `P(s'|s,a) ∈ {0,1}`, même état + même action → même résultat.
- L'aléa est injecté **uniquement au `reset`**, jamais dans `step`.

### Pourquoi aléa au `reset` seulement
- `step` déterministe = deux épisodes de même graine sont **identiques au pixel près**.
- Ça permet d'évaluer tous les agents (AG, aléatoire, heuristique) sur **exactement les mêmes parties** → comparaison équitable et **rejouable**.

### Rendu découplé de la simulation
- Le renderer (Pygame/ANSI) ne fait que **lire l'état public** ; il ne modifie rien.
- Conséquence : on entraîne **sans affichage** (mode rapide), puis on rejoue une partie graphiquement à l'identique.
- Bon design : « le rendu n'est pas la simulation, c'est sa représentation. »

### Boucle de jeu (PDA) + discrétisation du temps
- Cycle : **Perception** (`obs`) → **Décision** (`agent.act`) → **Action** (`env.step` met à jour l'état et renvoie le feedback).
- Jeu au **tour par tour** : pas de Δt physique, chaque `step` = exactement une transition. Le temps est échantillonné en pas discrets `t0, t1, …`.

### Pourquoi `info` ne sert pas à la décision
- `info` expose des variables d'analyse (score, cause de fin, nb obstacles).
- Si l'agent y avait accès, il pourrait exploiter des infos non perçues normalement → **fuite d'information** qui casse le réalisme « champ de perception limité » et fausse la généralisation.

---

## Bloc 2 — Agent / IA / neuroévolution

### Neuroévolution vs Deep RL
- Pas seulement « pas de gradient » :
  - Robuste aux récompenses **éparses / bruitées / non différentiables**.
  - Très simple à implémenter (NumPy pur, pas de framework).
  - **Parallélisable** trivialement (chaque génome s'évalue indépendamment).
- Adapté ici car le réseau est petit (243 poids) ; sur un gros réseau, le gradient (DQN/PPO) redevient plus efficace.

### D'où vient 243
- `W1` = 11×16 = 176, `b1` = 16 → 192.
- `W2` = 16×3 = 48, `b2` = 3 → 51.
- Total = **243**. Le génome est ce vecteur aplati.

### `tanh` caché + `argmax` sortie
- `tanh` : non-linéarité bornée [−1,1], stable, classique pour petit MLP.
- `argmax` : prend l'action au score max → **politique déterministe**.
- Conséquence : même obs → même action, donc évaluation **reproductible** (clé pour comparer les agents sur les mêmes graines).

### Fitness `100·s̄ + 0.5·t̄ + R̄` et le ×100
- `s̄` score moyen (priorité), `t̄` survie (bonus), `R̄` récompense cumulée.
- ×100 sur le score = il **domine** : la survie ne peut jamais compenser le fait de ne pas manger.
- Si ×1 : un serpent qui survit longtemps sans manger pourrait avoir une fitness comparable à un bon mangeur → on optimiserait le camping, pas l'objectif.

### Pourquoi 6 épisodes et pas 1
- 1 épisode = fitness **très bruitée** (la difficulté de carte varie : 3 à 10 obstacles).
- 6 épisodes → moyenne plus stable, mesure la **généralisation** plutôt qu'un coup de chance.
- Lien avec la limite : reste bruité, donc un bon individu peut être sous-évalué par malchance → convergence ralentie. Plus d'épisodes = moins de bruit mais plus cher.

### Graines renouvelées chaque génération (`10000 + gen·100 + i`)
- Chaque génération affronte des **cartes différentes**.
- Empêche le **surapprentissage** à un jeu fixe de cartes.
- Prix payé : courbes **bruitées** (la baisse ponctuelle = cartes plus dures, pas une régression réelle).

### Tournoi taille 5 (et taille 2 / 50)
- On tire 5 individus, le meilleur se reproduit → **pression de sélection modérée**.
- Taille 2 : pression faible → exploration forte, convergence lente, diversité préservée.
- Taille 50 : pression très forte → les meilleurs écrasent tout → **convergence prématurée**, perte de diversité.

### Croisement uniforme 0.7
- 0.7 = probabilité **d'appliquer** le croisement ; sinon clonage d'un parent.
- Uniforme (gène par gène, masque booléen) : chaque poids vient au hasard d'un parent.
- Préféré au point unique car les 243 poids n'ont pas d'ordre « logique » à préserver → le mélange fin explore mieux.

### Mutation σ décroissant 0.25 → 0.10
- Au début : grosses perturbations = **exploration** large de l'espace.
- En fin : petites perturbations = **affinage** (exploitation) autour des bonnes solutions.
- Analogue : **recuit simulé** / décroissance de pas en optimisation (compromis exploration/exploitation).

### Élitisme 12 %
- Les 12 % meilleurs passent **intacts** → on ne perd jamais la meilleure solution trouvée.
- Sans élitisme : croisement/mutation peuvent **détruire** le meilleur individu d'une génération à l'autre → progression instable.
- Avec 90 % d'élite : quasi pas de renouvellement → **stagnation**, diversité morte.

### Génotype vs phénotype
- Génotype = les **243 réels**. Phénotype = le **comportement de jeu** produit.
- La relation passe par l'argmax : un petit changement de poids peut **basculer** l'action choisie à un moment critique → effet domino sur toute la partie → grand saut de comportement (paysage de fitness non lisse).

### Pourquoi l'observation locale plombe la gestion d'obstacles
- L'agent ne voit que le **danger immédiat (1 case)**.
- Il ne détecte pas une **impasse / cul-de-sac** plusieurs cases à l'avance : il y entre, puis n'a plus d'action sûre → collision.
- D'où : excellent sans obstacle (18.9 ≈ heuristique 19.9), mais s'effondre quand la densité monte (2.5 à 12 obstacles).

### AG survit plus longtemps mais mange moins
- Interprétation : il a appris une **gestion prudente de l'espace** (évite la mort) mais une recherche de nourriture **moins optimale** sous obstacles que l'heuristique.
- Pas forcément une fitness mal calibrée : la survie n'est qu'un bonus (×0.5), le score domine. C'est plutôt la **limite de perception** qui l'empêche de convertir cette survie en score.

### NEAT vs CMA-ES vs observation enrichie
- Vraie limite = **perception locale**, pas l'optimiseur.
- Donc **observation enrichie** (vision en rayons, distance aux dangers, longueur du serpent) attaque la racine du problème → permet l'anticipation des impasses.
- CMA-ES / NEAT amélioreraient la **recherche** (covariance, topologie évolutive) mais ne donnent pas de nouvelle info à l'agent → gain plus faible sur ta limite réelle.

### « C'est du RL ? » — réponse propre
- On résout un **MDP** (états, actions, récompense, transitions) : cadre commun au RL.
- Mais on **n'utilise pas** la structure temporelle de la récompense (pas de Bellman, pas de gradient, pas de bootstrapping).
- C'est de l'**optimisation boîte noire** : on traite la partie entière comme une fonction `génome → fitness` à maximiser. Frontière : neuroévolution = optimisation directe de politique, sans crédit-assignment temporel.
