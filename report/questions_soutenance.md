# Questions de soutenance — Snake évolutif sur grille dynamique

## Bloc 1 — Environnement / Gymnasium / game dev

- Pourquoi avoir construit ton propre `SnakeEnv` au lieu de réutiliser un Snake existant ? Qu'est-ce que la conformité à l'API Gymnasium t'apporte concrètement ?
- Explique la différence entre **état global `s`** et **observation `o = φ(s)`**. Pourquoi cette distinction compte ici ?
- Ton observation fait 11 dims et est purement locale. Pourquoi pas la grille entière (144 cases) en entrée ? Quel est le compromis (généralisation vs anticipation) ?
- Pourquoi des **actions relatives (3)** plutôt qu'absolues (4) ? En quoi ça « élimine le demi-tour suicidaire » mécaniquement ?
- Quelle est exactement la différence entre `terminated` et `truncated` ? Donne un cas de chaque dans ton jeu.
- Ton reward shaping a 6 composantes. Justifie chaque terme. Pourquoi `−0.01` par pas, et le risque si tu l'enlèves ?
- C'est quoi le piège de la **récompense « trichable » (reward hacking)** ? Quels deux comportements dégénérés ta conception évite-t-elle, et comment ?
- En quoi ton jeu est « dynamique » ? La transition est-elle stochastique ou déterministe ? Où injectes-tu exactement l'aléa, et pourquoi pas dans le `step` ?
- Pourquoi placer l'aléa uniquement au `reset` est important pour la **reproductibilité** et la comparaison équitable des agents ?
- Le rendu (Pygame/ANSI) est « découplé » de la simulation. Qu'est-ce que ça veut dire et pourquoi c'est un bon choix d'ingénierie ?
- Décris ta boucle de jeu (cycle Perception → Décision → Action). Où le temps est-il discrétisé ?
- Le `info` dict ne sert pas à la décision. Pourquoi cette règle, et que se passerait-il si l'agent y avait accès ?

## Bloc 2 — Agent / IA / neuroévolution

- Pourquoi la **neuroévolution** et pas du Deep RL (DQN/PPO) ? Avantages réels vs simple absence de gradient ?
- D'où vient le chiffre **243** ? Recalcule la dimension du génome couche par couche.
- Pourquoi `tanh` en couche cachée et `argmax` en sortie ? Quelle conséquence l'argmax a-t-il (politique déterministe) sur la reproductibilité ?
- Détaille ta **fitness** : `100·s̄ + 0.5·t̄ + R̄`. Pourquoi le ×100 sur le score ? Que se passe-t-il si tu mets ×1 ?
- Pourquoi évaluer chaque génome sur **6 épisodes** et pas 1 ? Quel est le lien avec le « bruit de fitness » que tu cites en limite ?
- Tu renouvelles les graines d'éval à chaque génération (`10000 + gen·100 + i`). Pourquoi ? Quel surapprentissage ça empêche, et quel prix payes-tu (courbes bruitées) ?
- Explique tournoi taille 5 : que se passe-t-il si tu mets taille 2 ? taille 50 ? (pression de sélection vs diversité)
- Croisement uniforme à 0.7 : que veut dire le 0.7 ? Pourquoi du croisement gène par gène plutôt qu'à un point ?
- Mutation gaussienne avec σ décroissant 0.25 → 0.10 : pourquoi décroître ? Quel est l'analogue en optimisation (exploration vs exploitation, recuit) ?
- À quoi sert l'**élitisme 12 %** ? Que risques-tu sans élitisme ? Et avec 90 % d'élite ?
- Distingue **génotype (243 réels) et phénotype (comportement de jeu)**. Pourquoi un petit changement de génome peut donner un grand saut de comportement ?
- Ton AG bat le hasard ×70 mais reste sous l'heuristique en score. La robustesse montre que l'écart vient des **obstacles denses**, pas de la recherche de nourriture. Explique mécaniquement pourquoi l'observation locale cause ça (impasses, culs-de-sac).
- L'AG **survit plus longtemps** que l'heuristique (172.8 vs 153.1) mais mange moins. Comment interprètes-tu ça ? Est-ce un signe de fitness mal calibrée ?
- Si on te demande d'améliorer : **NEAT vs CMA-ES vs observation enrichie** — lequel attaque ta vraie limite, et pourquoi ?
- Question piège : « C'est de l'apprentissage par renforcement ? » Comment tu réponds proprement (frontière RL / optimisation boîte noire) ?
