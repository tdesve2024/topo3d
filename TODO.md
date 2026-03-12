# TODO — topo3d

## 1. Emboîtement des couches (zones fantômes)
Le lissage B-spline indépendant par couche casse l'emboîtement naturel du raster
(`altitude >= z_N` ⊃ `altitude >= z_{N+1}`). Les débordements atteignent 22 mm / 40 000 mm².
Le clipping post-extraction (emboîtement + chanfrein) a été implémenté mais ne corrige
pas complètement le problème visuellement. Piste : lisser **une seule fois** la couche
la plus basse, puis dériver les couches supérieures par intersection avec le raster,
ou bien lisser les contours de manière **cohérente entre couches** (spline partagée).

## 2. Format 16:9 de l'aperçu
L'aperçu généré dans l'interface web ne respecte pas le ratio 16:9 demandé.
Vérifier `run_preview` (figsize) et le rendu côté `templates/params.html` / endpoint
`/api/preview`.

## 3. Dernière couche vide
Si la dernière couche (sommet) n'a aucune pièce, ne pas générer de SVG pour elle.
Vérifier dans `run_generate` et `generer_guide_montage` — skipper les couches sans
polygone et ajuster la numérotation en conséquence.

## 4. Persistance des paramètres (serveur)
Certains paramètres ne sont pas sauvegardés entre deux sessions du serveur Python,
en particulier le ratio de la carte (largeur/hauteur). Vérifier `session.json`,
les clés sauvées dans `server.py` (load/save session), et le formulaire `params.html`
pour s'assurer que tous les champs sont persistés.
