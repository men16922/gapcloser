#!/usr/bin/env bash
# Publish site/ to the gh-pages branch (GitHub Pages serves it at https://<owner>.github.io/<repo>/).
set -euo pipefail
cd "$(dirname "$0")/.."
remote=$(git remote get-url origin)
tmp=$(mktemp -d)
cp -R site/. "$tmp/"
cd "$tmp"
git init -q -b gh-pages
git add -A
git -c user.name="$(git -C "$OLDPWD" config user.name)" -c user.email="$(git -C "$OLDPWD" config user.email)" \
    commit -q -m "Site: $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push -q --force "$remote" gh-pages
rm -rf "$tmp"
echo "pushed site/ to gh-pages"
