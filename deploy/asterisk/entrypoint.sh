#!/bin/sh
# Substitutes secrets into ari.conf at start so they never live in the image.
set -eu
: "${ASTERISK_ARI_PASSWORD:?must be set}"
sed "s|__ARI_PASSWORD__|${ASTERISK_ARI_PASSWORD}|" /etc/asterisk/ari.conf.tmpl > /etc/asterisk/ari.conf
exec /usr/sbin/asterisk -f -vvv -T -U root
