#!/bin/sh
# Substitutes secrets into the config at start so they never live in the image.
set -eu
: "${ASTERISK_ARI_PASSWORD:?must be set}"
sed "s|__ARI_PASSWORD__|${ASTERISK_ARI_PASSWORD}|" /etc/asterisk/ari.conf.tmpl > /etc/asterisk/ari.conf
# Without PHONE_SIP_PASSWORD the softphone login gets a random password, so it's closed rather than open.
phone_password="${PHONE_SIP_PASSWORD:-$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
desk_password="${DESK_SIP_PASSWORD:-$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
sed -i -e "s|__PHONE_SIP_PASSWORD__|${phone_password}|" -e "s|__DESK_SIP_PASSWORD__|${desk_password}|" \
       -e "s|__HOST_LAN_IP__|${HOST_LAN_IP:-127.0.0.1}|g" /etc/asterisk/pjsip.conf
exec /usr/sbin/asterisk -f -vvv -T -U root
