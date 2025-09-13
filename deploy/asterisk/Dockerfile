# Built from source because third-party image tags lag (docs/PLAN.md F11).
FROM debian:bookworm-slim AS build
ARG ASTERISK_VERSION=22.11.0
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl libedit-dev libjansson-dev libssl-dev \
      libsqlite3-dev libxml2-dev libxslt1-dev uuid-dev pkg-config xz-utils bzip2 patch python3 file
WORKDIR /src
RUN curl -fsSL "https://downloads.asterisk.org/pub/telephony/asterisk/releases/asterisk-${ASTERISK_VERSION}.tar.gz" -o a.tgz \
    && tar xzf a.tgz --strip-components=1 && rm a.tgz
RUN ./configure --with-pjproject-bundled --with-jansson --without-dahdi --without-pri --without-ss7 >/dev/null \
    && make menuselect.makeopts >/dev/null \
    && menuselect/menuselect --disable BUILD_NATIVE --disable-category MENUSELECT_CORE_SOUNDS \
         --disable-category MENUSELECT_MOH --disable-category MENUSELECT_EXTRA_SOUNDS menuselect.makeopts \
    && make -j1 third-party >/dev/null
# The bundled pjproject build fails under parallel make, so it gets its own serial step above.
RUN make -j"$(nproc)" >/dev/null && make install DESTDIR=/out >/dev/null && rm -rf /out/var/run

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libedit2 libjansson4 libssl3 libsqlite3-0 libxml2 libuuid1 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /
RUN ldconfig && mkdir -p /etc/asterisk /var/run/asterisk /var/log/asterisk /var/spool/asterisk /var/lib/asterisk
COPY conf/ /etc/asterisk/
COPY entrypoint.sh /deploy-entrypoint.sh
RUN chmod +x /deploy-entrypoint.sh
EXPOSE 5060/udp 8088/tcp
HEALTHCHECK --interval=5s --timeout=3s --retries=20 CMD asterisk -rx "ari show apps" >/dev/null || exit 1
ENTRYPOINT ["/deploy-entrypoint.sh"]
