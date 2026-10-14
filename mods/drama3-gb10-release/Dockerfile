# Minimal model-free ARM64 compact-head release; build from repository root.
FROM ghcr.io/mfellner/glm-5.3-flash-2x-dgx-sparks@sha256:5da66fdde5e5efaa54abfc9b4c8bcf8517960d4a7b69b87b32e93e2b1440bdaa
USER root
COPY mods/drama3-gb10-release/serve.py mods/drama3-gb10-release/run.sh mods/drama3-gb10-release/Dockerfile mods/drama3-gb10-release/NOTICE mods/drama3-gb10-release/manifest.sha256 /opt/drama-release/
COPY mods/drama3-gb10-perf/native_perf.py mods/drama3-gb10-perf/apply_patch.py mods/drama3-gb10-perf/native.patch mods/drama3-gb10-perf/source-pin.json mods/drama3-gb10-perf/NOTICE mods/drama3-gb10-perf/FISH_LICENSE.txt mods/drama3-gb10-perf/ADAPTER_LICENSE.txt /opt/drama-release/perf/
RUN cd /opt/drama-release && sha256sum --quiet -c manifest.sha256 \
 && /opt/drama-env/bin/python perf/apply_patch.py /opt/fish-speech \
 && chmod -R a+rX /opt/drama-release && chmod 755 /opt/drama-release/run.sh
ENV FISH_DRAMA3_COMPACT_HEAD=1 FISH_DRAMA3_SMALL_GRAPH_BUCKETS=0 FISH_DRAMA3_PREFIX_SNAPSHOT=0 \
    PYTHONPATH=/opt/drama:/opt/fish-speech
LABEL org.opencontainers.image.title="Drama 3 native GB10 compact-head" \
      org.opencontainers.image.source="https://github.com/mfellner/sparkrun-recipes" \
      org.opencontainers.image.version="drama3-fast-20261014" \
      io.sparkrun.fish-source="1ff3c3ed2f80d3106a83909dd25b83129c3d816f" \
      io.sparkrun.model-revision="dd64a85cd82e9521f7202610ede97b367fc83946"
USER drama
ENTRYPOINT ["/opt/drama-release/run.sh"]
