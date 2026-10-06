FROM node:22-bookworm-slim@sha256:43ac6c60b8f89723f746e8a92ce91abd5017e627ce1ddfe4238355d3a30b772c
# Pin the audited CLI independently of the chess services' installations.
RUN apt-get update && apt-get install -y --no-install-recommends python3 git ca-certificates curl bubblewrap \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global @openai/codex@0.154.0 \
    && npm cache clean --force
ENTRYPOINT ["codex"]
