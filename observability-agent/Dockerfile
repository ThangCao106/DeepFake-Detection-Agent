# Node 22 needed at RUNTIME here even though `@anthropic-ai/claude-code` itself
# gets npm-installed by the separate `login` service (docker-compose.yml) — that
# install writes into the shared `claude-home` volume, but the `claude` CLI it
# installs is a Node script, and claude_agent_sdk (Python) shells out to it on
# every query. So this image needs a working `node` binary even though it never
# runs `npm install` for claude-code itself. Copied from the official image
# rather than apt/NodeSource: Debian trixie's own repo only has Node 20 (too old
# for @anthropic-ai/claude-code, which wants >=22), and NodeSource's installer
# both doesn't reliably support trixie yet and would mean a curl download
# through the corporate SSL-inspecting proxy that was already unreliable
# (intermittent 503s) for a different domain.
FROM node:22-slim AS node

FROM python:3.13-slim

ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

# Silences harmless debconf "unable to initialize frontend" noise during apt installs
# below (no controlling tty during `docker build`) — doesn't affect the running container.
ENV DEBIAN_FRONTEND=noninteractive

# uv via pip (already present in this base image) instead of the astral.sh curl
# installer script — avoids an extra hop through the corporate proxy that was
# returning intermittent 503s for that domain.
RUN pip install --no-cache-dir uv

# Node 22 + npm, copied from the node:22-slim stage above (no apt/curl needed).
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version && npm --version

# Non-root user, UID/GID pinned to 1000:1000 (not left to auto-increment) so it's
# predictable from OUTSIDE the container too — if $HOME is bind-mounted from a host
# directory (e.g. `./claude-home:/home/agent` instead of a Docker-managed named
# volume), that host directory must be owned by this exact UID/GID or the container
# gets "Permission denied" writing anything under $HOME (uv's cache, claude's
# credentials, claude-mem's data, etc.). Before first run: `mkdir -p ./claude-home
# && chown -R 1000:1000 ./claude-home` on the host. App code lives at /app (outside
# $HOME) so it's never shadowed by the credential mount — mixing those two up
# (mounting something ON TOP of /app) would silently wipe out the baked-in app
# code and .venv on every start.
RUN useradd -m -u 1000 -s /bin/bash agent && mkdir -p /app && chown agent:agent /app
USER agent
ENV HOME=/home/agent
WORKDIR /app

# Copy CHỈ manifest trước rồi uv sync ngay — để layer cache của bước sync chỉ bị
# invalidate khi pyproject.toml/uv.lock thật sự đổi (thêm/bớt dependency), KHÔNG
# phải mỗi khi sửa code trong app/. Nếu copy code app TRƯỚC bước sync (thứ tự cũ),
# Docker sẽ coi mọi COPY sau layer đã đổi là "cache miss" luôn, kéo theo uv sync
# download lại toàn bộ package mỗi lần build dù dependency không đổi gì.
COPY --chown=agent:agent pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY --chown=agent:agent app ./app
COPY --chown=agent:agent static ./static
COPY --chown=agent:agent smoke_test.py ./

EXPOSE 8787

# 0.0.0.0 here is required for Docker's port mapping to reach the process at all.
# The "chỉ 127.0.0.1, không mở LAN" restriction is enforced one layer out, by
# binding the published port to 127.0.0.1 on the DOCKER HOST — see
# docker-compose.yml `ports:`. Do not "fix" this to 127.0.0.1 inside the
# container; that would make it unreachable even from the host.
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8787"]
