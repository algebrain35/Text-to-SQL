#!/bin/sh
# Inside the compose network Postgres and Neo4j are reached by service name, not localhost.
# Rewrite those two addresses in a copy of config.yml so one file serves both the host and the containers.
# A config that already points somewhere else (a managed database, say) is left alone.
# SCHEMAGRAPH_SOURCE_CONFIG and SCHEMAGRAPH_OUT_CONFIG exist so the rewrite can be tested outside a container.
set -eu

SRC="${SCHEMAGRAPH_SOURCE_CONFIG:-/app/config.yml}"
OUT="${SCHEMAGRAPH_OUT_CONFIG:-/tmp/config.yml}"

sed \
    -e 's@^\([[:space:]]*host:[[:space:]]*\)\(127\.0\.0\.1\|localhost\)\([[:space:]].*\)\{0,1\}$@\1db@' \
    -e 's@bolt://\(127\.0\.0\.1\|localhost\):@bolt://neo4j:@' \
    "$SRC" > "$OUT"

export SCHEMAGRAPH_CONFIG="$OUT"
exec "$@"
