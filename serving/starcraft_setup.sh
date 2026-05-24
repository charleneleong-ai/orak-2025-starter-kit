#!/usr/bin/env bash
# Install the public Blizzard SC2 Linux 4.10 distribution so the
# star_craft env (evaluation_utils/mcp_game_servers/star_craft/) can spawn
# a real SC2 process.
#
# burnysc2 (==7.1.1 in pyproject.toml) looks for the binary at
# ~/StarCraftII/Versions/<base>/SC2_x64 by default. This script downloads
# Blizzard's public Linux research package, extracts it with the
# documented EULA password, and (if SC2_INSTALL_DIR is a different path)
# symlinks ~/StarCraftII to it so burnysc2 finds it without env tweaks.
#
# Only the Flat64 map ships in this distro. Custom maps in
# LADDER_MAP_2023 (Ancient Cistern LIE, LastFantasyAIE) need a separate
# download — the MACLA root configs (configs/gemma_26b.yaml,
# configs/qwen35_a3b_int4.yaml) override star_craft/env to
# linux_default.yaml which pins map_idx=2 (Flat64) for that reason.
#
# Usage:
#   ./serving/starcraft_setup.sh                # default: install to /workspace/StarCraftII
#   SC2_INSTALL_DIR=/opt/StarCraftII ./serving/starcraft_setup.sh
#   SC2_KEEP_ZIP=1 ./serving/starcraft_setup.sh  # don't delete the 4GB zip after extract
#
# Env overrides:
#   SC2_INSTALL_DIR   target directory (default /workspace/StarCraftII)
#   SC2_ZIP_URL       Blizzard zip URL (default Linux 4.10)
#   SC2_ZIP_PASSWORD  unzip password (default iagreetotheeula — Blizzard's documented EULA word)
#   SC2_KEEP_ZIP      if set, keep the downloaded zip on success
set -euo pipefail

SC2_INSTALL_DIR="${SC2_INSTALL_DIR:-/workspace/StarCraftII}"
SC2_ZIP_URL="${SC2_ZIP_URL:-http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip}"
SC2_ZIP_PASSWORD="${SC2_ZIP_PASSWORD:-iagreetotheeula}"

HOME_LINK="${HOME:-/root}/StarCraftII"
STAGING_DIR="$(dirname "${SC2_INSTALL_DIR}")/_sc2_install"
ZIP_PATH="${STAGING_DIR}/SC2.4.10.zip"

echo "============================================"
echo "  SC2 Linux setup"
echo "  Install dir: ${SC2_INSTALL_DIR}"
echo "  Home link:   ${HOME_LINK}"
echo "  Staging:     ${STAGING_DIR}"
echo "  Zip URL:     ${SC2_ZIP_URL}"
echo "============================================"

if [[ -x "${SC2_INSTALL_DIR}/Versions" ]] || ls "${SC2_INSTALL_DIR}/Versions"/Base*/SC2_x64 >/dev/null 2>&1; then
    echo "[setup] SC2 already installed at ${SC2_INSTALL_DIR} — skipping download/extract."
else
    mkdir -p "${STAGING_DIR}"
    if [[ ! -s "${ZIP_PATH}" ]]; then
        echo "[setup] downloading ${SC2_ZIP_URL} -> ${ZIP_PATH} (~4.1 GB)"
        curl -fL -o "${ZIP_PATH}" "${SC2_ZIP_URL}"
    else
        echo "[setup] zip already present at ${ZIP_PATH} ($(du -h "${ZIP_PATH}" | cut -f1)), skipping download"
    fi

    echo "[setup] extracting to $(dirname "${SC2_INSTALL_DIR}")/"
    (cd "$(dirname "${SC2_INSTALL_DIR}")" && unzip -P "${SC2_ZIP_PASSWORD}" -q "${ZIP_PATH}")
    if [[ ! -d "${SC2_INSTALL_DIR}" ]]; then
        echo "ERROR: expected ${SC2_INSTALL_DIR} after extract but it doesn't exist."
        echo "       (zip top-level dir may not be 'StarCraftII'? check ${STAGING_DIR})"
        exit 1
    fi

    if [[ -z "${SC2_KEEP_ZIP+x}" ]]; then
        echo "[setup] removing zip ${ZIP_PATH} (set SC2_KEEP_ZIP=1 to keep)"
        rm -f "${ZIP_PATH}"
    fi
fi

# Symlink ~/StarCraftII -> SC2_INSTALL_DIR so burnysc2 finds the binary.
if [[ "${SC2_INSTALL_DIR}" != "${HOME_LINK}" ]]; then
    if [[ -L "${HOME_LINK}" ]]; then
        existing="$(readlink "${HOME_LINK}")"
        if [[ "${existing}" != "${SC2_INSTALL_DIR}" ]]; then
            echo "[setup] replacing stale symlink ${HOME_LINK} -> ${existing}"
            rm "${HOME_LINK}"
        fi
    elif [[ -e "${HOME_LINK}" ]]; then
        echo "ERROR: ${HOME_LINK} exists and is not a symlink."
        echo "       Refusing to clobber. Move/remove it manually if you want to reinstall."
        exit 1
    fi
    if [[ ! -L "${HOME_LINK}" ]]; then
        ln -s "${SC2_INSTALL_DIR}" "${HOME_LINK}"
        echo "[setup] symlinked ${HOME_LINK} -> ${SC2_INSTALL_DIR}"
    fi
fi

bin="$(ls "${SC2_INSTALL_DIR}"/Versions/Base*/SC2_x64 2>/dev/null | head -1 || true)"
if [[ -z "${bin}" ]]; then
    echo "ERROR: SC2_x64 binary not found under ${SC2_INSTALL_DIR}/Versions/Base*/"
    exit 1
fi
chmod +x "${bin}"
echo "[setup] SC2 binary: ${bin}"

# SC2 4.10's protocol lowercases the Maps/ path before sending it to the
# game process. Linux is case-sensitive, so without this symlink the
# game fails with InvalidMapPath even when the .SC2Map file exists.
if [[ ! -e "${SC2_INSTALL_DIR}/maps" ]]; then
    (cd "${SC2_INSTALL_DIR}" && ln -s Maps maps)
    echo "[setup] linked lowercase maps -> Maps (workaround for SC2 4.10 path normalisation)"
fi

# burnysc2 hardcodes `-eglpath libEGL.so` but distros only ship the
# versioned libEGL.so.1 / libGL.so.1 (the libegl-dev / libgl-dev
# packages provide the unversioned symlinks). Without them SC2 silently
# falls back to no-render and rgb_render_config -> map_image never
# populates -> the agent never sees the game image. Install
# project-local unversioned symlinks and bots.py will prepend
# $SC2_INSTALL_DIR/libs to LD_LIBRARY_PATH so SC2 finds them.
mkdir -p "${SC2_INSTALL_DIR}/libs"
for libname in libEGL libGL; do
    src=""
    for cand in "/usr/lib/x86_64-linux-gnu/${libname}.so.1" \
                "/usr/lib/${libname}.so.1" \
                "/lib/x86_64-linux-gnu/${libname}.so.1"; do
        if [[ -f "${cand}" ]]; then src="${cand}"; break; fi
    done
    if [[ -n "${src}" ]]; then
        ln -sf "${src}" "${SC2_INSTALL_DIR}/libs/${libname}.so"
        echo "[setup] symlinked ${SC2_INSTALL_DIR}/libs/${libname}.so -> ${src}"
    else
        echo "[setup] WARN: ${libname}.so.1 not found — image rendering won't work."
    fi
done

echo "[setup] maps available:"
ls "${SC2_INSTALL_DIR}"/Maps/ 2>/dev/null || echo "  (none — Maps directory missing)"

echo
echo "[setup] done. Smoke check:"
echo "    python -c 'from sc2 import maps; print(maps.get(\"Flat64\"))'"
echo
echo "[setup] then run the MACLA adapter on Flat64:"
echo "    ./serving/gemma_serve.sh &     # vLLM in another shell"
echo "    python run.py -c gemma_26b --local --games star_craft"
