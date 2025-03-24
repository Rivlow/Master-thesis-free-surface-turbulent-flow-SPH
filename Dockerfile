FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Configuration pour éviter les interactions pendant l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Variables pour la version Python
ARG PYTHON_VERSION=3.8
ARG PYTHON_EXECUTABLE=/usr/bin/python${PYTHON_VERSION}

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    freeglut3-dev \
    git \
    libgl1-mesa-glx \
    libglu1-mesa-dev \
    libglew-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    mesa-utils \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python${PYTHON_VERSION}-tk \
    python3-pip \
    python3-tk \
    python3-venv \
    rsync \
    vim \
    wget \
    xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configuration de Python
RUN update-alternatives --install /usr/bin/python python ${PYTHON_EXECUTABLE} 1 && \
    update-alternatives --install /usr/bin/python3 python3 ${PYTHON_EXECUTABLE} 1 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    ${PYTHON_EXECUTABLE} get-pip.py && \
    rm get-pip.py && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir wheel==0.38.4 pybind11==2.10.4 pipx==1.2.0 && \
    python3 -m pip install --no-cache-dir scipy==1.10.1 matplotlib==3.7.1 && \
    python3 -m pipx ensurepath

# Configuration des variables d'environnement
ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    MESA_GL_VERSION_OVERRIDE=3.3 \
    PYTHONPATH="/usr/lib/python${PYTHON_VERSION}/site-packages" \
    PYTHONHOME="/usr"

# Définition du répertoire de travail pour la compilation
WORKDIR /opt/splishsplash_build

# Clone et compilation de SPlisHSPlasH avec les bindings Python
RUN git clone https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git . && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DUSE_PYTHON_BINDINGS=On \
          -DUSE_EMBEDDED_PYTHON=On \
          -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} .. && \
    make -j $(nproc)

# Installation des bindings Python
RUN python3 setup.py bdist_wheel && \
    pip install build/dist/*.whl && \
    # Vérification de l'installation des bindings Python
    python3 -c "import pysplishsplash; print('Python bindings successfully installed!')"

# Modification du fichier source pour ignorer l'avertissement de scriptFile
RUN SCRIPT_FILE="/opt/splishsplash_build/Simulator/ScriptObject.cpp" && \
    sed -i 's/#ifndef USE_EMBEDDED_PYTHON/#if 0/' $SCRIPT_FILE && \
    cd build && \
    make -j $(nproc)

# Préparation du répertoire d'exécution
WORKDIR /opt/splishsplash

# Création du script d'initialisation
RUN echo '#!/bin/bash\n\
# Création des répertoires nécessaires\n\
mkdir -p /opt/splishsplash/bin\n\
mkdir -p /opt/splishsplash/data\n\
\n\
# Copie des binaires si le répertoire est vide\n\
if [ ! -d "/opt/splishsplash/bin" ] || [ -z "$(ls -A /opt/splishsplash/bin)" ]; then\n\
    echo "Initializing binaries..."\n\
    rsync -a --exclude build /opt/splishsplash_build/bin/ /opt/splishsplash/bin/\n\
fi\n\
\n\
# Copie des données si le répertoire est vide\n\
if [ ! -d "/opt/splishsplash/data" ] || [ -z "$(ls -A /opt/splishsplash/data)" ]; then\n\
    echo "Initializing data..."\n\
    rsync -a /opt/splishsplash_build/data/ /opt/splishsplash/data/\n\
fi\n\
\n\
# Configuration de l environnement Python\n\
export PYTHONPATH="/usr/lib/python'${PYTHON_VERSION}'/site-packages:${PYTHONPATH}"\n\
export PYTHONHOME="/usr"\n\
\n\
# Exécution de la commande fournie\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Configuration de l'entrée du conteneur
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]