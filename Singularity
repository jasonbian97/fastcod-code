Bootstrap: docker
From: ubuntu:18.04

# this script is adapted from https://github.com/MASILab/PreQual/blob/master/Singularity

%post -c /bin/bash

    cd /

    # Prepare directories for installing applications
    mkdir -p APPS
    mkdir -p INSTALLERS

    # Update all libraries
    apt-get -y update

    # Install xvfb, used to run graphical applications without displaying them on a screen.
    # apt-get -y install xvfb

    # Install ghostscript for pdf management
    # apt-get -y install ghostscript

    # Install MRTrix3
    apt-get -y install git g++ python python-numpy libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev python3-distutils
    cd APPS
    git clone https://github.com/MRtrix3/mrtrix3.git
    cd mrtrix3
    git checkout 3.0.3
    ./configure
    ./build
    cd /
    
    # Make custom folders
    mkdir -p INPUTS
    mkdir -p SUPPLEMENTAL
    mkdir -p CODE

    # Set Permissions
    chmod 755 /INPUTS
    chmod 755 /SUPPLEMENTAL
    chmod 755 /APPS
    chmod 755 /CODE
    
    # Install source code
    cd /
    apt-get -y install wget git gcc libpq-dev python-dev python-pip python3 python3.8 python3.8-venv python3.8-dev python3-dev python3-pip python3-venv python3-wheel libpng-dev libfreetype6-dev libblas3 liblapack3 libblas-dev liblapack-dev pkg-config
    cd /INSTALLERS
    git clone https://github.com/jasonbian97/fastcod-code.git
    cd fastcod-code
    mv src/* /APPS
    cd /APPS
    python3.6 -m venv fastcod_env
    source fastcod_env/bin/activate
    pip3 install wheel
    pip install -r /INSTALLERS/PreQual/venv/pip_install_synb0.txt
    deactivate

    cd /

    # Clean up
    rm -r /INSTALLERS

%environment

    # MRTrix3
    export PATH="/APPS/mrtrix3/bin:$PATH"

%runscript

    xvfb-run -a --server-num=$((65536+$$)) --server-args="-screen 0 1600x1280x24 -ac" bash /CODE/run_dtiQA.sh /INPUTS /OUTPUTS "$@"

    # "$@" passes any additional arguments provided to the container to the run_dtiQA.sh script.
    # xvfb-run is useful for running applications that require an X display server without actually having a physical display available.
