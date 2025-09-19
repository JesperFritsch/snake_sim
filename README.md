to build the package:

    python setup.py build
    
    python setup.py build_ext --inplace

install:

    pip install <repo root>

to run the simulation on a headless server 'xvfb' needs to be installed

to install xvfb:

    sudo apt update && sudo apt upgrade

    sudo apt install xvfb